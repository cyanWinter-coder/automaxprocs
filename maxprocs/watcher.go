// Copyright (c) 2026 Uber Technologies, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package maxprocs

import (
	"context"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	iruntime "go.uber.org/automaxprocs/internal/runtime"
)

const (
	// defaultRefreshInterval is how often the watcher re-reads the cgroup
	// CPU quota.
	defaultRefreshInterval = 10 * time.Second

	// stableSamples is the number of consecutive samples that must agree
	// before a new GOMAXPROCS value is applied. It filters out one-off
	// readings while a platform-driven quota change is settling.
	stableSamples = 2

	// hysteresisPercent is the minimum relative change (against the value
	// currently applied by the watcher) required before GOMAXPROCS is
	// adjusted. Small oscillations around the current value are ignored.
	hysteresisPercent = 10
)

// CPUQuotaSource provides the raw CPU quota and period enforced on the
// calling process. Both values are expressed in the same unit (microseconds
// for the cgroup CFS files); the effective CPU quota is quota/period.
//
// defined is false when no CPU quota is enforced (for example a cgroup with
// cpu.max set to "max"); err is non-nil when the quota could not be read.
// Implementations must keep the two cases distinct: a failed read must never
// be reported as an unlimited quota.
type CPUQuotaSource interface {
	CPUQuotaAndPeriod() (quota, period int64, defined bool, err error)
}

// defaultQuotaSource reads the quota from the Linux cgroup filesystem.
type defaultQuotaSource struct{}

func (defaultQuotaSource) CPUQuotaAndPeriod() (int64, int64, bool, error) {
	return iruntime.CgroupCPUQuotaAndPeriod()
}

// RefreshInterval sets how often Watch re-reads the CPU quota. Non-positive
// values are ignored and the default interval is used.
func RefreshInterval(d time.Duration) Option {
	return optionFunc(func(cfg *config) {
		if d > 0 {
			cfg.refresh = d
		}
	})
}

// QuotaSource injects the source used by Watch to read the CPU quota and
// period. This is mainly useful for tests and for platforms that expose CPU
// limits outside of the standard cgroup files.
func QuotaSource(source CPUQuotaSource) Option {
	return optionFunc(func(cfg *config) {
		if source != nil {
			cfg.quotaSource = source
		}
	})
}

// Watcher periodically re-reads the container CPU quota and keeps
// runtime.GOMAXPROCS in sync with it.
//
// The GOMAXPROCS environment variable, the minimum set with Min and the
// rounding strategy set with RoundQuotaFunc follow the same semantics as
// Set. A Watcher never treats a failed quota read as unlimited CPU: the last
// known good value is kept and reported as stale until reads succeed again.
type Watcher struct {
	cfg *config

	// ctx drives the polling goroutine. Cancelling it, either through the
	// context passed to Watch or through Stop, ends the watcher.
	ctx    context.Context
	cancel context.CancelFunc

	// done is closed once the polling goroutine has exited, guaranteeing
	// that no timer or goroutine outlives the watcher.
	done chan struct{}

	stopOnce sync.Once

	// originalProcs is the GOMAXPROCS value captured before the watcher
	// applied anything, and is restored on shutdown when the watcher still
	// owns the current value.
	originalProcs int

	// appliedProcs is the most recent GOMAXPROCS value applied by the
	// watcher. Zero means the watcher has never applied a value.
	appliedProcs int

	// candidateProcs and candidateHits track consecutive agreeing samples.
	candidateProcs int
	candidateHits  int

	// undefinedReported keeps the "quota undefined" log to a single line per
	// stretch of undefined readings, instead of one per poll.
	undefinedReported bool

	// stale indicates that the most recent quota read failed, so any value
	// currently applied is based on an older reading.
	stale atomic.Bool
}

// Watch starts a watcher that periodically adjusts runtime.GOMAXPROCS to
// match the calling process' CPU quota. The returned Watcher is stopped with
// its Stop method or by cancelling ctx; both paths stop the polling
// goroutine and release its timer.
//
// As with Set, Watch is a no-op when the GOMAXPROCS environment variable is
// set, and on platforms without a readable cgroup quota. The first sample is
// taken synchronously so the initial GOMAXPROCS adjustment happens before
// Watch returns.
func Watch(ctx context.Context, opts ...Option) *Watcher {
	cfg := newConfig(opts)

	// Honor the GOMAXPROCS environment variable if present, matching Set.
	if max, exists := os.LookupEnv(_maxProcsKey); exists {
		cfg.log("maxprocs: Honoring GOMAXPROCS=%q as set in environment", max)
		return newInertWatcher(ctx, cfg)
	}

	w := &Watcher{
		cfg:           cfg,
		originalProcs: currentMaxProcs(),
		done:          make(chan struct{}),
	}
	w.ctx, w.cancel = context.WithCancel(ctx)

	// Take the first sample synchronously so startup behavior matches Set.
	w.observe(true)

	go w.run()
	return w
}

// newInertWatcher returns a watcher whose quota source never produces a
// usable value. It performs no adjustments and has no goroutine or timer.
func newInertWatcher(ctx context.Context, cfg *config) *Watcher {
	w := &Watcher{
		cfg:           cfg,
		originalProcs: currentMaxProcs(),
		done:          make(chan struct{}),
	}
	w.ctx, w.cancel = context.WithCancel(ctx)
	close(w.done)
	return w
}

// Stale reports whether the latest quota read failed. While stale, the
// GOMAXPROCS value applied by the watcher is based on the last successful
// reading rather than the current cgroup state.
func (w *Watcher) Stale() bool {
	return w.stale.Load()
}

// run polls the quota source until the context is cancelled, then restores
// the original GOMAXPROCS when the watcher still owns the current value.
func (w *Watcher) run() {
	defer close(w.done)

	ticker := time.NewTicker(w.cfg.refresh)
	defer ticker.Stop()

	for {
		select {
		case <-w.ctx.Done():
			w.restore()
			return
		case <-ticker.C:
			w.observe(false)
		}
	}
}

// Stop ends the watcher. It is idempotent: repeated calls have no effect and
// never leak goroutines or timers. If the current GOMAXPROCS is still a value
// applied by this watcher, it is reset to the value captured when Watch was
// called; a value changed externally is left untouched.
func (w *Watcher) Stop() {
	w.stopOnce.Do(func() {
		w.cancel()
		<-w.done
	})
}

// restore resets GOMAXPROCS only while the watcher still owns the current
// value. A value set by other code is preserved.
func (w *Watcher) restore() {
	if w.appliedProcs == 0 {
		w.cfg.log("maxprocs: No GOMAXPROCS change to reset")
		return
	}
	if current := currentMaxProcs(); current != w.appliedProcs {
		w.cfg.log(
			"maxprocs: Leaving GOMAXPROCS=%v: changed externally from %v, not resetting",
			current, w.appliedProcs,
		)
		return
	}
	w.cfg.log("maxprocs: Resetting GOMAXPROCS to %v", w.originalProcs)
	runtime.GOMAXPROCS(w.originalProcs)
	w.appliedProcs = 0
}

// observe reads the quota source once and, depending on the reading, applies
// a new GOMAXPROCS value. The initial reading (initial == true) is applied
// immediately; later readings must be stable and clear the hysteresis
// threshold.
func (w *Watcher) observe(initial bool) {
	quota, period, defined, err := w.cfg.quotaSource.CPUQuotaAndPeriod()
	if err != nil {
		// Keep the last known good value. An unreadable quota is not the
		// same as unlimited CPU.
		if !w.stale.Swap(true) {
			w.cfg.log("maxprocs: Failed to read CPU quota: %v; keeping GOMAXPROCS=%v",
				err, currentMaxProcs())
		}
		w.candidateProcs = 0
		w.candidateHits = 0
		return
	}
	w.stale.Store(false)

	if !defined {
		// The quota is explicitly unlimited: leave GOMAXPROCS alone and do
		// not treat this reading as a candidate value.
		if !w.undefinedReported {
			w.cfg.log("maxprocs: CPU quota undefined; leaving GOMAXPROCS=%v", currentMaxProcs())
			w.undefinedReported = true
		}
		w.candidateProcs = 0
		w.candidateHits = 0
		return
	}
	w.undefinedReported = false

	desired := w.cfg.roundQuotaFunc(float64(quota) / float64(period))
	minUsed := false
	if w.cfg.minGOMAXPROCS > 0 && desired < w.cfg.minGOMAXPROCS {
		desired = w.cfg.minGOMAXPROCS
		minUsed = true
	}

	if initial {
		w.apply(desired, minUsed)
		return
	}

	if desired != w.candidateProcs {
		w.candidateProcs = desired
		w.candidateHits = 1
		return
	}
	w.candidateHits++
	if w.candidateHits < stableSamples {
		return
	}

	if desired == w.appliedProcs || !beyondHysteresis(w.appliedProcs, desired) {
		// The stable reading is equal to, or within the hysteresis band of,
		// the value currently in effect.
		w.candidateHits = 0
		return
	}

	w.apply(desired, minUsed)
}

// beyondHysteresis reports whether moving from current to next exceeds the
// relative hysteresis threshold. current is always a GOMAXPROCS value the
// watcher applied, so it is at least 1.
func beyondHysteresis(current, next int) bool {
	diff := next - current
	if diff < 0 {
		diff = -diff
	}
	return diff*100 >= current*hysteresisPercent
}

// apply updates GOMAXPROCS and records the watcher as the owner of the new
// value.
func (w *Watcher) apply(procs int, minUsed bool) {
	switch {
	case w.appliedProcs == procs:
		// Nothing to do.
	case minUsed:
		w.cfg.log("maxprocs: Updating GOMAXPROCS=%v: using minimum allowed GOMAXPROCS", procs)
		runtime.GOMAXPROCS(procs)
		w.appliedProcs = procs
	default:
		w.cfg.log("maxprocs: Updating GOMAXPROCS=%v: determined from CPU quota", procs)
		runtime.GOMAXPROCS(procs)
		w.appliedProcs = procs
	}
	w.candidateProcs = 0
	w.candidateHits = 0
}
