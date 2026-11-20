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
	"errors"
	"fmt"
	"math"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// fakeQuotaSource is a scriptable CPUQuotaSource. All methods are safe for
// concurrent use since the watcher polls from another goroutine.
type fakeQuotaSource struct {
	mu      sync.Mutex
	quota   int64
	period  int64
	defined bool
	err     error
	calls   int
}

func (f *fakeQuotaSource) CPUQuotaAndPeriod() (int64, int64, bool, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls++
	return f.quota, f.period, f.defined, f.err
}

func (f *fakeQuotaSource) setQuota(quota int64) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.quota = quota
	f.period = 100000
	f.defined = true
	f.err = nil
}

func (f *fakeQuotaSource) setUnlimited() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.quota = -1
	f.period = -1
	f.defined = false
	f.err = nil
}

func (f *fakeQuotaSource) setError(err error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.err = err
}

func (f *fakeQuotaSource) callCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.calls
}

// newTestWatcher builds a watcher without starting its polling goroutine so
// the sampling state machine can be driven deterministically. GOMAXPROCS is
// restored when the test ends.
func newTestWatcher(t *testing.T, src CPUQuotaSource, opts ...Option) *Watcher {
	t.Helper()

	prev := currentMaxProcs()
	t.Cleanup(func() {
		runtime.GOMAXPROCS(prev)
	})

	allOpts := append([]Option{QuotaSource(src), RefreshInterval(time.Hour)}, opts...)
	cfg := newConfig(allOpts)

	w := &Watcher{
		cfg:           cfg,
		originalProcs: prev,
		done:          make(chan struct{}),
	}
	w.ctx, w.cancel = context.WithCancel(context.Background())
	t.Cleanup(w.cancel)
	return w
}

func TestBeyondHysteresis(t *testing.T) {
	tests := []struct {
		current int
		next    int
		want    bool
	}{
		{10, 10, false}, // no change
		{10, 11, true},  // 10% of 10 is 1, and the threshold is inclusive
		{10, 9, true},
		{20, 21, false}, // 1 proc out of 20 is only 5%
		{20, 22, true},  // 10%
		{100, 105, false},
		{100, 110, true},
	}
	for _, tt := range tests {
		assert.Equal(t, tt.want, beyondHysteresis(tt.current, tt.next),
			"current=%d next=%d", tt.current, tt.next)
	}
}

func TestObserverInitialSample(t *testing.T) {
	t.Run("applies quota", func(t *testing.T) {
		src := &fakeQuotaSource{}
		src.setQuota(400000)
		w := newTestWatcher(t, src)

		w.observe(true)

		assert.Equal(t, 4, currentMaxProcs())
		assert.Equal(t, 4, w.appliedProcs)
		assert.False(t, w.Stale())
	})

	t.Run("honors minimum", func(t *testing.T) {
		src := &fakeQuotaSource{}
		src.setQuota(100000) // quota of 1 core
		w := newTestWatcher(t, src, Min(3))

		w.observe(true)

		assert.Equal(t, 3, currentMaxProcs())
		assert.Equal(t, 3, w.appliedProcs)
	})

	t.Run("honors rounding function", func(t *testing.T) {
		src := &fakeQuotaSource{}
		src.setQuota(250000) // 2.5 cores
		w := newTestWatcher(t, src, RoundQuotaFunc(func(v float64) int {
			return int(math.Ceil(v))
		}))

		w.observe(true)

		assert.Equal(t, 3, currentMaxProcs())
	})

	t.Run("undefined quota applies nothing", func(t *testing.T) {
		src := &fakeQuotaSource{}
		src.setUnlimited()
		w := newTestWatcher(t, src)

		w.observe(true)

		assert.Equal(t, w.originalProcs, currentMaxProcs())
		assert.Zero(t, w.appliedProcs)
		assert.False(t, w.Stale())
	})
}

func TestObserverStableSamples(t *testing.T) {
	src := &fakeQuotaSource{}
	src.setQuota(400000)
	w := newTestWatcher(t, src)
	w.observe(true)
	require.Equal(t, 4, currentMaxProcs())

	// The first sample of a new value only registers a candidate.
	src.setQuota(600000)
	w.observe(false)
	assert.Equal(t, 4, currentMaxProcs(), "single unstable sample must not change GOMAXPROCS")
	assert.Equal(t, 6, w.candidateProcs)
	assert.Equal(t, 1, w.candidateHits)

	// A disagreeing sample restarts the streak.
	src.setQuota(500000)
	w.observe(false)
	assert.Equal(t, 4, currentMaxProcs())
	src.setQuota(600000)
	w.observe(false)
	assert.Equal(t, 4, currentMaxProcs(), "streak must restart after a disagreeing sample")

	// Two consecutive agreeing samples apply the new value.
	w.observe(false)
	assert.Equal(t, 6, currentMaxProcs())
	assert.Equal(t, 6, w.appliedProcs)
}

func TestObserverHysteresis(t *testing.T) {
	src := &fakeQuotaSource{}
	src.setQuota(2000000) // 20 cores
	w := newTestWatcher(t, src)
	w.observe(true)
	require.Equal(t, 20, currentMaxProcs())

	// 21 is a stable new value but within the 10% hysteresis band of 20.
	src.setQuota(2100000)
	w.observe(false)
	w.observe(false)
	assert.Equal(t, 20, currentMaxProcs(), "sub-threshold change must be ignored")

	// 22 reaches the inclusive 10% threshold.
	src.setQuota(2200000)
	w.observe(false)
	w.observe(false)
	assert.Equal(t, 22, currentMaxProcs())

	// Scaling down uses the same threshold: 21 vs 22 is below 10%.
	src.setQuota(2100000)
	w.observe(false)
	w.observe(false)
	assert.Equal(t, 22, currentMaxProcs(), "sub-threshold scale-down must be ignored")

	src.setQuota(1900000) // 19 vs 22: ~13.6%, beyond the band
	w.observe(false)
	w.observe(false)
	assert.Equal(t, 19, currentMaxProcs())
}

func TestObserverReadFailureKeepsLastValue(t *testing.T) {
	src := &fakeQuotaSource{}
	src.setQuota(400000)
	w := newTestWatcher(t, src)
	w.observe(true)
	require.Equal(t, 4, currentMaxProcs())

	// A failed read must not be treated as unlimited CPU: the last known
	// good value stays in place and the watcher reports itself as stale.
	src.setError(errors.New("cgroup files vanished"))
	w.observe(false)
	assert.True(t, w.Stale())
	assert.Equal(t, 4, currentMaxProcs())
	assert.Zero(t, w.candidateProcs, "failure must discard pending candidates")

	// Repeated failures keep the stale state and the applied value.
	w.observe(false)
	assert.True(t, w.Stale())
	assert.Equal(t, 4, currentMaxProcs())

	// A successful read clears the stale flag.
	src.setQuota(400000)
	w.observe(false)
	assert.False(t, w.Stale())
	assert.Equal(t, 4, currentMaxProcs())
}

func TestObserverUnlimitedIsNotInfinite(t *testing.T) {
	src := &fakeQuotaSource{}
	src.setQuota(400000)
	w := newTestWatcher(t, src)
	w.observe(true)
	require.Equal(t, 4, currentMaxProcs())

	src.setUnlimited()
	w.observe(false)

	assert.False(t, w.Stale(), "explicitly unlimited is not a failed read")
	assert.Equal(t, 4, currentMaxProcs(), "unlimited quota must not raise GOMAXPROCS")
	assert.Zero(t, w.candidateProcs)
}

func TestObserverRestoreOnlyWhenOwned(t *testing.T) {
	src := &fakeQuotaSource{}
	src.setQuota(400000)
	w := newTestWatcher(t, src)
	w.observe(true)
	require.Equal(t, 4, currentMaxProcs())

	t.Run("restores when value still owned", func(t *testing.T) {
		w.restore()
		assert.Equal(t, w.originalProcs, currentMaxProcs())
		assert.Zero(t, w.appliedProcs)
	})

	t.Run("leaves externally changed value alone", func(t *testing.T) {
		w.observe(true)
		require.Equal(t, 4, currentMaxProcs())

		runtime.GOMAXPROCS(9)
		w.restore()
		assert.Equal(t, 9, currentMaxProcs(), "external GOMAXPROCS change must survive")
	})
}

func TestWatchEnvVarIsInert(t *testing.T) {
	withMax(t, 42, func() {
		prev := currentMaxProcs()
		src := &fakeQuotaSource{}
		src.setQuota(400000)

		w := Watch(context.Background(), QuotaSource(src))
		assert.True(t, isClosed(w.done), "inert watcher must not start a goroutine")
		assert.Equal(t, prev, currentMaxProcs())

		w.Stop()
		w.Stop() // repeated Stop is a no-op
		assert.Equal(t, prev, currentMaxProcs())
		assert.Zero(t, src.callCount(), "inert watcher must never read the quota")
	})
}

func TestWatchAppliesAndStops(t *testing.T) {
	prev := currentMaxProcs()
	t.Cleanup(func() { runtime.GOMAXPROCS(prev) })

	src := &fakeQuotaSource{}
	src.setQuota(400000)

	w := Watch(context.Background(), QuotaSource(src), RefreshInterval(time.Millisecond))
	assert.Equal(t, 4, currentMaxProcs(), "initial sample must apply before Watch returns")

	w.Stop()
	assert.True(t, isClosed(w.done), "Stop must wait for the polling goroutine")
	assert.Equal(t, prev, currentMaxProcs(), "Stop restores the original value while still owned")

	callsAfterStop := src.callCount()
	time.Sleep(20 * time.Millisecond)
	assert.Equal(t, callsAfterStop, src.callCount(), "no polling after Stop: timer/goroutine leaked")

	w.Stop() // idempotent
}

func TestWatchReactsToQuotaChange(t *testing.T) {
	prev := currentMaxProcs()
	t.Cleanup(func() { runtime.GOMAXPROCS(prev) })

	src := &fakeQuotaSource{}
	src.setQuota(200000)
	w := Watch(context.Background(), QuotaSource(src), RefreshInterval(time.Millisecond))
	defer w.Stop()
	require.Equal(t, 2, currentMaxProcs())

	src.setQuota(500000)
	require.Eventually(t, func() bool {
		return currentMaxProcs() == 5
	}, time.Second, time.Millisecond, "watcher should converge on the new quota")
}

func TestWatchStopsOnContextCancel(t *testing.T) {
	prev := currentMaxProcs()
	t.Cleanup(func() { runtime.GOMAXPROCS(prev) })

	src := &fakeQuotaSource{}
	src.setQuota(300000)
	ctx, cancel := context.WithCancel(context.Background())

	w := Watch(ctx, QuotaSource(src), RefreshInterval(time.Millisecond))
	require.Equal(t, 3, currentMaxProcs())

	cancel()
	assert.Eventually(t, func() bool {
		return isClosed(w.done)
	}, time.Second, time.Millisecond, "parent context cancellation must stop the watcher")
	assert.Equal(t, prev, currentMaxProcs(), "cancellation restores the original value while owned")

	w.Stop() // still safe after context-driven shutdown
}

func TestWatchStopPreservesExternalChange(t *testing.T) {
	prev := currentMaxProcs()
	t.Cleanup(func() { runtime.GOMAXPROCS(prev) })

	src := &fakeQuotaSource{}
	src.setQuota(300000)
	w := Watch(context.Background(), QuotaSource(src), RefreshInterval(time.Hour))
	require.Equal(t, 3, currentMaxProcs())

	runtime.GOMAXPROCS(7)
	w.Stop()
	assert.Equal(t, 7, currentMaxProcs(), "Stop must not overwrite an external change")
}

func TestWatchUndefinedQuotaIsInertButPolling(t *testing.T) {
	prev := currentMaxProcs()
	t.Cleanup(func() { runtime.GOMAXPROCS(prev) })

	src := &fakeQuotaSource{}
	src.setUnlimited()
	w := Watch(context.Background(), QuotaSource(src), RefreshInterval(time.Millisecond))
	defer w.Stop()

	assert.Equal(t, prev, currentMaxProcs())
	assert.Zero(t, w.appliedProcs)
	assert.Eventually(t, func() bool { return src.callCount() > 1 },
		time.Second, time.Millisecond, "watcher should keep polling for a quota to appear")

	src.setQuota(200000)
	assert.Eventually(t, func() bool {
		return currentMaxProcs() == 2
	}, time.Second, time.Millisecond, "a later quota should still be picked up")
}

func TestObserverLogDeduplication(t *testing.T) {
	t.Run("undefined quota logged once", func(t *testing.T) {
		src := &fakeQuotaSource{}
		src.setUnlimited()

		var logs []string
		w := newTestWatcher(t, src, Logger(func(format string, args ...interface{}) {
			logs = append(logs, fmt.Sprintf(format, args...))
		}))

		for i := 0; i < 3; i++ {
			w.observe(false)
		}
		assert.Equal(t, 1, countLogsContaining(logs, "quota undefined"),
			"undefined quota must not log on every poll")
	})

	t.Run("read failure logged once while stale", func(t *testing.T) {
		src := &fakeQuotaSource{}
		src.setQuota(400000)

		var logs []string
		w := newTestWatcher(t, src, Logger(func(format string, args ...interface{}) {
			logs = append(logs, fmt.Sprintf(format, args...))
		}))
		w.observe(true)

		src.setError(errors.New("boom"))
		w.observe(false)
		w.observe(false)
		assert.Equal(t, 1, countLogsContaining(logs, "Failed to read CPU quota"),
			"repeated failures must not spam the log")

		src.setQuota(400000)
		w.observe(false)
		src.setError(errors.New("boom again"))
		w.observe(false)
		assert.Equal(t, 2, countLogsContaining(logs, "Failed to read CPU quota"),
			"a new failure streak after recovery should log again")
	})
}

func countLogsContaining(logs []string, substr string) int {
	n := 0
	for _, line := range logs {
		if strings.Contains(line, substr) {
			n++
		}
	}
	return n
}

func TestRefreshIntervalOption(t *testing.T) {
	src := &fakeQuotaSource{}
	src.setQuota(400000)

	t.Run("default", func(t *testing.T) {
		cfg := newConfig([]Option{QuotaSource(src)})
		assert.Equal(t, defaultRefreshInterval, cfg.refresh)
	})

	t.Run("custom", func(t *testing.T) {
		cfg := newConfig([]Option{QuotaSource(src), RefreshInterval(time.Millisecond)})
		assert.Equal(t, time.Millisecond, cfg.refresh)
	})

	t.Run("non-positive ignored", func(t *testing.T) {
		cfg := newConfig([]Option{QuotaSource(src), RefreshInterval(0), RefreshInterval(-time.Second)})
		assert.Equal(t, defaultRefreshInterval, cfg.refresh)
	})
}

func isClosed(ch chan struct{}) bool {
	select {
	case <-ch:
		return true
	default:
		return false
	}
}
