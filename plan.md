# Plan: Accelerate HyperClockCache

## Overview

Add software prefetching and inline the hash function to reduce memory latency
on the Lookup/Eviction hot paths of HyperClockCache. All changes are
correctness-neutral — prefetch instructions are pure CPU hints, and the inline
hash produces identical results.

## Approach

Three categories of changes, ordered by expected impact:

1. **Prefetch in FixedHCC Lookup** (FindSlot probe loop)
2. **Prefetch in AutoHCC Lookup** (naive chain traversal)
3. **Prefetch in FixedHCC Eviction** (clock sweep)
4. **Inline BijectiveHash2x64** for cache-specific use

---

## 1. Prefetch in FixedHCC FindSlot

### What changes

In `FixedHyperClockTable::FindSlot` (`clock_cache.cc:1047-1081`), prefetch the
first probe slot before entering the loop, and prefetch the next probe slot after
computing the next `current` index.

### Code

**Before:**
```cpp
template <typename MatchFn, typename AbortFn, typename UpdateFn>
inline FixedHyperClockTable::HandleImpl* FixedHyperClockTable::FindSlot(
    const UniqueId64x2& hashed_key, const MatchFn& match_fn,
    const AbortFn& abort_fn, const UpdateFn& update_fn) {
  size_t base = static_cast<size_t>(hashed_key[1]);
  size_t increment = static_cast<size_t>(hashed_key[0]) | 1U;
  size_t first = ModTableSize(base);
  size_t current = first;
  bool is_last;
  do {
    HandleImpl* h = &array_[current];
    if (match_fn(h)) {
      return h;
    }
    if (abort_fn(h)) {
      return nullptr;
    }
    current = ModTableSize(current + increment);
    is_last = current == first;
    update_fn(h, is_last);
  } while (!is_last);
  return nullptr;
}
```

**After:**
```cpp
template <typename MatchFn, typename AbortFn, typename UpdateFn>
inline FixedHyperClockTable::HandleImpl* FixedHyperClockTable::FindSlot(
    const UniqueId64x2& hashed_key, const MatchFn& match_fn,
    const AbortFn& abort_fn, const UpdateFn& update_fn) {
  size_t base = static_cast<size_t>(hashed_key[1]);
  size_t increment = static_cast<size_t>(hashed_key[0]) | 1U;
  size_t first = ModTableSize(base);
  size_t current = first;
  // Prefetch the first probe slot
  PREFETCH(&array_[current], 0, 1);
  bool is_last;
  do {
    HandleImpl* h = &array_[current];
    if (match_fn(h)) {
      return h;
    }
    if (abort_fn(h)) {
      return nullptr;
    }
    current = ModTableSize(current + increment);
    is_last = current == first;
    // Prefetch the next probe slot while processing update_fn
    if (!is_last) {
      PREFETCH(&array_[current], 0, 1);
    }
    update_fn(h, is_last);
  } while (!is_last);
  return nullptr;
}
```

### Rationale

- The first prefetch is issued as early as possible, before the match_fn
  (which involves an atomic fetch_add). This gives ~10-20 cycles of overlap
  before the data is actually accessed.
- The next-slot prefetch is issued after computing the next index but before
  `update_fn` runs, giving some overlap if `update_fn` does displacement updates.
- For Lookup specifically, `update_fn` is a no-op, so the next-slot prefetch
  mainly helps if the current iteration's match_fn/abort_fn path takes enough
  cycles (the atomic ops) to overlap with the prefetch latency. Even partial
  overlap is beneficial.

---

## 2. Prefetch in AutoHCC Naive Lookup

### What changes

In `AutoHyperClockTable::Lookup` (`clock_cache.cc:3108-3145`), prefetch the next
chain entry while processing the current one. This follows the same pattern used
in `inlineskiplist.h`.

### Code

**Before (`clock_cache.cc:3108-3145`):**
```cpp
HandleImpl* const arr = array_.Get();
NextWithShift next_with_shift = arr[home].head_next_with_shift.LoadRelaxed();
for (size_t i = 0; !next_with_shift.IsEnd() && i < 10; ++i) {
    HandleImpl* h = &arr[next_with_shift.GetNext()];
    // ... speculative key match, atomic acquire, etc. ...
    next_with_shift = h->chain_next_with_shift.LoadRelaxed();
}
```

**After:**
```cpp
HandleImpl* const arr = array_.Get();
NextWithShift next_with_shift = arr[home].head_next_with_shift.LoadRelaxed();
// Prefetch the first chain entry
if (!next_with_shift.IsEnd()) {
    PREFETCH(&arr[next_with_shift.GetNext()], 0, 1);
}
for (size_t i = 0; !next_with_shift.IsEnd() && i < 10; ++i) {
    HandleImpl* h = &arr[next_with_shift.GetNext()];
    // ... speculative key match, atomic acquire, etc. (unchanged) ...
    next_with_shift = h->chain_next_with_shift.LoadRelaxed();
    // Prefetch the next chain entry
    if (!next_with_shift.IsEnd()) {
        PREFETCH(&arr[next_with_shift.GetNext()], 0, 1);
    }
}
```

### Rationale

Chain entries are at random positions in the array, so each pointer-follow is
a likely cache miss. Prefetching one step ahead overlaps the fetch of the next
entry with the key comparison and atomic operations on the current entry.

---

## 3. Prefetch in FixedHCC Eviction

### What changes

In `FixedHyperClockTable::Evict` (`clock_cache.cc:1104-1148`), prefetch the
next batch of `step_size` slots before processing the current batch.

### Code

**Before:**
```cpp
for (;;) {
    for (size_t i = 0; i < step_size; i++) {
      HandleImpl& h = array_[ModTableSize(Lower32of64(old_clock_pointer + i))];
      bool evicting = ClockUpdate(h, data);
      if (evicting) {
        Rollback(h.hashed_key, &h);
        TrackAndReleaseEvictedEntry(&h);
      }
    }
    // ... exit conditions ...
    old_clock_pointer = clock_pointer_.FetchAddRelaxed(step_size);
}
```

**After:**
```cpp
for (;;) {
    for (size_t i = 0; i < step_size; i++) {
      HandleImpl& h = array_[ModTableSize(Lower32of64(old_clock_pointer + i))];
      bool evicting = ClockUpdate(h, data);
      if (evicting) {
        Rollback(h.hashed_key, &h);
        TrackAndReleaseEvictedEntry(&h);
      }
    }
    // ... exit conditions ...
    old_clock_pointer = clock_pointer_.FetchAddRelaxed(step_size);
    // Prefetch the next batch of slots
    for (size_t i = 0; i < step_size; i++) {
      PREFETCH(&array_[ModTableSize(Lower32of64(old_clock_pointer + i))], 0, 1);
    }
}
```

### Rationale

The eviction sweep accesses consecutive cache lines. Prefetching the next batch
while the exit-condition checks run overlaps memory latency with the branch
logic. The prefetch addresses are trivially predictable (sequential indices).

---

## 4. Inline BijectiveHash2x64

### What changes

Move the `BijectiveHash2x64` (seeded version) implementation from `util/hash.cc`
to `util/hash.h` as an `inline` function. Keep the out-of-line version as well
(it just calls the inline one), so other callers and profiling are unaffected.

The helpers `XXH3_avalanche` and `Multiply64to128` are already inline in their
respective headers (`util/math128.h`, `util/math.h`). We just need to move the
`XXH3_avalanche` helper to `util/hash.h` and have the inline `BijectiveHash2x64`
use it.

### Code changes in `util/hash.h`

Add after the existing declarations, before the closing namespace brace:

```cpp
#include "util/math.h"
#include "util/math128.h"

namespace {
inline uint64_t XXH3_avalanche_inline(uint64_t h64) {
  h64 ^= h64 >> 37;
  h64 *= 0x165667919E3779F9U;
  h64 ^= h64 >> 32;
  return h64;
}
}  // namespace

inline void BijectiveHash2x64(uint64_t in_high64, uint64_t in_low64,
                               uint64_t seed, uint64_t* out_high64,
                               uint64_t* out_low64) {
  const uint64_t bitflipl = 0x59973f0033362349U - seed;
  const uint64_t bitfliph = 0xc202797692d63d58U + seed;
  Unsigned128 tmp128 =
      Multiply64to128(in_low64 ^ in_high64 ^ bitflipl, 0x9E3779B185EBCA87U);
  uint64_t lo = Lower64of128(tmp128);
  uint64_t hi = Upper64of128(tmp128);
  lo += 0x3c0000000000000U;
  in_high64 ^= bitfliph;
  hi += in_high64 + (Lower32of64(in_high64) * uint64_t{0x85EBCA76});
  lo ^= EndianSwapValue(hi);
  tmp128 = Multiply64to128(lo, 0xC2B2AE3D27D4EB4FU);
  lo = Lower64of128(tmp128);
  hi = Upper64of128(tmp128) + (hi * 0xC2B2AE3D27D4EB4FU);
  *out_low64 = XXH3_avalanche_inline(lo);
  *out_high64 = XXH3_avalanche_inline(hi);
}

inline void BijectiveHash2x64(uint64_t in_high64, uint64_t in_low64,
                               uint64_t* out_high64, uint64_t* out_low64) {
  BijectiveHash2x64(in_high64, in_low64, 0, out_high64, out_low64);
}
```

### Code changes in `util/hash.cc`

The existing out-of-line functions stay but are now wrappers that call the inline
version (or can be removed since the header provides the inline definition).
We simply remove the duplicate bodies from `hash.cc` since the inline definitions
in the header serve as the single source of truth. The out-of-line declarations
in `hash.h` (non-inline) are replaced by the inline definitions above.

### Rationale

- Eliminates 2 function calls per Lookup (~10-20 cycles saved)
- Enables the compiler to interleave hash computation with the prefetch
  instruction we're adding in FindSlot
- All building blocks (`Multiply64to128`, `EndianSwapValue`, `Lower64of128`,
  etc.) are already inline in their headers — only the top-level composition
  was out-of-line

---

## Files Modified

| File | Changes |
|------|---------|
| `cache/clock_cache.cc` | Add PREFETCH in FindSlot, Evict, AutoHCC Lookup |
| `cache/clock_cache.cc` | Add `#include "port/port.h"` (for PREFETCH macro) |
| `util/hash.h` | Add inline BijectiveHash2x64, add includes for math128.h |
| `util/hash.cc` | Remove duplicate bodies (now inline in header) |

**No new files. No build system changes needed** (no new .cc files added).

---

## Test Plan

### Unit Tests

No new unit tests are needed because:
- Prefetch instructions have no semantic effect — they cannot change behavior
- The inline hash produces identical results to the out-of-line version
- All existing tests (`cache_test`, `hash_test`) will validate correctness

### Existing Tests to Run

```bash
make -j$(nproc) cache_test && ./cache_test
make -j$(nproc) hash_test && ./hash_test
```

### Performance Validation

Build and run `cache_bench` before and after the change:

```bash
make clean && DEBUG_LEVEL=0 make -j$(nproc) cache_bench

# FixedHCC benchmark (primary target)
./cache_bench --cache_type=fixed_hyper_clock_cache --threads=16 \
  --cache_size=1073741824 --value_bytes=8192 --ops_per_thread=2000000

# AutoHCC benchmark
./cache_bench --cache_type=hyper_clock_cache --threads=16 \
  --cache_size=1073741824 --value_bytes=8192 --ops_per_thread=2000000

# High-collision stress test (tests multi-probe prefetching)
./cache_bench --cache_type=fixed_hyper_clock_cache --threads=16 \
  --degenerate_hash_bits=3 --ops_per_thread=2000000
```

Compare ops/sec/thread between before and after.

### Stress Tests

```bash
make -j$(nproc) cache_test && ./cache_test --gtest_filter="*Stress*"
```

---

## Metrics and Observability

No new metrics are needed. The existing `cache_bench` tool reports ops/sec/thread
which directly measures the impact of these changes. The existing
`ReportProblems()` diagnostics (occupancy, eviction effort exceeded count) are
unaffected.

---

## Considerations and Trade-offs

### Why not also prefetch in Insert's FindSlot?

Insert also calls FindSlot. The prefetch we add in FindSlot itself applies to
ALL callers (Lookup, Insert, Erase), so Insert benefits automatically.

### Why locality=1 for PREFETCH?

Following the convention in `inlineskiplist.h` and `ribbon_impl.h`. Locality=1
means "moderate temporal locality" — the data may be accessed again soon but
not immediately. This maps to prefetching into L2/L3 on x86, which is appropriate
for hash table slots that may or may not be the target entry.

### Why not locality=3 (L1)?

L1 prefetch (`locality=3`) would pollute the L1 cache more aggressively. Since
we're speculatively prefetching slots that might not be needed (e.g., the next
probe slot when the first probe might hit), moderate locality is more appropriate.

### Risk of inlining BijectiveHash2x64

The `hash.h` header comment says out-of-lining aids profiling. By keeping the
function signature and making it `inline` in the header, profilers that support
inlined frames (perf, VTune) will still show time spent in the function.
The out-of-line version in `hash.cc` is removed to avoid ODR violations.

### Alternatives considered

1. **Batched multi-key Lookup API**: Would maximize memory-level parallelism by
   hashing multiple keys and issuing all prefetches at once. Rejected because it
   requires API changes and caller modifications — much higher complexity for
   uncertain benefit.

2. **Linear probing instead of double hashing**: The existing TODO at
   `clock_cache.cc:1061` considers this. Rejected because (a) each slot is a full
   cache line so spatial locality from linear probing is minimal, and (b) linear
   probing has worse clustering behavior at high load factors.

3. **Larger eviction step_size**: Could be done alongside prefetching, but
   would change eviction behavior (coarser granularity per thread). Deferred as
   a separate change — prefetching the current step_size=4 is the minimal
   non-behavioral change.

---

## Task Checklist

### Phase 1: Prefetching

- [x] Add `#include "port/port.h"` to `cache/clock_cache.cc` (for PREFETCH macro)
- [x] Add prefetch of first probe slot in `FixedHyperClockTable::FindSlot`
- [x] Add prefetch of next probe slot in `FixedHyperClockTable::FindSlot` loop
- [x] Add prefetch of first chain entry in `AutoHyperClockTable::Lookup`
- [x] Add prefetch of next chain entry in `AutoHyperClockTable::Lookup` naive loop
- [x] Fix bug: `next_with_shift.IsEnd()` → `next_with_shift.GetNext()` in AutoHCC naive Lookup (line 3123)
- [x] Add prefetch of next batch in `FixedHyperClockTable::Evict`
- [x] Run `make format-auto`
- [x] Build and run `cache_test` to verify correctness (77/77 PASSED)
- [x] Build and run `hash_test` to verify correctness (16/16 PASSED)

### Phase 2: Inline BijectiveHash2x64

- [x] Add `#include "util/math128.h"` to `util/hash.h`
- [x] Add inline `detail::XXH3_avalanche` helper to `util/hash.h`
- [x] Add inline `BijectiveHash2x64` (seeded) to `util/hash.h`
- [x] Add inline `BijectiveHash2x64` (no-seed) to `util/hash.h`
- [x] Remove out-of-line `BijectiveHash2x64` bodies from `util/hash.cc`
- [x] Remove out-of-line `XXH3_avalanche` from `util/hash.cc` (now in header)
- [x] Keep `BijectiveUnhash2x64` out-of-line (cold path, only used for key recovery)
- [x] Run `make format-auto`
- [x] Build and run `hash_test` to verify correctness (16/16 PASSED)
- [x] Build and run `cache_test` to verify correctness (77/77 PASSED)

### Phase 3: Validation

- [ ] Build release binary: `make clean && DEBUG_LEVEL=0 make -j$(nproc) cache_bench`
- [ ] Run FixedHCC benchmark and record results
- [ ] Run AutoHCC benchmark and record results
- [ ] Run high-collision benchmark and record results
