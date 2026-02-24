# Research: Accelerating HyperClockCache

## 1. How HyperClockCache Works Today

### 1.1 Architecture Overview

HyperClockCache (HCC) is RocksDB's high-performance block cache, replacing LRUCache
for concurrent workloads. It comes in two variants:

- **FixedHyperClockCache** (`estimated_entry_charge > 0`): Pre-sized open-addressing
  hash table with double hashing. Fully lock/wait-free.
- **AutoHyperClockCache** (`estimated_entry_charge == 0`, default): Dynamically-growing
  hash table using linear hashing with chaining. Essentially wait-free.

**"Charge"** refers to the amount of memory (in bytes) that a cache entry is
accounted for. When you insert a block into the cache, you specify how many bytes
it "charges" against the cache's capacity. For example, a 4KB data block has a
charge of 4096. The sum of all charges is tracked as `usage_` and compared against
`capacity_` to decide when eviction is needed. The `estimated_entry_charge` option
tells FixedHCC what the average charge will be so it can pre-size the hash table.

**Lock-free vs wait-free**: Both are levels of non-blocking progress guarantees.
- **Lock-free**: At least one thread among all concurrent threads is guaranteed to
  make progress in a finite number of steps. Individual threads may starve (retry
  CAS loops indefinitely) but the system as a whole always moves forward. Example:
  a CAS loop that may fail and retry, but each failure means some other thread
  succeeded.
- **Wait-free**: Every thread is guaranteed to complete its operation in a bounded
  number of steps, regardless of what other threads do. No thread can ever starve.
  Example: a single `fetch_add` that always succeeds in one step.
- FixedHCC is **fully wait-free** because Lookup/Release use unconditional
  `fetch_add` (always completes in one step). AutoHCC is "essentially wait-free"
  because Lookup/Release are wait-free, but rare Insert/Erase/Grow operations
  use a per-chain lock (a spin-wait on a flag bit) that could theoretically wait,
  though only for very brief, localized operations.

Both variants share the same per-entry concurrency protocol (the `ClockHandle`
state machine and `SlotMeta` bit-packed atomic word) but differ in table
organization.

### 1.2 Key Files

| File | Purpose |
|------|---------|
| `cache/clock_cache.h` | All data structures, class declarations, slot state machine (~1245 lines) |
| `cache/clock_cache.cc` | Implementation (~3628 lines) |
| `cache/sharded_cache.h` | Template for sharding — dispatches ops by hash to shard |
| `include/rocksdb/cache.h` | Public API: `HyperClockCacheOptions`, factory functions |
| `include/rocksdb/advanced_cache.h` | `Cache` base class with `Handle`, priority levels |
| `util/hash.h` / `util/hash.cc` | `BijectiveHash2x64` hash function (NOT inlined) |
| `util/bit_fields.h` | `BitFields` framework for atomic bit-packed operations |
| `util/atomic.h` | `RelaxedAtomic`, `Atomic` wrappers enforcing explicit memory ordering |
| `cache/cache_bench_tool.cc` | `cache_bench` benchmark (~800 lines) |
| `port/port_posix.h` | `PREFETCH` macro definition |

**BijectiveHash2x64** is a hash function that maps 128 bits (two 64-bit words) to
128 bits (two 64-bit words) in a **bijective** (one-to-one and onto) manner. This
means:
- Every unique 128-bit input produces a unique 128-bit output (no collisions)
- The mapping is reversible — given the output, you can recover the exact input
  using `BijectiveUnhash2x64`
- It has good avalanche properties (small input changes cause large output changes)

The implementation (`util/hash.cc:148-166`) is adapted from XXH3's 9-to-16 byte
hashing path. It uses two 64×64→128 bit multiplications, byte-swap, and XXH3
avalanche mixing to achieve both bijectivity and good hash distribution.

The name breaks down as: "Bijective" (lossless/reversible) + "Hash" (mixing
function) + "2x64" (operates on two 64-bit words, i.e., 128 bits total).

HyperClockCache exploits bijectivity to use the hash output AS the stored key.
Since the hash is lossless, the original 16-byte cache key can be recovered from
the stored hash whenever needed (e.g., for eviction callbacks), eliminating the
need to store the original key separately. This saves 16 bytes per cache entry.

### 1.3 Core Data Structures

#### ClockHandleBasicData (40 bytes) — `clock_cache.h:295-310`
```cpp
struct ClockHandleBasicData : public Cache::Handle {
  void* value;                    // 8B — cached object pointer
  const CacheItemHelper* helper;  // 8B — callbacks for size/save/delete
  UniqueId64x2 hashed_key;        // 16B — bijective hash IS the key
  size_t total_charge;            // 8B — charge in bytes
};
```

The bijective hash trick: instead of storing raw key + hash, a single 16-byte
bijective (lossless, reversible) hash serves as both key and hash simultaneously.
This saves 16 bytes per entry versus storing both key and hash separately.

**How bijectivity works in detail**: A normal hash function loses information —
many inputs map to the same output. `BijectiveHash2x64` avoids this by using only
reversible operations on the 128-bit input:
1. XOR with constants (reversible: XOR again to undo)
2. Multiply by odd constants (reversible: multiply by the modular inverse)
3. XOR-shift (reversible: apply the inverse shift pattern)
4. Byte-swap (reversible: swap again)

Each step is individually invertible, so the entire chain is invertible. The
`BijectiveUnhash2x64` function (`hash.cc:168-190`) applies the inverse operations
in reverse order. This is similar to how a Feistel network in cryptography
creates a bijection from invertible round functions.

The key insight for the cache: since RocksDB block cache keys are exactly 16 bytes
(encoding file identity + block offset), and the bijective hash is also 16 bytes,
you can store ONLY the hash and recover the original key on demand. A traditional
cache would need to store both the key (16B) and a hash (8B) = 24B, but HCC
stores just the hashed key (16B), saving 8-16B per entry.

#### ClockHandle::SlotMeta (64-bit atomic word) — `clock_cache.h:320-395`
```
[bits 0-29]   AcquireCounter (30 bits)
[bits 30-59]  ReleaseCounter (30 bits)
[bit  60]     HitFlag (for secondary cache)
[bit  61]     OccupiedFlag
[bit  62]     ShareableFlag
[bit  63]     VisibleFlag
```

All concurrency control state packed into one 64-bit atomic word, enabling
single-instruction atomic updates (fetch_add) for Lookup and Release. This is
the heart of the lock-free design.

- Effective refcount = `AcquireCounter - ReleaseCounter`

  **Counter monotonicity and overflow handling**: Yes, both counters only increase
  (Lookup increments AcquireCounter, Release increments ReleaseCounter). They never
  decrease during normal operation (except for "undo" on mismatch, which decrements
  AcquireCounter). With 30 bits each, they can count up to ~1 billion before
  overflowing. The `CorrectNearOverflow` function (`clock_cache.cc:212-229`)
  handles this: on every Release, it checks if the ReleaseCounter has reached
  the "high" zone (top bit set + kMaxCountdown). If so, it atomically clears the
  top bit of BOTH counters simultaneously, effectively subtracting ~500M from each.
  This preserves:
  - The refcount (difference between counters is unchanged)
  - The clock countdown state (values ≥ kMaxCountdown are all treated equivalently)
  The check is a single predictable branch (`UNLIKELY`) that almost never triggers
  in practice — an entry would need millions of Lookup/Release cycles without
  being evicted.

- Effective clock countdown = `min(kMaxCountdown, AcquireCounter)` when refs == 0

  **Design heuristic**: This is a clever dual-use of the AcquireCounter. When an
  entry has zero refs (AcquireCounter == ReleaseCounter), the AcquireCounter value
  also serves as the CLOCK eviction countdown. Here's why this works:
  - On Insert, both counters are set to `initial_countdown` (1-3 based on priority).
    Since they're equal, refcount = 0, and countdown = initial_countdown.
  - Each Lookup does `fetch_add(+1)` on AcquireCounter, incrementing the countdown
    while also adding a reference. On Release, ReleaseCounter catches up, restoring
    refcount to 0, but AcquireCounter is now higher = longer countdown.
  - The eviction sweep, when it finds an unreferenced entry with countdown > 0,
    resets both counters to `min(acquire - 1, kMaxCountdown - 1)`, effectively
    decrementing the countdown by 1.
  - After enough sweeps without any Lookup refreshing the countdown, it reaches 0
    and the entry becomes evictable.
  This means frequently-accessed entries naturally get higher countdowns (more
  "lives") without any extra atomic operations — the Lookup's fetch_add serves
  double duty as both reference acquisition AND clock refresh.



#### FixedHyperClockTable::HandleImpl (exactly 64 bytes = 1 cache line) — `clock_cache.h:589-608`
```cpp
ALIGN_AS(64U) struct HandleImpl : public ClockHandle {
  RelaxedAtomic<uint32_t> displacements{};  // probe displacement tracking
  bool standalone = false;                   // heap-allocated standalone flag
  // padding to 64 bytes
};
static_assert(sizeof(HandleImpl) == 64U);  // clock_cache.cc:737
```

**`displacements`**: In open-addressing hash tables, when you delete an entry,
  you create a "hole" that could break probe sequences for other entries that
  hashed to an earlier slot but landed in a later slot due to collisions. The
  `displacements` counter on each slot tracks how many entries exist that hash
  to this slot or an earlier slot in their probe sequence but are stored at this
  slot or a later one. When Lookup reaches a slot with `displacements == 0`, it
  knows no entry it's looking for could be stored further along — it can stop
  probing immediately. This avoids the need for tombstones (which degrade
  performance over time). On Insert, all slots along the probe path get their
  `displacements` incremented; on removal (`Rollback`), they get decremented.

**`standalone`**: When Insert cannot find a slot in the hash table (e.g., table is
  too full, or occupancy limit is reached), HCC creates a "standalone" handle
  that is heap-allocated with `new` instead of residing in the table array. This
  handle is returned to the caller but is invisible to Lookup (it won't be found
  by other threads). It exists solely to satisfy the caller's need for a handle
  reference. When Released, it is `delete`d. The `standalone` flag distinguishes
  these heap-allocated handles from in-table handles so Release knows to use
  `delete` instead of marking the slot empty.

Each entry occupies exactly one CPU cache line (64 bytes), preventing false
sharing between adjacent entries.

### 1.4 Hash Function and Probing

#### Hash Computation — `clock_cache.h:1096-1105`
```cpp
static inline HashVal ComputeHash(const Slice& key, uint32_t seed) {
    UniqueId64x2 in, out;
    std::memcpy(&in, key.data(), kCacheKeySize);  // 16 bytes
    BijectiveHash2x64(in[1], in[0] ^ seed, &out[1], &out[0]);
    return out;
}
```

**Critical observation**: `BijectiveHash2x64` is defined in `util/hash.cc:148-166`
as a **non-inline** function. It involves:
1. Two `Multiply64to128` (128-bit multiply) operations
2. An `EndianSwapValue` (byte swap)
3. Two `XXH3_avalanche` passes (each: xor-shift-37, multiply, xor-shift-32)
4. Several XOR/add operations

The no-seed overload (`hash.cc:192-195`) adds another function call to the seeded
version, so there are effectively **two function calls** in the chain:
`BijectiveHash2x64(h,l,&oh,&ol)` → `BijectiveHash2x64(h,l,0,&oh,&ol)`.

Being out-of-line means:
1. Function call overhead (save/restore registers, jump)
2. The compiler cannot optimize across the call boundary
3. Hash computation cannot be overlapped with prefetch via instruction scheduling

#### Hash Usage
- **Sharding**: `Upper32of64(hash[0])` — upper 32 bits of first word
- **Primary probe (base)**: `hash[1]` — full second word
- **Secondary probe (increment)**: `hash[0] | 1U` — forced odd for coprimality with power-of-2 table size

#### Double Hashing Probe Sequence (FixedHCC) — `clock_cache.cc:1047-1081`
```cpp
size_t base = static_cast<size_t>(hashed_key[1]);
size_t increment = static_cast<size_t>(hashed_key[0]) | 1U;
size_t first = ModTableSize(base);
size_t current = first;
do {
    HandleImpl* h = &array_[current];
    if (match_fn(h)) return h;    // match: return
    if (abort_fn(h)) return nullptr; // displacements==0: stop
    current = ModTableSize(current + increment);
    is_last = current == first;
    update_fn(h, is_last);
} while (!is_last);
```

The increment is forced odd, guaranteeing every slot is visited exactly once
before cycling. Average probes at load factor 0.7 ≈ 1.43 (double hashing).

### 1.5 Slot State Machine

```
Empty → Construction   (Insert: atomic OR of OccupiedFlag)
Construction → Visible (Insert: store, exclusive ownership)
Visible → Invisible    (Erase: idempotent AND to clear VisibleFlag)
Shareable → Construction (Evict/Erase: CAS when refcount==0)
Construction → Empty   (Free: release store after freeing data)
```

### 1.6 Clock Eviction Algorithm — `clock_cache.cc:1104-1148`

Multiple threads evict in parallel. Each thread atomically advances `clock_pointer_`
by `step_size=4`, then processes those 4 slots:

```cpp
constexpr size_t step_size = 4;
uint64_t old_clock_pointer = clock_pointer_.FetchAddRelaxed(step_size);
uint64_t max_clock_pointer = old_clock_pointer + (kMaxCountdown << length_bits_);

for (;;) {
    for (size_t i = 0; i < step_size; i++) {
        HandleImpl& h = array_[ModTableSize(Lower32of64(old_clock_pointer + i))];
        bool evicting = ClockUpdate(h, data);
        if (evicting) {
            Rollback(h.hashed_key, &h);
            TrackAndReleaseEvictedEntry(&h);
        }
    }
    if (data->freed_charge >= requested_charge) return;
    if (old_clock_pointer >= max_clock_pointer) return;
    if (IsEvictionEffortExceeded(*data)) { ... return; }
    old_clock_pointer = clock_pointer_.FetchAddRelaxed(step_size);
}
```

`ClockUpdate` (`clock_cache.cc:97-156`) does:
1. Load meta (relaxed for FixedHCC, acquire for AutoHCC)
2. Skip non-shareable entries
3. Skip referenced entries (acquire_count != release_count) → `seen_pinned_count++`
4. If visible with countdown > 0: decrement countdown by 1 via CAS.
   Specifically, it sets both AcquireCounter and ReleaseCounter to
   `min(acquire_count - 1, kMaxCountdown - 1)`. This effectively reduces the
   countdown by 1 while keeping the refcount at 0. The `min` with
   `kMaxCountdown - 1` caps the countdown so that very hot entries don't
   accumulate unbounded countdown values. (`clock_cache.cc:131-138`)
5. If countdown == 0 or invisible: take ownership via CAS → evict

Countdown values by priority:
- HIGH: initial countdown = 3 (kHighCountdown)
- LOW: initial countdown = 2 (kLowCountdown)
- BOTTOM: initial countdown = 1 (kBottomCountdown)

### 1.7 Cache Line Separation for False Sharing Prevention

`BaseClockTable` (`clock_cache.h:519-578`) uses `ALIGN_AS(CACHE_LINE_SIZE)` to
separate hot data into distinct cache lines:

- **Group 1** (eviction hot): `clock_pointer_`, `yield_count_`, `eviction_effort_exceeded_count_`
- **Group 2** (insert/capacity hot): `occupancy_`, `usage_`, `standalone_usage_`, `capacity_`, `eec_and_scl_`
- **Group 3** (cold/read-only): `metadata_charge_policy_`, `allocator_`, `eviction_callback_`, `hash_seed_`

## 2. Hot Path Analysis

### 2.1 Lookup — THE Primary Hot Path

Full call chain:

```
ShardedCache::Lookup(key)                          [sharded_cache.h:198-206]
  1. hash = ComputeHash(key, seed)                 // BijectiveHash2x64 (OUT OF LINE)
  2. shard = GetShard(hash)                        // Upper32of64(hash[0]) & mask
  3. return shard.Lookup(key, hash)

ClockCacheShard::Lookup(key, hash)                 [clock_cache.cc:1274-1280]
  4. if (key.size() != 16) return null             // UNLIKELY branch
  5. return table_.Lookup(hash)

FixedHyperClockTable::Lookup(hashed_key)           [clock_cache.cc:827-886]
  6. FindSlot(hashed_key, match_fn, abort_fn, update_fn)
     For each slot in probe sequence:
       a. fetch_add(+1) on AcquireCounter          // 1 atomic (acq_rel)
       b. Check old_meta.IsVisible()               // branch on data in register
       c. Compare hashed_key (16 bytes)            // 2x 64-bit compare
       d. If match:
          - set HitFlag (1 relaxed atomic if eviction_callback_)
          - return handle
       e. If mismatch:
          - Unref: fetch_sub(-1) on AcquireCounter // 1 atomic (acq_rel)
       f. Check displacements == 0 for abort       // relaxed load
```

**Performance profile for first-probe hit (common case at load factor 0.7):**
- 1 out-of-line function call for BijectiveHash2x64 (~20-30 cycles compute + call overhead)
  **Call overhead breakdown**: A non-inline function call on x86-64 typically costs
  ~5-10 cycles: saving caller-saved registers to the stack (~2-3 cycles), the
  `call` instruction itself (~1 cycle, branch predictor usually predicts correctly
  for direct calls), the function prologue (stack frame setup, ~1-2 cycles), and
  the `ret` + epilogue (~1-2 cycles). Additionally, there's an indirect cost:
  the compiler cannot reorder instructions across the call boundary, so it cannot
  interleave the hash computation with a prefetch instruction that we'd like to
  issue early. For BijectiveHash2x64 specifically, the no-seed overload adds a
  second function call to the seeded version, doubling this overhead (~10-20
  cycles total call overhead on top of the ~20 cycles of actual hash computation).
- 1 cache line load for the slot (64 bytes) — **often an L2/L3 miss (~10-50ns)**
- 1 atomic fetch_add (acq_rel) for acquiring reference (~5-20ns)
- 1 16-byte key comparison (~1-2 cycles)
- 0-1 relaxed atomic for hit flag
- Total: dominated by **cache miss latency + hash computation**

### 2.2 Release — Second Most Frequent Hot Path

Common case (`useful=true`, not `erase_if_last_ref`):
```cpp
h->meta.Apply(ReleaseCounter::PlusTransformPromiseNoOverflow(1), &old_meta);
// CorrectNearOverflow check (almost never taken — highly predictable branch)
```

This is **1 atomic fetch_add** + 1 predictable branch. Already near-optimal.

### 2.3 Insert — Relatively Cold Path

Insert involves: optimistic occupancy increment, capacity check, potential
eviction, slot finding (FindSlot for FixedHCC), data placement. More expensive
than Lookup but called less frequently (block cache inserts happen on cache miss,
which is by definition less common than cache hit).

### 2.4 Eviction — Triggered During Insert

Parallel sweep with `step_size=4`. Each step processes 4 slots with
sequentially-increasing indices. Slots are consecutive in memory, so 4 consecutive
cache lines are accessed. No prefetching of the next batch is done.

## 3. Existing Performance Optimizations

1. **Single-atomic hot path**: Lookup and Release each = 1 `fetch_add` by packing
   all state into one 64-bit word
2. **Cache-line-aligned entries**: 64 bytes per slot, `ALIGN_AS(64U)`, no false sharing
3. **Bijective hash as key**: No separate key storage needed (16B savings per entry)
4. **Optimistic Lookup** (`clock_cache.cc:847`): Acquires ref before checking visibility,
   saving one atomic load in the common case (entry is visible)
5. **Speculative key match (AutoHCC)** (`clock_cache.cc:3120`): Reads key without
   holding ref first — data race but safe (false positives caught, false negatives
   fall through)
6. **Template-based FindSlot** (`clock_cache.h:707-710`): Uses templates instead of
   `std::function` to avoid heap-allocated closures and enable inlining
7. **Parallel eviction**: Multiple threads evict via atomic clock pointer increment
8. **Relaxed atomics**: Used where cross-thread synchronization is not needed
   (clock_pointer, usage, occupancy, displacements)
9. **Cache line separation in BaseClockTable**: Hot data groups separated with ALIGN_AS
10. **Overflow correction**: `CorrectNearOverflow` is highly predictable (almost never
    triggered), adding negligible branch cost

## 4. Identified Acceleration Opportunities

### 4.1 No Prefetching Anywhere in Cache Code (HIGH IMPACT)

**Current state**: Zero `PREFETCH` calls anywhere in `cache/clock_cache.cc` or
`cache/clock_cache.h`. The PREFETCH macro exists and is used extensively in other
RocksDB components:
- `memtable/inlineskiplist.h` — prefetch next node during traversal
- `table/cuckoo/cuckoo_table_reader.cc` — prefetch bucket range
- `util/ribbon_impl.h` — prefetch during filter construction
- `util/bloom_impl.h` — prefetch during bloom filter probing
- `util/dynamic_bloom.h` — prefetch during dynamic bloom operations

**Opportunity**: The dominant cost on the Lookup hot path for large caches is the
**cache line miss when accessing the slot** (`array_[ModTableSize(hash[1])]`).
This miss typically costs 10-50ns (L3 hit) or 100-300ns (DRAM access). The slot
address is computable from the hash result, but no prefetch is issued.

Adding `PREFETCH` at strategic points could overlap memory latency with
computation:

**A. Prefetch first probe slot early in Lookup**
After computing the hash and before entering the table Lookup function, the
first probe address is known. Issuing a prefetch here overlaps the memory fetch
with function call overhead and remaining setup.

**B. Prefetch next probe slot during probe loop**
When processing probe `i` and it's a miss, the address of probe `i+1` is
immediately computable. Prefetching it overlaps the fetch with the Unref
atomic operation on the current slot.

**C. Prefetch next batch in eviction sweep**
While processing the current batch of `step_size=4` slots, prefetch the next
batch. The addresses are sequential and predictable.

**D. Prefetch chain next in AutoHCC naive Lookup**
Follow the `inlineskiplist.h` pattern: prefetch the next chain entry while
processing the current one.

**Safety**: Prefetch instructions are pure CPU hints. They cannot cause faults
(even with invalid addresses), have no ordering constraints, and have no semantic
effect. They are safe to add anywhere.

**Expected impact**: For cache-cold accesses (the common case with multi-GB
caches), hiding even half the memory latency per Lookup translates to
significant throughput improvement. Bloom filter prefetching (a comparable
optimization in RocksDB) is known to provide 5-15% improvement.

### 4.2 BijectiveHash2x64 is Out-of-Line (MEDIUM IMPACT)

**Current state**: `BijectiveHash2x64` is defined in `util/hash.cc:148-166`.
The `util/hash.h` header comment says: *"implementation details are kept
out-of-line. Out-of-lining also aids in tracking the time spent in hashing
functions. Inlining is of limited benefit for runtime-sized hash inputs."*

However, HyperClockCache uses a **fixed-size 16-byte input**, which is different
from the general-purpose hash use case. The hash comment's reasoning about
"runtime-sized" doesn't apply here. Additionally, the no-seed overload adds
an extra function call (calls the seeded version with seed=0).

**Opportunity**: Create an inline version for the cache's fixed-size use case.
Benefits:
- Eliminates ~5-15 cycles of function call overhead per Lookup
- Enables the compiler to interleave hash computation with prefetch instructions
- Enables better register allocation across the hash+lookup boundary

**Approach**: Move `BijectiveHash2x64` implementation to `util/hash.h` as an
inline function (or create an inline variant). The internal helpers
`XXH3_avalanche`, `Multiply64to128`, `EndianSwapValue` are already inline in
their respective headers (`hash.cc`, `math128.h`, `math.h`).

### 4.3 Eviction Sweep Prefetching (MEDIUM IMPACT)

**Current state**: Eviction processes `step_size=4` consecutive slots per batch.
No prefetching of the next batch is done.

**Opportunity**: Prefetch the next `step_size` slots while processing the current
batch. Since the clock pointer advances sequentially, the next positions are
trivially predictable. This is especially valuable during heavy Insert workloads
that trigger significant eviction.

### 4.4 Probe Sequence Prefetching on Miss (MEDIUM IMPACT for high-collision scenarios)

**Current state**: On probe miss, the next slot address is computed but not
prefetched. Since each slot is a full cache line and the double-hash increment
is pseudo-random, probes beyond the first have poor spatial locality.

**Opportunity**: Prefetch the next probe slot while doing the Unref atomic
operation on the mismatched current slot. At load factor 0.7, ~30% of lookups
need a second probe, and prefetching could hide most of that miss penalty.

### 4.5 Eviction Step Size Tuning (LOW IMPACT)

**Current state**: `step_size = 4` is hardcoded with a TODO comment.

**Opportunity**: With prefetching, a larger step size (e.g., 8) may be more
efficient on modern hardware with wider memory pipelines, as it amortizes
the clock pointer atomic update cost over more slots.

## 5. Concurrency Patterns and Constraints

### 5.1 Memory Ordering Requirements

| Operation | Atomic op | Order | Can be relaxed? |
|-----------|-----------|-------|-----------------|
| Lookup acquire | fetch_add on AcquireCounter | acq_rel | No |
| Release | fetch_add on ReleaseCounter | acq_rel | No |
| Hit flag update | fetch_or on HitFlag | relaxed | Already optimal |
| Displacements check | load on displacements | relaxed | Already optimal |
| Eviction meta load | load on meta | relaxed (FixedHCC) | Already optimal |
| Clock pointer advance | fetch_add | relaxed | Already optimal |

The acq_rel ordering on Lookup/Release is fundamental to correctness.

### 5.2 Safe Points for Adding Prefetch

Prefetch instructions are pure hints with no semantic effect. They:
- Cannot cause faults (even with invalid/null addresses)
- Have no ordering constraints
- Can be inserted freely at any point
- Are a no-op on Windows (the PREFETCH macro is empty on port_win.h)

Optimal placement: as early as possible before the data is needed, to maximize
overlap between prefetch latency and other computation.

### 5.3 Shard Selection After Hash

After `ComputeHash`:
- `hash[0]` → upper 32 bits for sharding, full word for probe increment
- `hash[1]` → primary probe location

Both are available immediately. The shard pointer and the first slot address
can both be computed and prefetched before entering the table Lookup.

## 6. Related Patterns in RocksDB

### 6.1 PREFETCH Usage Examples

**inlineskiplist.h** (most relevant pattern):
```cpp
Node* next = x->Next(level);
if (next != nullptr) {
    PREFETCH(next->Next(level), 0, 1);
}
```
This prefetches one step ahead during linked-list traversal — directly
analogous to what we'd do in AutoHCC chain traversal.

**bloom_impl.h** (filter prefetch):
```cpp
PREFETCH(data + byte_offset, 0, 1);
```
Prefetches the cache line for bloom filter probing — analogous to prefetching
the hash table slot.

Convention: `PREFETCH(addr, 0 /*read*/, 1 /*moderate locality*/)` is most common.

### 6.2 Inline Hash Precedent

`XXH3_avalanche` and `XXH3_unavalanche` in `hash.cc` are already `inline` within
the translation unit. `Multiply64to128` in `math128.h` is already inline.
`EndianSwapValue` in `math.h` is already inline. So all the building blocks of
`BijectiveHash2x64` are already inline — only the top-level function is out-of-line.

## 7. Benchmark Infrastructure

**`cache/cache_bench_tool.cc`** — Dedicated cache microbenchmark:
- Build: `DEBUG_LEVEL=0 make cache_bench`
- Run: `./cache_bench --cache_type=fixed_hyper_clock_cache --threads=16`
- Key flags:
  - `--cache_size` (default 1GB)
  - `--threads` (default 16)
  - `--value_bytes` (default 8KB)
  - `--ops_per_thread` (default 2M)
  - `--resident_ratio` (default 0.25)
  - `--skew` (default 5, 0=uniform)
  - `--lookup_insert_percent` (default 82)
  - `--lookup_percent` (default 10)
- Reports ops/sec/thread and hit ratio

## 8. Potential Pitfalls and Constraints

1. **Prefetch address validity**: For FixedHCC, `array_` is a contiguous allocation
   sized to `1 << length_bits_`, so any `ModTableSize` result maps to a valid slot.
   For AutoHCC, slots within the current table size are valid.

2. **Inlining BijectiveHash2x64**: The `hash.h` header comment mentions out-of-lining
   aids profiling. An inline version should supplement (not replace) the existing
   out-of-line version, so profiling still works for other callers.

3. **Platform compatibility**: `PREFETCH` is defined for all platforms (no-op on
   Windows). `__builtin_prefetch` is available on GCC and Clang. ARM mapping
   in `port_posix.h` remaps locality for ARM's different prefetch semantics.

4. **Measurement requirements**: All optimizations must be validated with `cache_bench`.
   Build release binary: `make clean && DEBUG_LEVEL=0 make cache_bench`.

5. **Code complexity**: Prefetching adds minimal code (1-2 lines per site). Inlining
   the hash is slightly more involved but follows established patterns.

6. **AutoHCC vs FixedHCC**: Prefetching opportunities differ. FixedHCC has
   predictable slot addresses (array index). AutoHCC requires following chain
   pointers, making prefetch more complex but still beneficial.

## 9. Summary of Acceleration Opportunities (Ranked)

| # | Optimization | Impact | Complexity | Risk |
|---|-------------|--------|------------|------|
| 1 | Prefetch first probe slot in Lookup | HIGH | LOW | NONE |
| 2 | Prefetch next probe slot on miss (FindSlot) | HIGH | LOW | NONE |
| 3 | Inline BijectiveHash2x64 for cache | MEDIUM | LOW-MED | LOW |
| 4 | Prefetch in eviction sweep | MEDIUM | LOW | NONE |
| 5 | Prefetch chain next in AutoHCC Lookup | MEDIUM | LOW | NONE |
| 6 | Eviction step size increase (with prefetching) | LOW | LOW | LOW |

All proposed changes are **correctness-neutral** — prefetch instructions are pure
CPU hints that do not change data flow, synchronization, or algorithmic behavior.
The inline hash optimization produces identical results to the out-of-line version.
