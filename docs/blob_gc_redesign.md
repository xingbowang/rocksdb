# BlobDB GC Redesign: Per-File Garbage-Aware Collection

## Overview

This document proposes a redesign of the BlobDB garbage collection mechanism
to target individual blob files with high garbage ratios, rather than using
age-based batch selection.

## Current Design vs Proposed Design

### Current Design (Age-Based Batch)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CURRENT GC DESIGN                                │
└─────────────────────────────────────────────────────────────────────────┘

                    Blob Files (sorted by age, oldest first)
    ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
    │ File 1  │ File 2  │ File 3  │ File 4  │ File 5  │ File 6  │
    │ OLDEST  │         │         │         │         │ NEWEST  │
    │ 30% gar │ 10% gar │ 80% gar │ 90% gar │ 5% gar  │ 95% gar │
    └────┬────┴────┬────┴─────────┴─────────┴─────────┴─────────┘
         │         │
         └────┬────┘
              │
              ▼
    ┌─────────────────────┐
    │  Age Cutoff = 25%   │  (Only oldest 2 files eligible)
    │  Eligible: [1, 2]   │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────────────────┐
    │  Aggregate Garbage Calculation  │
    │  (30% + 10%) / 2 = 20%         │
    └──────────┬──────────────────────┘
               │
               ▼
    ┌─────────────────────────────────┐
    │  Force Threshold = 50%          │
    │  20% < 50% → NO GC TRIGGERED   │
    └─────────────────────────────────┘

    ❌ PROBLEM: Files 3,4,6 have 80-95% garbage but are IGNORED!
```

### Proposed Design (Per-File Garbage Targeting)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       PROPOSED GC DESIGN                                │
└─────────────────────────────────────────────────────────────────────────┘

                    Blob Files (any order)
    ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
    │ File 1  │ File 2  │ File 3  │ File 4  │ File 5  │ File 6  │
    │ 30% gar │ 10% gar │ 80% gar │ 90% gar │ 5% gar  │ 95% gar │
    │ 100MB   │ 200MB   │ 150MB   │ 80MB    │ 50MB    │ 120MB   │
    └─────────┴─────────┴────┬────┴────┬────┴─────────┴────┬────┘
                             │         │                   │
                             ▼         ▼                   ▼
    ┌────────────────────────────────────────────────────────────┐
    │         Per-File Garbage Ratio Check (threshold=50%)       │
    │                                                            │
    │  File 1: 30% < 50%  ❌ Skip                                │
    │  File 2: 10% < 50%  ❌ Skip                                │
    │  File 3: 80% >= 50% ✓ Candidate (120MB garbage)           │
    │  File 4: 90% >= 50% ✓ Candidate (72MB garbage)            │
    │  File 5: 5% < 50%   ❌ Skip                                │
    │  File 6: 95% >= 50% ✓ Candidate (114MB garbage)           │
    └────────────────────────────────────────────────────────────┘
                             │
                             ▼
    ┌────────────────────────────────────────────────────────────┐
    │              Priority Ranking (by garbage ratio)           │
    │                                                            │
    │  1. File 6: 95% garbage (highest ratio)                   │
    │  2. File 4: 90% garbage                                   │
    │  3. File 3: 80% garbage                                   │
    └────────────────────────────────────────────────────────────┘
                             │
                             ▼
    ┌────────────────────────────────────────────────────────────┐
    │         Select Top N Files (max_blob_files_per_gc=2)       │
    │                                                            │
    │  Selected: [File 6, File 4]                                │
    └────────────────────────────────────────────────────────────┘
                             │
                             ▼
    ┌────────────────────────────────────────────────────────────┐
    │            Find SSTs Referencing Selected Files            │
    │                                                            │
    │  File 6 → SSTs: [S10, S15, S20]                           │
    │  File 4 → SSTs: [S12, S18]                                │
    │                                                            │
    │  Mark for Compaction: [S10, S12, S15, S18, S20]           │
    └────────────────────────────────────────────────────────────┘

    ✓ Files with highest garbage are targeted regardless of age!
```

## Detailed Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    GARBAGE-AWARE GC ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│                         GARBAGE TRACKING LAYER                            │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                      BlobFileMetaData                               │  │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐│  │
│  │  │ total_blob_    │  │ garbage_blob_  │  │ garbage_ratio =       ││  │
│  │  │ bytes          │  │ bytes          │  │ garbage_bytes/total   ││  │
│  │  └────────────────┘  └────────────────┘  └────────────────────────┘│  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                         GC CANDIDATE SELECTOR                             │
│                    (NEW: BlobGCCandidateSelector class)                   │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  Input: All blob files from VersionStorageInfo                      │  │
│  │                                                                     │  │
│  │  Algorithm:                                                         │  │
│  │  1. Filter: garbage_ratio >= blob_file_garbage_threshold            │  │
│  │  2. Filter: file_size >= min_blob_file_size_for_gc                  │  │
│  │  3. Sort by priority (ratio, bytes, or age)                         │  │
│  │  4. Select top N files (max_blob_files_per_gc)                      │  │
│  │                                                                     │  │
│  │  Output: List of blob files marked for GC                           │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                        SST REFERENCE RESOLVER                             │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  For each selected blob file:                                       │  │
│  │    1. Get linked_ssts from BlobFileMetaData                         │  │
│  │    2. Lookup SST FileMetaData from VersionStorageInfo               │  │
│  │    3. Add to files_marked_for_forced_blob_gc_                       │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                        COMPACTION PICKER                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  NeedsCompaction() checks:                                          │  │
│  │    - !files_marked_for_forced_blob_gc_.empty()                      │  │
│  │                                                                     │  │
│  │  PickCompaction() selects:                                          │  │
│  │    - SSTs from files_marked_for_forced_blob_gc_                     │  │
│  │    - Sets CompactionReason::kForcedBlobGC                           │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                        COMPACTION JOB                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  During compaction:                                                 │  │
│  │    1. Read blobs from old files                                     │  │
│  │    2. Write live blobs to new blob file                             │  │
│  │    3. Update SST blob references                                    │  │
│  │    4. Track garbage via BlobGarbageMeter                            │  │
│  │    5. Mark old blob files for deletion                              │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────┘
```

## Component Changes

### 1. New Class: BlobGCCandidateSelector

```cpp
// db/blob/blob_gc_candidate_selector.h

class BlobGCCandidateSelector {
 public:
  struct Candidate {
    uint64_t blob_file_number;
    double garbage_ratio;
    uint64_t garbage_bytes;
    uint64_t total_bytes;
  };

  explicit BlobGCCandidateSelector(const AdvancedColumnFamilyOptions& opts);

  // Select blob files that exceed garbage threshold
  std::vector<Candidate> SelectCandidates(
      const std::map<uint64_t, std::shared_ptr<BlobFileMetaData>>& blob_files,
      size_t max_candidates) const;

 private:
  double garbage_threshold_;
  uint64_t min_file_size_;
  GCPriority priority_;
};
```

### 2. Modified: VersionStorageInfo

```cpp
// In version_set.h

class VersionStorageInfo {
  // Replace existing method
  void ComputeFilesMarkedForForcedBlobGC();

  // NEW: Per-file garbage tracking
  std::vector<uint64_t> GetHighGarbageBlobFiles(double threshold) const;

  // NEW: Statistics
  uint64_t GetTotalBlobGarbageBytes() const;
  double GetAverageBlobGarbageRatio() const;
};
```

### 3. Modified: BlobFileMetaData

```cpp
// In blob_file_meta.h

class BlobFileMetaData {
  // NEW: Computed property
  double GetGarbageRatio() const {
    if (GetTotalBlobBytes() == 0) return 0.0;
    return static_cast<double>(GetGarbageBlobBytes()) /
           static_cast<double>(GetTotalBlobBytes());
  }

  // NEW: Check if file is a GC candidate
  bool IsGCCandidate(double threshold, uint64_t min_size) const {
    return GetTotalBlobBytes() >= min_size &&
           GetGarbageRatio() >= threshold;
  }
};
```

## Algorithm: SelectCandidates()

```cpp
std::vector<Candidate> BlobGCCandidateSelector::SelectCandidates(
    const BlobFiles& blob_files, size_t max_candidates) const {

  std::vector<Candidate> candidates;

  // Step 1: Filter files that exceed threshold
  for (const auto& [file_num, meta] : blob_files) {
    double ratio = meta->GetGarbageRatio();
    uint64_t size = meta->GetTotalBlobBytes();

    if (ratio >= garbage_threshold_ && size >= min_file_size_) {
      candidates.push_back({
        .blob_file_number = file_num,
        .garbage_ratio = ratio,
        .garbage_bytes = meta->GetGarbageBlobBytes(),
        .total_bytes = size,
      });
    }
  }

  // Step 2: Sort by priority
  switch (priority_) {
    case GCPriority::HIGHEST_GARBAGE_RATIO:
      std::sort(candidates.begin(), candidates.end(),
                [](const auto& a, const auto& b) {
                  return a.garbage_ratio > b.garbage_ratio;
                });
      break;
    case GCPriority::LARGEST_GARBAGE_BYTES:
      std::sort(candidates.begin(), candidates.end(),
                [](const auto& a, const auto& b) {
                  return a.garbage_bytes > b.garbage_bytes;
                });
      break;
    case GCPriority::OLDEST_FIRST:
      std::sort(candidates.begin(), candidates.end(),
                [](const auto& a, const auto& b) {
                  return a.blob_file_number < b.blob_file_number;
                });
      break;
  }

  // Step 3: Limit to max candidates
  if (candidates.size() > max_candidates) {
    candidates.resize(max_candidates);
  }

  return candidates;
}
```

## Configuration Migration

### Old Configuration (Deprecated)
```cpp
options.enable_blob_garbage_collection = true;
options.blob_garbage_collection_age_cutoff = 0.25;        // DEPRECATED
options.blob_garbage_collection_force_threshold = 0.5;     // DEPRECATED
```

### New Configuration
```cpp
options.enable_blob_garbage_collection = true;
options.blob_file_garbage_threshold = 0.5;          // 50% garbage triggers GC
options.min_blob_file_size_for_gc = 64 << 20;       // 64MB minimum
options.max_blob_files_per_gc = 4;                  // Limit per round
options.blob_gc_priority = GCPriority::HIGHEST_GARBAGE_RATIO;
```

## Comparison: Old vs New

| Aspect | Old Design | New Design |
|--------|------------|------------|
| **Selection Criteria** | Age-based (oldest N%) | Garbage ratio-based |
| **Threshold Type** | Aggregate across batch | Per-file individual |
| **Files Considered** | Only oldest batch | All files |
| **Write Amplification Control** | Implicit (batch size) | Explicit (max_files_per_gc) |
| **Priority** | Always oldest | Configurable (ratio, bytes, age) |
| **Minimum Size Filter** | None | Configurable |

## Benefits

1. **Targeted GC**: High-garbage files are collected regardless of age
2. **Efficient Space Reclamation**: Prioritize files with most garbage
3. **Controlled Write Amplification**: Limit files per GC round
4. **Flexible Prioritization**: Choose what matters most for your workload
5. **Avoid Thrashing**: Minimum file size prevents GC of small files

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          COMPLETE GC DATA FLOW                              │
└─────────────────────────────────────────────────────────────────────────────┘

  [User Writes]
       │
       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  MemTable    │────▶│   Flush      │────▶│  SST File    │
└──────────────┘     └──────────────┘     │  (L0)        │
                                          │  blob_refs[] │
                                          └──────┬───────┘
                                                 │
       ┌─────────────────────────────────────────┼─────────────────────────┐
       │                                         │                         │
       ▼                                         ▼                         ▼
┌──────────────┐                          ┌──────────────┐          ┌──────────────┐
│  Blob File   │                          │  Blob File   │          │  Blob File   │
│  #1          │                          │  #2          │          │  #3          │
│  ┌─────────┐ │                          │  ┌─────────┐ │          │  ┌─────────┐ │
│  │total:   │ │                          │  │total:   │ │          │  │total:   │ │
│  │ 100MB   │ │                          │  │ 200MB   │ │          │  │ 150MB   │ │
│  │garbage: │ │                          │  │garbage: │ │          │  │garbage: │ │
│  │ 30MB    │ │                          │  │ 20MB    │ │          │  │ 120MB   │ │
│  │ratio:   │ │                          │  │ratio:   │ │          │  │ratio:   │ │
│  │ 30%     │ │                          │  │ 10%     │ │          │  │ 80%  ⚠️ │ │
│  └─────────┘ │                          │  └─────────┘ │          │  └─────────┘ │
└──────────────┘                          └──────────────┘          └──────┬───────┘
       │                                         │                         │
       └─────────────────────────────────────────┼─────────────────────────┘
                                                 │
                                                 ▼
                              ┌──────────────────────────────────┐
                              │   BlobGCCandidateSelector        │
                              │                                  │
                              │   threshold = 50%                │
                              │                                  │
                              │   File #1: 30% < 50%  ❌         │
                              │   File #2: 10% < 50%  ❌         │
                              │   File #3: 80% >= 50% ✅         │
                              │                                  │
                              │   Selected: [File #3]            │
                              └───────────────┬──────────────────┘
                                              │
                                              ▼
                              ┌──────────────────────────────────┐
                              │   SST Reference Lookup           │
                              │                                  │
                              │   File #3.linked_ssts = [S5, S8] │
                              │                                  │
                              │   Mark S5, S8 for compaction     │
                              └───────────────┬──────────────────┘
                                              │
                                              ▼
                              ┌──────────────────────────────────┐
                              │   Compaction Job                 │
                              │                                  │
                              │   1. Compact S5 + S8             │
                              │   2. Relocate live blobs         │
                              │   3. Create new SST S12          │
                              │   4. Create new Blob File #4     │
                              │   5. Delete old Blob File #3     │
                              └──────────────────────────────────┘
                                              │
                                              ▼
                              ┌──────────────────────────────────┐
                              │   Result                         │
                              │                                  │
                              │   Blob File #3 deleted           │
                              │   Blob File #4 created           │
                              │   (contains only live blobs)     │
                              │   Space reclaimed: 120MB         │
                              └──────────────────────────────────┘
```

## State Transition Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     BLOB FILE STATE TRANSITIONS                             │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌───────────────────────────────────────┐
                    │                                       │
                    ▼                                       │
           ┌───────────────┐                               │
           │   CREATED     │                               │
           │   ratio = 0%  │                               │
           └───────┬───────┘                               │
                   │                                       │
                   │ User deletes/updates blobs            │
                   ▼                                       │
           ┌───────────────┐                               │
           │   ACTIVE      │                               │
           │   ratio < 50% │◀──────────────────────────────┤
           └───────┬───────┘                               │
                   │                                       │
                   │ More garbage accumulates              │
                   ▼                                       │
           ┌───────────────┐                               │
           │   HIGH_GARBAGE│                               │
           │   ratio >= 50%│                               │
           └───────┬───────┘                               │
                   │                                       │
                   │ GC selector picks this file           │
                   ▼                                       │
           ┌───────────────┐                               │
           │   MARKED_GC   │                               │
           │   Pending     │                               │
           │   compaction  │                               │
           └───────┬───────┘                               │
                   │                                       │
                   │ Compaction runs                       │
                   ▼                                       │
           ┌───────────────┐    Live blobs    ┌───────────────┐
           │   COMPACTING  │ ───relocated───▶ │  NEW FILE     │
           │               │                  │  (CREATED)    │
           └───────┬───────┘                  └───────────────┘
                   │
                   │ All blobs processed
                   ▼
           ┌───────────────┐
           │   OBSOLETE    │
           │   Can delete  │
           └───────┬───────┘
                   │
                   │ File deletion
                   ▼
           ┌───────────────┐
           │   DELETED     │
           └───────────────┘
```

## Implementation Phases

### Phase 1: Core Infrastructure
- Add `GetGarbageRatio()` to BlobFileMetaData
- Implement BlobGCCandidateSelector class
- Add new configuration options

### Phase 2: Integration
- Modify `ComputeFilesMarkedForForcedBlobGC()` to use new selector
- Update compaction picker integration
- Add metrics/statistics

### Phase 3: Testing & Migration
- Add unit tests for new selector
- Add integration tests for GC behavior
- Deprecate old configuration options
- Write migration guide

## Metrics to Add

```cpp
// New statistics for monitoring
BLOB_GC_CANDIDATES_FOUND,       // Files exceeding threshold
BLOB_GC_FILES_COLLECTED,        // Files actually GC'd
BLOB_GC_BYTES_RECLAIMED,        // Space reclaimed
BLOB_GC_BYTES_RELOCATED,        // Live data moved
BLOB_GC_WRITE_AMPLIFICATION,    // Ratio of relocated/reclaimed
```

## Example: Before and After

### Scenario
- 6 blob files with varying garbage ratios
- Threshold = 50%
- Max files per GC = 2

### Old Behavior (age_cutoff = 0.25, force_threshold = 0.5)
```
Files by age: [F1:30%, F2:10%, F3:80%, F4:90%, F5:5%, F6:95%]
Eligible (oldest 25%): [F1, F2]  (only 1-2 files)
Aggregate ratio: (30% + 10%) / 2 = 20%
Result: 20% < 50% → NO GC

❌ Files F3, F4, F6 have 80-95% garbage but are IGNORED!
```

### New Behavior (threshold = 50%, max_files = 2)
```
Check all files:
  F1: 30% < 50% → Skip
  F2: 10% < 50% → Skip
  F3: 80% >= 50% → Candidate ✓
  F4: 90% >= 50% → Candidate ✓
  F5: 5% < 50% → Skip
  F6: 95% >= 50% → Candidate ✓

Sort by ratio: [F6:95%, F4:90%, F3:80%]
Select top 2: [F6, F4]

Result: GC Files F6 and F4 (highest garbage)
✓ Most garbage-heavy files are collected!
```
