//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include "rocksdb/advanced_options.h"
#include "rocksdb/rocksdb_namespace.h"

namespace ROCKSDB_NAMESPACE {

class BlobFileMetaData;

// Represents a blob file that is a candidate for garbage collection.
struct BlobGCCandidate {
  uint64_t blob_file_number;
  double garbage_ratio;
  uint64_t garbage_bytes;
  uint64_t total_bytes;
};

// Selects blob files for garbage collection based on per-file garbage metrics.
// This class replaces age-based batch selection with targeted, garbage-aware
// selection.
class BlobGCCandidateSelector {
 public:
  using BlobFiles = std::map<uint64_t, std::shared_ptr<BlobFileMetaData>>;

  // Creates a selector with the given parameters.
  // threshold: minimum garbage ratio for a file to be considered (0.0 to 1.0)
  // min_size: minimum total blob bytes for a file to be considered
  // max_candidates: maximum number of candidates to return
  // priority: determines the order in which candidates are returned
  BlobGCCandidateSelector(double threshold, uint64_t min_size,
                          size_t max_candidates, BlobGCPriority priority);

  // Selects blob files for garbage collection from the given blob files.
  // Returns candidates sorted according to the configured priority,
  // limited to max_candidates.
  std::vector<BlobGCCandidate> SelectCandidates(
      const BlobFiles& blob_files) const;

 private:
  double threshold_;
  uint64_t min_size_;
  size_t max_candidates_;
  BlobGCPriority priority_;
};

}  // namespace ROCKSDB_NAMESPACE
