//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "db/blob/blob_gc_candidate_selector.h"

#include <algorithm>

#include "db/blob/blob_file_meta.h"

namespace ROCKSDB_NAMESPACE {

BlobGCCandidateSelector::BlobGCCandidateSelector(double threshold,
                                                 uint64_t min_size,
                                                 size_t max_candidates,
                                                 BlobGCPriority priority)
    : threshold_(threshold),
      min_size_(min_size),
      max_candidates_(max_candidates),
      priority_(priority) {}

std::vector<BlobGCCandidate> BlobGCCandidateSelector::SelectCandidates(
    const BlobFiles& blob_files) const {
  std::vector<BlobGCCandidate> candidates;
  candidates.reserve(blob_files.size());

  // Filter blob files that meet the threshold and min size criteria
  for (const auto& [file_number, meta] : blob_files) {
    if (meta && meta->IsGCCandidate(threshold_, min_size_)) {
      candidates.push_back({file_number, meta->GetGarbageRatio(),
                            meta->GetGarbageBlobBytes(),
                            meta->GetTotalBlobBytes()});
    }
  }

  // Sort candidates based on priority
  switch (priority_) {
    case BlobGCPriority::kHighestGarbageRatio:
      std::sort(candidates.begin(), candidates.end(),
                [](const BlobGCCandidate& a, const BlobGCCandidate& b) {
                  // Higher garbage ratio comes first
                  if (a.garbage_ratio != b.garbage_ratio) {
                    return a.garbage_ratio > b.garbage_ratio;
                  }
                  // Tie-breaker: older files first (lower file number)
                  return a.blob_file_number < b.blob_file_number;
                });
      break;
    case BlobGCPriority::kLargestGarbageBytes:
      std::sort(candidates.begin(), candidates.end(),
                [](const BlobGCCandidate& a, const BlobGCCandidate& b) {
                  // Larger garbage bytes comes first
                  if (a.garbage_bytes != b.garbage_bytes) {
                    return a.garbage_bytes > b.garbage_bytes;
                  }
                  // Tie-breaker: older files first
                  return a.blob_file_number < b.blob_file_number;
                });
      break;
    case BlobGCPriority::kOldestFirst:
      std::sort(candidates.begin(), candidates.end(),
                [](const BlobGCCandidate& a, const BlobGCCandidate& b) {
                  // Lower file number (older) comes first
                  return a.blob_file_number < b.blob_file_number;
                });
      break;
  }

  // Limit to max_candidates
  if (candidates.size() > max_candidates_) {
    candidates.resize(max_candidates_);
  }

  return candidates;
}

}  // namespace ROCKSDB_NAMESPACE
