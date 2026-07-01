//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "db/blob/blob_gc_candidate_selector.h"

#include <memory>
#include <vector>

#include "db/blob/blob_file_meta.h"
#include "test_util/testharness.h"

namespace ROCKSDB_NAMESPACE {

class BlobGCCandidateSelectorTest : public testing::Test {
 protected:
  using BlobFiles = std::map<uint64_t, std::shared_ptr<BlobFileMetaData>>;

  std::shared_ptr<BlobFileMetaData> CreateBlobFileMeta(
      uint64_t blob_file_number, uint64_t total_blob_count,
      uint64_t total_blob_bytes, uint64_t garbage_blob_count,
      uint64_t garbage_blob_bytes) {
    auto shared_meta = SharedBlobFileMetaData::Create(
        blob_file_number, total_blob_count, total_blob_bytes,
        /*checksum_method=*/"", /*checksum_value=*/"");
    return BlobFileMetaData::Create(std::move(shared_meta),
                                    /*linked_ssts=*/{}, garbage_blob_count,
                                    garbage_blob_bytes);
  }
};

// Basic filtering tests

TEST_F(BlobGCCandidateSelectorTest, EmptyBlobFiles) {
  BlobGCCandidateSelector selector(
      /*threshold=*/0.5, /*min_size=*/0,
      /*max_candidates=*/10, BlobGCPriority::kHighestGarbageRatio);
  BlobFiles blob_files;
  auto candidates = selector.SelectCandidates(blob_files);
  ASSERT_TRUE(candidates.empty());
}

TEST_F(BlobGCCandidateSelectorTest, NoCandidatesAboveThreshold) {
  BlobGCCandidateSelector selector(
      /*threshold=*/0.5, /*min_size=*/0,
      /*max_candidates=*/10, BlobGCPriority::kHighestGarbageRatio);
  BlobFiles blob_files;
  // All files have < 50% garbage
  blob_files[1] = CreateBlobFileMeta(1, 100, 1000, 10, 100);  // 10%
  blob_files[2] = CreateBlobFileMeta(2, 100, 1000, 20, 200);  // 20%
  blob_files[3] = CreateBlobFileMeta(3, 100, 1000, 40, 400);  // 40%
  auto candidates = selector.SelectCandidates(blob_files);
  ASSERT_TRUE(candidates.empty());
}

TEST_F(BlobGCCandidateSelectorTest, FiltersByGarbageThreshold) {
  BlobGCCandidateSelector selector(
      /*threshold=*/0.5, /*min_size=*/0,
      /*max_candidates=*/10, BlobGCPriority::kHighestGarbageRatio);
  BlobFiles blob_files;
  blob_files[1] = CreateBlobFileMeta(1, 100, 1000, 10, 100);  // 10%
  blob_files[2] = CreateBlobFileMeta(2, 100, 1000, 50, 500);  // 50% - candidate
  blob_files[3] = CreateBlobFileMeta(3, 100, 1000, 80, 800);  // 80% - candidate
  auto candidates = selector.SelectCandidates(blob_files);
  ASSERT_EQ(candidates.size(), 2);
  // Sorted by highest garbage ratio first
  ASSERT_EQ(candidates[0].blob_file_number, 3);
  ASSERT_EQ(candidates[1].blob_file_number, 2);
}

TEST_F(BlobGCCandidateSelectorTest, FiltersByMinSize) {
  BlobGCCandidateSelector selector(
      /*threshold=*/0.5, /*min_size=*/500,
      /*max_candidates=*/10, BlobGCPriority::kHighestGarbageRatio);
  BlobFiles blob_files;
  blob_files[1] = CreateBlobFileMeta(1, 10, 100, 8, 80);  // 80% but too small
  blob_files[2] = CreateBlobFileMeta(2, 100, 1000, 60, 600);  // 60% - candidate
  auto candidates = selector.SelectCandidates(blob_files);
  ASSERT_EQ(candidates.size(), 1);
  ASSERT_EQ(candidates[0].blob_file_number, 2);
}

// Priority-based sorting tests

TEST_F(BlobGCCandidateSelectorTest, SortsByHighestGarbageRatio) {
  BlobGCCandidateSelector selector(
      /*threshold=*/0.5, /*min_size=*/0,
      /*max_candidates=*/10, BlobGCPriority::kHighestGarbageRatio);
  BlobFiles blob_files;
  blob_files[1] = CreateBlobFileMeta(1, 100, 1000, 60, 600);  // 60%
  blob_files[2] = CreateBlobFileMeta(2, 100, 1000, 90, 900);  // 90%
  blob_files[3] = CreateBlobFileMeta(3, 100, 1000, 75, 750);  // 75%
  auto candidates = selector.SelectCandidates(blob_files);
  ASSERT_EQ(candidates.size(), 3);
  ASSERT_EQ(candidates[0].blob_file_number, 2);  // 90%
  ASSERT_EQ(candidates[1].blob_file_number, 3);  // 75%
  ASSERT_EQ(candidates[2].blob_file_number, 1);  // 60%
}

TEST_F(BlobGCCandidateSelectorTest, SortsByLargestGarbageBytes) {
  BlobGCCandidateSelector selector(
      /*threshold=*/0.5, /*min_size=*/0,
      /*max_candidates=*/10, BlobGCPriority::kLargestGarbageBytes);
  BlobFiles blob_files;
  blob_files[1] =
      CreateBlobFileMeta(1, 100, 1000, 60, 600);  // 600 bytes garbage
  blob_files[2] =
      CreateBlobFileMeta(2, 100, 2000, 60, 1200);  // 1200 bytes garbage
  blob_files[3] =
      CreateBlobFileMeta(3, 100, 1500, 60, 900);  // 900 bytes garbage
  auto candidates = selector.SelectCandidates(blob_files);
  ASSERT_EQ(candidates.size(), 3);
  ASSERT_EQ(candidates[0].blob_file_number, 2);  // 1200 bytes
  ASSERT_EQ(candidates[1].blob_file_number, 3);  // 900 bytes
  ASSERT_EQ(candidates[2].blob_file_number, 1);  // 600 bytes
}

TEST_F(BlobGCCandidateSelectorTest, SortsByOldestFirst) {
  BlobGCCandidateSelector selector(
      /*threshold=*/0.5, /*min_size=*/0,
      /*max_candidates=*/10, BlobGCPriority::kOldestFirst);
  BlobFiles blob_files;
  blob_files[5] = CreateBlobFileMeta(5, 100, 1000, 60, 600);  // newer
  blob_files[2] = CreateBlobFileMeta(2, 100, 1000, 70, 700);  // older
  blob_files[8] = CreateBlobFileMeta(8, 100, 1000, 80, 800);  // newest
  auto candidates = selector.SelectCandidates(blob_files);
  ASSERT_EQ(candidates.size(), 3);
  ASSERT_EQ(candidates[0].blob_file_number, 2);  // oldest
  ASSERT_EQ(candidates[1].blob_file_number, 5);
  ASSERT_EQ(candidates[2].blob_file_number, 8);  // newest
}

// Max candidates limit test

TEST_F(BlobGCCandidateSelectorTest, LimitsMaxCandidates) {
  BlobGCCandidateSelector selector(
      /*threshold=*/0.5, /*min_size=*/0,
      /*max_candidates=*/2, BlobGCPriority::kHighestGarbageRatio);
  BlobFiles blob_files;
  blob_files[1] = CreateBlobFileMeta(1, 100, 1000, 60, 600);  // 60%
  blob_files[2] = CreateBlobFileMeta(2, 100, 1000, 90, 900);  // 90%
  blob_files[3] = CreateBlobFileMeta(3, 100, 1000, 75, 750);  // 75%
  blob_files[4] = CreateBlobFileMeta(4, 100, 1000, 55, 550);  // 55%
  auto candidates = selector.SelectCandidates(blob_files);
  ASSERT_EQ(candidates.size(), 2);
  // Top 2 by garbage ratio
  ASSERT_EQ(candidates[0].blob_file_number, 2);  // 90%
  ASSERT_EQ(candidates[1].blob_file_number, 3);  // 75%
}

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
