//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "db/blob/blob_file_meta.h"

#include <memory>

#include "test_util/testharness.h"

namespace ROCKSDB_NAMESPACE {

class BlobFileMetaDataTest : public testing::Test {
 protected:
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

// GetGarbageRatio() tests

TEST_F(BlobFileMetaDataTest, GetGarbageRatioZeroTotal) {
  // Empty file should return 0.0
  auto meta = CreateBlobFileMeta(/*blob_file_number=*/1,
                                 /*total_blob_count=*/0,
                                 /*total_blob_bytes=*/0,
                                 /*garbage_blob_count=*/0,
                                 /*garbage_blob_bytes=*/0);
  ASSERT_DOUBLE_EQ(meta->GetGarbageRatio(), 0.0);
}

TEST_F(BlobFileMetaDataTest, GetGarbageRatioNoGarbage) {
  // No garbage should return 0.0
  auto meta = CreateBlobFileMeta(/*blob_file_number=*/1,
                                 /*total_blob_count=*/100,
                                 /*total_blob_bytes=*/1000,
                                 /*garbage_blob_count=*/0,
                                 /*garbage_blob_bytes=*/0);
  ASSERT_DOUBLE_EQ(meta->GetGarbageRatio(), 0.0);
}

TEST_F(BlobFileMetaDataTest, GetGarbageRatioPartialGarbage) {
  // 25% garbage
  auto meta = CreateBlobFileMeta(/*blob_file_number=*/1,
                                 /*total_blob_count=*/100,
                                 /*total_blob_bytes=*/1000,
                                 /*garbage_blob_count=*/25,
                                 /*garbage_blob_bytes=*/250);
  ASSERT_DOUBLE_EQ(meta->GetGarbageRatio(), 0.25);
}

TEST_F(BlobFileMetaDataTest, GetGarbageRatioAllGarbage) {
  // 100% garbage
  auto meta = CreateBlobFileMeta(/*blob_file_number=*/1,
                                 /*total_blob_count=*/100,
                                 /*total_blob_bytes=*/1000,
                                 /*garbage_blob_count=*/100,
                                 /*garbage_blob_bytes=*/1000);
  ASSERT_DOUBLE_EQ(meta->GetGarbageRatio(), 1.0);
}

// IsGCCandidate() tests

TEST_F(BlobFileMetaDataTest, IsGCCandidateBelowThreshold) {
  // 25% garbage, threshold 50% -> not a candidate
  auto meta = CreateBlobFileMeta(/*blob_file_number=*/1,
                                 /*total_blob_count=*/100,
                                 /*total_blob_bytes=*/1000,
                                 /*garbage_blob_count=*/25,
                                 /*garbage_blob_bytes=*/250);
  ASSERT_FALSE(meta->IsGCCandidate(/*threshold=*/0.5, /*min_size=*/0));
}

TEST_F(BlobFileMetaDataTest, IsGCCandidateAboveThreshold) {
  // 75% garbage, threshold 50% -> candidate
  auto meta = CreateBlobFileMeta(/*blob_file_number=*/1,
                                 /*total_blob_count=*/100,
                                 /*total_blob_bytes=*/1000,
                                 /*garbage_blob_count=*/75,
                                 /*garbage_blob_bytes=*/750);
  ASSERT_TRUE(meta->IsGCCandidate(/*threshold=*/0.5, /*min_size=*/0));
}

TEST_F(BlobFileMetaDataTest, IsGCCandidateBelowMinSize) {
  // 75% garbage but file too small -> not a candidate
  auto meta = CreateBlobFileMeta(/*blob_file_number=*/1,
                                 /*total_blob_count=*/10,
                                 /*total_blob_bytes=*/100,
                                 /*garbage_blob_count=*/7,
                                 /*garbage_blob_bytes=*/75);
  ASSERT_FALSE(meta->IsGCCandidate(/*threshold=*/0.5, /*min_size=*/1000));
}

TEST_F(BlobFileMetaDataTest, IsGCCandidateExactlyAtThreshold) {
  // 50% garbage, threshold 50% -> candidate (>= not >)
  auto meta = CreateBlobFileMeta(/*blob_file_number=*/1,
                                 /*total_blob_count=*/100,
                                 /*total_blob_bytes=*/1000,
                                 /*garbage_blob_count=*/50,
                                 /*garbage_blob_bytes=*/500);
  ASSERT_TRUE(meta->IsGCCandidate(/*threshold=*/0.5, /*min_size=*/0));
}

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
