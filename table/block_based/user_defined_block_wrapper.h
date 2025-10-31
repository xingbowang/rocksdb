//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
//  *****************************************************************
//  EXPERIMENTAL - subject to change while under development
//  *****************************************************************

#pragma once

#include <memory>
#include <string>

#include "block.h"
#include "rocksdb/advanced_iterator.h"
#include "rocksdb/customizable.h"
#include "rocksdb/options.h"
#include "rocksdb/slice.h"
#include "rocksdb/status.h"
#include "rocksdb/user_defined_block.h"
#include "table/block_based/block_cache.h"

namespace ROCKSDB_NAMESPACE {

// User defined block warpper inherited from Block, and forward Block calls to
// external facing UserDefinedBlock object
class UserDefinedBlockWrapper : public Block_kData {
 public:
  UserDefinedBlockWrapper(std::unique_ptr<UserDefinedBlock> block,
                          BlockContents&& contents,
                          size_t read_amp_bytes_per_bit = 0,
                          Statistics* statistics = nullptr)
      : Block_kData(std::move(contents), read_amp_bytes_per_bit, statistics),
        block_(std::move(block)) {}

  Status InitBlock() { return block_->InitBlock(&contents_); }

  ~UserDefinedBlockWrapper() override;

  // Override Block methods to delegate to the user-defined block implementation
  size_t ApproximateMemoryUsage() const override {
    return block_->ApproximateMemoryUsage();
  }

  const Slice& ContentSlice() const override { return block_->ContentSlice(); }

  DataBlockIter* NewDataIterator(
      const Comparator* raw_ucmp, SequenceNumber global_seqno,
      DataBlockIter* input_iter = nullptr, Statistics* stats = nullptr,
      bool block_contents_pinned = false,
      bool user_defined_timestamps_persisted = true) override {
    return block_->NewDataIterator(raw_ucmp, global_seqno, input_iter, stats,
                                   block_contents_pinned,
                                   user_defined_timestamps_persisted);
  }

  std::unique_ptr<UserDefinedBlock> block_;
};

}  // namespace ROCKSDB_NAMESPACE
