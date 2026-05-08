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
#include <utility>

#include "block.h"
#include "rocksdb/advanced_iterator.h"
#include "rocksdb/customizable.h"
#include "rocksdb/options.h"
#include "rocksdb/slice.h"
#include "rocksdb/status.h"
#include "rocksdb/user_defined_block.h"
#include "table/block_based/block_cache.h"

namespace ROCKSDB_NAMESPACE {

// Wraps the public UserDefinedBlock API in the internal Block interface used by
// block cache and table iterators.
class UserDefinedBlockWrapper : public Block_kData {
 public:
  UserDefinedBlockWrapper(std::unique_ptr<UserDefinedBlock> block,
                          BlockContents&& contents,
                          size_t read_amp_bytes_per_bit = 0,
                          Statistics* statistics = nullptr,
                          uint32_t restart_interval = 1,
                          Status init_status = Status::OK())
      : Block_kData(std::move(contents), read_amp_bytes_per_bit, statistics,
                    restart_interval,
                    /*skip_initialization*/ true),
        block_(std::move(block)),
        init_status_(std::move(init_status)) {}

  Status InitBlock() {
    if (block_ == nullptr) {
      if (init_status_.ok()) {
        init_status_ =
            Status::InvalidArgument("user-defined block factory returned null");
      }
      return init_status_;
    }
    init_status_ = block_->InitBlock(&contents_);
    return init_status_;
  }

  ~UserDefinedBlockWrapper() override = default;

  // Override Block methods to delegate to the user-defined block implementation
  size_t ApproximateMemoryUsage() const override {
    if (!init_status_.ok() || block_ == nullptr) {
      return Block_kData::ApproximateMemoryUsage();
    }
    return block_->ApproximateMemoryUsage();
  }

  const Slice& ContentSlice() const override {
    if (!init_status_.ok() || block_ == nullptr) {
      return Block_kData::ContentSlice();
    }
    return block_->ContentSlice();
  }

  DataBlockIter* NewDataIterator(
      const Comparator* raw_ucmp, SequenceNumber global_seqno,
      DataBlockIter* input_iter, Statistics* stats, bool block_contents_pinned,
      bool user_defined_timestamps_persisted,
      void* user_defined_block_iterator_arg) override {
    if (!init_status_.ok() || block_ == nullptr) {
      DataBlockIter* iter =
          input_iter != nullptr ? input_iter : new DataBlockIter();
      iter->Invalidate(init_status_);
      return iter;
    }
    DataBlockIter* iter = block_->NewDataIterator(
        raw_ucmp, global_seqno, input_iter, stats, block_contents_pinned,
        user_defined_timestamps_persisted, user_defined_block_iterator_arg);
    if (iter == nullptr) {
      iter = input_iter != nullptr ? input_iter : new DataBlockIter();
      iter->Invalidate(
          Status::InvalidArgument("user-defined block returned null iterator"));
    }
    return iter;
  }

 private:
  std::unique_ptr<UserDefinedBlock> block_;
  Status init_status_;
};

}  // namespace ROCKSDB_NAMESPACE
