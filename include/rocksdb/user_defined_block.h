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

#include "rocksdb/advanced_iterator.h"
#include "rocksdb/comparator.h"
#include "rocksdb/customizable.h"
#include "rocksdb/options.h"
#include "rocksdb/slice.h"
#include "rocksdb/status.h"
#include "rocksdb/types.h"
#include "rocksdb/utilities/customizable_util.h"

namespace ROCKSDB_NAMESPACE {

// Forward declarations
struct BlockContents;
class BlockBuilder;
class DataBlockIter;
class Statistics;

// This is a public API for user-defined block builders and block formats.
// It allows users to define their own block format and build custom
// blocks during table building. Currently, only a monolithic block
// is supported (no partitioned block).
//
// This is currently supported only for a restricted set of use cases. A custom
// block implementation must preserve the raw key/value bytes it receives;
// DB-generated files can include internal-key type and sequence metadata, such
// as deletion tombstones. The same factory Name() recorded when writing the
// SST must be configured when reading it. Custom UDB formats are not compatible
// with block_protection_bytes_per_key or separate_key_value_in_data_block.

// Options for user defined block
struct UserDefinedBlockOption {
  const Comparator* comparator = BytewiseComparator();
};

// ============================================================================
// User-Defined Block Interface
// ============================================================================
//
// Abstract base class for user-defined block implementations.
// Users can implement their own block formats (e.g., columnar, compressed,
// encrypted) by inheriting from this class.
//
// Example: A columnar PAX-format block would parse BlockContents into
// column-oriented storage and provide efficient column access through
// iterators.
//
class UserDefinedBlock {
 public:
  virtual ~UserDefinedBlock() = default;

  // Initializes this block from raw block contents. The contents remain owned
  // by RocksDB and remain valid for the lifetime of this block object.
  virtual Status InitBlock(BlockContents* contents) = 0;

  // Returns the approximate memory usage of this block, including any
  // auxiliary data structures (indexes, caches, etc.).
  // This is used for cache accounting and memory budgeting.
  virtual size_t ApproximateMemoryUsage() const = 0;

  // Returns a Slice pointing to the exact serialized block contents passed to
  // InitBlock(). This is used for cache accounting and secondary-cache
  // save/restore. Returning parsed/re-encoded bytes can make a block fail to
  // round-trip through secondary cache. The returned Slice must remain valid
  // for the lifetime of this object.
  virtual const Slice& ContentSlice() const = 0;

  // Returns the size of the block content in bytes.
  virtual size_t size() const = 0;

  // Returns a pointer to the raw block data.
  // This may be needed for compatibility with legacy code paths.
  virtual const char* data() const = 0;

  // Returns true if this block owns the underlying memory.
  // If false, the memory is owned externally (e.g., mmap'd file).
  virtual bool own_bytes() const = 0;

  // Creates a new iterator for this block.
  // The caller takes ownership of the returned iterator.
  //
  // Parameters:
  //   raw_ucmp - User comparator (not wrapped)
  //   global_seqno - Global sequence number to apply, or
  //                  kDisableGlobalSequenceNumber if disabled
  //   input_iter - If not nullptr, reuse this iterator object instead of
  //                allocating a new one
  //   stats - Statistics object for recording metrics (may be nullptr)
  //   block_contents_pinned - If true, the block data will remain valid
  //                           even after the iterator is passed to other
  //                           objects (e.g., PinnableSlice)
  //   user_defined_timestamps_persisted - Whether user-defined timestamps
  //                                       are persisted in the block
  //   user_defined_block_iterator_arg - Opaque per-read argument from
  //                                     ReadOptions. The type and lifetime
  //                                     contract are defined by the configured
  //                                     UserDefinedBlockFactory.
  //
  // Returns: A pointer to the iterator. If input_iter was provided and
  //          successfully reused, returns input_iter; otherwise returns
  //          a newly allocated iterator.
  virtual DataBlockIter* NewDataIterator(
      const Comparator* raw_ucmp, SequenceNumber global_seqno,
      DataBlockIter* input_iter = nullptr, Statistics* stats = nullptr,
      bool block_contents_pinned = false,
      bool user_defined_timestamps_persisted = true,
      void* user_defined_block_iterator_arg = nullptr) = 0;
};

// ============================================================================
// User-Defined Block Factory Interface
// ============================================================================
//
// Factory for creating user-defined block builders and block instances.
// Users implement this interface to plug their custom block format into
// RocksDB.
// Name() must return a stable, non-empty identifier for the on-disk format.
// RocksDB records this value in table properties and uses it to reject factory
// mismatches when opening UDB SSTs.
//
// The factory is responsible for:
// 1. Creating block builders for writing (NewBuilder)
// 2. Creating reusable block iterator shells for reading (NewIterator)
// 3. Creating custom block instances when UsesCustomBlockFormat() is true
//
class UserDefinedBlockFactory : public Customizable {
 public:
  virtual ~UserDefinedBlockFactory() = default;

  static const char* Type() { return "UserDefinedBlockFactory"; }

  DataBlockIteratorType IteratorType() const {
    return DataBlockIteratorType::kUserDefinedDataBlockIter;
  }

  static Status CreateFromString(
      const ConfigOptions& config_options, const std::string& value,
      std::shared_ptr<UserDefinedBlockFactory>* factory) {
    return LoadSharedObject<UserDefinedBlockFactory>(config_options, value,
                                                     factory);
  }

  // -------------------------------------------------------------------------
  // WRITE PATH: Creating block builders
  // -------------------------------------------------------------------------

  // Creates a new block builder for writing data.
  // The builder is used during table building to incrementally add key-value
  // pairs and produce a serialized block.
  //
  // Parameters:
  //   option - Configuration options including comparator
  //   builder - Output parameter to receive the created builder
  //
  // Returns: OK on success, or an error status
  virtual Status NewBuilder(const UserDefinedBlockOption& /*option*/,
                            std::unique_ptr<BlockBuilder>& builder) const = 0;

  // -------------------------------------------------------------------------
  // READ PATH: Creating reusable block iterators
  // -------------------------------------------------------------------------

  // Creates a new iterator object for reading a user-defined block. RocksDB may
  // pass this object to Block::NewDataIterator() or
  // UserDefinedBlock::NewDataIterator() as input_iter so it can be initialized
  // for a specific block without another allocation.
  //
  // Parameters:
  //   option - Configuration options including comparator
  //   biter - Output parameter to receive the created iterator.
  //
  // Returns: OK on success, or an error status
  virtual Status NewIterator(const UserDefinedBlockOption& option,
                             DataBlockIter** biter) const = 0;

  // -------------------------------------------------------------------------
  // READ PATH: Creating block instances from raw bytes (NEW API)
  // -------------------------------------------------------------------------

  // Creates a new block instance. RocksDB will call InitBlock() with the raw
  // BlockContents after the object is created.
  // This is the primary API for custom block formats.
  //
  // The implementation should:
  // 1. Return a UserDefinedBlock object that knows the custom format.
  // 2. Parse BlockContents in UserDefinedBlock::InitBlock().
  // 3. Create iterators on demand through UserDefinedBlock::NewDataIterator().
  //
  // Parameters:
  //   option - Configuration options including comparator
  //   block - Output parameter to receive the created block
  //
  // Returns: OK on success, NotSupported if custom blocks aren't implemented,
  //          or another error status on failure
  //
  // Example implementation for a columnar format:
  //
  //   Status NewBlock(const UserDefinedBlockOption& option,
  //                   std::unique_ptr<UserDefinedBlock>* block) const {
  //     *block = std::make_unique<PAXBlock>(option.comparator);
  //     return Status::OK();
  //   }
  //
  virtual Status NewBlock(const UserDefinedBlockOption& /*option*/,
                          std::unique_ptr<UserDefinedBlock>* /*block*/) const {
    // Default implementation: not supported
    // Users who want custom block formats must override this method
    return Status::NotSupported(
        "Custom block format not implemented by this factory");
  }

  // -------------------------------------------------------------------------
  // CAPABILITY DETECTION
  // -------------------------------------------------------------------------

  // Returns true if this factory implements custom block formats via
  // NewBlock(). When true, RocksDB will use NewBlock() to create blocks instead
  // of the standard Block class.
  //
  // Override this to return true if you implement NewBlock().
  virtual bool UsesCustomBlockFormat() const { return false; }
};

}  // namespace ROCKSDB_NAMESPACE
