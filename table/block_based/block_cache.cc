//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "table/block_based/block_cache.h"

#include "rocksdb/user_defined_block.h"
#include "table/block_based/block_based_table_reader.h"

namespace ROCKSDB_NAMESPACE {

// Block_kUserDefinedData implementation
Block_kUserDefinedData::Block_kUserDefinedData(UserDefinedBlock* block)
    : block_(block) {}

Block_kUserDefinedData::~Block_kUserDefinedData() { delete block_; }

size_t Block_kUserDefinedData::ApproximateMemoryUsage() const {
  return block_->ApproximateMemoryUsage();
}

const Slice& Block_kUserDefinedData::ContentSlice() const {
  return block_->ContentSlice();
}

DataBlockIter* Block_kUserDefinedData::NewDataIterator(
    const Comparator* raw_ucmp, SequenceNumber global_seqno,
    DataBlockIter* input_iter, Statistics* stats, bool block_contents_pinned,
    bool user_defined_timestamps_persisted) {
  return block_->NewDataIterator(raw_ucmp, global_seqno, input_iter, stats,
                                 block_contents_pinned,
                                 user_defined_timestamps_persisted);
}

void BlockCreateContext::Create(std::unique_ptr<Block_kData>* parsed_out,
                                BlockContents&& block) {
  // Check if user-defined block factory is present and supports custom format
  if (table_options->user_defined_block_factory != nullptr &&
      table_options->user_defined_block_factory->UsesCustomBlockFormat()) {
    // Use the user-defined block factory to create a custom block
    std::unique_ptr<UserDefinedBlock> custom_block;
    UserDefinedBlockOption option;
    option.comparator = raw_ucmp;

    Status s = table_options->user_defined_block_factory->NewBlock(
        option, std::move(block), &custom_block);

    if (s.ok() && custom_block != nullptr) {
      // Wrap the user-defined block in Block_kUserDefinedData for cache
      // compatibility. Note: We use Block_kData* but it actually points to a
      // Block_kUserDefinedData. This works because cache only uses the
      // interface methods (ApproximateMemoryUsage, ContentSlice, etc.)
      // TODO: Refactor cache to use a common base class for all block types
      parsed_out->reset(reinterpret_cast<Block_kData*>(
          new Block_kUserDefinedData(custom_block.release())));
      return;
    }
    // If custom block creation failed, fall through to standard block
  }

  // Standard RocksDB block format
  parsed_out->reset(new Block_kData(
      std::move(block), table_options->read_amp_bytes_per_bit, statistics));
  parsed_out->get()->InitializeDataBlockProtectionInfo(protection_bytes_per_key,
                                                       raw_ucmp);
}
void BlockCreateContext::Create(std::unique_ptr<Block_kIndex>* parsed_out,
                                BlockContents&& block) {
  parsed_out->reset(new Block_kIndex(std::move(block),
                                     /*read_amp_bytes_per_bit*/ 0, statistics));
  parsed_out->get()->InitializeIndexBlockProtectionInfo(
      protection_bytes_per_key, raw_ucmp, index_value_is_full,
      index_has_first_key);
}
void BlockCreateContext::Create(
    std::unique_ptr<Block_kFilterPartitionIndex>* parsed_out,
    BlockContents&& block) {
  parsed_out->reset(new Block_kFilterPartitionIndex(
      std::move(block), /*read_amp_bytes_per_bit*/ 0, statistics));
  parsed_out->get()->InitializeIndexBlockProtectionInfo(
      protection_bytes_per_key, raw_ucmp, index_value_is_full,
      index_has_first_key);
}
void BlockCreateContext::Create(
    std::unique_ptr<Block_kRangeDeletion>* parsed_out, BlockContents&& block) {
  parsed_out->reset(new Block_kRangeDeletion(
      std::move(block), /*read_amp_bytes_per_bit*/ 0, statistics));
}
void BlockCreateContext::Create(std::unique_ptr<Block_kMetaIndex>* parsed_out,
                                BlockContents&& block) {
  parsed_out->reset(new Block_kMetaIndex(
      std::move(block), /*read_amp_bytes_per_bit*/ 0, statistics));
  parsed_out->get()->InitializeMetaIndexBlockProtectionInfo(
      protection_bytes_per_key);
}

void BlockCreateContext::Create(
    std::unique_ptr<Block_kUserDefinedIndex>* parsed_out,
    BlockContents&& block) {
  parsed_out->reset(new Block_kUserDefinedIndex(std::move(block)));
}

void BlockCreateContext::Create(
    std::unique_ptr<ParsedFullFilterBlock>* parsed_out, BlockContents&& block) {
  parsed_out->reset(new ParsedFullFilterBlock(
      table_options->filter_policy.get(), std::move(block)));
}

void BlockCreateContext::Create(std::unique_ptr<DecompressorDict>* parsed_out,
                                BlockContents&& block) {
  parsed_out->reset(new DecompressorDict(
      block.data, std::move(block.allocation), *decompressor));
}

namespace {
// For getting SecondaryCache-compatible helpers from a BlockType. This is
// useful for accessing block cache in untyped contexts, such as for generic
// cache warming in table builder.
const std::array<const Cache::CacheItemHelper*,
                 static_cast<unsigned>(BlockType::kInvalid) + 1>
    kCacheItemFullHelperForBlockType{{
        BlockCacheInterface<Block_kData>::GetFullHelper(),
        BlockCacheInterface<ParsedFullFilterBlock>::GetFullHelper(),
        BlockCacheInterface<Block_kFilterPartitionIndex>::GetFullHelper(),
        nullptr,  // kProperties
        BlockCacheInterface<DecompressorDict>::GetFullHelper(),
        BlockCacheInterface<Block_kRangeDeletion>::GetFullHelper(),
        nullptr,  // kHashIndexPrefixes
        nullptr,  // kHashIndexMetadata
        nullptr,  // kMetaIndex (not yet stored in block cache)
        BlockCacheInterface<Block_kIndex>::GetFullHelper(),
        nullptr,  // kInvalid
    }};

// For getting basic helpers from a BlockType (no SecondaryCache support)
const std::array<const Cache::CacheItemHelper*,
                 static_cast<unsigned>(BlockType::kInvalid) + 1>
    kCacheItemBasicHelperForBlockType{{
        BlockCacheInterface<Block_kData>::GetBasicHelper(),
        BlockCacheInterface<ParsedFullFilterBlock>::GetBasicHelper(),
        BlockCacheInterface<Block_kFilterPartitionIndex>::GetBasicHelper(),
        nullptr,  // kProperties
        BlockCacheInterface<DecompressorDict>::GetBasicHelper(),
        BlockCacheInterface<Block_kRangeDeletion>::GetBasicHelper(),
        nullptr,  // kHashIndexPrefixes
        nullptr,  // kHashIndexMetadata
        nullptr,  // kMetaIndex (not yet stored in block cache)
        BlockCacheInterface<Block_kIndex>::GetBasicHelper(),
        nullptr,  // kInvalid
    }};
}  // namespace

const Cache::CacheItemHelper* GetCacheItemHelper(
    BlockType block_type, CacheTier lowest_used_cache_tier) {
  if (lowest_used_cache_tier > CacheTier::kVolatileTier) {
    return kCacheItemFullHelperForBlockType[static_cast<unsigned>(block_type)];
  } else {
    return kCacheItemBasicHelperForBlockType[static_cast<unsigned>(block_type)];
  }
}

}  // namespace ROCKSDB_NAMESPACE
