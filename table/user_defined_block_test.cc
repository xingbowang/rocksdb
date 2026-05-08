//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "rocksdb/user_defined_block.h"

#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "db/dbformat.h"
#include "options/options_helper.h"
#include "port/stack_trace.h"
#include "rocksdb/db.h"
#include "rocksdb/env.h"
#include "rocksdb/flush_block_policy.h"
#include "rocksdb/iterator.h"
#include "rocksdb/options.h"
#include "rocksdb/slice.h"
#include "rocksdb/sst_file_reader.h"
#include "rocksdb/sst_file_writer.h"
#include "rocksdb/table.h"
#include "table/block_based/block.h"
#include "table/block_based/block_based_table_factory.h"
#include "table/block_based/block_builder.h"
#include "table/format.h"
#include "table/meta_blocks.h"
#include "test_util/testharness.h"
#include "test_util/testutil.h"
#include "util/coding.h"
#include "util/random.h"

namespace ROCKSDB_NAMESPACE {

class UserDefinedBlockTestBase : public testing::Test {
 public:
  class CustomFlushBlockPolicy : public FlushBlockPolicy {
   public:
    explicit CustomFlushBlockPolicy(int keys_per_block)
        : keys_in_current_block_(0), keys_per_block_(keys_per_block) {}

    bool Update(const Slice& /*key*/, const Slice& /*value*/) override {
      if (keys_in_current_block_ >= keys_per_block_) {
        keys_in_current_block_ = 1;
        return true;
      }
      keys_in_current_block_++;
      return false;
    }

   private:
    int keys_in_current_block_;
    int keys_per_block_;
  };

  class CustomFlushBlockPolicyFactory : public FlushBlockPolicyFactory {
   public:
    CustomFlushBlockPolicyFactory(int keys_per_block = 10)
        : keys_per_block_(keys_per_block) {}
    const char* Name() const override { return "CustomFlushBlockPolicy"; }
    FlushBlockPolicy* NewFlushBlockPolicy(const BlockBasedTableOptions&,
                                          const BlockBuilder&) const override {
      return new CustomFlushBlockPolicy(keys_per_block_);
    }
    int keys_per_block_;
  };

  // PAX Block Iterator - reads PAX formatted data
  class PAXBlockIterator : public DataBlockIter {
   public:
    PAXBlockIterator() : current_minipage_idx_(0), current_record_idx_(0) {}

    void Init(void* user_defined_block_iterator_arg, const Comparator* raw_ucmp,
              SequenceNumber global_seqno, const char* data,
              bool keys_are_internal) {
      Invalidate(Status::OK());
      // Now override with PAX-specific initialization
      user_defined_block_iterator_arg_ = user_defined_block_iterator_arg;
      keys_are_internal_ = keys_are_internal;
      global_seqno_ =
          keys_are_internal ? global_seqno : kDisableGlobalSequenceNumber;
      comparator_ = raw_ucmp;
      data_ = data;
      ParsePAXBlock();
      SeekToFirstImpl();
    }

    Slice value() const override {
      if (!Valid()) return Slice();
      return current_value_;
    }

    DataBlockIteratorType Type() const override {
      return DataBlockIteratorType::kUserDefinedDataBlockIter;
    }

   protected:
    Slice UserKey(const Slice& key) const {
      return keys_are_internal_ ? ExtractUserKey(key) : key;
    }

    void SeekToFirstImpl() override {
      if (records_.empty()) {
        current_minipage_idx_ = 0;
        current_record_idx_ = 0;
        current_ = restarts_;  // Mark as invalid
        return;
      }
      current_minipage_idx_ = 0;
      current_record_idx_ = 0;
      current_ = 0;
      LoadCurrentRecord();
    }

    void SeekToLastImpl() override {
      if (records_.empty()) {
        current_minipage_idx_ = 0;
        current_record_idx_ = 0;
        current_ = restarts_;
        return;
      }
      current_minipage_idx_ = static_cast<uint32_t>(records_.size() - 1);
      current_record_idx_ = 0;
      current_ = 0;
      LoadCurrentRecord();
    }

    void SeekImpl(const Slice& target) override {
      auto target_user_key = UserKey(target);
      if (records_.empty()) {
        current_ = restarts_;
        return;
      }

      // Binary search through records
      uint32_t left = 0;
      uint32_t right = static_cast<uint32_t>(records_.size());

      while (left < right) {
        uint32_t mid = (left + right) / 2;
        auto cmp = comparator_->CompareWithoutTimestamp(
            target_user_key, /*a_has_ts=*/false, UserKey(records_[mid].first),
            /*b_has_ts=*/false);
        if (cmp > 0) {
          left = mid + 1;
        } else if (cmp < 0) {
          right = mid;
        } else {
          current_minipage_idx_ = mid;
          current_record_idx_ = 0;
          current_ = 0;
          LoadCurrentRecord();
          return;
        }
      }

      if (left >= records_.size()) {
        current_ = restarts_;
        return;
      }

      current_minipage_idx_ = left;
      current_record_idx_ = 0;
      current_ = 0;
      LoadCurrentRecord();
    }

    void SeekForPrevImpl(const Slice& target) override {
      SeekImpl(target);
      if (!Valid()) {
        SeekToLastImpl();
        return;
      }

      // If current user key is greater than target's user key, move to
      // previous. The comparator_ is the raw user comparator, while SST-backed
      // iterator keys are internal keys.
      if (comparator_->CompareWithoutTimestamp(
              UserKey(key_), /*a_has_ts=*/false, UserKey(target),
              /*b_has_ts=*/false) > 0) {
        PrevImpl();
      }
    }

    void NextImpl() override {
      if (!Valid()) return;

      current_minipage_idx_++;
      if (current_minipage_idx_ >= records_.size()) {
        current_ = restarts_;
        return;
      }

      current_record_idx_ = 0;
      LoadCurrentRecord();
    }

    void PrevImpl() override {
      if (current_minipage_idx_ == 0) {
        current_ = restarts_;
        return;
      }

      current_minipage_idx_--;
      current_record_idx_ = 0;
      LoadCurrentRecord();
    }

   private:
    void ParsePAXBlock() {
      if (data_ == nullptr) return;

      const char* ptr = data_;

      // Read number of minipages
      uint32_t num_minipages = DecodeFixed32(ptr);
      ptr += 4;

      records_.clear();

      // Parse each minipage
      for (uint32_t mp = 0; mp < num_minipages; mp++) {
        // Read number of records in this minipage
        uint32_t num_records = DecodeFixed32(ptr);
        ptr += 4;

        // Parse keys column
        std::vector<Slice> keys;
        for (uint32_t i = 0; i < num_records; i++) {
          uint32_t key_len = DecodeFixed32(ptr);
          ptr += 4;
          keys.emplace_back(ptr, key_len);
          ptr += key_len;
        }

        // Parse values column
        std::vector<Slice> values;
        for (uint32_t i = 0; i < num_records; i++) {
          uint32_t val_len = DecodeFixed32(ptr);
          ptr += 4;
          values.emplace_back(ptr, val_len);
          ptr += val_len;
        }

        // Store key-value pairs
        for (uint32_t i = 0; i < num_records; i++) {
          records_.emplace_back(keys[i], values[i]);
        }
      }

      // Set restarts_ to records_.size() for Valid() check
      restarts_ = static_cast<uint32_t>(records_.size());
    }

    void LoadCurrentRecord() {
      if (current_minipage_idx_ >= records_.size()) {
        current_ = restarts_;
        return;
      }

      const auto& record = records_[current_minipage_idx_];
      key_ = record.first;
      current_value_ = record.second;
      if (keys_are_internal_) {
        raw_key_.SetInternalKey(key_, false);
      } else {
        raw_key_.SetUserKey(key_, false);
      }
    }

    void* user_defined_block_iterator_arg_;
    const Comparator* comparator_;
    bool keys_are_internal_;
    std::vector<std::pair<Slice, Slice>> records_;
    uint32_t current_minipage_idx_;
    uint32_t current_record_idx_;
    Slice current_value_;
  };

  // PAX (Partition Attributes Across) columnar format implementation
  // Data is organized in mini-pages with columnar storage within each page
  class PAXBlockBuilder : public BlockBuilder {
   public:
    explicit PAXBlockBuilder(int minipage_size = 100)
        : BlockBuilder(1, false, false,
                       BlockBasedTableOptions::kDataBlockBinarySearch),
          minipage_size_(minipage_size),
          current_minipage_records_(0) {}

    void Add(const Slice& key, const Slice& value,
             const Slice* const /*delta_value*/ = nullptr,
             bool /*skip_delta_encoding*/ = false) override {
      // Store keys and values separately (columnar within minipage)
      keys_.emplace_back(key.data(), key.size());
      values_.emplace_back(value.data(), value.size());
      current_minipage_records_++;
      total_key_count_++;

      // Flush minipage when it reaches the configured size
      if (current_minipage_records_ >= minipage_size_) {
        FlushMinipage();
      }
    }
    void AddWithLastKey(const Slice& key, const Slice& value,
                        const Slice& /*last_key_param*/,
                        const Slice* const /*delta_value*/,
                        bool /*skip_delta_encoding*/) override {
      Add(key, value, nullptr, false);
    }

    bool empty() const override { return total_key_count_ == 0; }

    void SwapAndReset(std::string& buffer) override {
      std::swap(buffer_, buffer);
      Reset();
    }

    Slice Finish() override {
      // Flush any remaining records
      if (current_minipage_records_ > 0) {
        FlushMinipage();
      }

      // Build final buffer with all minipages
      buffer_.clear();

      // Header: [num_minipages (4 bytes)]
      PutFixed32(&buffer_, static_cast<uint32_t>(minipages_.size()));

      // Append all minipages
      for (const auto& minipage : minipages_) {
        buffer_.append(minipage);
      }

      return Slice(buffer_);
    }

    size_t EstimateSizeAfterKV(const Slice& key,
                               const Slice& value) const override {
      return CurrentSizeEstimate() + key.size() + value.size() + 16;
    }

    size_t CurrentSizeEstimate() const override {
      size_t size = 4;  // num_minipages
      for (const auto& minipage : minipages_) {
        size += minipage.size();
      }
      // Add current minipage estimate
      for (const auto& k : keys_) {
        size += k.size() + 4;
      }
      for (const auto& v : values_) {
        size += v.size() + 4;
      }
      return size + 100;  // overhead
    }
    std::string& MutableBuffer() override { return buffer_; }

    void Reset() override {
      BlockBuilder::Reset();
      keys_.clear();
      values_.clear();
      minipages_.clear();
      buffer_.clear();
      current_minipage_records_ = 0;
      total_key_count_ = 0;
    }

   private:
    void FlushMinipage() {
      if (current_minipage_records_ == 0) return;

      std::string minipage;

      // Minipage header: [num_records (4 bytes)]
      PutFixed32(&minipage, current_minipage_records_);

      // Keys column: [key1_len, key1_data, key2_len, key2_data, ...]
      for (const auto& key : keys_) {
        PutFixed32(&minipage, static_cast<uint32_t>(key.size()));
        minipage.append(key);
      }

      // Values column: [val1_len, val1_data, val2_len, val2_data, ...]
      for (const auto& value : values_) {
        PutFixed32(&minipage, static_cast<uint32_t>(value.size()));
        minipage.append(value);
      }

      minipages_.push_back(std::move(minipage));
      keys_.clear();
      values_.clear();
      current_minipage_records_ = 0;
    }

    int minipage_size_;
    int current_minipage_records_;
    int total_key_count_{0};
    std::vector<std::string> keys_;
    std::vector<std::string> values_;
    std::vector<std::string> minipages_;
    std::string buffer_;
  };

  // PAXBlock: Implements UserDefinedBlock interface for columnar storage
  class PAXBlock : public UserDefinedBlock {
   public:
    Status InitBlock(BlockContents* contents) override {
      contents_ = contents;
      // The data is already parsed in the iterator, just validate
      if (contents_->data.empty()) {
        return Status::Corruption("Empty block contents");
      }
      return Status::OK();
    }

    ~PAXBlock() override = default;

    // UserDefinedBlock interface
    size_t ApproximateMemoryUsage() const override {
      return contents_->ApproximateMemoryUsage() + sizeof(*this);
    }

    const Slice& ContentSlice() const override { return contents_->data; }

    size_t size() const override { return contents_->data.size(); }

    const char* data() const override { return contents_->data.data(); }

    bool own_bytes() const override { return contents_->own_bytes(); }

    DataBlockIter* NewDataIterator(
        const Comparator* raw_ucmp, SequenceNumber global_seqno,
        DataBlockIter* input_iter, Statistics* /*stats*/,
        bool /*block_contents_pinned*/,
        bool /*user_defined_timestamps_persisted*/,
        void* user_defined_block_iterator_arg) override {
      PAXBlockIterator* iter = nullptr;
      if (input_iter != nullptr &&
          input_iter->Type() ==
              DataBlockIteratorType::kUserDefinedDataBlockIter) {
        iter = static_cast<PAXBlockIterator*>(input_iter);
      } else {
        iter = new PAXBlockIterator();
      }

      // Initialize the iterator with block data
      iter->Init(user_defined_block_iterator_arg, raw_ucmp, global_seqno,
                 contents_->data.data(), /*keys_are_internal=*/true);

      return iter;
    }

   private:
    BlockContents* contents_;
  };

  class TestUserDefinedBlockFactory : public UserDefinedBlockFactory {
   public:
    const char* Name() const override { return "test_pax_udb"; }

    Status NewBuilder(const UserDefinedBlockOption& /*option*/,
                      std::unique_ptr<BlockBuilder>& builder) const override {
      // Use PAX format with minipage size of 50 records
      builder = std::make_unique<PAXBlockBuilder>(50);
      return Status::OK();
    }

    // Creates the reusable iterator shell that will later be initialized with
    // a specific custom-format block.
    Status NewIterator(const UserDefinedBlockOption& /*option*/,
                       DataBlockIter** biter) const override {
      *biter = new PAXBlockIterator();
      return Status::OK();
    }

    // NEW API: Create custom block from raw bytes
    Status NewBlock(const UserDefinedBlockOption& /*option*/,
                    std::unique_ptr<UserDefinedBlock>* block) const override {
      *block = std::make_unique<PAXBlock>();
      return Status::OK();
    }

    // Indicate we use custom block format
    bool UsesCustomBlockFormat() const override { return true; }
  };

 protected:
  std::vector<std::pair<std::string, std::string>> generateKVWithValue(
      int key_count, const std::string& value) {
    std::vector<std::pair<std::string, std::string>> kvs(key_count);
    for (int i = 0; i < key_count; i++) {
      std::stringstream ss;
      ss << std::setw(2) << std::setfill('0') << i;
      std::string key = "key" + ss.str();
      kvs[i] = std::make_pair(key, value);
    }
    return kvs;
  }

  std::vector<std::pair<std::string, std::string>> generateKVs(
      int key_count, int value_size = 0) {
    std::vector<std::pair<std::string, std::string>> kvs(key_count);
    // Determine width based on key_count
    int width = key_count < 100 ? 2 : (key_count < 1000 ? 3 : 4);
    for (int i = 0; i < key_count; i++) {
      std::stringstream ss;
      ss << std::setw(width) << std::setfill('0') << i;
      std::string key = "key" + ss.str();
      std::string value;
      if (value_size != 0) {
        value = rnd.RandomString(value_size);
      } else {
        value = "value" + ss.str();
      }
      kvs[i] = std::make_pair(key, value);
    }
    return kvs;
  }

  void BasicTest();

  Options options_;
  Random rnd{301};
};

void UserDefinedBlockTestBase::BasicTest() {
  BlockBasedTableOptions table_options;
  std::string dbname = test::PerThreadDBPath("user_defined_block_test");
  std::string ingest_file = dbname + "test.sst";

  // Set up the user-defined block factory
  auto user_defined_block_factory =
      std::make_shared<TestUserDefinedBlockFactory>();
  table_options.user_defined_block_factory = user_defined_block_factory;

  // Set up custom flush block policy that flushes every 3 keys
  table_options.flush_block_policy_factory =
      std::make_shared<CustomFlushBlockPolicyFactory>();

  options_.table_factory.reset(NewBlockBasedTableFactory(table_options));
  options_.compression = kNoCompression;

  std::unique_ptr<SstFileWriter> writer;
  writer.reset(new SstFileWriter(EnvOptions(), options_));
  ASSERT_OK(writer->Open(ingest_file));

  auto kvs = generateKVs(/*key_count*/ 100);
  for (const auto& kv : kvs) {
    ASSERT_OK(writer->Put(kv.first, kv.second));
  }
  ASSERT_OK(writer->Finish());
  writer.reset();

  // Verify we can read the data using PAX format
  std::unique_ptr<SstFileReader> reader(new SstFileReader(options_));
  ASSERT_OK(reader->Open(ingest_file));

  ReadOptions ro;
  std::unique_ptr<Iterator> iter(reader->NewIterator(ro));
  ASSERT_NE(iter, nullptr);

  // Test that we can read all the keys
  int key_count = 0;
  for (iter->SeekToFirst(); iter->Valid() && iter->status().ok();
       iter->Next()) {
    key_count++;
  }
  ASSERT_EQ(key_count, 100);
  ASSERT_OK(iter->status());
  iter.reset();

  // Test seek to specific key
  iter.reset(reader->NewIterator(ro));
  ASSERT_NE(iter, nullptr);
  key_count = 0;
  for (iter->Seek("key040"); iter->Valid(); iter->Next()) {
    key_count++;
  }
  ASSERT_EQ(key_count, 60);
  ASSERT_OK(iter->status());

  // Test upper bound
  Slice ub("key075");
  ro.iterate_upper_bound = &ub;
  iter.reset(reader->NewIterator(ro));
  ASSERT_NE(iter, nullptr);

  key_count = 0;
  for (iter->Seek("key040"); iter->Valid(); iter->Next()) {
    key_count++;
  }
  ASSERT_EQ(key_count, 35);
  ASSERT_OK(iter->status());
  iter->Reset();
}

class UserDefinedBlockTest : public UserDefinedBlockTestBase {};

// Verifies the sample PAX builder/iterator contract without SST machinery by
// building an in-memory custom block and exercising basic iterator movement.
TEST_F(UserDefinedBlockTest, PAXBlockBuilderAndIteratorTest) {
  // Create PAX block builder
  PAXBlockBuilder builder(10);  // 10 records per minipage

  // Add sorted key-value pairs so Seek() can binary-search using the bytewise
  // comparator, as an SST block iterator would require.
  for (int i = 0; i < 25; i++) {
    std::stringstream ss;
    ss << std::setw(2) << std::setfill('0') << i;
    std::string key = "key" + ss.str();
    std::string value = "value" + ss.str();
    builder.Add(key, value);
  }

  // Finish building
  Slice block_data = builder.Finish();
  ASSERT_GT(block_data.size(), 0);

  // Create a copy of the data for the iterator
  std::string block_copy(block_data.data(), block_data.size());

  // Create PAX iterator
  PAXBlockIterator* iter = new PAXBlockIterator();
  iter->Init({}, BytewiseComparator(), kDisableGlobalSequenceNumber,
             block_copy.data(), /*keys_are_internal=*/false);

  // Test SeekToFirst
  iter->SeekToFirst();
  ASSERT_TRUE(iter->Valid());
  ASSERT_EQ(iter->key().ToString(), "key00");
  ASSERT_EQ(iter->value().ToString(), "value00");

  // Test Next
  iter->Next();
  ASSERT_TRUE(iter->Valid());
  ASSERT_EQ(iter->key().ToString(), "key01");
  ASSERT_EQ(iter->value().ToString(), "value01");

  // Test full iteration
  int count = 0;
  for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
    count++;
  }
  ASSERT_EQ(count, 25);
  ASSERT_OK(iter->status());

  // Test SeekToLast
  iter->SeekToLast();
  ASSERT_TRUE(iter->Valid());
  ASSERT_EQ(iter->key().ToString(), "key24");
  ASSERT_EQ(iter->value().ToString(), "value24");

  // Test Seek to middle and forward iteration from the seek target.
  iter->Seek("key15");
  ASSERT_TRUE(iter->Valid());
  ASSERT_EQ(iter->key().ToString(), "key15");

  count = 0;
  for (; iter->Valid(); iter->Next()) {
    count++;
  }
  ASSERT_EQ(count, 10);  // key15 through key24
  ASSERT_OK(iter->status());

  // Test Prev after SeekToLast to cover reverse movement in the PAX iterator.
  iter->SeekToLast();
  iter->Prev();
  ASSERT_TRUE(iter->Valid());
  ASSERT_EQ(iter->key().ToString(), "key23");
  ASSERT_OK(iter->status());

  delete iter;
}

// Verifies that the custom block format preserves empty values by serializing
// empty-value entries and scanning them back through the PAX iterator.
TEST_F(UserDefinedBlockTest, PAXBlockEmptyValues) {
  PAXBlockBuilder builder(5);

  // Add keys with empty values
  for (int i = 0; i < 10; i++) {
    std::string key = "key" + std::to_string(i);
    builder.Add(key, "");
  }

  Slice block_data = builder.Finish();
  std::string block_copy(block_data.data(), block_data.size());

  PAXBlockIterator* iter = new PAXBlockIterator();
  iter->Init({}, BytewiseComparator(), kDisableGlobalSequenceNumber,
             block_copy.data(), /*keys_are_internal=*/false);

  int count = 0;
  for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
    ASSERT_EQ(iter->value().ToString(), "");
    count++;
  }
  ASSERT_EQ(count, 10);
  ASSERT_OK(iter->status());

  delete iter;
}

// Verifies that the custom block format preserves large values by serializing
// 10KB values and checking their sizes through the PAX iterator.
TEST_F(UserDefinedBlockTest, PAXBlockLargeValues) {
  PAXBlockBuilder builder(3);

  // Add keys with large values
  for (int i = 0; i < 10; i++) {
    std::string key = "key" + std::to_string(i);
    std::string value(10240, 'x');  // 10KB value
    builder.Add(key, value);
  }

  Slice block_data = builder.Finish();
  std::string block_copy(block_data.data(), block_data.size());

  PAXBlockIterator* iter = new PAXBlockIterator();
  iter->Init({}, BytewiseComparator(), kDisableGlobalSequenceNumber,
             block_copy.data(), /*keys_are_internal=*/false);

  int count = 0;
  for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
    ASSERT_EQ(iter->value().size(), 10240);
    count++;
  }
  ASSERT_EQ(count, 10);
  ASSERT_OK(iter->status());

  delete iter;
}

// Verifies SstFileWriter/SstFileReader integration by writing a custom-format
// file, ingesting it through the test helper, and checking seek/scan behavior.
TEST_F(UserDefinedBlockTest, BasicTest) { BasicTest(); }

// Verifies dictionary-compression configurations do not route custom-format
// blocks through the buffered replay path that expects built-in block encoding.
TEST_F(UserDefinedBlockTest, DictCompressionDoesNotBufferCustomBlocks) {
  if (GetSupportedDictCompressions().empty()) {
    ROCKSDB_GTEST_SKIP("No supported dict compression");
    return;
  }

  BlockBasedTableOptions table_options;
  std::string dbname = test::PerThreadDBPath("user_defined_block_test");
  std::string ingest_file = dbname + "test_dict_compression.sst";

  auto user_defined_block_factory =
      std::make_shared<TestUserDefinedBlockFactory>();
  table_options.user_defined_block_factory = user_defined_block_factory;
  table_options.flush_block_policy_factory =
      std::make_shared<CustomFlushBlockPolicyFactory>(10);

  options_.table_factory.reset(NewBlockBasedTableFactory(table_options));
  options_.compression = GetSupportedDictCompressions()[0];
  options_.compression_opts.max_dict_bytes = 1024;
  options_.compression_opts.max_dict_buffer_bytes = 1024;

  std::unique_ptr<SstFileWriter> writer;
  writer.reset(new SstFileWriter(EnvOptions(), options_));
  ASSERT_OK(writer->Open(ingest_file));

  auto kvs = generateKVs(/*key_count*/ 200, /*value_size*/ 256);
  for (const auto& kv : kvs) {
    ASSERT_OK(writer->Put(kv.first, kv.second));
  }
  ASSERT_OK(writer->Finish());
  writer.reset();

  std::unique_ptr<SstFileReader> reader(new SstFileReader(options_));
  ASSERT_OK(reader->Open(ingest_file));

  ReadOptions ro;
  std::unique_ptr<Iterator> iter(reader->NewIterator(ro));
  ASSERT_NE(iter, nullptr);

  int key_count = 0;
  for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
    key_count++;
  }
  ASSERT_EQ(key_count, 200);
  ASSERT_OK(iter->status());
}

// Verifies direct SST create/read support by writing a custom-format file and
// checking first/last/previous/seek-for-prev iterator positioning through
// SstFileReader.
TEST_F(UserDefinedBlockTest, CreateAndReadBlock) {
  BlockBasedTableOptions table_options;
  std::string dbname = test::PerThreadDBPath("user_defined_block_test");
  std::string ingest_file = dbname + "test2.sst";

  auto user_defined_block_factory =
      std::make_shared<TestUserDefinedBlockFactory>();
  table_options.user_defined_block_factory = user_defined_block_factory;

  options_.table_factory.reset(NewBlockBasedTableFactory(table_options));

  std::unique_ptr<SstFileWriter> writer;
  writer.reset(new SstFileWriter(EnvOptions(), options_));
  ASSERT_OK(writer->Open(ingest_file));

  auto kvs = generateKVs(/*key_count*/ 50, /*value_size*/ 100);
  for (const auto& kv : kvs) {
    ASSERT_OK(writer->Put(kv.first, kv.second));
  }
  ASSERT_OK(writer->Finish());
  writer.reset();

  std::unique_ptr<SstFileReader> reader(new SstFileReader(options_));
  ASSERT_OK(reader->Open(ingest_file));

  ReadOptions ro;
  std::unique_ptr<Iterator> iter(reader->NewIterator(ro));
  ASSERT_NE(iter, nullptr);

  // Test SeekToFirst
  iter->SeekToFirst();
  ASSERT_TRUE(iter->Valid());
  ASSERT_EQ(iter->key().ToString(), "key00");

  // Test SeekToLast
  iter->SeekToLast();
  ASSERT_TRUE(iter->Valid());
  ASSERT_EQ(iter->key().ToString(), "key49");

  // Test Prev
  iter->Prev();
  ASSERT_TRUE(iter->Valid());
  ASSERT_EQ(iter->key().ToString(), "key48");

  // Test SeekForPrev on an exact key and a key between two existing keys.
  iter->SeekForPrev("key25");
  ASSERT_TRUE(iter->Valid());
  ASSERT_EQ(iter->key().ToString(), "key25");

  iter->SeekForPrev("key25a");
  ASSERT_TRUE(iter->Valid());
  ASSERT_EQ(iter->key().ToString(), "key25");
}

// Verifies large-value SST reads by writing 10KB values through the custom
// builder and checking that every value is recovered at its original size.
TEST_F(UserDefinedBlockTest, LargeValues) {
  BlockBasedTableOptions table_options;
  std::string dbname = test::PerThreadDBPath("user_defined_block_test");
  std::string ingest_file = dbname + "test_large_values.sst";

  auto user_defined_block_factory =
      std::make_shared<TestUserDefinedBlockFactory>();
  table_options.user_defined_block_factory = user_defined_block_factory;

  options_.table_factory.reset(NewBlockBasedTableFactory(table_options));

  std::unique_ptr<SstFileWriter> writer;
  writer.reset(new SstFileWriter(EnvOptions(), options_));
  ASSERT_OK(writer->Open(ingest_file));

  // Create 20 keys with large values (10KB each)
  auto kvs = generateKVs(/*key_count*/ 20, /*value_size*/ 10240);
  for (const auto& kv : kvs) {
    ASSERT_OK(writer->Put(kv.first, kv.second));
  }
  ASSERT_OK(writer->Finish());
  writer.reset();

  std::unique_ptr<SstFileReader> reader(new SstFileReader(options_));
  ASSERT_OK(reader->Open(ingest_file));

  ReadOptions ro;
  std::unique_ptr<Iterator> iter(reader->NewIterator(ro));
  ASSERT_NE(iter, nullptr);

  // Test that we can read all keys with large values
  int key_count = 0;
  for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
    key_count++;
    ASSERT_EQ(iter->value().size(), 10240);
  }
  ASSERT_EQ(key_count, 20);
  ASSERT_OK(iter->status());
}

// Verifies multi-block custom SST scans by forcing small data blocks, scanning
// 1000 keys, and seeking into the middle of the file.
TEST_F(UserDefinedBlockTest, ManyKeys) {
  BlockBasedTableOptions table_options;
  std::string dbname = test::PerThreadDBPath("user_defined_block_test");
  std::string ingest_file = dbname + "test_many_keys.sst";

  auto user_defined_block_factory =
      std::make_shared<TestUserDefinedBlockFactory>();
  table_options.user_defined_block_factory = user_defined_block_factory;

  // Use custom flush policy with more keys per block
  table_options.flush_block_policy_factory =
      std::make_shared<CustomFlushBlockPolicyFactory>(10);

  options_.table_factory.reset(NewBlockBasedTableFactory(table_options));

  std::unique_ptr<SstFileWriter> writer;
  writer.reset(new SstFileWriter(EnvOptions(), options_));
  ASSERT_OK(writer->Open(ingest_file));

  // Create 1000 keys
  auto kvs = generateKVs(/*key_count*/ 1000);
  for (const auto& kv : kvs) {
    ASSERT_OK(writer->Put(kv.first, kv.second));
  }
  ASSERT_OK(writer->Finish());
  writer.reset();

  std::unique_ptr<SstFileReader> reader(new SstFileReader(options_));
  ASSERT_OK(reader->Open(ingest_file));

  ReadOptions ro;
  std::unique_ptr<Iterator> iter(reader->NewIterator(ro));
  ASSERT_NE(iter, nullptr);

  // Test full iteration
  int key_count = 0;
  for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
    key_count++;
  }
  ASSERT_EQ(key_count, 1000);
  ASSERT_OK(iter->status());

  // Test seeking to middle
  iter->Seek("key0500");
  ASSERT_TRUE(iter->Valid());
  ASSERT_EQ(iter->key().ToString(), "key0500");

  // Count keys from middle to end
  key_count = 0;
  for (; iter->Valid(); iter->Next()) {
    key_count++;
  }
  ASSERT_EQ(key_count, 500);
  ASSERT_OK(iter->status());
}

// Verifies empty-value SST support by writing empty values through the custom
// builder and scanning them back through SstFileReader.
TEST_F(UserDefinedBlockTest, EmptyValue) {
  BlockBasedTableOptions table_options;
  std::string dbname = test::PerThreadDBPath("user_defined_block_test");
  std::string ingest_file = dbname + "test_empty_value.sst";

  auto user_defined_block_factory =
      std::make_shared<TestUserDefinedBlockFactory>();
  table_options.user_defined_block_factory = user_defined_block_factory;

  options_.table_factory.reset(NewBlockBasedTableFactory(table_options));

  std::unique_ptr<SstFileWriter> writer;
  writer.reset(new SstFileWriter(EnvOptions(), options_));
  ASSERT_OK(writer->Open(ingest_file));

  // Create keys with empty values
  auto kvs = generateKVWithValue(/*key_count*/ 30, /*value*/ "");
  for (const auto& kv : kvs) {
    ASSERT_OK(writer->Put(kv.first, kv.second));
  }
  ASSERT_OK(writer->Finish());
  writer.reset();

  std::unique_ptr<SstFileReader> reader(new SstFileReader(options_));
  ASSERT_OK(reader->Open(ingest_file));

  ReadOptions ro;
  std::unique_ptr<Iterator> iter(reader->NewIterator(ro));
  ASSERT_NE(iter, nullptr);

  // Test that we can read all keys with empty values
  int key_count = 0;
  for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
    key_count++;
    ASSERT_EQ(iter->value().ToString(), "");
  }
  ASSERT_EQ(key_count, 30);
  ASSERT_OK(iter->status());
}

// Verifies DB-level integration by writing, flushing, reading, updating,
// deleting, reopening, and checking persistence through custom data blocks.
TEST_F(UserDefinedBlockTest, DBReadWriteFlush) {
  BlockBasedTableOptions table_options;
  std::string dbname = test::PerThreadDBPath("user_defined_block_db_test");

  // Set up the user-defined block factory
  auto user_defined_block_factory =
      std::make_shared<TestUserDefinedBlockFactory>();
  table_options.user_defined_block_factory = user_defined_block_factory;

  // Set up custom flush block policy that flushes every 10 keys
  table_options.flush_block_policy_factory =
      std::make_shared<CustomFlushBlockPolicyFactory>(10);

  options_.table_factory.reset(NewBlockBasedTableFactory(table_options));
  options_.compression = kNoCompression;
  options_.create_if_missing = true;
  options_.write_buffer_size = 1024 * 1024;  // 1MB

  // Clean up any existing DB
  ASSERT_OK(DestroyDB(dbname, options_));

  // Open DB
  std::unique_ptr<DB> db;
  Status s = DB::Open(options_, dbname, &db);
  ASSERT_OK(s);
  ASSERT_NE(db.get(), nullptr);

  // Test 1: Basic Put and Seek operations
  WriteOptions write_options;
  ReadOptions read_options;

  // Write some data
  for (int i = 0; i < 100; i++) {
    std::stringstream ss;
    ss << std::setw(3) << std::setfill('0') << i;
    std::string key = "key" + ss.str();
    std::string value = "value" + ss.str();
    ASSERT_OK(db->Put(write_options, key, value));
  }

  // Read back the data from memtable using Seek
  Iterator* iter = db->NewIterator(read_options);
  for (int i = 0; i < 100; i++) {
    std::stringstream ss;
    ss << std::setw(3) << std::setfill('0') << i;
    std::string key = "key" + ss.str();
    std::string expected_value = "value" + ss.str();

    iter->Seek(key);
    ASSERT_TRUE(iter->Valid()) << "Failed to find key: " << key;
    ASSERT_EQ(iter->key().ToString(), key);
    ASSERT_EQ(iter->value().ToString(), expected_value);
  }
  ASSERT_OK(iter->status());
  delete iter;

  // Test 2: Flush and read from SST
  FlushOptions flush_options;
  flush_options.wait = true;
  ASSERT_OK(db->Flush(flush_options));

  // Read back the data from SST file with user-defined blocks using Seek
  iter = db->NewIterator(read_options);
  for (int i = 0; i < 100; i++) {
    std::stringstream ss;
    ss << std::setw(3) << std::setfill('0') << i;
    std::string key = "key" + ss.str();
    std::string expected_value = "value" + ss.str();

    iter->Seek(key);
    ASSERT_TRUE(iter->Valid()) << "Failed to find key: " << key;
    ASSERT_EQ(iter->key().ToString(), key);
    ASSERT_EQ(iter->value().ToString(), expected_value);
  }
  ASSERT_OK(iter->status());
  delete iter;

  // Test 3: Update existing keys and flush again
  for (int i = 0; i < 50; i++) {
    std::stringstream ss;
    ss << std::setw(3) << std::setfill('0') << i;
    std::string key = "key" + ss.str();
    std::string value = "updated_value" + ss.str();
    ASSERT_OK(db->Put(write_options, key, value));
  }

  ASSERT_OK(db->Flush(flush_options));

  // Verify updates using Seek
  iter = db->NewIterator(read_options);
  for (int i = 0; i < 50; i++) {
    std::stringstream ss;
    ss << std::setw(3) << std::setfill('0') << i;
    std::string key = "key" + ss.str();
    std::string expected_value = "updated_value" + ss.str();

    iter->Seek(key);
    ASSERT_TRUE(iter->Valid()) << "Failed to find key: " << key;
    ASSERT_EQ(iter->key().ToString(), key);
    ASSERT_EQ(iter->value().ToString(), expected_value);
  }

  // Verify non-updated keys using Seek
  for (int i = 50; i < 100; i++) {
    std::stringstream ss;
    ss << std::setw(3) << std::setfill('0') << i;
    std::string key = "key" + ss.str();
    std::string expected_value = "value" + ss.str();

    iter->Seek(key);
    ASSERT_TRUE(iter->Valid()) << "Failed to find key: " << key;
    ASSERT_EQ(iter->key().ToString(), key);
    ASSERT_EQ(iter->value().ToString(), expected_value);
  }
  ASSERT_OK(iter->status());
  delete iter;

  // Test 4: Iterator scan over user-defined blocks
  iter = db->NewIterator(read_options);
  ASSERT_NE(iter, nullptr);

  int key_count = 0;
  for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
    key_count++;
  }
  ASSERT_EQ(key_count, 100);
  ASSERT_OK(iter->status());

  // Test seek to specific key
  iter->Seek("key050");
  ASSERT_TRUE(iter->Valid());
  ASSERT_EQ(iter->key().ToString(), "key050");
  ASSERT_EQ(iter->value().ToString(), "value050");

  // Count remaining keys
  key_count = 0;
  for (; iter->Valid(); iter->Next()) {
    key_count++;
  }
  ASSERT_EQ(key_count, 50);
  ASSERT_OK(iter->status());

  delete iter;

  // Test 5: Delete operations
  for (int i = 0; i < 20; i++) {
    std::stringstream ss;
    ss << std::setw(3) << std::setfill('0') << i;
    std::string key = "key" + ss.str();
    ASSERT_OK(db->Delete(write_options, key));
  }

  ASSERT_OK(db->Flush(flush_options));

  // Verify deletions using Seek
  iter = db->NewIterator(read_options);
  for (int i = 0; i < 20; i++) {
    std::stringstream ss;
    ss << std::setw(3) << std::setfill('0') << i;
    std::string key = "key" + ss.str();

    iter->Seek(key);
    // After deletion, Seek should either go to the next key or be invalid
    if (iter->Valid()) {
      ASSERT_NE(iter->key().ToString(), key)
          << "Deleted key " << key << " should not be found";
    }
  }

  // Verify non-deleted keys still exist using Seek
  for (int i = 20; i < 100; i++) {
    std::stringstream ss;
    ss << std::setw(3) << std::setfill('0') << i;
    std::string key = "key" + ss.str();

    iter->Seek(key);
    ASSERT_TRUE(iter->Valid()) << "Failed to find key: " << key;
    ASSERT_EQ(iter->key().ToString(), key);
  }
  ASSERT_OK(iter->status());
  delete iter;

  // Test 6: Large values
  std::string large_value(10000, 'x');
  for (int i = 0; i < 10; i++) {
    std::stringstream ss;
    ss << "large_key" << i;
    std::string key = ss.str();
    ASSERT_OK(db->Put(write_options, key, large_value));
  }

  ASSERT_OK(db->Flush(flush_options));

  // Verify large values using Seek
  iter = db->NewIterator(read_options);
  for (int i = 0; i < 10; i++) {
    std::stringstream ss;
    ss << "large_key" << i;
    std::string key = ss.str();

    iter->Seek(key);
    ASSERT_TRUE(iter->Valid()) << "Failed to find key: " << key;
    ASSERT_EQ(iter->key().ToString(), key);
    ASSERT_EQ(iter->value().ToString(), large_value);
  }
  ASSERT_OK(iter->status());
  delete iter;

  // Clean up
  db.reset();

  // Test 7: Reopen and verify persistence
  ASSERT_OK(DB::Open(options_, dbname, &db));
  ASSERT_NE(db.get(), nullptr);

  // Verify data persisted correctly using Seek
  iter = db->NewIterator(read_options);
  for (int i = 20; i < 100; i++) {
    std::stringstream ss;
    ss << std::setw(3) << std::setfill('0') << i;
    std::string key = "key" + ss.str();

    iter->Seek(key);
    ASSERT_TRUE(iter->Valid()) << "Failed to find key: " << key;
    ASSERT_EQ(iter->key().ToString(), key);
  }

  // Verify large values using Seek
  for (int i = 0; i < 10; i++) {
    std::stringstream ss;
    ss << "large_key" << i;
    std::string key = ss.str();

    iter->Seek(key);
    ASSERT_TRUE(iter->Valid()) << "Failed to find key: " << key;
    ASSERT_EQ(iter->key().ToString(), key);
    ASSERT_EQ(iter->value().ToString(), large_value);
  }
  ASSERT_OK(iter->status());
  delete iter;

  // Final cleanup
  db.reset();
  ASSERT_OK(DestroyDB(dbname, options_));
}

// Verifies that external SST ingestion with UDB applies the global sequence
// number assigned during ingestion. The test takes a snapshot before ingesting
// a UDB SST and checks that the snapshot cannot see the ingested key.
TEST_F(UserDefinedBlockTest, IngestedFileRespectsGlobalSeqno) {
  BlockBasedTableOptions table_options;
  std::string dbname =
      test::PerThreadDBPath("user_defined_block_ingest_global_seqno");
  std::string ingest_file = dbname + "_ingest.sst";

  auto user_defined_block_factory =
      std::make_shared<TestUserDefinedBlockFactory>();
  table_options.user_defined_block_factory = user_defined_block_factory;

  options_.create_if_missing = true;
  options_.compression = kNoCompression;
  options_.table_factory.reset(NewBlockBasedTableFactory(table_options));

  ASSERT_OK(DestroyDB(dbname, options_));

  std::unique_ptr<DB> db;
  ASSERT_OK(DB::Open(options_, dbname, &db));

  ASSERT_OK(db->Put(WriteOptions(), "existing", "value0"));
  const Snapshot* snapshot = db->GetSnapshot();
  ASSERT_NE(snapshot, nullptr);

  std::unique_ptr<SstFileWriter> writer;
  writer.reset(new SstFileWriter(EnvOptions(), options_));
  ASSERT_OK(writer->Open(ingest_file));
  ASSERT_OK(writer->Put("ingested", "value1"));
  ASSERT_OK(writer->Finish());
  writer.reset();

  IngestExternalFileOptions ifo;
  ASSERT_OK(db->IngestExternalFile({ingest_file}, ifo));

  std::string value;
  ReadOptions snapshot_read_options;
  snapshot_read_options.snapshot = snapshot;
  ASSERT_TRUE(db->Get(snapshot_read_options, "ingested", &value).IsNotFound());

  ReadOptions latest_read_options;
  ASSERT_OK(db->Get(latest_read_options, "ingested", &value));
  ASSERT_EQ(value, "value1");

  db->ReleaseSnapshot(snapshot);
  db.reset();
  ASSERT_OK(DestroyDB(dbname, options_));
}

// Verifies that option validation rejects UDB with block per-key protection.
// UDB iterators own key/value interpretation, so RocksDB cannot enforce the
// built-in block_protection_bytes_per_key checksum contract for them.
TEST_F(UserDefinedBlockTest, BlockProtectionBytesPerKeyUnsupported) {
  BlockBasedTableOptions table_options;
  table_options.user_defined_block_factory =
      std::make_shared<TestUserDefinedBlockFactory>();

  std::unique_ptr<TableFactory> table_factory;
  table_factory.reset(NewBlockBasedTableFactory(table_options));

  Options options;
  options.block_protection_bytes_per_key = 8;
  Status s = table_factory->ValidateOptions(DBOptions(options),
                                            ColumnFamilyOptions(options));
  ASSERT_TRUE(s.IsNotSupported());
  ASSERT_NE(s.ToString().find("user_defined_block_factory"), std::string::npos);
}

// Verifies option validation rejects UDB with separated key/value blocks. UDB
// blocks own their physical layout, so the built-in separated-KV data block
// contract cannot be mixed with a custom block format.
TEST_F(UserDefinedBlockTest, SeparateKeyValueUnsupported) {
  BlockBasedTableOptions table_options;
  table_options.user_defined_block_factory =
      std::make_shared<TestUserDefinedBlockFactory>();
  table_options.separate_key_value_in_data_block = true;

  std::unique_ptr<TableFactory> table_factory;
  table_factory.reset(NewBlockBasedTableFactory(table_options));

  Options options;
  Status s = table_factory->ValidateOptions(DBOptions(options),
                                            ColumnFamilyOptions(options));
  ASSERT_TRUE(s.IsNotSupported());
  ASSERT_NE(s.ToString().find("separate_key_value_in_data_block"),
            std::string::npos);
}

// Verifies validation rejects a UDB factory with no stable identity. The table
// property written for UDB files depends on Name(), so an empty name would make
// factory mismatch checks unenforceable.
TEST_F(UserDefinedBlockTest, EmptyFactoryNameUnsupported) {
  class EmptyNameFactory : public TestUserDefinedBlockFactory {
   public:
    const char* Name() const override { return ""; }
  };

  BlockBasedTableOptions table_options;
  table_options.user_defined_block_factory =
      std::make_shared<EmptyNameFactory>();

  std::unique_ptr<TableFactory> table_factory;
  table_factory.reset(NewBlockBasedTableFactory(table_options));

  Options options;
  Status s = table_factory->ValidateOptions(DBOptions(options),
                                            ColumnFamilyOptions(options));
  ASSERT_TRUE(s.IsInvalidArgument());
  ASSERT_NE(s.ToString().find("non-empty Name"), std::string::npos);
}

// Verifies UDB SSTs record the writing factory identity. Opening the file
// without a factory or with a different factory must fail before any block is
// decoded by the wrong format implementation.
TEST_F(UserDefinedBlockTest, OpenRequiresMatchingFactory) {
  class DifferentNameFactory : public TestUserDefinedBlockFactory {
   public:
    const char* Name() const override { return "different_name_udb"; }
  };

  BlockBasedTableOptions table_options;
  table_options.user_defined_block_factory =
      std::make_shared<TestUserDefinedBlockFactory>();

  options_.compression = kNoCompression;
  options_.table_factory.reset(NewBlockBasedTableFactory(table_options));

  std::string sst_file =
      test::PerThreadDBPath("user_defined_block_factory_identity.sst");
  SstFileWriter writer(EnvOptions(), options_);
  ASSERT_OK(writer.Open(sst_file));
  ASSERT_OK(writer.Put("key", "value"));
  ASSERT_OK(writer.Finish());

  SstFileReader matching_reader(options_);
  ASSERT_OK(matching_reader.Open(sst_file));

  Options no_udb_options;
  SstFileReader no_udb_reader(no_udb_options);
  Status s = no_udb_reader.Open(sst_file);
  ASSERT_TRUE(s.IsInvalidArgument()) << s.ToString();
  ASSERT_NE(s.ToString().find("user_defined_block_factory"), std::string::npos);

  BlockBasedTableOptions mismatch_table_options;
  mismatch_table_options.user_defined_block_factory =
      std::make_shared<DifferentNameFactory>();
  Options mismatch_options;
  mismatch_options.table_factory.reset(
      NewBlockBasedTableFactory(mismatch_table_options));
  SstFileReader mismatch_reader(mismatch_options);
  s = mismatch_reader.Open(sst_file);
  ASSERT_TRUE(s.IsInvalidArgument()) << s.ToString();
  ASSERT_NE(s.ToString().find("mismatch"), std::string::npos);
}

// Verifies a standard-format SST remains readable when a UDB factory is
// configured. The table property, not the option alone, decides whether RocksDB
// routes data blocks through the custom parser.
TEST_F(UserDefinedBlockTest, StandardSstWithConfiguredFactoryUsesDefaultBlock) {
  class FailIfUsedFactory : public TestUserDefinedBlockFactory {
   public:
    const char* Name() const override { return "fail_if_used_udb"; }

    Status NewIterator(const UserDefinedBlockOption& /*option*/,
                       DataBlockIter** /*biter*/) const override {
      return Status::InvalidArgument("UDB iterator should not be used");
    }

    Status NewBlock(
        const UserDefinedBlockOption& /*option*/,
        std::unique_ptr<UserDefinedBlock>* /*block*/) const override {
      return Status::InvalidArgument("UDB block should not be used");
    }
  };

  Options standard_options;
  standard_options.compression = kNoCompression;

  std::string sst_file =
      test::PerThreadDBPath("standard_sst_with_configured_udb_factory.sst");
  SstFileWriter writer(EnvOptions(), standard_options);
  ASSERT_OK(writer.Open(sst_file));
  ASSERT_OK(writer.Put("key1", "value1"));
  ASSERT_OK(writer.Put("key2", "value2"));
  ASSERT_OK(writer.Finish());

  BlockBasedTableOptions table_options;
  table_options.user_defined_block_factory =
      std::make_shared<FailIfUsedFactory>();
  Options udb_options;
  udb_options.table_factory.reset(NewBlockBasedTableFactory(table_options));

  SstFileReader reader(udb_options);
  ASSERT_OK(reader.Open(sst_file));
  std::unique_ptr<Iterator> iter(reader.NewIterator(ReadOptions()));
  ASSERT_NE(iter, nullptr);

  iter->SeekToFirst();
  ASSERT_TRUE(iter->Valid());
  ASSERT_EQ(iter->key().ToString(), "key1");
  ASSERT_EQ(iter->value().ToString(), "value1");
  iter->Next();
  ASSERT_TRUE(iter->Valid());
  ASSERT_EQ(iter->key().ToString(), "key2");
  ASSERT_EQ(iter->value().ToString(), "value2");
  iter->Next();
  ASSERT_FALSE(iter->Valid());
  ASSERT_OK(iter->status());
}

// Verifies a factory that returns OK with a null reusable iterator fails reads
// cleanly. The test writes a UDB SST with that factory, then checks the first
// read reports InvalidArgument instead of falling back to the built-in parser.
TEST_F(UserDefinedBlockTest, NullFactoryIteratorFailsRead) {
  class NullIteratorFactory : public TestUserDefinedBlockFactory {
   public:
    const char* Name() const override { return "null_iterator_udb"; }

    Status NewIterator(const UserDefinedBlockOption& /*option*/,
                       DataBlockIter** biter) const override {
      *biter = nullptr;
      return Status::OK();
    }
  };

  BlockBasedTableOptions table_options;
  table_options.user_defined_block_factory =
      std::make_shared<NullIteratorFactory>();

  options_.compression = kNoCompression;
  options_.table_factory.reset(NewBlockBasedTableFactory(table_options));

  std::string sst_file =
      test::PerThreadDBPath("user_defined_block_null_factory_iterator.sst");
  SstFileWriter writer(EnvOptions(), options_);
  ASSERT_OK(writer.Open(sst_file));
  ASSERT_OK(writer.Put("key", "value"));
  ASSERT_OK(writer.Finish());

  SstFileReader reader(options_);
  ASSERT_OK(reader.Open(sst_file));

  std::unique_ptr<Iterator> iter(reader.NewIterator(ReadOptions()));
  ASSERT_NE(iter, nullptr);
  iter->SeekToFirst();
  ASSERT_FALSE(iter->Valid());
  ASSERT_TRUE(iter->status().IsInvalidArgument()) << iter->status().ToString();
  ASSERT_NE(iter->status().ToString().find("null block iterator"),
            std::string::npos);
}

// Verifies a custom block that returns a null initialized iterator fails reads
// cleanly. The test reaches the UserDefinedBlock wrapper path and checks it
// converts the null return into InvalidArgument instead of dereferencing it.
TEST_F(UserDefinedBlockTest, NullBlockIteratorFailsRead) {
  class NullDataIteratorBlock : public PAXBlock {
   public:
    DataBlockIter* NewDataIterator(
        const Comparator* /*raw_ucmp*/, SequenceNumber /*global_seqno*/,
        DataBlockIter* /*input_iter*/, Statistics* /*stats*/,
        bool /*block_contents_pinned*/,
        bool /*user_defined_timestamps_persisted*/,
        void* /*user_defined_block_iterator_arg*/) override {
      return nullptr;
    }
  };

  class NullDataIteratorFactory : public TestUserDefinedBlockFactory {
   public:
    const char* Name() const override { return "null_data_iterator_udb"; }

    Status NewBlock(const UserDefinedBlockOption& /*option*/,
                    std::unique_ptr<UserDefinedBlock>* block) const override {
      *block = std::make_unique<NullDataIteratorBlock>();
      return Status::OK();
    }
  };

  BlockBasedTableOptions table_options;
  table_options.user_defined_block_factory =
      std::make_shared<NullDataIteratorFactory>();

  options_.compression = kNoCompression;
  options_.table_factory.reset(NewBlockBasedTableFactory(table_options));

  std::string sst_file =
      test::PerThreadDBPath("user_defined_block_null_block_iterator.sst");
  SstFileWriter writer(EnvOptions(), options_);
  ASSERT_OK(writer.Open(sst_file));
  ASSERT_OK(writer.Put("key", "value"));
  ASSERT_OK(writer.Finish());

  SstFileReader reader(options_);
  ASSERT_OK(reader.Open(sst_file));

  std::unique_ptr<Iterator> iter(reader.NewIterator(ReadOptions()));
  ASSERT_NE(iter, nullptr);
  iter->SeekToFirst();
  ASSERT_FALSE(iter->Valid());
  ASSERT_TRUE(iter->status().IsInvalidArgument()) << iter->status().ToString();
  ASSERT_NE(iter->status().ToString().find("returned null iterator"),
            std::string::npos);
}

// Verifies DB MultiGet goes through the user-defined data-block iterator
// rather than the default stack DataBlockIter used by the built-in format.
TEST_F(UserDefinedBlockTest, DBMultiGet) {
  BlockBasedTableOptions table_options;
  std::string dbname = test::PerThreadDBPath("user_defined_block_multiget");

  auto user_defined_block_factory =
      std::make_shared<TestUserDefinedBlockFactory>();
  table_options.user_defined_block_factory = user_defined_block_factory;
  table_options.flush_block_policy_factory =
      std::make_shared<CustomFlushBlockPolicyFactory>(10);

  options_.table_factory.reset(NewBlockBasedTableFactory(table_options));
  options_.compression = kNoCompression;
  options_.create_if_missing = true;

  ASSERT_OK(DestroyDB(dbname, options_));

  std::unique_ptr<DB> db;
  ASSERT_OK(DB::Open(options_, dbname, &db));
  ASSERT_NE(db.get(), nullptr);

  WriteOptions write_options;
  for (int i = 0; i < 100; i++) {
    std::stringstream ss;
    ss << std::setw(3) << std::setfill('0') << i;
    std::string key = "key" + ss.str();
    std::string value = "value" + ss.str();
    ASSERT_OK(db->Put(write_options, key, value));
  }

  FlushOptions flush_options;
  flush_options.wait = true;
  ASSERT_OK(db->Flush(flush_options));

  std::vector<Slice> keys{Slice("key000"), Slice("key009"), Slice("key010"),
                          Slice("key099"), Slice("missing")};
  std::vector<std::string> values;
  std::vector<Status> statuses = db->MultiGet(ReadOptions(), keys, &values);

  ASSERT_EQ(statuses.size(), keys.size());
  ASSERT_EQ(values.size(), keys.size());
  ASSERT_OK(statuses[0]);
  ASSERT_EQ(values[0], "value000");
  ASSERT_OK(statuses[1]);
  ASSERT_EQ(values[1], "value009");
  ASSERT_OK(statuses[2]);
  ASSERT_EQ(values[2], "value010");
  ASSERT_OK(statuses[3]);
  ASSERT_EQ(values[3], "value099");
  ASSERT_TRUE(statuses[4].IsNotFound());

  db.reset();
  ASSERT_OK(DestroyDB(dbname, options_));
}

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
