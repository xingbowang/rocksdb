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

static const bool kVerbose = false;

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
    CustomFlushBlockPolicyFactory(int keys_per_block = 3)
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

    void Initialize(const Comparator* raw_ucmp, const char* data,
                    uint32_t /*restarts*/, uint32_t /*num_restarts*/,
                    SequenceNumber /*global_seqno*/,
                    BlockReadAmpBitmap* /*read_amp_bitmap*/,
                    bool /*block_contents_pinned*/,
                    bool /*user_defined_timestamps_persisted*/,
                    DataBlockHashIndex* /*data_block_hash_index*/,
                    uint8_t /*protection_bytes_per_key*/,
                    const char* /*kv_checksum*/,
                    uint32_t /*block_restart_interval*/) override {
      comparator_ = raw_ucmp;
      data_ = data;
      global_seqno_ = kDisableGlobalSequenceNumber;
      status_ = Status::OK();
      ParsePAXBlock();
      SeekToFirstImpl();
    }

    Slice value() const override {
      if (!Valid()) return Slice();
      return current_value_;
    }

   protected:
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
      if (records_.empty()) {
        current_ = restarts_;
        return;
      }

      // Binary search through records
      uint32_t left = 0;
      uint32_t right = static_cast<uint32_t>(records_.size());

      while (left < right) {
        uint32_t mid = (left + right) / 2;
        int cmp = target.compare(records_[mid].first);
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

      // If current key is greater than target, move to previous
      if (comparator_->Compare(key_, target) > 0) {
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
      raw_key_.SetKey(key_, false);
    }

    const Comparator* comparator_;
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

      // Flush minipage when it reaches the configured size
      if (current_minipage_records_ >= minipage_size_) {
        FlushMinipage();
      }
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

    void Reset() override {
      BlockBuilder::Reset();
      keys_.clear();
      values_.clear();
      minipages_.clear();
      buffer_.clear();
      current_minipage_records_ = 0;
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
    std::vector<std::string> keys_;
    std::vector<std::string> values_;
    std::vector<std::string> minipages_;
    std::string buffer_;
  };

  // PAXBlock: Implements UserDefinedBlock interface for columnar storage
  class PAXBlock : public UserDefinedBlock {
   public:
    explicit PAXBlock(BlockContents&& contents,
                      const Comparator* /*comparator*/)
        : contents_(std::move(contents)) {}

    ~PAXBlock() override = default;

    // Parse PAX format from raw bytes
    Status Parse() {
      // The data is already parsed in the iterator, just validate
      if (contents_.data.empty()) {
        return Status::Corruption("Empty block contents");
      }
      return Status::OK();
    }

    // UserDefinedBlock interface
    size_t ApproximateMemoryUsage() const override {
      return contents_.ApproximateMemoryUsage() + sizeof(*this);
    }

    const Slice& ContentSlice() const override { return contents_.data; }

    size_t size() const override { return contents_.data.size(); }

    const char* data() const override { return contents_.data.data(); }

    bool own_bytes() const override { return contents_.own_bytes(); }

    DataBlockIter* NewDataIterator(
        const Comparator* raw_ucmp, SequenceNumber global_seqno,
        DataBlockIter* input_iter = nullptr, Statistics* stats = nullptr,
        bool block_contents_pinned = false,
        bool user_defined_timestamps_persisted = true) override {
      (void)stats;
      PAXBlockIterator* iter = nullptr;
      if (input_iter != nullptr) {
        // Reuse existing iterator if provided
        // TODO : fix this
        delete input_iter;
        iter = new PAXBlockIterator();
      } else {
        iter = new PAXBlockIterator();
      }

      // Initialize the iterator with block data
      iter->Initialize(raw_ucmp, contents_.data.data(), 0, 0, global_seqno,
                       nullptr, block_contents_pinned,
                       user_defined_timestamps_persisted, nullptr, 0, nullptr,
                       1);

      return iter;
    }

   private:
    BlockContents contents_;
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

    // Legacy API - not used when UsesCustomBlockFormat() returns true
    Status NewIterator(const UserDefinedBlockOption& /*option*/,
                       DataBlockIter** biter) const override {
      *biter = new PAXBlockIterator();
      return Status::OK();
    }

    // NEW API: Create custom block from raw bytes
    Status NewBlock(const UserDefinedBlockOption& option,
                    BlockContents&& contents,
                    std::unique_ptr<UserDefinedBlock>* block) const override {
      auto pax_block =
          std::make_unique<PAXBlock>(std::move(contents), option.comparator);

      Status s = pax_block->Parse();
      if (!s.ok()) {
        return s;
      }

      *block = std::move(pax_block);
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
    if (kVerbose) {
      printf("key: %s, value: %s\n", iter->key().ToString().c_str(),
             iter->value().ToString().c_str());
    }
  }
  ASSERT_EQ(key_count, 100);
  ASSERT_OK(iter->status());
  iter.reset();

  // Test seek to specific key
  iter.reset(reader->NewIterator(ro));
  ASSERT_NE(iter, nullptr);
  key_count = 0;
  for (iter->Seek("key40"); iter->Valid(); iter->Next()) {
    key_count++;
  }
  ASSERT_EQ(key_count, 60);
  ASSERT_OK(iter->status());

  // Test upper bound
  Slice ub("key75");
  ro.iterate_upper_bound = &ub;
  iter.reset(reader->NewIterator(ro));
  ASSERT_NE(iter, nullptr);

  key_count = 0;
  for (iter->Seek("key40"); iter->Valid(); iter->Next()) {
    key_count++;
  }
  ASSERT_EQ(key_count, 35);
  ASSERT_OK(iter->status());
}

class UserDefinedBlockTest : public UserDefinedBlockTestBase {};

// Direct test of PAX Block Builder and Iterator (without SST infrastructure)
TEST_F(UserDefinedBlockTest, PAXBlockBuilderAndIteratorTest) {
  // Create PAX block builder
  PAXBlockBuilder builder(10);  // 10 records per minipage

  // Add 25 key-value pairs
  for (int i = 0; i < 25; i++) {
    std::string key = "key" + std::to_string(i);
    std::string value = "value" + std::to_string(i);
    builder.Add(key, value);
  }

  // Finish building
  Slice block_data = builder.Finish();
  ASSERT_GT(block_data.size(), 0);

  // Create a copy of the data for the iterator
  std::string block_copy(block_data.data(), block_data.size());

  // Create PAX iterator
  PAXBlockIterator* iter = new PAXBlockIterator();
  iter->Initialize(BytewiseComparator(), block_copy.data(), 0, 0, 0, nullptr,
                   true, true, nullptr, 0, nullptr, 1);

  // Test SeekToFirst
  iter->SeekToFirst();
  ASSERT_TRUE(iter->Valid());
  ASSERT_EQ(iter->key().ToString(), "key0");
  ASSERT_EQ(iter->value().ToString(), "value0");

  // Test Next
  iter->Next();
  ASSERT_TRUE(iter->Valid());
  ASSERT_EQ(iter->key().ToString(), "key1");
  ASSERT_EQ(iter->value().ToString(), "value1");

  // Test full iteration
  int count = 0;
  for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
    count++;
  }
  ASSERT_EQ(count, 25);

  // Test SeekToLast
  iter->SeekToLast();
  ASSERT_TRUE(iter->Valid());
  ASSERT_EQ(iter->key().ToString(), "key24");
  ASSERT_EQ(iter->value().ToString(), "value24");

  // Test Seek to middle
  iter->Seek("key15");
  ASSERT_TRUE(iter->Valid());
  ASSERT_EQ(iter->key().ToString(), "key15");

  // Count remaining
  count = 0;
  for (; iter->Valid(); iter->Next()) {
    count++;
  }
  ASSERT_EQ(count, 10);  // key15 through key24

  // Test Prev
  iter->SeekToLast();
  iter->Prev();
  ASSERT_TRUE(iter->Valid());
  ASSERT_EQ(iter->key().ToString(), "key23");

  delete iter;
}

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
  iter->Initialize(BytewiseComparator(), block_copy.data(), 0, 0, 0, nullptr,
                   true, true, nullptr, 0, nullptr, 1);

  int count = 0;
  for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
    ASSERT_EQ(iter->value().ToString(), "");
    count++;
  }
  ASSERT_EQ(count, 10);

  delete iter;
}

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
  iter->Initialize(BytewiseComparator(), block_copy.data(), 0, 0, 0, nullptr,
                   true, true, nullptr, 0, nullptr, 1);

  int count = 0;
  for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
    ASSERT_EQ(iter->value().size(), 10240);
    count++;
  }
  ASSERT_EQ(count, 10);

  delete iter;
}

TEST_F(UserDefinedBlockTest, BasicTest) { BasicTest(); }

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

  // Test SeekForPrev
  iter->SeekForPrev("key30");
  ASSERT_TRUE(iter->Valid());
  ASSERT_EQ(iter->key().ToString(), "key30");
}

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
  iter->Seek("key500");
  ASSERT_TRUE(iter->Valid());
  ASSERT_EQ(iter->key().ToString(), "key500");

  // Count keys from middle to end
  key_count = 0;
  for (; iter->Valid(); iter->Next()) {
    key_count++;
  }
  ASSERT_EQ(key_count, 500);
}

TEST_F(UserDefinedBlockTest, DISABLED_EmptyValue) {
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

TEST_F(UserDefinedBlockTest, L0Test) {
  BlockBasedTableOptions table_options;
  std::string dbname = test::PerThreadDBPath("L0Test");
  std::unique_ptr<DB> db;
  options_.create_if_missing = true;
  options_.disable_auto_compactions = true;
  // options_.compaction_style = kCompactionStyleLevel;
  // options_.level0_file_num_compaction_trigger = 2;
  // options_.compaction_options_universal.allow_trivial_move = true;
  Status s = DB::Open(options_, dbname, &db);
  ASSERT_OK(s);
  ASSERT_TRUE(db != nullptr);

  // Create 100 keys
  auto kvs = generateKVs(/*key_count*/ 100);
  // Add 10 keys at a time, then flush to create multiple L0 files
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      ASSERT_OK(db->Put(WriteOptions(), kvs[i * 10 + j].first,
                        kvs[i * 10 + j].second));
    }
    ASSERT_OK(db->Flush(FlushOptions()));
    Slice begin = kvs[i * 10].first;
    Slice end = kvs[(i + 1) * 10].first;

    ASSERT_OK(db->CompactRange(CompactRangeOptions(), &begin, &end));
  }

  // Search key 50
  std::string value;
  ASSERT_OK(db->Get(ReadOptions(), "key50", &value));
  ASSERT_EQ(value, "value50");

  // Search key 60
  ASSERT_OK(db->Get(ReadOptions(), "key60", &value));
  ASSERT_EQ(value, "value60");

  // Search key 70
  ASSERT_OK(db->Get(ReadOptions(), "key70", &value));
  ASSERT_EQ(value, "value70");

  ASSERT_OK(db->Close());
  ASSERT_OK(DestroyDB(dbname, options_));
}

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
