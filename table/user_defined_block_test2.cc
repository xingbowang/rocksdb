//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
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
#include "rocksdb/user_defined_block.h"
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

static const bool kVerbose = true;

class RexDBBase : public testing::Test {
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

  struct RexDBUdbIteratorArg {
    // This class
    // 1. stores projected columns for each query.
    // 2. stores copy of data for projected value for all the keys within the
    // scan range.
    // 3. set a flag to indicate whether it should read one page at a time.
    // 4. track upper bound for the scan
    // 5. track number of rows read
    //
    // TODO : discuss zero copy in standup

    std::vector<std::string> projected_columns;
    std::unordered_map<std::string, std::vector<std::string>> projected_values;
    bool read_one_page_at_a_time;
    std::string scan_upper_bound;
    int row_count;

    RexDBUdbIteratorArg() : read_one_page_at_a_time(false), row_count(0) {}
  };

  // RexDB Block Iterator - reads RexDB formatted data
  class RexDBBlockIterator : public DataBlockIter {
   public:
    RexDBBlockIterator() : current_record_idx_(0) {}

    void Init(void* user_defined_block_iterator_arg, const Comparator* raw_ucmp,
              const char* data) {
      // Now override with RexDB-specific initialization
      rexdb_udb_iterator_arg_ =
          static_cast<RexDBUdbIteratorArg*>(user_defined_block_iterator_arg);
      // make the updateKey assert pass
      global_seqno_ = kDisableGlobalSequenceNumber;
      comparator_ = raw_ucmp;
      data_ = data;
      ParseRexDBBlock();
    }

    Slice value() const override {
      if (!Valid()) return Slice();
      return current_value_;
    }

   protected:
    void SeekToFirstImpl() override {
      if (records_.empty()) {
        current_record_idx_ = 0;
        current_ = restarts_;
        return;
      }
      current_record_idx_ = 0;
      current_ = 0;
      LoadCurrentRecord();
    }

    void SeekToLastImpl() override {
      if (records_.empty()) {
        current_record_idx_ = 0;
        current_ = restarts_;
        return;
      }
      current_record_idx_ = static_cast<uint32_t>(records_.size() - 1);
      current_ = 0;
      LoadCurrentRecord();
    }

    void SeekImpl(const Slice& target) override {
      auto target_user_key = ExtractUserKey(target);
      if (records_.empty()) {
        current_ = restarts_;
        return;
      }

      uint32_t left = 0;
      uint32_t right = static_cast<uint32_t>(records_.size());

      while (left < right) {
        uint32_t mid = (left + right) / 2;
        auto cmp = comparator_->CompareWithoutTimestamp(
            target_user_key, /*a_has_ts=*/false,
            ExtractUserKey(records_[mid].first), /*b_has_ts=*/false);
        if (cmp > 0) {
          left = mid + 1;
        } else if (cmp < 0) {
          right = mid;
        } else {
          current_record_idx_ = mid;
          current_ = 0;
          LoadCurrentRecord();
          return;
        }
      }

      if (left >= records_.size()) {
        current_ = restarts_;
        return;
      }

      current_record_idx_ = left;
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

      current_record_idx_++;
      if (current_record_idx_ >= records_.size()) {
        current_ = restarts_;
        return;
      }

      LoadCurrentRecord();
    }

    void PrevImpl() override {
      if (current_record_idx_ == 0) {
        current_ = restarts_;
        return;
      }

      current_record_idx_--;
      LoadCurrentRecord();
    }

   private:
    void ParseRexDBBlock() {
      if (data_ == nullptr) return;

      const char* ptr = data_;

      uint32_t num_records = DecodeFixed32(ptr);
      ptr += 4;

      records_.clear();

      std::vector<Slice> keys;
      for (uint32_t i = 0; i < num_records; i++) {
        uint32_t key_len = DecodeFixed32(ptr);
        ptr += 4;
        keys.emplace_back(ptr, key_len);
        ptr += key_len;
      }

      std::vector<Slice> values;
      for (uint32_t i = 0; i < num_records; i++) {
        uint32_t val_len = DecodeFixed32(ptr);
        ptr += 4;
        values.emplace_back(ptr, val_len);
        ptr += val_len;
      }

      for (uint32_t i = 0; i < num_records; i++) {
        records_.emplace_back(keys[i], values[i]);
      }

      restarts_ = static_cast<uint32_t>(records_.size());
    }

    void LoadCurrentRecord() {
      if (current_record_idx_ >= records_.size()) {
        current_ = restarts_;
        return;
      }

      // If read_one_page_at_a_time is true, collect ALL remaining keys in this
      // block
      if (rexdb_udb_iterator_arg_ &&
          rexdb_udb_iterator_arg_->read_one_page_at_a_time) {
        if (kVerbose) {
          fprintf(stderr,
                  "LoadCurrentRecord: read_one_page_at_a_time mode, collecting "
                  "all keys from idx %u\n",
                  current_record_idx_);
        }

        // Iterate through all remaining records in this block
        for (uint32_t idx = current_record_idx_; idx < records_.size(); idx++) {
          const auto& record = records_[idx];
          std::string key_str = record.first.ToString();

          // Check if we've reached the upper bound
          if (!rexdb_udb_iterator_arg_->scan_upper_bound.empty() &&
              key_str >= rexdb_udb_iterator_arg_->scan_upper_bound) {
            if (kVerbose) {
              fprintf(stderr, "  Reached upper bound at key: %s\n",
                      key_str.c_str());
            }
            break;  // Stop collecting
          }

          // Determine the value to use (full or projected)
          Slice value_to_use;
          if (!rexdb_udb_iterator_arg_->projected_columns.empty()) {
            value_to_use = ProjectColumns(record.second);
          } else {
            value_to_use = record.second;
          }

          // Store the value in projected_values
          std::string value_str = value_to_use.ToString();
          rexdb_udb_iterator_arg_->projected_values[key_str].push_back(
              value_str);
          rexdb_udb_iterator_arg_->row_count++;

          if (kVerbose) {
            fprintf(stderr, "  Collected: key=%s, value=%s\n", key_str.c_str(),
                    value_str.c_str());
          }
        }

        if (kVerbose) {
          fprintf(stderr, "  Finished collecting block, total rows: %d\n",
                  rexdb_udb_iterator_arg_->row_count);
        }

        // Mark as invalid to force upper level iterator to move to next block
        current_ = restarts_;
        return;
      }

      // Normal mode: load single record
      const auto& record = records_[current_record_idx_];
      key_ = record.first;

      if (kVerbose) {
        fprintf(stderr,
                "LoadCurrentRecord: key=%s, rexdb_udb_iterator_arg_=%p\n",
                key_.ToString().c_str(), rexdb_udb_iterator_arg_);
        if (rexdb_udb_iterator_arg_) {
          fprintf(stderr, "  projected_columns.size()=%zu\n",
                  rexdb_udb_iterator_arg_->projected_columns.size());
        }
      }

      // Determine the value to use (full or projected)
      Slice value_to_use;
      if (rexdb_udb_iterator_arg_ &&
          !rexdb_udb_iterator_arg_->projected_columns.empty()) {
        if (kVerbose) {
          fprintf(stderr, "  Using projected columns!\n");
        }
        value_to_use = ProjectColumns(record.second);
      } else {
        if (kVerbose) {
          fprintf(stderr, "  Using full value\n");
        }
        value_to_use = record.second;
      }

      current_value_ = value_to_use;
      raw_key_.SetKey(key_, false);
    }

    Slice ProjectColumns(const Slice& full_value) {
      projected_value_buffer_.clear();
      const auto& proj_cols = rexdb_udb_iterator_arg_->projected_columns;
      std::unordered_set<std::string> proj_set(proj_cols.begin(),
                                               proj_cols.end());

      std::string val_str = full_value.ToString();
      size_t pos = 0;

      while (pos < val_str.size()) {
        size_t eq_pos = val_str.find('=', pos);
        if (eq_pos == std::string::npos) break;

        size_t semi_pos = val_str.find(';', eq_pos);
        if (semi_pos == std::string::npos) {
          semi_pos = val_str.size();
        }

        std::string col_name = val_str.substr(pos, eq_pos - pos);

        if (proj_set.find(col_name) != proj_set.end()) {
          projected_value_buffer_.append(
              val_str.substr(pos, semi_pos - pos + 1));
        }

        pos = semi_pos + 1;
      }

      return Slice(projected_value_buffer_);
    }

    RexDBUdbIteratorArg* rexdb_udb_iterator_arg_;
    const Comparator* comparator_;
    std::vector<std::pair<Slice, Slice>> records_;
    uint32_t current_record_idx_;
    Slice current_value_;
    std::string projected_value_buffer_;
  };

  // RexDB (Partition Attributes Across) columnar format implementation
  class RexDBBlockBuilder : public BlockBuilder {
   public:
    RexDBBlockBuilder()
        : BlockBuilder(1, false, false,
                       BlockBasedTableOptions::kDataBlockBinarySearch) {}

    void Add(const Slice& key, const Slice& value,
             const Slice* const /*delta_value*/ = nullptr,
             bool /*skip_delta_encoding*/ = false) override {
      keys_.emplace_back(key.data(), key.size());
      values_.emplace_back(value.data(), value.size());
      total_key_count_++;
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
      buffer_.clear();

      PutFixed32(&buffer_, static_cast<uint32_t>(keys_.size()));

      for (const auto& key : keys_) {
        PutFixed32(&buffer_, static_cast<uint32_t>(key.size()));
        buffer_.append(key);
      }

      for (const auto& value : values_) {
        PutFixed32(&buffer_, static_cast<uint32_t>(value.size()));
        buffer_.append(value);
      }

      return Slice(buffer_);
    }

    size_t EstimateSizeAfterKV(const Slice& key,
                               const Slice& value) const override {
      return CurrentSizeEstimate() + key.size() + value.size() + 16;
    }

    size_t CurrentSizeEstimate() const override {
      size_t size = 4;
      for (const auto& k : keys_) {
        size += k.size() + 4;
      }
      for (const auto& v : values_) {
        size += v.size() + 4;
      }
      return size + 100;
    }
    std::string& MutableBuffer() override { return buffer_; }

    void Reset() override {
      BlockBuilder::Reset();
      keys_.clear();
      values_.clear();
      buffer_.clear();
      total_key_count_ = 0;
    }

   private:
    int total_key_count_{0};
    std::vector<std::string> keys_;
    std::vector<std::string> values_;
    std::string buffer_;
  };

  // RexDBBlock: Implements UserDefinedBlock interface for columnar storage
  class RexDBBlock : public UserDefinedBlock {
   public:
    Status InitBlock(BlockContents* contents) override {
      contents_ = contents;
      // The data is already parsed in the iterator, just validate
      if (contents_->data.empty()) {
        return Status::Corruption("Empty block contents");
      }
      return Status::OK();
    }

    ~RexDBBlock() override = default;

    // UserDefinedBlock interface
    size_t ApproximateMemoryUsage() const override {
      return contents_->ApproximateMemoryUsage() + sizeof(*this);
    }

    const Slice& ContentSlice() const override { return contents_->data; }

    size_t size() const override { return contents_->data.size(); }

    const char* data() const override { return contents_->data.data(); }

    bool own_bytes() const override { return contents_->own_bytes(); }

    DataBlockIter* NewDataIterator(
        const Comparator* raw_ucmp, SequenceNumber /*global_seqno*/,
        DataBlockIter* input_iter, Statistics* /*stats*/,
        bool /*block_contents_pinned*/,
        bool /*user_defined_timestamps_persisted*/,
        void* user_defined_block_iterator_arg) override {
      RexDBBlockIterator* iter = nullptr;
      if (input_iter != nullptr) {
        // Reuse existing iterator if provided
        // TODO : fix this
        delete input_iter;
        iter = new RexDBBlockIterator();
      } else {
        iter = new RexDBBlockIterator();
      }

      // Initialize the iterator with block data
      iter->Init(user_defined_block_iterator_arg, raw_ucmp,
                 contents_->data.data());

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
      builder = std::make_unique<RexDBBlockBuilder>();
      return Status::OK();
    }

    // Legacy API - not used when UsesCustomBlockFormat() returns true
    Status NewIterator(const UserDefinedBlockOption& /*option*/,
                       DataBlockIter** biter) const override {
      *biter = new RexDBBlockIterator();
      return Status::OK();
    }

    // NEW API: Create custom block from raw bytes
    Status NewBlock(const UserDefinedBlockOption& /*option*/,
                    std::unique_ptr<UserDefinedBlock>* block) const override {
      *block = std::make_unique<RexDBBlock>();
      return Status::OK();
    }

    // Indicate we use custom block format
    bool UsesCustomBlockFormat() const override { return true; }
  };

 protected:
  std::vector<std::pair<std::string, std::string>> generateKVs(
      int key_count, int value_column_count) {
    std::vector<std::pair<std::string, std::string>> kvs(key_count);
    // Determine width based on key_count
    int width = key_count < 100 ? 2 : (key_count < 1000 ? 3 : 4);
    for (int i = 0; i < key_count; i++) {
      std::stringstream ss;
      ss << std::setw(width) << std::setfill('0') << i;
      std::string key = "key" + ss.str();
      std::string value;
      for (int j = 0; j < value_column_count; j++) {
        value += "col" + std::to_string(j) + "=value_col" + std::to_string(j) +
                 "_" + ss.str() + ";";
      }
      // print key and value
      if (kVerbose) {
        fprintf(stderr, "key: %s, value: %s\n", key.c_str(), value.c_str());
      }
      kvs[i] = std::make_pair(key, value);
    }
    return kvs;
  }

  void BasicTest();

  Options options_;
  Random rnd{301};
};

class RexDB : public RexDBBase {};

class MyFlushListener : public rocksdb::EventListener {
 public:
  void OnFlushCompleted(DB* db, const FlushJobInfo& info) override {
    // Your custom logic here
    std::cout << "Flush completed for column family: " << info.cf_name
              << std::endl;
    std::cout << "File path: " << info.file_path << std::endl;
    std::cout << "Job ID: " << info.job_id << std::endl;

    auto lmax = 6;
    // Call compactFiles to move the new flushed file from L0 to Lmax
    db->CompactFiles(CompactionOptions(), {info.file_path}, lmax);
  }
};

TEST_F(RexDB, RexDBPrototypeTest) {
  // Test user-defined blocks with DB operations including Put/Seek/Flush
  BlockBasedTableOptions table_options;
  std::string dbname = test::PerThreadDBPath("RexDB_DBPaxPrototypeTest");

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

  // Disable auto compactions, add trigger to move flushed files to Lmax
  options_.disable_auto_compactions = true;
  std::shared_ptr<MyFlushListener> listener =
      std::make_shared<MyFlushListener>();
  options_.listeners.push_back(listener);

  // Clean up any existing DB
  ASSERT_OK(DestroyDB(dbname, options_));

  // Open DB
  DB* db = nullptr;
  Status s = DB::Open(options_, dbname, &db);
  ASSERT_OK(s);
  ASSERT_NE(db, nullptr);

  // Test 1: Basic Put and Seek operations
  WriteOptions write_options;
  ReadOptions read_options;

  if (kVerbose) {
    fprintf(stderr, "\n\nAdd data to DB ...\n\n");
  }

  // Write some data
  auto kvs = generateKVs(100, 3);
  for (auto const& kv : kvs) {
    ASSERT_OK(db->Put(write_options, kv.first, kv.second));
  }

  db->Flush(FlushOptions());

  if (kVerbose) {
    fprintf(stderr, "\n\nRead data to DB ...\n\n");
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
    // print key value pair
    if (kVerbose) {
      fprintf(stderr, "key: %s, value: %s\n", iter->key().ToString().c_str(),
              iter->value().ToString().c_str());
    }
  }
  delete iter;

  if (kVerbose) {
    fprintf(stderr, "\n\nRead projected data from DB ...\n\n");
  }

  // 1. use mmap
  // 1. Use multi scan to pin multiple pages in memory
  // 2. Use another iterator to access the pages.
  // 3. The user defined block iterator will
  //    1. use user_defined_block_iterator_arg to decide the projected columns
  //    2. use user_defined_block_iterator_arg to decide the upper bound
  //    3. use user_defined_block_iterator_arg to collect pointers for the
  //    projected columns

  // To discuss on standup:
  // One issue is that the in-memory block structure is only valid when iterator
  // pointing to it. Once iterator moved on to next block, the previous block in
  // memory structor is destroyed. There are 2 options:
  //    1. This means the user_defined_block_iterator_arg needs to copy the
  //    structure information out.
  //    2. Iterator needs to copy the projected value to
  //    user_defined_block_iterator_arg directly. I don't know whether we could
  //    pass zippydb network write API down here directly, so that the value is
  //    written to the network system call directly. If so, we could achieve 0
  //    data copy. If not, we would have to copy it to a temporary buffer, then
  //    write it out through network system call, which would cause extra memory
  //    copy.

  // Test read_one_page_at_a_time mode
  if (kVerbose) {
    fprintf(stderr, "\n\nTest read_one_page_at_a_time mode ...\n\n");
  }

  RexDBUdbIteratorArg rexdb_udb_iterator_arg;
  rexdb_udb_iterator_arg.projected_columns = {"col0", "col2"};
  rexdb_udb_iterator_arg.scan_upper_bound = "key080";
  rexdb_udb_iterator_arg.read_one_page_at_a_time = true;

  read_options.user_defined_block_iterator_arg = &rexdb_udb_iterator_arg;

  // Scan through data blocks collecting projected values
  iter = db->NewIterator(read_options);

  // Start scanning from key025
  iter->Seek("key025");

  // The iterator will automatically collect values and move to next blocks
  // until we reach the upper bound
  int scan_count = 0;
  while (iter->Valid()) {
    scan_count++;
    if (kVerbose) {
      fprintf(stderr, "Scanning: key=%s\n", iter->key().ToString().c_str());
    }
    iter->Next();
  }

  if (kVerbose) {
    fprintf(stderr, "Scan completed after %d iterations\n", scan_count);
  }

  delete iter;

  // Print collected values
  if (kVerbose) {
    fprintf(stderr, "\n\nCollected projected values:\n");
    fprintf(stderr, "Total rows collected: %d\n",
            rexdb_udb_iterator_arg.row_count);
    fprintf(stderr, "Total unique keys: %zu\n",
            rexdb_udb_iterator_arg.projected_values.size());

    // Print first 10 entries as example
    for (const auto& kv : rexdb_udb_iterator_arg.projected_values) {
      fprintf(stderr, "Key: %s\n", kv.first.c_str());
      for (const auto& val : kv.second) {
        fprintf(stderr, "  Value: %s\n", val.c_str());
      }
    }
  }

  // Verify we collected data from key000 to key079 (before upper bound)
  ASSERT_GT(rexdb_udb_iterator_arg.row_count, 0);
  ASSERT_GT(rexdb_udb_iterator_arg.projected_values.size(), 0);

  // Verify upper bound was respected
  for (const auto& kv : rexdb_udb_iterator_arg.projected_values) {
    ASSERT_LT(kv.first, rexdb_udb_iterator_arg.scan_upper_bound);
  }

  // Final cleanup
  delete db;
  ASSERT_OK(DestroyDB(dbname, options_));

  if (kVerbose) {
    fprintf(stderr, "All DB tests passed!\n");
  }
}

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
