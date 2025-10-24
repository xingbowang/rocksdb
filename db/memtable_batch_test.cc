//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "db/dbformat.h"
#include "db/memtable.h"
#include "rocksdb/db.h"
#include "rocksdb/memtablerep.h"
#include "test_util/testharness.h"

namespace ROCKSDB_NAMESPACE {

class MemTableBatchTest : public testing::Test {
 public:
  MemTableBatchTest() : comparator_(BytewiseComparator()) {}

  void SetUp() override {
    ioptions_.memtable_factory.reset(new SkipListFactory);
    ioptions_.user_comparator = comparator_;
    ioptions_.stats = nullptr;

    memtable_ = new MemTable(InternalKeyComparator(comparator_), ioptions_,
                             MutableCFOptions(Options()), nullptr, 0, 0);
    memtable_->Ref();
  }

  void TearDown() override { delete memtable_->Unref(); }

  bool Get(const std::string& key, std::string* value) {
    LookupKey lkey(key, kMaxSequenceNumber);
    bool found = false;
    Status s;
    MergeContext merge_context;
    SequenceNumber max_covering_tombstone_seq = 0;
    SequenceNumber seq;
    ReadOptions ropts;
    found = memtable_->Get(lkey, value, nullptr, nullptr, &s, &merge_context,
                           &max_covering_tombstone_seq, &seq, ropts,
                           false /* immutable_memtable */);
    return found && s.ok();
  }

 protected:
  const Comparator* comparator_;
  ImmutableOptions ioptions_;
  MemTable* memtable_;
};

TEST_F(MemTableBatchTest, BasicBatchAdd) {
  const int kNumEntries = 100;
  std::vector<MemTable::MemTableEntry> entries;

  // Create batch of entries
  for (int i = 0; i < kNumEntries; i++) {
    MemTable::MemTableEntry entry;
    entry.seq = i + 1;
    entry.type = kTypeValue;
    entry.key = Slice("key" + std::to_string(i));
    entry.value = Slice("value" + std::to_string(i));
    entry.kv_prot_info = nullptr;
    entries.push_back(entry);
  }

  // Insert all entries using BatchAdd
  Status s = memtable_->BatchAdd(entries.data(), kNumEntries);
  ASSERT_OK(s);

  // Verify all entries were inserted
  ASSERT_EQ(memtable_->NumEntries(), kNumEntries);

  // Verify we can read back the values
  for (int i = 0; i < kNumEntries; i++) {
    std::string value;
    bool found = Get("key" + std::to_string(i), &value);
    ASSERT_TRUE(found);
    ASSERT_EQ(value, "value" + std::to_string(i));
  }
}

TEST_F(MemTableBatchTest, EmptyBatch) {
  Status s = memtable_->BatchAdd(nullptr, 0);
  ASSERT_OK(s);
  ASSERT_EQ(memtable_->NumEntries(), 0);
}

TEST_F(MemTableBatchTest, SingleEntry) {
  MemTable::MemTableEntry entry;
  entry.seq = 1;
  entry.type = kTypeValue;
  entry.key = Slice("single_key");
  entry.value = Slice("single_value");
  entry.kv_prot_info = nullptr;

  Status s = memtable_->BatchAdd(&entry, 1);
  ASSERT_OK(s);

  std::string value;
  bool found = Get("single_key", &value);
  ASSERT_TRUE(found);
  ASSERT_EQ(value, "single_value");
}

TEST_F(MemTableBatchTest, BatchVsSingleInsert) {
  const int kNumEntries = 50;

  // Use BatchAdd
  std::vector<MemTable::MemTableEntry> batch_entries;
  for (int i = 0; i < kNumEntries; i++) {
    MemTable::MemTableEntry entry;
    entry.seq = i + 1;
    entry.type = kTypeValue;
    entry.key = Slice("batch_key" + std::to_string(i));
    entry.value = Slice("batch_value" + std::to_string(i));
    entry.kv_prot_info = nullptr;
    batch_entries.push_back(entry);
  }
  Status s = memtable_->BatchAdd(batch_entries.data(), kNumEntries);
  ASSERT_OK(s);

  // Use single Add for comparison
  for (int i = 0; i < kNumEntries; i++) {
    s = memtable_->Add(i + 1000, kTypeValue,
                       Slice("single_key" + std::to_string(i)),
                       Slice("single_value" + std::to_string(i)), nullptr);
    ASSERT_OK(s);
  }

  // Verify total count
  ASSERT_EQ(memtable_->NumEntries(), kNumEntries * 2);

  // Verify batch entries
  for (int i = 0; i < kNumEntries; i++) {
    std::string value;
    bool found = Get("batch_key" + std::to_string(i), &value);
    ASSERT_TRUE(found);
    ASSERT_EQ(value, "batch_value" + std::to_string(i));
  }

  // Verify single entries
  for (int i = 0; i < kNumEntries; i++) {
    std::string value;
    bool found = Get("single_key" + std::to_string(i), &value);
    ASSERT_TRUE(found);
    ASSERT_EQ(value, "single_value" + std::to_string(i));
  }
}

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
