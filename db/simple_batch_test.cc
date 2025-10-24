//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include <iostream>
#include <vector>

#include "db/dbformat.h"
#include "db/memtable.h"
#include "rocksdb/db.h"
#include "rocksdb/memtablerep.h"

int main() {
  using namespace ROCKSDB_NAMESPACE;

  ImmutableOptions ioptions;
  ioptions.memtable_factory.reset(new SkipListFactory);
  ioptions.user_comparator = BytewiseComparator();
  ioptions.stats = nullptr;

  MemTable* memtable =
      new MemTable(InternalKeyComparator(BytewiseComparator()), ioptions,
                   MutableCFOptions(Options()), nullptr, 0, 0);
  memtable->Ref();

  const int kNumEntries = 100;
  std::vector<MemTable::MemTableEntry> entries;

  // Create batch of entries
  for (int i = 0; i < kNumEntries; i++) {
    MemTable::MemTableEntry entry;
    entry.seq = i + 1;
    entry.type = kTypeValue;
    std::string key_str = "key" + std::to_string(i);
    std::string value_str = "value" + std::to_string(i);
    entry.key = Slice(key_str);
    entry.value = Slice(value_str);
    entry.kv_prot_info = nullptr;
    entries.push_back(entry);
  }

  // Insert all entries using BatchAdd
  Status s = memtable->BatchAdd(entries.data(), kNumEntries);
  if (!s.ok()) {
    std::cerr << "BatchAdd failed: " << s.ToString() << std::endl;
    delete memtable->Unref();
    return 1;
  }

  // Verify all entries were inserted
  if (memtable->NumEntries() != kNumEntries) {
    std::cerr << "Expected " << kNumEntries << " entries, but got "
              << memtable->NumEntries() << std::endl;
    delete memtable->Unref();
    return 1;
  }

  std::cout << "Successfully inserted " << kNumEntries
            << " entries using BatchAdd!" << std::endl;
  std::cout << "MemTable test PASSED!" << std::endl;

  delete memtable->Unref();
  return 0;
}
