// Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Example code demonstrating how to use the batch insert feature with
// prefetching optimization for skiplist memtable.

#include <chrono>
#include <iostream>
#include <vector>

#include "memory/arena.h"
#include "memtable/inlineskiplist.h"
#include "rocksdb/comparator.h"
#include "rocksdb/slice.h"
#include "util/coding.h"

namespace ROCKSDB_NAMESPACE {

// Simple key comparator for testing
struct SimpleComparator {
  using DecodedType = Slice;

  int operator()(const char* a, const char* b) const {
    return BytewiseComparator()->Compare(Slice(a + 8, 8), Slice(b + 8, 8));
  }

  int operator()(const char* a, const DecodedType& b) const {
    return BytewiseComparator()->Compare(Slice(a + 8, 8), b);
  }

  DecodedType decode_key(const char* key) const { return Slice(key + 8, 8); }
};

// Helper function to create a key in RocksDB internal format
void MakeKey(char* buf, uint64_t seq, const std::string& user_key) {
  // Format: [8 bytes length prefix][user_key][8 bytes sequence + type]
  char* p = buf;
  p = EncodeVarint64(p, user_key.size() + 8);
  memcpy(p, user_key.data(), user_key.size());
  p += user_key.size();
  EncodeFixed64(p, (seq << 8) | 1);  // 1 is the value type
}

void BenchmarkBatchInsert() {
  std::cout << "Batch Insert with Prefetching Benchmark\n";
  std::cout << "==========================================\n\n";

  Arena arena;
  SimpleComparator cmp;
  InlineSkipList<SimpleComparator> skiplist(cmp, &arena);

  const size_t num_keys = 100000;
  const size_t batch_size = 8;  // Batch size for insert

  std::vector<const char*> keys;

  // Pre-allocate and prepare keys
  std::cout << "Preparing " << num_keys << " keys...\n";
  for (size_t i = 0; i < num_keys; ++i) {
    std::string user_key = "key_" + std::to_string(i * 100);  // Sparse keys
    size_t internal_key_size = 8 + user_key.size() + 8;
    char* key = skiplist.AllocateKey(internal_key_size);
    MakeKey(key, i, user_key);
    keys.push_back(key);
  }

  // Benchmark 1: Traditional single insert
  {
    InlineSkipList<SimpleComparator> skiplist_single(cmp, &arena);
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < num_keys; ++i) {
      std::string user_key = "key_" + std::to_string(i * 100);
      size_t internal_key_size = 8 + user_key.size() + 8;
      char* key = skiplist_single.AllocateKey(internal_key_size);
      MakeKey(key, i, user_key);
      skiplist_single.Insert(key);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Single Insert: " << duration.count() << " us\n";
    std::cout << "  Throughput: " << (num_keys * 1000000.0 / duration.count())
              << " ops/sec\n\n";
  }

  // Benchmark 2: Batch insert with prefetching
  {
    InlineSkipList<SimpleComparator> skiplist_batch(cmp, &arena);
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < num_keys; i += batch_size) {
      std::vector<const char*> batch_keys;
      size_t current_batch_size = std::min(batch_size, num_keys - i);

      // Allocate and prepare batch
      for (size_t j = 0; j < current_batch_size; ++j) {
        std::string user_key = "key_" + std::to_string((i + j) * 100);
        size_t internal_key_size = 8 + user_key.size() + 8;
        char* key = skiplist_batch.AllocateKey(internal_key_size);
        MakeKey(key, i + j, user_key);
        batch_keys.push_back(key);
      }

      // Batch insert
      skiplist_batch.InsertBatch(batch_keys.data(), current_batch_size);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Batch Insert (batch_size=" << batch_size
              << "): " << duration.count() << " us\n";
    std::cout << "  Throughput: " << (num_keys * 1000000.0 / duration.count())
              << " ops/sec\n\n";
  }

  // Benchmark 3: Larger batch size
  const size_t large_batch_size = 16;
  {
    InlineSkipList<SimpleComparator> skiplist_batch(cmp, &arena);
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < num_keys; i += large_batch_size) {
      std::vector<const char*> batch_keys;
      size_t current_batch_size = std::min(large_batch_size, num_keys - i);

      // Allocate and prepare batch
      for (size_t j = 0; j < current_batch_size; ++j) {
        std::string user_key = "key_" + std::to_string((i + j) * 100);
        size_t internal_key_size = 8 + user_key.size() + 8;
        char* key = skiplist_batch.AllocateKey(internal_key_size);
        MakeKey(key, i + j, user_key);
        batch_keys.push_back(key);
      }

      // Batch insert
      skiplist_batch.InsertBatch(batch_keys.data(), current_batch_size);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Batch Insert (batch_size=" << large_batch_size
              << "): " << duration.count() << " us\n";
    std::cout << "  Throughput: " << (num_keys * 1000000.0 / duration.count())
              << " ops/sec\n\n";
  }
}

}  // namespace ROCKSDB_NAMESPACE

int main() {
  ROCKSDB_NAMESPACE::BenchmarkBatchInsert();
  return 0;
}
