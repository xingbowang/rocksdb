//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "memory/arena.h"
#include "memtable/inlineskiplist.h"
#include "util/random.h"

// Simple assertion macros for standalone benchmark
#define ASSERT_TRUE(condition)                                                 \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "Assertion failed: %s at %s:%d\n", #condition, __FILE__, \
              __LINE__);                                                       \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#define ASSERT_EQ(a, b) ASSERT_TRUE((a) == (b))
#define ASSERT_NE(a, b) ASSERT_TRUE((a) != (b))
#define ASSERT_LT(a, b) ASSERT_TRUE((a) < (b))
#define ASSERT_LE(a, b) ASSERT_TRUE((a) <= (b))
#define ASSERT_GT(a, b) ASSERT_TRUE((a) > (b))
#define ASSERT_GE(a, b) ASSERT_TRUE((a) >= (b))

namespace ROCKSDB_NAMESPACE {

// Variable-length keys with a uint64_t ID prefix for ordering
struct VariableLengthKey {
  uint64_t id;
  size_t key_size;

  static uint64_t DecodeId(const char* key) {
    uint64_t id;
    memcpy(&id, key, sizeof(uint64_t));
    return id;
  }

  static void EncodeKey(char* buf, uint64_t id, size_t /*key_size*/,
                        Random* /*rnd*/) {
    // First 8 bytes: the ID for ordering
    memcpy(buf, &id, sizeof(uint64_t));
  }
};

struct TestComparator {
  using DecodedType = uint64_t;

  static DecodedType decode_key(const char* b) {
    return VariableLengthKey::DecodeId(b);
  }

  int operator()(const char* a, const char* b) const {
    uint64_t id_a = VariableLengthKey::DecodeId(a);
    uint64_t id_b = VariableLengthKey::DecodeId(b);
    if (id_a < id_b) {
      return -1;
    } else if (id_a > id_b) {
      return +1;
    } else {
      return 0;
    }
  }

  int operator()(const char* a, const DecodedType b) const {
    uint64_t id_a = VariableLengthKey::DecodeId(a);
    if (id_a < b) {
      return -1;
    } else if (id_a > b) {
      return +1;
    } else {
      return 0;
    }
  }
};

class SkipListBatchBenchmark {
 public:
  struct BenchResult {
    std::string name;
    size_t num_keys;
    size_t batch_size;
    double duration_ms;
    double throughput_ops_per_sec;
    double throughput_mb_per_sec;
    double speedup_vs_single;
    double speedup_vs_old_batch;
  };

  void Run() {
    PrintHeader();

    // Test with different key sizes (in bytes)
    std::vector<size_t> key_sizes = {100, 200, 500,
                                     1000};  // 100B, 500B, 1KB, 10KB

    // Test with different data sizes (in bytes)
    std::vector<size_t> key_counts = {1000 * 1000, 1000 * 1000,
                                      10 * 1000 * 1000};

    for (size_t key_size : key_sizes) {
      for (size_t key_count : key_counts) {
        // Calculate number of keys needed to reach the target data size
        size_t num_keys = key_count;

        // Cap at reasonable limits for very large workloads
        if (num_keys >
            10000000) {  // Cap at 10M keys for practical benchmarking
          num_keys = 10000000;
        }

        std::cout
            << "\n╔════════════════════════════════════════════════════════════"
               "══════════╗"
            << std::endl;
        std::cout << "║ Key Size: " << std::setw(6) << FormatSize(key_size)
                  << " | Num Keys: " << std::setw(10) << num_keys
                  << std::string(18, ' ') << "║" << std::endl;
        std::cout
            << "╚════════════════════════════════════════════════════════════"
               "══════════╝"
            << std::endl;

        // Test sequential keys
        std::cout << "\nSequential Keys:" << std::endl;
        RunBenchmarkSet(num_keys, key_size, false);

        // Test random keys
        std::cout << "\nRandom Keys:" << std::endl;
        RunBenchmarkSet(num_keys, key_size, true);
      }
    }
  }

 private:
  std::string FormatSize(size_t bytes) {
    if (bytes >= 1024ULL * 1024 * 1024) {
      return std::to_string(bytes / (1024ULL * 1024 * 1024)) + "GB";
    } else if (bytes >= 1024 * 1024) {
      return std::to_string(bytes / (1024 * 1024)) + "MB";
    } else if (bytes >= 1024) {
      return std::to_string(bytes / 1024) + "KB";
    } else {
      return std::to_string(bytes) + "B";
    }
  }

 private:
  void PrintHeader() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════"
                 "══════════╗\n";
    std::cout << "║   SkipList Insert Performance: Single vs Old Batch vs New "
                 "Batch     ║\n";
    std::cout << "║              (New Batch uses prefetching optimization)     "
                 "         ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════"
                 "══════════╝\n";
    std::cout << std::endl;
  }

  void RunBenchmarkSet(size_t num_keys, size_t key_size, bool random_keys) {
    std::vector<BenchResult> results;

    // Calculate throughput in MB/s
    double data_size_mb = (num_keys * key_size) / (1024.0 * 1024.0);

    // Baseline: single insert (one key at a time)
    double single_time = BenchmarkSingleInsert(num_keys, key_size, random_keys);
    BenchResult baseline = {"Single Insert (baseline)",
                            num_keys,
                            1,
                            single_time,
                            num_keys * 1000.0 / single_time,
                            data_size_mb * 1000.0 / single_time,
                            1.0,
                            0.0};
    results.push_back(baseline);

    // Test different batch sizes with both old and new APIs
    std::vector<size_t> batch_sizes = {2, 4, 8, 16, 32, 64};

    for (size_t batch_size : batch_sizes) {
      // Old approach: batch grouping but still using Insert() for each key
      double old_batch_time =
          BenchmarkOldBatchInsert(num_keys, key_size, batch_size, random_keys);
      BenchResult old_result = {"Old Batch (Insert loop)",
                                num_keys,
                                batch_size,
                                old_batch_time,
                                num_keys * 1000.0 / old_batch_time,
                                data_size_mb * 1000.0 / old_batch_time,
                                single_time / old_batch_time,
                                1.0};
      results.push_back(old_result);

      // New approach: using InsertBatch with prefetching
      double new_batch_time =
          BenchmarkNewBatchInsert(num_keys, key_size, batch_size, random_keys);
      BenchResult new_result = {"New Batch (w/prefetch)",
                                num_keys,
                                batch_size,
                                new_batch_time,
                                num_keys * 1000.0 / new_batch_time,
                                data_size_mb * 1000.0 / new_batch_time,
                                single_time / new_batch_time,
                                old_batch_time / new_batch_time};
      results.push_back(new_result);
    }

    // Print insert results
    PrintResults(results, key_size);

    // Now benchmark random read performance comparing Insert vs InsertBatch
    std::cout << "\n" << std::string(86, '=') << std::endl;
    std::cout << "Random Read Performance Comparison (Insert vs InsertBatch)"
              << std::endl;
    std::cout << std::string(86, '=') << std::endl;

    // Benchmark reads on skiplist built with Insert API
    std::cout << "\n[1] Skiplist built with Insert API:" << std::endl;
    double insert_read_throughput =
        BenchmarkRandomReadsWithInsert(num_keys, key_size, random_keys);

    // Benchmark reads on skiplist built with InsertBatch API
    std::cout << "\n[2] Skiplist built with InsertBatch API (batch_size=32):"
              << std::endl;
    double batch_read_throughput =
        BenchmarkRandomReadsWithBatch(num_keys, key_size, random_keys, 32);

    // Print comparison
    std::cout << "\n" << std::string(86, '-') << std::endl;
    std::cout << "Read Performance Summary:" << std::endl;
    std::cout << std::string(86, '-') << std::endl;
    std::cout << "  Insert API read throughput:      " << std::fixed
              << std::setprecision(2);
    if (insert_read_throughput >= 1000000) {
      std::cout << (insert_read_throughput / 1000000.0) << " M ops/sec"
                << std::endl;
    } else if (insert_read_throughput >= 1000) {
      std::cout << (insert_read_throughput / 1000.0) << " K ops/sec"
                << std::endl;
    } else {
      std::cout << insert_read_throughput << " ops/sec" << std::endl;
    }

    std::cout << "  InsertBatch API read throughput: " << std::fixed
              << std::setprecision(2);
    if (batch_read_throughput >= 1000000) {
      std::cout << (batch_read_throughput / 1000000.0) << " M ops/sec"
                << std::endl;
    } else if (batch_read_throughput >= 1000) {
      std::cout << (batch_read_throughput / 1000.0) << " K ops/sec"
                << std::endl;
    } else {
      std::cout << batch_read_throughput << " ops/sec" << std::endl;
    }

    double diff_percent = ((batch_read_throughput - insert_read_throughput) /
                           insert_read_throughput) *
                          100.0;
    std::cout << "  Difference:                      " << std::fixed
              << std::setprecision(2) << std::showpos << diff_percent
              << std::noshowpos << "%" << std::endl;
    std::cout << std::string(86, '-') << std::endl;
  }

  double BenchmarkSingleInsert(size_t num_keys, size_t key_size,
                               bool random_keys) {
    Arena arena;
    TestComparator cmp;
    InlineSkipList<TestComparator> skiplist(cmp, &arena);
    Random rnd(301);

    std::vector<uint64_t> key_ids = GenerateKeyIds(num_keys, random_keys, &rnd);

    // Calculate warmup and measurement split (20% warmup, 80% measured)
    size_t warmup_count = num_keys / 5;  // 20%

    // Warmup phase: insert first 20% of keys (not timed)
    for (size_t i = 0; i < warmup_count; ++i) {
      char* buf = skiplist.AllocateKey(key_size);
      VariableLengthKey::EncodeKey(buf, key_ids[i], key_size, &rnd);
      skiplist.Insert(buf);
    }

    // Measurement phase: time the remaining 80% of keys
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = warmup_count; i < num_keys; ++i) {
      char* buf = skiplist.AllocateKey(key_size);
      VariableLengthKey::EncodeKey(buf, key_ids[i], key_size, &rnd);
      skiplist.Insert(buf);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    return duration.count() / 1000.0;  // Convert to milliseconds
  }

  double BenchmarkOldBatchInsert(size_t num_keys, size_t key_size,
                                 size_t batch_size, bool random_keys) {
    Arena arena;
    TestComparator cmp;
    InlineSkipList<TestComparator> skiplist(cmp, &arena);
    Random rnd(301);

    std::vector<uint64_t> key_ids = GenerateKeyIds(num_keys, random_keys, &rnd);

    // Calculate warmup and measurement split (20% warmup, 80% measured)
    size_t warmup_count = num_keys / 5;  // 20%

    // Warmup phase: insert first 20% of keys (not timed)
    for (size_t i = 0; i < warmup_count; i += batch_size) {
      size_t current_batch_size = std::min(batch_size, warmup_count - i);

      std::vector<const char*> batch;
      batch.reserve(current_batch_size);
      for (size_t j = 0; j < current_batch_size; ++j) {
        char* buf = skiplist.AllocateKey(key_size);
        VariableLengthKey::EncodeKey(buf, key_ids[i + j], key_size, &rnd);
        batch.push_back(buf);
      }

      for (size_t j = 0; j < current_batch_size; ++j) {
        skiplist.Insert(batch[j]);
      }
    }

    // Measurement phase: time the remaining 80% of keys
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = warmup_count; i < num_keys; i += batch_size) {
      size_t current_batch_size = std::min(batch_size, num_keys - i);

      // Allocate all keys in the batch
      std::vector<const char*> batch;
      batch.reserve(current_batch_size);
      for (size_t j = 0; j < current_batch_size; ++j) {
        char* buf = skiplist.AllocateKey(key_size);
        VariableLengthKey::EncodeKey(buf, key_ids[i + j], key_size, &rnd);
        batch.push_back(buf);
      }

      // Insert using the old API (one at a time)
      for (size_t j = 0; j < current_batch_size; ++j) {
        skiplist.Insert(batch[j]);
      }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    return duration.count() / 1000.0;  // Convert to milliseconds
  }

  double BenchmarkNewBatchInsert(size_t num_keys, size_t key_size,
                                 size_t batch_size, bool random_keys) {
    Arena arena;
    TestComparator cmp;
    InlineSkipList<TestComparator> skiplist(cmp, &arena);
    Random rnd(301);

    std::vector<uint64_t> key_ids = GenerateKeyIds(num_keys, random_keys, &rnd);

    // Calculate warmup and measurement split (20% warmup, 80% measured)
    size_t warmup_count = num_keys / 5;  // 20%

    // Warmup phase: insert first 20% of keys (not timed)
    for (size_t i = 0; i < warmup_count; i += batch_size) {
      size_t current_batch_size = std::min(batch_size, warmup_count - i);
      std::vector<const char*> batch;
      batch.reserve(current_batch_size);

      for (size_t j = 0; j < current_batch_size; ++j) {
        char* buf = skiplist.AllocateKey(key_size);
        VariableLengthKey::EncodeKey(buf, key_ids[i + j], key_size, &rnd);
        batch.push_back(buf);
      }

      skiplist.InsertBatch(batch.data(), current_batch_size);
    }

    // Measurement phase: time the remaining 80% of keys
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = warmup_count; i < num_keys; i += batch_size) {
      size_t current_batch_size = std::min(batch_size, num_keys - i);
      std::vector<const char*> batch;
      batch.reserve(current_batch_size);

      for (size_t j = 0; j < current_batch_size; ++j) {
        char* buf = skiplist.AllocateKey(key_size);
        VariableLengthKey::EncodeKey(buf, key_ids[i + j], key_size, &rnd);
        batch.push_back(buf);
      }

      // Use the new InsertBatch API with prefetching
      skiplist.InsertBatch(batch.data(), current_batch_size);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    return duration.count() / 1000.0;  // Convert to milliseconds
  }

  double BenchmarkRandomReadsWithInsert(size_t num_keys, size_t key_size,
                                        bool random_keys) {
    Arena arena;
    TestComparator cmp;
    InlineSkipList<TestComparator> skiplist(cmp, &arena);
    Random rnd(301);

    std::vector<uint64_t> key_ids = GenerateKeyIds(num_keys, random_keys, &rnd);

    // Build skiplist using Insert API (not timed)
    std::cout << "  Building skiplist with Insert API (" << num_keys
              << " keys)..." << std::flush;
    for (size_t i = 0; i < num_keys; ++i) {
      char* buf = skiplist.AllocateKey(key_size);
      VariableLengthKey::EncodeKey(buf, key_ids[i], key_size, &rnd);
      skiplist.Insert(buf);
    }
    std::cout << " Done." << std::endl;

    // Benchmark random reads
    const size_t num_reads =
        std::min(num_keys, size_t(1000000));  // Cap at 1M reads
    std::cout << "  Performing " << num_reads << " random reads..."
              << std::flush;

    // Generate random read indices
    std::vector<size_t> read_indices;
    read_indices.reserve(num_reads);
    Random read_rnd(12345);
    for (size_t i = 0; i < num_reads; ++i) {
      read_indices.push_back(read_rnd.Next() % num_keys);
    }

    // Allocate and encode keys for reading
    std::vector<char*> read_keys;
    read_keys.reserve(num_reads);
    for (size_t i = 0; i < num_reads; ++i) {
      char* buf = skiplist.AllocateKey(key_size);
      VariableLengthKey::EncodeKey(buf, key_ids[read_indices[i]], key_size,
                                   &rnd);
      read_keys.push_back(buf);
    }

    // Time the random read operations using Contains()
    auto start = std::chrono::high_resolution_clock::now();

    size_t found_count = 0;
    for (size_t i = 0; i < num_reads; ++i) {
      if (skiplist.Contains(read_keys[i])) {
        found_count++;
      }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double duration_ms = duration.count() / 1000.0;
    double throughput_ops = num_reads * 1000.0 / duration_ms;

    std::cout << " Done." << std::endl;
    std::cout << "  Found: " << found_count << "/" << num_reads << std::endl;
    std::cout << "  Read time: " << std::fixed << std::setprecision(2)
              << duration_ms << " ms" << std::endl;

    return throughput_ops;
  }

  double BenchmarkRandomReadsWithBatch(size_t num_keys, size_t key_size,
                                       bool random_keys, size_t batch_size) {
    Arena arena;
    TestComparator cmp;
    InlineSkipList<TestComparator> skiplist(cmp, &arena);
    Random rnd(301);

    std::vector<uint64_t> key_ids = GenerateKeyIds(num_keys, random_keys, &rnd);

    // Build skiplist using InsertBatch API (not timed)
    std::cout << "  Building skiplist with InsertBatch API (" << num_keys
              << " keys, batch_size=" << batch_size << ")..." << std::flush;

    for (size_t i = 0; i < num_keys; i += batch_size) {
      size_t current_batch_size = std::min(batch_size, num_keys - i);
      std::vector<const char*> batch;
      batch.reserve(current_batch_size);

      for (size_t j = 0; j < current_batch_size; ++j) {
        char* buf = skiplist.AllocateKey(key_size);
        VariableLengthKey::EncodeKey(buf, key_ids[i + j], key_size, &rnd);
        batch.push_back(buf);
      }

      skiplist.InsertBatch(batch.data(), current_batch_size);
    }
    std::cout << " Done." << std::endl;

    // Benchmark random reads (same as Insert version)
    const size_t num_reads =
        std::min(num_keys, size_t(1000000));  // Cap at 1M reads
    std::cout << "  Performing " << num_reads << " random reads..."
              << std::flush;

    // Generate random read indices
    std::vector<size_t> read_indices;
    read_indices.reserve(num_reads);
    Random read_rnd(12345);
    for (size_t i = 0; i < num_reads; ++i) {
      read_indices.push_back(read_rnd.Next() % num_keys);
    }

    // Allocate and encode keys for reading
    std::vector<char*> read_keys;
    read_keys.reserve(num_reads);
    for (size_t i = 0; i < num_reads; ++i) {
      char* buf = skiplist.AllocateKey(key_size);
      VariableLengthKey::EncodeKey(buf, key_ids[read_indices[i]], key_size,
                                   &rnd);
      read_keys.push_back(buf);
    }

    // Time the random read operations using Contains()
    auto start = std::chrono::high_resolution_clock::now();

    size_t found_count = 0;
    for (size_t i = 0; i < num_reads; ++i) {
      if (skiplist.Contains(read_keys[i])) {
        found_count++;
      }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double duration_ms = duration.count() / 1000.0;
    double throughput_ops = num_reads * 1000.0 / duration_ms;

    std::cout << " Done." << std::endl;
    std::cout << "  Found: " << found_count << "/" << num_reads << std::endl;
    std::cout << "  Read time: " << std::fixed << std::setprecision(2)
              << duration_ms << " ms" << std::endl;

    return throughput_ops;
  }

  std::vector<uint64_t> GenerateKeyIds(size_t num_keys, bool random,
                                       Random* rnd) {
    std::vector<uint64_t> key_ids;
    key_ids.reserve(num_keys);

    if (random) {
      // Generate random key IDs
      for (size_t i = 0; i < num_keys; ++i) {
        key_ids.push_back(rnd->Next64());
      }
    } else {
      // Generate sequential key IDs with gaps
      for (size_t i = 0; i < num_keys; ++i) {
        key_ids.push_back(i * 100);
      }
    }

    return key_ids;
  }

  void PrintResults(const std::vector<BenchResult>& results,
                    size_t /*key_size*/) {
    std::cout << std::endl;
    std::cout << std::left << std::setw(26) << "Method" << std::setw(12)
              << "Batch Size" << std::setw(12) << "Time (ms)" << std::setw(14)
              << "Throughput" << std::setw(13) << "vs Single" << std::setw(13)
              << "vs Old Batch" << std::endl;
    std::cout << std::string(90, '-') << std::endl;

    for (const auto& result : results) {
      std::cout << std::left << std::setw(26) << result.name << std::setw(12)
                << result.batch_size << std::setw(12) << std::fixed
                << std::setprecision(2) << result.duration_ms;

      // Format throughput with appropriate units
      double throughput = result.throughput_ops_per_sec;
      if (throughput >= 1000000) {
        std::cout << std::setw(14) << std::fixed << std::setprecision(2)
                  << (throughput / 1000000.0) << "M/s";
      } else if (throughput >= 1000) {
        std::cout << std::setw(14) << std::fixed << std::setprecision(2)
                  << (throughput / 1000.0) << "K/s";
      } else {
        std::cout << std::setw(14) << std::fixed << std::setprecision(0)
                  << throughput << "/s";
      }

      // Speedup vs single
      if (result.speedup_vs_single > 0) {
        std::cout << std::setw(13) << std::fixed << std::setprecision(2)
                  << result.speedup_vs_single << "x";
      } else {
        std::cout << std::setw(13) << "-";
      }

      // Speedup vs old batch
      if (result.speedup_vs_old_batch > 0) {
        std::cout << std::setw(13) << std::fixed << std::setprecision(2)
                  << result.speedup_vs_old_batch << "x";
      } else {
        std::cout << std::setw(13) << "-";
      }

      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
};

}  // namespace ROCKSDB_NAMESPACE

int main(int /*argc*/, char** /*argv*/) {
  ROCKSDB_NAMESPACE::SkipListBatchBenchmark benchmark;
  benchmark.Run();
  return 0;
}
