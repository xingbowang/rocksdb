//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "memory/arena.h"
#include "memtable/inlineskiplist.h"
#include "util/random.h"

namespace ROCKSDB_NAMESPACE {

typedef uint64_t Key;

struct TestComparator {
  typedef Key DecodedType;

  static DecodedType decode_key(const char* b) {
    Key result;
    memcpy(&result, b, sizeof(Key));
    return result;
  }

  int operator()(const char* a, const char* b) const {
    Key ka = decode_key(a);
    Key kb = decode_key(b);
    if (ka < kb) {
      return -1;
    } else if (ka > kb) {
      return 1;
    } else {
      return 0;
    }
  }

  int operator()(const char* a, const DecodedType kb) const {
    Key ka = decode_key(a);
    if (ka < kb) {
      return -1;
    } else if (ka > kb) {
      return 1;
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
    double speedup;
  };

  void Run() {
    PrintHeader();

    // Test with different workload sizes
    std::vector<size_t> workload_sizes = {10000, 50000, 100000};

    for (size_t num_keys : workload_sizes) {
      std::cout << "\n=== Workload: " << num_keys << " keys ===" << std::endl;
      std::cout << std::string(80, '-') << std::endl;

      // Test sequential keys
      std::cout << "\nSequential Keys:" << std::endl;
      RunBenchmarkSet(num_keys, false);

      // Test random keys
      std::cout << "\nRandom Keys:" << std::endl;
      RunBenchmarkSet(num_keys, true);
    }
  }

 private:
  void PrintHeader() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════"
                 "══════════╗\n";
    std::cout << "║        SkipList Batch Insert Benchmark with Prefetching    "
                 "         ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════"
                 "══════════╝\n";
    std::cout << std::endl;
  }

  void RunBenchmarkSet(size_t num_keys, bool random_keys) {
    std::vector<BenchResult> results;

    // Baseline: single insert
    double baseline_time = BenchmarkSingleInsert(num_keys, random_keys);
    BenchResult baseline = {"Single Insert",
                            num_keys,
                            1,
                            baseline_time,
                            num_keys * 1000.0 / baseline_time,
                            1.0};
    results.push_back(baseline);

    // Test different batch sizes
    std::vector<size_t> batch_sizes = {2, 4, 8, 16, 32, 64};

    for (size_t batch_size : batch_sizes) {
      double time_ms = BenchmarkBatchInsert(num_keys, batch_size, random_keys);
      BenchResult result = {"Batch Insert",
                            num_keys,
                            batch_size,
                            time_ms,
                            num_keys * 1000.0 / time_ms,
                            baseline_time / time_ms};
      results.push_back(result);
    }

    // Print results
    PrintResults(results);
  }

  double BenchmarkSingleInsert(size_t num_keys, bool random_keys) {
    Arena arena;
    TestComparator cmp;
    InlineSkipList<TestComparator> skiplist(cmp, &arena);
    Random rnd(301);

    std::vector<Key> keys = GenerateKeys(num_keys, random_keys, &rnd);

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < num_keys; ++i) {
      char* buf = skiplist.AllocateKey(sizeof(Key));
      memcpy(buf, &keys[i], sizeof(Key));
      skiplist.Insert(buf);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    return duration.count() / 1000.0;  // Convert to milliseconds
  }

  double BenchmarkBatchInsert(size_t num_keys, size_t batch_size,
                              bool random_keys) {
    Arena arena;
    TestComparator cmp;
    InlineSkipList<TestComparator> skiplist(cmp, &arena);
    Random rnd(301);

    std::vector<Key> keys = GenerateKeys(num_keys, random_keys, &rnd);

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < num_keys; i += batch_size) {
      size_t current_batch_size = std::min(batch_size, num_keys - i);
      std::vector<const char*> batch;
      batch.reserve(current_batch_size);

      for (size_t j = 0; j < current_batch_size; ++j) {
        char* buf = skiplist.AllocateKey(sizeof(Key));
        memcpy(buf, &keys[i + j], sizeof(Key));
        batch.push_back(buf);
      }

      skiplist.InsertBatch(batch.data(), current_batch_size);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    return duration.count() / 1000.0;  // Convert to milliseconds
  }

  std::vector<Key> GenerateKeys(size_t num_keys, bool random, Random* rnd) {
    std::vector<Key> keys;
    keys.reserve(num_keys);

    if (random) {
      // Generate random keys
      for (size_t i = 0; i < num_keys; ++i) {
        keys.push_back(rnd->Next64());
      }
    } else {
      // Generate sequential keys with gaps
      for (size_t i = 0; i < num_keys; ++i) {
        keys.push_back(i * 100);
      }
    }

    return keys;
  }

  void PrintResults(const std::vector<BenchResult>& results) {
    std::cout << std::endl;
    std::cout << std::left << std::setw(16) << "Method" << std::setw(12)
              << "Batch Size" << std::setw(14) << "Time (ms)" << std::setw(18)
              << "Throughput (ops/s)" << std::setw(10) << "Speedup"
              << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    for (const auto& result : results) {
      std::cout << std::left << std::setw(16) << result.name << std::setw(12)
                << result.batch_size << std::setw(14) << std::fixed
                << std::setprecision(2) << result.duration_ms << std::setw(18)
                << std::fixed << std::setprecision(0)
                << result.throughput_ops_per_sec << std::setw(10) << std::fixed
                << std::setprecision(2) << result.speedup << "x" << std::endl;
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
