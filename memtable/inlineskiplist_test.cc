//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "memtable/inlineskiplist.h"

#include <set>
#include <unordered_set>

#include "memory/concurrent_arena.h"
#include "rocksdb/env.h"
#include "test_util/testharness.h"
#include "util/hash.h"
#include "util/random.h"

namespace ROCKSDB_NAMESPACE {

// Our test skip list stores 8-byte unsigned integers
using Key = uint64_t;

static const char* Encode(const uint64_t* key) {
  return reinterpret_cast<const char*>(key);
}

static Key Decode(const char* key) {
  Key rv;
  memcpy(&rv, key, sizeof(Key));
  return rv;
}

struct TestComparator {
  using DecodedType = Key;

  static DecodedType decode_key(const char* b) { return Decode(b); }

  int operator()(const char* a, const char* b) const {
    if (Decode(a) < Decode(b)) {
      return -1;
    } else if (Decode(a) > Decode(b)) {
      return +1;
    } else {
      return 0;
    }
  }

  int operator()(const char* a, const DecodedType b) const {
    if (Decode(a) < b) {
      return -1;
    } else if (Decode(a) > b) {
      return +1;
    } else {
      return 0;
    }
  }
};

using TestInlineSkipList = InlineSkipList<TestComparator>;

class InlineSkipTest : public testing::Test {
 public:
  void Insert(TestInlineSkipList* list, Key key) {
    char* buf = list->AllocateKey(sizeof(Key));
    memcpy(buf, &key, sizeof(Key));
    list->Insert(buf);
    keys_.insert(key);
  }

  bool InsertWithHint(TestInlineSkipList* list, Key key, void** hint) {
    char* buf = list->AllocateKey(sizeof(Key));
    memcpy(buf, &key, sizeof(Key));
    bool res = list->InsertWithHint(buf, hint);
    keys_.insert(key);
    return res;
  }

  void Validate(TestInlineSkipList* list) {
    // Check keys exist.
    for (Key key : keys_) {
      ASSERT_TRUE(list->Contains(Encode(&key)));
    }
    // Iterate over the list, make sure keys appears in order and no extra
    // keys exist.
    TestInlineSkipList::Iterator iter(list);
    ASSERT_FALSE(iter.Valid());
    Key zero = 0;
    iter.Seek(Encode(&zero));
    for (Key key : keys_) {
      ASSERT_TRUE(iter.Valid());
      ASSERT_EQ(key, Decode(iter.key()));
      iter.Next();
    }
    ASSERT_FALSE(iter.Valid());
    // Validate the list is well-formed.
    list->TEST_Validate();
  }

 private:
  std::set<Key> keys_;
};

TEST_F(InlineSkipTest, Empty) {
  Arena arena;
  TestComparator cmp;
  InlineSkipList<TestComparator> list(cmp, &arena);
  Key key = 10;
  ASSERT_TRUE(!list.Contains(Encode(&key)));

  InlineSkipList<TestComparator>::Iterator iter(&list);
  ASSERT_TRUE(!iter.Valid());
  iter.SeekToFirst();
  ASSERT_TRUE(!iter.Valid());
  key = 100;
  iter.Seek(Encode(&key));
  ASSERT_TRUE(!iter.Valid());
  iter.SeekForPrev(Encode(&key));
  ASSERT_TRUE(!iter.Valid());
  iter.SeekToLast();
  ASSERT_TRUE(!iter.Valid());
}

TEST_F(InlineSkipTest, InsertAndLookup) {
  const int N = 2000;
  const int R = 5000;
  Random rnd(1000);
  std::set<Key> keys;
  ConcurrentArena arena;
  TestComparator cmp;
  InlineSkipList<TestComparator> list(cmp, &arena);
  for (int i = 0; i < N; i++) {
    Key key = rnd.Next() % R;
    if (keys.insert(key).second) {
      char* buf = list.AllocateKey(sizeof(Key));
      memcpy(buf, &key, sizeof(Key));
      list.Insert(buf);
    }
  }

  for (Key i = 0; i < R; i++) {
    if (list.Contains(Encode(&i))) {
      ASSERT_EQ(keys.count(i), 1U);
    } else {
      ASSERT_EQ(keys.count(i), 0U);
    }
  }

  // Simple iterator tests
  {
    InlineSkipList<TestComparator>::Iterator iter(&list);
    ASSERT_TRUE(!iter.Valid());

    uint64_t zero = 0;
    iter.Seek(Encode(&zero));
    ASSERT_TRUE(iter.Valid());
    ASSERT_EQ(*(keys.begin()), Decode(iter.key()));

    uint64_t max_key = R - 1;
    iter.SeekForPrev(Encode(&max_key));
    ASSERT_TRUE(iter.Valid());
    ASSERT_EQ(*(keys.rbegin()), Decode(iter.key()));

    iter.SeekToFirst();
    ASSERT_TRUE(iter.Valid());
    ASSERT_EQ(*(keys.begin()), Decode(iter.key()));

    iter.SeekToLast();
    ASSERT_TRUE(iter.Valid());
    ASSERT_EQ(*(keys.rbegin()), Decode(iter.key()));
  }

  // Forward iteration test
  for (Key i = 0; i < R; i++) {
    InlineSkipList<TestComparator>::Iterator iter(&list);
    iter.Seek(Encode(&i));

    // Compare against model iterator
    std::set<Key>::iterator model_iter = keys.lower_bound(i);
    for (int j = 0; j < 3; j++) {
      if (model_iter == keys.end()) {
        ASSERT_TRUE(!iter.Valid());
        break;
      } else {
        ASSERT_TRUE(iter.Valid());
        ASSERT_EQ(*model_iter, Decode(iter.key()));
        ++model_iter;
        iter.Next();
      }
    }
  }

  // Backward iteration test
  for (Key i = 0; i < R; i++) {
    InlineSkipList<TestComparator>::Iterator iter(&list);
    iter.SeekForPrev(Encode(&i));

    // Compare against model iterator
    std::set<Key>::iterator model_iter = keys.upper_bound(i);
    for (int j = 0; j < 3; j++) {
      if (model_iter == keys.begin()) {
        ASSERT_TRUE(!iter.Valid());
        break;
      } else {
        ASSERT_TRUE(iter.Valid());
        ASSERT_EQ(*--model_iter, Decode(iter.key()));
        iter.Prev();
      }
    }
  }
}

TEST_F(InlineSkipTest, InsertWithHint_Sequential) {
  const int N = 100000;
  Arena arena;
  TestComparator cmp;
  TestInlineSkipList list(cmp, &arena);
  void* hint = nullptr;
  for (int i = 0; i < N; i++) {
    Key key = i;
    InsertWithHint(&list, key, &hint);
  }
  Validate(&list);
}

TEST_F(InlineSkipTest, InsertWithHint_MultipleHints) {
  const int N = 100000;
  const int S = 100;
  Random rnd(534);
  Arena arena;
  TestComparator cmp;
  TestInlineSkipList list(cmp, &arena);
  void* hints[S];
  Key last_key[S];
  for (int i = 0; i < S; i++) {
    hints[i] = nullptr;
    last_key[i] = 0;
  }
  for (int i = 0; i < N; i++) {
    Key s = rnd.Uniform(S);
    Key key = (s << 32) + (++last_key[s]);
    InsertWithHint(&list, key, &hints[s]);
  }
  Validate(&list);
}

TEST_F(InlineSkipTest, InsertWithHint_MultipleHintsRandom) {
  const int N = 100000;
  const int S = 100;
  Random rnd(534);
  Arena arena;
  TestComparator cmp;
  TestInlineSkipList list(cmp, &arena);
  void* hints[S];
  for (int i = 0; i < S; i++) {
    hints[i] = nullptr;
  }
  for (int i = 0; i < N; i++) {
    Key s = rnd.Uniform(S);
    Key key = (s << 32) + rnd.Next();
    InsertWithHint(&list, key, &hints[s]);
  }
  Validate(&list);
}

TEST_F(InlineSkipTest, InsertWithHint_CompatibleWithInsertWithoutHint) {
  const int N = 100000;
  const int S1 = 100;
  const int S2 = 100;
  Random rnd(534);
  Arena arena;
  TestComparator cmp;
  TestInlineSkipList list(cmp, &arena);
  std::unordered_set<Key> used;
  Key with_hint[S1];
  Key without_hint[S2];
  void* hints[S1];
  for (int i = 0; i < S1; i++) {
    hints[i] = nullptr;
    while (true) {
      Key s = rnd.Next();
      if (used.insert(s).second) {
        with_hint[i] = s;
        break;
      }
    }
  }
  for (int i = 0; i < S2; i++) {
    while (true) {
      Key s = rnd.Next();
      if (used.insert(s).second) {
        without_hint[i] = s;
        break;
      }
    }
  }
  for (int i = 0; i < N; i++) {
    Key s = rnd.Uniform(S1 + S2);
    if (s < S1) {
      Key key = (with_hint[s] << 32) + rnd.Next();
      InsertWithHint(&list, key, &hints[s]);
    } else {
      Key key = (without_hint[s - S1] << 32) + rnd.Next();
      Insert(&list, key);
    }
  }
  Validate(&list);
}

#if !defined(ROCKSDB_VALGRIND_RUN) || defined(ROCKSDB_FULL_VALGRIND_RUN)
// We want to make sure that with a single writer and multiple
// concurrent readers (with no synchronization other than when a
// reader's iterator is created), the reader always observes all the
// data that was present in the skip list when the iterator was
// constructor.  Because insertions are happening concurrently, we may
// also observe new values that were inserted since the iterator was
// constructed, but we should never miss any values that were present
// at iterator construction time.
//
// We generate multi-part keys:
//     <key,gen,hash>
// where:
//     key is in range [0..K-1]
//     gen is a generation number for key
//     hash is hash(key,gen)
//
// The insertion code picks a random key, sets gen to be 1 + the last
// generation number inserted for that key, and sets hash to Hash(key,gen).
//
// At the beginning of a read, we snapshot the last inserted
// generation number for each key.  We then iterate, including random
// calls to Next() and Seek().  For every key we encounter, we
// check that it is either expected given the initial snapshot or has
// been concurrently added since the iterator started.
class ConcurrentTest {
 public:
  static const uint32_t K = 8;

 private:
  static uint64_t key(Key key) { return (key >> 40); }
  static uint64_t gen(Key key) { return (key >> 8) & 0xffffffffu; }
  static uint64_t hash(Key key) { return key & 0xff; }

  static uint64_t HashNumbers(uint64_t k, uint64_t g) {
    uint64_t data[2] = {k, g};
    return Hash(reinterpret_cast<char*>(data), sizeof(data), 0);
  }

  static Key MakeKey(uint64_t k, uint64_t g) {
    assert(sizeof(Key) == sizeof(uint64_t));
    assert(k <= K);  // We sometimes pass K to seek to the end of the skiplist
    assert(g <= 0xffffffffu);
    return ((k << 40) | (g << 8) | (HashNumbers(k, g) & 0xff));
  }

  static bool IsValidKey(Key k) {
    return hash(k) == (HashNumbers(key(k), gen(k)) & 0xff);
  }

  static Key RandomTarget(Random* rnd) {
    switch (rnd->Next() % 10) {
      case 0:
        // Seek to beginning
        return MakeKey(0, 0);
      case 1:
        // Seek to end
        return MakeKey(K, 0);
      default:
        // Seek to middle
        return MakeKey(rnd->Next() % K, 0);
    }
  }

  // Per-key generation
  struct State {
    std::atomic<int> generation[K];
    void Set(int k, int v) {
      generation[k].store(v, std::memory_order_release);
    }
    int Get(int k) { return generation[k].load(std::memory_order_acquire); }

    State() {
      for (unsigned int k = 0; k < K; k++) {
        Set(k, 0);
      }
    }
  };

  // Current state of the test
  State current_;

  ConcurrentArena arena_;

  // InlineSkipList is not protected by mu_.  We just use a single writer
  // thread to modify it.
  InlineSkipList<TestComparator> list_;

 public:
  ConcurrentTest() : list_(TestComparator(), &arena_) {}

  // REQUIRES: No concurrent calls to WriteStep or ConcurrentWriteStep
  void WriteStep(Random* rnd) {
    const uint32_t k = rnd->Next() % K;
    const int g = current_.Get(k) + 1;
    const Key new_key = MakeKey(k, g);
    char* buf = list_.AllocateKey(sizeof(Key));
    memcpy(buf, &new_key, sizeof(Key));
    list_.Insert(buf);
    current_.Set(k, g);
  }

  // REQUIRES: No concurrent calls for the same k
  void ConcurrentWriteStep(uint32_t k, bool use_hint = false) {
    const int g = current_.Get(k) + 1;
    const Key new_key = MakeKey(k, g);
    char* buf = list_.AllocateKey(sizeof(Key));
    memcpy(buf, &new_key, sizeof(Key));
    if (use_hint) {
      void* hint = nullptr;
      list_.InsertWithHintConcurrently(buf, &hint);
      delete[] reinterpret_cast<char*>(hint);
    } else {
      list_.InsertConcurrently(buf);
    }
    ASSERT_EQ(g, current_.Get(k) + 1);
    current_.Set(k, g);
  }

  void ReadStep(Random* rnd) {
    // Remember the initial committed state of the skiplist.
    State initial_state;
    for (unsigned int k = 0; k < K; k++) {
      initial_state.Set(k, current_.Get(k));
    }

    Key pos = RandomTarget(rnd);
    InlineSkipList<TestComparator>::Iterator iter(&list_);
    iter.Seek(Encode(&pos));
    while (true) {
      Key current;
      if (!iter.Valid()) {
        current = MakeKey(K, 0);
      } else {
        current = Decode(iter.key());
        ASSERT_TRUE(IsValidKey(current)) << current;
      }
      ASSERT_LE(pos, current) << "should not go backwards";

      // Verify that everything in [pos,current) was not present in
      // initial_state.
      while (pos < current) {
        ASSERT_LT(key(pos), K) << pos;

        // Note that generation 0 is never inserted, so it is ok if
        // <*,0,*> is missing.
        ASSERT_TRUE((gen(pos) == 0U) ||
                    (gen(pos) > static_cast<uint64_t>(initial_state.Get(
                                    static_cast<int>(key(pos))))))
            << "key: " << key(pos) << "; gen: " << gen(pos)
            << "; initgen: " << initial_state.Get(static_cast<int>(key(pos)));

        // Advance to next key in the valid key space
        if (key(pos) < key(current)) {
          pos = MakeKey(key(pos) + 1, 0);
        } else {
          pos = MakeKey(key(pos), gen(pos) + 1);
        }
      }

      if (!iter.Valid()) {
        break;
      }

      if (rnd->Next() % 2) {
        iter.Next();
        pos = MakeKey(key(pos), gen(pos) + 1);
      } else {
        Key new_target = RandomTarget(rnd);
        if (new_target > pos) {
          pos = new_target;
          iter.Seek(Encode(&new_target));
        }
      }
    }
  }
};
const uint32_t ConcurrentTest::K;

// Simple test that does single-threaded testing of the ConcurrentTest
// scaffolding.
TEST_F(InlineSkipTest, ConcurrentReadWithoutThreads) {
  ConcurrentTest test;
  Random rnd(test::RandomSeed());
  for (int i = 0; i < 10000; i++) {
    test.ReadStep(&rnd);
    test.WriteStep(&rnd);
  }
}

TEST_F(InlineSkipTest, ConcurrentInsertWithoutThreads) {
  ConcurrentTest test;
  Random rnd(test::RandomSeed());
  for (int i = 0; i < 10000; i++) {
    test.ReadStep(&rnd);
    uint32_t base = rnd.Next();
    for (int j = 0; j < 4; ++j) {
      test.ConcurrentWriteStep((base + j) % ConcurrentTest::K);
    }
  }
}

class TestState {
 public:
  ConcurrentTest t_;
  bool use_hint_;
  int seed_;
  std::atomic<bool> quit_flag_;
  std::atomic<uint32_t> next_writer_;

  enum ReaderState { STARTING, RUNNING, DONE };

  explicit TestState(int s)
      : seed_(s),
        quit_flag_(false),
        state_(STARTING),
        pending_writers_(0),
        state_cv_(&mu_) {}

  void Wait(ReaderState s) {
    mu_.Lock();
    while (state_ != s) {
      state_cv_.Wait();
    }
    mu_.Unlock();
  }

  void Change(ReaderState s) {
    mu_.Lock();
    state_ = s;
    state_cv_.Signal();
    mu_.Unlock();
  }

  void AdjustPendingWriters(int delta) {
    mu_.Lock();
    pending_writers_ += delta;
    if (pending_writers_ == 0) {
      state_cv_.Signal();
    }
    mu_.Unlock();
  }

  void WaitForPendingWriters() {
    mu_.Lock();
    while (pending_writers_ != 0) {
      state_cv_.Wait();
    }
    mu_.Unlock();
  }

 private:
  port::Mutex mu_;
  ReaderState state_;
  int pending_writers_;
  port::CondVar state_cv_;
};

static void ConcurrentReader(void* arg) {
  TestState* state = static_cast<TestState*>(arg);
  Random rnd(state->seed_);
  int64_t reads = 0;
  state->Change(TestState::RUNNING);
  while (!state->quit_flag_.load(std::memory_order_acquire)) {
    state->t_.ReadStep(&rnd);
    ++reads;
  }
  (void)reads;
  state->Change(TestState::DONE);
}

static void ConcurrentWriter(void* arg) {
  TestState* state = static_cast<TestState*>(arg);
  uint32_t k = state->next_writer_++ % ConcurrentTest::K;
  state->t_.ConcurrentWriteStep(k, state->use_hint_);
  state->AdjustPendingWriters(-1);
}

static void RunConcurrentRead(int run) {
  const int seed = test::RandomSeed() + (run * 100);
  Random rnd(seed);
  const int N = 1000;
  const int kSize = 1000;
  for (int i = 0; i < N; i++) {
    if ((i % 100) == 0) {
      fprintf(stderr, "Run %d of %d\n", i, N);
    }
    TestState state(seed + 1);
    Env::Default()->SetBackgroundThreads(1);
    Env::Default()->Schedule(ConcurrentReader, &state);
    state.Wait(TestState::RUNNING);
    for (int k = 0; k < kSize; ++k) {
      state.t_.WriteStep(&rnd);
    }
    state.quit_flag_.store(true, std::memory_order_release);
    state.Wait(TestState::DONE);
  }
}

static void RunConcurrentInsert(int run, bool use_hint = false,
                                int write_parallelism = 4) {
  Env::Default()->SetBackgroundThreads(1 + write_parallelism,
                                       Env::Priority::LOW);
  const int seed = test::RandomSeed() + (run * 100);
  Random rnd(seed);
  const int N = 1000;
  const int kSize = 1000;
  for (int i = 0; i < N; i++) {
    if ((i % 100) == 0) {
      fprintf(stderr, "Run %d of %d\n", i, N);
    }
    TestState state(seed + 1);
    state.use_hint_ = use_hint;
    Env::Default()->Schedule(ConcurrentReader, &state);
    state.Wait(TestState::RUNNING);
    for (int k = 0; k < kSize; k += write_parallelism) {
      state.next_writer_ = rnd.Next();
      state.AdjustPendingWriters(write_parallelism);
      for (int p = 0; p < write_parallelism; ++p) {
        Env::Default()->Schedule(ConcurrentWriter, &state);
      }
      state.WaitForPendingWriters();
    }
    state.quit_flag_.store(true, std::memory_order_release);
    state.Wait(TestState::DONE);
  }
}

TEST_F(InlineSkipTest, ConcurrentRead1) { RunConcurrentRead(1); }
TEST_F(InlineSkipTest, ConcurrentRead2) { RunConcurrentRead(2); }
TEST_F(InlineSkipTest, ConcurrentRead3) { RunConcurrentRead(3); }
TEST_F(InlineSkipTest, ConcurrentRead4) { RunConcurrentRead(4); }
TEST_F(InlineSkipTest, ConcurrentRead5) { RunConcurrentRead(5); }
TEST_F(InlineSkipTest, ConcurrentInsert1) { RunConcurrentInsert(1); }
TEST_F(InlineSkipTest, ConcurrentInsert2) { RunConcurrentInsert(2); }
TEST_F(InlineSkipTest, ConcurrentInsert3) { RunConcurrentInsert(3); }
TEST_F(InlineSkipTest, ConcurrentInsertWithHint1) {
  RunConcurrentInsert(1, true);
}
TEST_F(InlineSkipTest, ConcurrentInsertWithHint2) {
  RunConcurrentInsert(2, true);
}
TEST_F(InlineSkipTest, ConcurrentInsertWithHint3) {
  RunConcurrentInsert(3, true);
}

// Batch Insert Tests
TEST_F(InlineSkipTest, BatchInsertEmpty) {
  Arena arena;
  TestComparator cmp;
  InlineSkipList<TestComparator> list(cmp, &arena);

  const char* keys[10];
  size_t inserted = list.InsertBatch(keys, 0);
  ASSERT_EQ(0, inserted);
}

TEST_F(InlineSkipTest, BatchInsertSingle) {
  Arena arena;
  TestComparator cmp;
  InlineSkipList<TestComparator> list(cmp, &arena);

  Key key = 100;
  char* buf = list.AllocateKey(sizeof(Key));
  memcpy(buf, &key, sizeof(Key));
  const char* keys[1] = {buf};

  size_t inserted = list.InsertBatch(keys, 1);
  ASSERT_EQ(1, inserted);
  ASSERT_TRUE(list.Contains(Encode(&key)));
}

TEST_F(InlineSkipTest, BatchInsertSequential) {
  Arena arena;
  TestComparator cmp;
  InlineSkipList<TestComparator> list(cmp, &arena);

  const int N = 100;
  std::vector<const char*> keys;

  // Allocate and prepare keys
  for (int i = 0; i < N; i++) {
    Key key = i * 10;
    char* buf = list.AllocateKey(sizeof(Key));
    memcpy(buf, &key, sizeof(Key));
    keys.push_back(buf);
  }

  // Batch insert
  size_t inserted = list.InsertBatch(keys.data(), N);
  ASSERT_EQ(N, inserted);

  // Verify all keys are present
  for (int i = 0; i < N; i++) {
    Key key = i * 10;
    ASSERT_TRUE(list.Contains(Encode(&key)));
  }

  // Verify ordering
  InlineSkipList<TestComparator>::Iterator iter(&list);
  iter.SeekToFirst();
  for (int i = 0; i < N; i++) {
    ASSERT_TRUE(iter.Valid());
    ASSERT_EQ(i * 10, Decode(iter.key()));
    iter.Next();
  }
  ASSERT_FALSE(iter.Valid());
}

TEST_F(InlineSkipTest, BatchInsertRandom) {
  Arena arena;
  TestComparator cmp;
  InlineSkipList<TestComparator> list(cmp, &arena);
  Random rnd(301);

  const int N = 100;
  std::vector<const char*> keys;
  std::set<Key> key_set;

  // Allocate and prepare random keys
  for (int i = 0; i < N; i++) {
    Key key = rnd.Next() % 10000;
    if (key_set.insert(key).second) {
      char* buf = list.AllocateKey(sizeof(Key));
      memcpy(buf, &key, sizeof(Key));
      keys.push_back(buf);
    }
  }

  // Batch insert
  size_t inserted = list.InsertBatch(keys.data(), keys.size());
  ASSERT_EQ(keys.size(), inserted);

  // Verify all keys are present
  for (Key key : key_set) {
    ASSERT_TRUE(list.Contains(Encode(&key)));
  }

  // Verify ordering
  InlineSkipList<TestComparator>::Iterator iter(&list);
  iter.SeekToFirst();
  for (Key key : key_set) {
    ASSERT_TRUE(iter.Valid());
    ASSERT_EQ(key, Decode(iter.key()));
    iter.Next();
  }
  ASSERT_FALSE(iter.Valid());
}

TEST_F(InlineSkipTest, BatchInsertWithDuplicates) {
  Arena arena;
  TestComparator cmp;
  InlineSkipList<TestComparator> list(cmp, &arena);

  const int N = 20;
  std::vector<const char*> keys;

  // Allocate keys with duplicates (10 unique keys, each repeated twice)
  for (int i = 0; i < N; i++) {
    Key key = (i / 2) * 10;  // 0, 0, 10, 10, 20, 20, ...
    char* buf = list.AllocateKey(sizeof(Key));
    memcpy(buf, &key, sizeof(Key));
    keys.push_back(buf);
  }

  // Batch insert should skip duplicates
  size_t inserted = list.InsertBatch(keys.data(), N);
  ASSERT_EQ(10, inserted);  // Only 10 unique keys

  // Verify only unique keys are present
  InlineSkipList<TestComparator>::Iterator iter(&list);
  iter.SeekToFirst();
  for (int i = 0; i < 10; i++) {
    ASSERT_TRUE(iter.Valid());
    ASSERT_EQ(i * 10, Decode(iter.key()));
    iter.Next();
  }
  ASSERT_FALSE(iter.Valid());
}

TEST_F(InlineSkipTest, BatchInsertWithExistingKeys) {
  Arena arena;
  TestComparator cmp;
  InlineSkipList<TestComparator> list(cmp, &arena);

  // Insert some keys using regular insert
  for (int i = 0; i < 10; i++) {
    Key key = i * 10;
    char* buf = list.AllocateKey(sizeof(Key));
    memcpy(buf, &key, sizeof(Key));
    list.Insert(buf);
  }

  // Batch insert with some existing and some new keys
  const int N = 15;
  std::vector<const char*> keys;
  for (int i = 5; i < 20; i++) {  // Keys 50-190
    Key key = i * 10;
    char* buf = list.AllocateKey(sizeof(Key));
    memcpy(buf, &key, sizeof(Key));
    keys.push_back(buf);
  }

  // Should insert only new keys (100-190)
  size_t inserted = list.InsertBatch(keys.data(), N);
  ASSERT_EQ(10, inserted);  // Only 10 new keys (100-190)

  // Verify all keys are present
  InlineSkipList<TestComparator>::Iterator iter(&list);
  iter.SeekToFirst();
  for (int i = 0; i < 20; i++) {
    ASSERT_TRUE(iter.Valid());
    ASSERT_EQ(i * 10, Decode(iter.key()));
    iter.Next();
  }
  ASSERT_FALSE(iter.Valid());
}

TEST_F(InlineSkipTest, BatchInsertLarge) {
  Arena arena;
  TestComparator cmp;
  InlineSkipList<TestComparator> list(cmp, &arena);

  const int N = 1000;
  std::vector<const char*> keys;

  // Allocate and prepare keys
  for (int i = 0; i < N; i++) {
    Key key = i;
    char* buf = list.AllocateKey(sizeof(Key));
    memcpy(buf, &key, sizeof(Key));
    keys.push_back(buf);
  }

  // Batch insert
  size_t inserted = list.InsertBatch(keys.data(), N);
  ASSERT_EQ(N, inserted);

  // Verify all keys are present
  for (int i = 0; i < N; i++) {
    Key key = i;
    ASSERT_TRUE(list.Contains(Encode(&key)));
  }

  // Validate skiplist structure
  list.TEST_Validate();
}

TEST_F(InlineSkipTest, BatchInsertMultipleBatches) {
  Arena arena;
  TestComparator cmp;
  InlineSkipList<TestComparator> list(cmp, &arena);

  const int BATCH_SIZE = 20;
  const int NUM_BATCHES = 10;

  // Insert multiple batches
  for (int batch = 0; batch < NUM_BATCHES; batch++) {
    std::vector<const char*> keys;
    for (int i = 0; i < BATCH_SIZE; i++) {
      Key key = batch * BATCH_SIZE + i;
      char* buf = list.AllocateKey(sizeof(Key));
      memcpy(buf, &key, sizeof(Key));
      keys.push_back(buf);
    }

    size_t inserted = list.InsertBatch(keys.data(), BATCH_SIZE);
    ASSERT_EQ(BATCH_SIZE, inserted);
  }

  // Verify all keys are present and ordered
  InlineSkipList<TestComparator>::Iterator iter(&list);
  iter.SeekToFirst();
  for (int i = 0; i < BATCH_SIZE * NUM_BATCHES; i++) {
    ASSERT_TRUE(iter.Valid());
    ASSERT_EQ(i, Decode(iter.key()));
    iter.Next();
  }
  ASSERT_FALSE(iter.Valid());
}

TEST_F(InlineSkipTest, BatchInsertVsSequential) {
  // Compare correctness of batch insert vs sequential insert
  Arena arena1, arena2;
  TestComparator cmp;
  InlineSkipList<TestComparator> list1(cmp, &arena1);
  InlineSkipList<TestComparator> list2(cmp, &arena2);
  Random rnd(401);

  const int N = 200;
  std::vector<Key> test_keys;

  // Generate test keys
  for (int i = 0; i < N; i++) {
    test_keys.push_back(rnd.Next() % 5000);
  }

  // Insert using sequential insert
  for (Key key : test_keys) {
    char* buf = list1.AllocateKey(sizeof(Key));
    memcpy(buf, &key, sizeof(Key));
    list1.Insert(buf);
  }

  // Insert using batch insert (in smaller batches)
  const int BATCH_SIZE = 16;
  for (size_t i = 0; i < test_keys.size(); i += BATCH_SIZE) {
    size_t batch_size = std::min(BATCH_SIZE, (int)(test_keys.size() - i));
    std::vector<const char*> keys;

    for (size_t j = 0; j < batch_size; j++) {
      Key key = test_keys[i + j];
      char* buf = list2.AllocateKey(sizeof(Key));
      memcpy(buf, &key, sizeof(Key));
      keys.push_back(buf);
    }

    list2.InsertBatch(keys.data(), batch_size);
  }

  // Both lists should contain the same keys
  InlineSkipList<TestComparator>::Iterator iter1(&list1);
  InlineSkipList<TestComparator>::Iterator iter2(&list2);

  iter1.SeekToFirst();
  iter2.SeekToFirst();

  while (iter1.Valid() && iter2.Valid()) {
    ASSERT_EQ(Decode(iter1.key()), Decode(iter2.key()));
    iter1.Next();
    iter2.Next();
  }

  ASSERT_FALSE(iter1.Valid());
  ASSERT_FALSE(iter2.Valid());

  // Validate both lists
  list1.TEST_Validate();
  list2.TEST_Validate();
}

TEST_F(InlineSkipTest, BatchInsertReversed) {
  Arena arena;
  TestComparator cmp;
  InlineSkipList<TestComparator> list(cmp, &arena);

  const int N = 100;
  std::vector<const char*> keys;

  // Allocate keys in reverse order
  for (int i = N - 1; i >= 0; i--) {
    Key key = i * 10;
    char* buf = list.AllocateKey(sizeof(Key));
    memcpy(buf, &key, sizeof(Key));
    keys.push_back(buf);
  }

  // Batch insert
  size_t inserted = list.InsertBatch(keys.data(), N);
  ASSERT_EQ(N, inserted);

  // Verify keys are properly ordered despite reverse insertion
  InlineSkipList<TestComparator>::Iterator iter(&list);
  iter.SeekToFirst();
  for (int i = 0; i < N; i++) {
    ASSERT_TRUE(iter.Valid());
    ASSERT_EQ(i * 10, Decode(iter.key()));
    iter.Next();
  }
  ASSERT_FALSE(iter.Valid());

  list.TEST_Validate();
}

TEST_F(InlineSkipTest, BatchInsertInterleaved) {
  Arena arena;
  TestComparator cmp;
  InlineSkipList<TestComparator> list(cmp, &arena);

  // First batch: even numbers
  const int N = 50;
  std::vector<const char*> keys1;
  for (int i = 0; i < N; i++) {
    Key key = i * 2;
    char* buf = list.AllocateKey(sizeof(Key));
    memcpy(buf, &key, sizeof(Key));
    keys1.push_back(buf);
  }

  size_t inserted1 = list.InsertBatch(keys1.data(), N);
  ASSERT_EQ(N, inserted1);

  // Second batch: odd numbers
  std::vector<const char*> keys2;
  for (int i = 0; i < N; i++) {
    Key key = i * 2 + 1;
    char* buf = list.AllocateKey(sizeof(Key));
    memcpy(buf, &key, sizeof(Key));
    keys2.push_back(buf);
  }

  size_t inserted2 = list.InsertBatch(keys2.data(), N);
  ASSERT_EQ(N, inserted2);

  // Verify all keys are properly interleaved
  InlineSkipList<TestComparator>::Iterator iter(&list);
  iter.SeekToFirst();
  for (int i = 0; i < 2 * N; i++) {
    ASSERT_TRUE(iter.Valid());
    ASSERT_EQ(i, Decode(iter.key()));
    iter.Next();
  }
  ASSERT_FALSE(iter.Valid());

  list.TEST_Validate();
}

#endif  // !defined(ROCKSDB_VALGRIND_RUN) || defined(ROCKSDB_FULL_VALGRIND_RUN)
}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
