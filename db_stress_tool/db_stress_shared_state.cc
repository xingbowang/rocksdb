//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
//

#ifdef GFLAGS
#include "db_stress_tool/db_stress_shared_state.h"

#include "db_stress_tool/db_stress_test_base.h"
#include "port/port.h"
#include "rocksdb/env.h"

namespace ROCKSDB_NAMESPACE {
namespace {

std::string SanitizePathComponent(const std::string& value) {
  std::string result;
  result.reserve(value.size());
  for (char c : value) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
        (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.') {
      result.push_back(c);
    } else {
      result.push_back('_');
    }
  }
  return result;
}

std::string SanitizeRecordValue(const std::string& value) {
  std::string result;
  result.reserve(value.size());
  for (char c : value) {
    if (c == '\n' || c == '\r' || c == '\t') {
      result.push_back(' ');
    } else {
      result.push_back(c);
    }
  }
  return result;
}

std::string OperationBreadcrumbFilePath(const ThreadState& thread) {
  std::string path = FLAGS_stress_diagnostics_dir;
  if (!path.empty() && path.back() != '/') {
    path.push_back('/');
  }
  const StressTest* const stress_test = thread.shared->GetStressTest();
  const std::string db_label =
      stress_test ? stress_test->GetDbLabel() : "unknown_db";
  path.append(SanitizePathComponent(db_label));
  path.append(".pid_");
  path.append(std::to_string(port::GetProcessID()));
  path.append(".thread_");
  path.append(std::to_string(thread.tid));
  path.append(".breadcrumbs.txt");
  return path;
}

}  // namespace

thread_local bool SharedState::ignore_read_error;

SharedState::SharedState(Env* env, StressTest* stress_test)
    : cv_(&mu_),
      env_(env != nullptr ? env : Env::Default()),
      seed_(static_cast<uint32_t>(FLAGS_seed)),
      max_key_(FLAGS_max_key),
      log2_keys_per_lock_(static_cast<uint32_t>(FLAGS_log2_keys_per_lock)),
      num_threads_(0),
      num_initialized_(0),
      num_populated_(0),
      vote_reopen_(0),
      num_done_(0),
      start_(false),
      start_verify_(false),
      operation_started_(false),
      operation_finished_(false),
      num_bg_threads_(0),
      should_stop_bg_thread_(false),
      bg_thread_finished_(0),
      stress_test_(stress_test),
      finished_ops_(0),
      successful_compactions_(0),
      successful_compactions_at_last_compaction_abort_(0),
      abort_resume_compactions_running_(false),
      verification_failure_(false),
      should_stop_test_(false),
      no_overwrite_ids_(GenerateNoOverwriteIds()),
      expected_state_manager_(nullptr),
      printing_verification_results_(false),
      start_timestamp_(env_->NowNanos()) {
  for (auto& completed_ops : completed_ops_by_type_) {
    completed_ops.store(0, std::memory_order_relaxed);
  }

  Status status;
  // TODO: We should introduce a way to explicitly disable verification
  // during shutdown. When that is disabled and FLAGS_expected_values_dir
  // is empty (disabling verification at startup), we can skip tracking
  // expected state. Only then should we permit bypassing the below feature
  // compatibility checks.
  const auto& expected_values_dir = stress_test_->GetExpectedValuesDir();
  if (!expected_values_dir.empty()) {
    if (!std::atomic<uint32_t>{}.is_lock_free() ||
        !std::atomic<uint64_t>{}.is_lock_free()) {
      std::ostringstream status_s;
      status_s << "Cannot use --expected_values_dir on platforms without "
                  "lock-free "
               << (!std::atomic<uint32_t>{}.is_lock_free()
                       ? "std::atomic<uint32_t>"
                       : "std::atomic<uint64_t>");
      status = Status::InvalidArgument(status_s.str());
    }

    if (status.ok() && FLAGS_clear_column_family_one_in > 0) {
      status = Status::InvalidArgument(
          "Cannot use --expected_values_dir on when "
          "--clear_column_family_one_in is greater than zero.");
    }
  }
  if (status.ok()) {
    if (expected_values_dir.empty()) {
      expected_state_manager_.reset(
          new AnonExpectedStateManager(FLAGS_max_key, FLAGS_column_families));
    } else {
      expected_state_manager_.reset(new FileExpectedStateManager(
          FLAGS_max_key, FLAGS_column_families, expected_values_dir));
    }
    status = expected_state_manager_->Open();
  }
  DB_STRESS_ASSERT_OK_MSG(status, "Failed setting up expected state");

  if (FLAGS_test_batches_snapshots) {
    fprintf(stdout, "No lock creation because test_batches_snapshots set\n");
    return;
  }

  long num_locks = static_cast<long>(max_key_ >> log2_keys_per_lock_);
  if (max_key_ & ((1 << log2_keys_per_lock_) - 1)) {
    num_locks++;
  }
  fprintf(stdout, "Creating %ld locks\n", num_locks * FLAGS_column_families);
  key_locks_.resize(FLAGS_column_families);

  for (int i = 0; i < FLAGS_column_families; ++i) {
    key_locks_[i].reset(new port::Mutex[num_locks]);
  }
  if (FLAGS_read_fault_one_in || FLAGS_metadata_read_fault_one_in) {
#ifdef NDEBUG
    // Unsupported in release mode because it relies on
    // `IGNORE_STATUS_IF_ERROR` to distinguish faults not expected to lead to
    // failure.
    fprintf(stderr,
            "Cannot set nonzero value for --read_fault_one_in in "
            "release mode.");
    exit(1);  // NOLINT(concurrency-mt-unsafe)
#else         // NDEBUG
    SyncPoint::GetInstance()->SetCallBack("FaultInjectionIgnoreError",
                                          IgnoreReadErrorCallback);
    SyncPoint::GetInstance()->EnableProcessing();
#endif        // NDEBUG
  }
}

bool SharedState::BeginOperation(uint32_t tid, StressOperationType type) {
  assert(tid < static_cast<uint32_t>(num_threads_));
  ThreadOperationState& state = thread_operation_states_[tid];
  const uint32_t active_type =
      state.active_type.load(std::memory_order_acquire);
  assert(active_type == static_cast<uint32_t>(StressOperationType::kNone));
  if (active_type != static_cast<uint32_t>(StressOperationType::kNone)) {
    return false;
  }
  state.started_micros.store(env_->NowMicros(), std::memory_order_relaxed);
  state.active_type.store(static_cast<uint32_t>(type),
                          std::memory_order_release);
  return true;
}

ThreadState::ThreadState(uint32_t index, SharedState* _shared)
    : tid(index),
      rand(1000 + index + _shared->GetSeed()),
      shared(_shared),
      operation_breadcrumb_pos(0),
      operation_breadcrumb_wrapped(false),
      diagnostic_io_disabled(false),
      operation_breadcrumb_failure_flushed(false),
      operation_ordinal(0),
      current_operation_ordinal(0) {
  if (OperationBreadcrumbsEnabled()) {
    operation_breadcrumbs.resize(
        static_cast<size_t>(FLAGS_stress_diagnostics_breadcrumb_entries));
  }
}

bool ThreadState::OperationBreadcrumbsEnabled() const {
  return FLAGS_stress_diagnostics_breadcrumbs &&
         FLAGS_stress_diagnostics_breadcrumb_entries > 0;
}

StressDiagnosticRecord& ThreadState::AppendOperationBreadcrumb(
    StressOperationType type, const char* phase) {
  assert(OperationBreadcrumbsEnabled());
  assert(!operation_breadcrumbs.empty());

  StressDiagnosticRecord& record =
      operation_breadcrumbs[operation_breadcrumb_pos];
  record.operation_ordinal = current_operation_ordinal;
  record.timestamp_micros = shared->GetEnv()->NowMicros();
  record.operation_type = type;
  record.phase = phase;
  record.details.clear();

  operation_breadcrumb_pos =
      (operation_breadcrumb_pos + 1) % operation_breadcrumbs.size();
  if (operation_breadcrumb_pos == 0) {
    operation_breadcrumb_wrapped = true;
  }
  return record;
}

std::string* ThreadState::RecordOperationEvent(StressOperationType type) {
  if (!OperationBreadcrumbsEnabled() || operation_breadcrumbs.empty()) {
    return nullptr;
  }
  return &AppendOperationBreadcrumb(type, "event").details;
}

bool ThreadState::BeginOperation(StressOperationType type) {
  ++operation_ordinal;
  current_operation_ordinal = operation_ordinal;

  if (OperationBreadcrumbsEnabled() && !operation_breadcrumbs.empty()) {
    AppendOperationBreadcrumb(type, "begin");
  }

  if (LivenessTrackingEnabled()) {
    return shared->BeginOperation(tid, type);
  }
  return false;
}

void ThreadState::RecordOperationEnd(StressOperationType type) {
  if (!OperationBreadcrumbsEnabled() || operation_breadcrumbs.empty()) {
    return;
  }
  StressDiagnosticRecord& record = AppendOperationBreadcrumb(type, "end");

  const bool verification_failed = shared->HasVerificationFailedYet();
  record.details = verification_failed ? "verification_failure=1" : "";

  if (verification_failed) {
    FlushOperationBreadcrumbsOnVerificationFailure();
    return;
  }

  if (FLAGS_stress_diagnostics_breadcrumb_flush_every > 0 &&
      current_operation_ordinal %
              FLAGS_stress_diagnostics_breadcrumb_flush_every ==
          0) {
    FlushOperationBreadcrumbs("periodic");
  }
}

void ThreadState::FlushOperationBreadcrumbsOnVerificationFailure() {
  if (operation_breadcrumb_failure_flushed) {
    return;
  }
  operation_breadcrumb_failure_flushed = true;
  FlushOperationBreadcrumbs("verification_failure");
}

void ThreadState::FlushOperationBreadcrumbs(const char* reason) {
  if (!OperationBreadcrumbsEnabled() || operation_breadcrumbs.empty() ||
      diagnostic_io_disabled || FLAGS_stress_diagnostics_dir.empty() ||
      (!operation_breadcrumb_wrapped && operation_breadcrumb_pos == 0)) {
    return;
  }

  Env* const env = Env::Default();
  Status s = env->CreateDirIfMissing(FLAGS_stress_diagnostics_dir);
  if (!s.ok()) {
    fprintf(stdout, "Failed to create stress diagnostics directory %s: %s\n",
            FLAGS_stress_diagnostics_dir.c_str(), s.ToString().c_str());
    diagnostic_io_disabled = true;
    return;
  }

  const std::string path = OperationBreadcrumbFilePath(*this);
  const std::string temp_path = path + ".tmp";
  std::unique_ptr<WritableFile> file;
  s = env->NewWritableFile(temp_path, &file, EnvOptions());
  if (!s.ok()) {
    fprintf(stdout, "Failed to create operation breadcrumb file %s: %s\n",
            temp_path.c_str(), s.ToString().c_str());
    diagnostic_io_disabled = true;
    return;
  }

  std::string output;
  output.append("# reason=");
  output.append(reason ? reason : "unknown");
  output.append(" pid=");
  output.append(std::to_string(port::GetProcessID()));
  output.append(" tid=");
  output.append(std::to_string(tid));
  output.append(" seed=");
  output.append(std::to_string(shared->GetSeed()));
  output.push_back('\n');

  const size_t entry_count = operation_breadcrumb_wrapped
                                 ? operation_breadcrumbs.size()
                                 : operation_breadcrumb_pos;
  const size_t first_entry =
      operation_breadcrumb_wrapped ? operation_breadcrumb_pos : 0;
  for (size_t i = 0; i < entry_count; ++i) {
    const size_t index = (first_entry + i) % operation_breadcrumbs.size();
    const StressDiagnosticRecord& record = operation_breadcrumbs[index];
    output.append("time_micros=");
    output.append(std::to_string(record.timestamp_micros));
    output.append(" tid=");
    output.append(std::to_string(tid));
    output.append(" ordinal=");
    output.append(std::to_string(record.operation_ordinal));
    output.append(" op=");
    output.append(StressOperationTypeName(record.operation_type));
    output.append(" phase=");
    output.append(record.phase);
    if (!record.details.empty()) {
      output.push_back(' ');
      output.append(SanitizeRecordValue(record.details));
    }
    output.push_back('\n');
  }

  Status write_status = file->Append(Slice(output));
  Status close_status = file->Close();
  file.reset();
  if (!write_status.ok()) {
    s = write_status;
  } else if (!close_status.ok()) {
    s = close_status;
  } else {
    s = env->RenameFile(temp_path, path);
  }
  if (!s.ok()) {
    env->DeleteFile(temp_path).PermitUncheckedError();
    fprintf(stdout, "Failed to write operation breadcrumb file %s: %s\n",
            path.c_str(), s.ToString().c_str());
    diagnostic_io_disabled = true;
  }
}

bool SharedState::ShouldVerifyAtBeginning() const {
  return !stress_test_->GetExpectedValuesDir().empty();
}

}  // namespace ROCKSDB_NAMESPACE
#endif  // GFLAGS
