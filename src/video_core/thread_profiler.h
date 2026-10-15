// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string>
#include <nlohmann/json.hpp>
#include "common/types.h"

namespace VideoCore {

/// True on a thread that is unwinding a sampled stack: an access violation there is expected to
/// happen now and then (unwind data of a thread caught mid-prologue) and is handled by the
/// profiler, so the emulator's crash handler must not treat it as fatal.
bool IsProfilerUnwinding() noexcept;

/// Samples the call stack of the thread whose name contains `thread_name` for `seconds` at about
/// `hz` samples per second (Windows x64 only; an error object elsewhere). Returns for the busiest
/// functions the share of samples where the function was running ("self") and where it was
/// anywhere on the stack ("inclusive"). Functions of the executable are given as an offset from
/// the image base (resolve with the linker map), others as module name and offset.
///
/// The thread is suspended for a few microseconds per sample, so the sampled thread slows down by
/// about hz * 30 us per second. For development only.
nlohmann::json ProfileThread(const std::string& thread_name, double seconds, u32 hz, u32 top);

} // namespace VideoCore
