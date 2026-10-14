// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <chrono>
#include <string>
#include "common/types.h"

// Stub of the fork's readback trace: the call sites stay, nothing is recorded.
namespace VideoCore::ReadbackTrace {

inline bool Enabled() {
    return false;
}

inline u64 NowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

inline void NoteMark(const char*) {}
inline void NoteMark(const std::string&) {}

} // namespace VideoCore::ReadbackTrace
