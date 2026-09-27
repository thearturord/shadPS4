// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>

#include "common/logging/log.h"
#include "common/spin_lock.h"
#include "common/thread.h"

#if _MSC_VER
#include <intrin.h>
#if _M_AMD64
#define __x86_64__ 1
#endif
#if _M_ARM64
#define __aarch64__ 1
#endif
#else
#if __x86_64__
#include <xmmintrin.h>
#endif
#endif

namespace {

void ThreadPause() {
#if __x86_64__
    _mm_pause();
#elif __aarch64__ && _MSC_VER
    __yield();
#elif __aarch64__
    asm("yield");
#endif
}

} // Anonymous namespace

namespace Common {

void SpinLock::lock() {
    if (lck.test_and_set(std::memory_order_acquire)) {
        LockSlow();
    }
    owner.store(std::this_thread::get_id(), std::memory_order_relaxed);
}

void SpinLock::LockSlow() {
    // Hang check: a spin lock is meant for short sections, so one held for seconds is a
    // deadlock, most likely this thread faulting while it already holds the lock.
    const auto start = std::chrono::steady_clock::now();
    u32 spins = 0;
    bool reported = false;
    while (lck.test_and_set(std::memory_order_acquire)) {
        ThreadPause();
        if (!reported && (++spins & 0xFFFFF) == 0 &&
            std::chrono::steady_clock::now() - start > std::chrono::seconds{2}) {
            reported = true;
            const bool self = owner.load(std::memory_order_relaxed) == std::this_thread::get_id();
            LOG_WARNING(Common, "Hang check: thread {} spinning >2s on spin lock {}{}",
                        Common::GetCurrentThreadName(), fmt::ptr(this),
                        self ? ", which this same thread already holds (self deadlock)" : "");
        }
    }
}

void SpinLock::unlock() {
    owner.store({}, std::memory_order_relaxed);
    lck.clear(std::memory_order_release);
}

bool SpinLock::try_lock() {
    if (lck.test_and_set(std::memory_order_acquire)) {
        return false;
    }
    owner.store(std::this_thread::get_id(), std::memory_order_relaxed);
    return true;
}

} // namespace Common
