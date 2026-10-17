// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <nlohmann/json_fwd.hpp>
#include "common/types.h"

// Explicit readbacks, step 1 (documents/tlg/native-explicit-readbacks.md): only measures. Records which memory the engine's copy and
// counter read-back records write for the CPU to read (their destinations), which ranges the CPU maps through `AsyncMap`, and which
// CPU read faults hit those ranges, the pages they share with other data, or neither. Debug server endpoint /readbacks.
namespace Core::NativeReadbacks {

/// A copy or counter read-back record writes [address, address + size): `usage` is the Usage of the destination resource (0 default,
/// 1 immutable, 2 dynamic, 3 staging, ~0 unknown), `kind` the record kind (9 counter, 10 copy), `caller` the Ghidra address of the call.
void NoteDestination(VAddr address, u64 size, u32 usage, u32 kind, u64 caller);

/// The engine mapped [address, address + size) for the CPU (`AsyncMap`), from the Ghidra address `caller`.
void NoteMap(VAddr address, u64 size, u64 caller, u32 first_dword = 0);

/// The CPU read a page the GPU wrote and the emulator is about to download it.
void NoteReadFault(VAddr address, u64 size);

/// Explicit staging readbacks are on (bit 33554432 of `native_sdk_strip`; needs the entry list, 8388608, and `readback_dma_async`).
bool ExplicitEnabled();

/// An explicit readback was done (a copy into a STAGING resource: downloaded in order, destination not marked GPU modified), or refused
/// (`reason`: 0 the copy took another path first, 1 larger than the download limit).
void NoteExplicit(bool done, u64 bytes, u32 reason);

nlohmann::json Report();

} // namespace Core::NativeReadbacks
