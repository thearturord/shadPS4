// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

// Guest function hooks (x86_64 only).
//
// Counts calls to chosen functions of the game's eboot, and optionally records their first six integer
// arguments and return address. Nothing happens unless <serial>.hooks.json exists in the custom configs
// directory. See the format below.
//
//   {
//     "ghidra_image_base": "0x1000000",           // optional, default 0x1000000
//     "ring_entries": 16384,                      // optional, power of two, for "args" hooks
//     "hooks": [
//       { "name": "Map",   "ghidra": "0x247f990" },                       // mode "count" (default)
//       { "name": "Copy",  "ghidra": "0x2480b70", "mode": "args" },       // count + argument ring
//       { "name": "Other", "offset": "0x123456" }                         // offset from the eboot base
//     ]
//   }
//
// Every hook replaces the first whole instructions (5 bytes or more) of the function with a jump to a
// stub in the module's trampoline area. The stub updates the counter or ring, runs the copied
// instructions, and jumps back. Functions whose first instructions contain a branch or a relative
// operand are refused and reported in the log.
//
// An "args" hook may also list "deref" entries: values read from guest memory at
// <argument register> + offset when the frame is written (not at call time, so objects must outlive the
// call; unreadable addresses give ffffffffffffffff):
//   { "name": "Map", "ghidra": "0x247f990", "mode": "args",
//     "deref": [ { "reg": "rsi", "off": "0x30", "size": 8 }, { "reg": "rsi", "off": "0x124", "size": 4 } ] }
//
// A hook with "mode": "snap" records, at call time, up to eight values read from [argument register +
// offset] (the objects the arguments point to), next to the return address:
//   { "name": "DrawIndexedCommon", "ghidra": "0x248e080", "mode": "snap",
//     "snap": [ { "reg": "rdi", "off": "0x4698", "size": 8 }, ... ] }
// They go to guest_hooks_snap.csv (frame,name,seq,ret,s0..s7); snap hooks write nothing to the args file.
//
// A hook with "mode": "call" and "handler": "<name>" calls a host function registered in native_draw.cpp at every
// call of the guest function, with the first six integer arguments and the return address. The handler returns
// non-zero to skip the guest function. Call hooks keep no counters or files.
//
// Output (log directory): guest_hooks.csv (frame,name,calls,total), for "args" hooks
// guest_hooks_args.csv (frame,name,seq,rdi,rsi,rdx,rcx,r8,r9,ret), for hooks with "deref"
// guest_hooks_deref.csv (frame,name,seq,d0..d5), and write_faults.csv (frame,page) with the pages
// the CPU wrote while the emulator tracked them (any hooks file turns this on).
namespace Core::GuestHooks {

/// Installs the hooks listed in the configuration file into the eboot loaded at `base`.
/// Must run before the guest starts executing.
void InstallForEboot(VAddr base, u64 image_size);

/// Called once per presented frame: writes the call counts and recorded arguments.
void OnFrame();

/// True when at least one hook is installed.
bool IsActive();

/// Records a CPU write to pages the emulator tracks: from the fault handler (size 1, the faulting guest
/// instruction was set with SetFaultRip) or from an explicit invalidation by a system call (rip 0). Cheap, any thread.
void NoteWriteFault(VAddr address, u64 size = 1);

/// The guest instruction that caused the write fault being handled on this thread (0 = not a fault), and the
/// first three stack words at that moment (the caller's return address when the faulting code is a leaf
/// function such as memcpy).
void SetFaultRip(u64 rip, const u64* stack_words = nullptr);

} // namespace Core::GuestHooks
