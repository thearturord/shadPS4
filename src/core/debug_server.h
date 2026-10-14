// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>
#include <map>
#include <string>
#include <nlohmann/json.hpp>
#include "common/types.h"

// A small read-only inspection server for development (setting debug_server_port, 0 = off).
//
// Listens on 127.0.0.1 only and answers GET requests with JSON, so tools (for example the MCP
// wrapper in the spike folder) can read live counters, guest memory and logs from the running
// emulator without restarting it. It never changes emulator state.
//
//   /              list of endpoints
//   /status        uptime, presented frames, fps
//   /threads       CPU use of the busiest threads since the previous call
//   /memory        ?addr=0x..&size=N (max 64 KB) guest memory as hex
//   /log           ?lines=N&filter=text&file=shad_log.txt tail of a log file
//   /faults        CPU write faults on tracked pages by 4 MB region since the previous call
//   /frames        ?n=600 the last n frame intervals with their time stamps
//   /periodic      last runs of periodic jobs (time and cost)
//   /policy        current readback policy switches (video_core/readback_policy.h)
//   /set           ?name=value changes one of them (development only)
//   /stats         figures published by the subsystems (Publish), latest values
//   other endpoints are registered by the subsystems (see Register)
namespace Core::DebugServer {

using Params = std::map<std::string, std::string>;
using Handler = std::function<nlohmann::json(const Params&)>;

/// Starts the server on `port` (127.0.0.1). Does nothing if it already runs or the port is 0.
void Start(u16 port);
void Stop();

/// Adds or replaces an endpoint. Can be called before the server starts.
void Register(const std::string& path, const std::string& description, Handler handler);
void Unregister(const std::string& path);

/// Stores the latest value of a named group of figures; /stats returns all groups. Any thread.
void Publish(const std::string& group, nlohmann::json value);

/// Milliseconds since the server's clock started (the time base of /frames and /periodic).
double UptimeMs();

/// Records that a periodic job named `name` just finished and took `cost_ms`; the last 24 runs of
/// each job are returned by /periodic, so dips in the frame rate can be matched to them.
void RecordPeriodic(const std::string& name, double cost_ms);

/// Frames presented so far.
u64 PresentCount();

/// Nanoseconds since the last present (0 before the first).
u64 SincePresentNs();

/// Called once per presented frame.
void NotePresent();

/// Called for every CPU write to a page the emulator tracks. Cheap, any thread.
void NoteWriteFault(VAddr address);

} // namespace Core::DebugServer
