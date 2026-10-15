// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <nlohmann/json.hpp>
#include "common/logging/log.h"
#include "core/debug_server.h"
#include "core/native_draw.h"

namespace Core::NativeDraw {

namespace {

// ---- reading the engine's objects -----------------------------------------------------------------------------
// Offsets: documents/tlg/spike/js_ctx_state.h (JsCtx / JsObj). Everything here reads guest memory directly; the
// context and the objects it points to are guest heap memory owned by the game.
template <typename T>
inline T Rd(u64 base, u64 offset) {
    T value;
    std::memcpy(&value, reinterpret_cast<const void*>(base + offset), sizeof(T));
    return value;
}

namespace Ctx {
constexpr u64 CurrentIndexBuffer = 0x58;
constexpr u64 InputLayout = 0x68;
constexpr u64 VertexBufferSlots = 0x70;
constexpr u64 VertexStrides = 0x170;
constexpr u64 VertexOffsets = 0x1f0;
constexpr u64 Topology = 0x4480;
constexpr u64 RenderTargetCount = 0x4484;
constexpr u64 RenderTargets = 0x4488;
constexpr u64 DepthStencilView = 0x44d0;
constexpr u64 Viewport = 0x44ec;
constexpr u64 PixelShader = 0x4690;
constexpr u64 VertexShader = 0x4698;
constexpr u64 RasterizerState = 0x46c8;
constexpr u64 DepthStencilState = 0x46e0;
constexpr u64 StencilRef = 0x46e8;
constexpr u64 BlendState = 0x46f0;
constexpr u64 BlendFactor = 0x46f8;
constexpr u64 InstanceCount = 0x4724;
constexpr u64 ConstantBufferTable(u64 s) { return 0x280 + s * 0x80; }
constexpr u64 ShaderResourceTable(u64 s) { return 0x580 + s * 0x400; }
constexpr u64 SamplerTable(u64 s) { return 0x1d80 + s * 0x80; }
} // namespace Ctx

namespace Shader {
// The stage block starts at shader + 0x30; its used-slot masks are at +0xb0 (constant buffers), +0xb4 (samplers),
// +0xc0/+0xc8 (shader resources, 128 bits).
constexpr u64 CbMask = 0x30 + 0xb0;
constexpr u64 SamplerMask = 0x30 + 0xb4;
constexpr u64 SrvMaskLo = 0x30 + 0xc0;
constexpr u64 SrvMaskHi = 0x30 + 0xc8;
} // namespace Shader

namespace Layout {
constexpr u64 ElementCount = 0x430;
constexpr u64 Elements = 0x38;
constexpr u64 ElementStride = 0x20;
} // namespace Layout

// ---- hashing -----------------------------------------------------------------------------------------------------
struct Hasher {
    u64 h = 0xcbf29ce484222325ull;
    void Add(u64 v) {
        h ^= v;
        h *= 0x100000001b3ull;
        h ^= h >> 29;
    }
};

enum Group : u32 {
    Shaders,
    InputObjects, // layout, index buffer, topology
    Targets,      // render target views, depth view
    States,       // blend, rasterizer, depth-stencil, viewport, factors
    ConstantBuffers,
    ShaderResources,
    Samplers,
    VertexBuffers, // vertex buffer objects, strides, offsets of the slots the layout reads
    Args,          // draw arguments
    NumGroups,
};
constexpr const char* GroupNames[NumGroups] = {"shaders",         "layout_ib_topology", "targets",
                                               "fixed_function",  "constant_buffers",   "shader_resources",
                                               "samplers",        "vertex_buffers",     "draw_args"};

struct Signature {
    std::array<u64, NumGroups> group{};
    bool valid = false;
};

void HashSlots(Hasher& h, u64 ctx, u64 table, u64 mask_lo, u64 mask_hi) {
    for (u32 slot = 0; slot < 128; ++slot) {
        const u64 bit = slot < 64 ? (mask_lo >> slot) & 1 : (mask_hi >> (slot - 64)) & 1;
        if (bit) {
            h.Add(Rd<u64>(ctx, table + slot * 8));
        }
    }
}

Signature Compute(u64 ctx, u64 a1, u64 a2, u64 a3, u32 kind) {
    Signature s;
    const u64 vs = Rd<u64>(ctx, Ctx::VertexShader);
    const u64 ps = Rd<u64>(ctx, Ctx::PixelShader);
    const u64 layout = Rd<u64>(ctx, Ctx::InputLayout);
    {
        Hasher h;
        h.Add(vs);
        h.Add(ps);
        s.group[Shaders] = h.h;
    }
    {
        Hasher h;
        h.Add(layout);
        h.Add(Rd<u64>(ctx, Ctx::CurrentIndexBuffer));
        h.Add(Rd<u32>(ctx, Ctx::Topology));
        s.group[InputObjects] = h.h;
    }
    {
        Hasher h;
        const u32 count = Rd<u32>(ctx, Ctx::RenderTargetCount);
        h.Add(count);
        for (u32 i = 0; i < 7; ++i) {
            h.Add(Rd<u64>(ctx, Ctx::RenderTargets + i * 8));
        }
        h.Add(Rd<u64>(ctx, Ctx::DepthStencilView));
        s.group[Targets] = h.h;
    }
    {
        Hasher h;
        h.Add(Rd<u64>(ctx, Ctx::BlendState));
        h.Add(Rd<u64>(ctx, Ctx::RasterizerState));
        h.Add(Rd<u64>(ctx, Ctx::DepthStencilState));
        h.Add(Rd<u8>(ctx, Ctx::StencilRef));
        h.Add(Rd<u64>(ctx, Ctx::BlendFactor));
        h.Add(Rd<u64>(ctx, Ctx::BlendFactor + 8));
        for (u32 i = 0; i < 6; ++i) {
            h.Add(Rd<u32>(ctx, Ctx::Viewport + i * 4));
        }
        s.group[States] = h.h;
    }
    {
        Hasher cb, srv, smp;
        const std::array<u64, 2> shaders{vs, ps};
        for (u64 stage = 0; stage < 2; ++stage) {
            const u64 shader = shaders[stage];
            if (!shader) {
                continue;
            }
            const u32 cb_mask = Rd<u32>(shader, Shader::CbMask) & 0xffff;
            for (u32 slot = 0; slot < 16; ++slot) {
                if ((cb_mask >> slot) & 1) {
                    cb.Add(Rd<u64>(ctx, Ctx::ConstantBufferTable(stage) + slot * 8));
                }
            }
            HashSlots(srv, ctx, Ctx::ShaderResourceTable(stage), Rd<u64>(shader, Shader::SrvMaskLo),
                      Rd<u64>(shader, Shader::SrvMaskHi));
            const u32 smp_mask = Rd<u32>(shader, Shader::SamplerMask) & 0xffff;
            for (u32 slot = 0; slot < 16; ++slot) {
                if ((smp_mask >> slot) & 1) {
                    smp.Add(Rd<u64>(ctx, Ctx::SamplerTable(stage) + slot * 8));
                }
            }
            cb.Add(0xffff0000 + stage);
            srv.Add(0xffff0000 + stage);
            smp.Add(0xffff0000 + stage);
        }
        s.group[ConstantBuffers] = cb.h;
        s.group[ShaderResources] = srv.h;
        s.group[Samplers] = smp.h;
    }
    {
        Hasher h;
        if (layout) {
            const u32 n = Rd<u32>(layout, Layout::ElementCount);
            for (u32 i = 0; i < n && i < 32; ++i) {
                const u32 slot = Rd<u32>(layout, Layout::Elements + i * Layout::ElementStride + 8);
                if (slot < 16) {
                    h.Add(Rd<u64>(ctx, Ctx::VertexBufferSlots + slot * 8));
                    h.Add(Rd<u32>(ctx, Ctx::VertexStrides + slot * 4));
                    h.Add(Rd<u32>(ctx, Ctx::VertexOffsets + slot * 4));
                }
            }
        }
        s.group[VertexBuffers] = h.h;
    }
    {
        Hasher h;
        h.Add(kind);
        h.Add(a1);
        h.Add(a2);
        h.Add(a3);
        h.Add(Rd<u32>(ctx, Ctx::InstanceCount));
        s.group[Args] = h.h;
    }
    s.valid = true;
    return s;
}

// ---- statistics --------------------------------------------------------------------------------------------------
enum Class : u32 { ZPass, GBuffer, Forward, Other, NumClasses };
constexpr const char* ClassNames[NumClasses] = {"z_prepass", "g_buffer", "forward", "other"};

struct Counters {
    std::atomic<u64> draws{};
    std::array<std::atomic<u64>, NumGroups> changed{};
    std::atomic<u64> state_changed{};     // any of shaders, layout/ib/topology, targets, fixed function
    std::atomic<u64> resources_changed{}; // any of constant buffers, shader resources, samplers, vertex buffers
    std::atomic<u64> nothing_changed{};   // everything but the draw arguments identical to the previous draw
    std::atomic<u64> only_args_and_vb{};  // state and cb/srv/samplers identical, only vertex buffers / args differ
};
std::array<Counters, NumClasses> counters;
std::atomic<VAddr> image_base{0};
std::atomic<u64> last_publish_ns{0};

// Ghidra address ranges of the three passes (documents/tlg/native-phase1-results-2026-10-15.md).
Class Classify(u64 ret) {
    const VAddr base = image_base.load(std::memory_order_relaxed);
    if (base == 0 || ret < base) {
        return Other;
    }
    const u64 g = ret - base + 0x1000000;
    if (g >= 0x1c74a00 && g < 0x1c76710) {
        return ZPass;
    }
    if (g >= 0x1c76710 && g < 0x1c779a0) {
        return GBuffer;
    }
    if (g >= 0x1c779a0 && g < 0x1c78f00) {
        return Forward;
    }
    return Other;
}

u64 NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void Publish() {
    nlohmann::json out = nlohmann::json::object();
    for (u32 c = 0; c < NumClasses; ++c) {
        auto& k = counters[c];
        const u64 draws = k.draws.exchange(0);
        if (draws == 0) {
            continue;
        }
        nlohmann::json entry = nlohmann::json::object();
        entry["draws"] = draws;
        const auto pct = [&](std::atomic<u64>& counter) {
            return 100.0 * static_cast<double>(counter.exchange(0)) / static_cast<double>(draws);
        };
        std::array<double, NumGroups> p{};
        nlohmann::json changed = nlohmann::json::object();
        for (u32 g = 0; g < NumGroups; ++g) {
            p[g] = pct(k.changed[g]);
            changed[GroupNames[g]] = p[g];
        }
        entry["percent_of_draws_where_this_differs_from_the_previous_draw"] = changed;
        entry["percent_any_state_changed"] = pct(k.state_changed);
        entry["percent_any_resource_changed"] = pct(k.resources_changed);
        entry["percent_only_vertex_buffers_and_args_differ"] = pct(k.only_args_and_vb);
        entry["percent_everything_but_args_identical"] = pct(k.nothing_changed);
        out[ClassNames[c]] = std::move(entry);
        LOG_INFO(Core,
                 "native draw probe {}: {} draws; differs from the previous draw: shaders {:.1f}% layout/ib {:.1f}% "
                 "targets {:.1f}% states {:.1f}% cb {:.1f}% srv {:.1f}% samplers {:.1f}% vb {:.1f}%",
                 ClassNames[c], draws, p[Shaders], p[InputObjects], p[Targets], p[States], p[ConstantBuffers],
                 p[ShaderResources], p[Samplers], p[VertexBuffers]);
    }
    DebugServer::Publish("native_draw_probe", std::move(out));
}

thread_local Signature previous;

u64 Probe(u64 ctx, u64 a1, u64 a2, u64 a3, u64 ret, u32 kind) {
    if (ctx == 0) {
        return 0;
    }
    const Class cls = Classify(ret);
    const Signature now = Compute(ctx, a1, a2, a3, kind);
    auto& k = counters[cls];
    k.draws.fetch_add(1, std::memory_order_relaxed);
    if (previous.valid) {
        bool state = false;
        for (u32 g = 0; g < NumGroups; ++g) {
            if (now.group[g] != previous.group[g]) {
                k.changed[g].fetch_add(1, std::memory_order_relaxed);
            }
        }
        for (u32 g : {Shaders, InputObjects, Targets, States}) {
            state |= now.group[g] != previous.group[g];
        }
        bool resources = false;
        for (u32 g : {ConstantBuffers, ShaderResources, Samplers, VertexBuffers}) {
            resources |= now.group[g] != previous.group[g];
        }
        const bool cb_srv_smp = now.group[ConstantBuffers] != previous.group[ConstantBuffers] ||
                                now.group[ShaderResources] != previous.group[ShaderResources] ||
                                now.group[Samplers] != previous.group[Samplers];
        if (state) {
            k.state_changed.fetch_add(1, std::memory_order_relaxed);
        }
        if (resources) {
            k.resources_changed.fetch_add(1, std::memory_order_relaxed);
        }
        if (!state && !cb_srv_smp) {
            k.only_args_and_vb.fetch_add(1, std::memory_order_relaxed);
        }
        if (!state && !resources) {
            k.nothing_changed.fetch_add(1, std::memory_order_relaxed);
        }
    }
    previous = now;

    const u64 t = NowNs();
    u64 last = last_publish_ns.load(std::memory_order_relaxed);
    if (t - last > 5'000'000'000ull && last_publish_ns.compare_exchange_strong(last, t)) {
        if (last != 0) {
            Publish();
        }
    }
    return 0;
}

u64 PS4_SYSV_ABI ProbeIndexed(u64 ctx, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret) {
    return Probe(ctx, a1, a2, a3, ret, 0);
}
u64 PS4_SYSV_ABI ProbeAuto(u64 ctx, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret) {
    return Probe(ctx, a1, a2, a3, ret, 1);
}
u64 PS4_SYSV_ABI ProbeIndirect(u64 ctx, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret) {
    return Probe(ctx, a1, a2, a3, ret, 2);
}

} // namespace

u64 FindHandler(const std::string& name) {
    if (name == "draw_probe_indexed") {
        return reinterpret_cast<u64>(&ProbeIndexed);
    }
    if (name == "draw_probe_auto") {
        return reinterpret_cast<u64>(&ProbeAuto);
    }
    if (name == "draw_probe_indirect") {
        return reinterpret_cast<u64>(&ProbeIndirect);
    }
    return 0;
}

void SetImageBase(VAddr base) {
    image_base.store(base, std::memory_order_relaxed);
}

} // namespace Core::NativeDraw
