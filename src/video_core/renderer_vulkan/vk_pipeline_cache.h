// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/amdgpu/draw_state.h"
#include <functional>
#include <memory>
#include <variant>
#include <boost/container/small_vector.hpp>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/specialization.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_fixed_state.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"
#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/image_view.h"
#include "vulkan/vulkan.hpp"

template <>
struct std::hash<vk::ShaderModule> {
    std::size_t operator()(const vk::ShaderModule& module) const noexcept {
        return std::hash<size_t>{}(reinterpret_cast<size_t>((VkShaderModule)module));
    }
};

namespace AmdGpu {
class Liverpool;
}

namespace Serialization {
struct Archive;
}

namespace Shader {
struct Info;
}

namespace Vulkan {

class Instance;
class Scheduler;
class ShaderCache;

struct Program {
    struct Module {
        vk::ShaderModule module;
        Shader::StageSpecialization spec;
    };
    static constexpr size_t MaxPermutations = 8;
    using ModuleList = boost::container::small_vector<Module, MaxPermutations>;

    Shader::Info info;
    ModuleList modules{};
    size_t last_perm{~size_t{0}}; ///< Variant picked by the last lookup.

    Program() = default;
    Program(Shader::HwStage stage, Shader::SwStage l_stage, Shader::ShaderParams params)
        : info{stage, l_stage, params} {}

    void AddPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec) {
        modules.emplace_back(module, std::move(spec));
    }

    void InsertPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec,
                      size_t perm_idx) {
        modules.resize(std::max(modules.size(), perm_idx + 1)); // <-- beware of realloc
        modules[perm_idx] = {module, std::move(spec)};
    }
};

struct DrawIndirectParams {
    u16 vertex_sgpr_offset;
    u32 instance_sgpr_offset;
    bool is_indirect;
};

/// Per-draw state of one shader stage, captured by the command processor thread when it looks the
/// pipeline up (cp_split_mode). `info` is the raster thread's own copy of the stage's Info: the
/// shared one is rewritten by the next lookup while this draw is still being recorded.
/// cp_split_prepare_images: the pure part of one image resource of a stage, decoded on the command
/// processor thread from the plan's own copy of the user data.
struct PreparedImage {
    AmdGpu::Image tsharp{};
    u32 num_bindings{1};
    boost::container::small_vector<VideoCore::ImageInfo, 1> infos;
    boost::container::small_vector<VideoCore::ImageViewInfo, 1> views;
};

struct StagePlan {
    Shader::Info* info{};
    boost::container::small_vector<PreparedImage, 8> images;
    bool images_prepared{};
    std::array<u32, Shader::NUM_USER_DATA_REGS> user_data{};
    u32 user_data_count{};
    boost::container::small_vector<u32, 64> flat;
    VAddr pgm_base{};
};

struct DrawPlan {
    const GraphicsPipeline* pipeline{}; ///< Null if the pipeline lookup failed.
    FixedState fixed{};
    std::array<StagePlan, MaxShaderStages> stages{};
};

struct DispatchPlan {
    const ComputePipeline* pipeline{}; ///< Null if the pipeline lookup failed.
    StagePlan stage{};
};

class PipelineCache {
public:
    explicit PipelineCache(const Instance& instance, Scheduler& scheduler,
                           AmdGpu::Liverpool* liverpool, u32 sparse_page_shift);
    ~PipelineCache();

    void WarmUp();
    void Sync();

    bool LoadComputePipeline(Serialization::Archive& ar);
    bool LoadGraphicsPipeline(Serialization::Archive& ar);
    bool LoadPipelineStage(Serialization::Archive& ar, size_t stage);

    const GraphicsPipeline* GetGraphicsPipeline(const DrawIndirectParams params = {});

    const ComputePipeline* GetComputePipeline();

    /// cp_split_mode: looks the pipeline up for the command processor's registers `regs` and
    /// captures the per-draw state of its stages. Runs on the command processor thread, the only
    /// thread that touches this cache then.
    void PlanGraphics(const AmdGpu::DrawRegs& regs, const FixedState& fixed,
                      DrawIndirectParams params, DrawPlan& plan);
    void PlanCompute(const AmdGpu::ComputeProgram& cs, DispatchPlan& plan);
    /// Raster thread: writes the captured state into the stage Infos the pipeline uses.
    static void ApplyPlan(const DrawPlan& plan);
    static void ApplyPlan(const DispatchPlan& plan);

    /// Plan statistics since the last call: stage inputs compared with the previous draw.
    struct PlanStats {
        u64 draws{};
        u64 draws_all_stages_same{};
        u64 stages{};
        u64 stages_same{};
    };
    PlanStats TakePlanStats() noexcept {
        return std::exchange(plan_stats, PlanStats{});
    }

    using Result = std::tuple<const Shader::Info*, vk::ShaderModule, u64>;
    Result GetProgram(Shader::HwStage hw_stage, Shader::SwStage sw_stage,
                      const Shader::ShaderParams& params, Shader::Backend::Bindings& binding);

    std::optional<vk::ShaderModule> ReplaceShader(vk::ShaderModule module,
                                                  std::span<const u32> spv_code);

    static std::string GetShaderName(Shader::HwStage stage, u64 hash,
                                     std::optional<size_t> perm = {});

    auto& GetProfile() const {
        return profile;
    }

private:
    Shader::Info* RasterInfoFor(const Shader::Info* shared);
    static void CapturePlan(StagePlan& out, Shader::Info* clone, const Shader::Info& shared);
    bool RefreshGraphicsKey();
    bool RefreshGraphicsStages();
    bool RefreshComputeKey();

    void DumpShader(std::span<const u32> code, u64 hash, Shader::HwStage stage, size_t perm_idx,
                    std::string_view ext);
    std::optional<std::vector<u32>> GetShaderPatch(u64 hash, Shader::HwStage stage, size_t perm_idx,
                                                   std::string_view ext);
    vk::ShaderModule CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                   const std::span<const u32>& code, size_t perm_idx,
                                   Shader::Backend::Bindings& binding);
    const Shader::RuntimeInfo& BuildRuntimeInfo(Shader::HwStage stage, Shader::SwStage l_stage);

    [[nodiscard]] bool IsPipelineCacheDirty() const {
        return num_new_pipelines > 0;
    }

private:
    const Instance& instance;
    Scheduler& scheduler;
    AmdGpu::Liverpool* liverpool;
    DescriptorHeap desc_heap;
    vk::UniquePipelineCache pipeline_cache;
    vk::UniquePipelineLayout pipeline_layout;
    Shader::Profile profile{};
    Shader::Pools pools;
    DrawIndirectParams draw_indirect_params{};
    /// Registers the current lookup reads: the raster mirror normally, the command processor's
    /// own registers (and the compute state of the dispatching queue) when planning.
    const AmdGpu::DrawRegs* lookup_regs{};
    const FixedState* lookup_fixed{};
    FixedState lookup_fixed_storage{};
    const AmdGpu::ComputeProgram* lookup_cs{};
    PlanStats plan_stats{};
    std::array<u64, MaxShaderStages> last_stage_hash{};
    bool lookup_regs_pinned{}; ///< The lookup reads lookup_regs/lookup_cs as set by the caller.
    /// cp_split_mode: the raster thread's copy of each program's Info.
    tsl::robin_map<const Shader::Info*, std::unique_ptr<Shader::Info>> raster_infos;
    tsl::robin_map<size_t, std::unique_ptr<Program>> program_cache;
    tsl::robin_map<ComputePipelineKey, std::unique_ptr<ComputePipeline>> compute_pipelines;
    tsl::robin_map<GraphicsPipelineKey, std::unique_ptr<GraphicsPipeline>> graphics_pipelines;
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    std::array<const Shader::Info*, MaxShaderStages> infos{};
    std::array<vk::ShaderModule, MaxShaderStages> modules{};
    Shader::Gcn::FetchShaderData* fetch_shader{};
    GraphicsPipelineKey graphics_key{};
    ComputePipelineKey compute_key{};
    u32 num_new_pipelines{};      // new pipelines added to the cache since the game start
    const u32 stage_lookup_reuse; // 0 = off, 1 = check the last variant first, 2 = verify
    tsl::robin_set<u64> logged_stage_mismatches;

    // Only if Config::collectShadersForDebug()
    tsl::robin_map<vk::ShaderModule,
                   std::vector<std::variant<GraphicsPipelineKey, ComputePipelineKey>>>
        module_related_pipelines;
};

} // namespace Vulkan
