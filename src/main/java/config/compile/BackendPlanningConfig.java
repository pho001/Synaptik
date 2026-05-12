package config.compile;

import config.optimizer.CpuRegionConfig;

import java.util.EnumSet;
import java.util.Objects;
import java.util.Set;

/**
 * Compile-time backend ownership planning policy.
 */
public record BackendPlanningConfig(
        BackendDiscoveryMode discoveryMode,
        BackendPlanningFailurePolicy failurePolicy,
        BackendPlanningRequirementScope requirementScope,
        Set<BackendTarget> targets,
        RegionOwnershipPlannerStrategy ownershipPlanner,
        PartitionSearchConfig search,
        CpuRegionConfig cpuRegions,
        BackendPlanningCostConfig cost
) {
    public BackendPlanningConfig {
        discoveryMode = discoveryMode == null ? BackendDiscoveryMode.CPU_ONLY : discoveryMode;
        failurePolicy = failurePolicy == null ? BackendPlanningFailurePolicy.OPTIONAL : failurePolicy;
        requirementScope = requirementScope == null ? BackendPlanningRequirementScope.ANY_TARGET : requirementScope;
        ownershipPlanner = ownershipPlanner == null ? RegionOwnershipPlannerStrategy.ANCHOR : ownershipPlanner;
        search = search == null ? PartitionSearchConfig.defaults() : search;
        cpuRegions = cpuRegions == null ? CpuRegionConfig.defaults() : cpuRegions;
        cost = cost == null ? BackendPlanningCostConfig.conservative() : cost;
        targets = normalizeTargets(discoveryMode, targets);
        validate(discoveryMode, failurePolicy, requirementScope, targets);
    }

    public static BackendPlanningConfig cpuOnly() {
        return new BackendPlanningConfig(
                BackendDiscoveryMode.CPU_ONLY,
                BackendPlanningFailurePolicy.OPTIONAL,
                BackendPlanningRequirementScope.ANY_TARGET,
                Set.of(),
                RegionOwnershipPlannerStrategy.ANCHOR,
                PartitionSearchConfig.defaults(),
                CpuRegionConfig.defaults(),
                BackendPlanningCostConfig.conservative()
        );
    }

    public static BackendPlanningConfig explicitOnly() {
        return new BackendPlanningConfig(
                BackendDiscoveryMode.EXPLICIT,
                BackendPlanningFailurePolicy.OPTIONAL,
                BackendPlanningRequirementScope.ALL_EXPLICIT_INTENTS,
                Set.of(BackendTarget.GPU_METAL, BackendTarget.GPU_CUDA),
                RegionOwnershipPlannerStrategy.ANCHOR,
                PartitionSearchConfig.defaults(),
                CpuRegionConfig.defaults(),
                BackendPlanningCostConfig.conservative()
        );
    }

    public static BackendPlanningConfig autoAccelerator() {
        return new BackendPlanningConfig(
                BackendDiscoveryMode.AUTO,
                BackendPlanningFailurePolicy.OPTIONAL,
                BackendPlanningRequirementScope.ANY_TARGET,
                Set.of(BackendTarget.GPU_METAL),
                RegionOwnershipPlannerStrategy.ANCHOR,
                PartitionSearchConfig.defaults(),
                CpuRegionConfig.defaults(),
                BackendPlanningCostConfig.conservative()
        );
    }

    public static BackendPlanningConfig requireAnyAcceleratorRegion() {
        return autoAccelerator().withFailurePolicy(
                BackendPlanningFailurePolicy.REQUIRE_ACCELERATOR_REGION,
                BackendPlanningRequirementScope.ANY_TARGET
        );
    }

    public static BackendPlanningConfig requireEachAcceleratorTarget() {
        return autoAccelerator().withFailurePolicy(
                BackendPlanningFailurePolicy.REQUIRE_ACCELERATOR_REGION,
                BackendPlanningRequirementScope.EACH_TARGET
        );
    }

    public static BackendPlanningConfig requireAllExplicitIntents() {
        return explicitOnly().withFailurePolicy(
                BackendPlanningFailurePolicy.REQUIRE_ALL_EXPLICIT_INTENTS,
                BackendPlanningRequirementScope.ALL_EXPLICIT_INTENTS
        );
    }

    public BackendPlanningConfig withDiscoveryMode(BackendDiscoveryMode newMode) {
        return new BackendPlanningConfig(
                newMode,
                failurePolicy,
                requirementScope,
                targets,
                ownershipPlanner,
                search,
                cpuRegions,
                cost
        );
    }

    public BackendPlanningConfig withOwnershipPlanner(RegionOwnershipPlannerStrategy newPlanner) {
        return new BackendPlanningConfig(
                discoveryMode,
                failurePolicy,
                requirementScope,
                targets,
                newPlanner,
                search,
                cpuRegions,
                cost
        );
    }

    public BackendPlanningConfig withTargets(Set<BackendTarget> newTargets) {
        return new BackendPlanningConfig(
                discoveryMode,
                failurePolicy,
                requirementScope,
                newTargets,
                ownershipPlanner,
                search,
                cpuRegions,
                cost
        );
    }

    public BackendPlanningConfig withSearch(PartitionSearchConfig newSearch) {
        return new BackendPlanningConfig(
                discoveryMode,
                failurePolicy,
                requirementScope,
                targets,
                ownershipPlanner,
                newSearch,
                cpuRegions,
                cost
        );
    }

    public BackendPlanningConfig withCpuRegions(CpuRegionConfig newCpuRegions) {
        return new BackendPlanningConfig(
                discoveryMode,
                failurePolicy,
                requirementScope,
                targets,
                ownershipPlanner,
                search,
                newCpuRegions,
                cost
        );
    }

    public BackendPlanningConfig withCost(BackendPlanningCostConfig newCost) {
        return new BackendPlanningConfig(
                discoveryMode,
                failurePolicy,
                requirementScope,
                targets,
                ownershipPlanner,
                search,
                cpuRegions,
                newCost
        );
    }

    public BackendPlanningConfig withFailurePolicy(
            BackendPlanningFailurePolicy newFailurePolicy,
            BackendPlanningRequirementScope newRequirementScope
    ) {
        return new BackendPlanningConfig(
                discoveryMode,
                newFailurePolicy,
                newRequirementScope,
                targets,
                ownershipPlanner,
                search,
                cpuRegions,
                cost
        );
    }

    private static Set<BackendTarget> normalizeTargets(BackendDiscoveryMode mode, Set<BackendTarget> input) {
        if (mode == BackendDiscoveryMode.CPU_ONLY) {
            return Set.of();
        }
        EnumSet<BackendTarget> normalized = EnumSet.noneOf(BackendTarget.class);
        if (input != null) {
            for (BackendTarget target : input) {
                normalized.add(Objects.requireNonNull(target, "targets cannot contain null"));
            }
        }
        if (normalized.isEmpty()) {
            normalized.add(BackendTarget.GPU_METAL);
        }
        normalized.remove(BackendTarget.CPU);
        return Set.copyOf(normalized);
    }

    private static void validate(
            BackendDiscoveryMode discoveryMode,
            BackendPlanningFailurePolicy failurePolicy,
            BackendPlanningRequirementScope requirementScope,
            Set<BackendTarget> targets
    ) {
        if (discoveryMode == BackendDiscoveryMode.CPU_ONLY
                && failurePolicy == BackendPlanningFailurePolicy.REQUIRE_ACCELERATOR_REGION) {
            throw new IllegalArgumentException("CPU_ONLY cannot require accelerator regions");
        }
        if (failurePolicy == BackendPlanningFailurePolicy.REQUIRE_ALL_EXPLICIT_INTENTS
                && discoveryMode == BackendDiscoveryMode.CPU_ONLY) {
            throw new IllegalArgumentException("CPU_ONLY cannot require explicit accelerator intents");
        }
        if (discoveryMode != BackendDiscoveryMode.CPU_ONLY && targets.isEmpty()) {
            throw new IllegalArgumentException("Accelerator planning requires at least one accelerator target");
        }
        if (failurePolicy == BackendPlanningFailurePolicy.REQUIRE_ALL_EXPLICIT_INTENTS
                && requirementScope != BackendPlanningRequirementScope.ALL_EXPLICIT_INTENTS) {
            throw new IllegalArgumentException("REQUIRE_ALL_EXPLICIT_INTENTS requires ALL_EXPLICIT_INTENTS scope");
        }
    }
}
