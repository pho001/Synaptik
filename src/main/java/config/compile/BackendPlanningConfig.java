package config.compile;

import config.optimizer.CpuPartitionConfig;

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
        PartitionOwnershipPlannerStrategy ownershipPlanner,
        PartitionSearchConfig search,
        CpuPartitionConfig cpuPartitions,
        BackendPlanningCostConfig cost
) {
    public BackendPlanningConfig {
        discoveryMode = discoveryMode == null ? BackendDiscoveryMode.CPU_ONLY : discoveryMode;
        failurePolicy = failurePolicy == null ? BackendPlanningFailurePolicy.OPTIONAL : failurePolicy;
        requirementScope = requirementScope == null ? BackendPlanningRequirementScope.ANY_TARGET : requirementScope;
        ownershipPlanner = ownershipPlanner == null ? PartitionOwnershipPlannerStrategy.ANCHOR : ownershipPlanner;
        search = search == null ? PartitionSearchConfig.defaults() : search;
        cpuPartitions = cpuPartitions == null ? CpuPartitionConfig.defaults() : cpuPartitions;
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
                PartitionOwnershipPlannerStrategy.ANCHOR,
                PartitionSearchConfig.defaults(),
                CpuPartitionConfig.defaults(),
                BackendPlanningCostConfig.conservative()
        );
    }

    public static BackendPlanningConfig explicitOnly() {
        return new BackendPlanningConfig(
                BackendDiscoveryMode.EXPLICIT,
                BackendPlanningFailurePolicy.OPTIONAL,
                BackendPlanningRequirementScope.ALL_EXPLICIT_INTENTS,
                Set.of(BackendTarget.GPU_METAL, BackendTarget.GPU_CUDA),
                PartitionOwnershipPlannerStrategy.ANCHOR,
                PartitionSearchConfig.defaults(),
                CpuPartitionConfig.defaults(),
                BackendPlanningCostConfig.conservative()
        );
    }

    public static BackendPlanningConfig autoAccelerator() {
        return new BackendPlanningConfig(
                BackendDiscoveryMode.AUTO,
                BackendPlanningFailurePolicy.OPTIONAL,
                BackendPlanningRequirementScope.ANY_TARGET,
                Set.of(BackendTarget.GPU_METAL),
                PartitionOwnershipPlannerStrategy.ANCHOR,
                PartitionSearchConfig.defaults(),
                CpuPartitionConfig.defaults(),
                BackendPlanningCostConfig.conservative()
        );
    }

    public static BackendPlanningConfig requireAnyAcceleratorPartition() {
        return autoAccelerator().withFailurePolicy(
                BackendPlanningFailurePolicy.REQUIRE_ACCELERATOR_PARTITION,
                BackendPlanningRequirementScope.ANY_TARGET
        );
    }

    public static BackendPlanningConfig requireEachAcceleratorTarget() {
        return autoAccelerator().withFailurePolicy(
                BackendPlanningFailurePolicy.REQUIRE_ACCELERATOR_PARTITION,
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
                cpuPartitions,
                cost
        );
    }

    public BackendPlanningConfig withOwnershipPlanner(PartitionOwnershipPlannerStrategy newPlanner) {
        return new BackendPlanningConfig(
                discoveryMode,
                failurePolicy,
                requirementScope,
                targets,
                newPlanner,
                search,
                cpuPartitions,
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
                cpuPartitions,
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
                cpuPartitions,
                cost
        );
    }

    public BackendPlanningConfig withCpuPartitions(CpuPartitionConfig newCpuPartitions) {
        return new BackendPlanningConfig(
                discoveryMode,
                failurePolicy,
                requirementScope,
                targets,
                ownershipPlanner,
                search,
                newCpuPartitions,
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
                cpuPartitions,
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
                cpuPartitions,
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
                && failurePolicy == BackendPlanningFailurePolicy.REQUIRE_ACCELERATOR_PARTITION) {
            throw new IllegalArgumentException("CPU_ONLY cannot require accelerator partitions");
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
