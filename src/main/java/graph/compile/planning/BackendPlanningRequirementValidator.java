package graph.compile.planning;

import config.compile.BackendPlanningConfig;
import config.compile.BackendPlanningFailurePolicy;
import config.compile.BackendPlanningRequirementScope;
import config.compile.BackendTarget;
import graph.compile.planning.partition.Partition;

import java.util.EnumSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Validates compile-time backend planning requirements against accepted partitions.
 */
final class BackendPlanningRequirementValidator {
    private BackendPlanningRequirementValidator() {
    }

    static void validateRequired(
            BackendPlanningConfig config,
            List<ExplicitBackendIntent> explicitIntents,
            List<Partition> partitions
    ) {
        if (config.failurePolicy() == BackendPlanningFailurePolicy.OPTIONAL) {
            return;
        }
        if (config.failurePolicy() == BackendPlanningFailurePolicy.REQUIRE_ACCELERATOR_REGION) {
            Set<BackendTarget> accepted = acceptedAcceleratorTargets(partitions);
            boolean ok = config.requirementScope() == BackendPlanningRequirementScope.EACH_TARGET
                    ? accepted.containsAll(config.targets())
                    : !accepted.isEmpty();
            if (!ok) {
                throw new IllegalStateException("Required accelerator backend planning produced no legal region");
            }
        }
        if (config.failurePolicy() == BackendPlanningFailurePolicy.REQUIRE_ALL_EXPLICIT_INTENTS) {
            List<ExplicitBackendIntent> missing = missingExplicitIntents(explicitIntents, partitions);
            if (!missing.isEmpty()) {
                throw new IllegalStateException("One or more explicit backend intents could not be planned: " + describe(missing));
            }
        }
    }

    private static List<ExplicitBackendIntent> missingExplicitIntents(
            List<ExplicitBackendIntent> explicitIntents,
            List<Partition> partitions
    ) {
        Set<ExplicitBackendIntent> accepted = acceptedExplicitIntents(partitions);
        return (explicitIntents == null ? List.<ExplicitBackendIntent>of() : explicitIntents).stream()
                .filter(intent -> !accepted.contains(intent))
                .toList();
    }

    private static Set<ExplicitBackendIntent> acceptedExplicitIntents(List<Partition> partitions) {
        LinkedHashSet<ExplicitBackendIntent> out = new LinkedHashSet<>();
        for (Partition partition : partitions == null ? List.<Partition>of() : partitions) {
            BackendTarget target = BackendTarget.fromPartitionTarget(partition.target());
            if (target == null || !target.accelerator()) {
                continue;
            }
            for (int nodeId : partition.orderedNodeIds()) {
                out.add(new ExplicitBackendIntent(nodeId, target));
            }
        }
        return Set.copyOf(out);
    }

    private static Set<BackendTarget> acceptedAcceleratorTargets(List<Partition> partitions) {
        EnumSet<BackendTarget> out = EnumSet.noneOf(BackendTarget.class);
        for (Partition partition : partitions == null ? List.<Partition>of() : partitions) {
            BackendTarget target = BackendTarget.fromPartitionTarget(partition.target());
            if (target != null && target.accelerator()) {
                out.add(target);
            }
        }
        return out;
    }

    private static String describe(List<ExplicitBackendIntent> intents) {
        return intents.stream()
                .map(intent -> "node " + intent.nodeId() + " -> " + intent.target())
                .toList()
                .toString();
    }
}
