package graph.compile;

import config.compile.BackendPlanningConfig;
import config.compile.BackendTarget;
import graph.optimizer.partition.Partition;
import graph.optimizer.partition.PartitionTarget;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BackendPlanningRequirementValidatorTest {

    @Test
    void requireAllExplicitIntentsValidatesEveryExplicitNode() {
        ArrayList<BackendPlanningDiagnostic> diagnostics = new ArrayList<>();

        IllegalStateException failure = assertThrows(IllegalStateException.class, () ->
                BackendPlanningRequirementValidator.validateRequired(
                        BackendPlanningConfig.requireAllExplicitIntents(),
                        List.of(
                                new ExplicitBackendIntent(1, BackendTarget.GPU_METAL),
                                new ExplicitBackendIntent(2, BackendTarget.GPU_METAL)
                        ),
                        List.of(partition("metal-1", PartitionTarget.GPU_METAL, List.of(1))),
                        diagnostics
                ));

        assertTrue(failure.getMessage().contains("node 2 -> GPU_METAL"));
        assertTrue(diagnostics.stream().anyMatch(diagnostic ->
                "REQUIRED_EXPLICIT_BACKEND_INTENT_MISSING".equals(diagnostic.code())
                        && diagnostic.message().contains("node 2 -> GPU_METAL")));
    }

    @Test
    void requireAllExplicitIntentsAcceptsMatchingNodeAndTargetPairs() {
        assertDoesNotThrow(() ->
                BackendPlanningRequirementValidator.validateRequired(
                        BackendPlanningConfig.requireAllExplicitIntents(),
                        List.of(
                                new ExplicitBackendIntent(1, BackendTarget.GPU_METAL),
                                new ExplicitBackendIntent(2, BackendTarget.GPU_CUDA)
                        ),
                        List.of(
                                partition("metal-1", PartitionTarget.GPU_METAL, List.of(1)),
                                partition("cuda-2", PartitionTarget.GPU_CUDA, List.of(2))
                        ),
                        new ArrayList<>()
                ));
    }

    private static Partition partition(String id, PartitionTarget target, List<Integer> nodeIds) {
        return new Partition(
                id,
                target,
                nodeIds,
                List.of(),
                List.of(),
                List.of(),
                List.of(),
                nodeIds.getFirst(),
                List.of(),
                List.of(),
                List.of(),
                nodeIds.size(),
                null,
                null,
                null
        );
    }
}
