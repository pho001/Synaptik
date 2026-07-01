package backend.lowering.partition;

import backend.lowering.LoweringFamily;
import planning.partition.PartitionTarget;
import org.junit.jupiter.api.Test;
import operations.Operation;
import tensor.DataType;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BackendPartitionExecutionPlanTest {
    @Test
    void requiresAnchorToBePartOfOrderedNodes() {
        assertThrows(IllegalArgumentException.class, () -> new BackendPartitionExecutionPlan(
                "partition",
                LoweringFamily.METAL_GRAPH_PARTITION,
                9,
                List.of(1, 2, 3),
                List.of(0),
                List.of(3),
                List.of(),
                List.of(),
                PartitionCost.ofWork(10L),
                PartitionDecision.selected("METAL_GRAPH_PARTITION", "test"),
                EmptyPartitionPayload.INSTANCE
        ));
    }

    @Test
    void normalizesNullCollectionsAndPayload() {
        BackendPartitionExecutionPlan plan = new BackendPartitionExecutionPlan(
                "partition",
                LoweringFamily.CUDA_GRAPH_PARTITION,
                2,
                List.of(1, 2),
                null,
                null,
                null,
                null,
                null,
                null,
                null
        );

        assertEquals(List.of(), plan.externalInputNodeIds());
        assertEquals(List.of(), plan.boundaryOutputNodeIds());
        assertEquals(List.of(), plan.nodePlans());
        assertEquals(List.of(), plan.executionGroups());
        assertTrue(plan.backendPayload() instanceof EmptyPartitionPayload);
        assertTrue(plan.decision().selected());
    }

    @Test
    void requiresPartitionId() {
        assertThrows(IllegalArgumentException.class, () -> new BackendPartitionExecutionPlan(
                "",
                LoweringFamily.DIRECT_KERNEL,
                1,
                List.of(1),
                List.of(),
                List.of(),
                List.of(),
                List.of(),
                PartitionCost.ofWork(1L),
                PartitionDecision.selected("DIRECT_KERNEL", "test"),
                EmptyPartitionPayload.INSTANCE
        ));
    }

    @Test
    void requiresUniqueOrderedAndBoundaryNodes() {
        assertThrows(IllegalArgumentException.class, () -> new BackendPartitionExecutionPlan(
                "partition",
                LoweringFamily.DIRECT_KERNEL,
                1,
                List.of(1, 1),
                List.of(),
                List.of(1),
                List.of(),
                List.of(),
                PartitionCost.ofWork(1L),
                PartitionDecision.selected("DIRECT_KERNEL", "test"),
                EmptyPartitionPayload.INSTANCE
        ));

        assertThrows(IllegalArgumentException.class, () -> new BackendPartitionExecutionPlan(
                "partition",
                LoweringFamily.DIRECT_KERNEL,
                1,
                List.of(1, 2),
                List.of(),
                List.of(2, 2),
                List.of(),
                List.of(),
                PartitionCost.ofWork(1L),
                PartitionDecision.selected("DIRECT_KERNEL", "test"),
                EmptyPartitionPayload.INSTANCE
        ));
    }

    @Test
    void requiresBoundaryOutputsToBelongToPartition() {
        assertThrows(IllegalArgumentException.class, () -> new BackendPartitionExecutionPlan(
                "partition",
                LoweringFamily.DIRECT_KERNEL,
                1,
                List.of(1, 2),
                List.of(),
                List.of(3),
                List.of(),
                List.of(),
                PartitionCost.ofWork(1L),
                PartitionDecision.selected("DIRECT_KERNEL", "test"),
                EmptyPartitionPayload.INSTANCE
        ));
    }

    @Test
    void requiresNodePlansAndExecutionGroupsToStayInsidePartition() {
        assertThrows(IllegalArgumentException.class, () -> new BackendPartitionExecutionPlan(
                "partition",
                LoweringFamily.DIRECT_KERNEL,
                1,
                List.of(1, 2),
                List.of(),
                List.of(2),
                List.of(nodePlan(3)),
                List.of(),
                PartitionCost.ofWork(1L),
                PartitionDecision.selected("DIRECT_KERNEL", "test"),
                EmptyPartitionPayload.INSTANCE
        ));

        assertThrows(IllegalArgumentException.class, () -> new BackendPartitionExecutionPlan(
                "partition",
                LoweringFamily.DIRECT_KERNEL,
                1,
                List.of(1, 2),
                List.of(),
                List.of(2),
                List.of(),
                List.of(new PartitionExecutionGroup(
                        "group",
                        List.of(1, 3),
                        PartitionExecutionKind.DIRECT_KERNEL,
                        "kernel",
                        List.of(),
                        List.of(2),
                        List.of(),
                        PartitionStorageContract.CPU_NATIVE,
                        "test"
                )),
                PartitionCost.ofWork(1L),
                PartitionDecision.selected("DIRECT_KERNEL", "test"),
                EmptyPartitionPayload.INSTANCE
        ));
    }

    private static PartitionNodePlan nodePlan(int nodeId) {
        return new PartitionNodePlan(
                nodeId,
                Operation.OpType.RELU,
                DataType.FLOAT32,
                PartitionRole.LOCAL_KERNEL,
                PartitionExecutionKind.DIRECT_KERNEL,
                "SEGMENT_SCALAR",
                PartitionStorageContract.CPU_NATIVE,
                List.of(),
                List.of(nodeId),
                PartitionLegalityStatus.SELECTED,
                "test"
        );
    }
}
