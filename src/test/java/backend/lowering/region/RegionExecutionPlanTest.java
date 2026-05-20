package backend.lowering.region;

import backend.lowering.LoweringFamily;
import graph.compile.planning.partition.PartitionTarget;
import org.junit.jupiter.api.Test;
import operations.Operation;
import tensor.DataType;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RegionExecutionPlanTest {
    @Test
    void requiresAnchorToBePartOfOrderedNodes() {
        assertThrows(IllegalArgumentException.class, () -> new RegionExecutionPlan(
                "region",
                PartitionTarget.GPU_METAL,
                LoweringFamily.METAL_GRAPH_REGION,
                9,
                List.of(1, 2, 3),
                List.of(0),
                List.of(3),
                List.of(),
                List.of(),
                RegionCost.ofWork(10L),
                RegionDecision.selected("METAL_GRAPH_REGION", "test"),
                EmptyRegionPayload.INSTANCE
        ));
    }

    @Test
    void normalizesNullCollectionsAndPayload() {
        RegionExecutionPlan plan = new RegionExecutionPlan(
                "region",
                PartitionTarget.GPU_CUDA,
                LoweringFamily.CUDA_GRAPH_REGION,
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
        assertTrue(plan.backendPayload() instanceof EmptyRegionPayload);
        assertTrue(plan.decision().selected());
    }

    @Test
    void requiresRegionId() {
        assertThrows(IllegalArgumentException.class, () -> new RegionExecutionPlan(
                "",
                PartitionTarget.CPU,
                LoweringFamily.CPU_NATIVE_REGION,
                1,
                List.of(1),
                List.of(),
                List.of(),
                List.of(),
                List.of(),
                RegionCost.ofWork(1L),
                RegionDecision.selected("CPU_NATIVE_REGION", "test"),
                EmptyRegionPayload.INSTANCE
        ));
    }

    @Test
    void requiresUniqueOrderedAndBoundaryNodes() {
        assertThrows(IllegalArgumentException.class, () -> new RegionExecutionPlan(
                "region",
                PartitionTarget.CPU,
                LoweringFamily.CPU_NATIVE_REGION,
                1,
                List.of(1, 1),
                List.of(),
                List.of(1),
                List.of(),
                List.of(),
                RegionCost.ofWork(1L),
                RegionDecision.selected("CPU_NATIVE_REGION", "test"),
                EmptyRegionPayload.INSTANCE
        ));

        assertThrows(IllegalArgumentException.class, () -> new RegionExecutionPlan(
                "region",
                PartitionTarget.CPU,
                LoweringFamily.CPU_NATIVE_REGION,
                1,
                List.of(1, 2),
                List.of(),
                List.of(2, 2),
                List.of(),
                List.of(),
                RegionCost.ofWork(1L),
                RegionDecision.selected("CPU_NATIVE_REGION", "test"),
                EmptyRegionPayload.INSTANCE
        ));
    }

    @Test
    void requiresBoundaryOutputsToBelongToRegion() {
        assertThrows(IllegalArgumentException.class, () -> new RegionExecutionPlan(
                "region",
                PartitionTarget.CPU,
                LoweringFamily.CPU_NATIVE_REGION,
                1,
                List.of(1, 2),
                List.of(),
                List.of(3),
                List.of(),
                List.of(),
                RegionCost.ofWork(1L),
                RegionDecision.selected("CPU_NATIVE_REGION", "test"),
                EmptyRegionPayload.INSTANCE
        ));
    }

    @Test
    void requiresNodePlansAndExecutionGroupsToStayInsideRegion() {
        assertThrows(IllegalArgumentException.class, () -> new RegionExecutionPlan(
                "region",
                PartitionTarget.CPU,
                LoweringFamily.CPU_NATIVE_REGION,
                1,
                List.of(1, 2),
                List.of(),
                List.of(2),
                List.of(nodePlan(3)),
                List.of(),
                RegionCost.ofWork(1L),
                RegionDecision.selected("CPU_NATIVE_REGION", "test"),
                EmptyRegionPayload.INSTANCE
        ));

        assertThrows(IllegalArgumentException.class, () -> new RegionExecutionPlan(
                "region",
                PartitionTarget.CPU,
                LoweringFamily.CPU_NATIVE_REGION,
                1,
                List.of(1, 2),
                List.of(),
                List.of(2),
                List.of(),
                List.of(new RegionExecutionGroup(
                        "group",
                        List.of(1, 3),
                        RegionExecutionKind.DIRECT_KERNEL,
                        "kernel",
                        List.of(),
                        List.of(2),
                        List.of(),
                        RegionStorageContract.CPU_NATIVE,
                        "test"
                )),
                RegionCost.ofWork(1L),
                RegionDecision.selected("CPU_NATIVE_REGION", "test"),
                EmptyRegionPayload.INSTANCE
        ));
    }

    private static RegionNodePlan nodePlan(int nodeId) {
        return new RegionNodePlan(
                nodeId,
                Operation.OpType.RELU,
                DataType.FLOAT32,
                RegionRole.LOCAL_KERNEL,
                RegionExecutionKind.DIRECT_KERNEL,
                "SEGMENT_SCALAR",
                RegionStorageContract.CPU_NATIVE,
                List.of(),
                List.of(nodeId),
                RegionLegalityStatus.SELECTED,
                "test"
        );
    }
}
