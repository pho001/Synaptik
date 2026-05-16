package backend.lowering.region;

import backend.lowering.LoweringFamily;
import graph.optimizer.partition.PartitionTarget;
import org.junit.jupiter.api.Test;

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
}
