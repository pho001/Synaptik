import tensor.layout.BroadcastPlan;
import tensor.layout.BroadcastPlanner;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class BroadcastPlannerTest {
    @Test
    public void testPlanAlignsRanksAndBroadcastsTrailingDims() {
        int[] aShape = new int[]{2, 1, 3};
        int[] aStrides = new int[]{3, 3, 1};
        int[] bShape = new int[]{1, 4, 3};
        int[] bStrides = new int[]{12, 3, 1};

        BroadcastPlan plan = BroadcastPlanner.plan(aShape, aStrides, bShape, bStrides);

        assertArrayEquals(new int[]{2, 4, 3}, plan.outShape());
        assertArrayEquals(new int[]{12, 3, 1}, plan.outStrides());
        assertArrayEquals(new int[]{3, 0, 1}, plan.aEffStrides());
        assertArrayEquals(new int[]{0, 3, 1}, plan.bEffStrides());
        assertArrayEquals(new int[]{1}, plan.reduceAxesForA());
        assertArrayEquals(new int[]{0}, plan.reduceAxesForB());
        assertFalse(plan.isNoBroadcast());
    }

    @Test
    public void testPlanRejectsIncompatibleShapes() {
        IllegalArgumentException ex = assertThrows(
                IllegalArgumentException.class,
                () -> BroadcastPlanner.plan(
                        new int[]{2, 3},
                        new int[]{3, 1},
                        new int[]{2, 4},
                        new int[]{4, 1}
                )
        );
        assertTrue(ex.getMessage().contains("Broadcast mismatch"));
    }

    @Test
    public void sameShapeZeroStrideViewIsStillBroadcast() {
        BroadcastPlan plan = BroadcastPlanner.plan(
                new int[]{2, 4},
                new int[]{4, 1},
                new int[]{2, 4},
                new int[]{0, 0}
        );

        assertArrayEquals(new int[]{2, 4}, plan.outShape());
        assertArrayEquals(new int[]{4, 1}, plan.aEffStrides());
        assertArrayEquals(new int[]{0, 0}, plan.bEffStrides());
        assertFalse(plan.isNoBroadcast());
    }
}
