package backend.cpu1;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.offset.Cpu1GenericOffsetPlan;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class Cpu1GenericOffsetPlanTest {
    @Test
    void genericPlanMapsRowMajorLogicalIndexToPhysicalStorageOffset() {
        Tensor base = new Tensor(
                new double[]{0.0, 1.0, 2.0, 3.0, 4.0, 5.0},
                new int[]{1, 1, 2, 1, 3},
                null,
                "base",
                DataType.FLOAT64
        );
        Tensor transposed = base.permute(4, 2, 0, 1, 3);
        Cpu1GenericOffsetPlan plan = Cpu1GenericOffsetPlan.forView(Cpu1TensorView.fromTensor(transposed));

        assertEquals(0, plan.offset(0));
        assertEquals(3, plan.offset(1));
        assertEquals(4, plan.offset(3));
        assertEquals(5, plan.offset(5));
    }

    @Test
    void genericPlanRejectsOutOfRangeLinearIndexWhenChecked() {
        Tensor tensor = new Tensor(new double[]{1.0, 2.0}, new int[]{1, 1, 1, 1, 2}, null, "values", DataType.FLOAT64);
        Cpu1GenericOffsetPlan plan = Cpu1GenericOffsetPlan.forView(Cpu1TensorView.fromTensor(tensor));

        assertThrows(IndexOutOfBoundsException.class, () -> plan.checkedOffset(-1));
        assertThrows(IndexOutOfBoundsException.class, () -> plan.checkedOffset(2));
    }
}
