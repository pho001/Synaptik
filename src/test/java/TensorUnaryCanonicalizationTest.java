import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;

public class TensorUnaryCanonicalizationTest {

    @Test
    void powCanonicalizesSpecialExponentsAtTensorSurface() {
        Tensor input = new Tensor(new double[]{2.0, 4.0}, new int[]{2}, null, "x", DataType.FLOAT64);

        Tensor pow0 = input.pow(0.0);
        Tensor pow1 = input.pow(1.0);
        Tensor powNeg1 = input.pow(-1.0);
        Tensor pow2 = input.pow(2.0);

        assertNull(pow0.getOperation());
        assertArrayEquals(new double[]{1.0, 1.0}, pow0.toDoubleArrayCopy(), 1e-9);

        assertSame(input, pow1);

        assertNotNull(powNeg1.getOperation());
        assertEquals(operations.Operation.OpType.INV, powNeg1.getOperation().opType());

        assertNotNull(pow2.getOperation());
        assertEquals(operations.Operation.OpType.MUL, pow2.getOperation().opType());
    }

    @Test
    void mulScalarCanonicalizesSpecialScalarsAtTensorSurface() {
        Tensor input = new Tensor(new double[]{2.0, 4.0}, new int[]{2}, null, "x", DataType.FLOAT64);

        Tensor mul0 = input.mul(0.0);
        Tensor mul1 = input.mul(1.0);
        Tensor mulNeg1 = input.mul(-1.0);

        assertNull(mul0.getOperation());
        assertArrayEquals(new double[]{0.0, 0.0}, mul0.toDoubleArrayCopy(), 1e-9);

        assertSame(input, mul1);

        assertNotNull(mulNeg1.getOperation());
        assertEquals(operations.Operation.OpType.NEG, mulNeg1.getOperation().opType());
    }

    @Test
    void invCanonicalizesDoubleInverseAtTensorSurface() {
        Tensor input = new Tensor(new double[]{2.0, 4.0}, new int[]{2}, null, "x", DataType.FLOAT64);

        Tensor inv = input.inv();
        Tensor invInv = inv.inv();

        assertSame(input, invInv);
    }

    @Test
    void clampCanonicalizesIdentityAndNestedThresholdsAtTensorSurface() {
        Tensor input = new Tensor(new double[]{2.0, 4.0}, new int[]{2}, null, "x", DataType.FLOAT64);

        Tensor clampMinIdentity = input.clampMin(Double.NEGATIVE_INFINITY);
        Tensor clampMaxIdentity = input.clampMax(Double.POSITIVE_INFINITY);
        Tensor nestedClampMin = input.clampMin(1.0).clampMin(3.0);
        Tensor nestedClampMax = input.clampMax(6.0).clampMax(4.0);

        assertSame(input, clampMinIdentity);
        assertSame(input, clampMaxIdentity);

        assertNotNull(nestedClampMin.getOperation());
        assertEquals(operations.Operation.OpType.CLAMP_MIN, nestedClampMin.getOperation().opType());
        assertEquals(3.0d, ((operations.elementwise.unary.clampMin) nestedClampMin.getOperation()).getMinValue(), 1e-12);

        assertNotNull(nestedClampMax.getOperation());
        assertEquals(operations.Operation.OpType.CLAMP_MAX, nestedClampMax.getOperation().opType());
        assertEquals(4.0d, ((operations.elementwise.unary.clampMax) nestedClampMax.getOperation()).getMaxValue(), 1e-12);
    }
}
