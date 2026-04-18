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
}
