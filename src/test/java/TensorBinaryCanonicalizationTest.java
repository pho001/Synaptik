import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TensorBinaryCanonicalizationTest {

    @Test
    void divCanonicalizesSpecialScalarCasesAtTensorSurface() {
        Tensor input = new Tensor(new double[]{2.0, 4.0}, new int[]{2}, null, "x", DataType.FLOAT64);

        Tensor divByOne = input.div(Tensor.scalar(1.0, DataType.FLOAT64));
        Tensor divByNegOne = input.div(Tensor.scalar(-1.0, DataType.FLOAT64));
        Tensor oneOverInput = Tensor.scalar(1.0, DataType.FLOAT64).div(input);
        Tensor divByTwo = input.div(Tensor.scalar(2.0, DataType.FLOAT64));

        assertSame(input, divByOne);

        assertNotNull(divByNegOne.getOperation());
        assertEquals(operations.Operation.OpType.NEG, divByNegOne.getOperation().opType());

        assertNotNull(oneOverInput.getOperation());
        assertEquals(operations.Operation.OpType.INV, oneOverInput.getOperation().opType());

        assertNotNull(divByTwo.getOperation());
        assertEquals(operations.Operation.OpType.MUL_SCALAR, divByTwo.getOperation().opType());
        assertTrue(divByTwo.getOperation() instanceof operations.mulScalar);
        assertEquals(0.5d, ((operations.mulScalar) divByTwo.getOperation()).getScalar(), 1e-12);
    }

    @Test
    void addSubMulCanonicalizeSpecialScalarCasesAtTensorSurface() {
        Tensor input = new Tensor(new double[]{2.0, 4.0}, new int[]{2}, null, "x", DataType.FLOAT64);

        Tensor addZero = input.add(Tensor.scalar(0.0, DataType.FLOAT64));
        Tensor zeroAdd = Tensor.scalar(0.0, DataType.FLOAT64).add(input);
        Tensor subZero = input.sub(Tensor.scalar(0.0, DataType.FLOAT64));
        Tensor zeroSub = Tensor.scalar(0.0, DataType.FLOAT64).sub(input);
        Tensor mulOne = input.mul(Tensor.scalar(1.0, DataType.FLOAT64));
        Tensor oneMul = Tensor.scalar(1.0, DataType.FLOAT64).mul(input);
        Tensor mulZero = input.mul(Tensor.scalar(0.0, DataType.FLOAT64));
        Tensor mulNegOne = input.mul(Tensor.scalar(-1.0, DataType.FLOAT64));

        assertSame(input, addZero);
        assertSame(input, zeroAdd);
        assertSame(input, subZero);
        assertSame(input, mulOne);
        assertSame(input, oneMul);

        assertNotNull(zeroSub.getOperation());
        assertEquals(operations.Operation.OpType.NEG, zeroSub.getOperation().opType());

        assertNull(mulZero.getOperation());
        assertArrayEquals(new double[]{0.0, 0.0}, mulZero.toDoubleArrayCopy(), 1e-12);

        assertNotNull(mulNegOne.getOperation());
        assertEquals(operations.Operation.OpType.NEG, mulNegOne.getOperation().opType());
    }
}
