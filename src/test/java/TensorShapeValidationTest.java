import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class TensorShapeValidationTest {
    @Test
    void rejectsZeroAndNegativeDimensions() {
        assertThrows(IllegalArgumentException.class,
                () -> new Tensor(new int[]{0, 2}, null, "zero", DataType.FLOAT32));
        assertThrows(IllegalArgumentException.class,
                () -> new Tensor(new int[]{2, -1}, null, "negative", DataType.FLOAT32));
    }

    @Test
    void rejectsFlatSizeOverflow() {
        assertThrows(IllegalArgumentException.class,
                () -> new Tensor(new int[]{Integer.MAX_VALUE, 2}, null, "overflow", DataType.FLOAT32));
    }

    @Test
    void scalarShapeNormalizesToSingleElementShape() {
        Tensor scalar = new Tensor(new double[]{7.0}, new int[]{}, null, "scalar", DataType.FLOAT64);

        assertArrayEquals(new int[]{1}, scalar.getShape());
    }
}
