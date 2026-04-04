import tensor.DataType;
import tensor.TensorDataFactory;
import tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class TensorDataFactoryTest {

    @Test
    void strictPrefixCopiesRequestedPrefix() {
        Tensor t = TensorDataFactory.prefixTensorStrict(
                "strict",
                new double[]{1, 2, 3, 4, 5, 6},
                true,
                DataType.FLOAT64,
                2,
                2
        );

        assertArrayEquals(new int[]{2, 2}, t.getShape());
        assertArrayEquals(new double[]{1, 2, 3, 4}, t.toDoubleArrayCopy(), 1e-12);
    }

    @Test
    void strictPrefixRejectsTooShortInput() {
        assertThrows(IllegalArgumentException.class, () ->
                TensorDataFactory.prefixTensorStrict(
                        "strictFail",
                        new double[]{1, 2, 3},
                        false,
                        DataType.FLOAT64,
                        2,
                        2
                ));
    }

    @Test
    void wrapPrefixRepeatsWhenInputIsTooShort() {
        Tensor t = TensorDataFactory.prefixTensorWrap(
                "wrap",
                new double[]{1, 2, 3},
                false,
                DataType.FLOAT64,
                2,
                2
        );

        assertArrayEquals(new double[]{1, 2, 3, 1}, t.toDoubleArrayCopy(), 1e-12);
    }
}
