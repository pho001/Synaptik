import runtime.contract.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorOps;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class UnfoldAxisExecutionTest {
    @Test
    void rankOneUnfoldMaterializesSlidingWindows() {
        Tensor input = new Tensor(new double[]{1, 2, 3, 4}, new int[]{4}, null, "input", DataType.FLOAT64);
        Tensor out = input.unfold(0, 2, 1);

        executeForward(out);

        assertArrayEquals(new int[]{3, 2}, out.getShape());
        assertArrayEquals(new double[]{1, 2, 2, 3, 3, 4}, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void rankTwoUnfoldSupportsNonUnitStep() {
        Tensor input = new Tensor(new double[]{
                1, 2, 3, 4, 5,
                6, 7, 8, 9, 10
        }, new int[]{2, 5}, null, "input", DataType.FLOAT64);
        Tensor out = TensorOps.unfold(input, 1, 2, 2);

        executeForward(out);

        assertArrayEquals(new int[]{2, 2, 2}, out.getShape());
        assertArrayEquals(new double[]{
                1, 2, 3, 4,
                6, 7, 8, 9
        }, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void rankNUnfoldAppendsWindowDimension() {
        double[] values = new double[30];
        for (int i = 0; i < values.length; i++) {
            values[i] = i + 1;
        }
        Tensor input = new Tensor(values, new int[]{2, 5, 3}, null, "input", DataType.FLOAT64);
        Tensor out = input.unfold(1, 3, 1);

        executeForward(out);

        assertArrayEquals(new int[]{2, 3, 3, 3}, out.getShape());
        assertArrayEquals(expectedRankN(values), out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void negativeAxisIsNormalized() {
        Tensor input = new Tensor(new double[]{
                1, 2, 3, 4,
                5, 6, 7, 8
        }, new int[]{2, 4}, null, "input", DataType.FLOAT64);
        Tensor out = input.unfold(-1, 2, 1);

        executeForward(out);

        assertArrayEquals(new int[]{2, 3, 2}, out.getShape());
        assertArrayEquals(new double[]{
                1, 2, 2, 3, 3, 4,
                5, 6, 6, 7, 7, 8
        }, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void invalidSizeAndStepAreRejected() {
        Tensor input = new Tensor(new double[]{1, 2, 3}, new int[]{3}, null, "input", DataType.FLOAT64);

        assertThrows(IllegalArgumentException.class, () -> input.unfold(0, 0, 1));
        assertThrows(IllegalArgumentException.class, () -> input.unfold(0, 2, 0));
        assertThrows(IllegalArgumentException.class, () -> input.unfold(0, 4, 1));
        assertThrows(IllegalArgumentException.class, () -> input.unfold(1, 2, 1));
    }

    @Test
    void unfoldPreservesBoolDType() {
        Tensor input = new Tensor(new byte[]{1, 0, 1, 0}, new int[]{4}, null, "input", DataType.BOOL);
        Tensor out = input.unfold(0, 2, 2);

        executeForward(out);

        assertEquals(DataType.BOOL, out.getDataType());
        assertArrayEquals(new int[]{2, 2}, out.getShape());
        assertArrayEquals(new byte[]{1, 0, 1, 0}, out.toBoolByteArrayCopy());
    }

    @Test
    void unfoldPreservesInt32DType() {
        Tensor input = new Tensor(new int[]{1, 2, 3, 4}, new int[]{4}, null, "input", DataType.INT32);
        Tensor out = input.unfold(0, 3, 1);

        executeForward(out);

        assertEquals(DataType.INT32, out.getDataType());
        assertArrayEquals(new int[]{2, 3}, out.getShape());
        assertArrayEquals(new int[]{1, 2, 3, 2, 3, 4}, out.toInt32ArrayCopy());
    }

    @Test
    void unfoldPreservesFloat64DType() {
        Tensor input = new Tensor(new double[]{1, 2, 3, 4}, new int[]{4}, null, "input", DataType.FLOAT64);
        Tensor out = input.unfold(0, 3, 1);

        executeForward(out);

        assertEquals(DataType.FLOAT64, out.getDataType());
        assertArrayEquals(new int[]{2, 3}, out.getShape());
        assertArrayEquals(new double[]{1, 2, 3, 2, 3, 4}, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void overlappingWindowsAccumulateGradient() {
        Tensor input = new Tensor(new double[]{1, 2, 3, 4}, new int[]{4}, null, "input", DataType.FLOAT64);
        input.setRequiresGrad(true);
        Tensor loss = input.unfold(0, 3, 1).sum();

        CompiledGraph.compile(loss, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.trainingDefaults())
                .execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{1, 2, 2, 1}, input.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    private static void executeForward(Tensor out) {
        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults())
                .execute(ExecutionMode.FORWARD);
    }

    private static double[] expectedRankN(double[] values) {
        double[] expected = new double[2 * 3 * 3 * 3];
        int p = 0;
        for (int b = 0; b < 2; b++) {
            for (int window = 0; window < 3; window++) {
                for (int c = 0; c < 3; c++) {
                    for (int offset = 0; offset < 3; offset++) {
                        expected[p++] = values[(b * 5 + window + offset) * 3 + c];
                    }
                }
            }
        }
        return expected;
    }
}
