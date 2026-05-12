import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class TensorShapeIndexOpsExecutionTest {
    @Test
    public void sliceWithStepsConcatAndCastExecute() {
        Tensor x = new Tensor(new float[]{
                1f, 2f, 3f, 4f,
                5f, 6f, 7f, 8f
        }, new int[]{2, 4}, null, "x", DataType.FLOAT32);

        Tensor sliced = x.slice(
                new int[]{0, 1},
                new int[]{2, 4},
                new int[]{0, 1},
                new int[]{1, 2}
        );
        Tensor out = Tensor.concat(0, sliced.cast(DataType.FLOAT64), sliced.cast(DataType.FLOAT64));

        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertEquals(DataType.FLOAT64, out.getDataType());
        assertArrayEquals(new int[]{4, 2}, out.getShape());
        assertArrayEquals(new double[]{2.0, 4.0, 6.0, 8.0, 2.0, 4.0, 6.0, 8.0},
                out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    public void castToSameDtypeReturnsInput() {
        Tensor x = new Tensor(new float[]{1f, 2f}, new int[]{2}, null, "x", DataType.FLOAT32);

        assertSame(x, x.cast(DataType.FLOAT32));
    }

    @Test
    public void concatRejectsMismatchedDtypesAndRanks() {
        Tensor f32 = new Tensor(new float[]{1f, 2f}, new int[]{2}, null, "f32", DataType.FLOAT32);
        Tensor f64 = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "f64", DataType.FLOAT64);
        Tensor rank2 = new Tensor(new float[]{1f, 2f}, new int[]{1, 2}, null, "rank2", DataType.FLOAT32);

        assertThrows(IllegalArgumentException.class, () -> Tensor.concat(0, f32, f64));
        assertThrows(IllegalArgumentException.class, () -> Tensor.concat(0, f32, rank2));
    }

    @Test
    public void sliceRejectsUnsupportedNegativeStepsAndDuplicateAxes() {
        Tensor x = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "x", DataType.FLOAT32);

        assertThrows(IllegalArgumentException.class,
                () -> x.slice(new int[]{0}, new int[]{2}, new int[]{0}, new int[]{-1}));
        assertThrows(IllegalArgumentException.class,
                () -> x.slice(new int[]{0, 0}, new int[]{1, 1}, new int[]{1, 1}, new int[]{1, 1}));
    }
}
