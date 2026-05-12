import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Int32Storage;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class Int32IndexDtypeTest {

    @Test
    void int32TensorUsesDedicatedStorage() {
        Tensor indices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "indices", DataType.INT32);
        assertSame(DataType.INT32, indices.getDataType());
        assertTrue(indices.getStorage() instanceof Int32Storage);
        assertArrayEquals(new int[]{2, 0}, indices.getInt32Data());
        assertArrayEquals(new double[]{2.0, 0.0}, indices.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void int32DisallowsImplicitConversionToFloatingDtypes() {
        Tensor indices = new Tensor(new int[]{1, 2}, new int[]{2}, null, "indices", DataType.INT32);
        assertThrows(UnsupportedOperationException.class, () -> indices.setDataType(DataType.FLOAT32));

        Tensor values = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "values", DataType.FLOAT64);
        assertThrows(UnsupportedOperationException.class, () -> values.setDataType(DataType.INT32));
    }

    @Test
    void gatherSupportsInt32Indices() {
        Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "indices", DataType.INT32);
        Tensor y = x.gather(indices, 1);

        CompiledGraph.compile(y, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{3.0, 4.0}, y.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void indexTargetLossFamilySupportsInt32Indices() {
        Tensor logits = new Tensor(new double[]{
                1.0, 2.0, 3.0,
                0.0, 0.0, 0.0
        }, new int[]{2, 3}, null, "logits", DataType.FLOAT64);
        Tensor targetIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "targetIndices", DataType.INT32);

        Tensor nll = logits.logSoftmax(1).nllLossFromIndices(targetIndices, 1);
        Tensor ce = logits.crossEntropyLossFromIndices(targetIndices, 1);

        CompiledGraph.compile(nll, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        CompiledGraph.compile(ce, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(nll.toDoubleArrayCopy(), ce.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void ignoreIndexSupportsInt32Indices() {
        Tensor logits = new Tensor(new double[]{
                1.0, 2.0, 3.0,
                0.0, 0.0, 0.0
        }, new int[]{2, 3}, null, "logits", DataType.FLOAT64);
        Tensor targetIndices = new Tensor(new int[]{2, -1}, new int[]{2}, null, "targetIndices", DataType.INT32);

        Tensor loss = logits.crossEntropyLossFromIndices(targetIndices, 1, -1);
        CompiledGraph.compile(loss, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{0.4076059644443804}, loss.toDoubleArrayCopy(), 1e-9);
    }
}
