import config.compile.CompileConfig;
import graph.CompiledGraph;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class CrossEntropyLossFromIndicesLoweringTest {

    @Test
    void defaultLoweringKeepsCanonicalLogSoftmaxAndIndexNllDecomposed() {
        Tensor logits = new Tensor(new double[]{
                1.0, 2.0, 3.0,
                0.0, 0.0, 0.0
        }, new int[]{2, 3}, null, "logits", DataType.FLOAT64);
        Tensor targetIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "targetIndices", DataType.INT32);

        Tensor canonical = logits.logSoftmax(1).nllLossFromIndices(targetIndices, 1);
        CompiledGraph compiled = CompiledGraph.compile(canonical, CompileConfig.inference());

        assertFalse(compiled.program().compiledNodes().stream()
                .map(graph.CompiledNode::operation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(opType -> opType == Operation.OpType.CROSS_ENTROPY_LOSS_INDICES));
        assertTrue(compiled.program().compiledNodes().stream()
                .map(graph.CompiledNode::operation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(opType -> opType == Operation.OpType.GATHER));
        assertTrue(compiled.program().compiledNodes().stream()
                .map(graph.CompiledNode::operation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(opType -> opType == Operation.OpType.MEAN));

        Tensor directLogits = new Tensor(new double[]{
                1.0, 2.0, 3.0,
                0.0, 0.0, 0.0
        }, new int[]{2, 3}, null, "directLogits", DataType.FLOAT64);
        Tensor direct = directLogits.crossEntropyLossFromIndices(targetIndices, 1);

        graph.CompiledGraph.compile(canonical, CompileConfig.inference())
                .prepare(config.runtime.RuntimeConfig.inferenceDefaults()).execute(backend.runtime.ExecutionMode.FORWARD);
        graph.CompiledGraph.compile(direct, CompileConfig.noGraphOptimizationBaseline())
                .prepare(config.runtime.RuntimeConfig.inferenceDefaults()).execute(backend.runtime.ExecutionMode.FORWARD);

        assertArrayEquals(direct.toDoubleArrayCopy(), canonical.toDoubleArrayCopy(), 1e-9);
    }
}
