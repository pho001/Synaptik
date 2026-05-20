package graph;

import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.compile.GraphOptimizationConfig;
import config.compile.SemanticCanonicalizationConfig;
import config.optimizer.PiecewiseLoweringConfig;
import config.optimizer.RewriteConfig;
import config.runtime.RuntimeConfig;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class SemanticForwardCanonicalizationCompileTest {
    @Test
    void preAutogradSimplificationDoesNotLowerLinearButPreservesForwardAndBackward() {
        Tensor manualInput = new Tensor(new double[]{
                1.0, 2.0,
                3.0, 4.0
        }, new int[]{2, 2}, null, "manualInput", DataType.FLOAT64);
        Tensor manualWeight = new Tensor(new double[]{
                5.0, 6.0,
                7.0, 8.0
        }, new int[]{2, 2}, null, "manualWeight", DataType.FLOAT64);
        Tensor manualBias = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "manualBias", DataType.FLOAT64);
        manualInput.setRequiresGrad(true);
        manualWeight.setRequiresGrad(true);
        manualBias.setRequiresGrad(true);

        Tensor manual = manualInput.matmul(manualWeight).add(manualBias).sum();
        CompiledGraph compiled = CompiledGraph.compile(manual, arOnlyTrainingConfig());
        compiled.execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        Tensor directInput = new Tensor(new double[]{
                1.0, 2.0,
                3.0, 4.0
        }, new int[]{2, 2}, null, "directInput", DataType.FLOAT64);
        Tensor directWeight = new Tensor(new double[]{
                5.0, 6.0,
                7.0, 8.0
        }, new int[]{2, 2}, null, "directWeight", DataType.FLOAT64);
        Tensor directBias = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "directBias", DataType.FLOAT64);
        directInput.setRequiresGrad(true);
        directWeight.setRequiresGrad(true);
        directBias.setRequiresGrad(true);

        Tensor direct = directInput.linear(directWeight, directBias).sum();
        CompiledGraph.compile(direct, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(direct.toDoubleArrayCopy(), manual.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(directInput.getGradient().toDoubleArrayCopy(), manualInput.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(directWeight.getGradient().toDoubleArrayCopy(), manualWeight.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(directBias.getGradient().toDoubleArrayCopy(), manualBias.getGradient().toDoubleArrayCopy(), 1e-9);
        assertFalse(compiled.compiledNodes().stream()
                .map(graph.CompiledNode::operation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(opType -> opType == Operation.OpType.LINEAR));
    }

    @Test
    void preAutogradSimplificationDoesNotLowerCrossEntropyButPreservesForwardAndBackward() {
        Tensor manualLogits = new Tensor(new double[]{
                1.0, 2.0, 3.0,
                0.0, 0.0, 0.0
        }, new int[]{2, 3}, null, "manualLogits", DataType.FLOAT64);
        Tensor targetIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "targetIndices", DataType.INT32);
        manualLogits.setRequiresGrad(true);

        Tensor manual = manualLogits.logSoftmax(1).nllLossFromIndices(targetIndices, 1);
        CompiledGraph compiled = CompiledGraph.compile(manual, arOnlyTrainingConfig());
        compiled.execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        Tensor directLogits = new Tensor(new double[]{
                1.0, 2.0, 3.0,
                0.0, 0.0, 0.0
        }, new int[]{2, 3}, null, "directLogits", DataType.FLOAT64);
        directLogits.setRequiresGrad(true);
        Tensor direct = directLogits.crossEntropyLossFromIndices(targetIndices, 1);
        CompiledGraph.compile(direct, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(direct.toDoubleArrayCopy(), manual.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(directLogits.getGradient().toDoubleArrayCopy(), manualLogits.getGradient().toDoubleArrayCopy(), 1e-9);
        assertFalse(compiled.compiledNodes().stream()
                .map(graph.CompiledNode::operation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(opType -> opType == Operation.OpType.CROSS_ENTROPY_LOSS_INDICES));
    }

    @Test
    void reluLikeWhereCanonicalizationSkipsLargeZeroTensorScan() {
        Tensor input = new Tensor(new double[4097], new int[]{4097}, null, "x", DataType.FLOAT64);
        Tensor out = Tensor.where(
                input.greaterThan(Tensor.scalar(0.0, DataType.FLOAT64)),
                input,
                Tensor.zerosLike(input)
        );

        CompiledGraph compiled = CompiledGraph.compile(out, semanticReluOnlyConfig());

        assertFalse(containsOp(compiled, Operation.OpType.RELU));
        assertTrue(containsOp(compiled, Operation.OpType.WHERE));
    }

    private static CompileConfig arOnlyTrainingConfig() {
        return CompileConfig.training().withGraphOptimization(GraphOptimizationConfig.stages(true, false, false, false, false));
    }

    private static CompileConfig semanticReluOnlyConfig() {
        RewriteConfig rewrite = RewriteConfig.defaults()
                .withPiecewiseLowering(new PiecewiseLoweringConfig(false, true, false));
        return CompileConfig.inference()
                .withGraphOptimization(GraphOptimizationConfig.noGraphOptimization())
                .withSemanticCanonicalization(new SemanticCanonicalizationConfig(true, rewrite));
    }

    private static boolean containsOp(CompiledGraph compiledGraph, Operation.OpType opType) {
        return compiledGraph.compiledNodes().stream()
                .map(graph.CompiledNode::operation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(type -> type == opType);
    }
}
