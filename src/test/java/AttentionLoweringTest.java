import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.compile.GraphOptimizationConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.options.AttentionOptions;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

public class AttentionLoweringTest {
    @Test
    void simplificationDoesNotLowerManualAttentionPatternWithoutMask() {
        Tensor qManual = matrix3d("q_manual");
        Tensor kManual = matrix3d("k_manual");
        Tensor vManual = values3d("v_manual");
        Tensor manual = qManual.matmul(kManual.permute(0, 2, 1)).mul(0.5).softmax(2).matmul(vManual);

        CompiledGraph compiled = CompiledGraph.compile(manual, arOnlyConfig());
        compiled.prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        Tensor qDirect = matrix3d("q_direct");
        Tensor kDirect = matrix3d("k_direct");
        Tensor vDirect = values3d("v_direct");
        Tensor direct = qDirect.scaledDotProductAttention(kDirect, vDirect, AttentionOptions.defaults().withScale(0.5));
        CompiledGraph.compile(direct, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(direct.toDoubleArrayCopy(), manual.toDoubleArrayCopy(), 1e-9);
        assertFalse(containsOp(compiled, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION));
    }

    @Test
    void simplificationDoesNotLowerManualAttentionPatternWithMask() {
        Tensor qManual = matrix3d("q_manual");
        Tensor kManual = matrix3d("k_manual");
        Tensor vManual = values3d("v_manual");
        Tensor maskManual = mask3d("mask_manual");
        Tensor scores = qManual.matmul(kManual.permute(0, 2, 1)).mul(0.5);
        Tensor manual = Tensor.where(maskManual, scores, Tensor.scalar(-1.0e30, DataType.FLOAT64)).softmax(2).matmul(vManual);

        CompiledGraph compiled = CompiledGraph.compile(manual, arOnlyConfig());
        compiled.prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        Tensor qDirect = matrix3d("q_direct");
        Tensor kDirect = matrix3d("k_direct");
        Tensor vDirect = values3d("v_direct");
        Tensor maskDirect = mask3d("mask_direct");
        Tensor direct = qDirect.scaledDotProductAttention(kDirect, vDirect, maskDirect, AttentionOptions.defaults().withScale(0.5));
        CompiledGraph.compile(direct, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(direct.toDoubleArrayCopy(), manual.toDoubleArrayCopy(), 1e-9);
        assertFalse(containsOp(compiled, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION));
    }

    @Test
    void simplificationKeepsManualAttentionBackwardGraphExplicit() {
        Tensor qManual = matrix3d("q_manual");
        Tensor kManual = matrix3d("k_manual");
        Tensor vManual = values3d("v_manual");
        qManual.setRequiresGrad(true);
        kManual.setRequiresGrad(true);
        vManual.setRequiresGrad(true);
        Tensor manual = qManual.matmul(kManual.permute(0, 2, 1)).mul(0.5).softmax(2).matmul(vManual).sum();

        CompiledGraph compiled = CompiledGraph.compile(manual, trainingArOnlyConfig());
        compiled.prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        Tensor qDirect = matrix3d("q_direct");
        Tensor kDirect = matrix3d("k_direct");
        Tensor vDirect = values3d("v_direct");
        qDirect.setRequiresGrad(true);
        kDirect.setRequiresGrad(true);
        vDirect.setRequiresGrad(true);
        Tensor direct = qDirect.scaledDotProductAttention(kDirect, vDirect, AttentionOptions.defaults().withScale(0.5)).sum();

        CompiledGraph.compile(direct, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(direct.toDoubleArrayCopy(), manual.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(qDirect.getGradient().toDoubleArrayCopy(), qManual.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(kDirect.getGradient().toDoubleArrayCopy(), kManual.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(vDirect.getGradient().toDoubleArrayCopy(), vManual.getGradient().toDoubleArrayCopy(), 1e-9);
        assertFalse(containsOp(compiled, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_BACKWARD));
    }

    @Test
    void directMaskedAttentionPreservesBackwardGradientsAgainstManualPattern() {
        Tensor qManual = matrix3d("q_manual");
        Tensor kManual = matrix3d("k_manual");
        Tensor vManual = values3d("v_manual");
        Tensor maskManual = mask3d("mask_manual");
        qManual.setRequiresGrad(true);
        kManual.setRequiresGrad(true);
        vManual.setRequiresGrad(true);
        Tensor manual = Tensor.where(
                maskManual,
                qManual.matmul(kManual.permute(0, 2, 1)).mul(0.5),
                Tensor.scalar(-1.0e30, DataType.FLOAT64)
        ).softmax(2).matmul(vManual).sum();

        CompiledGraph.compile(manual, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        Tensor qDirect = matrix3d("q_direct");
        Tensor kDirect = matrix3d("k_direct");
        Tensor vDirect = values3d("v_direct");
        Tensor maskDirect = mask3d("mask_direct");
        qDirect.setRequiresGrad(true);
        kDirect.setRequiresGrad(true);
        vDirect.setRequiresGrad(true);
        Tensor direct = qDirect.scaledDotProductAttention(kDirect, vDirect, maskDirect, AttentionOptions.defaults().withScale(0.5)).sum();

        CompiledGraph.compile(direct, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(direct.toDoubleArrayCopy(), manual.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(qDirect.getGradient().toDoubleArrayCopy(), qManual.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(kDirect.getGradient().toDoubleArrayCopy(), kManual.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(vDirect.getGradient().toDoubleArrayCopy(), vManual.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void directMaskedAttentionPreservesBackwardGradientsWhenWholeRowIsMasked() {
        Tensor qManual = matrix3d("q_manual");
        Tensor kManual = matrix3d("k_manual");
        Tensor vManual = values3d("v_manual");
        Tensor maskManual = fullyMaskedFirstRow("mask_manual");
        qManual.setRequiresGrad(true);
        kManual.setRequiresGrad(true);
        vManual.setRequiresGrad(true);
        Tensor manual = Tensor.where(
                maskManual,
                qManual.matmul(kManual.permute(0, 2, 1)).mul(0.5),
                Tensor.scalar(-1.0e30, DataType.FLOAT64)
        ).softmax(2).matmul(vManual).sum();

        CompiledGraph.compile(manual, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        Tensor qDirect = matrix3d("q_direct");
        Tensor kDirect = matrix3d("k_direct");
        Tensor vDirect = values3d("v_direct");
        Tensor maskDirect = fullyMaskedFirstRow("mask_direct");
        qDirect.setRequiresGrad(true);
        kDirect.setRequiresGrad(true);
        vDirect.setRequiresGrad(true);
        Tensor direct = qDirect.scaledDotProductAttention(kDirect, vDirect, maskDirect, AttentionOptions.defaults().withScale(0.5)).sum();

        CompiledGraph.compile(direct, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(direct.toDoubleArrayCopy(), manual.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(qDirect.getGradient().toDoubleArrayCopy(), qManual.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(kDirect.getGradient().toDoubleArrayCopy(), kManual.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(vDirect.getGradient().toDoubleArrayCopy(), vManual.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    private static Tensor matrix3d(String label) {
        return new Tensor(new double[]{
                1.0, 0.0,
                0.5, 1.0,
                0.75, 0.25,
                0.0, 1.0
        }, new int[]{2, 2, 2}, null, label, DataType.FLOAT64);
    }

    private static Tensor values3d(String label) {
        return new Tensor(new double[]{
                10.0, 1.0,
                1.0, 10.0,
                4.0, 2.0,
                2.0, 4.0
        }, new int[]{2, 2, 2}, null, label, DataType.FLOAT64);
    }

    private static Tensor mask3d(String label) {
        return new Tensor(new byte[]{
                1, 0,
                1, 1
        }, new int[]{1, 2, 2}, null, label, DataType.BOOL);
    }

    private static Tensor fullyMaskedFirstRow(String label) {
        return new Tensor(new byte[]{
                0, 0,
                1, 1
        }, new int[]{1, 2, 2}, null, label, DataType.BOOL);
    }

    private static boolean containsOp(CompiledGraph compiledGraph, Operation.OpType opType) {
        return compiledGraph.program().compiledNodes().stream()
                .map(graph.model.CompiledNode::operation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(type -> type == opType);
    }

    private static CompileConfig arOnlyConfig() {
        return CompileConfig.inference().withGraphOptimization(GraphOptimizationConfig.stages(true, false, false, false, false));
    }

    private static CompileConfig trainingArOnlyConfig() {
        return CompileConfig.training().withGraphOptimization(GraphOptimizationConfig.stages(true, false, false, false, false));
    }
}
