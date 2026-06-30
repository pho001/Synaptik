import runtime.contract.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.options.AttentionOptions;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class AttentionExecutionTest {

    @Test
    void scaledDotProductAttentionWithoutMaskMatchesExpectedValues() {
        Tensor q = new Tensor(new double[]{
                1, 0,
                0, 1
        }, new int[]{1, 2, 2}, null, "q", DataType.FLOAT64);
        Tensor k = new Tensor(new double[]{
                1, 0,
                0, 1
        }, new int[]{1, 2, 2}, null, "k", DataType.FLOAT64);
        Tensor v = new Tensor(new double[]{
                10, 1,
                1, 10
        }, new int[]{1, 2, 2}, null, "v", DataType.FLOAT64);

        Tensor out = q.scaledDotProductAttention(k, v, AttentionOptions.defaults().withScale(1.0));
        CompiledGraph compiledGraph = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline());
        compiledGraph.prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        double e = Math.exp(1.0);
        double denom = e + 1.0;
        assertArrayEquals(new int[]{1, 2, 2}, out.getShape());
        assertArrayEquals(new double[]{
                (10.0 * e + 1.0) / denom,
                (1.0 * e + 10.0) / denom,
                (10.0 + e) / denom,
                (1.0 + 10.0 * e) / denom
        }, out.toDoubleArrayCopy(), 1e-6);
        assertCanonicalAttentionGraph(compiledGraph);
    }

    @Test
    void scaledDotProductAttentionWithCausalMaskBlocksFutureTokens() {
        Tensor q = new Tensor(new double[]{
                1, 0,
                0, 1
        }, new int[]{1, 2, 2}, null, "q", DataType.FLOAT64);
        Tensor k = new Tensor(new double[]{
                1, 0,
                0, 1
        }, new int[]{1, 2, 2}, null, "k", DataType.FLOAT64);
        Tensor v = new Tensor(new double[]{
                10, 1,
                1, 10
        }, new int[]{1, 2, 2}, null, "v", DataType.FLOAT64);

        Tensor out = q.scaledDotProductAttention(k, v, AttentionOptions.causalDefaults().withScale(1.0));
        CompiledGraph compiledGraph = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline());
        compiledGraph.prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        double e = Math.exp(1.0);
        double denom = e + 1.0;
        assertArrayEquals(new double[]{
                10.0, 1.0,
                (10.0 + e) / denom,
                (1.0 + 10.0 * e) / denom
        }, out.toDoubleArrayCopy(), 1e-6);
        assertCanonicalAttentionGraph(compiledGraph);
    }

    @Test
    void scaledDotProductAttentionWithExternalMaskUsesBoolMask() {
        Tensor q = new Tensor(new double[]{1, 0}, new int[]{1, 1, 2}, null, "q", DataType.FLOAT64);
        Tensor k = new Tensor(new double[]{
                1, 0,
                0, 1
        }, new int[]{1, 2, 2}, null, "k", DataType.FLOAT64);
        Tensor v = new Tensor(new double[]{
                10, 1,
                1, 10
        }, new int[]{1, 2, 2}, null, "v", DataType.FLOAT64);
        Tensor mask = new Tensor(new byte[]{0, 1}, new int[]{1, 1, 2}, null, "mask", DataType.BOOL);

        Tensor out = q.scaledDotProductAttention(k, v, mask, AttentionOptions.defaults().withScale(1.0));
        CompiledGraph compiledGraph = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline());
        compiledGraph.prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{1.0, 10.0}, out.toDoubleArrayCopy(), 1e-6);
        assertCanonicalAttentionGraph(compiledGraph);
    }

    @Test
    void rankFourDefaultScaleMatchesExpectedValues() {
        Tensor q = new Tensor(new float[]{
                1f, 0f,
                0f, 1f
        }, new int[]{1, 1, 2, 2}, null, "q4", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{
                1f, 0f,
                0f, 1f
        }, new int[]{1, 1, 2, 2}, null, "k4", DataType.FLOAT32);
        Tensor v = new Tensor(new float[]{
                10f, 1f,
                1f, 10f
        }, new int[]{1, 1, 2, 2}, null, "v4", DataType.FLOAT32);

        Tensor out = q.scaledDotProductAttention(k, v, AttentionOptions.defaults());
        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        double scale = 1.0d / Math.sqrt(2.0d);
        double e = Math.exp(scale);
        double denom = e + 1.0d;
        assertArrayEquals(new int[]{1, 1, 2, 2}, out.getShape());
        assertArrayEquals(new double[]{
                (10.0 * e + 1.0) / denom,
                (1.0 * e + 10.0) / denom,
                (10.0 + e) / denom,
                (1.0 + 10.0 * e) / denom
        }, out.toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void attentionRejectsInvalidShapeDtypeAndMaskContracts() {
        Tensor q = new Tensor(new float[]{1f, 0f}, new int[]{1, 2}, null, "q", DataType.FLOAT32);
        Tensor kHeadMismatch = new Tensor(new float[]{1f, 0f, 0f}, new int[]{1, 3}, null, "kHeadMismatch", DataType.FLOAT32);
        Tensor v = new Tensor(new float[]{1f, 2f}, new int[]{1, 2}, null, "v", DataType.FLOAT32);
        assertThrows(IllegalArgumentException.class,
                () -> q.scaledDotProductAttention(kHeadMismatch, v, AttentionOptions.defaults()));

        Tensor qInt = new Tensor(new int[]{1, 2}, new int[]{1, 2}, null, "qInt", DataType.INT32);
        Tensor k = new Tensor(new float[]{1f, 0f}, new int[]{1, 2}, null, "k", DataType.FLOAT32);
        assertThrows(IllegalArgumentException.class,
                () -> qInt.scaledDotProductAttention(k, v, AttentionOptions.defaults()));

        Tensor maskNotBool = new Tensor(new float[]{1f}, new int[]{1, 1}, null, "maskNotBool", DataType.FLOAT32);
        assertThrows(IllegalArgumentException.class,
                () -> q.scaledDotProductAttention(k, v, maskNotBool, AttentionOptions.defaults()));
    }

    @Test
    void scaledDotProductAttentionBuildsCanonicalWeightsDag() {
        Tensor q = new Tensor(new double[]{
                1, 0,
                0, 1
        }, new int[]{1, 2, 2}, null, "q", DataType.FLOAT64);
        Tensor k = new Tensor(new double[]{
                1, 0,
                0, 1
        }, new int[]{1, 2, 2}, null, "k", DataType.FLOAT64);
        Tensor v = new Tensor(new double[]{
                10, 1,
                1, 10
        }, new int[]{1, 2, 2}, null, "v", DataType.FLOAT64);
        q.setRequiresGrad(true);
        k.setRequiresGrad(true);
        v.setRequiresGrad(true);

        Tensor out = q.scaledDotProductAttention(k, v, AttentionOptions.defaults().withScale(1.0));
        CompiledGraph compiledGraph = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline());
        compiledGraph
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD);

        double e = Math.exp(1.0);
        double denom = e + 1.0;
        assertArrayEquals(new int[]{1, 2, 2}, out.getShape());
        assertArrayEquals(new double[]{
                (10.0 * e + 1.0) / denom,
                (1.0 * e + 10.0) / denom,
                (10.0 + e) / denom,
                (1.0 + 10.0 * e) / denom
        }, out.toDoubleArrayCopy(), 1e-6);
        assertCanonicalAttentionGraph(compiledGraph);
        assertFalse(containsOp(compiledGraph, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS));
    }

    @Test
    void scaledDotProductAttentionBackwardUsesCanonicalPrimitiveDag() {
        Tensor q = new Tensor(new double[]{
                1, 0,
                0, 1
        }, new int[]{1, 2, 2}, null, "q", DataType.FLOAT64);
        Tensor k = new Tensor(new double[]{
                1, 0,
                0, 1
        }, new int[]{1, 2, 2}, null, "k", DataType.FLOAT64);
        Tensor v = new Tensor(new double[]{
                10, 1,
                1, 10
        }, new int[]{1, 2, 2}, null, "v", DataType.FLOAT64);
        q.setRequiresGrad(true);
        k.setRequiresGrad(true);
        v.setRequiresGrad(true);

        Tensor loss = q.scaledDotProductAttention(k, v, AttentionOptions.defaults().withScale(1.0)).sum();
        CompiledGraph compiledGraph = CompiledGraph.compile(loss, CompileConfig.noGraphOptimizationBaseline());
        compiledGraph.prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        assertCanonicalAttentionGraph(compiledGraph);
        assertFalse(containsOp(compiledGraph, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_BACKWARD));
    }

    @Test
    void bfloat16AttentionForwardUsesCanonicalPrimitiveDag() {
        Tensor q = new Tensor(new double[]{
                1, 0,
                0, 1
        }, new int[]{1, 2, 2}, null, "q", DataType.BFLOAT16);
        Tensor k = new Tensor(new double[]{
                1, 0,
                0, 1
        }, new int[]{1, 2, 2}, null, "k", DataType.BFLOAT16);
        Tensor v = new Tensor(new double[]{
                10, 1,
                1, 10
        }, new int[]{1, 2, 2}, null, "v", DataType.BFLOAT16);
        q.setRequiresGrad(true);
        k.setRequiresGrad(true);
        v.setRequiresGrad(true);

        Tensor out = q.scaledDotProductAttention(k, v, AttentionOptions.defaults().withScale(1.0));
        CompiledGraph compiledGraph = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline());
        compiledGraph
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD);

        double e = Math.exp(1.0);
        double denom = e + 1.0;
        assertArrayEquals(new double[]{
                (10.0 * e + 1.0) / denom,
                (1.0 * e + 10.0) / denom,
                (10.0 + e) / denom,
                (1.0 + 10.0 * e) / denom
        }, out.toDoubleArrayCopy(), 2e-2);
        assertCanonicalAttentionGraph(compiledGraph);
    }

    @Test
    void bfloat16AttentionBackwardMatchesFloat64Baseline() {
        Tensor q64 = new Tensor(new double[]{
                1, 0,
                0, 1
        }, new int[]{1, 2, 2}, null, "q64", DataType.FLOAT64);
        Tensor k64 = new Tensor(new double[]{
                1, 0,
                0, 1
        }, new int[]{1, 2, 2}, null, "k64", DataType.FLOAT64);
        Tensor v64 = new Tensor(new double[]{
                10, 1,
                1, 10
        }, new int[]{1, 2, 2}, null, "v64", DataType.FLOAT64);
        q64.setRequiresGrad(true);
        k64.setRequiresGrad(true);
        v64.setRequiresGrad(true);
        Tensor loss64 = q64.scaledDotProductAttention(k64, v64, AttentionOptions.defaults().withScale(1.0)).sum();
        CompiledGraph.compile(loss64, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        Tensor q16 = new Tensor(new double[]{
                1, 0,
                0, 1
        }, new int[]{1, 2, 2}, null, "q16", DataType.BFLOAT16);
        Tensor k16 = new Tensor(new double[]{
                1, 0,
                0, 1
        }, new int[]{1, 2, 2}, null, "k16", DataType.BFLOAT16);
        Tensor v16 = new Tensor(new double[]{
                10, 1,
                1, 10
        }, new int[]{1, 2, 2}, null, "v16", DataType.BFLOAT16);
        q16.setRequiresGrad(true);
        k16.setRequiresGrad(true);
        v16.setRequiresGrad(true);
        Tensor loss16 = q16.scaledDotProductAttention(k16, v16, AttentionOptions.defaults().withScale(1.0)).sum();
        CompiledGraph compiledGraph = CompiledGraph.compile(loss16, CompileConfig.noGraphOptimizationBaseline());
        compiledGraph.prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        assertFalse(containsOp(compiledGraph, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_BACKWARD));
        assertNotNull(q16.getGradient());
        assertNotNull(k16.getGradient());
        assertNotNull(v16.getGradient());
        assertArrayEquals(q64.getGradient().toDoubleArrayCopy(), q16.getGradient().toDoubleArrayCopy(), 8e-2);
        assertArrayEquals(k64.getGradient().toDoubleArrayCopy(), k16.getGradient().toDoubleArrayCopy(), 8e-2);
        assertArrayEquals(v64.getGradient().toDoubleArrayCopy(), v16.getGradient().toDoubleArrayCopy(), 8e-2);
    }

    private static boolean containsOp(CompiledGraph compiledGraph, Operation.OpType opType) {
        return compiledGraph.program().compiledNodes().stream()
                .map(graph.model.CompiledNode::operation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(type -> type == opType);
    }

    private static void assertCanonicalAttentionGraph(CompiledGraph compiledGraph) {
        assertFalse(containsOp(compiledGraph, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION));
        assertTrue(containsOp(compiledGraph, Operation.OpType.MATMUL));
        assertTrue(containsOp(compiledGraph, Operation.OpType.REDUCE_MAX));
        assertTrue(containsOp(compiledGraph, Operation.OpType.EXP));
        assertTrue(containsOp(compiledGraph, Operation.OpType.SUM));
    }
}
