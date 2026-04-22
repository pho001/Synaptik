import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.runtime.RuntimeConfig;
import config.runtime.FusedExecutionPolicy;
import config.runtime.FusedPrimaryBackend;
import graph.CompiledGraph;
import graph.execution.PreparedExecution;
import operations.Operation;
import backend.blas.BlasProvider;
import config.backend.KernelTuningConfig;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class PreparedExecutionBuildTest {
    @Test
    void inferenceOnlyGraphBuildsForwardOnlyPreparedExecution() {
        Tensor a = new Tensor(new double[]{1.0, 2.0, 3.0, 4.0}, new int[]{4}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{0.5, 1.5, -2.0, 3.0}, new int[]{4}, null, "b", DataType.FLOAT64);
        Tensor out = a.add(b).mul(a).sigmoid();

        RuntimeConfig runtimeConfig = RuntimeConfig.inferenceDefaults();
        CompiledGraph compiledGraph = CompiledGraph.compile(out, OptimizerConfig.noOptimization());
        PreparedExecution execution = compiledGraph.prepare(runtimeConfig);

        assertNotNull(execution);
        assertEquals(runtimeConfig, execution.runtimeConfig());
        assertFalse(execution.supportsBackward());
        assertFalse(execution.forwardSteps().isEmpty());
        assertTrue(execution.backwardSteps().isEmpty());

        execution.execute(ExecutionMode.FORWARD);
        assertEquals(4, out.toDoubleArrayCopy().length);
    }

    @Test
    void repeatedPrepareOnSameCompiledGraphBuildsIndependentPreparedExecutions() {
        Tensor a = new Tensor(new double[]{1.0, 2.0, 3.0}, new int[]{3}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{4.0, 5.0, 6.0}, new int[]{3}, null, "b", DataType.FLOAT64);
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);

        Tensor out = a.mul(b).add(a);
        CompiledGraph compiled = CompiledGraph.compile(out, OptimizerConfig.noOptimization());

        PreparedExecution first = compiled.prepare(RuntimeConfig.trainingDefaults());
        PreparedExecution second = compiled.prepare(RuntimeConfig.trainingDefaults());

        assertNotSame(first, second);
        assertNotSame(first.forwardSteps(), second.forwardSteps());
        assertEquals(first.forwardSteps().size(), second.forwardSteps().size());
        assertEquals(first.backwardSteps().size(), second.backwardSteps().size());

        first.execute(ExecutionMode.FORWARD_BACKWARD);
        Tensor firstGradient = a.getGradient();
        assertNotNull(firstGradient);
        firstGradient.setDataAt(0, 999.0);

        second.execute(ExecutionMode.FORWARD_BACKWARD);
        assertArrayEquals(new double[]{5.0, 6.0, 7.0}, a.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{1.0, 2.0, 3.0}, b.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void preparedExecutionStepViewsAreUnmodifiable() {
        Tensor a = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{3.0, 4.0}, new int[]{2}, null, "b", DataType.FLOAT64);
        Tensor out = a.add(b).relu();

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        assertThrows(UnsupportedOperationException.class, () -> execution.forwardSteps().clear());
        assertThrows(UnsupportedOperationException.class, () -> execution.backwardSteps().clear());
    }

    @Test
    void bfloat16FusedPrepareSkipsCompiledAsmKernel() {
        Tensor a = new Tensor(new double[]{1.0, 2.0, 3.0, 4.0}, new int[]{4}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(new double[]{0.5, 1.5, -2.0, 3.0}, new int[]{4}, null, "b", DataType.BFLOAT16);
        Tensor out = a.add(b).mul(a).sigmoid();

        PreparedExecution execution = CompiledGraph.compile(out, fuseOnlyInferenceConfig())
                .prepare(RuntimeConfig.inferenceDefaults());

        var fusedStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.FUSED)
                .findFirst()
                .orElseThrow();

        assertEquals("BFLOAT16", fusedStep.metadata().cpuPlan().computeContract().storageType().name());
        assertEquals("F32", fusedStep.metadata().cpuPlan().computeContract().computeType().name());
        assertEquals("CPU_FUSED", fusedStep.metadata().cpuPlan().computeContract().backend().name());
        assertNotNull(fusedStep.metadata().fusedExecutable());
    }

    @Test
    void float32FusedPrepareUsesAsmExecutable() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{0.5f, 1.5f, -2f, 3f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor out = a.add(b).mul(a).sigmoid();

        PreparedExecution execution = CompiledGraph.compile(out, fuseOnlyInferenceConfig())
                .prepare(new RuntimeConfig(
                        kernelWithVectorMin(1),
                        config.runtime.ApproximationConfig.defaults(),
                        config.runtime.BlasConfig.disabled(),
                        new FusedExecutionPolicy(FusedPrimaryBackend.ASM, true)
                ));

        var fusedStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.FUSED)
                .findFirst()
                .orElseThrow();

        assertEquals("FLOAT32", fusedStep.metadata().cpuPlan().computeContract().storageType().name());
        assertEquals("F32", fusedStep.metadata().cpuPlan().computeContract().computeType().name());
        assertEquals("CPU_FUSED", fusedStep.metadata().cpuPlan().computeContract().backend().name());
        assertTrue(fusedStep.metadata().fusedExecutable().getClass().getName().startsWith("graph.fused.asm."));
    }

    @Test
    void float64FusedPrepareUsesAsmExecutable() {
        Tensor a = new Tensor(new double[]{1.0, 2.0, 3.0, 4.0}, new int[]{4}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{0.5, 1.5, -2.0, 3.0}, new int[]{4}, null, "b", DataType.FLOAT64);
        Tensor out = a.add(b).mul(a).sigmoid();

        PreparedExecution execution = CompiledGraph.compile(out, fuseOnlyInferenceConfig())
                .prepare(new RuntimeConfig(
                        kernelWithVectorMin(1),
                        config.runtime.ApproximationConfig.defaults(),
                        config.runtime.BlasConfig.disabled(),
                        new FusedExecutionPolicy(FusedPrimaryBackend.ASM, true)
                ));

        var fusedStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.FUSED)
                .findFirst()
                .orElseThrow();

        assertEquals("FLOAT64", fusedStep.metadata().cpuPlan().computeContract().storageType().name());
        assertEquals("F64", fusedStep.metadata().cpuPlan().computeContract().computeType().name());
        assertEquals("CPU_FUSED", fusedStep.metadata().cpuPlan().computeContract().backend().name());
        assertTrue(fusedStep.metadata().fusedExecutable().getClass().getName().startsWith("graph.fused.asm."));
    }

    @Test
    void fusedAsmExecutableCacheSeparatesWidthSpecializations() {
        Tensor a1 = new Tensor(new double[]{1.0, 2.0, 3.0, 4.0}, new int[]{4}, null, "a1", DataType.FLOAT64);
        Tensor b1 = new Tensor(new double[]{0.5, 1.5, -2.0, 3.0}, new int[]{4}, null, "b1", DataType.FLOAT64);
        Tensor out1 = a1.add(b1).mul(a1).sigmoid();

        PreparedExecution width1Execution = CompiledGraph.compile(out1, fuseOnlyInferenceConfig())
                .prepare(runtimeWithFusedAsmWidth(1));

        var width1Fused = width1Execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.FUSED)
                .findFirst()
                .orElseThrow();

        Tensor a2 = new Tensor(new double[]{1.0, 2.0, 3.0, 4.0}, new int[]{4}, null, "a2", DataType.FLOAT64);
        Tensor b2 = new Tensor(new double[]{0.5, 1.5, -2.0, 3.0}, new int[]{4}, null, "b2", DataType.FLOAT64);
        Tensor out2 = a2.add(b2).mul(a2).sigmoid();

        PreparedExecution width2Execution = CompiledGraph.compile(out2, fuseOnlyInferenceConfig())
                .prepare(runtimeWithFusedAsmWidth(2));

        var width2Fused = width2Execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.FUSED)
                .findFirst()
                .orElseThrow();

        assertEquals(1, width1Fused.metadata().cpuPlan().dispatchHints().vectorWidth());
        assertEquals(2, width2Fused.metadata().cpuPlan().dispatchHints().vectorWidth());
        assertNotEquals(
                width1Fused.metadata().fusedExecutable().getClass().getName(),
                width2Fused.metadata().fusedExecutable().getClass().getName()
        );
        assertTrue(width1Fused.metadata().fusedExecutable().getClass().getName().endsWith("W1"));
        assertTrue(width2Fused.metadata().fusedExecutable().getClass().getName().endsWith("W2"));
    }

    @Test
    void float32FusedPrepareCanBeForcedToAsmFallbackByPolicy() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{0.5f, 1.5f, -2f, 3f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor out = a.add(b).mul(a).sigmoid();

        PreparedExecution execution = CompiledGraph.compile(out, fuseOnlyInferenceConfig())
                .prepare(new RuntimeConfig(
                        config.backend.KernelTuningConfig.defaultsInference(),
                        config.runtime.ApproximationConfig.defaults(),
                        config.runtime.BlasConfig.disabled(),
                        new FusedExecutionPolicy(FusedPrimaryBackend.ASM, true)
                ));

        var fusedStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.FUSED)
                .findFirst()
                .orElseThrow();

        assertTrue(fusedStep.metadata().fusedExecutable().getClass().getName().startsWith("graph.fused.asm."));
    }

    @Test
    void bfloat16LinearToReluPublishesFloatContinuationInInference() {
        Tensor input = new Tensor(new double[32 * 64], new int[]{32, 64}, null, "input", DataType.BFLOAT16);
        Tensor weight = new Tensor(new double[64 * 96], new int[]{64, 96}, null, "weight", DataType.BFLOAT16);
        Tensor bias = new Tensor(new double[96], new int[]{96}, null, "bias", DataType.BFLOAT16);
        Tensor out = input.linear(weight, bias).relu();

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .prepare(bfloat16BlasRuntime());

        var linearStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.LINEAR)
                .findFirst()
                .orElseThrow();

        assertEquals("BFLOAT16", linearStep.metadata().cpuPlan().computeContract().storageType().name());
        assertEquals("F32", linearStep.metadata().cpuPlan().computeContract().computeType().name());
        assertEquals("CPU_MATMUL_BLAS", linearStep.metadata().cpuPlan().computeContract().backend().name());
        assertTrue(linearStep.metadata().cpuPlan().publishFloatContinuation());
    }

    @Test
    void bfloat16MatmulToAddPublishesFloatContinuationInInference() {
        Tensor a = new Tensor(new double[64 * 64], new int[]{64, 64}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(new double[64 * 96], new int[]{64, 96}, null, "b", DataType.BFLOAT16);
        Tensor c = new Tensor(new double[64 * 96], new int[]{64, 96}, null, "c", DataType.BFLOAT16);
        Tensor out = a.matmul(b).add(c);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .prepare(bfloat16BlasRuntime());

        var matmulStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.MATMUL)
                .findFirst()
                .orElseThrow();

        assertEquals("BFLOAT16", matmulStep.metadata().cpuPlan().computeContract().storageType().name());
        assertEquals("F32", matmulStep.metadata().cpuPlan().computeContract().computeType().name());
        assertEquals("CPU_MATMUL_BLAS", matmulStep.metadata().cpuPlan().computeContract().backend().name());
        assertTrue(matmulStep.metadata().cpuPlan().publishFloatContinuation());
        assertNotNull(matmulStep.metadata().cpuPlan().matMulExecutable());
        assertEquals("BF16BlasMatMulExecutable", matmulStep.metadata().cpuPlan().matMulExecutable().getClass().getSimpleName());
    }

    @Test
    void bfloat16MatmulToBroadcastAddToReluPublishesFloatContinuationInInference() {
        Tensor a = new Tensor(new double[16 * 8], new int[]{16, 8}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(new double[8 * 12], new int[]{8, 12}, null, "b", DataType.BFLOAT16);
        Tensor bias = new Tensor(new double[12], new int[]{12}, null, "bias", DataType.BFLOAT16);
        Tensor out = a.matmul(b).add(bias).relu();

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var matmulStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.MATMUL)
                .findFirst()
                .orElseThrow();
        var addStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.ADD)
                .findFirst()
                .orElseThrow();

        assertTrue(matmulStep.metadata().cpuPlan().publishFloatContinuation());
        assertTrue(addStep.metadata().cpuPlan().publishFloatContinuation());
    }

    @Test
    void float64MatmulPrepareBuildsBlasExecutableWhenEligible() {
        Tensor a = new Tensor(new double[64 * 64], new int[]{64, 64}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[64 * 96], new int[]{64, 96}, null, "b", DataType.FLOAT64);
        Tensor out = a.matmul(b);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(bfloat16BlasRuntime());

        var matmulStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.MATMUL)
                .findFirst()
                .orElseThrow();

        assertNotNull(matmulStep.metadata().cpuPlan().matMulExecutable());
        assertEquals("F64BlasMatMulExecutable", matmulStep.metadata().cpuPlan().matMulExecutable().getClass().getSimpleName());
    }

    @Test
    void float32MatmulPrepareBuildsJavaExecutableWhenBlasDisabled() {
        Tensor a = new Tensor(new float[64 * 64], new int[]{64, 64}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[64 * 96], new int[]{64, 96}, null, "b", DataType.FLOAT32);
        Tensor out = a.matmul(b);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var matmulStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.MATMUL)
                .findFirst()
                .orElseThrow();

        assertNotNull(matmulStep.metadata().cpuPlan().matMulExecutable());
        assertEquals("F32JavaMatMulExecutable", matmulStep.metadata().cpuPlan().matMulExecutable().getClass().getSimpleName());
    }

    @Test
    void bfloat16MatmulToFusedNumericChainPublishesFloatContinuationInInference() {
        Tensor a = new Tensor(new double[64 * 64], new int[]{64, 64}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(new double[64 * 96], new int[]{64, 96}, null, "b", DataType.BFLOAT16);
        Tensor out = a.matmul(b).relu().abs().clampMax(1.0);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .prepare(bfloat16BlasRuntime());

        var matmulStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.MATMUL)
                .findFirst()
                .orElseThrow();

        assertEquals("BFLOAT16", matmulStep.metadata().cpuPlan().computeContract().storageType().name());
        assertEquals("F32", matmulStep.metadata().cpuPlan().computeContract().computeType().name());
        assertEquals("CPU_MATMUL_BLAS", matmulStep.metadata().cpuPlan().computeContract().backend().name());
        assertTrue(matmulStep.metadata().cpuPlan().publishFloatContinuation());
    }

    @Test
    void bfloat16MatmulToNegPublishesFloatContinuationInInference() {
        Tensor a = new Tensor(new double[64 * 64], new int[]{64, 64}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(new double[64 * 96], new int[]{64, 96}, null, "b", DataType.BFLOAT16);
        Tensor out = a.matmul(b).neg();

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .prepare(bfloat16BlasRuntime());

        var matmulStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.MATMUL)
                .findFirst()
                .orElseThrow();

        assertTrue(matmulStep.metadata().cpuPlan().publishFloatContinuation());
    }

    @Test
    void bfloat16MatmulToMulScalarPublishesFloatContinuationInInference() {
        Tensor a = new Tensor(new double[64 * 64], new int[]{64, 64}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(new double[64 * 96], new int[]{64, 96}, null, "b", DataType.BFLOAT16);
        Tensor out = a.matmul(b).mul(0.5);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .prepare(bfloat16BlasRuntime());

        var matmulStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.MATMUL)
                .findFirst()
                .orElseThrow();

        assertTrue(matmulStep.metadata().cpuPlan().publishFloatContinuation());
    }

    @Test
    void bfloat16MatmulToPowPublishesFloatContinuationInInference() {
        Tensor a = new Tensor(new double[64 * 64], new int[]{64, 64}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(new double[64 * 96], new int[]{64, 96}, null, "b", DataType.BFLOAT16);
        Tensor out = a.matmul(b).pow(1.5);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .prepare(bfloat16BlasRuntime());

        var matmulStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.MATMUL)
                .findFirst()
                .orElseThrow();

        assertTrue(matmulStep.metadata().cpuPlan().publishFloatContinuation());
    }

    @Test
    void bfloat16MatmulReshapeToNegPublishesFloatContinuationInInference() {
        Tensor a = new Tensor(new double[64 * 64], new int[]{64, 64}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(new double[64 * 96], new int[]{64, 96}, null, "b", DataType.BFLOAT16);
        Tensor out = a.matmul(b).reshape(32, 192).neg();

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .prepare(bfloat16BlasRuntime());

        var matmulStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.MATMUL)
                .findFirst()
                .orElseThrow();

        assertTrue(matmulStep.metadata().cpuPlan().publishFloatContinuation());
    }

    @Test
    void bfloat16MatmulToWhereToReluPublishesFloatContinuationInInference() {
        Tensor a = new Tensor(new double[8 * 8], new int[]{8, 8}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(new double[8 * 12], new int[]{8, 12}, null, "b", DataType.BFLOAT16);
        Tensor mask = new Tensor(new byte[8], new int[]{8, 1}, null, "mask", DataType.BOOL);
        Tensor fill = Tensor.scalar(-1.0, DataType.BFLOAT16);
        Tensor out = Tensor.where(mask, a.matmul(b), fill).relu();

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var matmulStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.MATMUL)
                .findFirst()
                .orElseThrow();
        var whereStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.WHERE)
                .findFirst()
                .orElseThrow();

        assertTrue(matmulStep.metadata().cpuPlan().publishFloatContinuation());
        assertTrue(whereStep.metadata().cpuPlan().publishFloatContinuation());
    }

    @Test
    void bfloat16LinearToLogSoftmaxPreparesBfloat16ReductionPath() {
        Tensor input = new Tensor(new double[32 * 64], new int[]{32, 64}, null, "input", DataType.BFLOAT16);
        Tensor weight = new Tensor(new double[64 * 96], new int[]{64, 96}, null, "weight", DataType.BFLOAT16);
        Tensor bias = new Tensor(new double[96], new int[]{96}, null, "bias", DataType.BFLOAT16);
        Tensor out = input.linear(weight, bias).logSoftmax(1);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .prepare(bfloat16BlasRuntime());

        var linearStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.LINEAR)
                .findFirst()
                .orElseThrow();
        var logSoftmaxStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.LOG_SOFTMAX)
                .findFirst()
                .orElseThrow();

        assertTrue(linearStep.metadata().cpuPlan().publishFloatContinuation());
        assertEquals("BFLOAT16", logSoftmaxStep.metadata().cpuPlan().computeContract().storageType().name());
        assertEquals("F32", logSoftmaxStep.metadata().cpuPlan().computeContract().computeType().name());
        assertEquals("CPU_REDUCTION", logSoftmaxStep.metadata().cpuPlan().computeContract().backend().name());
    }

    @Test
    void bfloat16SoftmaxToMeanKeepsFloatContinuationInInference() {
        Tensor input = new Tensor(new double[32 * 64], new int[]{32, 64}, null, "input", DataType.BFLOAT16);
        Tensor weight = new Tensor(new double[64 * 96], new int[]{64, 96}, null, "weight", DataType.BFLOAT16);
        Tensor bias = new Tensor(new double[96], new int[]{96}, null, "bias", DataType.BFLOAT16);
        Tensor out = input.linear(weight, bias).softmax(1).mean(1);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .prepare(bfloat16BlasRuntime());

        var linearStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.LINEAR)
                .findFirst()
                .orElseThrow();
        var softmaxStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.SOFTMAX)
                .findFirst()
                .orElseThrow();

        assertTrue(linearStep.metadata().cpuPlan().publishFloatContinuation());
        assertTrue(softmaxStep.metadata().cpuPlan().publishFloatContinuation());
        assertEquals("BFLOAT16", softmaxStep.metadata().cpuPlan().computeContract().storageType().name());
        assertEquals("F32", softmaxStep.metadata().cpuPlan().computeContract().computeType().name());
        assertEquals("CPU_REDUCTION", softmaxStep.metadata().cpuPlan().computeContract().backend().name());
    }

    @Test
    void bfloat16LayerNormToMeanKeepsFloatContinuationInInference() {
        Tensor input = new Tensor(new double[32 * 64], new int[]{32, 64}, null, "input", DataType.BFLOAT16);
        Tensor gamma = new Tensor(new double[64], new int[]{64}, null, "gamma", DataType.BFLOAT16);
        Tensor beta = new Tensor(new double[64], new int[]{64}, null, "beta", DataType.BFLOAT16);
        Tensor out = input.layerNorm(gamma, beta, 1e-5).mean(1);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .prepare(bfloat16BlasRuntime());

        var layerNormStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.LAYER_NORM)
                .findFirst()
                .orElseThrow();

        assertTrue(layerNormStep.metadata().cpuPlan().publishFloatContinuation());
        assertEquals("BFLOAT16", layerNormStep.metadata().cpuPlan().computeContract().storageType().name());
        assertEquals("F32", layerNormStep.metadata().cpuPlan().computeContract().computeType().name());
    }

    @Test
    void bfloat16RmsNormToMeanKeepsFloatContinuationInInference() {
        Tensor input = new Tensor(new double[32 * 64], new int[]{32, 64}, null, "input", DataType.BFLOAT16);
        Tensor gamma = new Tensor(new double[64], new int[]{64}, null, "gamma", DataType.BFLOAT16);
        Tensor out = input.rmsNorm(gamma, 1e-5).mean(1);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .prepare(bfloat16BlasRuntime());

        var rmsNormStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.RMS_NORM)
                .findFirst()
                .orElseThrow();

        assertTrue(rmsNormStep.metadata().cpuPlan().publishFloatContinuation());
        assertEquals("BFLOAT16", rmsNormStep.metadata().cpuPlan().computeContract().storageType().name());
        assertEquals("F32", rmsNormStep.metadata().cpuPlan().computeContract().computeType().name());
    }

    @Test
    void bfloat16LogSoftmaxToNllLossKeepsFloatContinuationInInference() {
        Tensor logits = new Tensor(new double[16 * 8], new int[]{16, 8}, null, "logits", DataType.BFLOAT16);
        Tensor targets = new Tensor(new double[16 * 8], new int[]{16, 8}, null, "targets", DataType.BFLOAT16);
        Tensor out = logits.logSoftmax(1).nllLoss(targets, 1);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .prepare(bfloat16BlasRuntime());

        var logSoftmaxStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.LOG_SOFTMAX)
                .findFirst()
                .orElseThrow();

        assertTrue(logSoftmaxStep.metadata().cpuPlan().publishFloatContinuation());
    }

    @Test
    void bfloat16MatmulToCrossEntropyLossPublishesFloatContinuationInInference() {
        Tensor a = new Tensor(new double[32 * 64], new int[]{32, 64}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(new double[64 * 96], new int[]{64, 96}, null, "b", DataType.BFLOAT16);
        Tensor targets = new Tensor(new double[32 * 96], new int[]{32, 96}, null, "targets", DataType.BFLOAT16);
        Tensor out = a.matmul(b).crossEntropyLoss(targets, 1);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .prepare(bfloat16BlasRuntime());

        var matmulStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.MATMUL)
                .findFirst()
                .orElseThrow();

        assertTrue(matmulStep.metadata().cpuPlan().publishFloatContinuation());
    }

    @Test
    void bfloat16SoftmaxGradPublishesFloatContinuationInTrainingUnaryChain() {
        Tensor input = new Tensor(new double[16 * 8], new int[]{16, 8}, null, "input", DataType.BFLOAT16);
        input.setRequiresGrad(true);
        Tensor out = input.exp().softmax(1).sum();

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.trainingDefaults())
                .prepare(bfloat16BlasRuntime());

        var softmaxGradStep = execution.backwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.SOFTMAX_GRAD)
                .findFirst()
                .orElseThrow();

        assertEquals("BFLOAT16", softmaxGradStep.metadata().cpuPlan().computeContract().storageType().name());
        assertEquals("F32", softmaxGradStep.metadata().cpuPlan().computeContract().computeType().name());
        assertEquals("CPU_REDUCTION", softmaxGradStep.metadata().cpuPlan().computeContract().backend().name());
        assertTrue(softmaxGradStep.metadata().cpuPlan().publishFloatContinuation());
    }

    @Test
    void bfloat16LogSoftmaxGradPublishesFloatContinuationInTrainingUnaryChain() {
        Tensor input = new Tensor(new double[16 * 8], new int[]{16, 8}, null, "input", DataType.BFLOAT16);
        input.setRequiresGrad(true);
        Tensor out = input.exp().logSoftmax(1).sum();

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.trainingDefaults())
                .prepare(bfloat16BlasRuntime());

        var logSoftmaxGradStep = execution.backwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.LOG_SOFTMAX_GRAD)
                .findFirst()
                .orElseThrow();

        assertEquals("BFLOAT16", logSoftmaxGradStep.metadata().cpuPlan().computeContract().storageType().name());
        assertEquals("F32", logSoftmaxGradStep.metadata().cpuPlan().computeContract().computeType().name());
        assertEquals("CPU_REDUCTION", logSoftmaxGradStep.metadata().cpuPlan().computeContract().backend().name());
        assertTrue(logSoftmaxGradStep.metadata().cpuPlan().publishFloatContinuation());
    }

    private static RuntimeConfig bfloat16BlasRuntime() {
        return new RuntimeConfig(
                KernelTuningConfig.defaultsInference(),
                config.runtime.ApproximationConfig.defaults(),
                new config.runtime.BlasConfig(
                        BlasProvider.OPENBLAS_FFM,
                        1L,
                        false,
                        100.0d,
                        false,
                        1
                )
        );
    }

    private static KernelTuningConfig kernelWithVectorMin(int vectorMinSize) {
        return kernelWithVectorMinAndFusedAsmWidth(vectorMinSize, KernelTuningConfig.defaultsInference().cpu().fusedCheapContiguousAsmVectorWidth());
    }

    private static RuntimeConfig runtimeWithFusedAsmWidth(int fusedAsmWidth) {
        return new RuntimeConfig(
                kernelWithVectorMinAndFusedAsmWidth(1, fusedAsmWidth),
                config.runtime.ApproximationConfig.defaults(),
                config.runtime.BlasConfig.disabled(),
                new FusedExecutionPolicy(FusedPrimaryBackend.ASM, true)
        );
    }

    private static KernelTuningConfig kernelWithVectorMinAndFusedAsmWidth(int vectorMinSize, int fusedAsmWidth) {
        var base = KernelTuningConfig.defaultsInference();
        var cpu = base.cpu();
        return new KernelTuningConfig(
                new config.backend.CpuKernelConfig(
                        cpu.loopUnrollFactor(),
                        cpu.matMulTileM(),
                        cpu.matMulTileN(),
                        cpu.matMulTileK(),
                        vectorMinSize,
                        vectorMinSize,
                        vectorMinSize,
                        vectorMinSize,
                        vectorMinSize,
                        cpu.parallelMinSize(),
                        cpu.parallelMinSize(),
                        cpu.fusedCheapParallelMinSize(),
                        cpu.fusedTranscendentalParallelMinSize(),
                        cpu.reductionParallelMinSize(),
                        cpu.contiguousMaterializeThreshold(),
                        cpu.lowCostTargetChunksPerWorker(),
                        cpu.mediumCostTargetChunksPerWorker(),
                        cpu.highCostTargetChunksPerWorker(),
                        cpu.minScalarChunkSize(),
                        cpu.minVectorChunkSize(),
                        cpu.minReductionChunkSize(),
                        cpu.commonPoolLowCostMaxWorkPerWorker(),
                        fusedAsmWidth,
                        fusedAsmWidth,
                        fusedAsmWidth,
                        fusedAsmWidth,
                        cpu.sumAccuracyMode(),
                        cpu.matMulParallelMinSize(),
                        cpu.attentionMatMulPolicy()
                ),
                base.cuda(),
                base.opencl()
        );
    }

    private static OptimizerConfig fuseOnlyInferenceConfig() {
        return new OptimizerConfig(
                java.util.List.of(config.optimizer.OptimizerStage.FUSE),
                config.optimizer.RewriteConfig.defaults(),
                config.optimizer.CseConfig.strictDefaults(),
                config.optimizer.FuseConfig.inferenceDefaults(),
                config.optimizer.MemoryConfig.defaults()
        );
    }
}
