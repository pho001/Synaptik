import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.runtime.RuntimeConfig;
import config.runtime.FusedExecutionPolicy;
import config.runtime.FusedPrimaryBackend;
import graph.CompiledGraph;
import graph.execution.PreparedExecution;
import operations.Operation;
import backend.blas.BlasProvider;
import backend.blas.BlasThreadPolicy;
import config.backend.KernelTuningConfig;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
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

        assertEquals("BF16_F32_COMPUTE", fusedStep.metadata().cpuPlan().computeMode().name());
        assertNotNull(fusedStep.metadata().fusedExecutable());
    }

    @Test
    void float32FusedPrepareUsesDirectVectorExecutable() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{0.5f, 1.5f, -2f, 3f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor out = a.add(b).mul(a).sigmoid();

        PreparedExecution execution = CompiledGraph.compile(out, fuseOnlyInferenceConfig())
                .prepare(new RuntimeConfig(
                        kernelWithVectorMin(1),
                        config.runtime.ApproximationConfig.defaults(),
                        config.runtime.BlasConfig.disabled(),
                        new FusedExecutionPolicy(FusedPrimaryBackend.DIRECT_VECTOR, true, true)
                ));

        var fusedStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.FUSED)
                .findFirst()
                .orElseThrow();

        assertEquals("F32", fusedStep.metadata().cpuPlan().computeMode().name());
        assertEquals("Float32PreparedFusedExecutable", fusedStep.metadata().fusedExecutable().getClass().getSimpleName());
    }

    @Test
    void float64FusedPrepareUsesDirectVectorExecutable() {
        Tensor a = new Tensor(new double[]{1.0, 2.0, 3.0, 4.0}, new int[]{4}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{0.5, 1.5, -2.0, 3.0}, new int[]{4}, null, "b", DataType.FLOAT64);
        Tensor out = a.add(b).mul(a).sigmoid();

        PreparedExecution execution = CompiledGraph.compile(out, fuseOnlyInferenceConfig())
                .prepare(new RuntimeConfig(
                        kernelWithVectorMin(1),
                        config.runtime.ApproximationConfig.defaults(),
                        config.runtime.BlasConfig.disabled(),
                        new FusedExecutionPolicy(FusedPrimaryBackend.DIRECT_VECTOR, true, true)
                ));

        var fusedStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.FUSED)
                .findFirst()
                .orElseThrow();

        assertEquals("F64", fusedStep.metadata().cpuPlan().computeMode().name());
        assertEquals("Float64PreparedFusedExecutable", fusedStep.metadata().fusedExecutable().getClass().getSimpleName());
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
                        new FusedExecutionPolicy(FusedPrimaryBackend.ASM, true, true)
                ));

        var fusedStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.FUSED)
                .findFirst()
                .orElseThrow();

        assertTrue(fusedStep.metadata().fusedExecutable().getClass().getName().startsWith("graph.fused.asm."));
    }

    @Test
    void float32FusedPrepareCanBeRoutedToAsmFallbackByVectorThresholdPolicy() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{0.5f, 1.5f, -2f, 3f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor out = a.add(b).mul(a).sigmoid();

        PreparedExecution execution = CompiledGraph.compile(out, fuseOnlyInferenceConfig())
                .prepare(new RuntimeConfig(
                        kernelWithVectorMin(1_000_000),
                        config.runtime.ApproximationConfig.defaults(),
                        config.runtime.BlasConfig.disabled(),
                        new FusedExecutionPolicy(FusedPrimaryBackend.DIRECT_VECTOR, true, true)
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

        assertEquals("BF16_BLAS", linearStep.metadata().cpuPlan().computeMode().name());
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

        assertEquals("BF16_BLAS", matmulStep.metadata().cpuPlan().computeMode().name());
        assertTrue(matmulStep.metadata().cpuPlan().publishFloatContinuation());
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

        assertEquals("BF16_BLAS", matmulStep.metadata().cpuPlan().computeMode().name());
        assertTrue(matmulStep.metadata().cpuPlan().publishFloatContinuation());
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
                        BlasThreadPolicy.FIXED,
                        1
                )
        );
    }

    private static KernelTuningConfig kernelWithVectorMin(int vectorMinSize) {
        var base = KernelTuningConfig.defaultsInference();
        var cpu = base.cpu();
        return new KernelTuningConfig(
                new config.backend.CpuKernelConfig(
                        cpu.loopUnrollFactor(),
                        cpu.matMulTileM(),
                        cpu.matMulTileN(),
                        cpu.matMulTileK(),
                        vectorMinSize,
                        cpu.parallelMinSize(),
                        cpu.contiguousMaterializeThreshold(),
                        cpu.lowCostTargetChunksPerWorker(),
                        cpu.mediumCostTargetChunksPerWorker(),
                        cpu.highCostTargetChunksPerWorker(),
                        cpu.minScalarChunkSize(),
                        cpu.minVectorChunkSize(),
                        cpu.minReductionChunkSize(),
                        cpu.commonPoolLowCostMaxWorkPerWorker(),
                        cpu.sumAccuracyMode(),
                        cpu.vectorPolicyCheap(),
                        cpu.vectorPolicyTranscendental(),
                        cpu.vectorPolicyReduction(),
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
