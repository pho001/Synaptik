package backend.cpu.nativecpu;

import backend.blas.BlasProvider;
import backend.blas.OpenBlasFfmBridge;
import backend.runtime.ExecutionMode;
import config.backend.KernelTuningConfig;
import config.compile.CompileConfig;
import config.compile.RegionOptimizationConfig;
import config.compile.SemanticCanonicalizationConfig;
import config.runtime.ApproximationConfig;
import config.runtime.BlasConfig;
import config.runtime.BlasStorageMode;
import config.runtime.CpuStorageProfile;
import config.runtime.NativeCpuFailurePolicy;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.execution.PreparedNodeExecution;
import graph.execution.PublicationPolicy;
import graph.execution.trace.ExecutionStepTrace;
import operations.Operation;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NativeCpuChainPlannerTest {
    @Test
    void cpuNativeMarksDenseElementwiseChainAsRequiredNativeSegment() {
        Tensor out = f32("left").add(f32("right")).relu();
        List<PreparedNodeExecution> steps = CompiledGraph.compile(out, compileConfig())
                .prepare(runtime(CpuStorageProfile.CPU_NATIVE))
                .forwardSteps();

        PreparedNativeCpuPlan add = nativePlan(steps, Operation.OpType.ADD);
        PreparedNativeCpuPlan relu = nativePlan(steps, Operation.OpType.RELU);

        assertEquals(NativeCpuChainDecision.REQUIRED_NATIVE, add.chainDecision());
        assertEquals(NativeCpuChainDecision.REQUIRED_NATIVE, relu.chainDecision());
        assertEquals("cpu-native-required", add.chainReason());
        assertEquals(add.chainSegmentId(), relu.chainSegmentId());
        assertTrue(add.chainSegmentId() >= 0);
    }

    @Test
    void autoRejectsSlowNonBlasNativeOpsWithoutBenchmarkEvidence() {
        Tensor out = f32("left").add(f32("right")).relu();
        List<PreparedNodeExecution> steps = CompiledGraph.compile(out, compileConfig())
                .prepare(runtime(CpuStorageProfile.AUTO))
                .forwardSteps();

        PreparedNativeCpuPlan add = nativePlan(steps, Operation.OpType.ADD);
        PreparedNativeCpuPlan relu = nativePlan(steps, Operation.OpType.RELU);

        assertEquals(PreparedNativeCpuRoute.NONE, add.route());
        assertEquals(NativeCpuChainDecision.AUTO_REJECTED_SLOW_OP, add.chainDecision());
        assertEquals("auto-rejected-slow-op:add", add.chainReason());
        assertEquals(-1, add.chainSegmentId());
        assertEquals(PreparedNativeCpuRoute.NONE, relu.route());
        assertEquals(NativeCpuChainDecision.AUTO_REJECTED_SLOW_OP, relu.chainDecision());
    }

    @Test
    void autoKeepsProviderAndViewOnlySegmentNativeWhenLegal() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat64GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor out = matrix(DataType.FLOAT64, "a").matmul(matrix(DataType.FLOAT64, "b")).reshape(4096);
        List<PreparedNodeExecution> steps = CompiledGraph.compile(out, compileConfig())
                .prepare(openBlasRuntime(CpuStorageProfile.AUTO))
                .forwardSteps();

        PreparedNativeCpuPlan matmul = nativePlan(steps, Operation.OpType.MATMUL);
        PreparedNativeCpuPlan reshape = nativePlan(steps, Operation.OpType.RESHAPE);

        assertEquals(PreparedNativeCpuRoute.NATIVE_EXECUTABLE, matmul.route());
        assertEquals(NativeCpuChainDecision.AUTO_FAST_NATIVE, matmul.chainDecision());
        assertEquals(PreparedNativeCpuRoute.VIEW_ALIAS, reshape.route());
        assertEquals(NativeCpuChainDecision.AUTO_FAST_NATIVE, reshape.chainDecision());
        assertEquals(matmul.chainSegmentId(), reshape.chainSegmentId());
    }

    @Test
    void unsupportedNativeOpGetsUnsupportedChainDecision() {
        Tensor out = f32("erf").erf();
        PreparedNativeCpuPlan plan = nativePlan(
                CompiledGraph.compile(out, compileConfig())
                        .prepare(runtime(CpuStorageProfile.CPU_NATIVE))
                        .forwardSteps(),
                Operation.OpType.ERF
        );

        assertEquals(PreparedNativeCpuRoute.FALLBACK_ONLY, plan.route());
        assertEquals(NativeCpuChainDecision.UNSUPPORTED_OP, plan.chainDecision());
        assertEquals("native-kernel-unsupported:erf", plan.chainReason());
        assertEquals(-1, plan.chainSegmentId());
    }

    @Test
    void cpuArrayHasNoNativeChainAnnotation() {
        Tensor out = f32("left").add(f32("right")).relu();
        PreparedNativeCpuPlan add = nativePlan(
                CompiledGraph.compile(out, compileConfig())
                        .prepare(runtime(CpuStorageProfile.CPU_ARRAY))
                        .forwardSteps(),
                Operation.OpType.ADD
        );

        assertEquals(NativeCpuChainDecision.NONE, add.chainDecision());
        assertEquals(-1, add.chainSegmentId());
        assertEquals("", add.chainReason());
    }

    @Test
    void tracedNativeSegmentIncludesChainAttrsAndDoesNotMaterializeIntermediateArray() {
        Tensor out = f32("left").add(f32("right")).relu();

        var prepared = CompiledGraph.compile(out, compileConfig()).prepare(runtime(CpuStorageProfile.CPU_NATIVE));
        int addNodeId = prepared.forwardSteps().stream()
                .filter(step -> step.compiledNode().operation() != null)
                .filter(step -> step.compiledNode().operation().opType() == Operation.OpType.ADD)
                .findFirst()
                .orElseThrow()
                .compiledNode()
                .id();
        var trace = prepared.executeTraced(ExecutionMode.FORWARD, PublicationPolicy.NONE);

        ExecutionStepTrace add = step(trace.steps(), Operation.OpType.ADD);
        ExecutionStepTrace relu = step(trace.steps(), Operation.OpType.RELU);
        Map<String, Object> addAttrs = add.metadata().attributes();
        Map<String, Object> reluAttrs = relu.metadata().attributes();

        assertEquals("REQUIRED_NATIVE", addAttrs.get("nativeCpuChainDecision"));
        assertEquals("cpu-native-required", addAttrs.get("nativeCpuChainReason"));
        assertEquals(addAttrs.get("nativeCpuChainSegmentId"), reluAttrs.get("nativeCpuChainSegmentId"));
        assertFalse(trace.cpuMaterializations().stream()
                .anyMatch(materialization -> materialization.nodeId() == addNodeId));
    }

    @Test
    void tracedAutoRejectedSlowOpIncludesChainDecision() {
        Tensor out = f32("left").add(f32("right")).relu();

        var trace = CompiledGraph.compile(out, compileConfig())
                .executeTraced(
                        runtime(CpuStorageProfile.AUTO),
                        ExecutionMode.FORWARD,
                        PublicationPolicy.NONE
                );

        Map<String, Object> addAttrs = step(trace.steps(), Operation.OpType.ADD).metadata().attributes();

        assertEquals("AUTO_REJECTED_SLOW_OP", addAttrs.get("nativeCpuChainDecision"));
        assertEquals("auto-rejected-slow-op:add", addAttrs.get("nativeCpuChainReason"));
        assertEquals(-1, addAttrs.get("nativeCpuChainSegmentId"));
    }

    private static PreparedNativeCpuPlan nativePlan(List<PreparedNodeExecution> steps, Operation.OpType opType) {
        return steps.stream()
                .filter(step -> step.compiledNode().operation() != null)
                .filter(step -> step.compiledNode().operation().opType() == opType)
                .findFirst()
                .orElseThrow()
                .metadata()
                .cpuPlan()
                .nativeCpuPlan();
    }

    private static ExecutionStepTrace step(List<ExecutionStepTrace> steps, Operation.OpType opType) {
        return steps.stream()
                .filter(step -> opType.name().equals(step.opType()))
                .findFirst()
                .orElseThrow();
    }

    private static RuntimeConfig runtime(CpuStorageProfile storageProfile) {
        return RuntimeConfig.inferenceDefaults()
                .withCpuStorageProfile(storageProfile)
                .withNativeCpuFailurePolicy(NativeCpuFailurePolicy.FALLBACK_TO_ARRAY);
    }

    private static RuntimeConfig openBlasRuntime(CpuStorageProfile storageProfile) {
        return new RuntimeConfig(
                KernelTuningConfig.defaultsInference(),
                ApproximationConfig.defaults(),
                new BlasConfig(
                        BlasProvider.OPENBLAS_FFM,
                        1L,
                        false,
                        100.0d,
                        false,
                        100.0d,
                        BlasStorageMode.AUTO,
                        false
                )
        )
                .withCpuStorageProfile(storageProfile)
                .withNativeCpuFailurePolicy(NativeCpuFailurePolicy.FALLBACK_TO_ARRAY);
    }

    private static CompileConfig compileConfig() {
        return CompileConfig.noGraphOptimizationBaseline()
                .withSemanticCanonicalization(SemanticCanonicalizationConfig.disabled())
                .withRegionOptimization(RegionOptimizationConfig.disabled());
    }

    private static Tensor f32(String label) {
        return new Tensor(new float[]{1f, -2f, 3f, -4f}, new int[]{2, 2}, null, label, DataType.FLOAT32);
    }

    private static Tensor matrix(DataType dataType, String label) {
        double[] values = new double[64 * 64];
        for (int i = 0; i < values.length; i++) {
            values[i] = Math.sin(i * 0.013d);
        }
        return new Tensor(values, new int[]{64, 64}, null, label, dataType);
    }
}
