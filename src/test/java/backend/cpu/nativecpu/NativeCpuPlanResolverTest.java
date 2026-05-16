package backend.cpu.nativecpu;

import backend.cpu.kernels.CpuDTypeOps;
import config.compile.CompileConfig;
import config.compile.RegionOptimizationConfig;
import config.compile.SemanticCanonicalizationConfig;
import config.runtime.CpuStorageProfile;
import config.runtime.NativeCpuFailurePolicy;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.execution.PreparedNodeExecution;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NativeCpuPlanResolverTest {
    @Test
    void cpuArrayAndAutoProfilesDoNotPrepareNativeNonBlasRoutes() {
        Tensor cpuArrayOut = f32("cpu_array_left").add(f32("cpu_array_right"));
        PreparedNativeCpuPlan cpuArray = nativePlan(cpuArrayOut, Operation.OpType.ADD, runtime(CpuStorageProfile.CPU_ARRAY));

        assertEquals(PreparedNativeCpuRoute.NONE, cpuArray.route());
        assertEquals(PreparedNativeCpuInputPolicy.ALL_CPU, cpuArray.inputPolicy());
        assertEquals(CpuStorageProfile.CPU_ARRAY, cpuArray.requestedStorage());
        assertEquals("cpu-storage-profile-not-native:cpu_array", cpuArray.fallbackReason());

        Tensor autoOut = f32("auto_left").add(f32("auto_right"));
        PreparedNativeCpuPlan auto = nativePlan(autoOut, Operation.OpType.ADD, runtime(CpuStorageProfile.AUTO));

        assertEquals(PreparedNativeCpuRoute.NONE, auto.route());
        assertEquals(PreparedNativeCpuInputPolicy.ALL_CPU, auto.inputPolicy());
        assertEquals(CpuStorageProfile.AUTO, auto.requestedStorage());
        assertEquals("cpu-storage-profile-not-native:auto", auto.fallbackReason());
    }

    @Test
    void cpuNativePreparesDenseF32ElementwiseAndReductionRoutes() {
        assertNativeExecutable(f32("add_left").add(f32("add_right")), Operation.OpType.ADD);
        assertNativeExecutable(f32("relu_input").relu(), Operation.OpType.RELU);
        assertNativeExecutable(f32("sum_input").sum(), Operation.OpType.SUM);
        assertNativeExecutable(f32("mean_input").mean(1), Operation.OpType.MEAN);
    }

    @Test
    void cpuNativePreparesBf16CastContiguousViewAndCompareRoutes() {
        assertNativeExecutable(bf16("bf16_add_left").add(bf16("bf16_add_right")), Operation.OpType.ADD);
        assertNativeExecutable(f32("cast_input").cast(DataType.BFLOAT16), Operation.OpType.CAST);
        assertNativeExecutable(f32("contiguous_input").contiguous(), Operation.OpType.CONTIGUOUS);
        assertNativeExecutable(f32("compare_left").greaterThan(f32("compare_right")), Operation.OpType.GT);

        PreparedNativeCpuPlan reshape = nativePlan(f32("reshape_input").reshape(4).relu(), Operation.OpType.RESHAPE, nativeRuntime());
        assertEquals(PreparedNativeCpuRoute.VIEW_ALIAS, reshape.route());
        assertEquals(PreparedNativeCpuInputPolicy.ALL_NATIVE, reshape.inputPolicy());
        assertEquals(NativeCpuKernelPerformanceStatus.VIEW_ONLY, reshape.coverageEntry().status());
    }

    @Test
    void cpuNativeWherePreparesConditionCpuValuesWithNativeBranches() {
        Tensor condition = new Tensor(new byte[]{1, 0, 1, 0}, new int[]{2, 2}, null, "where_condition", DataType.BOOL);
        Tensor out = Tensor.where(condition, f32("where_true"), f32("where_false"));

        PreparedNativeCpuPlan plan = nativePlan(out, Operation.OpType.WHERE, nativeRuntime());

        assertEquals(PreparedNativeCpuRoute.CONDITION_ARRAY_INPUT_NATIVE_OUTPUT, plan.route());
        assertEquals(PreparedNativeCpuInputPolicy.CONDITION_CPU_VALUES_NATIVE, plan.inputPolicy());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, plan.coverageEntry().status());
        assertEquals("", plan.fallbackReason());
    }

    @Test
    void cpuNativeUnsupportedAndIneligibleRoutesStayFallbackOnlyWithStableReasons() {
        PreparedNativeCpuPlan unsupported = nativePlan(f32("erf_input").erf(), Operation.OpType.ERF, nativeRuntime());
        assertEquals(PreparedNativeCpuRoute.FALLBACK_ONLY, unsupported.route());
        assertEquals(PreparedNativeCpuInputPolicy.ALL_CPU, unsupported.inputPolicy());
        assertEquals("native-kernel-unsupported:erf", unsupported.fallbackReason());

        PreparedNativeCpuPlan unsupportedCast = nativePlan(f32("cast_f64_input").cast(DataType.FLOAT64), Operation.OpType.CAST, nativeRuntime());
        assertEquals(PreparedNativeCpuRoute.FALLBACK_ONLY, unsupportedCast.route());
        assertEquals("native-kernel-ineligible:cast-dtype", unsupportedCast.fallbackReason());

        Tensor columnBias = new Tensor(new float[]{1f, -1f}, new int[]{2, 1}, null, "column_bias", DataType.FLOAT32);
        PreparedNativeCpuPlan broadcast = nativePlan(f32("broadcast_left").add(columnBias), Operation.OpType.ADD, nativeRuntime());
        assertEquals(PreparedNativeCpuRoute.FALLBACK_ONLY, broadcast.route());
        assertEquals("native-kernel-ineligible:add-broadcast", broadcast.fallbackReason());

        Tensor strided = f32("strided_contiguous_input").transpose().contiguous();
        PreparedNativeCpuPlan stridedCopy = nativePlan(strided, Operation.OpType.CONTIGUOUS, nativeRuntime());
        assertEquals(PreparedNativeCpuRoute.FALLBACK_ONLY, stridedCopy.route());
        assertEquals("native-kernel-ineligible:contiguous-strided", stridedCopy.fallbackReason());
    }

    private static void assertNativeExecutable(Tensor out, Operation.OpType opType) {
        PreparedNativeCpuPlan plan = nativePlan(out, opType, nativeRuntime());
        assertEquals(PreparedNativeCpuRoute.NATIVE_EXECUTABLE, plan.route(), opType.name());
        assertEquals(PreparedNativeCpuInputPolicy.ALL_NATIVE, plan.inputPolicy(), opType.name());
        assertNotNull(plan.coverageEntry(), opType.name());
        assertTrue(plan.coverageEntry().nativeSupported(), opType.name());
        assertEquals("", plan.fallbackReason(), opType.name());
    }

    private static PreparedNativeCpuPlan nativePlan(Tensor out, Operation.OpType opType, RuntimeConfig runtimeConfig) {
        PreparedNativeCpuPlan plan = step(out, opType, runtimeConfig).metadata().cpuPlan().nativeCpuPlan();
        assertNotNull(plan, opType.name());
        return plan;
    }

    private static PreparedNodeExecution step(Tensor out, Operation.OpType opType, RuntimeConfig runtimeConfig) {
        return CompiledGraph.compile(out, compileConfig())
                .prepare(runtimeConfig)
                .forwardSteps().stream()
                .filter(step -> step.compiledNode().operation() != null)
                .filter(step -> step.compiledNode().operation().opType() == opType)
                .findFirst()
                .orElseThrow();
    }

    private static RuntimeConfig nativeRuntime() {
        return runtime(CpuStorageProfile.CPU_NATIVE);
    }

    private static RuntimeConfig runtime(CpuStorageProfile storageProfile) {
        return RuntimeConfig.inferenceDefaults()
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

    private static Tensor bf16(String label) {
        return new Tensor(new short[]{
                CpuDTypeOps.toBFloat16Bits(1f),
                CpuDTypeOps.toBFloat16Bits(-2f),
                CpuDTypeOps.toBFloat16Bits(3f),
                CpuDTypeOps.toBFloat16Bits(-4f)
        }, new int[]{2, 2}, null, label, DataType.BFLOAT16);
    }
}
