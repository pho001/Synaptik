import tensor.dtype.TensorDTypeOps;
import runtime.contract.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.BFloat16TrainingPolicy;
import config.runtime.CpuStorageProfile;
import config.runtime.NativeCpuFailurePolicy;
import config.runtime.NativeCpuMemoryConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import runtime.execution.PreparedExecution;
import runtime.execution.PublicationPolicy;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import training.optimizer.AdamOptimizer;
import training.optimizer.SgdOptimizer;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TrainingOptimizerTest {
    @Test
    void trainableFlagEnablesGradButIsSeparateFromRequiresGrad() {
        Tensor parameter = new Tensor(new float[]{1.0f}, new int[]{1}, null, "parameter", DataType.FLOAT32);

        parameter.setTrainableParameter(true);

        assertTrue(parameter.getRequiresGrad());
        assertTrue(parameter.isTrainableParameter());

        parameter.setTrainableParameter(false);

        assertTrue(parameter.getRequiresGrad());
        assertEquals(false, parameter.isTrainableParameter());
    }

    @Test
    void sgdUpdatesOnlyTrainableParametersAndDoesNotPublishAllGradients() {
        Tensor w = new Tensor(new float[]{1.0f, 2.0f}, new int[]{2}, null, "w", DataType.FLOAT32);
        Tensor x = new Tensor(new float[]{2.0f, 3.0f}, new int[]{2}, null, "x", DataType.FLOAT32);
        w.setTrainableParameter(true);
        x.setRequiresGrad(true);
        Tensor loss = w.mul(x).sum();
        CompiledGraph graph = CompiledGraph.compile(loss, CompileConfig.training());
        PreparedExecution prepared = graph.prepare(RuntimeConfig.trainingDefaults());
        SgdOptimizer optimizer = new SgdOptimizer(0.1f);

        prepared.executeOptimizerStep(optimizer);

        assertArrayEquals(new float[]{0.8f, 1.7f}, w.toFloat32ArrayCopy(), 1.0e-6f);
        assertArrayEquals(new float[]{2.0f, 3.0f}, x.toFloat32ArrayCopy(), 0.0f);
        assertNull(w.getGradient());
        assertNull(x.getGradient());
    }

    @Test
    void nativeF32SgdMatchesArrayBaselineAndPublishesOnlyOnSync() {
        Tensor baselineW = new Tensor(new float[]{1.0f, 2.0f}, new int[]{2}, null, "baseline_w", DataType.FLOAT32);
        Tensor baselineX = new Tensor(new float[]{2.0f, 3.0f}, new int[]{2}, null, "baseline_x", DataType.FLOAT32);
        baselineW.setTrainableParameter(true);
        baselineX.setRequiresGrad(true);
        Tensor baselineLoss = baselineW.mul(baselineX).sum();
        CompiledGraph.compile(baselineLoss, CompileConfig.training())
                .prepare(RuntimeConfig.trainingDefaults())
                .executeOptimizerStep(new SgdOptimizer(0.1f), PublicationPolicy.OUTPUT_ONLY);

        Tensor nativeW = new Tensor(new float[]{1.0f, 2.0f}, new int[]{2}, null, "native_w", DataType.FLOAT32);
        Tensor nativeX = new Tensor(new float[]{2.0f, 3.0f}, new int[]{2}, null, "native_x", DataType.FLOAT32);
        nativeW.setTrainableParameter(true);
        nativeX.setRequiresGrad(true);
        Tensor nativeLoss = nativeW.mul(nativeX).sum();
        PreparedExecution prepared = CompiledGraph.compile(nativeLoss, CompileConfig.training())
                .prepare(nativeTrainingRuntime());
        SgdOptimizer optimizer = new SgdOptimizer(0.1f);

        var trace = prepared.executeOptimizerStepTraced(optimizer, PublicationPolicy.OUTPUT_ONLY);

        assertArrayEquals(new float[]{1.0f, 2.0f}, nativeW.toFloat32ArrayCopy(), 0.0f);
        assertNull(nativeW.getGradient());
        assertNull(nativeX.getGradient());
        assertEquals(1, trace.nativeOptimizers().size());
        assertEquals("CPU_NATIVE", trace.nativeOptimizers().getFirst().route());
        assertEquals("FLOAT32", trace.nativeOptimizers().getFirst().dataType());
        assertEquals("OUTPUT_ONLY", trace.nativeOptimizers().getFirst().publicationPolicy());
        assertEquals("SKIPPED", trace.nativeOptimizers().getFirst().gradientPublication());
        assertEquals("NONE", trace.nativeOptimizers().getFirst().optimizerStateStorage());
        assertEquals("", trace.nativeOptimizers().getFirst().bf16TrainingPolicy());
        assertEquals("FALLBACK_TO_ARRAY", trace.nativeOptimizers().getFirst().nativeCpuFailurePolicy());
        assertEquals("CPU_ARRAY", trace.nativeOptimizers().getFirst().parameterResidencyBefore());
        assertEquals("CPU_NATIVE", trace.nativeOptimizers().getFirst().parameterResidencyAfter());
        assertEquals("CPU_NATIVE", trace.nativeOptimizers().getFirst().gradientResidencyAfter());
        assertEquals("publication-policy-output-only", trace.nativeOptimizers().getFirst().publicationSkippedReason());

        optimizer.syncParametersToCpu();

        assertArrayEquals(baselineW.toFloat32ArrayCopy(), nativeW.toFloat32ArrayCopy(), 1.0e-6f);
    }

    @Test
    void nativeF32SgdReusesOptimizerOwnedParameterAcrossSteps() {
        Tensor w = new Tensor(new float[]{1.0f, 2.0f}, new int[]{2}, null, "w", DataType.FLOAT32);
        Tensor x = new Tensor(new float[]{2.0f, 3.0f}, new int[]{2}, null, "x", DataType.FLOAT32);
        w.setTrainableParameter(true);
        x.setRequiresGrad(true);
        Tensor loss = w.mul(x).sum();
        PreparedExecution prepared = CompiledGraph.compile(loss, CompileConfig.training())
                .prepare(nativeTrainingRuntime());
        SgdOptimizer optimizer = new SgdOptimizer(0.1f);

        var first = prepared.executeOptimizerStepTraced(optimizer, PublicationPolicy.OUTPUT_ONLY);
        var second = prepared.executeOptimizerStepTraced(optimizer, PublicationPolicy.OUTPUT_ONLY);

        assertEquals("CPU_NATIVE", first.nativeOptimizers().getFirst().route());
        assertEquals("CPU_NATIVE", second.nativeOptimizers().getFirst().route());
        assertArrayEquals(new float[]{0.8f, 1.7f}, w.toFloat32ArrayCopy(), 1.0e-6f);

        optimizer.syncParametersToCpu();

        assertArrayEquals(new float[]{0.6f, 1.4f}, w.toFloat32ArrayCopy(), 1.0e-6f);
        assertNull(w.getGradient());
        assertNull(x.getGradient());
    }

    @Test
    void nativeSgdTraceReportsArrayFallbackForUnsupportedDType() {
        Tensor w = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "w", DataType.FLOAT64);
        Tensor x = new Tensor(new double[]{2.0, 3.0}, new int[]{2}, null, "x", DataType.FLOAT64);
        w.setTrainableParameter(true);
        x.setRequiresGrad(true);
        Tensor loss = w.mul(x).sum();
        PreparedExecution prepared = CompiledGraph.compile(loss, CompileConfig.training())
                .prepare(nativeTrainingRuntime());

        var trace = prepared.executeOptimizerStepTraced(new SgdOptimizer(0.1f), PublicationPolicy.OUTPUT_ONLY);

        assertEquals(1, trace.nativeOptimizers().size());
        assertEquals("CPU_ARRAY", trace.nativeOptimizers().getFirst().route());
        assertTrue(trace.nativeOptimizers().getFirst().fallbackReason().contains("dtype-FLOAT64"));
        assertArrayEquals(new double[]{0.8, 1.7}, w.toFloat64ArrayCopy(), 1.0e-6);
        assertNull(w.getGradient());
        assertNull(x.getGradient());
    }

    @Test
    void requireNativeF32SgdUsesNativeRoute() {
        Tensor w = new Tensor(new float[]{1.0f, 2.0f}, new int[]{2}, null, "w", DataType.FLOAT32);
        Tensor x = new Tensor(new float[]{2.0f, 3.0f}, new int[]{2}, null, "x", DataType.FLOAT32);
        w.setTrainableParameter(true);
        x.setRequiresGrad(true);
        Tensor loss = w.mul(x);
        PreparedExecution prepared = CompiledGraph.compile(loss, CompileConfig.training())
                .prepare(nativeTrainingRuntime(NativeCpuFailurePolicy.REQUIRE_NATIVE));
        SgdOptimizer optimizer = new SgdOptimizer(0.1f);

        var trace = prepared.executeOptimizerStepTraced(optimizer, PublicationPolicy.OUTPUT_ONLY);

        assertEquals("CPU_NATIVE", trace.nativeOptimizers().getFirst().route());
        assertEquals("REQUIRE_NATIVE", trace.nativeOptimizers().getFirst().nativeCpuFailurePolicy());
        assertEquals("CPU_NATIVE", trace.nativeOptimizers().getFirst().parameterResidencyAfter());
        assertEquals("CPU_NATIVE", trace.nativeOptimizers().getFirst().gradientResidencyAfter());

        optimizer.syncParametersToCpu();

        assertArrayEquals(new float[]{0.8f, 1.7f}, w.toFloat32ArrayCopy(), 1.0e-6f);
    }

    @Test
    void requireNativeSgdRejectsUnsupportedDTypes() {
        PreparedExecution f64 = CompiledGraph.compile(float64TrainingLoss(), CompileConfig.training())
                .prepare(nativeTrainingRuntime(NativeCpuFailurePolicy.REQUIRE_NATIVE));
        IllegalStateException f64Failure = assertThrows(
                IllegalStateException.class,
                () -> f64.executeOptimizerStepTraced(new SgdOptimizer(0.1f), PublicationPolicy.OUTPUT_ONLY)
        );
        assertTrue(f64Failure.getMessage().contains("SgdOptimizer"));
        assertTrue(f64Failure.getMessage().contains("dtype-FLOAT64"));

        Tensor w = bf16Tensor(new float[]{1.0f, 2.0f}, "w");
        Tensor x = bf16Tensor(new float[]{2.0f, 3.0f}, "x");
        w.setTrainableParameter(true);
        x.setRequiresGrad(true);
        PreparedExecution bf16 = CompiledGraph.compile(w.mul(x), CompileConfig.training())
                .prepare(nativeTrainingRuntime(NativeCpuFailurePolicy.REQUIRE_NATIVE));
        IllegalStateException bf16Failure = assertThrows(
                IllegalStateException.class,
                () -> bf16.executeOptimizerStepTraced(new SgdOptimizer(0.1f), PublicationPolicy.OUTPUT_ONLY)
        );
        assertTrue(bf16Failure.getMessage().contains("SgdOptimizer"));
        assertTrue(bf16Failure.getMessage().contains("bf16-policy-ACTIVATIONS_ONLY"));
    }

    @Test
    void nativeSgdTraceReportsBf16ConservativeFallbackPolicy() {
        Tensor w = bf16Tensor(new float[]{1.0f, 2.0f}, "w");
        Tensor x = bf16Tensor(new float[]{2.0f, 3.0f}, "x");
        w.setTrainableParameter(true);
        x.setRequiresGrad(true);
        Tensor loss = w.mul(x).sum();
        PreparedExecution prepared = CompiledGraph.compile(loss, CompileConfig.training())
                .prepare(nativeTrainingRuntime());

        var trace = prepared.executeOptimizerStepTraced(new SgdOptimizer(0.1f), PublicationPolicy.OUTPUT_ONLY);

        assertEquals(1, trace.nativeOptimizers().size());
        assertEquals("CPU_ARRAY", trace.nativeOptimizers().getFirst().route());
        assertEquals("BFLOAT16", trace.nativeOptimizers().getFirst().dataType());
        assertEquals("ACTIVATIONS_ONLY", trace.nativeOptimizers().getFirst().bf16TrainingPolicy());
        assertEquals("SKIPPED", trace.nativeOptimizers().getFirst().gradientPublication());
        assertTrue(trace.nativeOptimizers().getFirst().fallbackReason().contains("bf16-policy-ACTIVATIONS_ONLY"));
        assertNull(w.getGradient());
        assertNull(x.getGradient());
    }

    @Test
    void nativeSgdSupportsExplicitExperimentalBf16ParameterPolicy() {
        Tensor w = bf16Tensor(new float[]{1.0f, 2.0f}, "w");
        Tensor x = bf16Tensor(new float[]{2.0f, 3.0f}, "x");
        w.setTrainableParameter(true);
        x.setRequiresGrad(true);
        Tensor loss = w.mul(x).sum();
        PreparedExecution prepared = CompiledGraph.compile(loss, CompileConfig.training())
                .prepare(nativeTrainingRuntime()
                        .withBFloat16TrainingPolicy(BFloat16TrainingPolicy.PARAMS_BF16_EXPERIMENTAL));
        SgdOptimizer optimizer = new SgdOptimizer(0.1f);

        var trace = prepared.executeOptimizerStepTraced(optimizer, PublicationPolicy.OUTPUT_ONLY);

        assertEquals("CPU_NATIVE", trace.nativeOptimizers().getFirst().route());
        assertEquals("BFLOAT16", trace.nativeOptimizers().getFirst().dataType());
        assertEquals("PARAMS_BF16_EXPERIMENTAL", trace.nativeOptimizers().getFirst().bf16TrainingPolicy());
        assertEquals("CPU_NATIVE", trace.nativeOptimizers().getFirst().parameterResidencyAfter());
        assertEquals("CPU_NATIVE", trace.nativeOptimizers().getFirst().gradientResidencyAfter());
        assertArrayEquals(new double[]{1.0, 2.0}, w.toDoubleArrayCopy(), 0.0);

        optimizer.syncParametersToCpu();

        assertArrayEquals(
                new double[]{
                        TensorDTypeOps.fromBFloat16Bits(TensorDTypeOps.toBFloat16Bits(0.8f)),
                        TensorDTypeOps.fromBFloat16Bits(TensorDTypeOps.toBFloat16Bits(1.7f))
                },
                w.toDoubleArrayCopy(),
                0.0
        );
    }

    @Test
    void nativeSgdReportsF32MasterBf16PolicyAsNotImplementedWithoutExperimentalUpdate() {
        Tensor w = bf16Tensor(new float[]{1.0f, 2.0f}, "w");
        Tensor x = bf16Tensor(new float[]{2.0f, 3.0f}, "x");
        w.setTrainableParameter(true);
        x.setRequiresGrad(true);
        Tensor loss = w.mul(x).sum();
        PreparedExecution prepared = CompiledGraph.compile(loss, CompileConfig.training())
                .prepare(nativeTrainingRuntime()
                        .withBFloat16TrainingPolicy(BFloat16TrainingPolicy.PARAMS_WITH_F32_MASTER));

        var trace = prepared.executeOptimizerStepTraced(new SgdOptimizer(0.1f), PublicationPolicy.OUTPUT_ONLY);

        assertEquals("CPU_ARRAY", trace.nativeOptimizers().getFirst().route());
        assertEquals("PARAMS_WITH_F32_MASTER", trace.nativeOptimizers().getFirst().bf16TrainingPolicy());
        assertTrue(trace.nativeOptimizers().getFirst().fallbackReason().contains("bf16-master-not-implemented"));
    }

    @Test
    void adamUpdatesOnlyTrainableParameters() {
        Tensor w = new Tensor(new float[]{1.0f, -2.0f}, new int[]{2}, null, "w", DataType.FLOAT32);
        Tensor x = new Tensor(new float[]{2.0f, -3.0f}, new int[]{2}, null, "x", DataType.FLOAT32);
        w.setTrainableParameter(true);
        x.setRequiresGrad(true);
        Tensor loss = w.mul(x).sum();
        CompiledGraph graph = CompiledGraph.compile(loss, CompileConfig.training());
        AdamOptimizer optimizer = new AdamOptimizer(0.1f);

        graph.prepare(RuntimeConfig.trainingDefaults()).executeOptimizerStep(optimizer);

        assertArrayEquals(new float[]{0.9f, -1.9f}, w.toFloat32ArrayCopy(), 1.0e-5f);
        assertArrayEquals(new float[]{2.0f, -3.0f}, x.toFloat32ArrayCopy(), 0.0f);
        assertNull(w.getGradient());
        assertNull(x.getGradient());
    }

    @Test
    void nativeF32AdamMatchesArrayBaselineAndPublishesOnlyOnSync() {
        Tensor baselineW = new Tensor(new float[]{1.0f, 2.0f}, new int[]{2}, null, "baseline_w", DataType.FLOAT32);
        Tensor baselineX = new Tensor(new float[]{2.0f, 3.0f}, new int[]{2}, null, "baseline_x", DataType.FLOAT32);
        baselineW.setTrainableParameter(true);
        baselineX.setRequiresGrad(true);
        Tensor baselineLoss = baselineW.mul(baselineX).sum();
        CompiledGraph.compile(baselineLoss, CompileConfig.training())
                .prepare(RuntimeConfig.trainingDefaults())
                .executeOptimizerStep(new AdamOptimizer(0.1f), PublicationPolicy.OUTPUT_ONLY);

        Tensor nativeW = new Tensor(new float[]{1.0f, 2.0f}, new int[]{2}, null, "native_w", DataType.FLOAT32);
        Tensor nativeX = new Tensor(new float[]{2.0f, 3.0f}, new int[]{2}, null, "native_x", DataType.FLOAT32);
        nativeW.setTrainableParameter(true);
        nativeX.setRequiresGrad(true);
        Tensor nativeLoss = nativeW.mul(nativeX).sum();
        PreparedExecution prepared = CompiledGraph.compile(nativeLoss, CompileConfig.training())
                .prepare(nativeTrainingRuntime());
        AdamOptimizer optimizer = new AdamOptimizer(0.1f);

        var trace = prepared.executeOptimizerStepTraced(optimizer, PublicationPolicy.OUTPUT_ONLY);

        assertArrayEquals(new float[]{1.0f, 2.0f}, nativeW.toFloat32ArrayCopy(), 0.0f);
        assertNull(nativeW.getGradient());
        assertNull(nativeX.getGradient());
        assertEquals(1, trace.nativeOptimizers().size());
        assertEquals("AdamOptimizer", trace.nativeOptimizers().getFirst().optimizer());
        assertEquals("CPU_NATIVE", trace.nativeOptimizers().getFirst().route());
        assertEquals("FLOAT32", trace.nativeOptimizers().getFirst().dataType());
        assertEquals("OUTPUT_ONLY", trace.nativeOptimizers().getFirst().publicationPolicy());
        assertEquals("SKIPPED", trace.nativeOptimizers().getFirst().gradientPublication());
        assertEquals("CPU_NATIVE", trace.nativeOptimizers().getFirst().optimizerStateStorage());
        assertEquals("FALLBACK_TO_ARRAY", trace.nativeOptimizers().getFirst().nativeCpuFailurePolicy());
        assertEquals("CPU_ARRAY", trace.nativeOptimizers().getFirst().parameterResidencyBefore());
        assertEquals("CPU_NATIVE", trace.nativeOptimizers().getFirst().parameterResidencyAfter());
        assertEquals("CPU_NATIVE", trace.nativeOptimizers().getFirst().gradientResidencyAfter());

        optimizer.syncParametersToCpu();

        assertArrayEquals(baselineW.toFloat32ArrayCopy(), nativeW.toFloat32ArrayCopy(), 1.0e-6f);
    }

    @Test
    void nativeF32AdamPreservesOptimizerStateAcrossSteps() {
        Tensor baselineW = new Tensor(new float[]{1.0f, 2.0f}, new int[]{2}, null, "baseline_w", DataType.FLOAT32);
        Tensor baselineX = new Tensor(new float[]{2.0f, 3.0f}, new int[]{2}, null, "baseline_x", DataType.FLOAT32);
        baselineW.setTrainableParameter(true);
        baselineX.setRequiresGrad(true);
        Tensor baselineLoss = baselineW.mul(baselineX).sum();
        PreparedExecution baselinePrepared = CompiledGraph.compile(baselineLoss, CompileConfig.training())
                .prepare(RuntimeConfig.trainingDefaults());
        AdamOptimizer baselineOptimizer = new AdamOptimizer(0.1f);
        baselinePrepared.executeOptimizerStep(baselineOptimizer, PublicationPolicy.OUTPUT_ONLY);
        baselinePrepared.executeOptimizerStep(baselineOptimizer, PublicationPolicy.OUTPUT_ONLY);

        Tensor nativeW = new Tensor(new float[]{1.0f, 2.0f}, new int[]{2}, null, "native_w", DataType.FLOAT32);
        Tensor nativeX = new Tensor(new float[]{2.0f, 3.0f}, new int[]{2}, null, "native_x", DataType.FLOAT32);
        nativeW.setTrainableParameter(true);
        nativeX.setRequiresGrad(true);
        Tensor nativeLoss = nativeW.mul(nativeX).sum();
        PreparedExecution nativePrepared = CompiledGraph.compile(nativeLoss, CompileConfig.training())
                .prepare(nativeTrainingRuntime());
        AdamOptimizer nativeOptimizer = new AdamOptimizer(0.1f);

        var first = nativePrepared.executeOptimizerStepTraced(nativeOptimizer, PublicationPolicy.OUTPUT_ONLY);
        var second = nativePrepared.executeOptimizerStepTraced(nativeOptimizer, PublicationPolicy.OUTPUT_ONLY);

        assertEquals("CPU_NATIVE", first.nativeOptimizers().getFirst().route());
        assertEquals("CPU_NATIVE", second.nativeOptimizers().getFirst().route());
        assertNull(nativeW.getGradient());
        assertNull(nativeX.getGradient());

        nativeOptimizer.syncParametersToCpu();

        assertArrayEquals(baselineW.toFloat32ArrayCopy(), nativeW.toFloat32ArrayCopy(), 1.0e-6f);
    }

    @Test
    void nativeAdamTraceReportsArrayFallbackForUnsupportedDType() {
        Tensor w = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "w", DataType.FLOAT64);
        Tensor x = new Tensor(new double[]{2.0, 3.0}, new int[]{2}, null, "x", DataType.FLOAT64);
        w.setTrainableParameter(true);
        x.setRequiresGrad(true);
        Tensor loss = w.mul(x).sum();
        PreparedExecution prepared = CompiledGraph.compile(loss, CompileConfig.training())
                .prepare(nativeTrainingRuntime());

        var trace = prepared.executeOptimizerStepTraced(new AdamOptimizer(0.1f), PublicationPolicy.OUTPUT_ONLY);

        assertEquals(1, trace.nativeOptimizers().size());
        assertEquals("AdamOptimizer", trace.nativeOptimizers().getFirst().optimizer());
        assertEquals("CPU_ARRAY", trace.nativeOptimizers().getFirst().route());
        assertTrue(trace.nativeOptimizers().getFirst().fallbackReason().contains("dtype-FLOAT64"));
        assertArrayEquals(new double[]{0.9, 1.9}, w.toFloat64ArrayCopy(), 1.0e-5);
        assertNull(w.getGradient());
        assertNull(x.getGradient());
    }

    @Test
    void requireNativeF32AdamUsesNativeRoute() {
        Tensor w = new Tensor(new float[]{1.0f, 2.0f}, new int[]{2}, null, "w", DataType.FLOAT32);
        Tensor x = new Tensor(new float[]{2.0f, 3.0f}, new int[]{2}, null, "x", DataType.FLOAT32);
        w.setTrainableParameter(true);
        x.setRequiresGrad(true);
        Tensor loss = w.mul(x);
        PreparedExecution prepared = CompiledGraph.compile(loss, CompileConfig.training())
                .prepare(nativeTrainingRuntime(NativeCpuFailurePolicy.REQUIRE_NATIVE));
        AdamOptimizer optimizer = new AdamOptimizer(0.1f);

        var trace = prepared.executeOptimizerStepTraced(optimizer, PublicationPolicy.OUTPUT_ONLY);

        assertEquals("CPU_NATIVE", trace.nativeOptimizers().getFirst().route());
        assertEquals("REQUIRE_NATIVE", trace.nativeOptimizers().getFirst().nativeCpuFailurePolicy());
        assertEquals("CPU_NATIVE", trace.nativeOptimizers().getFirst().optimizerStateStorage());
        assertEquals("CPU_NATIVE", trace.nativeOptimizers().getFirst().parameterResidencyAfter());
        assertEquals("CPU_NATIVE", trace.nativeOptimizers().getFirst().gradientResidencyAfter());

        optimizer.syncParametersToCpu();

        assertArrayEquals(new float[]{0.9f, 1.9f}, w.toFloat32ArrayCopy(), 1.0e-5f);
    }

    @Test
    void requireNativeAdamRejectsUnsupportedDTypes() {
        PreparedExecution f64 = CompiledGraph.compile(float64TrainingLoss(), CompileConfig.training())
                .prepare(nativeTrainingRuntime(NativeCpuFailurePolicy.REQUIRE_NATIVE));
        IllegalStateException f64Failure = assertThrows(
                IllegalStateException.class,
                () -> f64.executeOptimizerStepTraced(new AdamOptimizer(0.1f), PublicationPolicy.OUTPUT_ONLY)
        );
        assertTrue(f64Failure.getMessage().contains("AdamOptimizer"));
        assertTrue(f64Failure.getMessage().contains("dtype-FLOAT64"));

        Tensor w = bf16Tensor(new float[]{1.0f, 2.0f}, "w");
        Tensor x = bf16Tensor(new float[]{2.0f, 3.0f}, "x");
        w.setTrainableParameter(true);
        x.setRequiresGrad(true);
        PreparedExecution bf16 = CompiledGraph.compile(w.mul(x), CompileConfig.training())
                .prepare(nativeTrainingRuntime(NativeCpuFailurePolicy.REQUIRE_NATIVE));
        IllegalStateException bf16Failure = assertThrows(
                IllegalStateException.class,
                () -> bf16.executeOptimizerStepTraced(new AdamOptimizer(0.1f), PublicationPolicy.OUTPUT_ONLY)
        );
        assertTrue(bf16Failure.getMessage().contains("AdamOptimizer"));
        assertTrue(bf16Failure.getMessage().contains("bf16-policy-ACTIVATIONS_ONLY"));
    }

    @Test
    void nativeAdamTraceReportsBf16ConservativeFallbackPolicy() {
        Tensor w = bf16Tensor(new float[]{1.0f, 2.0f}, "w");
        Tensor x = bf16Tensor(new float[]{2.0f, 3.0f}, "x");
        w.setTrainableParameter(true);
        x.setRequiresGrad(true);
        Tensor loss = w.mul(x).sum();
        PreparedExecution prepared = CompiledGraph.compile(loss, CompileConfig.training())
                .prepare(nativeTrainingRuntime());

        var trace = prepared.executeOptimizerStepTraced(new AdamOptimizer(0.1f), PublicationPolicy.OUTPUT_ONLY);

        assertEquals(1, trace.nativeOptimizers().size());
        assertEquals("AdamOptimizer", trace.nativeOptimizers().getFirst().optimizer());
        assertEquals("CPU_ARRAY", trace.nativeOptimizers().getFirst().route());
        assertEquals("BFLOAT16", trace.nativeOptimizers().getFirst().dataType());
        assertEquals("CPU_ARRAY", trace.nativeOptimizers().getFirst().optimizerStateStorage());
        assertEquals("ACTIVATIONS_ONLY", trace.nativeOptimizers().getFirst().bf16TrainingPolicy());
        assertTrue(trace.nativeOptimizers().getFirst().fallbackReason().contains("bf16-policy-ACTIVATIONS_ONLY"));
        assertNull(w.getGradient());
        assertNull(x.getGradient());
    }

    @Test
    void nativeAdamReportsBf16ParameterPoliciesAsNotImplemented() {
        Tensor f32MasterW = bf16Tensor(new float[]{1.0f, 2.0f}, "f32MasterW");
        Tensor f32MasterX = bf16Tensor(new float[]{2.0f, 3.0f}, "f32MasterX");
        f32MasterW.setTrainableParameter(true);
        f32MasterX.setRequiresGrad(true);
        PreparedExecution f32MasterPrepared = CompiledGraph.compile(f32MasterW.mul(f32MasterX).sum(), CompileConfig.training())
                .prepare(nativeTrainingRuntime()
                        .withBFloat16TrainingPolicy(BFloat16TrainingPolicy.PARAMS_WITH_F32_MASTER));

        var f32MasterTrace = f32MasterPrepared.executeOptimizerStepTraced(new AdamOptimizer(0.1f), PublicationPolicy.OUTPUT_ONLY);

        assertEquals("CPU_ARRAY", f32MasterTrace.nativeOptimizers().getFirst().route());
        assertEquals("PARAMS_WITH_F32_MASTER", f32MasterTrace.nativeOptimizers().getFirst().bf16TrainingPolicy());
        assertTrue(f32MasterTrace.nativeOptimizers().getFirst().fallbackReason().contains("bf16-master-not-implemented"));

        Tensor experimentalW = bf16Tensor(new float[]{1.0f, 2.0f}, "experimentalW");
        Tensor experimentalX = bf16Tensor(new float[]{2.0f, 3.0f}, "experimentalX");
        experimentalW.setTrainableParameter(true);
        experimentalX.setRequiresGrad(true);
        PreparedExecution experimentalPrepared = CompiledGraph.compile(experimentalW.mul(experimentalX).sum(), CompileConfig.training())
                .prepare(nativeTrainingRuntime()
                        .withBFloat16TrainingPolicy(BFloat16TrainingPolicy.PARAMS_BF16_EXPERIMENTAL));

        var experimentalTrace = experimentalPrepared.executeOptimizerStepTraced(new AdamOptimizer(0.1f), PublicationPolicy.OUTPUT_ONLY);

        assertEquals("CPU_ARRAY", experimentalTrace.nativeOptimizers().getFirst().route());
        assertEquals("PARAMS_BF16_EXPERIMENTAL", experimentalTrace.nativeOptimizers().getFirst().bf16TrainingPolicy());
        assertTrue(experimentalTrace.nativeOptimizers().getFirst().fallbackReason()
                .contains("bf16-experimental-adam-not-implemented"));
    }

    @Test
    void nativeOptimizerStepWithPublicationNoneStillUpdatesWithoutPublishing() {
        Tensor w = new Tensor(new float[]{1.0f, 2.0f}, new int[]{2}, null, "w", DataType.FLOAT32);
        Tensor x = new Tensor(new float[]{2.0f, 3.0f}, new int[]{2}, null, "x", DataType.FLOAT32);
        w.setTrainableParameter(true);
        x.setRequiresGrad(true);
        Tensor loss = w.mul(x).sum();
        PreparedExecution prepared = CompiledGraph.compile(loss, CompileConfig.training())
                .prepare(nativeTrainingRuntime());
        SgdOptimizer optimizer = new SgdOptimizer(0.1f);

        var trace = prepared.executeOptimizerStepTraced(optimizer, PublicationPolicy.NONE);

        assertArrayEquals(new float[]{1.0f, 2.0f}, w.toFloat32ArrayCopy(), 0.0f);
        assertArrayEquals(new double[]{0.0}, loss.toDoubleArrayCopy(), 0.0);
        assertEquals("NONE", trace.nativeOptimizers().getFirst().publicationPolicy());
        assertEquals("NONE", trace.nativeOptimizers().getFirst().gradientPublication());
        assertNull(w.getGradient());
        assertNull(x.getGradient());

        optimizer.syncParametersToCpu();

        assertArrayEquals(new float[]{0.8f, 1.7f}, w.toFloat32ArrayCopy(), 1.0e-6f);
    }

    @Test
    void explicitOptimizerParameterListCannotUpdateNonTrainableGradientTensors() {
        Tensor w = new Tensor(new float[]{1.0f, 2.0f}, new int[]{2}, null, "w", DataType.FLOAT32);
        Tensor x = new Tensor(new float[]{2.0f, 3.0f}, new int[]{2}, null, "x", DataType.FLOAT32);
        w.setTrainableParameter(true);
        x.setRequiresGrad(true);
        Tensor loss = w.mul(x).sum();
        CompiledGraph graph = CompiledGraph.compile(loss, CompileConfig.training());
        SgdOptimizer optimizer = new SgdOptimizer(List.of(x), 0.1f);

        assertEquals(List.of(w), graph.trainableParameters());

        graph.prepare(RuntimeConfig.trainingDefaults()).executeOptimizerStep(optimizer);

        assertArrayEquals(new float[]{1.0f, 2.0f}, w.toFloat32ArrayCopy(), 0.0f);
        assertArrayEquals(new float[]{2.0f, 3.0f}, x.toFloat32ArrayCopy(), 0.0f);
        assertNull(w.getGradient());
        assertNull(x.getGradient());
    }

    @Test
    void normalForwardBackwardStillPublishesGradients() {
        Tensor w = new Tensor(new float[]{1.0f, 2.0f}, new int[]{2}, null, "w", DataType.FLOAT32);
        Tensor x = new Tensor(new float[]{2.0f, 3.0f}, new int[]{2}, null, "x", DataType.FLOAT32);
        w.setTrainableParameter(true);
        x.setRequiresGrad(true);
        Tensor loss = w.mul(x).sum();

        CompiledGraph.compile(loss, CompileConfig.training())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new float[]{2.0f, 3.0f}, w.getGradient().toFloat32ArrayCopy(), 1.0e-6f);
        assertArrayEquals(new float[]{1.0f, 2.0f}, x.getGradient().toFloat32ArrayCopy(), 1.0e-6f);
    }

    private static RuntimeConfig nativeTrainingRuntime() {
        return nativeTrainingRuntime(NativeCpuFailurePolicy.FALLBACK_TO_ARRAY);
    }

    private static RuntimeConfig nativeTrainingRuntime(NativeCpuFailurePolicy failurePolicy) {
        return RuntimeConfig.trainingDefaults()
                .withCpuStorageProfile(CpuStorageProfile.CPU_NATIVE)
                .withNativeCpuFailurePolicy(failurePolicy)
                .withNativeCpuMemory(NativeCpuMemoryConfig.perPreparedExecution(4096L));
    }

    private static Tensor float64TrainingLoss() {
        Tensor w = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "w", DataType.FLOAT64);
        Tensor x = new Tensor(new double[]{2.0, 3.0}, new int[]{2}, null, "x", DataType.FLOAT64);
        w.setTrainableParameter(true);
        x.setRequiresGrad(true);
        return w.mul(x);
    }

    private static Tensor bf16Tensor(float[] values, String label) {
        short[] bits = new short[values.length];
        for (int i = 0; i < values.length; i++) {
            bits[i] = TensorDTypeOps.toBFloat16Bits(values[i]);
        }
        return new Tensor(bits, new int[]{values.length}, null, label, DataType.BFLOAT16);
    }
}
