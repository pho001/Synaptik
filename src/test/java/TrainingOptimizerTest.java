import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.CpuStorageProfile;
import config.runtime.NativeCpuFailurePolicy;
import config.runtime.NativeCpuMemoryConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.execution.PreparedExecution;
import graph.execution.PublicationPolicy;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import training.optimizer.AdamOptimizer;
import training.optimizer.SgdOptimizer;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
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

        assertArrayEquals(new float[]{0.8f, 1.7f}, w.getFloat32Data(), 1.0e-6f);
        assertArrayEquals(new float[]{2.0f, 3.0f}, x.getFloat32Data(), 0.0f);
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

        assertArrayEquals(new float[]{1.0f, 2.0f}, nativeW.getFloat32Data(), 0.0f);
        assertNull(nativeW.getGradient());
        assertNull(nativeX.getGradient());
        assertEquals(1, trace.nativeOptimizers().size());
        assertEquals("CPU_NATIVE", trace.nativeOptimizers().getFirst().route());
        assertEquals(DataType.FLOAT32, trace.nativeOptimizers().getFirst().dataType());

        optimizer.syncParametersToCpu();

        assertArrayEquals(baselineW.getFloat32Data(), nativeW.getFloat32Data(), 1.0e-6f);
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
        assertArrayEquals(new float[]{0.8f, 1.7f}, w.getFloat32Data(), 1.0e-6f);

        optimizer.syncParametersToCpu();

        assertArrayEquals(new float[]{0.6f, 1.4f}, w.getFloat32Data(), 1.0e-6f);
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
        assertArrayEquals(new double[]{0.8, 1.7}, w.getFloat64Data(), 1.0e-6);
        assertNull(w.getGradient());
        assertNull(x.getGradient());
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

        graph.executeOptimizerStep(RuntimeConfig.trainingDefaults(), optimizer);

        assertArrayEquals(new float[]{0.9f, -1.9f}, w.getFloat32Data(), 1.0e-5f);
        assertArrayEquals(new float[]{2.0f, -3.0f}, x.getFloat32Data(), 0.0f);
        assertNull(w.getGradient());
        assertNull(x.getGradient());
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

        graph.executeOptimizerStep(RuntimeConfig.trainingDefaults(), optimizer);

        assertArrayEquals(new float[]{1.0f, 2.0f}, w.getFloat32Data(), 0.0f);
        assertArrayEquals(new float[]{2.0f, 3.0f}, x.getFloat32Data(), 0.0f);
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
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new float[]{2.0f, 3.0f}, w.getGradient().getFloat32Data(), 1.0e-6f);
        assertArrayEquals(new float[]{1.0f, 2.0f}, x.getGradient().getFloat32Data(), 1.0e-6f);
    }

    private static RuntimeConfig nativeTrainingRuntime() {
        return RuntimeConfig.trainingDefaults()
                .withCpuStorageProfile(CpuStorageProfile.CPU_NATIVE)
                .withNativeCpuFailurePolicy(NativeCpuFailurePolicy.FALLBACK_TO_ARRAY)
                .withNativeCpuMemory(NativeCpuMemoryConfig.perPreparedExecution(4096L));
    }
}
