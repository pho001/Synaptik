import backend.cpu.fused.asm.FusedAsmSpecializationKind;
import backend.cpu.fused.asm.emit.FusedOperationGenerator;
import backend.cpu.fused.plan.FusedOperation;
import backend.cpu.fused.numeric.FusedStorageKind;
import backend.memory.CpuMaterializationReason;
import backend.memory.StorageResidency;
import config.backend.CpuKernelConfig;
import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.compile.GraphOptimizationConfig;
import config.runtime.CpuStorageProfile;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.execution.PreparedExecution;
import graph.execution.PublicationPolicy;
import graph.execution.plan.InputResidencyRequirement;
import graph.execution.plan.OutputResidencyEffect;
import graph.execution.trace.RunTrace;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.dtype.BFloat16Bits;

import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class CpuFusedMemorySegmentExecutionTest {
    @Test
    void f32FusedSegmentWritesNativeOutputThenGraphOutputMaterializesExplicitly() {
        Tensor a = new Tensor(new float[]{1.0f, -4.0f, 3.0f, 5.0f}, new int[]{4}, null, "f32_a");
        Tensor b = new Tensor(new float[]{2.0f, 2.0f, -8.0f, 1.0f}, new int[]{4}, null, "f32_b");
        Tensor out = a.add(b).relu().exp();

        PreparedExecution prepared = prepare(out);
        assertSegmentPrepared(prepared);

        RunTrace trace = prepared.executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(
                new double[]{Math.exp(3.0), 1.0, 1.0, Math.exp(6.0)},
                out.toDoubleArrayCopy(),
                1e-5
        );
        assertGraphOutputMaterializedFromNativeFusedOutput(prepared, trace);
    }

    @Test
    void f32ContiguousFusedSegmentUsesVectorMemorySegmentPathWithScalarTail() {
        int size = 1031;
        float[] left = new float[size];
        float[] right = new float[size];
        for (int i = 0; i < size; i++) {
            left[i] = (i % 23) - 11.0f;
            right[i] = (i % 7) * 0.25f;
        }
        Tensor a = new Tensor(left, new int[]{size}, null, "f32_vec_a");
        Tensor b = new Tensor(right, new int[]{size}, null, "f32_vec_b");
        Tensor out = a.add(b).mul(0.5f).relu();

        PreparedExecution prepared = prepare(out, vectorNativeRuntime());
        assertSegmentPrepared(prepared);
        RunTrace trace = prepared.executeTraced(ExecutionMode.FORWARD);

        double[] values = out.toDoubleArrayCopy();
        assertEquals(Math.max(0.0, (left[0] + right[0]) * 0.5), values[0], 1e-5);
        assertEquals(Math.max(0.0, (left[997] + right[997]) * 0.5), values[997], 1e-5);
        assertEquals(Math.max(0.0, (left[size - 1] + right[size - 1]) * 0.5), values[size - 1], 1e-5);
        assertVectorSegmentTrace(trace);
    }

    @Test
    void f64ContiguousFusedSegmentUsesVectorMemorySegmentPathWithScalarTail() {
        int size = 515;
        double[] left = new double[size];
        double[] right = new double[size];
        for (int i = 0; i < size; i++) {
            left[i] = (i % 19) - 9.0;
            right[i] = (i % 5) * 0.125;
        }
        Tensor a = new Tensor(left, new int[]{size}, null, "f64_vec_a");
        Tensor b = new Tensor(right, new int[]{size}, null, "f64_vec_b");
        Tensor out = a.sub(b).abs().add(b);

        PreparedExecution prepared = prepare(out, vectorNativeRuntime());
        assertSegmentPrepared(prepared);
        RunTrace trace = prepared.executeTraced(ExecutionMode.FORWARD);

        double[] values = out.toDoubleArrayCopy();
        assertEquals(Math.abs(left[0] - right[0]) + right[0], values[0], 1e-9);
        assertEquals(Math.abs(left[331] - right[331]) + right[331], values[331], 1e-9);
        assertEquals(Math.abs(left[size - 1] - right[size - 1]) + right[size - 1], values[size - 1], 1e-9);
        assertVectorSegmentTrace(trace);
    }

    @Test
    void generatedF32SegmentVectorKernelUsesMemorySegmentApiWithoutArrayBindings() {
        Tensor a = new Tensor(new float[128], new int[]{128}, null, "bytecode_a");
        Tensor b = new Tensor(new float[128], new int[]{128}, null, "bytecode_b");
        Tensor out = a.add(b).relu();

        PreparedExecution prepared = prepare(out, vectorNativeRuntime());
        var fusedStep = fusedStep(prepared);
        FusedOperation fused = (FusedOperation) fusedStep.metadata().executionOperation();
        int vectorWidth = testsupport.MetadataArtifacts.cpuPlan(fusedStep.metadata()).dispatchHints().vectorWidth();

        byte[] bytecode = FusedOperationGenerator.generate(
                "debug/test/F32SegmentVectorKernel",
                fused.getPlan(),
                fused.getNumericContract(),
                fused.getApproximationContract(),
                vectorWidth,
                FusedAsmSpecializationKind.NONE
        );
        String constantPool = new String(bytecode, StandardCharsets.ISO_8859_1);

        assertTrue(constantPool.contains("fromMemorySegment"));
        assertTrue(constantPool.contains("intoMemorySegment"));
        assertFalse(constantPool.contains("TensorInternalAccess"));
        assertFalse(constantPool.contains("float32Data"));
    }

    @Test
    void defaultAndAutoFusedExecutionStayOnJavaArrayContract() {
        for (RuntimeConfig runtime : new RuntimeConfig[]{
                RuntimeConfig.inferenceDefaults(),
                RuntimeConfig.inferenceDefaults().withCpuStorageProfile(CpuStorageProfile.AUTO)
        }) {
            Tensor a = new Tensor(new float[]{1.0f, -4.0f, 3.0f, 5.0f}, new int[]{4}, null, "array_a");
            Tensor b = new Tensor(new float[]{2.0f, 2.0f, -8.0f, 1.0f}, new int[]{4}, null, "array_b");
            Tensor out = a.add(b).relu();

            PreparedExecution prepared = prepare(out, runtime);
            assertArrayPrepared(prepared);
            RunTrace trace = prepared.executeTraced(ExecutionMode.FORWARD, PublicationPolicy.NONE);

            assertEquals("CPU_JAVA_ARRAY", fusedTrace(trace).metadata().attributes().get("fusedInputStorageKind"));
            assertEquals("CPU_JAVA_ARRAY", fusedTrace(trace).metadata().attributes().get("fusedOutputStorageKind"));
            assertTrue(trace.cpuMaterializations().stream().noneMatch(entry ->
                            entry.detail().contains("array_to_native") || entry.detail().contains("native_to_array")),
                    () -> "array fused path should not bridge through native storage: " + trace.cpuMaterializations());
        }
    }

    @Test
    void f64FusedSegmentWritesNativeOutput() {
        Tensor a = new Tensor(new double[]{1.0, 2.0, 3.0}, new int[]{3}, null, "f64_a");
        Tensor b = new Tensor(new double[]{4.0, -8.0, 5.0}, new int[]{3}, null, "f64_b");
        Tensor out = a.add(b).relu();

        PreparedExecution prepared = prepare(out, parallelVectorNativeRuntime());
        assertSegmentPrepared(prepared);
        RunTrace trace = prepared.executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{5.0, 0.0, 8.0}, out.toDoubleArrayCopy(), 1e-9);
        assertGraphOutputMaterializedFromNativeFusedOutput(prepared, trace);
    }

    @Test
    void bf16FusedSegmentWritesNativeOutput() {
        Tensor a = bf16("bf16_a", 1.25f, -2.0f, 4.0f);
        Tensor b = bf16("bf16_b", 0.5f, 1.0f, -1.5f);
        Tensor out = a.add(b).relu();

        PreparedExecution prepared = prepare(out, vectorNativeRuntime());
        assertSegmentPrepared(prepared);
        RunTrace trace = prepared.executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{1.75, 0.0, 2.5}, out.toDoubleArrayCopy(), 1e-2);
        assertScalarOnlySegmentTrace(trace);
        assertGraphOutputMaterializedFromNativeFusedOutput(prepared, trace);
    }

    @Test
    void boolFusedSegmentWritesNativeOutput() {
        Tensor a = new Tensor(new float[]{1.0f, 5.0f, -1.0f, 7.0f}, new int[]{4}, null, "bool_a");
        Tensor b = new Tensor(new float[]{2.0f, 3.0f, -2.0f, 8.0f}, new int[]{4}, null, "bool_b");
        Tensor out = a.greaterThan(b).logicalNot();

        PreparedExecution prepared = prepare(out);
        assertSegmentPrepared(prepared);
        RunTrace trace = prepared.executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(new byte[]{1, 0, 0, 1}, out.toBoolByteArrayCopy());
        assertGraphOutputMaterializedFromNativeFusedOutput(prepared, trace);
    }

    @Test
    void segmentScalarPathHandlesOffsetInputAndBroadcast() {
        Tensor base = new Tensor(new float[]{1.0f, -2.0f, 3.0f, 4.0f, -5.0f, 6.0f}, new int[]{2, 3}, null, "base");
        Tensor bias = new Tensor(new float[]{0.5f}, new int[]{1}, null, "bias");
        Tensor out = base.select(0, 1).add(bias).relu();

        PreparedExecution prepared = prepare(out, vectorNativeRuntime());
        assertSegmentPrepared(prepared);
        RunTrace trace = prepared.executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{4.5, 0.0, 6.5}, out.toDoubleArrayCopy(), 1e-5);
        assertScalarOnlySegmentTrace(trace);
        assertGraphOutputMaterializedFromNativeFusedOutput(prepared, trace);
    }

    @Test
    void segmentParallelVectorDispatchBindsNativeSegmentsAndPublishesAfterVectorChunks() {
        int size = 140_000;
        float[] left = new float[size];
        float[] right = new float[size];
        for (int i = 0; i < size; i++) {
            left[i] = (i % 17) - 8.0f;
            right[i] = (i % 5) * 0.25f;
        }
        Tensor a = new Tensor(left, new int[]{size}, null, "parallel_a");
        Tensor b = new Tensor(right, new int[]{size}, null, "parallel_b");
        Tensor out = a.add(b).relu();

        PreparedExecution prepared = prepare(out, parallelVectorNativeRuntime());
        assertSegmentPrepared(prepared);
        RunTrace trace = prepared.executeTraced(ExecutionMode.FORWARD);

        double[] values = out.toDoubleArrayCopy();
        assertEquals(Math.max(0.0, left[0] + right[0]), values[0], 1e-5);
        assertEquals(Math.max(0.0, left[37_119] + right[37_119]), values[37_119], 1e-5);
        assertEquals(Math.max(0.0, left[size - 1] + right[size - 1]), values[size - 1], 1e-5);
        var fusedTrace = trace.steps().stream()
                .filter(step -> step.metadata().fused() != null)
                .findFirst()
                .orElseThrow();
        assertEquals("PARALLEL_VECTOR", fusedTrace.metadata().dispatch().mode());
        assertTrue(fusedTrace.metadata().dispatch().vectorWidth() > 1);
        assertEquals("NONE", fusedTrace.metadata().fused().vectorBlockReason());
        assertEquals("CPU_MEMORY_SEGMENT", fusedTrace.metadata().attributes().get("fusedInputStorageKind"));
        assertEquals("CPU_MEMORY_SEGMENT", fusedTrace.metadata().attributes().get("fusedOutputStorageKind"));
        assertGraphOutputMaterializedFromNativeFusedOutput(prepared, trace);
    }

    private static PreparedExecution prepare(Tensor out) {
        return prepare(out, nativeRuntime());
    }

    private static PreparedExecution prepare(Tensor out, RuntimeConfig runtimeConfig) {
        return CompiledGraph.compile(out, fusedOnlyConfig()).prepare(runtimeConfig);
    }

    private static void assertSegmentPrepared(PreparedExecution prepared) {
        var step = fusedStep(prepared);
        FusedOperation fused = (FusedOperation) step.metadata().executionOperation();
        assertEquals(FusedStorageKind.CPU_MEMORY_SEGMENT, fused.getNumericContract().inputStorageKind());
        assertEquals(FusedStorageKind.CPU_MEMORY_SEGMENT, fused.getNumericContract().outputStorageKind());
        assertEquals(InputResidencyRequirement.Mode.NONE, step.metadata().inputResidencyRequirement().mode());
        assertEquals(OutputResidencyEffect.Mode.NONE, step.metadata().outputResidencyEffect().mode());
    }

    private static void assertArrayPrepared(PreparedExecution prepared) {
        var step = fusedStep(prepared);
        FusedOperation fused = (FusedOperation) step.metadata().executionOperation();
        assertEquals(FusedStorageKind.CPU_JAVA_ARRAY, fused.getNumericContract().inputStorageKind());
        assertEquals(FusedStorageKind.CPU_JAVA_ARRAY, fused.getNumericContract().outputStorageKind());
        assertEquals(InputResidencyRequirement.Mode.CPU_READABLE_ALL, step.metadata().inputResidencyRequirement().mode());
        assertEquals(OutputResidencyEffect.Mode.CPU_CURRENT_PRESERVE_NATIVE, step.metadata().outputResidencyEffect().mode());
    }

    private static void assertGraphOutputMaterializedFromNativeFusedOutput(PreparedExecution prepared, RunTrace trace) {
        int fusedOutputNodeId = fusedStep(prepared).compiledNode().id();
        assertTrue(trace.cpuMaterializations().stream().anyMatch(entry ->
                        entry.nodeId() == fusedOutputNodeId
                                && (entry.reason() == CpuMaterializationReason.GRAPH_OUTPUT
                                || entry.reason() == CpuMaterializationReason.CPU_CONSUMER)
                                && entry.sourceResidency() == StorageResidency.CPU_NATIVE
                                && entry.completed()
                                && entry.detail().contains("native_to_array")),
                () -> "expected graph output publication to materialize fused CPU_NATIVE output: "
                        + trace.cpuMaterializations());
    }

    private static graph.execution.PreparedExecutionStep fusedStep(PreparedExecution prepared) {
        return prepared.forwardSteps().stream()
                .filter(step -> testsupport.MetadataArtifacts.fusedExecutable(step.metadata()) != null)
                .findFirst()
                .orElseThrow();
    }

    private static graph.execution.trace.ExecutionStepTrace fusedTrace(RunTrace trace) {
        return trace.steps().stream()
                .filter(step -> step.metadata().fused() != null)
                .findFirst()
                .orElseThrow();
    }

    private static void assertVectorSegmentTrace(RunTrace trace) {
        var fusedTrace = fusedTrace(trace);
        assertTrue(fusedTrace.metadata().dispatch().vectorWidth() > 1);
        assertEquals("NONE", fusedTrace.metadata().fused().vectorBlockReason());
        assertEquals("CPU_MEMORY_SEGMENT", fusedTrace.metadata().attributes().get("fusedInputStorageKind"));
        assertEquals("CPU_MEMORY_SEGMENT", fusedTrace.metadata().attributes().get("fusedOutputStorageKind"));
    }

    private static void assertScalarOnlySegmentTrace(RunTrace trace) {
        var fusedTrace = fusedTrace(trace);
        assertEquals(1, fusedTrace.metadata().dispatch().vectorWidth());
        assertEquals("MEMORY_SEGMENT_SCALAR_ONLY", fusedTrace.metadata().fused().vectorBlockReason());
        assertEquals("CPU_MEMORY_SEGMENT", fusedTrace.metadata().attributes().get("fusedInputStorageKind"));
        assertEquals("CPU_MEMORY_SEGMENT", fusedTrace.metadata().attributes().get("fusedOutputStorageKind"));
    }

    private static Tensor bf16(String label, float... values) {
        short[] bits = new short[values.length];
        for (int i = 0; i < values.length; i++) {
            bits[i] = BFloat16Bits.fromFloat(values[i]);
        }
        return new Tensor(bits, new int[]{values.length}, null, label, DataType.BFLOAT16);
    }

    private static RuntimeConfig nativeRuntime() {
        return RuntimeConfig.inferenceDefaults().withCpuStorageProfile(CpuStorageProfile.CPU_NATIVE);
    }

    private static RuntimeConfig vectorNativeRuntime() {
        return new RuntimeConfig(
                new CpuKernelConfig(4, 32, 32, 32, 1, Integer.MAX_VALUE),
                config.runtime.ApproximationConfig.defaults(),
                config.runtime.BlasConfig.disabled()
        ).withCpuStorageProfile(CpuStorageProfile.CPU_NATIVE);
    }

    private static RuntimeConfig parallelVectorNativeRuntime() {
        return new RuntimeConfig(
                new CpuKernelConfig(4, 32, 32, 32, 1, 1),
                config.runtime.ApproximationConfig.defaults(),
                config.runtime.BlasConfig.disabled()
        ).withCpuStorageProfile(CpuStorageProfile.CPU_NATIVE);
    }

    private static CompileConfig fusedOnlyConfig() {
        return CompileConfig.inference()
                .withGraphOptimization(GraphOptimizationConfig.noGraphOptimization());
    }
}
