import backend.contract.ComputeBackend;
import backend.cpu.CpuFusedExecutionArtifact;
import backend.cpu.fused.asm.FusedAsmSpecializationKind;
import backend.cpu.fused.asm.emit.FusedOperationGenerator;
import backend.cpu.fused.exec.FusedNativeSegmentBindings;
import backend.cpu.fused.exec.PreparedFusedExecutable;
import backend.cpu.fused.plan.FusedOperation;
import backend.cpu.fused.numeric.FusedStorageKind;
import runtime.contract.CpuMaterializationReason;
import runtime.contract.StorageResidency;
import config.backend.CpuKernelConfig;
import runtime.contract.ExecutionMode;
import runtime.execution.ExecutionContext;
import config.compile.CompileConfig;
import config.compile.GraphOptimizationConfig;
import config.runtime.CpuStorageProfile;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.execution.PreparedExecution;
import graph.execution.PreparedExecutionStep;
import graph.execution.PublicationPolicy;
import runtime.execution.PreparedStepMetadata;
import runtime.execution.InputResidencyRequirement;
import runtime.execution.OutputResidencyEffect;
import graph.execution.residency.RuntimeMemoryBinder;
import graph.execution.runner.PreparedExecutionRunner;
import runtime.execution.ExecutionState;
import trace.execution.RunTrace;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.dtype.BFloat16Bits;
import tensor.storage.NativeFloat32Storage;
import tensor.storage.NativeTensorStorage;

import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
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
    void f32SmallDirectFusedSegmentUsesScalarMemorySegmentPathBelowVectorThreshold() {
        Tensor a = new Tensor(new float[]{1.0f, -4.0f, 3.0f}, new int[]{3}, null, "f32_scalar_a");
        Tensor b = new Tensor(new float[]{2.0f, 2.0f, -8.0f}, new int[]{3}, null, "f32_scalar_b");
        Tensor out = a.add(b).relu();

        PreparedExecution prepared = prepare(out, vectorNativeRuntime());
        assertSegmentPrepared(prepared);
        RunTrace trace = prepared.executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{3.0, 0.0, 0.0}, out.toDoubleArrayCopy(), 1e-5);
        assertBelowThresholdSegmentTrace(trace);
        assertGraphOutputMaterializedFromNativeFusedOutput(prepared, trace);
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
    void f64SmallDirectFusedSegmentUsesScalarMemorySegmentPathBelowVectorThreshold() {
        Tensor a = new Tensor(new double[]{1.0, 2.0, 3.0}, new int[]{3}, null, "f64_scalar_a");
        Tensor b = new Tensor(new double[]{4.0, -8.0, 5.0}, new int[]{3}, null, "f64_scalar_b");
        Tensor out = a.add(b).relu();

        PreparedExecution prepared = prepare(out, vectorNativeRuntime());
        assertSegmentPrepared(prepared);
        RunTrace trace = prepared.executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{5.0, 0.0, 8.0}, out.toDoubleArrayCopy(), 1e-9);
        assertBelowThresholdSegmentTrace(trace);
        assertGraphOutputMaterializedFromNativeFusedOutput(prepared, trace);
    }

    @Test
    void f32ScalarBroadcastFusedSegmentUsesVectorMemorySegmentPath() {
        int size = 140_011;
        float[] left = new float[size];
        for (int i = 0; i < size; i++) {
            left[i] = (i % 29) - 14.0f;
        }
        Tensor a = new Tensor(left, new int[]{size}, null, "f32_broadcast_a");
        Tensor bias = new Tensor(new float[]{2.25f}, new int[]{1}, null, "f32_broadcast_bias");
        Tensor out = a.add(bias).relu();

        PreparedExecution prepared = prepare(out, vectorNativeRuntime());
        assertSegmentPrepared(prepared);
        RunTrace trace = prepared.executeTraced(ExecutionMode.FORWARD);

        double[] values = out.toDoubleArrayCopy();
        assertEquals(Math.max(0.0, left[0] + 2.25f), values[0], 1e-5);
        assertEquals(Math.max(0.0, left[997] + 2.25f), values[997], 1e-5);
        assertEquals(Math.max(0.0, left[size - 1] + 2.25f), values[size - 1], 1e-5);
        assertVectorSegmentTrace(trace);
        assertGraphOutputMaterializedFromNativeFusedOutput(prepared, trace);
    }

    @Test
    void f32AllZeroBroadcast2dFusedSegmentUsesVectorMemorySegmentPath() {
        int rows = 8192;
        int cols = 17;
        float[] left = new float[rows * cols];
        for (int i = 0; i < left.length; i++) {
            left[i] = (i % 31) - 15.0f;
        }
        Tensor a = new Tensor(left, new int[]{rows, cols}, null, "f32_broadcast2d_a");
        Tensor bias = new Tensor(new float[]{1.75f}, new int[]{1, 1}, null, "f32_broadcast2d_bias");
        Tensor out = a.add(bias).relu();

        PreparedExecution prepared = prepare(out, vectorNativeRuntime());
        assertSegmentPrepared(prepared);
        RunTrace trace = prepared.executeTraced(ExecutionMode.FORWARD);

        double[] values = out.toDoubleArrayCopy();
        assertEquals(Math.max(0.0, left[0] + 1.75f), values[0], 1e-5);
        assertEquals(Math.max(0.0, left[997] + 1.75f), values[997], 1e-5);
        assertEquals(Math.max(0.0, left[left.length - 1] + 1.75f), values[left.length - 1], 1e-5);
        assertVectorSegmentTrace(trace);
        assertGraphOutputMaterializedFromNativeFusedOutput(prepared, trace);
    }

    @Test
    void f64ScalarBroadcastFusedSegmentUsesVectorMemorySegmentPath() {
        int size = 140_003;
        double[] left = new double[size];
        for (int i = 0; i < size; i++) {
            left[i] = (i % 31) - 15.0;
        }
        Tensor a = new Tensor(left, new int[]{size}, null, "f64_broadcast_a");
        Tensor bias = new Tensor(new double[]{0.75}, new int[]{1}, null, "f64_broadcast_bias");
        Tensor out = a.sub(bias).abs();

        PreparedExecution prepared = prepare(out, vectorNativeRuntime());
        assertSegmentPrepared(prepared);
        RunTrace trace = prepared.executeTraced(ExecutionMode.FORWARD);

        double[] values = out.toDoubleArrayCopy();
        assertEquals(Math.abs(left[0] - 0.75), values[0], 1e-9);
        assertEquals(Math.abs(left[331] - 0.75), values[331], 1e-9);
        assertEquals(Math.abs(left[size - 1] - 0.75), values[size - 1], 1e-9);
        assertVectorSegmentTrace(trace);
        assertGraphOutputMaterializedFromNativeFusedOutput(prepared, trace);
    }

    @Test
    void f32GeneralBroadcastFusedSegmentUsesGeneratedGatherVectorPath() {
        int rows = 8192;
        int cols = 17;
        float[] rowValues = new float[rows];
        float[] colValues = new float[cols];
        for (int i = 0; i < rows; i++) {
            rowValues[i] = (i % 29) - 14.0f;
        }
        for (int i = 0; i < cols; i++) {
            colValues[i] = (i % 11) * 0.25f - 1.25f;
        }
        Tensor row = new Tensor(rowValues, new int[]{rows, 1}, null, "f32_segment_row");
        Tensor col = new Tensor(colValues, new int[]{1, cols}, null, "f32_segment_col");
        Tensor out = row.add(col).relu();

        PreparedExecution prepared = prepare(out, vectorNativeRuntime());
        assertSegmentPrepared(prepared);
        RunTrace trace = prepared.executeTraced(ExecutionMode.FORWARD);

        double[] values = out.toDoubleArrayCopy();
        assertEquals(Math.max(0.0, rowValues[0] + colValues[0]), values[0], 1e-5);
        assertEquals(Math.max(0.0, rowValues[0] + colValues[1]), values[1], 1e-5);
        assertEquals(Math.max(0.0, rowValues[1] + colValues[0]), values[cols], 1e-5);
        assertEquals(Math.max(0.0, rowValues[rows - 1] + colValues[cols - 1]), values[values.length - 1], 1e-5);
        assertVectorSegmentTrace(trace);
        assertGraphOutputMaterializedFromNativeFusedOutput(prepared, trace);
    }

    @Test
    void f64GeneralBroadcastFusedSegmentUsesGeneratedGatherVectorPath() {
        int rows = 8192;
        int cols = 23;
        double[] rowValues = new double[rows];
        double[] colValues = new double[cols];
        for (int i = 0; i < rows; i++) {
            rowValues[i] = (i % 19) - 9.0;
        }
        for (int i = 0; i < cols; i++) {
            colValues[i] = Math.cos(i * 0.15);
        }
        Tensor row = new Tensor(rowValues, new int[]{rows, 1}, null, "f64_segment_row", DataType.FLOAT64);
        Tensor col = new Tensor(colValues, new int[]{1, cols}, null, "f64_segment_col", DataType.FLOAT64);
        Tensor out = row.add(col).abs();

        PreparedExecution prepared = prepare(out, vectorNativeRuntime());
        assertSegmentPrepared(prepared);
        RunTrace trace = prepared.executeTraced(ExecutionMode.FORWARD);

        double[] values = out.toDoubleArrayCopy();
        assertEquals(Math.abs(rowValues[0] + colValues[0]), values[0], 1e-9);
        assertEquals(Math.abs(rowValues[0] + colValues[1]), values[1], 1e-9);
        assertEquals(Math.abs(rowValues[1] + colValues[0]), values[cols], 1e-9);
        assertEquals(Math.abs(rowValues[rows - 1] + colValues[cols - 1]), values[values.length - 1], 1e-9);
        assertVectorSegmentTrace(trace);
        assertGraphOutputMaterializedFromNativeFusedOutput(prepared, trace);
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
        assertNoArrayBackedSegmentBytecode(constantPool);
    }

    @Test
    void generatedF32UnsupportedFastSegmentKernelUsesScalarMemorySegmentApiWithoutArrayBindings() {
        Tensor a = new Tensor(new float[]{1.0f, 2.0f, 3.0f, 4.0f}, new int[]{4}, null, "bytecode_scalar_a");
        Tensor out = a.fastExp().relu();

        PreparedExecution prepared = prepare(out, vectorNativeRuntime());
        var fusedStep = fusedStep(prepared);
        FusedOperation fused = (FusedOperation) fusedStep.metadata().executionOperation();
        int vectorWidth = testsupport.MetadataArtifacts.cpuPlan(fusedStep.metadata()).dispatchHints().vectorWidth();

        byte[] bytecode = FusedOperationGenerator.generate(
                "debug/test/F32SegmentScalarKernel",
                fused.getPlan(),
                fused.getNumericContract(),
                fused.getApproximationContract(),
                vectorWidth,
                FusedAsmSpecializationKind.NONE
        );
        String constantPool = new String(bytecode, StandardCharsets.ISO_8859_1);

        assertEquals(1, vectorWidth);
        assertTrue(constantPool.contains("java/lang/foreign/MemorySegment"));
        assertTrue(constantPool.contains("JAVA_FLOAT"));
        assertTrue(constantPool.contains("get"));
        assertTrue(constantPool.contains("set"));
        assertFalse(constantPool.contains("fromMemorySegment"));
        assertFalse(constantPool.contains("intoMemorySegment"));
        assertNoArrayBackedSegmentBytecode(constantPool);
    }

    @Test
    void generatedF32SegmentBroadcastVectorKernelUsesMemorySegmentApiWithoutArrayBindings() {
        Tensor a = new Tensor(new float[128], new int[]{128}, null, "bytecode_broadcast_a");
        Tensor bias = new Tensor(new float[]{1.5f}, new int[]{1}, null, "bytecode_broadcast_bias");
        Tensor out = a.add(bias).relu();

        PreparedExecution prepared = prepare(out, vectorNativeRuntime());
        var fusedStep = fusedStep(prepared);
        FusedOperation fused = (FusedOperation) fusedStep.metadata().executionOperation();
        int vectorWidth = testsupport.MetadataArtifacts.cpuPlan(fusedStep.metadata()).dispatchHints().vectorWidth();

        byte[] bytecode = FusedOperationGenerator.generate(
                "debug/test/F32SegmentBroadcastVectorKernel",
                fused.getPlan(),
                fused.getNumericContract(),
                fused.getApproximationContract(),
                vectorWidth,
                FusedAsmSpecializationKind.NONE
        );
        String constantPool = new String(bytecode, StandardCharsets.ISO_8859_1);

        assertTrue(constantPool.contains("java/lang/foreign/MemorySegment"));
        assertTrue(constantPool.contains("JAVA_FLOAT"));
        assertTrue(constantPool.contains("broadcast"));
        assertTrue(constantPool.contains("intoMemorySegment"));
        assertNoArrayBackedSegmentBytecode(constantPool);
    }

    @Test
    void generatedF32SegmentGatherVectorKernelUsesMemorySegmentLaneLoadsAndScratchVectorLoad() {
        Tensor row = new Tensor(new float[64], new int[]{64, 1}, null, "bytecode_gather_row");
        Tensor col = new Tensor(new float[17], new int[]{1, 17}, null, "bytecode_gather_col");
        Tensor out = row.add(col).relu();

        PreparedExecution prepared = prepare(out, vectorNativeRuntime());
        var fusedStep = fusedStep(prepared);
        FusedOperation fused = (FusedOperation) fusedStep.metadata().executionOperation();
        int vectorWidth = testsupport.MetadataArtifacts.cpuPlan(fusedStep.metadata()).dispatchHints().vectorWidth();

        byte[] bytecode = FusedOperationGenerator.generate(
                "debug/test/F32SegmentGatherVectorKernel",
                fused.getPlan(),
                fused.getNumericContract(),
                fused.getApproximationContract(),
                vectorWidth,
                FusedAsmSpecializationKind.NONE
        );
        String constantPool = new String(bytecode, StandardCharsets.ISO_8859_1);

        assertTrue(vectorWidth > 1);
        assertTrue(constantPool.contains("java/lang/foreign/MemorySegment"));
        assertTrue(constantPool.contains("JAVA_FLOAT"));
        assertTrue(constantPool.contains("get"));
        assertTrue(constantPool.contains("fromArray"));
        assertTrue(constantPool.contains("intoMemorySegment"));
        assertFalse(constantPool.contains("fromMemorySegment"));
        assertFalse(constantPool.contains("FusedBroadcastVectorOps"));
        assertFalse(constantPool.contains("FusedStorageOps"));
        assertFalse(constantPool.contains("TensorInternalAccess"));
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
            assertEquals(fusedTrace(trace).metadata().fused().executionBackend(),
                    fusedTrace(trace).metadata().attributes().get("fusedExecutionClass"));
            assertEquals(fusedTrace(trace).metadata().fused().vectorFallbackReason(),
                    fusedTrace(trace).metadata().attributes().get("fusedVectorFallbackReason"));
            assertFalse(fusedTrace(trace).metadata().attributes().containsKey("fusedNativeOutputWritten"));
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
    void generatedBf16SegmentKernelStaysScalarOnlyWithoutArrayBindings() {
        Tensor a = bf16("bf16_bytecode_a", 1.0f, -2.0f, 3.0f, 4.0f);
        Tensor b = bf16("bf16_bytecode_b", 0.5f, 1.0f, -1.5f, 2.0f);
        Tensor out = a.add(b).relu();

        PreparedExecution prepared = prepare(out, vectorNativeRuntime());
        var fusedStep = fusedStep(prepared);
        FusedOperation fused = (FusedOperation) fusedStep.metadata().executionOperation();
        int vectorWidth = testsupport.MetadataArtifacts.cpuPlan(fusedStep.metadata()).dispatchHints().vectorWidth();

        byte[] bytecode = FusedOperationGenerator.generate(
                "debug/test/Bf16SegmentScalarOnlyKernel",
                fused.getPlan(),
                fused.getNumericContract(),
                fused.getApproximationContract(),
                vectorWidth,
                FusedAsmSpecializationKind.NONE
        );
        String constantPool = new String(bytecode, StandardCharsets.ISO_8859_1);

        assertEquals(1, vectorWidth);
        assertTrue(constantPool.contains("fromBFloat16Bits"));
        assertTrue(constantPool.contains("toBFloat16Bits"));
        assertTrue(constantPool.contains("JAVA_SHORT"));
        assertFalse(constantPool.contains("fromMemorySegment"));
        assertFalse(constantPool.contains("intoMemorySegment"));
        assertNoArrayBackedSegmentBytecode(constantPool);
        assertFalse(constantPool.contains("loadVectorBF16Array"));
        assertFalse(constantPool.contains("storeVectorBF16Array"));
    }

    @Test
    void boolFusedSegmentWritesNativeOutput() {
        Tensor a = new Tensor(new float[]{1.0f, 5.0f, -1.0f, 7.0f}, new int[]{4}, null, "bool_a");
        Tensor b = new Tensor(new float[]{2.0f, 3.0f, -2.0f, 8.0f}, new int[]{4}, null, "bool_b");
        Tensor out = a.greaterThan(b).logicalNot();

        PreparedExecution prepared = prepare(out, vectorNativeRuntime());
        assertSegmentPrepared(prepared);
        RunTrace trace = prepared.executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(new byte[]{1, 0, 0, 1}, out.toBoolByteArrayCopy());
        assertScalarOnlySegmentTrace(trace);
        assertGraphOutputMaterializedFromNativeFusedOutput(prepared, trace);
    }

    @Test
    void boolMaskFusedSegmentWritesNativeScalarOnlyOutput() {
        Tensor a = new Tensor(new byte[]{1, 0, 1, 0, 1}, new int[]{5}, null, "bool_mask_a", DataType.BOOL);
        Tensor b = new Tensor(new byte[]{1, 1, 0, 0, 1}, new int[]{5}, null, "bool_mask_b", DataType.BOOL);
        Tensor out = a.logicalAnd(b).logicalNot();

        PreparedExecution prepared = prepare(out, vectorNativeRuntime());
        assertSegmentPrepared(prepared);
        RunTrace trace = prepared.executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(new byte[]{0, 1, 1, 1, 0}, out.toBoolByteArrayCopy());
        assertScalarOnlySegmentTrace(trace);
        assertGraphOutputMaterializedFromNativeFusedOutput(prepared, trace);
    }

    @Test
    void generatedBoolCompareSegmentKernelStaysScalarOnlyWithoutArrayBindings() {
        Tensor a = new Tensor(new float[]{1.0f, 5.0f, -1.0f, 7.0f}, new int[]{4}, null, "bool_bytecode_a");
        Tensor b = new Tensor(new float[]{2.0f, 3.0f, -2.0f, 8.0f}, new int[]{4}, null, "bool_bytecode_b");
        Tensor out = a.greaterThan(b).logicalNot();

        PreparedExecution prepared = prepare(out, vectorNativeRuntime());
        var fusedStep = fusedStep(prepared);
        FusedOperation fused = (FusedOperation) fusedStep.metadata().executionOperation();
        int vectorWidth = testsupport.MetadataArtifacts.cpuPlan(fusedStep.metadata()).dispatchHints().vectorWidth();

        byte[] bytecode = FusedOperationGenerator.generate(
                "debug/test/BoolCompareSegmentScalarOnlyKernel",
                fused.getPlan(),
                fused.getNumericContract(),
                fused.getApproximationContract(),
                vectorWidth,
                FusedAsmSpecializationKind.NONE
        );
        String constantPool = new String(bytecode, StandardCharsets.ISO_8859_1);

        assertEquals(1, vectorWidth);
        assertTrue(constantPool.contains("java/lang/foreign/MemorySegment"));
        assertTrue(constantPool.contains("JAVA_FLOAT"));
        assertTrue(constantPool.contains("JAVA_BYTE"));
        assertFalse(constantPool.contains("storeMaskF32Array"));
        assertNoArrayBackedSegmentBytecode(constantPool);
    }

    @Test
    void segmentVectorPathHandlesInnermostBroadcastInputWithGeneratedGather() {
        int rows = 4096;
        int cols = 32;
        float[] data = new float[rows];
        for (int i = 0; i < data.length; i++) {
            data[i] = (i % 17) - 8.0f;
        }
        Tensor rowBias = new Tensor(data, new int[]{rows, 1}, null, "innermost_broadcast_bias");
        Tensor zeros = new Tensor(new float[rows * cols], new int[]{rows, cols}, null, "innermost_broadcast_zeros");
        Tensor out = rowBias.add(zeros).relu();

        PreparedExecution prepared = prepare(out, vectorNativeRuntime());
        assertSegmentPrepared(prepared);
        RunTrace trace = prepared.executeTraced(ExecutionMode.FORWARD);

        double[] values = out.toDoubleArrayCopy();
        assertEquals(Math.max(0.0, data[0]), values[0], 1e-5);
        assertEquals(Math.max(0.0, data[0]), values[1], 1e-5);
        assertEquals(Math.max(0.0, data[1]), values[cols], 1e-5);
        assertVectorSegmentTrace(trace);
        assertGraphOutputMaterializedFromNativeFusedOutput(prepared, trace);
    }

    @Test
    void segmentScalarPathHandlesUnsupportedWhereMaskInput() {
        int size = 257;
        byte[] maskData = new byte[size];
        float[] trueData = new float[size];
        float[] falseData = new float[size];
        for (int i = 0; i < size; i++) {
            maskData[i] = (byte) (i % 3 == 0 ? 1 : 0);
            trueData[i] = i * 0.25f;
            falseData[i] = -i * 0.5f;
        }
        Tensor mask = new Tensor(maskData, new int[]{size}, null, "where_mask", DataType.BOOL);
        Tensor ifTrue = new Tensor(trueData, new int[]{size}, null, "where_true");
        Tensor ifFalse = new Tensor(falseData, new int[]{size}, null, "where_false");
        Tensor out = Tensor.where(mask, ifTrue, ifFalse).relu();

        PreparedExecution prepared = prepare(out, vectorNativeRuntime());
        assertSegmentPrepared(prepared);
        RunTrace trace = prepared.executeTraced(ExecutionMode.FORWARD);

        double[] values = out.toDoubleArrayCopy();
        assertEquals(trueData[0], values[0], 1e-5);
        assertEquals(0.0, values[1], 1e-5);
        assertEquals(trueData[255], values[255], 1e-5);
        assertScalarOnlySegmentTrace(trace);
        assertGraphOutputMaterializedFromNativeFusedOutput(prepared, trace);
    }

    @Test
    void generatedWhereMaskSegmentKernelStaysScalarOnlyWithoutArrayBindings() {
        int size = 64;
        byte[] maskData = new byte[size];
        float[] trueData = new float[size];
        float[] falseData = new float[size];
        for (int i = 0; i < size; i++) {
            maskData[i] = (byte) (i % 2 == 0 ? 1 : 0);
            trueData[i] = i;
            falseData[i] = -i;
        }
        Tensor mask = new Tensor(maskData, new int[]{size}, null, "where_bytecode_mask", DataType.BOOL);
        Tensor ifTrue = new Tensor(trueData, new int[]{size}, null, "where_bytecode_true");
        Tensor ifFalse = new Tensor(falseData, new int[]{size}, null, "where_bytecode_false");
        Tensor out = Tensor.where(mask, ifTrue, ifFalse).relu();

        PreparedExecution prepared = prepare(out, vectorNativeRuntime());
        var fusedStep = fusedStep(prepared);
        FusedOperation fused = (FusedOperation) fusedStep.metadata().executionOperation();
        int vectorWidth = testsupport.MetadataArtifacts.cpuPlan(fusedStep.metadata()).dispatchHints().vectorWidth();

        byte[] bytecode = FusedOperationGenerator.generate(
                "debug/test/WhereMaskSegmentScalarOnlyKernel",
                fused.getPlan(),
                fused.getNumericContract(),
                fused.getApproximationContract(),
                vectorWidth,
                FusedAsmSpecializationKind.NONE
        );
        String constantPool = new String(bytecode, StandardCharsets.ISO_8859_1);

        assertEquals(1, vectorWidth);
        assertTrue(constantPool.contains("java/lang/foreign/MemorySegment"));
        assertTrue(constantPool.contains("JAVA_BYTE"));
        assertTrue(constantPool.contains("JAVA_FLOAT"));
        assertFalse(constantPool.contains("loadMaskF32Array"));
        assertNoArrayBackedSegmentBytecode(constantPool);
    }

    @Test
    void nonContiguousFusedInputStaysOnVisibleJavaArrayPathUnderCpuNativeProfile() {
        Tensor left = new Tensor(
                new float[]{1, 2, 3, 4, 5, 6, 7, 8},
                new int[]{2, 4},
                new int[]{1, 2},
                null,
                "noncontiguous_left",
                DataType.FLOAT32
        );
        Tensor right = new Tensor(new float[]{2, 3, 4, 5, 6, 7, 8, 9}, new int[]{2, 4}, null, "right");
        double[] leftLogical = left.toDoubleArrayCopy();
        Tensor out = left.mul(right).relu();

        PreparedExecution prepared = prepare(out, vectorNativeRuntime());
        assertArrayPrepared(prepared);
        RunTrace trace = prepared.executeTraced(ExecutionMode.FORWARD);

        double[] values = out.toDoubleArrayCopy();
        for (int i = 0; i < values.length; i++) {
            assertEquals(Math.max(0.0, leftLogical[i] * (i + 2.0)), values[i], 1e-5);
        }
        assertEquals("CPU_JAVA_ARRAY", fusedTrace(trace).metadata().attributes().get("fusedInputStorageKind"));
        assertEquals("CPU_JAVA_ARRAY", fusedTrace(trace).metadata().attributes().get("fusedOutputStorageKind"));
        assertFalse(fusedTrace(trace).metadata().attributes().containsKey("fusedNativeOutputWritten"));
    }

    @Test
    void explicitExpandFusedInputStaysOnVisibleJavaArrayPathUnderCpuNativeProfile() {
        Tensor row = new Tensor(new float[]{1.0f, -2.0f, 3.0f, -4.0f}, new int[]{1, 4}, null, "expanded_row");
        Tensor expanded = row.expand(3, 4);
        Tensor zeros = new Tensor(new float[12], new int[]{3, 4}, null, "expanded_zeros");
        Tensor out = expanded.add(zeros).relu();

        PreparedExecution prepared = prepare(out, vectorNativeRuntime());
        assertArrayPrepared(prepared);
        RunTrace trace = prepared.executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(
                new double[]{1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 3.0, 0.0},
                out.toDoubleArrayCopy(),
                1e-5
        );
        assertJavaArrayFusedTrace(trace);
    }

    @Test
    void stridedExpandBroadcastFusedInputStaysOnVisibleJavaArrayPathUnderCpuNativeProfile() {
        Tensor column = new Tensor(
                new float[]{1.0f, -2.0f, 3.0f, -4.0f},
                new int[]{4, 1},
                new int[]{1, 2},
                null,
                "strided_column",
                DataType.FLOAT32
        );
        Tensor expanded = column.expand(4, 3);
        Tensor zeros = new Tensor(new float[12], new int[]{4, 3}, null, "strided_expand_zeros");
        Tensor out = expanded.add(zeros).relu();

        PreparedExecution prepared = prepare(out, vectorNativeRuntime());
        assertArrayPrepared(prepared);
        RunTrace trace = prepared.executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(
                new double[]{1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0},
                out.toDoubleArrayCopy(),
                1e-5
        );
        assertJavaArrayFusedTrace(trace);
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
        assertEquals("NONE", fusedTrace.metadata().fused().vectorFallbackReason());
        assertEquals("CPU_MEMORY_SEGMENT", fusedTrace.metadata().attributes().get("fusedInputStorageKind"));
        assertEquals("CPU_MEMORY_SEGMENT", fusedTrace.metadata().attributes().get("fusedOutputStorageKind"));
        assertGraphOutputMaterializedFromNativeFusedOutput(prepared, trace);
    }

    @Test
    void segmentFusedExecutionLeavesCpuArrayStaleUntilExplicitMaterialization() {
        Tensor a = new Tensor(new float[]{1.0f, -4.0f, 3.0f, 5.0f}, new int[]{4}, null, "stale_a");
        Tensor b = new Tensor(new float[]{2.0f, 2.0f, -8.0f, 1.0f}, new int[]{4}, null, "stale_b");
        Tensor out = a.add(b).relu();
        FusedRunFixture fixture = runFixture(out, nativeRuntime());
        try {
            PreparedExecutionStep step = fusedStep(fixture.prepared());
            int outputNodeId = step.compiledNode().id();

            PreparedExecutionRunner.executeSteps(List.of(step), fixture.context(), false, null, 0);

            assertTrue(fixture.state().residencyForNodeId(outputNodeId).nativeCurrent());
            assertFalse(fixture.state().residencyForNodeId(outputNodeId).cpuCurrent());
            assertTrue(fixture.state().cpuMaterializationTraces().stream().noneMatch(entry ->
                            entry.nodeId() == outputNodeId && entry.detail().contains("native_to_array")),
                    () -> "native fused output should not materialize before an explicit CPU read: "
                            + fixture.state().cpuMaterializationTraces());

            fixture.state().requireCpuReadable(outputNodeId, CpuMaterializationReason.GRAPH_OUTPUT);

            assertTrue(fixture.state().residencyForNodeId(outputNodeId).cpuCurrent());
            assertArrayEquals(
                    new float[]{3.0f, 0.0f, 0.0f, 6.0f},
                    fixture.state().runtimeTensorForNodeId(outputNodeId).toFloat32ArrayCopy(),
                    0f
            );
        } finally {
            fixture.close();
        }
    }

    @Test
    void segmentFusedOutputReusesReservedNativeStorage() {
        Tensor a = new Tensor(new float[]{1.0f, -4.0f, 3.0f, 5.0f}, new int[]{4}, null, "reuse_a");
        Tensor b = new Tensor(new float[]{2.0f, 2.0f, -8.0f, 1.0f}, new int[]{4}, null, "reuse_b");
        Tensor out = a.add(b).relu();
        FusedRunFixture fixture = runFixture(out, nativeRuntime());
        try {
            PreparedExecutionStep step = fusedStep(fixture.prepared());
            int outputNodeId = step.compiledNode().id();
            NativeTensorStorage reserved = fixture.state().allocateNativeStorage(
                    DataType.FLOAT32,
                    step.compiledNode().flatDataSize(),
                    "reserved-fused-output"
            );
            fixture.state().reserveNativeOutputStorage(outputNodeId, reserved);

            PreparedExecutionRunner.executeSteps(List.of(step), fixture.context(), false, null, 0);

            assertSame(reserved, fixture.state().nativeStorageForNodeId(outputNodeId));
            assertTrue(fixture.state().residencyForNodeId(outputNodeId).nativeCurrent());
            fixture.state().requireCpuReadable(outputNodeId, CpuMaterializationReason.GRAPH_OUTPUT);
            assertArrayEquals(
                    new float[]{3.0f, 0.0f, 0.0f, 6.0f},
                    fixture.state().runtimeTensorForNodeId(outputNodeId).toFloat32ArrayCopy(),
                    0f
            );
        } finally {
            fixture.close();
        }
    }

    @Test
    void segmentFusedOutputAllocatesReplacementForWrongSizeNativeStorage() throws Exception {
        Tensor a = new Tensor(new float[]{1.0f, -4.0f, 3.0f, 5.0f}, new int[]{4}, null, "replace_a");
        Tensor b = new Tensor(new float[]{2.0f, 2.0f, -8.0f, 1.0f}, new int[]{4}, null, "replace_b");
        Tensor out = a.add(b).relu();
        FusedRunFixture fixture = runFixture(out, nativeRuntime());
        NativeTensorStorage wrongSize = fixture.state().allocateNativeStorage(
                DataType.FLOAT32,
                1,
                "wrong-size-fused-output"
        );
        try {
            PreparedExecutionStep step = fusedStep(fixture.prepared());
            int outputNodeId = step.compiledNode().id();
            forceNativeStorageBinding(fixture.state(), outputNodeId, wrongSize);

            PreparedExecutionRunner.executeSteps(List.of(step), fixture.context(), false, null, 0);

            NativeTensorStorage actual = fixture.state().nativeStorageForNodeId(outputNodeId);
            assertNotNull(actual);
            assertNotSame(wrongSize, actual);
            assertEquals(DataType.FLOAT32, actual.getType());
            assertEquals(step.compiledNode().flatDataSize(), actual.getSize());
            assertTrue(fixture.state().residencyForNodeId(outputNodeId).nativeCurrent());
        } finally {
            fixture.close();
            wrongSize.close();
        }
    }

    @Test
    void segmentFusedOutputAllocatesReplacementForWrongDtypeNativeStorage() throws Exception {
        Tensor a = new Tensor(new float[]{1.0f, -4.0f, 3.0f, 5.0f}, new int[]{4}, null, "replace_dtype_a");
        Tensor b = new Tensor(new float[]{2.0f, 2.0f, -8.0f, 1.0f}, new int[]{4}, null, "replace_dtype_b");
        Tensor out = a.add(b).relu();
        FusedRunFixture fixture = runFixture(out, nativeRuntime());
        NativeTensorStorage wrongDtype = fixture.state().allocateNativeStorage(
                DataType.FLOAT64,
                4,
                "wrong-dtype-fused-output"
        );
        try {
            PreparedExecutionStep step = fusedStep(fixture.prepared());
            int outputNodeId = step.compiledNode().id();
            forceNativeStorageBinding(fixture.state(), outputNodeId, wrongDtype);

            PreparedExecutionRunner.executeSteps(List.of(step), fixture.context(), false, null, 0);

            NativeTensorStorage actual = fixture.state().nativeStorageForNodeId(outputNodeId);
            assertNotNull(actual);
            assertNotSame(wrongDtype, actual);
            assertEquals(DataType.FLOAT32, actual.getType());
            assertEquals(step.compiledNode().flatDataSize(), actual.getSize());
            assertTrue(fixture.state().residencyForNodeId(outputNodeId).nativeCurrent());
        } finally {
            fixture.close();
            wrongDtype.close();
        }
    }

    @Test
    void segmentFusedOutputReservationDemotesReusedNativeCurrentStorageWhenGeneratedExecutionFails() {
        Tensor a = new Tensor(new float[]{1.0f, -4.0f, 3.0f, 5.0f}, new int[]{4}, null, "fail_a");
        Tensor b = new Tensor(new float[]{2.0f, 2.0f, -8.0f, 1.0f}, new int[]{4}, null, "fail_b");
        Tensor out = a.add(b).relu();
        FusedRunFixture fixture = runFixture(out, nativeRuntime());
        try {
            PreparedExecutionStep step = fusedStep(fixture.prepared());
            int outputNodeId = step.compiledNode().id();
            NativeTensorStorage preexisting = fixture.state().allocateNativeStorage(
                    DataType.FLOAT32,
                    step.compiledNode().flatDataSize(),
                    "preexisting-native-current-fused-output"
            );
            NativeFloat32Storage preexistingF32 = (NativeFloat32Storage) preexisting;
            preexistingF32.setFloat32At(0, -101.0f);
            preexistingF32.setFloat32At(1, -102.0f);
            preexistingF32.setFloat32At(2, -103.0f);
            preexistingF32.setFloat32At(3, -104.0f);
            fixture.state().attachNativeStorage(outputNodeId, preexisting, "preexisting native output");
            assertTrue(fixture.state().residencyForNodeId(outputNodeId).nativeCurrent());

            PreparedExecutionStep throwingStep = withThrowingFusedExecutable(step);

            IllegalStateException failure = assertThrows(
                    IllegalStateException.class,
                    () -> PreparedExecutionRunner.executeSteps(List.of(throwingStep), fixture.context(), false, null, 0)
            );

            assertTrue(failure.getMessage().contains("intentional fused segment failure"));
            assertSame(preexisting, fixture.state().nativeStorageForNodeId(outputNodeId));
            assertEquals(123.0f, preexistingF32.getFloat32At(0), 0f);
            assertFalse(fixture.state().residencyForNodeId(outputNodeId).nativeCurrent());
            assertFalse(fixture.state().residencyForNodeId(outputNodeId).cpuCurrent());
            assertEquals(StorageResidency.CPU_NATIVE, fixture.state().residencyForNodeId(outputNodeId).residency());
        } finally {
            fixture.close();
        }
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

    private static trace.execution.ExecutionStepTrace fusedTrace(RunTrace trace) {
        return trace.steps().stream()
                .filter(step -> step.metadata().fused() != null)
                .findFirst()
                .orElseThrow();
    }

    private static void assertVectorSegmentTrace(RunTrace trace) {
        var fusedTrace = fusedTrace(trace);
        assertTrue(fusedTrace.metadata().dispatch().vectorWidth() > 1);
        assertEquals("NONE", fusedTrace.metadata().fused().vectorFallbackReason());
        assertEquals("CPU_MEMORY_SEGMENT", fusedTrace.metadata().attributes().get("fusedInputStorageKind"));
        assertEquals("CPU_MEMORY_SEGMENT", fusedTrace.metadata().attributes().get("fusedOutputStorageKind"));
        assertEquals(fusedTrace.metadata().fused().executionBackend(),
                fusedTrace.metadata().attributes().get("fusedExecutionClass"));
        assertEquals("NONE", fusedTrace.metadata().attributes().get("fusedVectorFallbackReason"));
        assertEquals(true, fusedTrace.metadata().attributes().get("fusedVectorEligible"));
        assertNativeOutputWriteTrace(fusedTrace);
    }

    private static void assertScalarOnlySegmentTrace(RunTrace trace) {
        var fusedTrace = fusedTrace(trace);
        assertEquals(1, fusedTrace.metadata().dispatch().vectorWidth());
        assertEquals("VECTOR_PATH_UNSUPPORTED", fusedTrace.metadata().fused().vectorFallbackReason());
        assertEquals("CPU_MEMORY_SEGMENT", fusedTrace.metadata().attributes().get("fusedInputStorageKind"));
        assertEquals("CPU_MEMORY_SEGMENT", fusedTrace.metadata().attributes().get("fusedOutputStorageKind"));
        assertEquals(fusedTrace.metadata().fused().executionBackend(),
                fusedTrace.metadata().attributes().get("fusedExecutionClass"));
        assertEquals("VECTOR_PATH_UNSUPPORTED", fusedTrace.metadata().attributes().get("fusedVectorFallbackReason"));
        assertEquals(false, fusedTrace.metadata().attributes().get("fusedVectorEligible"));
        assertNativeOutputWriteTrace(fusedTrace);
    }

    private static void assertBelowThresholdSegmentTrace(RunTrace trace) {
        var fusedTrace = fusedTrace(trace);
        assertTrue(fusedTrace.metadata().dispatch().vectorWidth() > 1);
        assertEquals("SCALAR", fusedTrace.metadata().dispatch().mode());
        assertEquals("BELOW_VECTOR_THRESHOLD", fusedTrace.metadata().fused().vectorFallbackReason());
        assertEquals("CPU_MEMORY_SEGMENT", fusedTrace.metadata().attributes().get("fusedInputStorageKind"));
        assertEquals("CPU_MEMORY_SEGMENT", fusedTrace.metadata().attributes().get("fusedOutputStorageKind"));
        assertEquals("BELOW_VECTOR_THRESHOLD", fusedTrace.metadata().attributes().get("fusedVectorFallbackReason"));
        assertEquals(false, fusedTrace.metadata().attributes().get("fusedVectorEligible"));
        assertNativeOutputWriteTrace(fusedTrace);
    }

    private static void assertJavaArrayFusedTrace(RunTrace trace) {
        var fusedTrace = fusedTrace(trace);
        assertEquals("CPU_JAVA_ARRAY", fusedTrace.metadata().attributes().get("fusedInputStorageKind"));
        assertEquals("CPU_JAVA_ARRAY", fusedTrace.metadata().attributes().get("fusedOutputStorageKind"));
        assertFalse(fusedTrace.metadata().attributes().containsKey("fusedNativeOutputWritten"));
    }

    private static void assertNativeOutputWriteTrace(trace.execution.ExecutionStepTrace fusedTrace) {
        assertEquals(true, fusedTrace.metadata().attributes().get("fusedNativeOutputWritten"));
        assertEquals("CPU_NATIVE", fusedTrace.metadata().attributes().get("fusedNativeOutputResidency"));
        assertEquals("CPU fused MemorySegment wrote output",
                fusedTrace.metadata().attributes().get("fusedNativeOutputWriteReason"));
        assertEquals("CPU_NATIVE", fusedTrace.metadata().attributes().get("storageResidency"));
        assertEquals(false, fusedTrace.metadata().attributes().get("storageCpuCurrent"));
    }

    private static void assertNoArrayBackedSegmentBytecode(String constantPool) {
        assertFalse(constantPool.contains("TensorInternalAccess"));
        assertFalse(constantPool.contains("float32Data"));
        assertFalse(constantPool.contains("float64Data"));
        assertFalse(constantPool.contains("bfloat16Data"));
        assertFalse(constantPool.contains("boolData"));
        assertFalse(constantPool.contains("inputFloatContinuation"));
        assertFalse(constantPool.contains("fromArray"));
        assertFalse(constantPool.contains("intoArray"));
        assertFalse(constantPool.contains("loadVectorBF16Array"));
        assertFalse(constantPool.contains("storeVectorBF16Array"));
        assertFalse(constantPool.contains("loadMaskF32Array"));
        assertFalse(constantPool.contains("loadMaskF64Array"));
        assertFalse(constantPool.contains("storeMaskF32Array"));
        assertFalse(constantPool.contains("storeMaskF64Array"));
    }

    private static FusedRunFixture runFixture(Tensor out, RuntimeConfig runtimeConfig) {
        CompiledGraph compiled = CompiledGraph.compile(out, fusedOnlyConfig());
        PreparedExecution prepared = compiled.prepare(runtimeConfig);
        Map<Integer, PreparedStepMetadata> metadata = new HashMap<>();
        for (PreparedExecutionStep step : prepared.executionSteps()) {
            metadata.put(step.compiledNode().id(), step.metadata());
        }
        ExecutionState state = ExecutionState.create(
                compiled.program().compiledNodes(),
                compiled.program().descriptorIndex(),
                metadata,
                compiled.program().forwardOutputNode().id(),
                compiled.publication()
        );
        state.configureNativeCpuMemory(runtimeConfig.nativeCpuMemory());
        RuntimeMemoryBinder.bind(
                compiled.program().memoryPlan(),
                compiled.program().compiledNodes(),
                state
        );
        ExecutionContext context = ExecutionContext.fromRuntimeConfig(
                runtimeConfig,
                ExecutionMode.FORWARD,
                metadata,
                state
        );
        return new FusedRunFixture(prepared, state, context);
    }

    private static PreparedExecutionStep withThrowingFusedExecutable(PreparedExecutionStep step) {
        CpuFusedExecutionArtifact artifact = (CpuFusedExecutionArtifact) step.metadata().executable();
        PreparedFusedExecutable throwingExecutable = new PreparedFusedExecutable() {
            @Override
            public void applyRangeScalar(
                    List<Tensor> inputs,
                    Tensor out,
                    backend.cpu.execution.CpuKernelContext context,
                    int startInclusive,
                    int endExclusive
            ) {
                writePartialNativeOutput(context);
                throw new IllegalStateException("intentional fused segment failure");
            }

            @Override
            public void applyRangeVector(
                    List<Tensor> inputs,
                    Tensor out,
                    backend.cpu.execution.CpuKernelContext context,
                    int startInclusive,
                    int endExclusive
            ) {
                writePartialNativeOutput(context);
                throw new IllegalStateException("intentional fused segment failure");
            }
        };
        PreparedStepMetadata metadata = new PreparedStepMetadata(
                ComputeBackend.CPU,
                step.metadata().executionOperation(),
                step.metadata().executionInputNodeIds(),
                new CpuFusedExecutionArtifact(
                        artifact.cpuKernel(),
                        artifact.cpuPlan(),
                        throwingExecutable,
                        artifact.cpuWorkspace(),
                        artifact.vectorFallbackReason()
                ),
                step.metadata().inputResidencyRequirement(),
                step.metadata().outputResidencyEffect()
        );
        return new PreparedExecutionStep(
                step.compiledNode(),
                metadata,
                step.orderedNodeIds(),
                step.boundaryOutputNodeIds()
        );
    }

    private static void writePartialNativeOutput(backend.cpu.execution.CpuKernelContext context) {
        NativeTensorStorage output = FusedNativeSegmentBindings.outputStorage(context);
        if (output instanceof NativeFloat32Storage f32 && output.getSize() > 0) {
            f32.setFloat32At(0, 123.0f);
        }
    }

    @SuppressWarnings("unchecked")
    private static void forceNativeStorageBinding(
            ExecutionState state,
            int nodeId,
            NativeTensorStorage storage
    ) throws Exception {
        Field nativeStorageRegistryField = ExecutionState.class.getDeclaredField("nativeStorageRegistry");
        nativeStorageRegistryField.setAccessible(true);
        Object nativeStorageRegistry = nativeStorageRegistryField.get(state);
        Field storageByNodeIdField = nativeStorageRegistry.getClass().getDeclaredField("nativeStorageByNodeId");
        storageByNodeIdField.setAccessible(true);
        ((Map<Integer, NativeTensorStorage>) storageByNodeIdField.get(nativeStorageRegistry)).put(nodeId, storage);
    }

    private record FusedRunFixture(
            PreparedExecution prepared,
            ExecutionState state,
            ExecutionContext context
    ) {
        void close() {
            state.closeResources();
            prepared.close();
        }
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
