package backend.cpu.nativecpu;

import backend.blas.BlasProvider;
import backend.blas.OpenBlasFfmBridge;
import backend.cpu.kernels.CpuDTypeOps;
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
import graph.execution.PublicationPolicy;
import graph.execution.trace.ExecutionStepTrace;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NativeCpuElementwiseChainTest {
    @Test
    void cpuNativeMatmulReluKeepsReluOutputNative() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor out = a().matmul(b()).relu();

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> relu = attrs(trace.steps().stream()
                .filter(step -> "RELU".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertEquals("CPU_NATIVE", relu.get("cpuStorageProfile"));
        assertEquals("CPU_NATIVE", relu.get("requestedCpuStorage"));
        assertEquals("CPU_NATIVE", relu.get("actualCpuStorage"));
        assertEquals("NATIVE_CORRECT_BUT_SLOW", relu.get("nativeCpuKernelStatus"));
        assertEquals("SEGMENT_SCALAR", relu.get("nativeCpuKernelFamily"));
        assertEquals("", relu.get("nativeCpuFallbackReason"));
        assertEquals("CPU_NATIVE", relu.get("storageResidency"));
        assertEquals(false, relu.get("storageCpuCurrent"));
    }

    @Test
    void cpuNativeMatmulAddKeepsSameShapeAddOutputNative() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor c = tensor(new float[]{1f, -1f, 2f, -2f}, "c");
        Tensor out = a().matmul(b()).add(c);

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> add = attrs(trace.steps().stream()
                .filter(step -> "ADD".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertEquals("CPU_NATIVE", add.get("actualCpuStorage"));
        assertEquals("NATIVE_CORRECT_BUT_SLOW", add.get("nativeCpuKernelStatus"));
        assertEquals("SEGMENT_SCALAR", add.get("nativeCpuKernelFamily"));
        assertEquals("", add.get("nativeCpuFallbackReason"));
        assertEquals("CPU_NATIVE", add.get("storageResidency"));
    }

    @Test
    void cpuNativeMatmulBiasAddReluKeepsAddAndReluOutputNative() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor bias = vector(new float[]{1f, -100f}, "bias");
        Tensor out = a().matmul(b()).add(bias).relu();

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> add = attrs(trace.steps().stream()
                .filter(step -> "ADD".equals(step.opType()))
                .findFirst()
                .orElseThrow());
        Map<String, Object> relu = attrs(trace.steps().stream()
                .filter(step -> "RELU".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertEquals("CPU_NATIVE", add.get("actualCpuStorage"));
        assertEquals("NATIVE_CORRECT_BUT_SLOW", add.get("nativeCpuKernelStatus"));
        assertEquals("SEGMENT_SCALAR", add.get("nativeCpuKernelFamily"));
        assertEquals("", add.get("nativeCpuFallbackReason"));
        assertEquals("CPU_NATIVE", add.get("storageResidency"));
        assertEquals("CPU_NATIVE", relu.get("actualCpuStorage"));
        assertEquals("", relu.get("nativeCpuFallbackReason"));
        assertEquals("CPU_NATIVE", relu.get("storageResidency"));
    }

    @Test
    void cpuNativeMatmulBiasAddSupportsBiasOnLeftSide() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor bias = vector(new float[]{1f, -100f}, "bias");
        Tensor out = bias.add(a().matmul(b()));

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> add = attrs(trace.steps().stream()
                .filter(step -> "ADD".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertEquals("CPU_NATIVE", add.get("actualCpuStorage"));
        assertEquals("", add.get("nativeCpuFallbackReason"));
        assertEquals("CPU_NATIVE", add.get("storageResidency"));
    }

    @Test
    void cpuNativeMatmulMulReluKeepsMulAndReluOutputNative() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor scale = tensor(new float[]{2f, 3f, 4f, 5f}, "scale");
        Tensor out = a().matmul(b()).mul(scale).relu();

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> mul = attrs(trace.steps().stream()
                .filter(step -> "MUL".equals(step.opType()))
                .findFirst()
                .orElseThrow());
        Map<String, Object> relu = attrs(trace.steps().stream()
                .filter(step -> "RELU".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertEquals("CPU_NATIVE", mul.get("actualCpuStorage"));
        assertEquals("NATIVE_CORRECT_BUT_SLOW", mul.get("nativeCpuKernelStatus"));
        assertEquals("SEGMENT_SCALAR", mul.get("nativeCpuKernelFamily"));
        assertEquals("", mul.get("nativeCpuFallbackReason"));
        assertEquals("CPU_NATIVE", mul.get("storageResidency"));
        assertEquals("CPU_NATIVE", relu.get("actualCpuStorage"));
        assertEquals("", relu.get("nativeCpuFallbackReason"));
        assertEquals("CPU_NATIVE", relu.get("storageResidency"));
    }

    @Test
    void cpuNativeMatmulSubNegReluKeepsOutputsNative() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor offset = tensor(new float[]{100f, 100f, 100f, 100f}, "offset");
        Tensor out = a().matmul(b()).sub(offset).neg().relu();

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> sub = attrs(trace.steps().stream()
                .filter(step -> "SUB".equals(step.opType()))
                .findFirst()
                .orElseThrow());
        Map<String, Object> neg = attrs(trace.steps().stream()
                .filter(step -> "NEG".equals(step.opType()))
                .findFirst()
                .orElseThrow());
        Map<String, Object> relu = attrs(trace.steps().stream()
                .filter(step -> "RELU".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertEquals("CPU_NATIVE", sub.get("actualCpuStorage"));
        assertEquals("NATIVE_CORRECT_BUT_SLOW", sub.get("nativeCpuKernelStatus"));
        assertEquals("SEGMENT_SCALAR", sub.get("nativeCpuKernelFamily"));
        assertEquals("", sub.get("nativeCpuFallbackReason"));
        assertEquals("CPU_NATIVE", sub.get("storageResidency"));
        assertEquals("CPU_NATIVE", neg.get("actualCpuStorage"));
        assertEquals("NATIVE_CORRECT_BUT_SLOW", neg.get("nativeCpuKernelStatus"));
        assertEquals("SEGMENT_SCALAR", neg.get("nativeCpuKernelFamily"));
        assertEquals("", neg.get("nativeCpuFallbackReason"));
        assertEquals("CPU_NATIVE", neg.get("storageResidency"));
        assertEquals("CPU_NATIVE", relu.get("actualCpuStorage"));
        assertEquals("", relu.get("nativeCpuFallbackReason"));
        assertEquals("CPU_NATIVE", relu.get("storageResidency"));
    }

    @Test
    void cpuNativeMatmulDivKeepsDivOutputNative() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor divisor = tensor(new float[]{1f, 2f, 4f, 5f}, "divisor");
        Tensor out = a().matmul(b()).div(divisor);

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> div = attrs(trace.steps().stream()
                .filter(step -> "DIV".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertEquals("CPU_NATIVE", div.get("actualCpuStorage"));
        assertEquals("NATIVE_CORRECT_BUT_SLOW", div.get("nativeCpuKernelStatus"));
        assertEquals("SEGMENT_SCALAR", div.get("nativeCpuKernelFamily"));
        assertEquals("", div.get("nativeCpuFallbackReason"));
        assertEquals("CPU_NATIVE", div.get("storageResidency"));
    }

    @Test
    void cpuNativeMatmulBiasMulScalarReluKeepsMulScalarAndReluOutputNative() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor bias = vector(new float[]{1f, -100f}, "bias");
        Tensor out = a().matmul(b()).add(bias).mul(0.5d).relu();

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> mulScalar = attrs(trace.steps().stream()
                .filter(step -> "MUL_SCALAR".equals(step.opType()))
                .findFirst()
                .orElseThrow());
        Map<String, Object> relu = attrs(trace.steps().stream()
                .filter(step -> "RELU".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertEquals("CPU_NATIVE", mulScalar.get("actualCpuStorage"));
        assertEquals("NATIVE_CORRECT_BUT_SLOW", mulScalar.get("nativeCpuKernelStatus"));
        assertEquals("SEGMENT_SCALAR", mulScalar.get("nativeCpuKernelFamily"));
        assertEquals("", mulScalar.get("nativeCpuFallbackReason"));
        assertEquals("CPU_NATIVE", mulScalar.get("storageResidency"));
        assertEquals("CPU_NATIVE", relu.get("actualCpuStorage"));
        assertEquals("", relu.get("nativeCpuFallbackReason"));
        assertEquals("CPU_NATIVE", relu.get("storageResidency"));
    }

    @Test
    void cpuNativeMatmulWhereReluKeepsWhereAndReluOutputNative() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor condition = boolTensor(new byte[]{1, 0, 1, 0}, new int[]{2, 2}, "condition");
        Tensor fallback = tensor(new float[]{-100f, -100f, -100f, -100f}, "fallback");
        Tensor out = Tensor.where(condition, a().matmul(b()), fallback).relu();

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> where = attrs(trace.steps().stream()
                .filter(step -> "WHERE".equals(step.opType()))
                .findFirst()
                .orElseThrow());
        Map<String, Object> relu = attrs(trace.steps().stream()
                .filter(step -> "RELU".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertEquals("CPU_NATIVE", where.get("actualCpuStorage"));
        assertEquals("NATIVE_CORRECT_BUT_SLOW", where.get("nativeCpuKernelStatus"));
        assertEquals("SEGMENT_SCALAR", where.get("nativeCpuKernelFamily"));
        assertEquals("", where.get("nativeCpuFallbackReason"));
        assertEquals("CPU_NATIVE", where.get("storageResidency"));
        assertEquals("CPU_NATIVE", relu.get("actualCpuStorage"));
        assertEquals("", relu.get("nativeCpuFallbackReason"));
        assertEquals("CPU_NATIVE", relu.get("storageResidency"));
    }

    @Test
    void cpuNativeMatmulSumAllKeepsSumOutputNativeAndPublishesValue() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor out = a().matmul(b()).sum();

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD);

        Map<String, Object> sum = attrs(trace.steps().stream()
                .filter(step -> "SUM".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertEquals(134.0f, out.getFloat32Data()[0], 1.0e-5f);
        assertEquals("CPU_NATIVE", sum.get("actualCpuStorage"));
        assertEquals("NATIVE_CORRECT_BUT_SLOW", sum.get("nativeCpuKernelStatus"));
        assertEquals("SEGMENT_SCALAR", sum.get("nativeCpuKernelFamily"));
        assertEquals("", sum.get("nativeCpuFallbackReason"));
        assertEquals("CPU_NATIVE", sum.get("storageResidency"));
    }

    @Test
    void cpuNativeMatmulMeanAllKeepsMeanOutputNativeAndPublishesValue() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor out = a().matmul(b()).mean();

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD);

        Map<String, Object> mean = attrs(trace.steps().stream()
                .filter(step -> "MEAN".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertEquals(33.5f, out.getFloat32Data()[0], 1.0e-5f);
        assertEquals("CPU_NATIVE", mean.get("actualCpuStorage"));
        assertEquals("NATIVE_CORRECT_BUT_SLOW", mean.get("nativeCpuKernelStatus"));
        assertEquals("SEGMENT_SCALAR", mean.get("nativeCpuKernelFamily"));
        assertEquals("", mean.get("nativeCpuFallbackReason"));
        assertEquals("CPU_NATIVE", mean.get("storageResidency"));
    }

    @Test
    void cpuNativeMatmulSumAxisKeepsSumOutputNativeAndPublishesValues() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor sumColumns = a().matmul(b()).sum(0);
        Tensor sumRows = a().matmul(b()).sum(1);
        Tensor sumRowsKeepDims = a().matmul(b()).sum(1, true);

        var columnsTrace = CompiledGraph.compile(sumColumns, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD);
        var rowsTrace = CompiledGraph.compile(sumRows, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD);
        var keepDimsTrace = CompiledGraph.compile(sumRowsKeepDims, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD);

        assertArrayEquals(new float[]{62.0f, 72.0f}, sumColumns.getFloat32Data(), 1.0e-5f);
        assertArrayEquals(new float[]{41.0f, 93.0f}, sumRows.getFloat32Data(), 1.0e-5f);
        assertArrayEquals(new int[]{2, 1}, sumRowsKeepDims.getShape());
        assertArrayEquals(new float[]{41.0f, 93.0f}, sumRowsKeepDims.getFloat32Data(), 1.0e-5f);
        assertNativeReduction(columnsTrace.steps().stream()
                .filter(step -> "SUM".equals(step.opType()))
                .findFirst()
                .orElseThrow());
        assertNativeReduction(rowsTrace.steps().stream()
                .filter(step -> "SUM".equals(step.opType()))
                .findFirst()
                .orElseThrow());
        assertNativeReduction(keepDimsTrace.steps().stream()
                .filter(step -> "SUM".equals(step.opType()))
                .findFirst()
                .orElseThrow());
    }

    @Test
    void cpuNativeMatmulMeanAxisKeepsMeanOutputNativeAndPublishesValues() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor meanColumns = a().matmul(b()).mean(0);
        Tensor meanRows = a().matmul(b()).mean(1);
        Tensor meanRowsKeepDims = a().matmul(b()).mean(1, true);

        var columnsTrace = CompiledGraph.compile(meanColumns, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD);
        var rowsTrace = CompiledGraph.compile(meanRows, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD);
        var keepDimsTrace = CompiledGraph.compile(meanRowsKeepDims, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD);

        assertArrayEquals(new float[]{31.0f, 36.0f}, meanColumns.getFloat32Data(), 1.0e-5f);
        assertArrayEquals(new float[]{20.5f, 46.5f}, meanRows.getFloat32Data(), 1.0e-5f);
        assertArrayEquals(new int[]{2, 1}, meanRowsKeepDims.getShape());
        assertArrayEquals(new float[]{20.5f, 46.5f}, meanRowsKeepDims.getFloat32Data(), 1.0e-5f);
        assertNativeReduction(columnsTrace.steps().stream()
                .filter(step -> "MEAN".equals(step.opType()))
                .findFirst()
                .orElseThrow());
        assertNativeReduction(rowsTrace.steps().stream()
                .filter(step -> "MEAN".equals(step.opType()))
                .findFirst()
                .orElseThrow());
        assertNativeReduction(keepDimsTrace.steps().stream()
                .filter(step -> "MEAN".equals(step.opType()))
                .findFirst()
                .orElseThrow());
    }

    @Test
    void cpuNativeF32ToBf16CastWritesNativeOutputAndPublishesRawBits() {
        Tensor input = new Tensor(new float[]{
                1.0f,
                -0.0f,
                Float.POSITIVE_INFINITY,
                Float.NaN
        }, new int[]{2, 2}, null, "cast_f32", DataType.FLOAT32);
        Tensor out = input.cast(DataType.BFLOAT16);

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD);

        assertArrayEquals(new short[]{
                CpuDTypeOps.toBFloat16Bits(1.0f),
                CpuDTypeOps.toBFloat16Bits(-0.0f),
                CpuDTypeOps.toBFloat16Bits(Float.POSITIVE_INFINITY),
                CpuDTypeOps.toBFloat16Bits(Float.NaN)
        }, out.getBFloat16Data());
        assertNativeCast(trace.steps().stream()
                .filter(step -> "CAST".equals(step.opType()))
                .findFirst()
                .orElseThrow());
    }

    @Test
    void cpuNativeBf16ToF32CastWritesNativeOutputAndPublishesValues() {
        short[] bits = new short[]{
                (short) 0x3f80,
                (short) 0x8000,
                (short) 0x7f80,
                (short) 0x0001
        };
        Tensor input = new Tensor(bits.clone(), new int[]{2, 2}, null, "cast_bf16", DataType.BFLOAT16);
        Tensor out = input.cast(DataType.FLOAT32);

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD);

        assertEquals(Float.floatToRawIntBits(CpuDTypeOps.fromBFloat16Bits(bits[0])), Float.floatToRawIntBits(out.getFloat32Data()[0]));
        assertEquals(Float.floatToRawIntBits(CpuDTypeOps.fromBFloat16Bits(bits[1])), Float.floatToRawIntBits(out.getFloat32Data()[1]));
        assertEquals(Float.floatToRawIntBits(CpuDTypeOps.fromBFloat16Bits(bits[2])), Float.floatToRawIntBits(out.getFloat32Data()[2]));
        assertEquals(Float.floatToRawIntBits(CpuDTypeOps.fromBFloat16Bits(bits[3])), Float.floatToRawIntBits(out.getFloat32Data()[3]));
        assertNativeCast(trace.steps().stream()
                .filter(step -> "CAST".equals(step.opType()))
                .findFirst()
                .orElseThrow());
    }

    @Test
    void cpuNativeMatmulReshapeReluKeepsViewAndReluOutputsNative() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor out = a().matmul(b()).reshape(4).relu();

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        assertNativeView(trace.steps().stream()
                .filter(step -> "RESHAPE".equals(step.opType()))
                .findFirst()
                .orElseThrow());
        assertNativeSegmentScalar(trace.steps().stream()
                .filter(step -> "RELU".equals(step.opType()))
                .findFirst()
                .orElseThrow());
    }

    @Test
    void cpuNativeMatmulSqueezeExpandDimsAddKeepsViewOutputsNative() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor bias = vector(new float[]{1f, -100f}, "bias");
        Tensor out = a().matmul(b())
                .expandDims(0)
                .squeeze(0)
                .add(bias);

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        assertNativeView(trace.steps().stream()
                .filter(step -> "EXPAND_DIMS".equals(step.opType()))
                .findFirst()
                .orElseThrow());
        assertNativeView(trace.steps().stream()
                .filter(step -> "SQUEEZE".equals(step.opType()))
                .findFirst()
                .orElseThrow());
        assertNativeSegmentScalar(trace.steps().stream()
                .filter(step -> "ADD".equals(step.opType()))
                .findFirst()
                .orElseThrow());
    }

    @Test
    void cpuNativeBf16CastReshapeCastReadsNativeViewInput() {
        Tensor out = tensor(new float[]{1f, -2f, 3.5f, -4.5f}, "cast_input")
                .cast(DataType.BFLOAT16)
                .reshape(4)
                .cast(DataType.FLOAT32);

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD);

        assertArrayEquals(new float[]{1f, -2f, 3.5f, -4.5f}, out.getFloat32Data(), 1.0e-3f);
        assertNativeView(trace.steps().stream()
                .filter(step -> "RESHAPE".equals(step.opType()))
                .findFirst()
                .orElseThrow());
        assertNativeCast(trace.steps().stream()
                .filter(step -> "CAST".equals(step.opType()) && step.dataType() == DataType.FLOAT32)
                .findFirst()
                .orElseThrow());
    }

    @Test
    void cpuNativeF64AddMulNegChainKeepsOutputsNativeAndPublishesValues() {
        Tensor out = f64(new double[]{1.0d, -2.0d, 3.0d, -4.0d}, "f64_a")
                .add(f64(new double[]{0.5d, 2.0d, -1.0d, 8.0d}, "f64_b"))
                .mul(f64(new double[]{2.0d, 3.0d, 4.0d, 5.0d}, "f64_scale"))
                .neg();

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{-3.0d, -0.0d, -8.0d, -20.0d}, out.getFloat64Data(), 1.0e-12d);
        assertNativeSegmentScalar(trace.steps().stream()
                .filter(step -> "ADD".equals(step.opType()))
                .findFirst()
                .orElseThrow());
        assertNativeSegmentScalar(trace.steps().stream()
                .filter(step -> "MUL".equals(step.opType()))
                .findFirst()
                .orElseThrow());
        assertNativeSegmentScalar(trace.steps().stream()
                .filter(step -> "NEG".equals(step.opType()))
                .findFirst()
                .orElseThrow());
    }

    @Test
    void cpuNativeF64MulScalarKeepsOutputNativeAndPublishesValues() {
        Tensor out = f64(new double[]{1.0d, -2.0d, 3.5d, -4.5d}, "f64_a").mul(0.25d);

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{0.25d, -0.5d, 0.875d, -1.125d}, out.getFloat64Data(), 1.0e-12d);
        assertNativeSegmentScalar(trace.steps().stream()
                .filter(step -> "MUL_SCALAR".equals(step.opType()))
                .findFirst()
                .orElseThrow());
    }

    @Test
    void cpuNativeF64SumAndMeanKeepOutputsNativeAndPublishValues() {
        Tensor sumAll = f64Matrix(new double[]{1.0d, 2.0d, 3.0d, 4.0d}, new int[]{2, 2}, "f64_sum_all").sum();
        Tensor meanAll = f64Matrix(new double[]{1.0d, 2.0d, 3.0d, 4.0d}, new int[]{2, 2}, "f64_mean_all").mean();
        Tensor sumColumns = f64Matrix(new double[]{1.0d, 2.0d, 3.0d, 4.0d}, new int[]{2, 2}, "f64_sum_columns").sum(0);
        Tensor meanRows = f64Matrix(new double[]{1.0d, 2.0d, 3.0d, 4.0d}, new int[]{2, 2}, "f64_mean_rows").mean(1);

        var sumAllTrace = CompiledGraph.compile(sumAll, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD);
        var meanAllTrace = CompiledGraph.compile(meanAll, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD);
        var sumColumnsTrace = CompiledGraph.compile(sumColumns, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD);
        var meanRowsTrace = CompiledGraph.compile(meanRows, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{10.0d}, sumAll.getFloat64Data(), 1.0e-12d);
        assertArrayEquals(new double[]{2.5d}, meanAll.getFloat64Data(), 1.0e-12d);
        assertArrayEquals(new double[]{4.0d, 6.0d}, sumColumns.getFloat64Data(), 1.0e-12d);
        assertArrayEquals(new double[]{1.5d, 3.5d}, meanRows.getFloat64Data(), 1.0e-12d);
        assertNativeReduction(sumAllTrace.steps().stream()
                .filter(step -> "SUM".equals(step.opType()))
                .findFirst()
                .orElseThrow());
        assertNativeReduction(meanAllTrace.steps().stream()
                .filter(step -> "MEAN".equals(step.opType()))
                .findFirst()
                .orElseThrow());
        assertNativeReduction(sumColumnsTrace.steps().stream()
                .filter(step -> "SUM".equals(step.opType()))
                .findFirst()
                .orElseThrow());
        assertNativeReduction(meanRowsTrace.steps().stream()
                .filter(step -> "MEAN".equals(step.opType()))
                .findFirst()
                .orElseThrow());
    }

    @Test
    void cpuNativeF32CompareOpsReadNativeInputsAndPublishBoolArrays() {
        Tensor left = tensor(new float[]{1.0f, 2.0f, 3.0f, 4.0f}, "compare_left");
        Tensor right = tensor(new float[]{2.0f, 2.0f, 1.0f, 4.0f}, "compare_right");

        assertNativeCompare(left.greaterThan(right), "GT", new byte[]{0, 0, 1, 0});
        assertNativeCompare(left.greaterOrEqual(right), "GE", new byte[]{0, 1, 1, 1});
        assertNativeCompare(left.lessThan(right), "LT", new byte[]{1, 0, 0, 0});
        assertNativeCompare(left.lessOrEqual(right), "LE", new byte[]{1, 1, 0, 1});
        assertNativeCompare(left.equalTo(right), "EQ", new byte[]{0, 1, 0, 1});
        assertNativeCompare(left.notEqualTo(right), "NE", new byte[]{1, 0, 1, 0});
    }

    @Test
    void cpuNativeF64CompareReadsNativeInputsAndPublishesBoolArray() {
        Tensor out = f64(new double[]{1.0d, 2.0d, 3.0d, 4.0d}, "compare_left")
                .lessOrEqual(f64(new double[]{2.0d, 2.0d, 1.0d, 4.0d}, "compare_right"));

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD);

        assertArrayEquals(new byte[]{1, 1, 0, 1}, out.getBoolData());
        assertNativeCompareTrace(trace.steps().stream()
                .filter(step -> "LE".equals(step.opType()))
                .findFirst()
                .orElseThrow());
    }

    @Test
    void cpuNativeCompareConditionFeedsNativeWhere() {
        Tensor condition = tensor(new float[]{1.0f, 4.0f, 3.0f, 0.0f}, "compare_left")
                .greaterThan(tensor(new float[]{0.0f, 5.0f, 2.0f, 1.0f}, "compare_right"));
        Tensor out = Tensor.where(
                condition,
                tensor(new float[]{10.0f, 20.0f, 30.0f, 40.0f}, "if_true"),
                tensor(new float[]{-10.0f, -20.0f, -30.0f, -40.0f}, "if_false")
        );

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD);

        assertArrayEquals(new float[]{10.0f, -20.0f, 30.0f, -40.0f}, out.getFloat32Data(), 1.0e-6f);
        assertNativeCompareTrace(trace.steps().stream()
                .filter(step -> "GT".equals(step.opType()))
                .findFirst()
                .orElseThrow());
        assertNativeSegmentScalar(trace.steps().stream()
                .filter(step -> "WHERE".equals(step.opType()))
                .findFirst()
                .orElseThrow());
    }

    @Test
    void unsupportedCpuNativeBroadcastMulFallsBackToArrayWithTraceReason() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor scale = vector(new float[]{2f, 3f}, "scale");
        Tensor out = a().matmul(b()).mul(scale);

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> mul = attrs(trace.steps().stream()
                .filter(step -> "MUL".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertEquals("CPU_NATIVE", mul.get("requestedCpuStorage"));
        assertEquals("CPU_ARRAY", mul.get("actualCpuStorage"));
        assertEquals("native-kernel-ineligible:mul-broadcast", mul.get("nativeCpuFallbackReason"));
        assertEquals("CPU_ARRAY", mul.get("storageResidency"));
    }

    @Test
    void requireNativeRejectsUnsupportedCpuNativeBroadcastMul() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor scale = vector(new float[]{2f, 3f}, "scale");
        Tensor out = a().matmul(b()).mul(scale);

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                        .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.REQUIRE_NATIVE), ExecutionMode.FORWARD, PublicationPolicy.NONE)
        );

        assertTrue(failure.getMessage().contains("Native CPU execution required"));
        assertTrue(failure.getMessage().contains("native-kernel-ineligible:mul-broadcast"));
    }

    @Test
    void unsupportedCpuNativeBroadcastSubFallsBackToArrayWithTraceReason() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor offset = vector(new float[]{1f, 2f}, "offset");
        Tensor out = a().matmul(b()).sub(offset);

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> sub = attrs(trace.steps().stream()
                .filter(step -> "SUB".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertEquals("CPU_NATIVE", sub.get("requestedCpuStorage"));
        assertEquals("CPU_ARRAY", sub.get("actualCpuStorage"));
        assertEquals("native-kernel-ineligible:sub-broadcast", sub.get("nativeCpuFallbackReason"));
        assertEquals("CPU_ARRAY", sub.get("storageResidency"));
    }

    @Test
    void requireNativeRejectsUnsupportedCpuNativeBroadcastDiv() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor divisor = vector(new float[]{1f, 2f}, "divisor");
        Tensor out = a().matmul(b()).div(divisor);

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                        .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.REQUIRE_NATIVE), ExecutionMode.FORWARD, PublicationPolicy.NONE)
        );

        assertTrue(failure.getMessage().contains("Native CPU execution required"));
        assertTrue(failure.getMessage().contains("native-kernel-ineligible:div-broadcast"));
    }

    @Test
    void unsupportedCpuNativeBroadcastAddFallsBackToArrayWithTraceReason() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor columnBias = matrix(new float[]{1f, -100f}, new int[]{2, 1}, "column_bias");
        Tensor out = a().matmul(b()).add(columnBias);

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> add = attrs(trace.steps().stream()
                .filter(step -> "ADD".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertEquals("CPU_NATIVE", add.get("requestedCpuStorage"));
        assertEquals("CPU_ARRAY", add.get("actualCpuStorage"));
        assertEquals("native-kernel-ineligible:add-broadcast", add.get("nativeCpuFallbackReason"));
        assertEquals("CPU_ARRAY", add.get("storageResidency"));
    }

    @Test
    void unsupportedCpuNativeF64BroadcastAddFallsBackToArrayWithTraceReason() {
        Tensor out = f64Matrix(new double[]{1.0d, 2.0d, 3.0d, 4.0d}, new int[]{2, 2}, "f64_a")
                .add(f64Matrix(new double[]{10.0d, 20.0d}, new int[]{2, 1}, "f64_column_bias"));

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> add = attrs(trace.steps().stream()
                .filter(step -> "ADD".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertEquals("CPU_NATIVE", add.get("requestedCpuStorage"));
        assertEquals("CPU_ARRAY", add.get("actualCpuStorage"));
        assertEquals("native-kernel-ineligible:add-broadcast", add.get("nativeCpuFallbackReason"));
        assertEquals("CPU_ARRAY", add.get("storageResidency"));
    }

    @Test
    void requireNativeRejectsUnsupportedCpuNativeF64BroadcastAdd() {
        Tensor out = f64Matrix(new double[]{1.0d, 2.0d, 3.0d, 4.0d}, new int[]{2, 2}, "f64_a")
                .add(f64Matrix(new double[]{10.0d, 20.0d}, new int[]{2, 1}, "f64_column_bias"));

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                        .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.REQUIRE_NATIVE), ExecutionMode.FORWARD, PublicationPolicy.NONE)
        );

        assertTrue(failure.getMessage().contains("Native CPU execution required"));
        assertTrue(failure.getMessage().contains("native-kernel-ineligible:add-broadcast"));
    }

    @Test
    void unsupportedCpuNativeBroadcastCompareFallsBackToArrayWithTraceReason() {
        Tensor out = tensor(new float[]{1.0f, 2.0f, 3.0f, 4.0f}, "compare_left")
                .greaterThan(vector(new float[]{2.0f, 3.0f}, "compare_bias"));

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> gt = attrs(trace.steps().stream()
                .filter(step -> "GT".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertEquals("CPU_NATIVE", gt.get("requestedCpuStorage"));
        assertEquals("CPU_ARRAY", gt.get("actualCpuStorage"));
        assertEquals("native-kernel-ineligible:gt-broadcast", gt.get("nativeCpuFallbackReason"));
        assertEquals("CPU_ARRAY", gt.get("storageResidency"));
    }

    @Test
    void requireNativeRejectsUnsupportedCpuNativeBroadcastCompare() {
        Tensor out = tensor(new float[]{1.0f, 2.0f, 3.0f, 4.0f}, "compare_left")
                .greaterThan(vector(new float[]{2.0f, 3.0f}, "compare_bias"));

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                        .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.REQUIRE_NATIVE), ExecutionMode.FORWARD, PublicationPolicy.NONE)
        );

        assertTrue(failure.getMessage().contains("Native CPU execution required"));
        assertTrue(failure.getMessage().contains("native-kernel-ineligible:gt-broadcast"));
    }

    @Test
    void unsupportedCpuNativeBroadcastWhereFallsBackToArrayWithTraceReason() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor condition = boolTensor(new byte[]{1, 0}, new int[]{2, 1}, "condition");
        Tensor fallback = tensor(new float[]{-100f, -100f, -100f, -100f}, "fallback");
        Tensor out = Tensor.where(condition, a().matmul(b()), fallback);

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> where = attrs(trace.steps().stream()
                .filter(step -> "WHERE".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertEquals("CPU_NATIVE", where.get("requestedCpuStorage"));
        assertEquals("CPU_ARRAY", where.get("actualCpuStorage"));
        assertEquals("native-kernel-ineligible:where-broadcast", where.get("nativeCpuFallbackReason"));
        assertEquals("CPU_ARRAY", where.get("storageResidency"));
    }

    @Test
    void requireNativeRejectsUnsupportedCpuNativeBroadcastWhere() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor condition = boolTensor(new byte[]{1, 0}, new int[]{2, 1}, "condition");
        Tensor fallback = tensor(new float[]{-100f, -100f, -100f, -100f}, "fallback");
        Tensor out = Tensor.where(condition, a().matmul(b()), fallback);

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                        .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.REQUIRE_NATIVE), ExecutionMode.FORWARD, PublicationPolicy.NONE)
        );

        assertTrue(failure.getMessage().contains("Native CPU execution required"));
        assertTrue(failure.getMessage().contains("native-kernel-ineligible:where-broadcast"));
    }

    @Test
    void unsupportedCpuNativeStridedSumFallsBackToArrayWithTraceReason() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor out = a().matmul(b()).transpose().sum(1);

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> sum = attrs(trace.steps().stream()
                .filter(step -> "SUM".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertEquals("CPU_NATIVE", sum.get("requestedCpuStorage"));
        assertEquals("CPU_ARRAY", sum.get("actualCpuStorage"));
        assertEquals("native-kernel-ineligible:sum-strided", sum.get("nativeCpuFallbackReason"));
        assertEquals("CPU_ARRAY", sum.get("storageResidency"));
    }

    @Test
    void requireNativeRejectsUnsupportedCpuNativeStridedSum() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor out = a().matmul(b()).transpose().sum(1);

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                        .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.REQUIRE_NATIVE), ExecutionMode.FORWARD, PublicationPolicy.NONE)
        );

        assertTrue(failure.getMessage().contains("Native CPU execution required"));
        assertTrue(failure.getMessage().contains("native-kernel-ineligible:sum-strided"));
    }

    @Test
    void unsupportedCpuNativeF64StridedSumFallsBackToArrayWithTraceReason() {
        Tensor out = f64Matrix(new double[]{1.0d, 2.0d, 3.0d, 4.0d}, new int[]{2, 2}, "f64_a")
                .transpose()
                .sum(1);

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> sum = attrs(trace.steps().stream()
                .filter(step -> "SUM".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertEquals("CPU_NATIVE", sum.get("requestedCpuStorage"));
        assertEquals("CPU_ARRAY", sum.get("actualCpuStorage"));
        assertEquals("native-kernel-ineligible:sum-strided", sum.get("nativeCpuFallbackReason"));
        assertEquals("CPU_ARRAY", sum.get("storageResidency"));
    }

    @Test
    void requireNativeRejectsUnsupportedCpuNativeF64StridedSum() {
        Tensor out = f64Matrix(new double[]{1.0d, 2.0d, 3.0d, 4.0d}, new int[]{2, 2}, "f64_a")
                .transpose()
                .sum(1);

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                        .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.REQUIRE_NATIVE), ExecutionMode.FORWARD, PublicationPolicy.NONE)
        );

        assertTrue(failure.getMessage().contains("Native CPU execution required"));
        assertTrue(failure.getMessage().contains("native-kernel-ineligible:sum-strided"));
    }

    @Test
    void unsupportedCpuNativeCastFallsBackToArrayWithTraceReason() {
        Tensor out = tensor(new float[]{1f, 2f, 3f, 4f}, "cast_input").cast(DataType.FLOAT64);

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> cast = attrs(trace.steps().stream()
                .filter(step -> "CAST".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertEquals("CPU_NATIVE", cast.get("requestedCpuStorage"));
        assertEquals("CPU_ARRAY", cast.get("actualCpuStorage"));
        assertEquals("native-kernel-ineligible:cast-dtype", cast.get("nativeCpuFallbackReason"));
        assertEquals("CPU_ARRAY", cast.get("storageResidency"));
    }

    @Test
    void requireNativeRejectsUnsupportedCpuNativeCast() {
        Tensor out = tensor(new float[]{1f, 2f, 3f, 4f}, "cast_input").cast(DataType.FLOAT64);

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                        .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.REQUIRE_NATIVE), ExecutionMode.FORWARD, PublicationPolicy.NONE)
        );

        assertTrue(failure.getMessage().contains("Native CPU execution required"));
        assertTrue(failure.getMessage().contains("native-kernel-ineligible:cast-dtype"));
    }

    @Test
    void requireNativeRejectsUnsupportedCpuNativeBroadcastAdd() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor columnBias = matrix(new float[]{1f, -100f}, new int[]{2, 1}, "column_bias");
        Tensor out = a().matmul(b()).add(columnBias);

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                        .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.REQUIRE_NATIVE), ExecutionMode.FORWARD, PublicationPolicy.NONE)
        );

        assertTrue(failure.getMessage().contains("Native CPU execution required"));
        assertTrue(failure.getMessage().contains("native-kernel-ineligible:add-broadcast"));
    }

    @Test
    void unsupportedCpuNativeElementwiseFallsBackToArrayWithTraceReason() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor out = a().matmul(b()).log();

        var trace = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> log = attrs(trace.steps().stream()
                .filter(step -> "LOG".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertEquals("CPU_NATIVE", log.get("requestedCpuStorage"));
        assertEquals("CPU_ARRAY", log.get("actualCpuStorage"));
        assertEquals("NATIVE_UNSUPPORTED", log.get("nativeCpuKernelStatus"));
        assertEquals("ARRAY_ONLY", log.get("nativeCpuKernelFamily"));
        assertEquals("native-kernel-unsupported:log", log.get("nativeCpuFallbackReason"));
        assertEquals("CPU_ARRAY", log.get("storageResidency"));
    }

    @Test
    void requireNativeRejectsUnsupportedCpuNativeElementwise() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor out = a().matmul(b()).log();

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                        .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.REQUIRE_NATIVE), ExecutionMode.FORWARD, PublicationPolicy.NONE)
        );

        assertTrue(failure.getMessage().contains("Native CPU execution required"));
        assertTrue(failure.getMessage().contains("native-kernel-unsupported:log"));
    }

    @Test
    void autoStorageDoesNotUseNativeElementwiseSlice() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor out = a().matmul(b()).add(vector(new float[]{1f, -100f}, "bias"));

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.AUTO, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> add = attrs(trace.steps().stream()
                .filter(step -> "ADD".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertFalse(add.containsKey("nativeCpuKernelStatus"));
        assertEquals("CPU_ARRAY", add.get("storageResidency"));
    }

    @Test
    void autoStorageDoesNotUseNativeMulSlice() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor out = a().matmul(b()).mul(tensor(new float[]{2f, 3f, 4f, 5f}, "scale"));

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.AUTO, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> mul = attrs(trace.steps().stream()
                .filter(step -> "MUL".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertFalse(mul.containsKey("nativeCpuKernelStatus"));
        assertEquals("CPU_ARRAY", mul.get("storageResidency"));
    }

    @Test
    void autoStorageDoesNotUseNativeSubDivOrNegSlice() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor out = a().matmul(b())
                .sub(tensor(new float[]{1f, 2f, 3f, 4f}, "offset"))
                .div(tensor(new float[]{1f, 2f, 4f, 5f}, "divisor"))
                .neg();

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.AUTO, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> sub = attrs(trace.steps().stream()
                .filter(step -> "SUB".equals(step.opType()))
                .findFirst()
                .orElseThrow());
        Map<String, Object> div = attrs(trace.steps().stream()
                .filter(step -> "DIV".equals(step.opType()))
                .findFirst()
                .orElseThrow());
        Map<String, Object> neg = attrs(trace.steps().stream()
                .filter(step -> "NEG".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertFalse(sub.containsKey("nativeCpuKernelStatus"));
        assertEquals("CPU_ARRAY", sub.get("storageResidency"));
        assertFalse(div.containsKey("nativeCpuKernelStatus"));
        assertEquals("CPU_ARRAY", div.get("storageResidency"));
        assertFalse(neg.containsKey("nativeCpuKernelStatus"));
        assertEquals("CPU_ARRAY", neg.get("storageResidency"));
    }

    @Test
    void autoStorageDoesNotUseNativeMulScalarSlice() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor out = a().matmul(b()).mul(0.5d);

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.AUTO, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> mulScalar = attrs(trace.steps().stream()
                .filter(step -> "MUL_SCALAR".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertFalse(mulScalar.containsKey("nativeCpuKernelStatus"));
        assertEquals("CPU_ARRAY", mulScalar.get("storageResidency"));
    }

    @Test
    void autoStorageDoesNotUseNativeWhereSlice() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor condition = boolTensor(new byte[]{1, 0, 1, 0}, new int[]{2, 2}, "condition");
        Tensor fallback = tensor(new float[]{-100f, -100f, -100f, -100f}, "fallback");
        Tensor out = Tensor.where(condition, a().matmul(b()), fallback);

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.AUTO, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> where = attrs(trace.steps().stream()
                .filter(step -> "WHERE".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertFalse(where.containsKey("nativeCpuKernelStatus"));
        assertEquals("CPU_ARRAY", where.get("storageResidency"));
    }

    @Test
    void autoStorageDoesNotUseNativeSumSlice() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor out = a().matmul(b()).sum();

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.AUTO, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> sum = attrs(trace.steps().stream()
                .filter(step -> "SUM".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertFalse(sum.containsKey("nativeCpuKernelStatus"));
        assertEquals("CPU_ARRAY", sum.get("storageResidency"));
    }

    @Test
    void autoStorageDoesNotUseNativeCompareSlice() {
        Tensor out = tensor(new float[]{1.0f, 2.0f, 3.0f, 4.0f}, "compare_left")
                .greaterThan(tensor(new float[]{2.0f, 2.0f, 1.0f, 4.0f}, "compare_right"));

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.AUTO, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> gt = attrs(trace.steps().stream()
                .filter(step -> "GT".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertFalse(gt.containsKey("nativeCpuKernelStatus"));
        assertEquals("CPU_ARRAY", gt.get("storageResidency"));
    }

    @Test
    void autoStorageDoesNotUseNativeCastSlice() {
        Tensor out = tensor(new float[]{1f, 2f, 3f, 4f}, "cast_input").cast(DataType.BFLOAT16);

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.AUTO, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> cast = attrs(trace.steps().stream()
                .filter(step -> "CAST".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertFalse(cast.containsKey("nativeCpuKernelStatus"));
        assertEquals("CPU_ARRAY", cast.get("storageResidency"));
    }

    @Test
    void autoStorageDoesNotUseNativeF64ElementwiseOrReductionSlices() {
        Tensor out = f64(new double[]{1.0d, 2.0d, 3.0d, 4.0d}, "f64_a")
                .add(f64(new double[]{4.0d, 3.0d, 2.0d, 1.0d}, "f64_b"))
                .sum();

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.AUTO, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> add = attrs(trace.steps().stream()
                .filter(step -> "ADD".equals(step.opType()))
                .findFirst()
                .orElseThrow());
        Map<String, Object> sum = attrs(trace.steps().stream()
                .filter(step -> "SUM".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertFalse(add.containsKey("nativeCpuKernelStatus"));
        assertEquals("CPU_ARRAY", add.get("storageResidency"));
        assertFalse(sum.containsKey("nativeCpuKernelStatus"));
        assertEquals("CPU_ARRAY", sum.get("storageResidency"));
    }

    @Test
    void autoStorageDoesNotUseNativeViewSlice() {
        Tensor out = tensor(new float[]{1f, 2f, 3f, 4f}, "view_input")
                .reshape(4)
                .relu();

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.AUTO, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> reshape = attrs(trace.steps().stream()
                .filter(step -> "RESHAPE".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertFalse(reshape.containsKey("nativeCpuKernelStatus"));
        assertEquals("CPU_ARRAY", reshape.get("storageResidency"));
    }

    @Test
    void selectAndSliceStayOutsideNativeViewSlice() {
        Tensor selected = tensor(new float[]{1f, 2f, 3f, 4f}, "select_input")
                .select(0, 1)
                .relu();
        Tensor sliced = tensor(new float[]{1f, 2f, 3f, 4f}, "slice_input")
                .slice(new int[]{0, 0}, new int[]{1, 2}, new int[]{0, 1}, new int[]{1, 1})
                .relu();

        var selectTrace = CompiledGraph.compile(selected, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);
        var sliceTrace = CompiledGraph.compile(sliced, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> select = attrs(selectTrace.steps().stream()
                .filter(step -> "SELECT".equals(step.opType()))
                .findFirst()
                .orElseThrow());
        Map<String, Object> slice = attrs(sliceTrace.steps().stream()
                .filter(step -> "SLICE".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertFalse(select.containsKey("nativeCpuKernelStatus"));
        assertEquals("CPU_ARRAY", select.get("storageResidency"));
        assertFalse(slice.containsKey("nativeCpuKernelStatus"));
        assertEquals("CPU_ARRAY", slice.get("storageResidency"));
    }

    private static RuntimeConfig runtime(CpuStorageProfile storageProfile, NativeCpuFailurePolicy failurePolicy) {
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
                .withNativeCpuFailurePolicy(failurePolicy);
    }

    private static CompileConfig nativeElementwiseCompileConfig() {
        return CompileConfig.noGraphOptimizationBaseline()
                .withSemanticCanonicalization(SemanticCanonicalizationConfig.disabled())
                .withRegionOptimization(RegionOptimizationConfig.disabled());
    }

    private static Tensor a() {
        return tensor(new float[]{1f, 2f, 3f, 4f}, "a");
    }

    private static Tensor b() {
        return tensor(new float[]{5f, 6f, 7f, 8f}, "b");
    }

    private static Tensor tensor(float[] values, String label) {
        return new Tensor(values, new int[]{2, 2}, null, label, DataType.FLOAT32);
    }

    private static Tensor vector(float[] values, String label) {
        return new Tensor(values, new int[]{values.length}, null, label, DataType.FLOAT32);
    }

    private static Tensor matrix(float[] values, int[] shape, String label) {
        return new Tensor(values, shape, null, label, DataType.FLOAT32);
    }

    private static Tensor f64(double[] values, String label) {
        return new Tensor(values, new int[]{2, 2}, null, label, DataType.FLOAT64);
    }

    private static Tensor f64Matrix(double[] values, int[] shape, String label) {
        return new Tensor(values, shape, null, label, DataType.FLOAT64);
    }

    private static Tensor boolTensor(byte[] values, int[] shape, String label) {
        return new Tensor(values, shape, null, label, DataType.BOOL);
    }

    private static void assertNativeReduction(ExecutionStepTrace step) {
        Map<String, Object> reduction = attrs(step);
        assertEquals("CPU_NATIVE", reduction.get("actualCpuStorage"));
        assertEquals("NATIVE_CORRECT_BUT_SLOW", reduction.get("nativeCpuKernelStatus"));
        assertEquals("SEGMENT_SCALAR", reduction.get("nativeCpuKernelFamily"));
        assertEquals("", reduction.get("nativeCpuFallbackReason"));
        assertEquals("CPU_NATIVE", reduction.get("storageResidency"));
    }

    private static void assertNativeSegmentScalar(ExecutionStepTrace step) {
        Map<String, Object> attrs = attrs(step);
        assertEquals("CPU_NATIVE", attrs.get("actualCpuStorage"));
        assertEquals("NATIVE_CORRECT_BUT_SLOW", attrs.get("nativeCpuKernelStatus"));
        assertEquals("SEGMENT_SCALAR", attrs.get("nativeCpuKernelFamily"));
        assertEquals("", attrs.get("nativeCpuFallbackReason"));
        assertEquals("CPU_NATIVE", attrs.get("storageResidency"));
    }

    private static void assertNativeCompare(Tensor out, String opType, byte[] expected) {
        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD);

        assertArrayEquals(expected, out.getBoolData());
        assertNativeCompareTrace(trace.steps().stream()
                .filter(step -> opType.equals(step.opType()))
                .findFirst()
                .orElseThrow());
    }

    private static void assertNativeCompareTrace(ExecutionStepTrace step) {
        Map<String, Object> compare = attrs(step);
        assertEquals("CPU_NATIVE", compare.get("requestedCpuStorage"));
        assertEquals("CPU_ARRAY", compare.get("actualCpuStorage"));
        assertEquals("NATIVE_CORRECT_BUT_SLOW", compare.get("nativeCpuKernelStatus"));
        assertEquals("SEGMENT_SCALAR", compare.get("nativeCpuKernelFamily"));
        assertEquals("", compare.get("nativeCpuFallbackReason"));
        assertEquals("CPU_ARRAY", compare.get("storageResidency"));
    }

    private static void assertNativeCast(ExecutionStepTrace step) {
        Map<String, Object> cast = attrs(step);
        assertEquals("CPU_NATIVE", cast.get("actualCpuStorage"));
        assertEquals("NATIVE_CORRECT_BUT_SLOW", cast.get("nativeCpuKernelStatus"));
        assertEquals("SEGMENT_SCALAR", cast.get("nativeCpuKernelFamily"));
        assertEquals("", cast.get("nativeCpuFallbackReason"));
        assertEquals("CPU_NATIVE", cast.get("storageResidency"));
    }

    private static void assertNativeView(ExecutionStepTrace step) {
        Map<String, Object> view = attrs(step);
        assertEquals("CPU_NATIVE", view.get("cpuStorageProfile"));
        assertEquals("CPU_NATIVE", view.get("requestedCpuStorage"));
        assertEquals("CPU_NATIVE", view.get("actualCpuStorage"));
        assertEquals("VIEW_ONLY", view.get("nativeCpuKernelStatus"));
        assertEquals("VIEW_ONLY", view.get("nativeCpuKernelFamily"));
        assertEquals("", view.get("nativeCpuFallbackReason"));
        assertEquals("CPU_NATIVE", view.get("storageResidency"));
    }

    private static Map<String, Object> attrs(ExecutionStepTrace step) {
        return step.metadata().attributes();
    }
}
