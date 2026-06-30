package backend.cpu.kernels.linalg;

import backend.contract.ComputeBackend;
import backend.cpu.execution.CpuKernelContext;
import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.plan.CpuExecutionMode;
import backend.cpu.plan.CpuLayoutPlan;
import backend.cpu.plan.CpuNodeExecutionPlan;
import backend.cpu.plan.layout.StridedLayoutDecision;
import backend.cpu.plan.linalg.attention.ResolvedAttentionHints;
import backend.cpu.plan.linalg.attention.ResolvedScaledDotProductAttentionPlan;
import backend.cpu.storage.CpuStorageView;
import backend.runtime.ExecutionContext;
import backend.runtime.ExecutionMode;
import config.runtime.RuntimeConfig;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import operations.linalg.scaledDotProductAttention;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.dtype.TensorDTypeOps;

import java.lang.foreign.MemorySegment;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

class CpuScaledDotProductAttentionKernelStorageViewTest {
    @Test
    void f64MemorySegmentForwardHonorsOffsetsStridesAndBoolMask() {
        int[] shape = {1, 2, 2};
        int[] strides = {6, 3, 1};
        double[] queryStorage = {-9.0d, 1.0d, 0.0d, -9.0d, 0.0d, 1.0d};
        double[] keyStorage = {-9.0d, 1.0d, 0.0d, -9.0d, 0.0d, 1.0d};
        double[] valueStorage = {-9.0d, 10.0d, 1.0d, -9.0d, 1.0d, 10.0d};
        byte[] maskStorage = {1, 0, 1, 1};
        double[] outputStorage = {-7.0d, -7.0d, -7.0d, -7.0d, -7.0d, -7.0d};
        scaledDotProductAttention op = new scaledDotProductAttention(1.0d, true);

        Tensor query = tensor("q", DataType.FLOAT64, shape, strides, 1);
        Tensor key = tensor("k", DataType.FLOAT64, shape, strides, 1);
        Tensor value = tensor("v", DataType.FLOAT64, shape, strides, 1);
        Tensor mask = tensor("mask", DataType.BOOL, shape, new int[]{4, 2, 1}, 0);
        Tensor output = tensor("out", DataType.FLOAT64, shape, strides, 1, List.of(query, key, value, mask), op);

        new CpuScaledDotProductAttentionKernel().execute(call(
                op,
                List.of(query, key, value, mask),
                output,
                List.of(
                        segment(DataType.FLOAT64, MemorySegment.ofArray(queryStorage), shape, strides, 1),
                        segment(DataType.FLOAT64, MemorySegment.ofArray(keyStorage), shape, strides, 1),
                        segment(DataType.FLOAT64, MemorySegment.ofArray(valueStorage), shape, strides, 1),
                        segment(DataType.BOOL, MemorySegment.ofArray(maskStorage), shape, new int[]{4, 2, 1}, 0)
                ),
                segment(DataType.FLOAT64, MemorySegment.ofArray(outputStorage), shape, strides, 1)));

        double e = Math.exp(1.0d);
        double denom = e + 1.0d;
        assertArrayEquals(new double[]{
                -7.0d,
                10.0d,
                1.0d,
                -7.0d,
                (10.0d + e) / denom,
                (1.0d + 10.0d * e) / denom
        }, outputStorage, 1e-6);
    }

    @Test
    void f32MixedArrayAndMemorySegmentForwardMatchesDenseParity() {
        int[] shape = {1, 2, 2};
        float[] queryStorage = {1.0f, 0.0f, 0.0f, 1.0f};
        float[] keyStorage = {-9.0f, 1.0f, 0.0f, -9.0f, 0.0f, 1.0f};
        float[] valueStorage = {10.0f, 1.0f, 1.0f, 10.0f};
        float[] outputStorage = {-7.0f, -7.0f, -7.0f, -7.0f, -7.0f, -7.0f};
        scaledDotProductAttention op = new scaledDotProductAttention(1.0d, false);

        Tensor query = tensor("q", DataType.FLOAT32, shape, new int[]{4, 2, 1}, 0);
        Tensor key = tensor("k", DataType.FLOAT32, shape, new int[]{6, 3, 1}, 1);
        Tensor value = tensor("v", DataType.FLOAT32, shape, new int[]{4, 2, 1}, 0);
        Tensor output = tensor("out", DataType.FLOAT32, shape, new int[]{6, 3, 1}, 1, List.of(query, key, value), op);

        new CpuScaledDotProductAttentionKernel().execute(call(
                op,
                List.of(query, key, value),
                output,
                List.of(
                        array(DataType.FLOAT32, queryStorage, shape, new int[]{4, 2, 1}, 0),
                        segment(DataType.FLOAT32, MemorySegment.ofArray(keyStorage), shape, new int[]{6, 3, 1}, 1),
                        array(DataType.FLOAT32, valueStorage, shape, new int[]{4, 2, 1}, 0)
                ),
                segment(DataType.FLOAT32, MemorySegment.ofArray(outputStorage), shape, new int[]{6, 3, 1}, 1)));

        double e = Math.exp(1.0d);
        double denom = e + 1.0d;
        assertArrayEquals(new float[]{
                -7.0f,
                (float) ((10.0d * e + 1.0d) / denom),
                (float) ((1.0d * e + 10.0d) / denom),
                -7.0f,
                (float) ((10.0d + e) / denom),
                (float) ((1.0d + 10.0d * e) / denom)
        }, outputStorage, 1e-5f);
    }

    @Test
    void bf16MemorySegmentForwardWritesBf16OutputAndF32WeightsCache() {
        int[] shape = {1, 2, 2};
        short[] queryStorage = paddedBf16(-9.0f, 1.0f, 0.0f, -9.0f, 0.0f, 1.0f);
        short[] keyStorage = paddedBf16(-9.0f, 1.0f, 0.0f, -9.0f, 0.0f, 1.0f);
        short[] valueStorage = paddedBf16(-9.0f, 10.0f, 1.0f, -9.0f, 1.0f, 10.0f);
        short sentinel = TensorDTypeOps.toBFloat16Bits(-7.0f);
        short[] outputStorage = {sentinel, sentinel, sentinel, sentinel, sentinel, sentinel};
        scaledDotProductAttention op = new scaledDotProductAttention(1.0d, false);

        Tensor query = tensor("q", DataType.BFLOAT16, shape, new int[]{6, 3, 1}, 1);
        query.setRequiresGrad(true);
        Tensor key = tensor("k", DataType.BFLOAT16, shape, new int[]{6, 3, 1}, 1);
        Tensor value = tensor("v", DataType.BFLOAT16, shape, new int[]{6, 3, 1}, 1);
        Tensor output = tensor("out", DataType.BFLOAT16, shape, new int[]{6, 3, 1}, 1, List.of(query, key, value), op);
        CpuKernelContext context = context(op, plan(DataType.BFLOAT16));

        new CpuScaledDotProductAttentionKernel().execute(new CpuKernelCall(
                op,
                List.of(query, key, value),
                output,
                List.of(
                        segment(DataType.BFLOAT16, MemorySegment.ofArray(queryStorage), shape, new int[]{6, 3, 1}, 1),
                        segment(DataType.BFLOAT16, MemorySegment.ofArray(keyStorage), shape, new int[]{6, 3, 1}, 1),
                        segment(DataType.BFLOAT16, MemorySegment.ofArray(valueStorage), shape, new int[]{6, 3, 1}, 1)
                ),
                segment(DataType.BFLOAT16, MemorySegment.ofArray(outputStorage), shape, new int[]{6, 3, 1}, 1),
                plan(DataType.BFLOAT16),
                context,
                null));

        double e = Math.exp(1.0d);
        double denom = e + 1.0d;
        assertArrayEquals(new float[]{
                -7.0f,
                (float) ((10.0d * e + 1.0d) / denom),
                (float) ((1.0d * e + 10.0d) / denom),
                -7.0f,
                (float) ((10.0d + e) / denom),
                (float) ((1.0d + 10.0d * e) / denom)
        }, decodeBf16(outputStorage), 2e-2f);

        ScaledDotProductAttentionRuntimeCache cache =
                context.runtimeStateFor(output, ScaledDotProductAttentionRuntimeCache.class);
        assertNotNull(cache);
        assertArrayEquals(new double[]{
                e / denom,
                1.0d / denom,
                1.0d / denom,
                e / denom
        }, cache.weights().toDoubleArrayCopy(), 1e-5);
    }

    private static CpuKernelCall call(
            scaledDotProductAttention op,
            List<Tensor> inputTensors,
            Tensor output,
            List<CpuStorageView> inputViews,
            CpuStorageView outputView
    ) {
        CpuNodeExecutionPlan plan = plan(output.getDataType());
        return new CpuKernelCall(op, inputTensors, output, inputViews, outputView, plan, context(op, plan), null);
    }

    private static CpuKernelContext context(scaledDotProductAttention op, CpuNodeExecutionPlan plan) {
        CompiledNodeExecutionMetadata metadata = new CompiledNodeExecutionMetadata(ComputeBackend.CPU, op, List.of(), null);
        return new CpuKernelContext(
                1,
                List.of(0, 1, 2, 3),
                plan,
                ExecutionContext.fromRuntimeConfig(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD),
                metadata,
                List.of(),
                op);
    }

    private static CpuNodeExecutionPlan plan(DataType dtype) {
        ResolvedAttentionHints hints = new ResolvedAttentionHints(CpuExecutionMode.SCALAR, 1, 1, 1);
        return new CpuNodeExecutionPlan(
                new CpuLayoutPlan(StridedLayoutDecision.NONE, dtype, 0, null, null, List.of()),
                null,
                false,
                1,
                0,
                null,
                null,
                null,
                null,
                new ResolvedScaledDotProductAttentionPlan(hints, null, null, null, null, null));
    }

    private static Tensor tensor(String label, DataType dtype, int[] shape, int[] strides, int storageOffset) {
        return tensor(label, dtype, shape, strides, storageOffset, List.of(), null);
    }

    private static Tensor tensor(
            String label,
            DataType dtype,
            int[] shape,
            int[] strides,
            int storageOffset,
            List<Tensor> previous,
            scaledDotProductAttention op
    ) {
        return new Tensor(shape, strides, storageOffset, previous, op, label, dtype);
    }

    private static CpuStorageView array(DataType dtype, Object storage, int[] shape, int[] strides, int storageOffset) {
        return CpuStorageView.array(dtype, storage, shape, strides, storageOffset, logicalSize(shape));
    }

    private static CpuStorageView segment(
            DataType dtype,
            MemorySegment segment,
            int[] shape,
            int[] strides,
            int storageOffset
    ) {
        return CpuStorageView.segment(dtype, segment, shape, strides, storageOffset, logicalSize(shape));
    }

    private static int logicalSize(int[] shape) {
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        return size;
    }

    private static short[] paddedBf16(float... values) {
        short[] out = new short[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = TensorDTypeOps.toBFloat16Bits(values[i]);
        }
        return out;
    }

    private static float[] decodeBf16(short[] values) {
        float[] out = new float[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = TensorDTypeOps.fromBFloat16Bits(values[i]);
        }
        return out;
    }
}
