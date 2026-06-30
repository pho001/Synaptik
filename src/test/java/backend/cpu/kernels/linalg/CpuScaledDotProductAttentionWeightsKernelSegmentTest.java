package backend.cpu.kernels.linalg;

import backend.contract.ComputeBackend;
import backend.cpu.execution.CpuKernelContext;
import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.plan.CpuLayoutPlan;
import backend.cpu.plan.CpuNodeExecutionPlan;
import backend.cpu.plan.layout.StridedLayoutDecision;
import backend.cpu.storage.CpuStorageView;
import backend.runtime.ExecutionContext;
import backend.runtime.ExecutionMode;
import config.runtime.RuntimeConfig;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import operations.linalg.scaledDotProductAttentionWeights;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.dtype.TensorDTypeOps;

import java.lang.foreign.MemorySegment;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class CpuScaledDotProductAttentionWeightsKernelSegmentTest {
    @Test
    void exportsFloat64WeightsToStridedMemorySegmentOutput() {
        double[] outputStorage = {-7.0d, -7.0d, -7.0d, -7.0d, -7.0d, -7.0d};
        Tensor cached = new Tensor(new double[]{0.1d, 0.2d, 0.3d, 0.4d},
                new int[]{2, 2}, List.of(), "cached", DataType.FLOAT64);

        new CpuScaledDotProductAttentionWeightsKernel().execute(call(
                cached,
                outputTensor(DataType.FLOAT64),
                segment(DataType.FLOAT64, MemorySegment.ofArray(outputStorage))));

        assertArrayEquals(new double[]{-7.0d, 0.1d, 0.2d, 0.3d, 0.4d, -7.0d}, outputStorage, 0.0d);
    }

    @Test
    void exportsFloat32WeightsToStridedMemorySegmentOutput() {
        float[] outputStorage = {-7.0f, -7.0f, -7.0f, -7.0f, -7.0f, -7.0f};
        Tensor cached = new Tensor(new float[]{0.1f, 0.2f, 0.3f, 0.4f},
                new int[]{2, 2}, List.of(), "cached", DataType.FLOAT32);

        new CpuScaledDotProductAttentionWeightsKernel().execute(call(
                cached,
                outputTensor(DataType.FLOAT32),
                segment(DataType.FLOAT32, MemorySegment.ofArray(outputStorage))));

        assertArrayEquals(new float[]{-7.0f, 0.1f, 0.2f, 0.3f, 0.4f, -7.0f}, outputStorage, 0.0f);
    }

    @Test
    void exportsBfloat16WeightsToStridedMemorySegmentOutput() {
        short sentinel = TensorDTypeOps.toBFloat16Bits(-7.0f);
        short[] outputStorage = {sentinel, sentinel, sentinel, sentinel, sentinel, sentinel};
        short[] cachedBits = bf16(0.125f, 0.25f, 0.5f, 0.75f);
        Tensor cached = new Tensor(cachedBits, new int[]{2, 2}, List.of(), "cached", DataType.BFLOAT16);

        new CpuScaledDotProductAttentionWeightsKernel().execute(call(
                cached,
                outputTensor(DataType.BFLOAT16),
                segment(DataType.BFLOAT16, MemorySegment.ofArray(outputStorage))));

        assertArrayEquals(new short[]{sentinel, cachedBits[0], cachedBits[1], cachedBits[2], cachedBits[3], sentinel},
                outputStorage);
    }

    @Test
    void exportsFloat32CachedWeightsToBfloat16MemorySegmentOutput() {
        short sentinel = TensorDTypeOps.toBFloat16Bits(-7.0f);
        short[] outputStorage = {sentinel, sentinel, sentinel, sentinel, sentinel, sentinel};
        Tensor cached = new Tensor(new float[]{0.125f, 0.25f, 0.5f, 0.75f},
                new int[]{2, 2}, List.of(), "cached", DataType.FLOAT32);

        new CpuScaledDotProductAttentionWeightsKernel().execute(call(
                cached,
                outputTensor(DataType.BFLOAT16),
                segment(DataType.BFLOAT16, MemorySegment.ofArray(outputStorage))));

        assertArrayEquals(new short[]{
                sentinel,
                TensorDTypeOps.toBFloat16Bits(0.125f),
                TensorDTypeOps.toBFloat16Bits(0.25f),
                TensorDTypeOps.toBFloat16Bits(0.5f),
                TensorDTypeOps.toBFloat16Bits(0.75f),
                sentinel
        }, outputStorage);
    }

    private static CpuKernelCall call(Tensor cached, Tensor output, CpuStorageView outputView) {
        scaledDotProductAttentionWeights op = new scaledDotProductAttentionWeights();
        Tensor attentionOut = new Tensor(new int[]{2, 2}, List.of(), "attentionOut", output.getDataType());
        CpuNodeExecutionPlan plan = plan(output.getDataType());
        CompiledNodeExecutionMetadata metadata = new CompiledNodeExecutionMetadata(ComputeBackend.CPU, op, List.of(), null);
        CpuKernelContext context = new CpuKernelContext(
                1,
                List.of(0),
                plan,
                ExecutionContext.fromRuntimeConfig(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD),
                metadata,
                List.of(),
                op);
        context.putRuntimeState(attentionOut, new ScaledDotProductAttentionRuntimeCache(cached));
        return new CpuKernelCall(op, List.of(attentionOut), output, List.of(), outputView, plan, context, null);
    }

    private static Tensor outputTensor(DataType dtype) {
        scaledDotProductAttentionWeights op = new scaledDotProductAttentionWeights();
        Tensor attentionOut = new Tensor(new int[]{2, 2}, List.of(), "attentionOut", dtype);
        return new Tensor(new int[]{2, 2}, new int[]{2, 1}, 1, List.of(attentionOut), op, "output", dtype);
    }

    private static CpuStorageView segment(DataType dtype, MemorySegment segment) {
        return CpuStorageView.segment(dtype, segment, new int[]{2, 2}, new int[]{2, 1}, 1, 4);
    }

    private static CpuNodeExecutionPlan plan(DataType dtype) {
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
                null);
    }

    private static short[] bf16(float... values) {
        short[] bits = new short[values.length];
        for (int i = 0; i < values.length; i++) {
            bits[i] = TensorDTypeOps.toBFloat16Bits(values[i]);
        }
        return bits;
    }
}
