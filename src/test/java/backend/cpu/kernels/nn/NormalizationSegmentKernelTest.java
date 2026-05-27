package backend.cpu.kernels.nn;

import backend.ComputeBackend;
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
import operations.Operation;
import operations.normalization.layerNorm;
import operations.normalization.rmsNorm;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.dtype.TensorDTypeOps;

import java.lang.foreign.MemorySegment;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class NormalizationSegmentKernelTest {
    @Test
    void layerNormReadsAndWritesContiguousF32MemorySegments() {
        layerNorm op = new layerNorm(1, 1e-5);
        Tensor input = new Tensor(new int[]{2, 3}, new int[]{3, 1}, 1,
                null, null, "input", DataType.FLOAT32);
        Tensor gamma = new Tensor(new int[]{3}, new int[]{1}, 1,
                null, null, "gamma", DataType.FLOAT32);
        Tensor beta = new Tensor(new int[]{3}, new int[]{1}, 1,
                null, null, "beta", DataType.FLOAT32);
        Tensor output = new Tensor(new int[]{2, 3}, new int[]{3, 1}, 1,
                List.of(input, gamma, beta), op, "output", DataType.FLOAT32);

        float[] inputStorage = {-99.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, -99.0f};
        float[] gammaStorage = {-9.0f, 1.25f, 0.75f, 1.5f};
        float[] betaStorage = {-9.0f, 0.5f, -0.25f, 0.125f};
        float[] outputStorage = {-7.0f, -7.0f, -7.0f, -7.0f, -7.0f, -7.0f, -7.0f, -7.0f};

        new CpuLayerNormKernel().execute(call(
                op,
                List.of(input, gamma, beta),
                output,
                List.of(
                        f32Segment(inputStorage, new int[]{2, 3}, new int[]{3, 1}, 1),
                        f32Segment(gammaStorage, new int[]{3}, new int[]{1}, 1),
                        f32Segment(betaStorage, new int[]{3}, new int[]{1}, 1)
                ),
                f32Segment(outputStorage, new int[]{2, 3}, new int[]{3, 1}, 1)
        ));

        float[] expected = {-7.0f, 0.5f - 1.25f * invStdF32(1.0f, 2.0f, 3.0f),
                -0.25f, 0.125f + 1.5f * invStdF32(1.0f, 2.0f, 3.0f),
                0.5f - 1.25f * invStdF32(4.0f, 5.0f, 6.0f),
                -0.25f, 0.125f + 1.5f * invStdF32(4.0f, 5.0f, 6.0f), -7.0f};
        assertArrayEquals(expected, outputStorage, 1e-6f);
    }

    @Test
    void rmsNormReadsBf16SegmentInputAndGammaAndWritesBf16SegmentOutput() {
        rmsNorm op = new rmsNorm(1, 1e-5);
        Tensor input = new Tensor(new int[]{2, 2}, new int[]{2, 1}, 1,
                null, null, "input", DataType.BFLOAT16);
        Tensor gamma = new Tensor(new int[]{2}, new int[]{1}, 1,
                null, null, "gamma", DataType.BFLOAT16);
        Tensor output = new Tensor(new int[]{2, 2}, new int[]{2, 1}, 1,
                List.of(input, gamma), op, "output", DataType.BFLOAT16);

        short sentinel = TensorDTypeOps.toBFloat16Bits(-7.0f);
        short[] inputStorage = bf16(-9.0f, 3.0f, 4.0f, 5.0f, 6.0f, -9.0f);
        short[] gammaStorage = bf16(-9.0f, 1.0f, 0.5f);
        short[] outputStorage = {sentinel, sentinel, sentinel, sentinel, sentinel, sentinel};

        new CpuRmsNormKernel().execute(call(
                op,
                List.of(input, gamma),
                output,
                List.of(
                        bf16Segment(inputStorage, new int[]{2, 2}, new int[]{2, 1}, 1),
                        bf16Segment(gammaStorage, new int[]{2}, new int[]{1}, 1)
                ),
                bf16Segment(outputStorage, new int[]{2, 2}, new int[]{2, 1}, 1)
        ));

        short[] expected = {
                sentinel,
                rmsBf16(3.0f, 1.0f, 3.0f, 4.0f),
                rmsBf16(4.0f, 0.5f, 3.0f, 4.0f),
                rmsBf16(5.0f, 1.0f, 5.0f, 6.0f),
                rmsBf16(6.0f, 0.5f, 5.0f, 6.0f),
                sentinel
        };
        assertArrayEquals(expected, outputStorage);
    }

    private static float invStdF32(float a, float b, float c) {
        double total = (double) a + b + c;
        double totalSquares = (double) a * a + (double) b * b + (double) c * c;
        float mean = (float) (total / 3);
        float variance = (float) Math.max(totalSquares / 3 - mean * mean, 0.0d);
        return (float) (1.0d / Math.sqrt(variance + 1e-5f));
    }

    private static short rmsBf16(float value, float gamma, float a, float b) {
        float decodedValue = TensorDTypeOps.fromBFloat16Bits(TensorDTypeOps.toBFloat16Bits(value));
        float decodedGamma = TensorDTypeOps.fromBFloat16Bits(TensorDTypeOps.toBFloat16Bits(gamma));
        float decodedA = TensorDTypeOps.fromBFloat16Bits(TensorDTypeOps.toBFloat16Bits(a));
        float decodedB = TensorDTypeOps.fromBFloat16Bits(TensorDTypeOps.toBFloat16Bits(b));
        float invRms = (float) (1.0d / Math.sqrt(((double) decodedA * decodedA + (double) decodedB * decodedB) / 2.0d + 1e-5f));
        return TensorDTypeOps.toBFloat16Bits(decodedValue * decodedGamma * invRms);
    }

    private static short[] bf16(float... values) {
        short[] bits = new short[values.length];
        for (int i = 0; i < values.length; i++) {
            bits[i] = TensorDTypeOps.toBFloat16Bits(values[i]);
        }
        return bits;
    }

    private static CpuStorageView f32Segment(float[] storage, int[] shape, int[] strides, int storageOffset) {
        return CpuStorageView.segment(
                DataType.FLOAT32,
                MemorySegment.ofArray(storage),
                shape,
                strides,
                storageOffset,
                flatSize(shape));
    }

    private static CpuStorageView bf16Segment(short[] storage, int[] shape, int[] strides, int storageOffset) {
        return CpuStorageView.segment(
                DataType.BFLOAT16,
                MemorySegment.ofArray(storage),
                shape,
                strides,
                storageOffset,
                flatSize(shape));
    }

    private static int flatSize(int[] shape) {
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        return size;
    }

    private static CpuKernelCall call(
            Operation operation,
            List<Tensor> inputTensors,
            Tensor outputTensor,
            List<CpuStorageView> inputs,
            CpuStorageView output
    ) {
        CpuNodeExecutionPlan plan = plan(output.dtype());
        CompiledNodeExecutionMetadata metadata = new CompiledNodeExecutionMetadata(
                ComputeBackend.CPU,
                operation,
                List.of(),
                null);
        CpuKernelContext context = new CpuKernelContext(
                1,
                inputTensors.stream().map(tensor -> 0).toList(),
                plan,
                ExecutionContext.fromRuntimeConfig(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD),
                metadata,
                List.of(),
                operation);
        return new CpuKernelCall(operation, inputTensors, outputTensor, inputs, output, plan, context, null);
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
}
