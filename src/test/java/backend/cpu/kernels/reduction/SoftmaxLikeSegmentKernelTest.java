package backend.cpu.kernels.reduction;

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
import operations.reduction.logSoftmax;
import operations.reduction.softmax;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.dtype.TensorDTypeOps;

import java.lang.foreign.MemorySegment;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class SoftmaxLikeSegmentKernelTest {
    @Test
    void softmaxReadsAndWritesStridedF64MemorySegments() {
        softmax op = new softmax(1);
        Tensor input = new Tensor(new int[]{2, 3}, new int[]{1, 2}, 1,
                null, null, "input", DataType.FLOAT64);
        Tensor output = new Tensor(new int[]{2, 3}, new int[]{1, 2}, 1,
                List.of(input), op, "output", DataType.FLOAT64);
        double[] inputStorage = {-9.0d, 1.0d, 0.0d, 2.0d, 0.0d, 3.0d, 0.0d};
        double[] outputStorage = {-7.0d, -7.0d, -7.0d, -7.0d, -7.0d, -7.0d, -7.0d};

        new CpuSoftmaxKernel().execute(call(
                op,
                List.of(input),
                output,
                List.of(segment(DataType.FLOAT64, inputStorage, new int[]{2, 3}, new int[]{1, 2}, 1)),
                segment(DataType.FLOAT64, outputStorage, new int[]{2, 3}, new int[]{1, 2}, 1)));

        double[] row0 = softmaxRow(1.0d, 2.0d, 3.0d);
        assertArrayEquals(new double[]{
                -7.0d,
                row0[0], 1.0d / 3.0d,
                row0[1], 1.0d / 3.0d,
                row0[2], 1.0d / 3.0d
        }, outputStorage, 1e-12);
    }

    @Test
    void logSoftmaxReadsAndWritesF32MemorySegments() {
        logSoftmax op = new logSoftmax(1);
        Tensor input = new Tensor(new int[]{2, 3}, new int[]{3, 1}, 0,
                null, null, "input", DataType.FLOAT32);
        Tensor output = new Tensor(new int[]{2, 3}, new int[]{3, 1}, 0,
                List.of(input), op, "output", DataType.FLOAT32);
        float[] inputStorage = {1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.0f};
        float[] outputStorage = {-7.0f, -7.0f, -7.0f, -7.0f, -7.0f, -7.0f};

        new CpuLogSoftmaxKernel().execute(call(
                op,
                List.of(input),
                output,
                List.of(segment(DataType.FLOAT32, inputStorage, new int[]{2, 3}, new int[]{3, 1}, 0)),
                segment(DataType.FLOAT32, outputStorage, new int[]{2, 3}, new int[]{3, 1}, 0)));

        double[] row0 = logSoftmaxRow(1.0d, 2.0d, 3.0d);
        double[] row1 = logSoftmaxRow(0.0d, 0.0d, 0.0d);
        assertArrayEquals(new float[]{
                (float) row0[0], (float) row0[1], (float) row0[2],
                (float) row1[0], (float) row1[1], (float) row1[2]
        }, outputStorage, 1e-6f);
    }

    @Test
    void softmaxReadsAndWritesBf16MemorySegments() {
        softmax op = new softmax(1);
        Tensor input = new Tensor(new int[]{1, 3}, new int[]{3, 1}, 1,
                null, null, "input", DataType.BFLOAT16);
        Tensor output = new Tensor(new int[]{1, 3}, new int[]{3, 1}, 1,
                List.of(input), op, "output", DataType.BFLOAT16);
        short sentinel = TensorDTypeOps.toBFloat16Bits(-7.0f);
        short[] inputStorage = bf16(-9.0f, 1.0f, 2.0f, 3.0f, -9.0f);
        short[] outputStorage = {sentinel, sentinel, sentinel, sentinel, sentinel};

        new CpuSoftmaxKernel().execute(call(
                op,
                List.of(input),
                output,
                List.of(segment(DataType.BFLOAT16, inputStorage, new int[]{1, 3}, new int[]{3, 1}, 1)),
                segment(DataType.BFLOAT16, outputStorage, new int[]{1, 3}, new int[]{3, 1}, 1)));

        short[] expected = expectedBf16Softmax(1.0f, 2.0f, 3.0f);
        assertArrayEquals(new short[]{sentinel, expected[0], expected[1], expected[2], sentinel}, outputStorage);
    }

    private static CpuStorageView segment(DataType dtype, Object storage, int[] shape, int[] strides, int storageOffset) {
        MemorySegment segment;
        if (storage instanceof double[] data) {
            segment = MemorySegment.ofArray(data);
        } else if (storage instanceof float[] data) {
            segment = MemorySegment.ofArray(data);
        } else if (storage instanceof short[] data) {
            segment = MemorySegment.ofArray(data);
        } else {
            throw new IllegalArgumentException("Unsupported segment storage: " + storage.getClass());
        }
        return CpuStorageView.segment(
                dtype,
                segment,
                shape,
                strides,
                storageOffset,
                flatSize(shape));
    }

    private static short[] bf16(float... values) {
        short[] bits = new short[values.length];
        for (int i = 0; i < values.length; i++) {
            bits[i] = TensorDTypeOps.toBFloat16Bits(values[i]);
        }
        return bits;
    }

    private static short[] expectedBf16Softmax(float... values) {
        float[] decoded = new float[values.length];
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < values.length; i++) {
            decoded[i] = TensorDTypeOps.fromBFloat16Bits(TensorDTypeOps.toBFloat16Bits(values[i]));
            max = Math.max(max, decoded[i]);
        }
        float sum = 0.0f;
        short[] expBits = new short[values.length];
        for (int i = 0; i < values.length; i++) {
            float value = (float) Math.exp(decoded[i] - max);
            expBits[i] = TensorDTypeOps.toBFloat16Bits(value);
            sum += value;
        }
        float inv = 1.0f / sum;
        short[] out = new short[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = TensorDTypeOps.toBFloat16Bits(TensorDTypeOps.fromBFloat16Bits(expBits[i]) * inv);
        }
        return out;
    }

    private static double[] softmaxRow(double... values) {
        double[] log = logSoftmaxRow(values);
        double[] out = new double[log.length];
        for (int i = 0; i < log.length; i++) {
            out[i] = Math.exp(log[i]);
        }
        return out;
    }

    private static double[] logSoftmaxRow(double... values) {
        double max = Double.NEGATIVE_INFINITY;
        for (double value : values) {
            max = Math.max(max, value);
        }
        double sum = 0.0d;
        for (double value : values) {
            sum += Math.exp(value - max);
        }
        double logSumExp = max + Math.log(sum);
        double[] out = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = values[i] - logSumExp;
        }
        return out;
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
