package backend.kernels.cpu.reduction;

import backend.kernels.cpu.*;

import backend.kernels.cpu.CpuDTypeOps;
import backend.kernels.cpu.CpuKernelContext;
import backend.kernels.cpu.CpuThreadPool;
import backend.kernels.cpu.ResolvedReductionHints;
import tensor.Tensor;
import tensor.TensorMetadata;

public final class SoftmaxLoops {
    private SoftmaxLoops() {
    }

    public static void execute(Tensor input, Tensor node, int dimension, CpuKernelContext context) {
        validateShapes(input, node, dimension);
        double[] in = input.getFloat64Data();
        double[] out = node.getFloat64Data();
        runGroups(input.getShapeUnsafe(), input.getStridesUnsafe(), input.getStorageOffsetUnsafe(), node.getStridesUnsafe(), node.getStorageOffsetUnsafe(), dimension, group ->
                computeGroupF64(in, out, group.baseIn(), group.baseOut(), group.axisStrideIn(), group.axisStrideOut(), group.axisSize())
        , context.reductionHints());
    }

    public static void executeF32(Tensor input, Tensor node, int dimension, CpuKernelContext context) {
        validateShapes(input, node, dimension);
        float[] in = input.getFloat32Data();
        float[] out = node.getFloat32Data();
        runGroups(input.getShapeUnsafe(), input.getStridesUnsafe(), input.getStorageOffsetUnsafe(), node.getStridesUnsafe(), node.getStorageOffsetUnsafe(), dimension, group ->
                computeGroupF32(in, out, group.baseIn(), group.baseOut(), group.axisStrideIn(), group.axisStrideOut(), group.axisSize())
        , context.reductionHints());
    }

    public static void executeBF16(Tensor input, Tensor node, int dimension, CpuKernelContext context) {
        validateShapes(input, node, dimension);
        short[] in = input.getBFloat16Data();
        short[] out = node.getBFloat16Data();
        runGroups(input.getShapeUnsafe(), input.getStridesUnsafe(), input.getStorageOffsetUnsafe(), node.getStridesUnsafe(), node.getStorageOffsetUnsafe(), dimension, group ->
                computeGroupF16(in, out, group.baseIn(), group.baseOut(), group.axisStrideIn(), group.axisStrideOut(), group.axisSize())
        , context.reductionHints());
    }

    public static void executeF32ToBF16(Tensor input, float[] in, Tensor node, int dimension, CpuKernelContext context) {
        validateShapes(input, node, dimension);
        short[] out = node.getBFloat16Data();
        if (in == null) {
            throw new IllegalArgumentException("Float continuation input cannot be null");
        }
        runGroups(input.getShapeUnsafe(), input.getStridesUnsafe(), 0, node.getStridesUnsafe(), node.getStorageOffsetUnsafe(), dimension, group ->
                computeGroupF32ToBF16(in, out, group.baseIn(), group.baseOut(), group.axisStrideIn(), group.axisStrideOut(), group.axisSize())
        , context.reductionHints());
    }

    private static void validateShapes(Tensor input, Tensor node, int dimension) {
        int[] shape = input.getShapeUnsafe();
        if (shape == null || shape.length == 0) {
            throw new IllegalArgumentException("Input shape must not be empty");
        }
        if (dimension < 0 || dimension >= shape.length) {
            throw new IllegalArgumentException("Dimension out of bounds: " + dimension);
        }
        int[] outShape = node.getShapeUnsafe();
        if (outShape.length != shape.length) {
            throw new IllegalArgumentException("Softmax output rank must match input rank");
        }
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] != outShape[i]) {
                throw new IllegalArgumentException("Softmax output shape must match input shape");
            }
        }
    }

    private static void computeGroupF64(double[] in, double[] out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
            max = Math.max(max, in[inOffset]);
        }

        double sum = 0.0;
        for (int i = 0, inOffset = baseIn, outOffset = baseOut; i < axisSize; i++, inOffset += axisStrideIn, outOffset += axisStrideOut) {
            double value = Math.exp(in[inOffset] - max);
            out[outOffset] = value;
            sum += value;
        }

        double invSum = 1.0 / sum;
        for (int i = 0, outOffset = baseOut; i < axisSize; i++, outOffset += axisStrideOut) {
            out[outOffset] *= invSum;
        }
    }

    private static void computeGroupF32(float[] in, float[] out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
            max = Math.max(max, in[inOffset]);
        }

        float sum = 0.0f;
        for (int i = 0, inOffset = baseIn, outOffset = baseOut; i < axisSize; i++, inOffset += axisStrideIn, outOffset += axisStrideOut) {
            float value = (float) Math.exp(in[inOffset] - max);
            out[outOffset] = value;
            sum += value;
        }

        float invSum = 1.0f / sum;
        for (int i = 0, outOffset = baseOut; i < axisSize; i++, outOffset += axisStrideOut) {
            out[outOffset] *= invSum;
        }
    }

    private static void computeGroupF16(short[] in, short[] out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
            max = Math.max(max, CpuDTypeOps.fromBFloat16Bits(in[inOffset]));
        }

        float sum = 0.0f;
        for (int i = 0, inOffset = baseIn, outOffset = baseOut; i < axisSize; i++, inOffset += axisStrideIn, outOffset += axisStrideOut) {
            float value = (float) Math.exp(CpuDTypeOps.fromBFloat16Bits(in[inOffset]) - max);
            out[outOffset] = CpuDTypeOps.toBFloat16Bits(value);
            sum += value;
        }

        float invSum = 1.0f / sum;
        for (int i = 0, outOffset = baseOut; i < axisSize; i++, outOffset += axisStrideOut) {
            float normalized = CpuDTypeOps.fromBFloat16Bits(out[outOffset]) * invSum;
            out[outOffset] = CpuDTypeOps.toBFloat16Bits(normalized);
        }
    }

    private static void computeGroupF32ToBF16(float[] in, short[] out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
            max = Math.max(max, in[inOffset]);
        }

        float sum = 0.0f;
        for (int i = 0, inOffset = baseIn, outOffset = baseOut; i < axisSize; i++, inOffset += axisStrideIn, outOffset += axisStrideOut) {
            float value = (float) Math.exp(in[inOffset] - max);
            out[outOffset] = CpuDTypeOps.toBFloat16Bits(value);
            sum += value;
        }

        float invSum = 1.0f / sum;
        for (int i = 0, outOffset = baseOut; i < axisSize; i++, outOffset += axisStrideOut) {
            float normalized = CpuDTypeOps.fromBFloat16Bits(out[outOffset]) * invSum;
            out[outOffset] = CpuDTypeOps.toBFloat16Bits(normalized);
        }
    }

    private static void runGroups(
            int[] shape,
            int[] inStrides,
            int inBaseOffset,
            int[] outStrides,
            int outBaseOffset,
            int axis,
            GroupComputer groupComputer,
            ResolvedReductionHints hints
    ) {
        int[] reducedShape = reduceShape(shape, axis);
        int[] reducedDenseStrides = TensorMetadata.computeStrides(reducedShape);
        int groupCount = logicalSize(reducedShape);
        int axisSize = shape[axis];
        int axisStrideIn = inStrides[axis];
        int axisStrideOut = outStrides[axis];

        if (hints != null && hints.parallel() && groupCount > 1) {
            int chunkSize = Math.max(1, hints.chunkSize());
            int chunks = (groupCount + chunkSize - 1) / chunkSize;
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, groupCount);
                for (int group = start; group < end; group++) {
                    GroupState state = groupState(group, shape, inStrides, inBaseOffset, outStrides, outBaseOffset, axis, reducedDenseStrides, axisSize, axisStrideIn, axisStrideOut);
                    groupComputer.compute(state);
                }
            });
            return;
        }

        for (int group = 0; group < groupCount; group++) {
            GroupState state = groupState(group, shape, inStrides, inBaseOffset, outStrides, outBaseOffset, axis, reducedDenseStrides, axisSize, axisStrideIn, axisStrideOut);
            groupComputer.compute(state);
        }
    }

    private static GroupState groupState(
            int reducedIndex,
            int[] shape,
            int[] inStrides,
            int inBaseOffset,
            int[] outStrides,
            int outBaseOffset,
            int axis,
            int[] reducedDenseStrides,
            int axisSize,
            int axisStrideIn,
            int axisStrideOut
        ) {
        int rem = reducedIndex;
        int baseIn = inBaseOffset;
        int baseOut = outBaseOffset;
        for (int d = 0, rd = 0; d < shape.length; d++) {
            if (d == axis) {
                continue;
            }
            int coord = rem / reducedDenseStrides[rd];
            rem %= reducedDenseStrides[rd];
            baseIn += coord * inStrides[d];
            baseOut += coord * outStrides[d];
            rd++;
        }
        return new GroupState(baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize);
    }

    private static int[] reduceShape(int[] shape, int axis) {
        if (shape.length == 1) {
            return new int[]{1};
        }
        int[] reduced = new int[shape.length - 1];
        for (int i = 0, j = 0; i < shape.length; i++) {
            if (i != axis) {
                reduced[j++] = shape[i];
            }
        }
        return reduced;
    }

    private static int logicalSize(int[] shape) {
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        return size;
    }

    @FunctionalInterface
    private interface GroupComputer {
        void compute(GroupState state);
    }

    private record GroupState(int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
    }
}
