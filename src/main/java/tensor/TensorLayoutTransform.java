package tensor;

import java.util.Arrays;

public final class TensorLayoutTransform {
    private TensorLayoutTransform() {}

    public static void copyLinearized(Tensor src, Tensor dst) {
        if (src.getFlatDataSize() != dst.getFlatDataSize()) {
            throw new IllegalArgumentException("copyLinearized requires matching number of elements.");
        }
        int[] srcShape = src.getShape();
        int[] srcStrides = src.getStrides();
        int[] srcDenseStrides = denseStrides(srcShape);
        int srcBaseOffset = src.getStorageOffsetUnsafe();
        int size = src.getFlatDataSize();

        switch (src.getDataType()) {
            case FLOAT64 -> copyLinearizedF64(src.getFloat64Data(), dst, srcShape, srcStrides, srcDenseStrides, srcBaseOffset, size);
            case FLOAT32 -> copyLinearizedF32(src.getFloat32Data(), dst, srcShape, srcStrides, srcDenseStrides, srcBaseOffset, size);
            case BFLOAT16 -> copyLinearizedBF16(src.getBFloat16Data(), dst, srcShape, srcStrides, srcDenseStrides, srcBaseOffset, size);
            case INT32 -> copyLinearizedI32(src.getInt32Data(), dst, srcShape, srcStrides, srcDenseStrides, srcBaseOffset, size);
            case BOOL -> copyLinearizedBool(src.getBoolData(), dst, srcShape, srcStrides, srcDenseStrides, srcBaseOffset, size);
        }
    }

    public static void copyPermuted(Tensor src, Tensor dst, int[] axes) {
        int rank = src.getShape().length;
        int[] normalizedAxes = normalizeAxes(rank, axes);
        if (dst.getShape().length != rank) {
            throw new IllegalArgumentException("Destination rank must match source rank for permutation.");
        }
        for (int i = 0; i < rank; i++) {
            int expected = src.getShape()[normalizedAxes[i]];
            if (dst.getShape()[i] != expected) {
                throw new IllegalArgumentException("Destination shape does not match permutation.");
            }
        }

        int[] srcStrides = src.getStrides();
        int[] dstShape = dst.getShape();
        int[] dstDenseStrides = denseStrides(dstShape);
        int srcBaseOffset = src.getStorageOffsetUnsafe();
        int size = dst.getFlatDataSize();

        switch (src.getDataType()) {
            case FLOAT64 -> copyPermutedF64(src.getFloat64Data(), dst, normalizedAxes, srcStrides, dstDenseStrides, srcBaseOffset, size, rank);
            case FLOAT32 -> copyPermutedF32(src.getFloat32Data(), dst, normalizedAxes, srcStrides, dstDenseStrides, srcBaseOffset, size, rank);
            case BFLOAT16 -> copyPermutedBF16(src.getBFloat16Data(), dst, normalizedAxes, srcStrides, dstDenseStrides, srcBaseOffset, size, rank);
            case INT32 -> copyPermutedI32(src.getInt32Data(), dst, normalizedAxes, srcStrides, dstDenseStrides, srcBaseOffset, size, rank);
            case BOOL -> copyPermutedBool(src.getBoolData(), dst, normalizedAxes, srcStrides, dstDenseStrides, srcBaseOffset, size, rank);
        }
    }

    private static void copyLinearizedF64(
            double[] srcData,
            Tensor dst,
            int[] srcShape,
            int[] srcStrides,
            int[] srcDenseStrides,
            int srcBaseOffset,
            int size
    ) {
        double[] dstF64 = dst.getFloat64Data();
        if (dstF64 != null) {
            for (int i = 0; i < size; i++) {
                dstF64[i] = srcData[logicalToOffset(i, srcShape, srcStrides, srcDenseStrides, srcBaseOffset)];
            }
            return;
        }
        float[] dstF32 = dst.getFloat32Data();
        if (dstF32 != null) {
            for (int i = 0; i < size; i++) {
                dstF32[i] = (float) srcData[logicalToOffset(i, srcShape, srcStrides, srcDenseStrides, srcBaseOffset)];
            }
            return;
        }
        short[] dstF16 = dst.getBFloat16Data();
        if (dstF16 != null) {
            for (int i = 0; i < size; i++) {
                dstF16[i] = backend.cpu.kernels.CpuDTypeOps.toBFloat16Bits((float) srcData[logicalToOffset(i, srcShape, srcStrides, srcDenseStrides, srcBaseOffset)]);
            }
            return;
        }
        throw new IllegalStateException("Destination storage is missing");
    }

    private static void copyLinearizedF32(
            float[] srcData,
            Tensor dst,
            int[] srcShape,
            int[] srcStrides,
            int[] srcDenseStrides,
            int srcBaseOffset,
            int size
    ) {
        double[] dstF64 = dst.getFloat64Data();
        if (dstF64 != null) {
            for (int i = 0; i < size; i++) {
                dstF64[i] = srcData[logicalToOffset(i, srcShape, srcStrides, srcDenseStrides, srcBaseOffset)];
            }
            return;
        }
        float[] dstF32 = dst.getFloat32Data();
        if (dstF32 != null) {
            for (int i = 0; i < size; i++) {
                dstF32[i] = srcData[logicalToOffset(i, srcShape, srcStrides, srcDenseStrides, srcBaseOffset)];
            }
            return;
        }
        short[] dstF16 = dst.getBFloat16Data();
        if (dstF16 != null) {
            for (int i = 0; i < size; i++) {
                dstF16[i] = backend.cpu.kernels.CpuDTypeOps.toBFloat16Bits(srcData[logicalToOffset(i, srcShape, srcStrides, srcDenseStrides, srcBaseOffset)]);
            }
            return;
        }
        throw new IllegalStateException("Destination storage is missing");
    }

    private static void copyLinearizedBF16(
            short[] srcData,
            Tensor dst,
            int[] srcShape,
            int[] srcStrides,
            int[] srcDenseStrides,
            int srcBaseOffset,
            int size
    ) {
        double[] dstF64 = dst.getFloat64Data();
        if (dstF64 != null) {
            for (int i = 0; i < size; i++) {
                dstF64[i] = backend.cpu.kernels.CpuDTypeOps.fromBFloat16Bits(srcData[logicalToOffset(i, srcShape, srcStrides, srcDenseStrides, srcBaseOffset)]);
            }
            return;
        }
        float[] dstF32 = dst.getFloat32Data();
        if (dstF32 != null) {
            for (int i = 0; i < size; i++) {
                dstF32[i] = backend.cpu.kernels.CpuDTypeOps.fromBFloat16Bits(srcData[logicalToOffset(i, srcShape, srcStrides, srcDenseStrides, srcBaseOffset)]);
            }
            return;
        }
        short[] dstF16 = dst.getBFloat16Data();
        if (dstF16 != null) {
            for (int i = 0; i < size; i++) {
                dstF16[i] = srcData[logicalToOffset(i, srcShape, srcStrides, srcDenseStrides, srcBaseOffset)];
            }
            return;
        }
        throw new IllegalStateException("Destination storage is missing");
    }

    private static void copyLinearizedBool(
            byte[] srcData,
            Tensor dst,
            int[] srcShape,
            int[] srcStrides,
            int[] srcDenseStrides,
            int srcBaseOffset,
            int size
    ) {
        byte[] dstBool = dst.getBoolData();
        if (dstBool != null) {
            for (int i = 0; i < size; i++) {
                dstBool[i] = srcData[logicalToOffset(i, srcShape, srcStrides, srcDenseStrides, srcBaseOffset)];
            }
            return;
        }
        throw new IllegalStateException("Destination bool storage is missing");
    }

    private static void copyLinearizedI32(
            int[] srcData,
            Tensor dst,
            int[] srcShape,
            int[] srcStrides,
            int[] srcDenseStrides,
            int srcBaseOffset,
            int size
    ) {
        int[] dstI32 = dst.getInt32Data();
        if (dstI32 != null) {
            for (int i = 0; i < size; i++) {
                dstI32[i] = srcData[logicalToOffset(i, srcShape, srcStrides, srcDenseStrides, srcBaseOffset)];
            }
            return;
        }
        throw new IllegalStateException("Destination INT32 storage is missing");
    }

    private static void copyPermutedF64(
            double[] srcData,
            Tensor dst,
            int[] normalizedAxes,
            int[] srcStrides,
            int[] dstDenseStrides,
            int srcBaseOffset,
            int size,
            int rank
    ) {
        double[] dstF64 = dst.getFloat64Data();
        if (dstF64 != null) {
            for (int logicalIndex = 0; logicalIndex < size; logicalIndex++) {
                dstF64[logicalIndex] = srcData[permutedSourceOffset(logicalIndex, normalizedAxes, srcStrides, dstDenseStrides, srcBaseOffset, rank)];
            }
            return;
        }
        float[] dstF32 = dst.getFloat32Data();
        if (dstF32 != null) {
            for (int logicalIndex = 0; logicalIndex < size; logicalIndex++) {
                dstF32[logicalIndex] = (float) srcData[permutedSourceOffset(logicalIndex, normalizedAxes, srcStrides, dstDenseStrides, srcBaseOffset, rank)];
            }
            return;
        }
        short[] dstF16 = dst.getBFloat16Data();
        if (dstF16 != null) {
            for (int logicalIndex = 0; logicalIndex < size; logicalIndex++) {
                dstF16[logicalIndex] = backend.cpu.kernels.CpuDTypeOps.toBFloat16Bits((float) srcData[permutedSourceOffset(logicalIndex, normalizedAxes, srcStrides, dstDenseStrides, srcBaseOffset, rank)]);
            }
            return;
        }
        throw new IllegalStateException("Destination storage is missing");
    }

    private static void copyPermutedF32(
            float[] srcData,
            Tensor dst,
            int[] normalizedAxes,
            int[] srcStrides,
            int[] dstDenseStrides,
            int srcBaseOffset,
            int size,
            int rank
    ) {
        double[] dstF64 = dst.getFloat64Data();
        if (dstF64 != null) {
            for (int logicalIndex = 0; logicalIndex < size; logicalIndex++) {
                dstF64[logicalIndex] = srcData[permutedSourceOffset(logicalIndex, normalizedAxes, srcStrides, dstDenseStrides, srcBaseOffset, rank)];
            }
            return;
        }
        float[] dstF32 = dst.getFloat32Data();
        if (dstF32 != null) {
            for (int logicalIndex = 0; logicalIndex < size; logicalIndex++) {
                dstF32[logicalIndex] = srcData[permutedSourceOffset(logicalIndex, normalizedAxes, srcStrides, dstDenseStrides, srcBaseOffset, rank)];
            }
            return;
        }
        short[] dstF16 = dst.getBFloat16Data();
        if (dstF16 != null) {
            for (int logicalIndex = 0; logicalIndex < size; logicalIndex++) {
                dstF16[logicalIndex] = backend.cpu.kernels.CpuDTypeOps.toBFloat16Bits(srcData[permutedSourceOffset(logicalIndex, normalizedAxes, srcStrides, dstDenseStrides, srcBaseOffset, rank)]);
            }
            return;
        }
        throw new IllegalStateException("Destination storage is missing");
    }

    private static void copyPermutedBF16(
            short[] srcData,
            Tensor dst,
            int[] normalizedAxes,
            int[] srcStrides,
            int[] dstDenseStrides,
            int srcBaseOffset,
            int size,
            int rank
    ) {
        double[] dstF64 = dst.getFloat64Data();
        if (dstF64 != null) {
            for (int logicalIndex = 0; logicalIndex < size; logicalIndex++) {
                dstF64[logicalIndex] = backend.cpu.kernels.CpuDTypeOps.fromBFloat16Bits(srcData[permutedSourceOffset(logicalIndex, normalizedAxes, srcStrides, dstDenseStrides, srcBaseOffset, rank)]);
            }
            return;
        }
        float[] dstF32 = dst.getFloat32Data();
        if (dstF32 != null) {
            for (int logicalIndex = 0; logicalIndex < size; logicalIndex++) {
                dstF32[logicalIndex] = backend.cpu.kernels.CpuDTypeOps.fromBFloat16Bits(srcData[permutedSourceOffset(logicalIndex, normalizedAxes, srcStrides, dstDenseStrides, srcBaseOffset, rank)]);
            }
            return;
        }
        short[] dstF16 = dst.getBFloat16Data();
        if (dstF16 != null) {
            for (int logicalIndex = 0; logicalIndex < size; logicalIndex++) {
                dstF16[logicalIndex] = srcData[permutedSourceOffset(logicalIndex, normalizedAxes, srcStrides, dstDenseStrides, srcBaseOffset, rank)];
            }
            return;
        }
        throw new IllegalStateException("Destination storage is missing");
    }

    private static void copyPermutedBool(
            byte[] srcData,
            Tensor dst,
            int[] normalizedAxes,
            int[] srcStrides,
            int[] dstDenseStrides,
            int srcBaseOffset,
            int size,
            int rank
    ) {
        byte[] dstBool = dst.getBoolData();
        if (dstBool != null) {
            for (int logicalIndex = 0; logicalIndex < size; logicalIndex++) {
                dstBool[logicalIndex] = srcData[permutedSourceOffset(logicalIndex, normalizedAxes, srcStrides, dstDenseStrides, srcBaseOffset, rank)];
            }
            return;
        }
        throw new IllegalStateException("Destination bool storage is missing");
    }

    private static void copyPermutedI32(
            int[] srcData,
            Tensor dst,
            int[] normalizedAxes,
            int[] srcStrides,
            int[] dstDenseStrides,
            int srcBaseOffset,
            int size,
            int rank
    ) {
        int[] dstI32 = dst.getInt32Data();
        if (dstI32 != null) {
            for (int logicalIndex = 0; logicalIndex < size; logicalIndex++) {
                dstI32[logicalIndex] = srcData[permutedSourceOffset(logicalIndex, normalizedAxes, srcStrides, dstDenseStrides, srcBaseOffset, rank)];
            }
            return;
        }
        throw new IllegalStateException("Destination INT32 storage is missing");
    }

    private static int permutedSourceOffset(int logicalIndex, int[] normalizedAxes, int[] srcStrides, int[] dstDenseStrides, int srcBaseOffset, int rank) {
        int rem = logicalIndex;
        int srcOffset = srcBaseOffset;
        for (int dim = 0; dim < rank; dim++) {
            int coord = rem / dstDenseStrides[dim];
            rem %= dstDenseStrides[dim];
            srcOffset += coord * srcStrides[normalizedAxes[dim]];
        }
        return srcOffset;
    }

    public static int[] inferReshape(int[] oldShape, int[] requestedShape) {
        if (requestedShape == null || requestedShape.length == 0) {
            throw new IllegalArgumentException("Requested shape cannot be null/empty.");
        }
        int oldSize = size(oldShape);
        int[] out = requestedShape.clone();
        int minusOneIndex = -1;
        long knownProduct = 1L;
        for (int i = 0; i < out.length; i++) {
            int dim = out[i];
            if (dim == -1) {
                if (minusOneIndex != -1) {
                    throw new IllegalArgumentException("Only one -1 is allowed in reshape shape.");
                }
                minusOneIndex = i;
                continue;
            }
            if (dim <= 0) {
                throw new IllegalArgumentException("Reshape dimensions must be positive or -1.");
            }
            knownProduct *= dim;
        }
        if (minusOneIndex != -1) {
            if (knownProduct == 0 || oldSize % knownProduct != 0) {
                throw new IllegalArgumentException("Cannot infer reshape dimension for size=" + oldSize
                        + " and shape=" + Arrays.toString(requestedShape));
            }
            out[minusOneIndex] = (int) (oldSize / knownProduct);
        }
        if (size(out) != oldSize) {
            throw new IllegalArgumentException("Reshape size mismatch. oldSize=" + oldSize
                    + ", newShape=" + Arrays.toString(out));
        }
        return out;
    }

    public static int[] inferExpandShape(int[] oldShape, int[] requestedShape) {
        if (requestedShape == null || requestedShape.length == 0) {
            throw new IllegalArgumentException("Requested expand shape cannot be null/empty.");
        }
        int[] out = requestedShape.clone();
        for (int dim : out) {
            if (dim <= 0) {
                throw new IllegalArgumentException("Expand dimensions must be positive.");
            }
        }

        int oldRank = oldShape.length;
        int newRank = out.length;
        if (newRank < oldRank) {
            throw new IllegalArgumentException("Expanded rank cannot be smaller than source rank.");
        }

        int offset = newRank - oldRank;
        for (int d = 0; d < newRank; d++) {
            int srcDimIndex = d - offset;
            if (srcDimIndex < 0) {
                continue;
            }
            int srcDim = oldShape[srcDimIndex];
            int dstDim = out[d];
            if (srcDim != dstDim && srcDim != 1) {
                throw new IllegalArgumentException(
                        "Cannot expand non-singleton dimension " + srcDim + " to " + dstDim
                );
            }
        }
        return out;
    }

    public static int[] normalizeAxes(int rank, int[] axes) {
        if (axes == null || axes.length != rank) {
            throw new IllegalArgumentException("Axes length must equal tensor rank.");
        }
        boolean[] seen = new boolean[rank];
        int[] out = new int[rank];
        for (int i = 0; i < rank; i++) {
            int axis = axes[i];
            if (axis < 0) axis += rank;
            if (axis < 0 || axis >= rank) {
                throw new IllegalArgumentException("Axis out of range for rank " + rank + ": " + axes[i]);
            }
            if (seen[axis]) {
                throw new IllegalArgumentException("Duplicate axis in permutation: " + Arrays.toString(axes));
            }
            seen[axis] = true;
            out[i] = axis;
        }
        return out;
    }

    public static int[] inverseAxes(int[] axes) {
        int[] inv = new int[axes.length];
        for (int i = 0; i < axes.length; i++) {
            inv[axes[i]] = i;
        }
        return inv;
    }

    public static int normalizeInsertAxis(int axis, int rank) {
        int out = axis;
        if (out < 0) out += (rank + 1);
        if (out < 0 || out > rank) {
            throw new IllegalArgumentException("Axis out of range for expandDims: axis=" + axis + ", rank=" + rank);
        }
        return out;
    }

    public static int normalizeAxis(int axis, int rank) {
        int out = axis;
        if (out < 0) out += rank;
        if (out < 0 || out >= rank) {
            throw new IllegalArgumentException("Axis out of range: axis=" + axis + ", rank=" + rank);
        }
        return out;
    }

    private static int[] denseStrides(int[] shape) {
        int[] out = new int[shape.length];
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            out[i] = stride;
            stride *= shape[i];
        }
        return out;
    }

    private static int logicalToOffset(int logicalIndex, int[] shape, int[] strides, int[] denseStrides, int baseOffset) {
        int rem = logicalIndex;
        int offset = baseOffset;
        for (int dim = 0; dim < shape.length; dim++) {
            int coord = rem / denseStrides[dim];
            rem %= denseStrides[dim];
            offset += coord * strides[dim];
        }
        return offset;
    }

    private static int size(int[] shape) {
        int s = 1;
        for (int dim : shape) s *= dim;
        return s;
    }
}
