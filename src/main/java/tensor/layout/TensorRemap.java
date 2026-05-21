package tensor.layout;

import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.DataType;
import tensor.storage.TensorStorageAccess;

import java.util.Arrays;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;

public final class TensorRemap {
    private static final ConcurrentHashMap<Integer, ForkJoinPool> POOLS = new ConcurrentHashMap<>();

    private TensorRemap() {}

    public static final class RemapPlan {
        private final int[] srcShape;
        private final int[] dstShape;
        private final int[] srcStrides;
        private final int[] dstStrides;
        private final int srcBaseOffset;
        private final int dstBaseOffset;
        private final int[] denseStrides;
        private final int logicalSize;

        private RemapPlan(
                int[] srcShape,
                int[] dstShape,
                int[] srcStrides,
                int[] dstStrides,
                int srcBaseOffset,
                int dstBaseOffset,
                int[] denseStrides,
                int logicalSize
        ) {
            this.srcShape = srcShape;
            this.dstShape = dstShape;
            this.srcStrides = srcStrides;
            this.dstStrides = dstStrides;
            this.srcBaseOffset = srcBaseOffset;
            this.dstBaseOffset = dstBaseOffset;
            this.denseStrides = denseStrides;
            this.logicalSize = logicalSize;
        }
    }

    public static void copyLinearized(Tensor src, Tensor dst) {
        if (src == null || dst == null) {
            throw new IllegalArgumentException("Source and destination tensors cannot be null.");
        }
        if (src.getFlatDataSize() != dst.getFlatDataSize()) {
            throw new IllegalArgumentException("copyLinearized requires matching number of elements.");
        }
        if (src.getDataType() != dst.getDataType() && !canConvertNumeric(src.getDataType(), dst.getDataType())) {
            throw new IllegalArgumentException("copyLinearized requires matching tensor dtypes or supported numeric conversion. src="
                    + src.getDataType() + ", dst=" + dst.getDataType());
        }

        int[] srcShape = src.getShapeUnsafe();
        int[] dstShape = dst.getShapeUnsafe();
        int[] srcDenseStrides = denseStrides(srcShape);
        int[] dstDenseStrides = denseStrides(dstShape);
        int size = src.getFlatDataSize();
        copyLinearizedByType(
                src,
                dst,
                srcShape,
                src.getStridesUnsafe(),
                srcDenseStrides,
                src.getStorageOffsetUnsafe(),
                dstShape,
                dst.getStridesUnsafe(),
                dstDenseStrides,
                dst.getStorageOffsetUnsafe(),
                size
        );
        TensorInternalAccess.markStorageModified(dst);
    }

    public static void copyPermuted(Tensor src, Tensor dst, int[] axes) {
        if (src == null || dst == null) {
            throw new IllegalArgumentException("Source and destination tensors cannot be null.");
        }
        int rank = src.getShapeUnsafe().length;
        int[] normalizedAxes = TensorLayoutTransform.normalizeAxes(rank, axes);
        if (dst.getShapeUnsafe().length != rank) {
            throw new IllegalArgumentException("Destination rank must match source rank for permutation.");
        }
        for (int i = 0; i < rank; i++) {
            int expected = src.getShapeUnsafe()[normalizedAxes[i]];
            if (dst.getShapeUnsafe()[i] != expected) {
                throw new IllegalArgumentException("Destination shape does not match permutation.");
            }
        }
        if (src.getDataType() != dst.getDataType() && !canConvertNumeric(src.getDataType(), dst.getDataType())) {
            throw new IllegalArgumentException("copyPermuted requires matching tensor dtypes or supported numeric conversion. src="
                    + src.getDataType() + ", dst=" + dst.getDataType());
        }

        int[] dstShape = dst.getShapeUnsafe();
        int[] dstDenseStrides = denseStrides(dstShape);
        copyPermutedByType(
                src,
                dst,
                normalizedAxes,
                src.getStridesUnsafe(),
                src.getStorageOffsetUnsafe(),
                dstShape,
                dst.getStridesUnsafe(),
                dstDenseStrides,
                dst.getStorageOffsetUnsafe(),
                dst.getFlatDataSize(),
                rank
        );
        TensorInternalAccess.markStorageModified(dst);
    }

    public static RemapPlan buildPlan(Tensor src, Tensor dst) {
        if (src == null || dst == null) {
            return null;
        }
        return buildPlan(
                src.getShapeUnsafe(),
                src.getStridesUnsafe(),
                src.getStorageOffsetUnsafe(),
                dst.getShapeUnsafe(),
                dst.getStridesUnsafe(),
                dst.getStorageOffsetUnsafe()
        );
    }

    public static RemapPlan buildPlan(
            int[] srcShape,
            int[] srcStrides,
            int srcBaseOffset,
            int[] dstShape,
            int[] dstStrides,
            int dstBaseOffset
    ) {
        if (!Arrays.equals(srcShape, dstShape)) {
            throw new IllegalArgumentException("Source and destination tensors must have the same shape.");
        }
        int[] denseStrides = denseStrides(srcShape);
        int logicalSize = logicalSize(srcShape);
        return new RemapPlan(
                srcShape.clone(),
                dstShape.clone(),
                srcStrides.clone(),
                dstStrides.clone(),
                srcBaseOffset,
                dstBaseOffset,
                denseStrides,
                logicalSize
        );
    }

    public static void apply(Tensor src, Tensor dst, int parallelThreshold) {
        apply(src, dst, null, parallelThreshold);
    }

    public static void apply(Tensor src, Tensor dst, RemapPlan plan, int parallelThreshold) {
        RemapPlan effectivePlan = applyResolved(src, dst, plan, parallelThreshold);
        if (effectivePlan == null) {
            return;
        }

        dispatchApply(src, dst, effectivePlan, parallelThreshold);
    }

    public static void applyTrusted(Tensor src, Tensor dst, RemapPlan plan, int parallelThreshold) {
        RemapPlan effectivePlan = applyResolved(src, dst, plan, parallelThreshold);
        if (effectivePlan == null) {
            return;
        }
        dispatchApply(src, dst, effectivePlan, parallelThreshold);
    }

    private static RemapPlan applyResolved(Tensor src, Tensor dst, RemapPlan plan, int parallelThreshold) {
        if (src == null || dst == null) {
            throw new IllegalArgumentException("Source and destination tensors cannot be null.");
        }
        RemapPlan effectivePlan = resolvePlan(src, dst, plan);
        if (src == dst) {
            return null;
        }
        return effectivePlan;
    }

    private static RemapPlan resolvePlan(Tensor src, Tensor dst, RemapPlan plan) {
        if (plan != null && matches(src, dst, plan)) {
            return plan;
        }
        return buildPlan(src, dst);
    }

    private static boolean matches(Tensor src, Tensor dst, RemapPlan plan) {
        if (plan == null || src == null || dst == null) {
            return false;
        }
        return Arrays.equals(src.getShapeUnsafe(), plan.srcShape)
                && Arrays.equals(dst.getShapeUnsafe(), plan.dstShape)
                && Arrays.equals(src.getStridesUnsafe(), plan.srcStrides)
                && Arrays.equals(dst.getStridesUnsafe(), plan.dstStrides)
                && src.getStorageOffsetUnsafe() == plan.srcBaseOffset
                && dst.getStorageOffsetUnsafe() == plan.dstBaseOffset;
    }

    private static void dispatchApply(Tensor src, Tensor dst, RemapPlan plan, int parallelThreshold) {
        if (src.getDataType() != dst.getDataType()) {
            if (!canConvertNumeric(src.getDataType(), dst.getDataType())) {
                throw new IllegalArgumentException("TensorRemap requires matching tensor dtypes or supported numeric conversion. src="
                        + src.getDataType() + ", dst=" + dst.getDataType());
            }
            applyConverted(src, dst, plan, parallelThreshold);
            TensorInternalAccess.markStorageModified(dst);
            return;
        }
        switch (src.getDataType()) {
            case FLOAT64 -> applyF64(src, dst, plan, parallelThreshold);
            case FLOAT32 -> applyF32(src, dst, plan, parallelThreshold);
            case BFLOAT16 -> applyBF16(src, dst, plan, parallelThreshold);
            case INT32 -> applyI32(src, dst, plan, parallelThreshold);
            case INT64 -> applyI64(src, dst, plan, parallelThreshold);
            case BOOL -> applyBool(src, dst, plan, parallelThreshold);
        }
        TensorInternalAccess.markStorageModified(dst);
    }

    private static void copyLinearizedByType(
            Tensor src,
            Tensor dst,
            int[] srcShape,
            int[] srcStrides,
            int[] srcDenseStrides,
            int srcBaseOffset,
            int[] dstShape,
            int[] dstStrides,
            int[] dstDenseStrides,
            int dstBaseOffset,
            int size
    ) {
        if (src.getDataType() != dst.getDataType()) {
            copyLinearizedConverted(src, dst, srcShape, srcStrides, srcDenseStrides, srcBaseOffset, dstShape, dstStrides, dstDenseStrides, dstBaseOffset, size);
            return;
        }
        switch (src.getDataType()) {
            case FLOAT64 -> copyLinearizedF64(src, dst, srcShape, srcStrides, srcDenseStrides, srcBaseOffset, dstShape, dstStrides, dstDenseStrides, dstBaseOffset, size);
            case FLOAT32 -> copyLinearizedF32(src, dst, srcShape, srcStrides, srcDenseStrides, srcBaseOffset, dstShape, dstStrides, dstDenseStrides, dstBaseOffset, size);
            case BFLOAT16 -> copyLinearizedBF16(src, dst, srcShape, srcStrides, srcDenseStrides, srcBaseOffset, dstShape, dstStrides, dstDenseStrides, dstBaseOffset, size);
            case INT32 -> copyLinearizedI32(src, dst, srcShape, srcStrides, srcDenseStrides, srcBaseOffset, dstShape, dstStrides, dstDenseStrides, dstBaseOffset, size);
            case INT64 -> copyLinearizedI64(src, dst, srcShape, srcStrides, srcDenseStrides, srcBaseOffset, dstShape, dstStrides, dstDenseStrides, dstBaseOffset, size);
            case BOOL -> copyLinearizedBool(src, dst, srcShape, srcStrides, srcDenseStrides, srcBaseOffset, dstShape, dstStrides, dstDenseStrides, dstBaseOffset, size);
        }
    }

    private static void copyPermutedByType(
            Tensor src,
            Tensor dst,
            int[] normalizedAxes,
            int[] srcStrides,
            int srcBaseOffset,
            int[] dstShape,
            int[] dstStrides,
            int[] dstDenseStrides,
            int dstBaseOffset,
            int size,
            int rank
    ) {
        if (src.getDataType() != dst.getDataType()) {
            copyPermutedConverted(src, dst, normalizedAxes, srcStrides, srcBaseOffset, dstShape, dstStrides, dstDenseStrides, dstBaseOffset, size, rank);
            return;
        }
        switch (src.getDataType()) {
            case FLOAT64 -> copyPermutedF64(src, dst, normalizedAxes, srcStrides, srcBaseOffset, dstShape, dstStrides, dstDenseStrides, dstBaseOffset, size, rank);
            case FLOAT32 -> copyPermutedF32(src, dst, normalizedAxes, srcStrides, srcBaseOffset, dstShape, dstStrides, dstDenseStrides, dstBaseOffset, size, rank);
            case BFLOAT16 -> copyPermutedBF16(src, dst, normalizedAxes, srcStrides, srcBaseOffset, dstShape, dstStrides, dstDenseStrides, dstBaseOffset, size, rank);
            case INT32 -> copyPermutedI32(src, dst, normalizedAxes, srcStrides, srcBaseOffset, dstShape, dstStrides, dstDenseStrides, dstBaseOffset, size, rank);
            case INT64 -> copyPermutedI64(src, dst, normalizedAxes, srcStrides, srcBaseOffset, dstShape, dstStrides, dstDenseStrides, dstBaseOffset, size, rank);
            case BOOL -> copyPermutedBool(src, dst, normalizedAxes, srcStrides, srcBaseOffset, dstShape, dstStrides, dstDenseStrides, dstBaseOffset, size, rank);
        }
    }

    private static void copyLinearizedConverted(
            Tensor src,
            Tensor dst,
            int[] srcShape,
            int[] srcStrides,
            int[] srcDenseStrides,
            int srcBaseOffset,
            int[] dstShape,
            int[] dstStrides,
            int[] dstDenseStrides,
            int dstBaseOffset,
            int size
    ) {
        for (int i = 0; i < size; i++) {
            int srcOffset = logicalToOffset(i, srcShape, srcStrides, srcDenseStrides, srcBaseOffset);
            int dstOffset = logicalToOffset(i, dstShape, dstStrides, dstDenseStrides, dstBaseOffset);
            TensorInternalAccess.setByStorageOffset(dst, dstOffset, TensorInternalAccess.getByStorageOffset(src, srcOffset));
        }
    }

    private static void copyPermutedConverted(
            Tensor src,
            Tensor dst,
            int[] normalizedAxes,
            int[] srcStrides,
            int srcBaseOffset,
            int[] dstShape,
            int[] dstStrides,
            int[] dstDenseStrides,
            int dstBaseOffset,
            int size,
            int rank
    ) {
        for (int logicalIndex = 0; logicalIndex < size; logicalIndex++) {
            int srcOffset = permutedSourceOffset(logicalIndex, normalizedAxes, srcStrides, dstDenseStrides, srcBaseOffset, rank);
            int dstOffset = logicalToOffset(logicalIndex, dstShape, dstStrides, dstDenseStrides, dstBaseOffset);
            TensorInternalAccess.setByStorageOffset(dst, dstOffset, TensorInternalAccess.getByStorageOffset(src, srcOffset));
        }
    }

    private static void copyLinearizedF64(
            Tensor src,
            Tensor dst,
            int[] srcShape,
            int[] srcStrides,
            int[] srcDenseStrides,
            int srcBaseOffset,
            int[] dstShape,
            int[] dstStrides,
            int[] dstDenseStrides,
            int dstBaseOffset,
            int size
    ) {
        double[] srcData = TensorInternalAccess.float64Data(src);
        double[] dstData = TensorInternalAccess.float64Data(dst);
        for (int i = 0; i < size; i++) {
            dstData[logicalToOffset(i, dstShape, dstStrides, dstDenseStrides, dstBaseOffset)] =
                    srcData[logicalToOffset(i, srcShape, srcStrides, srcDenseStrides, srcBaseOffset)];
        }
    }

    private static void copyLinearizedF32(
            Tensor src,
            Tensor dst,
            int[] srcShape,
            int[] srcStrides,
            int[] srcDenseStrides,
            int srcBaseOffset,
            int[] dstShape,
            int[] dstStrides,
            int[] dstDenseStrides,
            int dstBaseOffset,
            int size
    ) {
        float[] srcData = TensorInternalAccess.float32Data(src);
        float[] dstData = TensorInternalAccess.float32Data(dst);
        for (int i = 0; i < size; i++) {
            dstData[logicalToOffset(i, dstShape, dstStrides, dstDenseStrides, dstBaseOffset)] =
                    srcData[logicalToOffset(i, srcShape, srcStrides, srcDenseStrides, srcBaseOffset)];
        }
    }

    private static void copyLinearizedBF16(
            Tensor src,
            Tensor dst,
            int[] srcShape,
            int[] srcStrides,
            int[] srcDenseStrides,
            int srcBaseOffset,
            int[] dstShape,
            int[] dstStrides,
            int[] dstDenseStrides,
            int dstBaseOffset,
            int size
    ) {
        short[] srcData = TensorInternalAccess.bfloat16Data(src);
        short[] dstData = TensorInternalAccess.bfloat16Data(dst);
        for (int i = 0; i < size; i++) {
            dstData[logicalToOffset(i, dstShape, dstStrides, dstDenseStrides, dstBaseOffset)] =
                    srcData[logicalToOffset(i, srcShape, srcStrides, srcDenseStrides, srcBaseOffset)];
        }
    }

    private static void copyLinearizedI32(
            Tensor src,
            Tensor dst,
            int[] srcShape,
            int[] srcStrides,
            int[] srcDenseStrides,
            int srcBaseOffset,
            int[] dstShape,
            int[] dstStrides,
            int[] dstDenseStrides,
            int dstBaseOffset,
            int size
    ) {
        int[] srcData = TensorInternalAccess.int32Data(src);
        int[] dstData = TensorInternalAccess.int32Data(dst);
        for (int i = 0; i < size; i++) {
            dstData[logicalToOffset(i, dstShape, dstStrides, dstDenseStrides, dstBaseOffset)] =
                    srcData[logicalToOffset(i, srcShape, srcStrides, srcDenseStrides, srcBaseOffset)];
        }
    }

    private static void copyLinearizedI64(
            Tensor src,
            Tensor dst,
            int[] srcShape,
            int[] srcStrides,
            int[] srcDenseStrides,
            int srcBaseOffset,
            int[] dstShape,
            int[] dstStrides,
            int[] dstDenseStrides,
            int dstBaseOffset,
            int size
    ) {
        long[] srcData = TensorInternalAccess.int64Data(src);
        long[] dstData = TensorInternalAccess.int64Data(dst);
        for (int i = 0; i < size; i++) {
            dstData[logicalToOffset(i, dstShape, dstStrides, dstDenseStrides, dstBaseOffset)] =
                    srcData[logicalToOffset(i, srcShape, srcStrides, srcDenseStrides, srcBaseOffset)];
        }
    }

    private static void copyLinearizedBool(
            Tensor src,
            Tensor dst,
            int[] srcShape,
            int[] srcStrides,
            int[] srcDenseStrides,
            int srcBaseOffset,
            int[] dstShape,
            int[] dstStrides,
            int[] dstDenseStrides,
            int dstBaseOffset,
            int size
    ) {
        byte[] srcData = TensorInternalAccess.boolData(src);
        byte[] dstData = TensorInternalAccess.boolData(dst);
        for (int i = 0; i < size; i++) {
            dstData[logicalToOffset(i, dstShape, dstStrides, dstDenseStrides, dstBaseOffset)] =
                    srcData[logicalToOffset(i, srcShape, srcStrides, srcDenseStrides, srcBaseOffset)];
        }
    }

    private static void copyPermutedF64(
            Tensor src,
            Tensor dst,
            int[] normalizedAxes,
            int[] srcStrides,
            int srcBaseOffset,
            int[] dstShape,
            int[] dstStrides,
            int[] dstDenseStrides,
            int dstBaseOffset,
            int size,
            int rank
    ) {
        double[] srcData = TensorInternalAccess.float64Data(src);
        double[] dstData = TensorInternalAccess.float64Data(dst);
        for (int logicalIndex = 0; logicalIndex < size; logicalIndex++) {
            dstData[logicalToOffset(logicalIndex, dstShape, dstStrides, dstDenseStrides, dstBaseOffset)] =
                    srcData[permutedSourceOffset(logicalIndex, normalizedAxes, srcStrides, dstDenseStrides, srcBaseOffset, rank)];
        }
    }

    private static void copyPermutedF32(
            Tensor src,
            Tensor dst,
            int[] normalizedAxes,
            int[] srcStrides,
            int srcBaseOffset,
            int[] dstShape,
            int[] dstStrides,
            int[] dstDenseStrides,
            int dstBaseOffset,
            int size,
            int rank
    ) {
        float[] srcData = TensorInternalAccess.float32Data(src);
        float[] dstData = TensorInternalAccess.float32Data(dst);
        for (int logicalIndex = 0; logicalIndex < size; logicalIndex++) {
            dstData[logicalToOffset(logicalIndex, dstShape, dstStrides, dstDenseStrides, dstBaseOffset)] =
                    srcData[permutedSourceOffset(logicalIndex, normalizedAxes, srcStrides, dstDenseStrides, srcBaseOffset, rank)];
        }
    }

    private static void copyPermutedBF16(
            Tensor src,
            Tensor dst,
            int[] normalizedAxes,
            int[] srcStrides,
            int srcBaseOffset,
            int[] dstShape,
            int[] dstStrides,
            int[] dstDenseStrides,
            int dstBaseOffset,
            int size,
            int rank
    ) {
        short[] srcData = TensorInternalAccess.bfloat16Data(src);
        short[] dstData = TensorInternalAccess.bfloat16Data(dst);
        for (int logicalIndex = 0; logicalIndex < size; logicalIndex++) {
            dstData[logicalToOffset(logicalIndex, dstShape, dstStrides, dstDenseStrides, dstBaseOffset)] =
                    srcData[permutedSourceOffset(logicalIndex, normalizedAxes, srcStrides, dstDenseStrides, srcBaseOffset, rank)];
        }
    }

    private static void copyPermutedI32(
            Tensor src,
            Tensor dst,
            int[] normalizedAxes,
            int[] srcStrides,
            int srcBaseOffset,
            int[] dstShape,
            int[] dstStrides,
            int[] dstDenseStrides,
            int dstBaseOffset,
            int size,
            int rank
    ) {
        int[] srcData = TensorInternalAccess.int32Data(src);
        int[] dstData = TensorInternalAccess.int32Data(dst);
        for (int logicalIndex = 0; logicalIndex < size; logicalIndex++) {
            dstData[logicalToOffset(logicalIndex, dstShape, dstStrides, dstDenseStrides, dstBaseOffset)] =
                    srcData[permutedSourceOffset(logicalIndex, normalizedAxes, srcStrides, dstDenseStrides, srcBaseOffset, rank)];
        }
    }

    private static void copyPermutedI64(
            Tensor src,
            Tensor dst,
            int[] normalizedAxes,
            int[] srcStrides,
            int srcBaseOffset,
            int[] dstShape,
            int[] dstStrides,
            int[] dstDenseStrides,
            int dstBaseOffset,
            int size,
            int rank
    ) {
        long[] srcData = TensorInternalAccess.int64Data(src);
        long[] dstData = TensorInternalAccess.int64Data(dst);
        for (int logicalIndex = 0; logicalIndex < size; logicalIndex++) {
            dstData[logicalToOffset(logicalIndex, dstShape, dstStrides, dstDenseStrides, dstBaseOffset)] =
                    srcData[permutedSourceOffset(logicalIndex, normalizedAxes, srcStrides, dstDenseStrides, srcBaseOffset, rank)];
        }
    }

    private static void copyPermutedBool(
            Tensor src,
            Tensor dst,
            int[] normalizedAxes,
            int[] srcStrides,
            int srcBaseOffset,
            int[] dstShape,
            int[] dstStrides,
            int[] dstDenseStrides,
            int dstBaseOffset,
            int size,
            int rank
    ) {
        byte[] srcData = TensorInternalAccess.boolData(src);
        byte[] dstData = TensorInternalAccess.boolData(dst);
        for (int logicalIndex = 0; logicalIndex < size; logicalIndex++) {
            dstData[logicalToOffset(logicalIndex, dstShape, dstStrides, dstDenseStrides, dstBaseOffset)] =
                    srcData[permutedSourceOffset(logicalIndex, normalizedAxes, srcStrides, dstDenseStrides, srcBaseOffset, rank)];
        }
    }

    private static void applyConverted(Tensor src, Tensor dst, RemapPlan plan, int parallelThreshold) {
        if (plan.logicalSize > parallelThreshold) {
            parallelApplyConverted(src, dst, plan);
        } else {
            sequentialApplyConverted(src, dst, plan);
        }
    }

    private static void sequentialApplyConverted(Tensor src, Tensor dst, RemapPlan plan) {
        walkOffsets(plan.srcShape, plan.denseStrides, plan.srcStrides, plan.dstStrides, plan.srcBaseOffset, plan.dstBaseOffset, 0, plan.logicalSize, (srcOffset, dstOffset) -> {
            TensorInternalAccess.setByStorageOffset(dst, dstOffset, TensorInternalAccess.getByStorageOffset(src, srcOffset));
        });
    }

    private static void parallelApplyConverted(Tensor src, Tensor dst, RemapPlan plan) {
        parallelForRanges(plan.logicalSize, (start, end) -> {
            walkOffsets(plan.srcShape, plan.denseStrides, plan.srcStrides, plan.dstStrides, plan.srcBaseOffset, plan.dstBaseOffset, start, end, (srcOffset, dstOffset) -> {
                TensorInternalAccess.setByStorageOffset(dst, dstOffset, TensorInternalAccess.getByStorageOffset(src, srcOffset));
            });
        });
    }

    private static void applyF64(Tensor src, Tensor dst, RemapPlan plan, int parallelThreshold) {
        if (tryFastCopyF64(src, dst, plan, plan.logicalSize)) {
            return;
        }
        if (plan.logicalSize > parallelThreshold) {
            parallelApplyF64(src, dst, plan);
        } else {
            sequentialApplyF64(src, dst, plan);
        }
    }

    private static void sequentialApplyF64(Tensor src, Tensor dst, RemapPlan plan) {
        double[] srcData = TensorInternalAccess.float64Data(src);
        double[] dstData = TensorInternalAccess.float64Data(dst);
        walkOffsets(plan.srcShape, plan.denseStrides, plan.srcStrides, plan.dstStrides, plan.srcBaseOffset, plan.dstBaseOffset, 0, plan.logicalSize, (srcOffset, dstOffset) -> {
            dstData[dstOffset] = srcData[srcOffset];
        });
    }

    private static void parallelApplyF64(Tensor src, Tensor dst, RemapPlan plan) {
        double[] srcData = TensorInternalAccess.float64Data(src);
        double[] dstData = TensorInternalAccess.float64Data(dst);
        parallelForRanges(plan.logicalSize, (start, end) -> {
            walkOffsets(plan.srcShape, plan.denseStrides, plan.srcStrides, plan.dstStrides, plan.srcBaseOffset, plan.dstBaseOffset, start, end, (srcOffset, dstOffset) -> {
                dstData[dstOffset] = srcData[srcOffset];
            });
        });
    }

    private static int[] denseStrides(int[] shape) {
        return TensorShape.contiguousStrides(shape);
    }

    private static int logicalSize(int[] shape) {
        return TensorShape.checkedFlatSize(shape);
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

    private static void applyF32(Tensor src, Tensor dst, RemapPlan plan, int parallelThreshold) {
        if (tryFastCopyF32(src, dst, plan, plan.logicalSize)) {
            return;
        }
        if (plan.logicalSize > parallelThreshold) {
            parallelApplyF32(src, dst, plan);
        } else {
            sequentialApplyF32(src, dst, plan);
        }
    }

    private static void sequentialApplyF32(Tensor src, Tensor dst, RemapPlan plan) {
        float[] srcData = TensorInternalAccess.float32Data(src);
        float[] dstData = TensorInternalAccess.float32Data(dst);
        walkOffsets(plan.srcShape, plan.denseStrides, plan.srcStrides, plan.dstStrides, plan.srcBaseOffset, plan.dstBaseOffset, 0, plan.logicalSize, (srcOffset, dstOffset) -> {
            dstData[dstOffset] = srcData[srcOffset];
        });
    }

    private static void parallelApplyF32(Tensor src, Tensor dst, RemapPlan plan) {
        float[] srcData = TensorInternalAccess.float32Data(src);
        float[] dstData = TensorInternalAccess.float32Data(dst);
        parallelForRanges(plan.logicalSize, (start, end) -> {
            walkOffsets(plan.srcShape, plan.denseStrides, plan.srcStrides, plan.dstStrides, plan.srcBaseOffset, plan.dstBaseOffset, start, end, (srcOffset, dstOffset) -> {
                dstData[dstOffset] = srcData[srcOffset];
            });
        });
    }

    private static void applyBF16(Tensor src, Tensor dst, RemapPlan plan, int parallelThreshold) {
        if (tryFastCopyF16(src, dst, plan, plan.logicalSize)) {
            return;
        }
        if (plan.logicalSize > parallelThreshold) {
            parallelApplyF16(src, dst, plan);
        } else {
            sequentialApplyF16(src, dst, plan);
        }
    }

    private static void applyBool(Tensor src, Tensor dst, RemapPlan plan, int parallelThreshold) {
        if (tryFastCopyBool(src, dst, plan, plan.logicalSize)) {
            return;
        }
        if (plan.logicalSize > parallelThreshold) {
            parallelApplyBool(src, dst, plan);
        } else {
            sequentialApplyBool(src, dst, plan);
        }
    }

    private static void applyI32(Tensor src, Tensor dst, RemapPlan plan, int parallelThreshold) {
        if (tryFastCopyI32(src, dst, plan, plan.logicalSize)) {
            return;
        }
        if (plan.logicalSize > parallelThreshold) {
            parallelApplyI32(src, dst, plan);
        } else {
            sequentialApplyI32(src, dst, plan);
        }
    }

    private static void applyI64(Tensor src, Tensor dst, RemapPlan plan, int parallelThreshold) {
        if (tryFastCopyI64(src, dst, plan, plan.logicalSize)) {
            return;
        }
        if (plan.logicalSize > parallelThreshold) {
            parallelApplyI64(src, dst, plan);
        } else {
            sequentialApplyI64(src, dst, plan);
        }
    }

    private static void sequentialApplyF16(Tensor src, Tensor dst, RemapPlan plan) {
        short[] srcData = TensorInternalAccess.bfloat16Data(src);
        short[] dstData = TensorInternalAccess.bfloat16Data(dst);
        walkOffsets(plan.srcShape, plan.denseStrides, plan.srcStrides, plan.dstStrides, plan.srcBaseOffset, plan.dstBaseOffset, 0, plan.logicalSize, (srcOffset, dstOffset) -> {
            dstData[dstOffset] = srcData[srcOffset];
        });
    }

    private static void parallelApplyF16(Tensor src, Tensor dst, RemapPlan plan) {
        short[] srcData = TensorInternalAccess.bfloat16Data(src);
        short[] dstData = TensorInternalAccess.bfloat16Data(dst);
        parallelForRanges(plan.logicalSize, (start, end) -> {
            walkOffsets(plan.srcShape, plan.denseStrides, plan.srcStrides, plan.dstStrides, plan.srcBaseOffset, plan.dstBaseOffset, start, end, (srcOffset, dstOffset) -> {
                dstData[dstOffset] = srcData[srcOffset];
            });
        });
    }

    private static void sequentialApplyBool(Tensor src, Tensor dst, RemapPlan plan) {
        byte[] srcData = TensorInternalAccess.boolData(src);
        byte[] dstData = TensorInternalAccess.boolData(dst);
        walkOffsets(plan.srcShape, plan.denseStrides, plan.srcStrides, plan.dstStrides, plan.srcBaseOffset, plan.dstBaseOffset, 0, plan.logicalSize, (srcOffset, dstOffset) -> {
            dstData[dstOffset] = srcData[srcOffset];
        });
    }

    private static void sequentialApplyI32(Tensor src, Tensor dst, RemapPlan plan) {
        int[] srcData = TensorInternalAccess.int32Data(src);
        int[] dstData = TensorInternalAccess.int32Data(dst);
        walkOffsets(plan.srcShape, plan.denseStrides, plan.srcStrides, plan.dstStrides, plan.srcBaseOffset, plan.dstBaseOffset, 0, plan.logicalSize, (srcOffset, dstOffset) -> {
            dstData[dstOffset] = srcData[srcOffset];
        });
    }

    private static void sequentialApplyI64(Tensor src, Tensor dst, RemapPlan plan) {
        long[] srcData = TensorInternalAccess.int64Data(src);
        long[] dstData = TensorInternalAccess.int64Data(dst);
        walkOffsets(plan.srcShape, plan.denseStrides, plan.srcStrides, plan.dstStrides, plan.srcBaseOffset, plan.dstBaseOffset, 0, plan.logicalSize, (srcOffset, dstOffset) -> {
            dstData[dstOffset] = srcData[srcOffset];
        });
    }

    private static void parallelApplyBool(Tensor src, Tensor dst, RemapPlan plan) {
        byte[] srcData = TensorInternalAccess.boolData(src);
        byte[] dstData = TensorInternalAccess.boolData(dst);
        parallelForRanges(plan.logicalSize, (start, end) -> {
            walkOffsets(plan.srcShape, plan.denseStrides, plan.srcStrides, plan.dstStrides, plan.srcBaseOffset, plan.dstBaseOffset, start, end, (srcOffset, dstOffset) -> {
                dstData[dstOffset] = srcData[srcOffset];
            });
        });
    }

    private static void parallelApplyI32(Tensor src, Tensor dst, RemapPlan plan) {
        int[] srcData = TensorInternalAccess.int32Data(src);
        int[] dstData = TensorInternalAccess.int32Data(dst);
        parallelForRanges(plan.logicalSize, (start, end) -> {
            walkOffsets(plan.srcShape, plan.denseStrides, plan.srcStrides, plan.dstStrides, plan.srcBaseOffset, plan.dstBaseOffset, start, end, (srcOffset, dstOffset) -> {
                dstData[dstOffset] = srcData[srcOffset];
            });
        });
    }

    private static void parallelApplyI64(Tensor src, Tensor dst, RemapPlan plan) {
        long[] srcData = TensorInternalAccess.int64Data(src);
        long[] dstData = TensorInternalAccess.int64Data(dst);
        parallelForRanges(plan.logicalSize, (start, end) -> {
            walkOffsets(plan.srcShape, plan.denseStrides, plan.srcStrides, plan.dstStrides, plan.srcBaseOffset, plan.dstBaseOffset, start, end, (srcOffset, dstOffset) -> {
                dstData[dstOffset] = srcData[srcOffset];
            });
        });
    }

    private static void parallelForRanges(int logicalSize, RangeConsumer body) {
        int workers = Math.max(1, Runtime.getRuntime().availableProcessors());
        int targetChunks = Math.max(workers, workers * 4);
        int chunkSize = Math.max(1024, (logicalSize + targetChunks - 1) / targetChunks);
        int chunks = (logicalSize + chunkSize - 1) / chunkSize;
        runChunks(chunks, workers, chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, logicalSize);
            body.accept(start, end);
        });
    }

    private static void runChunks(int chunks, int parallelism, IntConsumer chunkBody) {
        if (chunks <= 0) {
            return;
        }
        if (chunks == 1 || parallelism <= 1) {
            for (int i = 0; i < chunks; i++) {
                chunkBody.accept(i);
            }
            return;
        }
        ForkJoinPool pool = POOLS.computeIfAbsent(parallelism, ForkJoinPool::new);
        try {
            pool.submit(() -> IntStream.range(0, chunks).parallel().forEach(chunkBody)).get();
        } catch (Exception e) {
            throw new RuntimeException("Parallel tensor remap failed", e);
        }
    }

    private static void walkOffsets(
            int[] shape,
            int[] denseStrides,
            int[] srcStrides,
            int[] dstStrides,
            int srcBaseOffset,
            int dstBaseOffset,
            int startInclusive,
            int endExclusive,
            OffsetConsumer consumer
    ) {
        if (startInclusive >= endExclusive) {
            return;
        }
        int rank = shape.length;
        int[] coords = new int[rank];
        int temp = startInclusive;
        for (int d = 0; d < rank; d++) {
            coords[d] = temp / denseStrides[d];
            temp %= denseStrides[d];
        }

        int srcOffset = srcBaseOffset;
        int dstOffset = dstBaseOffset;
        for (int d = 0; d < rank; d++) {
            srcOffset += coords[d] * srcStrides[d];
            dstOffset += coords[d] * dstStrides[d];
        }

        for (int i = startInclusive; i < endExclusive; i++) {
            consumer.accept(srcOffset, dstOffset);
            for (int d = rank - 1; d >= 0; d--) {
                coords[d]++;
                srcOffset += srcStrides[d];
                dstOffset += dstStrides[d];
                if (coords[d] < shape[d]) {
                    break;
                }
                coords[d] = 0;
                srcOffset -= shape[d] * srcStrides[d];
                dstOffset -= shape[d] * dstStrides[d];
            }
        }
    }

    private static boolean tryFastCopyF64(Tensor src, Tensor dst, RemapPlan plan, int logicalSize) {
        double[] srcData = TensorStorageAccess.float64DataOrNull(TensorInternalAccess.storage(src));
        double[] dstData = TensorStorageAccess.float64DataOrNull(TensorInternalAccess.storage(dst));
        if (srcData == null || dstData == null) {
            return false;
        }
        if (!canUseRawCopy(src, dst, logicalSize)) {
            return false;
        }
        System.arraycopy(srcData, plan.srcBaseOffset, dstData, plan.dstBaseOffset, logicalSize);
        return true;
    }

    private static boolean tryFastCopyF32(Tensor src, Tensor dst, RemapPlan plan, int logicalSize) {
        float[] srcData = TensorStorageAccess.float32DataOrNull(TensorInternalAccess.storage(src));
        float[] dstData = TensorStorageAccess.float32DataOrNull(TensorInternalAccess.storage(dst));
        if (srcData == null || dstData == null) {
            return false;
        }
        if (!canUseRawCopy(src, dst, logicalSize)) {
            return false;
        }
        System.arraycopy(srcData, plan.srcBaseOffset, dstData, plan.dstBaseOffset, logicalSize);
        return true;
    }

    private static boolean tryFastCopyF16(Tensor src, Tensor dst, RemapPlan plan, int logicalSize) {
        short[] srcData = TensorStorageAccess.bfloat16DataOrNull(TensorInternalAccess.storage(src));
        short[] dstData = TensorStorageAccess.bfloat16DataOrNull(TensorInternalAccess.storage(dst));
        if (srcData == null || dstData == null) {
            return false;
        }
        if (!canUseRawCopy(src, dst, logicalSize)) {
            return false;
        }
        System.arraycopy(srcData, plan.srcBaseOffset, dstData, plan.dstBaseOffset, logicalSize);
        return true;
    }

    private static boolean tryFastCopyBool(Tensor src, Tensor dst, RemapPlan plan, int logicalSize) {
        byte[] srcData = TensorStorageAccess.boolDataOrNull(TensorInternalAccess.storage(src));
        byte[] dstData = TensorStorageAccess.boolDataOrNull(TensorInternalAccess.storage(dst));
        if (srcData == null || dstData == null) {
            return false;
        }
        if (!canUseRawCopy(src, dst, logicalSize)) {
            return false;
        }
        System.arraycopy(srcData, plan.srcBaseOffset, dstData, plan.dstBaseOffset, logicalSize);
        return true;
    }

    private static boolean tryFastCopyI32(Tensor src, Tensor dst, RemapPlan plan, int logicalSize) {
        int[] srcData = TensorStorageAccess.int32DataOrNull(TensorInternalAccess.storage(src));
        int[] dstData = TensorStorageAccess.int32DataOrNull(TensorInternalAccess.storage(dst));
        if (srcData == null || dstData == null) {
            return false;
        }
        if (!canUseRawCopy(src, dst, logicalSize)) {
            return false;
        }
        System.arraycopy(srcData, plan.srcBaseOffset, dstData, plan.dstBaseOffset, logicalSize);
        return true;
    }

    private static boolean tryFastCopyI64(Tensor src, Tensor dst, RemapPlan plan, int logicalSize) {
        long[] srcData = TensorStorageAccess.int64DataOrNull(TensorInternalAccess.storage(src));
        long[] dstData = TensorStorageAccess.int64DataOrNull(TensorInternalAccess.storage(dst));
        if (srcData == null || dstData == null) {
            return false;
        }
        if (!canUseRawCopy(src, dst, logicalSize)) {
            return false;
        }
        System.arraycopy(srcData, plan.srcBaseOffset, dstData, plan.dstBaseOffset, logicalSize);
        return true;
    }

    private static boolean canUseRawCopy(Tensor src, Tensor dst, int logicalSize) {
        return src.isContiguous()
                && dst.isContiguous()
                && src.getFlatDataSize() == logicalSize
                && dst.getFlatDataSize() == logicalSize;
    }

    private static boolean canConvertNumeric(DataType srcType, DataType dstType) {
        if (srcType == DataType.BOOL || dstType == DataType.BOOL
                || srcType == DataType.INT32 || dstType == DataType.INT32
                || srcType == DataType.INT64 || dstType == DataType.INT64) {
            return false;
        }
        return srcType != null && dstType != null;
    }

    @FunctionalInterface
    private interface OffsetConsumer {
        void accept(int srcOffset, int dstOffset);
    }

    @FunctionalInterface
    private interface RangeConsumer {
        void accept(int startInclusive, int endExclusive);
    }
}
