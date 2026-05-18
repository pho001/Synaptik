package tensor;

import java.util.Arrays;

public final class TensorRemap {
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

    public static RemapPlan buildPlan(Tensor src, Tensor dst) {
        if (src == null || dst == null) {
            return null;
        }
        int[] srcShape = src.getShapeUnsafe();
        int[] dstShape = dst.getShapeUnsafe();
        if (!Arrays.equals(srcShape, dstShape)) {
            throw new IllegalArgumentException("Source and destination tensors must have the same shape.");
        }
        int[] srcStrides = src.getStridesUnsafe();
        int[] dstStrides = dst.getStridesUnsafe();
        int[] denseStrides = denseStrides(srcShape);
        int logicalSize = logicalSize(srcShape);
        return new RemapPlan(
                srcShape.clone(),
                dstShape.clone(),
                srcStrides.clone(),
                dstStrides.clone(),
                src.getStorageOffsetUnsafe(),
                dst.getStorageOffsetUnsafe(),
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
            dst.markStorageModified();
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
        dst.markStorageModified();
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
            dst.setByStorageOffset(dstOffset, src.getByStorageOffset(srcOffset));
        });
    }

    private static void parallelApplyConverted(Tensor src, Tensor dst, RemapPlan plan) {
        parallelForRanges(plan.logicalSize, (start, end) -> {
            walkOffsets(plan.srcShape, plan.denseStrides, plan.srcStrides, plan.dstStrides, plan.srcBaseOffset, plan.dstBaseOffset, start, end, (srcOffset, dstOffset) -> {
                dst.setByStorageOffset(dstOffset, src.getByStorageOffset(srcOffset));
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
        double[] srcData = src.getFloat64Data();
        double[] dstData = dst.getFloat64Data();
        walkOffsets(plan.srcShape, plan.denseStrides, plan.srcStrides, plan.dstStrides, plan.srcBaseOffset, plan.dstBaseOffset, 0, plan.logicalSize, (srcOffset, dstOffset) -> {
            dstData[dstOffset] = srcData[srcOffset];
        });
    }

    private static void parallelApplyF64(Tensor src, Tensor dst, RemapPlan plan) {
        double[] srcData = src.getFloat64Data();
        double[] dstData = dst.getFloat64Data();
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
        float[] srcData = src.getFloat32Data();
        float[] dstData = dst.getFloat32Data();
        walkOffsets(plan.srcShape, plan.denseStrides, plan.srcStrides, plan.dstStrides, plan.srcBaseOffset, plan.dstBaseOffset, 0, plan.logicalSize, (srcOffset, dstOffset) -> {
            dstData[dstOffset] = srcData[srcOffset];
        });
    }

    private static void parallelApplyF32(Tensor src, Tensor dst, RemapPlan plan) {
        float[] srcData = src.getFloat32Data();
        float[] dstData = dst.getFloat32Data();
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
        short[] srcData = src.getBFloat16Data();
        short[] dstData = dst.getBFloat16Data();
        walkOffsets(plan.srcShape, plan.denseStrides, plan.srcStrides, plan.dstStrides, plan.srcBaseOffset, plan.dstBaseOffset, 0, plan.logicalSize, (srcOffset, dstOffset) -> {
            dstData[dstOffset] = srcData[srcOffset];
        });
    }

    private static void parallelApplyF16(Tensor src, Tensor dst, RemapPlan plan) {
        short[] srcData = src.getBFloat16Data();
        short[] dstData = dst.getBFloat16Data();
        parallelForRanges(plan.logicalSize, (start, end) -> {
            walkOffsets(plan.srcShape, plan.denseStrides, plan.srcStrides, plan.dstStrides, plan.srcBaseOffset, plan.dstBaseOffset, start, end, (srcOffset, dstOffset) -> {
                dstData[dstOffset] = srcData[srcOffset];
            });
        });
    }

    private static void sequentialApplyBool(Tensor src, Tensor dst, RemapPlan plan) {
        byte[] srcData = src.getBoolData();
        byte[] dstData = dst.getBoolData();
        walkOffsets(plan.srcShape, plan.denseStrides, plan.srcStrides, plan.dstStrides, plan.srcBaseOffset, plan.dstBaseOffset, 0, plan.logicalSize, (srcOffset, dstOffset) -> {
            dstData[dstOffset] = srcData[srcOffset];
        });
    }

    private static void sequentialApplyI32(Tensor src, Tensor dst, RemapPlan plan) {
        int[] srcData = src.getInt32Data();
        int[] dstData = dst.getInt32Data();
        walkOffsets(plan.srcShape, plan.denseStrides, plan.srcStrides, plan.dstStrides, plan.srcBaseOffset, plan.dstBaseOffset, 0, plan.logicalSize, (srcOffset, dstOffset) -> {
            dstData[dstOffset] = srcData[srcOffset];
        });
    }

    private static void sequentialApplyI64(Tensor src, Tensor dst, RemapPlan plan) {
        long[] srcData = src.getInt64Data();
        long[] dstData = dst.getInt64Data();
        walkOffsets(plan.srcShape, plan.denseStrides, plan.srcStrides, plan.dstStrides, plan.srcBaseOffset, plan.dstBaseOffset, 0, plan.logicalSize, (srcOffset, dstOffset) -> {
            dstData[dstOffset] = srcData[srcOffset];
        });
    }

    private static void parallelApplyBool(Tensor src, Tensor dst, RemapPlan plan) {
        byte[] srcData = src.getBoolData();
        byte[] dstData = dst.getBoolData();
        parallelForRanges(plan.logicalSize, (start, end) -> {
            walkOffsets(plan.srcShape, plan.denseStrides, plan.srcStrides, plan.dstStrides, plan.srcBaseOffset, plan.dstBaseOffset, start, end, (srcOffset, dstOffset) -> {
                dstData[dstOffset] = srcData[srcOffset];
            });
        });
    }

    private static void parallelApplyI32(Tensor src, Tensor dst, RemapPlan plan) {
        int[] srcData = src.getInt32Data();
        int[] dstData = dst.getInt32Data();
        parallelForRanges(plan.logicalSize, (start, end) -> {
            walkOffsets(plan.srcShape, plan.denseStrides, plan.srcStrides, plan.dstStrides, plan.srcBaseOffset, plan.dstBaseOffset, start, end, (srcOffset, dstOffset) -> {
                dstData[dstOffset] = srcData[srcOffset];
            });
        });
    }

    private static void parallelApplyI64(Tensor src, Tensor dst, RemapPlan plan) {
        long[] srcData = src.getInt64Data();
        long[] dstData = dst.getInt64Data();
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
        TensorParallelSupport.runChunks(chunks, workers, chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, logicalSize);
            body.accept(start, end);
        });
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
        double[] srcData = TensorStorageSupport.float64DataOrNull(src.getStorage());
        double[] dstData = TensorStorageSupport.float64DataOrNull(dst.getStorage());
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
        float[] srcData = TensorStorageSupport.float32DataOrNull(src.getStorage());
        float[] dstData = TensorStorageSupport.float32DataOrNull(dst.getStorage());
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
        short[] srcData = TensorStorageSupport.bfloat16DataOrNull(src.getStorage());
        short[] dstData = TensorStorageSupport.bfloat16DataOrNull(dst.getStorage());
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
        byte[] srcData = TensorStorageSupport.boolDataOrNull(src.getStorage());
        byte[] dstData = TensorStorageSupport.boolDataOrNull(dst.getStorage());
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
        int[] srcData = TensorStorageSupport.int32DataOrNull(src.getStorage());
        int[] dstData = TensorStorageSupport.int32DataOrNull(dst.getStorage());
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
        long[] srcData = TensorStorageSupport.int64DataOrNull(src.getStorage());
        long[] dstData = TensorStorageSupport.int64DataOrNull(dst.getStorage());
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
