package Tensor;

import Backend.kernels.cpu.CpuThreadPool;

import java.util.Arrays;

public final class TensorRemap {
    private TensorRemap() {}

    public static final class RemapPlan {
        private final int[] srcShape;
        private final int[] dstShape;
        private final int[] srcStrides;
        private final int[] dstStrides;
        private final int[] denseStrides;
        private final int logicalSize;

        private RemapPlan(
                int[] srcShape,
                int[] dstShape,
                int[] srcStrides,
                int[] dstStrides,
                int[] denseStrides,
                int logicalSize
        ) {
            this.srcShape = srcShape;
            this.dstShape = dstShape;
            this.srcStrides = srcStrides;
            this.dstStrides = dstStrides;
            this.denseStrides = denseStrides;
            this.logicalSize = logicalSize;
        }
    }

    public static RemapPlan buildPlan(Tensor src, Tensor dst) {
        if (src == null || dst == null) {
            return null;
        }
        int[] srcShape = src.getShape();
        int[] dstShape = dst.getShape();
        if (!Arrays.equals(srcShape, dstShape)) {
            throw new IllegalArgumentException("Source and destination tensors must have the same shape.");
        }
        int[] srcStrides = src.getStrides();
        int[] dstStrides = dst.getStrides();
        int[] denseStrides = denseStrides(srcShape);
        int logicalSize = logicalSize(srcShape);
        return new RemapPlan(
                srcShape.clone(),
                dstShape.clone(),
                srcStrides.clone(),
                dstStrides.clone(),
                denseStrides,
                logicalSize
        );
    }

    public static void apply(Tensor src, Tensor dst, int parallelThreshold) {
        apply(src, dst, null, parallelThreshold);
    }

    public static void apply(Tensor src, Tensor dst, RemapPlan plan, int parallelThreshold) {
        if (src == null || dst == null) {
            throw new IllegalArgumentException("Source and destination tensors cannot be null.");
        }
        RemapPlan effectivePlan = resolvePlan(src, dst, plan);
        if (src == dst) {
            return;
        }

        if (src.getDataType() == DataType.FLOAT32
                && src.getFloat32Data() != null
                && dst.getFloat32Data() != null) {
            applyF32(src, dst, effectivePlan, parallelThreshold);
            return;
        }
        if (src.getDataType() == DataType.FLOAT16
                && src.getFloat16Data() != null
                && dst.getFloat16Data() != null) {
            applyF16(src, dst, effectivePlan, parallelThreshold);
            return;
        }

        if (tryFastCopyF64(src, dst, effectivePlan.logicalSize)) {
            return;
        }
        if (tryFastCopyStorage(src, dst)) {
            return;
        }

        if (effectivePlan.logicalSize > parallelThreshold) {
            parallelApply(src, dst, effectivePlan);
        } else {
            sequentialApply(src, dst, effectivePlan);
        }
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
        return Arrays.equals(src.getShape(), plan.srcShape)
                && Arrays.equals(dst.getShape(), plan.dstShape)
                && Arrays.equals(src.getStrides(), plan.srcStrides)
                && Arrays.equals(dst.getStrides(), plan.dstStrides);
    }

    private static void sequentialApply(Tensor src, Tensor dst, RemapPlan plan) {
        TensorStorage srcStorage = src.getStorage();
        TensorStorage dstStorage = dst.getStorage();

        walkOffsets(plan.srcShape, plan.denseStrides, plan.srcStrides, plan.dstStrides, 0, plan.logicalSize, (srcOffset, dstOffset) -> {
            dstStorage.setAsDoubleAt(dstOffset, srcStorage.getAsDoubleAt(srcOffset));
        });
    }

    private static void parallelApply(Tensor src, Tensor dst, RemapPlan plan) {
        TensorStorage srcStorage = src.getStorage();
        TensorStorage dstStorage = dst.getStorage();

        parallelForRanges(plan.logicalSize, (start, end) -> {
            walkOffsets(plan.srcShape, plan.denseStrides, plan.srcStrides, plan.dstStrides, start, end, (srcOffset, dstOffset) -> {
                dstStorage.setAsDoubleAt(dstOffset, srcStorage.getAsDoubleAt(srcOffset));
            });
        });
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

    private static int logicalSize(int[] shape) {
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        return size;
    }

    private static void applyF32(Tensor src, Tensor dst, RemapPlan plan, int parallelThreshold) {
        if (tryFastCopyF32(src, dst, plan.logicalSize)) {
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
        walkOffsets(plan.srcShape, plan.denseStrides, plan.srcStrides, plan.dstStrides, 0, plan.logicalSize, (srcOffset, dstOffset) -> {
            dstData[dstOffset] = srcData[srcOffset];
        });
    }

    private static void parallelApplyF32(Tensor src, Tensor dst, RemapPlan plan) {
        float[] srcData = src.getFloat32Data();
        float[] dstData = dst.getFloat32Data();
        parallelForRanges(plan.logicalSize, (start, end) -> {
            walkOffsets(plan.srcShape, plan.denseStrides, plan.srcStrides, plan.dstStrides, start, end, (srcOffset, dstOffset) -> {
                dstData[dstOffset] = srcData[srcOffset];
            });
        });
    }

    private static void applyF16(Tensor src, Tensor dst, RemapPlan plan, int parallelThreshold) {
        if (tryFastCopyF16(src, dst, plan.logicalSize)) {
            return;
        }
        if (plan.logicalSize > parallelThreshold) {
            parallelApplyF16(src, dst, plan);
        } else {
            sequentialApplyF16(src, dst, plan);
        }
    }

    private static void sequentialApplyF16(Tensor src, Tensor dst, RemapPlan plan) {
        short[] srcData = src.getFloat16Data();
        short[] dstData = dst.getFloat16Data();
        walkOffsets(plan.srcShape, plan.denseStrides, plan.srcStrides, plan.dstStrides, 0, plan.logicalSize, (srcOffset, dstOffset) -> {
            dstData[dstOffset] = srcData[srcOffset];
        });
    }

    private static void parallelApplyF16(Tensor src, Tensor dst, RemapPlan plan) {
        short[] srcData = src.getFloat16Data();
        short[] dstData = dst.getFloat16Data();
        parallelForRanges(plan.logicalSize, (start, end) -> {
            walkOffsets(plan.srcShape, plan.denseStrides, plan.srcStrides, plan.dstStrides, start, end, (srcOffset, dstOffset) -> {
                dstData[dstOffset] = srcData[srcOffset];
            });
        });
    }

    private static void parallelForRanges(int logicalSize, RangeConsumer body) {
        int workers = Math.max(1, Runtime.getRuntime().availableProcessors());
        int targetChunks = Math.max(workers, workers * 4);
        int chunkSize = Math.max(1024, (logicalSize + targetChunks - 1) / targetChunks);
        int chunks = (logicalSize + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, workers, chunk -> {
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

        int srcOffset = 0;
        int dstOffset = 0;
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

    private static boolean tryFastCopyF64(Tensor src, Tensor dst, int logicalSize) {
        double[] srcData = src.getFloat64Data();
        double[] dstData = dst.getFloat64Data();
        if (srcData == null || dstData == null) {
            return false;
        }
        if (!canUseRawCopy(src, dst, logicalSize)) {
            return false;
        }
        System.arraycopy(srcData, 0, dstData, 0, Math.min(srcData.length, dstData.length));
        return true;
    }

    private static boolean tryFastCopyF32(Tensor src, Tensor dst, int logicalSize) {
        float[] srcData = src.getFloat32Data();
        float[] dstData = dst.getFloat32Data();
        if (srcData == null || dstData == null) {
            return false;
        }
        if (!canUseRawCopy(src, dst, logicalSize)) {
            return false;
        }
        System.arraycopy(srcData, 0, dstData, 0, Math.min(srcData.length, dstData.length));
        return true;
    }

    private static boolean tryFastCopyF16(Tensor src, Tensor dst, int logicalSize) {
        short[] srcData = src.getFloat16Data();
        short[] dstData = dst.getFloat16Data();
        if (srcData == null || dstData == null) {
            return false;
        }
        if (!canUseRawCopy(src, dst, logicalSize)) {
            return false;
        }
        System.arraycopy(srcData, 0, dstData, 0, Math.min(srcData.length, dstData.length));
        return true;
    }

    private static boolean tryFastCopyStorage(Tensor src, Tensor dst) {
        if (src.getStorage() == null || dst.getStorage() == null) {
            return false;
        }
        if (!Arrays.equals(src.getStrides(), dst.getStrides())) {
            return false;
        }
        int size = Math.min(src.getFlatDataSize(), dst.getFlatDataSize());
        for (int i = 0; i < size; i++) {
            dst.getStorage().setAsDoubleAt(i, src.getStorage().getAsDoubleAt(i));
        }
        return true;
    }

    private static boolean canUseRawCopy(Tensor src, Tensor dst, int logicalSize) {
        return (src.isContiguous() && dst.isContiguous())
                || Arrays.equals(src.getStrides(), dst.getStrides())
                || logicalSize <= 1;
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
