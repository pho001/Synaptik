package backend.cpu.kernels.linalg;

import backend.cpu.storage.CpuStorageView;
import tensor.DataType;
import tensor.dtype.TensorDTypeOps;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

final class CpuMatMulStorageLoops {
    private CpuMatMulStorageLoops() {
    }

    static void execute(CpuStorageView a, CpuStorageView b, CpuStorageView out) {
        validateDTypes(a, b, out);
        MatMulStoragePlan plan = MatMulStoragePlan.create(a, b, out);
        switch (out.dtype()) {
            case FLOAT64 -> executeF64(a, b, out, plan);
            case FLOAT32 -> executeF32(a, b, out, plan);
            case BFLOAT16 -> executeBF16(a, b, out, plan);
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException("MATMUL does not support " + out.dtype());
        }
    }

    static boolean isDenseZeroOffset(CpuStorageView view) {
        if (view.storageOffset() != 0) {
            return false;
        }
        int[] shape = view.shape();
        int[] strides = view.strides();
        int expected = 1;
        for (int dim = shape.length - 1; dim >= 0; dim--) {
            if (strides[dim] != expected) {
                return false;
            }
            expected = Math.multiplyExact(expected, shape[dim]);
        }
        return true;
    }

    static int requiredElementCapacity(CpuStorageView view) {
        int[] shape = view.shape();
        int[] strides = view.strides();
        long maxOffset = view.storageOffset();
        for (int i = 0; i < shape.length; i++) {
            maxOffset = Math.addExact(maxOffset, Math.multiplyExact((long) shape[i] - 1L, strides[i]));
            if (maxOffset > Integer.MAX_VALUE - 1L) {
                throw new IllegalArgumentException("MATMUL storage element capacity overflows int range");
            }
        }
        return (int) maxOffset + 1;
    }

    private static void executeF64(CpuStorageView a, CpuStorageView b, CpuStorageView out, MatMulStoragePlan plan) {
        double[] aArray = a.isArray() ? a.requireF64Array() : null;
        double[] bArray = b.isArray() ? b.requireF64Array() : null;
        double[] outArray = out.isArray() ? out.requireF64Array() : null;
        MemorySegment aSegment = a.isMemorySegment() ? a.requireSegment() : null;
        MemorySegment bSegment = b.isMemorySegment() ? b.requireSegment() : null;
        MemorySegment outSegment = out.isMemorySegment() ? out.requireSegment() : null;

        for (int batch = 0; batch < plan.batchCount; batch++) {
            int aBatchBase = a.storageOffset() + plan.aBatchOffsets[batch];
            int bBatchBase = b.storageOffset() + plan.bBatchOffsets[batch];
            int outBatchBase = out.storageOffset() + plan.outBatchOffsets[batch];
            for (int row = 0; row < plan.m; row++) {
                int aRowBase = aBatchBase + row * plan.aRowStride;
                int outRowBase = outBatchBase + row * plan.outRowStride;
                for (int col = 0; col < plan.n; col++) {
                    double sum = 0.0d;
                    int bColBase = bBatchBase + col * plan.bColStride;
                    for (int p = 0; p < plan.k; p++) {
                        sum += readF64(aArray, aSegment, aRowBase + p * plan.aColStride)
                                * readF64(bArray, bSegment, bColBase + p * plan.bRowStride);
                    }
                    writeF64(outArray, outSegment, outRowBase + col * plan.outColStride, sum);
                }
            }
        }
    }

    private static void executeF32(CpuStorageView a, CpuStorageView b, CpuStorageView out, MatMulStoragePlan plan) {
        float[] aArray = a.isArray() ? a.requireF32Array() : null;
        float[] bArray = b.isArray() ? b.requireF32Array() : null;
        float[] outArray = out.isArray() ? out.requireF32Array() : null;
        MemorySegment aSegment = a.isMemorySegment() ? a.requireSegment() : null;
        MemorySegment bSegment = b.isMemorySegment() ? b.requireSegment() : null;
        MemorySegment outSegment = out.isMemorySegment() ? out.requireSegment() : null;

        for (int batch = 0; batch < plan.batchCount; batch++) {
            int aBatchBase = a.storageOffset() + plan.aBatchOffsets[batch];
            int bBatchBase = b.storageOffset() + plan.bBatchOffsets[batch];
            int outBatchBase = out.storageOffset() + plan.outBatchOffsets[batch];
            for (int row = 0; row < plan.m; row++) {
                int aRowBase = aBatchBase + row * plan.aRowStride;
                int outRowBase = outBatchBase + row * plan.outRowStride;
                for (int col = 0; col < plan.n; col++) {
                    float sum = 0.0f;
                    int bColBase = bBatchBase + col * plan.bColStride;
                    for (int p = 0; p < plan.k; p++) {
                        sum += readF32(aArray, aSegment, aRowBase + p * plan.aColStride)
                                * readF32(bArray, bSegment, bColBase + p * plan.bRowStride);
                    }
                    writeF32(outArray, outSegment, outRowBase + col * plan.outColStride, sum);
                }
            }
        }
    }

    private static void executeBF16(CpuStorageView a, CpuStorageView b, CpuStorageView out, MatMulStoragePlan plan) {
        short[] aArray = a.isArray() ? a.requireBF16Array() : null;
        short[] bArray = b.isArray() ? b.requireBF16Array() : null;
        short[] outArray = out.isArray() ? out.requireBF16Array() : null;
        MemorySegment aSegment = a.isMemorySegment() ? a.requireSegment() : null;
        MemorySegment bSegment = b.isMemorySegment() ? b.requireSegment() : null;
        MemorySegment outSegment = out.isMemorySegment() ? out.requireSegment() : null;

        for (int batch = 0; batch < plan.batchCount; batch++) {
            int aBatchBase = a.storageOffset() + plan.aBatchOffsets[batch];
            int bBatchBase = b.storageOffset() + plan.bBatchOffsets[batch];
            int outBatchBase = out.storageOffset() + plan.outBatchOffsets[batch];
            for (int row = 0; row < plan.m; row++) {
                int aRowBase = aBatchBase + row * plan.aRowStride;
                int outRowBase = outBatchBase + row * plan.outRowStride;
                for (int col = 0; col < plan.n; col++) {
                    float sum = 0.0f;
                    int bColBase = bBatchBase + col * plan.bColStride;
                    for (int p = 0; p < plan.k; p++) {
                        sum += readBF16(aArray, aSegment, aRowBase + p * plan.aColStride)
                                * readBF16(bArray, bSegment, bColBase + p * plan.bRowStride);
                    }
                    writeBF16(outArray, outSegment, outRowBase + col * plan.outColStride, sum);
                }
            }
        }
    }

    private static void validateDTypes(CpuStorageView a, CpuStorageView b, CpuStorageView out) {
        DataType dtype = out.dtype();
        if (a.dtype() != dtype || b.dtype() != dtype) {
            throw new IllegalArgumentException("MATMUL storage dtype mismatch. a="
                    + a.dtype() + ", b=" + b.dtype() + ", out=" + dtype);
        }
    }

    private static double readF64(double[] array, MemorySegment segment, int offset) {
        return array != null ? array[offset] : segment.get(JAVA_DOUBLE, (long) offset * Double.BYTES);
    }

    private static void writeF64(double[] array, MemorySegment segment, int offset, double value) {
        if (array != null) {
            array[offset] = value;
        } else {
            segment.set(JAVA_DOUBLE, (long) offset * Double.BYTES, value);
        }
    }

    private static float readF32(float[] array, MemorySegment segment, int offset) {
        return array != null ? array[offset] : segment.get(JAVA_FLOAT, (long) offset * Float.BYTES);
    }

    private static void writeF32(float[] array, MemorySegment segment, int offset, float value) {
        if (array != null) {
            array[offset] = value;
        } else {
            segment.set(JAVA_FLOAT, (long) offset * Float.BYTES, value);
        }
    }

    private static float readBF16(short[] array, MemorySegment segment, int offset) {
        short bits = array != null ? array[offset] : segment.get(JAVA_SHORT, (long) offset * Short.BYTES);
        return TensorDTypeOps.fromBFloat16Bits(bits);
    }

    private static void writeBF16(short[] array, MemorySegment segment, int offset, float value) {
        short bits = TensorDTypeOps.toBFloat16Bits(value);
        if (array != null) {
            array[offset] = bits;
        } else {
            segment.set(JAVA_SHORT, (long) offset * Short.BYTES, bits);
        }
    }

    private record MatMulStoragePlan(
            int batchCount,
            int m,
            int n,
            int k,
            int aRowStride,
            int aColStride,
            int bRowStride,
            int bColStride,
            int outRowStride,
            int outColStride,
            int[] aBatchOffsets,
            int[] bBatchOffsets,
            int[] outBatchOffsets
    ) {
        static MatMulStoragePlan create(CpuStorageView a, CpuStorageView b, CpuStorageView out) {
            int[] aShape = a.shape();
            int[] bShape = b.shape();
            int[] outShape = out.shape();
            int[] aStrides = a.strides();
            int[] bStrides = b.strides();
            int[] outStrides = out.strides();
            validateShape(aShape, bShape, outShape);

            int batchCount = batchCount(outShape);
            return new MatMulStoragePlan(
                    batchCount,
                    outShape[outShape.length - 2],
                    outShape[outShape.length - 1],
                    aShape[aShape.length - 1],
                    aStrides[aStrides.length - 2],
                    aStrides[aStrides.length - 1],
                    bStrides[bStrides.length - 2],
                    bStrides[bStrides.length - 1],
                    outStrides[outStrides.length - 2],
                    outStrides[outStrides.length - 1],
                    batchOffsets(aShape, aStrides, outShape, true),
                    batchOffsets(bShape, bStrides, outShape, true),
                    batchOffsets(outShape, outStrides, outShape, false)
            );
        }

        private static void validateShape(int[] aShape, int[] bShape, int[] outShape) {
            if (aShape.length < 2 || bShape.length < 2 || outShape.length < 2) {
                throw new IllegalArgumentException("MATMUL requires rank >= 2. a="
                        + Arrays.toString(aShape) + ", b=" + Arrays.toString(bShape)
                        + ", out=" + Arrays.toString(outShape));
            }
            int aBatchRank = aShape.length - 2;
            int bBatchRank = bShape.length - 2;
            int outBatchRank = outShape.length - 2;
            int expectedOutRank = Math.max(aBatchRank, bBatchRank) + 2;
            if (outShape.length != expectedOutRank) {
                throw new IllegalArgumentException("MATMUL output rank mismatch. expected="
                        + expectedOutRank + ", actual=" + outShape.length);
            }
            int m = aShape[aShape.length - 2];
            int k = aShape[aShape.length - 1];
            int bK = bShape[bShape.length - 2];
            int n = bShape[bShape.length - 1];
            if (bK != k || outShape[outShape.length - 2] != m || outShape[outShape.length - 1] != n) {
                throw new IllegalArgumentException("MATMUL core dimensions mismatch. a="
                        + Arrays.toString(aShape) + ", b=" + Arrays.toString(bShape)
                        + ", out=" + Arrays.toString(outShape));
            }
            validateBroadcastBatch(aShape, outShape, outBatchRank);
            validateBroadcastBatch(bShape, outShape, outBatchRank);
        }

        private static void validateBroadcastBatch(int[] inputShape, int[] outShape, int outBatchRank) {
            int inputBatchRank = inputShape.length - 2;
            int shift = outBatchRank - inputBatchRank;
            if (shift < 0) {
                throw new IllegalArgumentException("MATMUL input batch rank exceeds output batch rank.");
            }
            for (int d = 0; d < outBatchRank; d++) {
                int inputDim = d < shift ? 1 : inputShape[d - shift];
                int outDim = outShape[d];
                if (inputDim != 1 && inputDim != outDim) {
                    throw new IllegalArgumentException("MATMUL batch dimensions are not broadcastable. input="
                            + Arrays.toString(inputShape) + ", out=" + Arrays.toString(outShape));
                }
            }
        }

        private static int[] batchOffsets(
                int[] shape,
                int[] strides,
                int[] outShape,
                boolean allowBroadcast
        ) {
            int outBatchRank = outShape.length - 2;
            int inputBatchRank = shape.length - 2;
            int shift = outBatchRank - inputBatchRank;
            int batchCount = batchCount(outShape);
            int[] offsets = new int[batchCount];
            if (outBatchRank == 0) {
                return offsets;
            }
            int[] outBatchShape = Arrays.copyOf(outShape, outBatchRank);
            int[] outBatchDenseStrides = denseStrides(outBatchShape);
            for (int batch = 0; batch < batchCount; batch++) {
                int tmp = batch;
                int offset = 0;
                for (int d = 0; d < outBatchRank; d++) {
                    int coord = tmp / outBatchDenseStrides[d];
                    tmp %= outBatchDenseStrides[d];
                    int inputDimIndex = d - shift;
                    if (inputDimIndex < 0) {
                        continue;
                    }
                    int inputDim = shape[inputDimIndex];
                    if (allowBroadcast && inputDim == 1) {
                        continue;
                    }
                    offset += coord * strides[inputDimIndex];
                }
                offsets[batch] = offset;
            }
            return offsets;
        }

        private static int batchCount(int[] outShape) {
            int count = 1;
            for (int i = 0; i < outShape.length - 2; i++) {
                count = Math.multiplyExact(count, outShape[i]);
            }
            return count;
        }

        private static int[] denseStrides(int[] shape) {
            int[] strides = new int[shape.length];
            int stride = 1;
            for (int i = shape.length - 1; i >= 0; i--) {
                strides[i] = stride;
                stride = Math.multiplyExact(stride, shape[i]);
            }
            return strides;
        }
    }
}
