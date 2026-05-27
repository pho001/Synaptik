package backend.cpu.kernels.linalg;

import backend.cpu.storage.CpuStorageView;
import tensor.DataType;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;

final class CpuLinearStorageLoops {
    private CpuLinearStorageLoops() {
    }

    static void execute(CpuStorageView input, CpuStorageView weight, CpuStorageView bias, CpuStorageView out) {
        validateDTypes(input, weight, bias, out);
        CpuMatMulStorageLoops.execute(input, weight, out);
        if (bias == null) {
            return;
        }
        BiasPlan plan = BiasPlan.create(out, bias);
        switch (out.dtype()) {
            case FLOAT64 -> addBiasF64(out, bias, plan);
            case FLOAT32 -> addBiasF32(out, bias, plan);
            case BFLOAT16 -> addBiasBF16(out, bias, plan);
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException("LINEAR does not support " + out.dtype());
        }
    }

    private static void addBiasF64(CpuStorageView out, CpuStorageView bias, BiasPlan plan) {
        double[] outArray = out.isArray() ? out.requireF64Array() : null;
        double[] biasArray = bias.isArray() ? bias.requireF64Array() : null;
        MemorySegment outSegment = out.isMemorySegment() ? out.requireSegment() : null;
        MemorySegment biasSegment = bias.isMemorySegment() ? bias.requireSegment() : null;

        for (int row = 0; row < plan.rowOffsets.length; row++) {
            int outBase = out.storageOffset() + plan.rowOffsets[row];
            for (int col = 0; col < plan.outFeatures; col++) {
                int outOffset = outBase + col * plan.outFeatureStride;
                double value = CpuMatMulStorageLoops.readF64(outArray, outSegment, outOffset)
                        + CpuMatMulStorageLoops.readF64(biasArray, biasSegment, plan.biasOffset(col));
                CpuMatMulStorageLoops.writeF64(outArray, outSegment, outOffset, value);
            }
        }
    }

    private static void addBiasF32(CpuStorageView out, CpuStorageView bias, BiasPlan plan) {
        float[] outArray = out.isArray() ? out.requireF32Array() : null;
        float[] biasArray = bias.isArray() ? bias.requireF32Array() : null;
        MemorySegment outSegment = out.isMemorySegment() ? out.requireSegment() : null;
        MemorySegment biasSegment = bias.isMemorySegment() ? bias.requireSegment() : null;

        for (int row = 0; row < plan.rowOffsets.length; row++) {
            int outBase = out.storageOffset() + plan.rowOffsets[row];
            for (int col = 0; col < plan.outFeatures; col++) {
                int outOffset = outBase + col * plan.outFeatureStride;
                float value = CpuMatMulStorageLoops.readF32(outArray, outSegment, outOffset)
                        + CpuMatMulStorageLoops.readF32(biasArray, biasSegment, plan.biasOffset(col));
                CpuMatMulStorageLoops.writeF32(outArray, outSegment, outOffset, value);
            }
        }
    }

    private static void addBiasBF16(CpuStorageView out, CpuStorageView bias, BiasPlan plan) {
        short[] outArray = out.isArray() ? out.requireBF16Array() : null;
        short[] biasArray = bias.isArray() ? bias.requireBF16Array() : null;
        MemorySegment outSegment = out.isMemorySegment() ? out.requireSegment() : null;
        MemorySegment biasSegment = bias.isMemorySegment() ? bias.requireSegment() : null;

        for (int row = 0; row < plan.rowOffsets.length; row++) {
            int outBase = out.storageOffset() + plan.rowOffsets[row];
            for (int col = 0; col < plan.outFeatures; col++) {
                int outOffset = outBase + col * plan.outFeatureStride;
                float value = CpuMatMulStorageLoops.readBF16(outArray, outSegment, outOffset)
                        + CpuMatMulStorageLoops.readBF16(biasArray, biasSegment, plan.biasOffset(col));
                CpuMatMulStorageLoops.writeBF16(outArray, outSegment, outOffset, value);
            }
        }
    }

    private static void validateDTypes(
            CpuStorageView input,
            CpuStorageView weight,
            CpuStorageView bias,
            CpuStorageView out
    ) {
        DataType dtype = out.dtype();
        if (input.dtype() != dtype || weight.dtype() != dtype || (bias != null && bias.dtype() != dtype)) {
            throw new IllegalArgumentException("LINEAR storage dtype mismatch. input="
                    + input.dtype() + ", weight=" + weight.dtype()
                    + ", bias=" + (bias == null ? "none" : bias.dtype())
                    + ", out=" + dtype);
        }
    }

    private record BiasPlan(
            int outFeatures,
            int outFeatureStride,
            int biasBase,
            int biasFeatureStride,
            int[] rowOffsets
    ) {
        static BiasPlan create(CpuStorageView out, CpuStorageView bias) {
            int[] outShape = out.shape();
            int[] outStrides = out.strides();
            int[] biasShape = bias.shape();
            int[] biasStrides = bias.strides();
            validateBiasShape(outShape, biasShape);

            int outRank = outShape.length;
            return new BiasPlan(
                    outShape[outRank - 1],
                    outStrides[outRank - 1],
                    bias.storageOffset(),
                    biasShape.length == 1 ? biasStrides[0] : biasStrides[1],
                    rowOffsets(outShape, outStrides)
            );
        }

        int biasOffset(int col) {
            return biasBase + col * biasFeatureStride;
        }

        private static void validateBiasShape(int[] outShape, int[] biasShape) {
            if (outShape.length < 2) {
                throw new IllegalArgumentException("LINEAR output must have rank >= 2. out="
                        + Arrays.toString(outShape));
            }
            int outFeatures = outShape[outShape.length - 1];
            boolean vectorBias = biasShape.length == 1 && biasShape[0] == outFeatures;
            boolean rowBias = biasShape.length == 2 && biasShape[0] == 1 && biasShape[1] == outFeatures;
            if (!vectorBias && !rowBias) {
                throw new IllegalArgumentException("LINEAR bias must have shape [outFeatures] or [1, outFeatures]. bias="
                        + Arrays.toString(biasShape) + ", out=" + Arrays.toString(outShape));
            }
        }

        private static int[] rowOffsets(int[] outShape, int[] outStrides) {
            int prefixRank = outShape.length - 1;
            int rowCount = 1;
            for (int dim = 0; dim < prefixRank; dim++) {
                rowCount = Math.multiplyExact(rowCount, outShape[dim]);
            }
            int[] offsets = new int[rowCount];
            if (prefixRank == 0) {
                return offsets;
            }
            int[] prefixShape = Arrays.copyOf(outShape, prefixRank);
            int[] densePrefixStrides = denseStrides(prefixShape);
            for (int row = 0; row < rowCount; row++) {
                int tmp = row;
                int offset = 0;
                for (int dim = 0; dim < prefixRank; dim++) {
                    int coord = tmp / densePrefixStrides[dim];
                    tmp %= densePrefixStrides[dim];
                    offset += coord * outStrides[dim];
                }
                offsets[row] = offset;
            }
            return offsets;
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
