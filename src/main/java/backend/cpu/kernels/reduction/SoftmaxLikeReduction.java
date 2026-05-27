package backend.cpu.kernels.reduction;

import tensor.dtype.TensorDTypeOps;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

enum SoftmaxLikeReduction {
    SOFTMAX {
        @Override
        void computeF64(double[] in, double[] out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
            if (canUseContiguousVectorPath(axisStrideIn, axisStrideOut, axisSize, F64_SPECIES.length())) {
                softmaxContiguousF64(in, out, baseIn, baseOut, axisSize);
                return;
            }
            double max = Double.NEGATIVE_INFINITY;
            for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
                max = Math.max(max, in[inOffset]);
            }
            double sum = 0.0d;
            for (int i = 0, inOffset = baseIn, outOffset = baseOut; i < axisSize; i++, inOffset += axisStrideIn, outOffset += axisStrideOut) {
                double value = Math.exp(in[inOffset] - max);
                out[outOffset] = value;
                sum += value;
            }
            double inv = 1.0d / sum;
            for (int i = 0, outOffset = baseOut; i < axisSize; i++, outOffset += axisStrideOut) {
                out[outOffset] *= inv;
            }
        }

        @Override
        void computeF32(float[] in, float[] out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
            if (canUseContiguousVectorPath(axisStrideIn, axisStrideOut, axisSize, F32_SPECIES.length())) {
                softmaxContiguousF32(in, out, baseIn, baseOut, axisSize);
                return;
            }
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
            float inv = 1.0f / sum;
            for (int i = 0, outOffset = baseOut; i < axisSize; i++, outOffset += axisStrideOut) {
                out[outOffset] *= inv;
            }
        }

        @Override
        void computeBF16(short[] in, short[] out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
            float max = Float.NEGATIVE_INFINITY;
            for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
                max = Math.max(max, TensorDTypeOps.fromBFloat16Bits(in[inOffset]));
            }
            float sum = 0.0f;
            for (int i = 0, inOffset = baseIn, outOffset = baseOut; i < axisSize; i++, inOffset += axisStrideIn, outOffset += axisStrideOut) {
                float value = (float) Math.exp(TensorDTypeOps.fromBFloat16Bits(in[inOffset]) - max);
                out[outOffset] = TensorDTypeOps.toBFloat16Bits(value);
                sum += value;
            }
            float inv = 1.0f / sum;
            for (int i = 0, outOffset = baseOut; i < axisSize; i++, outOffset += axisStrideOut) {
                out[outOffset] = TensorDTypeOps.toBFloat16Bits(TensorDTypeOps.fromBFloat16Bits(out[outOffset]) * inv);
            }
        }

        @Override
        void computeF32ToBF16(float[] in, short[] out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
            float max = Float.NEGATIVE_INFINITY;
            for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
                max = Math.max(max, in[inOffset]);
            }
            float sum = 0.0f;
            for (int i = 0, inOffset = baseIn, outOffset = baseOut; i < axisSize; i++, inOffset += axisStrideIn, outOffset += axisStrideOut) {
                float value = (float) Math.exp(in[inOffset] - max);
                out[outOffset] = TensorDTypeOps.toBFloat16Bits(value);
                sum += value;
            }
            float inv = 1.0f / sum;
            for (int i = 0, outOffset = baseOut; i < axisSize; i++, outOffset += axisStrideOut) {
                out[outOffset] = TensorDTypeOps.toBFloat16Bits(TensorDTypeOps.fromBFloat16Bits(out[outOffset]) * inv);
            }
        }

        @Override
        void computeF32ToFloat(float[] in, float[] out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
            computeF32(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize);
        }
    },
    LOG_SOFTMAX {
        @Override
        void computeF64(double[] in, double[] out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
            if (canUseContiguousVectorPath(axisStrideIn, axisStrideOut, axisSize, F64_SPECIES.length())) {
                logSoftmaxContiguousF64(in, out, baseIn, baseOut, axisSize);
                return;
            }
            double max = Double.NEGATIVE_INFINITY;
            for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
                max = Math.max(max, in[inOffset]);
            }
            double sum = 0.0d;
            for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
                sum += Math.exp(in[inOffset] - max);
            }
            double logSumExp = max + Math.log(sum);
            for (int i = 0, inOffset = baseIn, outOffset = baseOut; i < axisSize; i++, inOffset += axisStrideIn, outOffset += axisStrideOut) {
                out[outOffset] = in[inOffset] - logSumExp;
            }
        }

        @Override
        void computeF32(float[] in, float[] out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
            if (canUseContiguousVectorPath(axisStrideIn, axisStrideOut, axisSize, F32_SPECIES.length())) {
                logSoftmaxContiguousF32(in, out, baseIn, baseOut, axisSize);
                return;
            }
            float max = Float.NEGATIVE_INFINITY;
            for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
                max = Math.max(max, in[inOffset]);
            }
            float sum = 0.0f;
            for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
                sum += (float) Math.exp(in[inOffset] - max);
            }
            float logSumExp = (float) (max + Math.log(sum));
            for (int i = 0, inOffset = baseIn, outOffset = baseOut; i < axisSize; i++, inOffset += axisStrideIn, outOffset += axisStrideOut) {
                out[outOffset] = in[inOffset] - logSumExp;
            }
        }

        @Override
        void computeBF16(short[] in, short[] out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
            float max = Float.NEGATIVE_INFINITY;
            for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
                max = Math.max(max, TensorDTypeOps.fromBFloat16Bits(in[inOffset]));
            }
            float sum = 0.0f;
            for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
                sum += (float) Math.exp(TensorDTypeOps.fromBFloat16Bits(in[inOffset]) - max);
            }
            float logSumExp = (float) (max + Math.log(sum));
            for (int i = 0, inOffset = baseIn, outOffset = baseOut; i < axisSize; i++, inOffset += axisStrideIn, outOffset += axisStrideOut) {
                out[outOffset] = TensorDTypeOps.toBFloat16Bits(TensorDTypeOps.fromBFloat16Bits(in[inOffset]) - logSumExp);
            }
        }

        @Override
        void computeF32ToBF16(float[] in, short[] out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
            float max = Float.NEGATIVE_INFINITY;
            for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
                max = Math.max(max, in[inOffset]);
            }
            float sum = 0.0f;
            for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
                sum += (float) Math.exp(in[inOffset] - max);
            }
            float logSumExp = (float) (max + Math.log(sum));
            for (int i = 0, inOffset = baseIn, outOffset = baseOut; i < axisSize; i++, inOffset += axisStrideIn, outOffset += axisStrideOut) {
                out[outOffset] = TensorDTypeOps.toBFloat16Bits(in[inOffset] - logSumExp);
            }
        }

        @Override
        void computeF32ToFloat(float[] in, float[] out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
            computeF32(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize);
        }
    };

    abstract void computeF64(double[] in, double[] out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize);

    abstract void computeF32(float[] in, float[] out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize);

    abstract void computeBF16(short[] in, short[] out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize);

    abstract void computeF32ToBF16(float[] in, short[] out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize);

    abstract void computeF32ToFloat(float[] in, float[] out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize);

    void computeF64(MemorySegment in, MemorySegment out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
        if (this == SOFTMAX) {
            softmaxF64(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize);
        } else {
            logSoftmaxF64(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize);
        }
    }

    void computeF64(double[] in, MemorySegment out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
        if (this == SOFTMAX) {
            softmaxF64(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize);
        } else {
            logSoftmaxF64(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize);
        }
    }

    void computeF64(MemorySegment in, double[] out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
        if (this == SOFTMAX) {
            softmaxF64(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize);
        } else {
            logSoftmaxF64(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize);
        }
    }

    void computeF32(MemorySegment in, MemorySegment out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
        if (this == SOFTMAX) {
            softmaxF32(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize);
        } else {
            logSoftmaxF32(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize);
        }
    }

    void computeF32(float[] in, MemorySegment out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
        if (this == SOFTMAX) {
            softmaxF32(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize);
        } else {
            logSoftmaxF32(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize);
        }
    }

    void computeF32(MemorySegment in, float[] out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
        if (this == SOFTMAX) {
            softmaxF32(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize);
        } else {
            logSoftmaxF32(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize);
        }
    }

    void computeBF16(MemorySegment in, MemorySegment out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
        if (this == SOFTMAX) {
            softmaxBF16(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize);
        } else {
            logSoftmaxBF16(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize);
        }
    }

    void computeBF16(short[] in, MemorySegment out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
        if (this == SOFTMAX) {
            softmaxBF16(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize);
        } else {
            logSoftmaxBF16(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize);
        }
    }

    void computeBF16(MemorySegment in, short[] out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
        if (this == SOFTMAX) {
            softmaxBF16(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize);
        } else {
            logSoftmaxBF16(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize);
        }
    }

    void computeF32ToBF16(float[] in, MemorySegment out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
        if (this == SOFTMAX) {
            softmaxF32ToBF16(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize);
        } else {
            logSoftmaxF32ToBF16(in, out, baseIn, baseOut, axisStrideIn, axisStrideOut, axisSize);
        }
    }

    private static final VectorSpecies<Float> F32_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Double> F64_SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final int MIN_VECTOR_AXIS_MULTIPLIER = 4;

    private static boolean canUseContiguousVectorPath(int axisStrideIn, int axisStrideOut, int axisSize, int speciesLength) {
        return axisStrideIn == 1
                && axisStrideOut == 1
                && speciesLength > 1
                && axisSize >= speciesLength * MIN_VECTOR_AXIS_MULTIPLIER;
    }

    private static void softmaxF64(MemorySegment in, MemorySegment out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
            max = Math.max(max, readF64(in, inOffset));
        }
        double sum = 0.0d;
        for (int i = 0, inOffset = baseIn, outOffset = baseOut; i < axisSize; i++, inOffset += axisStrideIn, outOffset += axisStrideOut) {
            double value = Math.exp(readF64(in, inOffset) - max);
            writeF64(out, outOffset, value);
            sum += value;
        }
        double inv = 1.0d / sum;
        for (int i = 0, outOffset = baseOut; i < axisSize; i++, outOffset += axisStrideOut) {
            writeF64(out, outOffset, readF64(out, outOffset) * inv);
        }
    }

    private static void softmaxF64(double[] in, MemorySegment out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
            max = Math.max(max, in[inOffset]);
        }
        double sum = 0.0d;
        for (int i = 0, inOffset = baseIn, outOffset = baseOut; i < axisSize; i++, inOffset += axisStrideIn, outOffset += axisStrideOut) {
            double value = Math.exp(in[inOffset] - max);
            writeF64(out, outOffset, value);
            sum += value;
        }
        double inv = 1.0d / sum;
        for (int i = 0, outOffset = baseOut; i < axisSize; i++, outOffset += axisStrideOut) {
            writeF64(out, outOffset, readF64(out, outOffset) * inv);
        }
    }

    private static void softmaxF64(MemorySegment in, double[] out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
            max = Math.max(max, readF64(in, inOffset));
        }
        double sum = 0.0d;
        for (int i = 0, inOffset = baseIn, outOffset = baseOut; i < axisSize; i++, inOffset += axisStrideIn, outOffset += axisStrideOut) {
            double value = Math.exp(readF64(in, inOffset) - max);
            out[outOffset] = value;
            sum += value;
        }
        double inv = 1.0d / sum;
        for (int i = 0, outOffset = baseOut; i < axisSize; i++, outOffset += axisStrideOut) {
            out[outOffset] *= inv;
        }
    }

    private static void logSoftmaxF64(MemorySegment in, MemorySegment out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
            max = Math.max(max, readF64(in, inOffset));
        }
        double sum = 0.0d;
        for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
            sum += Math.exp(readF64(in, inOffset) - max);
        }
        double logSumExp = max + Math.log(sum);
        for (int i = 0, inOffset = baseIn, outOffset = baseOut; i < axisSize; i++, inOffset += axisStrideIn, outOffset += axisStrideOut) {
            writeF64(out, outOffset, readF64(in, inOffset) - logSumExp);
        }
    }

    private static void logSoftmaxF64(double[] in, MemorySegment out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
            max = Math.max(max, in[inOffset]);
        }
        double sum = 0.0d;
        for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
            sum += Math.exp(in[inOffset] - max);
        }
        double logSumExp = max + Math.log(sum);
        for (int i = 0, inOffset = baseIn, outOffset = baseOut; i < axisSize; i++, inOffset += axisStrideIn, outOffset += axisStrideOut) {
            writeF64(out, outOffset, in[inOffset] - logSumExp);
        }
    }

    private static void logSoftmaxF64(MemorySegment in, double[] out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
            max = Math.max(max, readF64(in, inOffset));
        }
        double sum = 0.0d;
        for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
            sum += Math.exp(readF64(in, inOffset) - max);
        }
        double logSumExp = max + Math.log(sum);
        for (int i = 0, inOffset = baseIn, outOffset = baseOut; i < axisSize; i++, inOffset += axisStrideIn, outOffset += axisStrideOut) {
            out[outOffset] = readF64(in, inOffset) - logSumExp;
        }
    }

    private static void softmaxF32(MemorySegment in, MemorySegment out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
            max = Math.max(max, readF32(in, inOffset));
        }
        float sum = 0.0f;
        for (int i = 0, inOffset = baseIn, outOffset = baseOut; i < axisSize; i++, inOffset += axisStrideIn, outOffset += axisStrideOut) {
            float value = (float) Math.exp(readF32(in, inOffset) - max);
            writeF32(out, outOffset, value);
            sum += value;
        }
        float inv = 1.0f / sum;
        for (int i = 0, outOffset = baseOut; i < axisSize; i++, outOffset += axisStrideOut) {
            writeF32(out, outOffset, readF32(out, outOffset) * inv);
        }
    }

    private static void softmaxF32(float[] in, MemorySegment out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
            max = Math.max(max, in[inOffset]);
        }
        float sum = 0.0f;
        for (int i = 0, inOffset = baseIn, outOffset = baseOut; i < axisSize; i++, inOffset += axisStrideIn, outOffset += axisStrideOut) {
            float value = (float) Math.exp(in[inOffset] - max);
            writeF32(out, outOffset, value);
            sum += value;
        }
        float inv = 1.0f / sum;
        for (int i = 0, outOffset = baseOut; i < axisSize; i++, outOffset += axisStrideOut) {
            writeF32(out, outOffset, readF32(out, outOffset) * inv);
        }
    }

    private static void softmaxF32(MemorySegment in, float[] out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
            max = Math.max(max, readF32(in, inOffset));
        }
        float sum = 0.0f;
        for (int i = 0, inOffset = baseIn, outOffset = baseOut; i < axisSize; i++, inOffset += axisStrideIn, outOffset += axisStrideOut) {
            float value = (float) Math.exp(readF32(in, inOffset) - max);
            out[outOffset] = value;
            sum += value;
        }
        float inv = 1.0f / sum;
        for (int i = 0, outOffset = baseOut; i < axisSize; i++, outOffset += axisStrideOut) {
            out[outOffset] *= inv;
        }
    }

    private static void logSoftmaxF32(MemorySegment in, MemorySegment out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
            max = Math.max(max, readF32(in, inOffset));
        }
        float sum = 0.0f;
        for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
            sum += (float) Math.exp(readF32(in, inOffset) - max);
        }
        float logSumExp = (float) (max + Math.log(sum));
        for (int i = 0, inOffset = baseIn, outOffset = baseOut; i < axisSize; i++, inOffset += axisStrideIn, outOffset += axisStrideOut) {
            writeF32(out, outOffset, readF32(in, inOffset) - logSumExp);
        }
    }

    private static void logSoftmaxF32(float[] in, MemorySegment out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
            max = Math.max(max, in[inOffset]);
        }
        float sum = 0.0f;
        for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
            sum += (float) Math.exp(in[inOffset] - max);
        }
        float logSumExp = (float) (max + Math.log(sum));
        for (int i = 0, inOffset = baseIn, outOffset = baseOut; i < axisSize; i++, inOffset += axisStrideIn, outOffset += axisStrideOut) {
            writeF32(out, outOffset, in[inOffset] - logSumExp);
        }
    }

    private static void logSoftmaxF32(MemorySegment in, float[] out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
            max = Math.max(max, readF32(in, inOffset));
        }
        float sum = 0.0f;
        for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
            sum += (float) Math.exp(readF32(in, inOffset) - max);
        }
        float logSumExp = (float) (max + Math.log(sum));
        for (int i = 0, inOffset = baseIn, outOffset = baseOut; i < axisSize; i++, inOffset += axisStrideIn, outOffset += axisStrideOut) {
            out[outOffset] = readF32(in, inOffset) - logSumExp;
        }
    }

    private static void softmaxBF16(MemorySegment in, MemorySegment out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
            max = Math.max(max, TensorDTypeOps.fromBFloat16Bits(readBF16(in, inOffset)));
        }
        float sum = 0.0f;
        for (int i = 0, inOffset = baseIn, outOffset = baseOut; i < axisSize; i++, inOffset += axisStrideIn, outOffset += axisStrideOut) {
            float value = (float) Math.exp(TensorDTypeOps.fromBFloat16Bits(readBF16(in, inOffset)) - max);
            writeBF16(out, outOffset, TensorDTypeOps.toBFloat16Bits(value));
            sum += value;
        }
        float inv = 1.0f / sum;
        for (int i = 0, outOffset = baseOut; i < axisSize; i++, outOffset += axisStrideOut) {
            writeBF16(out, outOffset, TensorDTypeOps.toBFloat16Bits(TensorDTypeOps.fromBFloat16Bits(readBF16(out, outOffset)) * inv));
        }
    }

    private static void softmaxBF16(short[] in, MemorySegment out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
            max = Math.max(max, TensorDTypeOps.fromBFloat16Bits(in[inOffset]));
        }
        float sum = 0.0f;
        for (int i = 0, inOffset = baseIn, outOffset = baseOut; i < axisSize; i++, inOffset += axisStrideIn, outOffset += axisStrideOut) {
            float value = (float) Math.exp(TensorDTypeOps.fromBFloat16Bits(in[inOffset]) - max);
            writeBF16(out, outOffset, TensorDTypeOps.toBFloat16Bits(value));
            sum += value;
        }
        float inv = 1.0f / sum;
        for (int i = 0, outOffset = baseOut; i < axisSize; i++, outOffset += axisStrideOut) {
            writeBF16(out, outOffset, TensorDTypeOps.toBFloat16Bits(TensorDTypeOps.fromBFloat16Bits(readBF16(out, outOffset)) * inv));
        }
    }

    private static void softmaxBF16(MemorySegment in, short[] out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
            max = Math.max(max, TensorDTypeOps.fromBFloat16Bits(readBF16(in, inOffset)));
        }
        float sum = 0.0f;
        for (int i = 0, inOffset = baseIn, outOffset = baseOut; i < axisSize; i++, inOffset += axisStrideIn, outOffset += axisStrideOut) {
            float value = (float) Math.exp(TensorDTypeOps.fromBFloat16Bits(readBF16(in, inOffset)) - max);
            out[outOffset] = TensorDTypeOps.toBFloat16Bits(value);
            sum += value;
        }
        float inv = 1.0f / sum;
        for (int i = 0, outOffset = baseOut; i < axisSize; i++, outOffset += axisStrideOut) {
            out[outOffset] = TensorDTypeOps.toBFloat16Bits(TensorDTypeOps.fromBFloat16Bits(out[outOffset]) * inv);
        }
    }

    private static void logSoftmaxBF16(MemorySegment in, MemorySegment out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
            max = Math.max(max, TensorDTypeOps.fromBFloat16Bits(readBF16(in, inOffset)));
        }
        float sum = 0.0f;
        for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
            sum += (float) Math.exp(TensorDTypeOps.fromBFloat16Bits(readBF16(in, inOffset)) - max);
        }
        float logSumExp = (float) (max + Math.log(sum));
        for (int i = 0, inOffset = baseIn, outOffset = baseOut; i < axisSize; i++, inOffset += axisStrideIn, outOffset += axisStrideOut) {
            writeBF16(out, outOffset, TensorDTypeOps.toBFloat16Bits(TensorDTypeOps.fromBFloat16Bits(readBF16(in, inOffset)) - logSumExp));
        }
    }

    private static void logSoftmaxBF16(short[] in, MemorySegment out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
            max = Math.max(max, TensorDTypeOps.fromBFloat16Bits(in[inOffset]));
        }
        float sum = 0.0f;
        for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
            sum += (float) Math.exp(TensorDTypeOps.fromBFloat16Bits(in[inOffset]) - max);
        }
        float logSumExp = (float) (max + Math.log(sum));
        for (int i = 0, inOffset = baseIn, outOffset = baseOut; i < axisSize; i++, inOffset += axisStrideIn, outOffset += axisStrideOut) {
            writeBF16(out, outOffset, TensorDTypeOps.toBFloat16Bits(TensorDTypeOps.fromBFloat16Bits(in[inOffset]) - logSumExp));
        }
    }

    private static void logSoftmaxBF16(MemorySegment in, short[] out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
            max = Math.max(max, TensorDTypeOps.fromBFloat16Bits(readBF16(in, inOffset)));
        }
        float sum = 0.0f;
        for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
            sum += (float) Math.exp(TensorDTypeOps.fromBFloat16Bits(readBF16(in, inOffset)) - max);
        }
        float logSumExp = (float) (max + Math.log(sum));
        for (int i = 0, inOffset = baseIn, outOffset = baseOut; i < axisSize; i++, inOffset += axisStrideIn, outOffset += axisStrideOut) {
            out[outOffset] = TensorDTypeOps.toBFloat16Bits(TensorDTypeOps.fromBFloat16Bits(readBF16(in, inOffset)) - logSumExp);
        }
    }

    private static void softmaxF32ToBF16(float[] in, MemorySegment out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
            max = Math.max(max, in[inOffset]);
        }
        float sum = 0.0f;
        for (int i = 0, inOffset = baseIn, outOffset = baseOut; i < axisSize; i++, inOffset += axisStrideIn, outOffset += axisStrideOut) {
            float value = (float) Math.exp(in[inOffset] - max);
            writeBF16(out, outOffset, TensorDTypeOps.toBFloat16Bits(value));
            sum += value;
        }
        float inv = 1.0f / sum;
        for (int i = 0, outOffset = baseOut; i < axisSize; i++, outOffset += axisStrideOut) {
            writeBF16(out, outOffset, TensorDTypeOps.toBFloat16Bits(TensorDTypeOps.fromBFloat16Bits(readBF16(out, outOffset)) * inv));
        }
    }

    private static void logSoftmaxF32ToBF16(float[] in, MemorySegment out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
            max = Math.max(max, in[inOffset]);
        }
        float sum = 0.0f;
        for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
            sum += (float) Math.exp(in[inOffset] - max);
        }
        float logSumExp = (float) (max + Math.log(sum));
        for (int i = 0, inOffset = baseIn, outOffset = baseOut; i < axisSize; i++, inOffset += axisStrideIn, outOffset += axisStrideOut) {
            writeBF16(out, outOffset, TensorDTypeOps.toBFloat16Bits(in[inOffset] - logSumExp));
        }
    }

    private static double readF64(MemorySegment segment, int offset) {
        return segment.get(JAVA_DOUBLE, (long) offset * Double.BYTES);
    }

    private static void writeF64(MemorySegment segment, int offset, double value) {
        segment.set(JAVA_DOUBLE, (long) offset * Double.BYTES, value);
    }

    private static float readF32(MemorySegment segment, int offset) {
        return segment.get(JAVA_FLOAT, (long) offset * Float.BYTES);
    }

    private static void writeF32(MemorySegment segment, int offset, float value) {
        segment.set(JAVA_FLOAT, (long) offset * Float.BYTES, value);
    }

    private static short readBF16(MemorySegment segment, int offset) {
        return segment.get(JAVA_SHORT, (long) offset * Short.BYTES);
    }

    private static void writeBF16(MemorySegment segment, int offset, short value) {
        segment.set(JAVA_SHORT, (long) offset * Short.BYTES, value);
    }

    private static void softmaxContiguousF64(double[] in, double[] out, int baseIn, int baseOut, int axisSize) {
        double max = maxContiguousF64(in, baseIn, axisSize);
        double sum = expIntoOutAndAccumulateF64(in, out, baseIn, baseOut, axisSize, max);
        scaleOutContiguousF64(out, baseOut, axisSize, 1.0d / sum);
    }

    private static void softmaxContiguousF32(float[] in, float[] out, int baseIn, int baseOut, int axisSize) {
        float max = maxContiguousF32(in, baseIn, axisSize);
        float sum = expIntoOutAndAccumulateF32(in, out, baseIn, baseOut, axisSize, max);
        scaleOutContiguousF32(out, baseOut, axisSize, 1.0f / sum);
    }

    private static void logSoftmaxContiguousF64(double[] in, double[] out, int baseIn, int baseOut, int axisSize) {
        double max = maxContiguousF64(in, baseIn, axisSize);
        double sum = expAccumulateOnlyF64(in, baseIn, axisSize, max);
        double logSumExp = max + Math.log(sum);
        subtractScalarIntoOutF64(in, out, baseIn, baseOut, axisSize, logSumExp);
    }

    private static void logSoftmaxContiguousF32(float[] in, float[] out, int baseIn, int baseOut, int axisSize) {
        float max = maxContiguousF32(in, baseIn, axisSize);
        float sum = expAccumulateOnlyF32(in, baseIn, axisSize, max);
        float logSumExp = (float) (max + Math.log(sum));
        subtractScalarIntoOutF32(in, out, baseIn, baseOut, axisSize, logSumExp);
    }

    private static double maxContiguousF64(double[] in, int base, int length) {
        int upper = F64_SPECIES.loopBound(length);
        DoubleVector vectorMax = DoubleVector.broadcast(F64_SPECIES, Double.NEGATIVE_INFINITY);
        int i = 0;
        for (; i < upper; i += F64_SPECIES.length()) {
            vectorMax = vectorMax.max(DoubleVector.fromArray(F64_SPECIES, in, base + i));
        }
        double max = vectorMax.reduceLanes(VectorOperators.MAX);
        for (; i < length; i++) {
            max = Math.max(max, in[base + i]);
        }
        return max;
    }

    private static float maxContiguousF32(float[] in, int base, int length) {
        int upper = F32_SPECIES.loopBound(length);
        FloatVector vectorMax = FloatVector.broadcast(F32_SPECIES, Float.NEGATIVE_INFINITY);
        int i = 0;
        for (; i < upper; i += F32_SPECIES.length()) {
            vectorMax = vectorMax.max(FloatVector.fromArray(F32_SPECIES, in, base + i));
        }
        float max = vectorMax.reduceLanes(VectorOperators.MAX);
        for (; i < length; i++) {
            max = Math.max(max, in[base + i]);
        }
        return max;
    }

    private static double expIntoOutAndAccumulateF64(double[] in, double[] out, int baseIn, int baseOut, int length, double max) {
        int upper = F64_SPECIES.loopBound(length);
        DoubleVector sumVector = DoubleVector.zero(F64_SPECIES);
        DoubleVector maxVector = DoubleVector.broadcast(F64_SPECIES, max);
        int i = 0;
        for (; i < upper; i += F64_SPECIES.length()) {
            DoubleVector values = DoubleVector.fromArray(F64_SPECIES, in, baseIn + i).sub(maxVector).lanewise(VectorOperators.EXP);
            values.intoArray(out, baseOut + i);
            sumVector = sumVector.add(values);
        }
        double sum = sumVector.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            double value = Math.exp(in[baseIn + i] - max);
            out[baseOut + i] = value;
            sum += value;
        }
        return sum;
    }

    private static float expIntoOutAndAccumulateF32(float[] in, float[] out, int baseIn, int baseOut, int length, float max) {
        int upper = F32_SPECIES.loopBound(length);
        FloatVector sumVector = FloatVector.zero(F32_SPECIES);
        FloatVector maxVector = FloatVector.broadcast(F32_SPECIES, max);
        int i = 0;
        for (; i < upper; i += F32_SPECIES.length()) {
            FloatVector values = FloatVector.fromArray(F32_SPECIES, in, baseIn + i).sub(maxVector).lanewise(VectorOperators.EXP);
            values.intoArray(out, baseOut + i);
            sumVector = sumVector.add(values);
        }
        float sum = sumVector.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            float value = (float) Math.exp(in[baseIn + i] - max);
            out[baseOut + i] = value;
            sum += value;
        }
        return sum;
    }

    private static double expAccumulateOnlyF64(double[] in, int baseIn, int length, double max) {
        int upper = F64_SPECIES.loopBound(length);
        DoubleVector sumVector = DoubleVector.zero(F64_SPECIES);
        DoubleVector maxVector = DoubleVector.broadcast(F64_SPECIES, max);
        int i = 0;
        for (; i < upper; i += F64_SPECIES.length()) {
            DoubleVector values = DoubleVector.fromArray(F64_SPECIES, in, baseIn + i).sub(maxVector).lanewise(VectorOperators.EXP);
            sumVector = sumVector.add(values);
        }
        double sum = sumVector.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            sum += Math.exp(in[baseIn + i] - max);
        }
        return sum;
    }

    private static float expAccumulateOnlyF32(float[] in, int baseIn, int length, float max) {
        int upper = F32_SPECIES.loopBound(length);
        FloatVector sumVector = FloatVector.zero(F32_SPECIES);
        FloatVector maxVector = FloatVector.broadcast(F32_SPECIES, max);
        int i = 0;
        for (; i < upper; i += F32_SPECIES.length()) {
            FloatVector values = FloatVector.fromArray(F32_SPECIES, in, baseIn + i).sub(maxVector).lanewise(VectorOperators.EXP);
            sumVector = sumVector.add(values);
        }
        float sum = sumVector.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            sum += (float) Math.exp(in[baseIn + i] - max);
        }
        return sum;
    }

    private static void scaleOutContiguousF64(double[] out, int baseOut, int length, double scale) {
        int upper = F64_SPECIES.loopBound(length);
        DoubleVector scaleVector = DoubleVector.broadcast(F64_SPECIES, scale);
        int i = 0;
        for (; i < upper; i += F64_SPECIES.length()) {
            DoubleVector.fromArray(F64_SPECIES, out, baseOut + i).mul(scaleVector).intoArray(out, baseOut + i);
        }
        for (; i < length; i++) {
            out[baseOut + i] *= scale;
        }
    }

    private static void scaleOutContiguousF32(float[] out, int baseOut, int length, float scale) {
        int upper = F32_SPECIES.loopBound(length);
        FloatVector scaleVector = FloatVector.broadcast(F32_SPECIES, scale);
        int i = 0;
        for (; i < upper; i += F32_SPECIES.length()) {
            FloatVector.fromArray(F32_SPECIES, out, baseOut + i).mul(scaleVector).intoArray(out, baseOut + i);
        }
        for (; i < length; i++) {
            out[baseOut + i] *= scale;
        }
    }

    private static void subtractScalarIntoOutF64(double[] in, double[] out, int baseIn, int baseOut, int length, double scalar) {
        int upper = F64_SPECIES.loopBound(length);
        DoubleVector scalarVector = DoubleVector.broadcast(F64_SPECIES, scalar);
        int i = 0;
        for (; i < upper; i += F64_SPECIES.length()) {
            DoubleVector.fromArray(F64_SPECIES, in, baseIn + i).sub(scalarVector).intoArray(out, baseOut + i);
        }
        for (; i < length; i++) {
            out[baseOut + i] = in[baseIn + i] - scalar;
        }
    }

    private static void subtractScalarIntoOutF32(float[] in, float[] out, int baseIn, int baseOut, int length, float scalar) {
        int upper = F32_SPECIES.loopBound(length);
        FloatVector scalarVector = FloatVector.broadcast(F32_SPECIES, scalar);
        int i = 0;
        for (; i < upper; i += F32_SPECIES.length()) {
            FloatVector.fromArray(F32_SPECIES, in, baseIn + i).sub(scalarVector).intoArray(out, baseOut + i);
        }
        for (; i < length; i++) {
            out[baseOut + i] = in[baseIn + i] - scalar;
        }
    }
}
