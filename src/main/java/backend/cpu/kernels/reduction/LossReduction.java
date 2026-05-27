package backend.cpu.kernels.reduction;

import tensor.dtype.TensorDTypeOps;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

enum LossReduction {
    NLL {
        @Override
        double computeF64(double[] aData, double[] bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
            double loss = 0.0d;
            for (int i = 0, aOffset = baseA, bOffset = baseB; i < axisSize; i++, aOffset += axisStrideA, bOffset += axisStrideB) {
                loss -= bData[bOffset] * aData[aOffset];
            }
            return loss;
        }

        @Override
        double computeF32(float[] aData, float[] bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
            double loss = 0.0d;
            for (int i = 0, aOffset = baseA, bOffset = baseB; i < axisSize; i++, aOffset += axisStrideA, bOffset += axisStrideB) {
                loss -= bData[bOffset] * aData[aOffset];
            }
            return loss;
        }

        @Override
        double computeBF16(short[] aData, short[] bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
            double loss = 0.0d;
            for (int i = 0, aOffset = baseA, bOffset = baseB; i < axisSize; i++, aOffset += axisStrideA, bOffset += axisStrideB) {
                loss -= TensorDTypeOps.fromBFloat16Bits(bData[bOffset]) * TensorDTypeOps.fromBFloat16Bits(aData[aOffset]);
            }
            return loss;
        }

        @Override
        double computeF32ToBF16(float[] aData, short[] bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
            double loss = 0.0d;
            for (int i = 0, aOffset = baseA, bOffset = baseB; i < axisSize; i++, aOffset += axisStrideA, bOffset += axisStrideB) {
                loss -= TensorDTypeOps.fromBFloat16Bits(bData[bOffset]) * aData[aOffset];
            }
            return loss;
        }
    },
    CROSS_ENTROPY {
        @Override
        double computeF64(double[] aData, double[] bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
            double max = Double.NEGATIVE_INFINITY;
            for (int i = 0, offset = baseA; i < axisSize; i++, offset += axisStrideA) {
                max = Math.max(max, aData[offset]);
            }
            double sumExp = 0.0d;
            double weightedLogits = 0.0d;
            double targetSum = 0.0d;
            for (int i = 0, aOffset = baseA, bOffset = baseB; i < axisSize; i++, aOffset += axisStrideA, bOffset += axisStrideB) {
                double target = bData[bOffset];
                double logit = aData[aOffset];
                sumExp += Math.exp(logit - max);
                weightedLogits += target * logit;
                targetSum += target;
            }
            return targetSum * (max + Math.log(sumExp)) - weightedLogits;
        }

        @Override
        double computeF32(float[] aData, float[] bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
            float max = Float.NEGATIVE_INFINITY;
            for (int i = 0, offset = baseA; i < axisSize; i++, offset += axisStrideA) {
                max = Math.max(max, aData[offset]);
            }
            double sumExp = 0.0d;
            double weightedLogits = 0.0d;
            double targetSum = 0.0d;
            for (int i = 0, aOffset = baseA, bOffset = baseB; i < axisSize; i++, aOffset += axisStrideA, bOffset += axisStrideB) {
                double target = bData[bOffset];
                double logit = aData[aOffset];
                sumExp += Math.exp(logit - max);
                weightedLogits += target * logit;
                targetSum += target;
            }
            return targetSum * (max + Math.log(sumExp)) - weightedLogits;
        }

        @Override
        double computeBF16(short[] aData, short[] bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
            float max = Float.NEGATIVE_INFINITY;
            for (int i = 0, offset = baseA; i < axisSize; i++, offset += axisStrideA) {
                max = Math.max(max, TensorDTypeOps.fromBFloat16Bits(aData[offset]));
            }
            double sumExp = 0.0d;
            double weightedLogits = 0.0d;
            double targetSum = 0.0d;
            for (int i = 0, aOffset = baseA, bOffset = baseB; i < axisSize; i++, aOffset += axisStrideA, bOffset += axisStrideB) {
                double target = TensorDTypeOps.fromBFloat16Bits(bData[bOffset]);
                double logit = TensorDTypeOps.fromBFloat16Bits(aData[aOffset]);
                sumExp += Math.exp(logit - max);
                weightedLogits += target * logit;
                targetSum += target;
            }
            return targetSum * (max + Math.log(sumExp)) - weightedLogits;
        }

        @Override
        double computeF32ToBF16(float[] aData, short[] bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
            float max = Float.NEGATIVE_INFINITY;
            for (int i = 0, offset = baseA; i < axisSize; i++, offset += axisStrideA) {
                max = Math.max(max, aData[offset]);
            }
            double sumExp = 0.0d;
            double weightedLogits = 0.0d;
            double targetSum = 0.0d;
            for (int i = 0, aOffset = baseA, bOffset = baseB; i < axisSize; i++, aOffset += axisStrideA, bOffset += axisStrideB) {
                double target = TensorDTypeOps.fromBFloat16Bits(bData[bOffset]);
                double logit = aData[aOffset];
                sumExp += Math.exp(logit - max);
                weightedLogits += target * logit;
                targetSum += target;
            }
            return targetSum * (max + Math.log(sumExp)) - weightedLogits;
        }
    };

    abstract double computeF64(double[] aData, double[] bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize);

    abstract double computeF32(float[] aData, float[] bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize);

    abstract double computeBF16(short[] aData, short[] bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize);

    abstract double computeF32ToBF16(float[] aData, short[] bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize);

    double computeF64(MemorySegment aData, MemorySegment bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
        return this == NLL
                ? nllF64(aData, bData, baseA, baseB, axisStrideA, axisStrideB, axisSize)
                : crossEntropyF64(aData, bData, baseA, baseB, axisStrideA, axisStrideB, axisSize);
    }

    double computeF64(double[] aData, MemorySegment bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
        return this == NLL
                ? nllF64(aData, bData, baseA, baseB, axisStrideA, axisStrideB, axisSize)
                : crossEntropyF64(aData, bData, baseA, baseB, axisStrideA, axisStrideB, axisSize);
    }

    double computeF64(MemorySegment aData, double[] bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
        return this == NLL
                ? nllF64(aData, bData, baseA, baseB, axisStrideA, axisStrideB, axisSize)
                : crossEntropyF64(aData, bData, baseA, baseB, axisStrideA, axisStrideB, axisSize);
    }

    double computeF32(MemorySegment aData, MemorySegment bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
        return this == NLL
                ? nllF32(aData, bData, baseA, baseB, axisStrideA, axisStrideB, axisSize)
                : crossEntropyF32(aData, bData, baseA, baseB, axisStrideA, axisStrideB, axisSize);
    }

    double computeF32(float[] aData, MemorySegment bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
        return this == NLL
                ? nllF32(aData, bData, baseA, baseB, axisStrideA, axisStrideB, axisSize)
                : crossEntropyF32(aData, bData, baseA, baseB, axisStrideA, axisStrideB, axisSize);
    }

    double computeF32(MemorySegment aData, float[] bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
        return this == NLL
                ? nllF32(aData, bData, baseA, baseB, axisStrideA, axisStrideB, axisSize)
                : crossEntropyF32(aData, bData, baseA, baseB, axisStrideA, axisStrideB, axisSize);
    }

    double computeBF16(MemorySegment aData, MemorySegment bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
        return this == NLL
                ? nllBF16(aData, bData, baseA, baseB, axisStrideA, axisStrideB, axisSize)
                : crossEntropyBF16(aData, bData, baseA, baseB, axisStrideA, axisStrideB, axisSize);
    }

    double computeBF16(short[] aData, MemorySegment bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
        return this == NLL
                ? nllBF16(aData, bData, baseA, baseB, axisStrideA, axisStrideB, axisSize)
                : crossEntropyBF16(aData, bData, baseA, baseB, axisStrideA, axisStrideB, axisSize);
    }

    double computeBF16(MemorySegment aData, short[] bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
        return this == NLL
                ? nllBF16(aData, bData, baseA, baseB, axisStrideA, axisStrideB, axisSize)
                : crossEntropyBF16(aData, bData, baseA, baseB, axisStrideA, axisStrideB, axisSize);
    }

    double computeF32ToBF16(float[] aData, MemorySegment bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
        return this == NLL
                ? nllF32ToBF16(aData, bData, baseA, baseB, axisStrideA, axisStrideB, axisSize)
                : crossEntropyF32ToBF16(aData, bData, baseA, baseB, axisStrideA, axisStrideB, axisSize);
    }

    private static double nllF64(MemorySegment aData, MemorySegment bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
        double loss = 0.0d;
        for (int i = 0, aOffset = baseA, bOffset = baseB; i < axisSize; i++, aOffset += axisStrideA, bOffset += axisStrideB) {
            loss -= readF64(bData, bOffset) * readF64(aData, aOffset);
        }
        return loss;
    }

    private static double nllF64(double[] aData, MemorySegment bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
        double loss = 0.0d;
        for (int i = 0, aOffset = baseA, bOffset = baseB; i < axisSize; i++, aOffset += axisStrideA, bOffset += axisStrideB) {
            loss -= readF64(bData, bOffset) * aData[aOffset];
        }
        return loss;
    }

    private static double nllF64(MemorySegment aData, double[] bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
        double loss = 0.0d;
        for (int i = 0, aOffset = baseA, bOffset = baseB; i < axisSize; i++, aOffset += axisStrideA, bOffset += axisStrideB) {
            loss -= bData[bOffset] * readF64(aData, aOffset);
        }
        return loss;
    }

    private static double nllF32(MemorySegment aData, MemorySegment bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
        double loss = 0.0d;
        for (int i = 0, aOffset = baseA, bOffset = baseB; i < axisSize; i++, aOffset += axisStrideA, bOffset += axisStrideB) {
            loss -= readF32(bData, bOffset) * readF32(aData, aOffset);
        }
        return loss;
    }

    private static double nllF32(float[] aData, MemorySegment bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
        double loss = 0.0d;
        for (int i = 0, aOffset = baseA, bOffset = baseB; i < axisSize; i++, aOffset += axisStrideA, bOffset += axisStrideB) {
            loss -= readF32(bData, bOffset) * aData[aOffset];
        }
        return loss;
    }

    private static double nllF32(MemorySegment aData, float[] bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
        double loss = 0.0d;
        for (int i = 0, aOffset = baseA, bOffset = baseB; i < axisSize; i++, aOffset += axisStrideA, bOffset += axisStrideB) {
            loss -= bData[bOffset] * readF32(aData, aOffset);
        }
        return loss;
    }

    private static double nllBF16(MemorySegment aData, MemorySegment bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
        double loss = 0.0d;
        for (int i = 0, aOffset = baseA, bOffset = baseB; i < axisSize; i++, aOffset += axisStrideA, bOffset += axisStrideB) {
            loss -= TensorDTypeOps.fromBFloat16Bits(readBF16(bData, bOffset))
                    * TensorDTypeOps.fromBFloat16Bits(readBF16(aData, aOffset));
        }
        return loss;
    }

    private static double nllBF16(short[] aData, MemorySegment bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
        double loss = 0.0d;
        for (int i = 0, aOffset = baseA, bOffset = baseB; i < axisSize; i++, aOffset += axisStrideA, bOffset += axisStrideB) {
            loss -= TensorDTypeOps.fromBFloat16Bits(readBF16(bData, bOffset)) * TensorDTypeOps.fromBFloat16Bits(aData[aOffset]);
        }
        return loss;
    }

    private static double nllBF16(MemorySegment aData, short[] bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
        double loss = 0.0d;
        for (int i = 0, aOffset = baseA, bOffset = baseB; i < axisSize; i++, aOffset += axisStrideA, bOffset += axisStrideB) {
            loss -= TensorDTypeOps.fromBFloat16Bits(bData[bOffset]) * TensorDTypeOps.fromBFloat16Bits(readBF16(aData, aOffset));
        }
        return loss;
    }

    private static double nllF32ToBF16(float[] aData, MemorySegment bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
        double loss = 0.0d;
        for (int i = 0, aOffset = baseA, bOffset = baseB; i < axisSize; i++, aOffset += axisStrideA, bOffset += axisStrideB) {
            loss -= TensorDTypeOps.fromBFloat16Bits(readBF16(bData, bOffset)) * aData[aOffset];
        }
        return loss;
    }

    private static double crossEntropyF64(MemorySegment aData, MemorySegment bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0, offset = baseA; i < axisSize; i++, offset += axisStrideA) {
            max = Math.max(max, readF64(aData, offset));
        }
        double sumExp = 0.0d;
        double weightedLogits = 0.0d;
        double targetSum = 0.0d;
        for (int i = 0, aOffset = baseA, bOffset = baseB; i < axisSize; i++, aOffset += axisStrideA, bOffset += axisStrideB) {
            double target = readF64(bData, bOffset);
            double logit = readF64(aData, aOffset);
            sumExp += Math.exp(logit - max);
            weightedLogits += target * logit;
            targetSum += target;
        }
        return targetSum * (max + Math.log(sumExp)) - weightedLogits;
    }

    private static double crossEntropyF64(double[] aData, MemorySegment bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0, offset = baseA; i < axisSize; i++, offset += axisStrideA) {
            max = Math.max(max, aData[offset]);
        }
        double sumExp = 0.0d;
        double weightedLogits = 0.0d;
        double targetSum = 0.0d;
        for (int i = 0, aOffset = baseA, bOffset = baseB; i < axisSize; i++, aOffset += axisStrideA, bOffset += axisStrideB) {
            double target = readF64(bData, bOffset);
            double logit = aData[aOffset];
            sumExp += Math.exp(logit - max);
            weightedLogits += target * logit;
            targetSum += target;
        }
        return targetSum * (max + Math.log(sumExp)) - weightedLogits;
    }

    private static double crossEntropyF64(MemorySegment aData, double[] bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0, offset = baseA; i < axisSize; i++, offset += axisStrideA) {
            max = Math.max(max, readF64(aData, offset));
        }
        double sumExp = 0.0d;
        double weightedLogits = 0.0d;
        double targetSum = 0.0d;
        for (int i = 0, aOffset = baseA, bOffset = baseB; i < axisSize; i++, aOffset += axisStrideA, bOffset += axisStrideB) {
            double target = bData[bOffset];
            double logit = readF64(aData, aOffset);
            sumExp += Math.exp(logit - max);
            weightedLogits += target * logit;
            targetSum += target;
        }
        return targetSum * (max + Math.log(sumExp)) - weightedLogits;
    }

    private static double crossEntropyF32(MemorySegment aData, MemorySegment bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0, offset = baseA; i < axisSize; i++, offset += axisStrideA) {
            max = Math.max(max, readF32(aData, offset));
        }
        double sumExp = 0.0d;
        double weightedLogits = 0.0d;
        double targetSum = 0.0d;
        for (int i = 0, aOffset = baseA, bOffset = baseB; i < axisSize; i++, aOffset += axisStrideA, bOffset += axisStrideB) {
            double target = readF32(bData, bOffset);
            double logit = readF32(aData, aOffset);
            sumExp += Math.exp(logit - max);
            weightedLogits += target * logit;
            targetSum += target;
        }
        return targetSum * (max + Math.log(sumExp)) - weightedLogits;
    }

    private static double crossEntropyF32(float[] aData, MemorySegment bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0, offset = baseA; i < axisSize; i++, offset += axisStrideA) {
            max = Math.max(max, aData[offset]);
        }
        double sumExp = 0.0d;
        double weightedLogits = 0.0d;
        double targetSum = 0.0d;
        for (int i = 0, aOffset = baseA, bOffset = baseB; i < axisSize; i++, aOffset += axisStrideA, bOffset += axisStrideB) {
            double target = readF32(bData, bOffset);
            double logit = aData[aOffset];
            sumExp += Math.exp(logit - max);
            weightedLogits += target * logit;
            targetSum += target;
        }
        return targetSum * (max + Math.log(sumExp)) - weightedLogits;
    }

    private static double crossEntropyF32(MemorySegment aData, float[] bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0, offset = baseA; i < axisSize; i++, offset += axisStrideA) {
            max = Math.max(max, readF32(aData, offset));
        }
        double sumExp = 0.0d;
        double weightedLogits = 0.0d;
        double targetSum = 0.0d;
        for (int i = 0, aOffset = baseA, bOffset = baseB; i < axisSize; i++, aOffset += axisStrideA, bOffset += axisStrideB) {
            double target = bData[bOffset];
            double logit = readF32(aData, aOffset);
            sumExp += Math.exp(logit - max);
            weightedLogits += target * logit;
            targetSum += target;
        }
        return targetSum * (max + Math.log(sumExp)) - weightedLogits;
    }

    private static double crossEntropyBF16(MemorySegment aData, MemorySegment bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0, offset = baseA; i < axisSize; i++, offset += axisStrideA) {
            max = Math.max(max, TensorDTypeOps.fromBFloat16Bits(readBF16(aData, offset)));
        }
        double sumExp = 0.0d;
        double weightedLogits = 0.0d;
        double targetSum = 0.0d;
        for (int i = 0, aOffset = baseA, bOffset = baseB; i < axisSize; i++, aOffset += axisStrideA, bOffset += axisStrideB) {
            double target = TensorDTypeOps.fromBFloat16Bits(readBF16(bData, bOffset));
            double logit = TensorDTypeOps.fromBFloat16Bits(readBF16(aData, aOffset));
            sumExp += Math.exp(logit - max);
            weightedLogits += target * logit;
            targetSum += target;
        }
        return targetSum * (max + Math.log(sumExp)) - weightedLogits;
    }

    private static double crossEntropyBF16(short[] aData, MemorySegment bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0, offset = baseA; i < axisSize; i++, offset += axisStrideA) {
            max = Math.max(max, TensorDTypeOps.fromBFloat16Bits(aData[offset]));
        }
        double sumExp = 0.0d;
        double weightedLogits = 0.0d;
        double targetSum = 0.0d;
        for (int i = 0, aOffset = baseA, bOffset = baseB; i < axisSize; i++, aOffset += axisStrideA, bOffset += axisStrideB) {
            double target = TensorDTypeOps.fromBFloat16Bits(readBF16(bData, bOffset));
            double logit = TensorDTypeOps.fromBFloat16Bits(aData[aOffset]);
            sumExp += Math.exp(logit - max);
            weightedLogits += target * logit;
            targetSum += target;
        }
        return targetSum * (max + Math.log(sumExp)) - weightedLogits;
    }

    private static double crossEntropyBF16(MemorySegment aData, short[] bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0, offset = baseA; i < axisSize; i++, offset += axisStrideA) {
            max = Math.max(max, TensorDTypeOps.fromBFloat16Bits(readBF16(aData, offset)));
        }
        double sumExp = 0.0d;
        double weightedLogits = 0.0d;
        double targetSum = 0.0d;
        for (int i = 0, aOffset = baseA, bOffset = baseB; i < axisSize; i++, aOffset += axisStrideA, bOffset += axisStrideB) {
            double target = TensorDTypeOps.fromBFloat16Bits(bData[bOffset]);
            double logit = TensorDTypeOps.fromBFloat16Bits(readBF16(aData, aOffset));
            sumExp += Math.exp(logit - max);
            weightedLogits += target * logit;
            targetSum += target;
        }
        return targetSum * (max + Math.log(sumExp)) - weightedLogits;
    }

    private static double crossEntropyF32ToBF16(float[] aData, MemorySegment bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0, offset = baseA; i < axisSize; i++, offset += axisStrideA) {
            max = Math.max(max, aData[offset]);
        }
        double sumExp = 0.0d;
        double weightedLogits = 0.0d;
        double targetSum = 0.0d;
        for (int i = 0, aOffset = baseA, bOffset = baseB; i < axisSize; i++, aOffset += axisStrideA, bOffset += axisStrideB) {
            double target = TensorDTypeOps.fromBFloat16Bits(readBF16(bData, bOffset));
            double logit = aData[aOffset];
            sumExp += Math.exp(logit - max);
            weightedLogits += target * logit;
            targetSum += target;
        }
        return targetSum * (max + Math.log(sumExp)) - weightedLogits;
    }

    private static double readF64(MemorySegment segment, int offset) {
        return segment.get(JAVA_DOUBLE, (long) offset * Double.BYTES);
    }

    private static float readF32(MemorySegment segment, int offset) {
        return segment.get(JAVA_FLOAT, (long) offset * Float.BYTES);
    }

    private static short readBF16(MemorySegment segment, int offset) {
        return segment.get(JAVA_SHORT, (long) offset * Short.BYTES);
    }
}
