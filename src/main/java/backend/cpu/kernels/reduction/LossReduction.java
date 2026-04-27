package backend.cpu.kernels.reduction;

import backend.cpu.kernels.CpuDTypeOps;

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
                loss -= CpuDTypeOps.fromBFloat16Bits(bData[bOffset]) * CpuDTypeOps.fromBFloat16Bits(aData[aOffset]);
            }
            return loss;
        }

        @Override
        double computeF32ToBF16(float[] aData, short[] bData, int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
            double loss = 0.0d;
            for (int i = 0, aOffset = baseA, bOffset = baseB; i < axisSize; i++, aOffset += axisStrideA, bOffset += axisStrideB) {
                loss -= CpuDTypeOps.fromBFloat16Bits(bData[bOffset]) * aData[aOffset];
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
                max = Math.max(max, CpuDTypeOps.fromBFloat16Bits(aData[offset]));
            }
            double sumExp = 0.0d;
            double weightedLogits = 0.0d;
            double targetSum = 0.0d;
            for (int i = 0, aOffset = baseA, bOffset = baseB; i < axisSize; i++, aOffset += axisStrideA, bOffset += axisStrideB) {
                double target = CpuDTypeOps.fromBFloat16Bits(bData[bOffset]);
                double logit = CpuDTypeOps.fromBFloat16Bits(aData[aOffset]);
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
                double target = CpuDTypeOps.fromBFloat16Bits(bData[bOffset]);
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
}
