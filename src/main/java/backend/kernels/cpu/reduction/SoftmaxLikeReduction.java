package backend.kernels.cpu.reduction;

import backend.kernels.cpu.CpuDTypeOps;

enum SoftmaxLikeReduction {
    SOFTMAX {
        @Override
        void computeF64(double[] in, double[] out, int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisSize) {
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
                max = Math.max(max, CpuDTypeOps.fromBFloat16Bits(in[inOffset]));
            }
            float sum = 0.0f;
            for (int i = 0, inOffset = baseIn, outOffset = baseOut; i < axisSize; i++, inOffset += axisStrideIn, outOffset += axisStrideOut) {
                float value = (float) Math.exp(CpuDTypeOps.fromBFloat16Bits(in[inOffset]) - max);
                out[outOffset] = CpuDTypeOps.toBFloat16Bits(value);
                sum += value;
            }
            float inv = 1.0f / sum;
            for (int i = 0, outOffset = baseOut; i < axisSize; i++, outOffset += axisStrideOut) {
                out[outOffset] = CpuDTypeOps.toBFloat16Bits(CpuDTypeOps.fromBFloat16Bits(out[outOffset]) * inv);
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
                out[outOffset] = CpuDTypeOps.toBFloat16Bits(value);
                sum += value;
            }
            float inv = 1.0f / sum;
            for (int i = 0, outOffset = baseOut; i < axisSize; i++, outOffset += axisStrideOut) {
                out[outOffset] = CpuDTypeOps.toBFloat16Bits(CpuDTypeOps.fromBFloat16Bits(out[outOffset]) * inv);
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
                max = Math.max(max, CpuDTypeOps.fromBFloat16Bits(in[inOffset]));
            }
            float sum = 0.0f;
            for (int i = 0, inOffset = baseIn; i < axisSize; i++, inOffset += axisStrideIn) {
                sum += (float) Math.exp(CpuDTypeOps.fromBFloat16Bits(in[inOffset]) - max);
            }
            float logSumExp = (float) (max + Math.log(sum));
            for (int i = 0, inOffset = baseIn, outOffset = baseOut; i < axisSize; i++, inOffset += axisStrideIn, outOffset += axisStrideOut) {
                out[outOffset] = CpuDTypeOps.toBFloat16Bits(CpuDTypeOps.fromBFloat16Bits(in[inOffset]) - logSumExp);
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
                out[outOffset] = CpuDTypeOps.toBFloat16Bits(in[inOffset] - logSumExp);
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
}
