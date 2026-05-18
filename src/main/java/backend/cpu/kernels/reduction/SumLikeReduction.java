package backend.cpu.kernels.reduction;

import tensor.TensorInternalAccess;

import backend.cpu.kernels.CpuDTypeOps;
import tensor.Tensor;

enum SumLikeReduction {
    SUM {
        @Override
        void finalizeF64(Tensor node, Tensor input, int dimension) {
        }

        @Override
        void finalizeF32(Tensor node, Tensor input, int dimension) {
        }

        @Override
        void finalizeBF16(Tensor node, Tensor input, int dimension) {
        }
    },
    MEAN {
        @Override
        void finalizeF64(Tensor node, Tensor input, int dimension) {
            double[] out = TensorInternalAccess.float64Data(node);
            double scale = 1.0d / divisor(input, dimension);
            int baseOffset = node.getStorageOffsetUnsafe();
            for (int i = 0; i < node.getFlatDataSize(); i++) {
                out[baseOffset + i] *= scale;
            }
        }

        @Override
        void finalizeF32(Tensor node, Tensor input, int dimension) {
            float[] out = TensorInternalAccess.float32Data(node);
            float scale = 1.0f / divisor(input, dimension);
            int baseOffset = node.getStorageOffsetUnsafe();
            for (int i = 0; i < node.getFlatDataSize(); i++) {
                out[baseOffset + i] *= scale;
            }
        }

        @Override
        void finalizeBF16(Tensor node, Tensor input, int dimension) {
            short[] out = TensorInternalAccess.bfloat16Data(node);
            float scale = 1.0f / divisor(input, dimension);
            int baseOffset = node.getStorageOffsetUnsafe();
            for (int i = 0; i < node.getFlatDataSize(); i++) {
                int idx = baseOffset + i;
                out[idx] = CpuDTypeOps.toBFloat16Bits(CpuDTypeOps.fromBFloat16Bits(out[idx]) * scale);
            }
        }

        private int divisor(Tensor input, int dimension) {
            return dimension == -1 ? input.getFlatDataSize() : input.getShapeUnsafe()[dimension];
        }
    };

    abstract void finalizeF64(Tensor node, Tensor input, int dimension);

    abstract void finalizeF32(Tensor node, Tensor input, int dimension);

    abstract void finalizeBF16(Tensor node, Tensor input, int dimension);
}
