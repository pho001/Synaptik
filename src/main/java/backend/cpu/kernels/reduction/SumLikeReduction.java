package backend.cpu.kernels.reduction;

import backend.cpu.storage.CpuStorageView;
import tensor.dtype.TensorDTypeOps;

import java.lang.foreign.MemorySegment;

enum SumLikeReduction {
    SUM {
        @Override
        void finalizeF64(CpuStorageView output, CpuStorageView input, int dimension) {
        }

        @Override
        void finalizeF32(CpuStorageView output, CpuStorageView input, int dimension) {
        }

        @Override
        void finalizeBF16(CpuStorageView output, CpuStorageView input, int dimension) {
        }
    },
    MEAN {
        @Override
        void finalizeF64(CpuStorageView output, CpuStorageView input, int dimension) {
            double[] out = ReductionStorageAccess.f64Array(output);
            MemorySegment outSegment = ReductionStorageAccess.f64Segment(output);
            double scale = 1.0d / divisor(input, dimension);
            int[] shape = output.shape();
            int[] strides = output.strides();
            for (int i = 0; i < output.logicalSize(); i++) {
                int offset = ReductionStorageAccess.logicalToOffset(i, shape, strides, output.storageOffset());
                ReductionStorageAccess.writeF64(
                        out,
                        outSegment,
                        offset,
                        ReductionStorageAccess.readF64(out, outSegment, offset) * scale
                );
            }
        }

        @Override
        void finalizeF32(CpuStorageView output, CpuStorageView input, int dimension) {
            float[] out = ReductionStorageAccess.f32Array(output);
            MemorySegment outSegment = ReductionStorageAccess.f32Segment(output);
            float scale = 1.0f / divisor(input, dimension);
            int[] shape = output.shape();
            int[] strides = output.strides();
            for (int i = 0; i < output.logicalSize(); i++) {
                int offset = ReductionStorageAccess.logicalToOffset(i, shape, strides, output.storageOffset());
                ReductionStorageAccess.writeF32(
                        out,
                        outSegment,
                        offset,
                        ReductionStorageAccess.readF32(out, outSegment, offset) * scale
                );
            }
        }

        @Override
        void finalizeBF16(CpuStorageView output, CpuStorageView input, int dimension) {
            short[] out = ReductionStorageAccess.bf16Array(output);
            MemorySegment outSegment = ReductionStorageAccess.bf16Segment(output);
            float scale = 1.0f / divisor(input, dimension);
            int[] shape = output.shape();
            int[] strides = output.strides();
            for (int i = 0; i < output.logicalSize(); i++) {
                int offset = ReductionStorageAccess.logicalToOffset(i, shape, strides, output.storageOffset());
                short value = ReductionStorageAccess.readBF16(out, outSegment, offset);
                ReductionStorageAccess.writeBF16(
                        out,
                        outSegment,
                        offset,
                        TensorDTypeOps.toBFloat16Bits(TensorDTypeOps.fromBFloat16Bits(value) * scale)
                );
            }
        }

        private int divisor(CpuStorageView input, int dimension) {
            return dimension == -1 ? input.logicalSize() : input.shape()[dimension];
        }
    };

    abstract void finalizeF64(CpuStorageView output, CpuStorageView input, int dimension);

    abstract void finalizeF32(CpuStorageView output, CpuStorageView input, int dimension);

    abstract void finalizeBF16(CpuStorageView output, CpuStorageView input, int dimension);
}
