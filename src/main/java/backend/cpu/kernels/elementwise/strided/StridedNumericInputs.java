package backend.cpu.kernels.elementwise.strided;

import tensor.Tensor;

import java.util.List;

final class StridedNumericInputs {
    private StridedNumericInputs() {
    }

    record F64(
            double[] a,
            double[] b,
            int[] aStrides,
            int[] bStrides,
            int aBaseOffset,
            int bBaseOffset,
            double[] out,
            int[] outShape,
            int[] outStrides,
            int outBaseOffset,
            int rank,
            int logicalSize
    ) {
    }

    record F32(
            float[] a,
            float[] b,
            int[] aStrides,
            int[] bStrides,
            int aBaseOffset,
            int bBaseOffset,
            float[] out,
            int[] outShape,
            int[] outStrides,
            int outBaseOffset,
            int rank,
            int logicalSize
    ) {
    }

    record BF16(
            short[] a,
            short[] b,
            int[] aStrides,
            int[] bStrides,
            int aBaseOffset,
            int bBaseOffset,
            short[] out,
            int[] outShape,
            int[] outStrides,
            int outBaseOffset,
            int rank,
            int logicalSize
    ) {
    }

    static F64 prepareF64(List<Tensor> inputs, Tensor node) {
        double[] out = node.getFloat64Data();
        if (out == null) {
            return null;
        }

        int[] outShape = node.getShapeUnsafe();
        int[] outStrides = node.getStridesUnsafe();
        int outBaseOffset = node.getStorageOffsetUnsafe();
        int rank = outShape.length;

        double[] a = null;
        double[] b = null;
        int[] aStrides = null;
        int[] bStrides = null;
        int aBaseOffset = 0;
        int bBaseOffset = 0;
        if (!inputs.isEmpty()) {
            Tensor ta = inputs.get(0);
            a = ta.getFloat64Data();
            aStrides = ta.getStridesUnsafe();
            aBaseOffset = ta.getStorageOffsetUnsafe();
        }
        if (inputs.size() > 1) {
            Tensor tb = inputs.get(1);
            b = tb.getFloat64Data();
            bStrides = tb.getStridesUnsafe();
            bBaseOffset = tb.getStorageOffsetUnsafe();
        }
        return new F64(a, b, aStrides, bStrides, aBaseOffset, bBaseOffset, out, outShape, outStrides, outBaseOffset, rank, node.getFlatDataSize());
    }

    static F32 prepareF32(List<Tensor> inputs, Tensor node) {
        float[] out = node.getFloat32Data();
        if (out == null) {
            return null;
        }

        int[] outShape = node.getShapeUnsafe();
        int[] outStrides = node.getStridesUnsafe();
        int outBaseOffset = node.getStorageOffsetUnsafe();
        int rank = outShape.length;

        float[] a = null;
        float[] b = null;
        int[] aStrides = null;
        int[] bStrides = null;
        int aBaseOffset = 0;
        int bBaseOffset = 0;
        if (!inputs.isEmpty()) {
            Tensor ta = inputs.get(0);
            a = ta.getFloat32Data();
            aStrides = ta.getStridesUnsafe();
            aBaseOffset = ta.getStorageOffsetUnsafe();
        }
        if (inputs.size() > 1) {
            Tensor tb = inputs.get(1);
            b = tb.getFloat32Data();
            bStrides = tb.getStridesUnsafe();
            bBaseOffset = tb.getStorageOffsetUnsafe();
        }
        return new F32(a, b, aStrides, bStrides, aBaseOffset, bBaseOffset, out, outShape, outStrides, outBaseOffset, rank, node.getFlatDataSize());
    }

    static BF16 prepareBF16(List<Tensor> inputs, Tensor node) {
        short[] out = node.getBFloat16Data();
        if (out == null) {
            return null;
        }

        int[] outShape = node.getShapeUnsafe();
        int[] outStrides = node.getStridesUnsafe();
        int outBaseOffset = node.getStorageOffsetUnsafe();
        int rank = outShape.length;

        short[] a = null;
        short[] b = null;
        int[] aStrides = null;
        int[] bStrides = null;
        int aBaseOffset = 0;
        int bBaseOffset = 0;
        if (!inputs.isEmpty()) {
            Tensor ta = inputs.get(0);
            a = ta.getBFloat16Data();
            aStrides = ta.getStridesUnsafe();
            aBaseOffset = ta.getStorageOffsetUnsafe();
        }
        if (inputs.size() > 1) {
            Tensor tb = inputs.get(1);
            b = tb.getBFloat16Data();
            bStrides = tb.getStridesUnsafe();
            bBaseOffset = tb.getStorageOffsetUnsafe();
        }
        return new BF16(a, b, aStrides, bStrides, aBaseOffset, bBaseOffset, out, outShape, outStrides, outBaseOffset, rank, node.getFlatDataSize());
    }
}
