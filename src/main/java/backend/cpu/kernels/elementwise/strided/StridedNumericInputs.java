package backend.cpu.kernels.elementwise.strided;

import backend.cpu.kernels.elementwise.ElementwiseLayoutPlan;
import tensor.TensorInternalAccess;

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
        double[] out = TensorInternalAccess.float64Data(node);
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
            a = TensorInternalAccess.float64Data(ta);
            ElementwiseLayoutPlan.Operand layout = ElementwiseLayoutPlan.inputOperand(ta, outShape);
            aStrides = layout.strides();
            aBaseOffset = layout.baseOffset();
        }
        if (inputs.size() > 1) {
            Tensor tb = inputs.get(1);
            b = TensorInternalAccess.float64Data(tb);
            ElementwiseLayoutPlan.Operand layout = ElementwiseLayoutPlan.inputOperand(tb, outShape);
            bStrides = layout.strides();
            bBaseOffset = layout.baseOffset();
        }
        return new F64(a, b, aStrides, bStrides, aBaseOffset, bBaseOffset, out, outShape, outStrides, outBaseOffset, rank, node.getFlatDataSize());
    }

    static F32 prepareF32(List<Tensor> inputs, Tensor node) {
        float[] out = TensorInternalAccess.float32Data(node);
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
            a = TensorInternalAccess.float32Data(ta);
            ElementwiseLayoutPlan.Operand layout = ElementwiseLayoutPlan.inputOperand(ta, outShape);
            aStrides = layout.strides();
            aBaseOffset = layout.baseOffset();
        }
        if (inputs.size() > 1) {
            Tensor tb = inputs.get(1);
            b = TensorInternalAccess.float32Data(tb);
            ElementwiseLayoutPlan.Operand layout = ElementwiseLayoutPlan.inputOperand(tb, outShape);
            bStrides = layout.strides();
            bBaseOffset = layout.baseOffset();
        }
        return new F32(a, b, aStrides, bStrides, aBaseOffset, bBaseOffset, out, outShape, outStrides, outBaseOffset, rank, node.getFlatDataSize());
    }

    static BF16 prepareBF16(List<Tensor> inputs, Tensor node) {
        short[] out = TensorInternalAccess.bfloat16Data(node);
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
            a = TensorInternalAccess.bfloat16Data(ta);
            ElementwiseLayoutPlan.Operand layout = ElementwiseLayoutPlan.inputOperand(ta, outShape);
            aStrides = layout.strides();
            aBaseOffset = layout.baseOffset();
        }
        if (inputs.size() > 1) {
            Tensor tb = inputs.get(1);
            b = TensorInternalAccess.bfloat16Data(tb);
            ElementwiseLayoutPlan.Operand layout = ElementwiseLayoutPlan.inputOperand(tb, outShape);
            bStrides = layout.strides();
            bBaseOffset = layout.baseOffset();
        }
        return new BF16(a, b, aStrides, bStrides, aBaseOffset, bBaseOffset, out, outShape, outStrides, outBaseOffset, rank, node.getFlatDataSize());
    }
}
