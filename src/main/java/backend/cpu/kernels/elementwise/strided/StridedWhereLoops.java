package backend.cpu.kernels.elementwise.strided;

import backend.cpu.kernels.elementwise.ElementwiseLayoutPlan;
import tensor.TensorInternalAccess;

import tensor.Tensor;

import java.util.List;

final class StridedWhereLoops {
    private StridedWhereLoops() {
    }

    static void forwardF64(List<Tensor> inputs, Tensor node) {
        double[] out = TensorInternalAccess.float64Data(node);
        if (out == null) {
            return;
        }
        byte[] cond = TensorInternalAccess.boolData(inputs.get(0));
        double[] ifTrue = TensorInternalAccess.float64Data(inputs.get(1));
        double[] ifFalse = TensorInternalAccess.float64Data(inputs.get(2));
        int[] outShape = node.getShapeUnsafe();
        int[] outStrides = node.getStridesUnsafe();
        ElementwiseLayoutPlan.Operand condLayout = ElementwiseLayoutPlan.inputOperand(inputs.get(0), outShape);
        ElementwiseLayoutPlan.Operand trueLayout = ElementwiseLayoutPlan.inputOperand(inputs.get(1), outShape);
        ElementwiseLayoutPlan.Operand falseLayout = ElementwiseLayoutPlan.inputOperand(inputs.get(2), outShape);
        int[] condStrides = condLayout.strides();
        int[] trueStrides = trueLayout.strides();
        int[] falseStrides = falseLayout.strides();
        int outBaseOffset = node.getStorageOffsetUnsafe();
        int condBaseOffset = condLayout.baseOffset();
        int trueBaseOffset = trueLayout.baseOffset();
        int falseBaseOffset = falseLayout.baseOffset();

        StridedScalarLoops.genericWhereF64(
                cond,
                ifTrue,
                ifFalse,
                condStrides,
                trueStrides,
                falseStrides,
                condBaseOffset,
                trueBaseOffset,
                falseBaseOffset,
                out,
                outShape,
                outStrides,
                outBaseOffset,
                node.getFlatDataSize()
        );
    }

    static void forwardF32(List<Tensor> inputs, Tensor node, float[] out) {
        byte[] cond = TensorInternalAccess.boolData(inputs.get(0));
        float[] ifTrue = TensorInternalAccess.float32Data(inputs.get(1));
        float[] ifFalse = TensorInternalAccess.float32Data(inputs.get(2));
        int[] outShape = node.getShapeUnsafe();
        int[] outStrides = node.getStridesUnsafe();
        ElementwiseLayoutPlan.Operand condLayout = ElementwiseLayoutPlan.inputOperand(inputs.get(0), outShape);
        ElementwiseLayoutPlan.Operand trueLayout = ElementwiseLayoutPlan.inputOperand(inputs.get(1), outShape);
        ElementwiseLayoutPlan.Operand falseLayout = ElementwiseLayoutPlan.inputOperand(inputs.get(2), outShape);
        int[] condStrides = condLayout.strides();
        int[] trueStrides = trueLayout.strides();
        int[] falseStrides = falseLayout.strides();
        int outBaseOffset = node.getStorageOffsetUnsafe();
        int condBaseOffset = condLayout.baseOffset();
        int trueBaseOffset = trueLayout.baseOffset();
        int falseBaseOffset = falseLayout.baseOffset();

        StridedScalarLoops.genericWhereF32(
                cond,
                ifTrue,
                ifFalse,
                condStrides,
                trueStrides,
                falseStrides,
                condBaseOffset,
                trueBaseOffset,
                falseBaseOffset,
                out,
                outShape,
                outStrides,
                outBaseOffset,
                node.getFlatDataSize()
        );
    }

    static void forwardBF16(List<Tensor> inputs, Tensor node, short[] out) {
        byte[] cond = TensorInternalAccess.boolData(inputs.get(0));
        short[] ifTrue = TensorInternalAccess.bfloat16Data(inputs.get(1));
        short[] ifFalse = TensorInternalAccess.bfloat16Data(inputs.get(2));
        int[] outShape = node.getShapeUnsafe();
        int[] outStrides = node.getStridesUnsafe();
        ElementwiseLayoutPlan.Operand condLayout = ElementwiseLayoutPlan.inputOperand(inputs.get(0), outShape);
        ElementwiseLayoutPlan.Operand trueLayout = ElementwiseLayoutPlan.inputOperand(inputs.get(1), outShape);
        ElementwiseLayoutPlan.Operand falseLayout = ElementwiseLayoutPlan.inputOperand(inputs.get(2), outShape);
        int[] condStrides = condLayout.strides();
        int[] trueStrides = trueLayout.strides();
        int[] falseStrides = falseLayout.strides();
        int outBaseOffset = node.getStorageOffsetUnsafe();
        int condBaseOffset = condLayout.baseOffset();
        int trueBaseOffset = trueLayout.baseOffset();
        int falseBaseOffset = falseLayout.baseOffset();

        StridedScalarLoops.genericWhereBF16(
                cond,
                ifTrue,
                ifFalse,
                condStrides,
                trueStrides,
                falseStrides,
                condBaseOffset,
                trueBaseOffset,
                falseBaseOffset,
                out,
                outShape,
                outStrides,
                outBaseOffset,
                node.getFlatDataSize()
        );
    }
}
