package backend.cpu.kernels.elementwise.strided;

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
        int[] condStrides = inputs.get(0).getStridesUnsafe();
        int[] trueStrides = inputs.get(1).getStridesUnsafe();
        int[] falseStrides = inputs.get(2).getStridesUnsafe();
        int outBaseOffset = node.getStorageOffsetUnsafe();
        int condBaseOffset = inputs.get(0).getStorageOffsetUnsafe();
        int trueBaseOffset = inputs.get(1).getStorageOffsetUnsafe();
        int falseBaseOffset = inputs.get(2).getStorageOffsetUnsafe();

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
        int[] condStrides = inputs.get(0).getStridesUnsafe();
        int[] trueStrides = inputs.get(1).getStridesUnsafe();
        int[] falseStrides = inputs.get(2).getStridesUnsafe();
        int outBaseOffset = node.getStorageOffsetUnsafe();
        int condBaseOffset = inputs.get(0).getStorageOffsetUnsafe();
        int trueBaseOffset = inputs.get(1).getStorageOffsetUnsafe();
        int falseBaseOffset = inputs.get(2).getStorageOffsetUnsafe();

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
        int[] condStrides = inputs.get(0).getStridesUnsafe();
        int[] trueStrides = inputs.get(1).getStridesUnsafe();
        int[] falseStrides = inputs.get(2).getStridesUnsafe();
        int outBaseOffset = node.getStorageOffsetUnsafe();
        int condBaseOffset = inputs.get(0).getStorageOffsetUnsafe();
        int trueBaseOffset = inputs.get(1).getStorageOffsetUnsafe();
        int falseBaseOffset = inputs.get(2).getStorageOffsetUnsafe();

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
