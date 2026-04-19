package backend.kernels.cpu.elementwise.strided;

import tensor.Tensor;

import java.util.List;

final class StridedWhereLoops {
    private StridedWhereLoops() {
    }

    static void forwardF64(List<Tensor> inputs, Tensor node) {
        double[] out = node.getFloat64Data();
        if (out == null) {
            return;
        }
        byte[] cond = inputs.get(0).getBoolData();
        double[] ifTrue = inputs.get(1).getFloat64Data();
        double[] ifFalse = inputs.get(2).getFloat64Data();
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
        byte[] cond = inputs.get(0).getBoolData();
        float[] ifTrue = inputs.get(1).getFloat32Data();
        float[] ifFalse = inputs.get(2).getFloat32Data();
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
        byte[] cond = inputs.get(0).getBoolData();
        short[] ifTrue = inputs.get(1).getBFloat16Data();
        short[] ifFalse = inputs.get(2).getBFloat16Data();
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
