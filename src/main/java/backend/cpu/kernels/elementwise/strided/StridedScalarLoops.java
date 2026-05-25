package backend.cpu.kernels.elementwise.strided;

import backend.cpu.kernels.elementwise.ElementwiseOffsetCursor;

final class StridedScalarLoops {
    private StridedScalarLoops() {
    }

    static void rank1BoolUnary(
            StridedElementWiseSemantics.BoolUnaryOp unary,
            byte[] a,
            int strideA,
            int aBaseOffset,
            byte[] out,
            int outStride,
            int outBaseOffset,
            int logicalSize
    ) {
        for (int i = 0; i < logicalSize; i++) {
            out[outBaseOffset + i * outStride] = unary.apply(a[aBaseOffset + i * strideA]);
        }
    }

    static void rank1BoolBinary(
            StridedElementWiseSemantics.BoolBinaryOp binary,
            byte[] a,
            byte[] b,
            int strideA,
            int strideB,
            int aBaseOffset,
            int bBaseOffset,
            byte[] out,
            int outStride,
            int outBaseOffset,
            int logicalSize
    ) {
        for (int i = 0; i < logicalSize; i++) {
            out[outBaseOffset + i * outStride] = binary.apply(a[aBaseOffset + i * strideA], b[bBaseOffset + i * strideB]);
        }
    }

    static void rank1CompareF64(
            StridedElementWiseSemantics.F64CompareOp compare,
            double[] a,
            double[] b,
            int strideA,
            int strideB,
            int aBaseOffset,
            int bBaseOffset,
            byte[] out,
            int outStride,
            int outBaseOffset,
            int logicalSize
    ) {
        for (int i = 0; i < logicalSize; i++) {
            out[outBaseOffset + i * outStride] = compare.apply(a[aBaseOffset + i * strideA], b[bBaseOffset + i * strideB]);
        }
    }

    static void rank1CompareF32(
            StridedElementWiseSemantics.F32CompareOp compare,
            float[] a,
            float[] b,
            int strideA,
            int strideB,
            int aBaseOffset,
            int bBaseOffset,
            byte[] out,
            int outStride,
            int outBaseOffset,
            int logicalSize
    ) {
        for (int i = 0; i < logicalSize; i++) {
            out[outBaseOffset + i * outStride] = compare.apply(a[aBaseOffset + i * strideA], b[bBaseOffset + i * strideB]);
        }
    }

    static void rank1CompareBF16(
            StridedElementWiseSemantics.BF16CompareOp compare,
            short[] a,
            short[] b,
            int strideA,
            int strideB,
            int aBaseOffset,
            int bBaseOffset,
            byte[] out,
            int outStride,
            int outBaseOffset,
            int logicalSize
    ) {
        for (int i = 0; i < logicalSize; i++) {
            out[outBaseOffset + i * outStride] = compare.apply(a[aBaseOffset + i * strideA], b[bBaseOffset + i * strideB]);
        }
    }

    static void rank1F64Binary(
            StridedElementWiseSemantics.F64BinaryOp binary,
            double[] a,
            double[] b,
            int strideA,
            int strideB,
            int aBaseOffset,
            int bBaseOffset,
            double[] out,
            int outStride,
            int outBaseOffset,
            int logicalSize
    ) {
        for (int i = 0; i < logicalSize; i++) {
            out[outBaseOffset + i * outStride] = binary.apply(a[aBaseOffset + i * strideA], b[bBaseOffset + i * strideB]);
        }
    }

    static void rank1F32Binary(
            StridedElementWiseSemantics.F32BinaryOp binary,
            float[] a,
            float[] b,
            int strideA,
            int strideB,
            int aBaseOffset,
            int bBaseOffset,
            float[] out,
            int outStride,
            int outBaseOffset,
            int logicalSize
    ) {
        for (int i = 0; i < logicalSize; i++) {
            out[outBaseOffset + i * outStride] = binary.apply(a[aBaseOffset + i * strideA], b[bBaseOffset + i * strideB]);
        }
    }

    static void rank1BF16Binary(
            StridedElementWiseSemantics.BF16BinaryOp binary,
            short[] a,
            short[] b,
            int strideA,
            int strideB,
            int aBaseOffset,
            int bBaseOffset,
            short[] out,
            int outStride,
            int outBaseOffset,
            int logicalSize
    ) {
        for (int i = 0; i < logicalSize; i++) {
            out[outBaseOffset + i * outStride] = binary.apply(a[aBaseOffset + i * strideA], b[bBaseOffset + i * strideB]);
        }
    }

    static void rank1F64Unary(
            StridedElementWiseSemantics.F64UnaryOp unary,
            double[] a,
            int strideA,
            int aBaseOffset,
            double[] out,
            int outStride,
            int outBaseOffset,
            int logicalSize
    ) {
        for (int i = 0; i < logicalSize; i++) {
            out[outBaseOffset + i * outStride] = unary.apply(a[aBaseOffset + i * strideA]);
        }
    }

    static void rank1F32Unary(
            StridedElementWiseSemantics.F32UnaryOp unary,
            float[] a,
            int strideA,
            int aBaseOffset,
            float[] out,
            int outStride,
            int outBaseOffset,
            int logicalSize
    ) {
        for (int i = 0; i < logicalSize; i++) {
            out[outBaseOffset + i * outStride] = unary.apply(a[aBaseOffset + i * strideA]);
        }
    }

    static void rank1BF16Unary(
            StridedElementWiseSemantics.BF16UnaryOp unary,
            short[] a,
            int strideA,
            int aBaseOffset,
            short[] out,
            int outStride,
            int outBaseOffset,
            int logicalSize
    ) {
        for (int i = 0; i < logicalSize; i++) {
            out[outBaseOffset + i * outStride] = unary.apply(a[aBaseOffset + i * strideA]);
        }
    }

    static void genericBoolUnary(
            StridedElementWiseSemantics.BoolUnaryOp unary,
            byte[] a,
            int[] aStrides,
            int aBaseOffset,
            byte[] out,
            int[] shape,
            int[] outStrides,
            int outBaseOffset,
            int logicalSize
    ) {
        if (logicalSize <= 0) {
            return;
        }
        ElementwiseOffsetCursor cursor = new ElementwiseOffsetCursor(
                shape,
                new int[][]{outStrides, aStrides},
                new int[]{outBaseOffset, aBaseOffset}
        );
        for (int i = 0; i < logicalSize; i++) {
            out[cursor.offset(0)] = unary.apply(a[cursor.offset(1)]);
            if (i + 1 < logicalSize) {
                cursor.step();
            }
        }
    }

    static void genericBoolBinary(
            StridedElementWiseSemantics.BoolBinaryOp binary,
            byte[] a,
            byte[] b,
            int[] aStrides,
            int[] bStrides,
            int aBaseOffset,
            int bBaseOffset,
            byte[] out,
            int[] shape,
            int[] outStrides,
            int outBaseOffset,
            int logicalSize
    ) {
        if (logicalSize <= 0) {
            return;
        }
        ElementwiseOffsetCursor cursor = new ElementwiseOffsetCursor(
                shape,
                new int[][]{outStrides, aStrides, bStrides},
                new int[]{outBaseOffset, aBaseOffset, bBaseOffset}
        );
        for (int i = 0; i < logicalSize; i++) {
            out[cursor.offset(0)] = binary.apply(a[cursor.offset(1)], b[cursor.offset(2)]);
            if (i + 1 < logicalSize) {
                cursor.step();
            }
        }
    }

    static void genericCompareF64(
            StridedElementWiseSemantics.F64CompareOp compare,
            double[] a,
            double[] b,
            int[] aStrides,
            int[] bStrides,
            int aBaseOffset,
            int bBaseOffset,
            byte[] out,
            int[] shape,
            int[] outStrides,
            int outBaseOffset,
            int logicalSize
    ) {
        if (logicalSize <= 0) {
            return;
        }
        ElementwiseOffsetCursor cursor = new ElementwiseOffsetCursor(
                shape,
                new int[][]{outStrides, aStrides, bStrides},
                new int[]{outBaseOffset, aBaseOffset, bBaseOffset}
        );
        for (int i = 0; i < logicalSize; i++) {
            out[cursor.offset(0)] = compare.apply(a[cursor.offset(1)], b[cursor.offset(2)]);
            if (i + 1 < logicalSize) {
                cursor.step();
            }
        }
    }

    static void genericCompareF32(
            StridedElementWiseSemantics.F32CompareOp compare,
            float[] a,
            float[] b,
            int[] aStrides,
            int[] bStrides,
            int aBaseOffset,
            int bBaseOffset,
            byte[] out,
            int[] shape,
            int[] outStrides,
            int outBaseOffset,
            int logicalSize
    ) {
        if (logicalSize <= 0) {
            return;
        }
        ElementwiseOffsetCursor cursor = new ElementwiseOffsetCursor(
                shape,
                new int[][]{outStrides, aStrides, bStrides},
                new int[]{outBaseOffset, aBaseOffset, bBaseOffset}
        );
        for (int i = 0; i < logicalSize; i++) {
            out[cursor.offset(0)] = compare.apply(a[cursor.offset(1)], b[cursor.offset(2)]);
            if (i + 1 < logicalSize) {
                cursor.step();
            }
        }
    }

    static void genericCompareBF16(
            StridedElementWiseSemantics.BF16CompareOp compare,
            short[] a,
            short[] b,
            int[] aStrides,
            int[] bStrides,
            int aBaseOffset,
            int bBaseOffset,
            byte[] out,
            int[] shape,
            int[] outStrides,
            int outBaseOffset,
            int logicalSize
    ) {
        if (logicalSize <= 0) {
            return;
        }
        ElementwiseOffsetCursor cursor = new ElementwiseOffsetCursor(
                shape,
                new int[][]{outStrides, aStrides, bStrides},
                new int[]{outBaseOffset, aBaseOffset, bBaseOffset}
        );
        for (int i = 0; i < logicalSize; i++) {
            out[cursor.offset(0)] = compare.apply(a[cursor.offset(1)], b[cursor.offset(2)]);
            if (i + 1 < logicalSize) {
                cursor.step();
            }
        }
    }

    static void genericF64Binary(
            StridedElementWiseSemantics.F64BinaryOp binary,
            double[] a,
            double[] b,
            int[] aStrides,
            int[] bStrides,
            int aBaseOffset,
            int bBaseOffset,
            double[] out,
            int[] shape,
            int[] outStrides,
            int outBaseOffset,
            int logicalSize
    ) {
        if (logicalSize <= 0) {
            return;
        }
        ElementwiseOffsetCursor cursor = new ElementwiseOffsetCursor(
                shape,
                new int[][]{outStrides, aStrides, bStrides},
                new int[]{outBaseOffset, aBaseOffset, bBaseOffset}
        );
        for (int i = 0; i < logicalSize; i++) {
            out[cursor.offset(0)] = binary.apply(a[cursor.offset(1)], b[cursor.offset(2)]);
            if (i + 1 < logicalSize) {
                cursor.step();
            }
        }
    }

    static void genericF32Binary(
            StridedElementWiseSemantics.F32BinaryOp binary,
            float[] a,
            float[] b,
            int[] aStrides,
            int[] bStrides,
            int aBaseOffset,
            int bBaseOffset,
            float[] out,
            int[] shape,
            int[] outStrides,
            int outBaseOffset,
            int logicalSize
    ) {
        if (logicalSize <= 0) {
            return;
        }
        ElementwiseOffsetCursor cursor = new ElementwiseOffsetCursor(
                shape,
                new int[][]{outStrides, aStrides, bStrides},
                new int[]{outBaseOffset, aBaseOffset, bBaseOffset}
        );
        for (int i = 0; i < logicalSize; i++) {
            out[cursor.offset(0)] = binary.apply(a[cursor.offset(1)], b[cursor.offset(2)]);
            if (i + 1 < logicalSize) {
                cursor.step();
            }
        }
    }

    static void genericBF16Binary(
            StridedElementWiseSemantics.BF16BinaryOp binary,
            short[] a,
            short[] b,
            int[] aStrides,
            int[] bStrides,
            int aBaseOffset,
            int bBaseOffset,
            short[] out,
            int[] shape,
            int[] outStrides,
            int outBaseOffset,
            int logicalSize
    ) {
        if (logicalSize <= 0) {
            return;
        }
        ElementwiseOffsetCursor cursor = new ElementwiseOffsetCursor(
                shape,
                new int[][]{outStrides, aStrides, bStrides},
                new int[]{outBaseOffset, aBaseOffset, bBaseOffset}
        );
        for (int i = 0; i < logicalSize; i++) {
            out[cursor.offset(0)] = binary.apply(a[cursor.offset(1)], b[cursor.offset(2)]);
            if (i + 1 < logicalSize) {
                cursor.step();
            }
        }
    }

    static void genericF64Unary(
            StridedElementWiseSemantics.F64UnaryOp unary,
            double[] a,
            int[] aStrides,
            int aBaseOffset,
            double[] out,
            int[] shape,
            int[] outStrides,
            int outBaseOffset,
            int logicalSize
    ) {
        if (logicalSize <= 0) {
            return;
        }
        ElementwiseOffsetCursor cursor = new ElementwiseOffsetCursor(
                shape,
                new int[][]{outStrides, aStrides},
                new int[]{outBaseOffset, aBaseOffset}
        );
        for (int i = 0; i < logicalSize; i++) {
            out[cursor.offset(0)] = unary.apply(a[cursor.offset(1)]);
            if (i + 1 < logicalSize) {
                cursor.step();
            }
        }
    }

    static void genericF32Unary(
            StridedElementWiseSemantics.F32UnaryOp unary,
            float[] a,
            int[] aStrides,
            int aBaseOffset,
            float[] out,
            int[] shape,
            int[] outStrides,
            int outBaseOffset,
            int logicalSize
    ) {
        if (logicalSize <= 0) {
            return;
        }
        ElementwiseOffsetCursor cursor = new ElementwiseOffsetCursor(
                shape,
                new int[][]{outStrides, aStrides},
                new int[]{outBaseOffset, aBaseOffset}
        );
        for (int i = 0; i < logicalSize; i++) {
            out[cursor.offset(0)] = unary.apply(a[cursor.offset(1)]);
            if (i + 1 < logicalSize) {
                cursor.step();
            }
        }
    }

    static void genericBF16Unary(
            StridedElementWiseSemantics.BF16UnaryOp unary,
            short[] a,
            int[] aStrides,
            int aBaseOffset,
            short[] out,
            int[] shape,
            int[] outStrides,
            int outBaseOffset,
            int logicalSize
    ) {
        if (logicalSize <= 0) {
            return;
        }
        ElementwiseOffsetCursor cursor = new ElementwiseOffsetCursor(
                shape,
                new int[][]{outStrides, aStrides},
                new int[]{outBaseOffset, aBaseOffset}
        );
        for (int i = 0; i < logicalSize; i++) {
            out[cursor.offset(0)] = unary.apply(a[cursor.offset(1)]);
            if (i + 1 < logicalSize) {
                cursor.step();
            }
        }
    }

    static void genericWhereF64(
            byte[] cond,
            double[] ifTrue,
            double[] ifFalse,
            int[] condStrides,
            int[] trueStrides,
            int[] falseStrides,
            int condBaseOffset,
            int trueBaseOffset,
            int falseBaseOffset,
            double[] out,
            int[] shape,
            int[] outStrides,
            int outBaseOffset,
            int logicalSize
    ) {
        if (logicalSize <= 0) {
            return;
        }
        ElementwiseOffsetCursor cursor = new ElementwiseOffsetCursor(
                shape,
                new int[][]{outStrides, condStrides, trueStrides, falseStrides},
                new int[]{outBaseOffset, condBaseOffset, trueBaseOffset, falseBaseOffset}
        );
        for (int i = 0; i < logicalSize; i++) {
            out[cursor.offset(0)] = cond[cursor.offset(1)] != 0 ? ifTrue[cursor.offset(2)] : ifFalse[cursor.offset(3)];
            if (i + 1 < logicalSize) {
                cursor.step();
            }
        }
    }

    static void genericWhereF32(
            byte[] cond,
            float[] ifTrue,
            float[] ifFalse,
            int[] condStrides,
            int[] trueStrides,
            int[] falseStrides,
            int condBaseOffset,
            int trueBaseOffset,
            int falseBaseOffset,
            float[] out,
            int[] shape,
            int[] outStrides,
            int outBaseOffset,
            int logicalSize
    ) {
        if (logicalSize <= 0) {
            return;
        }
        ElementwiseOffsetCursor cursor = new ElementwiseOffsetCursor(
                shape,
                new int[][]{outStrides, condStrides, trueStrides, falseStrides},
                new int[]{outBaseOffset, condBaseOffset, trueBaseOffset, falseBaseOffset}
        );
        for (int i = 0; i < logicalSize; i++) {
            out[cursor.offset(0)] = cond[cursor.offset(1)] != 0 ? ifTrue[cursor.offset(2)] : ifFalse[cursor.offset(3)];
            if (i + 1 < logicalSize) {
                cursor.step();
            }
        }
    }

    static void genericWhereBF16(
            byte[] cond,
            short[] ifTrue,
            short[] ifFalse,
            int[] condStrides,
            int[] trueStrides,
            int[] falseStrides,
            int condBaseOffset,
            int trueBaseOffset,
            int falseBaseOffset,
            short[] out,
            int[] shape,
            int[] outStrides,
            int outBaseOffset,
            int logicalSize
    ) {
        if (logicalSize <= 0) {
            return;
        }
        ElementwiseOffsetCursor cursor = new ElementwiseOffsetCursor(
                shape,
                new int[][]{outStrides, condStrides, trueStrides, falseStrides},
                new int[]{outBaseOffset, condBaseOffset, trueBaseOffset, falseBaseOffset}
        );
        for (int i = 0; i < logicalSize; i++) {
            out[cursor.offset(0)] = cond[cursor.offset(1)] != 0 ? ifTrue[cursor.offset(2)] : ifFalse[cursor.offset(3)];
            if (i + 1 < logicalSize) {
                cursor.step();
            }
        }
    }
}
