package backend.kernels.cpu.elementwise.strided;

import operations.Operation;
import tensor.Tensor;

import java.util.List;

final class StridedBooleanLoops {
    private StridedBooleanLoops() {
    }

    static void forward(Operation op, List<Tensor> inputs, Tensor node) {
        byte[] out = node.getBoolData();
        if (out == null) {
            return;
        }

        int[] outShape = node.getShapeUnsafe();
        int[] outStrides = node.getStridesUnsafe();
        int outBaseOffset = node.getStorageOffsetUnsafe();
        int rank = outShape.length;
        int logicalSize = node.getFlatDataSize();

        if (op.opType() == Operation.OpType.LOGICAL_NOT) {
            Tensor ta = inputs.getFirst();
            byte[] a = ta.getBoolData();
            int[] aStrides = ta.getStridesUnsafe();
            int aBaseOffset = ta.getStorageOffsetUnsafe();
            if (rank == 1) {
                rank1BoolUnary(a, aStrides[0], aBaseOffset, out, outStrides[0], outBaseOffset, logicalSize);
                return;
            }
            genericBoolUnary(op, a, aStrides, aBaseOffset, out, outShape, outStrides, outBaseOffset, logicalSize);
            return;
        }

        Tensor ta = inputs.get(0);
        Tensor tb = inputs.get(1);
        if (ta.getDataType() == tensor.DataType.BOOL) {
            byte[] a = ta.getBoolData();
            byte[] b = tb.getBoolData();
            int[] aStrides = ta.getStridesUnsafe();
            int[] bStrides = tb.getStridesUnsafe();
            int aBaseOffset = ta.getStorageOffsetUnsafe();
            int bBaseOffset = tb.getStorageOffsetUnsafe();
            if (rank == 1) {
                rank1BoolBinary(op, a, b, aStrides[0], bStrides[0], aBaseOffset, bBaseOffset, out, outStrides[0], outBaseOffset, logicalSize);
                return;
            }
            genericBoolBinary(op, a, b, aStrides, bStrides, aBaseOffset, bBaseOffset, out, outShape, outStrides, outBaseOffset, logicalSize);
            return;
        }

        switch (ta.getDataType()) {
            case FLOAT64 -> {
                double[] a = ta.getFloat64Data();
                double[] b = tb.getFloat64Data();
                int[] aStrides = ta.getStridesUnsafe();
                int[] bStrides = tb.getStridesUnsafe();
                int aBaseOffset = ta.getStorageOffsetUnsafe();
                int bBaseOffset = tb.getStorageOffsetUnsafe();
                if (rank == 1) {
                    rank1CompareF64(op, a, b, aStrides[0], bStrides[0], aBaseOffset, bBaseOffset, out, outStrides[0], outBaseOffset, logicalSize);
                    return;
                }
                genericCompareF64(op, a, b, aStrides, bStrides, aBaseOffset, bBaseOffset, out, outShape, outStrides, outBaseOffset, logicalSize);
            }
            case FLOAT32 -> {
                float[] a = ta.getFloat32Data();
                float[] b = tb.getFloat32Data();
                int[] aStrides = ta.getStridesUnsafe();
                int[] bStrides = tb.getStridesUnsafe();
                int aBaseOffset = ta.getStorageOffsetUnsafe();
                int bBaseOffset = tb.getStorageOffsetUnsafe();
                if (rank == 1) {
                    rank1CompareF32(op, a, b, aStrides[0], bStrides[0], aBaseOffset, bBaseOffset, out, outStrides[0], outBaseOffset, logicalSize);
                    return;
                }
                genericCompareF32(op, a, b, aStrides, bStrides, aBaseOffset, bBaseOffset, out, outShape, outStrides, outBaseOffset, logicalSize);
            }
            case BFLOAT16 -> {
                short[] a = ta.getBFloat16Data();
                short[] b = tb.getBFloat16Data();
                int[] aStrides = ta.getStridesUnsafe();
                int[] bStrides = tb.getStridesUnsafe();
                int aBaseOffset = ta.getStorageOffsetUnsafe();
                int bBaseOffset = tb.getStorageOffsetUnsafe();
                if (rank == 1) {
                    rank1CompareBF16(op, a, b, aStrides[0], bStrides[0], aBaseOffset, bBaseOffset, out, outStrides[0], outBaseOffset, logicalSize);
                    return;
                }
                genericCompareBF16(op, a, b, aStrides, bStrides, aBaseOffset, bBaseOffset, out, outShape, outStrides, outBaseOffset, logicalSize);
            }
            case INT32, BOOL -> throw new UnsupportedOperationException("Unsupported BOOL strided input contract for opType=" + op.opType());
        }
    }

    private static void rank1BoolUnary(
            byte[] a,
            int strideA,
            int aBaseOffset,
            byte[] out,
            int outStride,
            int outBaseOffset,
            int logicalSize
    ) {
        StridedScalarLoops.rank1BoolUnary(
                StridedElementWiseSemantics.boolNot(),
                a,
                strideA,
                aBaseOffset,
                out,
                outStride,
                outBaseOffset,
                logicalSize
        );
    }

    private static void rank1BoolBinary(
            Operation op,
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
        StridedElementWiseSemantics.BoolBinaryOp binary = StridedElementWiseSemantics.resolveBoolBinary(op);
        if (binary == null) {
            throw new UnsupportedOperationException("Unsupported bool strided opType=" + op.opType());
        }
        StridedScalarLoops.rank1BoolBinary(binary, a, b, strideA, strideB, aBaseOffset, bBaseOffset, out, outStride, outBaseOffset, logicalSize);
    }

    private static void rank1CompareF64(
            Operation op,
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
        StridedElementWiseSemantics.F64CompareOp compare = StridedElementWiseSemantics.resolveF64Compare(op);
        if (compare == null) {
            throw new UnsupportedOperationException("Unsupported compare strided opType=" + op.opType());
        }
        StridedScalarLoops.rank1CompareF64(compare, a, b, strideA, strideB, aBaseOffset, bBaseOffset, out, outStride, outBaseOffset, logicalSize);
    }

    private static void rank1CompareF32(
            Operation op,
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
        StridedElementWiseSemantics.F32CompareOp compare = StridedElementWiseSemantics.resolveF32Compare(op);
        if (compare == null) {
            throw new UnsupportedOperationException("Unsupported compare strided opType=" + op.opType());
        }
        StridedScalarLoops.rank1CompareF32(compare, a, b, strideA, strideB, aBaseOffset, bBaseOffset, out, outStride, outBaseOffset, logicalSize);
    }

    private static void rank1CompareBF16(
            Operation op,
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
        StridedElementWiseSemantics.BF16CompareOp compare = StridedElementWiseSemantics.resolveBF16Compare(op);
        if (compare == null) {
            throw new UnsupportedOperationException("Unsupported compare strided opType=" + op.opType());
        }
        StridedScalarLoops.rank1CompareBF16(compare, a, b, strideA, strideB, aBaseOffset, bBaseOffset, out, outStride, outBaseOffset, logicalSize);
    }

    private static void genericCompareF64(
            Operation op,
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
        StridedElementWiseSemantics.F64CompareOp compare = StridedElementWiseSemantics.resolveF64Compare(op);
        if (compare == null) {
            throw new UnsupportedOperationException("Unsupported compare strided opType=" + op.opType());
        }
        StridedScalarLoops.genericCompareF64(compare, a, b, aStrides, bStrides, aBaseOffset, bBaseOffset, out, shape, outStrides, outBaseOffset, logicalSize);
    }

    private static void genericCompareF32(
            Operation op,
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
        StridedElementWiseSemantics.F32CompareOp compare = StridedElementWiseSemantics.resolveF32Compare(op);
        if (compare == null) {
            throw new UnsupportedOperationException("Unsupported compare strided opType=" + op.opType());
        }
        StridedScalarLoops.genericCompareF32(compare, a, b, aStrides, bStrides, aBaseOffset, bBaseOffset, out, shape, outStrides, outBaseOffset, logicalSize);
    }

    private static void genericCompareBF16(
            Operation op,
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
        StridedElementWiseSemantics.BF16CompareOp compare = StridedElementWiseSemantics.resolveBF16Compare(op);
        if (compare == null) {
            throw new UnsupportedOperationException("Unsupported compare strided opType=" + op.opType());
        }
        StridedScalarLoops.genericCompareBF16(compare, a, b, aStrides, bStrides, aBaseOffset, bBaseOffset, out, shape, outStrides, outBaseOffset, logicalSize);
    }

    private static void genericBoolUnary(
            Operation op,
            byte[] a,
            int[] aStrides,
            int aBaseOffset,
            byte[] out,
            int[] shape,
            int[] outStrides,
            int outBaseOffset,
            int logicalSize
    ) {
        StridedElementWiseSemantics.BoolUnaryOp unary = StridedElementWiseSemantics.resolveBoolUnary(op);
        if (unary == null) {
            throw new UnsupportedOperationException("Unsupported bool strided opType=" + op.opType());
        }
        StridedScalarLoops.genericBoolUnary(unary, a, aStrides, aBaseOffset, out, shape, outStrides, outBaseOffset, logicalSize);
    }

    private static void genericBoolBinary(
            Operation op,
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
        StridedElementWiseSemantics.BoolBinaryOp binary = StridedElementWiseSemantics.resolveBoolBinary(op);
        if (binary == null) {
            throw new UnsupportedOperationException("Unsupported bool strided opType=" + op.opType());
        }
        StridedScalarLoops.genericBoolBinary(binary, a, b, aStrides, bStrides, aBaseOffset, bBaseOffset, out, shape, outStrides, outBaseOffset, logicalSize);
    }
}
