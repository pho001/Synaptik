package backend.kernels.cpu;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import operations.Operation;
import operations.clampMax;
import operations.clampMin;
import operations.mulScalar;
import operations.pow;
import tensor.Tensor;
import utils.FastExp;

import java.util.List;

public final class CpuStridedElementWise {
    private static final VectorSpecies<Double> F64 = DoubleVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Float> F32 = FloatVector.SPECIES_PREFERRED;

    private CpuStridedElementWise() {}

    public static boolean supports(Operation op) {
        if (op == null) return false;
        return switch (op.opType()) {
            case ADD, SUB, MUL, DIV, MIN, MAX, GT, GE, LT, LE, EQ, NE, WHERE,
                    LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT,
                    NEG, INV, LOG, EXP, FAST_EXP, TANH, FAST_TANH, POW, SQRT, ABS, MUL_SCALAR, RELU, CLAMP_MIN, CLAMP_MAX, SIGMOID -> true;
            default -> false;
        };
    }

    public static void forward(Operation op, List<Tensor> inputs, Tensor node) {
        forward(op, inputs, node, null);
    }

    public static void forward(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (op == null) {
            return;
        }
        boolean useFastExpApprox = context != null && context.useFastExpApprox();
        boolean useFastTanhApprox = context != null && context.useFastTanhApprox();
        switch (node.getDataType()) {
            case FLOAT32 -> {
                forwardF32(op, inputs, node, useFastExpApprox, useFastTanhApprox);
                return;
            }
            case BFLOAT16 -> {
                forwardBF16(op, inputs, node, useFastExpApprox, useFastTanhApprox);
                return;
            }
            case FLOAT64 -> {
                if (op.opType() == Operation.OpType.WHERE) {
                    forwardWhereF64(inputs, node);
                    return;
                }
                // continue with existing F64 path below
            }
            case BOOL -> {
                forwardBOOL(op, inputs, node);
                return;
            }
            case INT32 -> throw new UnsupportedOperationException("INT32 is not supported by CpuStridedElementWise.");
        }

        double[] out = node.getFloat64Data();
        if (out == null) {
            return;
        }
        if (op.opType() == Operation.OpType.WHERE) {
            forwardWhereF64(inputs, node);
            return;
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

        if (rank == 1) {
            forwardRank1F64(
                    op,
                    a,
                    b,
                    aStrides,
                    bStrides,
                    aBaseOffset,
                    bBaseOffset,
                    out,
                    outStrides[0],
                    outBaseOffset,
                    node.getFlatDataSize(),
                    useFastExpApprox,
                    useFastTanhApprox
            );
            return;
        }

        if (rank == 2 && tryForwardRank2F64(
                op,
                a,
                b,
                aStrides,
                bStrides,
                aBaseOffset,
                bBaseOffset,
                out,
                outShape[0],
                outShape[1],
                outStrides,
                outBaseOffset,
                useFastExpApprox,
                useFastTanhApprox
        )) {
            return;
        }

        forwardGenericF64(
                op,
                a,
                b,
                aStrides,
                bStrides,
                aBaseOffset,
                bBaseOffset,
                out,
                outShape,
                outStrides,
                outBaseOffset,
                node.getFlatDataSize(),
                useFastExpApprox,
                useFastTanhApprox
        );
    }

    private static void forwardF32(
            Operation op,
            List<Tensor> inputs,
            Tensor node,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        float[] out = node.getFloat32Data();
        if (out == null) {
            return;
        }
        if (op.opType() == Operation.OpType.WHERE) {
            forwardWhereF32(inputs, node, out);
            return;
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

        if (rank == 1) {
            forwardRank1F32(
                    op,
                    a,
                    b,
                    aStrides,
                    bStrides,
                    aBaseOffset,
                    bBaseOffset,
                    out,
                    outStrides[0],
                    outBaseOffset,
                    node.getFlatDataSize(),
                    useFastExpApprox,
                    useFastTanhApprox
            );
            return;
        }

        if (rank == 2 && tryForwardRank2F32(
                op,
                a,
                b,
                aStrides,
                bStrides,
                aBaseOffset,
                bBaseOffset,
                out,
                outShape[0],
                outShape[1],
                outStrides,
                outBaseOffset,
                useFastExpApprox,
                useFastTanhApprox
        )) {
            return;
        }

        forwardGenericF32(
                op,
                a,
                b,
                aStrides,
                bStrides,
                aBaseOffset,
                bBaseOffset,
                out,
                outShape,
                outStrides,
                outBaseOffset,
                node.getFlatDataSize(),
                useFastExpApprox,
                useFastTanhApprox
        );
    }

    private static void forwardBF16(
            Operation op,
            List<Tensor> inputs,
            Tensor node,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        short[] out = node.getBFloat16Data();
        if (out == null) {
            return;
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

        if (rank == 1) {
            forwardRank1BF16(
                    op,
                    a,
                    b,
                    aStrides,
                    bStrides,
                    aBaseOffset,
                    bBaseOffset,
                    out,
                    outStrides[0],
                    outBaseOffset,
                    node.getFlatDataSize(),
                    useFastExpApprox,
                    useFastTanhApprox
            );
            return;
        }

        forwardGenericBF16(
                op,
                a,
                b,
                aStrides,
                bStrides,
                aBaseOffset,
                bBaseOffset,
                out,
                outShape,
                outStrides,
                outBaseOffset,
                node.getFlatDataSize(),
                useFastExpApprox,
                useFastTanhApprox
        );
    }

    private static void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node) {
        byte[] out = node.getBoolData();
        if (out == null) {
            return;
        }

        int[] outShape = node.getShapeUnsafe();
        int[] outStrides = node.getStridesUnsafe();
        int outBaseOffset = node.getStorageOffsetUnsafe();
        int rank = outShape.length;

        if (op.opType() == Operation.OpType.LOGICAL_NOT) {
            Tensor ta = inputs.getFirst();
            byte[] a = ta.getBoolData();
            int[] aStrides = ta.getStridesUnsafe();
            int aBaseOffset = ta.getStorageOffsetUnsafe();
            if (rank == 1) {
                forwardRank1BoolUnary(
                        a,
                        aStrides[0],
                        aBaseOffset,
                        out,
                        outStrides[0],
                        outBaseOffset,
                        node.getFlatDataSize()
                );
                return;
            }
            forwardGenericBoolUnary(op, a, aStrides, aBaseOffset, out, outShape, outStrides, outBaseOffset, node.getFlatDataSize());
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
                forwardRank1BoolBinary(
                        op,
                        a,
                        b,
                        aStrides[0],
                        bStrides[0],
                        aBaseOffset,
                        bBaseOffset,
                        out,
                        outStrides[0],
                        outBaseOffset,
                        node.getFlatDataSize()
                );
                return;
            }
            forwardGenericBoolBinary(op, a, b, aStrides, bStrides, aBaseOffset, bBaseOffset, out, outShape, outStrides, outBaseOffset, node.getFlatDataSize());
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
                    forwardRank1CompareF64(
                            op,
                            a,
                            b,
                            aStrides[0],
                            bStrides[0],
                            aBaseOffset,
                            bBaseOffset,
                            out,
                            outStrides[0],
                            outBaseOffset,
                            node.getFlatDataSize()
                    );
                    return;
                }
                forwardGenericCompareF64(op, a, b, aStrides, bStrides, aBaseOffset, bBaseOffset, out, outShape, outStrides, outBaseOffset, node.getFlatDataSize());
            }
            case FLOAT32 -> {
                float[] a = ta.getFloat32Data();
                float[] b = tb.getFloat32Data();
                int[] aStrides = ta.getStridesUnsafe();
                int[] bStrides = tb.getStridesUnsafe();
                int aBaseOffset = ta.getStorageOffsetUnsafe();
                int bBaseOffset = tb.getStorageOffsetUnsafe();
                if (rank == 1) {
                    forwardRank1CompareF32(
                            op,
                            a,
                            b,
                            aStrides[0],
                            bStrides[0],
                            aBaseOffset,
                            bBaseOffset,
                            out,
                            outStrides[0],
                            outBaseOffset,
                            node.getFlatDataSize()
                    );
                    return;
                }
                forwardGenericCompareF32(op, a, b, aStrides, bStrides, aBaseOffset, bBaseOffset, out, outShape, outStrides, outBaseOffset, node.getFlatDataSize());
            }
            case BFLOAT16 -> {
                short[] a = ta.getBFloat16Data();
                short[] b = tb.getBFloat16Data();
                int[] aStrides = ta.getStridesUnsafe();
                int[] bStrides = tb.getStridesUnsafe();
                int aBaseOffset = ta.getStorageOffsetUnsafe();
                int bBaseOffset = tb.getStorageOffsetUnsafe();
                if (rank == 1) {
                    forwardRank1CompareBF16(
                            op,
                            a,
                            b,
                            aStrides[0],
                            bStrides[0],
                            aBaseOffset,
                            bBaseOffset,
                            out,
                            outStrides[0],
                            outBaseOffset,
                            node.getFlatDataSize()
                    );
                    return;
                }
                forwardGenericCompareBF16(op, a, b, aStrides, bStrides, aBaseOffset, bBaseOffset, out, outShape, outStrides, outBaseOffset, node.getFlatDataSize());
            }
            case INT32, BOOL -> throw new UnsupportedOperationException("Unsupported BOOL strided input contract for opType=" + op.opType());
        }
    }

    private static void forwardWhereF64(List<Tensor> inputs, Tensor node) {
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
        int rank = outShape.length;

        forwardGenericWhereF64(
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

    private static void forwardWhereF32(List<Tensor> inputs, Tensor node, float[] out) {
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
        int rank = outShape.length;

        forwardGenericWhereF32(
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

    private static void forwardWhereF16(List<Tensor> inputs, Tensor node, short[] out) {
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
        int rank = outShape.length;

        forwardGenericWhereF16(
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

    private static void forwardRank1BoolUnary(
            byte[] a,
            int strideA,
            int aBaseOffset,
            byte[] out,
            int outStride,
            int outBaseOffset,
            int logicalSize
    ) {
        for (int i = 0; i < logicalSize; i++) {
            out[outBaseOffset + i * outStride] = a[aBaseOffset + i * strideA] == 0 ? (byte) 1 : (byte) 0;
        }
    }

    private static void forwardRank1BoolBinary(
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
        switch (op.opType()) {
            case LOGICAL_AND -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] =
                            (a[aBaseOffset + i * strideA] != 0 && b[bBaseOffset + i * strideB] != 0) ? (byte) 1 : (byte) 0;
                }
            }
            case LOGICAL_OR -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] =
                            (a[aBaseOffset + i * strideA] != 0 || b[bBaseOffset + i * strideB] != 0) ? (byte) 1 : (byte) 0;
                }
            }
            default -> throw new UnsupportedOperationException("Unsupported bool strided opType=" + op.opType());
        }
    }

    private static void forwardRank1CompareF64(
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
        switch (op.opType()) {
            case GT -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = a[aBaseOffset + i * strideA] > b[bBaseOffset + i * strideB] ? (byte) 1 : (byte) 0;
                }
            }
            case GE -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = a[aBaseOffset + i * strideA] >= b[bBaseOffset + i * strideB] ? (byte) 1 : (byte) 0;
                }
            }
            case LT -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = a[aBaseOffset + i * strideA] < b[bBaseOffset + i * strideB] ? (byte) 1 : (byte) 0;
                }
            }
            case LE -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = a[aBaseOffset + i * strideA] <= b[bBaseOffset + i * strideB] ? (byte) 1 : (byte) 0;
                }
            }
            case EQ -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = a[aBaseOffset + i * strideA] == b[bBaseOffset + i * strideB] ? (byte) 1 : (byte) 0;
                }
            }
            case NE -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = a[aBaseOffset + i * strideA] != b[bBaseOffset + i * strideB] ? (byte) 1 : (byte) 0;
                }
            }
            default -> throw new UnsupportedOperationException("Unsupported compare strided opType=" + op.opType());
        }
    }

    private static void forwardRank1CompareF32(
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
        switch (op.opType()) {
            case GT -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = a[aBaseOffset + i * strideA] > b[bBaseOffset + i * strideB] ? (byte) 1 : (byte) 0;
                }
            }
            case GE -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = a[aBaseOffset + i * strideA] >= b[bBaseOffset + i * strideB] ? (byte) 1 : (byte) 0;
                }
            }
            case LT -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = a[aBaseOffset + i * strideA] < b[bBaseOffset + i * strideB] ? (byte) 1 : (byte) 0;
                }
            }
            case LE -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = a[aBaseOffset + i * strideA] <= b[bBaseOffset + i * strideB] ? (byte) 1 : (byte) 0;
                }
            }
            case EQ -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = a[aBaseOffset + i * strideA] == b[bBaseOffset + i * strideB] ? (byte) 1 : (byte) 0;
                }
            }
            case NE -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = a[aBaseOffset + i * strideA] != b[bBaseOffset + i * strideB] ? (byte) 1 : (byte) 0;
                }
            }
            default -> throw new UnsupportedOperationException("Unsupported compare strided opType=" + op.opType());
        }
    }

    private static void forwardRank1CompareBF16(
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
        switch (op.opType()) {
            case GT -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = CpuDTypeOps.fromBFloat16Bits(a[aBaseOffset + i * strideA])
                            > CpuDTypeOps.fromBFloat16Bits(b[bBaseOffset + i * strideB]) ? (byte) 1 : (byte) 0;
                }
            }
            case GE -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = CpuDTypeOps.fromBFloat16Bits(a[aBaseOffset + i * strideA])
                            >= CpuDTypeOps.fromBFloat16Bits(b[bBaseOffset + i * strideB]) ? (byte) 1 : (byte) 0;
                }
            }
            case LT -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = CpuDTypeOps.fromBFloat16Bits(a[aBaseOffset + i * strideA])
                            < CpuDTypeOps.fromBFloat16Bits(b[bBaseOffset + i * strideB]) ? (byte) 1 : (byte) 0;
                }
            }
            case LE -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = CpuDTypeOps.fromBFloat16Bits(a[aBaseOffset + i * strideA])
                            <= CpuDTypeOps.fromBFloat16Bits(b[bBaseOffset + i * strideB]) ? (byte) 1 : (byte) 0;
                }
            }
            case EQ -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = CpuDTypeOps.fromBFloat16Bits(a[aBaseOffset + i * strideA])
                            == CpuDTypeOps.fromBFloat16Bits(b[bBaseOffset + i * strideB]) ? (byte) 1 : (byte) 0;
                }
            }
            case NE -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = CpuDTypeOps.fromBFloat16Bits(a[aBaseOffset + i * strideA])
                            != CpuDTypeOps.fromBFloat16Bits(b[bBaseOffset + i * strideB]) ? (byte) 1 : (byte) 0;
                }
            }
            default -> throw new UnsupportedOperationException("Unsupported compare strided opType=" + op.opType());
        }
    }

    private static void forwardRank1F64(
            Operation op,
            double[] a,
            double[] b,
            int[] aStrides,
            int[] bStrides,
            int aBaseOffset,
            int bBaseOffset,
            double[] out,
            int outStride,
            int outBaseOffset,
            int logicalSize,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        int strideA = a != null ? aStrides[0] : 0;
        int strideB = b != null ? bStrides[0] : 0;
        switch (op.opType()) {
            case ADD -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = a[aBaseOffset + i * strideA] + b[bBaseOffset + i * strideB];
                }
            }
            case SUB -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = a[aBaseOffset + i * strideA] - b[bBaseOffset + i * strideB];
                }
            }
            case MUL -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = a[aBaseOffset + i * strideA] * b[bBaseOffset + i * strideB];
                }
            }
            case DIV -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = a[aBaseOffset + i * strideA] / b[bBaseOffset + i * strideB];
                }
            }
            case MIN -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = Math.min(a[aBaseOffset + i * strideA], b[bBaseOffset + i * strideB]);
                }
            }
            case MAX -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = Math.max(a[aBaseOffset + i * strideA], b[bBaseOffset + i * strideB]);
                }
            }
            case NEG -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = -a[aBaseOffset + i * strideA];
                }
            }
            case INV -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = 1.0 / a[aBaseOffset + i * strideA];
                }
            }
            case LOG -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = Math.log(a[aBaseOffset + i * strideA]);
                }
            }
            case EXP -> {
                for (int i = 0; i < logicalSize; i++) {
                    double value = a[aBaseOffset + i * strideA];
                    out[outBaseOffset + i * outStride] = useFastExpApprox ? FastExp.fastExpF64(value) : Math.exp(value);
                }
            }
            case FAST_EXP -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = FastExp.fastExpF64(a[aBaseOffset + i * strideA]);
                }
            }
            case TANH -> {
                for (int i = 0; i < logicalSize; i++) {
                    double value = a[aBaseOffset + i * strideA];
                    out[outBaseOffset + i * outStride] = useFastTanhApprox ? FastExp.fastTanhF64(value) : Math.tanh(value);
                }
            }
            case FAST_TANH -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = FastExp.fastTanhF64(a[aBaseOffset + i * strideA]);
                }
            }
            case SQRT -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = Math.sqrt(a[aBaseOffset + i * strideA]);
                }
            }
            case ABS -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = Math.abs(a[aBaseOffset + i * strideA]);
                }
            }
            case RELU -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = Math.max(0.0, a[aBaseOffset + i * strideA]);
                }
            }
            case CLAMP_MIN -> {
                double minValue = ((clampMin) op).getMinValue();
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = Math.max(minValue, a[aBaseOffset + i * strideA]);
                }
            }
            case CLAMP_MAX -> {
                double maxValue = ((clampMax) op).getMaxValue();
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = Math.min(maxValue, a[aBaseOffset + i * strideA]);
                }
            }
            case SIGMOID -> {
                for (int i = 0; i < logicalSize; i++) {
                    double value = a[aBaseOffset + i * strideA];
                    out[outBaseOffset + i * outStride] = 1.0 / (1.0 + Math.exp(-value));
                }
            }
            case MUL_SCALAR -> {
                double scalar = ((mulScalar) op).getScalar();
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = a[aBaseOffset + i * strideA] * scalar;
                }
            }
            case POW -> {
                double exponent = ((pow) op).getExponent();
                for (int i = 0; i < logicalSize; i++) {
                    double value = a[aBaseOffset + i * strideA];
                    out[outBaseOffset + i * outStride] = CpuPowSupport.applyF64(value, exponent);
                }
            }
            default -> throw new UnsupportedOperationException("Unsupported strided opType=" + op.opType());
        }
    }

    private static void forwardRank1F32(
            Operation op,
            float[] a,
            float[] b,
            int[] aStrides,
            int[] bStrides,
            int aBaseOffset,
            int bBaseOffset,
            float[] out,
            int outStride,
            int outBaseOffset,
            int logicalSize,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        int strideA = a != null ? aStrides[0] : 0;
        int strideB = b != null ? bStrides[0] : 0;
        switch (op.opType()) {
            case ADD -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = a[aBaseOffset + i * strideA] + b[bBaseOffset + i * strideB];
                }
            }
            case SUB -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = a[aBaseOffset + i * strideA] - b[bBaseOffset + i * strideB];
                }
            }
            case MUL -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = a[aBaseOffset + i * strideA] * b[bBaseOffset + i * strideB];
                }
            }
            case DIV -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = a[aBaseOffset + i * strideA] / b[bBaseOffset + i * strideB];
                }
            }
            case MIN -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = Math.min(a[aBaseOffset + i * strideA], b[bBaseOffset + i * strideB]);
                }
            }
            case MAX -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = Math.max(a[aBaseOffset + i * strideA], b[bBaseOffset + i * strideB]);
                }
            }
            case NEG -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = -a[aBaseOffset + i * strideA];
                }
            }
            case INV -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = 1.0f / a[aBaseOffset + i * strideA];
                }
            }
            case LOG -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = (float) Math.log(a[aBaseOffset + i * strideA]);
                }
            }
            case EXP -> {
                for (int i = 0; i < logicalSize; i++) {
                    float value = a[aBaseOffset + i * strideA];
                    out[outBaseOffset + i * outStride] = useFastExpApprox ? FastExp.fastExpF32(value) : (float) Math.exp(value);
                }
            }
            case FAST_EXP -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = FastExp.fastExpF32(a[aBaseOffset + i * strideA]);
                }
            }
            case TANH -> {
                for (int i = 0; i < logicalSize; i++) {
                    float value = a[aBaseOffset + i * strideA];
                    out[outBaseOffset + i * outStride] = useFastTanhApprox ? FastExp.fastTanhF32(value) : (float) Math.tanh(value);
                }
            }
            case FAST_TANH -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = FastExp.fastTanhF32(a[aBaseOffset + i * strideA]);
                }
            }
            case SQRT -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = (float) Math.sqrt(a[aBaseOffset + i * strideA]);
                }
            }
            case ABS -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = Math.abs(a[aBaseOffset + i * strideA]);
                }
            }
            case RELU -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = Math.max(0.0f, a[aBaseOffset + i * strideA]);
                }
            }
            case CLAMP_MIN -> {
                float minValue = ((clampMin) op).getMinValueF32();
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = Math.max(minValue, a[aBaseOffset + i * strideA]);
                }
            }
            case CLAMP_MAX -> {
                float maxValue = ((clampMax) op).getMaxValueF32();
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = Math.min(maxValue, a[aBaseOffset + i * strideA]);
                }
            }
            case SIGMOID -> {
                for (int i = 0; i < logicalSize; i++) {
                    float value = a[aBaseOffset + i * strideA];
                    out[outBaseOffset + i * outStride] = (float) (1.0 / (1.0 + Math.exp(-value)));
                }
            }
            case MUL_SCALAR -> {
                float scalar = ((mulScalar) op).getScalarF32();
                for (int i = 0; i < logicalSize; i++) {
                    out[outBaseOffset + i * outStride] = a[aBaseOffset + i * strideA] * scalar;
                }
            }
            case POW -> {
                float exponent = ((pow) op).getExponentF32();
                for (int i = 0; i < logicalSize; i++) {
                    float value = a[aBaseOffset + i * strideA];
                    out[outBaseOffset + i * outStride] = CpuPowSupport.applyF32(value, exponent);
                }
            }
            default -> throw new UnsupportedOperationException("Unsupported strided opType=" + op.opType());
        }
    }

    private static void forwardRank1BF16(
            Operation op,
            short[] a,
            short[] b,
            int[] aStrides,
            int[] bStrides,
            int aBaseOffset,
            int bBaseOffset,
            short[] out,
            int outStride,
            int outBaseOffset,
            int logicalSize,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        int strideA = a != null ? aStrides[0] : 0;
        int strideB = b != null ? bStrides[0] : 0;
        switch (op.opType()) {
            case ADD -> {
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[aBaseOffset + i * strideA]);
                    float bv = CpuDTypeOps.fromBFloat16Bits(b[bBaseOffset + i * strideB]);
                    out[outBaseOffset + i * outStride] = CpuDTypeOps.toBFloat16Bits(av + bv);
                }
            }
            case SUB -> {
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[aBaseOffset + i * strideA]);
                    float bv = CpuDTypeOps.fromBFloat16Bits(b[bBaseOffset + i * strideB]);
                    out[outBaseOffset + i * outStride] = CpuDTypeOps.toBFloat16Bits(av - bv);
                }
            }
            case MUL -> {
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[aBaseOffset + i * strideA]);
                    float bv = CpuDTypeOps.fromBFloat16Bits(b[bBaseOffset + i * strideB]);
                    out[outBaseOffset + i * outStride] = CpuDTypeOps.toBFloat16Bits(av * bv);
                }
            }
            case DIV -> {
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[aBaseOffset + i * strideA]);
                    float bv = CpuDTypeOps.fromBFloat16Bits(b[bBaseOffset + i * strideB]);
                    out[outBaseOffset + i * outStride] = CpuDTypeOps.toBFloat16Bits(av / bv);
                }
            }
            case MIN -> {
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[aBaseOffset + i * strideA]);
                    float bv = CpuDTypeOps.fromBFloat16Bits(b[bBaseOffset + i * strideB]);
                    out[outBaseOffset + i * outStride] = CpuDTypeOps.toBFloat16Bits(Math.min(av, bv));
                }
            }
            case MAX -> {
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[aBaseOffset + i * strideA]);
                    float bv = CpuDTypeOps.fromBFloat16Bits(b[bBaseOffset + i * strideB]);
                    out[outBaseOffset + i * outStride] = CpuDTypeOps.toBFloat16Bits(Math.max(av, bv));
                }
            }
            case NEG -> {
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[aBaseOffset + i * strideA]);
                    out[outBaseOffset + i * outStride] = CpuDTypeOps.toBFloat16Bits(-av);
                }
            }
            case INV -> {
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[aBaseOffset + i * strideA]);
                    out[outBaseOffset + i * outStride] = CpuDTypeOps.toBFloat16Bits(1.0f / av);
                }
            }
            case LOG -> {
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[aBaseOffset + i * strideA]);
                    out[outBaseOffset + i * outStride] = CpuDTypeOps.toBFloat16Bits((float) Math.log(av));
                }
            }
            case EXP -> {
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[aBaseOffset + i * strideA]);
                    out[outBaseOffset + i * outStride] = CpuDTypeOps.toBFloat16Bits(
                            useFastExpApprox ? FastExp.fastExpF32(av) : (float) Math.exp(av)
                    );
                }
            }
            case FAST_EXP -> {
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[aBaseOffset + i * strideA]);
                    out[outBaseOffset + i * outStride] = CpuDTypeOps.toBFloat16Bits(FastExp.fastExpF32(av));
                }
            }
            case TANH -> {
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[aBaseOffset + i * strideA]);
                    out[outBaseOffset + i * outStride] = CpuDTypeOps.toBFloat16Bits(
                            useFastTanhApprox ? FastExp.fastTanhF32(av) : (float) Math.tanh(av)
                    );
                }
            }
            case FAST_TANH -> {
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[aBaseOffset + i * strideA]);
                    out[outBaseOffset + i * outStride] = CpuDTypeOps.toBFloat16Bits(FastExp.fastTanhF32(av));
                }
            }
            case SQRT -> {
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[aBaseOffset + i * strideA]);
                    out[outBaseOffset + i * outStride] = CpuDTypeOps.toBFloat16Bits((float) Math.sqrt(av));
                }
            }
            case ABS -> {
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[aBaseOffset + i * strideA]);
                    out[outBaseOffset + i * outStride] = CpuDTypeOps.toBFloat16Bits(Math.abs(av));
                }
            }
            case RELU -> {
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[aBaseOffset + i * strideA]);
                    out[outBaseOffset + i * outStride] = CpuDTypeOps.toBFloat16Bits(Math.max(0.0f, av));
                }
            }
            case CLAMP_MIN -> {
                float minValue = ((clampMin) op).getMinValueF32();
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[aBaseOffset + i * strideA]);
                    out[outBaseOffset + i * outStride] = CpuDTypeOps.toBFloat16Bits(Math.max(minValue, av));
                }
            }
            case CLAMP_MAX -> {
                float maxValue = ((clampMax) op).getMaxValueF32();
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[aBaseOffset + i * strideA]);
                    out[outBaseOffset + i * outStride] = CpuDTypeOps.toBFloat16Bits(Math.min(maxValue, av));
                }
            }
            case SIGMOID -> {
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[aBaseOffset + i * strideA]);
                    out[outBaseOffset + i * outStride] = CpuDTypeOps.toBFloat16Bits((float) (1.0 / (1.0 + Math.exp(-av))));
                }
            }
            case MUL_SCALAR -> {
                float scalar = ((mulScalar) op).getScalarF32();
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[aBaseOffset + i * strideA]);
                    out[outBaseOffset + i * outStride] = CpuDTypeOps.toBFloat16Bits(av * scalar);
                }
            }
            case POW -> {
                float exponent = ((pow) op).getExponentF32();
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[aBaseOffset + i * strideA]);
                    out[outBaseOffset + i * outStride] = CpuDTypeOps.toBFloat16Bits(CpuPowSupport.applyF32(av, exponent));
                }
            }
            default -> throw new UnsupportedOperationException("Unsupported strided opType=" + op.opType());
        }
    }

    private static void forwardGenericF64(
            Operation op,
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
            int logicalSize,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        if (logicalSize <= 0) {
            return;
        }
        MultiOffsetCursor cursor = new MultiOffsetCursor(
                shape,
                new int[][]{outStrides, aStrides, bStrides},
                new int[]{outBaseOffset, aBaseOffset, bBaseOffset}
        );
        switch (op.opType()) {
            case ADD -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = a[cursor.offset(1)] + b[cursor.offset(2)];
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case SUB -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = a[cursor.offset(1)] - b[cursor.offset(2)];
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case MUL -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = a[cursor.offset(1)] * b[cursor.offset(2)];
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case DIV -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = a[cursor.offset(1)] / b[cursor.offset(2)];
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case MIN -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = Math.min(a[cursor.offset(1)], b[cursor.offset(2)]);
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case MAX -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = Math.max(a[cursor.offset(1)], b[cursor.offset(2)]);
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case NEG -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = -a[cursor.offset(1)];
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case INV -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = 1.0 / a[cursor.offset(1)];
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case LOG -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = Math.log(a[cursor.offset(1)]);
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case EXP -> {
                for (int i = 0; i < logicalSize; i++) {
                    double value = a[cursor.offset(1)];
                    out[cursor.offset(0)] = useFastExpApprox ? FastExp.fastExpF64(value) : Math.exp(value);
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case FAST_EXP -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = FastExp.fastExpF64(a[cursor.offset(1)]);
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case TANH -> {
                for (int i = 0; i < logicalSize; i++) {
                    double value = a[cursor.offset(1)];
                    out[cursor.offset(0)] = useFastTanhApprox ? FastExp.fastTanhF64(value) : Math.tanh(value);
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case FAST_TANH -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = FastExp.fastTanhF64(a[cursor.offset(1)]);
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case SQRT -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = Math.sqrt(a[cursor.offset(1)]);
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case ABS -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = Math.abs(a[cursor.offset(1)]);
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case RELU -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = Math.max(0.0, a[cursor.offset(1)]);
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case CLAMP_MIN -> {
                double minValue = ((clampMin) op).getMinValue();
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = Math.max(minValue, a[cursor.offset(1)]);
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case CLAMP_MAX -> {
                double maxValue = ((clampMax) op).getMaxValue();
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = Math.min(maxValue, a[cursor.offset(1)]);
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case SIGMOID -> {
                for (int i = 0; i < logicalSize; i++) {
                    double value = a[cursor.offset(1)];
                    out[cursor.offset(0)] = 1.0 / (1.0 + Math.exp(-value));
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case MUL_SCALAR -> {
                double scalar = ((mulScalar) op).getScalar();
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = a[cursor.offset(1)] * scalar;
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case POW -> {
                double exponent = ((pow) op).getExponent();
                for (int i = 0; i < logicalSize; i++) {
                    double value = a[cursor.offset(1)];
                    out[cursor.offset(0)] = CpuPowSupport.applyF64(value, exponent);
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            default -> throw new UnsupportedOperationException("Unsupported strided opType=" + op.opType());
        }
    }

    private static void forwardGenericF32(
            Operation op,
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
            int logicalSize,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        if (logicalSize <= 0) {
            return;
        }
        MultiOffsetCursor cursor = new MultiOffsetCursor(
                shape,
                new int[][]{outStrides, aStrides, bStrides},
                new int[]{outBaseOffset, aBaseOffset, bBaseOffset}
        );
        switch (op.opType()) {
            case ADD -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = a[cursor.offset(1)] + b[cursor.offset(2)];
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case SUB -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = a[cursor.offset(1)] - b[cursor.offset(2)];
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case MUL -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = a[cursor.offset(1)] * b[cursor.offset(2)];
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case DIV -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = a[cursor.offset(1)] / b[cursor.offset(2)];
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case MIN -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = Math.min(a[cursor.offset(1)], b[cursor.offset(2)]);
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case MAX -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = Math.max(a[cursor.offset(1)], b[cursor.offset(2)]);
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case NEG -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = -a[cursor.offset(1)];
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case INV -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = 1.0f / a[cursor.offset(1)];
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case LOG -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = (float) Math.log(a[cursor.offset(1)]);
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case EXP -> {
                for (int i = 0; i < logicalSize; i++) {
                    float value = a[cursor.offset(1)];
                    out[cursor.offset(0)] = useFastExpApprox ? FastExp.fastExpF32(value) : (float) Math.exp(value);
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case FAST_EXP -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = FastExp.fastExpF32(a[cursor.offset(1)]);
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case TANH -> {
                for (int i = 0; i < logicalSize; i++) {
                    float value = a[cursor.offset(1)];
                    out[cursor.offset(0)] = useFastTanhApprox ? FastExp.fastTanhF32(value) : (float) Math.tanh(value);
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case FAST_TANH -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = FastExp.fastTanhF32(a[cursor.offset(1)]);
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case SQRT -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = (float) Math.sqrt(a[cursor.offset(1)]);
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case ABS -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = Math.abs(a[cursor.offset(1)]);
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case RELU -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = Math.max(0.0f, a[cursor.offset(1)]);
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case CLAMP_MIN -> {
                float minValue = ((clampMin) op).getMinValueF32();
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = Math.max(minValue, a[cursor.offset(1)]);
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case CLAMP_MAX -> {
                float maxValue = ((clampMax) op).getMaxValueF32();
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = Math.min(maxValue, a[cursor.offset(1)]);
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case SIGMOID -> {
                for (int i = 0; i < logicalSize; i++) {
                    float value = a[cursor.offset(1)];
                    out[cursor.offset(0)] = (float) (1.0 / (1.0 + Math.exp(-value)));
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case MUL_SCALAR -> {
                float scalar = ((mulScalar) op).getScalarF32();
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = a[cursor.offset(1)] * scalar;
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case POW -> {
                float exponent = ((pow) op).getExponentF32();
                for (int i = 0; i < logicalSize; i++) {
                    float value = a[cursor.offset(1)];
                    out[cursor.offset(0)] = CpuPowSupport.applyF32(value, exponent);
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            default -> throw new UnsupportedOperationException("Unsupported strided opType=" + op.opType());
        }
    }

    private static void forwardGenericBF16(
            Operation op,
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
            int logicalSize,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        if (logicalSize <= 0) {
            return;
        }
        MultiOffsetCursor cursor = new MultiOffsetCursor(
                shape,
                new int[][]{outStrides, aStrides, bStrides},
                new int[]{outBaseOffset, aBaseOffset, bBaseOffset}
        );
        switch (op.opType()) {
            case ADD -> {
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[cursor.offset(1)]);
                    float bv = CpuDTypeOps.fromBFloat16Bits(b[cursor.offset(2)]);
                    out[cursor.offset(0)] = CpuDTypeOps.toBFloat16Bits(av + bv);
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case SUB -> {
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[cursor.offset(1)]);
                    float bv = CpuDTypeOps.fromBFloat16Bits(b[cursor.offset(2)]);
                    out[cursor.offset(0)] = CpuDTypeOps.toBFloat16Bits(av - bv);
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case MUL -> {
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[cursor.offset(1)]);
                    float bv = CpuDTypeOps.fromBFloat16Bits(b[cursor.offset(2)]);
                    out[cursor.offset(0)] = CpuDTypeOps.toBFloat16Bits(av * bv);
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case DIV -> {
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[cursor.offset(1)]);
                    float bv = CpuDTypeOps.fromBFloat16Bits(b[cursor.offset(2)]);
                    out[cursor.offset(0)] = CpuDTypeOps.toBFloat16Bits(av / bv);
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case MIN -> {
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[cursor.offset(1)]);
                    float bv = CpuDTypeOps.fromBFloat16Bits(b[cursor.offset(2)]);
                    out[cursor.offset(0)] = CpuDTypeOps.toBFloat16Bits(Math.min(av, bv));
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case MAX -> {
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[cursor.offset(1)]);
                    float bv = CpuDTypeOps.fromBFloat16Bits(b[cursor.offset(2)]);
                    out[cursor.offset(0)] = CpuDTypeOps.toBFloat16Bits(Math.max(av, bv));
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case NEG -> {
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[cursor.offset(1)]);
                    out[cursor.offset(0)] = CpuDTypeOps.toBFloat16Bits(-av);
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case INV -> {
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[cursor.offset(1)]);
                    out[cursor.offset(0)] = CpuDTypeOps.toBFloat16Bits(1.0f / av);
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case LOG -> {
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[cursor.offset(1)]);
                    out[cursor.offset(0)] = CpuDTypeOps.toBFloat16Bits((float) Math.log(av));
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case EXP -> {
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[cursor.offset(1)]);
                    out[cursor.offset(0)] = CpuDTypeOps.toBFloat16Bits(
                            useFastExpApprox ? FastExp.fastExpF32(av) : (float) Math.exp(av)
                    );
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case FAST_EXP -> {
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[cursor.offset(1)]);
                    out[cursor.offset(0)] = CpuDTypeOps.toBFloat16Bits(FastExp.fastExpF32(av));
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case TANH -> {
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[cursor.offset(1)]);
                    out[cursor.offset(0)] = CpuDTypeOps.toBFloat16Bits(
                            useFastTanhApprox ? FastExp.fastTanhF32(av) : (float) Math.tanh(av)
                    );
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case FAST_TANH -> {
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[cursor.offset(1)]);
                    out[cursor.offset(0)] = CpuDTypeOps.toBFloat16Bits(FastExp.fastTanhF32(av));
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case SQRT -> {
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[cursor.offset(1)]);
                    out[cursor.offset(0)] = CpuDTypeOps.toBFloat16Bits((float) Math.sqrt(av));
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case ABS -> {
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[cursor.offset(1)]);
                    out[cursor.offset(0)] = CpuDTypeOps.toBFloat16Bits(Math.abs(av));
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case RELU -> {
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[cursor.offset(1)]);
                    out[cursor.offset(0)] = CpuDTypeOps.toBFloat16Bits(Math.max(0.0f, av));
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case CLAMP_MIN -> {
                float minValue = ((clampMin) op).getMinValueF32();
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[cursor.offset(1)]);
                    out[cursor.offset(0)] = CpuDTypeOps.toBFloat16Bits(Math.max(minValue, av));
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case CLAMP_MAX -> {
                float maxValue = ((clampMax) op).getMaxValueF32();
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[cursor.offset(1)]);
                    out[cursor.offset(0)] = CpuDTypeOps.toBFloat16Bits(Math.min(maxValue, av));
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case SIGMOID -> {
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[cursor.offset(1)]);
                    out[cursor.offset(0)] = CpuDTypeOps.toBFloat16Bits((float) (1.0 / (1.0 + Math.exp(-av))));
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case MUL_SCALAR -> {
                float scalar = ((mulScalar) op).getScalarF32();
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[cursor.offset(1)]);
                    out[cursor.offset(0)] = CpuDTypeOps.toBFloat16Bits(av * scalar);
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case POW -> {
                float exponent = ((pow) op).getExponentF32();
                for (int i = 0; i < logicalSize; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[cursor.offset(1)]);
                    out[cursor.offset(0)] = CpuDTypeOps.toBFloat16Bits(CpuPowSupport.applyF32(av, exponent));
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            default -> throw new UnsupportedOperationException("Unsupported strided opType=" + op.opType());
        }
    }

    private static void forwardGenericCompareF64(
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
        if (logicalSize <= 0) {
            return;
        }
        MultiOffsetCursor cursor = new MultiOffsetCursor(
                shape,
                new int[][]{outStrides, aStrides, bStrides},
                new int[]{outBaseOffset, aBaseOffset, bBaseOffset}
        );
        switch (op.opType()) {
            case GT -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = a[cursor.offset(1)] > b[cursor.offset(2)] ? (byte) 1 : (byte) 0;
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case GE -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = a[cursor.offset(1)] >= b[cursor.offset(2)] ? (byte) 1 : (byte) 0;
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case LT -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = a[cursor.offset(1)] < b[cursor.offset(2)] ? (byte) 1 : (byte) 0;
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case LE -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = a[cursor.offset(1)] <= b[cursor.offset(2)] ? (byte) 1 : (byte) 0;
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case EQ -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = a[cursor.offset(1)] == b[cursor.offset(2)] ? (byte) 1 : (byte) 0;
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case NE -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = a[cursor.offset(1)] != b[cursor.offset(2)] ? (byte) 1 : (byte) 0;
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            default -> throw new UnsupportedOperationException("Unsupported compare strided opType=" + op.opType());
        }
    }

    private static void forwardGenericCompareF32(
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
        if (logicalSize <= 0) {
            return;
        }
        MultiOffsetCursor cursor = new MultiOffsetCursor(
                shape,
                new int[][]{outStrides, aStrides, bStrides},
                new int[]{outBaseOffset, aBaseOffset, bBaseOffset}
        );
        switch (op.opType()) {
            case GT -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = a[cursor.offset(1)] > b[cursor.offset(2)] ? (byte) 1 : (byte) 0;
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case GE -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = a[cursor.offset(1)] >= b[cursor.offset(2)] ? (byte) 1 : (byte) 0;
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case LT -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = a[cursor.offset(1)] < b[cursor.offset(2)] ? (byte) 1 : (byte) 0;
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case LE -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = a[cursor.offset(1)] <= b[cursor.offset(2)] ? (byte) 1 : (byte) 0;
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case EQ -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = a[cursor.offset(1)] == b[cursor.offset(2)] ? (byte) 1 : (byte) 0;
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case NE -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = a[cursor.offset(1)] != b[cursor.offset(2)] ? (byte) 1 : (byte) 0;
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            default -> throw new UnsupportedOperationException("Unsupported compare strided opType=" + op.opType());
        }
    }

    private static void forwardGenericCompareBF16(
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
        if (logicalSize <= 0) {
            return;
        }
        MultiOffsetCursor cursor = new MultiOffsetCursor(
                shape,
                new int[][]{outStrides, aStrides, bStrides},
                new int[]{outBaseOffset, aBaseOffset, bBaseOffset}
        );
        switch (op.opType()) {
            case GT -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = CpuDTypeOps.fromBFloat16Bits(a[cursor.offset(1)])
                            > CpuDTypeOps.fromBFloat16Bits(b[cursor.offset(2)]) ? (byte) 1 : (byte) 0;
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case GE -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = CpuDTypeOps.fromBFloat16Bits(a[cursor.offset(1)])
                            >= CpuDTypeOps.fromBFloat16Bits(b[cursor.offset(2)]) ? (byte) 1 : (byte) 0;
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case LT -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = CpuDTypeOps.fromBFloat16Bits(a[cursor.offset(1)])
                            < CpuDTypeOps.fromBFloat16Bits(b[cursor.offset(2)]) ? (byte) 1 : (byte) 0;
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case LE -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = CpuDTypeOps.fromBFloat16Bits(a[cursor.offset(1)])
                            <= CpuDTypeOps.fromBFloat16Bits(b[cursor.offset(2)]) ? (byte) 1 : (byte) 0;
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case EQ -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = CpuDTypeOps.fromBFloat16Bits(a[cursor.offset(1)])
                            == CpuDTypeOps.fromBFloat16Bits(b[cursor.offset(2)]) ? (byte) 1 : (byte) 0;
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case NE -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = CpuDTypeOps.fromBFloat16Bits(a[cursor.offset(1)])
                            != CpuDTypeOps.fromBFloat16Bits(b[cursor.offset(2)]) ? (byte) 1 : (byte) 0;
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            default -> throw new UnsupportedOperationException("Unsupported compare strided opType=" + op.opType());
        }
    }

    private static void forwardGenericBoolUnary(
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
        if (logicalSize <= 0) {
            return;
        }
        MultiOffsetCursor cursor = new MultiOffsetCursor(
                shape,
                new int[][]{outStrides, aStrides},
                new int[]{outBaseOffset, aBaseOffset}
        );
        for (int i = 0; i < logicalSize; i++) {
            out[cursor.offset(0)] = a[cursor.offset(1)] == 0 ? (byte) 1 : (byte) 0;
            if (i + 1 < logicalSize) cursor.step();
        }
    }

    private static void forwardGenericBoolBinary(
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
        if (logicalSize <= 0) {
            return;
        }
        MultiOffsetCursor cursor = new MultiOffsetCursor(
                shape,
                new int[][]{outStrides, aStrides, bStrides},
                new int[]{outBaseOffset, aBaseOffset, bBaseOffset}
        );
        switch (op.opType()) {
            case LOGICAL_AND -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = (a[cursor.offset(1)] != 0 && b[cursor.offset(2)] != 0) ? (byte) 1 : (byte) 0;
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            case LOGICAL_OR -> {
                for (int i = 0; i < logicalSize; i++) {
                    out[cursor.offset(0)] = (a[cursor.offset(1)] != 0 || b[cursor.offset(2)] != 0) ? (byte) 1 : (byte) 0;
                    if (i + 1 < logicalSize) cursor.step();
                }
            }
            default -> throw new UnsupportedOperationException("Unsupported bool strided opType=" + op.opType());
        }
    }

    private static void forwardGenericWhereF64(
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
        MultiOffsetCursor cursor = new MultiOffsetCursor(
                shape,
                new int[][]{outStrides, condStrides, trueStrides, falseStrides},
                new int[]{outBaseOffset, condBaseOffset, trueBaseOffset, falseBaseOffset}
        );
        for (int i = 0; i < logicalSize; i++) {
            out[cursor.offset(0)] = cond[cursor.offset(1)] != 0
                    ? ifTrue[cursor.offset(2)]
                    : ifFalse[cursor.offset(3)];
            if (i + 1 < logicalSize) cursor.step();
        }
    }

    private static void forwardGenericWhereF32(
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
        MultiOffsetCursor cursor = new MultiOffsetCursor(
                shape,
                new int[][]{outStrides, condStrides, trueStrides, falseStrides},
                new int[]{outBaseOffset, condBaseOffset, trueBaseOffset, falseBaseOffset}
        );
        for (int i = 0; i < logicalSize; i++) {
            out[cursor.offset(0)] = cond[cursor.offset(1)] != 0
                    ? ifTrue[cursor.offset(2)]
                    : ifFalse[cursor.offset(3)];
            if (i + 1 < logicalSize) cursor.step();
        }
    }

    private static void forwardGenericWhereF16(
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
        MultiOffsetCursor cursor = new MultiOffsetCursor(
                shape,
                new int[][]{outStrides, condStrides, trueStrides, falseStrides},
                new int[]{outBaseOffset, condBaseOffset, trueBaseOffset, falseBaseOffset}
        );
        for (int i = 0; i < logicalSize; i++) {
            out[cursor.offset(0)] = cond[cursor.offset(1)] != 0
                    ? ifTrue[cursor.offset(2)]
                    : ifFalse[cursor.offset(3)];
            if (i + 1 < logicalSize) cursor.step();
        }
    }

    private static boolean tryForwardRank2F64(
            Operation op,
            double[] a,
            double[] b,
            int[] aStrides,
            int[] bStrides,
            int aBaseOffset,
            int bBaseOffset,
            double[] out,
            int rows,
            int cols,
            int[] outStrides,
            int outBaseOffset,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        if (out == null || outStrides == null || outStrides.length != 2) {
            return false;
        }
        return switch (op.opType()) {
            case ADD, SUB, MUL, DIV, MIN, MAX ->
                    tryForwardRank2BinaryF64(op, a, b, aStrides, bStrides, aBaseOffset, bBaseOffset, out, rows, cols, outStrides, outBaseOffset);
            case NEG, INV, LOG, EXP, FAST_EXP, TANH, FAST_TANH, POW, SQRT, ABS, MUL_SCALAR, RELU, CLAMP_MIN, CLAMP_MAX, SIGMOID ->
                    tryForwardRank2UnaryF64(op, a, aStrides, aBaseOffset, out, rows, cols, outStrides, outBaseOffset, useFastExpApprox, useFastTanhApprox);
            default -> false;
        };
    }

    private static boolean tryForwardRank2F32(
            Operation op,
            float[] a,
            float[] b,
            int[] aStrides,
            int[] bStrides,
            int aBaseOffset,
            int bBaseOffset,
            float[] out,
            int rows,
            int cols,
            int[] outStrides,
            int outBaseOffset,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        if (out == null || outStrides == null || outStrides.length != 2) {
            return false;
        }
        return switch (op.opType()) {
            case ADD, SUB, MUL, DIV, MIN, MAX ->
                    tryForwardRank2BinaryF32(op, a, b, aStrides, bStrides, aBaseOffset, bBaseOffset, out, rows, cols, outStrides, outBaseOffset);
            case NEG, INV, LOG, EXP, FAST_EXP, TANH, FAST_TANH, POW, SQRT, ABS, MUL_SCALAR, RELU, CLAMP_MIN, CLAMP_MAX, SIGMOID ->
                    tryForwardRank2UnaryF32(op, a, aStrides, aBaseOffset, out, rows, cols, outStrides, outBaseOffset, useFastExpApprox, useFastTanhApprox);
            default -> false;
        };
    }

    private static boolean tryForwardRank2BinaryF64(
            Operation op,
            double[] a,
            double[] b,
            int[] aStrides,
            int[] bStrides,
            int aBaseOffset,
            int bBaseOffset,
            double[] out,
            int rows,
            int cols,
            int[] outStrides,
            int outBaseOffset
    ) {
        if (a == null || b == null || aStrides == null || bStrides == null || aStrides.length != 2 || bStrides.length != 2) {
            return false;
        }

        int outRowStride = outStrides[0];
        int outColStride = outStrides[1];
        int aRowStride = aStrides[0];
        int aColStride = aStrides[1];
        int bRowStride = bStrides[0];
        int bColStride = bStrides[1];

        if (outColStride == 1 && aColStride == 1 && bColStride == 0) {
            for (int row = 0; row < rows; row++) {
                int outRowBase = outBaseOffset + row * outRowStride;
                int aRowBase = aBaseOffset + row * aRowStride;
                double right = b[bBaseOffset + row * bRowStride];
                vectorBinaryBroadcastRightF64(op, a, aRowBase, right, out, outRowBase, cols);
            }
            return true;
        }

        if (outColStride == 1 && aColStride == 0 && bColStride == 1) {
            for (int row = 0; row < rows; row++) {
                int outRowBase = outBaseOffset + row * outRowStride;
                int bRowBase = bBaseOffset + row * bRowStride;
                double left = a[aBaseOffset + row * aRowStride];
                vectorBinaryBroadcastLeftF64(op, left, b, bRowBase, out, outRowBase, cols);
            }
            return true;
        }

        switch (op.opType()) {
            case ADD -> {
                for (int row = 0; row < rows; row++) {
                    int outRowBase = outBaseOffset + row * outRowStride;
                    int aRowBase = aBaseOffset + row * aRowStride;
                    int bRowBase = bBaseOffset + row * bRowStride;
                    for (int col = 0; col < cols; col++) {
                        out[outRowBase + col * outColStride] = a[aRowBase + col * aColStride] + b[bRowBase + col * bColStride];
                    }
                }
            }
            case SUB -> {
                for (int row = 0; row < rows; row++) {
                    int outRowBase = outBaseOffset + row * outRowStride;
                    int aRowBase = aBaseOffset + row * aRowStride;
                    int bRowBase = bBaseOffset + row * bRowStride;
                    for (int col = 0; col < cols; col++) {
                        out[outRowBase + col * outColStride] = a[aRowBase + col * aColStride] - b[bRowBase + col * bColStride];
                    }
                }
            }
            case MUL -> {
                for (int row = 0; row < rows; row++) {
                    int outRowBase = outBaseOffset + row * outRowStride;
                    int aRowBase = aBaseOffset + row * aRowStride;
                    int bRowBase = bBaseOffset + row * bRowStride;
                    for (int col = 0; col < cols; col++) {
                        out[outRowBase + col * outColStride] = a[aRowBase + col * aColStride] * b[bRowBase + col * bColStride];
                    }
                }
            }
            case DIV -> {
                for (int row = 0; row < rows; row++) {
                    int outRowBase = outBaseOffset + row * outRowStride;
                    int aRowBase = aBaseOffset + row * aRowStride;
                    int bRowBase = bBaseOffset + row * bRowStride;
                    for (int col = 0; col < cols; col++) {
                        out[outRowBase + col * outColStride] = a[aRowBase + col * aColStride] / b[bRowBase + col * bColStride];
                    }
                }
            }
            case MIN -> {
                for (int row = 0; row < rows; row++) {
                    int outRowBase = outBaseOffset + row * outRowStride;
                    int aRowBase = aBaseOffset + row * aRowStride;
                    int bRowBase = bBaseOffset + row * bRowStride;
                    for (int col = 0; col < cols; col++) {
                        out[outRowBase + col * outColStride] = Math.min(a[aRowBase + col * aColStride], b[bRowBase + col * bColStride]);
                    }
                }
            }
            case MAX -> {
                for (int row = 0; row < rows; row++) {
                    int outRowBase = outBaseOffset + row * outRowStride;
                    int aRowBase = aBaseOffset + row * aRowStride;
                    int bRowBase = bBaseOffset + row * bRowStride;
                    for (int col = 0; col < cols; col++) {
                        out[outRowBase + col * outColStride] = Math.max(a[aRowBase + col * aColStride], b[bRowBase + col * bColStride]);
                    }
                }
            }
            default -> throw new UnsupportedOperationException("Unsupported rank-2 F64 binary op: " + op.opType());
        }
        return true;
    }

    private static boolean tryForwardRank2BinaryF32(
            Operation op,
            float[] a,
            float[] b,
            int[] aStrides,
            int[] bStrides,
            int aBaseOffset,
            int bBaseOffset,
            float[] out,
            int rows,
            int cols,
            int[] outStrides,
            int outBaseOffset
    ) {
        if (a == null || b == null || aStrides == null || bStrides == null || aStrides.length != 2 || bStrides.length != 2) {
            return false;
        }

        int outRowStride = outStrides[0];
        int outColStride = outStrides[1];
        int aRowStride = aStrides[0];
        int aColStride = aStrides[1];
        int bRowStride = bStrides[0];
        int bColStride = bStrides[1];

        if (outColStride == 1 && aColStride == 1 && bColStride == 0) {
            for (int row = 0; row < rows; row++) {
                int outRowBase = outBaseOffset + row * outRowStride;
                int aRowBase = aBaseOffset + row * aRowStride;
                float right = b[bBaseOffset + row * bRowStride];
                vectorBinaryBroadcastRightF32(op, a, aRowBase, right, out, outRowBase, cols);
            }
            return true;
        }

        if (outColStride == 1 && aColStride == 0 && bColStride == 1) {
            for (int row = 0; row < rows; row++) {
                int outRowBase = outBaseOffset + row * outRowStride;
                int bRowBase = bBaseOffset + row * bRowStride;
                float left = a[aBaseOffset + row * aRowStride];
                vectorBinaryBroadcastLeftF32(op, left, b, bRowBase, out, outRowBase, cols);
            }
            return true;
        }

        switch (op.opType()) {
            case ADD -> {
                for (int row = 0; row < rows; row++) {
                    int outRowBase = outBaseOffset + row * outRowStride;
                    int aRowBase = aBaseOffset + row * aRowStride;
                    int bRowBase = bBaseOffset + row * bRowStride;
                    for (int col = 0; col < cols; col++) {
                        out[outRowBase + col * outColStride] = a[aRowBase + col * aColStride] + b[bRowBase + col * bColStride];
                    }
                }
            }
            case SUB -> {
                for (int row = 0; row < rows; row++) {
                    int outRowBase = outBaseOffset + row * outRowStride;
                    int aRowBase = aBaseOffset + row * aRowStride;
                    int bRowBase = bBaseOffset + row * bRowStride;
                    for (int col = 0; col < cols; col++) {
                        out[outRowBase + col * outColStride] = a[aRowBase + col * aColStride] - b[bRowBase + col * bColStride];
                    }
                }
            }
            case MUL -> {
                for (int row = 0; row < rows; row++) {
                    int outRowBase = outBaseOffset + row * outRowStride;
                    int aRowBase = aBaseOffset + row * aRowStride;
                    int bRowBase = bBaseOffset + row * bRowStride;
                    for (int col = 0; col < cols; col++) {
                        out[outRowBase + col * outColStride] = a[aRowBase + col * aColStride] * b[bRowBase + col * bColStride];
                    }
                }
            }
            case DIV -> {
                for (int row = 0; row < rows; row++) {
                    int outRowBase = outBaseOffset + row * outRowStride;
                    int aRowBase = aBaseOffset + row * aRowStride;
                    int bRowBase = bBaseOffset + row * bRowStride;
                    for (int col = 0; col < cols; col++) {
                        out[outRowBase + col * outColStride] = a[aRowBase + col * aColStride] / b[bRowBase + col * bColStride];
                    }
                }
            }
            case MIN -> {
                for (int row = 0; row < rows; row++) {
                    int outRowBase = outBaseOffset + row * outRowStride;
                    int aRowBase = aBaseOffset + row * aRowStride;
                    int bRowBase = bBaseOffset + row * bRowStride;
                    for (int col = 0; col < cols; col++) {
                        out[outRowBase + col * outColStride] = Math.min(a[aRowBase + col * aColStride], b[bRowBase + col * bColStride]);
                    }
                }
            }
            case MAX -> {
                for (int row = 0; row < rows; row++) {
                    int outRowBase = outBaseOffset + row * outRowStride;
                    int aRowBase = aBaseOffset + row * aRowStride;
                    int bRowBase = bBaseOffset + row * bRowStride;
                    for (int col = 0; col < cols; col++) {
                        out[outRowBase + col * outColStride] = Math.max(a[aRowBase + col * aColStride], b[bRowBase + col * bColStride]);
                    }
                }
            }
            default -> throw new UnsupportedOperationException("Unsupported rank-2 F32 binary op: " + op.opType());
        }
        return true;
    }

    private static boolean tryForwardRank2UnaryF64(
            Operation op,
            double[] a,
            int[] aStrides,
            int aBaseOffset,
            double[] out,
            int rows,
            int cols,
            int[] outStrides,
            int outBaseOffset,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        if (a == null || aStrides == null || aStrides.length != 2) {
            return false;
        }

        int outRowStride = outStrides[0];
        int outColStride = outStrides[1];
        int aRowStride = aStrides[0];
        int aColStride = aStrides[1];

        switch (op.opType()) {
            case NEG -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) fillRowF64(out, outBaseOffset + row * outRowStride, cols, -a[aBaseOffset + row * aRowStride]);
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = -a[aRowBase + col * aColStride];
                    }
                }
            }
            case INV -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) fillRowF64(out, outBaseOffset + row * outRowStride, cols, 1.0 / a[aBaseOffset + row * aRowStride]);
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = 1.0 / a[aRowBase + col * aColStride];
                    }
                }
            }
            case LOG -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) fillRowF64(out, outBaseOffset + row * outRowStride, cols, Math.log(a[aBaseOffset + row * aRowStride]));
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = Math.log(a[aRowBase + col * aColStride]);
                    }
                }
            }
            case EXP -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) {
                        double value = a[aBaseOffset + row * aRowStride];
                        fillRowF64(out, outBaseOffset + row * outRowStride, cols, useFastExpApprox ? FastExp.fastExpF64(value) : Math.exp(value));
                    }
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) {
                            double value = a[aRowBase + col * aColStride];
                            out[outRowBase + col * outColStride] = useFastExpApprox ? FastExp.fastExpF64(value) : Math.exp(value);
                        }
                    }
                }
            }
            case FAST_EXP -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) fillRowF64(out, outBaseOffset + row * outRowStride, cols, FastExp.fastExpF64(a[aBaseOffset + row * aRowStride]));
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = FastExp.fastExpF64(a[aRowBase + col * aColStride]);
                    }
                }
            }
            case TANH -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) {
                        double value = a[aBaseOffset + row * aRowStride];
                        fillRowF64(out, outBaseOffset + row * outRowStride, cols, useFastTanhApprox ? FastExp.fastTanhF64(value) : Math.tanh(value));
                    }
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) {
                            double value = a[aRowBase + col * aColStride];
                            out[outRowBase + col * outColStride] = useFastTanhApprox ? FastExp.fastTanhF64(value) : Math.tanh(value);
                        }
                    }
                }
            }
            case FAST_TANH -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) fillRowF64(out, outBaseOffset + row * outRowStride, cols, FastExp.fastTanhF64(a[aBaseOffset + row * aRowStride]));
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = FastExp.fastTanhF64(a[aRowBase + col * aColStride]);
                    }
                }
            }
            case SQRT -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) fillRowF64(out, outBaseOffset + row * outRowStride, cols, Math.sqrt(a[aBaseOffset + row * aRowStride]));
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = Math.sqrt(a[aRowBase + col * aColStride]);
                    }
                }
            }
            case ABS -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) fillRowF64(out, outBaseOffset + row * outRowStride, cols, Math.abs(a[aBaseOffset + row * aRowStride]));
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = Math.abs(a[aRowBase + col * aColStride]);
                    }
                }
            }
            case MUL_SCALAR -> {
                double scalar = ((mulScalar) op).getScalar();
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) fillRowF64(out, outBaseOffset + row * outRowStride, cols, a[aBaseOffset + row * aRowStride] * scalar);
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = a[aRowBase + col * aColStride] * scalar;
                    }
                }
            }
            case RELU -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) fillRowF64(out, outBaseOffset + row * outRowStride, cols, Math.max(0.0, a[aBaseOffset + row * aRowStride]));
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = Math.max(0.0, a[aRowBase + col * aColStride]);
                    }
                }
            }
            case CLAMP_MIN -> {
                double minValue = ((clampMin) op).getMinValue();
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) fillRowF64(out, outBaseOffset + row * outRowStride, cols, Math.max(minValue, a[aBaseOffset + row * aRowStride]));
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = Math.max(minValue, a[aRowBase + col * aColStride]);
                    }
                }
            }
            case CLAMP_MAX -> {
                double maxValue = ((clampMax) op).getMaxValue();
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) fillRowF64(out, outBaseOffset + row * outRowStride, cols, Math.min(maxValue, a[aBaseOffset + row * aRowStride]));
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = Math.min(maxValue, a[aRowBase + col * aColStride]);
                    }
                }
            }
            case SIGMOID -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) {
                        double value = a[aBaseOffset + row * aRowStride];
                        fillRowF64(out, outBaseOffset + row * outRowStride, cols, 1.0 / (1.0 + Math.exp(-value)));
                    }
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) {
                            double value = a[aRowBase + col * aColStride];
                            out[outRowBase + col * outColStride] = 1.0 / (1.0 + Math.exp(-value));
                        }
                    }
                }
            }
            case POW -> {
                double exponent = ((pow) op).getExponent();
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) {
                        double value = a[aBaseOffset + row * aRowStride];
                        fillRowF64(out, outBaseOffset + row * outRowStride, cols, CpuPowSupport.applyF64(value, exponent));
                    }
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) {
                            double value = a[aRowBase + col * aColStride];
                            out[outRowBase + col * outColStride] = CpuPowSupport.applyF64(value, exponent);
                        }
                    }
                }
            }
            default -> throw new UnsupportedOperationException("Unsupported rank-2 F64 unary op: " + op.opType());
        }
        return true;
    }

    private static boolean tryForwardRank2UnaryF32(
            Operation op,
            float[] a,
            int[] aStrides,
            int aBaseOffset,
            float[] out,
            int rows,
            int cols,
            int[] outStrides,
            int outBaseOffset,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        if (a == null || aStrides == null || aStrides.length != 2) {
            return false;
        }

        int outRowStride = outStrides[0];
        int outColStride = outStrides[1];
        int aRowStride = aStrides[0];
        int aColStride = aStrides[1];

        switch (op.opType()) {
            case NEG -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) fillRowF32(out, outBaseOffset + row * outRowStride, cols, -a[aBaseOffset + row * aRowStride]);
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = -a[aRowBase + col * aColStride];
                    }
                }
            }
            case INV -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) fillRowF32(out, outBaseOffset + row * outRowStride, cols, 1.0f / a[aBaseOffset + row * aRowStride]);
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = 1.0f / a[aRowBase + col * aColStride];
                    }
                }
            }
            case LOG -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) fillRowF32(out, outBaseOffset + row * outRowStride, cols, (float) Math.log(a[aBaseOffset + row * aRowStride]));
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = (float) Math.log(a[aRowBase + col * aColStride]);
                    }
                }
            }
            case EXP -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) {
                        float value = a[aBaseOffset + row * aRowStride];
                        fillRowF32(out, outBaseOffset + row * outRowStride, cols, useFastExpApprox ? FastExp.fastExpF32(value) : (float) Math.exp(value));
                    }
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) {
                            float value = a[aRowBase + col * aColStride];
                            out[outRowBase + col * outColStride] = useFastExpApprox ? FastExp.fastExpF32(value) : (float) Math.exp(value);
                        }
                    }
                }
            }
            case FAST_EXP -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) fillRowF32(out, outBaseOffset + row * outRowStride, cols, FastExp.fastExpF32(a[aBaseOffset + row * aRowStride]));
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = FastExp.fastExpF32(a[aRowBase + col * aColStride]);
                    }
                }
            }
            case TANH -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) {
                        float value = a[aBaseOffset + row * aRowStride];
                        fillRowF32(out, outBaseOffset + row * outRowStride, cols, useFastTanhApprox ? FastExp.fastTanhF32(value) : (float) Math.tanh(value));
                    }
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) {
                            float value = a[aRowBase + col * aColStride];
                            out[outRowBase + col * outColStride] = useFastTanhApprox ? FastExp.fastTanhF32(value) : (float) Math.tanh(value);
                        }
                    }
                }
            }
            case FAST_TANH -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) fillRowF32(out, outBaseOffset + row * outRowStride, cols, FastExp.fastTanhF32(a[aBaseOffset + row * aRowStride]));
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = FastExp.fastTanhF32(a[aRowBase + col * aColStride]);
                    }
                }
            }
            case SQRT -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) fillRowF32(out, outBaseOffset + row * outRowStride, cols, (float) Math.sqrt(a[aBaseOffset + row * aRowStride]));
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = (float) Math.sqrt(a[aRowBase + col * aColStride]);
                    }
                }
            }
            case ABS -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) fillRowF32(out, outBaseOffset + row * outRowStride, cols, Math.abs(a[aBaseOffset + row * aRowStride]));
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = Math.abs(a[aRowBase + col * aColStride]);
                    }
                }
            }
            case MUL_SCALAR -> {
                float scalar = ((mulScalar) op).getScalarF32();
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) fillRowF32(out, outBaseOffset + row * outRowStride, cols, a[aBaseOffset + row * aRowStride] * scalar);
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = a[aRowBase + col * aColStride] * scalar;
                    }
                }
            }
            case RELU -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) fillRowF32(out, outBaseOffset + row * outRowStride, cols, Math.max(0.0f, a[aBaseOffset + row * aRowStride]));
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = Math.max(0.0f, a[aRowBase + col * aColStride]);
                    }
                }
            }
            case CLAMP_MIN -> {
                float minValue = ((clampMin) op).getMinValueF32();
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) fillRowF32(out, outBaseOffset + row * outRowStride, cols, Math.max(minValue, a[aBaseOffset + row * aRowStride]));
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = Math.max(minValue, a[aRowBase + col * aColStride]);
                    }
                }
            }
            case CLAMP_MAX -> {
                float maxValue = ((clampMax) op).getMaxValueF32();
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) fillRowF32(out, outBaseOffset + row * outRowStride, cols, Math.min(maxValue, a[aBaseOffset + row * aRowStride]));
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) out[outRowBase + col * outColStride] = Math.min(maxValue, a[aRowBase + col * aColStride]);
                    }
                }
            }
            case SIGMOID -> {
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) {
                        float value = a[aBaseOffset + row * aRowStride];
                        fillRowF32(out, outBaseOffset + row * outRowStride, cols, (float) (1.0 / (1.0 + Math.exp(-value))));
                    }
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) {
                            float value = a[aRowBase + col * aColStride];
                            out[outRowBase + col * outColStride] = (float) (1.0 / (1.0 + Math.exp(-value)));
                        }
                    }
                }
            }
            case POW -> {
                float exponent = ((pow) op).getExponentF32();
                if (outColStride == 1 && aColStride == 0) {
                    for (int row = 0; row < rows; row++) {
                        float value = a[aBaseOffset + row * aRowStride];
                        fillRowF32(out, outBaseOffset + row * outRowStride, cols, CpuPowSupport.applyF32(value, exponent));
                    }
                } else {
                    for (int row = 0; row < rows; row++) {
                        int outRowBase = outBaseOffset + row * outRowStride;
                        int aRowBase = aBaseOffset + row * aRowStride;
                        for (int col = 0; col < cols; col++) {
                            float value = a[aRowBase + col * aColStride];
                            out[outRowBase + col * outColStride] = CpuPowSupport.applyF32(value, exponent);
                        }
                    }
                }
            }
            default -> throw new UnsupportedOperationException("Unsupported rank-2 F32 unary op: " + op.opType());
        }
        return true;
    }

    private static void vectorBinaryBroadcastRightF64(
            Operation op,
            double[] left,
            int leftBase,
            double right,
            double[] out,
            int outBase,
            int cols
    ) {
        int width = F64.length();
        int upper = cols - (cols % width);
        int col = 0;
        DoubleVector rightVector = DoubleVector.broadcast(F64, right);
        switch (op.opType()) {
            case ADD -> {
                for (; col < upper; col += width) DoubleVector.fromArray(F64, left, leftBase + col).add(rightVector).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = left[leftBase + col] + right;
            }
            case SUB -> {
                for (; col < upper; col += width) DoubleVector.fromArray(F64, left, leftBase + col).sub(rightVector).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = left[leftBase + col] - right;
            }
            case MUL -> {
                for (; col < upper; col += width) DoubleVector.fromArray(F64, left, leftBase + col).mul(rightVector).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = left[leftBase + col] * right;
            }
            case DIV -> {
                for (; col < upper; col += width) DoubleVector.fromArray(F64, left, leftBase + col).div(rightVector).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = left[leftBase + col] / right;
            }
            case MIN -> {
                for (; col < upper; col += width) DoubleVector.fromArray(F64, left, leftBase + col).min(rightVector).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = Math.min(left[leftBase + col], right);
            }
            case MAX -> {
                for (; col < upper; col += width) DoubleVector.fromArray(F64, left, leftBase + col).max(rightVector).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = Math.max(left[leftBase + col], right);
            }
            default -> throw new UnsupportedOperationException("Unsupported vector rank-2 F64 binary op: " + op.opType());
        }
    }

    private static void vectorBinaryBroadcastRightF32(
            Operation op,
            float[] left,
            int leftBase,
            float right,
            float[] out,
            int outBase,
            int cols
    ) {
        int width = F32.length();
        int upper = cols - (cols % width);
        int col = 0;
        FloatVector rightVector = FloatVector.broadcast(F32, right);
        switch (op.opType()) {
            case ADD -> {
                for (; col < upper; col += width) FloatVector.fromArray(F32, left, leftBase + col).add(rightVector).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = left[leftBase + col] + right;
            }
            case SUB -> {
                for (; col < upper; col += width) FloatVector.fromArray(F32, left, leftBase + col).sub(rightVector).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = left[leftBase + col] - right;
            }
            case MUL -> {
                for (; col < upper; col += width) FloatVector.fromArray(F32, left, leftBase + col).mul(rightVector).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = left[leftBase + col] * right;
            }
            case DIV -> {
                for (; col < upper; col += width) FloatVector.fromArray(F32, left, leftBase + col).div(rightVector).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = left[leftBase + col] / right;
            }
            case MIN -> {
                for (; col < upper; col += width) FloatVector.fromArray(F32, left, leftBase + col).min(rightVector).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = Math.min(left[leftBase + col], right);
            }
            case MAX -> {
                for (; col < upper; col += width) FloatVector.fromArray(F32, left, leftBase + col).max(rightVector).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = Math.max(left[leftBase + col], right);
            }
            default -> throw new UnsupportedOperationException("Unsupported vector rank-2 F32 binary op: " + op.opType());
        }
    }

    private static void vectorBinaryBroadcastLeftF64(
            Operation op,
            double left,
            double[] right,
            int rightBase,
            double[] out,
            int outBase,
            int cols
    ) {
        int width = F64.length();
        int upper = cols - (cols % width);
        int col = 0;
        DoubleVector leftVector = DoubleVector.broadcast(F64, left);
        switch (op.opType()) {
            case ADD -> {
                for (; col < upper; col += width) leftVector.add(DoubleVector.fromArray(F64, right, rightBase + col)).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = left + right[rightBase + col];
            }
            case SUB -> {
                for (; col < upper; col += width) leftVector.sub(DoubleVector.fromArray(F64, right, rightBase + col)).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = left - right[rightBase + col];
            }
            case MUL -> {
                for (; col < upper; col += width) leftVector.mul(DoubleVector.fromArray(F64, right, rightBase + col)).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = left * right[rightBase + col];
            }
            case DIV -> {
                for (; col < upper; col += width) leftVector.div(DoubleVector.fromArray(F64, right, rightBase + col)).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = left / right[rightBase + col];
            }
            case MIN -> {
                for (; col < upper; col += width) leftVector.min(DoubleVector.fromArray(F64, right, rightBase + col)).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = Math.min(left, right[rightBase + col]);
            }
            case MAX -> {
                for (; col < upper; col += width) leftVector.max(DoubleVector.fromArray(F64, right, rightBase + col)).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = Math.max(left, right[rightBase + col]);
            }
            default -> throw new UnsupportedOperationException("Unsupported vector rank-2 F64 binary op: " + op.opType());
        }
    }

    private static void vectorBinaryBroadcastLeftF32(
            Operation op,
            float left,
            float[] right,
            int rightBase,
            float[] out,
            int outBase,
            int cols
    ) {
        int width = F32.length();
        int upper = cols - (cols % width);
        int col = 0;
        FloatVector leftVector = FloatVector.broadcast(F32, left);
        switch (op.opType()) {
            case ADD -> {
                for (; col < upper; col += width) leftVector.add(FloatVector.fromArray(F32, right, rightBase + col)).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = left + right[rightBase + col];
            }
            case SUB -> {
                for (; col < upper; col += width) leftVector.sub(FloatVector.fromArray(F32, right, rightBase + col)).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = left - right[rightBase + col];
            }
            case MUL -> {
                for (; col < upper; col += width) leftVector.mul(FloatVector.fromArray(F32, right, rightBase + col)).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = left * right[rightBase + col];
            }
            case DIV -> {
                for (; col < upper; col += width) leftVector.div(FloatVector.fromArray(F32, right, rightBase + col)).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = left / right[rightBase + col];
            }
            case MIN -> {
                for (; col < upper; col += width) leftVector.min(FloatVector.fromArray(F32, right, rightBase + col)).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = Math.min(left, right[rightBase + col]);
            }
            case MAX -> {
                for (; col < upper; col += width) leftVector.max(FloatVector.fromArray(F32, right, rightBase + col)).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = Math.max(left, right[rightBase + col]);
            }
            default -> throw new UnsupportedOperationException("Unsupported vector rank-2 F32 binary op: " + op.opType());
        }
    }

    private static void fillRowF64(double[] out, int outBase, int cols, double value) {
        int width = F64.length();
        int upper = cols - (cols % width);
        int col = 0;
        DoubleVector vector = DoubleVector.broadcast(F64, value);
        for (; col < upper; col += width) {
            vector.intoArray(out, outBase + col);
        }
        for (; col < cols; col++) {
            out[outBase + col] = value;
        }
    }

    private static void fillRowF32(float[] out, int outBase, int cols, float value) {
        int width = F32.length();
        int upper = cols - (cols % width);
        int col = 0;
        FloatVector vector = FloatVector.broadcast(F32, value);
        for (; col < upper; col += width) {
            vector.intoArray(out, outBase + col);
        }
        for (; col < cols; col++) {
            out[outBase + col] = value;
        }
    }

    private static final class MultiOffsetCursor {
        private final int[] shape;
        private final int[] coords;
        private final int[][] strides;
        private final int[] offsets;

        private MultiOffsetCursor(int[] shape, int[][] strides, int[] baseOffsets) {
            this.shape = shape != null ? shape : new int[0];
            this.coords = new int[this.shape.length];
            this.strides = strides;
            this.offsets = baseOffsets.clone();
        }

        private int offset(int slot) {
            return offsets[slot];
        }

        private void step() {
            for (int dim = shape.length - 1; dim >= 0; dim--) {
                int nextCoord = coords[dim] + 1;
                if (nextCoord < shape[dim]) {
                    coords[dim] = nextCoord;
                    for (int slot = 0; slot < offsets.length; slot++) {
                        int[] slotStrides = strides[slot];
                        if (slotStrides != null) {
                            offsets[slot] += slotStrides[dim];
                        }
                    }
                    return;
                }

                int currentCoord = coords[dim];
                if (currentCoord != 0) {
                    for (int slot = 0; slot < offsets.length; slot++) {
                        int[] slotStrides = strides[slot];
                        if (slotStrides != null) {
                            offsets[slot] -= currentCoord * slotStrides[dim];
                        }
                    }
                }
                coords[dim] = 0;
            }
        }
    }
}
