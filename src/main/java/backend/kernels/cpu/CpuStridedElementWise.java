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
        int[] outDenseStrides = tensor.TensorMetadata.computeStrides(outShape);
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
            forwardRank1(
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

        for (int i = 0; i < node.getFlatDataSize(); i++) {
            int outIdx = remapIndex(i, outDenseStrides, outStrides, rank, outBaseOffset);
            int aIdx = a != null ? remapIndex(i, outDenseStrides, aStrides, rank, aBaseOffset) : -1;
            int bIdx = b != null ? remapIndex(i, outDenseStrides, bStrides, rank, bBaseOffset) : -1;
            out[outIdx] = eval(op, a, b, aIdx, bIdx, useFastExpApprox, useFastTanhApprox);
        }
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
        int[] outDenseStrides = tensor.TensorMetadata.computeStrides(outShape);
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
            int strideA = a != null ? aStrides[0] : 0;
            int strideB = b != null ? bStrides[0] : 0;
            for (int i = 0; i < node.getFlatDataSize(); i++) {
                int outIdx = outBaseOffset + i * outStrides[0];
                int aIdx = a != null ? aBaseOffset + i * strideA : -1;
                int bIdx = b != null ? bBaseOffset + i * strideB : -1;
                out[outIdx] = evalF32(op, a, b, aIdx, bIdx, useFastExpApprox, useFastTanhApprox);
            }
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

        for (int i = 0; i < node.getFlatDataSize(); i++) {
            int outIdx = remapIndex(i, outDenseStrides, outStrides, rank, outBaseOffset);
            int aIdx = a != null ? remapIndex(i, outDenseStrides, aStrides, rank, aBaseOffset) : -1;
            int bIdx = b != null ? remapIndex(i, outDenseStrides, bStrides, rank, bBaseOffset) : -1;
            out[outIdx] = evalF32(op, a, b, aIdx, bIdx, useFastExpApprox, useFastTanhApprox);
        }
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
        int[] outDenseStrides = tensor.TensorMetadata.computeStrides(outShape);
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
            int strideA = a != null ? aStrides[0] : 0;
            int strideB = b != null ? bStrides[0] : 0;
            for (int i = 0; i < node.getFlatDataSize(); i++) {
                int outIdx = outBaseOffset + i * outStrides[0];
                int aIdx = a != null ? aBaseOffset + i * strideA : -1;
                int bIdx = b != null ? bBaseOffset + i * strideB : -1;
                out[outIdx] = evalF16(op, a, b, aIdx, bIdx, useFastExpApprox, useFastTanhApprox);
            }
            return;
        }

        for (int i = 0; i < node.getFlatDataSize(); i++) {
            int outIdx = remapIndex(i, outDenseStrides, outStrides, rank, outBaseOffset);
            int aIdx = a != null ? remapIndex(i, outDenseStrides, aStrides, rank, aBaseOffset) : -1;
            int bIdx = b != null ? remapIndex(i, outDenseStrides, bStrides, rank, bBaseOffset) : -1;
            out[outIdx] = evalF16(op, a, b, aIdx, bIdx, useFastExpApprox, useFastTanhApprox);
        }
    }

    private static void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node) {
        byte[] out = node.getBoolData();
        if (out == null) {
            return;
        }

        int[] outShape = node.getShapeUnsafe();
        int[] outDenseStrides = tensor.TensorMetadata.computeStrides(outShape);
        int[] outStrides = node.getStridesUnsafe();
        int outBaseOffset = node.getStorageOffsetUnsafe();
        int rank = outShape.length;

        if (op.opType() == Operation.OpType.LOGICAL_NOT) {
            Tensor ta = inputs.getFirst();
            byte[] a = ta.getBoolData();
            int[] aStrides = ta.getStridesUnsafe();
            int aBaseOffset = ta.getStorageOffsetUnsafe();
            if (rank == 1) {
                for (int i = 0; i < node.getFlatDataSize(); i++) {
                    out[outBaseOffset + i * outStrides[0]] = evalBoolUnary(op, a[aBaseOffset + i * aStrides[0]]);
                }
                return;
            }
            for (int i = 0; i < node.getFlatDataSize(); i++) {
                int outIdx = remapIndex(i, outDenseStrides, outStrides, rank, outBaseOffset);
                int aIdx = remapIndex(i, outDenseStrides, aStrides, rank, aBaseOffset);
                out[outIdx] = evalBoolUnary(op, a[aIdx]);
            }
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
                for (int i = 0; i < node.getFlatDataSize(); i++) {
                    out[outBaseOffset + i * outStrides[0]] =
                            evalBoolBinary(op, a[aBaseOffset + i * aStrides[0]], b[bBaseOffset + i * bStrides[0]]);
                }
                return;
            }
            for (int i = 0; i < node.getFlatDataSize(); i++) {
                int outIdx = remapIndex(i, outDenseStrides, outStrides, rank, outBaseOffset);
                int aIdx = remapIndex(i, outDenseStrides, aStrides, rank, aBaseOffset);
                int bIdx = remapIndex(i, outDenseStrides, bStrides, rank, bBaseOffset);
                out[outIdx] = evalBoolBinary(op, a[aIdx], b[bIdx]);
            }
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
                    for (int i = 0; i < node.getFlatDataSize(); i++) {
                        out[outBaseOffset + i * outStrides[0]] =
                                evalCompare(op, a[aBaseOffset + i * aStrides[0]], b[bBaseOffset + i * bStrides[0]]);
                    }
                    return;
                }
                for (int i = 0; i < node.getFlatDataSize(); i++) {
                    int outIdx = remapIndex(i, outDenseStrides, outStrides, rank, outBaseOffset);
                    int aIdx = remapIndex(i, outDenseStrides, aStrides, rank, aBaseOffset);
                    int bIdx = remapIndex(i, outDenseStrides, bStrides, rank, bBaseOffset);
                    out[outIdx] = evalCompare(op, a[aIdx], b[bIdx]);
                }
            }
            case FLOAT32 -> {
                float[] a = ta.getFloat32Data();
                float[] b = tb.getFloat32Data();
                int[] aStrides = ta.getStridesUnsafe();
                int[] bStrides = tb.getStridesUnsafe();
                int aBaseOffset = ta.getStorageOffsetUnsafe();
                int bBaseOffset = tb.getStorageOffsetUnsafe();
                if (rank == 1) {
                    for (int i = 0; i < node.getFlatDataSize(); i++) {
                        out[outBaseOffset + i * outStrides[0]] =
                                evalCompare(op, a[aBaseOffset + i * aStrides[0]], b[bBaseOffset + i * bStrides[0]]);
                    }
                    return;
                }
                for (int i = 0; i < node.getFlatDataSize(); i++) {
                    int outIdx = remapIndex(i, outDenseStrides, outStrides, rank, outBaseOffset);
                    int aIdx = remapIndex(i, outDenseStrides, aStrides, rank, aBaseOffset);
                    int bIdx = remapIndex(i, outDenseStrides, bStrides, rank, bBaseOffset);
                    out[outIdx] = evalCompare(op, a[aIdx], b[bIdx]);
                }
            }
            case BFLOAT16 -> {
                short[] a = ta.getBFloat16Data();
                short[] b = tb.getBFloat16Data();
                int[] aStrides = ta.getStridesUnsafe();
                int[] bStrides = tb.getStridesUnsafe();
                int aBaseOffset = ta.getStorageOffsetUnsafe();
                int bBaseOffset = tb.getStorageOffsetUnsafe();
                if (rank == 1) {
                    for (int i = 0; i < node.getFlatDataSize(); i++) {
                        out[outBaseOffset + i * outStrides[0]] = evalCompare(
                                op,
                                CpuDTypeOps.fromBFloat16Bits(a[aBaseOffset + i * aStrides[0]]),
                                CpuDTypeOps.fromBFloat16Bits(b[bBaseOffset + i * bStrides[0]])
                        );
                    }
                    return;
                }
                for (int i = 0; i < node.getFlatDataSize(); i++) {
                    int outIdx = remapIndex(i, outDenseStrides, outStrides, rank, outBaseOffset);
                    int aIdx = remapIndex(i, outDenseStrides, aStrides, rank, aBaseOffset);
                    int bIdx = remapIndex(i, outDenseStrides, bStrides, rank, bBaseOffset);
                    out[outIdx] = evalCompare(op, CpuDTypeOps.fromBFloat16Bits(a[aIdx]), CpuDTypeOps.fromBFloat16Bits(b[bIdx]));
                }
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
        int[] outDenseStrides = tensor.TensorMetadata.computeStrides(outShape);
        int[] outStrides = node.getStridesUnsafe();
        int[] condStrides = inputs.get(0).getStridesUnsafe();
        int[] trueStrides = inputs.get(1).getStridesUnsafe();
        int[] falseStrides = inputs.get(2).getStridesUnsafe();
        int outBaseOffset = node.getStorageOffsetUnsafe();
        int condBaseOffset = inputs.get(0).getStorageOffsetUnsafe();
        int trueBaseOffset = inputs.get(1).getStorageOffsetUnsafe();
        int falseBaseOffset = inputs.get(2).getStorageOffsetUnsafe();
        int rank = outShape.length;

        for (int i = 0; i < node.getFlatDataSize(); i++) {
            int outIdx = remapIndex(i, outDenseStrides, outStrides, rank, outBaseOffset);
            int condIdx = remapIndex(i, outDenseStrides, condStrides, rank, condBaseOffset);
            int trueIdx = remapIndex(i, outDenseStrides, trueStrides, rank, trueBaseOffset);
            int falseIdx = remapIndex(i, outDenseStrides, falseStrides, rank, falseBaseOffset);
            out[outIdx] = cond[condIdx] != 0 ? ifTrue[trueIdx] : ifFalse[falseIdx];
        }
    }

    private static void forwardWhereF32(List<Tensor> inputs, Tensor node, float[] out) {
        byte[] cond = inputs.get(0).getBoolData();
        float[] ifTrue = inputs.get(1).getFloat32Data();
        float[] ifFalse = inputs.get(2).getFloat32Data();
        int[] outShape = node.getShapeUnsafe();
        int[] outDenseStrides = tensor.TensorMetadata.computeStrides(outShape);
        int[] outStrides = node.getStridesUnsafe();
        int[] condStrides = inputs.get(0).getStridesUnsafe();
        int[] trueStrides = inputs.get(1).getStridesUnsafe();
        int[] falseStrides = inputs.get(2).getStridesUnsafe();
        int outBaseOffset = node.getStorageOffsetUnsafe();
        int condBaseOffset = inputs.get(0).getStorageOffsetUnsafe();
        int trueBaseOffset = inputs.get(1).getStorageOffsetUnsafe();
        int falseBaseOffset = inputs.get(2).getStorageOffsetUnsafe();
        int rank = outShape.length;

        for (int i = 0; i < node.getFlatDataSize(); i++) {
            int outIdx = remapIndex(i, outDenseStrides, outStrides, rank, outBaseOffset);
            int condIdx = remapIndex(i, outDenseStrides, condStrides, rank, condBaseOffset);
            int trueIdx = remapIndex(i, outDenseStrides, trueStrides, rank, trueBaseOffset);
            int falseIdx = remapIndex(i, outDenseStrides, falseStrides, rank, falseBaseOffset);
            out[outIdx] = cond[condIdx] != 0 ? ifTrue[trueIdx] : ifFalse[falseIdx];
        }
    }

    private static void forwardWhereF16(List<Tensor> inputs, Tensor node, short[] out) {
        byte[] cond = inputs.get(0).getBoolData();
        short[] ifTrue = inputs.get(1).getBFloat16Data();
        short[] ifFalse = inputs.get(2).getBFloat16Data();
        int[] outShape = node.getShapeUnsafe();
        int[] outDenseStrides = tensor.TensorMetadata.computeStrides(outShape);
        int[] outStrides = node.getStridesUnsafe();
        int[] condStrides = inputs.get(0).getStridesUnsafe();
        int[] trueStrides = inputs.get(1).getStridesUnsafe();
        int[] falseStrides = inputs.get(2).getStridesUnsafe();
        int outBaseOffset = node.getStorageOffsetUnsafe();
        int condBaseOffset = inputs.get(0).getStorageOffsetUnsafe();
        int trueBaseOffset = inputs.get(1).getStorageOffsetUnsafe();
        int falseBaseOffset = inputs.get(2).getStorageOffsetUnsafe();
        int rank = outShape.length;

        for (int i = 0; i < node.getFlatDataSize(); i++) {
            int outIdx = remapIndex(i, outDenseStrides, outStrides, rank, outBaseOffset);
            int condIdx = remapIndex(i, outDenseStrides, condStrides, rank, condBaseOffset);
            int trueIdx = remapIndex(i, outDenseStrides, trueStrides, rank, trueBaseOffset);
            int falseIdx = remapIndex(i, outDenseStrides, falseStrides, rank, falseBaseOffset);
            out[outIdx] = cond[condIdx] != 0 ? ifTrue[trueIdx] : ifFalse[falseIdx];
        }
    }

    private static void forwardRank1(
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
        for (int i = 0; i < logicalSize; i++) {
            int outIdx = outBaseOffset + i * outStride;
            int aIdx = a != null ? aBaseOffset + i * strideA : -1;
            int bIdx = b != null ? bBaseOffset + i * strideB : -1;
            out[outIdx] = eval(op, a, b, aIdx, bIdx, useFastExpApprox, useFastTanhApprox);
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

        for (int row = 0; row < rows; row++) {
            int outRowBase = outBaseOffset + row * outRowStride;
            int aRowBase = aBaseOffset + row * aRowStride;
            int bRowBase = bBaseOffset + row * bRowStride;
            for (int col = 0; col < cols; col++) {
                out[outRowBase + col * outColStride] = switch (op.opType()) {
                    case ADD -> a[aRowBase + col * aColStride] + b[bRowBase + col * bColStride];
                    case SUB -> a[aRowBase + col * aColStride] - b[bRowBase + col * bColStride];
                    case MUL -> a[aRowBase + col * aColStride] * b[bRowBase + col * bColStride];
                    case DIV -> a[aRowBase + col * aColStride] / b[bRowBase + col * bColStride];
                    case MIN -> Math.min(a[aRowBase + col * aColStride], b[bRowBase + col * bColStride]);
                    case MAX -> Math.max(a[aRowBase + col * aColStride], b[bRowBase + col * bColStride]);
                    default -> throw new UnsupportedOperationException("Unsupported rank-2 F64 binary op: " + op.opType());
                };
            }
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

        for (int row = 0; row < rows; row++) {
            int outRowBase = outBaseOffset + row * outRowStride;
            int aRowBase = aBaseOffset + row * aRowStride;
            int bRowBase = bBaseOffset + row * bRowStride;
            for (int col = 0; col < cols; col++) {
                out[outRowBase + col * outColStride] = switch (op.opType()) {
                    case ADD -> a[aRowBase + col * aColStride] + b[bRowBase + col * bColStride];
                    case SUB -> a[aRowBase + col * aColStride] - b[bRowBase + col * bColStride];
                    case MUL -> a[aRowBase + col * aColStride] * b[bRowBase + col * bColStride];
                    case DIV -> a[aRowBase + col * aColStride] / b[bRowBase + col * bColStride];
                    case MIN -> Math.min(a[aRowBase + col * aColStride], b[bRowBase + col * bColStride]);
                    case MAX -> Math.max(a[aRowBase + col * aColStride], b[bRowBase + col * bColStride]);
                    default -> throw new UnsupportedOperationException("Unsupported rank-2 F32 binary op: " + op.opType());
                };
            }
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

        if (outColStride == 1 && aColStride == 0) {
            for (int row = 0; row < rows; row++) {
                int outRowBase = outBaseOffset + row * outRowStride;
                double value = evalUnaryF64(op, a[aBaseOffset + row * aRowStride], useFastExpApprox, useFastTanhApprox);
                fillRowF64(out, outRowBase, cols, value);
            }
            return true;
        }

        for (int row = 0; row < rows; row++) {
            int outRowBase = outBaseOffset + row * outRowStride;
            int aRowBase = aBaseOffset + row * aRowStride;
            for (int col = 0; col < cols; col++) {
                out[outRowBase + col * outColStride] = evalUnaryF64(
                        op,
                        a[aRowBase + col * aColStride],
                        useFastExpApprox,
                        useFastTanhApprox
                );
            }
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

        if (outColStride == 1 && aColStride == 0) {
            for (int row = 0; row < rows; row++) {
                int outRowBase = outBaseOffset + row * outRowStride;
                float value = evalUnaryF32(op, a[aBaseOffset + row * aRowStride], useFastExpApprox, useFastTanhApprox);
                fillRowF32(out, outRowBase, cols, value);
            }
            return true;
        }

        for (int row = 0; row < rows; row++) {
            int outRowBase = outBaseOffset + row * outRowStride;
            int aRowBase = aBaseOffset + row * aRowStride;
            for (int col = 0; col < cols; col++) {
                out[outRowBase + col * outColStride] = evalUnaryF32(
                        op,
                        a[aRowBase + col * aColStride],
                        useFastExpApprox,
                        useFastTanhApprox
                );
            }
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
        for (; col < upper; col += width) {
            DoubleVector leftVector = DoubleVector.fromArray(F64, left, leftBase + col);
            DoubleVector result = switch (op.opType()) {
                case ADD -> leftVector.add(rightVector);
                case SUB -> leftVector.sub(rightVector);
                case MUL -> leftVector.mul(rightVector);
                case DIV -> leftVector.div(rightVector);
                case MIN -> leftVector.min(rightVector);
                case MAX -> leftVector.max(rightVector);
                default -> throw new UnsupportedOperationException("Unsupported vector rank-2 F64 binary op: " + op.opType());
            };
            result.intoArray(out, outBase + col);
        }
        for (; col < cols; col++) {
            out[outBase + col] = switch (op.opType()) {
                case ADD -> left[leftBase + col] + right;
                case SUB -> left[leftBase + col] - right;
                case MUL -> left[leftBase + col] * right;
                case DIV -> left[leftBase + col] / right;
                case MIN -> Math.min(left[leftBase + col], right);
                case MAX -> Math.max(left[leftBase + col], right);
                default -> throw new UnsupportedOperationException("Unsupported scalar rank-2 F64 binary op: " + op.opType());
            };
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
        for (; col < upper; col += width) {
            FloatVector leftVector = FloatVector.fromArray(F32, left, leftBase + col);
            FloatVector result = switch (op.opType()) {
                case ADD -> leftVector.add(rightVector);
                case SUB -> leftVector.sub(rightVector);
                case MUL -> leftVector.mul(rightVector);
                case DIV -> leftVector.div(rightVector);
                case MIN -> leftVector.min(rightVector);
                case MAX -> leftVector.max(rightVector);
                default -> throw new UnsupportedOperationException("Unsupported vector rank-2 F32 binary op: " + op.opType());
            };
            result.intoArray(out, outBase + col);
        }
        for (; col < cols; col++) {
            out[outBase + col] = switch (op.opType()) {
                case ADD -> left[leftBase + col] + right;
                case SUB -> left[leftBase + col] - right;
                case MUL -> left[leftBase + col] * right;
                case DIV -> left[leftBase + col] / right;
                case MIN -> Math.min(left[leftBase + col], right);
                case MAX -> Math.max(left[leftBase + col], right);
                default -> throw new UnsupportedOperationException("Unsupported scalar rank-2 F32 binary op: " + op.opType());
            };
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
        for (; col < upper; col += width) {
            DoubleVector rightVector = DoubleVector.fromArray(F64, right, rightBase + col);
            DoubleVector result = switch (op.opType()) {
                case ADD -> leftVector.add(rightVector);
                case SUB -> leftVector.sub(rightVector);
                case MUL -> leftVector.mul(rightVector);
                case DIV -> leftVector.div(rightVector);
                case MIN -> leftVector.min(rightVector);
                case MAX -> leftVector.max(rightVector);
                default -> throw new UnsupportedOperationException("Unsupported vector rank-2 F64 binary op: " + op.opType());
            };
            result.intoArray(out, outBase + col);
        }
        for (; col < cols; col++) {
            out[outBase + col] = switch (op.opType()) {
                case ADD -> left + right[rightBase + col];
                case SUB -> left - right[rightBase + col];
                case MUL -> left * right[rightBase + col];
                case DIV -> left / right[rightBase + col];
                case MIN -> Math.min(left, right[rightBase + col]);
                case MAX -> Math.max(left, right[rightBase + col]);
                default -> throw new UnsupportedOperationException("Unsupported scalar rank-2 F64 binary op: " + op.opType());
            };
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
        for (; col < upper; col += width) {
            FloatVector rightVector = FloatVector.fromArray(F32, right, rightBase + col);
            FloatVector result = switch (op.opType()) {
                case ADD -> leftVector.add(rightVector);
                case SUB -> leftVector.sub(rightVector);
                case MUL -> leftVector.mul(rightVector);
                case DIV -> leftVector.div(rightVector);
                case MIN -> leftVector.min(rightVector);
                case MAX -> leftVector.max(rightVector);
                default -> throw new UnsupportedOperationException("Unsupported vector rank-2 F32 binary op: " + op.opType());
            };
            result.intoArray(out, outBase + col);
        }
        for (; col < cols; col++) {
            out[outBase + col] = switch (op.opType()) {
                case ADD -> left + right[rightBase + col];
                case SUB -> left - right[rightBase + col];
                case MUL -> left * right[rightBase + col];
                case DIV -> left / right[rightBase + col];
                case MIN -> Math.min(left, right[rightBase + col]);
                case MAX -> Math.max(left, right[rightBase + col]);
                default -> throw new UnsupportedOperationException("Unsupported scalar rank-2 F32 binary op: " + op.opType());
            };
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

    private static double evalUnaryF64(
            Operation op,
            double value,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        return switch (op.opType()) {
            case NEG -> -value;
            case INV -> 1.0 / value;
            case LOG -> Math.log(value);
            case EXP -> useFastExpApprox ? FastExp.fastExpF64(value) : Math.exp(value);
            case FAST_EXP -> FastExp.fastExpF64(value);
            case TANH -> useFastTanhApprox ? FastExp.fastTanhF64(value) : Math.tanh(value);
            case FAST_TANH -> FastExp.fastTanhF64(value);
            case SQRT -> Math.sqrt(value);
            case ABS -> Math.abs(value);
            case RELU -> Math.max(0.0, value);
            case CLAMP_MIN -> Math.max(((clampMin) op).getMinValue(), value);
            case CLAMP_MAX -> Math.min(((clampMax) op).getMaxValue(), value);
            case SIGMOID -> 1.0 / (1.0 + Math.exp(-value));
            case MUL_SCALAR -> value * ((mulScalar) op).getScalar();
            case POW -> {
                double exponent = ((pow) op).getExponent();
                if (exponent == 0.0) yield 1.0;
                if (exponent == 1.0) yield value;
                if (exponent == 2.0) yield value * value;
                yield Math.pow(value, exponent);
            }
            default -> throw new UnsupportedOperationException("Unsupported rank-2 F64 unary op: " + op.opType());
        };
    }

    private static float evalUnaryF32(
            Operation op,
            float value,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        return switch (op.opType()) {
            case NEG -> -value;
            case INV -> 1.0f / value;
            case LOG -> (float) Math.log(value);
            case EXP -> useFastExpApprox ? FastExp.fastExpF32(value) : (float) Math.exp(value);
            case FAST_EXP -> FastExp.fastExpF32(value);
            case TANH -> useFastTanhApprox ? FastExp.fastTanhF32(value) : (float) Math.tanh(value);
            case FAST_TANH -> FastExp.fastTanhF32(value);
            case SQRT -> (float) Math.sqrt(value);
            case ABS -> Math.abs(value);
            case RELU -> Math.max(0.0f, value);
            case CLAMP_MIN -> Math.max(((clampMin) op).getMinValueF32(), value);
            case CLAMP_MAX -> Math.min(((clampMax) op).getMaxValueF32(), value);
            case SIGMOID -> (float) (1.0 / (1.0 + Math.exp(-value)));
            case MUL_SCALAR -> value * ((mulScalar) op).getScalarF32();
            case POW -> {
                float exponent = ((pow) op).getExponentF32();
                if (exponent == 0.0f) yield 1.0f;
                if (exponent == 1.0f) yield value;
                if (exponent == 2.0f) yield value * value;
                yield (float) Math.pow(value, exponent);
            }
            default -> throw new UnsupportedOperationException("Unsupported rank-2 F32 unary op: " + op.opType());
        };
    }

    private static int remapIndex(int flatOut, int[] denseStrides, int[] targetStrides, int rank, int baseOffset) {
        int idx = flatOut;
        int targetFlat = baseOffset;
        for (int d = 0; d < rank; d++) {
            int coord = idx / denseStrides[d];
            idx %= denseStrides[d];
            targetFlat += coord * targetStrides[d];
        }
        return targetFlat;
    }

    private static byte evalCompare(Operation op, double left, double right) {
        boolean value = switch (op.opType()) {
            case GT -> left > right;
            case GE -> left >= right;
            case LT -> left < right;
            case LE -> left <= right;
            case EQ -> left == right;
            case NE -> left != right;
            default -> throw new UnsupportedOperationException("Unsupported compare strided opType=" + op.opType());
        };
        return value ? (byte) 1 : (byte) 0;
    }

    private static byte evalBoolBinary(Operation op, byte left, byte right) {
        boolean l = left != 0;
        boolean r = right != 0;
        boolean value = switch (op.opType()) {
            case LOGICAL_AND -> l && r;
            case LOGICAL_OR -> l || r;
            default -> throw new UnsupportedOperationException("Unsupported bool strided opType=" + op.opType());
        };
        return value ? (byte) 1 : (byte) 0;
    }

    private static byte evalBoolUnary(Operation op, byte value) {
        boolean v = value != 0;
        boolean out = switch (op.opType()) {
            case LOGICAL_NOT -> !v;
            default -> throw new UnsupportedOperationException("Unsupported bool unary strided opType=" + op.opType());
        };
        return out ? (byte) 1 : (byte) 0;
    }

    private static double eval(
            Operation op,
            double[] a,
            double[] b,
            int aIdx,
            int bIdx,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        return switch (op.opType()) {
            case ADD -> a[aIdx] + b[bIdx];
            case SUB -> a[aIdx] - b[bIdx];
            case MUL -> a[aIdx] * b[bIdx];
            case DIV -> a[aIdx] / b[bIdx];
            case MIN -> Math.min(a[aIdx], b[bIdx]);
            case MAX -> Math.max(a[aIdx], b[bIdx]);
            case NEG -> -a[aIdx];
            case INV -> 1.0 / a[aIdx];
            case LOG -> Math.log(a[aIdx]);
            case EXP -> useFastExpApprox ? FastExp.fastExpF64(a[aIdx]) : Math.exp(a[aIdx]);
            case FAST_EXP -> FastExp.fastExpF64(a[aIdx]);
            case TANH -> useFastTanhApprox ? FastExp.fastTanhF64(a[aIdx]) : Math.tanh(a[aIdx]);
            case FAST_TANH -> FastExp.fastTanhF64(a[aIdx]);
            case SQRT -> Math.sqrt(a[aIdx]);
            case ABS -> Math.abs(a[aIdx]);
            case RELU -> Math.max(0.0, a[aIdx]);
            case CLAMP_MIN -> Math.max(((clampMin) op).getMinValue(), a[aIdx]);
            case CLAMP_MAX -> Math.min(((clampMax) op).getMaxValue(), a[aIdx]);
            case SIGMOID -> 1.0 / (1.0 + Math.exp(-a[aIdx]));
            case MUL_SCALAR -> a[aIdx] * ((mulScalar) op).getScalar();
            case POW -> {
                double exponent = ((pow) op).getExponent();
                double v = a[aIdx];
                if (exponent == 0.0) yield 1.0;
                if (exponent == 1.0) yield v;
                if (exponent == 2.0) yield v * v;
                yield Math.pow(v, exponent);
            }
            default -> throw new UnsupportedOperationException("Unsupported strided opType=" + op.opType());
        };
    }

    private static float evalF32(
            Operation op,
            float[] a,
            float[] b,
            int aIdx,
            int bIdx,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        return switch (op.opType()) {
            case ADD -> a[aIdx] + b[bIdx];
            case SUB -> a[aIdx] - b[bIdx];
            case MUL -> a[aIdx] * b[bIdx];
            case DIV -> a[aIdx] / b[bIdx];
            case MIN -> Math.min(a[aIdx], b[bIdx]);
            case MAX -> Math.max(a[aIdx], b[bIdx]);
            case NEG -> -a[aIdx];
            case INV -> 1.0f / a[aIdx];
            case LOG -> (float) Math.log(a[aIdx]);
            case EXP -> useFastExpApprox ? FastExp.fastExpF32(a[aIdx]) : (float) Math.exp(a[aIdx]);
            case FAST_EXP -> FastExp.fastExpF32(a[aIdx]);
            case TANH -> useFastTanhApprox ? FastExp.fastTanhF32(a[aIdx]) : (float) Math.tanh(a[aIdx]);
            case FAST_TANH -> FastExp.fastTanhF32(a[aIdx]);
            case SQRT -> (float) Math.sqrt(a[aIdx]);
            case ABS -> Math.abs(a[aIdx]);
            case RELU -> Math.max(0.0f, a[aIdx]);
            case CLAMP_MIN -> Math.max(((clampMin) op).getMinValueF32(), a[aIdx]);
            case CLAMP_MAX -> Math.min(((clampMax) op).getMaxValueF32(), a[aIdx]);
            case SIGMOID -> (float) (1.0 / (1.0 + Math.exp(-a[aIdx])));
            case MUL_SCALAR -> a[aIdx] * ((mulScalar) op).getScalarF32();
            case POW -> {
                float exponent = ((pow) op).getExponentF32();
                float v = a[aIdx];
                if (exponent == 0.0f) yield 1.0f;
                if (exponent == 1.0f) yield v;
                if (exponent == 2.0f) yield v * v;
                yield (float) Math.pow(v, exponent);
            }
            default -> throw new UnsupportedOperationException("Unsupported strided opType=" + op.opType());
        };
    }

    private static short evalF16(
            Operation op,
            short[] a,
            short[] b,
            int aIdx,
            int bIdx,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        float av = a != null ? CpuDTypeOps.fromBFloat16Bits(a[aIdx]) : 0.0f;
        float bv = b != null ? CpuDTypeOps.fromBFloat16Bits(b[bIdx]) : 0.0f;
        float value = switch (op.opType()) {
            case ADD -> av + bv;
            case SUB -> av - bv;
            case MUL -> av * bv;
            case DIV -> av / bv;
            case MIN -> Math.min(av, bv);
            case MAX -> Math.max(av, bv);
            case NEG -> -av;
            case INV -> 1.0f / av;
            case LOG -> (float) Math.log(av);
            case EXP -> useFastExpApprox ? FastExp.fastExpF32(av) : (float) Math.exp(av);
            case FAST_EXP -> FastExp.fastExpF32(av);
            case TANH -> useFastTanhApprox ? FastExp.fastTanhF32(av) : (float) Math.tanh(av);
            case FAST_TANH -> FastExp.fastTanhF32(av);
            case SQRT -> (float) Math.sqrt(av);
            case ABS -> Math.abs(av);
            case RELU -> Math.max(0.0f, av);
            case CLAMP_MIN -> Math.max(((clampMin) op).getMinValueF32(), av);
            case CLAMP_MAX -> Math.min(((clampMax) op).getMaxValueF32(), av);
            case SIGMOID -> (float) (1.0 / (1.0 + Math.exp(-av)));
            case MUL_SCALAR -> av * ((mulScalar) op).getScalarF32();
            case POW -> {
                float exponent = ((pow) op).getExponentF32();
                if (exponent == 0.0f) yield 1.0f;
                if (exponent == 1.0f) yield av;
                if (exponent == 2.0f) yield av * av;
                yield (float) Math.pow(av, exponent);
            }
            default -> throw new UnsupportedOperationException("Unsupported strided opType=" + op.opType());
        };
        return CpuDTypeOps.toBFloat16Bits(value);
    }
}
