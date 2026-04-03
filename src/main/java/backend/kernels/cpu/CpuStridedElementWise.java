package backend.kernels.cpu;

import operations.Operation;
import operations.clampMax;
import operations.clampMin;
import operations.mulScalar;
import operations.pow;
import tensor.Tensor;
import utils.FastExp;

import java.util.List;

public final class CpuStridedElementWise {
    private CpuStridedElementWise() {}

    public static boolean supports(Operation op) {
        if (op == null) return false;
        return switch (op.opType()) {
            case ADD, SUB, MUL, DIV, MIN, MAX, NEG, INV, LOG, EXP, FAST_EXP, TANH, FAST_TANH, POW, SQRT, ABS, MUL_SCALAR, RELU, CLAMP_MIN, CLAMP_MAX, SIGMOID -> true;
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
            case FLOAT16 -> {
                forwardF16(op, inputs, node, useFastExpApprox, useFastTanhApprox);
                return;
            }
            case FLOAT64 -> {
                // continue with existing F64 path below
            }
        }

        double[] out = node.getData();
        if (out == null) {
            return;
        }

        int[] outShape = node.getShapeUnsafe();
        int[] outStrides = node.getStridesUnsafe();
        int rank = outShape.length;

        double[] a = null;
        double[] b = null;
        int[] aStrides = null;
        int[] bStrides = null;
        if (!inputs.isEmpty()) {
            Tensor ta = inputs.get(0);
            a = ta.getData();
            aStrides = ta.getStridesUnsafe();
        }
        if (inputs.size() > 1) {
            Tensor tb = inputs.get(1);
            b = tb.getData();
            bStrides = tb.getStridesUnsafe();
        }

        if (rank == 1) {
            forwardRank1(op, a, b, aStrides, bStrides, out, useFastExpApprox, useFastTanhApprox);
            return;
        }

        for (int i = 0; i < out.length; i++) {
            int aIdx = a != null ? remapIndex(i, outStrides, aStrides, rank) : -1;
            int bIdx = b != null ? remapIndex(i, outStrides, bStrides, rank) : -1;
            out[i] = eval(op, a, b, aIdx, bIdx, useFastExpApprox, useFastTanhApprox);
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

        int[] outShape = node.getShapeUnsafe();
        int[] outStrides = node.getStridesUnsafe();
        int rank = outShape.length;

        float[] a = null;
        float[] b = null;
        int[] aStrides = null;
        int[] bStrides = null;
        if (!inputs.isEmpty()) {
            Tensor ta = inputs.get(0);
            a = ta.getFloat32Data();
            aStrides = ta.getStridesUnsafe();
        }
        if (inputs.size() > 1) {
            Tensor tb = inputs.get(1);
            b = tb.getFloat32Data();
            bStrides = tb.getStridesUnsafe();
        }

        if (rank == 1) {
            int strideA = a != null ? aStrides[0] : 0;
            int strideB = b != null ? bStrides[0] : 0;
            for (int i = 0; i < out.length; i++) {
                int aIdx = a != null ? i * strideA : -1;
                int bIdx = b != null ? i * strideB : -1;
                out[i] = evalF32(op, a, b, aIdx, bIdx, useFastExpApprox, useFastTanhApprox);
            }
            return;
        }

        for (int i = 0; i < out.length; i++) {
            int aIdx = a != null ? remapIndex(i, outStrides, aStrides, rank) : -1;
            int bIdx = b != null ? remapIndex(i, outStrides, bStrides, rank) : -1;
            out[i] = evalF32(op, a, b, aIdx, bIdx, useFastExpApprox, useFastTanhApprox);
        }
    }

    private static void forwardF16(
            Operation op,
            List<Tensor> inputs,
            Tensor node,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        short[] out = node.getFloat16Data();
        if (out == null) {
            return;
        }

        int[] outShape = node.getShapeUnsafe();
        int[] outStrides = node.getStridesUnsafe();
        int rank = outShape.length;

        short[] a = null;
        short[] b = null;
        int[] aStrides = null;
        int[] bStrides = null;
        if (!inputs.isEmpty()) {
            Tensor ta = inputs.get(0);
            a = ta.getFloat16Data();
            aStrides = ta.getStridesUnsafe();
        }
        if (inputs.size() > 1) {
            Tensor tb = inputs.get(1);
            b = tb.getFloat16Data();
            bStrides = tb.getStridesUnsafe();
        }

        if (rank == 1) {
            int strideA = a != null ? aStrides[0] : 0;
            int strideB = b != null ? bStrides[0] : 0;
            for (int i = 0; i < out.length; i++) {
                int aIdx = a != null ? i * strideA : -1;
                int bIdx = b != null ? i * strideB : -1;
                out[i] = evalF16(op, a, b, aIdx, bIdx, useFastExpApprox, useFastTanhApprox);
            }
            return;
        }

        for (int i = 0; i < out.length; i++) {
            int aIdx = a != null ? remapIndex(i, outStrides, aStrides, rank) : -1;
            int bIdx = b != null ? remapIndex(i, outStrides, bStrides, rank) : -1;
            out[i] = evalF16(op, a, b, aIdx, bIdx, useFastExpApprox, useFastTanhApprox);
        }
    }

    private static void forwardRank1(
            Operation op,
            double[] a,
            double[] b,
            int[] aStrides,
            int[] bStrides,
            double[] out,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        int strideA = a != null ? aStrides[0] : 0;
        int strideB = b != null ? bStrides[0] : 0;
        for (int i = 0; i < out.length; i++) {
            int aIdx = a != null ? i * strideA : -1;
            int bIdx = b != null ? i * strideB : -1;
            out[i] = eval(op, a, b, aIdx, bIdx, useFastExpApprox, useFastTanhApprox);
        }
    }

    private static int remapIndex(int flatOut, int[] outStrides, int[] inStrides, int rank) {
        int idx = flatOut;
        int inFlat = 0;
        for (int d = 0; d < rank; d++) {
            int coord = idx / outStrides[d];
            idx %= outStrides[d];
            inFlat += coord * inStrides[d];
        }
        return inFlat;
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
        float av = a != null ? CpuDTypeOps.fromHalfBits(a[aIdx]) : 0.0f;
        float bv = b != null ? CpuDTypeOps.fromHalfBits(b[bIdx]) : 0.0f;
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
        return CpuDTypeOps.toHalfBits(value);
    }
}
