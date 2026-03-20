package Backend.kernels.cpu;

import Operations.Operation;
import Operations.mulScalar;
import Operations.pow;
import Tensor.Tensor;

import java.util.List;

public final class CpuStridedElementWise {
    private CpuStridedElementWise() {}

    public static boolean supports(Operation op) {
        if (op == null) return false;
        return switch (op.opType()) {
            case ADD, SUB, MUL, DIV, NEG, INV, LOG, EXP, TANH, POW, SQRT, MUL_SCALAR, RELU, SIGMOID -> true;
            default -> false;
        };
    }

    public static void forward(Operation op, List<Tensor> inputs, Tensor node) {
        if (op == null) {
            return;
        }
        double[] out = node.getData();
        if (out == null) {
            return;
        }

        int[] outShape = node.getShape();
        int[] outStrides = node.getStrides();
        int rank = outShape.length;

        double[] a = null;
        double[] b = null;
        int[] aStrides = null;
        int[] bStrides = null;
        if (!inputs.isEmpty()) {
            Tensor ta = inputs.get(0);
            a = ta.getData();
            aStrides = ta.getStrides();
        }
        if (inputs.size() > 1) {
            Tensor tb = inputs.get(1);
            b = tb.getData();
            bStrides = tb.getStrides();
        }

        if (rank == 1) {
            forwardRank1(op, a, b, aStrides, bStrides, out);
            return;
        }

        for (int i = 0; i < out.length; i++) {
            int aIdx = a != null ? remapIndex(i, outStrides, aStrides, rank) : -1;
            int bIdx = b != null ? remapIndex(i, outStrides, bStrides, rank) : -1;
            out[i] = eval(op, a, b, aIdx, bIdx);
        }
    }

    private static void forwardRank1(
            Operation op,
            double[] a,
            double[] b,
            int[] aStrides,
            int[] bStrides,
            double[] out
    ) {
        int strideA = a != null ? aStrides[0] : 0;
        int strideB = b != null ? bStrides[0] : 0;
        for (int i = 0; i < out.length; i++) {
            int aIdx = a != null ? i * strideA : -1;
            int bIdx = b != null ? i * strideB : -1;
            out[i] = eval(op, a, b, aIdx, bIdx);
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

    private static double eval(Operation op, double[] a, double[] b, int aIdx, int bIdx) {
        return switch (op.opType()) {
            case ADD -> a[aIdx] + b[bIdx];
            case SUB -> a[aIdx] - b[bIdx];
            case MUL -> a[aIdx] * b[bIdx];
            case DIV -> a[aIdx] / b[bIdx];
            case NEG -> -a[aIdx];
            case INV -> 1.0 / a[aIdx];
            case LOG -> Math.log(a[aIdx]);
            case EXP -> Math.exp(a[aIdx]);
            case TANH -> Math.tanh(a[aIdx]);
            case SQRT -> Math.sqrt(a[aIdx]);
            case RELU -> Math.max(0.0, a[aIdx]);
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
}
