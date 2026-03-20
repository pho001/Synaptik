package Graph.codegen;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

import java.util.function.DoubleUnaryOperator;

public final class FusedVectorOps {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final DoubleVector ZERO = DoubleVector.zero(SPECIES);
    private static final DoubleVector ONE = DoubleVector.broadcast(SPECIES, 1.0);

    private FusedVectorOps() {}

    public static int width() {
        return SPECIES.length();
    }

    public static Object fromArray(double[] data, int index) {
        return DoubleVector.fromArray(SPECIES, data, index);
    }

    public static void intoArray(Object vector, double[] out, int index) {
        ((DoubleVector) vector).intoArray(out, index);
    }

    public static Object add(Object a, Object b) {
        return ((DoubleVector) a).add((DoubleVector) b);
    }

    public static Object add(Object a, Object b, int mode) {
        if (mode == FusedDTypeOps.MODE_F64) {
            return add(a, b);
        }
        return mapBinary(a, b, mode, BinaryOp.ADD);
    }

    public static Object sub(Object a, Object b) {
        return ((DoubleVector) a).sub((DoubleVector) b);
    }

    public static Object sub(Object a, Object b, int mode) {
        if (mode == FusedDTypeOps.MODE_F64) {
            return sub(a, b);
        }
        return mapBinary(a, b, mode, BinaryOp.SUB);
    }

    public static Object mul(Object a, Object b) {
        return ((DoubleVector) a).mul((DoubleVector) b);
    }

    public static Object mul(Object a, Object b, int mode) {
        if (mode == FusedDTypeOps.MODE_F64) {
            return mul(a, b);
        }
        return mapBinary(a, b, mode, BinaryOp.MUL);
    }

    public static Object div(Object a, Object b) {
        return ((DoubleVector) a).div((DoubleVector) b);
    }

    public static Object div(Object a, Object b, int mode) {
        if (mode == FusedDTypeOps.MODE_F64) {
            return div(a, b);
        }
        return mapBinary(a, b, mode, BinaryOp.DIV);
    }

    public static Object neg(Object a) {
        return ((DoubleVector) a).neg();
    }

    public static Object neg(Object a, int mode) {
        if (mode == FusedDTypeOps.MODE_F64) {
            return neg(a);
        }
        return mapUnary(a, x -> FusedDTypeOps.neg(x, mode));
    }

    public static Object inv(Object a) {
        return ONE.div((DoubleVector) a);
    }

    public static Object inv(Object a, int mode) {
        if (mode == FusedDTypeOps.MODE_F64) {
            return inv(a);
        }
        return mapUnary(a, x -> FusedDTypeOps.inv(x, mode));
    }

    public static Object mulScalar(Object a, double scalar) {
        return ((DoubleVector) a).mul(scalar);
    }

    public static Object mulScalar(Object a, double scalar, int mode) {
        if (mode == FusedDTypeOps.MODE_F64) {
            return mulScalar(a, scalar);
        }
        return mapUnary(a, x -> FusedDTypeOps.mulScalar(x, scalar, mode));
    }

    public static Object relu(Object a) {
        return ((DoubleVector) a).max(ZERO);
    }

    public static Object relu(Object a, int mode) {
        if (mode == FusedDTypeOps.MODE_F64) {
            return relu(a);
        }
        return mapUnary(a, x -> FusedDTypeOps.relu(x, mode));
    }

    public static Object noop(Object a) {
        return a;
    }

    public static Object noop(Object a, int mode) {
        if (mode == FusedDTypeOps.MODE_F64) {
            return noop(a);
        }
        return mapUnary(a, x -> FusedDTypeOps.noop(x, mode));
    }

    public static Object sqrt(Object a) {
        return mapUnary(a, Math::sqrt);
    }

    public static Object sqrt(Object a, int mode) {
        return mapUnary(a, x -> FusedDTypeOps.sqrt(x, mode));
    }

    public static Object exp(Object a) {
        return mapUnary(a, Math::exp);
    }

    public static Object exp(Object a, int mode) {
        return mapUnary(a, x -> FusedDTypeOps.exp(x, mode));
    }

    public static Object log(Object a) {
        return mapUnary(a, Math::log);
    }

    public static Object log(Object a, int mode) {
        return mapUnary(a, x -> FusedDTypeOps.log(x, mode));
    }

    public static Object tanh(Object a) {
        return mapUnary(a, Math::tanh);
    }

    public static Object tanh(Object a, int mode) {
        return mapUnary(a, x -> FusedDTypeOps.tanh(x, mode));
    }

    public static Object sigmoid(Object a) {
        return mapUnary(a, x -> 1.0 / (1.0 + Math.exp(-x)));
    }

    public static Object sigmoid(Object a, int mode) {
        return mapUnary(a, x -> FusedDTypeOps.sigmoid(x, mode));
    }

    public static Object pow(Object a, double exponent) {
        return mapUnary(a, x -> Math.pow(x, exponent));
    }

    public static Object pow(Object a, double exponent, int mode) {
        return mapUnary(a, x -> FusedDTypeOps.pow(x, exponent, mode));
    }

    private static Object mapUnary(Object vector, DoubleUnaryOperator fn) {
        DoubleVector v = (DoubleVector) vector;
        double[] buf = new double[SPECIES.length()];
        v.intoArray(buf, 0);
        for (int i = 0; i < buf.length; i++) {
            buf[i] = fn.applyAsDouble(buf[i]);
        }
        return DoubleVector.fromArray(SPECIES, buf, 0);
    }

    private enum BinaryOp { ADD, SUB, MUL, DIV }

    private static Object mapBinary(Object left, Object right, int mode, BinaryOp op) {
        double[] a = new double[SPECIES.length()];
        double[] b = new double[SPECIES.length()];
        ((DoubleVector) left).intoArray(a, 0);
        ((DoubleVector) right).intoArray(b, 0);
        for (int i = 0; i < a.length; i++) {
            a[i] = switch (op) {
                case ADD -> FusedDTypeOps.add(a[i], b[i], mode);
                case SUB -> FusedDTypeOps.sub(a[i], b[i], mode);
                case MUL -> FusedDTypeOps.mul(a[i], b[i], mode);
                case DIV -> FusedDTypeOps.div(a[i], b[i], mode);
            };
        }
        return DoubleVector.fromArray(SPECIES, a, 0);
    }
}
