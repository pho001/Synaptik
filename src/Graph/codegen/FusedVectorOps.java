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

    public static Object sub(Object a, Object b) {
        return ((DoubleVector) a).sub((DoubleVector) b);
    }

    public static Object mul(Object a, Object b) {
        return ((DoubleVector) a).mul((DoubleVector) b);
    }

    public static Object div(Object a, Object b) {
        return ((DoubleVector) a).div((DoubleVector) b);
    }

    public static Object neg(Object a) {
        return ((DoubleVector) a).neg();
    }

    public static Object inv(Object a) {
        return ONE.div((DoubleVector) a);
    }

    public static Object mulScalar(Object a, double scalar) {
        return ((DoubleVector) a).mul(scalar);
    }

    public static Object relu(Object a) {
        return ((DoubleVector) a).max(ZERO);
    }

    public static Object noop(Object a) {
        return a;
    }

    public static Object sqrt(Object a) {
        return mapUnary(a, Math::sqrt);
    }

    public static Object exp(Object a) {
        return mapUnary(a, Math::exp);
    }

    public static Object log(Object a) {
        return mapUnary(a, Math::log);
    }

    public static Object tanh(Object a) {
        return mapUnary(a, Math::tanh);
    }

    public static Object sigmoid(Object a) {
        return mapUnary(a, x -> 1.0 / (1.0 + Math.exp(-x)));
    }

    public static Object pow(Object a, double exponent) {
        return mapUnary(a, x -> Math.pow(x, exponent));
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
}
