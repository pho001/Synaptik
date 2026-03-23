package Graph.codegen;

import Backend.ComputeEngine;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import Utils.FastExp;

import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

public final class FusedVectorOps {
    private static final VectorSpecies<Double> DOUBLE_SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final DoubleVector DOUBLE_ZERO = DoubleVector.zero(DOUBLE_SPECIES);
    private static final DoubleVector DOUBLE_ONE = DoubleVector.broadcast(DOUBLE_SPECIES, 1.0);
    private static final FloatVector FLOAT_ZERO = FloatVector.zero(FLOAT_SPECIES);
    private static final FloatVector FLOAT_ONE = FloatVector.broadcast(FLOAT_SPECIES, 1.0f);

    private FusedVectorOps() {}

    public static int width(int mode) {
        return mode == FusedDTypeOps.MODE_F32 ? FLOAT_SPECIES.length() : DOUBLE_SPECIES.length();
    }

    public static int widthF64() {
        return DOUBLE_SPECIES.length();
    }

    public static int widthF32() {
        return FLOAT_SPECIES.length();
    }

    public static boolean isRecommended(int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F32 -> FLOAT_SPECIES.length() >= 4;
            case FusedDTypeOps.MODE_F64, FusedDTypeOps.MODE_F16 -> DOUBLE_SPECIES.length() >= 4;
            default -> false;
        };
    }

    public static Object add(Object a, Object b, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> ((DoubleVector) a).add((DoubleVector) b);
            case FusedDTypeOps.MODE_F32 -> ((FloatVector) a).add((FloatVector) b);
            case FusedDTypeOps.MODE_F16 -> mapBinaryD(a, b, (x, y) -> FusedDTypeOps.add(x, y, mode));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector addF64(DoubleVector a, DoubleVector b) { return a.add(b); }
    public static FloatVector addF32(FloatVector a, FloatVector b) { return a.add(b); }

    public static Object sub(Object a, Object b, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> ((DoubleVector) a).sub((DoubleVector) b);
            case FusedDTypeOps.MODE_F32 -> ((FloatVector) a).sub((FloatVector) b);
            case FusedDTypeOps.MODE_F16 -> mapBinaryD(a, b, (x, y) -> FusedDTypeOps.sub(x, y, mode));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector subF64(DoubleVector a, DoubleVector b) { return a.sub(b); }
    public static FloatVector subF32(FloatVector a, FloatVector b) { return a.sub(b); }

    public static Object mul(Object a, Object b, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> ((DoubleVector) a).mul((DoubleVector) b);
            case FusedDTypeOps.MODE_F32 -> ((FloatVector) a).mul((FloatVector) b);
            case FusedDTypeOps.MODE_F16 -> mapBinaryD(a, b, (x, y) -> FusedDTypeOps.mul(x, y, mode));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector mulF64(DoubleVector a, DoubleVector b) { return a.mul(b); }
    public static FloatVector mulF32(FloatVector a, FloatVector b) { return a.mul(b); }

    public static Object div(Object a, Object b, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> ((DoubleVector) a).div((DoubleVector) b);
            case FusedDTypeOps.MODE_F32 -> ((FloatVector) a).div((FloatVector) b);
            case FusedDTypeOps.MODE_F16 -> mapBinaryD(a, b, (x, y) -> FusedDTypeOps.div(x, y, mode));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector divF64(DoubleVector a, DoubleVector b) { return a.div(b); }
    public static FloatVector divF32(FloatVector a, FloatVector b) { return a.div(b); }

    public static Object min(Object a, Object b, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> ((DoubleVector) a).min((DoubleVector) b);
            case FusedDTypeOps.MODE_F32 -> ((FloatVector) a).min((FloatVector) b);
            case FusedDTypeOps.MODE_F16 -> mapBinaryD(a, b, (x, y) -> FusedDTypeOps.min(x, y, mode));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector minF64(DoubleVector a, DoubleVector b) { return a.min(b); }
    public static FloatVector minF32(FloatVector a, FloatVector b) { return a.min(b); }

    public static Object max(Object a, Object b, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> ((DoubleVector) a).max((DoubleVector) b);
            case FusedDTypeOps.MODE_F32 -> ((FloatVector) a).max((FloatVector) b);
            case FusedDTypeOps.MODE_F16 -> mapBinaryD(a, b, (x, y) -> FusedDTypeOps.max(x, y, mode));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector maxF64(DoubleVector a, DoubleVector b) { return a.max(b); }
    public static FloatVector maxF32(FloatVector a, FloatVector b) { return a.max(b); }

    public static Object neg(Object a, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> ((DoubleVector) a).neg();
            case FusedDTypeOps.MODE_F32 -> ((FloatVector) a).neg();
            case FusedDTypeOps.MODE_F16 -> mapUnaryD(a, x -> FusedDTypeOps.neg(x, mode));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector negF64(DoubleVector a) { return a.neg(); }
    public static FloatVector negF32(FloatVector a) { return a.neg(); }

    public static Object inv(Object a, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> DOUBLE_ONE.div((DoubleVector) a);
            case FusedDTypeOps.MODE_F32 -> FLOAT_ONE.div((FloatVector) a);
            case FusedDTypeOps.MODE_F16 -> mapUnaryD(a, x -> FusedDTypeOps.inv(x, mode));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector invF64(DoubleVector a) { return DOUBLE_ONE.div(a); }
    public static FloatVector invF32(FloatVector a) { return FLOAT_ONE.div(a); }

    public static Object mulScalar(Object a, double scalar, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> ((DoubleVector) a).mul(scalar);
            case FusedDTypeOps.MODE_F32 -> ((FloatVector) a).mul((float) scalar);
            case FusedDTypeOps.MODE_F16 -> mapUnaryD(a, x -> FusedDTypeOps.mulScalar(x, scalar, mode));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector mulScalarF64(DoubleVector a, double scalar) { return a.mul(scalar); }
    public static FloatVector mulScalarF32(FloatVector a, double scalar) { return a.mul((float) scalar); }

    public static Object relu(Object a, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> ((DoubleVector) a).max(DOUBLE_ZERO);
            case FusedDTypeOps.MODE_F32 -> ((FloatVector) a).max(FLOAT_ZERO);
            case FusedDTypeOps.MODE_F16 -> mapUnaryD(a, x -> FusedDTypeOps.relu(x, mode));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector reluF64(DoubleVector a) { return a.max(DOUBLE_ZERO); }
    public static FloatVector reluF32(FloatVector a) { return a.max(FLOAT_ZERO); }

    public static Object noop(Object a, int mode) {
        return a;
    }

    public static DoubleVector noopF64(DoubleVector a) { return a; }
    public static FloatVector noopF32(FloatVector a) { return a; }

    public static Object sqrt(Object a, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> mapUnaryD(a, Math::sqrt);
            case FusedDTypeOps.MODE_F32 -> mapUnaryF(a, x -> (float) Math.sqrt(x));
            case FusedDTypeOps.MODE_F16 -> mapUnaryD(a, x -> FusedDTypeOps.sqrt(x, mode));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector sqrtF64(DoubleVector a) { return (DoubleVector) mapUnaryD(a, Math::sqrt); }
    public static FloatVector sqrtF32(FloatVector a) { return (FloatVector) mapUnaryF(a, x -> (float) Math.sqrt(x)); }

    public static Object exp(Object a, int mode) {
        if (ComputeEngine.useFastExpApprox()) {
            return fastExp(a, mode);
        }
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> mapUnaryD(a, Math::exp);
            case FusedDTypeOps.MODE_F32 -> mapUnaryF(a, x -> (float) Math.exp(x));
            case FusedDTypeOps.MODE_F16 -> mapUnaryD(a, x -> FusedDTypeOps.exp(x, mode));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector expF64(DoubleVector a) { return (DoubleVector) mapUnaryD(a, Math::exp); }
    public static FloatVector expF32(FloatVector a) {
        if (ComputeEngine.useFastExpApprox()) {
            return fastExpF32(a);
        }
        return (FloatVector) mapUnaryF(a, x -> (float) Math.exp(x));
    }

    public static Object fastExp(Object a, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> mapUnaryD(a, FastExp::fastExpF64);
            case FusedDTypeOps.MODE_F32 -> mapUnaryF(a, FastExp::fastExpF32);
            case FusedDTypeOps.MODE_F16 -> mapUnaryD(a, x -> FusedDTypeOps.fastExp(x, mode));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector fastExpF64(DoubleVector a) { return (DoubleVector) mapUnaryD(a, FastExp::fastExpF64); }
    public static FloatVector fastExpF32(FloatVector a) { return (FloatVector) mapUnaryF(a, FastExp::fastExpF32); }

    public static Object log(Object a, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> mapUnaryD(a, Math::log);
            case FusedDTypeOps.MODE_F32 -> mapUnaryF(a, x -> (float) Math.log(x));
            case FusedDTypeOps.MODE_F16 -> mapUnaryD(a, x -> FusedDTypeOps.log(x, mode));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector logF64(DoubleVector a) { return (DoubleVector) mapUnaryD(a, Math::log); }
    public static FloatVector logF32(FloatVector a) { return (FloatVector) mapUnaryF(a, x -> (float) Math.log(x)); }

    public static Object tanh(Object a, int mode) {
        if (ComputeEngine.useFastTanhApprox()) {
            return fastTanh(a, mode);
        }
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> mapUnaryD(a, Math::tanh);
            case FusedDTypeOps.MODE_F32 -> mapUnaryF(a, x -> (float) Math.tanh(x));
            case FusedDTypeOps.MODE_F16 -> mapUnaryD(a, x -> FusedDTypeOps.tanh(x, mode));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector tanhF64(DoubleVector a) {
        if (ComputeEngine.useFastTanhApprox()) {
            return fastTanhF64(a);
        }
        return (DoubleVector) mapUnaryD(a, Math::tanh);
    }
    public static FloatVector tanhF32(FloatVector a) {
        if (ComputeEngine.useFastTanhApprox()) {
            return fastTanhF32(a);
        }
        return (FloatVector) mapUnaryF(a, x -> (float) Math.tanh(x));
    }

    public static Object fastTanh(Object a, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> mapUnaryD(a, FastExp::fastTanhF64);
            case FusedDTypeOps.MODE_F32 -> mapUnaryF(a, FastExp::fastTanhF32);
            case FusedDTypeOps.MODE_F16 -> mapUnaryD(a, x -> FusedDTypeOps.fastTanh(x, mode));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector fastTanhF64(DoubleVector a) { return (DoubleVector) mapUnaryD(a, FastExp::fastTanhF64); }
    public static FloatVector fastTanhF32(FloatVector a) { return (FloatVector) mapUnaryF(a, FastExp::fastTanhF32); }

    public static Object sigmoid(Object a, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> mapUnaryD(a, x -> 1.0 / (1.0 + Math.exp(-x)));
            case FusedDTypeOps.MODE_F32 -> mapUnaryF(a, x -> (float) (1.0 / (1.0 + Math.exp(-x))));
            case FusedDTypeOps.MODE_F16 -> mapUnaryD(a, x -> FusedDTypeOps.sigmoid(x, mode));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector sigmoidF64(DoubleVector a) { return (DoubleVector) mapUnaryD(a, x -> 1.0 / (1.0 + Math.exp(-x))); }
    public static FloatVector sigmoidF32(FloatVector a) { return (FloatVector) mapUnaryF(a, x -> (float) (1.0 / (1.0 + Math.exp(-x)))); }

    public static Object pow(Object a, double exponent, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> mapUnaryD(a, x -> Math.pow(x, exponent));
            case FusedDTypeOps.MODE_F32 -> mapUnaryF(a, x -> (float) Math.pow(x, exponent));
            case FusedDTypeOps.MODE_F16 -> mapUnaryD(a, x -> FusedDTypeOps.pow(x, exponent, mode));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector powF64(DoubleVector a, double exponent) { return (DoubleVector) mapUnaryD(a, x -> Math.pow(x, exponent)); }
    public static FloatVector powF32(FloatVector a, double exponent) { return (FloatVector) mapUnaryF(a, x -> (float) Math.pow(x, exponent)); }

    private static Object mapUnaryD(Object vector, DoubleUnaryOperator fn) {
        DoubleVector v = (DoubleVector) vector;
        double[] buf = new double[DOUBLE_SPECIES.length()];
        v.intoArray(buf, 0);
        for (int i = 0; i < buf.length; i++) {
            buf[i] = fn.applyAsDouble(buf[i]);
        }
        return DoubleVector.fromArray(DOUBLE_SPECIES, buf, 0);
    }

    private static Object mapUnaryF(Object vector, FloatUnaryOperator fn) {
        FloatVector v = (FloatVector) vector;
        float[] buf = new float[FLOAT_SPECIES.length()];
        v.intoArray(buf, 0);
        for (int i = 0; i < buf.length; i++) {
            buf[i] = fn.applyAsFloat(buf[i]);
        }
        return FloatVector.fromArray(FLOAT_SPECIES, buf, 0);
    }

    private static Object mapBinaryD(Object left, Object right, DoubleBinaryOperator fn) {
        double[] a = new double[DOUBLE_SPECIES.length()];
        double[] b = new double[DOUBLE_SPECIES.length()];
        ((DoubleVector) left).intoArray(a, 0);
        ((DoubleVector) right).intoArray(b, 0);
        for (int i = 0; i < a.length; i++) {
            a[i] = fn.applyAsDouble(a[i], b[i]);
        }
        return DoubleVector.fromArray(DOUBLE_SPECIES, a, 0);
    }

    private static Object mapBinaryF(Object left, Object right, FloatBinaryOperator fn) {
        float[] a = new float[FLOAT_SPECIES.length()];
        float[] b = new float[FLOAT_SPECIES.length()];
        ((FloatVector) left).intoArray(a, 0);
        ((FloatVector) right).intoArray(b, 0);
        for (int i = 0; i < a.length; i++) {
            a[i] = fn.applyAsFloat(a[i], b[i]);
        }
        return FloatVector.fromArray(FLOAT_SPECIES, a, 0);
    }

    @FunctionalInterface
    private interface FloatUnaryOperator {
        float applyAsFloat(float value);
    }

    @FunctionalInterface
    private interface FloatBinaryOperator {
        float applyAsFloat(float left, float right);
    }
}
