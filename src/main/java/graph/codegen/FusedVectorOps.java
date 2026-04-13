package graph.codegen;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import utils.FastExp;

import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

public final class FusedVectorOps {
    private static final VectorSpecies<Double> DOUBLE_SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;

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

    public static Object add(Object a, Object b, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> ((DoubleVector) a).add((DoubleVector) b);
            case FusedDTypeOps.MODE_F32 -> ((FloatVector) a).add((FloatVector) b);
            case FusedDTypeOps.MODE_BF16 -> mapBinaryD(a, b, (x, y) -> FusedDTypeOps.add(x, y, mode));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector addF64(DoubleVector a, DoubleVector b) { return a.add(b); }
    public static FloatVector addF32(FloatVector a, FloatVector b) { return a.add(b); }

    public static Object sub(Object a, Object b, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> ((DoubleVector) a).sub((DoubleVector) b);
            case FusedDTypeOps.MODE_F32 -> ((FloatVector) a).sub((FloatVector) b);
            case FusedDTypeOps.MODE_BF16 -> mapBinaryD(a, b, (x, y) -> FusedDTypeOps.sub(x, y, mode));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector subF64(DoubleVector a, DoubleVector b) { return a.sub(b); }
    public static FloatVector subF32(FloatVector a, FloatVector b) { return a.sub(b); }

    public static Object mul(Object a, Object b, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> ((DoubleVector) a).mul((DoubleVector) b);
            case FusedDTypeOps.MODE_F32 -> ((FloatVector) a).mul((FloatVector) b);
            case FusedDTypeOps.MODE_BF16 -> mapBinaryD(a, b, (x, y) -> FusedDTypeOps.mul(x, y, mode));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector mulF64(DoubleVector a, DoubleVector b) { return a.mul(b); }
    public static FloatVector mulF32(FloatVector a, FloatVector b) { return a.mul(b); }

    public static Object div(Object a, Object b, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> ((DoubleVector) a).div((DoubleVector) b);
            case FusedDTypeOps.MODE_F32 -> ((FloatVector) a).div((FloatVector) b);
            case FusedDTypeOps.MODE_BF16 -> mapBinaryD(a, b, (x, y) -> FusedDTypeOps.div(x, y, mode));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector divF64(DoubleVector a, DoubleVector b) { return a.div(b); }
    public static FloatVector divF32(FloatVector a, FloatVector b) { return a.div(b); }

    public static Object min(Object a, Object b, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> ((DoubleVector) a).min((DoubleVector) b);
            case FusedDTypeOps.MODE_F32 -> ((FloatVector) a).min((FloatVector) b);
            case FusedDTypeOps.MODE_BF16 -> mapBinaryD(a, b, (x, y) -> FusedDTypeOps.min(x, y, mode));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector minF64(DoubleVector a, DoubleVector b) { return a.min(b); }
    public static FloatVector minF32(FloatVector a, FloatVector b) { return a.min(b); }

    public static Object max(Object a, Object b, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> ((DoubleVector) a).max((DoubleVector) b);
            case FusedDTypeOps.MODE_F32 -> ((FloatVector) a).max((FloatVector) b);
            case FusedDTypeOps.MODE_BF16 -> mapBinaryD(a, b, (x, y) -> FusedDTypeOps.max(x, y, mode));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector maxF64(DoubleVector a, DoubleVector b) { return a.max(b); }
    public static FloatVector maxF32(FloatVector a, FloatVector b) { return a.max(b); }

    public static Object neg(Object a, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> ((DoubleVector) a).neg();
            case FusedDTypeOps.MODE_F32 -> ((FloatVector) a).neg();
            case FusedDTypeOps.MODE_BF16 -> mapUnaryD(a, x -> FusedDTypeOps.neg(x, mode));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector negF64(DoubleVector a) { return a.neg(); }
    public static FloatVector negF32(FloatVector a) { return a.neg(); }

    public static Object inv(Object a, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> doubleOne((DoubleVector) a).div((DoubleVector) a);
            case FusedDTypeOps.MODE_F32 -> floatOne((FloatVector) a).div((FloatVector) a);
            case FusedDTypeOps.MODE_BF16 -> mapUnaryD(a, x -> FusedDTypeOps.inv(x, mode));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector invF64(DoubleVector a) { return doubleOne(a).div(a); }
    public static FloatVector invF32(FloatVector a) { return floatOne(a).div(a); }

    public static Object mulScalar(Object a, double scalar, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> ((DoubleVector) a).mul(scalar);
            case FusedDTypeOps.MODE_F32 -> ((FloatVector) a).mul((float) scalar);
            case FusedDTypeOps.MODE_BF16 -> mapUnaryD(a, x -> FusedDTypeOps.mulScalar(x, scalar, mode));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector mulScalarF64(DoubleVector a, double scalar) { return a.mul(scalar); }
    public static FloatVector mulScalarF32(FloatVector a, float scalar) { return a.mul(scalar); }

    public static Object relu(Object a, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> ((DoubleVector) a).max(doubleZero((DoubleVector) a));
            case FusedDTypeOps.MODE_F32 -> ((FloatVector) a).max(floatZero((FloatVector) a));
            case FusedDTypeOps.MODE_BF16 -> mapUnaryD(a, x -> FusedDTypeOps.relu(x, mode));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector reluF64(DoubleVector a) { return a.max(doubleZero(a)); }
    public static FloatVector reluF32(FloatVector a) { return a.max(floatZero(a)); }

    public static Object abs(Object a, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> ((DoubleVector) a).abs();
            case FusedDTypeOps.MODE_F32 -> ((FloatVector) a).abs();
            case FusedDTypeOps.MODE_BF16 -> mapUnaryD(a, x -> FusedDTypeOps.abs(x, mode));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector absF64(DoubleVector a) { return a.abs(); }
    public static FloatVector absF32(FloatVector a) { return a.abs(); }

    public static Object clampMin(Object a, double minValue, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> ((DoubleVector) a).max(minValue);
            case FusedDTypeOps.MODE_F32 -> ((FloatVector) a).max((float) minValue);
            case FusedDTypeOps.MODE_BF16 -> mapUnaryD(a, x -> FusedDTypeOps.clampMin(x, minValue, mode));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector clampMinF64(DoubleVector a, double minValue) { return a.max(minValue); }
    public static FloatVector clampMinF32(FloatVector a, float minValue) { return a.max(minValue); }

    public static Object clampMax(Object a, double maxValue, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> ((DoubleVector) a).min(maxValue);
            case FusedDTypeOps.MODE_F32 -> ((FloatVector) a).min((float) maxValue);
            case FusedDTypeOps.MODE_BF16 -> mapUnaryD(a, x -> FusedDTypeOps.clampMax(x, maxValue, mode));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector clampMaxF64(DoubleVector a, double maxValue) { return a.min(maxValue); }
    public static FloatVector clampMaxF32(FloatVector a, float maxValue) { return a.min(maxValue); }

    public static Object noop(Object a, int mode) {
        return a;
    }

    public static DoubleVector noopF64(DoubleVector a) { return a; }
    public static FloatVector noopF32(FloatVector a) { return a; }

    public static Object sqrt(Object a, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> mapUnaryD(a, Math::sqrt);
            case FusedDTypeOps.MODE_F32 -> mapUnaryF(a, x -> (float) Math.sqrt(x));
            case FusedDTypeOps.MODE_BF16 -> mapUnaryD(a, x -> FusedDTypeOps.sqrt(x, mode));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector sqrtF64(DoubleVector a) { return (DoubleVector) mapUnaryD(a, Math::sqrt); }
    public static FloatVector sqrtF32(FloatVector a) { return (FloatVector) mapUnaryF(a, x -> (float) Math.sqrt(x)); }

    public static Object exp(Object a, int mode) {
        return exp(a, mode, false);
    }

    public static Object exp(Object a, int mode, boolean useFastExpApprox) {
        if (useFastExpApprox) {
            return fastExp(a, mode);
        }
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> mapUnaryD(a, Math::exp);
            case FusedDTypeOps.MODE_F32 -> mapUnaryF(a, x -> (float) Math.exp(x));
            case FusedDTypeOps.MODE_BF16 -> mapUnaryD(a, x -> FusedDTypeOps.exp(x, mode));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector expF64(DoubleVector a) { return expF64(a, false); }
    public static DoubleVector expF64(DoubleVector a, boolean useFastExpApprox) {
        if (useFastExpApprox) {
            return fastExpF64(a);
        }
        return (DoubleVector) mapUnaryD(a, Math::exp);
    }
    public static FloatVector expF32(FloatVector a) { return expF32(a, false); }
    public static FloatVector expF32(FloatVector a, boolean useFastExpApprox) {
        if (useFastExpApprox) {
            return fastExpF32(a);
        }
        return (FloatVector) mapUnaryF(a, x -> (float) Math.exp(x));
    }

    public static Object fastExp(Object a, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> mapUnaryD(a, FastExp::fastExpF64);
            case FusedDTypeOps.MODE_F32 -> mapUnaryF(a, FastExp::fastExpF32);
            case FusedDTypeOps.MODE_BF16 -> mapUnaryD(a, x -> FusedDTypeOps.fastExp(x, mode));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector fastExpF64(DoubleVector a) { return (DoubleVector) mapUnaryD(a, FastExp::fastExpF64); }
    public static FloatVector fastExpF32(FloatVector a) { return (FloatVector) mapUnaryF(a, FastExp::fastExpF32); }

    public static Object log(Object a, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> mapUnaryD(a, Math::log);
            case FusedDTypeOps.MODE_F32 -> mapUnaryF(a, x -> (float) Math.log(x));
            case FusedDTypeOps.MODE_BF16 -> mapUnaryD(a, x -> FusedDTypeOps.log(x, mode));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector logF64(DoubleVector a) { return (DoubleVector) mapUnaryD(a, Math::log); }
    public static FloatVector logF32(FloatVector a) { return (FloatVector) mapUnaryF(a, x -> (float) Math.log(x)); }

    public static Object tanh(Object a, int mode) {
        return tanh(a, mode, false);
    }

    public static Object tanh(Object a, int mode, boolean useFastTanhApprox) {
        if (useFastTanhApprox) {
            return fastTanh(a, mode);
        }
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> mapUnaryD(a, Math::tanh);
            case FusedDTypeOps.MODE_F32 -> mapUnaryF(a, x -> (float) Math.tanh(x));
            case FusedDTypeOps.MODE_BF16 -> mapUnaryD(a, x -> FusedDTypeOps.tanh(x, mode));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector tanhF64(DoubleVector a) { return tanhF64(a, false); }
    public static DoubleVector tanhF64(DoubleVector a, boolean useFastTanhApprox) {
        if (useFastTanhApprox) {
            return fastTanhF64(a);
        }
        return (DoubleVector) mapUnaryD(a, Math::tanh);
    }
    public static FloatVector tanhF32(FloatVector a) { return tanhF32(a, false); }
    public static FloatVector tanhF32(FloatVector a, boolean useFastTanhApprox) {
        if (useFastTanhApprox) {
            return fastTanhF32(a);
        }
        return (FloatVector) mapUnaryF(a, x -> (float) Math.tanh(x));
    }

    public static Object fastTanh(Object a, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> mapUnaryD(a, FastExp::fastTanhF64);
            case FusedDTypeOps.MODE_F32 -> mapUnaryF(a, FastExp::fastTanhF32);
            case FusedDTypeOps.MODE_BF16 -> mapUnaryD(a, x -> FusedDTypeOps.fastTanh(x, mode));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector fastTanhF64(DoubleVector a) { return (DoubleVector) mapUnaryD(a, FastExp::fastTanhF64); }
    public static FloatVector fastTanhF32(FloatVector a) { return (FloatVector) mapUnaryF(a, FastExp::fastTanhF32); }

    public static Object sigmoid(Object a, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> mapUnaryD(a, x -> 1.0 / (1.0 + Math.exp(-x)));
            case FusedDTypeOps.MODE_F32 -> mapUnaryF(a, x -> (float) (1.0 / (1.0 + Math.exp(-x))));
            case FusedDTypeOps.MODE_BF16 -> mapUnaryD(a, x -> FusedDTypeOps.sigmoid(x, mode));
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector sigmoidF64(DoubleVector a) { return (DoubleVector) mapUnaryD(a, x -> 1.0 / (1.0 + Math.exp(-x))); }
    public static FloatVector sigmoidF32(FloatVector a) { return (FloatVector) mapUnaryF(a, x -> (float) (1.0 / (1.0 + Math.exp(-x)))); }

    public static Object pow(Object a, double exponent, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> powF64((DoubleVector) a, exponent);
            case FusedDTypeOps.MODE_F32 -> powF32((FloatVector) a, (float) exponent);
            case FusedDTypeOps.MODE_BF16 -> powBF16(a, exponent);
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static DoubleVector powF64(DoubleVector a, double exponent) {
        if (exponent == 0.0d) {
            return doubleOne(a);
        }
        if (exponent == 1.0d) {
            return a;
        }
        if (exponent == 2.0d) {
            return a.mul(a);
        }
        if (exponent == 0.5d) {
            return a.lanewise(VectorOperators.SQRT);
        }
        if (exponent == -1.0d) {
            return doubleOne(a).div(a);
        }
        return (DoubleVector) mapUnaryD(a, x -> Math.pow(x, exponent));
    }

    public static FloatVector powF32(FloatVector a, float exponent) {
        if (exponent == 0.0f) {
            return floatOne(a);
        }
        if (exponent == 1.0f) {
            return a;
        }
        if (exponent == 2.0f) {
            return a.mul(a);
        }
        if (exponent == 0.5f) {
            return a.lanewise(VectorOperators.SQRT);
        }
        if (exponent == -1.0f) {
            return floatOne(a).div(a);
        }
        return (FloatVector) mapUnaryF(a, x -> (float) Math.pow(x, exponent));
    }

    private static Object powBF16(Object a, double exponent) {
        if (exponent == 0.0d) {
            return mapUnaryD(a, x -> FusedDTypeOps.cast(1.0d, FusedDTypeOps.MODE_BF16));
        }
        if (exponent == 1.0d) {
            return a;
        }
        if (exponent == 2.0d) {
            return mapUnaryD(a, x -> FusedDTypeOps.cast(x * x, FusedDTypeOps.MODE_BF16));
        }
        if (exponent == 0.5d) {
            return mapUnaryD(a, x -> FusedDTypeOps.cast(Math.sqrt(x), FusedDTypeOps.MODE_BF16));
        }
        if (exponent == -1.0d) {
            return mapUnaryD(a, x -> FusedDTypeOps.cast(1.0d / x, FusedDTypeOps.MODE_BF16));
        }
        return mapUnaryD(a, x -> FusedDTypeOps.pow(x, exponent, FusedDTypeOps.MODE_BF16));
    }

    public static Object gt(Object a, Object b, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> ((DoubleVector) a).compare(VectorOperators.GT, (DoubleVector) b);
            case FusedDTypeOps.MODE_F32 -> ((FloatVector) a).compare(VectorOperators.GT, (FloatVector) b);
            default -> throw new IllegalArgumentException("Vector compare GT is supported only for F32/F64.");
        };
    }

    public static Object ge(Object a, Object b, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> ((DoubleVector) a).compare(VectorOperators.GE, (DoubleVector) b);
            case FusedDTypeOps.MODE_F32 -> ((FloatVector) a).compare(VectorOperators.GE, (FloatVector) b);
            default -> throw new IllegalArgumentException("Vector compare GE is supported only for F32/F64.");
        };
    }

    public static Object lt(Object a, Object b, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> ((DoubleVector) a).compare(VectorOperators.LT, (DoubleVector) b);
            case FusedDTypeOps.MODE_F32 -> ((FloatVector) a).compare(VectorOperators.LT, (FloatVector) b);
            default -> throw new IllegalArgumentException("Vector compare LT is supported only for F32/F64.");
        };
    }

    public static Object le(Object a, Object b, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> ((DoubleVector) a).compare(VectorOperators.LE, (DoubleVector) b);
            case FusedDTypeOps.MODE_F32 -> ((FloatVector) a).compare(VectorOperators.LE, (FloatVector) b);
            default -> throw new IllegalArgumentException("Vector compare LE is supported only for F32/F64.");
        };
    }

    public static Object eq(Object a, Object b, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> ((DoubleVector) a).compare(VectorOperators.EQ, (DoubleVector) b);
            case FusedDTypeOps.MODE_F32 -> ((FloatVector) a).compare(VectorOperators.EQ, (FloatVector) b);
            default -> throw new IllegalArgumentException("Vector compare EQ is supported only for F32/F64.");
        };
    }

    public static Object ne(Object a, Object b, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> ((DoubleVector) a).compare(VectorOperators.NE, (DoubleVector) b);
            case FusedDTypeOps.MODE_F32 -> ((FloatVector) a).compare(VectorOperators.NE, (FloatVector) b);
            default -> throw new IllegalArgumentException("Vector compare NE is supported only for F32/F64.");
        };
    }

    public static Object logicalAnd(Object a, Object b, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> ((VectorMask<Double>) a).and((VectorMask<Double>) b);
            case FusedDTypeOps.MODE_F32 -> ((VectorMask<Float>) a).and((VectorMask<Float>) b);
            default -> throw new IllegalArgumentException("Vector logicalAnd is supported only for F32/F64.");
        };
    }

    public static Object logicalOr(Object a, Object b, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> ((VectorMask<Double>) a).or((VectorMask<Double>) b);
            case FusedDTypeOps.MODE_F32 -> ((VectorMask<Float>) a).or((VectorMask<Float>) b);
            default -> throw new IllegalArgumentException("Vector logicalOr is supported only for F32/F64.");
        };
    }

    public static Object logicalNot(Object a, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> ((VectorMask<Double>) a).not();
            case FusedDTypeOps.MODE_F32 -> ((VectorMask<Float>) a).not();
            default -> throw new IllegalArgumentException("Vector logicalNot is supported only for F32/F64.");
        };
    }

    public static Object where(Object condition, Object ifTrue, Object ifFalse, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> ((DoubleVector) ifFalse).blend((DoubleVector) ifTrue, (VectorMask<Double>) condition);
            case FusedDTypeOps.MODE_F32 -> ((FloatVector) ifFalse).blend((FloatVector) ifTrue, (VectorMask<Float>) condition);
            default -> throw new IllegalArgumentException("Vector where is supported only for F32/F64.");
        };
    }

    private static Object mapUnaryD(Object vector, DoubleUnaryOperator fn) {
        DoubleVector v = (DoubleVector) vector;
        VectorSpecies<Double> species = v.species();
        double[] buf = new double[species.length()];
        v.intoArray(buf, 0);
        for (int i = 0; i < buf.length; i++) {
            buf[i] = fn.applyAsDouble(buf[i]);
        }
        return DoubleVector.fromArray(species, buf, 0);
    }

    private static Object mapUnaryF(Object vector, FloatUnaryOperator fn) {
        FloatVector v = (FloatVector) vector;
        VectorSpecies<Float> species = v.species();
        float[] buf = new float[species.length()];
        v.intoArray(buf, 0);
        for (int i = 0; i < buf.length; i++) {
            buf[i] = fn.applyAsFloat(buf[i]);
        }
        return FloatVector.fromArray(species, buf, 0);
    }

    private static Object mapBinaryD(Object left, Object right, DoubleBinaryOperator fn) {
        DoubleVector lv = (DoubleVector) left;
        DoubleVector rv = (DoubleVector) right;
        VectorSpecies<Double> species = lv.species();
        double[] a = new double[species.length()];
        double[] b = new double[species.length()];
        lv.intoArray(a, 0);
        rv.intoArray(b, 0);
        for (int i = 0; i < a.length; i++) {
            a[i] = fn.applyAsDouble(a[i], b[i]);
        }
        return DoubleVector.fromArray(species, a, 0);
    }

    private static Object mapBinaryF(Object left, Object right, FloatBinaryOperator fn) {
        FloatVector lv = (FloatVector) left;
        FloatVector rv = (FloatVector) right;
        VectorSpecies<Float> species = lv.species();
        float[] a = new float[species.length()];
        float[] b = new float[species.length()];
        lv.intoArray(a, 0);
        rv.intoArray(b, 0);
        for (int i = 0; i < a.length; i++) {
            a[i] = fn.applyAsFloat(a[i], b[i]);
        }
        return FloatVector.fromArray(species, a, 0);
    }

    private static DoubleVector doubleZero(DoubleVector vector) {
        return DoubleVector.zero(vector.species());
    }

    private static DoubleVector doubleOne(DoubleVector vector) {
        return DoubleVector.broadcast(vector.species(), 1.0d);
    }

    private static FloatVector floatZero(FloatVector vector) {
        return FloatVector.zero(vector.species());
    }

    private static FloatVector floatOne(FloatVector vector) {
        return FloatVector.broadcast(vector.species(), 1.0f);
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
