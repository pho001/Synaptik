package backend.cpu.fused.runtime;

import backend.cpu.kernels.CpuDTypeOps;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import utils.FastTranscendentals;

/**
 * Internal vector arithmetic entrypoints invoked by generated fused kernels.
 */
public final class FusedVectorOps {
    private static final VectorSpecies<Double> DOUBLE_SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;

    private FusedVectorOps() {}

    public static int widthF64() {
        return DOUBLE_SPECIES.length();
    }

    public static int widthF32() {
        return FLOAT_SPECIES.length();
    }

    public static DoubleVector addF64(DoubleVector a, DoubleVector b) { return a.add(b); }
    public static FloatVector addF32(FloatVector a, FloatVector b) { return a.add(b); }
    public static FloatVector addBF16(FloatVector a, FloatVector b) { return mapBinaryBF16(a, b, (x, y) -> x + y); }

    public static DoubleVector subF64(DoubleVector a, DoubleVector b) { return a.sub(b); }
    public static FloatVector subF32(FloatVector a, FloatVector b) { return a.sub(b); }
    public static FloatVector subBF16(FloatVector a, FloatVector b) { return mapBinaryBF16(a, b, (x, y) -> x - y); }

    public static DoubleVector mulF64(DoubleVector a, DoubleVector b) { return a.mul(b); }
    public static FloatVector mulF32(FloatVector a, FloatVector b) { return a.mul(b); }
    public static FloatVector mulBF16(FloatVector a, FloatVector b) { return mapBinaryBF16(a, b, (x, y) -> x * y); }

    public static DoubleVector divF64(DoubleVector a, DoubleVector b) { return a.div(b); }
    public static FloatVector divF32(FloatVector a, FloatVector b) { return a.div(b); }
    public static FloatVector divBF16(FloatVector a, FloatVector b) { return mapBinaryBF16(a, b, (x, y) -> x / y); }

    public static DoubleVector minF64(DoubleVector a, DoubleVector b) { return a.min(b); }
    public static FloatVector minF32(FloatVector a, FloatVector b) { return a.min(b); }
    public static FloatVector minBF16(FloatVector a, FloatVector b) { return mapBinaryBF16(a, b, Math::min); }

    public static DoubleVector maxF64(DoubleVector a, DoubleVector b) { return a.max(b); }
    public static FloatVector maxF32(FloatVector a, FloatVector b) { return a.max(b); }
    public static FloatVector maxBF16(FloatVector a, FloatVector b) { return mapBinaryBF16(a, b, Math::max); }

    public static DoubleVector negF64(DoubleVector a) { return a.neg(); }
    public static FloatVector negF32(FloatVector a) { return a.neg(); }
    public static FloatVector negBF16(FloatVector a) { return mapUnaryBF16(a, x -> -x); }

    public static DoubleVector invF64(DoubleVector a) { return doubleOne(a).div(a); }
    public static FloatVector invF32(FloatVector a) { return floatOne(a).div(a); }
    public static FloatVector invBF16(FloatVector a) { return mapUnaryBF16(a, x -> 1.0d / x); }

    public static DoubleVector mulScalarF64(DoubleVector a, double scalar) { return a.mul(scalar); }
    public static FloatVector mulScalarF32(FloatVector a, float scalar) { return a.mul(scalar); }
    public static FloatVector mulScalarBF16(FloatVector a, double scalar) {
        float bf16Scalar = quantizeBF16(scalar);
        return mapUnaryBF16(a, x -> x * bf16Scalar);
    }

    public static DoubleVector constantF64(double value) {
        return DoubleVector.broadcast(DOUBLE_SPECIES, value);
    }

    public static FloatVector constantF32(float value) {
        return FloatVector.broadcast(FLOAT_SPECIES, value);
    }

    public static FloatVector constantBF16(double value) {
        return FloatVector.broadcast(FLOAT_SPECIES, quantizeBF16(value));
    }

    public static DoubleVector reluF64(DoubleVector a) { return a.max(doubleZero(a)); }
    public static FloatVector reluF32(FloatVector a) { return a.max(floatZero(a)); }
    public static FloatVector reluBF16(FloatVector a) { return mapUnaryBF16(a, x -> Math.max(x, 0.0d)); }

    public static DoubleVector absF64(DoubleVector a) { return a.abs(); }
    public static FloatVector absF32(FloatVector a) { return a.abs(); }
    public static FloatVector absBF16(FloatVector a) { return mapUnaryBF16(a, Math::abs); }

    public static DoubleVector clampMinF64(DoubleVector a, double minValue) { return a.max(minValue); }
    public static FloatVector clampMinF32(FloatVector a, float minValue) { return a.max(minValue); }
    public static FloatVector clampMinBF16(FloatVector a, double minValue) {
        float bf16Min = quantizeBF16(minValue);
        return mapUnaryBF16(a, x -> Math.max(x, bf16Min));
    }

    public static DoubleVector clampMaxF64(DoubleVector a, double maxValue) { return a.min(maxValue); }
    public static FloatVector clampMaxF32(FloatVector a, float maxValue) { return a.min(maxValue); }
    public static FloatVector clampMaxBF16(FloatVector a, double maxValue) {
        float bf16Max = quantizeBF16(maxValue);
        return mapUnaryBF16(a, x -> Math.min(x, bf16Max));
    }

    public static DoubleVector noopF64(DoubleVector a) { return a; }
    public static FloatVector noopF32(FloatVector a) { return a; }
    public static FloatVector noopBF16(FloatVector a) { return a; }

    public static DoubleVector sqrtF64(DoubleVector a) { return mapUnaryF64(a, Math::sqrt); }
    public static FloatVector sqrtF32(FloatVector a) { return mapUnaryF32(a, x -> (float) Math.sqrt(x)); }
    public static FloatVector sqrtBF16(FloatVector a) { return mapUnaryBF16(a, Math::sqrt); }

    public static DoubleVector expF64(DoubleVector a) {
        return mapUnaryF64(a, Math::exp);
    }

    public static FloatVector expF32(FloatVector a) {
        return mapUnaryF32(a, x -> (float) Math.exp(x));
    }

    public static FloatVector expBF16(FloatVector a) {
        return mapUnaryBF16(a, Math::exp);
    }

    public static DoubleVector fastExpF64(DoubleVector a) { return mapUnaryF64(a, FastTranscendentals::fastExpF64); }
    public static FloatVector fastExpF32(FloatVector a) { return mapUnaryF32(a, FastTranscendentals::fastExpF32); }
    public static FloatVector fastExpBF16(FloatVector a) {
        return mapUnaryBF16(a, x -> FastTranscendentals.fastExpF32((float) x));
    }

    public static DoubleVector logF64(DoubleVector a) { return mapUnaryF64(a, Math::log); }
    public static FloatVector logF32(FloatVector a) { return mapUnaryF32(a, x -> (float) Math.log(x)); }
    public static FloatVector logBF16(FloatVector a) { return mapUnaryBF16(a, Math::log); }

    public static DoubleVector tanhF64(DoubleVector a) {
        return mapUnaryF64(a, Math::tanh);
    }

    public static FloatVector tanhF32(FloatVector a) {
        return mapUnaryF32(a, x -> (float) Math.tanh(x));
    }

    public static FloatVector tanhBF16(FloatVector a) {
        return mapUnaryBF16(a, Math::tanh);
    }

    public static DoubleVector fastTanhF64(DoubleVector a) { return mapUnaryF64(a, FastTranscendentals::fastTanhF64); }
    public static FloatVector fastTanhF32(FloatVector a) { return mapUnaryF32(a, FastTranscendentals::fastTanhF32); }
    public static FloatVector fastTanhBF16(FloatVector a) {
        return mapUnaryBF16(a, x -> FastTranscendentals.fastTanhF32((float) x));
    }

    public static DoubleVector sigmoidF64(DoubleVector a) {
        return mapUnaryF64(a, x -> 1.0d / (1.0d + Math.exp(-x)));
    }

    public static FloatVector sigmoidF32(FloatVector a) {
        return mapUnaryF32(a, x -> (float) (1.0d / (1.0d + Math.exp(-x))));
    }

    public static FloatVector sigmoidBF16(FloatVector a) {
        return mapUnaryBF16(a, x -> 1.0d / (1.0d + Math.exp(-x)));
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
        return mapUnaryF64(a, x -> Math.pow(x, exponent));
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
        return mapUnaryF32(a, x -> (float) Math.pow(x, exponent));
    }

    public static FloatVector powBF16(FloatVector a, double exponent) {
        if (exponent == 0.0d) {
            return mapUnaryBF16(a, ignored -> 1.0d);
        }
        if (exponent == 1.0d) {
            return a;
        }
        if (exponent == 2.0d) {
            return mapUnaryBF16(a, x -> x * x);
        }
        if (exponent == 0.5d) {
            return mapUnaryBF16(a, Math::sqrt);
        }
        if (exponent == -1.0d) {
            return mapUnaryBF16(a, x -> 1.0d / x);
        }
        return mapUnaryBF16(a, x -> Math.pow(x, exponent));
    }

    public static VectorMask<Double> gtF64(DoubleVector a, DoubleVector b) { return a.compare(VectorOperators.GT, b); }
    public static VectorMask<Float> gtF32(FloatVector a, FloatVector b) { return a.compare(VectorOperators.GT, b); }
    public static VectorMask<Float> gtBF16(FloatVector a, FloatVector b) { return a.compare(VectorOperators.GT, b); }

    public static VectorMask<Double> geF64(DoubleVector a, DoubleVector b) { return a.compare(VectorOperators.GE, b); }
    public static VectorMask<Float> geF32(FloatVector a, FloatVector b) { return a.compare(VectorOperators.GE, b); }
    public static VectorMask<Float> geBF16(FloatVector a, FloatVector b) { return a.compare(VectorOperators.GE, b); }

    public static VectorMask<Double> ltF64(DoubleVector a, DoubleVector b) { return a.compare(VectorOperators.LT, b); }
    public static VectorMask<Float> ltF32(FloatVector a, FloatVector b) { return a.compare(VectorOperators.LT, b); }
    public static VectorMask<Float> ltBF16(FloatVector a, FloatVector b) { return a.compare(VectorOperators.LT, b); }

    public static VectorMask<Double> leF64(DoubleVector a, DoubleVector b) { return a.compare(VectorOperators.LE, b); }
    public static VectorMask<Float> leF32(FloatVector a, FloatVector b) { return a.compare(VectorOperators.LE, b); }
    public static VectorMask<Float> leBF16(FloatVector a, FloatVector b) { return a.compare(VectorOperators.LE, b); }

    public static VectorMask<Double> eqF64(DoubleVector a, DoubleVector b) { return a.compare(VectorOperators.EQ, b); }
    public static VectorMask<Float> eqF32(FloatVector a, FloatVector b) { return a.compare(VectorOperators.EQ, b); }
    public static VectorMask<Float> eqBF16(FloatVector a, FloatVector b) { return a.compare(VectorOperators.EQ, b); }

    public static VectorMask<Double> neF64(DoubleVector a, DoubleVector b) { return a.compare(VectorOperators.NE, b); }
    public static VectorMask<Float> neF32(FloatVector a, FloatVector b) { return a.compare(VectorOperators.NE, b); }
    public static VectorMask<Float> neBF16(FloatVector a, FloatVector b) { return a.compare(VectorOperators.NE, b); }

    public static VectorMask<Double> logicalAndF64(VectorMask<Double> a, VectorMask<Double> b) { return a.and(b); }
    public static VectorMask<Float> logicalAndF32(VectorMask<Float> a, VectorMask<Float> b) { return a.and(b); }
    public static VectorMask<Float> logicalAndBF16(VectorMask<Float> a, VectorMask<Float> b) { return a.and(b); }

    public static VectorMask<Double> logicalOrF64(VectorMask<Double> a, VectorMask<Double> b) { return a.or(b); }
    public static VectorMask<Float> logicalOrF32(VectorMask<Float> a, VectorMask<Float> b) { return a.or(b); }
    public static VectorMask<Float> logicalOrBF16(VectorMask<Float> a, VectorMask<Float> b) { return a.or(b); }

    public static VectorMask<Double> logicalNotF64(VectorMask<Double> a) { return a.not(); }
    public static VectorMask<Float> logicalNotF32(VectorMask<Float> a) { return a.not(); }
    public static VectorMask<Float> logicalNotBF16(VectorMask<Float> a) { return a.not(); }

    public static DoubleVector whereF64(VectorMask<Double> condition, DoubleVector ifTrue, DoubleVector ifFalse) {
        return ifFalse.blend(ifTrue, condition);
    }

    public static FloatVector whereF32(VectorMask<Float> condition, FloatVector ifTrue, FloatVector ifFalse) {
        return ifFalse.blend(ifTrue, condition);
    }

    public static FloatVector whereBF16(VectorMask<Float> condition, FloatVector ifTrue, FloatVector ifFalse) {
        return ifFalse.blend(ifTrue, condition);
    }

    private static DoubleVector mapUnaryF64(DoubleVector vector, DoubleUnaryOperator fn) {
        VectorSpecies<Double> species = vector.species();
        double[] lanes = new double[species.length()];
        vector.intoArray(lanes, 0);
        for (int i = 0; i < lanes.length; i++) {
            lanes[i] = fn.applyAsDouble(lanes[i]);
        }
        return DoubleVector.fromArray(species, lanes, 0);
    }

    private static FloatVector mapUnaryF32(FloatVector vector, FloatUnaryOperator fn) {
        VectorSpecies<Float> species = vector.species();
        float[] lanes = new float[species.length()];
        vector.intoArray(lanes, 0);
        for (int i = 0; i < lanes.length; i++) {
            lanes[i] = fn.applyAsFloat(lanes[i]);
        }
        return FloatVector.fromArray(species, lanes, 0);
    }

    private static FloatVector mapUnaryBF16(FloatVector vector, DoubleUnaryOperator fn) {
        VectorSpecies<Float> species = vector.species();
        float[] lanes = new float[species.length()];
        vector.intoArray(lanes, 0);
        for (int i = 0; i < lanes.length; i++) {
            lanes[i] = quantizeBF16(fn.applyAsDouble(lanes[i]));
        }
        return FloatVector.fromArray(species, lanes, 0);
    }

    private static FloatVector mapBinaryBF16(FloatVector left, FloatVector right, DoubleBinaryOperator fn) {
        VectorSpecies<Float> species = left.species();
        float[] a = new float[species.length()];
        float[] b = new float[species.length()];
        left.intoArray(a, 0);
        right.intoArray(b, 0);
        for (int i = 0; i < a.length; i++) {
            a[i] = quantizeBF16(fn.applyAsDouble(a[i], b[i]));
        }
        return FloatVector.fromArray(species, a, 0);
    }

    private static float quantizeBF16(double value) {
        return CpuDTypeOps.fromBFloat16Bits(CpuDTypeOps.toBFloat16Bits((float) value));
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
    private interface DoubleUnaryOperator {
        double applyAsDouble(double value);
    }

    @FunctionalInterface
    private interface DoubleBinaryOperator {
        double applyAsDouble(double left, double right);
    }

    @FunctionalInterface
    private interface FloatUnaryOperator {
        float applyAsFloat(float value);
    }
}
