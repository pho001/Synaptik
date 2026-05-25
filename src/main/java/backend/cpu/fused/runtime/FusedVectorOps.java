package backend.cpu.fused.runtime;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Allocation-free Vector API primitives invoked by generated fused kernels.
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

    public static DoubleVector subF64(DoubleVector a, DoubleVector b) { return a.sub(b); }
    public static FloatVector subF32(FloatVector a, FloatVector b) { return a.sub(b); }

    public static DoubleVector mulF64(DoubleVector a, DoubleVector b) { return a.mul(b); }
    public static FloatVector mulF32(FloatVector a, FloatVector b) { return a.mul(b); }

    public static DoubleVector divF64(DoubleVector a, DoubleVector b) { return a.div(b); }
    public static FloatVector divF32(FloatVector a, FloatVector b) { return a.div(b); }

    public static DoubleVector minF64(DoubleVector a, DoubleVector b) { return a.min(b); }
    public static FloatVector minF32(FloatVector a, FloatVector b) { return a.min(b); }

    public static DoubleVector maxF64(DoubleVector a, DoubleVector b) { return a.max(b); }
    public static FloatVector maxF32(FloatVector a, FloatVector b) { return a.max(b); }

    public static DoubleVector negF64(DoubleVector a) { return a.neg(); }
    public static FloatVector negF32(FloatVector a) { return a.neg(); }

    public static DoubleVector reciprocalF64(DoubleVector a) { return DoubleVector.broadcast(a.species(), 1.0d).div(a); }
    public static FloatVector reciprocalF32(FloatVector a) { return FloatVector.broadcast(a.species(), 1.0f).div(a); }
    public static DoubleVector reciprocalSquareF64(DoubleVector a) { return reciprocalF64(squareF64(a)); }
    public static FloatVector reciprocalSquareF32(FloatVector a) { return reciprocalF32(squareF32(a)); }

    public static DoubleVector mulScalarF64(DoubleVector a, double scalar) { return a.mul(scalar); }
    public static FloatVector mulScalarF32(FloatVector a, float scalar) { return a.mul(scalar); }

    public static DoubleVector constantF64(int width, double value) {
        return DoubleVector.broadcast(FusedVectorSpecies.f64(width), value);
    }

    public static FloatVector constantF32(int width, float value) {
        return FloatVector.broadcast(FusedVectorSpecies.f32(width), value);
    }

    public static DoubleVector reluF64(DoubleVector a) { return a.max(0.0d); }
    public static FloatVector reluF32(FloatVector a) { return a.max(0.0f); }

    public static DoubleVector absF64(DoubleVector a) { return a.abs(); }
    public static FloatVector absF32(FloatVector a) { return a.abs(); }

    public static DoubleVector clampMinF64(DoubleVector a, double minValue) { return a.max(minValue); }
    public static FloatVector clampMinF32(FloatVector a, float minValue) { return a.max(minValue); }

    public static DoubleVector clampMaxF64(DoubleVector a, double maxValue) { return a.min(maxValue); }
    public static FloatVector clampMaxF32(FloatVector a, float maxValue) { return a.min(maxValue); }

    public static DoubleVector noopF64(DoubleVector a) { return a; }
    public static FloatVector noopF32(FloatVector a) { return a; }

    public static DoubleVector squareF64(DoubleVector a) { return a.mul(a); }
    public static FloatVector squareF32(FloatVector a) { return a.mul(a); }

    public static DoubleVector sqrtF64(DoubleVector a) { return a.lanewise(VectorOperators.SQRT); }
    public static FloatVector sqrtF32(FloatVector a) { return a.lanewise(VectorOperators.SQRT); }

    public static VectorMask<Double> gtF64(DoubleVector a, DoubleVector b) { return a.compare(VectorOperators.GT, b); }
    public static VectorMask<Float> gtF32(FloatVector a, FloatVector b) { return a.compare(VectorOperators.GT, b); }

    public static VectorMask<Double> geF64(DoubleVector a, DoubleVector b) { return a.compare(VectorOperators.GE, b); }
    public static VectorMask<Float> geF32(FloatVector a, FloatVector b) { return a.compare(VectorOperators.GE, b); }

    public static VectorMask<Double> ltF64(DoubleVector a, DoubleVector b) { return a.compare(VectorOperators.LT, b); }
    public static VectorMask<Float> ltF32(FloatVector a, FloatVector b) { return a.compare(VectorOperators.LT, b); }

    public static VectorMask<Double> leF64(DoubleVector a, DoubleVector b) { return a.compare(VectorOperators.LE, b); }
    public static VectorMask<Float> leF32(FloatVector a, FloatVector b) { return a.compare(VectorOperators.LE, b); }

    public static VectorMask<Double> eqF64(DoubleVector a, DoubleVector b) { return a.compare(VectorOperators.EQ, b); }
    public static VectorMask<Float> eqF32(FloatVector a, FloatVector b) { return a.compare(VectorOperators.EQ, b); }

    public static VectorMask<Double> neF64(DoubleVector a, DoubleVector b) { return a.compare(VectorOperators.NE, b); }
    public static VectorMask<Float> neF32(FloatVector a, FloatVector b) { return a.compare(VectorOperators.NE, b); }

    public static VectorMask<Double> logicalAndF64(VectorMask<Double> a, VectorMask<Double> b) { return a.and(b); }
    public static VectorMask<Float> logicalAndF32(VectorMask<Float> a, VectorMask<Float> b) { return a.and(b); }

    public static VectorMask<Double> logicalOrF64(VectorMask<Double> a, VectorMask<Double> b) { return a.or(b); }
    public static VectorMask<Float> logicalOrF32(VectorMask<Float> a, VectorMask<Float> b) { return a.or(b); }

    public static VectorMask<Double> logicalNotF64(VectorMask<Double> a) { return a.not(); }
    public static VectorMask<Float> logicalNotF32(VectorMask<Float> a) { return a.not(); }

    public static DoubleVector whereF64(VectorMask<Double> condition, DoubleVector ifTrue, DoubleVector ifFalse) {
        return ifFalse.blend(ifTrue, condition);
    }

    public static FloatVector whereF32(VectorMask<Float> condition, FloatVector ifTrue, FloatVector ifFalse) {
        return ifFalse.blend(ifTrue, condition);
    }
}
