package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Pure preferred-species vector mathematics used by generated exact/default CPU entries.
 *
 * <p>The overloads preserve the selected lane type throughout evaluation. The binary32
 * coefficient tables are the exact round-to-nearest binary32 values derived from the retained
 * binary64 tables; no runtime narrowing or route selection occurs here.</p>
 */
final class CpuVectorMath {
    private static final VectorSpecies<Double> DOUBLE_SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;

    // Cephes erf/erfc rational coefficients, documented at
    // https://netlib.org/cephes/doubldoc.html and published in netlib cephes ndtr.c.
    private static final double[] DOUBLE_ERF_T = {9.60497373987051638749E0,
            9.00260197203842689217E1, 2.23200534594684319226E3,
            7.00332514112805075473E3, 5.55923013010394962768E4};
    private static final double[] DOUBLE_ERF_U = {3.35617141647503099647E1,
            5.21357949780152679795E2, 4.59432382970980127987E3,
            2.26290000613890934246E4, 4.92673942608635921086E4};
    private static final double[] DOUBLE_ERFC_P = {2.46196981473530512524E-10,
            5.64189564831068821977E-1, 7.46321056442269912687E0,
            4.86371970985681366614E1, 1.96520832956077098242E2,
            5.26445194995477358631E2, 9.34528527171957607540E2,
            1.02755188689515710272E3, 5.57535335369399327526E2};
    private static final double[] DOUBLE_ERFC_Q = {1.32281951154744992508E1,
            8.67072140885989742329E1, 3.54937778887819891062E2,
            9.75708501743205489753E2, 1.82390916687909736289E3,
            2.24633760818710981792E3, 1.65666309194161350182E3,
            5.57535340817727675546E2};
    private static final double[] DOUBLE_ERFC_R = {5.64189583547755073984E-1,
            1.27536670759978104416E0, 5.01905042251180477414E0,
            6.16021097993053585195E0, 7.40974269950448939160E0,
            2.97886665372100240670E0};
    private static final double[] DOUBLE_ERFC_S = {2.26052863220117276590E0,
            9.39603524938001434673E0, 1.20489539808096656605E1,
            1.70814450747565897222E1, 9.60896809063285878198E0,
            3.36907645100081516050E0};

    // Exact IEEE-754 binary32 roundings of the corresponding Cephes binary64 tables above;
    // https://netlib.org/cephes/singldoc.html is provenance review evidence, not an erff substitution.
    private static final float[] FLOAT_ERF_T = {0x1.335bf2p3f, 0x1.681aa4p6f,
            0x1.17002cp11f, 0x1.b5b534p12f, 0x1.b2509ap15f};
    private static final float[] FLOAT_ERF_U = {0x1.0c7e64p5f, 0x1.04add2p9f,
            0x1.1f252ep12f, 0x1.6194p14f, 0x1.80e6cap15f};
    private static final float[] FLOAT_ERFC_P = {0x1.0eb24ap-32f, 0x1.20dd74p-1f,
            0x1.dda53ep2f, 0x1.8518fap5f, 0x1.890aaap7f, 0x1.0738fcp9f,
            0x1.d343a6p9f, 0x1.00e352p10f, 0x1.16c486p9f};
    private static final float[] FLOAT_ERFC_Q = {0x1.a74d6p3f, 0x1.5ad43p6f,
            0x1.62f012p8f, 0x1.e7dabp9f, 0x1.c7fa3p10f, 0x1.18cacep11f,
            0x1.9e2a7p10f, 0x1.16c486p9f};
    private static final float[] FLOAT_ERFC_R = {0x1.20dd76p-1f, 0x1.467e6ep0f,
            0x1.41382p2f, 0x1.8a40e6p2f, 0x1.da393ap2f, 0x1.7d4b8p1f};
    private static final float[] FLOAT_ERFC_S = {0x1.2159p1f, 0x1.2cac52p3f,
            0x1.819108p3f, 0x1.114d9ap4f, 0x1.337caap3f, 0x1.af3de6p1f};

    private CpuVectorMath() { }

    /** Returns binary64 lane-wise absolute values.
     * @param value non-null input vector
     * @return non-null lane-wise absolute values
     */
    static DoubleVector abs(DoubleVector value) { return value.abs(); }
    /** Returns binary32 lane-wise absolute values.
     * @param value non-null input vector
     * @return non-null lane-wise absolute values
     */
    static FloatVector abs(FloatVector value) { return value.abs(); }

    /** Returns exact binary64 positive one divided by every lane.
     * @param value non-null input vector
     * @return non-null lane-wise reciprocal values
     */
    static DoubleVector reciprocal(DoubleVector value) { return positiveOne().div(value); }
    /** Returns exact binary32 positive one divided by every lane.
     * @param value non-null input vector
     * @return non-null lane-wise reciprocal values
     */
    static FloatVector reciprocal(FloatVector value) { return positiveOneFloat().div(value); }

    /** Returns binary64 lane-wise natural logarithms.
     * @param value non-null input vector
     * @return non-null lane-wise natural logarithms
     */
    static DoubleVector log(DoubleVector value) { return value.lanewise(VectorOperators.LOG); }
    /** Returns binary32 lane-wise natural logarithms.
     * @param value non-null input vector
     * @return non-null lane-wise natural logarithms
     */
    static FloatVector log(FloatVector value) { return value.lanewise(VectorOperators.LOG); }
    /** Returns binary64 lane-wise logarithms of one plus the input.
     * @param value non-null input vector
     * @return non-null lane-wise {@code log1p} values
     */
    static DoubleVector log1p(DoubleVector value) { return value.lanewise(VectorOperators.LOG1P); }
    /** Returns binary32 lane-wise logarithms of one plus the input.
     * @param value non-null input vector
     * @return non-null lane-wise {@code log1p} values
     */
    static FloatVector log1p(FloatVector value) { return value.lanewise(VectorOperators.LOG1P); }
    /** Returns binary64 lane-wise natural exponentials.
     * @param value non-null input vector
     * @return non-null lane-wise natural exponentials
     */
    static DoubleVector exp(DoubleVector value) { return value.lanewise(VectorOperators.EXP); }
    /** Returns binary32 lane-wise natural exponentials.
     * @param value non-null input vector
     * @return non-null lane-wise natural exponentials
     */
    static FloatVector exp(FloatVector value) { return value.lanewise(VectorOperators.EXP); }
    /** Returns binary64 lane-wise natural exponentials minus one.
     * @param value non-null input vector
     * @return non-null lane-wise {@code expm1} values
     */
    static DoubleVector expm1(DoubleVector value) { return value.lanewise(VectorOperators.EXPM1); }
    /** Returns binary32 lane-wise natural exponentials minus one.
     * @param value non-null input vector
     * @return non-null lane-wise {@code expm1} values
     */
    static FloatVector expm1(FloatVector value) { return value.lanewise(VectorOperators.EXPM1); }

    /**
     * Evaluates the retained Cephes-family error-function approximation in binary64.
     * @param value non-null input vector
     * @return the lane-wise error function, preserving signed zero and infinity behavior
     */
    static DoubleVector erf(DoubleVector value) {
        DoubleVector x = value.abs();
        DoubleVector z = x.mul(x);
        DoubleVector small = polevl(z, DOUBLE_ERF_T).div(p1evl(z, DOUBLE_ERF_U)).mul(x);
        DoubleVector near = polevl(x, DOUBLE_ERFC_P).div(p1evl(x, DOUBLE_ERFC_Q));
        DoubleVector far = polevl(x, DOUBLE_ERFC_R).div(p1evl(x, DOUBLE_ERFC_S));
        DoubleVector ratio = far.blend(near, x.lt(8.0d));
        DoubleVector large = positiveOne().sub(z.neg().lanewise(VectorOperators.EXP).mul(ratio));
        DoubleVector result = large.blend(small, x.compare(VectorOperators.LE, 1.0d));
        result = result.blend(1.0d, x.eq(Double.POSITIVE_INFINITY));
        return result.blend(result.neg(), value.test(VectorOperators.IS_NEGATIVE));
    }

    /**
     * Evaluates the retained Cephes-family error-function approximation in binary32.
     * @param value non-null input vector
     * @return the lane-wise error function, preserving signed zero and infinity behavior
     */
    static FloatVector erf(FloatVector value) {
        FloatVector x = value.abs();
        FloatVector z = x.mul(x);
        FloatVector small = polevl(z, FLOAT_ERF_T).div(p1evl(z, FLOAT_ERF_U)).mul(x);
        FloatVector near = polevl(x, FLOAT_ERFC_P).div(p1evl(x, FLOAT_ERFC_Q));
        FloatVector far = polevl(x, FLOAT_ERFC_R).div(p1evl(x, FLOAT_ERFC_S));
        FloatVector ratio = far.blend(near, x.lt(8.0f));
        FloatVector large = positiveOneFloat().sub(z.neg().lanewise(VectorOperators.EXP).mul(ratio));
        FloatVector result = large.blend(small, x.compare(VectorOperators.LE, 1.0f));
        result = result.blend(1.0f, x.eq(Float.POSITIVE_INFINITY));
        return result.blend(result.neg(), value.test(VectorOperators.IS_NEGATIVE));
    }

    /** Returns binary64 lane-wise principal square roots.
     * @param value non-null input vector
     * @return non-null lane-wise principal square roots
     */
    static DoubleVector sqrt(DoubleVector value) { return value.lanewise(VectorOperators.SQRT); }
    /** Returns binary32 lane-wise principal square roots.
     * @param value non-null input vector
     * @return non-null lane-wise principal square roots
     */
    static FloatVector sqrt(FloatVector value) { return value.lanewise(VectorOperators.SQRT); }
    /** Returns one divided by each binary64 lane's principal square root.
     * @param value non-null input vector
     * @return non-null lane-wise reciprocal-square-root values
     */
    static DoubleVector rsqrt(DoubleVector value) { return reciprocal(sqrt(value)); }
    /** Returns one divided by each binary32 lane's principal square root.
     * @param value non-null input vector
     * @return non-null lane-wise reciprocal-square-root values
     */
    static FloatVector rsqrt(FloatVector value) { return reciprocal(sqrt(value)); }
    /** Returns binary64 lane-wise signs with both zero signs and NaN preserved.
     * @param value non-null input vector
     * @return exact negative one, the original signed zero or NaN, or exact positive one per lane
     */
    static DoubleVector sign(DoubleVector value) {
        DoubleVector result = positiveOne().blend(-1.0d, value.lt(0.0d));
        return result.blend(value, value.eq(0.0d).or(value.test(VectorOperators.IS_NAN)));
    }
    /** Returns binary32 lane-wise signs with both zero signs and NaN preserved.
     * @param value non-null input vector
     * @return exact negative one, the original signed zero or NaN, or exact positive one per lane
     */
    static FloatVector sign(FloatVector value) {
        FloatVector result = positiveOneFloat().blend(-1.0f, value.lt(0.0f));
        return result.blend(value, value.eq(0.0f).or(value.test(VectorOperators.IS_NAN)));
    }
    /** Returns binary64 lane-wise hyperbolic tangents.
     * @param value non-null input vector
     * @return non-null lane-wise hyperbolic tangents
     */
    static DoubleVector tanh(DoubleVector value) { return value.lanewise(VectorOperators.TANH); }
    /** Returns binary32 lane-wise hyperbolic tangents.
     * @param value non-null input vector
     * @return non-null lane-wise hyperbolic tangents
     */
    static FloatVector tanh(FloatVector value) { return value.lanewise(VectorOperators.TANH); }

    /**
     * Evaluates exact/default GELU in binary64 using the corresponding {@link #erf(DoubleVector)}.
     * @param value non-null vector of pre-activation values
     * @return the lane-wise GELU result, including negative-infinity signed-zero correction
     */
    static DoubleVector gelu(DoubleVector value) {
        DoubleVector result = value.mul(0.5d).mul(erf(value.div(Math.sqrt(2.0d))).add(1.0d));
        return result.blend(-0.0d, value.eq(Double.NEGATIVE_INFINITY));
    }

    /**
     * Evaluates exact/default GELU in binary32 using the corresponding {@link #erf(FloatVector)}.
     * @param value non-null vector of pre-activation values
     * @return the lane-wise GELU result, including negative-infinity signed-zero correction
     */
    static FloatVector gelu(FloatVector value) {
        FloatVector result = value.mul(0.5f).mul(erf(value.div((float) Math.sqrt(2.0d))).add(1.0f));
        return result.blend(-0.0f, value.eq(Float.NEGATIVE_INFINITY));
    }

    /** Produces exact positive one in every preferred binary64 lane.
     * @return a non-null preferred-species binary64 vector filled with exact positive one
     */
    static DoubleVector positiveOne() { return DoubleVector.broadcast(DOUBLE_SPECIES, 1.0d); }
    /** Produces exact positive one in every preferred binary32 lane.
     * @return a non-null preferred-species binary32 vector filled with exact positive one
     */
    static FloatVector positiveOneFloat() { return FloatVector.broadcast(FLOAT_SPECIES, 1.0f); }

    private static DoubleVector polevl(DoubleVector x, double[] coefficients) {
        DoubleVector result = DoubleVector.broadcast(DOUBLE_SPECIES, coefficients[0]);
        for (int i = 1; i < coefficients.length; i++) result = result.mul(x).add(coefficients[i]);
        return result;
    }

    private static DoubleVector p1evl(DoubleVector x, double[] coefficients) {
        DoubleVector result = x.add(coefficients[0]);
        for (int i = 1; i < coefficients.length; i++) result = result.mul(x).add(coefficients[i]);
        return result;
    }

    private static FloatVector polevl(FloatVector x, float[] coefficients) {
        FloatVector result = FloatVector.broadcast(FLOAT_SPECIES, coefficients[0]);
        for (int i = 1; i < coefficients.length; i++) result = result.mul(x).add(coefficients[i]);
        return result;
    }

    private static FloatVector p1evl(FloatVector x, float[] coefficients) {
        FloatVector result = x.add(coefficients[0]);
        for (int i = 1; i < coefficients.length; i++) result = result.mul(x).add(coefficients[i]);
        return result;
    }
}
