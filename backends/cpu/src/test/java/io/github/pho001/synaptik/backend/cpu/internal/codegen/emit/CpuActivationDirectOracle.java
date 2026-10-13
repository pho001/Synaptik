package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode;
import io.github.pho001.synaptik.model.datatype.DataType;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;

/**
 * Independently authored direct Java oracle for CPU0022. No production reference, emitter,
 * vector helper, or low codec is called. Rational coefficients are the repository's retained
 * numerical data, not newly imported source. Named composite sites are explicit Java floats;
 * the elementary ERF algorithm is independently checked against integral/known-value tests.
 */
final class CpuActivationDirectOracle {
    private CpuActivationDirectOracle() {}

    /** Evaluates the guarded elementary ERF algorithm without a polynomial outside [0,6).
     * @param value input including special values
     * @return double elementary result, retaining zero sign and NaN class
     */
    static double erf(double value) {
        if (Double.isNaN(value) || value == 0) return value;
        double x = Math.abs(value);
        if (x >= 6) return Math.copySign(1, value);
        double result;
        if (x <= 1) {
            double z = x * x;
            double numerator = (((9.60497373987051638749E0 * z
                    + 9.00260197203842689217E1) * z + 2.23200534594684319226E3) * z
                    + 7.00332514112805075473E3) * z + 5.55923013010394962768E4;
            double denominator = ((((z + 3.35617141647503099647E1) * z
                    + 5.21357949780152679795E2) * z + 4.59432382970980127987E3) * z
                    + 2.26290000613890934246E4) * z + 4.92673942608635921086E4;
            result = x * numerator / denominator;
        } else {
            double numerator = (((((((2.46196981473530512524E-10 * x
                    + 5.64189564831068821977E-1) * x + 7.46321056442269912687E0) * x
                    + 4.86371970985681366614E1) * x + 1.96520832956077098242E2) * x
                    + 5.26445194995477358631E2) * x + 9.34528527171957607540E2) * x
                    + 1.02755188689515710272E3) * x + 5.57535335369399327526E2;
            double denominator = (((((((x + 1.32281951154744992508E1) * x
                    + 8.67072140885989742329E1) * x + 3.54937778887819891062E2) * x
                    + 9.75708501743205489753E2) * x + 1.82390916687909736289E3) * x
                    + 2.24633760818710981792E3) * x + 1.65666309194161350182E3) * x
                    + 5.57535340817727675546E2;
            result = 1 - Math.exp(-x * x) * (numerator / denominator);
        }
        return Math.copySign(result, value);
    }

    /** Evaluates every composite with FLOAT32 named sites and no intermediate low narrowing.
     * @param kind one of the five composite opcodes
     * @param x exactly decoded low input
     * @return FLOAT32 working result
     */
    static float composite(CpuPointwiseOpcode kind, float x) {
        return switch (kind) {
            case RSQRT -> 1f / (float) StrictMath.sqrt(x);
            case SIGMOID -> sigmoid(x);
            case GELU_EXACT -> gelu(x);
            case GELU_TANH_APPROXIMATION -> geluTanh(x);
            case SILU -> silu(x);
            default -> throw new AssertionError(kind);
        };
    }

    /** Evaluates the stable sigmoid branches with rounded FLOAT32 exponential results.
     * @param x represented input, including special values
     * @return FLOAT32 sign-guarded sigmoid
     */
    static float sigmoid(float x) {
        if (x >= 0) return 1f / (1f + (float) StrictMath.exp(-x));
        float exponential = (float) StrictMath.exp(x);
        return exponential / (1f + exponential);
    }

    /** Evaluates GELU with a FLOAT32 boundary after each named elementary/arithmetic site.
     * @param x represented input, including special values
     * @return FLOAT32 named-site exact GELU
     */
    static float gelu(float x) {
        if (x == Float.NEGATIVE_INFINITY) return -0f;
        float root = (float) StrictMath.sqrt(2f);
        float argument = x / root;
        float elementary = (float) erf(argument);
        return (0.5f * x) * (1f + elementary);
    }

    /** Evaluates the fixed tanh GELU formula with constants rounded before their first use.
     * @param x represented input, including special values
     * @return FLOAT32 fixed-form tanh GELU
     */
    static float geluTanh(float x) {
        if (x == Float.NEGATIVE_INFINITY) return -0f;
        float square = x * x;
        float cube = square * x;
        float ratio = 2f / (float) Math.PI;
        float scale = (float) StrictMath.sqrt(ratio);
        float inner = x + 0.044715f * cube;
        float elementary = (float) StrictMath.tanh(scale * inner);
        return (0.5f * x) * (1f + elementary);
    }

    /** Evaluates stable SiLU branches and its negative-infinity extension.
     * @param x represented input, including special values
     * @return FLOAT32 sign-guarded SiLU
     */
    static float silu(float x) {
        if (x == Float.NEGATIVE_INFINITY) return -0f;
        if (x >= 0) return x / (1f + (float) StrictMath.exp(-x));
        float exponential = (float) StrictMath.exp(x);
        return x * exponential / (1f + exponential);
    }

    /** Decodes a low representation without an intermediate narrowing.
     * @param type either low storage type
     * @param bits raw storage
     * @return exactly decoded input
     */
    static float decode(DataType type, short bits) {
        return type == DataType.BFLOAT16 ? Float.intBitsToFloat(bits << 16)
                : Float.float16ToFloat(bits);
    }

    /** Performs one low narrowing with the generated producer's canonical NaN representation.
     * @param type either low storage type
     * @param x working result
     * @return one ties-even narrowing
     */
    static short encode(DataType type, float x) {
        if (Float.isNaN(x)) return (short) (type == DataType.BFLOAT16 ? 0x7fc0 : 0x7e00);
        if (type == DataType.FLOAT16) return Float.floatToFloat16(x);
        int bits = Float.floatToRawIntBits(x);
        return (short) ((bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16);
    }

    /** Direct FLOAT32 vector oracle with bounded, straight-line Horner arithmetic.
     * @param value non-null preferred-species vector
     * @return lane-wise ERF, including exact saturation and special classes
     */
    static FloatVector erf(FloatVector value) {
        FloatVector x = value.abs().min(6f);
        FloatVector z = x.mul(x);
        FloatVector smallNumerator = z.mul(0x1.335bf2p3f).add(0x1.681aa4p6f)
                .mul(z).add(0x1.17002cp11f).mul(z).add(0x1.b5b534p12f)
                .mul(z).add(0x1.b2509ap15f);
        FloatVector smallDenominator = z.add(0x1.0c7e64p5f).mul(z).add(0x1.04add2p9f)
                .mul(z).add(0x1.1f252ep12f).mul(z).add(0x1.6194p14f)
                .mul(z).add(0x1.80e6cap15f);
        FloatVector numerator = x.mul(0x1.0eb24ap-32f).add(0x1.20dd74p-1f)
                .mul(x).add(0x1.dda53ep2f).mul(x).add(0x1.8518fap5f)
                .mul(x).add(0x1.890aaap7f).mul(x).add(0x1.0738fcp9f)
                .mul(x).add(0x1.d343a6p9f).mul(x).add(0x1.00e352p10f)
                .mul(x).add(0x1.16c486p9f);
        FloatVector denominator = x.add(0x1.a74d6p3f).mul(x).add(0x1.5ad43p6f)
                .mul(x).add(0x1.62f012p8f).mul(x).add(0x1.e7dabp9f)
                .mul(x).add(0x1.c7fa3p10f).mul(x).add(0x1.18cacep11f)
                .mul(x).add(0x1.9e2a7p10f).mul(x).add(0x1.16c486p9f);
        FloatVector one = FloatVector.broadcast(value.species(), 1f);
        FloatVector large = one.sub(z.neg().lanewise(VectorOperators.EXP)
                .mul(numerator.div(denominator)));
        FloatVector magnitude = large.blend(smallNumerator.div(smallDenominator).mul(x),
                x.compare(VectorOperators.LE, 1f)).blend(1f, x.eq(6f));
        return magnitude.blend(magnitude.neg(), value.test(VectorOperators.IS_NEGATIVE));
    }

    /** Direct FLOAT64 vector oracle with the same bounded dataflow and double coefficients.
     * @param value non-null preferred-species vector
     * @return lane-wise ERF, preserving saturation and special classes
     */
    static DoubleVector erf(DoubleVector value) {
        DoubleVector x = value.abs().min(6d);
        DoubleVector z = x.mul(x);
        DoubleVector smallNumerator = z.mul(9.60497373987051638749E0).add(9.00260197203842689217E1)
                .mul(z).add(2.23200534594684319226E3).mul(z).add(7.00332514112805075473E3)
                .mul(z).add(5.55923013010394962768E4);
        DoubleVector smallDenominator = z.add(3.35617141647503099647E1).mul(z).add(5.21357949780152679795E2)
                .mul(z).add(4.59432382970980127987E3).mul(z).add(2.26290000613890934246E4)
                .mul(z).add(4.92673942608635921086E4);
        DoubleVector numerator = x.mul(2.46196981473530512524E-10).add(5.64189564831068821977E-1)
                .mul(x).add(7.46321056442269912687E0).mul(x).add(4.86371970985681366614E1)
                .mul(x).add(1.96520832956077098242E2).mul(x).add(5.26445194995477358631E2)
                .mul(x).add(9.34528527171957607540E2).mul(x).add(1.02755188689515710272E3)
                .mul(x).add(5.57535335369399327526E2);
        DoubleVector denominator = x.add(1.32281951154744992508E1).mul(x).add(8.67072140885989742329E1)
                .mul(x).add(3.54937778887819891062E2).mul(x).add(9.75708501743205489753E2)
                .mul(x).add(1.82390916687909736289E3).mul(x).add(2.24633760818710981792E3)
                .mul(x).add(1.65666309194161350182E3).mul(x).add(5.57535340817727675546E2);
        DoubleVector one = DoubleVector.broadcast(value.species(), 1d);
        DoubleVector large = one.sub(z.neg().lanewise(VectorOperators.EXP).mul(numerator.div(denominator)));
        DoubleVector magnitude = large.blend(smallNumerator.div(smallDenominator).mul(x),
                x.compare(VectorOperators.LE, 1d)).blend(1d, x.eq(6d));
        return magnitude.blend(magnitude.neg(), value.test(VectorOperators.IS_NEGATIVE));
    }
}
