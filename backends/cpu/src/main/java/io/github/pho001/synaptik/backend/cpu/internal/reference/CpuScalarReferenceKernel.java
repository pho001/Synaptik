package io.github.pho001.synaptik.backend.cpu.internal.reference;

/**
 * Scalar conformance realization for the exact CPU-0005A fused semantics.
 * It is an unsupported cold-test/reference contract and is never a Runtime IR interpreter.
 */
public final class CpuScalarReferenceKernel {
    private static final double[] ERF_T = {9.60497373987051638749E0,
            9.00260197203842689217E1, 2.23200534594684319226E3,
            7.00332514112805075473E3, 5.55923013010394962768E4};
    private static final double[] ERF_U = {3.35617141647503099647E1,
            5.21357949780152679795E2, 4.59432382970980127987E3,
            2.26290000613890934246E4, 4.92673942608635921086E4};
    private static final double[] ERFC_P = {2.46196981473530512524E-10,
            5.64189564831068821977E-1, 7.46321056442269912687E0,
            4.86371970985681366614E1, 1.96520832956077098242E2,
            5.26445194995477358631E2, 9.34528527171957607540E2,
            1.02755188689515710272E3, 5.57535335369399327526E2};
    private static final double[] ERFC_Q = {1.32281951154744992508E1,
            8.67072140885989742329E1, 3.54937778887819891062E2,
            9.75708501743205489753E2, 1.82390916687909736289E3,
            2.24633760818710981792E3, 1.65666309194161350182E3,
            5.57535340817727675546E2};
    private static final double[] ERFC_R = {5.64189583547755073984E-1,
            1.27536670759978104416E0, 5.01905042251180477414E0,
            6.16021097993053585195E0, 7.40974269950448939160E0,
            2.97886665372100240670E0};
    private static final double[] ERFC_S = {2.26052863220117276590E0,
            9.39603524938001434673E0, 1.20489539808096656605E1,
            1.70814450747565897222E1, 9.60896809063285878198E0,
            3.36907645100081516050E0};
    private CpuScalarReferenceKernel() { }

    /**
     * Evaluates the exact Model GELU target in fixed operation order.
     *
     * @param value input value, including IEEE 754 special values
     * @return {@code 0.5 * value * (1 + erf(value / sqrt(2)))}, preserving the implementation's
     *     documented special-value classifications
     */
    public static double gelu(double value) {
        return 0.5d * value * (1.0d + erf(value / Math.sqrt(2.0d)));
    }

    /**
     * Evaluates a portable scalar approximation of the Gaussian error function.
     * The approximation is shared by generated and reference realizations and preserves NaN,
     * infinities, and signed zero classifications.
     *
     * @param value input value, including IEEE 754 special values
     * @return the finite approximation to the Gaussian error function, or the corresponding
     *     preserved NaN, infinity, or signed-zero classification
     */
    public static double erf(double value) {
        if (Double.isNaN(value)) return Double.NaN;
        if (value == 0.0d) return value;
        if (value == Double.POSITIVE_INFINITY) return 1.0d;
        if (value == Double.NEGATIVE_INFINITY) return -1.0d;
        double x = Math.abs(value);
        double result;
        if (x <= 1.0d) {
            double z = x * x;
            result = x * polevl(z, ERF_T) / p1evl(z, ERF_U);
        } else {
            double erfc = Math.exp(-x * x) * (x < 8.0d
                    ? polevl(x, ERFC_P) / p1evl(x, ERFC_Q)
                    : polevl(x, ERFC_R) / p1evl(x, ERFC_S));
            result = 1.0d - erfc;
        }
        return Math.copySign(result, value);
    }

    /**
     * Executes the fused reference calculation over one half-open range.
     *
     * @param a non-null first ADD input; not mutated
     * @param b non-null second ADD input; not mutated
     * @param c non-null MUL input; not mutated
     * @param output non-null destination mutated only in {@code [start, end)}
     * @param start non-negative inclusive element index
     * @param end exclusive element index no greater than any array length
     * @throws NullPointerException if an array is {@code null}
     * @throws IllegalArgumentException if the half-open range is negative, reversed, or exceeds an
     *     input or output array
     */
    public static void execute(double[] a, double[] b, double[] c, double[] output,
            long start, long end) {
        if (start < 0 || end < start || end > a.length || end > b.length || end > c.length
                || end > output.length) throw new IllegalArgumentException("invalid reference bounds");
        for (long index = start; index < end; index++) {
            double sum = a[(int) index] + b[(int) index];
            double activated = gelu(sum);
            output[(int) index] = activated * c[(int) index];
        }
    }

    private static double polevl(double x, double[] coefficients) {
        double result = coefficients[0];
        for (int i = 1; i < coefficients.length; i++) result = result * x + coefficients[i];
        return result;
    }

    private static double p1evl(double x, double[] coefficients) {
        double result = x + coefficients[0];
        for (int i = 1; i < coefficients.length; i++) result = result * x + coefficients[i];
        return result;
    }
}
