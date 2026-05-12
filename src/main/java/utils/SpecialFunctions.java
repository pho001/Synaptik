package utils;

/**
 * Numerically focused scalar special functions used by kernels and compile-time folding.
 */
public final class SpecialFunctions {
    private SpecialFunctions() {
    }

    /**
     * Approximates the Gaussian error function with maximum error around 1.5e-7.
     *
     * <p>Java does not expose {@code erf} in {@link Math}, so Synaptik uses the standard
     * Abramowitz-Stegun 7.1.26 approximation for ONNX interchange and CPU execution.</p>
     */
    public static double erf(double value) {
        if (Double.isNaN(value)) {
            return Double.NaN;
        }
        if (Double.isInfinite(value)) {
            return Math.copySign(1.0d, value);
        }
        double sign = Math.copySign(1.0d, value);
        double x = Math.abs(value);
        double t = 1.0d / (1.0d + 0.3275911d * x);
        double polynomial = (((((1.061405429d * t - 1.453152027d) * t) + 1.421413741d) * t - 0.284496736d) * t + 0.254829592d) * t;
        return sign * (1.0d - polynomial * Math.exp(-x * x));
    }

    public static float erf(float value) {
        return (float) erf((double) value);
    }
}
