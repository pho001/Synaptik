package numerics;

import tensor.DataType;

/**
 * User-facing diagnostics policy that turns aggregate numeric drift metrics into a stability verdict.
 */
public final class NumericsPolicy {
    private final double absTol;
    private final double relTol;
    private final long maxUlpTol;

    /**
     * Creates a policy with absolute, relative, and ULP tolerances.
     *
     * @param absTol maximum absolute error before escalation
     * @param relTol relative tolerance scaled by the observed maximum magnitude
     * @param maxUlpTol maximum ULP distance before escalation
     */
    public NumericsPolicy(double absTol, double relTol, long maxUlpTol) {
        this.absTol = absTol;
        this.relTol = relTol;
        this.maxUlpTol = maxUlpTol;
    }

    /**
     * Returns default tolerances for a floating-point data type used by the numerics harness.
     *
     * @param dtype data type being compared
     * @return default comparison policy for the data type
     */
    public static NumericsPolicy defaultsFor(DataType dtype) {
        if (dtype == DataType.BOOL) {
            throw new UnsupportedOperationException("BOOL is not supported by numerics policy.");
        }
        if (dtype == DataType.FLOAT64) {
            return new NumericsPolicy(1e-12, 1e-12, 16L);
        }
        return new NumericsPolicy(1e-5, 1e-5, 128L);
    }

    /**
     * Evaluates aggregate metrics and classifies them as safe, borderline, or unsafe.
     *
     * @param m aggregate metrics from one numerics run
     * @return verdict explaining the highest-severity observed drift
     */
    public Verdict evaluate(NumericsMetrics.AggregateMetrics m) {
        if (m.invalidCount > 0) {
            return Verdict.unsafe("invalidCount=" + m.invalidCount);
        }
        double tol = absTol + relTol * Math.max(1.0, m.maxAbs);
        if (m.maxAbs > tol) {
            if (m.maxUlp <= maxUlpTol) {
                return Verdict.borderline("maxAbs=" + m.maxAbs + " > tol=" + tol + " but maxUlp=" + m.maxUlp + " <= " + maxUlpTol);
            }
            return Verdict.unsafe("maxAbs=" + m.maxAbs + " > tol=" + tol + ", maxUlp=" + m.maxUlp);
        }
        if (m.maxUlp > maxUlpTol) {
            return Verdict.borderline("maxUlp=" + m.maxUlp + " > " + maxUlpTol + " with maxAbs within tol");
        }
        return Verdict.safe("within abs/rel and ulp tolerance");
    }

    /**
     * Severity levels reported by the user-facing numerics diagnostics.
     */
    public enum Status { SAFE, BORDERLINE, UNSAFE }

    /**
     * Immutable user-facing decision with a severity and short diagnostic reason.
     */
    public static final class Verdict {
        /** Diagnostic severity assigned by the policy. */
        public final Status status;
        /** Human-readable reason for the status. */
        public final String reason;

        private Verdict(Status status, String reason) {
            this.status = status;
            this.reason = reason;
        }

        /**
         * Creates a safe verdict.
         *
         * @param reason human-readable reason
         * @return safe verdict
         */
        public static Verdict safe(String reason) { return new Verdict(Status.SAFE, reason); }
        /**
         * Creates a borderline verdict.
         *
         * @param reason human-readable reason
         * @return borderline verdict
         */
        public static Verdict borderline(String reason) { return new Verdict(Status.BORDERLINE, reason); }
        /**
         * Creates an unsafe verdict.
         *
         * @param reason human-readable reason
         * @return unsafe verdict
         */
        public static Verdict unsafe(String reason) { return new Verdict(Status.UNSAFE, reason); }
    }
}
