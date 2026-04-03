package numerics;

import tensor.DataType;

public final class NumericsPolicy {
    private final double absTol;
    private final double relTol;
    private final long maxUlpTol;

    public NumericsPolicy(double absTol, double relTol, long maxUlpTol) {
        this.absTol = absTol;
        this.relTol = relTol;
        this.maxUlpTol = maxUlpTol;
    }

    public static NumericsPolicy defaultsFor(DataType dtype) {
        if (dtype == DataType.BOOL) {
            throw new UnsupportedOperationException("BOOL is not supported by numerics policy.");
        }
        if (dtype == DataType.FLOAT64) {
            return new NumericsPolicy(1e-12, 1e-12, 16L);
        }
        return new NumericsPolicy(1e-5, 1e-5, 128L);
    }

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

    public enum Status { SAFE, BORDERLINE, UNSAFE }

    public static final class Verdict {
        public final Status status;
        public final String reason;

        private Verdict(Status status, String reason) {
            this.status = status;
            this.reason = reason;
        }

        public static Verdict safe(String reason) { return new Verdict(Status.SAFE, reason); }
        public static Verdict borderline(String reason) { return new Verdict(Status.BORDERLINE, reason); }
        public static Verdict unsafe(String reason) { return new Verdict(Status.UNSAFE, reason); }
    }
}
