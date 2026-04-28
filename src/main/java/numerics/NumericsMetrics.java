package numerics;

import tensor.DataType;

import java.util.Arrays;

/**
 * User-facing diagnostics utilities for comparing numeric signals produced by two execution profiles.
 */
public final class NumericsMetrics {
    private NumericsMetrics() {}

    /**
     * Compares two FLOAT64 signals and returns absolute, relative, and ULP drift metrics.
     *
     * @param baseline reference signal
     * @param candidate candidate signal
     * @return drift metrics for the signal pair
     */
    public static SignalMetrics compare(double[] baseline, double[] candidate) {
        return compare(baseline, candidate, DataType.FLOAT64);
    }

    /**
     * Compares two signals using the ULP scale appropriate for the supplied data type.
     *
     * @param baseline reference signal
     * @param candidate candidate signal
     * @param dtype floating-point data type represented by the signal values
     * @return drift metrics for the signal pair
     */
    public static SignalMetrics compare(double[] baseline, double[] candidate, DataType dtype) {
        if (baseline.length != candidate.length) {
            throw new IllegalArgumentException("Mismatched lengths: " + baseline.length + " vs " + candidate.length);
        }

        double maxAbs = 0.0;
        double maxRel = 0.0;
        double sumAbs = 0.0;
        long maxUlp = 0L;
        int finiteCount = 0;
        int invalidCount = 0;
        long[] ulps = new long[baseline.length];
        int ulpCount = 0;

        for (int i = 0; i < baseline.length; i++) {
            double a = baseline[i];
            double b = candidate[i];
            if (!Double.isFinite(a) || !Double.isFinite(b)) {
                invalidCount++;
                continue;
            }
            double abs = Math.abs(a - b);
            double rel = abs / Math.max(1.0, Math.abs(a));
            long ulp = toDTypeUlpDistance(a, b, dtype);
            if (abs > maxAbs) maxAbs = abs;
            if (rel > maxRel) maxRel = rel;
            if (ulp > maxUlp) maxUlp = ulp;
            sumAbs += abs;
            finiteCount++;
            ulps[ulpCount++] = ulp;
        }

        double avgAbs = finiteCount == 0 ? 0.0 : (sumAbs / finiteCount);
        if (ulpCount == 0) {
            return new SignalMetrics(maxAbs, avgAbs, maxRel, maxUlp, 0L, 0L, finiteCount, invalidCount);
        }

        Arrays.sort(ulps, 0, ulpCount);
        long p50Ulp = ulps[(int) Math.floor((ulpCount - 1) * 0.50)];
        long p95Ulp = ulps[(int) Math.floor((ulpCount - 1) * 0.95)];
        return new SignalMetrics(maxAbs, avgAbs, maxRel, maxUlp, p50Ulp, p95Ulp, finiteCount, invalidCount);
    }

    /**
     * User-facing drift summary for one numeric signal.
     */
    public static final class SignalMetrics {
        /** Maximum absolute difference across finite values. */
        public final double maxAbs;
        /** Mean absolute difference across finite values. */
        public final double avgAbs;
        /** Maximum relative difference across finite values. */
        public final double maxRel;
        /** Maximum observed ULP distance. */
        public final long maxUlp;
        /** Median observed ULP distance. */
        public final long p50Ulp;
        /** 95th-percentile observed ULP distance. */
        public final long p95Ulp;
        /** Number of positions where both values were finite. */
        public final int finiteCount;
        /** Number of positions containing a non-finite value in either signal. */
        public final int invalidCount;

        /**
         * Creates a signal-level metrics record.
         */
        public SignalMetrics(
                double maxAbs,
                double avgAbs,
                double maxRel,
                long maxUlp,
                long p50Ulp,
                long p95Ulp,
                int finiteCount,
                int invalidCount
        ) {
            this.maxAbs = maxAbs;
            this.avgAbs = avgAbs;
            this.maxRel = maxRel;
            this.maxUlp = maxUlp;
            this.p50Ulp = p50Ulp;
            this.p95Ulp = p95Ulp;
            this.finiteCount = finiteCount;
            this.invalidCount = invalidCount;
        }
    }

    /**
     * User-facing aggregate drift summary across all signals in a numerics run.
     */
    public static final class AggregateMetrics {
        /** Maximum absolute difference across all compared signals. */
        public final double maxAbs;
        /** Maximum relative difference across all compared signals. */
        public final double maxRel;
        /** Maximum ULP distance across all compared signals. */
        public final long maxUlp;
        /** Total count of non-finite comparison positions across all signals. */
        public final int invalidCount;

        /**
         * Creates an aggregate metrics record.
         */
        public AggregateMetrics(double maxAbs, double maxRel, long maxUlp, int invalidCount) {
            this.maxAbs = maxAbs;
            this.maxRel = maxRel;
            this.maxUlp = maxUlp;
            this.invalidCount = invalidCount;
        }
    }

    /**
     * Combines per-signal metrics into the aggregate used by {@link NumericsPolicy}.
     *
     * @return maximum drift and total invalid count across all supplied signals
     */
    public static AggregateMetrics aggregate(
            SignalMetrics out,
            SignalMetrics gradA,
            SignalMetrics gradB,
            SignalMetrics gradC,
            SignalMetrics broadcast
    ) {
        double maxAbs = Math.max(Math.max(out.maxAbs, gradA.maxAbs), Math.max(Math.max(gradB.maxAbs, gradC.maxAbs), broadcast.maxAbs));
        double maxRel = Math.max(Math.max(out.maxRel, gradA.maxRel), Math.max(Math.max(gradB.maxRel, gradC.maxRel), broadcast.maxRel));
        long maxUlp = Math.max(Math.max(out.maxUlp, gradA.maxUlp), Math.max(Math.max(gradB.maxUlp, gradC.maxUlp), broadcast.maxUlp));
        int invalidCount = out.invalidCount + gradA.invalidCount + gradB.invalidCount + gradC.invalidCount + broadcast.invalidCount;
        return new AggregateMetrics(maxAbs, maxRel, maxUlp, invalidCount);
    }

    /**
     * Computes the monotonic IEEE-754 ULP distance between two double values.
     *
     * @param a first value
     * @param b second value
     * @return absolute ULP distance
     */
    public static long ulpDistance(double a, double b) {
        long ia = Double.doubleToLongBits(a);
        long ib = Double.doubleToLongBits(b);
        long oa = ia < 0 ? Long.MIN_VALUE - ia : ia;
        long ob = ib < 0 ? Long.MIN_VALUE - ib : ib;
        long d = oa - ob;
        return d < 0 ? -d : d;
    }

    /**
     * Computes the monotonic IEEE-754 ULP distance between two float values.
     *
     * @param a first value
     * @param b second value
     * @return absolute ULP distance
     */
    public static long ulpDistance(float a, float b) {
        int ia = Float.floatToIntBits(a);
        int ib = Float.floatToIntBits(b);
        int oa = ia < 0 ? Integer.MIN_VALUE - ia : ia;
        int ob = ib < 0 ? Integer.MIN_VALUE - ib : ib;
        int d = oa - ob;
        return Math.abs((long) d);
    }

    /**
     * Computes ULP distance after applying the comparison precision implied by a data type.
     *
     * @param a first value
     * @param b second value
     * @param dtype comparison precision
     * @return absolute ULP distance at the requested precision
     */
    public static long toDTypeUlpDistance(double a, double b, DataType dtype) {
        if (dtype == DataType.FLOAT32) {
            return ulpDistance((float) a, (float) b);
        }
        if (dtype == DataType.BOOL) {
            throw new UnsupportedOperationException("BOOL is not supported by numerics ULP metrics.");
        }
        return ulpDistance(a, b);
    }
}
