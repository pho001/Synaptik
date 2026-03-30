package numerics;

import tensor.DataType;

import java.util.Arrays;

public final class NumericsMetrics {
    private NumericsMetrics() {}

    public static SignalMetrics compare(double[] baseline, double[] candidate) {
        return compare(baseline, candidate, DataType.FLOAT64);
    }

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

    public static final class SignalMetrics {
        public final double maxAbs;
        public final double avgAbs;
        public final double maxRel;
        public final long maxUlp;
        public final long p50Ulp;
        public final long p95Ulp;
        public final int finiteCount;
        public final int invalidCount;

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

    public static final class AggregateMetrics {
        public final double maxAbs;
        public final double maxRel;
        public final long maxUlp;
        public final int invalidCount;

        public AggregateMetrics(double maxAbs, double maxRel, long maxUlp, int invalidCount) {
            this.maxAbs = maxAbs;
            this.maxRel = maxRel;
            this.maxUlp = maxUlp;
            this.invalidCount = invalidCount;
        }
    }

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

    public static long ulpDistance(double a, double b) {
        long ia = Double.doubleToLongBits(a);
        long ib = Double.doubleToLongBits(b);
        long oa = ia < 0 ? Long.MIN_VALUE - ia : ia;
        long ob = ib < 0 ? Long.MIN_VALUE - ib : ib;
        long d = oa - ob;
        return d < 0 ? -d : d;
    }

    public static long ulpDistance(float a, float b) {
        int ia = Float.floatToIntBits(a);
        int ib = Float.floatToIntBits(b);
        int oa = ia < 0 ? Integer.MIN_VALUE - ia : ia;
        int ob = ib < 0 ? Integer.MIN_VALUE - ib : ib;
        int d = oa - ob;
        return Math.abs((long) d);
    }

    public static long toDTypeUlpDistance(double a, double b, DataType dtype) {
        if (dtype == DataType.FLOAT32) {
            return ulpDistance((float) a, (float) b);
        }
        return ulpDistance(a, b);
    }
}
