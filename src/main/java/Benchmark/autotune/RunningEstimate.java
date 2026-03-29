package Benchmark.autotune;

public final class RunningEstimate {
    private final double confidenceZ;
    private int count;
    private double mean;
    private double m2;

    public RunningEstimate(double confidenceZ) {
        this.confidenceZ = Math.max(0.0, confidenceZ);
    }

    public void add(double value) {
        count++;
        double delta = value - mean;
        mean += delta / count;
        double delta2 = value - mean;
        m2 += delta * delta2;
    }

    public int count() {
        return count;
    }

    public double mean() {
        return count == 0 ? Double.POSITIVE_INFINITY : mean;
    }

    public double variance() {
        if (count < 2) {
            return Double.POSITIVE_INFINITY;
        }
        return m2 / (count - 1);
    }

    public double stderr() {
        if (count < 2) {
            return Double.POSITIVE_INFINITY;
        }
        return Math.sqrt(Math.max(0.0, variance()) / count);
    }

    public double optimistic() {
        if (count == 0) {
            return Double.NEGATIVE_INFINITY;
        }
        double stderr = stderr();
        if (!Double.isFinite(stderr)) {
            return mean;
        }
        return mean - (confidenceZ * stderr);
    }

    public double conservative() {
        if (count == 0) {
            return Double.POSITIVE_INFINITY;
        }
        double stderr = stderr();
        if (!Double.isFinite(stderr)) {
            return mean;
        }
        return mean + (confidenceZ * stderr);
    }
}
