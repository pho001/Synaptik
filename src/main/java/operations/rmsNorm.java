package operations;

public final class rmsNorm implements Operation {
    private final int normalizedRank;
    private final double epsilon;

    public rmsNorm(int normalizedRank, double epsilon) {
        if (normalizedRank < 1) {
            throw new IllegalArgumentException("normalizedRank must be >= 1");
        }
        if (!(epsilon > 0.0d)) {
            throw new IllegalArgumentException("epsilon must be > 0");
        }
        this.normalizedRank = normalizedRank;
        this.epsilon = epsilon;
    }

    @Override
    public OpType opType() {
        return OpType.RMS_NORM;
    }

    @Override
    public String getExpression() {
        return "rmsNorm";
    }

    public int getNormalizedRank() {
        return normalizedRank;
    }

    public double getEpsilon() {
        return epsilon;
    }
}
