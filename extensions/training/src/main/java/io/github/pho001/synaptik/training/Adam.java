package io.github.pho001.synaptik.training;

/**
 * Immutable Adam configuration with coupled L2 weight decay.
 *
 * <p>On each successful global optimizer step, Adam adds {@code weightDecay * masterParameter}
 * to the selected gradient before updating its first and second moments. Bias-corrected moments
 * produce the adaptive subtraction. A session owns the moments; this shareable value owns none.
 * FLOAT32, BFLOAT16, and FLOAT16 parameters use coefficients narrowed once to FLOAT32 at open;
 * FLOAT64 parameters use the configured binary64 coefficients.</p>
 */
public final class Adam implements Optimizer {
    private final double learningRate;
    private final double beta1;
    private final double beta2;
    private final double epsilon;
    private final double weightDecay;

    /**
     * Creates one coupled-decay Adam configuration.
     *
     * @param learningRate finite, strictly positive update scale
     * @param beta1 finite first-moment coefficient in {@code [0, 1)}
     * @param beta2 finite second-moment coefficient in {@code [0, 1)}
     * @param epsilon finite, strictly positive denominator offset
     * @param weightDecay finite, non-negative coupled L2 coefficient
     * @throws IllegalArgumentException if any coefficient violates its range
     */
    public Adam(double learningRate, double beta1, double beta2, double epsilon,
            double weightDecay) {
        AdaptiveCoefficients.validate(learningRate, beta1, beta2, epsilon, weightDecay);
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.weightDecay = weightDecay;
    }

    /**
     * Returns the update scale applied after moment bias correction.
     *
     * @return the finite, strictly positive learning rate
     */
    public double learningRate() {
        return learningRate;
    }

    /**
     * Returns the decay coefficient of the first-moment history.
     *
     * @return the finite first-moment coefficient in {@code [0, 1)}
     */
    public double beta1() {
        return beta1;
    }

    /**
     * Returns the decay coefficient of the squared-gradient history.
     *
     * @return the finite second-moment coefficient in {@code [0, 1)}
     */
    public double beta2() {
        return beta2;
    }

    /**
     * Returns the offset added after the square root of the corrected second moment.
     *
     * @return the finite, strictly positive denominator offset
     */
    public double epsilon() {
        return epsilon;
    }

    /**
     * Returns the L2 coefficient added to the gradient before moment updates.
     *
     * @return the finite, non-negative coupled decay coefficient
     */
    public double weightDecay() {
        return weightDecay;
    }

    @Override
    public boolean equals(Object other) {
        return this == other || other instanceof Adam candidate
                && AdaptiveCoefficients.equal(learningRate, candidate.learningRate)
                && AdaptiveCoefficients.equal(beta1, candidate.beta1)
                && AdaptiveCoefficients.equal(beta2, candidate.beta2)
                && AdaptiveCoefficients.equal(epsilon, candidate.epsilon)
                && AdaptiveCoefficients.equal(weightDecay, candidate.weightDecay);
    }

    @Override
    public int hashCode() {
        return AdaptiveCoefficients.hash(learningRate, beta1, beta2, epsilon, weightDecay);
    }

    @Override
    public String toString() {
        return "Adam[learningRate=" + learningRate + ", beta1=" + beta1 + ", beta2=" + beta2
                + ", epsilon=" + epsilon + ", weightDecay=" + weightDecay + ']';
    }
}
