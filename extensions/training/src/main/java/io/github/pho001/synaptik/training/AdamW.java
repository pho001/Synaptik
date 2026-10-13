package io.github.pho001.synaptik.training;

/**
 * Immutable AdamW configuration with decoupled weight decay.
 *
 * <p>On each successful global optimizer step, moments use the selected gradient without decay.
 * The master parameter first receives the factor {@code 1 - learningRate * weightDecay}, then
 * the bias-corrected adaptive subtraction. A session owns the moments; this shareable value owns
 * none. FLOAT32, BFLOAT16, and FLOAT16 parameters use coefficients narrowed once to FLOAT32 at
 * open; FLOAT64 parameters use the configured binary64 coefficients.</p>
 */
public final class AdamW implements Optimizer {
    private final double learningRate;
    private final double beta1;
    private final double beta2;
    private final double epsilon;
    private final double weightDecay;

    /**
     * Creates one decoupled-decay AdamW configuration.
     *
     * @param learningRate finite, strictly positive update scale
     * @param beta1 finite first-moment coefficient in {@code [0, 1)}
     * @param beta2 finite second-moment coefficient in {@code [0, 1)}
     * @param epsilon finite, strictly positive denominator offset
     * @param weightDecay finite, non-negative decoupled decay coefficient
     * @throws IllegalArgumentException if any coefficient violates its range
     */
    public AdamW(double learningRate, double beta1, double beta2, double epsilon,
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
     * Returns the coefficient used to scale the master parameter separately from gradients.
     *
     * @return the finite, non-negative decoupled decay coefficient
     */
    public double weightDecay() {
        return weightDecay;
    }

    @Override
    public boolean equals(Object other) {
        return this == other || other instanceof AdamW candidate
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
        return "AdamW[learningRate=" + learningRate + ", beta1=" + beta1 + ", beta2=" + beta2
                + ", epsilon=" + epsilon + ", weightDecay=" + weightDecay + ']';
    }
}
