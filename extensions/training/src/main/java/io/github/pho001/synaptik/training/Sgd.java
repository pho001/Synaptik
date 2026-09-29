package io.github.pho001.synaptik.training;

/**
 * Immutable stochastic-gradient-descent configuration with optional momentum, dampening, coupled
 * weight decay, and Nesterov momentum.
 *
 * <p>For parameter value {@code p}, published gradient {@code g}, weight decay {@code wd},
 * momentum {@code m}, dampening {@code damp}, and learning rate {@code lr}, the adjusted gradient
 * is {@code d = g + wd * p}. With momentum disabled the update is {@code p = p - lr * d}. With
 * momentum enabled, the first successful optimizer step initializes the parameter's session-owned
 * slot to {@code d}; dampening is deliberately not applied on that first step. Later steps set the
 * slot to {@code m * previous + (1 - damp) * d}. Ordinary momentum uses the updated slot.
 * Nesterov uses {@code d + m * updatedSlot}, requires positive momentum, and is compatible only
 * with zero dampening. FLOAT32, BFLOAT16, and FLOAT16 parameter sessions narrow all numeric
 * configuration values once and perform binary32 master-state arithmetic; FLOAT64 parameter
 * sessions use binary64 arithmetic.</p>
 *
 * <p>The value owns no parameters or mutable state and may be shared by sessions. Each session
 * initializes independent zero momentum slots and retains this exact configuration, including
 * dampening, in its immutable state snapshots.</p>
 */
public final class Sgd implements Optimizer {
    private final double learningRate;
    private final double momentum;
    private final double dampening;
    private final double weightDecay;
    private final boolean nesterov;

    /**
     * Creates plain SGD without momentum, weight decay, or Nesterov behavior.
     *
     * @param learningRate finite strictly positive update scale
     * @throws IllegalArgumentException if {@code learningRate} is not finite and positive
     */
    public Sgd(double learningRate) {
        this(learningRate, 0.0d, 0.0d, 0.0d, false);
    }

    /**
     * Creates an SGD configuration with zero dampening.
     *
     * @param learningRate finite strictly positive update scale
     * @param momentum finite coefficient in the half-open interval {@code [0, 1)}
     * @param weightDecay finite non-negative coupled L2 coefficient
     * @param nesterov whether to use Nesterov momentum; requires positive {@code momentum}
     * @throws IllegalArgumentException if a numeric constraint is violated or Nesterov is selected
     *     with zero momentum
     */
    public Sgd(
            double learningRate,
            double momentum,
            double weightDecay,
            boolean nesterov) {
        this(learningRate, momentum, 0.0d, weightDecay, nesterov);
    }

    /**
     * Creates one complete SGD configuration.
     *
     * @param learningRate finite strictly positive update scale
     * @param momentum finite coefficient in the half-open interval {@code [0, 1)}
     * @param dampening finite coefficient in {@code [0, 1]}; not applied on the first momentum step
     * @param weightDecay finite non-negative coupled L2 coefficient
     * @param nesterov whether to use Nesterov momentum; requires positive {@code momentum} and zero
     *     {@code dampening}
     * @throws IllegalArgumentException if a numeric constraint is violated or Nesterov is
     *     incompatible with the supplied momentum or dampening
     */
    public Sgd(
            double learningRate,
            double momentum,
            double dampening,
            double weightDecay,
            boolean nesterov) {
        if (!Double.isFinite(learningRate) || learningRate <= 0.0d) {
            throw new IllegalArgumentException(
                    "learningRate must be finite and positive: " + learningRate);
        }
        if (!Double.isFinite(momentum) || momentum < 0.0d || momentum >= 1.0d) {
            throw new IllegalArgumentException(
                    "momentum must be finite and in [0, 1): " + momentum);
        }
        if (!Double.isFinite(dampening) || dampening < 0.0d || dampening > 1.0d) {
            throw new IllegalArgumentException(
                    "dampening must be finite and in [0, 1]: " + dampening);
        }
        if (!Double.isFinite(weightDecay) || weightDecay < 0.0d) {
            throw new IllegalArgumentException(
                    "weightDecay must be finite and non-negative: " + weightDecay);
        }
        if (nesterov && momentum == 0.0d) {
            throw new IllegalArgumentException("Nesterov momentum requires positive momentum");
        }
        if (nesterov && dampening != 0.0d) {
            throw new IllegalArgumentException("Nesterov momentum requires zero dampening");
        }
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.dampening = dampening;
        this.weightDecay = weightDecay;
        this.nesterov = nesterov;
    }

    /**
     * Returns the configured update scale.
     *
     * @return the finite strictly positive learning rate
     */
    public double learningRate() {
        return learningRate;
    }

    /**
     * Returns the configured momentum coefficient.
     *
     * @return a finite value in {@code [0, 1)}; zero disables momentum
     */
    public double momentum() {
        return momentum;
    }

    /**
     * Returns the configured dampening coefficient.
     *
     * @return a finite value in {@code [0, 1]}; ignored on the first momentum step
     */
    public double dampening() {
        return dampening;
    }

    /**
     * Returns the coupled L2 coefficient added to each gradient.
     *
     * @return a finite non-negative coefficient; zero disables weight decay
     */
    public double weightDecay() {
        return weightDecay;
    }

    /**
     * Reports whether updates use Nesterov momentum.
     *
     * @return {@code true} only when positive momentum and Nesterov behavior are configured
     */
    public boolean nesterov() {
        return nesterov;
    }

    @Override
    public boolean equals(Object other) {
        return this == other || other instanceof Sgd candidate
                && Double.doubleToLongBits(learningRate)
                        == Double.doubleToLongBits(candidate.learningRate)
                && Double.doubleToLongBits(momentum)
                        == Double.doubleToLongBits(candidate.momentum)
                && Double.doubleToLongBits(dampening)
                        == Double.doubleToLongBits(candidate.dampening)
                && Double.doubleToLongBits(weightDecay)
                        == Double.doubleToLongBits(candidate.weightDecay)
                && nesterov == candidate.nesterov;
    }

    @Override
    public int hashCode() {
        int result = Long.hashCode(Double.doubleToLongBits(learningRate));
        result = 31 * result + Long.hashCode(Double.doubleToLongBits(momentum));
        result = 31 * result + Long.hashCode(Double.doubleToLongBits(dampening));
        result = 31 * result + Long.hashCode(Double.doubleToLongBits(weightDecay));
        return 31 * result + Boolean.hashCode(nesterov);
    }

    @Override
    public String toString() {
        return "Sgd[learningRate=" + learningRate
                + ", momentum=" + momentum
                + ", dampening=" + dampening
                + ", weightDecay=" + weightDecay
                + ", nesterov=" + nesterov + ']';
    }
}
