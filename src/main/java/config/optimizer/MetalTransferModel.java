package config.optimizer;

/**
 * Graph-level transfer-cost preset for scored Metal accelerator region planning.
 *
 * <p>The model changes profitability scoring only. It does not change Metal legality, dtype support,
 * native bridge availability, or tensor storage synchronization rules. The conservative model matches
 * the current copy-based FFM bridge; less conservative models are intended for research benchmarking
 * and future shared-buffer/device-resident execution.</p>
 */
public enum MetalTransferModel {
    /**
     * Penalizes Metal boundaries strongly because the current bridge copies inputs and outputs.
     */
    CONSERVATIVE(0.05d, 0.10d, 0.025d),

    /**
     * Intermediate preset for experiments where some transfer costs may be amortized or measured lower.
     */
    MEASURED(0.025d, 0.05d, 0.05d),

    /**
     * Low-penalty research preset for exploring larger Metal regions.
     */
    AGGRESSIVE(0.01d, 0.02d, 0.10d);

    private final double inputBytePenalty;
    private final double outputBytePenalty;
    private final double avoidedIntermediateByteCredit;

    MetalTransferModel(double inputBytePenalty, double outputBytePenalty, double avoidedIntermediateByteCredit) {
        this.inputBytePenalty = inputBytePenalty;
        this.outputBytePenalty = outputBytePenalty;
        this.avoidedIntermediateByteCredit = avoidedIntermediateByteCredit;
    }

    /**
     * Returns the score penalty per byte entering a Metal region.
     *
     * @return non-negative input byte penalty
     */
    public double inputBytePenalty() {
        return inputBytePenalty;
    }

    /**
     * Returns the score penalty per byte leaving a Metal region.
     *
     * @return non-negative output byte penalty
     */
    public double outputBytePenalty() {
        return outputBytePenalty;
    }

    /**
     * Returns the score credit per internal intermediate byte kept inside a Metal region.
     *
     * @return non-negative avoided intermediate byte credit
     */
    public double avoidedIntermediateByteCredit() {
        return avoidedIntermediateByteCredit;
    }
}
