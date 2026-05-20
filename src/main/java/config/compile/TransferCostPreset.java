package config.compile;

/**
 * Backend-neutral transfer-cost preset for scored accelerator region planning.
 *
 * <p>The preset changes profitability scoring only. It does not change backend legality, dtype support,
 * native bridge availability, or tensor storage synchronization rules.</p>
 */
public enum TransferCostPreset {
    /**
     * Penalizes accelerator boundaries strongly when transfers or result copies dominate.
     */
    CONSERVATIVE(0.05d, 0.10d, 0.025d),

    /**
     * Intermediate preset for experiments where some transfer costs may be amortized or measured lower.
     */
    MEASURED(0.025d, 0.05d, 0.05d),

    /**
     * Low-penalty research preset for exploring larger accelerator regions.
     */
    AGGRESSIVE(0.01d, 0.02d, 0.10d);

    private final double inputBytePenalty;
    private final double outputBytePenalty;
    private final double avoidedIntermediateByteCredit;

    TransferCostPreset(double inputBytePenalty, double outputBytePenalty, double avoidedIntermediateByteCredit) {
        this.inputBytePenalty = inputBytePenalty;
        this.outputBytePenalty = outputBytePenalty;
        this.avoidedIntermediateByteCredit = avoidedIntermediateByteCredit;
    }

    /**
     * Returns the score penalty per byte entering an accelerator region.
     *
     * @return non-negative input byte penalty
     */
    public double inputBytePenalty() {
        return inputBytePenalty;
    }

    /**
     * Returns the score penalty per byte leaving an accelerator region.
     *
     * @return non-negative output byte penalty
     */
    public double outputBytePenalty() {
        return outputBytePenalty;
    }

    /**
     * Returns the score credit per internal intermediate byte kept inside an accelerator region.
     *
     * @return non-negative avoided intermediate byte credit
     */
    public double avoidedIntermediateByteCredit() {
        return avoidedIntermediateByteCredit;
    }
}
