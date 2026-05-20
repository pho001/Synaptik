package config.compile;

/**
 * Compile-time cost facts consumed by backend ownership planning.
 */
public record PlanningCostProfile(
        TransferCostPreset transferCostPreset
) {
    public PlanningCostProfile {
        transferCostPreset = transferCostPreset == null ? TransferCostPreset.CONSERVATIVE : transferCostPreset;
    }

    public static PlanningCostProfile conservative() {
        return new PlanningCostProfile(TransferCostPreset.CONSERVATIVE);
    }

    public static PlanningCostProfile measuredTransfer() {
        return new PlanningCostProfile(TransferCostPreset.MEASURED);
    }

    public static PlanningCostProfile aggressiveTransfer() {
        return new PlanningCostProfile(TransferCostPreset.AGGRESSIVE);
    }
}
