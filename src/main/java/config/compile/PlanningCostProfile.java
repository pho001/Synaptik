package config.compile;

import config.optimizer.MetalTransferModel;

/**
 * Compile-time cost facts consumed by backend ownership planning.
 */
public record PlanningCostProfile(
        MetalTransferModel metalTransferModel
) {
    public PlanningCostProfile {
        metalTransferModel = metalTransferModel == null ? MetalTransferModel.CONSERVATIVE : metalTransferModel;
    }

    public static PlanningCostProfile conservative() {
        return new PlanningCostProfile(MetalTransferModel.CONSERVATIVE);
    }

    public static PlanningCostProfile measuredTransfer() {
        return new PlanningCostProfile(MetalTransferModel.MEASURED);
    }

    public static PlanningCostProfile aggressiveTransfer() {
        return new PlanningCostProfile(MetalTransferModel.AGGRESSIVE);
    }
}
