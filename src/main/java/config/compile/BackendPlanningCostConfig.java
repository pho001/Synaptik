package config.compile;

/**
 * Cost configuration for backend ownership planning.
 */
public record BackendPlanningCostConfig(
        PlanningCostProfile planningCostProfile
) {
    public BackendPlanningCostConfig {
        planningCostProfile = planningCostProfile == null ? PlanningCostProfile.conservative() : planningCostProfile;
    }

    public static BackendPlanningCostConfig conservative() {
        return new BackendPlanningCostConfig(PlanningCostProfile.conservative());
    }

    public static BackendPlanningCostConfig measuredTransfer() {
        return new BackendPlanningCostConfig(PlanningCostProfile.measuredTransfer());
    }
}
