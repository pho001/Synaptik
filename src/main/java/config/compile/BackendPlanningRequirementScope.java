package config.compile;

/**
 * Scope used when backend planning has required accelerator work.
 */
public enum BackendPlanningRequirementScope {
    ANY_TARGET,
    EACH_TARGET,
    ALL_EXPLICIT_INTENTS
}
