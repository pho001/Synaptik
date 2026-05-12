package config.compile;

/**
 * Failure behavior for accelerator backend planning.
 */
public enum BackendPlanningFailurePolicy {
    OPTIONAL,
    REQUIRE_ACCELERATOR_REGION,
    REQUIRE_ALL_EXPLICIT_INTENTS
}
