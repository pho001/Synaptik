package config.compile;

/**
 * Failure behavior for accelerator backend planning.
 */
public enum BackendPlanningFailurePolicy {
    OPTIONAL,
    REQUIRE_ACCELERATOR_PARTITION,
    REQUIRE_ALL_EXPLICIT_INTENTS
}
