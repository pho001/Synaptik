package planning.memory;

/**
 * Role assigned to a tensor storage owner during memory planning.
 */
public enum MemoryRole {
    LEAF,
    FORWARD_TEMP,
    SAVED_FORWARD,
    GRADIENT_TARGET,
    BACKWARD_TEMP,
    VIEW_ALIAS
}
