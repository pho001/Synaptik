package graph.optimizer.memory;

public enum MemoryRole {
    LEAF,
    FORWARD_TEMP,
    SAVED_FORWARD,
    GRADIENT_TARGET,
    BACKWARD_TEMP,
    VIEW_ALIAS
}
