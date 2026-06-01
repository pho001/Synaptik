package backend.cpu1.prepare.dispatch;

/**
 * Coarse prepare-time cost bucket for cpu1 dispatch decisions.
 */
public enum Cpu1CostClass {
    CHEAP_ELEMENTWISE,
    EXPENSIVE_ELEMENTWISE,
    REDUCTION,
    MATMUL
}
