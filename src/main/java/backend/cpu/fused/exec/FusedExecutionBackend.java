package backend.cpu.fused.exec;

import backend.cpu.fused.plan.FusedExecutionPlan;

/**
 * Internal SPI for executable backends that run lowered CPU fused operations.
 */
public interface FusedExecutionBackend {
    /**
     * Returns whether this backend can prepare the supplied fused execution plan.
     */
    boolean supports(FusedExecutionPlan plan);

    /**
     * Compiles or otherwise prepares an executable for the supplied fused plan.
     */
    PreparedFusedExecutable prepare(FusedExecutionPlan plan);

    /**
     * Returns a short backend identifier for diagnostics and policy decisions.
     */
    String name();
}
