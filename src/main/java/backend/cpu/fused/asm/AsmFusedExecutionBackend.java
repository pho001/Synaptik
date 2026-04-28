package backend.cpu.fused.asm;

import backend.cpu.fused.exec.FusedExecutionBackend;
import backend.cpu.fused.plan.FusedExecutionPlan;
import backend.cpu.fused.exec.PreparedFusedExecutable;

/**
 * ASM-backed fused execution backend.
 *
 * <p>This internal backend generates JVM bytecode for fused expressions and
 * delegates class caching to {@link AsmPreparedFusedExecutableFactory}.</p>
 */
public final class AsmFusedExecutionBackend implements FusedExecutionBackend {
    private static final AsmPreparedFusedExecutableFactory FACTORY = new AsmPreparedFusedExecutableFactory();

    /**
     * Returns whether the ASM backend can attempt to prepare the plan.
     */
    @Override
    public boolean supports(FusedExecutionPlan plan) {
        return plan != null;
    }

    /**
     * Generates or reuses bytecode for the fused plan.
     */
    @Override
    public PreparedFusedExecutable prepare(FusedExecutionPlan plan) {
        return FACTORY.create(plan);
    }

    /**
     * Returns the backend id used in diagnostics.
     */
    @Override
    public String name() {
        return "asm";
    }
}
