package backend.cpu.fused.exec;

import backend.cpu.fused.asm.AsmPreparedFusedExecutableFactory;
import backend.cpu.fused.plan.FusedExecutionPlan;
import config.runtime.FusedExecutionPolicy;

/**
 * Prepares the executable for the single production CPU fused compiled path.
 */
public final class FusedExecutablePreparer {
    private final AsmPreparedFusedExecutableFactory asmFactory = new AsmPreparedFusedExecutableFactory();

    public PreparedFusedExecutable prepare(FusedExecutionPlan plan, FusedExecutionPolicy policy) {
        if (plan == null) {
            throw new IllegalArgumentException("plan cannot be null");
        }
        FusedExecutionPolicy effectivePolicy = policy == null
                ? FusedExecutionPolicy.defaultsInference()
                : policy;
        boolean memorySegmentStorage = plan.descriptor().getNumericContract().usesMemorySegmentStorage();
        try {
            return asmFactory.create(plan);
        } catch (RuntimeException ex) {
            if (memorySegmentStorage) {
                throw new IllegalStateException(
                        "CPU fused MemorySegment ASM preparation failed; refusing Java-array interpreter fallback.",
                        ex
                );
            }
            if (!effectivePolicy.allowBackendFallback()) {
                throw ex;
            }
            return new InterpretedPreparedFusedExecutable(
                    plan.descriptor().getPlan(),
                    plan.descriptor().getNumericContract(),
                    plan.descriptor().getApproximationContract()
            );
        }
    }
}
