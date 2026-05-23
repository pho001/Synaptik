package backend.cpu.fused.exec;

import backend.cpu.fused.asm.AsmPreparedFusedExecutableFactory;
import backend.cpu.fused.plan.FusedExecutionPlan;
import config.runtime.FusedExecutionPolicy;

/**
 * Prepares the executable for the single production CPU fused compiled path.
 */
public final class FusedExecutablePreparer {
    private final AsmCompiler asmCompiler;

    public FusedExecutablePreparer() {
        this(new AsmPreparedFusedExecutableFactory()::create);
    }

    FusedExecutablePreparer(AsmCompiler asmCompiler) {
        if (asmCompiler == null) {
            throw new IllegalArgumentException("asmCompiler cannot be null");
        }
        this.asmCompiler = asmCompiler;
    }

    public PreparedFusedExecutable prepare(FusedExecutionPlan plan, FusedExecutionPolicy policy) {
        if (plan == null) {
            throw new IllegalArgumentException("plan cannot be null");
        }
        FusedExecutionPolicy effectivePolicy = policy == null
                ? FusedExecutionPolicy.defaultsInference()
                : policy;
        boolean asmOnlyMemorySegmentContract = plan.descriptor().getNumericContract().usesMemorySegmentStorage();
        try {
            return asmCompiler.create(plan);
        } catch (RuntimeException ex) {
            if (asmOnlyMemorySegmentContract) {
                throw memorySegmentAsmPreparationFailure(ex);
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

    private static IllegalStateException memorySegmentAsmPreparationFailure(RuntimeException cause) {
        return new IllegalStateException(
                "CPU_MEMORY_SEGMENT fused execution is ASM-only; "
                        + "refusing Java-array interpreter fallback after ASM preparation failure.",
                cause
        );
    }

    @FunctionalInterface
    interface AsmCompiler {
        PreparedFusedExecutable create(FusedExecutionPlan plan);
    }
}
