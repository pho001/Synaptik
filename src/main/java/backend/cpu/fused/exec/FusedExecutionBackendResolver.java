package backend.cpu.fused.exec;

import backend.cpu.fused.asm.AsmFusedExecutionBackend;
import backend.cpu.fused.plan.FusedExecutionPlan;
import config.runtime.FusedExecutionPolicy;

/**
 * Selects the concrete backend for a fused CPU execution plan.
 *
 * <p>The resolver currently routes all supported plans to the ASM backend and
 * fails fast when no executable backend accepts the plan.</p>
 */
public final class FusedExecutionBackendResolver {
    private final FusedExecutionBackend asm = new AsmFusedExecutionBackend();

    /**
     * Resolves and prepares an executable according to the supplied policy.
     */
    public PreparedFusedExecutable resolve(FusedExecutionPlan plan, FusedExecutionPolicy policy) {
        if (policy == null) {
            policy = FusedExecutionPolicy.defaultsInference();
        }
        if (asm.supports(plan)) {
            try {
                return asm.prepare(plan);
            } catch (RuntimeException | LinkageError e) {
                if (!policy.allowBackendFallback()) {
                    throw e;
                }
                return new InterpretedPreparedFusedExecutable(plan.descriptor().getPlan(), plan.descriptor().getPrecisionMode());
            }
        }
        if (policy.allowBackendFallback()) {
            return new InterpretedPreparedFusedExecutable(plan.descriptor().getPlan(), plan.descriptor().getPrecisionMode());
        }
        throw new IllegalStateException("ASM fused execution backend does not support plan for " + plan.descriptor().getExpression());
    }
}
