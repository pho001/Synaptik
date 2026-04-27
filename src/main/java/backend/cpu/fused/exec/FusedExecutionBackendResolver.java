package backend.cpu.fused.exec;

import backend.cpu.fused.asm.AsmFusedExecutionBackend;
import backend.cpu.fused.plan.FusedExecutionPlan;
import config.runtime.FusedExecutionPolicy;

public final class FusedExecutionBackendResolver {
    private final FusedExecutionBackend asm = new AsmFusedExecutionBackend();

    public PreparedFusedExecutable resolve(FusedExecutionPlan plan, FusedExecutionPolicy policy) {
        if (policy == null) {
            policy = FusedExecutionPolicy.defaultsInference();
        }
        if (asm.supports(plan)) {
            return asm.prepare(plan);
        }
        throw new IllegalStateException("ASM fused execution backend does not support plan for " + plan.descriptor().getExpression());
    }
}
