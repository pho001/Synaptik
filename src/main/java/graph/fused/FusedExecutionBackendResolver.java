package graph.fused;

import config.runtime.FusedExecutionPolicy;
import graph.fused.asm.AsmFusedExecutionBackend;

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
