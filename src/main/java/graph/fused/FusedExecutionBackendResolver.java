package graph.fused;

import config.runtime.FusedExecutionPolicy;
import config.runtime.FusedPrimaryBackend;
import operations.Operation;
import graph.fused.asm.AsmFusedExecutionBackend;
import graph.fused.vector.DirectFusedExecutionBackend;

import java.util.List;

public final class FusedExecutionBackendResolver {
    private final List<FusedExecutionBackend> backends = List.of(
            new DirectFusedExecutionBackend(),
            new AsmFusedExecutionBackend()
    );

    public PreparedFusedExecutable resolve(FusedExecutionPlan plan, FusedExecutionPolicy policy) {
        FusedExecutionPolicy effectivePolicy = policy == null ? FusedExecutionPolicy.defaultsInference() : policy;
        FusedExecutionBackend direct = backends.get(0);
        FusedExecutionBackend asm = backends.get(1);

        boolean directSupported = direct.supports(plan);
        boolean asmSupported = asm.supports(plan);
        boolean directPreferred = directSupported && shouldPreferDirect(plan, effectivePolicy);

        if (effectivePolicy.primaryBackend() == FusedPrimaryBackend.DIRECT_VECTOR) {
            if (directPreferred) {
                return direct.prepare(plan);
            }
            if (asmSupported && effectivePolicy.allowBackendFallback()) {
                return asm.prepare(plan);
            }
            if (directSupported) {
                return direct.prepare(plan);
            }
            if (asmSupported) {
                return asm.prepare(plan);
            }
        } else {
            if (asmSupported) {
                return asm.prepare(plan);
            }
            if (directSupported && effectivePolicy.allowBackendFallback()) {
                return direct.prepare(plan);
            }
            if (directSupported) {
                return direct.prepare(plan);
            }
        }
        throw new IllegalStateException("No fused execution backend supports plan for " + plan.descriptor().getExpression());
    }

    private boolean shouldPreferDirect(FusedExecutionPlan plan, FusedExecutionPolicy policy) {
        if (plan == null || policy == null) {
            return true;
        }
        if (plan.outputLength() < plan.cpuVectorMinSize()) {
            return false;
        }
        if (!policy.preferDirectForCompareSelect() && containsCompareSelect(plan)) {
            return false;
        }
        return true;
    }

    private static boolean containsCompareSelect(FusedExecutionPlan plan) {
        for (var node : plan.descriptor().getPlan().nodes()) {
            Operation.OpType type = node.opType();
            switch (type) {
                case GT, GE, LT, LE, EQ, NE, LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT, WHERE -> {
                    return true;
                }
                default -> {
                }
            }
        }
        return false;
    }
}
