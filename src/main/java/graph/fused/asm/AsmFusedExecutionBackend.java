package graph.fused.asm;

import graph.fused.FusedExecutionBackend;
import graph.fused.FusedExecutionPlan;
import graph.fused.PreparedFusedExecutable;

public final class AsmFusedExecutionBackend implements FusedExecutionBackend {
    private static final AsmPreparedFusedExecutableFactory FACTORY = new AsmPreparedFusedExecutableFactory();

    @Override
    public boolean supports(FusedExecutionPlan plan) {
        return plan != null;
    }

    @Override
    public PreparedFusedExecutable prepare(FusedExecutionPlan plan) {
        return FACTORY.create(plan.descriptor());
    }

    @Override
    public String name() {
        return "asm";
    }
}
