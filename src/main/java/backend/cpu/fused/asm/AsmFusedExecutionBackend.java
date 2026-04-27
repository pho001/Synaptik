package backend.cpu.fused.asm;

import backend.cpu.fused.exec.FusedExecutionBackend;
import backend.cpu.fused.plan.FusedExecutionPlan;
import backend.cpu.fused.exec.PreparedFusedExecutable;

public final class AsmFusedExecutionBackend implements FusedExecutionBackend {
    private static final AsmPreparedFusedExecutableFactory FACTORY = new AsmPreparedFusedExecutableFactory();

    @Override
    public boolean supports(FusedExecutionPlan plan) {
        return plan != null;
    }

    @Override
    public PreparedFusedExecutable prepare(FusedExecutionPlan plan) {
        return FACTORY.create(plan);
    }

    @Override
    public String name() {
        return "asm";
    }
}
