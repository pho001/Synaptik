package backend.cpu.fused.exec;

import backend.cpu.fused.plan.FusedExecutionPlan;

public interface FusedExecutionBackend {
    boolean supports(FusedExecutionPlan plan);

    PreparedFusedExecutable prepare(FusedExecutionPlan plan);

    String name();
}
