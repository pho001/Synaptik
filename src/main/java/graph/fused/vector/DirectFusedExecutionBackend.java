package graph.fused.vector;

import backend.kernels.cpu.CpuComputeMode;
import graph.fused.FusedExecutionBackend;
import graph.fused.FusedExecutionPlan;
import graph.fused.PreparedFusedExecutable;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;

public final class DirectFusedExecutionBackend implements FusedExecutionBackend {
    @Override
    public boolean supports(FusedExecutionPlan plan) {
        if (plan == null) {
            return false;
        }
        return switch (plan.computeMode()) {
            case BF16_F32_COMPUTE, F32 -> FloatVector.SPECIES_PREFERRED.length() > 1;
            case F64 -> DoubleVector.SPECIES_PREFERRED.length() > 1;
            default -> false;
        };
    }

    @Override
    public PreparedFusedExecutable prepare(FusedExecutionPlan plan) {
        return switch (plan.computeMode()) {
            case BF16_F32_COMPUTE -> new BFloat16PreparedFusedExecutable(plan.descriptor());
            case F32 -> new Float32PreparedFusedExecutable(plan.descriptor());
            case F64 -> new Float64PreparedFusedExecutable(plan.descriptor());
            default -> throw new IllegalStateException("Unsupported vector fused compute mode: " + plan.computeMode());
        };
    }

    @Override
    public String name() {
        return "direct";
    }
}
