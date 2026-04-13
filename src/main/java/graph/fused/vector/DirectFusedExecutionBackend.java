package graph.fused.vector;

import backend.kernels.cpu.CpuComputeDType;
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
        boolean vectorCapable = switch (plan.computeContract().computeType()) {
            case F32, BF16_NATIVE -> FloatVector.SPECIES_PREFERRED.length() > 1;
            case F64 -> DoubleVector.SPECIES_PREFERRED.length() > 1;
            default -> false;
        };
        return vectorCapable
                && DirectFusedPlanSupport.supportsPlan(
                plan.descriptor().getPlan(),
                plan.computeContract().storageType()
        );
    }

    @Override
    public PreparedFusedExecutable prepare(FusedExecutionPlan plan) {
        return switch (plan.computeContract().storageType()) {
            case FLOAT32 -> new Float32PreparedFusedExecutable(plan.descriptor(), DirectLinearF32Program.lower(plan.descriptor().getPlan()));
            case FLOAT64 -> new Float64PreparedFusedExecutable(plan.descriptor(), DirectLinearF64Program.lower(plan.descriptor().getPlan()));
            default -> throw new IllegalStateException(
                    "Unsupported vector fused compute contract: storage="
                            + plan.computeContract().storageType()
                            + ", compute=" + plan.computeContract().computeType()
            );
        };
    }

    @Override
    public String name() {
        return "direct";
    }
}
