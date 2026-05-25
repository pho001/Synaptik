package backend.cpu.plan;

import tensor.DataType;

import java.util.Objects;

public record ResolvedCpuComputeContract(
        DataType storageType,
        CpuComputeDType computeType,
        CpuExecutionBackend backend,
        CpuAccumulateDType accumulateType
) {
    public ResolvedCpuComputeContract {
        Objects.requireNonNull(storageType, "storageType cannot be null");
        computeType = computeType == null ? CpuComputeDType.F64 : computeType;
        backend = backend == null ? CpuExecutionBackend.CPU_GENERIC : backend;
        accumulateType = accumulateType == null ? CpuAccumulateDType.NONE : accumulateType;
    }
}
