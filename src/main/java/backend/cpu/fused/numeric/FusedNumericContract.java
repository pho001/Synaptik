package backend.cpu.fused.numeric;

import java.util.Objects;

/**
 * Numeric correctness contract consumed by fused ASM generation and interpreted fallback.
 */
public record FusedNumericContract(
        FusedStorageKind inputStorageKind,
        FusedStorageKind outputStorageKind,
        FusedValueLane inputValueLane,
        FusedComputeKind computeKind,
        FusedValueLane outputValueLane
) {
    public FusedNumericContract {
        inputStorageKind = Objects.requireNonNull(inputStorageKind, "inputStorageKind cannot be null");
        outputStorageKind = Objects.requireNonNull(outputStorageKind, "outputStorageKind cannot be null");
        inputValueLane = Objects.requireNonNull(inputValueLane, "inputValueLane cannot be null");
        computeKind = Objects.requireNonNull(computeKind, "computeKind cannot be null");
        outputValueLane = Objects.requireNonNull(outputValueLane, "outputValueLane cannot be null");
    }

    public boolean usesFloatCompute() {
        return computeKind == FusedComputeKind.F32;
    }

    public boolean usesDoubleCompute() {
        return computeKind == FusedComputeKind.F64;
    }

    public boolean usesMemorySegmentStorage() {
        return inputStorageKind == FusedStorageKind.CPU_MEMORY_SEGMENT
                || outputStorageKind == FusedStorageKind.CPU_MEMORY_SEGMENT;
    }

    public String signatureToken() {
        return inputStorageKind + ":" + inputValueLane
                + "->" + computeKind
                + "->" + outputStorageKind + ":" + outputValueLane;
    }
}
