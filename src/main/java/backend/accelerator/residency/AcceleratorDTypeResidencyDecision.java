package backend.accelerator.residency;

import backend.ComputeBackend;
import backend.accelerator.lowering.GpuLoweringUnsupportedReason;
import tensor.DataType;

import java.util.Objects;

/**
 * Role-specific dtype residency decision for accelerator-visible values.
 *
 * <p>This is a diagnostic/runtime planning contract. It describes whether a dtype can be represented
 * and whether it is legal for a native role; it does not imply broad native arithmetic support.</p>
 */
public record AcceleratorDTypeResidencyDecision(
        ComputeBackend backend,
        DataType dataType,
        String role,
        boolean residentRepresentable,
        boolean nativeInputLegal,
        boolean nativeOutputLegal,
        boolean nativeComputeLegal,
        GpuLoweringUnsupportedReason reason,
        String detail
) {
    public AcceleratorDTypeResidencyDecision {
        backend = Objects.requireNonNull(backend, "backend cannot be null");
        dataType = Objects.requireNonNull(dataType, "dataType cannot be null");
        role = role == null || role.isBlank() ? "unknown" : role;
        detail = detail == null ? "" : detail;
    }

    /**
     * Returns whether this decision rejected the dtype for the requested role.
     *
     * @return true when a stable unsupported reason is attached
     */
    public boolean rejected() {
        return reason != null;
    }
}
