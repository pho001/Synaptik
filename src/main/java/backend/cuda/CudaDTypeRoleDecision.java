package backend.cuda;

import tensor.DataType;

import java.util.Objects;

/**
 * One CUDA dtype role decision.
 */
public record CudaDTypeRoleDecision(
        DataType dataType,
        CudaDTypeRole role,
        boolean supported,
        String reasonCode,
        String detail
) {
    public CudaDTypeRoleDecision {
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(role, "role");
        reasonCode = reasonCode == null ? "" : reasonCode.strip();
        detail = detail == null ? "" : detail.strip();
    }
}
