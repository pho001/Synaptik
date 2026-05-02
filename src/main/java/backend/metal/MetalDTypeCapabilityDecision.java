package backend.metal;

import tensor.DataType;

/**
 * Role-specific Metal dtype capability decision.
 *
 * @param role dtype role being checked
 * @param dataType tensor dtype
 * @param supported whether this role is currently legal for Metal native execution
 * @param storageRepresentable whether runtime storage/residency can represent the dtype bytes
 * @param nativeCompute whether Metal can compute this dtype natively in the current bridge
 * @param nativeOutput whether Metal can publish this dtype natively in the current bridge
 * @param reasonCode stable reason code
 * @param detail readable diagnostic detail
 */
public record MetalDTypeCapabilityDecision(
        MetalDTypeRole role,
        DataType dataType,
        boolean supported,
        boolean storageRepresentable,
        boolean nativeCompute,
        boolean nativeOutput,
        MetalDTypeReasonCode reasonCode,
        String detail
) {
    public MetalDTypeCapabilityDecision {
        if (role == null) {
            throw new IllegalArgumentException("role cannot be null");
        }
        if (dataType == null) {
            throw new IllegalArgumentException("dataType cannot be null");
        }
        reasonCode = reasonCode == null ? MetalDTypeReasonCode.UNSUPPORTED_OPERATION_DTYPE : reasonCode;
        detail = detail == null ? "" : detail;
    }
}
