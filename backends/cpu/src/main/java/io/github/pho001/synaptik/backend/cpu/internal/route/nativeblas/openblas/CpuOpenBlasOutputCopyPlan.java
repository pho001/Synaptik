package io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAffineCopyIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.Objects;

/**
 * Immutable OpenBLAS-route copy from a canonical native result workspace to the represented
 * logical output. This is deliberately distinct from external-read input materialization.
 *
 * @param outputBoundaryPosition stable logical-output boundary position
 * @param dataType exact represented type copied without conversion
 * @param sourceBinding canonical dense workspace read geometry
 * @param destinationCarrier cold-proved logical-output carrier
 * @param destinationBinding proved injective logical-output write geometry
 * @param elementCount positive logical output element count
 * @param byteCount exact canonical workspace byte count
 * @param workspaceRequirementId analysis-local workspace identity
 * @param byteAlignment exact type-width alignment
 * @param copyIr exact existing affine-copy IR
 * @param copySpecialization exact generated copy specialization
 * @param affineAddressPairs caller-owned alternating workspace-source and logical-output element
 *     addresses; cloned during construction and on access
 */
public record CpuOpenBlasOutputCopyPlan(int outputBoundaryPosition, DataType dataType,
        CpuAccessPlan.Binding sourceBinding,
        CpuKernelSpecialization.CarrierAccess destinationCarrier,
        CpuAccessPlan.Binding destinationBinding, long elementCount, long byteCount,
        int workspaceRequirementId, long byteAlignment, CpuAffineCopyIr copyIr,
        CpuKernelSpecialization copySpecialization, long[] affineAddressPairs) {
    /**
     * Validates and snapshots the complete route-local generated copy contract.
     *
     * @throws NullPointerException if a required type, binding, carrier, IR, or specialization is
     *     {@code null}
     * @throws IllegalArgumentException if direction, boundary identity, data type, geometry,
     *     workspace identity/alignment, generated specialization, or address cardinality disagrees
     * @throws ArithmeticException if exact byte or address cardinality arithmetic overflows
     */
    public CpuOpenBlasOutputCopyPlan {
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(sourceBinding, "sourceBinding");
        Objects.requireNonNull(destinationCarrier, "destinationCarrier");
        Objects.requireNonNull(destinationBinding, "destinationBinding");
        Objects.requireNonNull(copyIr, "copyIr");
        Objects.requireNonNull(copySpecialization, "copySpecialization");
        affineAddressPairs = affineAddressPairs.clone();
        if (outputBoundaryPosition != 2 || elementCount <= 0
                || byteCount != Math.multiplyExact(elementCount, dataType.byteWidth())
                || workspaceRequirementId < 8 || workspaceRequirementId > 10
                || byteAlignment != dataType.byteWidth()
                || sourceBinding.plan().accessKind() != CpuAccessPlan.AccessKind.READ
                || sourceBinding.plan().regime() != CpuAccessPlan.Regime.DENSE_LINEAR
                || sourceBinding.baseElementOffset() != 0
                || sourceBinding.elementCount() != elementCount
                || sourceBinding.start() != 0 || sourceBinding.end() != elementCount
                || sourceBinding.referencedElementSpan() != elementCount
                || destinationBinding.plan().accessKind() != CpuAccessPlan.AccessKind.WRITE
                || destinationBinding.elementCount() != elementCount
                || destinationBinding.start() != 0 || destinationBinding.end() != elementCount
                || copyIr.dataType() != dataType
                || !copyIr.sourceAccess().equals(sourceBinding.plan())
                || !copyIr.resultAccess().equals(destinationBinding.plan())
                || !copySpecialization.carrierPattern().equals(java.util.List.of(
                        CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT,
                        destinationCarrier))
                || copySpecialization.materializedSourcePosition() != -1
                || affineAddressPairs.length != Math.multiplyExact(elementCount, 2)) {
            throw new IllegalArgumentException("invalid OpenBLAS output-copy plan");
        }
    }

    /**
     * Returns the immutable plan's address table without exposing its retained array.
     *
     * @return a new array of alternating workspace-source and logical-output element addresses
     */
    @Override public long[] affineAddressPairs() { return affineAddressPairs.clone(); }
}
