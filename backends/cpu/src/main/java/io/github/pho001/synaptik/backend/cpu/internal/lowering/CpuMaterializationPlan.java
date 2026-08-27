package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAffineCopyIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFusionDecision;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRepresentationDecision;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.List;
import java.util.Objects;

/**
 * Immutable plan for one complete external-read copy candidate into a dedicated contiguous
 * workspace. Source geometry and generated copy identity are fixed during analysis. Compatible
 * repeated or cross-unit consumers share this one plan and therefore one copy operation.
 * Consumer positions are stable unit/boundary coordinates and contain no graph or physical
 * resource identity. Ordinary preparation retains such plans as candidate data and selects
 * direct execution; an explicitly selected compatible future plan can finalize and execute them.
 *
 * @param sourceBoundaryIndex stable complete-plan boundary position
 * @param dataType exact represented type copied without conversion
 * @param sourceCarrier original generated carrier form
 * @param sourceBinding proved original read geometry
 * @param consumerBinding canonical dense zero-offset read geometry
 * @param consumers ordered compatible computation consumers; copied defensively
 * @param elementCount non-negative logical element count
 * @param byteCount exact represented byte count
 * @param workspaceRequirementId fixed analysis-local identity 8 or 9
 * @param byteAlignment exact type-width workspace alignment
 * @param useCount positive total semantic instruction-use count
 * @param expectedRunCount positive policy run estimate
 * @param directCost checked direct cost for all uses
 * @param copyCost checked fixed-plus-element copy cost for one run
 * @param contiguousCost checked contiguous cost for one use and one run
 * @param copiedTotalCost checked repeated copy-plus-consumer cost
 * @param netBenefit checked direct-minus-copied diagnostic cost, which may be non-positive
 * @param benefitBasisPoints checked diagnostic benefit ratio in basis points
 * @param copyStrategy scalar or parallel-scalar generated copy orchestration
 * @param copyIr existing structural CONTIGUOUS affine-copy form
 * @param copySpecialization exact existing affine-copy specialization
 * @param affineAddressPairs alternating source and dense-result element addresses
 */
public record CpuMaterializationPlan(int sourceBoundaryIndex, DataType dataType,
        CpuKernelSpecialization.CarrierAccess sourceCarrier,
        CpuAccessPlan.Binding sourceBinding, CpuAccessPlan.Binding consumerBinding,
        List<CpuRepresentationDecision.ConsumerPosition> consumers,
        long elementCount, long byteCount, int workspaceRequirementId, long byteAlignment,
        long useCount, long expectedRunCount, long directCost, long copyCost,
        long contiguousCost, long copiedTotalCost, long netBenefit, int benefitBasisPoints,
        CpuRepresentationDecision.CopyStrategy copyStrategy, CpuAffineCopyIr copyIr,
        CpuKernelSpecialization copySpecialization, long[] affineAddressPairs) {
    /** Validates one complete copy-candidate plan. */
    public CpuMaterializationPlan {
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(sourceCarrier, "sourceCarrier");
        Objects.requireNonNull(sourceBinding, "sourceBinding");
        Objects.requireNonNull(consumerBinding, "consumerBinding");
        consumers = List.copyOf(consumers);
        Objects.requireNonNull(copyStrategy, "copyStrategy");
        Objects.requireNonNull(copyIr, "copyIr");
        Objects.requireNonNull(copySpecialization, "copySpecialization");
        affineAddressPairs = affineAddressPairs.clone();
        long countedUses = consumers.stream().mapToLong(
                CpuRepresentationDecision.ConsumerPosition::instructionUseCount)
                .reduce(0, Math::addExact);
        if (sourceBoundaryIndex < 0 || consumers.isEmpty() || consumers.size() > 8
                || useCount != countedUses || elementCount < 0
                || byteCount != Math.multiplyExact(elementCount, dataType.byteWidth())
                || workspaceRequirementId < 8 || workspaceRequirementId > 9
                || byteAlignment != dataType.byteWidth() || expectedRunCount <= 0
                || directCost < 0 || copyCost < 0 || contiguousCost < 0
                || copiedTotalCost < 0
                || copiedTotalCost != Math.subtractExact(directCost, netBenefit)
                || sourceBinding.plan().accessKind() != CpuAccessPlan.AccessKind.READ
                || consumerBinding.plan().regime() != CpuAccessPlan.Regime.DENSE_LINEAR
                || consumerBinding.baseElementOffset() != 0
                || consumerBinding.referencedElementSpan() != elementCount
                || copyIr.dataType() != dataType
                || !copyIr.sourceAccess().equals(sourceBinding.plan())
                || !copyIr.resultAccess().equals(writePlan(consumerBinding.plan()))
                || copySpecialization.materializedSourcePosition() != -1
                || !copySpecialization.carrierPattern().equals(List.of(sourceCarrier,
                        CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT))
                || affineAddressPairs.length != Math.multiplyExact(elementCount, 2)) {
            throw new IllegalArgumentException("invalid CPU materialization plan");
        }
    }

    /** Returns copy geometry without exposing retained mutable state.
     * @return a defensive copy of alternating source/result addresses
     */
    @Override public long[] affineAddressPairs() { return affineAddressPairs.clone(); }

    /** Returns the legacy stable diagnostic text without granting selection authority.
     * @return stable non-null compatibility text retained by existing diagnostics
     */
    public String selectionReason() { return "selected: closed profitable representation"; }

    /** Returns the graph-identity-free retained identity.
     * @return complete immutable structural copy identity
     */
    public CpuRepresentationDecision.MaterializationIdentity identity() {
        return new CpuRepresentationDecision.MaterializationIdentity(sourceBoundaryIndex,
                dataType, sourceCarrier, sourceBinding,
                CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT, consumerBinding, consumers,
                useCount, (int) consumers.stream().map(
                        CpuRepresentationDecision.ConsumerPosition::unitPosition).distinct().count(),
                consumers.size(), elementCount, byteCount, workspaceRequirementId, byteCount,
                byteAlignment, copyStrategy,
                CpuFusionDecision.StructuralKey.fromHex(copyIr.structuralKey()),
                copySpecialization);
    }

    private static CpuAccessPlan writePlan(CpuAccessPlan source) {
        return new CpuAccessPlan(CpuAccessPlan.AccessKind.WRITE, source.regime(),
                source.iterationRank(), source.axisRoles(), source.contiguousSuffix());
    }
}
