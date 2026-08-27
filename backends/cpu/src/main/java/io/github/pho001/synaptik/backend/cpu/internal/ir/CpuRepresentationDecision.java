package io.github.pho001.synaptik.backend.cpu.internal.ir;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Closed CPU-private facts for bounded external-read representation candidates and ordinary
 * selection. Retained facts describe direct, eligible single-copy, eligible disjoint-consumer
 * pair, and typed rejected pair forms with source-relative structural geometry only. They
 * deliberately exclude graph values, assignments, artifacts, timings, caches, and Runtime state.
 * Ordinary complete enabled preparation retains materialized forms as candidates but selects the
 * direct representation with {@link SelectionReason#DIRECT_MATERIALIZATION_UNPROVED}.
 */
public sealed interface CpuRepresentationDecision permits CpuRepresentationDecision.Variant,
        CpuRepresentationDecision.Rejection, CpuRepresentationDecision.Selection {
    /** Maximum complete topology/representation variants. */
    int MAX_VARIANTS = 2_368;
    /** Maximum existing plus representation decision facts. */
    int MAX_TOTAL_DECISION_FACTS = 2_753;

    /** Generated copy orchestration fixed during cold analysis. */
    enum CopyStrategy {
        /** One scalar generated copy call. */ SCALAR,
        /** Bounded parallel-scalar generated copy calls. */ PARALLEL_SCALAR
    }
    /** Closed reason why a variant is not selectable. */
    enum RejectionReason {
        /** Source access cannot be represented by a copy. */ INELIGIBLE_SOURCE,
        /** Repeated uses do not share compatible copy geometry. */ INCOMPATIBLE_REUSE,
        /** One represented instruction consumes both proposed copied sources. */ CO_CONSUMED_PAIR,
        /** Candidate exceeds its byte ceiling. */ BYTE_LIMIT,
        /** Candidate exceeds its resource ceiling. */ RESOURCE_LIMIT,
        /** An individual copy has no positive estimated benefit. */ NON_POSITIVE_INDIVIDUAL_BENEFIT,
        /** Candidate misses a configured diagnostic policy margin. */ INSUFFICIENT_POLICY_MARGIN,
        /** Candidate misses the topology comparison margin. */ INSUFFICIENT_TOPOLOGY_MARGIN,
        /** Candidate exceeds profitable code or live-value pressure. */ STRUCTURAL_PRESSURE,
        /** Checked comparison arithmetic is uncertain. */ UNCERTAIN_ARITHMETIC,
        /** Bounded enumeration is not complete. */ INCOMPLETE_ENUMERATION
    }
    /** Closed reason for the representation selected during ordinary CPU analysis. */
    enum SelectionReason {
        /** Materialization policy is disabled, so the selected representation is direct. */
        DIRECT_POLICY_DISABLED,
        /** The canonical direct representation is the safe fallback. */
        DIRECT_CANONICAL_FALLBACK,
        /** A tied comparison retains direct representation. */
        DIRECT_TIE,
        /** Incomplete or uncertain facts require the canonical direct representation. */
        DIRECT_UNCERTAINTY,
        /** Complete copied candidates exist, but end-to-end promotion remains unproved. */
        DIRECT_MATERIALIZATION_UNPROVED,
        /** A direct representation of a profitable topology is selected. */
        DIRECT_PROFITABLE_TOPOLOGY,
        /** A copied representation is explicitly selected by a compatible non-ordinary owner. */
        COPIED_PROFITABLE
    }

    /**
     * One stable unit-local consumer of a copied source.
     *
     * @param unitPosition zero-based represented computation-unit position
     * @param boundaryPosition zero-based boundary position within that unit
     * @param instructionUseCount positive semantic instruction-use count
     */
    record ConsumerPosition(int unitPosition, int boundaryPosition, long instructionUseCount) {
        /** Validates bounded non-negative positions and a positive use count. */
        public ConsumerPosition {
            if (unitPosition < 0 || unitPosition >= 8 || boundaryPosition < 0
                    || instructionUseCount <= 0) throw new IllegalArgumentException(
                            "CPU representation consumer is invalid");
        }
    }

    /**
     * Complete graph-identity-free identity of one copy and all compatible consumers.
     *
     * @param sourceBoundaryPosition stable complete-plan source-boundary position
     * @param dataType exact represented type
     * @param sourceCarrier original source carrier form
     * @param sourceBinding exact original source geometry
     * @param consumerCarrier contiguous workspace consumer carrier form
     * @param consumerBinding canonical dense consumer geometry
     * @param consumers ordered compatible consumers; copied defensively
     * @param instructionUseCount total semantic uses across {@code consumers}
     * @param reuseUnitCount number of distinct represented consumer units
     * @param reusePositionCount number of represented consumer positions
     * @param elementCount non-negative copied logical element count
     * @param byteCount non-negative exact represented byte count
     * @param workspaceRequirementId analysis-local workspace identity 8 or 9
     * @param workspaceBytes exact workspace byte count, equal to {@code byteCount}
     * @param workspaceAlignment positive workspace alignment in bytes
     * @param copyStrategy exact generated copy orchestration
     * @param copyStructuralKey exact generated affine-copy structural key
     * @param copySpecialization exact generated affine-copy specialization
     */
    record MaterializationIdentity(int sourceBoundaryPosition, DataType dataType,
            CpuKernelSpecialization.CarrierAccess sourceCarrier,
            CpuAccessPlan.Binding sourceBinding,
            CpuKernelSpecialization.CarrierAccess consumerCarrier,
            CpuAccessPlan.Binding consumerBinding, List<ConsumerPosition> consumers,
            long instructionUseCount, int reuseUnitCount, int reusePositionCount,
            long elementCount, long byteCount, int workspaceRequirementId,
            long workspaceBytes, long workspaceAlignment, CopyStrategy copyStrategy,
            CpuFusionDecision.StructuralKey copyStructuralKey,
            CpuKernelSpecialization copySpecialization) {
        /** Snapshots and validates one bounded ordered identity. */
        public MaterializationIdentity {
            Objects.requireNonNull(dataType, "dataType");
            Objects.requireNonNull(sourceCarrier, "sourceCarrier");
            Objects.requireNonNull(sourceBinding, "sourceBinding");
            Objects.requireNonNull(consumerCarrier, "consumerCarrier");
            Objects.requireNonNull(consumerBinding, "consumerBinding");
            consumers = List.copyOf(consumers);
            Objects.requireNonNull(copyStrategy, "copyStrategy");
            Objects.requireNonNull(copyStructuralKey, "copyStructuralKey");
            Objects.requireNonNull(copySpecialization, "copySpecialization");
            long uses = consumers.stream().mapToLong(ConsumerPosition::instructionUseCount)
                    .reduce(0, Math::addExact);
            long units = consumers.stream().map(ConsumerPosition::unitPosition).distinct().count();
            if (sourceBoundaryPosition < 0 || sourceBoundaryPosition >= 64
                    || consumers.isEmpty() || consumers.size() > 8
                    || instructionUseCount != uses || reuseUnitCount != units
                    || reusePositionCount != consumers.size() || elementCount < 0 || byteCount < 0
                    || workspaceRequirementId < 8 || workspaceRequirementId > 9
                    || workspaceBytes != byteCount || workspaceAlignment <= 0
                    || consumerCarrier != CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT) {
                throw new IllegalArgumentException("CPU materialization identity is inconsistent");
            }
        }
    }

    /**
     * Complete topology and its ordered zero-, one-, or two-copy representation.
     *
     * @param topology complete CPU 0008D topology identity
     * @param materializations ordered distinct-source copy identities; copied defensively
     */
    record VariantIdentity(CpuFusionDecision.CandidateIdentity topology,
            List<MaterializationIdentity> materializations) {
        /** Snapshots and validates stable distinct source order. */
        public VariantIdentity {
            Objects.requireNonNull(topology, "topology");
            materializations = List.copyOf(materializations);
            if (materializations.size() > 2) throw new IllegalArgumentException(
                    "CPU representation selects at most two copies");
            int previous = -1;
            for (MaterializationIdentity value : materializations) {
                if (value.sourceBoundaryPosition() <= previous) throw new IllegalArgumentException(
                        "CPU materializations must be in stable distinct source order");
                previous = value.sourceBoundaryPosition();
            }
        }
    }

    /**
     * One completely scored direct or copied variant.
     *
     * @param identity complete topology/representation identity
     * @param topologyScore checked CPU 0008D topology score
     * @param representationCost checked direct-or-copied representation cost
     * @param comparisonScore checked sum of topology and representation costs
     * @param copiedBytes combined copied workspace bytes
     * @param resourceCount exact candidate resource count
     * @param stableTopologyRank stable CPU 0008D topology rank
     * @param canonicalSplit whether the topology is canonical split
     * @param compatibilityBaseline whether the topology is the CPU 0008B baseline
     * @param selectedDirectCost direct-cost diagnostic for a copied variant, otherwise empty
     * @param selectedCopiedCost copied-cost diagnostic for a copied variant, otherwise empty
     * @param netBenefit direct-minus-copied diagnostic, otherwise empty
     * @param benefitBasisPoints diagnostic benefit ratio, otherwise empty
     */
    record Variant(VariantIdentity identity, long topologyScore, long representationCost,
            long comparisonScore, long copiedBytes, int resourceCount, int stableTopologyRank,
            boolean canonicalSplit, boolean compatibilityBaseline,
            Optional<Long> selectedDirectCost, Optional<Long> selectedCopiedCost,
            Optional<Long> netBenefit, Optional<Integer> benefitBasisPoints)
            implements CpuRepresentationDecision {
        /** Validates exact score composition and bounded facts. */
        public Variant {
            Objects.requireNonNull(identity, "identity");
            selectedDirectCost = Objects.requireNonNull(selectedDirectCost, "selectedDirectCost");
            selectedCopiedCost = Objects.requireNonNull(selectedCopiedCost, "selectedCopiedCost");
            netBenefit = Objects.requireNonNull(netBenefit, "netBenefit");
            benefitBasisPoints = Objects.requireNonNull(benefitBasisPoints, "benefitBasisPoints");
            boolean direct = identity.materializations().isEmpty();
            if (representationCost < 0 || comparisonScore != Math.addExact(topologyScore,
                    representationCost) || copiedBytes < 0 || resourceCount < 0
                    || resourceCount > 74 || stableTopologyRank < 0
                    || direct != selectedDirectCost.isEmpty()
                    || selectedDirectCost.isPresent() != selectedCopiedCost.isPresent()
                    || selectedDirectCost.isPresent() != netBenefit.isPresent()
                    || selectedDirectCost.isPresent() != benefitBasisPoints.isPresent()) {
                throw new IllegalArgumentException("CPU representation variant is inconsistent");
            }
        }
    }

    /**
     * Typed diagnostic rejection for one complete identity.
     *
     * @param identity complete rejected topology/representation identity
     * @param reason closed rejection reason
     */
    record Rejection(VariantIdentity identity, RejectionReason reason)
            implements CpuRepresentationDecision {
        /** Retains one non-null identity and closed reason. */
        public Rejection {
            Objects.requireNonNull(identity, "identity");
            Objects.requireNonNull(reason, "reason");
        }
    }

    /**
     * Final selected identity and exact direct canonical fallback.
     *
     * @param selected exact selected representation identity
     * @param canonicalDirect exact canonical direct fallback identity
     * @param stableRank non-negative stable selected rank
     * @param reason closed selection reason
     */
    record Selection(VariantIdentity selected, VariantIdentity canonicalDirect,
            int stableRank, SelectionReason reason) implements CpuRepresentationDecision {
        /** Validates one final selection. */
        public Selection {
            Objects.requireNonNull(selected, "selected");
            Objects.requireNonNull(canonicalDirect, "canonicalDirect");
            Objects.requireNonNull(reason, "reason");
            if (stableRank < 0 || reason != SelectionReason.COPIED_PROFITABLE
                    && reason != SelectionReason.DIRECT_PROFITABLE_TOPOLOGY
                    && reason != SelectionReason.DIRECT_POLICY_DISABLED
                    && reason != SelectionReason.DIRECT_MATERIALIZATION_UNPROVED
                    && !selected.equals(canonicalDirect)) throw new IllegalArgumentException(
                            "CPU representation selection is inconsistent");
        }
    }
}
