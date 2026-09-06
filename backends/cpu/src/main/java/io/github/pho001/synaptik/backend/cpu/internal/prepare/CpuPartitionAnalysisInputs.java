package io.github.pho001.synaptik.backend.cpu.internal.prepare;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import java.util.List;

/**
 * Checked immutable CPU inputs that do not contain graph semantics or instance resources.
 *
 * @param loweringManifestEnabled whether cold diagnostics should retain a lowering manifest
 * @param carrierPattern non-null ordered direct carrier forms for the derived boundaries;
 *     membership is snapshotted and contains no physical carrier object; an empty list selects
 *     one exact {@code MEMORY_SEGMENT} form per derived boundary
 * @param portableExecution non-null immutable cold compute-preference and parallelism inputs
 * @param materializationPolicy non-null dimensionless cold candidate-comparison policy; disabled
 *     retains only direct forms, while enabled ordinary analysis retains complete bounded copied
 *     candidates but still selects direct execution
 * @param conv2dMaterializedSuffixUnit whether this input belongs to the sole tagged pointwise
 *     suffix unit of a two-unit Conv2d plan
 * @param partialReductionEvidence non-null diagnostic partial-reduction evidence; current
 *     production preparation treats it as non-authoritative and remains fail-closed
 */
public record CpuPartitionAnalysisInputs(boolean loweringManifestEnabled,
        List<CarrierAccess> carrierPattern, PortableExecutionConfig portableExecution,
        MaterializationPolicy materializationPolicy, boolean conv2dMaterializedSuffixUnit,
        PartialReductionEvidence partialReductionEvidence)
        implements BackendAnalysisInputs {
    /**
     * Compatibility constructor that deliberately admits no partial route.
     *
     * @param loweringManifestEnabled whether diagnostics are retained
     * @param carrierPattern requested carrier forms
     * @param portableExecution cold execution limits
     * @param materializationPolicy cold materialization policy
     * @param conv2dMaterializedSuffixUnit tagged Conv2d suffix marker
     */
    public CpuPartitionAnalysisInputs(boolean loweringManifestEnabled,
            List<CarrierAccess> carrierPattern, PortableExecutionConfig portableExecution,
            MaterializationPolicy materializationPolicy, boolean conv2dMaterializedSuffixUnit) {
        this(loweringManifestEnabled, carrierPattern, portableExecution, materializationPolicy,
                conv2dMaterializedSuffixUnit, PartialReductionEvidence.NONE);
    }
    /**
     * Default input: manifest and materialization disabled, scalar single-thread execution, and
     * one exact-segment carrier selected for every boundary derived by lowering.
     */
    public static final CpuPartitionAnalysisInputs DEFAULT = new CpuPartitionAnalysisInputs(false,
            List.of(),
            PortableExecutionConfig.DEFAULT, MaterializationPolicy.DISABLED, false,
            PartialReductionEvidence.NONE);

    /**
     * Immutable diagnostic record for the deliberately narrow partial-reduction route.
     *
     * <p>This record is not a production admission credential.  Until a separately scoped
     * trusted reader verifies a complete frozen evidence root, the current selector ignores even
     * a syntactically passing instance and keeps the whole-cell route.</p>
     *
     * @param passed claimed matching-evidence outcome; it cannot authorize the current selector
     * @param kind exact ordinary aggregate kind authorized by a passing row
     * @param dataType exact represented primitive type authorized by a passing row
     * @param form exact aggregate form authorized by a passing row
     * @param partialCount exact prepared count, restricted to two or four when passed
     */
    public record PartialReductionEvidence(boolean passed, CpuAggregateIr.Kind kind,
            DataType dataType, CpuAggregateIr.Form form, int partialCount) {
        /** Fail-closed default with no admitted partial count. */
        public static final PartialReductionEvidence NONE = new PartialReductionEvidence(false,
                null, null, null, 0);
        /**
         * Compatibility constructor for the former untyped evidence shape.
         *
         * <p>Only the fail-closed {@link #NONE} form is meaningful without an aggregate identity.
         * A passing row must name the exact kind, type, and form that it authorizes.</p>
         *
         * @param passed whether the evidence row passed
         * @param partialCount requested partial count
         */
        public PartialReductionEvidence(boolean passed, int partialCount) {
            this(passed, null, null, null, partialCount);
        }
        /** Validates the diagnostic shape without performing measurement or admission. */
        public PartialReductionEvidence {
            if (passed && (kind == null || dataType == null || form == null
                    || partialCount != 2 && partialCount != 4)
                    || !passed && (kind != null || dataType != null || form != null
                    || partialCount != 0))
                throw new IllegalArgumentException("partial-reduction evidence facts disagree");
        }
    }

    /**
     * Creates ordinary analysis inputs outside the tagged Conv2d suffix exception.
     *
     * @param loweringManifestEnabled whether cold diagnostics retain a lowering manifest
     * @param carrierPattern ordered direct carrier forms; copied defensively by the canonical
     *     constructor
     * @param portableExecution immutable compute and parallelism preferences
     * @param materializationPolicy dimensionless cold materialization policy
     * @throws NullPointerException if a required reference or list element is {@code null}
     */
    public CpuPartitionAnalysisInputs(boolean loweringManifestEnabled,
            List<CarrierAccess> carrierPattern, PortableExecutionConfig portableExecution,
            MaterializationPolicy materializationPolicy) {
        this(loweringManifestEnabled, carrierPattern, portableExecution, materializationPolicy,
                false, PartialReductionEvidence.NONE);
    }

    /**
     * Cold, dimensionless materialization evidence. No value is measured during preparation.
     *
     * @param enabled whether analysis may retain bounded direct, single-copy, and two-copy
     *     complete representation candidates; it does not authorize ordinary copy promotion
     * @param copyFixedCostUnits non-negative fixed copy estimate per run
     * @param copyCostUnitsPerElement non-negative copy estimate per logical element
     * @param directKernelCostUnitsPerElement non-negative direct-consumer estimate per element/use
     * @param contiguousKernelCostUnitsPerElement non-negative contiguous-consumer estimate per
     *     element/use
     * @param expectedRunCount positive repeated-run estimate
     * @param maximumAdditionalBytes non-negative combined copy-candidate workspace byte ceiling
     * @param minimumNetBenefitCostUnits non-negative diagnostic absolute benefit threshold
     * @param minimumBenefitBasisPoints diagnostic relative benefit threshold from {@code 0} through
     *     {@code 10_000}
     */
    public record MaterializationPolicy(boolean enabled, long copyFixedCostUnits,
            long copyCostUnitsPerElement, long directKernelCostUnitsPerElement,
            long contiguousKernelCostUnitsPerElement, long expectedRunCount,
            long maximumAdditionalBytes, long minimumNetBenefitCostUnits,
            int minimumBenefitBasisPoints) {
        /** Direct-only compatibility policy. */
        public static final MaterializationPolicy DISABLED = new MaterializationPolicy(false,
                0, 0, 0, 0, 1, 0, 0, 0);
        /**
         * Validates one cold comparison policy.
         *
         * @throws IllegalArgumentException if a cost or byte limit is negative, expected runs are
         *     not positive, or the basis-point threshold is outside {@code [0, 10_000]}
         */
        public MaterializationPolicy {
            if (copyFixedCostUnits < 0 || copyCostUnitsPerElement < 0
                    || directKernelCostUnitsPerElement < 0
                    || contiguousKernelCostUnitsPerElement < 0 || expectedRunCount <= 0
                    || maximumAdditionalBytes < 0 || minimumNetBenefitCostUnits < 0
                    || minimumBenefitBasisPoints < 0 || minimumBenefitBasisPoints > 10_000) {
                throw new IllegalArgumentException("invalid materialization policy");
            }
        }
    }

    /**
     * Cold CPU-private strategy-selection inputs. CPU analysis bounds usable parallelism by the
     * configured and available snapshots, then applies the minimum range size. These facts do not
     * discover workers, retain a worker group, or enter generated artifact identity.
     *
     * @param computePreference non-null scalar or vector-if-eligible preference
     * @param configuredMaximumParallelism positive caller-configured upper bound
     * @param availableParallelism positive composition/platform availability snapshot
     * @param minimumElementsPerWorker positive minimum logical elements per parallel chunk
     */
    public record PortableExecutionConfig(ComputePreference computePreference,
            int configuredMaximumParallelism, int availableParallelism,
            long minimumElementsPerWorker) {
        /** Requested compute-axis preference with deterministic scalar fallback. */
        public enum ComputePreference {
            /** Require scalar generated compute. */ SCALAR,
            /** Prefer vector compute when every boundary is eligible; otherwise use scalar. */
            VECTOR_IF_ELIGIBLE
        }
        /** Compatibility configuration used by {@link CpuPartitionAnalysisInputs#DEFAULT}. */
        public static final PortableExecutionConfig DEFAULT = new PortableExecutionConfig(
                ComputePreference.SCALAR, 1, 1, 1);

        /**
         * Validates one immutable cold strategy-selection input.
         *
         * @param computePreference non-null scalar or vector-if-eligible preference
         * @param configuredMaximumParallelism positive caller-configured upper bound
         * @param availableParallelism positive availability snapshot
         * @param minimumElementsPerWorker positive minimum logical elements per parallel chunk
         * @throws NullPointerException if {@code computePreference} is {@code null}
         * @throws IllegalArgumentException if a parallelism or minimum-range value is not positive
         */
        public PortableExecutionConfig {
            java.util.Objects.requireNonNull(computePreference, "computePreference");
            if (configuredMaximumParallelism <= 0 || availableParallelism <= 0
                    || minimumElementsPerWorker <= 0) {
                throw new IllegalArgumentException("portable execution limits must be positive");
            }
        }
    }

    /**
     * Creates compatibility inputs selecting scalar, single-thread execution.
     *
     * @param loweringManifestEnabled whether to retain cold lowering diagnostics
     * @param carrierPattern non-null ordered derived-boundary carrier pattern, or empty for the
     *     exact-segment-per-boundary policy; snapshotted
     * @throws NullPointerException if {@code carrierPattern} or an entry is {@code null}
     */
    public CpuPartitionAnalysisInputs(boolean loweringManifestEnabled,
            List<CarrierAccess> carrierPattern) {
        this(loweringManifestEnabled, carrierPattern, PortableExecutionConfig.DEFAULT,
                MaterializationPolicy.DISABLED, false, PartialReductionEvidence.NONE);
    }

    /**
     * Creates direct-only inputs with explicit portable execution selection.
     *
     * @param loweringManifestEnabled whether to retain cold lowering diagnostics
     * @param carrierPattern non-null ordered derived-boundary carrier pattern, or empty for the
     *     exact-segment-per-boundary policy; snapshotted
     * @param portableExecution non-null immutable cold execution inputs
     * @throws NullPointerException if a reference or carrier entry is {@code null}
     */
    public CpuPartitionAnalysisInputs(boolean loweringManifestEnabled,
            List<CarrierAccess> carrierPattern, PortableExecutionConfig portableExecution) {
        this(loweringManifestEnabled, carrierPattern, portableExecution,
                MaterializationPolicy.DISABLED, false, PartialReductionEvidence.NONE);
    }

    /**
     * Snapshots the non-null ordered pattern, rejecting null entries in encounter order.
     *
     * @param loweringManifestEnabled whether to retain cold lowering diagnostics
     * @param carrierPattern non-null ordered derived-boundary carrier pattern, or empty for the
     *     exact-segment-per-boundary policy; copied defensively
     * @param portableExecution non-null immutable cold execution inputs
     * @param materializationPolicy non-null dimensionless cold materialization policy
     * @throws NullPointerException if {@code carrierPattern}, an entry, or
     *     either policy/configuration component is {@code null}
     */
    public CpuPartitionAnalysisInputs {
        carrierPattern = List.copyOf(carrierPattern);
        java.util.Objects.requireNonNull(portableExecution, "portableExecution");
        java.util.Objects.requireNonNull(materializationPolicy, "materializationPolicy");
        java.util.Objects.requireNonNull(partialReductionEvidence, "partialReductionEvidence");
    }
}
