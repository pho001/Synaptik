package io.github.pho001.synaptik.backend.cpu.internal.prepare;

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
 * @param materializationPolicy non-null dimensionless cold comparison policy; disabled means
 *     direct access only
 * @param conv2dMaterializedSuffixUnit whether this input belongs to the sole tagged pointwise
 *     suffix unit of a two-unit Conv2d plan
 */
public record CpuPartitionAnalysisInputs(boolean loweringManifestEnabled,
        List<CarrierAccess> carrierPattern, PortableExecutionConfig portableExecution,
        MaterializationPolicy materializationPolicy, boolean conv2dMaterializedSuffixUnit)
        implements BackendAnalysisInputs {
    /**
     * Default input: manifest and materialization disabled, scalar single-thread execution, and
     * one exact-segment carrier selected for every boundary derived by lowering.
     */
    public static final CpuPartitionAnalysisInputs DEFAULT = new CpuPartitionAnalysisInputs(false,
            List.of(),
            PortableExecutionConfig.DEFAULT, MaterializationPolicy.DISABLED, false);

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
                false);
    }

    /**
     * Cold, dimensionless materialization evidence. No value is measured during preparation.
     *
     * @param enabled whether analysis may compare one-input contiguous-copy candidates
     * @param copyFixedCostUnits non-negative fixed copy estimate per run
     * @param copyCostUnitsPerElement non-negative copy estimate per logical element
     * @param directKernelCostUnitsPerElement non-negative direct-consumer estimate per element/use
     * @param contiguousKernelCostUnitsPerElement non-negative contiguous-consumer estimate per
     *     element/use
     * @param expectedRunCount positive repeated-run estimate
     * @param maximumAdditionalBytes non-negative workspace byte ceiling
     * @param minimumNetBenefitCostUnits non-negative absolute selection threshold
     * @param minimumBenefitBasisPoints relative selection threshold from {@code 0} through
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
                MaterializationPolicy.DISABLED, false);
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
                MaterializationPolicy.DISABLED, false);
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
    }
}
