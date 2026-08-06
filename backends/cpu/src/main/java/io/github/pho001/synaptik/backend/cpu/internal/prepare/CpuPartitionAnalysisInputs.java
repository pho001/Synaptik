package io.github.pho001.synaptik.backend.cpu.internal.prepare;

import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import java.util.List;

/**
 * Checked immutable CPU inputs that do not contain graph semantics or instance resources.
 *
 * @param loweringManifestEnabled whether cold diagnostics should retain a lowering manifest
 * @param carrierPattern non-null ordered direct carrier forms for the current four boundaries;
 *     membership is snapshotted and contains no physical carrier object
 * @param portableExecution non-null immutable cold compute-preference and parallelism inputs
 */
public record CpuPartitionAnalysisInputs(boolean loweringManifestEnabled,
        List<CarrierAccess> carrierPattern, PortableExecutionConfig portableExecution)
        implements BackendAnalysisInputs {
    /** Default current-topology input: manifest disabled and four direct segment boundaries. */
    public static final CpuPartitionAnalysisInputs DEFAULT = new CpuPartitionAnalysisInputs(false,
            List.of(CarrierAccess.MEMORY_SEGMENT, CarrierAccess.MEMORY_SEGMENT,
                    CarrierAccess.MEMORY_SEGMENT, CarrierAccess.MEMORY_SEGMENT),
            PortableExecutionConfig.DEFAULT);

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
     * @param carrierPattern non-null ordered current-topology carrier pattern; snapshotted
     * @throws NullPointerException if {@code carrierPattern} or an entry is {@code null}
     */
    public CpuPartitionAnalysisInputs(boolean loweringManifestEnabled,
            List<CarrierAccess> carrierPattern) {
        this(loweringManifestEnabled, carrierPattern, PortableExecutionConfig.DEFAULT);
    }

    /**
     * Snapshots the non-null ordered pattern, rejecting null entries in encounter order.
     *
     * @param loweringManifestEnabled whether to retain cold lowering diagnostics
     * @param carrierPattern non-null ordered current-topology carrier pattern; copied defensively
     * @param portableExecution non-null immutable cold execution inputs
     * @throws NullPointerException if {@code carrierPattern}, an entry, or
     *     {@code portableExecution} is {@code null}
     */
    public CpuPartitionAnalysisInputs {
        carrierPattern = List.copyOf(carrierPattern);
        java.util.Objects.requireNonNull(portableExecution, "portableExecution");
    }
}
