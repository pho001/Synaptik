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
 */
public record CpuPartitionAnalysisInputs(boolean loweringManifestEnabled,
        List<CarrierAccess> carrierPattern)
        implements BackendAnalysisInputs {
    /** Default current-topology input: manifest disabled and four direct segment boundaries. */
    public static final CpuPartitionAnalysisInputs DEFAULT = new CpuPartitionAnalysisInputs(false,
            List.of(CarrierAccess.MEMORY_SEGMENT, CarrierAccess.MEMORY_SEGMENT,
                    CarrierAccess.MEMORY_SEGMENT, CarrierAccess.MEMORY_SEGMENT));

    /**
     * Snapshots the non-null ordered pattern, rejecting null entries in encounter order.
     *
     * @throws NullPointerException if {@code carrierPattern} or an entry is {@code null}
     */
    public CpuPartitionAnalysisInputs {
        carrierPattern = List.copyOf(carrierPattern);
    }
}
