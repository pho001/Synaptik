package backend.select;

import config.runtime.RuntimeConfig;
import planning.partition.PlannedPartition;

import java.util.List;

/**
 * Strategy for selecting backend partition plans that should run as accelerator/CPU partitions.
 */
public interface BackendSelectionPolicy {
    /**
     * Selects executable backend plans from optimizer-produced candidates.
     *
     * @param plannedPartitions backend planned partitions from graph compilation
     * @param runtimeConfig runtime policy controlling accelerator enablement and thresholds
     * @return selected planned partitions plus trace metadata explaining accepted and rejected candidates
     */
    BackendSelectionResult select(
            List<PlannedPartition> plannedPartitions,
            RuntimeConfig runtimeConfig
    );
}
