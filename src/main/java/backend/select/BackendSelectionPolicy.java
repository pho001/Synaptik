package backend.select;

import config.runtime.RuntimeConfig;
import graph.optimizer.partition.BackendCandidatePartition;

import java.util.List;

/**
 * Strategy for selecting backend partition plans that should run as accelerator/CPU regions.
 */
public interface BackendSelectionPolicy {
    /**
     * Selects executable backend plans from optimizer-produced candidates.
     *
     * @param candidates backend candidate partitions from graph optimization
     * @param runtimeConfig runtime policy controlling accelerator enablement and thresholds
     * @return selected plans plus trace metadata explaining accepted and rejected candidates
     */
    BackendSelectionResult select(
            List<BackendCandidatePartition> candidates,
            RuntimeConfig runtimeConfig
    );
}
