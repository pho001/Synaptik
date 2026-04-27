package backend.select;

import config.runtime.RuntimeConfig;
import graph.optimizer.partition.BackendCandidatePartition;

import java.util.List;

public interface BackendSelectionPolicy {
    BackendSelectionResult select(
            List<BackendCandidatePartition> candidates,
            RuntimeConfig runtimeConfig
    );
}
