package backend.accelerator.select;

import config.runtime.RuntimeConfig;
import graph.optimizer.partition.AcceleratorCandidatePartition;

import java.util.List;

public interface AcceleratorSelectionPolicy {
    AcceleratorSelectionResult select(
            List<AcceleratorCandidatePartition> candidates,
            RuntimeConfig runtimeConfig
    );
}
