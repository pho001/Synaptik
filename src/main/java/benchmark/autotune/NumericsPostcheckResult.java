package benchmark.autotune;

import benchmark.OptimizerCandidate;

import java.nio.file.Path;
import java.util.List;

public record NumericsPostcheckResult(
        List<OptimizerCandidate> keptCandidates,
        int checked,
        int markedUnsafe,
        Path reportPath,
        List<NumericsPostcheckDrop> droppedUnsafe
) {
    public int droppedCount() {
        return droppedUnsafe.size();
    }
}
