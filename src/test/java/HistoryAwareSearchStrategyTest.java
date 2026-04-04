import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tuning.candidate.Candidate;
import tuning.candidate.CandidateFingerprint;
import tuning.candidate.ListCandidateSpace;
import tuning.search.FirstKSearchStrategy;
import tuning.search.HistoryAwareSearchStrategy;
import tuning.search.SearchContext;
import tuning.session.AutotuneRequest;
import tuning.store.*;
import tuning.workload.TensorRootWorkloadSpec;
import tuning.workload.WorkloadKind;
import tuning.workload.WorkloadMetadata;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.OffsetDateTime;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class HistoryAwareSearchStrategyTest {
    @Test
    void historyAwareSearchPrefersPersistedBestProfile() throws Exception {
        var workload = new TensorRootWorkloadSpec(
                "history_workload",
                WorkloadKind.GENERIC,
                environment -> tensor.Tensor.scalar(1.0),
                environment -> tuning.validate.ValidationReference.none(),
                environment -> WorkloadMetadata.of("history_workload", WorkloadKind.GENERIC)
        );

        Candidate a = candidate("a");
        Candidate b = candidate("b");
        Candidate c = candidate("c");
        var request = new AutotuneRequest(
                workload,
                new ListCandidateSpace(List.of(a, b, c)),
                tuning.measure.MeasurementPolicy.defaults(),
                tuning.validate.ValidationPolicy.disabled(),
                new tuning.search.SearchPolicy(3, 1, 1, false),
                new PersistencePolicy(true, true, Files.createTempFile("best-", ".json"), Files.createTempFile("hist-", ".jsonl"))
        );

        WorkloadFingerprint workloadFp = WorkloadFingerprint.of(workload, WorkloadMetadata.of("history_workload", WorkloadKind.GENERIC), a.profile());
        HardwareFingerprint hardwareFp = HardwareFingerprint.capture();
        new JsonFileBestProfileStore().save(request.persistence().bestProfilePath(), new BestProfileRecord(
                hardwareFp,
                workloadFp,
                b.profile(),
                1.0d,
                OffsetDateTime.now()
        ));

        var strategy = new HistoryAwareSearchStrategy(
                new FirstKSearchStrategy(1),
                new FileBestProfileResolver(new JsonFileBestProfileStore()),
                new JsonFileTuningHistoryStore(),
                request.persistence().bestProfilePath(),
                request.persistence().historyPath()
        );
        var result = strategy.search(new SearchContext(request, request.candidateSpace()));

        assertEquals("b", result.selectedCandidates().getFirst().name());
    }

    @Test
    void historyAwareSearchCanPruneHistoricallyInvalidCandidates() throws Exception {
        var workload = new TensorRootWorkloadSpec(
                "history_prune",
                WorkloadKind.GENERIC,
                environment -> tensor.Tensor.scalar(1.0),
                environment -> tuning.validate.ValidationReference.none(),
                environment -> WorkloadMetadata.of("history_prune", WorkloadKind.GENERIC)
        );
        Candidate invalid = candidate("invalid");
        Candidate valid = candidate("valid");
        Path historyPath = Files.createTempFile("hist-prune-", ".jsonl");
        var request = new AutotuneRequest(
                workload,
                new ListCandidateSpace(List.of(invalid, valid)),
                tuning.measure.MeasurementPolicy.defaults(),
                tuning.validate.ValidationPolicy.disabled(),
                new tuning.search.SearchPolicy(4, 1, 1, true),
                new PersistencePolicy(false, true, null, historyPath)
        );

        WorkloadFingerprint workloadFp = WorkloadFingerprint.of(workload, WorkloadMetadata.of("history_prune", WorkloadKind.GENERIC), invalid.profile());
        HardwareFingerprint hardwareFp = HardwareFingerprint.capture();
        new JsonFileTuningHistoryStore().append(historyPath, new TuningHistoryEntry(
                CandidateFingerprint.of(invalid),
                invalid.name(),
                false,
                Double.POSITIVE_INFINITY,
                Double.POSITIVE_INFINITY,
                Double.POSITIVE_INFINITY,
                "invalid",
                "summary",
                OffsetDateTime.now(),
                hardwareFp,
                workloadFp
        ));

        var strategy = new HistoryAwareSearchStrategy(
                new tuning.search.ExhaustiveSearchStrategy(),
                new FileBestProfileResolver(new JsonFileBestProfileStore()),
                new JsonFileTuningHistoryStore(),
                null,
                historyPath
        );
        var result = strategy.search(new SearchContext(request, request.candidateSpace()));

        assertEquals(1, result.selectedCandidates().size());
        assertEquals("valid", result.selectedCandidates().getFirst().name());
    }

    private static Candidate candidate(String name) {
        return new Candidate(name, new ExecutionProfile(
                name,
                name,
                tensor.DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.noOptimization(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        ));
    }
}
