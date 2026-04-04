import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tuning.candidate.Candidate;
import tuning.candidate.CandidateFingerprint;
import tuning.candidate.ListCandidateSpace;
import tuning.candidate.ProfileGridCandidateSpace;
import tuning.candidate.ProfileMutators;
import tuning.search.CompositeSearchStrategy;
import tuning.search.ExhaustiveSearchStrategy;
import tuning.search.FirstKSearchStrategy;
import tuning.search.RefinementSearchStrategy;
import tuning.search.SearchContext;
import tuning.session.AutotuneRequest;
import tuning.workload.TensorRootWorkloadSpec;
import tuning.workload.WorkloadKind;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class SearchStrategiesTest {
    @Test
    void compositeSearchStrategyMergesSelections() {
        var workload = new TensorRootWorkloadSpec(
                "search_workload",
                WorkloadKind.GENERIC,
                environment -> tensor.Tensor.scalar(1.0)
        );
        var candidates = List.of(candidate("c0"), candidate("c1"), candidate("c2"));
        var request = new AutotuneRequest(
                workload,
                new ListCandidateSpace(candidates),
                tuning.measure.MeasurementPolicy.defaults(),
                tuning.validate.ValidationPolicy.disabled(),
                new tuning.search.SearchPolicy(3, 2, 1, false),
                tuning.store.PersistencePolicy.disabled()
        );

        var strategy = new CompositeSearchStrategy(List.of(
                new FirstKSearchStrategy(1),
                new ExhaustiveSearchStrategy()
        ));
        var result = strategy.search(new SearchContext(request, request.candidateSpace()));

        assertEquals(3, result.selectedCandidates().size());
        assertEquals("c0", result.preferredCandidate().name());
    }

    @Test
    void refinementStrategyExpandsNeighborsFromBestCandidates() {
        var workload = tuning.workload.StandardWorkloads.conv2d(
                "conv_search",
                1, 8, 8, 8, 8, 3, 3,
                tensor.Conv2dOptions.defaults().withPadding(1, 1),
                true
        );
        var base = new ExecutionProfile(
                "base",
                "base",
                tensor.DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.inferenceDefaults(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
        var space = new ProfileGridCandidateSpace(
                base,
                List.of(ProfileMutators.conv2dLoweringModes(List.of(
                        config.optimizer.Conv2dLoweringMode.HEURISTIC,
                        config.optimizer.Conv2dLoweringMode.OFF,
                        config.optimizer.Conv2dLoweringMode.ALWAYS
                )))
        );
        var request = new AutotuneRequest(
                workload,
                space,
                tuning.measure.MeasurementPolicy.defaults(),
                tuning.validate.ValidationPolicy.disabled(),
                new tuning.search.SearchPolicy(8, 1, 3, false),
                tuning.store.PersistencePolicy.disabled()
        );

        var refinement = new RefinementSearchStrategy(new FirstKSearchStrategy(1), 1, 8);
        var initial = refinement.search(new SearchContext(request, request.candidateSpace()));
        assertEquals(1, initial.selectedCandidates().size());

        var measured = java.util.List.of(
                tuning.report.BenchmarkCandidateReport.success(
                        initial.selectedCandidates().getFirst(),
                        tuning.validate.ValidationResult.skipped(),
                        new tuning.measure.MeasurementResult(
                                tuning.measure.MeasurementPolicy.defaults(),
                                new graph.execution.trace.ExecutionTrace(null, null, graph.execution.trace.RunTrace.empty(ExecutionMode.FORWARD)),
                                new tuning.measure.MeasurementStatistics(1.0, 1.0, 1.0)
                        )
                )
        );
        var refined = refinement.refine(
                new SearchContext(request, request.candidateSpace()),
                measured,
                1,
                new java.util.HashSet<>(java.util.List.of(CandidateFingerprint.of(initial.selectedCandidates().getFirst())))
        );

        assertTrue(refined.selectedCandidates().size() >= 1);
        assertTrue(refined.selectedCandidates().stream().noneMatch(c ->
                CandidateFingerprint.of(c).equals(CandidateFingerprint.of(initial.selectedCandidates().getFirst()))
        ));
    }

    private static Candidate candidate(String name) {
        return new Candidate(
                name,
                new ExecutionProfile(
                        name,
                        name,
                        tensor.DataType.FLOAT64,
                        ExecutionMode.FORWARD,
                        config.optimizer.OptimizerConfig.noOptimization(),
                        config.runtime.RuntimeConfig.inferenceDefaults(),
                        WorkloadProfile.none()
                )
        );
    }
}
