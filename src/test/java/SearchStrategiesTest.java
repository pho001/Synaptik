import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tuning.candidate.Candidate;
import tuning.candidate.ListCandidateSpace;
import tuning.search.CompositeSearchStrategy;
import tuning.search.ExhaustiveSearchStrategy;
import tuning.search.FirstKSearchStrategy;
import tuning.search.SearchContext;
import tuning.session.AutotuneRequest;
import tuning.workload.TensorRootWorkloadSpec;
import tuning.workload.WorkloadKind;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

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
