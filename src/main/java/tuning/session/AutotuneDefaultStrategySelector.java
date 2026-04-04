package tuning.session;

import tuning.candidate.CandidateSpace;
import tuning.candidate.RefinableCandidateSpace;
import tuning.search.BranchAndBoundSearchStrategy;
import tuning.search.ExhaustiveSearchStrategy;
import tuning.search.FirstKSearchStrategy;
import tuning.search.SearchStrategy;
import tuning.search.TreeBeamSearchStrategy;
import tuning.search.MedianSteadyStateScoreModel;
import tuning.search.WorkloadAwareBoundModel;

public final class AutotuneDefaultStrategySelector {
    private static final int MIN_TREE_CANDIDATES = 8;

    private AutotuneDefaultStrategySelector() {
    }

    public static SearchStrategy select(AutotuneRequest request) {
        if (request == null) {
            throw new IllegalArgumentException("request cannot be null");
        }
        CandidateSpace space = request.candidateSpace();
        int candidateCount = space.generate(request.workload()).size();
        int beamWidth = Math.max(1, request.search().beamWidth());

        if (!(space instanceof RefinableCandidateSpace)) {
            return new ExhaustiveSearchStrategy();
        }

        if (candidateCount >= Math.max(MIN_TREE_CANDIDATES, beamWidth * 3)) {
            return new BranchAndBoundSearchStrategy(
                    new FirstKSearchStrategy(Math.min(beamWidth, candidateCount)),
                    new MedianSteadyStateScoreModel(),
                    new WorkloadAwareBoundModel(),
                    beamWidth,
                    Math.max(beamWidth, 2)
            );
        }

        if (candidateCount > beamWidth) {
            return new TreeBeamSearchStrategy(
                    new FirstKSearchStrategy(Math.min(beamWidth, candidateCount)),
                    beamWidth,
                    Math.max(beamWidth, 2)
            );
        }

        return new ExhaustiveSearchStrategy();
    }
}
