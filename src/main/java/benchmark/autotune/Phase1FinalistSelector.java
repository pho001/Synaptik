package benchmark.autotune;

import benchmark.OptimizerCandidate;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Function;
import java.util.function.ToDoubleFunction;

public final class Phase1FinalistSelector {
    private Phase1FinalistSelector() {}

    public static <T> List<T> selectFinalists(
            List<T> phase1,
            ToDoubleFunction<T> trainingScoreFn,
            ToDoubleFunction<T> inferenceScoreFn,
            Function<T, OptimizerCandidate> candidateFn,
            int topK
    ) {
        List<T> byTraining = new ArrayList<>(phase1);
        byTraining.sort((a, b) -> Double.compare(trainingScoreFn.applyAsDouble(a), trainingScoreFn.applyAsDouble(b)));
        List<T> byInference = new ArrayList<>(phase1);
        byInference.sort((a, b) -> Double.compare(inferenceScoreFn.applyAsDouble(a), inferenceScoreFn.applyAsDouble(b)));

        Set<OptimizerCandidate> finalistsSet = new LinkedHashSet<>();
        int kTrain = Math.min(topK, byTraining.size());
        int kInf = Math.min(topK, byInference.size());
        for (int i = 0; i < kTrain; i++) finalistsSet.add(candidateFn.apply(byTraining.get(i)));
        for (int i = 0; i < kInf; i++) finalistsSet.add(candidateFn.apply(byInference.get(i)));

        List<T> out = new ArrayList<>(finalistsSet.size());
        for (T item : byTraining) {
            if (finalistsSet.contains(candidateFn.apply(item))) {
                out.add(item);
            }
        }
        return out;
    }
}
