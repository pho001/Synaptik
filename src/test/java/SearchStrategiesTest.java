import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tuning.candidate.Candidate;
import tuning.candidate.CandidateFingerprint;
import tuning.candidate.ListCandidateSpace;
import tuning.candidate.ProfileGridCandidateSpace;
import tuning.candidate.explicit.ExplicitProfileMutators;
import tuning.search.BestFirstTreeSearchStrategy;
import tuning.search.BranchAndBoundSearchStrategy;
import tuning.search.MedianSteadyStateScoreModel;
import tuning.search.ParentScoreBoundModel;
import tuning.search.WorkloadAwareBoundModel;
import tuning.search.CompositeSearchStrategy;
import tuning.search.ExhaustiveSearchStrategy;
import tuning.search.FirstKSearchStrategy;
import tuning.search.JsonSearchTreeReportRenderer;
import tuning.search.RefinementSearchStrategy;
import tuning.search.SearchContext;
import tuning.search.TextSearchTreeReportRenderer;
import tuning.search.TreeBeamSearchStrategy;
import tuning.autotune.AutotuneRequest;
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
                tensor.options.Conv2dOptions.defaults().withPadding(1, 1),
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
                List.of(ExplicitProfileMutators.conv2dLoweringModes(List.of(
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
                tuning.benchmark.report.BenchmarkCandidateReport.success(
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

    @Test
    void treeBeamSearchStrategyTracksLineageAcrossRounds() {
        var workload = tuning.workload.StandardWorkloads.conv2d(
                "conv_tree",
                1, 8, 8, 8, 8, 3, 3,
                tensor.options.Conv2dOptions.defaults().withPadding(1, 1),
                true
        );
        var base = new ExecutionProfile(
                "tree",
                "tree",
                tensor.DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.inferenceDefaults(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
        var space = new ProfileGridCandidateSpace(
                base,
                List.of(ExplicitProfileMutators.conv2dLoweringModes(List.of(
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

        var strategy = new TreeBeamSearchStrategy(new FirstKSearchStrategy(1), 1, 8);
        var initial = strategy.search(new SearchContext(request, request.candidateSpace()));
        assertEquals(1, initial.selectedCandidates().size());
        assertEquals(1, strategy.snapshot().nodes().size());

        var measured = java.util.List.of(
                tuning.benchmark.report.BenchmarkCandidateReport.success(
                        initial.selectedCandidates().getFirst(),
                        tuning.validate.ValidationResult.skipped(),
                        new tuning.measure.MeasurementResult(
                                tuning.measure.MeasurementPolicy.defaults(),
                                new graph.execution.trace.ExecutionTrace(null, null, graph.execution.trace.RunTrace.empty(ExecutionMode.FORWARD)),
                                new tuning.measure.MeasurementStatistics(1.0, 1.0, 1.0)
                        )
                )
        );
        var refined = strategy.refine(
                new SearchContext(request, request.candidateSpace()),
                measured,
                1,
                new java.util.HashSet<>(java.util.List.of(CandidateFingerprint.of(initial.selectedCandidates().getFirst())))
        );

        assertTrue(refined.selectedCandidates().size() >= 1);
        assertTrue(strategy.snapshot().nodes().size() > 1);
        assertTrue(strategy.snapshot().nodes().stream().anyMatch(node -> node.parentFingerprint() != null));
        assertEquals(refined.selectedCandidates().size(), strategy.snapshot().frontierFingerprints().size());
    }

    @Test
    void treeSearchReportRenderersProduceReadableOutput() {
        var workload = tuning.workload.StandardWorkloads.conv2d(
                "conv_tree_report",
                1, 8, 8, 8, 8, 3, 3,
                tensor.options.Conv2dOptions.defaults().withPadding(1, 1),
                true
        );
        var base = new ExecutionProfile(
                "tree-report",
                "tree-report",
                tensor.DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.inferenceDefaults(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
        var space = new ProfileGridCandidateSpace(
                base,
                List.of(ExplicitProfileMutators.conv2dLoweringModes(List.of(
                        config.optimizer.Conv2dLoweringMode.HEURISTIC,
                        config.optimizer.Conv2dLoweringMode.OFF
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

        var strategy = new TreeBeamSearchStrategy(new FirstKSearchStrategy(1), 1, 8);
        strategy.search(new SearchContext(request, request.candidateSpace()));
        String text = TextSearchTreeReportRenderer.render(strategy.report());
        String json = JsonSearchTreeReportRenderer.render(strategy.report());

        assertTrue(text.contains("Search Tree Report"));
        assertTrue(text.contains("strategy=TreeBeamSearchStrategy"));
        assertTrue(json.contains("\"strategyName\": \"TreeBeamSearchStrategy\""));
        assertTrue(json.contains("\"nodes\": ["));
    }

    @Test
    void bestFirstTreeSearchStrategyExpandsBestMeasuredNode() {
        var workload = tuning.workload.StandardWorkloads.conv2d(
                "conv_best_first",
                1, 8, 8, 8, 8, 3, 3,
                tensor.options.Conv2dOptions.defaults().withPadding(1, 1),
                true
        );
        var base = new ExecutionProfile(
                "best-first",
                "best-first",
                tensor.DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.inferenceDefaults(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
        var space = new ProfileGridCandidateSpace(
                base,
                List.of(ExplicitProfileMutators.conv2dLoweringModes(List.of(
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

        var seed = new ExhaustiveSearchStrategy();
        var strategy = new BestFirstTreeSearchStrategy(seed, new MedianSteadyStateScoreModel(), 2);
        var initial = strategy.search(new SearchContext(request, request.candidateSpace()));
        assertTrue(initial.selectedCandidates().size() >= 3);

        var measured = java.util.List.of(
                report(initial.selectedCandidates().get(0), 3.0),
                report(initial.selectedCandidates().get(1), 1.0),
                report(initial.selectedCandidates().get(2), 2.0)
        );
        var refined = strategy.refine(
                new SearchContext(request, request.candidateSpace()),
                measured,
                1,
                new java.util.HashSet<>(measured.stream().map(r -> CandidateFingerprint.of(r.candidate())).toList())
        );

        assertTrue(refined.selectedCandidates().size() >= 1);
        assertTrue(strategy.snapshot().nodes().size() > initial.selectedCandidates().size());
    }

    @Test
    void branchAndBoundSearchStrategyCanPruneFrontierBranch() {
        var workload = tuning.workload.StandardWorkloads.conv2d(
                "conv_bb",
                1, 8, 8, 8, 8, 3, 3,
                tensor.options.Conv2dOptions.defaults().withPadding(1, 1),
                true
        );
        var base = new ExecutionProfile(
                "bb",
                "bb",
                tensor.DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.inferenceDefaults(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
        var space = new ProfileGridCandidateSpace(
                base,
                List.of(ExplicitProfileMutators.conv2dLoweringModes(List.of(
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

        var strategy = new BranchAndBoundSearchStrategy(
                new ExhaustiveSearchStrategy(),
                new MedianSteadyStateScoreModel(),
                new ParentScoreBoundModel(),
                1,
                2
        );
        var initial = strategy.search(new SearchContext(request, request.candidateSpace()));
        var measured = java.util.List.of(
                report(initial.selectedCandidates().get(0), 10.0),
                report(initial.selectedCandidates().get(1), 1.0),
                report(initial.selectedCandidates().get(2), 2.0)
        );

        var refined = strategy.refine(
                new SearchContext(request, request.candidateSpace()),
                measured,
                1,
                new java.util.HashSet<>(measured.stream().map(r -> CandidateFingerprint.of(r.candidate())).toList())
        );

        assertTrue(strategy.prunedFingerprints().size() >= 1);
        assertTrue(refined.selectedCandidates().size() >= 1);
    }

    @Test
    void workloadAwareBoundModelPenalizesConv2dOffMode() {
        var workload = tuning.workload.StandardWorkloads.conv2d(
                "conv_bound",
                1, 8, 8, 8, 8, 3, 3,
                tensor.options.Conv2dOptions.defaults().withPadding(1, 1),
                true
        );
        var request = new AutotuneRequest(
                workload,
                candidate("seed").profile(),
                new ListCandidateSpace(List.of()),
                tuning.measure.MeasurementPolicy.defaults(),
                tuning.validate.ValidationPolicy.disabled(),
                new tuning.search.SearchPolicy(8, 1, 3, false),
                tuning.store.PersistencePolicy.disabled()
        );
        var scoreModel = new MedianSteadyStateScoreModel();
        var boundModel = new WorkloadAwareBoundModel();

        Candidate heuristic = candidate("conv2dLowering=HEURISTIC");
        Candidate off = candidate("conv2dLowering=OFF");
        double heuristicBound = boundModel.optimisticBound(
                report(heuristic, 1.0),
                new tuning.search.SearchTreeNode("h", heuristic.name(), null, 0, 0),
                scoreModel,
                new SearchContext(request, request.candidateSpace())
        );
        double offBound = boundModel.optimisticBound(
                report(off, 1.0),
                new tuning.search.SearchTreeNode("o", off.name(), null, 0, 0),
                scoreModel,
                new SearchContext(request, request.candidateSpace())
        );

        assertTrue(heuristicBound < offBound);
    }

    @Test
    void workloadAwareBoundModelPenalizesTransformerForceOff() {
        var request = new AutotuneRequest(
                tuning.workload.StandardWorkloads.transformerHotPath("transformer"),
                candidate("seed").profile(),
                new ListCandidateSpace(List.of()),
                tuning.measure.MeasurementPolicy.defaults(),
                tuning.validate.ValidationPolicy.disabled(),
                new tuning.search.SearchPolicy(8, 1, 3, false),
                tuning.store.PersistencePolicy.disabled()
        );
        var scoreModel = new MedianSteadyStateScoreModel();
        var boundModel = new WorkloadAwareBoundModel();

        Candidate auto = candidate("attentionMatMul=AUTO");
        Candidate forceOff = candidate("attentionMatMul=FORCE_OFF");
        double autoBound = boundModel.optimisticBound(
                report(auto, 1.0),
                new tuning.search.SearchTreeNode("a", auto.name(), null, 0, 0),
                scoreModel,
                new SearchContext(request, request.candidateSpace())
        );
        double offBound = boundModel.optimisticBound(
                report(forceOff, 1.0),
                new tuning.search.SearchTreeNode("f", forceOff.name(), null, 0, 0),
                scoreModel,
                new SearchContext(request, request.candidateSpace())
        );

        assertTrue(autoBound < offBound);
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

    private static tuning.benchmark.report.BenchmarkCandidateReport report(Candidate candidate, double medianMs) {
        return tuning.benchmark.report.BenchmarkCandidateReport.success(
                candidate,
                tuning.validate.ValidationResult.skipped(),
                new tuning.measure.MeasurementResult(
                        tuning.measure.MeasurementPolicy.defaults(),
                        new graph.execution.trace.ExecutionTrace(null, null, graph.execution.trace.RunTrace.empty(ExecutionMode.FORWARD)),
                        new tuning.measure.MeasurementStatistics(medianMs, medianMs, medianMs)
                )
        );
    }
}
