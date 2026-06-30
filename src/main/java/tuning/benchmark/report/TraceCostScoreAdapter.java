package tuning.benchmark.report;

import graph.optimizer.cost.CostComponent;
import graph.optimizer.cost.CostScore;
import trace.compile.MaterializationCostTrace;
import trace.compile.PartitionDecisionTrace;

import java.util.List;
import java.util.Objects;

final class TraceCostScoreAdapter {
    private TraceCostScoreAdapter() {
    }

    static CostScore toCostScore(MaterializationCostTrace trace) {
        Objects.requireNonNull(trace, "trace cannot be null");
        return CostScore.of(
                "AcceleratorPartitionCostModel",
                "accelerator-partition-materialization",
                List.of(
                        CostComponent.higherIsBetter("finalScore", trace.finalScore(),
                                "materialization-aware accelerator partition score"),
                        CostComponent.higherIsBetter("estimatedComputeWork", trace.estimatedComputeWork(),
                                "larger accelerator work can amortize dispatch and transfer cost"),
                        CostComponent.higherIsBetter("avoidedIntermediateBytes", trace.avoidedIntermediateBytes(),
                                "intermediate bytes retained inside the accelerator region"),
                        CostComponent.lowerIsBetter("boundaryCount", trace.boundaryCount(),
                                "CPU/accelerator boundaries introduce handoff cost"),
                        CostComponent.lowerIsBetter("estimatedTransferBytes", trace.estimatedTransferBytes(),
                                "estimated bytes copied across accelerator boundaries"),
                        CostComponent.lowerIsBetter("layoutFallbackBytes", trace.layoutFallbackBytes(),
                                "bytes affected by layout fallback or dense materialization"),
                        CostComponent.lowerIsBetter("dispatchCost", trace.dispatchCost(),
                                "fixed accelerator dispatch cost applied by the preset"),
                        CostComponent.informational("preset", 0.0d, trace.preset()),
                        CostComponent.informational("fallbackMode", 0.0d, trace.fallbackMode()),
                        CostComponent.informational("layoutClass", 0.0d, trace.layoutClass())
                )
        );
    }

    static CostScore toCostScore(PartitionDecisionTrace.CandidateCostTrace trace) {
        Objects.requireNonNull(trace, "trace cannot be null");
        return CostScore.of(
                "AcceleratorPartitionCostModel",
                "accelerator-partition-finalist",
                List.of(
                        CostComponent.higherIsBetter("finalScore", trace.finalScore(),
                                "materialization-aware accelerator partition finalist score"),
                        CostComponent.higherIsBetter("estimatedComputeWork", trace.estimatedComputeWork(),
                                "larger accelerator work can amortize dispatch and transfer cost"),
                        CostComponent.lowerIsBetter("boundaryCount", trace.boundaryCount(),
                                "CPU/accelerator boundaries introduce handoff cost"),
                        CostComponent.lowerIsBetter("estimatedTransferBytes", trace.estimatedTransferBytes(),
                                "estimated bytes copied across accelerator boundaries"),
                        CostComponent.lowerIsBetter("layoutFallbackBytes", trace.layoutFallbackBytes(),
                                "bytes affected by layout fallback or dense materialization"),
                        CostComponent.informational("preset", 0.0d, trace.preset())
                )
        );
    }
}
