package backend.select;

import backend.ComputeBackend;
import backend.accelerator.select.AcceleratorPlanCostModel;
import backend.accelerator.select.AcceleratorRuntimeAvailability;
import config.runtime.RuntimeConfig;
import graph.execution.trace.BackendSelectionDecisionTrace;
import graph.execution.trace.BackendSelectionTrace;
import graph.optimizer.partition.BackendCandidatePartition;
import graph.optimizer.partition.PartitionPlan;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Default backend selection policy used by prepared execution building.
 *
 * <p>The policy filters optimizer-produced backend candidates by compatibility, runtime accelerator
 * enablement, runtime availability requirements, and cost-model acceptance. Every accepted or rejected
 * candidate contributes a decision to the returned trace.</p>
 */
public final class DefaultBackendSelectionPolicy implements BackendSelectionPolicy {
    /**
     * Selects backend plans using runtime accelerator policy and the accelerator cost model.
     *
     * @param candidates candidate partitions from graph optimization; {@code null} yields an empty result
     * @param runtimeConfig runtime policy; when {@code null}, backend enablement checks are skipped
     * @return selected plans and decision trace
     */
    @Override
    public BackendSelectionResult select(
            List<BackendCandidatePartition> candidates,
            RuntimeConfig runtimeConfig
    ) {
        List<PartitionPlan> out = new ArrayList<>();
        List<BackendSelectionDecisionTrace> decisions = new ArrayList<>();
        if (candidates == null) {
            return BackendSelectionResult.empty();
        }
        for (BackendCandidatePartition candidate : candidates) {
            if (candidate == null) {
                continue;
            }
            List<ComputeBackend> compatibleBackends = candidate.compatibleBackends().stream()
                    .sorted(Comparator.comparing(Enum::name))
                    .toList();
            PartitionPlan plan = candidate.plan();
            if (plan == null) {
                decisions.add(new BackendSelectionDecisionTrace(
                        candidate.anchorNodeId(),
                        candidate.nodeIds(),
                        compatibleBackends,
                        false,
                        null,
                        "missing-backend-plan",
                        0L
                ));
                continue;
            }
            if (!candidate.compatibleBackends().contains(plan.backend())) {
                var summary = AcceleratorPlanCostModel.summarize(plan);
                decisions.add(new BackendSelectionDecisionTrace(
                        candidate.anchorNodeId(),
                        candidate.nodeIds(),
                        compatibleBackends,
                        false,
                        null,
                        "backend-not-compatible",
                        plan.estimatedWork(),
                        summary,
                        List.of()
                ));
                continue;
            }
            if (runtimeConfig != null && !runtimeConfig.accelerator().forBackend(plan.backend()).enabled()) {
                var summary = AcceleratorPlanCostModel.summarize(plan);
                decisions.add(new BackendSelectionDecisionTrace(
                        candidate.anchorNodeId(),
                        candidate.nodeIds(),
                        compatibleBackends,
                        false,
                        null,
                        "backend-disabled",
                        plan.estimatedWork(),
                        summary,
                        List.of()
                ));
                continue;
            }
            boolean requireAvailability = runtimeConfig != null
                    && runtimeConfig.accelerator().forBackend(plan.backend()).requireRuntimeAvailability();
            boolean available = !requireAvailability || AcceleratorRuntimeAvailability.isAvailable(plan.backend());
            if (!available) {
                var summary = AcceleratorPlanCostModel.summarize(plan);
                decisions.add(new BackendSelectionDecisionTrace(
                        candidate.anchorNodeId(),
                        candidate.nodeIds(),
                        compatibleBackends,
                        false,
                        null,
                        "runtime-unavailable",
                        plan.estimatedWork(),
                        summary,
                        List.of()
                ));
                continue;
            }
            AcceleratorPlanCostModel.Decision decision = AcceleratorPlanCostModel.decide(plan, runtimeConfig);
            if (decision.accepted()) {
                out.add(plan);
                decisions.add(new BackendSelectionDecisionTrace(
                        candidate.anchorNodeId(),
                        candidate.nodeIds(),
                        compatibleBackends,
                        true,
                        plan.backend(),
                        "selected",
                        plan.estimatedWork(),
                        decision.costSummary(),
                        List.of(),
                        plan.gpuLoweredRegionManifest()
                ));
            } else {
                decisions.add(new BackendSelectionDecisionTrace(
                        candidate.anchorNodeId(),
                        candidate.nodeIds(),
                        compatibleBackends,
                        false,
                        null,
                        decision.reason(),
                        plan.estimatedWork(),
                        decision.costSummary(),
                        List.of()
                ));
            }
        }
        int selectedCount = (int) decisions.stream().filter(BackendSelectionDecisionTrace::selected).count();
        return new BackendSelectionResult(
                List.copyOf(out),
                new BackendSelectionTrace(decisions.size(), selectedCount, decisions.size() - selectedCount, decisions)
        );
    }
}
