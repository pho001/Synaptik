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

public final class DefaultBackendSelectionPolicy implements BackendSelectionPolicy {
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
                decisions.add(new BackendSelectionDecisionTrace(
                        candidate.anchorNodeId(),
                        candidate.nodeIds(),
                        compatibleBackends,
                        false,
                        null,
                        "backend-not-compatible",
                        plan.estimatedWork()
                ));
                continue;
            }
            if (runtimeConfig != null && !runtimeConfig.accelerator().forBackend(plan.backend()).enabled()) {
                decisions.add(new BackendSelectionDecisionTrace(
                        candidate.anchorNodeId(),
                        candidate.nodeIds(),
                        compatibleBackends,
                        false,
                        null,
                        "backend-disabled",
                        plan.estimatedWork()
                ));
                continue;
            }
            boolean requireAvailability = runtimeConfig != null
                    && runtimeConfig.accelerator().forBackend(plan.backend()).requireRuntimeAvailability();
            boolean available = !requireAvailability || AcceleratorRuntimeAvailability.isAvailable(plan.backend());
            if (!available) {
                decisions.add(new BackendSelectionDecisionTrace(
                        candidate.anchorNodeId(),
                        candidate.nodeIds(),
                        compatibleBackends,
                        false,
                        null,
                        "runtime-unavailable",
                        plan.estimatedWork()
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
                        plan.estimatedWork()
                ));
            } else {
                decisions.add(new BackendSelectionDecisionTrace(
                        candidate.anchorNodeId(),
                        candidate.nodeIds(),
                        compatibleBackends,
                        false,
                        null,
                        decision.reason(),
                        plan.estimatedWork()
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
