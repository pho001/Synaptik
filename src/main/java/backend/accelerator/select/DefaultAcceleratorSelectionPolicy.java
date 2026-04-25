package backend.accelerator.select;

import backend.ComputeBackend;
import config.runtime.RuntimeConfig;
import graph.execution.trace.AcceleratorSelectionDecisionTrace;
import graph.execution.trace.AcceleratorSelectionTrace;
import graph.optimizer.partition.AcceleratorCandidatePartition;
import graph.optimizer.partition.AcceleratorPartitionPlan;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public final class DefaultAcceleratorSelectionPolicy implements AcceleratorSelectionPolicy {
    @Override
    public AcceleratorSelectionResult select(
            List<AcceleratorCandidatePartition> candidates,
            RuntimeConfig runtimeConfig
    ) {
        List<AcceleratorPartitionPlan> out = new ArrayList<>();
        List<AcceleratorSelectionDecisionTrace> decisions = new ArrayList<>();
        if (candidates == null) {
            return AcceleratorSelectionResult.empty();
        }
        for (AcceleratorCandidatePartition candidate : candidates) {
            if (candidate == null) {
                continue;
            }
            List<ComputeBackend> compatibleBackends = candidate.compatibleBackends().stream()
                    .sorted(Comparator.comparing(Enum::name))
                    .toList();
            AcceleratorPartitionPlan plan = candidate.plan();
            if (plan == null) {
                decisions.add(new AcceleratorSelectionDecisionTrace(
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
                decisions.add(new AcceleratorSelectionDecisionTrace(
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
                decisions.add(new AcceleratorSelectionDecisionTrace(
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
                decisions.add(new AcceleratorSelectionDecisionTrace(
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
                decisions.add(new AcceleratorSelectionDecisionTrace(
                        candidate.anchorNodeId(),
                        candidate.nodeIds(),
                        compatibleBackends,
                        true,
                        plan.backend(),
                        "selected",
                        plan.estimatedWork()
                ));
            } else {
                decisions.add(new AcceleratorSelectionDecisionTrace(
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
        int selectedCount = (int) decisions.stream().filter(AcceleratorSelectionDecisionTrace::selected).count();
        return new AcceleratorSelectionResult(
                List.copyOf(out),
                new AcceleratorSelectionTrace(decisions.size(), selectedCount, decisions.size() - selectedCount, decisions)
        );
    }
}
