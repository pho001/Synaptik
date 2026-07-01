package backend.select;

import backend.contract.ComputeBackend;
import backend.accelerator.lowering.GpuLoweredPartitionManifest;
import backend.accelerator.lowering.AcceleratorPartitionPlan;
import backend.accelerator.select.AcceleratorPlanCostModel;
import backend.accelerator.select.AcceleratorRuntimeAvailability;
import config.runtime.RuntimeConfig;
import trace.prepare.BackendSelectionDecisionTrace;
import trace.prepare.BackendSelectionTrace;
import trace.prepare.GpuLoweredPartitionTrace;
import trace.compile.MaterializationCostTrace;
import planning.partition.cost.AcceleratorPartitionScoreModel;
import planning.partition.PartitionPlan;
import planning.partition.PlannedPartition;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Default backend selection policy used by prepared execution building.
 *
 * <p>The policy filters planned partitions by compatibility, runtime accelerator
 * enablement, runtime availability requirements, and cost-model acceptance. Every accepted or rejected
 * planned partition contributes a decision to the returned trace.</p>
 */
public final class DefaultBackendSelectionPolicy implements BackendSelectionPolicy {
    /**
     * Selects backend planned partitions using runtime accelerator policy and the accelerator cost model.
     *
     * @param plannedPartitions planned partitions from graph compilation; {@code null} yields an empty result
     * @param runtimeConfig runtime policy; when {@code null}, backend enablement checks are skipped
     * @return selected planned partitions and decision trace
     */
    @Override
    public BackendSelectionResult select(
            List<PlannedPartition> plannedPartitions,
            RuntimeConfig runtimeConfig
    ) {
        List<PlannedPartition> out = new ArrayList<>();
        List<BackendSelectionDecisionTrace> decisions = new ArrayList<>();
        if (plannedPartitions == null) {
            return BackendSelectionResult.empty();
        }
        for (PlannedPartition plannedPartition : plannedPartitions) {
            if (plannedPartition == null) {
                continue;
            }
            List<String> compatibleBackends = plannedPartition.compatibleBackends().stream()
                    .sorted(Comparator.comparing(Enum::name))
                    .map(Enum::name)
                    .toList();
            PartitionPlan plan = plannedPartition.plan();
            if (plan != null && plan.backend() == ComputeBackend.CPU) {
                continue;
            }
            if (plan == null) {
                decisions.add(new BackendSelectionDecisionTrace(
                        plannedPartition.anchorNodeId(),
                        plannedPartition.nodeIds(),
                        compatibleBackends,
                        false,
                        null,
                        "missing-backend-plan",
                        0L
                ));
                continue;
            }
            if (!plannedPartition.compatibleBackends().contains(plan.backend())) {
                var summary = AcceleratorPlanCostModel.summarize(plan);
                decisions.add(new BackendSelectionDecisionTrace(
                        plannedPartition.anchorNodeId(),
                        plannedPartition.nodeIds(),
                        compatibleBackends,
                        false,
                        null,
                        "backend-not-compatible",
                        plan.estimatedWork(),
                        traceCost(summary),
                        List.of()
                ));
                continue;
            }
            if (runtimeConfig != null && !runtimeConfig.accelerator().forBackend(plan.backend()).enabled()) {
                var summary = AcceleratorPlanCostModel.summarize(plan);
                decisions.add(new BackendSelectionDecisionTrace(
                        plannedPartition.anchorNodeId(),
                        plannedPartition.nodeIds(),
                        compatibleBackends,
                        false,
                        null,
                        "backend-disabled",
                        plan.estimatedWork(),
                        traceCost(summary),
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
                        plannedPartition.anchorNodeId(),
                        plannedPartition.nodeIds(),
                        compatibleBackends,
                        false,
                        null,
                        "runtime-unavailable",
                        plan.estimatedWork(),
                        traceCost(summary),
                        List.of()
                ));
                continue;
            }
            AcceleratorPlanCostModel.Decision decision = AcceleratorPlanCostModel.decide(plan, runtimeConfig);
            if (decision.accepted()) {
                out.add(plannedPartition);
                decisions.add(new BackendSelectionDecisionTrace(
                        plannedPartition.anchorNodeId(),
                        plannedPartition.nodeIds(),
                        compatibleBackends,
                        true,
                        plan.backend().name(),
                        "selected",
                        plan.estimatedWork(),
                        traceCost(decision.costSummary()),
                        List.of(),
                        traceManifest(plan instanceof AcceleratorPartitionPlan acceleratorPlan
                                ? acceleratorPlan.gpuLoweredPartitionManifest()
                                : null)
                ));
            } else {
                decisions.add(new BackendSelectionDecisionTrace(
                        plannedPartition.anchorNodeId(),
                        plannedPartition.nodeIds(),
                        compatibleBackends,
                        false,
                        null,
                        decision.reason(),
                        plan.estimatedWork(),
                        traceCost(decision.costSummary()),
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

    private static MaterializationCostTrace traceCost(
            AcceleratorPartitionScoreModel.MaterializationCostSummary source
    ) {
        if (source == null) {
            return null;
        }
        return new MaterializationCostTrace(
                source.preset(), source.boundaryCount(), source.estimatedTransferBytes(),
                source.layoutFallbackBytes(), source.estimatedComputeWork(),
                source.avoidedIntermediateBytes(), source.dispatchCost(), source.finalScore(),
                source.reasonCode(), source.fallbackMode(), source.layoutClass()
        );
    }

    private static GpuLoweredPartitionTrace traceManifest(GpuLoweredPartitionManifest source) {
        if (source == null) {
            return null;
        }
        return new GpuLoweredPartitionTrace(
                source.partitionId(),
                source.backend().name(),
                source.anchorNodeId(),
                source.orderedNodeIds(),
                source.externalInputNodeIds(),
                source.outputNodeIds(),
                source.selectedPartitionLength(),
                source.originalOps().stream()
                        .map(op -> new GpuLoweredPartitionTrace.OriginalOperation(
                                op.nodeId(), op.opType(), op.inputNodeIds(), op.outputNodeIds(),
                                op.dataType().name(), op.shape(), op.loweredPrimitiveIds(),
                                op.aggregatedReasons().stream().map(Enum::name).toList()
                        ))
                        .toList(),
                source.loweredPrimitives().stream()
                        .map(primitive -> new GpuLoweredPartitionTrace.LoweredPrimitive(
                                primitive.primitiveId(), primitive.primitiveType(), primitive.sourceOriginalNodeIds(),
                                primitive.inputRefs(), primitive.outputRef(), primitive.dataType().name(),
                                primitive.shape(), primitive.reasons().stream().map(Enum::name).toList()
                        ))
                        .toList(),
                source.inputAssumptions().stream().map(DefaultBackendSelectionPolicy::traceAssumption).toList(),
                source.outputAssumptions().stream().map(DefaultBackendSelectionPolicy::traceAssumption).toList(),
                new GpuLoweredPartitionTrace.CompoundSummary(
                        source.fusedSummary().backend().name(), source.fusedSummary().patternType().name(),
                        source.fusedSummary().supported(), source.fusedSummary().reason().name(),
                        source.fusedSummary().orderedNodeIds(), source.fusedSummary().externalInputNodeIds(),
                        source.fusedSummary().outputNodeIds(), source.fusedSummary().dagNodeTypes(),
                        source.fusedSummary().postOps(), source.fusedSummary().detail()
                ),
                source.fusedSubpatterns().stream()
                        .map(summary -> new GpuLoweredPartitionTrace.FusedSubpattern(
                                summary.patternType().name(), summary.supported(),
                                summary.originalOperationNodeIds(), summary.loweredPrimitiveIds(),
                                summary.loweredPrimitiveCount(), summary.reason().name(), summary.detail()
                        ))
                        .toList(),
                source.rejections().stream()
                        .map(rejection -> new GpuLoweredPartitionTrace.Rejection(
                                rejection.level(), rejection.originalNodeId(), rejection.primitiveId(),
                                rejection.fusedPatternType(), rejection.reason().name(), rejection.detail()
                        ))
                        .toList(),
                new GpuLoweredPartitionTrace.CandidateSpan(
                        source.candidateSpan().originalCandidateNodeIds(), source.candidateSpan().acceptedNodeIds(),
                        source.candidateSpan().rejectedOriginalNodeId(), source.candidateSpan().rejectedPrimitiveId(),
                        source.candidateSpan().reason().name()
                ),
                source.backendExtensions()
        );
    }

    private static GpuLoweredPartitionTrace.ValueAssumption traceAssumption(
            backend.accelerator.lowering.GpuLoweredPartitionValueAssumption source
    ) {
        return new GpuLoweredPartitionTrace.ValueAssumption(
                source.nodeId(), source.role(), source.dataType().name(), source.rank(), source.shape(),
                source.layout(), source.contiguous(), source.hasStorageOffset(), source.storageOffset()
        );
    }
}
