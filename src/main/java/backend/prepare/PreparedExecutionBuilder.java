package backend.prepare;

import backend.accelerator.exec.PartitionExecutionRole;
import backend.select.BackendSelectionResult;
import backend.select.DefaultBackendSelectionPolicy;
import backend.lowering.BackendCapabilities;
import backend.lowering.LoweringInput;
import backend.lowering.LoweringContext;
import backend.lowering.LoweringPipeline;
import backend.partition.BackendPartitionDescriptorRegistry;
import graph.CompiledNode;
import graph.compile.CompileArtifacts;
import graph.execution.CompiledNodeExecutionMetadata;
import graph.execution.PreparedExecution;
import graph.execution.PreparedNodeExecution;
import graph.execution.trace.PrepareTrace;
import graph.optimizer.partition.PartitionPlan;
import graph.optimizer.partition.PlannedPartition;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

public final class PreparedExecutionBuilder {
    private PreparedExecutionBuilder() {
    }

    public static PreparedExecution prepare(CompileArtifacts artifacts, config.runtime.RuntimeConfig runtimeConfig) {
        Objects.requireNonNull(artifacts, "artifacts cannot be null");
        Objects.requireNonNull(runtimeConfig, "runtimeConfig cannot be null");
        long t0 = System.nanoTime();
        List<CompiledNode> compiledNodes = artifacts.compiledNodes();
        Map<Integer, List<CompiledNode>> consumers = buildConsumerMap(compiledNodes);
        BackendPrepareContext context = new BackendPrepareContext(
                runtimeConfig,
                artifacts.supportsBackward(),
                compiledNodes,
                artifacts.descriptorIndex(),
                consumers
        );
        BackendSelectionResult selection = new DefaultBackendSelectionPolicy().select(
                artifacts.plannedPartitions(),
                runtimeConfig
        );
        context.publishBackendPlans(selection.selectedPlans());
        LoweringInput loweringInput = artifacts.loweringInput();
        publishLoweredRegions(artifacts, compiledNodes, context, runtimeConfig, selection, loweringInput);
        BackendPrepareDispatcher dispatcher = BackendPrepareDispatcher.from(runtimeConfig);

        List<PreparedNodeExecution> executionSteps = new ArrayList<>();
        List<PreparedNodeExecution> forwardSteps = new ArrayList<>();
        List<PreparedNodeExecution> backwardSteps = new ArrayList<>();
        for (CompiledNode node : compiledNodes) {
            if (node.operation() == null || node.inputIds().isEmpty()) {
                continue;
            }
            CompiledNodeExecutionMetadata metadata = dispatcher.prepare(node, context);
            context.publishPreparedMetadata(node.id(), metadata);
            if (metadata.partitionRole() == PartitionExecutionRole.INTERIOR) {
                continue;
            }
            PreparedNodeExecution step = new PreparedNodeExecution(node, metadata);
            executionSteps.add(step);
            if (node.id() <= artifacts.forwardBoundaryNodeId()) {
                forwardSteps.add(step);
            } else {
                backwardSteps.add(step);
            }
        }
        return new PreparedExecution(
                runtimeConfig,
                artifacts.supportsBackward(),
                executionSteps,
                forwardSteps,
                backwardSteps,
                compiledNodes,
                artifacts.descriptorIndex(),
                artifacts.gradientBindings(),
                artifacts.rootTensor(),
                artifacts.forwardOutputNode(),
                artifacts.forwardSeedGradient(),
                loweringInput == null ? artifacts.memoryPlan() : loweringInput.memoryPlan(),
                new PrepareTrace(
                        true,
                        System.nanoTime() - t0,
                        forwardSteps.size(),
                        backwardSteps.size(),
                        selection.trace(),
                        BackendPrepareTraceContributors.diagnostics(selection, loweringInput)
                )
        );
    }

    private static void publishLoweredRegions(
            CompileArtifacts artifacts,
            List<CompiledNode> compiledNodes,
            BackendPrepareContext context,
            config.runtime.RuntimeConfig runtimeConfig,
            BackendSelectionResult selection,
            LoweringInput loweringInput
    ) {
        if (loweringInput == null || loweringInput.optimizedRegions().isEmpty() || loweringInput.memoryPlan() == null) {
            return;
        }
        LoweringPipeline pipeline = new LoweringPipeline(BackendPartitionDescriptorRegistry.defaults().lowerers());
        Map<String, PartitionPlan> selectedPlansByPartitionId =
                selectedPlansByPartitionId(selection);
        Set<backend.ComputeBackend> supportedBackends = new java.util.LinkedHashSet<>();
        supportedBackends.add(backend.ComputeBackend.CPU);
        for (PartitionPlan plan : selectedPlansByPartitionId.values()) {
            if (plan != null) {
                supportedBackends.add(plan.backend());
            }
        }
        var lowered = pipeline.lower(
                loweringInput,
                new BackendCapabilities(supportedBackends),
                new LoweringContext(
                        runtimeConfig,
                        compiledNodes,
                        artifacts.descriptorIndex(),
                        selectedPlansByPartitionId
                )
        );
        context.publishLoweredRegions(lowered.lowered().loweredRegions());
    }

    private static Map<String, PartitionPlan> selectedPlansByPartitionId(BackendSelectionResult selection) {
        if (selection == null || selection.selectedPartitions().isEmpty()) {
            return Map.of();
        }
        HashMap<String, PartitionPlan> out = new HashMap<>();
        for (PlannedPartition selectedPartition : selection.selectedPartitions()) {
            if (selectedPartition == null
                    || selectedPartition.partition() == null
                    || selectedPartition.plan() == null) {
                continue;
            }
            String partitionId = selectedPartition.partition().partitionId();
            if (partitionId != null && !partitionId.isBlank()) {
                out.put(partitionId, selectedPartition.plan());
            }
        }
        return Map.copyOf(out);
    }

    private static Map<Integer, List<CompiledNode>> buildConsumerMap(List<CompiledNode> graph) {
        Map<Integer, List<CompiledNode>> consumers = new HashMap<>();
        for (CompiledNode node : graph) {
            consumers.computeIfAbsent(node.id(), ignored -> new ArrayList<>());
        }
        for (CompiledNode node : graph) {
            for (int inputId : node.inputIds()) {
                consumers.computeIfAbsent(inputId, ignored -> new ArrayList<>()).add(node);
            }
        }
        return consumers;
    }

}
