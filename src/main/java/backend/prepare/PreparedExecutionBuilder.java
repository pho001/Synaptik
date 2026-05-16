package backend.prepare;

import backend.accelerator.exec.PartitionExecutionRole;
import backend.select.BackendSelectionResult;
import backend.select.DefaultBackendSelectionPolicy;
import backend.lowering.BackendCapabilities;
import backend.lowering.LoweringContext;
import backend.lowering.LoweringPipeline;
import backend.partition.BackendPartitionDescriptorRegistry;
import backend.cpu.nativecpu.NativeCpuChainPlanner;
import graph.CompiledNode;
import graph.compile.CompileArtifacts;
import graph.execution.CompiledNodeExecutionMetadata;
import graph.execution.PreparedExecution;
import graph.execution.PreparedNodeExecution;
import graph.execution.trace.PrepareTrace;
import graph.optimizer.partition.BackendCandidatePartition;
import graph.optimizer.partition.PartitionPlan;
import graph.optimizer.state.OptimizerState;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.IdentityHashMap;
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
                artifacts.backendSelectionCandidates(),
                runtimeConfig
        );
        context.publishBackendPlans(selection.selectedPlans());
        OptimizerState loweringInput = artifacts.requireLoweringReadyOptimizerState();
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
        executionSteps = new ArrayList<>(NativeCpuChainPlanner.annotate(executionSteps, runtimeConfig));
        forwardSteps = new ArrayList<>();
        backwardSteps = new ArrayList<>();
        for (PreparedNodeExecution step : executionSteps) {
            if (step.compiledNode().id() <= artifacts.forwardBoundaryNodeId()) {
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
                        selection.trace()
                )
        );
    }

    private static void publishLoweredRegions(
            CompileArtifacts artifacts,
            List<CompiledNode> compiledNodes,
            BackendPrepareContext context,
            config.runtime.RuntimeConfig runtimeConfig,
            BackendSelectionResult selection,
            OptimizerState loweringInput
    ) {
        if (loweringInput == null || loweringInput.optimizedRegions().isEmpty() || loweringInput.memoryPlan() == null) {
            return;
        }
        LoweringPipeline pipeline = new LoweringPipeline(BackendPartitionDescriptorRegistry.defaults().lowerers());
        Map<String, PartitionPlan> selectedPlansByPartitionId =
                selectedPlansByPartitionId(artifacts, selection);
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

    private static Map<String, PartitionPlan> selectedPlansByPartitionId(
            CompileArtifacts artifacts,
            BackendSelectionResult selection
    ) {
        if (selection == null || selection.selectedPlans().isEmpty()) {
            return Map.of();
        }
        IdentityHashMap<PartitionPlan, String> partitionIdByPlan = new IdentityHashMap<>();
        for (BackendCandidatePartition candidate : artifacts.backendSelectionCandidates()) {
            if (candidate == null || candidate.plan() == null || candidate.partition() == null) {
                continue;
            }
            partitionIdByPlan.put(candidate.plan(), candidate.partition().partitionId());
        }
        HashMap<String, PartitionPlan> out = new HashMap<>();
        for (PartitionPlan selectedPlan : selection.selectedPlans()) {
            String partitionId = partitionIdByPlan.get(selectedPlan);
            if (partitionId != null && !partitionId.isBlank()) {
                out.put(partitionId, selectedPlan);
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
