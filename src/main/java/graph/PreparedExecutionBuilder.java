package graph;

import backend.accelerator.exec.PartitionExecutionRole;
import backend.accelerator.select.DefaultAcceleratorSelectionPolicy;
import backend.accelerator.select.AcceleratorSelectionResult;
import backend.prepare.BackendPrepareContext;
import backend.prepare.BackendPrepareDispatcher;
import graph.execution.CompiledNodeExecutionMetadata;
import graph.execution.PreparedExecution;
import graph.execution.PreparedNodeExecution;
import graph.execution.trace.PrepareTrace;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

final class PreparedExecutionBuilder {
    private PreparedExecutionBuilder() {
    }

    static PreparedExecution prepare(CompiledGraph graph, config.runtime.RuntimeConfig runtimeConfig) {
        long t0 = System.nanoTime();
        List<CompiledNode> compiledNodes = graph.compiledNodesView();
        Map<Integer, List<CompiledNode>> consumers = buildConsumerMap(compiledNodes);
        BackendPrepareContext context = new BackendPrepareContext(
                runtimeConfig,
                graph.supportsBackward(),
                compiledNodes,
                consumers
        );
        AcceleratorSelectionResult selection = new DefaultAcceleratorSelectionPolicy().select(
                graph.compiledAcceleratorCandidatesView(),
                runtimeConfig
        );
        context.publishAcceleratorPlans(selection.selectedPlans());
        BackendPrepareDispatcher dispatcher = BackendPrepareDispatcher.from(runtimeConfig);

        List<PreparedNodeExecution> forwardSteps = new ArrayList<>();
        List<PreparedNodeExecution> backwardSteps = new ArrayList<>();
        for (CompiledNode node : compiledNodes) {
            if (node.operation() == null || node.inputTensors().isEmpty()) {
                continue;
            }
            CompiledNodeExecutionMetadata metadata = dispatcher.prepare(node, context);
            context.publishPreparedMetadata(node.id(), metadata);
            if (metadata.partitionRole() == PartitionExecutionRole.INTERIOR) {
                continue;
            }
            PreparedNodeExecution step = new PreparedNodeExecution(node, metadata);
            if (node.id() <= graph.forwardBoundaryNodeId()) {
                forwardSteps.add(step);
            } else {
                backwardSteps.add(step);
            }
        }

        return new PreparedExecution(
                runtimeConfig,
                graph.supportsBackward(),
                forwardSteps,
                backwardSteps,
                compiledNodes,
                graph.compiledGradientBindings(),
                graph.getRootTensor(),
                graph.compiledForwardOutputNode(),
                graph.forwardSeedGradient(),
                new PrepareTrace(
                        true,
                        System.nanoTime() - t0,
                        forwardSteps.size(),
                        backwardSteps.size(),
                        selection.trace()
                )
        );
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
