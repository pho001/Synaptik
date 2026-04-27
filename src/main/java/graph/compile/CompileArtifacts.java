package graph.compile;

import graph.CompiledGradientBinding;
import graph.CompiledNode;
import graph.execution.trace.PartitionCompileTrace;
import graph.optimizer.memory.MemoryPlan;
import graph.optimizer.partition.BackendCandidatePartition;
import graph.optimizer.partition.Partition;
import graph.optimizer.partition.PartitionPlan;
import graph.optimizer.state.OptimizerState;
import tensor.Tensor;

import java.util.List;
import java.util.Map;
import java.util.Objects;

public record CompileArtifacts(
        Tensor rootTensor,
        List<Tensor> finalGraph,
        List<CompiledNode> compiledNodes,
        Map<Tensor, CompiledGradientBinding> gradientBindings,
        CompiledGradientBinding forwardSeedGradient,
        CompiledNode forwardOutputNode,
        MemoryPlan memoryPlan,
        OptimizerState optimizerState,
        List<Partition> partitions,
        List<PartitionPlan> backendPlans,
        List<BackendCandidatePartition> backendSelectionCandidates,
        boolean supportsBackward,
        int forwardBoundaryNodeId,
        PartitionCompileTrace partitionPlanningTrace
) {
    public CompileArtifacts {
        rootTensor = Objects.requireNonNull(rootTensor, "rootTensor cannot be null");
        finalGraph = List.copyOf(finalGraph == null ? List.of() : finalGraph);
        compiledNodes = List.copyOf(compiledNodes == null ? List.of() : compiledNodes);
        gradientBindings = Map.copyOf(gradientBindings == null ? Map.of() : gradientBindings);
        partitions = List.copyOf(partitions == null ? List.of() : partitions);
        backendPlans = List.copyOf(backendPlans == null ? List.of() : backendPlans);
        backendSelectionCandidates = List.copyOf(backendSelectionCandidates == null ? List.of() : backendSelectionCandidates);
        partitionPlanningTrace = partitionPlanningTrace == null ? PartitionCompileTrace.empty() : partitionPlanningTrace;
    }

    public boolean requiresLoweringReadyOptimizerState() {
        return !backendSelectionCandidates.isEmpty() && !partitions.isEmpty();
    }

    public OptimizerState requireLoweringReadyOptimizerState() {
        if (!requiresLoweringReadyOptimizerState()) {
            return optimizerState;
        }
        if (optimizerState == null || optimizerState.optimizedRegions().isEmpty() || optimizerState.memoryPlan() == null) {
            throw new IllegalStateException("Compile artifacts are missing lowering-ready optimizer state.");
        }
        return optimizerState;
    }
}
