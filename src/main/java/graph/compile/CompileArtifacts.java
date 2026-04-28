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

/**
 * Immutable output of graph compilation.
 *
 * <p>Artifacts are the boundary between compile and prepare. They contain the optimized tensor graph, stable
 * {@link CompiledNode} snapshots, gradient publication bindings, optional backend partition plans, the optimizer state
 * needed by lowering, and the memory plan consumed by runtime binding. Preparation must treat this record as read-only.
 *
 * @param rootTensor source root tensor that initiated compilation
 * @param finalGraph optimized tensors in execution order
 * @param compiledNodes immutable node snapshots derived from {@code finalGraph}
 * @param gradientBindings mappings used to publish compiled backward outputs to source tensor gradients
 * @param forwardSeedGradient binding used to seed the root gradient for backward execution
 * @param forwardOutputNode compiled node that represents the forward output value
 * @param memoryPlan storage reuse and region handoff plan, if memory planning ran
 * @param optimizerState final optimizer state retained for prepare-time lowering
 * @param partitions accepted backend partitions
 * @param backendPlans backend-specific plans attached to accepted partitions
 * @param backendSelectionCandidates candidate partitions considered during backend selection
 * @param supportsBackward whether the artifact bundle contains backward execution work
 * @param forwardBoundaryNodeId id of the last forward node in the compiled graph
 * @param partitionPlanningTrace partition planning diagnostics captured during compilation
 */
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

    /**
     * Returns whether prepare-time lowering requires a complete optimizer state.
     *
     * @return {@code true} when backend candidates and partitions exist and optimized regions/memory plan must be
     * present
     */
    public boolean requiresLoweringReadyOptimizerState() {
        return !backendSelectionCandidates.isEmpty() && !partitions.isEmpty();
    }

    /**
     * Returns optimizer state suitable for backend lowering, or fails if required state is incomplete.
     *
     * @return optimizer state, possibly {@code null} when no lowering-ready state is required
     * @throws IllegalStateException if partitions require optimized regions or memory planning data that is absent
     */
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
