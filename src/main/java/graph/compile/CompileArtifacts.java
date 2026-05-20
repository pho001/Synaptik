package graph.compile;

import backend.lowering.LoweringInput;
import graph.CompiledGradientBinding;
import graph.CompiledNode;
import graph.execution.trace.PartitionCompileTrace;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.compile.planning.memory.MemoryPlan;
import graph.compile.planning.partition.Partition;
import graph.compile.planning.partition.PartitionPlan;
import graph.compile.planning.partition.PlannedPartition;
import graph.compile.planning.region.OptimizedRegion;
import tensor.Tensor;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Immutable output of graph compilation.
 *
 * <p>Artifacts are the boundary between compile and prepare. They contain the optimized tensor graph, stable
 * {@link CompiledNode} snapshots, gradient publication bindings, planned backend partitions, optimized regions needed
 * by lowering, and the memory plan consumed by runtime binding. Preparation must treat this record as read-only.
 *
 * @param rootTensor source root tensor that initiated compilation
 * @param graphContract user-visible graph structure captured at compilation
 * @param finalGraph optimized tensors in execution order
 * @param compiledNodes immutable node snapshots derived from {@code finalGraph}
 * @param descriptorIndex immutable tensor descriptor index derived from {@code compiledNodes}
 * @param gradientBindings mappings used to publish compiled backward outputs to source tensor gradients
 * @param forwardSeedGradient binding used to seed the root gradient for backward execution
 * @param forwardOutputNode compiled node that represents the forward output value
 * @param memoryPlan storage reuse and region handoff plan, if memory planning ran
 * @param optimizedRegions optimized regions derived from accepted partitions
 * @param plannedPartitions accepted backend partitions with attached backend plans
 * @param supportsBackward whether the artifact bundle contains backward execution work
 * @param forwardBoundaryNodeId id of the last forward node in the compiled graph
 * @param partitionPlanningTrace partition planning diagnostics captured during compilation
 */
public record CompileArtifacts(
        Tensor rootTensor,
        GraphStructureContract graphContract,
        List<Tensor> finalGraph,
        List<CompiledNode> compiledNodes,
        CompiledTensorDescriptorIndex descriptorIndex,
        Map<Tensor, CompiledGradientBinding> gradientBindings,
        CompiledGradientBinding forwardSeedGradient,
        CompiledNode forwardOutputNode,
        MemoryPlan memoryPlan,
        List<OptimizedRegion> optimizedRegions,
        List<PlannedPartition> plannedPartitions,
        boolean supportsBackward,
        int forwardBoundaryNodeId,
        PartitionCompileTrace partitionPlanningTrace
) {
    public CompileArtifacts {
        rootTensor = Objects.requireNonNull(rootTensor, "rootTensor cannot be null");
        graphContract = graphContract == null ? GraphStructureContract.unchecked() : graphContract;
        finalGraph = List.copyOf(finalGraph == null ? List.of() : finalGraph);
        compiledNodes = List.copyOf(compiledNodes == null ? List.of() : compiledNodes);
        descriptorIndex = Objects.requireNonNull(descriptorIndex, "descriptorIndex cannot be null");
        gradientBindings = Map.copyOf(gradientBindings == null ? Map.of() : gradientBindings);
        optimizedRegions = List.copyOf(optimizedRegions == null ? List.of() : optimizedRegions);
        plannedPartitions = List.copyOf(plannedPartitions == null ? List.of() : plannedPartitions);
        partitionPlanningTrace = partitionPlanningTrace == null ? PartitionCompileTrace.empty() : partitionPlanningTrace;
    }

    /**
     * Returns accepted backend partitions.
     *
     * @return partitions derived from planned partitions
     */
    public List<Partition> partitions() {
        return plannedPartitions.stream()
                .map(PlannedPartition::partition)
                .toList();
    }

    /**
     * Returns non-null backend plans attached to accepted partitions.
     *
     * @return backend plans derived from planned partitions
     */
    public List<PartitionPlan> backendPlans() {
        return plannedPartitions.stream()
                .map(PlannedPartition::plan)
                .filter(Objects::nonNull)
                .toList();
    }

    /**
     * Returns finalized lowering input for prepare-time backend lowering.
     *
     * @return lowering input, or {@code null} when no planned optimized regions require lowering
     * @throws IllegalStateException if planned optimized regions exist but memory plan is missing
     */
    public LoweringInput loweringInput() {
        if (!requiresLoweringInput()) {
            return null;
        }
        if (optimizedRegions.isEmpty() || memoryPlan == null) {
            throw new IllegalStateException("Compile artifacts are missing lowering input.");
        }
        return new LoweringInput(optimizedRegions, memoryPlan, planByPartitionId());
    }

    public boolean requiresLoweringInput() {
        return !plannedPartitions.isEmpty() && !optimizedRegions.isEmpty();
    }

    private Map<String, PartitionPlan> planByPartitionId() {
        java.util.HashMap<String, PartitionPlan> out = new java.util.HashMap<>();
        for (PlannedPartition plannedPartition : plannedPartitions) {
            if (plannedPartition == null || plannedPartition.partition() == null || plannedPartition.plan() == null) {
                continue;
            }
            out.put(plannedPartition.partition().partitionId(), plannedPartition.plan());
        }
        return Map.copyOf(out);
    }
}
