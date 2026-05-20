package graph.compile;

import backend.lowering.LoweringInput;
import graph.CompiledNode;
import graph.CompiledProgram;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.compile.planning.memory.MemoryPlan;
import graph.compile.planning.partition.Partition;
import graph.compile.planning.partition.PartitionPlan;
import graph.compile.planning.partition.PlannedPartition;
import graph.compile.planning.region.OptimizedRegion;
import graph.compile.publication.PublicationPlan;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Thin compile-result aggregate that separates executable program data from publication bindings.
 */
public record CompileArtifacts(
        CompiledProgram program,
        PublicationPlan publication
) {
    public CompileArtifacts {
        program = Objects.requireNonNull(program, "program cannot be null");
        publication = Objects.requireNonNull(publication, "publication cannot be null");
    }

    public List<CompiledNode> compiledNodes() {
        return program.compiledNodes();
    }

    public CompiledTensorDescriptorIndex descriptorIndex() {
        return program.descriptorIndex();
    }

    public boolean supportsBackward() {
        return program.supportsBackward();
    }

    public int forwardBoundaryNodeId() {
        return program.forwardBoundaryNodeId();
    }

    public int forwardOutputNodeId() {
        return program.forwardOutputNodeId();
    }

    public CompiledNode forwardOutputNode() {
        return program.forwardOutputNode();
    }

    public MemoryPlan memoryPlan() {
        return program.memoryPlan();
    }

    public List<OptimizedRegion> optimizedRegions() {
        return program.optimizedRegions();
    }

    public List<PlannedPartition> plannedPartitions() {
        return program.plannedPartitions();
    }

    public List<Partition> partitions() {
        return program.partitions();
    }

    public List<PartitionPlan> backendPlans() {
        return program.backendPlans();
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
        if (program.optimizedRegions().isEmpty() || program.memoryPlan() == null) {
            throw new IllegalStateException("Compile artifacts are missing lowering input.");
        }
        return new LoweringInput(program.optimizedRegions(), program.memoryPlan(), planByPartitionId());
    }

    public boolean requiresLoweringInput() {
        return !program.plannedPartitions().isEmpty() && !program.optimizedRegions().isEmpty();
    }

    private Map<String, PartitionPlan> planByPartitionId() {
        java.util.HashMap<String, PartitionPlan> out = new java.util.HashMap<>();
        for (PlannedPartition plannedPartition : program.plannedPartitions()) {
            if (plannedPartition == null || plannedPartition.partition() == null || plannedPartition.plan() == null) {
                continue;
            }
            out.put(plannedPartition.partition().partitionId(), plannedPartition.plan());
        }
        return Map.copyOf(out);
    }
}
