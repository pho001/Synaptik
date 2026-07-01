package graph.compile;

import backend.lowering.LoweringInput;
import planning.descriptor.CompiledTensorDescriptorIndex;
import planning.memory.MemoryPlan;
import planning.partition.Partition;
import planning.partition.PartitionPlan;
import planning.partition.ExecutablePartitionPlan;
import planning.partition.PlannedPartition;
import graph.compile.publication.PublicationPlan;
import graph.model.CompiledNode;

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

    public List<ExecutablePartitionPlan> executablePartitions() {
        return program.executablePartitions();
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
     * @return lowering input, or {@code null} when no executable partitions require lowering
     * @throws IllegalStateException if executable partitions exist but memory plan is missing
     */
    public LoweringInput loweringInput() {
        if (!requiresLoweringInput()) {
            return null;
        }
        if (program.executablePartitions().isEmpty() || program.memoryPlan() == null) {
            throw new IllegalStateException("Compile artifacts are missing lowering input.");
        }
        return new LoweringInput(program.executablePartitions(), program.memoryPlan());
    }

    public boolean requiresLoweringInput() {
        return program.partitionExecutionEnabled() && !program.executablePartitions().isEmpty();
    }
}
