package backend.lowering;

import planning.partition.ExecutablePartitionPlan;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Runs registered backend lowerers over executable partitions after memory planning.
 *
 * <p>Lowering is a prepare-time bridge between compile planning and backend execution. It receives executable partitions
 * with a finalized memory plan, invokes backend-specific {@link PartitionLowerer}s for each partition, and returns lowering
 * artifacts plus a simple trace.</p>
 */
public final class LoweringPipeline {
    private final List<PartitionLowerer> lowerers;

    /**
     * Creates a lowering pipeline.
     *
     * @param lowerers lowerers tried in order for each optimized partition; {@code null} becomes empty
     */
    public LoweringPipeline(List<PartitionLowerer> lowerers) {
        this.lowerers = List.copyOf(lowerers == null ? List.of() : lowerers);
    }

    /**
     * Lowers optimized partitions into backend artifacts.
     *
     * @param input lowering input after partition/fusion/memory planning
     * @param capabilities backend capabilities available for this prepare step
     * @param context lowering context containing runtime config and selected partition plans
     * @return lowering state containing artifacts and trace
     * @throws IllegalStateException if the input has no finalized memory plan
     */
    public LoweringState lower(
            LoweringInput input,
            BackendCapabilities capabilities,
            LoweringContext context
    ) {
        Objects.requireNonNull(input, "input cannot be null");
        Objects.requireNonNull(capabilities, "capabilities cannot be null");
        Objects.requireNonNull(context, "context cannot be null");
        if (input.memoryPlan() == null) {
            throw new IllegalStateException("Lowering requires a finalized memory plan.");
        }
        List<LoweredPartition> loweredPartitions = new ArrayList<>();
        List<BackendWorkspaceRequirement> requirements = new ArrayList<>();
        List<String> events = new ArrayList<>();
        for (ExecutablePartitionPlan executablePartition : input.executablePartitions()) {
            LoweringRequest request = new LoweringRequest(executablePartition, input.memoryPlan(), capabilities, context);
            LoweringResult result = lowerPartition(request);
            if (result == null || result.loweredPartition() == null) {
                continue;
            }
            loweredPartitions.add(result.loweredPartition());
            requirements.addAll(result.workspaceRequirements());
            events.add("lowered:" + executablePartition.partition().partitionId());
        }
        return new LoweringState(
                input,
                new LoweringArtifacts(loweredPartitions, requirements),
                new LoweringTrace(events)
        );
    }

    private LoweringResult lowerPartition(LoweringRequest request) {
        for (PartitionLowerer lowerer : lowerers) {
            LoweringResult result = lowerer.lowerPartition(request);
            if (result != null && result.loweredPartition() != null) {
                return result;
            }
        }
        return null;
    }
}
