package backend.lowering;

import planning.region.PlannedRegion;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Runs registered backend lowerers over optimized regions after partition/fusion/memory planning.
 *
 * <p>Lowering is a prepare-time bridge between compile planning and backend execution. It receives optimized regions
 * with a finalized memory plan, invokes backend-specific {@link RegionLowerer}s for each region, and returns lowering
 * artifacts plus a simple trace.</p>
 */
public final class LoweringPipeline {
    private final List<RegionLowerer> lowerers;

    /**
     * Creates a lowering pipeline.
     *
     * @param lowerers lowerers tried in order for each optimized region; {@code null} becomes empty
     */
    public LoweringPipeline(List<RegionLowerer> lowerers) {
        this.lowerers = List.copyOf(lowerers == null ? List.of() : lowerers);
    }

    /**
     * Lowers optimized regions into backend artifacts.
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
        LoweringInput effectiveInput = context.partitionPlansById().isEmpty()
                ? input
                : input.withPartitionPlans(context.partitionPlansById());
        LoweringContext effectiveContext = context.withPartitionPlans(
                context.partitionPlansById().isEmpty()
                        ? input.partitionPlansById()
                        : context.partitionPlansById()
        );
        List<LoweredRegion> loweredRegions = new ArrayList<>();
        List<BackendWorkspaceRequirement> requirements = new ArrayList<>();
        List<String> events = new ArrayList<>();
        for (PlannedRegion region : effectiveInput.plannedRegions()) {
            LoweringRequest request = new LoweringRequest(region, effectiveInput.memoryPlan(), capabilities, effectiveContext);
            LoweringResult result = lowerRegion(request);
            if (result == null || result.loweredRegion() == null) {
                continue;
            }
            loweredRegions.add(result.loweredRegion());
            requirements.addAll(result.workspaceRequirements());
            events.add("lowered:" + region.regionId());
        }
        return new LoweringState(
                effectiveInput,
                new LoweringArtifacts(loweredRegions, requirements),
                new LoweringTrace(events)
        );
    }

    private LoweringResult lowerRegion(LoweringRequest request) {
        for (RegionLowerer lowerer : lowerers) {
            LoweringResult result = lowerer.lower(request);
            if (result != null && result.loweredRegion() != null) {
                return result;
            }
        }
        return null;
    }
}
