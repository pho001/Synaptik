package backend.lowering;

import graph.optimizer.region.OptimizedRegion;
import graph.optimizer.state.OptimizerState;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Runs registered backend lowerers over optimized regions after partition/fusion/memory planning.
 *
 * <p>Lowering is a prepare-time bridge between graph optimization and backend execution. It receives an
 * optimizer state with finalized memory planning, invokes backend-specific {@link RegionLowerer}s for
 * each optimized region, and returns lowering artifacts plus a simple trace.</p>
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
     * @param optimized optimizer state after partition/fusion/memory planning
     * @param capabilities backend capabilities available for this prepare step
     * @param context lowering context containing runtime config and selected partition plans
     * @return lowering state containing artifacts and trace
     * @throws IllegalStateException if the optimizer state has no finalized memory plan
     */
    public LoweringState lower(
            OptimizerState optimized,
            BackendCapabilities capabilities,
            LoweringContext context
    ) {
        Objects.requireNonNull(optimized, "optimized cannot be null");
        Objects.requireNonNull(capabilities, "capabilities cannot be null");
        Objects.requireNonNull(context, "context cannot be null");
        if (optimized.memoryPlan() == null) {
            throw new IllegalStateException("Lowering requires a finalized memory plan.");
        }
        LoweringContext effectiveContext = context.withPartitionPlans(
                context.partitionPlansById().isEmpty()
                        ? optimized.partitionPlansById()
                        : context.partitionPlansById()
        );
        List<LoweredRegion> loweredRegions = new ArrayList<>();
        List<BackendWorkspaceRequirement> requirements = new ArrayList<>();
        List<String> events = new ArrayList<>();
        for (OptimizedRegion region : optimized.optimizedRegions()) {
            LoweringRequest request = new LoweringRequest(region, optimized.memoryPlan(), capabilities, effectiveContext);
            LoweringResult result = lowerRegion(request);
            if (result == null || result.loweredRegion() == null) {
                continue;
            }
            loweredRegions.add(result.loweredRegion());
            requirements.addAll(result.workspaceRequirements());
            events.add("lowered:" + region.regionId());
        }
        return new LoweringState(
                optimized,
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
