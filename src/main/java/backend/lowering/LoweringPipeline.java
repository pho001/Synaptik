package backend.lowering;

import graph.optimizer.region.OptimizedRegion;
import graph.optimizer.state.OptimizerState;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public final class LoweringPipeline {
    private final List<RegionLowerer> lowerers;

    public LoweringPipeline(List<RegionLowerer> lowerers) {
        this.lowerers = List.copyOf(lowerers == null ? List.of() : lowerers);
    }

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
