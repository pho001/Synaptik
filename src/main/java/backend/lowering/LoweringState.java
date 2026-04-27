package backend.lowering;

import graph.optimizer.state.OptimizerState;

import java.util.Objects;

public record LoweringState(
        OptimizerState optimized,
        LoweringArtifacts lowered,
        LoweringTrace trace
) {
    public LoweringState {
        optimized = Objects.requireNonNull(optimized, "optimized cannot be null");
        lowered = lowered == null ? LoweringArtifacts.empty() : lowered;
        trace = trace == null ? LoweringTrace.empty() : trace;
    }
}
