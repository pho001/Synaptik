package backend.lowering;

import java.util.Objects;

public record LoweringState(
        LoweringInput input,
        LoweringArtifacts lowered,
        LoweringTrace trace
) {
    public LoweringState {
        input = Objects.requireNonNull(input, "input cannot be null");
        lowered = lowered == null ? LoweringArtifacts.empty() : lowered;
        trace = trace == null ? LoweringTrace.empty() : trace;
    }
}
