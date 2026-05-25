package backend.cpu.plan.fused;

import backend.cpu.plan.elementwise.ResolvedDispatchHints;
import backend.cpu.fused.plan.FusedVectorFallbackReason;

import java.util.Objects;

public record PreparedFusedDispatch(
        ResolvedDispatchHints dispatchHints,
        int cpuVectorMinSize,
        int asmVectorWidth,
        FusedVectorFallbackReason vectorFallbackReason
) {
    public PreparedFusedDispatch {
        Objects.requireNonNull(dispatchHints, "dispatchHints cannot be null");
        cpuVectorMinSize = Math.max(1, cpuVectorMinSize);
        asmVectorWidth = Math.max(1, asmVectorWidth);
        vectorFallbackReason = vectorFallbackReason == null ? FusedVectorFallbackReason.NONE : vectorFallbackReason;
    }
}
