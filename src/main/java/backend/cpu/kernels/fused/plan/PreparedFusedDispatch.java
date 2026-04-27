package backend.cpu.kernels.fused.plan;

import backend.cpu.kernels.elementwise.plan.ResolvedDispatchHints;

import java.util.Objects;

public record PreparedFusedDispatch(
        ResolvedDispatchHints dispatchHints,
        int cpuVectorMinSize,
        int asmVectorWidth
) {
    public PreparedFusedDispatch {
        Objects.requireNonNull(dispatchHints, "dispatchHints cannot be null");
        cpuVectorMinSize = Math.max(1, cpuVectorMinSize);
        asmVectorWidth = Math.max(1, asmVectorWidth);
    }
}
