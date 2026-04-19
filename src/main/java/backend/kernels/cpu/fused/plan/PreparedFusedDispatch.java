package backend.kernels.cpu.fused.plan;

import backend.kernels.cpu.elementwise.plan.ResolvedDispatchHints;

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
