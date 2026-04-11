package backend.kernels.cpu.fused;

import backend.kernels.cpu.*;

public record FusedExecutionOptions(
        boolean useFastExpApprox,
        boolean useFastTanhApprox
) {
    private static final FusedExecutionOptions EXACT = new FusedExecutionOptions(false, false);

    public static FusedExecutionOptions exact() {
        return EXACT;
    }
}
