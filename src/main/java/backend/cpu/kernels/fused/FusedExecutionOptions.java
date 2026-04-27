package backend.cpu.kernels.fused;

import backend.cpu.kernels.*;

public record FusedExecutionOptions(
        boolean useFastExpApprox,
        boolean useFastTanhApprox
) {
    private static final FusedExecutionOptions EXACT = new FusedExecutionOptions(false, false);

    public static FusedExecutionOptions exact() {
        return EXACT;
    }
}
