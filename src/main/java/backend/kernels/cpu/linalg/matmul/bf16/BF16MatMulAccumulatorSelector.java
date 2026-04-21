package backend.kernels.cpu.linalg.matmul.bf16;

import config.backend.CpuMatMulMicroKernel;

final class BF16MatMulAccumulatorSelector {
    private BF16MatMulAccumulatorSelector() {
    }

    static BF16AccumKernel select(CpuMatMulMicroKernel microKernel) {
        return switch (microKernel) {
            case BF16_2X4 -> BF16MatMulKernel2x4::accumulate;
            case BF16_4X4 -> BF16MatMulKernel4x4::accumulate;
            default -> BF16MatMulKernel4x2::accumulate;
        };
    }
}
