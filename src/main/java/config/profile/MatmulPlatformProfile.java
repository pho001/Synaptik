package config.profile;

import backend.blas.BlasProvider;
import config.backend.CpuMatMulMicroKernel;

import java.util.Objects;

/**
 * Calibrated matmul and attention-matmul runtime settings.
 *
 * <p>This profile section stores the BLAS dispatch thresholds, Java/CPU tile sizes, parallel threshold,
 * and selected microkernels used by matmul execution. Attention-specific tile and microkernel fields
 * allow calibration to tune attention workloads separately while still keeping one matmul runtime
 * profile section.</p>
 *
 * @param blasProvider BLAS provider eligible for matmul dispatch
 * @param blasMatmulMinWork minimum regular matmul work before BLAS dispatch is eligible
 * @param blasThreads BLAS thread count; currently normalized to {@code 0} for provider default behavior
 * @param f32RequireMgeK whether F32 regular BLAS dispatch requires {@code M >= K}
 * @param f32MaxNOverK maximum {@code N / K} ratio for regular F32 BLAS dispatch
 * @param f32WideRequireMgeK whether F32 wide BLAS dispatch requires {@code M >= K}
 * @param f32WideMaxNOverK maximum {@code N / K} ratio for wide F32 BLAS dispatch
 * @param loopUnrollFactor scalar loop unroll factor used by CPU kernels
 * @param matMulTileM CPU matmul tile size in the M dimension
 * @param matMulTileN CPU matmul tile size in the N dimension
 * @param matMulTileK CPU matmul tile size in the K dimension
 * @param attentionMatMulTileM attention matmul tile size in the M dimension
 * @param attentionMatMulTileN attention matmul tile size in the N dimension
 * @param attentionMatMulTileK attention matmul tile size in the K dimension
 * @param matMulParallelMinSize minimum matmul work before CPU parallel dispatch is eligible
 * @param matMulMicroKernel selected regular matmul microkernel; {@code null} becomes {@code AUTO}
 * @param attentionMatMulMicroKernel selected attention matmul microkernel; {@code null} falls back to regular microkernel
 */
public record MatmulPlatformProfile(
        BlasProvider blasProvider,
        long blasMatmulMinWork,
        int blasThreads,
        boolean f32RequireMgeK,
        double f32MaxNOverK,
        boolean f32WideRequireMgeK,
        double f32WideMaxNOverK,
        int loopUnrollFactor,
        int matMulTileM,
        int matMulTileN,
        int matMulTileK,
        int attentionMatMulTileM,
        int attentionMatMulTileN,
        int attentionMatMulTileK,
        int matMulParallelMinSize,
        CpuMatMulMicroKernel matMulMicroKernel,
        CpuMatMulMicroKernel attentionMatMulMicroKernel
) {
    public MatmulPlatformProfile {
        blasProvider = Objects.requireNonNull(blasProvider, "blasProvider cannot be null");
        blasThreads = 0;
        attentionMatMulTileM = attentionMatMulTileM <= 0 ? matMulTileM : attentionMatMulTileM;
        attentionMatMulTileN = attentionMatMulTileN <= 0 ? matMulTileN : attentionMatMulTileN;
        attentionMatMulTileK = attentionMatMulTileK <= 0 ? matMulTileK : attentionMatMulTileK;
        matMulMicroKernel = matMulMicroKernel == null ? CpuMatMulMicroKernel.AUTO : matMulMicroKernel;
        attentionMatMulMicroKernel = attentionMatMulMicroKernel == null ? matMulMicroKernel : attentionMatMulMicroKernel;
        f32WideMaxNOverK = f32WideMaxNOverK > 0.0d ? f32WideMaxNOverK : f32MaxNOverK;
    }

    /**
     * Creates a matmul profile whose wide-F32 dispatch settings mirror regular F32 dispatch.
     *
     * @param blasProvider BLAS provider
     * @param blasMatmulMinWork minimum BLAS work
     * @param blasThreads BLAS thread count; normalized to {@code 0}
     * @param f32RequireMgeK regular F32 {@code M >= K} requirement
     * @param f32MaxNOverK regular F32 {@code N / K} ratio limit
     * @param loopUnrollFactor scalar loop unroll factor
     * @param matMulTileM regular matmul M tile
     * @param matMulTileN regular matmul N tile
     * @param matMulTileK regular matmul K tile
     * @param attentionMatMulTileM attention M tile
     * @param attentionMatMulTileN attention N tile
     * @param attentionMatMulTileK attention K tile
     * @param matMulParallelMinSize parallel dispatch threshold
     * @param matMulMicroKernel regular microkernel
     * @param attentionMatMulMicroKernel attention microkernel
     */
    public MatmulPlatformProfile(
            BlasProvider blasProvider,
            long blasMatmulMinWork,
            int blasThreads,
            boolean f32RequireMgeK,
            double f32MaxNOverK,
            int loopUnrollFactor,
            int matMulTileM,
            int matMulTileN,
            int matMulTileK,
            int attentionMatMulTileM,
            int attentionMatMulTileN,
            int attentionMatMulTileK,
            int matMulParallelMinSize,
            CpuMatMulMicroKernel matMulMicroKernel,
            CpuMatMulMicroKernel attentionMatMulMicroKernel
    ) {
        this(
                blasProvider,
                blasMatmulMinWork,
                blasThreads,
                f32RequireMgeK,
                f32MaxNOverK,
                f32RequireMgeK,
                f32MaxNOverK,
                loopUnrollFactor,
                matMulTileM,
                matMulTileN,
                matMulTileK,
                attentionMatMulTileM,
                attentionMatMulTileN,
                attentionMatMulTileK,
                matMulParallelMinSize,
                matMulMicroKernel,
                attentionMatMulMicroKernel
        );
    }
}
