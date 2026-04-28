package config.profile;

/**
 * Calibrated dispatch thresholds for non-fused elementwise CPU kernels.
 *
 * <p>Cheap operations and transcendental operations have separate vector and parallel thresholds
 * because their per-element cost differs. The values are measured during platform calibration and later
 * copied into {@link config.backend.CpuKernelConfig}.</p>
 *
 * @param cheapVectorMinSize minimum element count before vectorizing cheap elementwise operations
 * @param transcendentalVectorMinSize minimum element count before vectorizing transcendental operations
 * @param cheapParallelMinSize minimum element count before parallelizing cheap operations
 * @param transcendentalParallelMinSize minimum element count before parallelizing transcendental operations
 */
public record ElementwiseDispatchPlatformProfile(
        int cheapVectorMinSize,
        int transcendentalVectorMinSize,
        int cheapParallelMinSize,
        int transcendentalParallelMinSize
) {}
