package backend.lowering;

/**
 * Backend-specific lowering hook for optimized graph regions.
 *
 * <p>Lowerers translate optimizer regions into backend execution units or accelerator DAG artifacts.
 * A lowerer returns {@code null} or a result with no lowered region when it cannot handle the request,
 * allowing the lowering pipeline to try the next registered lowerer.</p>
 */
public interface RegionLowerer {
    /**
     * Attempts to lower one optimized region.
     *
     * @param request lowering request containing region, memory plan, capabilities, and context
     * @return lowering result, or {@code null} when this lowerer does not support the region
     */
    LoweringResult lower(LoweringRequest request);
}
