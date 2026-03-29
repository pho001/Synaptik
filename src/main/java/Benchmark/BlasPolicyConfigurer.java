package Benchmark;

import Backend.blas.BlasRuntime;

public final class BlasPolicyConfigurer {
    private BlasPolicyConfigurer() {}

    public static void apply(TuningKnobs knobs) {
        if (knobs == null) {
            return;
        }
        System.setProperty(BlasRuntime.PROP_PROVIDER, knobs.blasProvider());
        System.setProperty(BlasRuntime.PROP_MATMUL_MIN_WORK, Long.toString(knobs.blasMatMulMinWork()));
        System.setProperty(BlasRuntime.PROP_F32_REQUIRE_M_GE_K, Boolean.toString(knobs.blasF32RequireMgeK()));
        System.setProperty(BlasRuntime.PROP_F32_MAX_N_OVER_K, Double.toString(knobs.blasF32MaxNOverK()));
    }
}
