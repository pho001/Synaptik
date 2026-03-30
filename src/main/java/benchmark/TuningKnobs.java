package benchmark;

import config.backend.KernelTuningConfig;
import config.optimizer.FuseConfig;
import backend.blas.BlasRuntime;

/**
 * Unified tuning knobs for optimizer/runtime search.
 * Some fields are currently placeholders until relevant kernels are implemented.
 */
public final class TuningKnobs {
    // Ready today (Fusion rule consumes only the booleans through candidate policy)
    private final boolean strictCseSafety;
    private final FuseConfig fuseConfig;

    // Backend/kernel tuning profile (prepared, can be gradually wired into kernels/codegen)
    private final KernelTuningConfig kernelConfig;
    // Runtime BLAS routing policy (used by CPU matmul path when provider is enabled).
    private final String blasProvider;
    private final long blasMatMulMinWork;
    private final boolean blasF32RequireMgeK;
    private final double blasF32MaxNOverK;

    public TuningKnobs(
            boolean strictCseSafety,
            FuseConfig fuseConfig,
            KernelTuningConfig kernelConfig
    ) {
        this(
                strictCseSafety,
                fuseConfig,
                kernelConfig,
                BlasRuntime.DEFAULT_PROVIDER,
                BlasRuntime.DEFAULT_MATMUL_MIN_WORK,
                BlasRuntime.DEFAULT_F32_REQUIRE_M_GE_K,
                BlasRuntime.DEFAULT_F32_MAX_N_OVER_K
        );
    }

    public TuningKnobs(
            boolean strictCseSafety,
            FuseConfig fuseConfig,
            KernelTuningConfig kernelConfig,
            String blasProvider,
            long blasMatMulMinWork,
            boolean blasF32RequireMgeK,
            double blasF32MaxNOverK
    ) {
        this.strictCseSafety = strictCseSafety;
        this.fuseConfig = fuseConfig;
        this.kernelConfig = kernelConfig;
        this.blasProvider = (blasProvider == null || blasProvider.isBlank())
                ? BlasRuntime.DEFAULT_PROVIDER
                : blasProvider.trim();
        this.blasMatMulMinWork = blasMatMulMinWork > 0 ? blasMatMulMinWork : BlasRuntime.DEFAULT_MATMUL_MIN_WORK;
        this.blasF32RequireMgeK = blasF32RequireMgeK;
        this.blasF32MaxNOverK = blasF32MaxNOverK > 0.0d ? blasF32MaxNOverK : BlasRuntime.DEFAULT_F32_MAX_N_OVER_K;
    }

    public boolean strictCseSafety() {
        return strictCseSafety;
    }

    public boolean preserveSharedExpensiveNodes() {
        return fuseConfig.preserveSharedExpensiveNodes();
    }

    public FuseConfig fuseConfig() {
        return fuseConfig;
    }

    public KernelTuningConfig kernelConfig() {
        return kernelConfig;
    }

    public int loopUnrollFactor() {
        return kernelConfig.cpu().loopUnrollFactor();
    }

    public String blasProvider() {
        return blasProvider;
    }

    public long blasMatMulMinWork() {
        return blasMatMulMinWork;
    }

    public boolean blasF32RequireMgeK() {
        return blasF32RequireMgeK;
    }

    public double blasF32MaxNOverK() {
        return blasF32MaxNOverK;
    }

    public TuningKnobs withBlasPolicy(
            String provider,
            long matMulMinWork,
            boolean f32RequireMgeK,
            double f32MaxNOverK
    ) {
        return new TuningKnobs(
                strictCseSafety,
                fuseConfig,
                kernelConfig,
                provider,
                matMulMinWork,
                f32RequireMgeK,
                f32MaxNOverK
        );
    }

    public static TuningKnobs trainingDefaults() {
        return new TuningKnobs(
                true,
                FuseConfig.trainingDefaults(),
                KernelTuningConfig.defaultsTraining()
        );
    }

    public static TuningKnobs inferencePerfDefaults() {
        return new TuningKnobs(
                false,
                FuseConfig.inferencePerfDefaults(),
                KernelTuningConfig.defaultsInference()
        );
    }
}
