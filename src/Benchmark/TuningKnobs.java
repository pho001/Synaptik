package Benchmark;

import Config.backend.KernelTuningConfig;
import Config.optimizer.FuseConfig;

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

    public TuningKnobs(
            boolean strictCseSafety,
            FuseConfig fuseConfig,
            KernelTuningConfig kernelConfig
    ) {
        this.strictCseSafety = strictCseSafety;
        this.fuseConfig = fuseConfig;
        this.kernelConfig = kernelConfig;
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
