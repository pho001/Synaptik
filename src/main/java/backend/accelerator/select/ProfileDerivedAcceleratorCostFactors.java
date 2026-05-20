package backend.accelerator.select;

import backend.ComputeBackend;
import config.runtime.RuntimeConfig;
import graph.compile.planning.partition.cost.AcceleratorPartitionScoreModel;

public record ProfileDerivedAcceleratorCostFactors(
        String presetName,
        long minimumEstimatedWork,
        int contiguousMaterializeThreshold,
        double dispatchOverhead,
        double uploadBytePenalty,
        double downloadBytePenalty,
        double layoutFallbackBytePenalty
) {
    private static final int DEFAULT_CONTIGUOUS_MATERIALIZE_THRESHOLD = 4096;

    public ProfileDerivedAcceleratorCostFactors {
        presetName = presetName == null || presetName.isBlank() ? "PROFILE_DERIVED" : presetName;
        minimumEstimatedWork = Math.max(0L, minimumEstimatedWork);
        contiguousMaterializeThreshold = Math.max(1, contiguousMaterializeThreshold);
        dispatchOverhead = Math.max(0.0d, dispatchOverhead);
        uploadBytePenalty = Math.max(0.0d, uploadBytePenalty);
        downloadBytePenalty = Math.max(0.0d, downloadBytePenalty);
        layoutFallbackBytePenalty = Math.max(0.0d, layoutFallbackBytePenalty);
    }

    public static ProfileDerivedAcceleratorCostFactors fromRuntimeConfig(
            RuntimeConfig runtimeConfig,
            ComputeBackend backend
    ) {
        var conservative = AcceleratorPartitionScoreModel.StaticCostPreset.conservative();
        long minimumEstimatedWork = runtimeConfig == null || backend == null
                ? 0L
                : runtimeConfig.accelerator().forBackend(backend).minimumEstimatedWork();
        int contiguousMaterializeThreshold = runtimeConfig == null
                ? DEFAULT_CONTIGUOUS_MATERIALIZE_THRESHOLD
                : runtimeConfig.kernel().cpu().contiguousMaterializeThreshold();
        double scale = 4096.0d / Math.max(1024, contiguousMaterializeThreshold);
        return new ProfileDerivedAcceleratorCostFactors(
                "PROFILE_DERIVED",
                minimumEstimatedWork,
                contiguousMaterializeThreshold,
                conservative.dispatchOverhead() + Math.min(10_000.0d, minimumEstimatedWork * 0.001d),
                conservative.uploadBytePenalty() * scale,
                conservative.downloadBytePenalty() * scale,
                conservative.layoutFallbackBytePenalty() * scale
        );
    }

    public AcceleratorPartitionScoreModel.StaticCostPreset toStaticCostPreset() {
        var conservative = AcceleratorPartitionScoreModel.StaticCostPreset.conservative();
        return new AcceleratorPartitionScoreModel.StaticCostPreset(
                presetName,
                conservative.boundaryPenalty(),
                uploadBytePenalty,
                downloadBytePenalty,
                conservative.tensorArrayFallbackBytePenalty(),
                layoutFallbackBytePenalty,
                conservative.avoidedIntermediateByteCredit(),
                dispatchOverhead,
                conservative.computeWorkCredit()
        );
    }
}
