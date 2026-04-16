package debug;

import backend.runtime.ExecutionMode;
import config.backend.CpuKernelConfig;
import config.backend.KernelTuningConfig;
import config.optimizer.OptimizerConfig;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import config.profile.WorkloadKind;
import config.runtime.ApproximationConfig;
import config.runtime.BlasConfig;
import config.runtime.FusedExecutionPolicy;
import config.runtime.FusedPrimaryBackend;
import config.runtime.RuntimeConfig;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tuning.report.TextBenchmarkReportRenderer;
import tuning.session.BenchmarkEntry;
import tuning.session.BenchmarkRequest;
import tuning.session.BenchmarkSession;
import tuning.validate.ValidationPolicy;
import tuning.workload.StandardWorkloads;

import java.util.List;

final class TransformerFusedProfileComparisonTest {
    private static final tuning.measure.MeasurementPolicy COMPARISON_MEASUREMENT = DebugMeasurementPolicies.STANDARD;

    @Test
    void compareDefaultVsFusedVecTransformerProfile() {
        var request = new BenchmarkRequest(
                StandardWorkloads.transformerHotPath("compare_transformer_fused_profiles"),
                List.of(
                        BenchmarkEntry.baseline("default-inference", transformerInferenceProfile("default-inference")),
                        BenchmarkEntry.candidate("noncheap-strided-w2", transformerFamilySpecificProfile("noncheap-strided-w2", 1, 1, 1, 2)),
                        BenchmarkEntry.candidate("noncheap-strided-w4", transformerFamilySpecificProfile("noncheap-strided-w4", 1, 1, 1, 4))
                ),
                COMPARISON_MEASUREMENT,
                ValidationPolicy.disabled(),
                tuning.report.ReportPolicy.defaults()
        );

        var report = BenchmarkSession.create(request).run();
        System.out.println();
        System.out.println(TextBenchmarkReportRenderer.render(report));
    }

    private static ExecutionProfile transformerInferenceProfile(String candidateName) {
        return new ExecutionProfile(
                candidateName,
                candidateName,
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                OptimizerConfig.inferenceDefaults(),
                RuntimeConfig.inferenceDefaults(),
                new WorkloadProfile(
                        WorkloadKind.TRANSFORMER_HOT_PATH,
                        4,
                        8,
                        64,
                        32,
                        32,
                        512,
                        true
                )
        );
    }

    private static ExecutionProfile transformerFamilySpecificProfile(
            String candidateName,
            int cheapContiguous,
            int cheapStrided,
            int nonCheapContiguous,
            int nonCheapStrided
    ) {
        return new ExecutionProfile(
                candidateName,
                candidateName,
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                OptimizerConfig.inferenceDefaults(),
                runtimeWithFusedAsmWidths(cheapContiguous, cheapStrided, nonCheapContiguous, nonCheapStrided),
                new WorkloadProfile(
                        WorkloadKind.TRANSFORMER_HOT_PATH,
                        4,
                        8,
                        64,
                        32,
                        32,
                        512,
                        true
                )
        );
    }

    private static RuntimeConfig runtimeWithFusedAsmWidths(
            int cheapContiguous,
            int cheapStrided,
            int nonCheapContiguous,
            int nonCheapStrided
    ) {
        var base = RuntimeConfig.inferenceDefaults();
        CpuKernelConfig cpu = base.kernel().cpu();
        return new RuntimeConfig(
                new KernelTuningConfig(
                        new CpuKernelConfig(
                                cpu.loopUnrollFactor(),
                                cpu.matMulTileM(),
                                cpu.matMulTileN(),
                                cpu.matMulTileK(),
                                cpu.cheapVectorMinSize(),
                                cpu.transcendentalVectorMinSize(),
                                cpu.fusedCheapVectorMinSize(),
                                cpu.fusedTranscendentalVectorMinSize(),
                                cpu.reductionVectorMinSize(),
                                cpu.cheapParallelMinSize(),
                                cpu.transcendentalParallelMinSize(),
                                cpu.fusedCheapParallelMinSize(),
                                cpu.fusedTranscendentalParallelMinSize(),
                                cpu.reductionParallelMinSize(),
                                cpu.contiguousMaterializeThreshold(),
                                cpu.lowCostTargetChunksPerWorker(),
                                cpu.mediumCostTargetChunksPerWorker(),
                                cpu.highCostTargetChunksPerWorker(),
                                cpu.minScalarChunkSize(),
                                cpu.minVectorChunkSize(),
                                cpu.minReductionChunkSize(),
                                cpu.commonPoolLowCostMaxWorkPerWorker(),
                                cheapContiguous,
                                cheapStrided,
                                nonCheapContiguous,
                                nonCheapStrided,
                                cpu.sumAccuracyMode(),
                                cpu.matMulParallelMinSize(),
                                cpu.attentionMatMulPolicy()
                        ),
                        base.kernel().cuda(),
                        base.kernel().opencl()
                ),
                ApproximationConfig.defaults(),
                BlasConfig.disabled(),
                new FusedExecutionPolicy(FusedPrimaryBackend.ASM, true)
        );
    }
}
