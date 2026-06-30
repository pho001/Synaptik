package debug;

import runtime.contract.ExecutionMode;
import config.backend.CpuKernelConfig;
import config.backend.KernelTuningConfig;
import config.compile.CompileConfig;
import config.profile.ExecutionProfile;
import config.profile.WorkloadKind;
import config.profile.WorkloadProfile;
import config.runtime.ApproximationConfig;
import config.runtime.BlasConfig;
import config.runtime.FusedExecutionPolicy;
import config.runtime.RuntimeConfig;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tuning.benchmark.report.TextBenchmarkReportRenderer;
import tuning.benchmark.BenchmarkEntry;
import tuning.benchmark.BenchmarkRequest;
import tuning.benchmark.BenchmarkSession;
import tuning.validate.ValidationPolicy;
import tuning.workload.CalibrationWorkloads;
import tuning.workload.StandardWorkloads;
import tuning.workload.WorkloadSpec;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class HotPathAnalysisTest {
    private static final tuning.measure.MeasurementPolicy ANALYSIS_MEASUREMENT = DebugMeasurementPolicies.STANDARD_WITH_TRACE;

    @Test
    void printRepresentativeInferenceHotPaths() {
        run(
                CalibrationWorkloads.fusedCheapElementwise("analysis_fused_cheap", 1_048_576),
                inferenceProfile("analysis-inference")
        );
        run(
                CalibrationWorkloads.fusedCheapStridedElementwise("analysis_fused_cheap_strided", 512, 512),
                inferenceProfile("analysis-inference")
        );
        run(
                CalibrationWorkloads.fusedTranscendental("analysis_fused_transcendental", 1_048_576),
                inferenceProfile("analysis-inference")
        );
        run(
                CalibrationWorkloads.reductionSum("analysis_reduction_sum", 1_048_576),
                inferenceProfile("analysis-inference")
        );
        run(
                StandardWorkloads.matmul("analysis_matmul_attention_like", 8, 128, 64, 64),
                inferenceProfile("analysis-inference")
        );
        run(
                StandardWorkloads.transformerHotPath("analysis_transformer_hot_path"),
                transformerInferenceProfile("analysis-transformer")
        );
        run(
                StandardWorkloads.transformerHotPath("analysis_transformer_hot_path_fused_vec"),
                transformerFusedVectorProfile("analysis-transformer-fused-vec")
        );
    }

    private static void run(WorkloadSpec workload, ExecutionProfile profile) {
        var report = BenchmarkSession.create(new BenchmarkRequest(
                workload,
                java.util.List.of(BenchmarkEntry.candidate(profile.candidateName(), profile)),
                ANALYSIS_MEASUREMENT,
                ValidationPolicy.disabled(),
                tuning.reporting.ReportPolicy.defaults()
        )).run();

        System.out.println();
        System.out.println("=== HOT PATH ANALYSIS :: " + workload.name() + " :: " + profile.candidateName() + " ===");
        System.out.println(TextBenchmarkReportRenderer.render(report));

        assertEquals(1, report.successCount(), "Expected analyzed workload to execute successfully.");
        assertTrue(report.candidates().getFirst().measurement().trace().run().durationNs() >= 0L);
    }

    private static ExecutionProfile inferenceProfile(String candidateName) {
        return new ExecutionProfile(
                candidateName,
                candidateName,
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                CompileConfig.inference(),
                RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
    }

    private static ExecutionProfile transformerInferenceProfile(String candidateName) {
        return new ExecutionProfile(
                candidateName,
                candidateName,
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                CompileConfig.inference(),
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

    private static ExecutionProfile transformerFusedVectorProfile(String candidateName) {
        return new ExecutionProfile(
                candidateName,
                candidateName,
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                CompileConfig.inference(),
                runtimeWithFusedAsmWidths(4, 4, 4, 4),
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
                                cpu.sumAccuracyMode(),
                                cpu.matMulParallelMinSize(),
                                cpu.attentionMatMulPolicy()
                        ),
                        base.kernel().cuda(),
                        base.kernel().opencl()
                ),
                ApproximationConfig.defaults(),
                BlasConfig.disabled(),
                new FusedExecutionPolicy(true)
        );
    }
}
