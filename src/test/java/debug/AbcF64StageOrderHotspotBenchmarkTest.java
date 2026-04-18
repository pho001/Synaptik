package debug;

import backend.runtime.ExecutionMode;
import config.backend.CpuKernelConfig;
import config.backend.KernelTuningConfig;
import config.optimizer.OptimizerStage;
import config.profile.ExecutionProfile;
import config.runtime.RuntimeConfig;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tuning.report.TextBenchmarkReportRenderer;
import tuning.session.BenchmarkEntry;
import tuning.session.BenchmarkRequest;
import tuning.session.BenchmarkSession;
import tuning.store.HardwareFingerprint;
import tuning.store.JsonFileBestProfileStore;
import tuning.store.PlatformCalibrationPaths;
import tuning.validate.ValidationPolicy;
import tuning.workload.StandardWorkloads;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

final class AbcF64StageOrderHotspotBenchmarkTest {
    private static final tuning.measure.MeasurementPolicy MEASUREMENT = DebugMeasurementPolicies.STANDARD_WITH_TRACE;

    @Test
    void compareArBeforeFuseVariants() {
        ExecutionProfile best = loadBestProfile();
        ExecutionProfile arCseFuse = withStageOrder(best, "ar-cse-fuse", List.of(
                OptimizerStage.AR,
                OptimizerStage.CSE,
                OptimizerStage.FUSE
        ));
        ExecutionProfile bestNonCheapStridedW2 = withRuntime(best, "noncheap-strided-w2", runtimeWithNonCheapStridedAsmWidth(best.runtime(), 2));
        ExecutionProfile cseArFuse = withStageOrder(best, "cse-ar-fuse", List.of(
                OptimizerStage.CSE,
                OptimizerStage.AR,
                OptimizerStage.FUSE
        ));

        BenchmarkRequest request = new BenchmarkRequest(
                StandardWorkloads.abcSequenceMatmulBlasBenchmark("abc_sequence_matmul_f64_stage_order_probe"),
                List.of(
                        BenchmarkEntry.candidate("best-current", best),
                        BenchmarkEntry.candidate("best-current-noncheap-strided-w2", bestNonCheapStridedW2),
                        BenchmarkEntry.candidate("ar-cse-fuse", arCseFuse),
                        BenchmarkEntry.candidate("cse-ar-fuse", cseArFuse)
                ),
                MEASUREMENT,
                ValidationPolicy.disabled(),
                tuning.report.ReportPolicy.defaults()
        );

        var report = BenchmarkSession.create(request).run();
        System.out.println();
        System.out.println("ABC_F64_STAGE_ORDER_HOTSPOT_BENCHMARK");
        System.out.println(TextBenchmarkReportRenderer.render(report));
    }

    private static ExecutionProfile loadBestProfile() {
        Path profilePath = resolveExisting(
                Path.of("profiles", "platform", PlatformCalibrationPaths.platformId(HardwareFingerprint.capture()), "tuning", "abc", "f64-best-profile.json"),
                Path.of("build", "tuning", "best-profiles", "abc-f64-best-profile.json")
        );
        return new JsonFileBestProfileStore()
                .load(profilePath)
                .orElseThrow(() -> new IllegalStateException("Missing best profile for f64"))
                .profile();
    }

    private static ExecutionProfile withStageOrder(
            ExecutionProfile base,
            String suffix,
            List<OptimizerStage> stageOrder
    ) {
        return new ExecutionProfile(
                base.profileName() + "-" + suffix,
                base.candidateName() + "-" + suffix,
                DataType.FLOAT64,
                ExecutionMode.FORWARD_BACKWARD,
                base.optimizer().withStageOrder(stageOrder),
                base.runtime(),
                base.workload()
        );
    }

    private static ExecutionProfile withRuntime(
            ExecutionProfile base,
            String suffix,
            RuntimeConfig runtime
    ) {
        return new ExecutionProfile(
                base.profileName() + "-" + suffix,
                base.candidateName() + "-" + suffix,
                DataType.FLOAT64,
                ExecutionMode.FORWARD_BACKWARD,
                base.optimizer(),
                runtime,
                base.workload()
        );
    }

    private static RuntimeConfig runtimeWithNonCheapStridedAsmWidth(RuntimeConfig base, int width) {
        CpuKernelConfig cpu = base.kernel().cpu();
        CpuKernelConfig patchedCpu = new CpuKernelConfig(
                cpu.loopUnrollFactor(),
                cpu.matMulTileM(),
                cpu.matMulTileN(),
                cpu.matMulTileK(),
                cpu.cheapVectorMinSize(),
                cpu.transcendentalVectorMinSize(),
                cpu.fusedCheapVectorMinSize(),
                cpu.fusedTranscendentalVectorMinSize(),
                cpu.reductionVectorMinSize(),
                cpu.attentionVectorMinSize(),
                cpu.cheapParallelMinSize(),
                cpu.transcendentalParallelMinSize(),
                cpu.fusedCheapParallelMinSize(),
                cpu.fusedTranscendentalParallelMinSize(),
                cpu.reductionParallelMinSize(),
                cpu.attentionParallelMinSize(),
                cpu.contiguousMaterializeThreshold(),
                cpu.lowCostTargetChunksPerWorker(),
                cpu.mediumCostTargetChunksPerWorker(),
                cpu.highCostTargetChunksPerWorker(),
                cpu.minScalarChunkSize(),
                cpu.minVectorChunkSize(),
                cpu.minReductionChunkSize(),
                cpu.commonPoolLowCostMaxWorkPerWorker(),
                cpu.fusedCheapContiguousAsmVectorWidth(),
                cpu.fusedCheapStridedAsmVectorWidth(),
                cpu.fusedNonCheapContiguousAsmVectorWidth(),
                width,
                cpu.sumAccuracyMode(),
                cpu.matMulParallelMinSize(),
                cpu.attentionMatMulPolicy(),
                cpu.matMulMicroKernel(),
                cpu.attentionMatMulMicroKernel(),
                cpu.attentionMatMulTileM(),
                cpu.attentionMatMulTileN(),
                cpu.attentionMatMulTileK()
        );
        return new RuntimeConfig(
                new KernelTuningConfig(patchedCpu, base.kernel().cuda(), base.kernel().opencl()),
                base.approximation(),
                base.blas(),
                base.fused()
        );
    }

    private static Path resolveExisting(Path preferred, Path fallback) {
        return Files.exists(preferred) ? preferred : fallback;
    }
}
