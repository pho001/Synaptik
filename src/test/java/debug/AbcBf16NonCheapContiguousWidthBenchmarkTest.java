package debug;

import runtime.contract.ExecutionMode;
import config.backend.CpuKernelConfig;
import config.backend.KernelTuningConfig;
import config.profile.ExecutionProfile;
import config.runtime.RuntimeConfig;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tuning.benchmark.report.TextBenchmarkReportRenderer;
import tuning.benchmark.BenchmarkEntry;
import tuning.benchmark.BenchmarkRequest;
import tuning.benchmark.BenchmarkSession;
import tuning.store.HardwareFingerprint;
import tuning.store.JsonFileBestProfileStore;
import tuning.calibration.store.PlatformCalibrationPaths;
import tuning.validate.ValidationPolicy;
import tuning.workload.StandardWorkloads;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

final class AbcBf16NonCheapContiguousWidthBenchmarkTest {
    private static final tuning.measure.MeasurementPolicy MEASUREMENT = DebugMeasurementPolicies.STANDARD_WITH_TRACE;

    @Test
    void compareCurrentBestAgainstNonCheapContiguousWidths() {
        ExecutionProfile best = loadBestProfile();
        ExecutionProfile width2 = withRuntime(best, "noncheap-contiguous-w2", runtimeWithNonCheapContiguousAsmWidth(best.runtime(), 2));
        ExecutionProfile width4 = withRuntime(best, "noncheap-contiguous-w4", runtimeWithNonCheapContiguousAsmWidth(best.runtime(), 4));

        BenchmarkRequest request = new BenchmarkRequest(
                StandardWorkloads.abcSequenceMatmulBlasBenchmark("abc_sequence_matmul_bf16_noncheap_contiguous_probe"),
                List.of(
                        BenchmarkEntry.candidate("best-current", best),
                        BenchmarkEntry.candidate("best-current-noncheap-contiguous-w2", width2),
                        BenchmarkEntry.candidate("best-current-noncheap-contiguous-w4", width4)
                ),
                MEASUREMENT,
                ValidationPolicy.disabled(),
                tuning.reporting.ReportPolicy.defaults()
        );

        var report = BenchmarkSession.create(request).run();
        System.out.println();
        System.out.println("ABC_BF16_NONCHEAP_CONTIGUOUS_WIDTH_BENCHMARK");
        System.out.println(TextBenchmarkReportRenderer.render(report));
    }

    private static ExecutionProfile loadBestProfile() {
        Path profilePath = resolveExisting(
                Path.of("profiles", "platform", PlatformCalibrationPaths.platformId(HardwareFingerprint.capture()), "tuning", "abc", "bf16-best-profile.json"),
                Path.of("build", "tuning", "best-profiles", "abc-bf16-best-profile.json")
        );
        return new JsonFileBestProfileStore()
                .load(profilePath)
                .orElseThrow(() -> new IllegalStateException("Missing best profile for bf16"))
                .profile();
    }

    private static ExecutionProfile withRuntime(
            ExecutionProfile base,
            String suffix,
            RuntimeConfig runtime
    ) {
        return new ExecutionProfile(
                base.profileName() + "-" + suffix,
                base.candidateName() + "-" + suffix,
                DataType.BFLOAT16,
                ExecutionMode.FORWARD_BACKWARD,
                base.compile(),
                runtime,
                base.workload()
        );
    }

    private static RuntimeConfig runtimeWithNonCheapContiguousAsmWidth(RuntimeConfig base, int width) {
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
