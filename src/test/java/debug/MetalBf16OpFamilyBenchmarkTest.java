package debug;

import runtime.contract.ExecutionMode;
import config.compile.BackendPlanningConfig;
import config.compile.CompileConfig;
import config.optimizer.CpuFusionConfig;
import config.optimizer.CpuRegionConfig;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import config.runtime.RuntimeConfig;
import runtime.execution.PublicationPolicy;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.loss.LossReduction;
import tuning.benchmark.BenchmarkEntry;
import tuning.benchmark.BenchmarkRequest;
import tuning.benchmark.BenchmarkSession;
import tuning.benchmark.report.BenchmarkCandidateReport;
import tuning.measure.MeasurementPolicy;
import tuning.validate.ValidationPolicy;
import tuning.workload.NormalizationWorkloadSpec;
import tuning.workload.StandardWorkloads;
import tuning.workload.WorkloadSpec;

import java.util.List;
import java.util.Locale;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

final class MetalBf16OpFamilyBenchmarkTest {
    private static final MeasurementPolicy MEASUREMENT = new MeasurementPolicy(
            3,
            8,
            3,
            true,
            true,
            true,
            true,
            true,
            PublicationPolicy.NONE
    );

    @Test
    void benchmarkMetalBf16VsF32OpFamilies() {
        assumeMetalMpsConfigured();
        List<Family> families = List.of(
                new Family(
                        "matmul",
                        StandardWorkloads.matmul("metal_dtype_family_matmul", 1, 512, 512, 512)
                ),
                new Family(
                        "mlp",
                        StandardWorkloads.mlpClassification(
                                "metal_dtype_family_mlp",
                                64,
                                256,
                                512,
                                256,
                                32,
                                LossReduction.MEAN
                        )
                ),
                new Family(
                        "layer_norm",
                        StandardWorkloads.normalization(
                                "metal_dtype_family_layer_norm",
                                NormalizationWorkloadSpec.NormalizationKind.LAYER_NORM,
                                16,
                                256,
                                16,
                                1,
                                1e-5
                        )
                ),
                new Family(
                        "reduction",
                        StandardWorkloads.reductionChain("metal_dtype_family_reduction", 64, 1024)
                ),
                new Family(
                        "transformer_block_medium_train",
                        StandardWorkloads.transformerBlockHotPath(
                                "metal_dtype_family_transformer_block_medium_train",
                                WorkloadProfile.transformerHotPathMedium()
                        ),
                        ExecutionMode.FORWARD_BACKWARD
                )
        );

        System.out.println();
        System.out.println("METAL_BF16_OP_FAMILY_BENCHMARK");
        System.out.printf(
                Locale.US,
                "%-34s %-12s %-12s %-12s %-12s %-12s %-24s %-12s%n",
                "family",
                "f32Ms",
                "bf16Ms",
                "bf16/f32",
                "f32Status",
                "bf16Status",
                "bf16Evidence",
                "winner"
        );
        for (Family family : families) {
            var report = BenchmarkSession.create(request(family)).run();
            BenchmarkCandidateReport f32 = report.candidates().stream()
                    .filter(candidate -> candidate.entry().name().equals("f32-metal"))
                    .findFirst()
                    .orElseThrow();
            BenchmarkCandidateReport bf16 = report.candidates().stream()
                    .filter(candidate -> candidate.entry().name().equals("bf16-metal"))
                    .findFirst()
                    .orElseThrow();
            double f32Ms = medianMs(f32);
            double bf16Ms = medianMs(bf16);
            double ratio = f32Ms > 0.0d && bf16Ms > 0.0d ? bf16Ms / f32Ms : Double.NaN;
            System.out.printf(
                    Locale.US,
                    "%-34s %-12s %-12s %-12s %-12s %-12s %-24s %-12s%n",
                    family.name(),
                    formatMs(f32Ms),
                    formatMs(bf16Ms),
                    Double.isNaN(ratio) ? "n/a" : String.format(Locale.US, "%.3f", ratio),
                    f32.success() ? "OK" : "FAIL",
                    bf16.success() ? "OK" : "FAIL",
                    bf16Evidence(f32, bf16, ratio),
                    report.bestCandidateName().isBlank() ? "n/a" : report.bestCandidateName()
            );
            if (!f32.success() || !bf16.success()) {
                System.out.println("  f32Failure=" + f32.failureReason());
                System.out.println("  bf16Failure=" + bf16.failureReason());
            }
        }
    }

    private static BenchmarkRequest request(Family family) {
        return new BenchmarkRequest(
                family.workload(),
                List.of(
                        BenchmarkEntry.candidate("f32-metal", profile(family, DataType.FLOAT32, "f32")),
                        BenchmarkEntry.candidate("bf16-metal", profile(family, DataType.BFLOAT16, "bf16"))
                ),
                MEASUREMENT,
                ValidationPolicy.disabled(),
                tuning.reporting.ReportPolicy.defaults()
        );
    }

    private static ExecutionProfile profile(Family family, DataType dataType, String dtypeId) {
        return new ExecutionProfile(
                "metal-" + family.name() + "-" + dtypeId,
                dtypeId + "-metal",
                dataType,
                family.mode(),
                compileConfig(family.mode()),
                family.mode() == ExecutionMode.FORWARD_BACKWARD
                        ? RuntimeConfig.trainingDefaults()
                        : RuntimeConfig.inferenceDefaults(),
                family.mode() == ExecutionMode.FORWARD_BACKWARD
                        ? WorkloadProfile.transformerHotPathMedium()
                        : WorkloadProfile.none()
        );
    }

    private static CompileConfig compileConfig(ExecutionMode mode) {
        CompileConfig base = mode == ExecutionMode.FORWARD_BACKWARD
                ? CompileConfig.training()
                : CompileConfig.inference();
        return base
                .withBackendPlanning(BackendPlanningConfig.autoAccelerator().withCpuRegions(CpuRegionConfig.defaults()))
                .withRegionOptimization(base.regionOptimization().withCpuFusion(CpuFusionConfig.defaults()));
    }

    private static double medianMs(BenchmarkCandidateReport candidate) {
        return candidate.measurement() == null ? Double.NaN : candidate.measurement().steadyStateStats().medianMs();
    }

    private static String bf16Evidence(BenchmarkCandidateReport f32, BenchmarkCandidateReport bf16, double ratio) {
        if (!f32.success()) {
            return "f32_baseline_failed";
        }
        if (!bf16.success()) {
            return "bf16_unsupported_or_bug";
        }
        if (Double.isNaN(ratio)) {
            return "insufficient_data";
        }
        if (ratio <= 0.75d) {
            return "native_likely";
        }
        if (ratio <= 0.95d) {
            return "mixed_or_promoted_likely";
        }
        return "promoted_or_overhead_bound";
    }

    private static String formatMs(double value) {
        return Double.isNaN(value) ? "n/a" : String.format(Locale.US, "%.6f", value);
    }

    private static void assumeMetalMpsConfigured() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank(), "Set -Dsynaptik.metal.mps.lib to run Metal benchmark.");
    }

    private record Family(String name, WorkloadSpec workload, ExecutionMode mode) {
        private Family(String name, WorkloadSpec workload) {
            this(name, workload, ExecutionMode.FORWARD);
        }
    }
}
