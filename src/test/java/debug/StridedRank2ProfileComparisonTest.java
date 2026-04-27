package debug;

import backend.runtime.ExecutionMode;
import config.backend.CpuKernelConfig;
import config.optimizer.OptimizerConfig;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import config.runtime.RuntimeConfig;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tuning.benchmark.report.TextBenchmarkReportRenderer;
import tuning.benchmark.BenchmarkEntry;
import tuning.benchmark.BenchmarkRequest;
import tuning.benchmark.BenchmarkSession;
import tuning.validate.ValidationPolicy;
import tuning.workload.CalibrationWorkloads;

import java.util.List;

final class StridedRank2ProfileComparisonTest {
    private static final tuning.measure.MeasurementPolicy MEASUREMENT = DebugMeasurementPolicies.STANDARD;

    @Test
    void compareF64StridedVsMaterialized() {
        run(DataType.FLOAT64, "f64");
    }

    @Test
    void compareF32StridedVsMaterialized() {
        run(DataType.FLOAT32, "f32");
    }

    private static void run(DataType dataType, String dtypeId) {
        BenchmarkRequest request = new BenchmarkRequest(
                CalibrationWorkloads.materializationStridedElementwise("rank2_strided_compare_" + dtypeId, 512, 512),
                List.of(
                        BenchmarkEntry.candidate("strided-path", profile(dataType, dtypeId + "-strided", 1_000_000)),
                        BenchmarkEntry.candidate("forced-materialize", profile(dataType, dtypeId + "-materialized", 0))
                ),
                MEASUREMENT,
                ValidationPolicy.disabled(),
                tuning.reporting.ReportPolicy.defaults()
        );

        var report = BenchmarkSession.create(request).run();
        System.out.println();
        System.out.println("STRIDED_RANK2_PROFILE_COMPARISON :: " + dtypeId);
        System.out.println(TextBenchmarkReportRenderer.render(report));
    }

    private static ExecutionProfile profile(DataType dataType, String candidateName, int materializeThreshold) {
        return new ExecutionProfile(
                candidateName,
                candidateName,
                dataType,
                ExecutionMode.FORWARD,
                OptimizerConfig.noOptimization(),
                new RuntimeConfig(
                        new CpuKernelConfig(4, 32, 32, 32, 1_024, 100_000, materializeThreshold),
                        config.runtime.ApproximationConfig.defaults(),
                        config.runtime.BlasConfig.disabled()
                ),
                WorkloadProfile.none()
        );
    }
}
