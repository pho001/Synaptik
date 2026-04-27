package debug;

import backend.runtime.ExecutionMode;
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
import tuning.workload.StandardWorkloads;
import tuning.workload.WorkloadSpec;

import java.util.List;

final class Conv2dCurrentStateBenchmarkTest {
    private static final tuning.measure.MeasurementPolicy MEASUREMENT = DebugMeasurementPolicies.STANDARD;

    @Test
    void benchmarkF64ForwardBackward() {
        runBenchmark(DataType.FLOAT64);
    }

    @Test
    void benchmarkF32ForwardBackward() {
        runBenchmark(DataType.FLOAT32);
    }

    private static void runBenchmark(DataType dataType) {
        WorkloadSpec workload = StandardWorkloads.conv2d(
                "debug_conv2d_resnet_3x3_28",
                2, 64, 128, 28, 28, 3, 3,
                tensor.options.Conv2dOptions.defaults().withPadding(1, 1),
                true
        );
        BenchmarkRequest request = new BenchmarkRequest(
                workload,
                List.of(
                        BenchmarkEntry.baseline("train-default", defaultProfile(dataType))
                ),
                MEASUREMENT,
                ValidationPolicy.disabled(),
                tuning.reporting.ReportPolicy.defaults()
        );

        var report = BenchmarkSession.create(request).run();
        System.out.println();
        System.out.println(TextBenchmarkReportRenderer.render(report));
    }
    private static ExecutionProfile defaultProfile(DataType dataType) {
        return new ExecutionProfile(
                "train-default",
                "train-default",
                dataType,
                ExecutionMode.FORWARD_BACKWARD,
                OptimizerConfig.trainingDefaults(),
                RuntimeConfig.trainingDefaults(),
                WorkloadProfile.none()
        );
    }
}
