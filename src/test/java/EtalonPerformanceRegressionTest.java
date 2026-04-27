import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import tuning.etalon.FrameworkEtalon;
import tuning.benchmark.report.BenchmarkCandidateReport;
import tuning.benchmark.report.BenchmarkSuiteReport;
import tuning.benchmark.BenchmarkSuiteSession;
import tuning.store.JsonFileBenchmarkReportStore;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Properties;

import static org.junit.jupiter.api.Assertions.assertTrue;

@Tag("benchmark")
public class EtalonPerformanceRegressionTest {
    private static final double WORKLOAD_REGRESSION_TOLERANCE = 0.40d;
    private static final double WORKLOAD_ABSOLUTE_NOISE_FLOOR_MS = 0.050d;

    @Test
    void inferenceEtalonMatchesPerformanceBaseline() throws IOException {
        BenchmarkSuiteReport report = BenchmarkSuiteSession
                .create(FrameworkEtalon.inferenceRegressionSuite())
                .run();

        Path out = Path.of("build", "tuning-etalon-regression", "current-inference-suite.json");
        new JsonFileBenchmarkReportStore().saveSuite(out, report);

        Properties baseline = loadBaseline();
        List<String> regressions = new ArrayList<>();
        List<String> observations = new ArrayList<>();

        verifyWorkloadMetric(report, baseline, "etalon_matmul_small", "f32_infer_default", regressions, observations);
        verifyWorkloadMetric(report, baseline, "etalon_matmul_small", "f64_infer_default", regressions, observations);
        verifyWorkloadMetric(report, baseline, "etalon_abc_sequence_small", "f32_infer_default", regressions, observations);
        verifyWorkloadMetric(report, baseline, "etalon_abc_sequence_small", "f64_infer_default", regressions, observations);
        verifyWorkloadMetric(report, baseline, "etalon_conv2d_resnet_3x3", "f64_infer_default", regressions, observations);
        verifyWorkloadMetric(report, baseline, "etalon_layer_norm_small", "f32_infer_default", regressions, observations);
        verifyWorkloadMetric(report, baseline, "etalon_layer_norm_small", "f64_infer_default", regressions, observations);
        verifyWorkloadMetric(report, baseline, "etalon_max_pool2d_small", "f32_infer_default", regressions, observations);
        verifyWorkloadMetric(report, baseline, "etalon_max_pool2d_small", "f64_infer_default", regressions, observations);

        observations.sort(Comparator.naturalOrder());
        StringBuilder message = new StringBuilder(2048);
        message.append("Inference etalon performance regression check.\n");
        message.append("Current suite JSON: ").append(out).append('\n');
        if (!observations.isEmpty()) {
            message.append("Observed deltas:\n");
            for (String observation : observations) {
                message.append("  ").append(observation).append('\n');
            }
        }
        if (!regressions.isEmpty()) {
            message.append("Regressions:\n");
            for (String regression : regressions) {
                message.append("  ").append(regression).append('\n');
            }
        }

        assertTrue(regressions.isEmpty(), message.toString());
    }

    private static Properties loadBaseline() throws IOException {
        Properties properties = new Properties();
        try (InputStream in = EtalonPerformanceRegressionTest.class.getClassLoader()
                .getResourceAsStream("tuning/etalon/inference-performance-baseline.properties")) {
            if (in == null) {
                throw new IllegalStateException("Missing inference performance baseline resource.");
            }
            properties.load(in);
        }
        return properties;
    }

    private static void verifyWorkloadMetric(
            BenchmarkSuiteReport report,
            Properties baseline,
            String workloadName,
            String candidateName,
            List<String> regressions,
            List<String> observations
    ) {
        double expectedMedian = Double.parseDouble(require(baseline, "workload." + workloadName + "." + candidateName));
        BenchmarkCandidateReport candidate = report.workloadReports().stream()
                .filter(r -> r.workloadName().equals(workloadName))
                .findFirst()
                .flatMap(r -> r.candidates().stream().filter(c -> c.entry().name().equals(candidateName)).findFirst())
                .orElseThrow(() -> new IllegalStateException("Missing workload candidate " + workloadName + "/" + candidateName));

        if (!candidate.success() || candidate.measurement() == null) {
            regressions.add("workload." + workloadName + "." + candidateName + " missing successful measurement");
            return;
        }

        compareMetric(
                "workload." + workloadName + "." + candidateName,
                expectedMedian,
                candidate.measurement().steadyStateStats().medianMs(),
                WORKLOAD_REGRESSION_TOLERANCE,
                regressions,
                observations
        );
    }

    private static void compareMetric(
            String key,
            double baseline,
            double current,
            double tolerance,
            List<String> regressions,
            List<String> observations
    ) {
        if (!Double.isFinite(current)) {
            regressions.add(key + " is not finite");
            return;
        }
        double ratio = baseline == 0.0d ? Double.NaN : current / baseline;
        double deltaPct = baseline == 0.0d ? Double.NaN : ((current - baseline) / baseline) * 100.0d;
        observations.add(String.format(Locale.US,
                "%s baseline=%.6f current=%.6f delta=%+.2f%% ratio=%.3f",
                key, baseline, current, deltaPct, ratio));
        double absoluteDelta = current - baseline;
        if (current > baseline * (1.0d + tolerance) && absoluteDelta > WORKLOAD_ABSOLUTE_NOISE_FLOOR_MS) {
            regressions.add(String.format(Locale.US,
                    "%s regressed beyond %.0f%% tolerance and %.3fms noise floor: baseline=%.6f current=%.6f delta=%+.2f%%",
                    key, tolerance * 100.0d, WORKLOAD_ABSOLUTE_NOISE_FLOOR_MS, baseline, current, deltaPct));
        }
    }

    private static String require(Properties properties, String key) {
        String value = properties.getProperty(key);
        if (value == null || value.isBlank()) {
            throw new IllegalStateException("Missing baseline property: " + key);
        }
        return value;
    }
}
