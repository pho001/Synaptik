package tuning.etalon;

import tuning.benchmark.report.JsonBenchmarkSuiteReportRenderer;
import tuning.benchmark.report.TextBenchmarkSuiteReportRenderer;
import tuning.benchmark.BenchmarkSuiteSession;
import tuning.preset.TuningPreset;
import tuning.store.JsonFileBenchmarkReportStore;

import java.nio.file.Path;

public final class FrameworkEtalonCli {
    private FrameworkEtalonCli() {
    }

    public static void main(String[] args) {
        String suite = System.getProperty("etalon.suite", "all").trim().toLowerCase(java.util.Locale.ROOT);
        TuningPreset preset = TuningPreset.valueOf(
                System.getProperty("etalon.preset", TuningPreset.BALANCED.name()).trim().toUpperCase(java.util.Locale.ROOT)
        );
        Path outDir = Path.of(System.getProperty("etalon.outDir", "build/tuning-etalon"));

        if (suite.equals("inference") || suite.equals("all")) {
            var report = BenchmarkSuiteSession.create(FrameworkEtalon.inferenceSuite(preset)).run();
            System.out.println("=== Inference Etalon ===");
            System.out.println(TextBenchmarkSuiteReportRenderer.render(report));
            new JsonFileBenchmarkReportStore().saveSuite(outDir.resolve("inference-suite.json"), report);
        }
        if (suite.equals("training") || suite.equals("all")) {
            var report = BenchmarkSuiteSession.create(FrameworkEtalon.trainingSuite(preset)).run();
            System.out.println("=== Training Etalon ===");
            System.out.println(TextBenchmarkSuiteReportRenderer.render(report));
            new JsonFileBenchmarkReportStore().saveSuite(outDir.resolve("training-suite.json"), report);
        }
    }
}
