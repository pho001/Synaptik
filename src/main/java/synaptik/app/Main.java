package synaptik.app;

import config.profile.ExecutionProfile;
import config.profile.PlatformRuntimeProfile;
import tensor.DataType;
import tuning.api.Synaptik;
import tuning.benchmark.report.BenchmarkReport;
import tuning.benchmark.report.TextBenchmarkReportRenderer;
import tuning.calibration.PlatformCalibrationResult;
import tuning.calibration.run.CalibrationCommand;
import tuning.preset.TuningPreset;
import tuning.workload.StandardWorkloads;
import tuning.workload.WorkloadSpec;

import java.nio.file.Path;
import java.util.List;

/**
 * Programmatic tuning entry point.
 *
 * <p>This class is intentionally not a CLI parser. It shows the regular Java API shape for running
 * calibration and benchmarking through the public {@code tuning.api} fluent facade. The command-line
 * tool remains {@link TuningCli}; this entry point is useful when the application should be
 * configured from Java code instead of command tokens.</p>
 */
public final class Main {
    private static final Path PROFILE_ROOT = Path.of("profiles");
    private static final DataType DTYPE = DataType.FLOAT64;
    private static final TuningPreset PRESET = TuningPreset.QUICK;

    private Main() {
    }

    /**
     * Runs a small programmatic calibration and benchmark flow.
     *
     * @param args ignored; this class is not a command-line parser
     */
    public static void main(String[] args) {
        List<PlatformCalibrationResult> calibrationResults = Synaptik.tuning()
                .calibration()
                .dtypes().single(DTYPE)
                .families().all()
                .preset(PRESET)
                .mode().training()
                .measurement().iterations(1, 3, 1)
                .progress().live()
                .color().auto()
                .outputRoot(PROFILE_ROOT)
                .run();

        PlatformRuntimeProfile calibratedRuntime = calibrationResults.getLast().finalRuntimeProfile();
        BenchmarkReport benchmark = benchmarkCalibratedRuntime(DTYPE, calibratedRuntime);
        System.out.println(TextBenchmarkReportRenderer.render(benchmark));
    }

    private static BenchmarkReport benchmarkCalibratedRuntime(DataType dtype, PlatformRuntimeProfile calibratedRuntime) {
        String dtypeId = CalibrationCommand.dtypeId(dtype);
        WorkloadSpec workload = StandardWorkloads.abcSequenceMatmulBlasBenchmark(
                "main_abc_sequence_matmul_" + dtypeId
        );
        ExecutionProfile baseline = Synaptik.tuning()
                .profile()
                .name("main-baseline-no-opt-" + dtypeId)
                .candidate("baseline-no-opt")
                .dtype(dtype)
                .mode().training()
                .optimizer().noOptimization()
                .runtime().noOptNoVecNoPar()
                .build();
        ExecutionProfile calibrated = Synaptik.tuning()
                .profile()
                .name("main-calibrated-runtime-" + dtypeId)
                .candidate("calibrated-runtime")
                .dtype(dtype)
                .mode().training()
                .optimizer().trainingDefaults()
                .runtime().fromPlatformProfile(calibratedRuntime)
                .build();

        return Synaptik.tuning()
                .benchmark()
                .workload(workload)
                .quick()
                .report()
                        .hotStepLimit(5)
                        .includeTrace()
                        .includeFailedCandidates()
                        .done()
                .compare()
                .baseline("baseline-no-opt", baseline)
                .candidate("calibrated-runtime", calibrated)
                .run();
    }
}
