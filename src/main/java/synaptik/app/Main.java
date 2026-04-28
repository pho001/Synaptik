package synaptik.app;

import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.profile.ExecutionProfile;
import config.profile.PlatformRuntimeProfile;
import config.runtime.RuntimeConfig;
import tensor.DataType;
import tuning.autotune.TuningDefaults;
import tuning.benchmark.BenchmarkEntry;
import tuning.benchmark.BenchmarkSession;
import tuning.benchmark.report.BenchmarkReport;
import tuning.benchmark.report.TextBenchmarkReportRenderer;
import tuning.calibration.PlatformCalibrationResult;
import tuning.calibration.family.CalibrationFamilyId;
import tuning.calibration.run.CalibrationCommand;
import tuning.calibration.run.CalibrationRunner;
import tuning.calibration.run.CalibrationScope;
import tuning.measure.MeasurementPolicy;
import tuning.preset.TuningPreset;
import tuning.workload.StandardWorkloads;
import tuning.workload.WorkloadSpec;

import java.nio.file.Path;
import java.util.List;

/**
 * Programmatic tuning entry point.
 *
 * <p>This class is intentionally not a CLI parser. It shows the regular Java API shape for running
 * calibration and then benchmarking calibrated runtime settings. The command-line tool remains
 * {@link TuningCli}; this entry point is useful when the application should be configured from Java
 * code instead of command tokens.</p>
 */
public final class Main {
    private static final Path PROFILE_ROOT = Path.of("profiles");
    private static final DataType DTYPE = DataType.FLOAT64;
    private static final ExecutionMode MODE = ExecutionMode.FORWARD_BACKWARD;
    private static final TuningPreset PRESET = TuningPreset.QUICK;

    private Main() {
    }

    /**
     * Runs a small programmatic calibration and benchmark flow.
     *
     * @param args ignored; this class is not a command-line parser
     */
    public static void main(String[] args) {
        List<PlatformCalibrationResult> calibrationResults = calibration()
                .dtypes().single(DTYPE)
                .families().all()
                .preset(PRESET)
                .mode().training()
                .measurement().iterations(1, 3, 1)
                .progress().lines()
                .color().auto()
                .outputRoot(PROFILE_ROOT)
                .run();

        PlatformRuntimeProfile calibratedRuntime = calibrationResults.getLast().finalRuntimeProfile();
        BenchmarkReport benchmark = benchmarkCalibratedRuntime(DTYPE, calibratedRuntime);
        System.out.println(TextBenchmarkReportRenderer.render(benchmark));
    }

    private static CalibrationDsl calibration() {
        return new CalibrationDsl();
    }

    private static BenchmarkReport benchmarkCalibratedRuntime(DataType dtype, PlatformRuntimeProfile calibratedRuntime) {
        WorkloadSpec workload = StandardWorkloads.abcSequenceMatmulBlasBenchmark(
                "main_abc_sequence_matmul_" + CalibrationCommand.dtypeId(dtype)
        );
        ExecutionProfile baseline = new ExecutionProfile(
                "main-baseline-no-opt-" + CalibrationCommand.dtypeId(dtype),
                "baseline-no-opt",
                dtype,
                MODE,
                OptimizerConfig.noOptimization(),
                RuntimeConfig.noOptNoVecNoPar()
        );
        ExecutionProfile calibrated = new ExecutionProfile(
                "main-calibrated-runtime-" + CalibrationCommand.dtypeId(dtype),
                "calibrated-runtime",
                dtype,
                MODE,
                OptimizerConfig.trainingDefaults(),
                calibratedRuntime.toRuntimeConfig()
        );

        return BenchmarkSession.create(TuningDefaults.quickBenchmark(
                workload,
                List.of(
                        BenchmarkEntry.baseline("baseline-no-opt", baseline),
                        BenchmarkEntry.candidate("calibrated-runtime", calibrated)
                )
        )).run();
    }

    private static final class CalibrationDsl {
        private List<DataType> dataTypes = List.of(DTYPE);
        private CalibrationFamilyId family;
        private CalibrationScope scope = CalibrationScope.ALL_FAMILIES;
        private TuningPreset preset = PRESET;
        private ExecutionMode mode = MODE;
        private MeasurementPolicy measurement;
        private String colorMode = "auto";
        private String progressMode = "live";
        private Path outputRoot = PROFILE_ROOT;
        private boolean includeAccelerators;

        private DTypeStep dtypes() {
            return new DTypeStep(this);
        }

        private FamilyStep families() {
            return new FamilyStep(this);
        }

        private CalibrationDsl preset(TuningPreset value) {
            preset = value == null ? TuningPreset.BALANCED : value;
            return this;
        }

        private CalibrationDsl quick() {
            return preset(TuningPreset.QUICK);
        }

        private CalibrationDsl balanced() {
            return preset(TuningPreset.BALANCED);
        }

        private CalibrationDsl thorough() {
            return preset(TuningPreset.THOROUGH);
        }

        private ModeStep mode() {
            return new ModeStep(this);
        }

        private MeasurementStep measurement() {
            return new MeasurementStep(this);
        }

        private ProgressStep progress() {
            return new ProgressStep(this);
        }

        private ColorStep color() {
            return new ColorStep(this);
        }

        private CalibrationDsl outputRoot(Path value) {
            outputRoot = value == null ? PROFILE_ROOT : value;
            return this;
        }

        private CalibrationDsl includeAccelerators() {
            includeAccelerators = true;
            return this;
        }

        private CalibrationCommand toCommand() {
            return new CalibrationCommand(
                    dataTypes,
                    family,
                    scope,
                    preset,
                    mode,
                    measurement,
                    colorMode,
                    progressMode,
                    outputRoot,
                    includeAccelerators
            );
        }

        private List<PlatformCalibrationResult> run() {
            return CalibrationRunner.create().run(toCommand());
        }
    }

    private record DTypeStep(CalibrationDsl parent) {
        private CalibrationDsl single(DataType dataType) {
            parent.dataTypes = List.of(dataType);
            return parent;
        }

        private CalibrationDsl all() {
            parent.dataTypes = List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16);
            return parent;
        }
    }

    private record FamilyStep(CalibrationDsl parent) {
        private CalibrationDsl single(CalibrationFamilyId family) {
            parent.family = family;
            parent.scope = CalibrationScope.SINGLE_FAMILY;
            return parent;
        }

        private CalibrationDsl all() {
            parent.family = null;
            parent.scope = CalibrationScope.ALL_FAMILIES;
            return parent;
        }
    }

    private record ModeStep(CalibrationDsl parent) {
        private CalibrationDsl forward() {
            parent.mode = ExecutionMode.FORWARD;
            return parent;
        }

        private CalibrationDsl training() {
            parent.mode = ExecutionMode.FORWARD_BACKWARD;
            return parent;
        }

        private CalibrationDsl forwardBackward() {
            return training();
        }
    }

    private record MeasurementStep(CalibrationDsl parent) {
        private CalibrationDsl iterations(int warmupIters, int measureIters, int repeats) {
            MeasurementPolicy base = parent.preset.benchmarkMeasurement();
            parent.measurement = new MeasurementPolicy(
                    warmupIters,
                    measureIters,
                    repeats,
                    base.measureCompile(),
                    base.measurePrepare(),
                    base.measureColdRun(),
                    base.measureSteadyState(),
                    base.captureStepTrace()
            );
            return parent;
        }

        private CalibrationDsl policy(MeasurementPolicy policy) {
            parent.measurement = policy;
            return parent;
        }
    }

    private record ProgressStep(CalibrationDsl parent) {
        private CalibrationDsl live() {
            parent.progressMode = "live";
            return parent;
        }

        private CalibrationDsl lines() {
            parent.progressMode = "lines";
            return parent;
        }

        private CalibrationDsl quiet() {
            parent.progressMode = "quiet";
            return parent;
        }
    }

    private record ColorStep(CalibrationDsl parent) {
        private CalibrationDsl auto() {
            parent.colorMode = "auto";
            return parent;
        }

        private CalibrationDsl always() {
            parent.colorMode = "always";
            return parent;
        }

        private CalibrationDsl never() {
            parent.colorMode = "never";
            return parent;
        }
    }
}
