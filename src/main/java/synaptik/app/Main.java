package synaptik.app;

import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.ExecutionProfileAssembler;
import config.profile.GraphExecutionPolicy;
import config.profile.PlatformRuntimeProfile;
import config.profile.PlatformRuntimeProfileIO;
import config.profile.WorkloadProfile;
import tensor.DataType;
import tuning.candidate.ProfileGridCandidateSpace;
import tuning.candidate.ProfileMutators;
import tuning.measure.MeasurementPolicy;
import tuning.report.TextBenchmarkReportRenderer;
import tuning.report.TextPlatformCalibrationResultRenderer;
import tuning.report.TextTuningResultRenderer;
import tuning.search.ExhaustiveSearchStrategy;
import tuning.session.AutotuneSession;
import tuning.session.BenchmarkEntry;
import tuning.session.BenchmarkSession;
import tuning.session.LoggingPlatformCalibrationProgressListener;
import tuning.session.PlatformCalibrationDefaults;
import tuning.session.PlatformCalibrationFamily;
import tuning.session.PlatformCalibrationRequest;
import tuning.session.PlatformCalibrationSession;
import tuning.session.PlatformCalibrationStep;
import tuning.session.TuningDefaults;
import tuning.session.TuningPreset;
import tuning.store.JsonFileBestProfileStore;
import tuning.store.JsonFileTuningHistoryStore;
import tuning.store.HardwareFingerprint;
import tuning.store.PersistencePolicy;
import tuning.store.PlatformCalibrationLayout;
import tuning.store.PlatformCalibrationPaths;
import tuning.store.PlatformCalibrationSaveHelper;
import tuning.validate.DefaultValidationEngine;
import tuning.workload.StandardWorkloads;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;

public final class Main {
    private Main() {
    }

    public static void main(String[] args) {
        if (args.length == 0) {
            runFull(DTypeTarget.F64);
            return;
        }

        if (args.length < 2) {
            printUsage();
            return;
        }

        Phase phase = Phase.parse(args[0]);
        DTypeTarget dtype = DTypeTarget.parse(args[1]);
        if (phase == null || dtype == null) {
            printUsage();
            return;
        }

        switch (phase) {
            case FULL -> {
                requireExactArgCount(args, 2);
                runFull(dtype);
            }
            case CALIBRATE -> runCalibration(dtype, parseCalibrationOptions(args));
            case AUTOTUNE -> {
                requireExactArgCount(args, 2);
                runAutotune(dtype);
            }
            case BENCHMARK_WINNER -> {
                requireExactArgCount(args, 2);
                runWinnerBenchmark(dtype);
            }
            case BENCHMARK_STAGE_SPACE -> {
                requireExactArgCount(args, 2);
                runStageSpaceBenchmark(dtype);
            }
        }
    }

    private static void runFull(DTypeTarget dtype) {
        System.out.println(header(dtype, "full flow"));
        System.out.println("note=convenience flow for local iteration; for the cleanest performance numbers prefer running phases separately");
        runCalibration(dtype, CalibrationCliOptions.defaults());
        runAutotune(dtype);
        runWinnerBenchmark(dtype);
    }

    private static void runCalibration(DTypeTarget dtype, CalibrationCliOptions options) {
        CalibrationCliOptions effective = options == null ? CalibrationCliOptions.defaults() : options;
        String title = effective.family() == null
                ? "training calibration"
                : "training calibration [" + effective.family().cliName() + "]";
        System.out.println(header(dtype, title));
        if (effective.family() != null) {
            System.out.println("family=" + effective.family().cliName());
        }
        printCalibrationMeasurement(effective.measurement());
        ExecutionProfile seed = trainingSeedProfile(dtype);
        PlatformCalibrationLayout layout = calibrationLayout(dtype, seed);
        PlatformCalibrationRequest baseRequest = calibrationRequest(dtype, seed, layout, effective);
        PlatformCalibrationRequest request = new PlatformCalibrationRequest(
                baseRequest.platformId(),
                baseRequest.profileName(),
                baseRequest.dataType(),
                baseRequest.executionMode(),
                baseRequest.graphPolicy(),
                baseRequest.seedRuntimeProfile(),
                baseRequest.steps(),
                baseRequest.outputProfilePath(),
                effective.measurement(),
                LoggingPlatformCalibrationProgressListener.defaults()
        );

        var result = PlatformCalibrationSession.create(request).run();
        PlatformCalibrationSaveHelper.saveAll(
                result,
                layout.profilePath(),
                layout.jsonReportPath(),
                layout.textReportPath()
        );

        System.out.println(TextPlatformCalibrationResultRenderer.render(result));
        System.out.println("profilePath=" + result.outputProfilePath());
    }

    private static PlatformCalibrationRequest calibrationRequest(
            DTypeTarget dtype,
            ExecutionProfile seed,
            PlatformCalibrationLayout layout,
            CalibrationCliOptions options
    ) {
        if (options.family() == null) {
            return PlatformCalibrationDefaults.balancedTrainingFull(
                    layout.platformId(),
                    seed,
                    layout.profilePath()
            );
        }
        return PlatformCalibrationRequest.fromSeedExecutionProfile(
                layout.platformId(),
                seed,
                options.family().createSteps(
                        "calib-" + options.family().cliName() + "-train",
                        TuningPreset.BALANCED,
                        dtype.dataType
                ),
                layout.profilePath()
        );
    }

    private static void printCalibrationMeasurement(MeasurementPolicy measurement) {
        if (measurement == null) {
            return;
        }
        System.out.println("measurementOverride="
                + "warmup=" + measurement.warmupIters()
                + ", measure=" + measurement.measureIters()
                + ", repeats=" + measurement.repeats());
    }

    private static void runAutotune(DTypeTarget dtype) {
        System.out.println(header(dtype, "ABC autotune"));
        ExecutionProfile calibratedSeed = calibratedAbcSeed(dtype);
        var workload = abcWorkload(dtype);
        var candidateSpace = stageCandidateSpace(calibratedSeed);
        PersistencePolicy persistence = tuningPersistence(dtype);

        var request = TuningDefaults.balancedAutotune(
                workload,
                calibratedSeed,
                candidateSpace,
                persistence
        );
        var result = AutotuneSession.create(
                request,
                new ExhaustiveSearchStrategy(),
                new tuning.measure.DefaultMeasurementEngine(),
                new DefaultValidationEngine(),
                new JsonFileBestProfileStore(),
                new JsonFileTuningHistoryStore()
        ).run();

        System.out.println(TextTuningResultRenderer.render(result));
        System.out.println("autotuneBestProfilePath=" + persistence.bestProfilePath());
        System.out.println("autotuneHistoryPath=" + persistence.historyPath());
    }

    private static void runWinnerBenchmark(DTypeTarget dtype) {
        System.out.println(header(dtype, "ABC benchmark: baseline vs winner"));
        ExecutionProfile baseline = baselineProfile(dtype);
        ExecutionProfile winner = loadWinnerProfile(dtype);

        var request = TuningDefaults.benchmark(
                TuningPreset.BALANCED,
                abcWorkload(dtype),
                List.of(
                        BenchmarkEntry.baseline("baseline-no-opt", baseline),
                        BenchmarkEntry.candidate("best-profile", winner)
                )
        );
        var report = BenchmarkSession.create(request).run();
        System.out.println(TextBenchmarkReportRenderer.render(report));
    }

    private static void runStageSpaceBenchmark(DTypeTarget dtype) {
        System.out.println(header(dtype, "ABC benchmark: stage space exploration"));
        ExecutionProfile baseline = baselineProfile(dtype);
        ExecutionProfile calibratedSeed = calibratedAbcSeed(dtype);
        var workload = abcWorkload(dtype);
        var candidateSpace = stageCandidateSpace(calibratedSeed);

        List<BenchmarkEntry> entries = new ArrayList<>();
        entries.add(BenchmarkEntry.baseline("baseline-no-opt", baseline));
        candidateSpace.generate(workload).forEach(candidate ->
                entries.add(BenchmarkEntry.candidate(candidate.name(), candidate.profile()))
        );

        var request = TuningDefaults.benchmark(
                TuningPreset.BALANCED,
                workload,
                entries
        );
        var report = BenchmarkSession.create(request).run();
        System.out.println(TextBenchmarkReportRenderer.render(report));
    }

    private static ExecutionProfile trainingSeedProfile(DTypeTarget dtype) {
        return new ExecutionProfile(
                "platform-seed-" + dtype.id + "-training",
                "platform-seed-" + dtype.id + "-training",
                dtype.dataType,
                ExecutionMode.FORWARD_BACKWARD,
                config.optimizer.OptimizerConfig.trainingDefaults(),
                config.runtime.RuntimeConfig.trainingDefaults(),
                WorkloadProfile.none()
        );
    }

    private static PlatformCalibrationLayout calibrationLayout(DTypeTarget dtype, ExecutionProfile seed) {
        HardwareFingerprint hardware = HardwareFingerprint.capture();
        String platformId = PlatformCalibrationPaths.platformId(hardware);
        String variant = PlatformCalibrationPaths.variantId(seed);
        Path root = Path.of("profiles", "platform", platformId);
        return new PlatformCalibrationLayout(
                platformId,
                hardware,
                root.resolve("calibration").resolve(variant + ".json"),
                root.resolve("reports").resolve("calibration-" + variant + ".json"),
                root.resolve("reports").resolve("calibration-" + variant + ".txt")
        );
    }

    private static PlatformCalibrationLayout legacyCalibrationLayout(DTypeTarget dtype, ExecutionProfile seed) {
        return PlatformCalibrationPaths.defaultLayout(
                Path.of("build", "platform-calibration", dtype.id),
                seed
        );
    }

    private static PlatformRuntimeProfile loadCalibrationProfile(DTypeTarget dtype) {
        ExecutionProfile seed = trainingSeedProfile(dtype);
        PlatformCalibrationLayout layout = calibrationLayout(dtype, seed);
        Path path = ensurePreferredArtifact(
                layout.profilePath(),
                legacyCalibrationLayout(dtype, seed).profilePath(),
                "calibration profile"
        );
        if (!Files.exists(path)) {
            throw new IllegalStateException("Missing calibration profile: " + path + ". Run `calibrate " + dtype.id + "` first.");
        }
        PlatformRuntimeProfile fallback = PlatformRuntimeProfile.fromExecutionProfile(
                layout.platformId(),
                layout.hardware().key(),
                "fallback",
                seed
        );
        return PlatformRuntimeProfileIO.loadOrDefault(path, fallback);
    }

    private static ExecutionProfile calibratedAbcSeed(DTypeTarget dtype) {
        return ExecutionProfileAssembler.assemble(
                "abc-" + dtype.id + "-calibrated",
                "abc-" + dtype.id + "-calibrated",
                dtype.dataType,
                ExecutionMode.FORWARD_BACKWARD,
                loadCalibrationProfile(dtype),
                GraphExecutionPolicy.trainingDefaults(),
                WorkloadProfile.none()
        );
    }

    private static ProfileGridCandidateSpace stageCandidateSpace(ExecutionProfile seed) {
        return new ProfileGridCandidateSpace(
                seed,
                List.of(ProfileMutators.constrainedStageOrderSpace())
        );
    }

    private static ExecutionProfile baselineProfile(DTypeTarget dtype) {
        return new ExecutionProfile(
                "abc-baseline-no-opt-" + dtype.id,
                "abc-baseline-no-opt-" + dtype.id,
                dtype.dataType,
                ExecutionMode.FORWARD_BACKWARD,
                config.optimizer.OptimizerConfig.noOptimization(),
                config.runtime.RuntimeConfig.noOptNoVecNoPar(),
                WorkloadProfile.none()
        );
    }

    private static ExecutionProfile loadWinnerProfile(DTypeTarget dtype) {
        PersistencePolicy preferred = tuningPersistence(dtype);
        PersistencePolicy legacy = legacyTuningPersistence(dtype);
        ensurePreferredArtifact(
                preferred.bestProfilePath(),
                legacy.bestProfilePath(),
                "best profile"
        );
        ensurePreferredArtifact(
                preferred.historyPath(),
                legacy.historyPath(),
                "tuning history"
        );
        Path path = resolveExisting(
                tuningPersistence(dtype).bestProfilePath(),
                legacyTuningPersistence(dtype).bestProfilePath()
        );
        return new JsonFileBestProfileStore()
                .load(path)
                .orElseThrow(() -> new IllegalStateException("Missing best profile: " + path + ". Run `autotune " + dtype.id + "` first."))
                .profile();
    }

    private static PersistencePolicy tuningPersistence(DTypeTarget dtype) {
        String platformId = PlatformCalibrationPaths.platformId(HardwareFingerprint.capture());
        Path root = Path.of("profiles", "platform", platformId, "tuning", "abc");
        return new PersistencePolicy(
                true,
                true,
                root.resolve(dtype.id + "-best-profile.json"),
                root.resolve(dtype.id + "-history.jsonl")
        );
    }

    private static PersistencePolicy legacyTuningPersistence(DTypeTarget dtype) {
        return new PersistencePolicy(
                true,
                true,
                Path.of("build", "tuning", "best-profiles", "abc-" + dtype.id + "-best-profile.json"),
                Path.of("build", "tuning", "history", "abc-" + dtype.id + "-history.jsonl")
        );
    }

    private static Path resolveExisting(Path preferred, Path fallback) {
        if (preferred != null && Files.exists(preferred)) {
            return preferred;
        }
        return fallback;
    }

    private static Path ensurePreferredArtifact(Path preferred, Path legacy, String label) {
        if (preferred == null || Files.exists(preferred)) {
            return preferred;
        }
        if (legacy == null || !Files.exists(legacy)) {
            return preferred;
        }
        try {
            Files.createDirectories(preferred.getParent());
            Files.copy(legacy, preferred, StandardCopyOption.REPLACE_EXISTING);
            System.out.println("migratedLegacyArtifact=" + label + " -> " + preferred);
            return preferred;
        } catch (Exception e) {
            throw new IllegalStateException(
                    "Failed to migrate legacy " + label + " from " + legacy + " to " + preferred,
                    e
            );
        }
    }

    private static tuning.workload.WorkloadSpec abcWorkload(DTypeTarget dtype) {
        return StandardWorkloads.abcSequenceMatmulBlasBenchmark("abc_sequence_matmul_" + dtype.id);
    }

    private static String header(DTypeTarget dtype, String title) {
        return "\n==============================\n"
                + "Synaptik " + dtype.id.toUpperCase() + " " + title + "\n"
                + "==============================";
    }

    private static void printUsage() {
        System.out.println("""
                Usage:
                  ./gradlew run --args="full <f64|f32|bf16>"
                  ./gradlew run --args="calibrate <f64|f32|bf16> [family] [warmup measure repeats]"
                  ./gradlew run --args="autotune <f64|f32|bf16>"
                  ./gradlew run --args="benchmark-winner <f64|f32|bf16>"
                  ./gradlew run --args="benchmark-stage-space <f64|f32|bf16>"

                Notes:
                  - no args defaults to `full f64`
                  - run phases separately to avoid cross-phase JVM warmup bias
                  - `calibrate` optionally accepts a single family, e.g. `calibrate f64 conv2d`
                  - `calibrate` optionally accepts explicit measurement override, e.g. `calibrate f64 conv2d 30 100 2`
                  - supported calibration families: %s
                  - `autotune` expects an existing calibration profile
                  - `benchmark-winner` expects an existing best-profile artifact
                """.formatted(CalibrationFamilyTarget.supportedCliNames()));
    }

    private static void requireExactArgCount(String[] args, int expected) {
        if (args.length != expected) {
            printUsage();
            throw new IllegalArgumentException("Unexpected argument count.");
        }
    }

    static CalibrationCliOptions parseCalibrationOptions(String[] args) {
        CalibrationFamilyTarget family = null;
        int measurementStart = 2;
        if (args.length > 2 && !isInteger(args[2])) {
            family = CalibrationFamilyTarget.parse(args[2]);
            if (family == null) {
                printUsage();
                throw new IllegalArgumentException(
                        "Unknown calibration family `" + args[2] + "`. Supported families: " + CalibrationFamilyTarget.supportedCliNames()
                );
            }
            measurementStart = 3;
        }
        int remaining = args.length - measurementStart;
        if (remaining == 0) {
            return new CalibrationCliOptions(family, null);
        }
        if (remaining != 3) {
            printUsage();
            throw new IllegalArgumentException(
                    "Calibration measurement override expects exactly 3 integers: warmup measure repeats."
            );
        }
        return new CalibrationCliOptions(family, parseCalibrationMeasurement(args, measurementStart));
    }

    private static MeasurementPolicy parseCalibrationMeasurement(String[] args, int startIndex) {
        try {
            int warmup = Integer.parseInt(args[startIndex]);
            int measure = Integer.parseInt(args[startIndex + 1]);
            int repeats = Integer.parseInt(args[startIndex + 2]);
            MeasurementPolicy base = TuningPreset.BALANCED.benchmarkMeasurement();
            return new MeasurementPolicy(
                    warmup,
                    measure,
                    repeats,
                    base.measureCompile(),
                    base.measurePrepare(),
                    base.measureColdRun(),
                    base.measureSteadyState(),
                    base.captureStepTrace()
            );
        } catch (NumberFormatException ex) {
            printUsage();
            throw new IllegalArgumentException("Calibration measurement override must be integers.", ex);
        }
    }

    private static boolean isInteger(String value) {
        if (value == null || value.isBlank()) {
            return false;
        }
        int start = value.charAt(0) == '-' ? 1 : 0;
        if (start == value.length()) {
            return false;
        }
        for (int i = start; i < value.length(); i++) {
            if (!Character.isDigit(value.charAt(i))) {
                return false;
            }
        }
        return true;
    }

    private enum Phase {
        FULL("full"),
        CALIBRATE("calibrate"),
        AUTOTUNE("autotune"),
        BENCHMARK_WINNER("benchmark-winner"),
        BENCHMARK_STAGE_SPACE("benchmark-stage-space");

        private final String cliName;

        Phase(String cliName) {
            this.cliName = cliName;
        }

        private static Phase parse(String value) {
            if (value == null) {
                return null;
            }
            for (Phase phase : values()) {
                if (phase.cliName.equalsIgnoreCase(value)) {
                    return phase;
                }
            }
            return null;
        }
    }

    private enum DTypeTarget {
        F64("f64", DataType.FLOAT64),
        F32("f32", DataType.FLOAT32),
        BF16("bf16", DataType.BFLOAT16);

        private final String id;
        private final DataType dataType;

        DTypeTarget(String id, DataType dataType) {
            this.id = id;
            this.dataType = dataType;
        }

        private static DTypeTarget parse(String value) {
            if (value == null) {
                return null;
            }
            for (DTypeTarget target : values()) {
                if (target.id.equalsIgnoreCase(value)) {
                    return target;
                }
            }
            return null;
        }
    }

    record CalibrationCliOptions(
            CalibrationFamilyTarget family,
            MeasurementPolicy measurement
    ) {
        static CalibrationCliOptions defaults() {
            return new CalibrationCliOptions(null, null);
        }
    }

    enum CalibrationFamilyTarget {
        MATMUL("matmul", null) {
            @Override
            List<PlatformCalibrationStep> createSteps(String name, TuningPreset preset, DataType dataType) {
                return List.of(
                        PlatformCalibrationDefaults.matmulJavaStep(name + "-java", preset),
                        PlatformCalibrationDefaults.matmulBlasDispatchStep(name + "-blas", preset),
                        PlatformCalibrationDefaults.matmulBlasWideDispatchStep(name + "-blas-wide", preset)
                );
            }
        },
        MATMUL_JAVA("matmul-java", PlatformCalibrationFamily.MATMUL_JAVA) {
            @Override
            List<PlatformCalibrationStep> createSteps(String name, TuningPreset preset, DataType dataType) {
                return List.of(PlatformCalibrationDefaults.matmulJavaStep(name, preset));
            }
        },
        MATMUL_BLAS("matmul-blas", null) {
            @Override
            List<PlatformCalibrationStep> createSteps(String name, TuningPreset preset, DataType dataType) {
                return List.of(
                        PlatformCalibrationDefaults.matmulBlasDispatchStep(name + "-regular", preset),
                        PlatformCalibrationDefaults.matmulBlasWideDispatchStep(name + "-wide", preset)
                );
            }
        },
        MATMUL_BLAS_REGULAR("matmul-blas-regular", PlatformCalibrationFamily.MATMUL_BLAS_DISPATCH) {
            @Override
            List<PlatformCalibrationStep> createSteps(String name, TuningPreset preset, DataType dataType) {
                return List.of(PlatformCalibrationDefaults.matmulBlasDispatchStep(name, preset));
            }
        },
        MATMUL_BLAS_WIDE("matmul-blas-wide", PlatformCalibrationFamily.MATMUL_BLAS_DISPATCH_WIDE) {
            @Override
            List<PlatformCalibrationStep> createSteps(String name, TuningPreset preset, DataType dataType) {
                return List.of(PlatformCalibrationDefaults.matmulBlasWideDispatchStep(name, preset));
            }
        },
        ATTENTION_MATMUL("attention-matmul", PlatformCalibrationFamily.ATTENTION_MATMUL) {
            @Override
            List<PlatformCalibrationStep> createSteps(String name, TuningPreset preset, DataType dataType) {
                return List.of(PlatformCalibrationDefaults.attentionMatmulStep(name, preset));
            }
        },
        CONV2D("conv2d", null) {
            @Override
            List<PlatformCalibrationStep> createSteps(String name, TuningPreset preset, DataType dataType) {
                return List.of(PlatformCalibrationDefaults.conv2dGemmDispatchStep(name, preset, dataType));
            }
        },
        FUSED_THRESHOLDS("fused-thresholds", PlatformCalibrationFamily.FUSED_THRESHOLDS) {
            @Override
            List<PlatformCalibrationStep> createSteps(String name, TuningPreset preset, DataType dataType) {
                return List.of(PlatformCalibrationDefaults.fusedDispatchStep(name, preset));
            }
        },
        FUSED_CHEAP_CONTIGUOUS("fused-cheap-contiguous", PlatformCalibrationFamily.FUSED_CHEAP_CONTIGUOUS) {
            @Override
            List<PlatformCalibrationStep> createSteps(String name, TuningPreset preset, DataType dataType) {
                return List.of(PlatformCalibrationDefaults.fusedCheapContiguousStep(name, preset, dataType));
            }
        },
        FUSED_CHEAP_STRIDED("fused-cheap-strided", PlatformCalibrationFamily.FUSED_CHEAP_STRIDED) {
            @Override
            List<PlatformCalibrationStep> createSteps(String name, TuningPreset preset, DataType dataType) {
                return List.of(PlatformCalibrationDefaults.fusedCheapStridedStep(name, preset, dataType));
            }
        },
        FUSED_NON_CHEAP_CONTIGUOUS("fused-noncheap-contiguous", PlatformCalibrationFamily.FUSED_NON_CHEAP_CONTIGUOUS) {
            @Override
            List<PlatformCalibrationStep> createSteps(String name, TuningPreset preset, DataType dataType) {
                return List.of(PlatformCalibrationDefaults.fusedNonCheapContiguousStep(name, preset, dataType));
            }
        },
        FUSED_NON_CHEAP_STRIDED("fused-noncheap-strided", PlatformCalibrationFamily.FUSED_NON_CHEAP_STRIDED) {
            @Override
            List<PlatformCalibrationStep> createSteps(String name, TuningPreset preset, DataType dataType) {
                return List.of(PlatformCalibrationDefaults.fusedNonCheapStridedStep(name, preset, dataType));
            }
        },
        ELEMENTWISE("elementwise", PlatformCalibrationFamily.ELEMENTWISE_DISPATCH) {
            @Override
            List<PlatformCalibrationStep> createSteps(String name, TuningPreset preset, DataType dataType) {
                return List.of(PlatformCalibrationDefaults.elementwiseDispatchStep(name, preset));
            }
        },
        REDUCTION("reduction", PlatformCalibrationFamily.REDUCTION) {
            @Override
            List<PlatformCalibrationStep> createSteps(String name, TuningPreset preset, DataType dataType) {
                return List.of(PlatformCalibrationDefaults.reductionStep(name, preset));
            }
        },
        ATTENTION_THRESHOLDS("attention-thresholds", PlatformCalibrationFamily.ATTENTION_THRESHOLDS) {
            @Override
            List<PlatformCalibrationStep> createSteps(String name, TuningPreset preset, DataType dataType) {
                return List.of(PlatformCalibrationDefaults.attentionStep(name, preset));
            }
        },
        SCHEDULER("scheduler", PlatformCalibrationFamily.SCHEDULER) {
            @Override
            List<PlatformCalibrationStep> createSteps(String name, TuningPreset preset, DataType dataType) {
                return List.of(PlatformCalibrationDefaults.schedulerStep(name, preset));
            }
        },
        MATERIALIZATION("materialization", PlatformCalibrationFamily.MATERIALIZATION) {
            @Override
            List<PlatformCalibrationStep> createSteps(String name, TuningPreset preset, DataType dataType) {
                return List.of(
                        PlatformCalibrationDefaults.materializationStep(name, preset),
                        PlatformCalibrationDefaults.whereMaterializationStep(name + "-where", preset)
                );
            }
        },
        NUMERICS("numerics", PlatformCalibrationFamily.NUMERICS) {
            @Override
            List<PlatformCalibrationStep> createSteps(String name, TuningPreset preset, DataType dataType) {
                return List.of(PlatformCalibrationDefaults.numericsStep(name, preset));
            }
        },
        ACCELERATOR_METAL_SELECTION("accelerator-metal", PlatformCalibrationFamily.ACCELERATOR_METAL_SELECTION) {
            @Override
            List<PlatformCalibrationStep> createSteps(String name, TuningPreset preset, DataType dataType) {
                return List.of(PlatformCalibrationDefaults.acceleratorMetalSelectionStep(name, preset));
            }
        };

        private final String cliName;
        private final PlatformCalibrationFamily family;

        CalibrationFamilyTarget(String cliName, PlatformCalibrationFamily family) {
            this.cliName = cliName;
            this.family = family;
        }

        String cliName() {
            return cliName;
        }

        PlatformCalibrationFamily family() {
            return family;
        }

        abstract List<PlatformCalibrationStep> createSteps(String name, TuningPreset preset, DataType dataType);

        static CalibrationFamilyTarget parse(String value) {
            if (value == null) {
                return null;
            }
            for (CalibrationFamilyTarget target : values()) {
                if (target.cliName.equalsIgnoreCase(value)) {
                    return target;
                }
            }
            return null;
        }

        static String supportedCliNames() {
            return String.join(", ", java.util.Arrays.stream(values()).map(CalibrationFamilyTarget::cliName).toList());
        }
    }
}
