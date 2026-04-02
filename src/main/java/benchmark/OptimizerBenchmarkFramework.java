package benchmark;

import backend.ComputeEngine;
import backend.kernels.cpu.CpuSchedulerAdvisor;
import benchmark.autotune.CandidateEvalCache;
import benchmark.autotune.CandidateGraphIndex;
import benchmark.autotune.CandidatePerf;
import benchmark.autotune.CoarseKnobSignature;
import benchmark.autotune.CorrectnessVerdict;
import benchmark.autotune.FamilyScoutStats;
import benchmark.autotune.AutoTuneBestResults;
import benchmark.autotune.AutoTuneFinalizationConfig;
import benchmark.autotune.AutoTuneFinalizationResult;
import benchmark.autotune.AutoTuneFinalizer;
import benchmark.autotune.AutoTunePersistencePort;
import benchmark.autotune.AutoTuneProfilePersistence;
import benchmark.autotune.AutoTuneProfilePersistenceResult;
import benchmark.autotune.AutoTuneProgressTracker;
import benchmark.autotune.AutoTuneResult;
import benchmark.autotune.AutoTuneSessionConfig;
import benchmark.autotune.AutoTuneSessionResult;
import benchmark.autotune.AutoTuneSessionRunner;
import benchmark.autotune.FinalistPreparation;
import benchmark.autotune.FinalistPreparationResult;
import benchmark.autotune.GraphScoutConfig;
import benchmark.autotune.GraphScoutReducer;
import benchmark.autotune.NumericsPostcheckConfig;
import benchmark.autotune.NumericsPostcheckResult;
import benchmark.autotune.NumericsPostcheckRunner;
import benchmark.autotune.Phase1Counters;
import graph.CompiledGraph;
import benchmark.autotune.Phase1CandidateEvaluator;
import benchmark.autotune.Phase1CandidateResult;
import benchmark.autotune.Phase1Step;
import benchmark.autotune.RunningEstimate;
import benchmark.autotune.Phase1FinalistSelector;
import benchmark.autotune.RefineConfig;
import benchmark.autotune.RefinedCandidate;
import benchmark.autotune.RefineProgressUpdate;
import benchmark.autotune.RefineRunner;
import benchmark.autotune.UnsafeCandidateHistory;
import benchmark.autotune.AutotuneSearchSupport;
import benchmark.autotune.BeamSearchConfig;
import benchmark.measure.MeasurementObjective;
import benchmark.measure.MeasurementScoring;
import benchmark.measure.NanoClock;
import benchmark.measure.CandidateMeasurementCachePort;
import benchmark.measure.CandidateMeasurementHarness;
import benchmark.measure.CandidateMeasurementResult;
import benchmark.measure.MeasuredBenchmarkScenario;
import benchmark.measure.MeasuredBroadcastScenario;
import benchmark.scenario.BenchmarkScenarioFactory;
import benchmark.scenario.LinearGraphShape;
import benchmark.scenario.PreparedBenchmarkScenario;
import benchmark.scenario.PreparedBroadcastScenario;
import graph.optimizer.GraphOptimizer;
import numerics.NumericsHarness;
import numerics.NumericsMetrics;
import numerics.NumericsPolicy;
import numerics.NumericsReport;
import tensor.DataType;
import tensor.Tensor;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.LocalDateTime;
import java.time.OffsetDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;

public final class OptimizerBenchmarkFramework {
    private OptimizerBenchmarkFramework() {}

    private static final int SIZE = 1_000_000;
    private static final int WARMUP_ITERS = 200;
    private static final int MEASURE_ITERS = 1000;
    private static final int STAGE_WARMUP_ITERS = 50;
    private static final int STAGE_MEASURE_ITERS = 300;
    private static final int AUTOTUNE_SIZE = 200_000;
    private static final int AUTOTUNE_SAFETY_SIZE = Math.max(
            4096,
            Integer.getInteger("benchmark.autotuneSafetySize", 128)
    );
    private static final int AUTOTUNE_WARMUP_ITERS = 12;
    private static final int AUTOTUNE_MEASURE_ITERS = 40;
    private static final int AUTOTUNE_MAX_CANDIDATES = Math.max(
            1,
            Integer.getInteger("benchmark.autotuneMaxCandidates", 500)
    );
    private static final int AUTOTUNE_REFINE_TOP_K = 8;
    private static final int AUTOTUNE_REFINE_WARMUP_ITERS = 50;
    private static final int AUTOTUNE_REFINE_MEASURE_ITERS = 300;
    private static final int AUTOTUNE_REFINE_REPEATS = 3;
    private static final int AUTOTUNE_BROADCAST_B0 = 128;
    private static final int AUTOTUNE_BROADCAST_B1 = 8;
    private static final int AUTOTUNE_BROADCAST_F = 128;
    private static final int AUTOTUNE_SAFETY_BROADCAST_B0 = Math.max(
            1,
            Integer.getInteger("benchmark.autotuneSafetyBroadcastB0", Math.min(16, AUTOTUNE_BROADCAST_B0))
    );
    private static final int AUTOTUNE_SAFETY_BROADCAST_B1 = Math.max(
            1,
            Integer.getInteger("benchmark.autotuneSafetyBroadcastB1", Math.min(4, AUTOTUNE_BROADCAST_B1))
    );
    private static final int AUTOTUNE_SAFETY_BROADCAST_F = Math.max(
            1,
            Integer.getInteger("benchmark.autotuneSafetyBroadcastF", Math.min(16, AUTOTUNE_BROADCAST_F))
    );
    private static final boolean AUTOTUNE_ENABLE_SAFETY_PRECHECK =
            Boolean.parseBoolean(System.getProperty("benchmark.autotuneSafetyPrecheck", "true"));
    private static final int BENCH_LINEAR_BATCH = 64;
    private static final int BENCH_LINEAR_IN = 64;
    private static final int BENCH_LINEAR_H1 = 64;
    private static final int BENCH_LINEAR_H2 = 64;
    private static final int BENCH_LINEAR_OUT = 64;
    private static final LinearGraphShape BENCH_LINEAR_SHAPE =
            new LinearGraphShape(BENCH_LINEAR_BATCH, BENCH_LINEAR_IN, BENCH_LINEAR_H1, BENCH_LINEAR_H2, BENCH_LINEAR_OUT);
    private static final int BENCH_GRAPH_BLOCKS = Math.max(
            1,
            Integer.getInteger("benchmark.graphBlocks", 12)
    );
    private static final int AUTOTUNE_GRAPH_BLOCKS = Math.max(
            1,
            Integer.getInteger("benchmark.autotuneGraphBlocks", Math.max(2, BENCH_GRAPH_BLOCKS / 2))
    );
    private static final boolean ENABLE_AUTOTUNE =
            Boolean.parseBoolean(System.getProperty("benchmark.enableAutotune", "true"));
    private static final boolean AUTOTUNE_SCAN_ALL_CANDIDATES =
            Boolean.parseBoolean(System.getProperty("benchmark.autotuneScanAllCandidates", "false"));
    private static final boolean AUTOTUNE_SAFETY_SWEEP_ONLY =
            Boolean.parseBoolean(System.getProperty("benchmark.autotuneSafetySweepOnly", "false"));
    private static final boolean AUTOTUNE_RESCAN_UNSAFE =
            Boolean.parseBoolean(System.getProperty("benchmark.autotuneRescanUnsafe", "false"));
    private static final boolean AUTOTUNE_MEM_ONLY_REPLAY_UNSAFE =
            Boolean.parseBoolean(System.getProperty("benchmark.autotuneMemOnlyReplayUnsafe", "false"));
    private static final int AUTOTUNE_MEM_ONLY_REPLAY_LIMIT =
            Integer.getInteger("benchmark.autotuneMemOnlyReplayLimit", Integer.MAX_VALUE);
    private static final int AUTOTUNE_CANDIDATE_START =
            Math.max(0, Integer.getInteger("benchmark.autotuneCandidateStart", 0));
    private static final int AUTOTUNE_CANDIDATE_COUNT =
            Math.max(1, Integer.getInteger("benchmark.autotuneCandidateCount", Integer.MAX_VALUE));
    private static final String AUTOTUNE_REPLAY_STAGE =
            System.getProperty("benchmark.autotuneReplayStage", "MEM").trim().toUpperCase(Locale.ROOT);
    private static final String AUTOTUNE_PRINT_CANDIDATES =
            System.getProperty("benchmark.autotunePrintCandidates", "").trim();
    private static final String AUTOTUNE_COMPARE_NOOPT_MEM_CANDIDATES =
            System.getProperty("benchmark.autotuneCompareNoOptMemCandidates", "").trim();
    private static final String AUTOTUNE_DEBUG_CANDIDATE_INDICES =
            System.getProperty("benchmark.autotuneDebugCandidateIndices", "").trim();
    private static final boolean AUTOTUNE_RESET_CANDIDATE_RUNTIME =
            Boolean.parseBoolean(System.getProperty("benchmark.autotuneResetCandidateRuntime", "false"));
    private static final boolean AUTOTUNE_SAFETY_STATELESS =
            Boolean.parseBoolean(System.getProperty("benchmark.autotuneSafetyStateless", "false"));
    private static final boolean AUTOTUNE_TRACE_CANDIDATES =
            Boolean.parseBoolean(System.getProperty("benchmark.autotuneTraceCandidates", "false"));
    private static final boolean AUTOTUNE_NUMERICS_POSTCHECK =
            Boolean.parseBoolean(System.getProperty("benchmark.autotuneNumericsPostcheck", "false"));
    private static final int AUTOTUNE_NUMERICS_POSTCHECK_TOP_N =
            Math.max(1, Integer.getInteger("benchmark.autotuneNumericsPostcheckTopN", Integer.MAX_VALUE));
    private static final long AUTOTUNE_NUMERICS_POSTCHECK_SEED =
            Long.getLong("benchmark.autotuneNumericsPostcheckSeed", 42L);
    private static final String AUTOTUNE_SEARCH_MODE =
            System.getProperty("benchmark.autotuneSearchMode", "GRAPH_SCOUT").trim().toUpperCase(Locale.ROOT);
    private static final int AUTOTUNE_STAGE_SCOUT_SAMPLE_PER_STAGE =
            Math.max(1, Integer.getInteger("benchmark.autotuneStageScoutSamplePerStage", 4));
    private static final int AUTOTUNE_STAGE_SCOUT_MAX_SAMPLES_PER_STAGE =
            Math.max(AUTOTUNE_STAGE_SCOUT_SAMPLE_PER_STAGE,
                    Integer.getInteger("benchmark.autotuneStageScoutMaxSamplesPerStage",
                            AUTOTUNE_STAGE_SCOUT_SAMPLE_PER_STAGE * 2));
    private static final int AUTOTUNE_STAGE_SCOUT_MAX_ROUNDS =
            Math.max(1, Integer.getInteger("benchmark.autotuneStageScoutMaxRounds", 4));
    private static final int AUTOTUNE_STAGE_SCOUT_MIN_ACTIVE_FAMILIES =
            Math.max(1, Integer.getInteger("benchmark.autotuneStageScoutMinActiveFamilies", 8));
    private static final double AUTOTUNE_STAGE_SCOUT_CONFIDENCE_Z =
            Math.max(0.0, Double.parseDouble(System.getProperty("benchmark.autotuneStageScoutConfidenceZ", "2.0")));
    private static final int AUTOTUNE_STAGE_SCOUT_WARMUP_ITERS =
            Math.max(0, Integer.getInteger("benchmark.autotuneStageScoutWarmupIters", 1));
    private static final int AUTOTUNE_STAGE_SCOUT_MEASURE_ITERS =
            Math.max(1, Integer.getInteger("benchmark.autotuneStageScoutMeasureIters", 3));
    private static final int AUTOTUNE_STAGE_SCOUT_TOP_TRAIN =
            Math.max(1, Integer.getInteger("benchmark.autotuneStageScoutTopTrain", 3));
    private static final int AUTOTUNE_STAGE_SCOUT_TOP_INF =
            Math.max(1, Integer.getInteger("benchmark.autotuneStageScoutTopInference", 3));
    private static final int AUTOTUNE_PRESCREEN_WARMUP_ITERS =
            Math.max(0, Integer.getInteger("benchmark.autotunePrescreenWarmupIters", 1));
    private static final int AUTOTUNE_PRESCREEN_MEASURE_ITERS =
            Math.max(1, Integer.getInteger("benchmark.autotunePrescreenMeasureIters", 3));
    private static final int AUTOTUNE_PRESCREEN_KEEP_TRAIN =
            Math.max(1, Integer.getInteger("benchmark.autotunePrescreenKeepTrain", 128));
    private static final int AUTOTUNE_PRESCREEN_KEEP_INF =
            Math.max(1, Integer.getInteger("benchmark.autotunePrescreenKeepInference", 128));
    private static final int AUTOTUNE_PRESCREEN_DIVERSITY_SEEDS_PER_FAMILY =
            Math.max(0, Integer.getInteger("benchmark.autotunePrescreenDiversitySeedsPerFamily", 1));
    private static final int AUTOTUNE_PRESCREEN_MAX_PER_STAGE_ORDER =
            Math.max(1, Integer.getInteger("benchmark.autotunePrescreenMaxPerStageOrder", 64));
    private static final int AUTOTUNE_FUSED_EARLY_TIER_PREWARM_ITERS =
            Math.max(0, Integer.getInteger("benchmark.autotuneFusedEarlyTierPrewarmIters", 8));
    private static final int AUTOTUNE_BEAM_ROUNDS =
            Math.max(1, Integer.getInteger("benchmark.autotuneBeamRounds", 4));
    private static final int AUTOTUNE_BEAM_SEED_TRAIN =
            Math.max(1, Integer.getInteger("benchmark.autotuneBeamSeedTrain", 8));
    private static final int AUTOTUNE_BEAM_SEED_INF =
            Math.max(1, Integer.getInteger("benchmark.autotuneBeamSeedInference", 8));
    private static final int AUTOTUNE_BEAM_WIDTH_TRAIN =
            Math.max(1, Integer.getInteger("benchmark.autotuneBeamWidthTrain", 8));
    private static final int AUTOTUNE_BEAM_WIDTH_INF =
            Math.max(1, Integer.getInteger("benchmark.autotuneBeamWidthInference", 8));
    private static final int AUTOTUNE_BEAM_KEEP_TRAIN =
            Math.max(1, Integer.getInteger("benchmark.autotuneBeamKeepTrain", 32));
    private static final int AUTOTUNE_BEAM_KEEP_INF =
            Math.max(1, Integer.getInteger("benchmark.autotuneBeamKeepInference", 32));
    private static final int AUTOTUNE_BEAM_MAX_PER_STAGE =
            Math.max(1, Integer.getInteger("benchmark.autotuneBeamMaxPerStage", 3));
    private static final double ABS_TOL_FLOAT64 = 1e-12;
    private static final double REL_TOL_FLOAT64 = 1e-12;
    private static final double ABS_TOL_FLOAT32 = 1e-5;
    private static final double REL_TOL_FLOAT32 = 1e-5;
    private static final double AUTOTUNE_TRAIN_BROADCAST_WEIGHT = 0.15;
    private static final double AUTOTUNE_INF_BROADCAST_WEIGHT = 0.30;
    private static final Random RNG = new Random(42);
    private static final DataType BENCH_DTYPE = resolveBenchDataType();
    private static final double ABS_TOL =
            BENCH_DTYPE == DataType.FLOAT64 ? ABS_TOL_FLOAT64 : ABS_TOL_FLOAT32;
    private static final double REL_TOL =
            BENCH_DTYPE == DataType.FLOAT64 ? REL_TOL_FLOAT64 : REL_TOL_FLOAT32;

    private static final String RESET = "\u001B[0m";
    private static final String BOLD = "\u001B[1m";
    private static final String CYAN = "\u001B[36m";
    private static final String GREEN = "\u001B[32m";
    private static final String YELLOW = "\u001B[33m";
    private static final String RED = "\u001B[31m";
    private static final String GRAY = "\u001B[90m";
    private static final Path PROFILE_PATH = Path.of("config", "optimizer-profile.json");
    private static final Path AUTOTUNE_BEST_PATH = Path.of("build", "optimizer-autotune", "best-profile.json");
    private static final Path AUTOTUNE_BEST_TRAINING_PATH = Path.of("build", "optimizer-autotune", "best-profile-training.json");
    private static final Path AUTOTUNE_BEST_INFERENCE_PATH = Path.of("build", "optimizer-autotune", "best-profile-inference.json");
    private static final Path HW_PROFILE_PATH = Path.of("config", "optimizer-hw-profiles.tsv");
    private static final Path AUTOTUNE_HISTORY_PATH = Path.of("build", "optimizer-autotune", "candidate-history.tsv");
    private static final Path AUTOTUNE_PROGRESS_PATH = Path.of("build", "optimizer-autotune", "progress.json");
    private static final Path AUTOTUNE_PROGRESS_ROWS_PATH = Path.of("build", "optimizer-autotune", "progress-rows.tsv");
    private static final Path AUTOTUNE_NUMERICS_REPORT_DIR = Path.of("build", "numerics");
    private static final int AUTOTUNE_HISTORY_SCHEMA_VERSION = 1;
    private static final int AUTOTUNE_ENGINE_VERSION = 3;
    private static final int HW_PROFILE_MAX_BUCKETS = 10;
    private static final int AUTOTUNE_PROGRESS_LOG_EVERY =
            Math.max(1, Integer.getInteger("benchmark.autotuneProgressLogEvery", 100));
    private static final long AUTOTUNE_PROGRESS_MIN_INTERVAL_MS =
            Math.max(0L, Long.getLong("benchmark.autotuneProgressMinIntervalMs", 2_000L));
    private static final DateTimeFormatter AUTOTUNE_NUMERICS_TS_FORMAT =
            DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss");
    private static final CandidateMeasurementHarness CANDIDATE_MEASUREMENT_HARNESS =
            new CandidateMeasurementHarness(
                    OptimizerBenchmarkFramework::createMeasuredBenchmarkScenario,
                    OptimizerBenchmarkFramework::createMeasuredBroadcastScenario,
                    AUTOTUNE_GRAPH_BLOCKS,
                    AUTOTUNE_FUSED_EARLY_TIER_PREWARM_ITERS,
                    NanoClock.SYSTEM
            );

    public static void run() {
        if (ENABLE_AUTOTUNE && AUTOTUNE_SAFETY_SWEEP_ONLY) {
            autoTune();
            return;
        }

        List<OptimizerCandidate> candidates = applyProfileToDefaults(OptimizerCandidateFactory.defaultCandidates());
        OptimizerCandidate noOptCandidate = findByName(candidates, "NO_OPT");
        OptimizerCandidate recommended = findByName(candidates, "RECOMMENDED");
        OptimizerCandidate inferencePerf = findByName(candidates, "INFERENCE_PERF");

        runScalarSanityCheck(OptimizerBuilder.build(recommended));

        double[] baseA = randomData(SIZE);
        double[] baseB = randomData(SIZE);
        double[] baseC = randomData(SIZE);

        // Forward benchmark běží přes inference-only pipeline (bez backward grafu).
        PreparedBenchmarkScenario noOptForward = newBenchState(baseA, baseB, baseC, noOptCandidate, false, BENCH_GRAPH_BLOCKS);
        PreparedBenchmarkScenario optForward = newBenchState(baseA, baseB, baseC, inferencePerf, false, BENCH_GRAPH_BLOCKS);
        // Training benchmark běží přes training pipeline.
        PreparedBenchmarkScenario noOptTrain = newBenchState(baseA, baseB, baseC, noOptCandidate, true, BENCH_GRAPH_BLOCKS);
        PreparedBenchmarkScenario optTrain = newBenchState(baseA, baseB, baseC, recommended, true, BENCH_GRAPH_BLOCKS);

        int compiledNoOpt = noOptTrain.compiledGraph().getCompiledGraphAsList().size();
        int compiledOpt = optTrain.compiledGraph().getCompiledGraphAsList().size();

        for (int i = 0; i < WARMUP_ITERS; i++) {
            noOptForward.compute();
            optForward.compute();
        }

        long t0 = System.nanoTime();
        for (int i = 0; i < MEASURE_ITERS; i++) noOptForward.compute();
        long t1 = System.nanoTime();
        for (int i = 0; i < MEASURE_ITERS; i++) optForward.compute();
        long t2 = System.nanoTime();

        double forwardNoOptMs = (t1 - t0) / 1_000_000.0 / MEASURE_ITERS;
        double forwardOptMs = (t2 - t1) / 1_000_000.0 / MEASURE_ITERS;
        double forwardSpeedup = forwardOptMs == 0.0 ? Double.POSITIVE_INFINITY : forwardNoOptMs / forwardOptMs;

        for (int i = 0; i < WARMUP_ITERS; i++) {
            noOptTrain.compute();
            optTrain.compute();
        }

        long t3 = System.nanoTime();
        for (int i = 0; i < MEASURE_ITERS; i++) noOptTrain.compute();
        long t4 = System.nanoTime();
        for (int i = 0; i < MEASURE_ITERS; i++) optTrain.compute();
        long t5 = System.nanoTime();

        double trainNoOptMs = (t4 - t3) / 1_000_000.0 / MEASURE_ITERS;
        double trainOptMs = (t5 - t4) / 1_000_000.0 / MEASURE_ITERS;
        double trainSpeedup = trainOptMs == 0.0 ? Double.POSITIVE_INFINITY : trainNoOptMs / trainOptMs;

        RunResult freshNoOpt = runFresh(baseA, baseB, baseC, noOptCandidate, BENCH_GRAPH_BLOCKS);
        RunResult freshOpt = runFresh(baseA, baseB, baseC, recommended, BENCH_GRAPH_BLOCKS);

        Diff ta7Diff = diff(freshNoOpt.ta7, freshOpt.ta7);
        Diff gradADiff = diff(freshNoOpt.gradA, freshOpt.gradA);
        Diff gradBDiff = diff(freshNoOpt.gradB, freshOpt.gradB);
        Diff gradCDiff = diff(freshNoOpt.gradC, freshOpt.gradC);

        boolean okTa7 = ta7Diff.ok();
        boolean okA = gradADiff.ok();
        boolean okB = gradBDiff.ok();
        boolean okC = gradCDiff.ok();
        boolean allOk = okTa7 && okA && okB && okC;

        String allColor = allOk ? GREEN : RED;

        System.out.println(BOLD + CYAN + "=== Optimizer Benchmark ===" + RESET);
        System.out.println(GRAY + "DType=" + BENCH_DTYPE + RESET);
        System.out.println(GRAY + "Size=" + SIZE + ", graphBlocks=" + BENCH_GRAPH_BLOCKS
                + ", warmup=" + WARMUP_ITERS + ", measure=" + MEASURE_ITERS + RESET);
        System.out.println();
        System.out.println(BOLD + CYAN + "[Graph Size]" + RESET);
        System.out.println(YELLOW + "Compiled graph no-opt: " + RESET + compiledNoOpt);
        System.out.println(GREEN + "Compiled graph opt:    " + RESET + compiledOpt);
        System.out.println();
        System.out.println(BOLD + CYAN + "[Forward]" + RESET);
        System.out.println(YELLOW + "Avg no-opt: " + RESET + forwardNoOptMs + " ms");
        System.out.println(GREEN + "Avg opt:    " + RESET + forwardOptMs + " ms");
        System.out.println(CYAN + "Speedup:    " + RESET + forwardSpeedup + "x");
        System.out.println();
        System.out.println(BOLD + CYAN + "[Training]" + RESET);
        System.out.println(YELLOW + "Avg no-opt: " + RESET + trainNoOptMs + " ms");
        System.out.println(GREEN + "Avg opt:    " + RESET + trainOptMs + " ms");
        System.out.println(CYAN + "Speedup:    " + RESET + trainSpeedup + "x");
        System.out.println();
        System.out.println(BOLD + CYAN + "[Correctness: fresh run, same base inputs]" + RESET);
        System.out.println("Ta7 equal:    " + okTa7 + " | " + ta7Diff);
        System.out.println("Grad A equal: " + okA + " | " + gradADiff);
        System.out.println("Grad B equal: " + okB + " | " + gradBDiff);
        System.out.println("Grad C equal: " + okC + " | " + gradCDiff);
        System.out.println(allColor + "All equal:    " + allOk + RESET);

        benchmarkByStage(baseA, baseB, baseC, candidates);

        if (ENABLE_AUTOTUNE) {
            autoTune();
        }
    }

    private static List<OptimizerCandidate> applyProfileToDefaults(List<OptimizerCandidate> defaults) {
        List<OptimizerCandidate> out = new ArrayList<>(defaults.size());
        TuningKnobs tuned = OptimizerProfileIO.loadKnobsOrDefault(PROFILE_PATH, TuningKnobs.trainingDefaults());
        String hwBucket = OptimizerProfileIO.hardwareBucketKey();
        for (OptimizerCandidate c : defaults) {
            if ("RECOMMENDED".equals(c.name())) {
                out.add(new OptimizerCandidate(c.name(), c.stageOrder(), tuned));
            } else {
                out.add(c);
            }
        }

        OptimizerCandidate profiledRecommended = findByName(out, "RECOMMENDED");
        OptimizerCandidate overriddenRecommended = OptimizerProfileIO.loadRecommendedOverrideOrDefault(AUTOTUNE_BEST_TRAINING_PATH, profiledRecommended);
        if (overriddenRecommended != profiledRecommended) {
            replaceCandidate(out, overriddenRecommended);
            profiledRecommended = overriddenRecommended;
        }
        OptimizerCandidate profiledInference = findByName(out, "INFERENCE_PERF");
        OptimizerCandidate overriddenInference = OptimizerProfileIO.loadRecommendedOverrideOrDefault(AUTOTUNE_BEST_INFERENCE_PATH, profiledInference);
        if (overriddenInference != profiledInference) {
            replaceCandidate(out, overriddenInference);
        }
        OptimizerCandidate currentRecommended = findByName(out, "RECOMMENDED");
        OptimizerCandidate archRecommended = OptimizerProfileIO.loadArchitectureDefaultOverrideOrDefault("TRAINING", currentRecommended);
        if (archRecommended != currentRecommended) {
            replaceCandidate(out, archRecommended);
        }
        OptimizerCandidate currentInference = findByName(out, "INFERENCE_PERF");
        OptimizerCandidate archInference = OptimizerProfileIO.loadArchitectureDefaultOverrideOrDefault("INFERENCE", currentInference);
        if (archInference != currentInference) {
            replaceCandidate(out, archInference);
        }
        OptimizerCandidate hwRecommended = OptimizerProfileIO.loadHardwareOverrideOrDefault(
                HW_PROFILE_PATH,
                hwBucket,
                "TRAINING",
                findByName(out, "RECOMMENDED")
        );
        replaceCandidate(out, hwRecommended);
        OptimizerCandidate hwInference = OptimizerProfileIO.loadHardwareOverrideOrDefault(
                HW_PROFILE_PATH,
                hwBucket,
                "INFERENCE",
                findByName(out, "INFERENCE_PERF")
        );
        replaceCandidate(out, hwInference);

        try {
            runScalarSanityCore(OptimizerBuilder.build(profiledRecommended));
            runScalarSanityCore(OptimizerBuilder.build(findByName(out, "INFERENCE_PERF")));
            if (Files.exists(AUTOTUNE_BEST_TRAINING_PATH)) {
                System.out.println(GRAY + "Using autotune training profile from " + AUTOTUNE_BEST_TRAINING_PATH.toAbsolutePath() + RESET);
            } else if (Files.exists(PROFILE_PATH)) {
                System.out.println(GRAY + "Using optimizer profile from " + PROFILE_PATH.toAbsolutePath() + RESET);
            }
            if (Files.exists(AUTOTUNE_BEST_INFERENCE_PATH)) {
                System.out.println(GRAY + "Using autotune inference profile from " + AUTOTUNE_BEST_INFERENCE_PATH.toAbsolutePath() + RESET);
            }
            if (Files.exists(HW_PROFILE_PATH)) {
                System.out.println(GRAY + "Using HW bucket profile from " + HW_PROFILE_PATH.toAbsolutePath() + " (" + hwBucket + ")" + RESET);
            } else {
                System.out.println(GRAY + "Using architecture preset for os.arch=" + System.getProperty("os.arch", "unknown") + RESET);
            }
            return out;
        } catch (IllegalStateException e) {
            System.out.println(RED + "Optimizer profile validation failed; falling back to defaults." + RESET);
            System.out.println(RED + "Reason: " + e.getMessage() + RESET);
            return defaults;
        }
    }

    private static void replaceCandidate(List<OptimizerCandidate> candidates, OptimizerCandidate replacement) {
        for (int i = 0; i < candidates.size(); i++) {
            if (candidates.get(i).name().equals(replacement.name())) {
                candidates.set(i, replacement);
                return;
            }
        }
    }

    private static void benchmarkByStage(double[] baseA, double[] baseB, double[] baseC, List<OptimizerCandidate> stages) {
        RunResult baseline = runFresh(baseA, baseB, baseC, findByName(stages, "NO_OPT"), BENCH_GRAPH_BLOCKS);

        System.out.println();
        System.out.println(BOLD + CYAN + "[Stage Breakdown]" + RESET);
        System.out.println(GRAY + "Per-stage benchmark: warmup=" + STAGE_WARMUP_ITERS + ", measure=" + STAGE_MEASURE_ITERS + RESET);
        final String headerFmt = "%-18s %8s %10s %12s %12s %9s %9s %10s %12s";
        final String rowFmt = "%-18s %8d %10d %12.4f %12.4f %9s %9s %10s %12s";
        String header = String.format(headerFmt, "NAME", "GRAPH_INF", "GRAPH_TRN", "FWD_MS", "TRAIN_MS", "FWD_X", "TRN_X", "CHECK", "MAX_ABS");
        System.out.println(GRAY + header + RESET);
        System.out.println(GRAY + "-".repeat(header.length()) + RESET);

        double baseForward = -1.0;
        double baseTrain = -1.0;

        for (int idx = 0; idx < stages.size(); idx++) {
            OptimizerCandidate stage = stages.get(idx);
            PreparedBenchmarkScenario stateForward = newBenchState(baseA, baseB, baseC, stage, false, BENCH_GRAPH_BLOCKS);
            int graphSize = stateForward.compiledGraph().getCompiledGraphAsList().size();

            for (int i = 0; i < STAGE_WARMUP_ITERS; i++) stateForward.compute();
            long t0 = System.nanoTime();
            for (int i = 0; i < STAGE_MEASURE_ITERS; i++) stateForward.compute();
            long t1 = System.nanoTime();
            double forwardMs = (t1 - t0) / 1_000_000.0 / STAGE_MEASURE_ITERS;

            PreparedBenchmarkScenario stateTrain = newBenchState(baseA, baseB, baseC, stage, true, BENCH_GRAPH_BLOCKS);
            int trainingGraphSize = stateTrain.compiledGraph().getCompiledGraphAsList().size();
            for (int i = 0; i < STAGE_WARMUP_ITERS; i++) stateTrain.compute();
            long t2 = System.nanoTime();
            for (int i = 0; i < STAGE_MEASURE_ITERS; i++) stateTrain.compute();
            long t3 = System.nanoTime();
            double trainMs = (t3 - t2) / 1_000_000.0 / STAGE_MEASURE_ITERS;

            RunResult current = runFresh(baseA, baseB, baseC, stage, BENCH_GRAPH_BLOCKS);
            Diff dOut = diff(baseline.ta7, current.ta7);
            Diff dA = diff(baseline.gradA, current.gradA);
            Diff dB = diff(baseline.gradB, current.gradB);
            Diff dC = diff(baseline.gradC, current.gradC);
            boolean ok = dOut.ok() && dA.ok() && dB.ok() && dC.ok();
            double maxAbs = Math.max(Math.max(dOut.maxAbs, dA.maxAbs), Math.max(dB.maxAbs, dC.maxAbs));

            if (idx == 0) {
                baseForward = forwardMs;
                baseTrain = trainMs;
            }

            double fwdSpeed = forwardMs == 0.0 ? Double.POSITIVE_INFINITY : baseForward / forwardMs;
            double trainSpeed = trainMs == 0.0 ? Double.POSITIVE_INFINITY : baseTrain / trainMs;

            String color = ok ? GREEN : RED;
            System.out.println(color + String.format(
                    rowFmt,
                    stage.name(),
                    graphSize,
                    trainingGraphSize,
                    forwardMs,
                    trainMs,
                    String.format("%.3fx", fwdSpeed),
                    String.format("%.3fx", trainSpeed),
                    ok ? "OK" : "MISMATCH",
                    String.format(Locale.US, "%.3e", maxAbs)
            ) + RESET);
            if (!ok) {
                System.out.println(RED + "  diff[" + stage.name() + "]"
                        + " out=" + String.format(Locale.US, "%.3e", dOut.maxAbs)
                        + ", gradA=" + String.format(Locale.US, "%.3e", dA.maxAbs)
                        + ", gradB=" + String.format(Locale.US, "%.3e", dB.maxAbs)
                        + ", gradC=" + String.format(Locale.US, "%.3e", dC.maxAbs)
                        + RESET);
            }
        }

        System.out.println();
        System.out.println(GRAY + "Legend:" + RESET);
        System.out.println(GRAY + "  NAME     = optimizer configuration" + RESET);
        System.out.println(GRAY + "  GRAPH_INF= number of nodes in forward-only compiled graph (inference pipeline)" + RESET);
        System.out.println(GRAY + "  GRAPH_TRN= number of nodes in training compiled graph (forward+backward pipeline)" + RESET);
        System.out.println(GRAY + "  FWD_MS   = average time per forward run [ms]" + RESET);
        System.out.println(GRAY + "  TRAIN_MS = average time per training run [ms]" + RESET);
        System.out.println(GRAY + "  FWD_X    = forward speedup vs NO_OPT (higher is better)" + RESET);
        System.out.println(GRAY + "  TRN_X    = training speedup vs NO_OPT (higher is better)" + RESET);
        System.out.println(GRAY + "  CHECK    = matches Ta7 + grad(A,B,C) vs NO_OPT (OK/MISMATCH)" + RESET);
        System.out.println(GRAY + "  MAX_ABS  = max absolute diff across {Ta7, gradA, gradB, gradC}" + RESET);
        System.out.println();

        System.out.println(GRAY + "Tuning knobs note:" + RESET);
        System.out.println(GRAY + "  matmul tiling and loop unrolling fields are prepared in TuningKnobs" + RESET);
        System.out.println(GRAY + "  and can be wired once corresponding kernels/codegen paths are implemented." + RESET);
        System.out.println();

        OptimizerCandidate recommended = findByName(stages, "RECOMMENDED");
        var rk = recommended.knobs();
        var rf = rk.fuseConfig();
        var rkc = rk.kernelConfig();
        System.out.println(BOLD + CYAN + "[RECOMMENDED Params]" + RESET);
        System.out.println(GRAY + "  stageOrder=" + recommended.stageOrder() + RESET);
        System.out.println(GRAY + "  strictCseSafety=" + rk.strictCseSafety() + RESET);
        System.out.println(GRAY + "  fuse.maxClusterNodes=" + rf.maxClusterNodes()
                + ", scoreThreshold=" + String.format(Locale.US, "%.6f", rf.scoreThreshold())
                + ", internalEdgeBonus=" + String.format(Locale.US, "%.6f", rf.internalEdgeBonus())
                + ", externalInputPenalty=" + String.format(Locale.US, "%.6f", rf.externalInputPenalty())
                + ", sharedExpensivePenalty=" + String.format(Locale.US, "%.6f", rf.sharedExpensivePenalty())
                + ", nonCheapBonus=" + String.format(Locale.US, "%.6f", rf.nonCheapBonus())
                + ", preserveSharedExpensiveNodes=" + rf.preserveSharedExpensiveNodes() + RESET);
        System.out.println(GRAY + "  kernel.cpu=[unroll=" + rkc.cpu().loopUnrollFactor()
                + ", tileM=" + rkc.cpu().matMulTileM()
                + ", tileN=" + rkc.cpu().matMulTileN()
                + ", tileK=" + rkc.cpu().matMulTileK()
                + ", vecMin=" + rkc.cpu().vectorMinSize()
                + ", parMin=" + rkc.cpu().parallelMinSize()
                + ", matMulParMin=" + rkc.cpu().matMulParallelMinSize()
                + ", par=" + rkc.cpu().parallelism()
                + ", chunksPerWorker=" + rkc.cpu().chunksPerWorker()
                + ", minChunk=" + rkc.cpu().minChunkSize()
                + ", contigMatThreshold=" + rkc.cpu().contiguousMaterializeThreshold()
                + ", lowCostNsPerElemThreshold=" + String.format(Locale.US, "%.4f", rkc.cpu().lowCostNsPerElementThreshold())
                + ", vecPolicyCheap=" + rkc.cpu().vectorPolicyCheap()
                + ", vecPolicyTrans=" + rkc.cpu().vectorPolicyTranscendental()
                + ", vecPolicyRed=" + rkc.cpu().vectorPolicyReduction() + "]" + RESET);
        System.out.println(GRAY + "  kernel.cuda=[unroll=" + rkc.cuda().loopUnrollFactor()
                + ", tileM=" + rkc.cuda().matMulTileM()
                + ", tileN=" + rkc.cuda().matMulTileN()
                + ", tileK=" + rkc.cuda().matMulTileK() + "]" + RESET);
        System.out.println(GRAY + "  kernel.opencl=[unroll=" + rkc.opencl().loopUnrollFactor()
                + ", tileM=" + rkc.opencl().matMulTileM()
                + ", tileN=" + rkc.opencl().matMulTileN()
                + ", tileK=" + rkc.opencl().matMulTileK() + "]" + RESET);
        System.out.println();

        OptimizerCandidate inferencePerf = findByName(stages, "INFERENCE_PERF");
        var ik = inferencePerf.knobs();
        var inf = ik.fuseConfig();
        var ikc = ik.kernelConfig();
        System.out.println(BOLD + CYAN + "[INFERENCE_PERF Params]" + RESET);
        System.out.println(GRAY + "  stageOrder=" + inferencePerf.stageOrder() + RESET);
        System.out.println(GRAY + "  strictCseSafety=" + ik.strictCseSafety() + RESET);
        System.out.println(GRAY + "  fuse.maxClusterNodes=" + inf.maxClusterNodes()
                + ", scoreThreshold=" + String.format(Locale.US, "%.6f", inf.scoreThreshold())
                + ", internalEdgeBonus=" + String.format(Locale.US, "%.6f", inf.internalEdgeBonus())
                + ", externalInputPenalty=" + String.format(Locale.US, "%.6f", inf.externalInputPenalty())
                + ", sharedExpensivePenalty=" + String.format(Locale.US, "%.6f", inf.sharedExpensivePenalty())
                + ", nonCheapBonus=" + String.format(Locale.US, "%.6f", inf.nonCheapBonus())
                + ", preserveSharedExpensiveNodes=" + inf.preserveSharedExpensiveNodes() + RESET);
        System.out.println(GRAY + "  kernel.cpu=[unroll=" + ikc.cpu().loopUnrollFactor()
                + ", tileM=" + ikc.cpu().matMulTileM()
                + ", tileN=" + ikc.cpu().matMulTileN()
                + ", tileK=" + ikc.cpu().matMulTileK()
                + ", vecMin=" + ikc.cpu().vectorMinSize()
                + ", parMin=" + ikc.cpu().parallelMinSize()
                + ", matMulParMin=" + ikc.cpu().matMulParallelMinSize()
                + ", par=" + ikc.cpu().parallelism()
                + ", chunksPerWorker=" + ikc.cpu().chunksPerWorker()
                + ", minChunk=" + ikc.cpu().minChunkSize()
                + ", contigMatThreshold=" + ikc.cpu().contiguousMaterializeThreshold()
                + ", lowCostNsPerElemThreshold=" + String.format(Locale.US, "%.4f", ikc.cpu().lowCostNsPerElementThreshold())
                + ", vecPolicyCheap=" + ikc.cpu().vectorPolicyCheap()
                + ", vecPolicyTrans=" + ikc.cpu().vectorPolicyTranscendental()
                + ", vecPolicyRed=" + ikc.cpu().vectorPolicyReduction() + "]" + RESET);
        System.out.println(GRAY + "  kernel.cuda=[unroll=" + ikc.cuda().loopUnrollFactor()
                + ", tileM=" + ikc.cuda().matMulTileM()
                + ", tileN=" + ikc.cuda().matMulTileN()
                + ", tileK=" + ikc.cuda().matMulTileK() + "]" + RESET);
        System.out.println(GRAY + "  kernel.opencl=[unroll=" + ikc.opencl().loopUnrollFactor()
                + ", tileM=" + ikc.opencl().matMulTileM()
                + ", tileN=" + ikc.opencl().matMulTileN()
                + ", tileK=" + ikc.opencl().matMulTileK() + "]" + RESET);
        System.out.println();
    }

    private static void autoTune() {
        double[] baseA = randomData(AUTOTUNE_SIZE);
        double[] baseB = randomData(AUTOTUNE_SIZE);
        double[] baseC = randomData(AUTOTUNE_SIZE);
        double[] baseBroadcastA = randomData(AUTOTUNE_BROADCAST_B0 * AUTOTUNE_BROADCAST_F);
        double[] baseBroadcastB = randomData(AUTOTUNE_BROADCAST_B1 * AUTOTUNE_BROADCAST_F);
        double[] baseBroadcastC = randomData(AUTOTUNE_BROADCAST_B0 * AUTOTUNE_BROADCAST_B1 * AUTOTUNE_BROADCAST_F);
        double[] safetyA = randomData(AUTOTUNE_SAFETY_SIZE);
        double[] safetyB = randomData(AUTOTUNE_SAFETY_SIZE);
        double[] safetyC = randomData(AUTOTUNE_SAFETY_SIZE);
        double[] safetyBroadcastA = randomData(AUTOTUNE_SAFETY_BROADCAST_B0 * AUTOTUNE_SAFETY_BROADCAST_F);
        double[] safetyBroadcastB = randomData(AUTOTUNE_SAFETY_BROADCAST_B1 * AUTOTUNE_SAFETY_BROADCAST_F);
        double[] safetyBroadcastC = randomData(AUTOTUNE_SAFETY_BROADCAST_B0 * AUTOTUNE_SAFETY_BROADCAST_B1 * AUTOTUNE_SAFETY_BROADCAST_F);

        RunResult baseline = runFresh(
                baseA,
                baseB,
                baseC,
                new OptimizerCandidate("NO_OPT", List.of(), TuningKnobs.trainingDefaults()),
                AUTOTUNE_GRAPH_BLOCKS
        );
        BroadcastRunResult baselineBroadcast = runBroadcastFresh(
                baseBroadcastA,
                baseBroadcastB,
                baseBroadcastC,
                new OptimizerCandidate("NO_OPT", List.of(), TuningKnobs.trainingDefaults())
        );
        RunResult safetyBaseline = runFresh(
                safetyA,
                safetyB,
                safetyC,
                new OptimizerCandidate("NO_OPT", List.of(), TuningKnobs.trainingDefaults()),
                AUTOTUNE_GRAPH_BLOCKS
        );
        BroadcastRunResult safetyBaselineBroadcast = runBroadcastFresh(
                safetyBroadcastA,
                safetyBroadcastB,
                safetyBroadcastC,
                new OptimizerCandidate("NO_OPT", List.of(), TuningKnobs.trainingDefaults()),
                AUTOTUNE_SAFETY_BROADCAST_B0,
                AUTOTUNE_SAFETY_BROADCAST_B1,
                AUTOTUNE_SAFETY_BROADCAST_F
        );
        List<OptimizerCandidate> all = OptimizerCandidateFactory.autotuneCandidates();
        if (!AUTOTUNE_DEBUG_CANDIDATE_INDICES.isEmpty()) {
            runDebugCandidateSequence(
                    all,
                    safetyBaseline,
                    safetyBaselineBroadcast,
                    safetyA,
                    safetyB,
                    safetyC,
                    safetyBroadcastA,
                    safetyBroadcastB,
                    safetyBroadcastC
            );
            return;
        }
        if (!AUTOTUNE_PRINT_CANDIDATES.isEmpty()) {
            printCandidatesByName(all, AUTOTUNE_PRINT_CANDIDATES);
            return;
        }
        if (!AUTOTUNE_COMPARE_NOOPT_MEM_CANDIDATES.isEmpty()) {
            compareNoOptVsMemForCandidates(
                    all,
                    AUTOTUNE_COMPARE_NOOPT_MEM_CANDIDATES,
                    baseline,
                    baselineBroadcast,
                    safetyBaseline,
                    safetyBaselineBroadcast,
                    baseA,
                    baseB,
                    baseC,
                    baseBroadcastA,
                    baseBroadcastB,
                    baseBroadcastC,
                    safetyA,
                    safetyB,
                    safetyC,
                    safetyBroadcastA,
                    safetyBroadcastB,
                    safetyBroadcastC
            );
            return;
        }
        List<OptimizerCandidate> candidates = AUTOTUNE_SCAN_ALL_CANDIDATES
                ? all
                : capCandidatesDeterministic(all, AUTOTUNE_MAX_CANDIDATES);
        candidates = sliceCandidates(candidates, AUTOTUNE_CANDIDATE_START, AUTOTUNE_CANDIDATE_COUNT);
        if (!AUTOTUNE_SAFETY_SWEEP_ONLY && "GRAPH_SCOUT".equals(AUTOTUNE_SEARCH_MODE)) {
            candidates = reduceCandidatesViaGraphScout(
                    candidates,
                    baseA,
                    baseB,
                    baseC,
                    baseBroadcastA,
                    baseBroadcastB,
                    baseBroadcastC
            );
        }
        String historyContext = autoTuneHistoryContextSignature();
        Path historyPath = resolveAutotuneHistoryPath();
        Path progressPath = resolveAutotuneProgressPath();
        Path progressRowsPath = resolveAutotuneProgressRowsPath();
        UnsafeCandidateHistory history = UnsafeCandidateHistory.load(historyPath, historyContext);
        AutoTuneProgressTracker progress = new AutoTuneProgressTracker(
                progressPath,
                progressRowsPath,
                BENCH_DTYPE,
                AUTOTUNE_CANDIDATE_START,
                candidates.size(),
                AUTOTUNE_PROGRESS_LOG_EVERY,
                AUTOTUNE_PROGRESS_MIN_INTERVAL_MS
        );

        AutoTuneResult bestTraining = null;
        AutoTuneResult bestInference = null;

        System.out.println(BOLD + CYAN + "[Auto-Tune]" + RESET);
        System.out.println(GRAY + "Two-phase mode" + RESET);
        System.out.println(GRAY + "Candidates total=" + all.size()
                + ", evaluated=" + candidates.size()
                + ", size=" + AUTOTUNE_SIZE
                + ", graphBlocks=" + AUTOTUNE_GRAPH_BLOCKS
                + ", broadcastShape=[" + AUTOTUNE_BROADCAST_B0 + ",1," + AUTOTUNE_BROADCAST_F + "]x[1," + AUTOTUNE_BROADCAST_B1 + "," + AUTOTUNE_BROADCAST_F + "]"
                + ", safetyPrecheck=" + AUTOTUNE_ENABLE_SAFETY_PRECHECK
                + ", safetySweepOnly=" + AUTOTUNE_SAFETY_SWEEP_ONLY
                + ", scanAllCandidates=" + AUTOTUNE_SCAN_ALL_CANDIDATES
                + ", rescanUnsafe=" + AUTOTUNE_RESCAN_UNSAFE
                + ", safetySize=" + AUTOTUNE_SAFETY_SIZE
                + ", safetyBroadcastShape=[" + AUTOTUNE_SAFETY_BROADCAST_B0 + ",1," + AUTOTUNE_SAFETY_BROADCAST_F + "]x[1," + AUTOTUNE_SAFETY_BROADCAST_B1 + "," + AUTOTUNE_SAFETY_BROADCAST_F + "]"
                + ", warmup=" + AUTOTUNE_WARMUP_ITERS
                + ", measure=" + AUTOTUNE_MEASURE_ITERS
                + ", refineTopK=" + AUTOTUNE_REFINE_TOP_K
                + ", refineWarmup=" + AUTOTUNE_REFINE_WARMUP_ITERS
                + ", refineMeasure=" + AUTOTUNE_REFINE_MEASURE_ITERS
                + ", refineRepeats=" + AUTOTUNE_REFINE_REPEATS
                + ", memOnlyReplayUnsafe=" + AUTOTUNE_MEM_ONLY_REPLAY_UNSAFE
                + ", replayStage=" + AUTOTUNE_REPLAY_STAGE
                + ", candidateStart=" + AUTOTUNE_CANDIDATE_START
                + ", candidateCount=" + AUTOTUNE_CANDIDATE_COUNT
                + ", unsafeHistory=" + historyPath.toAbsolutePath() + RESET);

        if (AUTOTUNE_MEM_ONLY_REPLAY_UNSAFE) {
            runMemOnlyReplayForUnsafeCandidates(
                    all,
                    history,
                    baseline,
                    baselineBroadcast,
                    safetyBaseline,
                    safetyBaselineBroadcast,
                    baseA,
                    baseB,
                    baseC,
                    baseBroadcastA,
                    baseBroadcastB,
                    baseBroadcastC,
                    safetyA,
                    safetyB,
                    safetyC,
                    safetyBroadcastA,
                    safetyBroadcastB,
                    safetyBroadcastC
            );
            return;
        }

        final int phase1TotalCandidates = candidates.size();
        final Phase1Counters[] phase1CountersHolder = {Phase1Counters.zero()};
        final String hwBucket = OptimizerProfileIO.hardwareBucketKey();
        AutoTuneSessionResult session;
        try {
            session = AutoTuneSessionRunner.run(
                    candidates,
                    new AutoTuneSessionConfig(
                            AUTOTUNE_SAFETY_SWEEP_ONLY,
                            AUTOTUNE_SAFETY_STATELESS,
                            new AutoTuneFinalizationConfig(
                                    AUTOTUNE_REFINE_TOP_K,
                                    AUTOTUNE_NUMERICS_POSTCHECK,
                                    new RefineConfig(AUTOTUNE_REFINE_REPEATS, AUTOTUNE_REFINE_WARMUP_ITERS, AUTOTUNE_REFINE_MEASURE_ITERS)
                            )
                    ),
                    candidate -> {
                        String candidateKey = candidateFingerprint(candidate);
                        return Phase1CandidateEvaluator.evaluate(
                                candidate,
                                !AUTOTUNE_SAFETY_STATELESS && !AUTOTUNE_RESCAN_UNSAFE && history.isUnsafe(candidateKey),
                                AUTOTUNE_RESCAN_UNSAFE,
                                AUTOTUNE_ENABLE_SAFETY_PRECHECK,
                                AUTOTUNE_SAFETY_SWEEP_ONLY,
                                OptimizerBenchmarkFramework::resetCandidateRuntimeState,
                                c -> {
                                    CorrectnessCheck safety = checkCandidateCorrectness(
                                            safetyBaseline,
                                            safetyBaselineBroadcast,
                                            safetyA,
                                            safetyB,
                                            safetyC,
                                            safetyBroadcastA,
                                            safetyBroadcastB,
                                            safetyBroadcastC,
                                            c,
                                            AUTOTUNE_GRAPH_BLOCKS,
                                            AUTOTUNE_SAFETY_BROADCAST_B0,
                                            AUTOTUNE_SAFETY_BROADCAST_B1,
                                            AUTOTUNE_SAFETY_BROADCAST_F
                                    );
                                    return new CorrectnessVerdict(safety.ok(), maxAbs(safety));
                                },
                                c -> measureCandidatePerf(
                                        c,
                                        baseA,
                                        baseB,
                                        baseC,
                                        baseBroadcastA,
                                        baseBroadcastB,
                                        baseBroadcastC,
                                        AUTOTUNE_WARMUP_ITERS,
                                        AUTOTUNE_MEASURE_ITERS,
                                        "PHASE1",
                                        null
                                ),
                                (c, perf) -> {
                                    CorrectnessCheck full = checkCandidateCorrectness(
                                            baseline,
                                            baselineBroadcast,
                                            baseA,
                                            baseB,
                                            baseC,
                                            baseBroadcastA,
                                            baseBroadcastB,
                                            baseBroadcastC,
                                            c,
                                            AUTOTUNE_GRAPH_BLOCKS,
                                            AUTOTUNE_BROADCAST_B0,
                                            AUTOTUNE_BROADCAST_B1,
                                            AUTOTUNE_BROADCAST_F
                                    );
                                    return new CorrectnessVerdict(full.ok(), maxAbs(full));
                                }
                        );
                    },
                    step -> {
                        OptimizerCandidate candidate = step.candidate();
                        String candidateKey = candidateFingerprint(candidate);
                        Phase1CandidateResult evaluated = step.result();
                        Phase1Counters counters = step.counters();
                        phase1CountersHolder[0] = counters;
                        switch (evaluated.status()) {
                            case SKIPPED_UNSAFE_HISTORY -> {
                                progress.recordPhase1(
                                        "SKIPPED_UNSAFE_HISTORY",
                                        candidate,
                                        counters.processed(),
                                        counters.valid(),
                                        counters.mismatch(),
                                        counters.skippedUnsafe(),
                                        counters.mismatchSafety(),
                                        counters.mismatchFull(),
                                        null,
                                        null,
                                        step.rowMs(),
                                        Double.NaN,
                                        Double.NaN,
                                        Double.NaN,
                                        -1,
                                        -1
                                );
                                if (AUTOTUNE_SAFETY_SWEEP_ONLY && (counters.processed() % 1000 == 0)) {
                                    System.out.println(GRAY + "Safety sweep progress: " + counters.processed() + "/" + phase1TotalCandidates
                                            + " | safe=" + counters.safetySweepSafe()
                                            + ", unsafe=" + counters.mismatchSafety()
                                            + ", skippedUnsafe=" + counters.skippedUnsafe() + RESET);
                                }
                            }
                            case MISMATCH_SAFETY -> {
                                if (AUTOTUNE_TRACE_CANDIDATES) {
                                    System.out.println(GRAY + "trace idx=" + (counters.processed() - 1)
                                            + ", candidate=" + candidate.name()
                                            + ", ok=false"
                                            + ", maxAbs=" + String.format(Locale.US, "%.3e", evaluated.safetyVerdict().maxAbs())
                                            + RESET);
                                }
                                if (!AUTOTUNE_SAFETY_STATELESS) {
                                    history.markUnsafe(candidateKey, candidate.name(), evaluated.unsafeReason());
                                }
                                progress.recordPhase1(
                                        "MISMATCH_SAFETY",
                                        candidate,
                                        counters.processed(),
                                        counters.valid(),
                                        counters.mismatch(),
                                        counters.skippedUnsafe(),
                                        counters.mismatchSafety(),
                                        counters.mismatchFull(),
                                        null,
                                        null,
                                        step.rowMs(),
                                        Double.NaN,
                                        Double.NaN,
                                        Double.NaN,
                                        -1,
                                        -1
                                );
                                if (AUTOTUNE_SAFETY_SWEEP_ONLY && (counters.processed() % 1000 == 0)) {
                                    if (!AUTOTUNE_SAFETY_STATELESS) {
                                        history.save(historyPath);
                                    }
                                    System.out.println(GRAY + "Safety sweep progress: " + counters.processed() + "/" + phase1TotalCandidates
                                            + " | safe=" + counters.safetySweepSafe()
                                            + ", unsafe=" + counters.mismatchSafety()
                                            + ", skippedUnsafe=" + counters.skippedUnsafe() + RESET);
                                }
                            }
                            case SAFE_SWEEP -> {
                                if (AUTOTUNE_TRACE_CANDIDATES) {
                                    System.out.println(GRAY + "trace idx=" + (counters.processed() - 1)
                                            + ", candidate=" + candidate.name()
                                            + ", ok=true"
                                            + ", maxAbs=" + String.format(Locale.US, "%.3e", evaluated.safetyVerdict().maxAbs())
                                            + RESET);
                                }
                                progress.recordPhase1(
                                        "SAFE_SWEEP",
                                        candidate,
                                        counters.processed(),
                                        counters.valid(),
                                        counters.mismatch(),
                                        counters.skippedUnsafe(),
                                        counters.mismatchSafety(),
                                        counters.mismatchFull(),
                                        null,
                                        null,
                                        step.rowMs(),
                                        Double.NaN,
                                        Double.NaN,
                                        Double.NaN,
                                        -1,
                                        -1
                                );
                                if (counters.processed() % 1000 == 0) {
                                    if (!AUTOTUNE_SAFETY_STATELESS) {
                                        history.save(historyPath);
                                    }
                                    System.out.println(GRAY + "Safety sweep progress: " + counters.processed() + "/" + phase1TotalCandidates
                                            + " | safe=" + counters.safetySweepSafe()
                                            + ", unsafe=" + counters.mismatchSafety()
                                            + ", skippedUnsafe=" + counters.skippedUnsafe() + RESET);
                                }
                            }
                            case MISMATCH_FULL -> {
                                history.markUnsafe(candidateKey, candidate.name(), evaluated.unsafeReason());
                                CandidatePerf perf = evaluated.perf();
                                progress.recordPhase1(
                                        "MISMATCH_FULL",
                                        candidate,
                                        counters.processed(),
                                        counters.valid(),
                                        counters.mismatch(),
                                        counters.skippedUnsafe(),
                                        counters.mismatchSafety(),
                                        counters.mismatchFull(),
                                        null,
                                        null,
                                        step.rowMs(),
                                        perf.forwardMs(),
                                        perf.trainMs(),
                                        perf.broadcastMs(),
                                        perf.graphInfSize(),
                                        perf.graphTrnSize()
                                );
                            }
                            case VALID_PHASE1 -> {
                                CandidatePerf perf = evaluated.perf();
                                progress.recordPhase1(
                                        "VALID_PHASE1",
                                        candidate,
                                        counters.processed(),
                                        counters.valid(),
                                        counters.mismatch(),
                                        counters.skippedUnsafe(),
                                        counters.mismatchSafety(),
                                        counters.mismatchFull(),
                                        null,
                                        null,
                                        step.rowMs(),
                                        perf.forwardMs(),
                                        perf.trainMs(),
                                        perf.broadcastMs(),
                                        perf.graphInfSize(),
                                        perf.graphTrnSize()
                                );
                            }
                        }
                    },
                    finalists -> applyNumericsPostcheck(finalists, history),
                    (candidate, warmupIters, measureIters, tier, cache) -> measureCandidatePerf(
                            candidate,
                            baseA,
                            baseB,
                            baseC,
                            baseBroadcastA,
                            baseBroadcastB,
                            baseBroadcastC,
                            warmupIters,
                            measureIters,
                            tier,
                            null
                    ),
                    prepared -> {
                        if (prepared.status() == FinalistPreparationResult.Status.OK) {
                            Phase1Counters counters = phase1CountersHolder[0];
                            System.out.println(GRAY + "Phase1 valid=" + counters.valid() + ", mismatch=" + counters.mismatch()
                                    + " (safety=" + counters.mismatchSafety() + ", full=" + counters.mismatchFull() + ")"
                                    + ", skippedUnsafe=" + counters.skippedUnsafe()
                                    + ", finalists=" + prepared.finalists().size() + RESET);
                        }
                    },
                    update -> {
                        progress.recordRefine(
                                update.candidate(),
                                update.refinedIndex(),
                                update.finalists(),
                                update.bestTraining(),
                                update.bestInference(),
                                update.rowMs(),
                                update.fwdMs(),
                                update.trainMs(),
                                update.broadcastMs()
                        );
                    },
                    (bestTrainingCandidate, bestInferenceCandidate, validCount, mismatchCount) ->
                            AutoTuneProfilePersistence.persist(
                                    bestTrainingCandidate,
                                    bestInferenceCandidate,
                                    validCount,
                                    mismatchCount,
                                    AUTOTUNE_BEST_TRAINING_PATH,
                                    AUTOTUNE_BEST_INFERENCE_PATH,
                                    AUTOTUNE_BEST_PATH,
                                    PROFILE_PATH,
                                    HW_PROFILE_PATH,
                                    hwBucket,
                                    HW_PROFILE_MAX_BUCKETS
                            ),
                    () -> history.save(historyPath),
                    System::nanoTime
            );
        } catch (IOException e) {
            throw new IllegalStateException("Failed to write autotune profile", e);
        }

        Phase1Counters phase1Counters = session.counters();
        int processedCandidates = phase1Counters.processed();
        int validCount = phase1Counters.valid();
        int mismatchCount = phase1Counters.mismatch();
        int mismatchSafetyCount = phase1Counters.mismatchSafety();
        int mismatchFullCount = phase1Counters.mismatchFull();
        int skippedUnsafeCount = phase1Counters.skippedUnsafe();
        int safetySweepSafeCount = phase1Counters.safetySweepSafe();

        if (session.status() == AutoTuneSessionResult.Status.SAFE_SWEEP_DONE) {
            progress.complete("SAFE_SWEEP_DONE", processedCandidates, validCount, mismatchCount, skippedUnsafeCount, bestTraining, bestInference);
            int checked = candidates.size() - skippedUnsafeCount;
            int unsafe = mismatchSafetyCount;
            int safe = safetySweepSafeCount;
            System.out.println(CYAN + "Safety sweep completed." + RESET);
            System.out.println(GRAY + "Checked=" + checked
                    + ", safe=" + safe
                    + ", unsafe(mismatch)=" + unsafe
                    + ", skippedUnsafe=" + skippedUnsafeCount + RESET);
            if (AUTOTUNE_SAFETY_STATELESS) {
                System.out.println(CYAN + "Safety sweep stateless mode: history update skipped." + RESET);
            } else {
                System.out.println(CYAN + "Unsafe history updated: " + RESET + historyPath.toAbsolutePath());
            }
            System.out.println();
            return;
        }

        if (session.status() == AutoTuneSessionResult.Status.NO_VALID_CANDIDATE) {
            progress.complete("NO_VALID_CANDIDATE", processedCandidates, validCount, mismatchCount, skippedUnsafeCount, bestTraining, bestInference);
            System.out.println(RED + "No valid candidate passed correctness filter." + RESET);
            return;
        }
        if (session.status() == AutoTuneSessionResult.Status.EMPTY_AFTER_POSTCHECK) {
            System.out.println(RED + "No finalist left after numerics post-check." + RESET);
            return;
        }

        bestTraining = session.bestResults().training();
        bestInference = session.bestResults().inference();

        System.out.println(GREEN + "Best (TRAINING): " + bestTraining.candidate().name()
                + " | graph_trn=" + bestTraining.graphTrnSize()
                + " | fwd=" + String.format("%.4f", bestTraining.forwardMs()) + " ms"
                + " | train=" + String.format("%.4f", bestTraining.trainMs()) + " ms"
                + " | bcast=" + String.format("%.4f", bestTraining.broadcastMs()) + " ms"
                + " | score=" + String.format("%.4f", bestTraining.score()) + RESET);
        System.out.println(GREEN + "Best (INFERENCE): " + bestInference.candidate().name()
                + " | graph_inf=" + bestInference.graphInfSize()
                + " | fwd=" + String.format("%.4f", bestInference.forwardMs()) + " ms"
                + " | bcast=" + String.format("%.4f", bestInference.broadcastMs()) + " ms"
                + " | score=" + String.format("%.4f", bestInference.score()) + RESET);
        System.out.println(GRAY + "Valid=" + validCount + ", mismatch=" + mismatchCount
                + " (safety=" + mismatchSafetyCount + ", full=" + mismatchFullCount + ")" + RESET);
        AutoTuneProfilePersistenceResult persisted = session.persistenceResult();
        if (persisted.trainingImproved()) {
            System.out.println(CYAN + "Saved improved training profile: " + RESET + AUTOTUNE_BEST_TRAINING_PATH.toAbsolutePath());
            System.out.println(CYAN + "Updated runtime profile (training): " + RESET + PROFILE_PATH.toAbsolutePath());
        } else {
            System.out.println(GRAY + "Training profile kept (existing score="
                    + String.format("%.6f", persisted.previousTrainingScore())
                    + " <= new score="
                    + String.format("%.6f", bestTraining.score()) + ")." + RESET);
        }

        if (persisted.inferenceImproved()) {
            System.out.println(CYAN + "Saved improved inference profile: " + RESET + AUTOTUNE_BEST_INFERENCE_PATH.toAbsolutePath());
        } else {
            System.out.println(GRAY + "Inference profile kept (existing score="
                    + String.format("%.6f", persisted.previousInferenceScore())
                    + " <= new score="
                    + String.format("%.6f", bestInference.score()) + ")." + RESET);
        }

        if (persisted.hwTrainingImproved() || persisted.hwInferenceImproved()) {
            System.out.println(CYAN + "Updated HW profiles: " + RESET + HW_PROFILE_PATH.toAbsolutePath()
                    + " (bucket=" + hwBucket + ")");
        } else {
            System.out.println(GRAY + "HW profiles kept (no score improvement for bucket " + hwBucket + ")." + RESET);
        }
        progress.complete("DONE", processedCandidates, validCount, mismatchCount, skippedUnsafeCount, bestTraining, bestInference);
        System.out.println();
    }

    private static Path resolveAutotuneHistoryPath() {
        String raw = System.getProperty("benchmark.autotuneHistoryPath", "").trim();
        if (raw.isEmpty()) {
            return AUTOTUNE_HISTORY_PATH;
        }
        return Path.of(raw);
    }

    private static Path resolveAutotuneProgressPath() {
        String raw = System.getProperty("benchmark.autotuneProgressPath", "").trim();
        if (raw.isEmpty()) {
            return AUTOTUNE_PROGRESS_PATH;
        }
        return Path.of(raw);
    }

    private static Path resolveAutotuneProgressRowsPath() {
        String raw = System.getProperty("benchmark.autotuneProgressRowsPath", "").trim();
        if (raw.isEmpty()) {
            return AUTOTUNE_PROGRESS_ROWS_PATH;
        }
        return Path.of(raw);
    }

    private static List<OptimizerCandidate> sliceCandidates(List<OptimizerCandidate> candidates, int start, int count) {
        if (candidates.isEmpty()) {
            return candidates;
        }
        if (start <= 0 && count >= candidates.size()) {
            return candidates;
        }
        if (start >= candidates.size()) {
            return List.of();
        }
        int toIndex = Math.min(candidates.size(), start + Math.max(1, count));
        return new ArrayList<>(candidates.subList(start, toIndex));
    }

    private static double elapsedRowMs(long rowStartNs) {
        return (System.nanoTime() - rowStartNs) / 1_000_000.0;
    }

    private static List<OptimizerCandidate> reduceCandidatesViaGraphScout(
            List<OptimizerCandidate> candidates,
            double[] baseA,
            double[] baseB,
            double[] baseC,
            double[] baseBroadcastA,
            double[] baseBroadcastB,
            double[] baseBroadcastC
    ) {
        if (candidates.isEmpty()) {
            return candidates;
        }

        CandidateEvalCache evalCache = new CandidateEvalCache(
                BENCH_DTYPE,
                AUTOTUNE_GRAPH_BLOCKS,
                AUTOTUNE_BROADCAST_B0,
                AUTOTUNE_BROADCAST_B1,
                AUTOTUNE_BROADCAST_F,
                baseA.length,
                baseBroadcastA.length,
                baseBroadcastB.length,
                baseBroadcastC.length,
                OptimizerBenchmarkFramework::candidateFingerprint
        );
        return GraphScoutReducer.reduceCandidates(
                candidates,
                new GraphScoutConfig(
                        AUTOTUNE_STAGE_SCOUT_SAMPLE_PER_STAGE,
                        AUTOTUNE_STAGE_SCOUT_MAX_SAMPLES_PER_STAGE,
                        AUTOTUNE_STAGE_SCOUT_MAX_ROUNDS,
                        AUTOTUNE_STAGE_SCOUT_MIN_ACTIVE_FAMILIES,
                        AUTOTUNE_STAGE_SCOUT_WARMUP_ITERS,
                        AUTOTUNE_STAGE_SCOUT_MEASURE_ITERS,
                        AUTOTUNE_STAGE_SCOUT_TOP_TRAIN,
                        AUTOTUNE_STAGE_SCOUT_TOP_INF,
                        AUTOTUNE_PRESCREEN_KEEP_TRAIN,
                        AUTOTUNE_PRESCREEN_KEEP_INF,
                        AUTOTUNE_PRESCREEN_DIVERSITY_SEEDS_PER_FAMILY,
                        AUTOTUNE_PRESCREEN_MAX_PER_STAGE_ORDER,
                        AUTOTUNE_PRESCREEN_WARMUP_ITERS,
                        AUTOTUNE_PRESCREEN_MEASURE_ITERS,
                        AUTOTUNE_PROGRESS_LOG_EVERY,
                        AUTOTUNE_PROGRESS_MIN_INTERVAL_MS,
                        AUTOTUNE_STAGE_SCOUT_CONFIDENCE_Z,
                        new BeamSearchConfig(
                                AUTOTUNE_BEAM_ROUNDS,
                                AUTOTUNE_BEAM_SEED_TRAIN,
                                AUTOTUNE_BEAM_SEED_INF,
                                AUTOTUNE_BEAM_WIDTH_TRAIN,
                                AUTOTUNE_BEAM_WIDTH_INF,
                                AUTOTUNE_BEAM_KEEP_TRAIN,
                                AUTOTUNE_BEAM_KEEP_INF,
                                AUTOTUNE_BEAM_MAX_PER_STAGE
                        )
                ),
                evalCache,
                (candidate, warmupIters, measureIters, tier, cache) -> measureCandidatePerf(
                        candidate,
                        baseA,
                        baseB,
                        baseC,
                        baseBroadcastA,
                        baseBroadcastB,
                        baseBroadcastC,
                        warmupIters,
                        measureIters,
                        tier,
                        cache
                ),
                OptimizerBenchmarkFramework::candidateFingerprint,
                msg -> System.out.println(GRAY + msg + RESET),
                System::nanoTime
        );
    }

    private static List<OptimizerCandidate> selectCandidatesViaBeam(
            List<CandidatePerf> prescreen,
            Set<OptimizerCandidate> seedCandidates
    ) {
        return AutotuneSearchSupport.selectCandidatesViaBeam(
                prescreen,
                seedCandidates,
                new BeamSearchConfig(
                        AUTOTUNE_BEAM_ROUNDS,
                        AUTOTUNE_BEAM_SEED_TRAIN,
                        AUTOTUNE_BEAM_SEED_INF,
                        AUTOTUNE_BEAM_WIDTH_TRAIN,
                        AUTOTUNE_BEAM_WIDTH_INF,
                        AUTOTUNE_BEAM_KEEP_TRAIN,
                        AUTOTUNE_BEAM_KEEP_INF,
                        AUTOTUNE_BEAM_MAX_PER_STAGE
                ),
                AutotuneSearchSupport::stageOrderNeighbors,
                OptimizerBenchmarkFramework::candidateFingerprint,
                msg -> System.out.println(GRAY + msg + RESET)
        );
    }

    private static String stageOrderKey(OptimizerCandidate candidate) {
        return AutotuneSearchSupport.stageOrderKey(candidate);
    }

    private static List<String> stageOrderNeighbors(String stageOrderKey) {
        return AutotuneSearchSupport.stageOrderNeighbors(stageOrderKey);
    }

    private static CandidatePerf measureCandidatePerf(
            OptimizerCandidate candidate,
            double[] baseA,
            double[] baseB,
            double[] baseC,
            double[] baseBroadcastA,
            double[] baseBroadcastB,
            double[] baseBroadcastC,
            int warmupIters,
            int measureIters,
            String tier,
            CandidateEvalCache cache
    ) {
        CandidateMeasurementResult measured = CANDIDATE_MEASUREMENT_HARNESS.measure(
                candidate,
                baseA,
                baseB,
                baseC,
                baseBroadcastA,
                baseBroadcastB,
                baseBroadcastC,
                warmupIters,
                measureIters,
                tier,
                cache
        );
        return new CandidatePerf(
                measured.candidate(),
                stageOrderKey(measured.candidate()),
                CoarseKnobSignature.of(measured.candidate()),
                measured.graphInfSize(),
                measured.graphTrnSize(),
                measured.forwardMs(),
                measured.trainMs(),
                measured.broadcastMs()
        );
    }

    private static double scoreCandidate(
            double forwardMs,
            double trainMs,
            double broadcastMs,
            int graphInfSize,
            int graphTrnSize,
            TuneObjective objective
    ) {
        return MeasurementScoring.score(
                forwardMs,
                trainMs,
                broadcastMs,
                graphInfSize,
                graphTrnSize,
                objective == TuneObjective.INFERENCE ? MeasurementObjective.INFERENCE : MeasurementObjective.TRAINING
        );
    }

    private static MeasuredBenchmarkScenario createMeasuredBenchmarkScenario(
            double[] baseA,
            double[] baseB,
            double[] baseC,
            OptimizerCandidate candidate,
            boolean requiresGrad,
            int graphBlocks
    ) {
        PreparedBenchmarkScenario scenario = newBenchState(baseA, baseB, baseC, candidate, requiresGrad, graphBlocks);
        return new MeasuredBenchmarkScenario() {
            @Override
            public int graphSize() {
                return scenario.compiledGraph().getCompiledGraphAsList().size();
            }

            @Override
            public void setTrainingMode(boolean trainingMode) {
                scenario.setTrainingMode(trainingMode);
            }

            @Override
            public void compute() {
                scenario.compute();
            }
        };
    }

    private static MeasuredBroadcastScenario createMeasuredBroadcastScenario(
            double[] baseA,
            double[] baseB,
            double[] baseC,
            OptimizerCandidate candidate
    ) {
        PreparedBroadcastScenario scenario = newBroadcastBenchState(baseA, baseB, baseC, candidate);
        return scenario::compute;
    }

    private static List<OptimizerCandidate> applyNumericsPostcheck(
            List<OptimizerCandidate> finalistsList,
            UnsafeCandidateHistory history
    ) {
        NumericsHarness.Config cfg = new NumericsHarness.Config();
        cfg.dtype = BENCH_DTYPE;
        cfg.size = AUTOTUNE_SIZE;
        cfg.graphBlocks = AUTOTUNE_GRAPH_BLOCKS;
        cfg.b0 = AUTOTUNE_BROADCAST_B0;
        cfg.b1 = AUTOTUNE_BROADCAST_B1;
        cfg.f = AUTOTUNE_BROADCAST_F;
        cfg.seed = AUTOTUNE_NUMERICS_POSTCHECK_SEED;

        NumericsHarness harness = new NumericsHarness(cfg);
        NumericsPolicy policy = NumericsPolicy.defaultsFor(BENCH_DTYPE);
        NumericsPostcheckResult result = NumericsPostcheckRunner.run(
                finalistsList,
                new NumericsPostcheckConfig(
                        BENCH_DTYPE,
                        AUTOTUNE_NUMERICS_POSTCHECK_TOP_N,
                        AUTOTUNE_NUMERICS_REPORT_DIR,
                        AUTOTUNE_NUMERICS_TS_FORMAT
                ),
                (baselineNoOptSameKnobs, finalist) -> harness.run(baselineNoOptSameKnobs, finalist, policy),
                history,
                OptimizerBenchmarkFramework::candidateFingerprint
        );
        for (var dropped : result.droppedUnsafe()) {
            System.out.println(YELLOW + "Numerics post-check filtered finalist=" + dropped.candidateName()
                    + " | reason=" + dropped.reason() + RESET);
        }
        System.out.println(GRAY + "Numerics post-check enabled: checked=" + result.checked()
                + ", kept=" + result.keptCandidates().size()
                + ", dropped=" + result.droppedCount()
                + ", markedUnsafe=" + result.markedUnsafe() + RESET);
        if (result.reportPath() != null) {
            System.out.println(CYAN + "Numerics post-check report: " + RESET + result.reportPath().toAbsolutePath());
        }
        return result.keptCandidates();
    }

    private static void resetCandidateRuntimeState() {
        if (!AUTOTUNE_RESET_CANDIDATE_RUNTIME) {
            return;
        }
        CpuSchedulerAdvisor.reset();
    }

    private static String sanitizeTsv(String value) {
        if (value == null) {
            return "";
        }
        return value.replace('\t', ' ')
                .replace('\n', ' ')
                .replace('\r', ' ');
    }

    private static String fmtDouble(double v) {
        return String.format(Locale.US, "%.12e", v);
    }

    private static List<OptimizerCandidate> capCandidatesDeterministic(List<OptimizerCandidate> all, int maxCount) {
        if (all.size() <= maxCount) return all;
        List<OptimizerCandidate> out = new java.util.ArrayList<>(maxCount);
        double step = (double) all.size() / maxCount;
        double pos = 0.0;
        for (int i = 0; i < maxCount; i++) {
            out.add(all.get((int) pos));
            pos += step;
        }
        return out;
    }

    private static void runDebugCandidateSequence(
            List<OptimizerCandidate> all,
            RunResult safetyBaseline,
            BroadcastRunResult safetyBaselineBroadcast,
            double[] safetyA,
            double[] safetyB,
            double[] safetyC,
            double[] safetyBroadcastA,
            double[] safetyBroadcastB,
            double[] safetyBroadcastC
    ) {
        List<Integer> indices = new ArrayList<>();
        for (String raw : AUTOTUNE_DEBUG_CANDIDATE_INDICES.split(",")) {
            String token = raw.trim();
            if (token.isEmpty()) {
                continue;
            }
            indices.add(Integer.parseInt(token));
        }
        long baselineHashBefore = arrayHash(safetyBaseline.ta7) ^ arrayHash(safetyBaseline.gradA)
                ^ arrayHash(safetyBaseline.gradB) ^ arrayHash(safetyBaseline.gradC)
                ^ arrayHash(safetyBaselineBroadcast.out);
        long safetyInputsHashBefore = arrayHash(safetyA) ^ arrayHash(safetyB) ^ arrayHash(safetyC)
                ^ arrayHash(safetyBroadcastA) ^ arrayHash(safetyBroadcastB) ^ arrayHash(safetyBroadcastC);
        UnsafeCandidateHistory localHistory = UnsafeCandidateHistory.empty("debug-sequence");
        System.out.println(CYAN + "Debug candidate sequence: " + indices + RESET);
        System.out.println(GRAY + "baselineHashBefore=" + baselineHashBefore
                + ", safetyInputsHashBefore=" + safetyInputsHashBefore + RESET);
        for (int index : indices) {
            OptimizerCandidate candidate = all.get(index);
            String candidateKey = candidateFingerprint(candidate);
            CorrectnessCheck safety = checkCandidateCorrectness(
                    safetyBaseline,
                    safetyBaselineBroadcast,
                    safetyA,
                    safetyB,
                    safetyC,
                    safetyBroadcastA,
                    safetyBroadcastB,
                    safetyBroadcastC,
                    candidate,
                    AUTOTUNE_GRAPH_BLOCKS,
                    AUTOTUNE_SAFETY_BROADCAST_B0,
                    AUTOTUNE_SAFETY_BROADCAST_B1,
                    AUTOTUNE_SAFETY_BROADCAST_F
            );
            if (!safety.ok()) {
                localHistory.markUnsafe(candidateKey, candidate.name(), "DEBUG_MISMATCH");
            }
            long baselineHashAfter = arrayHash(safetyBaseline.ta7) ^ arrayHash(safetyBaseline.gradA)
                    ^ arrayHash(safetyBaseline.gradB) ^ arrayHash(safetyBaseline.gradC)
                    ^ arrayHash(safetyBaselineBroadcast.out);
            long safetyInputsHashAfter = arrayHash(safetyA) ^ arrayHash(safetyB) ^ arrayHash(safetyC)
                    ^ arrayHash(safetyBroadcastA) ^ arrayHash(safetyBroadcastB) ^ arrayHash(safetyBroadcastC);
            System.out.println(GRAY + "idx=" + index
                    + ", candidate=" + candidate.name()
                    + ", ok=" + safety.ok()
                    + ", maxAbs=" + String.format(Locale.US, "%.3e", maxAbs(safety))
                    + ", out=" + String.format(Locale.US, "%.3e", safety.out.maxAbs)
                    + ", outIdx=" + safety.out.argMaxIndex
                    + ", outBase=" + String.format(Locale.US, "%.6f", safety.out.leftAtMax)
                    + ", outCand=" + String.format(Locale.US, "%.6f", safety.out.rightAtMax)
                    + ", gradA=" + String.format(Locale.US, "%.3e", safety.gradA.maxAbs)
                    + ", gradB=" + String.format(Locale.US, "%.3e", safety.gradB.maxAbs)
                    + ", gradC=" + String.format(Locale.US, "%.3e", safety.gradC.maxAbs)
                    + ", bcast=" + String.format(Locale.US, "%.3e", safety.broadcast.maxAbs)
                    + ", baselineHashAfter=" + baselineHashAfter
                    + ", safetyInputsHashAfter=" + safetyInputsHashAfter + RESET);
        }
        System.out.println();
    }

    private static long arrayHash(double[] values) {
        long acc = 1125899906842597L;
        for (double value : values) {
            acc = (acc * 31L) ^ Double.doubleToLongBits(value);
        }
        return acc;
    }

    private static void runScalarSanityCheck(GraphOptimizer optimizer) {
        runScalarSanityCore(optimizer);
        System.out.println(BOLD + CYAN + "[Scalar Sanity]" + RESET);
        System.out.println(GREEN + "OK: Te7 + grad(A,B,C) for no-opt and opt match expected scalar values." + RESET);
        System.out.println();
    }

    private static void runScalarSanityCore(GraphOptimizer optimizer) {
        Tensor A0 = Tensor.scalar(10.0, BENCH_DTYPE);
        Tensor B0 = Tensor.scalar(2.0, BENCH_DTYPE);
        Tensor C0 = Tensor.scalar(5.0, BENCH_DTYPE);
        A0.setRequiresGrad(true);
        B0.setRequiresGrad(true);
        C0.setRequiresGrad(true);
        Tensor Te7No = buildTa7(A0, B0, C0);
        CompiledGraph.compile(Te7No, config.optimizer.OptimizerConfig.noOptimization())
                .execute(config.runtime.RuntimeConfig.trainingDefaults(), backend.runtime.ExecutionMode.FORWARD_BACKWARD);

        Tensor A1 = Tensor.scalar(10.0, BENCH_DTYPE);
        Tensor B1 = Tensor.scalar(2.0, BENCH_DTYPE);
        Tensor C1 = Tensor.scalar(5.0, BENCH_DTYPE);
        A1.setRequiresGrad(true);
        B1.setRequiresGrad(true);
        C1.setRequiresGrad(true);
        Tensor Te7Opt = buildTa7(A1, B1, C1);
        CompiledGraph.compile(Te7Opt, optimizer)
                .execute(config.runtime.RuntimeConfig.trainingDefaults(), backend.runtime.ExecutionMode.FORWARD_BACKWARD);

        double expectedTe7 = 64.0;
        double expectedGradA = -12.8;
        double expectedGradB = -48.0;
        double expectedGradC = 41.6;

        checkClose("Scalar no-opt Te7", Te7No.toDoubleArrayCopy()[0], expectedTe7, 1e-5);
        checkClose("Scalar no-opt gradA", A0.getGradient().toDoubleArrayCopy()[0], expectedGradA, 1e-5);
        checkClose("Scalar no-opt gradB", B0.getGradient().toDoubleArrayCopy()[0], expectedGradB, 1e-5);
        checkClose("Scalar no-opt gradC", C0.getGradient().toDoubleArrayCopy()[0], expectedGradC, 1e-5);

        checkClose("Scalar opt Te7", Te7Opt.toDoubleArrayCopy()[0], expectedTe7, 1e-5);
        checkClose("Scalar opt gradA", A1.getGradient().toDoubleArrayCopy()[0], expectedGradA, 1e-5);
        checkClose("Scalar opt gradB", B1.getGradient().toDoubleArrayCopy()[0], expectedGradB, 1e-5);
        checkClose("Scalar opt gradC", C1.getGradient().toDoubleArrayCopy()[0], expectedGradC, 1e-5);
    }

    private static void checkClose(String name, double actual, double expected, double tol) {
        if (!Double.isFinite(actual)) {
            throw new IllegalStateException(name + " is not finite: " + actual);
        }
        double d = Math.abs(actual - expected);
        if (d > tol) {
            throw new IllegalStateException(name + " mismatch. actual=" + actual + ", expected=" + expected + ", diff=" + d);
        }
    }

    private static RunResult runFresh(
            double[] baseA,
            double[] baseB,
            double[] baseC,
            OptimizerCandidate candidate,
            int graphBlocks
    ) {
        PreparedBenchmarkScenario s = newBenchState(baseA, baseB, baseC, candidate, true, graphBlocks);
        s.compute();
        return new RunResult(
                s.ta7.toDoubleArrayCopy().clone(),
                s.A.getGradient().toDoubleArrayCopy().clone(),
                s.B.getGradient().toDoubleArrayCopy().clone(),
                s.C.getGradient().toDoubleArrayCopy().clone()
        );
    }

    private static BroadcastRunResult runBroadcastFresh(
            double[] baseA,
            double[] baseB,
            double[] baseC,
            OptimizerCandidate candidate
    ) {
        PreparedBroadcastScenario s = newBroadcastBenchState(
                baseA,
                baseB,
                baseC,
                candidate,
                AUTOTUNE_BROADCAST_B0,
                AUTOTUNE_BROADCAST_B1,
                AUTOTUNE_BROADCAST_F
        );
        s.compute();
        return new BroadcastRunResult(s.out.toDoubleArrayCopy().clone());
    }

    private static BroadcastRunResult runBroadcastFresh(
            double[] baseA,
            double[] baseB,
            double[] baseC,
            OptimizerCandidate candidate,
            int b0,
            int b1,
            int f
    ) {
        PreparedBroadcastScenario s = newBroadcastBenchState(baseA, baseB, baseC, candidate, b0, b1, f);
        s.compute();
        return new BroadcastRunResult(s.out.toDoubleArrayCopy().clone());
    }

    private static PreparedBenchmarkScenario newBenchState(
            double[] baseA,
            double[] baseB,
            double[] baseC,
            OptimizerCandidate candidate,
            boolean requiresGrad,
            int graphBlocks
    ) {
        return BenchmarkScenarioFactory.createOptimizerBenchmarkScenario(
                baseA,
                baseB,
                baseC,
                candidate,
                BENCH_DTYPE,
                requiresGrad,
                graphBlocks,
                BENCH_LINEAR_SHAPE
        );
    }

    private static PreparedBroadcastScenario newBroadcastBenchState(
            double[] baseA,
            double[] baseB,
            double[] baseC,
            OptimizerCandidate candidate
    ) {
        return newBroadcastBenchState(
                baseA,
                baseB,
                baseC,
                candidate,
                AUTOTUNE_BROADCAST_B0,
                AUTOTUNE_BROADCAST_B1,
                AUTOTUNE_BROADCAST_F
        );
    }

    private static PreparedBroadcastScenario newBroadcastBenchState(
            double[] baseA,
            double[] baseB,
            double[] baseC,
            OptimizerCandidate candidate,
            int b0,
            int b1,
            int f
    ) {
        return BenchmarkScenarioFactory.createBroadcastScenario(baseA, baseB, baseC, candidate, BENCH_DTYPE, b0, b1, f);
    }

    private static CorrectnessCheck checkCandidateCorrectness(
            RunResult baseline,
            BroadcastRunResult baselineBroadcast,
            double[] baseA,
            double[] baseB,
            double[] baseC,
            double[] baseBroadcastA,
            double[] baseBroadcastB,
            double[] baseBroadcastC,
            OptimizerCandidate candidate,
            int graphBlocks,
            int b0,
            int b1,
            int f
    ) {
        RunResult rr = runFresh(baseA, baseB, baseC, candidate, graphBlocks);
        Diff dOut = diff(baseline.ta7, rr.ta7);
        Diff dA = diff(baseline.gradA, rr.gradA);
        Diff dB = diff(baseline.gradB, rr.gradB);
        Diff dC = diff(baseline.gradC, rr.gradC);
        BroadcastRunResult br = runBroadcastFresh(baseBroadcastA, baseBroadcastB, baseBroadcastC, candidate, b0, b1, f);
        Diff dBroadcast = diff(baselineBroadcast.out, br.out);
        return new CorrectnessCheck(dOut, dA, dB, dC, dBroadcast);
    }

    private static void runMemOnlyReplayForUnsafeCandidates(
            List<OptimizerCandidate> allCandidates,
            UnsafeCandidateHistory history,
            RunResult baselineFull,
            BroadcastRunResult baselineBroadcastFull,
            RunResult baselineSafety,
            BroadcastRunResult baselineBroadcastSafety,
            double[] baseA,
            double[] baseB,
            double[] baseC,
            double[] baseBroadcastA,
            double[] baseBroadcastB,
            double[] baseBroadcastC,
            double[] safetyA,
            double[] safetyB,
            double[] safetyC,
            double[] safetyBroadcastA,
            double[] safetyBroadcastB,
            double[] safetyBroadcastC
    ) {
        List<OptimizerCandidate> unsafeCandidates = new ArrayList<>();
        for (OptimizerCandidate c : allCandidates) {
            if (history.isUnsafe(candidateFingerprint(c))) {
                unsafeCandidates.add(c);
            }
        }

        int totalUnsafe = unsafeCandidates.size();
        int limit = Math.max(1, AUTOTUNE_MEM_ONLY_REPLAY_LIMIT);
        int toCheck = Math.min(totalUnsafe, limit);
        int okBoth = 0;
        int failedSafety = 0;
        int failedFull = 0;
        int failedAny = 0;
        int printed = 0;
        List<Double> failSafetyAbs = new ArrayList<>();
        List<Double> failFullAbs = new ArrayList<>();
        List<MemReplayFailure> failures = new ArrayList<>();

        System.out.println(CYAN + "Mem-only replay over UNSAFE history candidates" + RESET);
        System.out.println(GRAY + "Unsafe in history=" + totalUnsafe + ", replayLimit=" + limit + ", checking=" + toCheck + RESET);
        OptimizationStage replayStage = resolveReplayStage();
        if (replayStage == null && !"NONE".equals(AUTOTUNE_REPLAY_STAGE)) {
            System.out.println(RED + "Invalid benchmark.autotuneReplayStage=" + AUTOTUNE_REPLAY_STAGE
                    + " (allowed: MEM, FUSE, AR, CSE, NONE)" + RESET);
            return;
        }
        if (replayStage == null) {
            System.out.println(GRAY + "Replay stage override=[NONE]" + RESET);
        } else {
            System.out.println(GRAY + "Replay stage override=[" + replayStage + "]" + RESET);
        }

        for (int i = 0; i < toCheck; i++) {
            OptimizerCandidate original = unsafeCandidates.get(i);
            OptimizerCandidate memOnly = new OptimizerCandidate(
                    "MEM_REPLAY_" + i,
                    replayStage == null ? List.of() : List.of(replayStage),
                    original.knobs()
            );

            CorrectnessCheck safety = checkCandidateCorrectness(
                    baselineSafety,
                    baselineBroadcastSafety,
                    safetyA,
                    safetyB,
                    safetyC,
                    safetyBroadcastA,
                    safetyBroadcastB,
                    safetyBroadcastC,
                    memOnly,
                    AUTOTUNE_GRAPH_BLOCKS,
                    AUTOTUNE_SAFETY_BROADCAST_B0,
                    AUTOTUNE_SAFETY_BROADCAST_B1,
                    AUTOTUNE_SAFETY_BROADCAST_F
            );
            CorrectnessCheck full = checkCandidateCorrectness(
                    baselineFull,
                    baselineBroadcastFull,
                    baseA,
                    baseB,
                    baseC,
                    baseBroadcastA,
                    baseBroadcastB,
                    baseBroadcastC,
                    memOnly,
                    AUTOTUNE_GRAPH_BLOCKS,
                    AUTOTUNE_BROADCAST_B0,
                    AUTOTUNE_BROADCAST_B1,
                    AUTOTUNE_BROADCAST_F
            );

            boolean safetyOk = safety.ok();
            boolean fullOk = full.ok();
            if (safetyOk && fullOk) {
                okBoth++;
                continue;
            }

            failedAny++;
            if (!safetyOk) {
                failedSafety++;
                failSafetyAbs.add(maxAbs(safety));
            }
            if (!fullOk) {
                failedFull++;
                failFullAbs.add(maxAbs(full));
            }
            failures.add(new MemReplayFailure(original.name(), safetyOk, fullOk, maxAbs(safety), maxAbs(full)));

            if (printed < 20) {
                printed++;
                double maxSafety = maxAbs(safety);
                double maxFull = maxAbs(full);
                System.out.println(RED + "MEM replay fail"
                        + " | original=" + original.name()
                        + " | safetyOk=" + safetyOk
                        + " | fullOk=" + fullOk
                        + " | maxAbsSafety=" + String.format(Locale.US, "%.3e", maxSafety)
                        + " | maxAbsFull=" + String.format(Locale.US, "%.3e", maxFull)
                        + RESET);
            }
        }

        System.out.println(CYAN + "Mem-only replay completed." + RESET);
        System.out.println(GRAY + "Checked=" + toCheck
                + ", okBoth=" + okBoth
                + ", failedAny=" + failedAny
                + ", failedSafety=" + failedSafety
                + ", failedFull=" + failedFull + RESET);
        if (!failSafetyAbs.isEmpty()) {
            System.out.println(GRAY + "Safety maxAbs stats"
                    + " | min=" + String.format(Locale.US, "%.3e", min(failSafetyAbs))
                    + ", p50=" + String.format(Locale.US, "%.3e", percentile(failSafetyAbs, 0.50))
                    + ", p95=" + String.format(Locale.US, "%.3e", percentile(failSafetyAbs, 0.95))
                    + ", max=" + String.format(Locale.US, "%.3e", max(failSafetyAbs))
                    + RESET);
        }
        if (!failFullAbs.isEmpty()) {
            System.out.println(GRAY + "Full maxAbs stats"
                    + " | min=" + String.format(Locale.US, "%.3e", min(failFullAbs))
                    + ", p50=" + String.format(Locale.US, "%.3e", percentile(failFullAbs, 0.50))
                    + ", p95=" + String.format(Locale.US, "%.3e", percentile(failFullAbs, 0.95))
                    + ", max=" + String.format(Locale.US, "%.3e", max(failFullAbs))
                    + RESET);
        }
        if (!failures.isEmpty()) {
            failures.sort((a, b) -> Double.compare(b.maxAbsFull, a.maxAbsFull));
            int n = Math.min(10, failures.size());
            System.out.println(GRAY + "Top MEM replay failures by full maxAbs:" + RESET);
            for (int i = 0; i < n; i++) {
                MemReplayFailure f = failures.get(i);
                System.out.println(GRAY + "  " + (i + 1) + ". " + f.originalName
                        + " | safetyOk=" + f.safetyOk
                        + ", fullOk=" + f.fullOk
                        + ", maxAbsSafety=" + String.format(Locale.US, "%.3e", f.maxAbsSafety)
                        + ", maxAbsFull=" + String.format(Locale.US, "%.3e", f.maxAbsFull)
                        + RESET);
            }
        }
        if (toCheck < totalUnsafe) {
            System.out.println(YELLOW + "Replay truncated by benchmark.autotuneMemOnlyReplayLimit." + RESET);
        }
        System.out.println();
    }

    private static double maxAbs(CorrectnessCheck c) {
        return Math.max(
                Math.max(c.out.maxAbs, c.gradA.maxAbs),
                Math.max(Math.max(c.gradB.maxAbs, c.gradC.maxAbs), c.broadcast.maxAbs)
        );
    }

    private static double min(List<Double> values) {
        double m = Double.POSITIVE_INFINITY;
        for (double v : values) {
            m = Math.min(m, v);
        }
        return m;
    }

    private static double max(List<Double> values) {
        double m = Double.NEGATIVE_INFINITY;
        for (double v : values) {
            m = Math.max(m, v);
        }
        return m;
    }

    private static double percentile(List<Double> values, double p) {
        if (values.isEmpty()) {
            return Double.NaN;
        }
        List<Double> sorted = new ArrayList<>(values);
        Collections.sort(sorted);
        int idx = (int) Math.floor((sorted.size() - 1) * p);
        idx = Math.max(0, Math.min(sorted.size() - 1, idx));
        return sorted.get(idx);
    }

    private static void printCandidatesByName(List<OptimizerCandidate> all, String csvNames) {
        Set<String> requested = new LinkedHashSet<>();
        for (String raw : csvNames.split(",")) {
            String name = raw.trim();
            if (!name.isEmpty()) {
                requested.add(name);
            }
        }
        if (requested.isEmpty()) {
            System.out.println(YELLOW + "No candidate names provided in benchmark.autotunePrintCandidates." + RESET);
            return;
        }
        Map<String, OptimizerCandidate> byName = new HashMap<>();
        for (OptimizerCandidate c : all) {
            byName.put(c.name(), c);
        }
        System.out.println(CYAN + "Auto-tune candidate debug dump" + RESET);
        for (String name : requested) {
            OptimizerCandidate c = byName.get(name);
            if (c == null) {
                System.out.println(RED + "Missing candidate: " + name + RESET);
                continue;
            }
            System.out.println(GRAY + "candidate=" + c.name() + RESET);
            System.out.println(GRAY + "  stageOrder=" + c.stageOrder() + RESET);
            System.out.println(GRAY + "  " + candidateCanonicalSpec(c) + RESET);
        }
        System.out.println();
    }

    private static void compareNoOptVsMemForCandidates(
            List<OptimizerCandidate> all,
            String csvNames,
            RunResult baselineFull,
            BroadcastRunResult baselineBroadcastFull,
            RunResult baselineSafety,
            BroadcastRunResult baselineBroadcastSafety,
            double[] baseA,
            double[] baseB,
            double[] baseC,
            double[] baseBroadcastA,
            double[] baseBroadcastB,
            double[] baseBroadcastC,
            double[] safetyA,
            double[] safetyB,
            double[] safetyC,
            double[] safetyBroadcastA,
            double[] safetyBroadcastB,
            double[] safetyBroadcastC
    ) {
        Set<String> requested = new LinkedHashSet<>();
        for (String raw : csvNames.split(",")) {
            String name = raw.trim();
            if (!name.isEmpty()) {
                requested.add(name);
            }
        }
        if (requested.isEmpty()) {
            System.out.println(YELLOW + "No candidate names provided in benchmark.autotuneCompareNoOptMemCandidates." + RESET);
            return;
        }
        Map<String, OptimizerCandidate> byName = new HashMap<>();
        for (OptimizerCandidate c : all) {
            byName.put(c.name(), c);
        }
        System.out.println(CYAN + "Compare NO_OPT(same knobs) vs MEM_ONLY(same knobs)" + RESET);
        for (String name : requested) {
            OptimizerCandidate c = byName.get(name);
            if (c == null) {
                System.out.println(RED + "Missing candidate: " + name + RESET);
                continue;
            }
            OptimizerCandidate noOptSameKnobs = new OptimizerCandidate(name + "_NOOPT", List.of(), c.knobs());
            OptimizerCandidate memOnlySameKnobs = new OptimizerCandidate(name + "_MEM", List.of(OptimizationStage.MEM), c.knobs());

            CorrectnessCheck noOptSafety = checkCandidateCorrectness(
                    baselineSafety, baselineBroadcastSafety,
                    safetyA, safetyB, safetyC, safetyBroadcastA, safetyBroadcastB, safetyBroadcastC,
                    noOptSameKnobs, AUTOTUNE_GRAPH_BLOCKS,
                    AUTOTUNE_SAFETY_BROADCAST_B0, AUTOTUNE_SAFETY_BROADCAST_B1, AUTOTUNE_SAFETY_BROADCAST_F
            );
            CorrectnessCheck memSafety = checkCandidateCorrectness(
                    baselineSafety, baselineBroadcastSafety,
                    safetyA, safetyB, safetyC, safetyBroadcastA, safetyBroadcastB, safetyBroadcastC,
                    memOnlySameKnobs, AUTOTUNE_GRAPH_BLOCKS,
                    AUTOTUNE_SAFETY_BROADCAST_B0, AUTOTUNE_SAFETY_BROADCAST_B1, AUTOTUNE_SAFETY_BROADCAST_F
            );

            CorrectnessCheck noOptFull = checkCandidateCorrectness(
                    baselineFull, baselineBroadcastFull,
                    baseA, baseB, baseC, baseBroadcastA, baseBroadcastB, baseBroadcastC,
                    noOptSameKnobs, AUTOTUNE_GRAPH_BLOCKS,
                    AUTOTUNE_BROADCAST_B0, AUTOTUNE_BROADCAST_B1, AUTOTUNE_BROADCAST_F
            );
            CorrectnessCheck memFull = checkCandidateCorrectness(
                    baselineFull, baselineBroadcastFull,
                    baseA, baseB, baseC, baseBroadcastA, baseBroadcastB, baseBroadcastC,
                    memOnlySameKnobs, AUTOTUNE_GRAPH_BLOCKS,
                    AUTOTUNE_BROADCAST_B0, AUTOTUNE_BROADCAST_B1, AUTOTUNE_BROADCAST_F
            );

            System.out.println(GRAY + "candidate=" + name + RESET);
            System.out.println(GRAY + "  noopt: safetyOk=" + noOptSafety.ok() + ", fullOk=" + noOptFull.ok()
                    + ", maxSafety=" + String.format(Locale.US, "%.3e", maxAbs(noOptSafety))
                    + ", maxFull=" + String.format(Locale.US, "%.3e", maxAbs(noOptFull)) + RESET);
            System.out.println(GRAY + "  mem  : safetyOk=" + memSafety.ok() + ", fullOk=" + memFull.ok()
                    + ", maxSafety=" + String.format(Locale.US, "%.3e", maxAbs(memSafety))
                    + ", maxFull=" + String.format(Locale.US, "%.3e", maxAbs(memFull)) + RESET);
        }
        System.out.println();
    }

    private static final class MemReplayFailure {
        private final String originalName;
        private final boolean safetyOk;
        private final boolean fullOk;
        private final double maxAbsSafety;
        private final double maxAbsFull;

        private MemReplayFailure(String originalName, boolean safetyOk, boolean fullOk, double maxAbsSafety, double maxAbsFull) {
            this.originalName = originalName;
            this.safetyOk = safetyOk;
            this.fullOk = fullOk;
            this.maxAbsSafety = maxAbsSafety;
            this.maxAbsFull = maxAbsFull;
        }
    }

    private static OptimizationStage resolveReplayStage() {
        if ("NONE".equals(AUTOTUNE_REPLAY_STAGE)) {
            return null;
        }
        try {
            return OptimizationStage.valueOf(AUTOTUNE_REPLAY_STAGE);
        } catch (IllegalArgumentException ex) {
            return null;
        }
    }

    private static Tensor buildTa7(Tensor A, Tensor B, Tensor C) {
        Tensor Ta1 = A.div(B);
        Tensor Ta2 = A.sub(C);
        Tensor Ta3 = B.add(C);
        Tensor Ta4 = Ta1.div(Ta2);
        Tensor Ta5 = Ta3.mul(Ta4);
        Tensor Ta6 = Ta4.add(Ta5);
        return Ta6.pow(2);
    }

    private static DataType resolveBenchDataType() {
        String raw = System.getProperty("benchmark.dtype", DataType.FLOAT32.name()).trim().toUpperCase(Locale.ROOT);
        try {
            DataType parsed = DataType.valueOf(raw);
            if (parsed != DataType.FLOAT32 && parsed != DataType.FLOAT64) {
                throw new IllegalArgumentException("Unsupported benchmark.dtype: " + raw + ". Allowed: FLOAT32, FLOAT64");
            }
            return parsed;
        } catch (IllegalArgumentException ex) {
            throw new IllegalStateException("Invalid benchmark.dtype: " + raw + ". Allowed: FLOAT32, FLOAT64", ex);
        }
    }

    private static double[] randomData(int n) {
        double[] d = new double[n];
        for (int i = 0; i < n; i++) d[i] = RNG.nextDouble(0.1, 1.0);
        return d;
    }

    private static Diff diff(double[] left, double[] right) {
        if (left.length != right.length) {
            return new Diff(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY,
                    0, left.length, -1, Double.NaN, Double.NaN);
        }
        double maxAbs = 0.0;
        double maxRel = 0.0;
        double maxTolRatio = 0.0;
        double sumAbs = 0.0;
        int finiteCount = 0;
        int invalidCount = 0;
        int argMaxIndex = -1;
        double leftAtMax = Double.NaN;
        double rightAtMax = Double.NaN;
        for (int i = 0; i < left.length; i++) {
            double a = left[i];
            double b = right[i];
            if (!Double.isFinite(a) || !Double.isFinite(b)) {
                invalidCount++;
                continue;
            }
            double d = Math.abs(a - b);
            double scale = Math.max(1.0, Math.max(Math.abs(a), Math.abs(b)));
            double rel = d / scale;
            double tol = ABS_TOL + REL_TOL * scale;
            double tolRatio = tol > 0.0 ? (d / tol) : Double.POSITIVE_INFINITY;
            sumAbs += d;
            finiteCount++;
            if (d > maxAbs) {
                maxAbs = d;
                argMaxIndex = i;
                leftAtMax = a;
                rightAtMax = b;
            }
            if (rel > maxRel) {
                maxRel = rel;
            }
            if (tolRatio > maxTolRatio) {
                maxTolRatio = tolRatio;
            }
        }
        double avgAbs = finiteCount == 0 ? 0.0 : sumAbs / finiteCount;
        return new Diff(maxAbs, avgAbs, maxRel, maxTolRatio, finiteCount, invalidCount, argMaxIndex, leftAtMax, rightAtMax);
    }

    private static OptimizerCandidate findByName(List<OptimizerCandidate> candidates, String name) {
        for (OptimizerCandidate c : candidates) {
            if (c.name().equals(name)) return c;
        }
        throw new IllegalStateException("Missing candidate: " + name);
    }

    private static String autoTuneHistoryContextSignature() {
        String osName = normalizeContextValue(System.getProperty("os.name", "unknown"));
        String osArch = normalizeContextValue(System.getProperty("os.arch", "unknown"));
        String javaVersion = normalizeContextValue(System.getProperty("java.version", "unknown"));
        String vmName = normalizeContextValue(System.getProperty("java.vm.name", "unknown"));
        String vmVendor = normalizeContextValue(System.getProperty("java.vendor", "unknown"));
        int cores = Runtime.getRuntime().availableProcessors();
        return "schema=" + AUTOTUNE_HISTORY_SCHEMA_VERSION
                + "|engine=" + AUTOTUNE_ENGINE_VERSION
                + "|dtype=" + BENCH_DTYPE
                + "|absTol=" + ABS_TOL
                + "|relTol=" + REL_TOL
                + "|size=" + AUTOTUNE_SIZE
                + "|safetyPrecheck=" + AUTOTUNE_ENABLE_SAFETY_PRECHECK
                + "|safetySweepOnly=" + AUTOTUNE_SAFETY_SWEEP_ONLY
                + "|scanAllCandidates=" + AUTOTUNE_SCAN_ALL_CANDIDATES
                + "|rescanUnsafe=" + AUTOTUNE_RESCAN_UNSAFE
                + "|safetySize=" + AUTOTUNE_SAFETY_SIZE
                + "|graphBlocks=" + AUTOTUNE_GRAPH_BLOCKS
                + "|bshape=" + AUTOTUNE_BROADCAST_B0 + "x" + AUTOTUNE_BROADCAST_B1 + "x" + AUTOTUNE_BROADCAST_F
                + "|safetyBshape=" + AUTOTUNE_SAFETY_BROADCAST_B0 + "x" + AUTOTUNE_SAFETY_BROADCAST_B1 + "x" + AUTOTUNE_SAFETY_BROADCAST_F
                + "|os=" + osName
                + "|arch=" + osArch
                + "|jvm=" + vmName
                + "|java=" + javaVersion
                + "|vendor=" + vmVendor
                + "|blasProviderProp=" + normalizeContextValue(System.getProperty("cg.cpu.blas.provider", "NONE"))
                + "|blasMinWorkProp=" + normalizeContextValue(System.getProperty("cg.cpu.blas.matmulMinWork", "2000000"))
                + "|blasF32RequireMgeKProp=" + normalizeContextValue(System.getProperty("cg.cpu.blas.f32RequireMgeK", "true"))
                + "|blasF32MaxNOverKProp=" + normalizeContextValue(System.getProperty("cg.cpu.blas.f32MaxNOverK", "3.0"))
                + "|cores=" + cores;
    }

    private static String normalizeContextValue(String value) {
        if (value == null || value.isBlank()) {
            return "unknown";
        }
        String cleaned = value.replace('|', '_')
                .replace('\t', '_')
                .replace('\n', '_')
                .replace('\r', '_')
                .trim();
        return cleaned.isEmpty() ? "unknown" : cleaned;
    }

    private static String candidateFingerprint(OptimizerCandidate candidate) {
        String canonical = candidateCanonicalSpec(candidate);
        return sha256Hex(canonical);
    }

    private static String candidateCanonicalSpec(OptimizerCandidate candidate) {
        StringBuilder sb = new StringBuilder(512);
        sb.append("name=").append(candidate.name()).append('|');
        sb.append("stageOrder=");
        for (OptimizationStage stage : candidate.stageOrder()) {
            sb.append(stage.name()).append(',');
        }
        sb.append('|');

        TuningKnobs knobs = candidate.knobs();
        sb.append("strictCseSafety=").append(knobs.strictCseSafety()).append('|');

        var fuse = knobs.fuseConfig();
        sb.append("fuse.maxClusterNodes=").append(fuse.maxClusterNodes()).append('|');
        sb.append("fuse.scoreThreshold=").append(String.format(Locale.US, "%.12f", fuse.scoreThreshold())).append('|');
        sb.append("fuse.internalEdgeBonus=").append(String.format(Locale.US, "%.12f", fuse.internalEdgeBonus())).append('|');
        sb.append("fuse.externalInputPenalty=").append(String.format(Locale.US, "%.12f", fuse.externalInputPenalty())).append('|');
        sb.append("fuse.sharedExpensivePenalty=").append(String.format(Locale.US, "%.12f", fuse.sharedExpensivePenalty())).append('|');
        sb.append("fuse.nonCheapBonus=").append(String.format(Locale.US, "%.12f", fuse.nonCheapBonus())).append('|');
        sb.append("fuse.preserveSharedExpensiveNodes=").append(fuse.preserveSharedExpensiveNodes()).append('|');

        var kernel = knobs.kernelConfig();
        var cpu = kernel.cpu();
        sb.append("cpu.unroll=").append(cpu.loopUnrollFactor()).append('|');
        sb.append("cpu.tileM=").append(cpu.matMulTileM()).append('|');
        sb.append("cpu.tileN=").append(cpu.matMulTileN()).append('|');
        sb.append("cpu.tileK=").append(cpu.matMulTileK()).append('|');
        sb.append("cpu.vecMin=").append(cpu.vectorMinSize()).append('|');
        sb.append("cpu.parMin=").append(cpu.parallelMinSize()).append('|');
        sb.append("cpu.matMulParMin=").append(cpu.matMulParallelMinSize()).append('|');
        sb.append("cpu.parallelism=").append(cpu.parallelism()).append('|');
        sb.append("cpu.chunksPerWorker=").append(cpu.chunksPerWorker()).append('|');
        sb.append("cpu.minChunk=").append(cpu.minChunkSize()).append('|');
        sb.append("cpu.contigThreshold=").append(cpu.contiguousMaterializeThreshold()).append('|');
        sb.append("cpu.sumAcc=").append(cpu.sumAccuracyMode()).append('|');
        sb.append("cpu.lowCostNs=").append(String.format(Locale.US, "%.12f", cpu.lowCostNsPerElementThreshold())).append('|');
        sb.append("cpu.vecPolicyCheap=").append(cpu.vectorPolicyCheap()).append('|');
        sb.append("cpu.vecPolicyTrans=").append(cpu.vectorPolicyTranscendental()).append('|');
        sb.append("cpu.vecPolicyRed=").append(cpu.vectorPolicyReduction()).append('|');

        var cuda = kernel.cuda();
        sb.append("cuda.unroll=").append(cuda.loopUnrollFactor()).append('|');
        sb.append("cuda.tileM=").append(cuda.matMulTileM()).append('|');
        sb.append("cuda.tileN=").append(cuda.matMulTileN()).append('|');
        sb.append("cuda.tileK=").append(cuda.matMulTileK()).append('|');

        var opencl = kernel.opencl();
        sb.append("opencl.unroll=").append(opencl.loopUnrollFactor()).append('|');
        sb.append("opencl.tileM=").append(opencl.matMulTileM()).append('|');
        sb.append("opencl.tileN=").append(opencl.matMulTileN()).append('|');
        sb.append("opencl.tileK=").append(opencl.matMulTileK()).append('|');

        sb.append("blas.provider=").append(knobs.blasProvider()).append('|');
        sb.append("blas.matmulMinWork=").append(knobs.blasMatMulMinWork()).append('|');
        sb.append("blas.f32RequireMgeK=").append(knobs.blasF32RequireMgeK()).append('|');
        sb.append("blas.f32MaxNOverK=").append(String.format(Locale.US, "%.12f", knobs.blasF32MaxNOverK()));
        return sb.toString();
    }

    private static String sha256Hex(String text) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest(text.getBytes(StandardCharsets.UTF_8));
            StringBuilder sb = new StringBuilder(hash.length * 2);
            for (byte b : hash) {
                sb.append(Character.forDigit((b >> 4) & 0xF, 16));
                sb.append(Character.forDigit(b & 0xF, 16));
            }
            return sb.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("Missing SHA-256 algorithm", e);
        }
    }

    private static final class Diff {
        private final double maxAbs;
        private final double avgAbs;
        private final double maxRel;
        private final double maxTolRatio;
        private final int finiteCount;
        private final int invalidCount;
        private final int argMaxIndex;
        private final double leftAtMax;
        private final double rightAtMax;

        private Diff(
                double maxAbs,
                double avgAbs,
                double maxRel,
                double maxTolRatio,
                int finiteCount,
                int invalidCount,
                int argMaxIndex,
                double leftAtMax,
                double rightAtMax
        ) {
            this.maxAbs = maxAbs;
            this.avgAbs = avgAbs;
            this.maxRel = maxRel;
            this.maxTolRatio = maxTolRatio;
            this.finiteCount = finiteCount;
            this.invalidCount = invalidCount;
            this.argMaxIndex = argMaxIndex;
            this.leftAtMax = leftAtMax;
            this.rightAtMax = rightAtMax;
        }

        private boolean ok() {
            if (invalidCount > 0) return false;
            return maxTolRatio <= 1.0;
        }

        @Override
        public String toString() {
            return "Diff[maxAbs=" + maxAbs
                    + ", avgAbs=" + avgAbs
                    + ", maxRel=" + maxRel
                    + ", maxTolRatio=" + maxTolRatio
                    + ", finiteCount=" + finiteCount
                    + ", invalidCount=" + invalidCount
                    + ", argMaxIndex=" + argMaxIndex
                    + ", leftAtMax=" + leftAtMax
                    + ", rightAtMax=" + rightAtMax
                    + "]";
        }
    }

    private static final class CorrectnessCheck {
        private final Diff out;
        private final Diff gradA;
        private final Diff gradB;
        private final Diff gradC;
        private final Diff broadcast;

        private CorrectnessCheck(Diff out, Diff gradA, Diff gradB, Diff gradC, Diff broadcast) {
            this.out = out;
            this.gradA = gradA;
            this.gradB = gradB;
            this.gradC = gradC;
            this.broadcast = broadcast;
        }

        private boolean ok() {
            return out.ok() && gradA.ok() && gradB.ok() && gradC.ok() && broadcast.ok();
        }
    }

    private static final class RunResult {
        private final double[] ta7;
        private final double[] gradA;
        private final double[] gradB;
        private final double[] gradC;

        private RunResult(double[] ta7, double[] gradA, double[] gradB, double[] gradC) {
            this.ta7 = ta7;
            this.gradA = gradA;
            this.gradB = gradB;
            this.gradC = gradC;
        }
    }

    private static final class BroadcastRunResult {
        private final double[] out;

        private BroadcastRunResult(double[] out) {
            this.out = out;
        }
    }

    private enum TuneObjective {
        TRAINING,
        INFERENCE
    }
}
