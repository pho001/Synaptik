package Benchmark;

import Backend.ComputeEngine;
import Graph.optimizer.GraphOptimizer;
import Tensor.DataType;
import Tensor.Tensor;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.HashMap;
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
    private static final int AUTOTUNE_WARMUP_ITERS = 12;
    private static final int AUTOTUNE_MEASURE_ITERS = 40;
    private static final int AUTOTUNE_MAX_CANDIDATES = 500;
    private static final int AUTOTUNE_REFINE_TOP_K = 8;
    private static final int AUTOTUNE_REFINE_WARMUP_ITERS = 50;
    private static final int AUTOTUNE_REFINE_MEASURE_ITERS = 300;
    private static final int AUTOTUNE_REFINE_REPEATS = 3;
    private static final int AUTOTUNE_BROADCAST_B0 = 128;
    private static final int AUTOTUNE_BROADCAST_B1 = 8;
    private static final int AUTOTUNE_BROADCAST_F = 128;
    private static final boolean ENABLE_AUTOTUNE =
            Boolean.parseBoolean(System.getProperty("benchmark.enableAutotune", "true"));
    private static final double ABS_TOL = 1e-9;
    private static final double REL_TOL = 1e-7;
    private static final double AUTOTUNE_TRAIN_BROADCAST_WEIGHT = 0.15;
    private static final double AUTOTUNE_INF_BROADCAST_WEIGHT = 0.30;
    private static final Random RNG = new Random(42);
    private static final DataType BENCH_DTYPE = resolveBenchDataType();

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
    private static final int AUTOTUNE_HISTORY_SCHEMA_VERSION = 1;
    private static final int AUTOTUNE_ENGINE_VERSION = 2;
    private static final int HW_PROFILE_MAX_BUCKETS = 10;

    public static void run() {
        List<OptimizerCandidate> candidates = applyProfileToDefaults(OptimizerCandidateFactory.defaultCandidates());
        OptimizerCandidate noOptCandidate = findByName(candidates, "NO_OPT");
        OptimizerCandidate recommended = findByName(candidates, "RECOMMENDED");
        OptimizerCandidate inferencePerf = findByName(candidates, "INFERENCE_PERF");

        runScalarSanityCheck(OptimizerBuilder.build(recommended));

        double[] baseA = randomData(SIZE);
        double[] baseB = randomData(SIZE);
        double[] baseC = randomData(SIZE);

        // Forward benchmark běží přes inference-only pipeline (bez backward grafu).
        BenchState noOptForward = newBenchState(baseA, baseB, baseC, noOptCandidate, false);
        BenchState optForward = newBenchState(baseA, baseB, baseC, inferencePerf, false);
        // Training benchmark běží přes training pipeline.
        BenchState noOptTrain = newBenchState(baseA, baseB, baseC, noOptCandidate, true);
        BenchState optTrain = newBenchState(baseA, baseB, baseC, recommended, true);

        int compiledNoOpt = noOptTrain.ta7.getCompiledGraph().getCompiledGraphAsList().size();
        int compiledOpt = optTrain.ta7.getCompiledGraph().getCompiledGraphAsList().size();

        noOptForward.ta7.getCompiledGraph().setTrainingModeOff();
        optForward.ta7.getCompiledGraph().setTrainingModeOff();
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

        noOptTrain.ta7.getCompiledGraph().setTrainingModeOn();
        optTrain.ta7.getCompiledGraph().setTrainingModeOn();
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

        RunResult freshNoOpt = runFresh(baseA, baseB, baseC, noOptCandidate);
        RunResult freshOpt = runFresh(baseA, baseB, baseC, recommended);

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
        System.out.println(GRAY + "Size=" + SIZE + ", warmup=" + WARMUP_ITERS + ", measure=" + MEASURE_ITERS + RESET);
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
        RunResult baseline = runFresh(baseA, baseB, baseC, findByName(stages, "NO_OPT"));

        System.out.println();
        System.out.println(BOLD + CYAN + "[Stage Breakdown]" + RESET);
        System.out.println(GRAY + "Per-stage benchmark: warmup=" + STAGE_WARMUP_ITERS + ", measure=" + STAGE_MEASURE_ITERS + RESET);
        final String headerFmt = "%-18s %8s %10s %12s %12s %9s %9s %10s";
        final String rowFmt = "%-18s %8d %10d %12.4f %12.4f %9s %9s %10s";
        String header = String.format(headerFmt, "NAME", "GRAPH_INF", "GRAPH_TRN", "FWD_MS", "TRAIN_MS", "FWD_X", "TRN_X", "CHECK");
        System.out.println(GRAY + header + RESET);
        System.out.println(GRAY + "-".repeat(header.length()) + RESET);

        double baseForward = -1.0;
        double baseTrain = -1.0;

        for (int idx = 0; idx < stages.size(); idx++) {
            OptimizerCandidate stage = stages.get(idx);
            BenchState stateForward = newBenchState(baseA, baseB, baseC, stage, false);
            int graphSize = stateForward.ta7.getCompiledGraph().getCompiledGraphAsList().size();

            stateForward.ta7.getCompiledGraph().setTrainingModeOff();
            for (int i = 0; i < STAGE_WARMUP_ITERS; i++) stateForward.compute();
            long t0 = System.nanoTime();
            for (int i = 0; i < STAGE_MEASURE_ITERS; i++) stateForward.compute();
            long t1 = System.nanoTime();
            double forwardMs = (t1 - t0) / 1_000_000.0 / STAGE_MEASURE_ITERS;

            BenchState stateTrain = newBenchState(baseA, baseB, baseC, stage, true);
            int trainingGraphSize = stateTrain.ta7.getCompiledGraph().getCompiledGraphAsList().size();
            stateTrain.ta7.getCompiledGraph().setTrainingModeOn();
            for (int i = 0; i < STAGE_WARMUP_ITERS; i++) stateTrain.compute();
            long t2 = System.nanoTime();
            for (int i = 0; i < STAGE_MEASURE_ITERS; i++) stateTrain.compute();
            long t3 = System.nanoTime();
            double trainMs = (t3 - t2) / 1_000_000.0 / STAGE_MEASURE_ITERS;

            RunResult current = runFresh(baseA, baseB, baseC, stage);
            boolean ok = diff(baseline.ta7, current.ta7).ok()
                    && diff(baseline.gradA, current.gradA).ok()
                    && diff(baseline.gradB, current.gradB).ok()
                    && diff(baseline.gradC, current.gradC).ok();

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
                    ok ? "OK" : "MISMATCH"
            ) + RESET);
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

        RunResult baseline = runFresh(baseA, baseB, baseC, new OptimizerCandidate("NO_OPT", List.of(), TuningKnobs.trainingDefaults()));
        BroadcastRunResult baselineBroadcast = runBroadcastFresh(
                baseBroadcastA,
                baseBroadcastB,
                baseBroadcastC,
                new OptimizerCandidate("NO_OPT", List.of(), TuningKnobs.trainingDefaults())
        );
        List<OptimizerCandidate> all = OptimizerCandidateFactory.autotuneCandidates();
        List<OptimizerCandidate> candidates = capCandidatesDeterministic(all, AUTOTUNE_MAX_CANDIDATES);
        String historyContext = autoTuneHistoryContextSignature();
        CandidateHistory history = CandidateHistory.load(AUTOTUNE_HISTORY_PATH, historyContext);

        AutoTuneResult bestTraining = null;
        AutoTuneResult bestInference = null;
        int validCount = 0;
        int mismatchCount = 0;
        int skippedUnsafeCount = 0;
        List<AutoTuneResult> validPhase1 = new ArrayList<>();

        System.out.println(BOLD + CYAN + "[Auto-Tune]" + RESET);
        System.out.println(GRAY + "Two-phase mode" + RESET);
        System.out.println(GRAY + "Candidates total=" + all.size()
                + ", evaluated=" + candidates.size()
                + ", size=" + AUTOTUNE_SIZE
                + ", broadcastShape=[" + AUTOTUNE_BROADCAST_B0 + ",1," + AUTOTUNE_BROADCAST_F + "]x[1," + AUTOTUNE_BROADCAST_B1 + "," + AUTOTUNE_BROADCAST_F + "]"
                + ", warmup=" + AUTOTUNE_WARMUP_ITERS
                + ", measure=" + AUTOTUNE_MEASURE_ITERS
                + ", refineTopK=" + AUTOTUNE_REFINE_TOP_K
                + ", refineWarmup=" + AUTOTUNE_REFINE_WARMUP_ITERS
                + ", refineMeasure=" + AUTOTUNE_REFINE_MEASURE_ITERS
                + ", refineRepeats=" + AUTOTUNE_REFINE_REPEATS
                + ", unsafeHistory=" + AUTOTUNE_HISTORY_PATH.toAbsolutePath() + RESET);

        for (OptimizerCandidate candidate : candidates) {
            String candidateKey = candidateFingerprint(candidate);
            if (history.isUnsafe(candidateKey)) {
                skippedUnsafeCount++;
                continue;
            }
            BenchState stateForward = newBenchState(baseA, baseB, baseC, candidate, false);
            int graphInfSize = stateForward.ta7.getCompiledGraph().getCompiledGraphAsList().size();

            stateForward.ta7.getCompiledGraph().setTrainingModeOff();
            for (int i = 0; i < AUTOTUNE_WARMUP_ITERS; i++) stateForward.compute();
            long f0 = System.nanoTime();
            for (int i = 0; i < AUTOTUNE_MEASURE_ITERS; i++) stateForward.compute();
            long f1 = System.nanoTime();
            double fwdMs = (f1 - f0) / 1_000_000.0 / AUTOTUNE_MEASURE_ITERS;

            BenchState stateTrain = newBenchState(baseA, baseB, baseC, candidate, true);
            int graphTrnSize = stateTrain.ta7.getCompiledGraph().getCompiledGraphAsList().size();
            stateTrain.ta7.getCompiledGraph().setTrainingModeOn();
            for (int i = 0; i < AUTOTUNE_WARMUP_ITERS; i++) stateTrain.compute();
            long t0 = System.nanoTime();
            for (int i = 0; i < AUTOTUNE_MEASURE_ITERS; i++) stateTrain.compute();
            long t1 = System.nanoTime();
            double trainMs = (t1 - t0) / 1_000_000.0 / AUTOTUNE_MEASURE_ITERS;

            BroadcastBenchState stateBroadcast = newBroadcastBenchState(baseBroadcastA, baseBroadcastB, baseBroadcastC, candidate);
            for (int i = 0; i < AUTOTUNE_WARMUP_ITERS; i++) stateBroadcast.compute();
            long b0 = System.nanoTime();
            for (int i = 0; i < AUTOTUNE_MEASURE_ITERS; i++) stateBroadcast.compute();
            long b1 = System.nanoTime();
            double broadcastMs = (b1 - b0) / 1_000_000.0 / AUTOTUNE_MEASURE_ITERS;

            RunResult rr = runFresh(baseA, baseB, baseC, candidate);
            Diff dOut = diff(baseline.ta7, rr.ta7);
            Diff dA = diff(baseline.gradA, rr.gradA);
            Diff dB = diff(baseline.gradB, rr.gradB);
            Diff dC = diff(baseline.gradC, rr.gradC);
            BroadcastRunResult br = runBroadcastFresh(baseBroadcastA, baseBroadcastB, baseBroadcastC, candidate);
            Diff dBroadcast = diff(baselineBroadcast.out, br.out);
            boolean ok = dOut.ok() && dA.ok() && dB.ok() && dC.ok() && dBroadcast.ok();

            if (!ok) {
                mismatchCount++;
                history.markUnsafe(candidateKey, candidate.name(), "MISMATCH");
                continue;
            }
            validCount++;

            double score = scoreCandidate(fwdMs, trainMs, broadcastMs, graphInfSize, graphTrnSize, TuneObjective.TRAINING);
            AutoTuneResult cur = new AutoTuneResult(candidate, graphInfSize, graphTrnSize, fwdMs, trainMs, broadcastMs, score);
            validPhase1.add(cur);
        }

        if (validPhase1.isEmpty()) {
            System.out.println(RED + "No valid candidate passed correctness filter." + RESET);
            return;
        }

        List<AutoTuneResult> byTraining = new ArrayList<>(validPhase1);
        byTraining.sort((a, b) -> Double.compare(
                scoreCandidate(a.forwardMs, a.trainMs, a.broadcastMs, a.graphInfSize, a.graphTrnSize, TuneObjective.TRAINING),
                scoreCandidate(b.forwardMs, b.trainMs, b.broadcastMs, b.graphInfSize, b.graphTrnSize, TuneObjective.TRAINING)
        ));
        List<AutoTuneResult> byInference = new ArrayList<>(validPhase1);
        byInference.sort((a, b) -> Double.compare(
                scoreCandidate(a.forwardMs, a.trainMs, a.broadcastMs, a.graphInfSize, a.graphTrnSize, TuneObjective.INFERENCE),
                scoreCandidate(b.forwardMs, b.trainMs, b.broadcastMs, b.graphInfSize, b.graphTrnSize, TuneObjective.INFERENCE)
        ));
        Set<OptimizerCandidate> finalistsSet = new LinkedHashSet<>();
        int kTrain = Math.min(AUTOTUNE_REFINE_TOP_K, byTraining.size());
        int kInf = Math.min(AUTOTUNE_REFINE_TOP_K, byInference.size());
        for (int i = 0; i < kTrain; i++) finalistsSet.add(byTraining.get(i).candidate);
        for (int i = 0; i < kInf; i++) finalistsSet.add(byInference.get(i).candidate);
        List<OptimizerCandidate> finalistsList = new ArrayList<>(finalistsSet);
        int finalists = finalistsList.size();
        System.out.println(GRAY + "Phase1 valid=" + validCount + ", mismatch=" + mismatchCount
                + ", skippedUnsafe=" + skippedUnsafeCount
                + ", finalists=" + finalists + RESET);

        for (int i = 0; i < finalists; i++) {
            OptimizerCandidate candidate = finalistsList.get(i);
            double sumFwdMs = 0.0;
            double sumTrainMs = 0.0;
            double sumBroadcastMs = 0.0;
            int graphInfSize = -1;
            int graphTrnSize = -1;

            for (int r = 0; r < AUTOTUNE_REFINE_REPEATS; r++) {
                BenchState stateForward = newBenchState(baseA, baseB, baseC, candidate, false);
                graphInfSize = stateForward.ta7.getCompiledGraph().getCompiledGraphAsList().size();

                stateForward.ta7.getCompiledGraph().setTrainingModeOff();
                for (int j = 0; j < AUTOTUNE_REFINE_WARMUP_ITERS; j++) stateForward.compute();
                long f0 = System.nanoTime();
                for (int j = 0; j < AUTOTUNE_REFINE_MEASURE_ITERS; j++) stateForward.compute();
                long f1 = System.nanoTime();
                sumFwdMs += (f1 - f0) / 1_000_000.0 / AUTOTUNE_REFINE_MEASURE_ITERS;

                BenchState stateTrain = newBenchState(baseA, baseB, baseC, candidate, true);
                graphTrnSize = stateTrain.ta7.getCompiledGraph().getCompiledGraphAsList().size();
                stateTrain.ta7.getCompiledGraph().setTrainingModeOn();
                for (int j = 0; j < AUTOTUNE_REFINE_WARMUP_ITERS; j++) stateTrain.compute();
                long t0 = System.nanoTime();
                for (int j = 0; j < AUTOTUNE_REFINE_MEASURE_ITERS; j++) stateTrain.compute();
                long t1 = System.nanoTime();
                sumTrainMs += (t1 - t0) / 1_000_000.0 / AUTOTUNE_REFINE_MEASURE_ITERS;

                BroadcastBenchState stateBroadcast = newBroadcastBenchState(baseBroadcastA, baseBroadcastB, baseBroadcastC, candidate);
                for (int j = 0; j < AUTOTUNE_REFINE_WARMUP_ITERS; j++) stateBroadcast.compute();
                long b0 = System.nanoTime();
                for (int j = 0; j < AUTOTUNE_REFINE_MEASURE_ITERS; j++) stateBroadcast.compute();
                long b1 = System.nanoTime();
                sumBroadcastMs += (b1 - b0) / 1_000_000.0 / AUTOTUNE_REFINE_MEASURE_ITERS;
            }

            double fwdMs = sumFwdMs / AUTOTUNE_REFINE_REPEATS;
            double trainMs = sumTrainMs / AUTOTUNE_REFINE_REPEATS;
            double broadcastMs = sumBroadcastMs / AUTOTUNE_REFINE_REPEATS;
            double trainScore = scoreCandidate(fwdMs, trainMs, broadcastMs, graphInfSize, graphTrnSize, TuneObjective.TRAINING);
            double infScore = scoreCandidate(fwdMs, trainMs, broadcastMs, graphInfSize, graphTrnSize, TuneObjective.INFERENCE);

            AutoTuneResult refinedTraining = new AutoTuneResult(candidate, graphInfSize, graphTrnSize, fwdMs, trainMs, broadcastMs, trainScore);
            AutoTuneResult refinedInference = new AutoTuneResult(candidate, graphInfSize, graphTrnSize, fwdMs, trainMs, broadcastMs, infScore);
            if (bestTraining == null || refinedTraining.score < bestTraining.score) {
                bestTraining = refinedTraining;
            }
            if (bestInference == null || refinedInference.score < bestInference.score) {
                bestInference = refinedInference;
            }
        }

        System.out.println(GREEN + "Best (TRAINING): " + bestTraining.candidate.name()
                + " | graph_trn=" + bestTraining.graphTrnSize
                + " | fwd=" + String.format("%.4f", bestTraining.forwardMs) + " ms"
                + " | train=" + String.format("%.4f", bestTraining.trainMs) + " ms"
                + " | bcast=" + String.format("%.4f", bestTraining.broadcastMs) + " ms"
                + " | score=" + String.format("%.4f", bestTraining.score) + RESET);
        System.out.println(GREEN + "Best (INFERENCE): " + bestInference.candidate.name()
                + " | graph_inf=" + bestInference.graphInfSize
                + " | fwd=" + String.format("%.4f", bestInference.forwardMs) + " ms"
                + " | bcast=" + String.format("%.4f", bestInference.broadcastMs) + " ms"
                + " | score=" + String.format("%.4f", bestInference.score) + RESET);
        System.out.println(GRAY + "Valid=" + validCount + ", mismatch=" + mismatchCount + RESET);

        try {
            Files.createDirectories(AUTOTUNE_BEST_TRAINING_PATH.getParent());
            double previousTrainingScore = OptimizerProfileIO.loadScoreOrInfinity(AUTOTUNE_BEST_TRAINING_PATH);
            double previousInferenceScore = OptimizerProfileIO.loadScoreOrInfinity(AUTOTUNE_BEST_INFERENCE_PATH);
            boolean trainingImproved = bestTraining.score + 1e-12 < previousTrainingScore;
            boolean inferenceImproved = bestInference.score + 1e-12 < previousInferenceScore;
            String hwBucket = OptimizerProfileIO.hardwareBucketKey();

            if (trainingImproved) {
                Files.writeString(AUTOTUNE_BEST_TRAINING_PATH, bestTraining.toJson(validCount, mismatchCount), StandardCharsets.UTF_8);
                // Backward-compatible alias for tooling that expects single best profile.
                Files.writeString(AUTOTUNE_BEST_PATH, bestTraining.toJson(validCount, mismatchCount), StandardCharsets.UTF_8);
                // Keep RECOMMENDED as training-oriented by default.
                OptimizerProfileIO.saveKnobs(PROFILE_PATH, bestTraining.candidate.knobs(), bestTraining.candidate.name());
                System.out.println(CYAN + "Saved improved training profile: " + RESET + AUTOTUNE_BEST_TRAINING_PATH.toAbsolutePath());
                System.out.println(CYAN + "Updated runtime profile (training): " + RESET + PROFILE_PATH.toAbsolutePath());
            } else {
                System.out.println(GRAY + "Training profile kept (existing score="
                        + String.format("%.6f", previousTrainingScore)
                        + " <= new score="
                        + String.format("%.6f", bestTraining.score) + ")." + RESET);
            }

            if (inferenceImproved) {
                Files.writeString(AUTOTUNE_BEST_INFERENCE_PATH, bestInference.toJson(validCount, mismatchCount), StandardCharsets.UTF_8);
                System.out.println(CYAN + "Saved improved inference profile: " + RESET + AUTOTUNE_BEST_INFERENCE_PATH.toAbsolutePath());
            } else {
                System.out.println(GRAY + "Inference profile kept (existing score="
                        + String.format("%.6f", previousInferenceScore)
                        + " <= new score="
                        + String.format("%.6f", bestInference.score) + ")." + RESET);
            }

            boolean hwTrainingImproved = OptimizerProfileIO.saveHardwareProfileIfImproved(
                    HW_PROFILE_PATH,
                    hwBucket,
                    "TRAINING",
                    bestTraining.candidate,
                    bestTraining.score,
                    HW_PROFILE_MAX_BUCKETS
            );
            boolean hwInferenceImproved = OptimizerProfileIO.saveHardwareProfileIfImproved(
                    HW_PROFILE_PATH,
                    hwBucket,
                    "INFERENCE",
                    bestInference.candidate,
                    bestInference.score,
                    HW_PROFILE_MAX_BUCKETS
            );
            if (hwTrainingImproved || hwInferenceImproved) {
                System.out.println(CYAN + "Updated HW profiles: " + RESET + HW_PROFILE_PATH.toAbsolutePath()
                        + " (bucket=" + hwBucket + ")");
            } else {
                System.out.println(GRAY + "HW profiles kept (no score improvement for bucket " + hwBucket + ")." + RESET);
            }
        } catch (IOException e) {
            throw new IllegalStateException("Failed to write autotune profile", e);
        }
        history.save(AUTOTUNE_HISTORY_PATH);
        System.out.println();
    }

    private static double scoreCandidate(
            double forwardMs,
            double trainMs,
            double broadcastMs,
            int graphInfSize,
            int graphTrnSize,
            TuneObjective objective
    ) {
        if (objective == TuneObjective.INFERENCE) {
            double weightedForward = (1.0 - AUTOTUNE_INF_BROADCAST_WEIGHT) * forwardMs;
            double weightedBroadcast = AUTOTUNE_INF_BROADCAST_WEIGHT * broadcastMs;
            return weightedForward + weightedBroadcast + (0.0005 * graphInfSize);
        }
        // Training objective: training dominates, with broadcast pressure and mild graph-size preference.
        return (0.35 * forwardMs)
                + (0.50 * trainMs)
                + (AUTOTUNE_TRAIN_BROADCAST_WEIGHT * broadcastMs)
                + (0.0005 * graphTrnSize);
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
        Te7No.compute();
        Te7No.getCompiledGraph().setTrainingModeOn();
        Te7No.compute();

        Tensor A1 = Tensor.scalar(10.0, BENCH_DTYPE);
        Tensor B1 = Tensor.scalar(2.0, BENCH_DTYPE);
        Tensor C1 = Tensor.scalar(5.0, BENCH_DTYPE);
        A1.setRequiresGrad(true);
        B1.setRequiresGrad(true);
        C1.setRequiresGrad(true);
        Tensor Te7Opt = buildTa7(A1, B1, C1);
        Te7Opt.compute(optimizer);
        Te7Opt.getCompiledGraph().setTrainingModeOn();
        Te7Opt.compute(optimizer);

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

    private static RunResult runFresh(double[] baseA, double[] baseB, double[] baseC, OptimizerCandidate candidate) {
        BenchState s = newBenchState(baseA, baseB, baseC, candidate, true);
        s.ta7.getCompiledGraph().setTrainingModeOn();
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
        BroadcastBenchState s = newBroadcastBenchState(baseA, baseB, baseC, candidate);
        s.compute();
        return new BroadcastRunResult(s.out.toDoubleArrayCopy().clone());
    }

    private static BenchState newBenchState(double[] baseA, double[] baseB, double[] baseC, OptimizerCandidate candidate, boolean requiresGrad) {
        ComputeEngine.setCpuKernelConfig(candidate.knobs().kernelConfig().cpu());

        Tensor A = inputTensor("A", baseA, requiresGrad);
        Tensor B = inputTensor("B", baseB, requiresGrad);
        Tensor C = inputTensor("C", baseC, requiresGrad);
        Tensor Ta7 = buildTa7(A, B, C);

        GraphOptimizer optimizer = OptimizerBuilder.build(candidate);
        Ta7.compute(optimizer);

        return new BenchState(A, B, C, Ta7, optimizer);
    }

    private static BroadcastBenchState newBroadcastBenchState(
            double[] baseA,
            double[] baseB,
            double[] baseC,
            OptimizerCandidate candidate
    ) {
        ComputeEngine.setCpuKernelConfig(candidate.knobs().kernelConfig().cpu());

        Tensor A = inputTensor("BA", baseA, false, new int[]{AUTOTUNE_BROADCAST_B0, 1, AUTOTUNE_BROADCAST_F});
        Tensor B = inputTensor("BB", baseB, false, new int[]{1, AUTOTUNE_BROADCAST_B1, AUTOTUNE_BROADCAST_F});
        Tensor C = inputTensor("BC", baseC, false, new int[]{AUTOTUNE_BROADCAST_B0, AUTOTUNE_BROADCAST_B1, AUTOTUNE_BROADCAST_F});
        Tensor out = buildBroadcastExpr(A, B, C);

        GraphOptimizer optimizer = OptimizerBuilder.build(candidate);
        out.compute(optimizer);

        return new BroadcastBenchState(out, optimizer);
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

    private static Tensor buildBroadcastExpr(Tensor A, Tensor B, Tensor C) {
        return A.add(B).mul(C).add(A).sigmoid();
    }

    private static Tensor inputTensor(String label, double[] data, boolean requiresGrad) {
        Tensor t = new Tensor(new int[]{data.length}, null, label, BENCH_DTYPE);
        t.setData(data.clone());
        t.setRequiresGrad(requiresGrad);
        return t;
    }

    private static Tensor inputTensor(String label, double[] data, boolean requiresGrad, int[] shape) {
        Tensor t = new Tensor(shape, null, label, BENCH_DTYPE);
        t.setData(data.clone());
        t.setRequiresGrad(requiresGrad);
        return t;
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
        if (left.length != right.length) return new Diff(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, 0, left.length);
        double maxAbs = 0.0;
        double sumAbs = 0.0;
        int finiteCount = 0;
        int invalidCount = 0;
        for (int i = 0; i < left.length; i++) {
            double a = left[i];
            double b = right[i];
            if (!Double.isFinite(a) || !Double.isFinite(b)) {
                invalidCount++;
                continue;
            }
            double d = Math.abs(a - b);
            sumAbs += d;
            finiteCount++;
            if (d > maxAbs) maxAbs = d;
        }
        double avgAbs = finiteCount == 0 ? 0.0 : sumAbs / finiteCount;
        return new Diff(maxAbs, avgAbs, finiteCount, invalidCount);
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
                + "|bshape=" + AUTOTUNE_BROADCAST_B0 + "x" + AUTOTUNE_BROADCAST_B1 + "x" + AUTOTUNE_BROADCAST_F
                + "|os=" + osName
                + "|arch=" + osArch
                + "|jvm=" + vmName
                + "|java=" + javaVersion
                + "|vendor=" + vmVendor
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
        sb.append("opencl.tileK=").append(opencl.matMulTileK());
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

    private static final class CandidateHistory {
        private final String contextSignature;
        private final Map<String, UnsafeCandidateRecord> unsafeByFingerprint;
        private boolean dirty;

        private CandidateHistory(String contextSignature, Map<String, UnsafeCandidateRecord> unsafeByFingerprint) {
            this.contextSignature = contextSignature;
            this.unsafeByFingerprint = unsafeByFingerprint;
        }

        private static CandidateHistory load(Path path, String contextSignature) {
            if (!Files.exists(path)) {
                return new CandidateHistory(contextSignature, new HashMap<>());
            }
            Map<String, UnsafeCandidateRecord> unsafe = new HashMap<>();
            try {
                List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
                for (String line : lines) {
                    if (line == null || line.isBlank() || line.startsWith("#")) {
                        continue;
                    }
                    String[] cols = line.split("\t", 5);
                    if (cols.length < 5) {
                        continue;
                    }
                    if (!"UNSAFE".equals(cols[1])) {
                        continue;
                    }
                    if (!contextSignature.equals(cols[4])) {
                        continue;
                    }
                    unsafe.put(cols[0], new UnsafeCandidateRecord(cols[0], cols[2], cols[3], cols[4]));
                }
                return new CandidateHistory(contextSignature, unsafe);
            } catch (IOException e) {
                return new CandidateHistory(contextSignature, new HashMap<>());
            }
        }

        private boolean isUnsafe(String fingerprint) {
            return unsafeByFingerprint.containsKey(fingerprint);
        }

        private void markUnsafe(String fingerprint, String candidateName, String reason) {
            if (unsafeByFingerprint.containsKey(fingerprint)) {
                return;
            }
            String now = OffsetDateTime.now().toString();
            String cleanReason = sanitize(reason + " candidate=" + candidateName);
            unsafeByFingerprint.put(
                    fingerprint,
                    new UnsafeCandidateRecord(fingerprint, cleanReason, now, contextSignature)
            );
            dirty = true;
        }

        private void save(Path path) {
            if (!dirty) {
                return;
            }
            try {
                Files.createDirectories(path.getParent());
                List<String> lines = new ArrayList<>();
                lines.add("# fingerprint\tstatus\treason\ttimestamp\tcontext");
                for (UnsafeCandidateRecord record : unsafeByFingerprint.values()) {
                    lines.add(record.toLine());
                }
                Files.write(path, lines, StandardCharsets.UTF_8);
                dirty = false;
            } catch (IOException e) {
                throw new IllegalStateException("Failed to write autotune candidate history", e);
            }
        }

        private static String sanitize(String value) {
            return value
                    .replace('\t', ' ')
                    .replace('\n', ' ')
                    .replace('\r', ' ');
        }
    }

    private static final class UnsafeCandidateRecord {
        private final String fingerprint;
        private final String reason;
        private final String timestamp;
        private final String context;

        private UnsafeCandidateRecord(String fingerprint, String reason, String timestamp, String context) {
            this.fingerprint = fingerprint;
            this.reason = reason;
            this.timestamp = timestamp;
            this.context = context;
        }

        private String toLine() {
            return fingerprint + "\tUNSAFE\t" + reason + "\t" + timestamp + "\t" + context;
        }
    }

    private static final class Diff {
        private final double maxAbs;
        private final double avgAbs;
        private final int finiteCount;
        private final int invalidCount;

        private Diff(double maxAbs, double avgAbs, int finiteCount, int invalidCount) {
            this.maxAbs = maxAbs;
            this.avgAbs = avgAbs;
            this.finiteCount = finiteCount;
            this.invalidCount = invalidCount;
        }

        private boolean ok() {
            if (invalidCount > 0) return false;
            double tol = ABS_TOL + REL_TOL * Math.max(1.0, maxAbs);
            return maxAbs <= tol;
        }

        @Override
        public String toString() {
            return "Diff[maxAbs=" + maxAbs + ", avgAbs=" + avgAbs + ", finiteCount=" + finiteCount + ", invalidCount=" + invalidCount + "]";
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

    private static final class BenchState {
        private final Tensor A;
        private final Tensor B;
        private final Tensor C;
        private final Tensor ta7;
        private final GraphOptimizer optimizer;

        private BenchState(Tensor a, Tensor b, Tensor c, Tensor ta7, GraphOptimizer optimizer) {
            this.A = a;
            this.B = b;
            this.C = c;
            this.ta7 = ta7;
            this.optimizer = optimizer;
        }

        private void compute() {
            ta7.compute(optimizer);
        }
    }

    private static final class BroadcastBenchState {
        private final Tensor out;
        private final GraphOptimizer optimizer;

        private BroadcastBenchState(Tensor out, GraphOptimizer optimizer) {
            this.out = out;
            this.optimizer = optimizer;
        }

        private void compute() {
            out.compute(optimizer);
        }
    }

    private static final class AutoTuneResult {
        private final OptimizerCandidate candidate;
        private final int graphInfSize;
        private final int graphTrnSize;
        private final double forwardMs;
        private final double trainMs;
        private final double broadcastMs;
        private final double score;

        private AutoTuneResult(
                OptimizerCandidate candidate,
                int graphInfSize,
                int graphTrnSize,
                double forwardMs,
                double trainMs,
                double broadcastMs,
                double score
        ) {
            this.candidate = candidate;
            this.graphInfSize = graphInfSize;
            this.graphTrnSize = graphTrnSize;
            this.forwardMs = forwardMs;
            this.trainMs = trainMs;
            this.broadcastMs = broadcastMs;
            this.score = score;
        }

        private String toJson(int validCount, int mismatchCount) {
            var knobs = candidate.knobs();
            var fuse = knobs.fuseConfig();
            var kernels = knobs.kernelConfig();
            StringBuilder sb = new StringBuilder();
            sb.append("{\n");
            sb.append("  \"timestamp\": \"").append(OffsetDateTime.now()).append("\",\n");
            sb.append("  \"candidateName\": \"").append(candidate.name()).append("\",\n");
            sb.append("  \"stageOrder\": ").append(stageOrderJson(candidate.stageOrder())).append(",\n");
            sb.append("  \"knobs\": {\n");
            sb.append("    \"strictCseSafety\": ").append(knobs.strictCseSafety()).append(",\n");
            sb.append("    \"kernel\": {\n");
            sb.append("      \"cpu\": {\n");
            sb.append("        \"cpuLoopUnrollFactor\": ").append(kernels.cpu().loopUnrollFactor()).append(",\n");
            sb.append("        \"cpuMatMulTileM\": ").append(kernels.cpu().matMulTileM()).append(",\n");
            sb.append("        \"cpuMatMulTileN\": ").append(kernels.cpu().matMulTileN()).append(",\n");
            sb.append("        \"cpuMatMulTileK\": ").append(kernels.cpu().matMulTileK()).append(",\n");
            sb.append("        \"cpuVectorMinSize\": ").append(kernels.cpu().vectorMinSize()).append(",\n");
            sb.append("        \"cpuParallelMinSize\": ").append(kernels.cpu().parallelMinSize()).append(",\n");
            sb.append("        \"cpuMatMulParallelMinSize\": ").append(kernels.cpu().matMulParallelMinSize()).append(",\n");
            sb.append("        \"cpuParallelism\": ").append(kernels.cpu().parallelism()).append(",\n");
            sb.append("        \"cpuChunksPerWorker\": ").append(kernels.cpu().chunksPerWorker()).append(",\n");
            sb.append("        \"cpuMinChunkSize\": ").append(kernels.cpu().minChunkSize()).append(",\n");
            sb.append("        \"cpuContiguousMaterializeThreshold\": ").append(kernels.cpu().contiguousMaterializeThreshold()).append(",\n");
            sb.append("        \"cpuLowCostNsPerElementThreshold\": ").append(String.format(Locale.US, "%.8f", kernels.cpu().lowCostNsPerElementThreshold())).append(",\n");
            sb.append("        \"cpuVectorPolicyCheap\": \"").append(kernels.cpu().vectorPolicyCheap().name()).append("\",\n");
            sb.append("        \"cpuVectorPolicyTranscendental\": \"").append(kernels.cpu().vectorPolicyTranscendental().name()).append("\",\n");
            sb.append("        \"cpuVectorPolicyReduction\": \"").append(kernels.cpu().vectorPolicyReduction().name()).append("\"\n");
            sb.append("      },\n");
            sb.append("      \"cuda\": {\n");
            sb.append("        \"cudaLoopUnrollFactor\": ").append(kernels.cuda().loopUnrollFactor()).append(",\n");
            sb.append("        \"cudaMatMulTileM\": ").append(kernels.cuda().matMulTileM()).append(",\n");
            sb.append("        \"cudaMatMulTileN\": ").append(kernels.cuda().matMulTileN()).append(",\n");
            sb.append("        \"cudaMatMulTileK\": ").append(kernels.cuda().matMulTileK()).append("\n");
            sb.append("      },\n");
            sb.append("      \"opencl\": {\n");
            sb.append("        \"openclLoopUnrollFactor\": ").append(kernels.opencl().loopUnrollFactor()).append(",\n");
            sb.append("        \"openclMatMulTileM\": ").append(kernels.opencl().matMulTileM()).append(",\n");
            sb.append("        \"openclMatMulTileN\": ").append(kernels.opencl().matMulTileN()).append(",\n");
            sb.append("        \"openclMatMulTileK\": ").append(kernels.opencl().matMulTileK()).append("\n");
            sb.append("      }\n");
            sb.append("    },\n");
            sb.append("    \"fuse\": {\n");
            sb.append("      \"maxClusterNodes\": ").append(fuse.maxClusterNodes()).append(",\n");
            sb.append("      \"scoreThreshold\": ").append(String.format(Locale.US, "%.8f", fuse.scoreThreshold())).append(",\n");
            sb.append("      \"internalEdgeBonus\": ").append(String.format(Locale.US, "%.8f", fuse.internalEdgeBonus())).append(",\n");
            sb.append("      \"externalInputPenalty\": ").append(String.format(Locale.US, "%.8f", fuse.externalInputPenalty())).append(",\n");
            sb.append("      \"sharedExpensivePenalty\": ").append(String.format(Locale.US, "%.8f", fuse.sharedExpensivePenalty())).append(",\n");
            sb.append("      \"nonCheapBonus\": ").append(String.format(Locale.US, "%.8f", fuse.nonCheapBonus())).append(",\n");
            sb.append("      \"preserveSharedExpensiveNodes\": ").append(fuse.preserveSharedExpensiveNodes()).append("\n");
            sb.append("    }\n");
            sb.append("  },\n");
            sb.append("  \"metrics\": {\n");
            sb.append("    \"graphInfSize\": ").append(graphInfSize).append(",\n");
            sb.append("    \"graphTrnSize\": ").append(graphTrnSize).append(",\n");
            sb.append("    \"forwardMs\": ").append(String.format(Locale.US, "%.8f", forwardMs)).append(",\n");
            sb.append("    \"trainMs\": ").append(String.format(Locale.US, "%.8f", trainMs)).append(",\n");
            sb.append("    \"broadcastMs\": ").append(String.format(Locale.US, "%.8f", broadcastMs)).append(",\n");
            sb.append("    \"score\": ").append(String.format(Locale.US, "%.8f", score)).append(",\n");
            sb.append("    \"validCandidates\": ").append(validCount).append(",\n");
            sb.append("    \"mismatchedCandidates\": ").append(mismatchCount).append("\n");
            sb.append("  }\n");
            sb.append("}\n");
            return sb.toString();
        }

        private static String stageOrderJson(List<OptimizationStage> stages) {
            if (stages.isEmpty()) return "[]";
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < stages.size(); i++) {
                if (i > 0) sb.append(", ");
                sb.append("\"").append(stages.get(i).name()).append("\"");
            }
            sb.append("]");
            return sb.toString();
        }
    }

    private enum TuneObjective {
        TRAINING,
        INFERENCE
    }
}
