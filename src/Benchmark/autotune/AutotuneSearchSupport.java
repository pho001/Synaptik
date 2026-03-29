package Benchmark.autotune;

import Benchmark.OptimizationStage;
import Benchmark.OptimizerCandidate;
import Benchmark.measure.MeasurementObjective;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Consumer;
import java.util.function.Function;

public final class AutotuneSearchSupport {
    private AutotuneSearchSupport() {}

    public static List<OptimizerCandidate> selectCandidatesViaBeam(
            List<CandidatePerf> prescreen,
            Set<OptimizerCandidate> seedCandidates,
            BeamSearchConfig config,
            Function<String, List<String>> stageOrderNeighborsFn,
            Function<OptimizerCandidate, String> fingerprintFn,
            Consumer<String> logger
    ) {
        if (prescreen.isEmpty()) {
            return List.of();
        }
        if (prescreen.size() <= Math.max(config.keepTrain(), config.keepInference())) {
            List<OptimizerCandidate> direct = new ArrayList<>(prescreen.size());
            for (CandidatePerf perf : prescreen) {
                direct.add(perf.candidate());
            }
            return direct;
        }

        Consumer<String> effectiveLogger = logger == null ? msg -> {} : logger;
        CandidateGraphIndex graph = new CandidateGraphIndex(prescreen, stageOrderNeighborsFn);
        Set<OptimizerCandidate> out = new LinkedHashSet<>();
        runBeamForObjective(MeasurementObjective.TRAINING, prescreen, seedCandidates, graph, out, config, fingerprintFn, effectiveLogger);
        runBeamForObjective(MeasurementObjective.INFERENCE, prescreen, seedCandidates, graph, out, config, fingerprintFn, effectiveLogger);
        return new ArrayList<>(out);
    }

    public static List<FamilyScoutStats> pruneFamilyScoutRound(
            List<FamilyScoutStats> activeFamilies,
            int targetActive,
            int trainKeep,
            int infKeep
    ) {
        if (activeFamilies.size() <= targetActive) {
            return new ArrayList<>(activeFamilies);
        }
        List<FamilyScoutStats> byTrain = new ArrayList<>(activeFamilies);
        byTrain.sort((a, b) -> Double.compare(a.trainingOptimistic(), b.trainingOptimistic()));
        List<FamilyScoutStats> byInf = new ArrayList<>(activeFamilies);
        byInf.sort((a, b) -> Double.compare(a.inferenceOptimistic(), b.inferenceOptimistic()));
        Set<FamilyScoutStats> survivors = new LinkedHashSet<>();
        for (int i = 0; i < Math.min(trainKeep, byTrain.size()); i++) {
            survivors.add(byTrain.get(i));
        }
        for (int i = 0; i < Math.min(infKeep, byInf.size()); i++) {
            survivors.add(byInf.get(i));
        }
        if (survivors.size() < targetActive) {
            List<FamilyScoutStats> byCombined = new ArrayList<>(activeFamilies);
            byCombined.sort((a, b) -> Double.compare(a.combinedOptimistic(), b.combinedOptimistic()));
            for (FamilyScoutStats family : byCombined) {
                survivors.add(family);
                if (survivors.size() >= targetActive) {
                    break;
                }
            }
        }
        return new ArrayList<>(survivors);
    }

    public static List<OptimizerCandidate> sampleCandidatesEvenly(List<OptimizerCandidate> candidates, int sampleCount) {
        if (candidates.isEmpty()) {
            return List.of();
        }
        if (candidates.size() <= sampleCount) {
            return new ArrayList<>(candidates);
        }
        List<OptimizerCandidate> out = new ArrayList<>(sampleCount);
        if (sampleCount == 1) {
            out.add(candidates.get(candidates.size() / 2));
            return out;
        }
        int lastIndex = candidates.size() - 1;
        for (int i = 0; i < sampleCount; i++) {
            int idx = (int) Math.round((i * lastIndex) / (double) (sampleCount - 1));
            out.add(candidates.get(idx));
        }
        return out;
    }

    public static List<OptimizerCandidate> capCandidatesPerStageOrder(List<OptimizerCandidate> candidates, int maxPerStageOrder) {
        if (candidates.isEmpty() || maxPerStageOrder <= 0) {
            return candidates;
        }
        Map<String, List<OptimizerCandidate>> byStageOrder = new LinkedHashMap<>();
        for (OptimizerCandidate candidate : candidates) {
            byStageOrder.computeIfAbsent(stageOrderKey(candidate), key -> new ArrayList<>()).add(candidate);
        }
        List<OptimizerCandidate> out = new ArrayList<>(Math.min(candidates.size(), byStageOrder.size() * maxPerStageOrder));
        for (List<OptimizerCandidate> group : byStageOrder.values()) {
            out.addAll(sampleCandidatesEvenly(group, maxPerStageOrder));
        }
        return out;
    }

    public static String stageOrderKey(OptimizerCandidate candidate) {
        if (candidate == null || candidate.stageOrder().isEmpty()) {
            return "NONE";
        }
        StringBuilder sb = new StringBuilder(32);
        for (OptimizationStage stage : candidate.stageOrder()) {
            if (sb.length() > 0) {
                sb.append(',');
            }
            sb.append(stage.name());
        }
        return sb.toString();
    }

    public static List<String> stageOrderNeighbors(String stageOrderKey) {
        List<OptimizationStage> stages = parseStageOrderKey(stageOrderKey);
        boolean hasMem = !stages.isEmpty() && stages.get(stages.size() - 1) == OptimizationStage.MEM;
        List<OptimizationStage> nonMem = new ArrayList<>(stages);
        if (hasMem) {
            nonMem.remove(nonMem.size() - 1);
        }

        LinkedHashSet<String> out = new LinkedHashSet<>();
        OptimizationStage[] baseStages = new OptimizationStage[]{OptimizationStage.AR, OptimizationStage.CSE, OptimizationStage.FUSE};

        for (int i = 0; i < nonMem.size(); i++) {
            List<OptimizationStage> next = new ArrayList<>(nonMem);
            next.remove(i);
            out.add(toStageOrderKey(next, hasMem));
        }

        for (OptimizationStage stage : baseStages) {
            if (nonMem.contains(stage)) {
                continue;
            }
            for (int pos = 0; pos <= nonMem.size(); pos++) {
                List<OptimizationStage> next = new ArrayList<>(nonMem);
                next.add(pos, stage);
                out.add(toStageOrderKey(next, hasMem));
            }
        }

        for (int i = 0; i + 1 < nonMem.size(); i++) {
            List<OptimizationStage> next = new ArrayList<>(nonMem);
            Collections.swap(next, i, i + 1);
            out.add(toStageOrderKey(next, hasMem));
        }

        out.remove(stageOrderKey);
        return new ArrayList<>(out);
    }

    private static void runBeamForObjective(
            MeasurementObjective objective,
            List<CandidatePerf> prescreen,
            Set<OptimizerCandidate> seedCandidates,
            CandidateGraphIndex graph,
            Set<OptimizerCandidate> out,
            BeamSearchConfig config,
            Function<OptimizerCandidate, String> fingerprintFn,
            Consumer<String> logger
    ) {
        Comparator<CandidatePerf> comparator = objective == MeasurementObjective.TRAINING
                ? Comparator.comparingDouble(CandidatePerf::trainingScore)
                : Comparator.comparingDouble(CandidatePerf::inferenceScore);
        int seedLimit = objective == MeasurementObjective.TRAINING ? config.seedTrain() : config.seedInference();
        int beamWidth = objective == MeasurementObjective.TRAINING ? config.beamWidthTrain() : config.beamWidthInference();
        int keep = objective == MeasurementObjective.TRAINING ? config.keepTrain() : config.keepInference();

        List<CandidatePerf> sorted = new ArrayList<>(prescreen);
        sorted.sort(comparator);

        LinkedHashSet<CandidatePerf> seedFrontier = new LinkedHashSet<>();
        for (CandidatePerf perf : sorted) {
            if (seedCandidates.contains(perf.candidate())) {
                seedFrontier.add(perf);
                if (seedFrontier.size() >= seedLimit) {
                    break;
                }
            }
        }
        for (CandidatePerf perf : sorted) {
            if (seedFrontier.size() >= seedLimit) {
                break;
            }
            seedFrontier.add(perf);
        }

        Set<String> visited = new LinkedHashSet<>();
        List<CandidatePerf> frontier = new ArrayList<>(seedFrontier);
        List<CandidatePerf> explored = new ArrayList<>(seedFrontier);
        for (CandidatePerf perf : seedFrontier) {
            visited.add(fingerprintFn.apply(perf.candidate()));
        }

        for (int round = 1; round <= config.rounds() && !frontier.isEmpty(); round++) {
            List<CandidatePerf> pool = new ArrayList<>();
            Set<String> poolSeen = new HashSet<>();
            for (CandidatePerf perf : frontier) {
                for (CandidatePerf neighbor : graph.neighbors(perf)) {
                    String fp = fingerprintFn.apply(neighbor.candidate());
                    if (visited.contains(fp) || !poolSeen.add(fp)) {
                        continue;
                    }
                    pool.add(neighbor);
                }
            }
            if (pool.isEmpty()) {
                logger.accept("Beam " + objective + " round " + round + " | no new neighbors");
                break;
            }
            pool.sort(comparator);
            frontier = pickWithStageDiversity(pool, beamWidth, config.maxPerStage(), fingerprintFn);
            for (CandidatePerf perf : frontier) {
                String fp = fingerprintFn.apply(perf.candidate());
                if (visited.add(fp)) {
                    explored.add(perf);
                }
            }
            logger.accept("Beam " + objective
                    + " round=" + round
                    + " | newNeighbors=" + pool.size()
                    + " | selected=" + frontier.size()
                    + " | best=" + frontier.get(0).candidate().name()
                    + " | bestScore=" + String.format(java.util.Locale.US, "%.4f",
                    objective == MeasurementObjective.TRAINING ? frontier.get(0).trainingScore() : frontier.get(0).inferenceScore()));
        }

        explored.sort(comparator);
        List<CandidatePerf> finalists = pickWithStageDiversity(explored, keep, config.maxPerStage(), fingerprintFn);
        for (CandidatePerf perf : finalists) {
            out.add(perf.candidate());
        }
    }

    private static List<CandidatePerf> pickWithStageDiversity(
            List<CandidatePerf> candidates,
            int limit,
            int maxPerStage,
            Function<OptimizerCandidate, String> fingerprintFn
    ) {
        List<CandidatePerf> out = new ArrayList<>(Math.min(limit, candidates.size()));
        Map<String, Integer> counts = new HashMap<>();
        for (CandidatePerf perf : candidates) {
            String stageOrder = perf.stageOrderKey();
            int used = counts.getOrDefault(stageOrder, 0);
            if (used >= maxPerStage) {
                continue;
            }
            out.add(perf);
            counts.put(stageOrder, used + 1);
            if (out.size() >= limit) {
                break;
            }
        }
        if (out.size() < limit) {
            Set<String> seen = new HashSet<>();
            for (CandidatePerf perf : out) {
                seen.add(fingerprintFn.apply(perf.candidate()));
            }
            for (CandidatePerf perf : candidates) {
                String fp = fingerprintFn.apply(perf.candidate());
                if (seen.add(fp)) {
                    out.add(perf);
                    if (out.size() >= limit) {
                        break;
                    }
                }
            }
        }
        return out;
    }

    private static List<OptimizationStage> parseStageOrderKey(String stageOrderKey) {
        if (stageOrderKey == null || stageOrderKey.isBlank() || "NONE".equals(stageOrderKey)) {
            return new ArrayList<>();
        }
        List<OptimizationStage> out = new ArrayList<>();
        for (String token : stageOrderKey.split(",")) {
            String trimmed = token.trim();
            if (!trimmed.isEmpty()) {
                out.add(OptimizationStage.valueOf(trimmed));
            }
        }
        return out;
    }

    private static String toStageOrderKey(List<OptimizationStage> nonMemStages, boolean hasMem) {
        List<OptimizationStage> all = new ArrayList<>(nonMemStages);
        if (hasMem) {
            all.add(OptimizationStage.MEM);
        }
        if (all.isEmpty()) {
            return "NONE";
        }
        StringBuilder sb = new StringBuilder(32);
        for (OptimizationStage stage : all) {
            if (sb.length() > 0) {
                sb.append(',');
            }
            sb.append(stage.name());
        }
        return sb.toString();
    }
}
