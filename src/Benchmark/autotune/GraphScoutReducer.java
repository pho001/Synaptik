package Benchmark.autotune;

import Benchmark.OptimizerCandidate;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.LongSupplier;

public final class GraphScoutReducer {
    private GraphScoutReducer() {}

    public static List<OptimizerCandidate> reduceCandidates(
            List<OptimizerCandidate> candidates,
            GraphScoutConfig config,
            CandidateEvalCache evalCache,
            CandidatePerfSource perfSource,
            Function<OptimizerCandidate, String> fingerprintFn,
            Consumer<String> logger,
            LongSupplier nanoTimeSource
    ) {
        if (candidates.isEmpty()) {
            return candidates;
        }
        Consumer<String> effectiveLogger = logger == null ? msg -> {} : logger;
        LongSupplier clock = nanoTimeSource == null ? System::nanoTime : nanoTimeSource;

        Map<String, List<OptimizerCandidate>> groups = new HashMap<>();
        for (OptimizerCandidate candidate : candidates) {
            groups.computeIfAbsent(AutotuneSearchSupport.stageOrderKey(candidate), key -> new ArrayList<>()).add(candidate);
        }

        effectiveLogger.accept("Graph-scout search enabled"
                + " | groups=" + groups.size()
                + ", stageScoutSamplesPerRound=" + config.stageScoutSamplePerRound()
                + ", stageScoutMaxSamples=" + config.stageScoutMaxSamplesPerStage()
                + ", stageScoutMaxRounds=" + config.stageScoutMaxRounds()
                + ", stageScoutMinActiveFamilies=" + config.stageScoutMinActiveFamilies()
                + ", stageScoutConfidenceZ=" + String.format(java.util.Locale.US, "%.2f", config.confidenceZ())
                + ", prescreenWarmup=" + config.prescreenWarmupIters()
                + ", prescreenMeasure=" + config.prescreenMeasureIters()
                + ", prescreenKeepTrain=" + config.prescreenKeepTrain()
                + ", prescreenKeepInf=" + config.prescreenKeepInference()
                + ", prescreenDiversitySeedsPerFamily=" + config.prescreenDiversitySeedsPerFamily());

        List<String> sortedKeys = new ArrayList<>(groups.keySet());
        Collections.sort(sortedKeys);
        List<FamilyScoutStats> families = new ArrayList<>(sortedKeys.size());
        for (String key : sortedKeys) {
            List<OptimizerCandidate> group = groups.get(key);
            List<OptimizerCandidate> samplePool = AutotuneSearchSupport.sampleCandidatesEvenly(group, config.stageScoutMaxSamplesPerStage());
            families.add(new FamilyScoutStats(key, group, samplePool, config.confidenceZ()));
        }

        List<FamilyScoutStats> activeFamilies = new ArrayList<>(families);
        int round = 0;
        while (round < config.stageScoutMaxRounds() && !activeFamilies.isEmpty()) {
            round++;
            int groupIndex = 0;
            int sampledFamilies = 0;
            for (FamilyScoutStats family : activeFamilies) {
                groupIndex++;
                int evaluated = 0;
                while (evaluated < config.stageScoutSamplePerRound() && family.hasRemainingSamples()) {
                    OptimizerCandidate sampledCandidate = family.nextSample();
                    CandidatePerf perf = perfSource.measure(
                            sampledCandidate,
                            config.stageScoutWarmupIters(),
                            config.stageScoutMeasureIters(),
                            "SCOUT",
                            evalCache
                    );
                    family.record(perf);
                    evaluated++;
                }
                if (evaluated > 0) {
                    sampledFamilies++;
                }
                effectiveLogger.accept("Stage scout round " + round
                        + " | family=" + groupIndex + "/" + activeFamilies.size()
                        + " | stageOrder=" + family.stageOrder()
                        + " | n=" + family.samples()
                        + " | trainMean=" + String.format(java.util.Locale.US, "%.4f", family.trainingMean())
                        + " | trainCI=[" + String.format(java.util.Locale.US, "%.4f", family.trainingOptimistic()) + ", "
                        + String.format(java.util.Locale.US, "%.4f", family.trainingConservative()) + "]"
                        + " | infMean=" + String.format(java.util.Locale.US, "%.4f", family.inferenceMean())
                        + " | infCI=[" + String.format(java.util.Locale.US, "%.4f", family.inferenceOptimistic()) + ", "
                        + String.format(java.util.Locale.US, "%.4f", family.inferenceConservative()) + "]");
            }
            if (sampledFamilies == 0) {
                break;
            }
            if (activeFamilies.size() <= config.stageScoutMinActiveFamilies()) {
                break;
            }
            int targetActive = Math.max(
                    config.stageScoutMinActiveFamilies(),
                    (activeFamilies.size() + 1) / 2
            );
            activeFamilies = AutotuneSearchSupport.pruneFamilyScoutRound(
                    activeFamilies,
                    targetActive,
                    Math.max(config.stageScoutTopTrain(), Math.max(1, targetActive / 2)),
                    Math.max(config.stageScoutTopInference(), Math.max(1, targetActive / 2))
            );
            effectiveLogger.accept("Stage scout round " + round
                    + " selected activeFamilies=" + activeFamilies.size()
                    + " | target=" + targetActive);
        }

        Set<String> selectedStageOrders = new LinkedHashSet<>();
        List<FamilyScoutStats> finalByTrain = new ArrayList<>(activeFamilies);
        finalByTrain.sort((a, b) -> Double.compare(a.trainingConservative(), b.trainingConservative()));
        List<FamilyScoutStats> finalByInf = new ArrayList<>(activeFamilies);
        finalByInf.sort((a, b) -> Double.compare(a.inferenceConservative(), b.inferenceConservative()));
        for (int i = 0; i < Math.min(config.stageScoutTopTrain(), finalByTrain.size()); i++) {
            selectedStageOrders.add(finalByTrain.get(i).stageOrder());
        }
        for (int i = 0; i < Math.min(config.stageScoutTopInference(), finalByInf.size()); i++) {
            selectedStageOrders.add(finalByInf.get(i).stageOrder());
        }
        for (FamilyScoutStats family : activeFamilies) {
            selectedStageOrders.add(family.stageOrder());
        }

        List<OptimizerCandidate> filtered = new ArrayList<>();
        for (OptimizerCandidate candidate : candidates) {
            if (selectedStageOrders.contains(AutotuneSearchSupport.stageOrderKey(candidate))) {
                filtered.add(candidate);
            }
        }

        effectiveLogger.accept("Stage scout selected orders=" + selectedStageOrders.size()
                + " | orders=" + selectedStageOrders
                + " | candidates=" + filtered.size() + "/" + candidates.size());

        List<OptimizerCandidate> stageCapped = AutotuneSearchSupport.capCandidatesPerStageOrder(filtered, config.prescreenMaxPerStageOrder());
        if (stageCapped.size() != filtered.size()) {
            effectiveLogger.accept("Stage scout per-order cap="
                    + config.prescreenMaxPerStageOrder()
                    + " | candidates=" + stageCapped.size()
                    + "/" + filtered.size());
        }
        filtered = stageCapped;

        if (filtered.size() <= Math.max(config.prescreenKeepTrain(), config.prescreenKeepInference())) {
            return filtered;
        }

        List<CandidatePerf> prescreen = new ArrayList<>(filtered.size());
        long prescreenStartNs = clock.getAsLong();
        long lastLogNs = prescreenStartNs;
        for (int i = 0; i < filtered.size(); i++) {
            OptimizerCandidate candidate = filtered.get(i);
            long rowStartNs = clock.getAsLong();
            CandidatePerf perf = perfSource.measure(
                    candidate,
                    config.prescreenWarmupIters(),
                    config.prescreenMeasureIters(),
                    "PRESCREEN",
                    evalCache
            );
            prescreen.add(perf);
            long now = clock.getAsLong();
            int processed = i + 1;
            boolean byCount = processed <= 1 || processed % config.progressLogEvery() == 0;
            boolean byTime = ((now - lastLogNs) / 1_000_000L) >= config.progressMinIntervalMs();
            if (byCount || byTime) {
                lastLogNs = now;
                double rowMs = (now - rowStartNs) / 1_000_000.0;
                double avgRowMs = (now - prescreenStartNs) / 1_000_000.0 / processed;
                double etaSec = (avgRowMs / 1000.0) * Math.max(0, filtered.size() - processed);
                effectiveLogger.accept("Prescreen progress: " + processed + "/" + filtered.size()
                        + " | rowMs=" + fmtMillis(rowMs)
                        + ", avgRowMs=" + fmtMillis(avgRowMs)
                        + ", etaSec=" + fmtSeconds(etaSec)
                        + " | candidate=" + candidate.name());
            }
        }

        Set<OptimizerCandidate> survivors = new LinkedHashSet<>();
        List<CandidatePerf> prescreenByTrain = new ArrayList<>(prescreen);
        prescreenByTrain.sort((a, b) -> Double.compare(a.trainingScore(), b.trainingScore()));
        List<CandidatePerf> prescreenByInf = new ArrayList<>(prescreen);
        prescreenByInf.sort((a, b) -> Double.compare(a.inferenceScore(), b.inferenceScore()));
        for (int i = 0; i < Math.min(config.prescreenKeepTrain(), prescreenByTrain.size()); i++) {
            survivors.add(prescreenByTrain.get(i).candidate());
        }
        for (int i = 0; i < Math.min(config.prescreenKeepInference(), prescreenByInf.size()); i++) {
            survivors.add(prescreenByInf.get(i).candidate());
        }
        if (config.prescreenDiversitySeedsPerFamily() > 0) {
            Map<String, List<CandidatePerf>> byFamily = new HashMap<>();
            for (CandidatePerf perf : prescreen) {
                byFamily.computeIfAbsent(AutotuneSearchSupport.stageOrderKey(perf.candidate()), key -> new ArrayList<>()).add(perf);
            }
            for (String stageOrder : selectedStageOrders) {
                List<CandidatePerf> family = byFamily.get(stageOrder);
                if (family == null || family.isEmpty()) {
                    continue;
                }
                family.sort((a, b) -> Double.compare(
                        Math.min(a.trainingScore(), a.inferenceScore()),
                        Math.min(b.trainingScore(), b.inferenceScore())
                ));
                for (int i = 0; i < Math.min(config.prescreenDiversitySeedsPerFamily(), family.size()); i++) {
                    survivors.add(family.get(i).candidate());
                }
            }
        }

        effectiveLogger.accept("Prescreen selected candidates=" + survivors.size()
                + "/" + filtered.size());

        List<OptimizerCandidate> beamCandidates = AutotuneSearchSupport.selectCandidatesViaBeam(
                prescreen,
                survivors,
                config.beamSearchConfig(),
                AutotuneSearchSupport::stageOrderNeighbors,
                fingerprintFn,
                msg -> effectiveLogger.accept(msg)
        );
        effectiveLogger.accept("Beam selected candidates=" + beamCandidates.size()
                + "/" + prescreen.size());
        return beamCandidates;
    }

    private static String fmtMillis(double value) {
        return String.format(java.util.Locale.US, "%.3f", value);
    }

    private static String fmtSeconds(double value) {
        return String.format(java.util.Locale.US, "%.1f", value);
    }
}
