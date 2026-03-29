package Benchmark.autotune;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

public final class CandidateGraphIndex {
    private final Map<String, CandidatePerf> byStageAndKnob = new HashMap<>();
    private final Map<String, List<CandidatePerf>> byStageOrder = new HashMap<>();
    private final Function<String, List<String>> stageOrderNeighborsFn;

    public CandidateGraphIndex(List<CandidatePerf> candidates, Function<String, List<String>> stageOrderNeighborsFn) {
        this.stageOrderNeighborsFn = stageOrderNeighborsFn;
        for (CandidatePerf perf : candidates) {
            byStageAndKnob.put(compositeKey(perf.stageOrderKey(), perf.coarseKnobSignature()), perf);
            byStageOrder.computeIfAbsent(perf.stageOrderKey(), key -> new ArrayList<>()).add(perf);
        }
    }

    public List<CandidatePerf> neighbors(CandidatePerf perf) {
        List<CandidatePerf> out = new ArrayList<>();
        for (String stageNeighbor : stageOrderNeighborsFn.apply(perf.stageOrderKey())) {
            CandidatePerf candidate = byStageAndKnob.get(compositeKey(stageNeighbor, perf.coarseKnobSignature()));
            if (candidate != null) {
                out.add(candidate);
            }
        }
        for (CandidatePerf candidate : byStageOrder.getOrDefault(perf.stageOrderKey(), List.of())) {
            if (candidate == perf) {
                continue;
            }
            if (perf.coarseKnobSignature().distance(candidate.coarseKnobSignature()) == 1) {
                out.add(candidate);
            }
        }
        return out;
    }

    private static String compositeKey(String stageOrder, CoarseKnobSignature signature) {
        return stageOrder + "||" + signature.strictCseSafety()
                + "||" + signature.fuseProfileKey()
                + "||" + signature.kernelProfileKey()
                + "||" + signature.blasProfileKey();
    }
}
