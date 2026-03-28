package Benchmark;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public final class OptimizerCandidate {
    private final String name;
    private final List<OptimizationStage> stageOrder;
    private final TuningKnobs knobs;

    public OptimizerCandidate(String name, List<OptimizationStage> stageOrder, TuningKnobs knobs) {
        this.name = name;
        this.stageOrder = Collections.unmodifiableList(normalizeStageOrder(stageOrder));
        this.knobs = knobs;
    }

    public String name() {
        return name;
    }

    public List<OptimizationStage> stageOrder() {
        return stageOrder;
    }

    public TuningKnobs knobs() {
        return knobs;
    }

    private static List<OptimizationStage> normalizeStageOrder(List<OptimizationStage> stageOrder) {
        if (stageOrder == null || stageOrder.isEmpty()) {
            return List.of();
        }
        List<OptimizationStage> normalized = new ArrayList<>(stageOrder.size());
        boolean hasMem = false;
        for (OptimizationStage stage : stageOrder) {
            if (stage == null) {
                continue;
            }
            if (stage == OptimizationStage.MEM) {
                hasMem = true;
                continue;
            }
            normalized.add(stage);
        }
        if (hasMem) {
            normalized.add(OptimizationStage.MEM);
        }
        return normalized;
    }
}
