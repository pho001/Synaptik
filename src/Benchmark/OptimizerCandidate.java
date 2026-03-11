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
        this.stageOrder = Collections.unmodifiableList(new ArrayList<>(stageOrder));
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
}
