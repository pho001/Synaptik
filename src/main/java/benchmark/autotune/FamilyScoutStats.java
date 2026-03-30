package benchmark.autotune;

import benchmark.OptimizerCandidate;

import java.util.List;

public final class FamilyScoutStats {
    private final String stageOrder;
    private final List<OptimizerCandidate> group;
    private final List<OptimizerCandidate> samplePool;
    private final RunningEstimate training;
    private final RunningEstimate inference;
    private int nextSampleIndex;

    public FamilyScoutStats(String stageOrder, List<OptimizerCandidate> group, List<OptimizerCandidate> samplePool, double confidenceZ) {
        this.stageOrder = stageOrder;
        this.group = group;
        this.samplePool = samplePool;
        this.training = new RunningEstimate(confidenceZ);
        this.inference = new RunningEstimate(confidenceZ);
    }

    public String stageOrder() {
        return stageOrder;
    }

    public List<OptimizerCandidate> group() {
        return group;
    }

    public boolean hasRemainingSamples() {
        return nextSampleIndex < samplePool.size();
    }

    public OptimizerCandidate nextSample() {
        return samplePool.get(nextSampleIndex++);
    }

    public void record(CandidatePerf perf) {
        training.add(perf.trainingScore());
        inference.add(perf.inferenceScore());
    }

    public int samples() {
        return training.count();
    }

    public double trainingMean() {
        return training.mean();
    }

    public double trainingOptimistic() {
        return training.optimistic();
    }

    public double trainingConservative() {
        return training.conservative();
    }

    public double inferenceMean() {
        return inference.mean();
    }

    public double inferenceOptimistic() {
        return inference.optimistic();
    }

    public double inferenceConservative() {
        return inference.conservative();
    }

    public double combinedOptimistic() {
        return Math.min(trainingOptimistic(), inferenceOptimistic());
    }
}
