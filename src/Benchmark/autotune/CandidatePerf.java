package Benchmark.autotune;

import Benchmark.OptimizerCandidate;
import Benchmark.measure.MeasurementObjective;
import Benchmark.measure.MeasurementScoring;

public final class CandidatePerf {
    private final OptimizerCandidate candidate;
    private final String stageOrderKey;
    private final CoarseKnobSignature coarseKnobSignature;
    private final int graphInfSize;
    private final int graphTrnSize;
    private final double forwardMs;
    private final double trainMs;
    private final double broadcastMs;

    public CandidatePerf(
            OptimizerCandidate candidate,
            String stageOrderKey,
            CoarseKnobSignature coarseKnobSignature,
            int graphInfSize,
            int graphTrnSize,
            double forwardMs,
            double trainMs,
            double broadcastMs
    ) {
        this.candidate = candidate;
        this.stageOrderKey = stageOrderKey;
        this.coarseKnobSignature = coarseKnobSignature;
        this.graphInfSize = graphInfSize;
        this.graphTrnSize = graphTrnSize;
        this.forwardMs = forwardMs;
        this.trainMs = trainMs;
        this.broadcastMs = broadcastMs;
    }

    public OptimizerCandidate candidate() {
        return candidate;
    }

    public String stageOrderKey() {
        return stageOrderKey;
    }

    public CoarseKnobSignature coarseKnobSignature() {
        return coarseKnobSignature;
    }

    public int graphInfSize() {
        return graphInfSize;
    }

    public int graphTrnSize() {
        return graphTrnSize;
    }

    public double forwardMs() {
        return forwardMs;
    }

    public double trainMs() {
        return trainMs;
    }

    public double broadcastMs() {
        return broadcastMs;
    }

    public double trainingScore() {
        return MeasurementScoring.score(
                forwardMs,
                trainMs,
                broadcastMs,
                graphInfSize,
                graphTrnSize,
                MeasurementObjective.TRAINING
        );
    }

    public double inferenceScore() {
        return MeasurementScoring.score(
                forwardMs,
                trainMs,
                broadcastMs,
                graphInfSize,
                graphTrnSize,
                MeasurementObjective.INFERENCE
        );
    }
}
