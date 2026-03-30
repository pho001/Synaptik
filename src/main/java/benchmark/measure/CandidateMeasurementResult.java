package benchmark.measure;

import benchmark.OptimizerCandidate;

public record CandidateMeasurementResult(
        OptimizerCandidate candidate,
        int graphInfSize,
        int graphTrnSize,
        double forwardMs,
        double trainMs,
        double broadcastMs
) {
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
