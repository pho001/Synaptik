package Benchmark.autotune;

public record AutoTuneBestResults(AutoTuneResult training, AutoTuneResult inference) {

    public AutoTuneBestResults update(CandidatePerf perf) {
        AutoTuneResult nextTraining = training;
        AutoTuneResult nextInference = inference;

        AutoTuneResult candidateTraining = AutoTuneResult.forTraining(perf);
        if (nextTraining == null || candidateTraining.score() < nextTraining.score()) {
            nextTraining = candidateTraining;
        }

        AutoTuneResult candidateInference = AutoTuneResult.forInference(perf);
        if (nextInference == null || candidateInference.score() < nextInference.score()) {
            nextInference = candidateInference;
        }

        return new AutoTuneBestResults(nextTraining, nextInference);
    }
}
