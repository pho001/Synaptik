package Benchmark.measure;

public interface MeasuredBenchmarkScenario {
    int graphSize();

    void setTrainingMode(boolean trainingMode);

    void compute();
}
