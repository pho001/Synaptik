package Numerics;

import Backend.ComputeEngine;
import Benchmark.OptimizationStage;
import Benchmark.OptimizerBuilder;
import Benchmark.OptimizerCandidate;
import Benchmark.TuningKnobs;
import Benchmark.scenario.BenchmarkGraphRecipes;
import Benchmark.scenario.ScenarioTensorFactory;
import Graph.optimizer.GraphOptimizer;
import Tensor.DataType;
import Tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public final class NumericsHarness {
    public static final class Config {
        public DataType dtype = DataType.FLOAT32;
        public int size = 200_000;
        public int graphBlocks = 6;
        public int b0 = 128;
        public int b1 = 8;
        public int f = 128;
        public long seed = 42L;
    }

    private final Config config;

    public NumericsHarness(Config config) {
        this.config = config;
    }

    public NumericsReport run(OptimizerCandidate a, OptimizerCandidate b, NumericsPolicy policy) {
        InputSet input = new InputSet(config.size, config.seed);
        OutputSet outA = runCandidate(a, input);
        OutputSet outB = runCandidate(b, input);

        NumericsMetrics.SignalMetrics mOut = NumericsMetrics.compare(outA.out, outB.out, config.dtype);
        NumericsMetrics.SignalMetrics mGradA = NumericsMetrics.compare(outA.gradA, outB.gradA, config.dtype);
        NumericsMetrics.SignalMetrics mGradB = NumericsMetrics.compare(outA.gradB, outB.gradB, config.dtype);
        NumericsMetrics.SignalMetrics mGradC = NumericsMetrics.compare(outA.gradC, outB.gradC, config.dtype);
        NumericsMetrics.SignalMetrics mBroadcast = NumericsMetrics.compare(outA.broadcastOut, outB.broadcastOut, config.dtype);
        NumericsMetrics.AggregateMetrics agg = NumericsMetrics.aggregate(mOut, mGradA, mGradB, mGradC, mBroadcast);

        return new NumericsReport(
                "benchmark-like",
                a.name(),
                b.name(),
                mOut,
                mGradA,
                mGradB,
                mGradC,
                mBroadcast,
                agg,
                policy.evaluate(agg)
        );
    }

    public OptimizerCandidate candidate(String name, List<OptimizationStage> stages) {
        return new OptimizerCandidate(name, stages, TuningKnobs.trainingDefaults());
    }

    private OutputSet runCandidate(OptimizerCandidate candidate, InputSet input) {
        ComputeEngine.setCpuKernelConfig(candidate.knobs().kernelConfig().cpu());
        GraphOptimizer optimizer = OptimizerBuilder.build(candidate);

        Tensor A = ScenarioTensorFactory.flatTensor("A", input.baseA, true, config.dtype);
        Tensor B = ScenarioTensorFactory.flatTensor("B", input.baseB, true, config.dtype);
        Tensor C = ScenarioTensorFactory.flatTensor("C", input.baseC, true, config.dtype);

        Tensor linearIn = ScenarioTensorFactory.prefixTensorWrap("LIN_IN", input.baseA, true, config.dtype, 64, 64);
        Tensor w1 = ScenarioTensorFactory.prefixTensorWrap("LIN_W1", input.baseB, false, config.dtype, 64, 64);
        Tensor b1 = ScenarioTensorFactory.prefixTensorWrap("LIN_B1", input.baseC, false, config.dtype, 64, 64);
        Tensor w2 = ScenarioTensorFactory.prefixTensorWrap("LIN_W2", input.baseC, false, config.dtype, 64, 64);
        Tensor b2 = ScenarioTensorFactory.prefixTensorWrap("LIN_B2", input.baseA, false, config.dtype, 64, 64);
        Tensor w3 = ScenarioTensorFactory.prefixTensorWrap("LIN_W3", input.baseA, false, config.dtype, 64, 64);
        Tensor b3 = ScenarioTensorFactory.prefixTensorWrap("LIN_B3", input.baseB, false, config.dtype, 64, 64);

        Tensor out = BenchmarkGraphRecipes.buildOptimizerBenchmarkGraph(
                A, B, C, linearIn, w1, b1, w2, b2, w3, b3, config.graphBlocks
        );
        out.compute(optimizer);
        out.getCompiledGraph().setTrainingModeOn();
        out.compute(optimizer);

        Tensor BA = ScenarioTensorFactory.prefixTensorWrap("BA", input.baseA, false, config.dtype, config.b0, 1, config.f);
        Tensor BB = ScenarioTensorFactory.prefixTensorWrap("BB", input.baseB, false, config.dtype, 1, config.b1, config.f);
        Tensor BC = ScenarioTensorFactory.prefixTensorWrap("BC", input.baseC, false, config.dtype, config.b0, config.b1, config.f);
        Tensor broadcastOut = BenchmarkGraphRecipes.buildBroadcastGraph(BA, BB, BC);
        broadcastOut.compute(optimizer);

        return new OutputSet(
                out.toDoubleArrayCopy().clone(),
                A.getGradient().toDoubleArrayCopy().clone(),
                B.getGradient().toDoubleArrayCopy().clone(),
                C.getGradient().toDoubleArrayCopy().clone(),
                broadcastOut.toDoubleArrayCopy().clone()
        );
    }

    private static final class InputSet {
        final double[] baseA;
        final double[] baseB;
        final double[] baseC;

        private InputSet(int size, long seed) {
            Random rng = new Random(seed);
            this.baseA = randomArray(size, rng);
            this.baseB = randomArray(size, rng);
            this.baseC = randomArray(size, rng);
        }

        private static double[] randomArray(int size, Random rng) {
            double[] out = new double[size];
            for (int i = 0; i < size; i++) {
                out[i] = rng.nextDouble();
            }
            return out;
        }
    }

    private static final class OutputSet {
        final double[] out;
        final double[] gradA;
        final double[] gradB;
        final double[] gradC;
        final double[] broadcastOut;

        private OutputSet(double[] out, double[] gradA, double[] gradB, double[] gradC, double[] broadcastOut) {
            this.out = out;
            this.gradA = gradA;
            this.gradB = gradB;
            this.gradC = gradC;
            this.broadcastOut = broadcastOut;
        }
    }

    public static List<OptimizationStage> parseStages(String stageSpec) {
        String spec = stageSpec == null ? "" : stageSpec.trim();
        if (spec.isEmpty() || "NONE".equalsIgnoreCase(spec)) {
            return List.of();
        }
        String[] parts = spec.split("[+,]");
        List<OptimizationStage> out = new ArrayList<>();
        for (String p : parts) {
            String s = p.trim();
            if (s.isEmpty()) continue;
            out.add(OptimizationStage.valueOf(s.toUpperCase()));
        }
        return out;
    }
}
