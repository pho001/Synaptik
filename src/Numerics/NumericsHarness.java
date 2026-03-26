package Numerics;

import Backend.ComputeEngine;
import Benchmark.OptimizationStage;
import Benchmark.OptimizerBuilder;
import Benchmark.OptimizerCandidate;
import Benchmark.TuningKnobs;
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

        Tensor A = inputTensor("A", input.baseA, true);
        Tensor B = inputTensor("B", input.baseB, true);
        Tensor C = inputTensor("C", input.baseC, true);

        Tensor linearIn = inputTensorWithShapePrefix("LIN_IN", input.baseA, true, 64, 64);
        Tensor w1 = inputTensorWithShapePrefix("LIN_W1", input.baseB, false, 64, 64);
        Tensor b1 = inputTensorWithShapePrefix("LIN_B1", input.baseC, false, 64, 64);
        Tensor w2 = inputTensorWithShapePrefix("LIN_W2", input.baseC, false, 64, 64);
        Tensor b2 = inputTensorWithShapePrefix("LIN_B2", input.baseA, false, 64, 64);
        Tensor w3 = inputTensorWithShapePrefix("LIN_W3", input.baseA, false, 64, 64);
        Tensor b3 = inputTensorWithShapePrefix("LIN_B3", input.baseB, false, 64, 64);

        Tensor out = buildBenchmarkGraph(A, B, C, linearIn, w1, b1, w2, b2, w3, b3, config.graphBlocks);
        out.compute(optimizer);
        out.getCompiledGraph().setTrainingModeOn();
        out.compute(optimizer);

        Tensor BA = inputTensorWithShapePrefix("BA", input.baseA, false, config.b0, 1, config.f);
        Tensor BB = inputTensorWithShapePrefix("BB", input.baseB, false, 1, config.b1, config.f);
        Tensor BC = inputTensorWithShapePrefix("BC", input.baseC, false, config.b0, config.b1, config.f);
        Tensor broadcastOut = BA.add(BB).mul(BC).add(BA).sigmoid();
        broadcastOut.compute(optimizer);

        return new OutputSet(
                out.toDoubleArrayCopy().clone(),
                A.getGradient().toDoubleArrayCopy().clone(),
                B.getGradient().toDoubleArrayCopy().clone(),
                C.getGradient().toDoubleArrayCopy().clone(),
                broadcastOut.toDoubleArrayCopy().clone()
        );
    }

    private Tensor buildBenchmarkGraph(
            Tensor A,
            Tensor B,
            Tensor C,
            Tensor linearIn,
            Tensor w1,
            Tensor b1,
            Tensor w2,
            Tensor b2,
            Tensor w3,
            Tensor b3,
            int graphBlocks
    ) {
        int blocks = Math.max(1, graphBlocks);
        Tensor x = A.mul(0.50).add(B.mul(0.30)).sub(C.mul(0.20));
        for (int i = 0; i < blocks; i++) {
            x = x.mul(0.70).add(B.mul(0.20));
            x = x.sub(C.mul(0.10));
            x = x.add(A.mul(0.05));
            x = x.mul(0.95).add(B.mul(0.03)).sub(C.mul(0.02));
        }
        Tensor linear1 = linearIn.matmul(w1).add(b1);
        Tensor linear2 = linear1.matmul(w2).add(b2);
        Tensor linear3 = linear2.matmul(w3).add(b3);
        Tensor linearScalar = linear3.sum();
        return x.mul(x).add(B.mul(0.01)).add(linearScalar);
    }

    private Tensor inputTensor(String label, double[] data, boolean requiresGrad) {
        Tensor t = new Tensor(new int[]{data.length}, null, label, config.dtype);
        t.setData(data.clone());
        t.setRequiresGrad(requiresGrad);
        return t;
    }

    private Tensor inputTensor(String label, double[] data, boolean requiresGrad, int[] shape) {
        Tensor t = new Tensor(shape, null, label, config.dtype);
        t.setData(data.clone());
        t.setRequiresGrad(requiresGrad);
        return t;
    }

    private Tensor inputTensorWithShapePrefix(String label, double[] data, boolean requiresGrad, int... shape) {
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        if (size <= 0) {
            throw new IllegalArgumentException("Invalid shape size for " + label);
        }
        double[] sliced = new double[size];
        if (data.length >= size) {
            System.arraycopy(data, 0, sliced, 0, size);
        } else {
            for (int i = 0; i < size; i++) {
                sliced[i] = data[i % data.length];
            }
        }
        Tensor t = new Tensor(shape, null, label, config.dtype);
        t.setData(sliced);
        t.setRequiresGrad(requiresGrad);
        return t;
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
