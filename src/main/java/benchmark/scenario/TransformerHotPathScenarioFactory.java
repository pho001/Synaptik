package benchmark.scenario;

import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.WorkloadKind;
import config.profile.WorkloadProfile;
import graph.CompiledGraph;
import graph.execution.PreparedExecution;
import tensor.AttentionOptions;
import tensor.DataType;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

public final class TransformerHotPathScenarioFactory {
    private TransformerHotPathScenarioFactory() {
    }

    public static List<PreparedHotPathScenario> create(ExecutionProfile profile) {
        if (profile == null) {
            throw new IllegalArgumentException("profile cannot be null");
        }
        WorkloadProfile workload = profile.workload();
        if (workload.kind() != WorkloadKind.TRANSFORMER_HOT_PATH) {
            throw new IllegalArgumentException("Transformer hot-path scenario factory requires TRANSFORMER_HOT_PATH workload profile.");
        }

        List<PreparedHotPathScenario> scenarios = new ArrayList<>();
        scenarios.add(prepareScenario("batched_matmul_qk", buildBatchedMatMul(workload, profile.dataType(), profile.mode()), profile));
        scenarios.add(prepareScenario("score_mask_softmax", buildScoreMaskSoftmax(workload, profile.dataType(), profile.mode()), profile));
        scenarios.add(prepareScenario("weights_value_matmul", buildWeightsValueMatMul(workload, profile.dataType(), profile.mode()), profile));
        scenarios.add(prepareScenario("attention", buildAttention(workload, profile.dataType(), profile.mode()), profile));
        scenarios.add(prepareScenario("layer_norm", buildLayerNorm(workload, profile.dataType(), profile.mode()), profile));
        scenarios.add(prepareScenario("rms_norm", buildRmsNorm(workload, profile.dataType(), profile.mode()), profile));
        scenarios.add(prepareScenario("batch_norm", buildBatchNorm(workload, profile.dataType(), profile.mode()), profile));
        scenarios.add(prepareScenario("transformer_block", buildTransformerBlock(workload, profile.dataType(), profile.mode()), profile));
        return List.copyOf(scenarios);
    }

    private static PreparedHotPathScenario prepareScenario(String name, Tensor root, ExecutionProfile profile) {
        CompiledGraph compiledGraph = CompiledGraph.compile(root, profile.optimizer());
        PreparedExecution execution = compiledGraph.prepare(profile.runtime());
        return new PreparedHotPathScenario(name, root, execution, profile.mode());
    }

    private static Tensor buildBatchedMatMul(WorkloadProfile workload, DataType dataType, ExecutionMode mode) {
        Tensor q = tensor("Q_MATMUL", 11, dataType, mode == ExecutionMode.FORWARD_BACKWARD,
                workload.batch(), workload.heads(), workload.seqLen(), workload.headDim());
        Tensor k = tensor("K_MATMUL", 12, dataType, mode == ExecutionMode.FORWARD_BACKWARD,
                workload.batch(), workload.heads(), workload.seqLen(), workload.headDim());
        Tensor out = q.matmul(swapLastTwoAxes(k));
        return finalizeRoot(out, mode);
    }

    private static Tensor buildAttention(WorkloadProfile workload, DataType dataType, ExecutionMode mode) {
        Tensor q = tensor("Q_ATTN", 21, dataType, mode == ExecutionMode.FORWARD_BACKWARD,
                workload.batch(), workload.heads(), workload.seqLen(), workload.headDim());
        Tensor k = tensor("K_ATTN", 22, dataType, mode == ExecutionMode.FORWARD_BACKWARD,
                workload.batch(), workload.heads(), workload.seqLen(), workload.headDim());
        Tensor v = tensor("V_ATTN", 23, dataType, mode == ExecutionMode.FORWARD_BACKWARD,
                workload.batch(), workload.heads(), workload.seqLen(), workload.valueDim());
        Tensor out = q.scaledDotProductAttention(k, v,
                new AttentionOptions(workload.causal(), null));
        return finalizeRoot(out, mode);
    }

    private static Tensor buildScoreMaskSoftmax(WorkloadProfile workload, DataType dataType, ExecutionMode mode) {
        boolean requiresGrad = mode == ExecutionMode.FORWARD_BACKWARD;
        Tensor q = tensor("Q_SCORE", 24, dataType, requiresGrad,
                workload.batch(), workload.heads(), workload.seqLen(), workload.headDim());
        Tensor k = tensor("K_SCORE", 25, dataType, requiresGrad,
                workload.batch(), workload.heads(), workload.seqLen(), workload.headDim());
        Tensor scores = q.matmul(swapLastTwoAxes(k)).mul(1.0d / Math.sqrt(workload.headDim()));
        if (workload.causal()) {
            Tensor mask = causalMask(workload.batch(), workload.heads(), workload.seqLen());
            scores = Tensor.where(mask, scores, Tensor.scalar(maskFillValue(dataType), dataType));
        }
        Tensor weights = scores.softmax(scores.getShapeUnsafe().length - 1);
        return finalizeRoot(weights, mode);
    }

    private static Tensor buildWeightsValueMatMul(WorkloadProfile workload, DataType dataType, ExecutionMode mode) {
        boolean requiresGrad = mode == ExecutionMode.FORWARD_BACKWARD;
        Tensor scores = tensor("SCORES_WV", 26, dataType, requiresGrad,
                workload.batch(), workload.heads(), workload.seqLen(), workload.seqLen());
        Tensor weights = scores.softmax(scores.getShapeUnsafe().length - 1);
        Tensor value = tensor("V_WV", 27, dataType, requiresGrad,
                workload.batch(), workload.heads(), workload.seqLen(), workload.valueDim());
        Tensor out = weights.matmul(value);
        return finalizeRoot(out, mode);
    }

    private static Tensor buildLayerNorm(WorkloadProfile workload, DataType dataType, ExecutionMode mode) {
        Tensor x = tensor("X_LN", 31, dataType, mode == ExecutionMode.FORWARD_BACKWARD,
                workload.batch(), workload.seqLen(), workload.modelDim());
        Tensor gamma = tensor("GAMMA_LN", 32, dataType, mode == ExecutionMode.FORWARD_BACKWARD,
                workload.modelDim());
        Tensor beta = tensor("BETA_LN", 33, dataType, mode == ExecutionMode.FORWARD_BACKWARD,
                workload.modelDim());
        Tensor out = x.layerNorm(gamma, beta, 1e-5);
        return finalizeRoot(out, mode);
    }

    private static Tensor buildRmsNorm(WorkloadProfile workload, DataType dataType, ExecutionMode mode) {
        Tensor x = tensor("X_RMS", 41, dataType, mode == ExecutionMode.FORWARD_BACKWARD,
                workload.batch(), workload.seqLen(), workload.modelDim());
        Tensor gamma = tensor("GAMMA_RMS", 42, dataType, mode == ExecutionMode.FORWARD_BACKWARD,
                workload.modelDim());
        Tensor out = x.rmsNorm(gamma, 1e-5);
        return finalizeRoot(out, mode);
    }

    private static Tensor buildBatchNorm(WorkloadProfile workload, DataType dataType, ExecutionMode mode) {
        Tensor x = tensor("X_BN", 51, dataType, mode == ExecutionMode.FORWARD_BACKWARD,
                workload.batch(), workload.modelDim(), workload.seqLen(), 1);
        Tensor gamma = tensor("GAMMA_BN", 52, dataType, mode == ExecutionMode.FORWARD_BACKWARD,
                workload.modelDim());
        Tensor beta = tensor("BETA_BN", 53, dataType, mode == ExecutionMode.FORWARD_BACKWARD,
                workload.modelDim());
        Tensor out = x.batchNorm(gamma, beta, 1, 1e-5);
        return finalizeRoot(out, mode);
    }

    private static Tensor buildTransformerBlock(WorkloadProfile workload, DataType dataType, ExecutionMode mode) {
        boolean requiresGrad = mode == ExecutionMode.FORWARD_BACKWARD;
        int modelDim = workload.modelDim();

        Tensor x = tensor("X_BLOCK", 61, dataType, requiresGrad, workload.batch(), workload.seqLen(), modelDim);
        Tensor q = tensor("Q_BLOCK", 62, dataType, requiresGrad, workload.batch(), workload.heads(), workload.seqLen(), workload.headDim());
        Tensor k = tensor("K_BLOCK", 63, dataType, requiresGrad, workload.batch(), workload.heads(), workload.seqLen(), workload.headDim());
        Tensor v = tensor("V_BLOCK", 64, dataType, requiresGrad, workload.batch(), workload.heads(), workload.seqLen(), workload.valueDim());

        Tensor attn = q.scaledDotProductAttention(k, v, new AttentionOptions(workload.causal(), null));
        Tensor merged = attn.permute(0, 2, 1, 3).reshape(workload.batch(), workload.seqLen(), modelDim);
        Tensor residual = x.add(merged);

        Tensor lnGamma = tensor("LN_GAMMA", 65, dataType, requiresGrad, modelDim);
        Tensor lnBeta = tensor("LN_BETA", 66, dataType, requiresGrad, modelDim);
        Tensor norm = residual.layerNorm(lnGamma, lnBeta, 1e-5);

        Tensor w1 = tensor("FF_W1", 67, dataType, requiresGrad, modelDim, workload.ffHiddenDim());
        Tensor b1 = tensor("FF_B1", 68, dataType, requiresGrad, workload.ffHiddenDim());
        Tensor w2 = tensor("FF_W2", 69, dataType, requiresGrad, workload.ffHiddenDim(), modelDim);
        Tensor b2 = tensor("FF_B2", 70, dataType, requiresGrad, modelDim);

        Tensor ff = norm.linear(w1, b1).relu().linear(w2, b2);
        Tensor rmsGamma = tensor("RMS_GAMMA", 71, dataType, requiresGrad, modelDim);
        Tensor out = ff.add(residual).rmsNorm(rmsGamma, 1e-5);
        return finalizeRoot(out, mode);
    }

    private static Tensor finalizeRoot(Tensor out, ExecutionMode mode) {
        return mode == ExecutionMode.FORWARD_BACKWARD ? out.sum() : out.sum();
    }

    private static Tensor swapLastTwoAxes(Tensor tensor) {
        int rank = tensor.getShapeUnsafe().length;
        if (rank == 2) {
            return tensor.transpose();
        }
        int[] axes = new int[rank];
        for (int i = 0; i < rank; i++) {
            axes[i] = i;
        }
        int tmp = axes[rank - 1];
        axes[rank - 1] = axes[rank - 2];
        axes[rank - 2] = tmp;
        return tensor.permute(axes);
    }

    private static Tensor causalMask(int batch, int heads, int seqLen) {
        int[] shape = new int[]{batch, heads, seqLen, seqLen};
        int flat = batch * heads * seqLen * seqLen;
        byte[] data = new byte[flat];
        int offset = 0;
        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < heads; h++) {
                for (int q = 0; q < seqLen; q++) {
                    for (int k = 0; k < seqLen; k++) {
                        data[offset++] = (byte) (k <= q ? 1 : 0);
                    }
                }
            }
        }
        return new Tensor(data, shape, null, "CAUSAL_MASK", DataType.BOOL);
    }

    private static double maskFillValue(DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> -1.0e30d;
            case FLOAT32 -> -1.0e9d;
            case FLOAT16 -> -65504.0d;
            case INT32, BOOL -> throw new IllegalArgumentException("mask fill requires floating dtype");
        };
    }

    private static Tensor tensor(String label, int seed, DataType dataType, boolean requiresGrad, int... shape) {
        double[] data = randomData(flatSize(shape), seed);
        return ScenarioTensorFactory.shapedTensor(label, data, requiresGrad, dataType, shape);
    }

    private static int flatSize(int[] shape) {
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        return size;
    }

    private static double[] randomData(int size, int seed) {
        java.util.Random random = new java.util.Random(seed);
        double[] out = new double[size];
        for (int i = 0; i < size; i++) {
            out[i] = Math.sin(i * 0.013 + seed * 0.1) + (random.nextDouble() - 0.5) * 0.2;
        }
        return out;
    }
}
