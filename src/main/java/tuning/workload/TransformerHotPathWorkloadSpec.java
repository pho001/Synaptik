package tuning.workload;

import config.profile.ExecutionProfile;
import config.profile.WorkloadKind;
import config.profile.WorkloadProfile;
import tensor.DataType;
import tensor.Tensor;
import tensor.options.AttentionOptions;
import tuning.validate.ValidationReference;
import tuning.validate.ValidationTarget;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class TransformerHotPathWorkloadSpec implements WorkloadSpec {
    private final String name;

    public TransformerHotPathWorkloadSpec(String name) {
        this.name = (name == null || name.isBlank()) ? "transformer_hot_path" : name;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public tuning.workload.WorkloadKind kind() {
        return tuning.workload.WorkloadKind.TRANSFORMER_HOT_PATH;
    }

    @Override
    public WorkloadInstance instantiate(WorkloadEnvironment environment) {
        ExecutionProfile profile = environment.profile();
        WorkloadProfile requested = profile.workload();
        WorkloadProfile workload = requested.kind() == WorkloadKind.NONE
                ? WorkloadProfile.transformerHotPathDefaults()
                : requested;
        if (workload.kind() != WorkloadKind.TRANSFORMER_HOT_PATH) {
            throw new IllegalArgumentException("TransformerHotPathWorkloadSpec requires profile.workload.kind == TRANSFORMER_HOT_PATH");
        }

        boolean requiresGrad = profile.mode() == backend.runtime.ExecutionMode.FORWARD_BACKWARD;
        DataType dataType = profile.dataType();
        int modelDim = workload.modelDim();

        Tensor x = tensor("X_BLOCK", 301, dataType, requiresGrad, workload.batch(), workload.seqLen(), modelDim);
        Tensor q = tensor("Q_BLOCK", 302, dataType, requiresGrad, workload.batch(), workload.heads(), workload.seqLen(), workload.headDim());
        Tensor k = tensor("K_BLOCK", 303, dataType, requiresGrad, workload.batch(), workload.heads(), workload.seqLen(), workload.headDim());
        Tensor v = tensor("V_BLOCK", 304, dataType, requiresGrad, workload.batch(), workload.heads(), workload.seqLen(), workload.valueDim());

        Tensor attn = q.scaledDotProductAttention(k, v, new AttentionOptions(workload.causal(), null));
        Tensor merged = attn.permute(0, 2, 1, 3).reshape(workload.batch(), workload.seqLen(), modelDim);
        Tensor residual = x.add(merged);

        Tensor lnGamma = tensor("LN_GAMMA", 305, dataType, requiresGrad, modelDim);
        Tensor lnBeta = tensor("LN_BETA", 306, dataType, requiresGrad, modelDim);
        Tensor norm = residual.layerNorm(lnGamma, lnBeta, 1e-5);

        Tensor w1 = tensor("FF_W1", 307, dataType, requiresGrad, modelDim, workload.ffHiddenDim());
        Tensor b1 = tensor("FF_B1", 308, dataType, requiresGrad, workload.ffHiddenDim());
        Tensor w2 = tensor("FF_W2", 309, dataType, requiresGrad, workload.ffHiddenDim(), modelDim);
        Tensor b2 = tensor("FF_B2", 310, dataType, requiresGrad, modelDim);

        Tensor ff = norm.linear(w1, b1).relu().linear(w2, b2);
        Tensor rmsGamma = tensor("RMS_GAMMA", 311, dataType, requiresGrad, modelDim);
        Tensor output = ff.add(residual).rmsNorm(rmsGamma, 1e-5);
        Tensor root = finalizeRoot(output, profile.mode());
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("batch", workload.batch());
        metadata.put("heads", workload.heads());
        metadata.put("seqLen", workload.seqLen());
        metadata.put("headDim", workload.headDim());
        metadata.put("valueDim", workload.valueDim());
        metadata.put("ffHiddenDim", workload.ffHiddenDim());
        metadata.put("causal", workload.causal());
        List<String> gradientLabels = new ArrayList<>();
        if (requiresGrad) {
            gradientLabels.add("X_BLOCK");
            gradientLabels.add("Q_BLOCK");
            gradientLabels.add("K_BLOCK");
            gradientLabels.add("V_BLOCK");
            gradientLabels.add("LN_GAMMA");
            gradientLabels.add("LN_BETA");
            gradientLabels.add("FF_W1");
            gradientLabels.add("FF_B1");
            gradientLabels.add("FF_W2");
            gradientLabels.add("FF_B2");
            gradientLabels.add("RMS_GAMMA");
        }

        return new DefaultWorkloadInstance(
                root,
                ValidationTarget.label(output.getLabel()),
                ValidationReference.baselineProfile(
                        WorkloadValidationProfiles.baselineFor(profile),
                        gradientLabels
                ),
                new WorkloadMetadata(
                        name,
                        tuning.workload.WorkloadKind.TRANSFORMER_HOT_PATH,
                        metadata
                )
        );
    }

    private static Tensor finalizeRoot(Tensor out, backend.runtime.ExecutionMode mode) {
        return mode == backend.runtime.ExecutionMode.FORWARD_BACKWARD ? out.sum() : out.sum();
    }

    private static Tensor tensor(String label, int seed, DataType dataType, boolean requiresGrad, int... shape) {
        double[] data = randomData(flatSize(shape), seed);
        return tensor.TensorDataFactory.shapedTensor(label, data, requiresGrad, dataType, shape);
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
            out[i] = Math.sin(i * 0.011 + seed * 0.07) + (random.nextDouble() - 0.5) * 0.15;
        }
        return out;
    }
}
