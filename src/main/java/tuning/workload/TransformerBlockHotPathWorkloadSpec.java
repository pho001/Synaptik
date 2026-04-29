package tuning.workload;

import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.WorkloadKind;
import config.profile.WorkloadProfile;
import tensor.DataType;
import tensor.Tensor;
import tensor.options.AttentionOptions;
import tuning.validate.ValidationReference;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Transformer-block shaped hot-path workload for graph autotune and backend benchmarking.
 *
 * <p>The graph intentionally combines projection matmuls, layout transforms, scaled dot-product attention, residual
 * elementwise chains, feed-forward matmuls, and a scalar loss. This makes it a better end-to-end optimizer stressor
 * than isolated matmul or the legacy ABC sequence workload: partitioning, fusion, memory planning, BLAS dispatch,
 * accelerator offload, and backward execution all have meaningful work to do.</p>
 */
public final class TransformerBlockHotPathWorkloadSpec implements WorkloadSpec {
    private final String name;
    private final WorkloadProfile defaultProfile;

    /**
     * Creates a transformer-block workload with the given stable name.
     *
     * @param name workload name used in reports and tuning persistence
     */
    public TransformerBlockHotPathWorkloadSpec(String name) {
        this(name, WorkloadProfile.transformerHotPathDefaults());
    }

    /**
     * Creates a transformer-block workload with an explicit default shape profile.
     *
     * <p>The default is used when the candidate execution profile does not carry
     * specialized workload metadata. Candidate profiles that do carry
     * {@link WorkloadProfile} still win, which lets autotune and benchmark
     * entries keep shape identity in persistence fingerprints.</p>
     *
     * @param name workload name used in reports and tuning persistence
     * @param defaultProfile default transformer shape; must be {@code TRANSFORMER_HOT_PATH}
     */
    public TransformerBlockHotPathWorkloadSpec(String name, WorkloadProfile defaultProfile) {
        this.name = name == null || name.isBlank() ? "transformer_block_hot_path" : name;
        this.defaultProfile = defaultProfile == null ? WorkloadProfile.transformerHotPathDefaults() : defaultProfile;
        if (this.defaultProfile.kind() != WorkloadKind.TRANSFORMER_HOT_PATH) {
            throw new IllegalArgumentException("TransformerBlockHotPathWorkloadSpec default profile must be TRANSFORMER_HOT_PATH.");
        }
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
        WorkloadProfile workload = resolveWorkloadProfile(profile, defaultProfile);
        boolean requiresGrad = profile.mode() == ExecutionMode.FORWARD_BACKWARD;
        DataType dataType = profile.dataType();

        int batch = workload.batch();
        int seqLen = workload.seqLen();
        int heads = workload.heads();
        int headDim = workload.headDim();
        int valueDim = workload.valueDim();
        int modelDim = workload.modelDim();
        int tokenCount = batch * seqLen;
        int qkProjectionDim = heads * headDim;
        int valueProjectionDim = heads * valueDim;

        Tensor x = tensor("TBLOCK_X", 901, dataType, requiresGrad, 0.12, batch, seqLen, modelDim);
        Tensor x2d = x.reshape(tokenCount, modelDim);

        Tensor wq = tensor("TBLOCK_WQ", 902, dataType, requiresGrad, 0.035, modelDim, qkProjectionDim);
        Tensor bq = tensor("TBLOCK_BQ", 903, dataType, requiresGrad, 0.010, qkProjectionDim);
        Tensor wk = tensor("TBLOCK_WK", 904, dataType, requiresGrad, 0.035, modelDim, qkProjectionDim);
        Tensor bk = tensor("TBLOCK_BK", 905, dataType, requiresGrad, 0.010, qkProjectionDim);
        Tensor wv = tensor("TBLOCK_WV", 906, dataType, requiresGrad, 0.035, modelDim, valueProjectionDim);
        Tensor bv = tensor("TBLOCK_BV", 907, dataType, requiresGrad, 0.010, valueProjectionDim);
        Tensor wo = tensor("TBLOCK_WO", 908, dataType, requiresGrad, 0.035, valueProjectionDim, modelDim);
        Tensor bo = tensor("TBLOCK_BO", 909, dataType, requiresGrad, 0.010, modelDim);

        Tensor q = x2d.linear(wq, bq).reshape(batch, seqLen, heads, headDim).permute(0, 2, 1, 3);
        Tensor k = x2d.linear(wk, bk).reshape(batch, seqLen, heads, headDim).permute(0, 2, 1, 3);
        Tensor v = x2d.linear(wv, bv).reshape(batch, seqLen, heads, valueDim).permute(0, 2, 1, 3);

        Tensor attention = q.scaledDotProductAttention(k, v, new AttentionOptions(workload.causal(), null));
        Tensor attentionMerged = attention.permute(0, 2, 1, 3).reshape(tokenCount, valueProjectionDim);
        Tensor projected = attentionMerged.linear(wo, bo);
        Tensor residual1 = x2d.add(projected);

        Tensor w1 = tensor("TBLOCK_W1", 910, dataType, requiresGrad, 0.030, modelDim, workload.ffHiddenDim());
        Tensor b1 = tensor("TBLOCK_B1", 911, dataType, requiresGrad, 0.010, workload.ffHiddenDim());
        Tensor w2 = tensor("TBLOCK_W2", 912, dataType, requiresGrad, 0.030, workload.ffHiddenDim(), modelDim);
        Tensor b2 = tensor("TBLOCK_B2", 913, dataType, requiresGrad, 0.010, modelDim);

        Tensor ff1 = residual1.linear(w1, b1);
        Tensor geluLike = ff1.mul(0.5).mul(ff1.tanh().add(Tensor.scalar(1.0, dataType)));
        Tensor ff2 = geluLike.linear(w2, b2);
        Tensor output = residual1.add(ff2);
        Tensor loss = output.mul(output).mean();

        return new DefaultWorkloadInstance(
                loss,
                ValidationReference.baselineProfile(
                        WorkloadValidationProfiles.baselineFor(profile),
                        requiresGrad ? gradientLabels() : List.of()
                ),
                new WorkloadMetadata(name, tuning.workload.WorkloadKind.TRANSFORMER_HOT_PATH, metadata(workload))
        );
    }

    private static WorkloadProfile resolveWorkloadProfile(ExecutionProfile profile, WorkloadProfile defaultProfile) {
        WorkloadProfile requested = profile.workload();
        if (requested.kind() == WorkloadKind.NONE) {
            return defaultProfile;
        }
        if (requested.kind() != WorkloadKind.TRANSFORMER_HOT_PATH) {
            throw new IllegalArgumentException("TransformerBlockHotPathWorkloadSpec requires TRANSFORMER_HOT_PATH workload metadata.");
        }
        return requested;
    }

    private static List<String> gradientLabels() {
        return List.of(
                "TBLOCK_X",
                "TBLOCK_WQ", "TBLOCK_BQ",
                "TBLOCK_WK", "TBLOCK_BK",
                "TBLOCK_WV", "TBLOCK_BV",
                "TBLOCK_WO", "TBLOCK_BO",
                "TBLOCK_W1", "TBLOCK_B1",
                "TBLOCK_W2", "TBLOCK_B2"
        );
    }

    private static Map<String, Object> metadata(WorkloadProfile workload) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("batch", workload.batch());
        metadata.put("heads", workload.heads());
        metadata.put("seqLen", workload.seqLen());
        metadata.put("headDim", workload.headDim());
        metadata.put("valueDim", workload.valueDim());
        metadata.put("modelDim", workload.modelDim());
        metadata.put("ffHiddenDim", workload.ffHiddenDim());
        metadata.put("causal", workload.causal());
        metadata.put("shapePreset", workload.transformerPresetName());
        return metadata;
    }

    private static Tensor tensor(
            String label,
            int seed,
            DataType dataType,
            boolean requiresGrad,
            double scale,
            int... shape
    ) {
        return tensor.TensorDataFactory.shapedTensor(
                label,
                randomData(flatSize(shape), seed, scale),
                requiresGrad,
                dataType,
                shape
        );
    }

    private static int flatSize(int[] shape) {
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        return size;
    }

    private static double[] randomData(int size, int seed, double scale) {
        java.util.Random random = new java.util.Random(seed);
        double[] out = new double[size];
        for (int i = 0; i < size; i++) {
            double wave = Math.sin(i * 0.013 + seed * 0.017) + Math.cos(i * 0.007 + seed * 0.031);
            out[i] = scale * (wave + (random.nextDouble() - 0.5) * 0.2);
        }
        return out;
    }
}
