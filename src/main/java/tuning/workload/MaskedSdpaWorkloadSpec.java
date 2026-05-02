package tuning.workload;

import config.profile.ExecutionProfile;
import tensor.DataType;
import tensor.Tensor;
import tensor.options.AttentionOptions;
import tuning.validate.ValidationReference;
import tuning.validate.ValidationTarget;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class MaskedSdpaWorkloadSpec implements WorkloadSpec {
    private final String name;
    private final int batch;
    private final int heads;
    private final int tokens;
    private final int headDim;
    private final int valueDim;

    public MaskedSdpaWorkloadSpec(String name, int batch, int heads, int tokens, int headDim, int valueDim) {
        if (batch < 1 || heads < 1 || tokens < 2 || headDim < 1 || valueDim < 1) {
            throw new IllegalArgumentException("Masked SDPA workload requires positive dimensions and tokens >= 2");
        }
        this.name = name == null || name.isBlank() ? "masked_sdpa" : name;
        this.batch = batch;
        this.heads = heads;
        this.tokens = tokens;
        this.headDim = headDim;
        this.valueDim = valueDim;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public WorkloadKind kind() {
        return WorkloadKind.GENERIC;
    }

    @Override
    public WorkloadInstance instantiate(WorkloadEnvironment environment) {
        ExecutionProfile profile = environment.profile();
        boolean requiresGrad = profile.mode() == backend.runtime.ExecutionMode.FORWARD_BACKWARD;
        DataType dataType = profile.dataType();

        Tensor q = tensor("MASKED_SDPA_Q", 1101, dataType, requiresGrad, batch, heads, tokens, headDim);
        Tensor k = tensor("MASKED_SDPA_K", 1102, dataType, requiresGrad, batch, heads, tokens, headDim);
        Tensor v = tensor("MASKED_SDPA_V", 1103, dataType, requiresGrad, batch, heads, tokens, valueDim);
        Tensor mask = new Tensor(maskData(), new int[]{batch, heads, tokens, tokens}, null, "MASKED_SDPA_MASK", DataType.BOOL);
        Tensor attention = q.scaledDotProductAttention(k, v, mask, AttentionOptions.defaults());
        Tensor root = profile.mode() == backend.runtime.ExecutionMode.FORWARD ? attention : attention.sum();

        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("batch", batch);
        metadata.put("heads", heads);
        metadata.put("tokens", tokens);
        metadata.put("headDim", headDim);
        metadata.put("valueDim", valueDim);
        metadata.put("ops", List.of("SCALED_DOT_PRODUCT_ATTENTION"));
        metadata.put("maskMode", "EXTERNAL_BOOL_MASK");

        return new DefaultWorkloadInstance(
                root,
                ValidationTarget.root(),
                ValidationReference.baselineProfile(
                        WorkloadValidationProfiles.baselineFor(profile),
                        requiresGrad ? List.of("MASKED_SDPA_Q", "MASKED_SDPA_K", "MASKED_SDPA_V") : List.of()
                ),
                new WorkloadMetadata(name, WorkloadKind.GENERIC, metadata)
        );
    }

    private byte[] maskData() {
        byte[] out = new byte[batch * heads * tokens * tokens];
        int index = 0;
        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < heads; h++) {
                for (int q = 0; q < tokens; q++) {
                    for (int k = 0; k < tokens; k++) {
                        out[index++] = (byte) (((q + k + h) % 3) == 0 ? 0 : 1);
                    }
                }
            }
        }
        return out;
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
            out[i] = Math.sin(i * 0.017 + seed * 0.01) + (random.nextDouble() - 0.5) * 0.2;
        }
        return out;
    }
}
