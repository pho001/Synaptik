package tuning.workload;

import config.profile.ExecutionProfile;
import tensor.DataType;
import tensor.Tensor;
import tuning.validate.ValidationReference;
import tuning.validate.ValidationTarget;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class NormalizationWorkloadSpec implements WorkloadSpec {
    public enum NormalizationKind {
        BATCH_NORM,
        LAYER_NORM,
        RMS_NORM
    }

    private final String name;
    private final NormalizationKind normalizationKind;
    private final int batch;
    private final int channels;
    private final int height;
    private final int width;
    private final double epsilon;

    public NormalizationWorkloadSpec(
            String name,
            NormalizationKind normalizationKind,
            int batch,
            int channels,
            int height,
            int width,
            double epsilon
    ) {
        if (batch < 1 || channels < 1 || height < 1 || width < 1) {
            throw new IllegalArgumentException("Normalization workload dimensions must be >= 1");
        }
        if (!(epsilon > 0.0d)) {
            throw new IllegalArgumentException("Normalization workload epsilon must be > 0");
        }
        this.name = (name == null || name.isBlank()) ? "normalization" : name;
        this.normalizationKind = normalizationKind == null ? NormalizationKind.LAYER_NORM : normalizationKind;
        this.batch = batch;
        this.channels = channels;
        this.height = height;
        this.width = width;
        this.epsilon = epsilon;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public WorkloadKind kind() {
        return WorkloadKind.NORMALIZATION;
    }

    @Override
    public WorkloadInstance instantiate(WorkloadEnvironment environment) {
        ExecutionProfile profile = environment.profile();
        boolean requiresGrad = profile.mode() == backend.runtime.ExecutionMode.FORWARD_BACKWARD;
        DataType dataType = profile.dataType();

        Tensor output;
        List<String> gradientLabels = new ArrayList<>();
        switch (normalizationKind) {
            case BATCH_NORM -> {
                Tensor input = tensor("BN_INPUT", 401, dataType, requiresGrad, batch, channels, height, width);
                Tensor gamma = tensor("BN_GAMMA", 402, dataType, requiresGrad, channels);
                Tensor beta = tensor("BN_BETA", 403, dataType, requiresGrad, channels);
                output = input.batchNorm(gamma, beta, 1, epsilon);
                if (requiresGrad) {
                    gradientLabels.add("BN_INPUT");
                    gradientLabels.add("BN_GAMMA");
                    gradientLabels.add("BN_BETA");
                }
            }
            case LAYER_NORM -> {
                Tensor input = tensor("LN_INPUT", 404, dataType, requiresGrad, batch, height, channels);
                Tensor gamma = tensor("LN_GAMMA", 405, dataType, requiresGrad, channels);
                Tensor beta = tensor("LN_BETA", 406, dataType, requiresGrad, channels);
                output = input.layerNorm(gamma, beta, epsilon);
                if (requiresGrad) {
                    gradientLabels.add("LN_INPUT");
                    gradientLabels.add("LN_GAMMA");
                    gradientLabels.add("LN_BETA");
                }
            }
            case RMS_NORM -> {
                Tensor input = tensor("RMS_INPUT", 407, dataType, requiresGrad, batch, height, channels);
                Tensor gamma = tensor("RMS_GAMMA", 408, dataType, requiresGrad, channels);
                output = input.rmsNorm(gamma, epsilon);
                if (requiresGrad) {
                    gradientLabels.add("RMS_INPUT");
                    gradientLabels.add("RMS_GAMMA");
                }
            }
            default -> throw new IllegalStateException("Unsupported normalization kind: " + normalizationKind);
        }

        Tensor root = finalizeRoot(output, profile.mode());
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("normalizationKind", normalizationKind.name());
        metadata.put("batch", batch);
        metadata.put("channels", channels);
        metadata.put("height", height);
        metadata.put("width", width);
        metadata.put("epsilon", epsilon);

        return new DefaultWorkloadInstance(
                root,
                ValidationTarget.label(output.getLabel()),
                ValidationReference.baselineProfile(
                        WorkloadValidationProfiles.baselineFor(profile),
                        gradientLabels
                ),
                new WorkloadMetadata(name, WorkloadKind.NORMALIZATION, metadata)
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
            out[i] = Math.sin(i * 0.019 + seed * 0.05) + (random.nextDouble() - 0.5) * 0.15;
        }
        return out;
    }
}
