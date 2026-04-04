package tuning.workload;

import config.profile.ExecutionProfile;
import tensor.Conv2dOptions;
import tensor.DataType;
import tensor.Tensor;
import tuning.validate.ValidationReference;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class Conv2dWorkloadSpec implements WorkloadSpec {
    private final String name;
    private final int batch;
    private final int inChannels;
    private final int outChannels;
    private final int height;
    private final int width;
    private final int kernelH;
    private final int kernelW;
    private final Conv2dOptions options;
    private final boolean withBias;

    public Conv2dWorkloadSpec(
            String name,
            int batch,
            int inChannels,
            int outChannels,
            int height,
            int width,
            int kernelH,
            int kernelW,
            Conv2dOptions options,
            boolean withBias
    ) {
        if (batch < 1 || inChannels < 1 || outChannels < 1 || height < 1 || width < 1 || kernelH < 1 || kernelW < 1) {
            throw new IllegalArgumentException("Conv2d workload dimensions must be >= 1");
        }
        this.name = (name == null || name.isBlank()) ? "conv2d" : name;
        this.batch = batch;
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.height = height;
        this.width = width;
        this.kernelH = kernelH;
        this.kernelW = kernelW;
        this.options = options == null ? Conv2dOptions.defaults() : options;
        this.withBias = withBias;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public WorkloadKind kind() {
        return WorkloadKind.CONV2D;
    }

    @Override
    public WorkloadInstance instantiate(WorkloadEnvironment environment) {
        ExecutionProfile profile = environment.profile();
        boolean requiresGrad = profile.mode() == backend.runtime.ExecutionMode.FORWARD_BACKWARD;
        DataType dataType = profile.dataType();

        Tensor input = tensor("CONV_INPUT", 201, dataType, requiresGrad, batch, inChannels, height, width);
        Tensor weight = tensor(
                "CONV_WEIGHT",
                202,
                dataType,
                requiresGrad,
                outChannels,
                inChannels / options.groups(),
                kernelH,
                kernelW
        );
        Tensor root = withBias
                ? input.conv2d(weight, tensor("CONV_BIAS", 203, dataType, requiresGrad, outChannels), options)
                : input.conv2d(weight, options);
        root = finalizeRoot(root, profile.mode());
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("batch", batch);
        metadata.put("inChannels", inChannels);
        metadata.put("outChannels", outChannels);
        metadata.put("height", height);
        metadata.put("width", width);
        metadata.put("kernelH", kernelH);
        metadata.put("kernelW", kernelW);
        metadata.put("groups", options.groups());
        metadata.put("strideH", options.strideH());
        metadata.put("strideW", options.strideW());
        metadata.put("padH", options.padH());
        metadata.put("padW", options.padW());
        metadata.put("withBias", withBias);
        List<String> gradientLabels = new ArrayList<>();
        if (requiresGrad) {
            gradientLabels.add("CONV_INPUT");
            gradientLabels.add("CONV_WEIGHT");
            if (withBias) {
                gradientLabels.add("CONV_BIAS");
            }
        }

        return new DefaultWorkloadInstance(
                root,
                ValidationReference.baselineProfile(
                        WorkloadValidationProfiles.baselineFor(profile),
                        gradientLabels
                ),
                new WorkloadMetadata(
                        name,
                        WorkloadKind.CONV2D,
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
            out[i] = Math.cos(i * 0.013 + seed * 0.1) + (random.nextDouble() - 0.5) * 0.2;
        }
        return out;
    }
}
