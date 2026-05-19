package tuning.workload;

import config.profile.ExecutionProfile;
import tensor.DataType;
import tensor.Tensor;
import tensor.options.Pool2dOptions;
import tuning.validate.ValidationReference;
import tuning.validate.ValidationTarget;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class Pool2dWorkloadSpec implements WorkloadSpec {
    public enum PoolKind {
        MAX,
        AVG
    }

    private final String name;
    private final PoolKind poolKind;
    private final int batch;
    private final int channels;
    private final int height;
    private final int width;
    private final Pool2dOptions options;

    public Pool2dWorkloadSpec(
            String name,
            PoolKind poolKind,
            int batch,
            int channels,
            int height,
            int width,
            Pool2dOptions options
    ) {
        if (batch < 1 || channels < 1 || height < 1 || width < 1) {
            throw new IllegalArgumentException("Pool2d workload dimensions must be >= 1");
        }
        this.name = (name == null || name.isBlank()) ? "pool2d" : name;
        this.poolKind = poolKind == null ? PoolKind.MAX : poolKind;
        this.batch = batch;
        this.channels = channels;
        this.height = height;
        this.width = width;
        this.options = options == null ? Pool2dOptions.square(2) : options;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public WorkloadKind kind() {
        return WorkloadKind.POOL2D;
    }

    @Override
    public WorkloadInstance instantiate(WorkloadEnvironment environment) {
        ExecutionProfile profile = environment.profile();
        boolean requiresGrad = profile.mode() == backend.runtime.ExecutionMode.FORWARD_BACKWARD;
        DataType dataType = profile.dataType();

        Tensor input = tensor("POOL_INPUT", 501, dataType, requiresGrad, batch, channels, height, width);
        Tensor output = switch (poolKind) {
            case MAX -> input.maxPool2d(options);
            case AVG -> input.avgPool2d(options);
        };
        Tensor root = finalizeRoot(output, profile.mode());

        List<String> gradientLabels = requiresGrad ? List.of("POOL_INPUT") : List.of();
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("poolKind", poolKind.name());
        metadata.put("batch", batch);
        metadata.put("channels", channels);
        metadata.put("height", height);
        metadata.put("width", width);
        metadata.put("kernelH", options.kernelH());
        metadata.put("kernelW", options.kernelW());
        metadata.put("strideH", options.strideH());
        metadata.put("strideW", options.strideW());
        metadata.put("padH", options.padH());
        metadata.put("padW", options.padW());
        metadata.put("countIncludePad", options.countIncludePad());

        return new DefaultWorkloadInstance(
                root,
                ValidationTarget.label(output.getLabel()),
                ValidationReference.baselineProfile(
                        WorkloadValidationProfiles.baselineFor(profile),
                        gradientLabels
                ),
                new WorkloadMetadata(name, WorkloadKind.POOL2D, metadata)
        );
    }

    private static Tensor finalizeRoot(Tensor out, backend.runtime.ExecutionMode mode) {
        return mode == backend.runtime.ExecutionMode.FORWARD_BACKWARD ? out.sum() : out.sum();
    }

    private static Tensor tensor(String label, int seed, DataType dataType, boolean requiresGrad, int... shape) {
        double[] data = randomData(flatSize(shape), seed);
        return tensor.factory.TensorDataFactory.shapedTensor(label, data, requiresGrad, dataType, shape);
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
            out[i] = Math.cos(i * 0.015 + seed * 0.05) + (random.nextDouble() - 0.5) * 0.15;
        }
        return out;
    }
}
