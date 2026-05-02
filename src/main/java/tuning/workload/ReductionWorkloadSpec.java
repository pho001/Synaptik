package tuning.workload;

import config.profile.ExecutionProfile;
import tensor.DataType;
import tensor.Tensor;
import tuning.validate.ValidationReference;
import tuning.validate.ValidationTarget;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class ReductionWorkloadSpec implements WorkloadSpec {
    private final String name;
    private final int batch;
    private final int features;

    public ReductionWorkloadSpec(String name, int batch, int features) {
        if (batch < 1 || features < 1) {
            throw new IllegalArgumentException("Reduction workload dimensions must be >= 1");
        }
        this.name = name == null || name.isBlank() ? "reduction_chain" : name;
        this.batch = batch;
        this.features = features;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public WorkloadKind kind() {
        return WorkloadKind.REDUCTION;
    }

    @Override
    public WorkloadInstance instantiate(WorkloadEnvironment environment) {
        ExecutionProfile profile = environment.profile();
        boolean requiresGrad = profile.mode() == backend.runtime.ExecutionMode.FORWARD_BACKWARD;
        DataType dataType = profile.dataType();

        Tensor input = tensor("REDUCTION_INPUT", 701, dataType, requiresGrad, batch, features);
        Tensor scaled = input.mul(Tensor.scalar(0.75d).expand(batch, features));
        Tensor reduced = scaled.sum(1, true)
                .add(scaled.mean(1, true))
                .add(scaled.min(1, true))
                .add(scaled.max(1, true));
        Tensor root = reduced.sum();

        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("batch", batch);
        metadata.put("features", features);
        metadata.put("ops", List.of("SUM", "MEAN", "REDUCE_MIN", "REDUCE_MAX"));

        return new DefaultWorkloadInstance(
                root,
                ValidationTarget.root(),
                ValidationReference.baselineProfile(
                        WorkloadValidationProfiles.baselineFor(profile),
                        requiresGrad ? List.of("REDUCTION_INPUT") : List.of()
                ),
                new WorkloadMetadata(name, WorkloadKind.REDUCTION, metadata)
        );
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
            out[i] = Math.sin(i * 0.017 + seed * 0.03) + (random.nextDouble() - 0.5) * 0.1;
        }
        return out;
    }
}
