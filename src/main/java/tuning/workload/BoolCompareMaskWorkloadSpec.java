package tuning.workload;

import config.profile.ExecutionProfile;
import tensor.DataType;
import tensor.Tensor;
import tuning.validate.ValidationReference;
import tuning.validate.ValidationTarget;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class BoolCompareMaskWorkloadSpec implements WorkloadSpec {
    private final String name;
    private final int batch;
    private final int features;

    public BoolCompareMaskWorkloadSpec(String name, int batch, int features) {
        if (batch < 1 || features < 1) {
            throw new IllegalArgumentException("Bool compare workload dimensions must be >= 1");
        }
        this.name = name == null || name.isBlank() ? "bool_compare_where" : name;
        this.batch = batch;
        this.features = features;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public WorkloadKind kind() {
        return WorkloadKind.BOOL_COMPARE;
    }

    @Override
    public WorkloadInstance instantiate(WorkloadEnvironment environment) {
        ExecutionProfile profile = environment.profile();
        boolean requiresGrad = profile.mode() == runtime.contract.ExecutionMode.FORWARD_BACKWARD;
        DataType dataType = profile.dataType();

        Tensor input = tensor("BOOL_COMPARE_INPUT", 801, dataType, requiresGrad, batch, features);
        Tensor threshold = tensor("BOOL_COMPARE_THRESHOLD", 802, dataType, false, 1, features);
        Tensor fallback = Tensor.zerosLike(input);
        Tensor mask = input.greaterThan(threshold);
        Tensor selected = Tensor.where(mask, input, fallback).relu();
        Tensor root = selected.sum();

        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("batch", batch);
        metadata.put("features", features);
        metadata.put("ops", List.of("GT", "WHERE"));

        return new DefaultWorkloadInstance(
                root,
                ValidationTarget.root(),
                ValidationReference.baselineProfile(
                        WorkloadValidationProfiles.baselineFor(profile),
                        requiresGrad ? List.of("BOOL_COMPARE_INPUT") : List.of()
                ),
                new WorkloadMetadata(name, WorkloadKind.BOOL_COMPARE, metadata)
        );
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
            out[i] = Math.sin(i * 0.023 + seed * 0.02) + (random.nextDouble() - 0.5) * 0.2;
        }
        return out;
    }
}
