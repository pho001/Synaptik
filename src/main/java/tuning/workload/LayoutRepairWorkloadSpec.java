package tuning.workload;

import config.profile.ExecutionProfile;
import tensor.DataType;
import tensor.Tensor;
import tuning.validate.ValidationReference;
import tuning.validate.ValidationTarget;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class LayoutRepairWorkloadSpec implements WorkloadSpec {
    private final String name;
    private final int batch;
    private final int features;

    public LayoutRepairWorkloadSpec(String name, int batch, int features) {
        if (batch < 2 || features < 1) {
            throw new IllegalArgumentException("Layout repair workload requires batch>=2 and features>=1");
        }
        this.name = name == null || name.isBlank() ? "layout_broadcast_repair" : name;
        this.batch = batch;
        this.features = features;
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
        DataType dataType = profile.dataType();

        Tensor input = tensor("LAYOUT_REPAIR_INPUT", 1001, dataType, 1, features);
        Tensor bias = tensor("LAYOUT_REPAIR_BIAS", 1002, dataType, 1, features);
        Tensor base = input.add(bias);
        Tensor expanded = base.expand(batch, features);
        Tensor root = expanded.contiguous();

        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("batch", batch);
        metadata.put("features", features);
        metadata.put("ops", List.of("ADD", "EXPAND", "CONTIGUOUS"));
        metadata.put("layoutRoute", "BROADCAST_GPU_MATERIALIZATION");

        return new DefaultWorkloadInstance(
                root,
                ValidationTarget.root(),
                ValidationReference.baselineProfile(
                        WorkloadValidationProfiles.baselineFor(profile),
                        List.of()
                ),
                new WorkloadMetadata(name, WorkloadKind.GENERIC, metadata)
        );
    }

    private static Tensor tensor(String label, int seed, DataType dataType, int... shape) {
        double[] data = randomData(flatSize(shape), seed);
        return tensor.factory.TensorDataFactory.shapedTensor(label, data, false, dataType, shape);
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
            out[i] = Math.sin(i * 0.041 + seed * 0.01) + (random.nextDouble() - 0.5) * 0.1;
        }
        return out;
    }
}
