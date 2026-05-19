package tuning.workload;

import config.profile.ExecutionProfile;
import tensor.DataType;
import tensor.Tensor;
import tuning.validate.ValidationReference;
import tuning.validate.ValidationTarget;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class GatherTakeWorkloadSpec implements WorkloadSpec {
    private final String name;
    private final int batch;
    private final int features;
    private final int picks;

    public GatherTakeWorkloadSpec(String name, int batch, int features, int picks) {
        if (batch < 1 || features < 2 || picks < 1) {
            throw new IllegalArgumentException("Gather/take workload requires batch>=1, features>=2, picks>=1");
        }
        this.name = name == null || name.isBlank() ? "gather_take" : name;
        this.batch = batch;
        this.features = features;
        this.picks = picks;
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

        Tensor input = tensor("GATHER_TAKE_INPUT", 901, dataType, batch, features);
        Tensor bias = tensor("GATHER_TAKE_BIAS", 902, dataType, 1, features);
        Tensor values = input.add(bias).relu();
        Tensor gather = values.gather(gatherIndices(), 1);
        Tensor take = values.takeAlongAxis(takeIndices(), 1);
        Tensor root = gather.sum().add(take.sum());

        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("batch", batch);
        metadata.put("features", features);
        metadata.put("picks", picks);
        metadata.put("ops", List.of("ADD", "RELU", "GATHER", "TAKE_ALONG_AXIS"));

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

    private Tensor gatherIndices() {
        int[] data = new int[batch];
        for (int i = 0; i < data.length; i++) {
            data[i] = (i * 3 + 1) % features;
        }
        return new Tensor(data, new int[]{batch}, null, "GATHER_TAKE_GATHER_INDICES", DataType.INT32);
    }

    private Tensor takeIndices() {
        int[] data = new int[batch * picks];
        for (int row = 0; row < batch; row++) {
            for (int col = 0; col < picks; col++) {
                data[row * picks + col] = (row + col * 2) % features;
            }
        }
        return new Tensor(data, new int[]{batch, picks}, null, "GATHER_TAKE_TAKE_INDICES", DataType.INT32);
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
            out[i] = Math.cos(i * 0.031 + seed * 0.01) + (random.nextDouble() - 0.5) * 0.2;
        }
        return out;
    }
}
