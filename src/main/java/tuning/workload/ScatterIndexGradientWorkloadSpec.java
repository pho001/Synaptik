package tuning.workload;

import config.profile.ExecutionProfile;
import operations.index.ScatterReduction;
import tensor.DataType;
import tensor.Tensor;
import tuning.validate.ValidationReference;
import tuning.validate.ValidationTarget;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class ScatterIndexGradientWorkloadSpec implements WorkloadSpec {
    private final String name;
    private final int batch;
    private final int features;
    private final int picks;

    public ScatterIndexGradientWorkloadSpec(String name, int batch, int features, int picks) {
        if (batch < 1 || features < 2 || picks < 1) {
            throw new IllegalArgumentException("Scatter/index-gradient workload requires batch>=1, features>=2, picks>=1");
        }
        this.name = name == null || name.isBlank() ? "scatter_index_gradient" : name;
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

        Tensor input = tensor("SCATTER_INDEX_INPUT", 931, dataType, batch, features);
        Tensor bias = tensor("SCATTER_INDEX_BIAS", 932, dataType, 1, features);
        Tensor base = input.add(bias).relu();
        Tensor scatter = base.scatterAdd(scatterIndices(), tensor("SCATTER_INDEX_SRC", 933, dataType, batch), 1);
        Tensor zeroBase = Tensor.zerosLike(base);
        Tensor gatherGrad = zeroBase.scatterAdd(
                gatherGradIndices(),
                tensor("SCATTER_INDEX_GATHER_OUT_GRAD", 934, dataType, batch),
                1
        );
        Tensor takeGrad = zeroBase.scatterElements(
                takeGradIndices(),
                tensor("SCATTER_INDEX_TAKE_OUT_GRAD", 935, dataType, batch, picks),
                1,
                ScatterReduction.ADD
        );
        Tensor root = scatter.add(gatherGrad).add(takeGrad).sum();

        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("batch", batch);
        metadata.put("features", features);
        metadata.put("picks", picks);
        metadata.put("ops", List.of("SCATTER_ADD", "SCATTER_ELEMENTS"));

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

    private Tensor scatterIndices() {
        int[] data = new int[batch];
        for (int row = 0; row < batch; row++) {
            data[row] = row % features;
        }
        return new Tensor(data, new int[]{batch}, null, "SCATTER_INDEX_SCATTER_INDICES", DataType.INT32);
    }

    private Tensor gatherGradIndices() {
        int[] data = new int[batch];
        for (int row = 0; row < batch; row++) {
            data[row] = (row * 2 + 1) % features;
        }
        return new Tensor(data, new int[]{batch}, null, "SCATTER_INDEX_GATHER_GRAD_INDICES", DataType.INT32);
    }

    private Tensor takeGradIndices() {
        int[] data = new int[batch * picks];
        for (int row = 0; row < batch; row++) {
            int repeated = (row + 2) % features;
            for (int col = 0; col < picks; col++) {
                data[row * picks + col] = col < 2 ? repeated : (row + col) % features;
            }
        }
        return new Tensor(data, new int[]{batch, picks}, null, "SCATTER_INDEX_TAKE_GRAD_INDICES", DataType.INT32);
    }

    private static Tensor tensor(String label, int seed, DataType dataType, int... shape) {
        double[] data = randomData(flatSize(shape), seed);
        return tensor.TensorDataFactory.shapedTensor(label, data, false, dataType, shape);
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
            out[i] = Math.sin(i * 0.037 + seed * 0.01) + (random.nextDouble() - 0.5) * 0.15;
        }
        return out;
    }
}
