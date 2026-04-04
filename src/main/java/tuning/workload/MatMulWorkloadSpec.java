package tuning.workload;

import config.profile.ExecutionProfile;
import tensor.DataType;
import tensor.Tensor;
import tuning.validate.ValidationReference;

import java.util.List;
import java.util.Map;

public final class MatMulWorkloadSpec implements WorkloadSpec {
    private final String name;
    private final int batch;
    private final int m;
    private final int k;
    private final int n;

    public MatMulWorkloadSpec(String name, int batch, int m, int k, int n) {
        if (batch < 1 || m < 1 || k < 1 || n < 1) {
            throw new IllegalArgumentException("MatMul workload dimensions must be >= 1");
        }
        this.name = (name == null || name.isBlank()) ? "matmul" : name;
        this.batch = batch;
        this.m = m;
        this.k = k;
        this.n = n;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public WorkloadKind kind() {
        return WorkloadKind.MATMUL;
    }

    @Override
    public WorkloadInstance instantiate(WorkloadEnvironment environment) {
        ExecutionProfile profile = environment.profile();
        boolean requiresGrad = profile.mode() == backend.runtime.ExecutionMode.FORWARD_BACKWARD;
        DataType dataType = profile.dataType();

        Tensor left;
        Tensor right;
        if (batch == 1) {
            left = tensor("MATMUL_A", 101, dataType, requiresGrad, m, k);
            right = tensor("MATMUL_B", 102, dataType, requiresGrad, k, n);
        } else {
            left = tensor("MATMUL_A", 101, dataType, requiresGrad, batch, m, k);
            right = tensor("MATMUL_B", 102, dataType, requiresGrad, batch, k, n);
        }

        Tensor root = finalizeRoot(left.matmul(right), profile.mode());
        return new DefaultWorkloadInstance(
                root,
                ValidationReference.baselineProfile(
                        WorkloadValidationProfiles.baselineFor(profile),
                        requiresGrad ? List.of("MATMUL_A", "MATMUL_B") : List.of()
                ),
                new WorkloadMetadata(
                        name,
                        WorkloadKind.MATMUL,
                        Map.of("batch", batch, "m", m, "k", k, "n", n)
                )
        );
    }

    private static Tensor finalizeRoot(Tensor out, backend.runtime.ExecutionMode mode) {
        return mode == backend.runtime.ExecutionMode.FORWARD_BACKWARD ? out.sum() : out.sum();
    }

    private static Tensor tensor(String label, int seed, DataType dataType, boolean requiresGrad, int... shape) {
        double[] data = randomData(flatSize(shape), seed);
        return benchmark.scenario.ScenarioTensorFactory.shapedTensor(label, data, requiresGrad, dataType, shape);
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
            out[i] = Math.sin(i * 0.017 + seed * 0.1) + (random.nextDouble() - 0.5) * 0.2;
        }
        return out;
    }
}
