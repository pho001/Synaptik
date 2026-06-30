package tuning.workload;

import config.profile.ExecutionProfile;
import tensor.DataType;
import tensor.Tensor;
import tuning.validate.ValidationReference;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class AbcSequenceMatmulWorkloadSpec implements WorkloadSpec {
    private final String name;
    private final int batch;
    private final int features;

    public AbcSequenceMatmulWorkloadSpec(String name, int batch, int features) {
        if (batch < 2 || features < 2) {
            throw new IllegalArgumentException("ABC sequence matmul workload dimensions must be >= 2.");
        }
        this.name = name == null || name.isBlank() ? "abc_sequence_matmul" : name;
        this.batch = batch;
        this.features = features;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public WorkloadKind kind() {
        return WorkloadKind.ABC_SEQUENCE_MATMUL;
    }

    @Override
    public WorkloadInstance instantiate(WorkloadEnvironment environment) {
        ExecutionProfile profile = environment.profile();
        boolean requiresGrad = profile.mode() == runtime.contract.ExecutionMode.FORWARD_BACKWARD;
        DataType dataType = profile.dataType();

        Tensor A = tensor("A", 801, dataType, requiresGrad, batch, features, 1.5, 0.9);
        Tensor B = tensor("B", 802, dataType, requiresGrad, batch, features, 1.1, 0.7);
        Tensor C = tensor("C", 803, dataType, requiresGrad, batch, features, 0.2, 0.15);

        Tensor T1 = A.div(B);
        Tensor T2 = A.sub(C);
        Tensor T3 = B.add(C);
        Tensor T4 = T1.div(T2);
        Tensor T5 = T3.mul(T4);
        Tensor T6 = T4.add(T5);
        Tensor T7 = T6.pow(2.0);

        Tensor matmul = A.matmul(B.transpose());
        Tensor matmulProjected = matmul.mean(1, true).expand(T7.getShapeUnsafe());
        Tensor root = T7.add(matmulProjected).mean();

        Map<String, Object> attributes = new LinkedHashMap<>();
        attributes.put("batch", batch);
        attributes.put("features", features);

        return new DefaultWorkloadInstance(
                root,
                ValidationReference.baselineProfile(
                        WorkloadValidationProfiles.baselineFor(profile),
                        requiresGrad ? List.of("A", "B", "C") : List.of()
                ),
                new WorkloadMetadata(name, WorkloadKind.ABC_SEQUENCE_MATMUL, attributes)
        );
    }

    private static Tensor tensor(
            String label,
            int seed,
            DataType dataType,
            boolean requiresGrad,
            int rows,
            int cols,
            double base,
            double amplitude
    ) {
        double[] data = new double[rows * cols];
        java.util.Random random = new java.util.Random(seed);
        for (int i = 0; i < data.length; i++) {
            data[i] = base + Math.abs(Math.sin(seed * 0.01 + i * 0.031)) * amplitude + random.nextDouble() * 0.03;
        }
        return tensor.factory.TensorDataFactory.shapedTensor(label, data, requiresGrad, dataType, rows, cols);
    }
}
