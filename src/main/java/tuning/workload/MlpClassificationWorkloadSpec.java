package tuning.workload;

import config.profile.ExecutionProfile;
import tensor.DataType;
import tensor.Tensor;
import tensor.loss.LossReduction;
import tuning.validate.ValidationReference;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class MlpClassificationWorkloadSpec implements WorkloadSpec {
    private final String name;
    private final int batch;
    private final int inputFeatures;
    private final int hidden1;
    private final int hidden2;
    private final int classes;
    private final LossReduction reduction;

    public MlpClassificationWorkloadSpec(
            String name,
            int batch,
            int inputFeatures,
            int hidden1,
            int hidden2,
            int classes,
            LossReduction reduction
    ) {
        if (batch < 1 || inputFeatures < 2 || hidden1 < 2 || hidden2 < 2 || classes < 2) {
            throw new IllegalArgumentException("MLP classification workload dimensions must be valid.");
        }
        this.name = name == null || name.isBlank() ? "mlp_classification" : name;
        this.batch = batch;
        this.inputFeatures = inputFeatures;
        this.hidden1 = hidden1;
        this.hidden2 = hidden2;
        this.classes = classes;
        this.reduction = reduction == null ? LossReduction.MEAN : reduction;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public WorkloadKind kind() {
        return WorkloadKind.MLP_CLASSIFICATION;
    }

    @Override
    public WorkloadInstance instantiate(WorkloadEnvironment environment) {
        ExecutionProfile profile = environment.profile();
        boolean requiresGrad = profile.mode() == runtime.contract.ExecutionMode.FORWARD_BACKWARD;
        DataType dataType = profile.dataType();

        Tensor encoded = tensor("MLP_ENCODED", 701, dataType, requiresGrad, batch, inputFeatures);
        Tensor w1 = tensor("MLP_W1", 702, dataType, requiresGrad, inputFeatures, hidden1);
        Tensor b1 = tensor("MLP_B1", 703, dataType, requiresGrad, hidden1);
        Tensor w2 = tensor("MLP_W2", 704, dataType, requiresGrad, hidden1, hidden2);
        Tensor b2 = tensor("MLP_B2", 705, dataType, requiresGrad, hidden2);
        Tensor w3 = tensor("MLP_W3", 706, dataType, requiresGrad, hidden2, classes);
        Tensor b3 = tensor("MLP_B3", 707, dataType, requiresGrad, classes);
        Tensor targets = targetIndices(batch, classes);

        Tensor logits = encoded
                .linear(w1, b1).relu()
                .linear(w2, b2).relu()
                .linear(w3, b3);
        Tensor root = logits.crossEntropyLossFromIndices(targets, 1, reduction);

        Map<String, Object> attributes = new LinkedHashMap<>();
        attributes.put("batch", batch);
        attributes.put("inputFeatures", inputFeatures);
        attributes.put("hidden1", hidden1);
        attributes.put("hidden2", hidden2);
        attributes.put("classes", classes);
        attributes.put("reduction", reduction.name());

        return new DefaultWorkloadInstance(
                root,
                ValidationReference.baselineProfile(
                        WorkloadValidationProfiles.baselineFor(profile),
                        requiresGrad
                                ? List.of("MLP_ENCODED", "MLP_W1", "MLP_B1", "MLP_W2", "MLP_B2", "MLP_W3", "MLP_B3")
                                : List.of()
                ),
                new WorkloadMetadata(name, WorkloadKind.MLP_CLASSIFICATION, attributes)
        );
    }

    private static Tensor tensor(String label, int seed, DataType dataType, boolean requiresGrad, int... shape) {
        return tensor.factory.TensorDataFactory.shapedTensor(label, encodedData(flatSize(shape), seed), requiresGrad, dataType, shape);
    }

    private static Tensor targetIndices(int batch, int classes) {
        int[] data = new int[batch];
        for (int i = 0; i < batch; i++) {
            data[i] = (i * 3 + 1) % classes;
        }
        return new Tensor(data, new int[]{batch}, null, "MLP_TARGETS", DataType.INT32);
    }

    private static int flatSize(int[] shape) {
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        return size;
    }

    private static double[] encodedData(int size, int seed) {
        java.util.Random random = new java.util.Random(seed);
        double[] out = new double[size];
        for (int i = 0; i < size; i++) {
            out[i] = Math.sin(seed * 0.01 + i * 0.071) + Math.cos(seed * 0.03 + i * 0.017) + (random.nextDouble() - 0.5) * 0.05;
        }
        return out;
    }
}
