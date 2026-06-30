package tuning.workload;

import config.profile.ExecutionProfile;
import tensor.DataType;
import tensor.Tensor;
import tensor.loss.LossReduction;
import tuning.validate.ValidationReference;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class LossWorkloadSpec implements WorkloadSpec {
    public enum LossKind {
        CROSS_ENTROPY_FROM_INDICES,
        NLL_FROM_INDICES,
        DENSE_CROSS_ENTROPY_AND_NLL
    }

    private final String name;
    private final LossKind lossKind;
    private final int batch;
    private final int classes;
    private final LossReduction reduction;

    public LossWorkloadSpec(
            String name,
            LossKind lossKind,
            int batch,
            int classes,
            LossReduction reduction
    ) {
        if (batch < 1 || classes < 2) {
            throw new IllegalArgumentException("Loss workload dimensions must be valid");
        }
        this.name = (name == null || name.isBlank()) ? "loss" : name;
        this.lossKind = lossKind == null ? LossKind.CROSS_ENTROPY_FROM_INDICES : lossKind;
        this.batch = batch;
        this.classes = classes;
        this.reduction = reduction == null ? LossReduction.MEAN : reduction;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public WorkloadKind kind() {
        return WorkloadKind.LOSS;
    }

    @Override
    public WorkloadInstance instantiate(WorkloadEnvironment environment) {
        ExecutionProfile profile = environment.profile();
        boolean requiresGrad = profile.mode() == runtime.contract.ExecutionMode.FORWARD_BACKWARD;
        DataType dataType = profile.dataType();

        Tensor logits = tensor("LOSS_LOGITS", 601, dataType, requiresGrad, batch, classes);
        Tensor targetIndices = targetIndices(batch, classes);
        Tensor denseTargets = denseTargets(batch, classes, dataType);
        Tensor root = switch (lossKind) {
            case CROSS_ENTROPY_FROM_INDICES -> logits.crossEntropyLossFromIndices(targetIndices, 1, reduction);
            case NLL_FROM_INDICES -> logits.logSoftmax(1).nllLossFromIndices(targetIndices, 1, reduction);
            case DENSE_CROSS_ENTROPY_AND_NLL -> {
                Tensor denseCe = logits.crossEntropyLoss(denseTargets, 1);
                Tensor denseNll = logits.logSoftmax(1).nllLoss(denseTargets, 1);
                yield denseCe.add(denseNll);
            }
        };
        root = finalizeRoot(root, profile.mode());

        List<String> gradientLabels = requiresGrad ? List.of("LOSS_LOGITS") : List.of();
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("lossKind", lossKind.name());
        metadata.put("batch", batch);
        metadata.put("classes", classes);
        metadata.put("reduction", reduction.name());

        return new DefaultWorkloadInstance(
                root,
                ValidationReference.baselineProfile(
                        WorkloadValidationProfiles.baselineFor(profile),
                        gradientLabels
                ),
                new WorkloadMetadata(name, WorkloadKind.LOSS, metadata)
        );
    }

    private static Tensor finalizeRoot(Tensor out, runtime.contract.ExecutionMode mode) {
        return mode == runtime.contract.ExecutionMode.FORWARD_BACKWARD ? out.sum() : out.sum();
    }

    private static Tensor tensor(String label, int seed, DataType dataType, boolean requiresGrad, int... shape) {
        double[] data = randomData(flatSize(shape), seed);
        return tensor.factory.TensorDataFactory.shapedTensor(label, data, requiresGrad, dataType, shape);
    }

    private static Tensor targetIndices(int batch, int classes) {
        int[] data = new int[batch];
        for (int i = 0; i < batch; i++) {
            data[i] = i % classes;
        }
        return new Tensor(data, new int[]{batch}, null, "LOSS_TARGETS", DataType.INT32);
    }

    private static Tensor denseTargets(int batch, int classes, DataType dataType) {
        double[] data = new double[batch * classes];
        for (int row = 0; row < batch; row++) {
            data[row * classes + (row % classes)] = 1.0d;
        }
        return new Tensor(data, new int[]{batch, classes}, null, "LOSS_DENSE_TARGETS", dataType);
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
            out[i] = Math.sin(i * 0.021 + seed * 0.05) + (random.nextDouble() - 0.5) * 0.2;
        }
        return out;
    }
}
