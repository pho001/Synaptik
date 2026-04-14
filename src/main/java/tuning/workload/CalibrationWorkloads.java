package tuning.workload;

import tuning.validate.ValidationReference;

public final class CalibrationWorkloads {
    private CalibrationWorkloads() {
    }

    public static WorkloadCatalog defaultCatalog() {
        return new WorkloadCatalog()
                .register(matmulSquare("calib_matmul_square", 128))
                .register(matmulTallSkinny("calib_matmul_tall_skinny", 256, 64, 64))
                .register(matmulBatchedAttentionLike("calib_matmul_attention_like", 8, 128, 64, 64))
                .register(fusedCheapElementwise("calib_fused_cheap", 65_536))
                .register(fusedCheapStridedElementwise("calib_fused_cheap_strided", 256, 256))
                .register(fusedTranscendental("calib_fused_transcendental", 65_536))
                .register(fusedTranscendentalStrided("calib_fused_transcendental_strided", 256, 256))
                .register(reductionSum("calib_reduction_sum", 262_144))
                .register(schedulerCheapParallel("calib_scheduler_cheap", 262_144))
                .register(materializationStridedElementwise("calib_materialization_strided", 256, 256))
                .register(conv2dResnet3x3("calib_conv2d_resnet_3x3"));
    }

    public static MatMulWorkloadSpec matmulSquare(String name, int dim) {
        return StandardWorkloads.matmul(name, 1, dim, dim, dim);
    }

    public static MatMulWorkloadSpec matmulTallSkinny(String name, int m, int k, int n) {
        return StandardWorkloads.matmul(name, 1, m, k, n);
    }

    public static MatMulWorkloadSpec matmulBatchedAttentionLike(String name, int batch, int m, int k, int n) {
        return StandardWorkloads.matmul(name, batch, m, k, n);
    }

    public static TensorRootWorkloadSpec fusedCheapElementwise(String name, int size) {
        return new TensorRootWorkloadSpec(
                name,
                WorkloadKind.GENERIC,
                environment -> {
                    boolean requiresGrad = environment.profile().mode() == backend.runtime.ExecutionMode.FORWARD_BACKWARD;
                    tensor.Tensor a = tensor.TensorDataFactory.shapedTensor("A", random(size, 101), requiresGrad, environment.profile().dataType(), size);
                    tensor.Tensor b = tensor.TensorDataFactory.shapedTensor("B", random(size, 102), requiresGrad, environment.profile().dataType(), size);
                    return a.add(b).mul(a).sub(b).relu().sum();
                },
                environment -> ValidationReference.none(),
                environment -> WorkloadMetadata.of(name, WorkloadKind.GENERIC)
        );
    }

    public static TensorRootWorkloadSpec fusedTranscendental(String name, int size) {
        return new TensorRootWorkloadSpec(
                name,
                WorkloadKind.GENERIC,
                environment -> {
                    boolean requiresGrad = environment.profile().mode() == backend.runtime.ExecutionMode.FORWARD_BACKWARD;
                    tensor.Tensor a = tensor.TensorDataFactory.shapedTensor("A", random(size, 103), requiresGrad, environment.profile().dataType(), size);
                    return a.exp().tanh().log().sigmoid().sum();
                },
                environment -> ValidationReference.none(),
                environment -> WorkloadMetadata.of(name, WorkloadKind.GENERIC)
        );
    }

    public static TensorRootWorkloadSpec fusedCheapStridedElementwise(String name, int rows, int cols) {
        return new TensorRootWorkloadSpec(
                name,
                WorkloadKind.GENERIC,
                environment -> {
                    int size = rows * cols;
                    boolean requiresGrad = environment.profile().mode() == backend.runtime.ExecutionMode.FORWARD_BACKWARD;
                    tensor.Tensor a = tensor.TensorDataFactory.shapedTensor("A", random(size, 109), requiresGrad, environment.profile().dataType(), rows, cols);
                    tensor.Tensor b = tensor.TensorDataFactory.shapedTensor("B", random(size, 110), requiresGrad, environment.profile().dataType(), rows, cols);
                    tensor.Tensor aT = a.transpose();
                    tensor.Tensor bT = b.transpose();
                    return aT.add(bT).mul(aT).sub(bT).relu().sum();
                },
                environment -> ValidationReference.none(),
                environment -> WorkloadMetadata.of(name, WorkloadKind.GENERIC)
        );
    }

    public static TensorRootWorkloadSpec fusedTranscendentalStrided(String name, int rows, int cols) {
        return new TensorRootWorkloadSpec(
                name,
                WorkloadKind.GENERIC,
                environment -> {
                    int size = rows * cols;
                    boolean requiresGrad = environment.profile().mode() == backend.runtime.ExecutionMode.FORWARD_BACKWARD;
                    tensor.Tensor a = tensor.TensorDataFactory.shapedTensor("A", random(size, 111), requiresGrad, environment.profile().dataType(), rows, cols);
                    tensor.Tensor aT = a.transpose();
                    return aT.exp().tanh().log().sigmoid().sum();
                },
                environment -> ValidationReference.none(),
                environment -> WorkloadMetadata.of(name, WorkloadKind.GENERIC)
        );
    }

    public static TensorRootWorkloadSpec reductionSum(String name, int size) {
        return new TensorRootWorkloadSpec(
                name,
                WorkloadKind.GENERIC,
                environment -> {
                    boolean requiresGrad = environment.profile().mode() == backend.runtime.ExecutionMode.FORWARD_BACKWARD;
                    return tensor.TensorDataFactory.shapedTensor("REDUCE", random(size, 104), requiresGrad, environment.profile().dataType(), size).sum();
                },
                environment -> ValidationReference.none(),
                environment -> WorkloadMetadata.of(name, WorkloadKind.GENERIC)
        );
    }

    public static TensorRootWorkloadSpec schedulerCheapParallel(String name, int size) {
        return new TensorRootWorkloadSpec(
                name,
                WorkloadKind.GENERIC,
                environment -> {
                    boolean requiresGrad = environment.profile().mode() == backend.runtime.ExecutionMode.FORWARD_BACKWARD;
                    tensor.Tensor a = tensor.TensorDataFactory.shapedTensor("A", random(size, 105), requiresGrad, environment.profile().dataType(), size);
                    tensor.Tensor b = tensor.TensorDataFactory.shapedTensor("B", random(size, 106), requiresGrad, environment.profile().dataType(), size);
                    return a.add(b).mul(a).add(b).sum();
                },
                environment -> ValidationReference.none(),
                environment -> WorkloadMetadata.of(name, WorkloadKind.GENERIC)
        );
    }

    public static TensorRootWorkloadSpec materializationStridedElementwise(String name, int rows, int cols) {
        return new TensorRootWorkloadSpec(
                name,
                WorkloadKind.GENERIC,
                environment -> {
                    int size = rows * cols;
                    boolean requiresGrad = environment.profile().mode() == backend.runtime.ExecutionMode.FORWARD_BACKWARD;
                    tensor.Tensor a = tensor.TensorDataFactory.shapedTensor("A", random(size, 107), requiresGrad, environment.profile().dataType(), rows, cols);
                    tensor.Tensor b = tensor.TensorDataFactory.shapedTensor("B", random(size, 108), requiresGrad, environment.profile().dataType(), rows, cols);
                    tensor.Tensor aT = a.transpose();
                    tensor.Tensor bT = b.transpose();
                    return aT.add(bT).mul(aT).sum();
                },
                environment -> ValidationReference.none(),
                environment -> WorkloadMetadata.of(name, WorkloadKind.GENERIC)
        );
    }

    public static Conv2dWorkloadSpec conv2dResnet3x3(String name) {
        return StandardWorkloads.conv2d(
                name,
                2, 64, 128, 56, 56, 3, 3,
                tensor.Conv2dOptions.defaults().withPadding(1, 1),
                true
        );
    }

    private static double[] random(int size, int seed) {
        java.util.Random random = new java.util.Random(seed);
        double[] out = new double[size];
        for (int i = 0; i < size; i++) {
            out[i] = Math.sin(i * 0.013 + seed * 0.07) + (random.nextDouble() - 0.5) * 0.1;
        }
        return out;
    }
}
