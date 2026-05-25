package tuning.workload;

import backend.ComputeBackend;
import tensor.dtype.TensorDTypeOps;
import backend.runtime.ExecutionMode;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tuning.validate.ValidationReference;
import graph.compile.intent.BackendIntentPlan;

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
                .register(fusedAffineRationalStrided("calib_fused_affine_rational_strided", 256, 2048))
                .register(reductionSum("calib_reduction_sum", 262_144))
                .register(schedulerCheapParallel("calib_scheduler_cheap", 262_144))
                .register(materializationStridedElementwise("calib_materialization_strided", 256, 256))
                .register(materializationStridedWhere("calib_materialization_where", 256, 256))
                .register(conv2dPointwiseProjection("calib_conv2d_pointwise_low", 4, 128, 64, 8, 8))
                .register(conv2dPointwiseProjection("calib_conv2d_pointwise_edge_1m", 4, 128, 128, 8, 8))
                .register(conv2dPointwiseProjection("calib_conv2d_pointwise_edge_2m", 4, 128, 256, 8, 8))
                .register(conv2dPointwiseProjection("calib_conv2d_pointwise_edge_4m", 1, 128, 128, 16, 16))
                .register(conv2dResnet3x3("calib_conv2d_resnet_8_mid", 8, 64, 64, 8, 8))
                .register(conv2dResnet3x3("calib_conv2d_resnet_8_high", 8, 64, 128, 8, 8))
                .register(conv2dResnet3x3("calib_conv2d_resnet_3x3"));
    }

    public static MatMulWorkloadSpec matmulSquare(String name, int dim) {
        return StandardWorkloads.matmul(name, 1, dim, dim, dim);
    }

    public static MatMulWorkloadSpec matmulTallSkinny(String name, int m, int k, int n) {
        return StandardWorkloads.matmul(name, 1, m, k, n);
    }

    public static MatMulWorkloadSpec matmulWide(String name, int m, int k, int n) {
        return StandardWorkloads.matmul(name, 1, m, k, n);
    }

    public static MatMulWorkloadSpec matmulBatchedAttentionLike(String name, int batch, int m, int k, int n) {
        return StandardWorkloads.matmul(name, batch, m, k, n);
    }

    public static TensorRootWorkloadSpec appleMetalMatmulAddTanh(String name, int m, int k, int n) {
        return TensorRootWorkloadSpec.fromGraphFactory(
                name,
                WorkloadKind.MATMUL,
                (WorkloadGraphFactory) environment -> {
                    boolean requiresGrad = environment.profile().mode() == ExecutionMode.FORWARD_BACKWARD;
                    DataType dataType = environment.profile().dataType();
                    Tensor a = tensor(random(m * k, 701), new int[]{m, k}, "A", dataType);
                    Tensor b = tensor(random(k * n, 702), new int[]{k, n}, "B", dataType);
                    Tensor bias = tensor(random(n, 703), new int[]{n}, "BIAS", dataType);
                    if (requiresGrad) {
                        a.setRequiresGrad(true);
                        b.setRequiresGrad(true);
                        bias.setRequiresGrad(true);
                    }
                    Tensor matmul = a.matmul(b);
                    Tensor add = matmul.add(bias);
                    Tensor out = add.tanh();
                    Tensor root = environment.profile().mode() == ExecutionMode.FORWARD ? out : out.sum();
                    BackendIntentPlan backendIntentPlan = BackendIntentPlan.of(
                            ComputeBackend.GPU_METAL,
                            matmul,
                            add,
                            out
                    );
                    return new WorkloadGraph(root, backendIntentPlan);
                },
                environment -> ValidationReference.none(),
                environment -> WorkloadMetadata.of(name, WorkloadKind.MATMUL)
        );
    }

    public static TensorRootWorkloadSpec fusedCheapElementwise(String name, int size) {
        return new TensorRootWorkloadSpec(
                name,
                WorkloadKind.GENERIC,
                environment -> {
                    boolean requiresGrad = environment.profile().mode() == backend.runtime.ExecutionMode.FORWARD_BACKWARD;
                    tensor.Tensor a = tensor.factory.TensorDataFactory.shapedTensor("A", random(size, 101), requiresGrad, environment.profile().dataType(), size);
                    tensor.Tensor b = tensor.factory.TensorDataFactory.shapedTensor("B", random(size, 102), requiresGrad, environment.profile().dataType(), size);
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
                    tensor.Tensor a = tensor.factory.TensorDataFactory.shapedTensor("A", random(size, 103), requiresGrad, environment.profile().dataType(), size);
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
                    tensor.Tensor a = tensor.factory.TensorDataFactory.shapedTensor("A", random(size, 109), requiresGrad, environment.profile().dataType(), rows, cols);
                    tensor.Tensor b = tensor.factory.TensorDataFactory.shapedTensor("B", random(size, 110), requiresGrad, environment.profile().dataType(), rows, cols);
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
                    tensor.Tensor a = tensor.factory.TensorDataFactory.shapedTensor("A", random(size, 111), requiresGrad, environment.profile().dataType(), rows, cols);
                    tensor.Tensor aT = a.transpose();
                    return aT.exp().tanh().log().sigmoid().sum();
                },
                environment -> ValidationReference.none(),
                environment -> WorkloadMetadata.of(name, WorkloadKind.GENERIC)
        );
    }

    public static TensorRootWorkloadSpec fusedAffineRationalStrided(String name, int rows, int cols) {
        return new TensorRootWorkloadSpec(
                name,
                WorkloadKind.GENERIC,
                environment -> {
                    int size = rows * cols;
                    boolean requiresGrad = environment.profile().mode() == backend.runtime.ExecutionMode.FORWARD_BACKWARD;
                    tensor.Tensor stridedBase = tensor.factory.TensorDataFactory.shapedTensor(
                            "STRIDED_BASE",
                            random(size, 311),
                            requiresGrad,
                            environment.profile().dataType(),
                            cols,
                            rows
                    );
                    tensor.Tensor scale = tensor.factory.TensorDataFactory.shapedTensor("SCALE", random(size, 312), requiresGrad, environment.profile().dataType(), rows, cols);
                    tensor.Tensor bias = tensor.factory.TensorDataFactory.shapedTensor("BIAS", random(size, 313), requiresGrad, environment.profile().dataType(), rows, cols);
                    tensor.Tensor grad = tensor.factory.TensorDataFactory.shapedTensor("GRAD", random(size, 314), requiresGrad, environment.profile().dataType(), rows, cols);
                    tensor.Tensor denom = tensor.factory.TensorDataFactory.shapedTensor("DENOM", positiveRandom(size, 315, 0.75), requiresGrad, environment.profile().dataType(), rows, cols);
                    tensor.Tensor residual = tensor.factory.TensorDataFactory.shapedTensor("RESIDUAL", random(size, 316), requiresGrad, environment.profile().dataType(), rows, cols);
                    tensor.Tensor strided = stridedBase.transpose();
                    return strided.neg()
                            .mul(scale)
                            .add(bias)
                            .mul(grad)
                            .div(denom)
                            .add(residual)
                            .sum();
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
                    return tensor.factory.TensorDataFactory.shapedTensor("REDUCE", random(size, 104), requiresGrad, environment.profile().dataType(), size).sum();
                },
                environment -> ValidationReference.none(),
                environment -> WorkloadMetadata.of(name, WorkloadKind.GENERIC)
        );
    }

    public static TensorRootWorkloadSpec maskedAttention(String name, int batch, int heads, int tokens, int headDim, int valueDim) {
        return new TensorRootWorkloadSpec(
                name,
                WorkloadKind.GENERIC,
                environment -> {
                    DataType dataType = environment.profile().dataType();
                    ExecutionMode mode = environment.profile().mode();
                    Tensor q = tensor(queryData(batch, heads, tokens, headDim), new int[]{batch, heads, tokens, headDim}, "q", dataType);
                    Tensor k = tensor(keyData(batch, heads, tokens, headDim), new int[]{batch, heads, tokens, headDim}, "k", dataType);
                    Tensor v = tensor(valueData(batch, heads, tokens, valueDim), new int[]{batch, heads, tokens, valueDim}, "v", dataType);
                    if (mode == ExecutionMode.FORWARD_BACKWARD) {
                        q.setRequiresGrad(true);
                        k.setRequiresGrad(true);
                        v.setRequiresGrad(true);
                    }
                    Tensor mask = new Tensor(maskData(batch, heads, tokens), new int[]{batch, heads, tokens, tokens}, null, "mask", DataType.BOOL);
                    double scale = 1.0d / Math.sqrt(headDim);
                    Tensor scores = q.matmul(k.permute(0, 1, 3, 2)).mul(scale);
                    Tensor out = Tensor.where(mask, scores, Tensor.scalar(maskFillValue(dataType), dataType))
                            .softmax(3)
                            .matmul(v);
                    return mode == ExecutionMode.FORWARD ? out : out.sum();
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
                    tensor.Tensor a = tensor.factory.TensorDataFactory.shapedTensor("A", random(size, 105), requiresGrad, environment.profile().dataType(), size);
                    tensor.Tensor b = tensor.factory.TensorDataFactory.shapedTensor("B", random(size, 106), requiresGrad, environment.profile().dataType(), size);
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
                    tensor.Tensor a = tensor.factory.TensorDataFactory.shapedTensor("A", random(size, 107), requiresGrad, environment.profile().dataType(), rows, cols);
                    tensor.Tensor b = tensor.factory.TensorDataFactory.shapedTensor("B", random(size, 108), requiresGrad, environment.profile().dataType(), rows, cols);
                    tensor.Tensor aT = a.transpose();
                    tensor.Tensor bT = b.transpose();
                    return aT.add(bT).mul(aT).sum();
                },
                environment -> ValidationReference.none(),
                environment -> WorkloadMetadata.of(name, WorkloadKind.GENERIC)
        );
    }

    public static TensorRootWorkloadSpec materializationStridedWhere(String name, int rows, int cols) {
        return new TensorRootWorkloadSpec(
                name,
                WorkloadKind.GENERIC,
                environment -> {
                    int size = rows * cols;
                    boolean requiresGrad = environment.profile().mode() == backend.runtime.ExecutionMode.FORWARD_BACKWARD;
                    tensor.Tensor a = tensor.factory.TensorDataFactory.shapedTensor("A", random(size, 211), requiresGrad, environment.profile().dataType(), rows, cols);
                    tensor.Tensor b = tensor.factory.TensorDataFactory.shapedTensor("B", random(size, 212), requiresGrad, environment.profile().dataType(), rows, cols);
                    tensor.Tensor maskBase = new Tensor(maskPattern(size), new int[]{rows, cols}, null, "MASK", DataType.BOOL);
                    tensor.Tensor aT = a.transpose();
                    tensor.Tensor bT = b.transpose();
                    tensor.Tensor maskT = maskBase.transpose();
                    return Tensor.where(maskT, aT, bT).add(aT).sum();
                },
                environment -> ValidationReference.none(),
                environment -> WorkloadMetadata.of(name, WorkloadKind.GENERIC)
        );
    }

    public static Conv2dWorkloadSpec conv2dResnet3x3(String name) {
        return conv2dResnet3x3(name, 2, 64, 128, 56, 56);
    }

    public static Conv2dWorkloadSpec conv2dResnet3x3(
            String name,
            int batch,
            int inChannels,
            int outChannels,
            int height,
            int width
    ) {
        return StandardWorkloads.conv2d(
                name,
                batch, inChannels, outChannels, height, width, 3, 3,
                tensor.options.Conv2dOptions.defaults().withPadding(1, 1),
                true
        );
    }

    public static Conv2dWorkloadSpec conv2dPointwiseProjection(
            String name,
            int batch,
            int inChannels,
            int outChannels,
            int height,
            int width
    ) {
        return StandardWorkloads.conv2d(
                name,
                batch, inChannels, outChannels, height, width, 1, 1,
                tensor.options.Conv2dOptions.defaults(),
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

    private static double[] positiveRandom(int size, int seed, double floor) {
        java.util.Random random = new java.util.Random(seed);
        double[] out = new double[size];
        for (int i = 0; i < size; i++) {
            out[i] = floor + Math.abs(Math.sin(i * 0.017 + seed * 0.11)) + random.nextDouble() * 0.25;
        }
        return out;
    }

    private static byte[] maskPattern(int size) {
        byte[] out = new byte[size];
        for (int i = 0; i < size; i++) {
            out[i] = (byte) (((i * 17 + 3) & 1) == 0 ? 1 : 0);
        }
        return out;
    }

    private static Tensor tensor(double[] values, int[] shape, String label, DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> new Tensor(values.clone(), shape, null, label, DataType.FLOAT64);
            case FLOAT32 -> new Tensor(toF32(values), shape, null, label, DataType.FLOAT32);
            case BFLOAT16 -> new Tensor(toBf16(values), shape, null, label, DataType.BFLOAT16);
            case INT32, INT64, BOOL -> throw new IllegalArgumentException("attention workload requires floating dtype");
        };
    }

    private static float[] toF32(double[] src) {
        float[] out = new float[src.length];
        for (int i = 0; i < src.length; i++) {
            out[i] = (float) src[i];
        }
        return out;
    }

    private static short[] toBf16(double[] src) {
        short[] out = new short[src.length];
        for (int i = 0; i < src.length; i++) {
            out[i] = TensorDTypeOps.toBFloat16Bits((float) src[i]);
        }
        return out;
    }

    private static double maskFillValue(DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> -1.0e30d;
            case FLOAT32 -> -1.0e9d;
            case BFLOAT16 -> -1.0e9d;
            case INT32, INT64, BOOL -> throw new IllegalArgumentException("attention workload requires floating dtype");
        };
    }

    private static double[] queryData(int batch, int heads, int tokens, int headDim) {
        int size = batch * heads * tokens * headDim;
        double[] out = new double[size];
        for (int i = 0; i < size; i++) {
            out[i] = Math.sin(i * 0.013) + Math.cos(i * 0.003) * 0.25 + (i % 7) * 0.03125;
        }
        return out;
    }

    private static double[] keyData(int batch, int heads, int tokens, int headDim) {
        int size = batch * heads * tokens * headDim;
        double[] out = new double[size];
        for (int i = 0; i < size; i++) {
            out[i] = Math.cos(i * 0.017) + Math.sin(i * 0.005) * 0.2 + (i % 11) * 0.015625;
        }
        return out;
    }

    private static double[] valueData(int batch, int heads, int tokens, int valueDim) {
        int size = batch * heads * tokens * valueDim;
        double[] out = new double[size];
        for (int i = 0; i < size; i++) {
            out[i] = Math.sin(i * 0.019) * 0.75 + Math.cos(i * 0.007) * 0.5 + ((i / valueDim) % 13) * 0.015625;
        }
        return out;
    }

    private static byte[] maskData(int batch, int heads, int tokens) {
        int size = batch * heads * tokens * tokens;
        byte[] out = new byte[size];
        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < heads; h++) {
                for (int q = 0; q < tokens; q++) {
                    for (int k = 0; k < tokens; k++) {
                        int index = ((b * heads + h) * tokens + q) * tokens + k;
                        out[index] = (byte) ((k <= q || ((q + k + h) & 3) != 0) ? 1 : 0);
                    }
                }
            }
        }
        return out;
    }
}
