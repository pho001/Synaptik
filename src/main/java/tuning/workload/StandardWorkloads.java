package tuning.workload;

import config.profile.WorkloadProfile;
import tensor.Conv2dOptions;

public final class StandardWorkloads {
    private StandardWorkloads() {
    }

    public static MatMulWorkloadSpec matmul(String name, int batch, int m, int k, int n) {
        return new MatMulWorkloadSpec(name, batch, m, k, n);
    }

    public static Conv2dWorkloadSpec conv2d(
            String name,
            int batch,
            int inChannels,
            int outChannels,
            int height,
            int width,
            int kernelH,
            int kernelW,
            Conv2dOptions options,
            boolean withBias
    ) {
        return new Conv2dWorkloadSpec(name, batch, inChannels, outChannels, height, width, kernelH, kernelW, options, withBias);
    }

    public static TransformerHotPathWorkloadSpec transformerHotPath(String name) {
        return new TransformerHotPathWorkloadSpec(name);
    }

    public static NormalizationWorkloadSpec normalization(
            String name,
            NormalizationWorkloadSpec.NormalizationKind kind,
            int batch,
            int channels,
            int height,
            int width,
            double epsilon
    ) {
        return new NormalizationWorkloadSpec(name, kind, batch, channels, height, width, epsilon);
    }

    public static Pool2dWorkloadSpec pool2d(
            String name,
            Pool2dWorkloadSpec.PoolKind kind,
            int batch,
            int channels,
            int height,
            int width,
            tensor.Pool2dOptions options
    ) {
        return new Pool2dWorkloadSpec(name, kind, batch, channels, height, width, options);
    }

    public static LossWorkloadSpec indexedLoss(
            String name,
            LossWorkloadSpec.LossKind kind,
            int batch,
            int classes,
            tensor.LossReduction reduction
    ) {
        return new LossWorkloadSpec(name, kind, batch, classes, reduction);
    }

    public static WorkloadCatalog defaultCatalog() {
        return new WorkloadCatalog()
                .register(matmul("matmul_small", 1, 64, 64, 64))
                .register(matmul("matmul_batched_attention_like", 8, 128, 64, 64))
                .register(conv2d(
                        "conv2d_resnet_3x3",
                        2, 64, 128, 56, 56, 3, 3,
                        new Conv2dOptions(1, 1, 1, 1, 1, 1, 1),
                        true
                ))
                .register(normalization("layer_norm_small", NormalizationWorkloadSpec.NormalizationKind.LAYER_NORM, 4, 64, 8, 1, 1e-5))
                .register(pool2d("max_pool2d_small", Pool2dWorkloadSpec.PoolKind.MAX, 2, 8, 16, 16, tensor.Pool2dOptions.square(2)))
                .register(indexedLoss("cross_entropy_small", LossWorkloadSpec.LossKind.CROSS_ENTROPY_FROM_INDICES, 8, 16, tensor.LossReduction.MEAN))
                .register(transformerHotPath("transformer_hot_path"));
    }

    public static WorkloadProfile transformerHotPathDefaults() {
        return WorkloadProfile.transformerHotPathDefaults();
    }
}
