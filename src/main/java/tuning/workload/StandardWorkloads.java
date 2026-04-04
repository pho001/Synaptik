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
                .register(transformerHotPath("transformer_hot_path"));
    }

    public static WorkloadProfile transformerHotPathDefaults() {
        return WorkloadProfile.transformerHotPathDefaults();
    }
}
