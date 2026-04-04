package config.optimizer;

public record RewriteConfig(
        Conv2dLoweringConfig conv2dLowering
) {
    public RewriteConfig {
        conv2dLowering = conv2dLowering == null ? Conv2dLoweringConfig.defaults() : conv2dLowering;
    }

    public static RewriteConfig defaults() {
        return new RewriteConfig(Conv2dLoweringConfig.defaults());
    }

    public RewriteConfig withConv2dLowering(Conv2dLoweringConfig newConv2dLowering) {
        return new RewriteConfig(newConv2dLowering);
    }
}
