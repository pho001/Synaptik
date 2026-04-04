package backend.runtime;

import backend.blas.BlasThreadController;

import java.util.Objects;

public final class ExecutionContext {
    private final RuntimeConfig runtimeConfig;
    private final ExecutionMode mode;
    private final backend.kernels.cpu.CpuExecutionPlanner cpuPlanner;
    private final boolean useFastExpApprox;
    private final boolean useFastTanhApprox;

    public ExecutionContext(RuntimeConfig runtimeConfig, ExecutionMode mode) {
        this.runtimeConfig = Objects.requireNonNull(runtimeConfig, "runtimeConfig cannot be null");
        this.mode = Objects.requireNonNull(mode, "mode cannot be null");
        BlasThreadController.apply(runtimeConfig.blasConfig());
        this.cpuPlanner = backend.kernels.cpu.CpuExecutionPlanner.from(runtimeConfig.cpuKernelConfig());
        boolean backwardEnabled = mode == ExecutionMode.FORWARD_BACKWARD;
        this.useFastExpApprox = runtimeConfig.approximationConfig().useFastExp(backwardEnabled);
        this.useFastTanhApprox = runtimeConfig.approximationConfig().useFastTanh(backwardEnabled);
    }

    public static ExecutionContext forwardBackward(RuntimeConfig runtimeConfig) {
        return new ExecutionContext(runtimeConfig, ExecutionMode.FORWARD_BACKWARD);
    }

    public static ExecutionContext forward(RuntimeConfig runtimeConfig) {
        return new ExecutionContext(runtimeConfig, ExecutionMode.FORWARD);
    }

    public RuntimeConfig runtimeConfig() {
        return runtimeConfig;
    }

    public ExecutionMode mode() {
        return mode;
    }

    public boolean runsBackwardPass() {
        return mode == ExecutionMode.FORWARD_BACKWARD;
    }

    public boolean useFastExpApprox() {
        return useFastExpApprox;
    }

    public boolean useFastTanhApprox() {
        return useFastTanhApprox;
    }

    public backend.kernels.cpu.CpuExecutionPlanner cpuPlanner() {
        return cpuPlanner;
    }
}
