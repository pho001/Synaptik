package graph.execution;

import backend.ComputeEngine;
import backend.kernels.cpu.CpuDTypeOps;
import backend.runtime.ExecutionContext;
import backend.runtime.ExecutionMode;
import config.runtime.RuntimeConfig;
import tensor.Tensor;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public final class PreparedExecution {
    private final RuntimeConfig runtimeConfig;
    private final boolean supportsBackward;
    private final List<PreparedNodeExecution> forwardSteps;
    private final List<PreparedNodeExecution> backwardSteps;
    private final List<Tensor> allNodes;
    private final Tensor rootTensor;
    private final Tensor forwardOutput;

    public PreparedExecution(
            RuntimeConfig runtimeConfig,
            boolean supportsBackward,
            List<PreparedNodeExecution> forwardSteps,
            List<PreparedNodeExecution> backwardSteps,
            List<Tensor> allNodes,
            Tensor rootTensor,
            Tensor forwardOutput
    ) {
        this.runtimeConfig = Objects.requireNonNull(runtimeConfig, "runtimeConfig cannot be null");
        this.supportsBackward = supportsBackward;
        this.forwardSteps = List.copyOf(forwardSteps == null ? List.of() : forwardSteps);
        this.backwardSteps = List.copyOf(backwardSteps == null ? List.of() : backwardSteps);
        this.allNodes = List.copyOf(allNodes == null ? List.of() : allNodes);
        this.rootTensor = Objects.requireNonNull(rootTensor, "rootTensor cannot be null");
        this.forwardOutput = Objects.requireNonNull(forwardOutput, "forwardOutput cannot be null");
    }

    public RuntimeConfig runtimeConfig() {
        return runtimeConfig;
    }

    public boolean supportsBackward() {
        return supportsBackward;
    }

    public List<PreparedNodeExecution> forwardSteps() {
        return forwardSteps;
    }

    public List<PreparedNodeExecution> backwardSteps() {
        return backwardSteps;
    }

    public void execute(ExecutionMode mode) {
        Objects.requireNonNull(mode, "mode cannot be null");
        if (mode == ExecutionMode.FORWARD_BACKWARD && !supportsBackward) {
            throw new IllegalStateException("Prepared execution does not support backward execution.");
        }

        ExecutionContext context = new ExecutionContext(runtimeConfig.toBackendRuntimeConfig(), mode);
        for (PreparedNodeExecution step : forwardSteps) {
            ComputeEngine.compute(step.node(), step.metadata(), context);
        }

        syncRootData(mode);

        if (mode == ExecutionMode.FORWARD_BACKWARD) {
            zeroGrad();
            if (rootTensor.getGradient() != null) {
                fillGradientOnes(rootTensor.getGradient());
            }
            for (PreparedNodeExecution step : backwardSteps) {
                ComputeEngine.compute(step.node(), step.metadata(), context);
            }
        }
    }

    public void backward() {
        if (!supportsBackward) {
            System.out.println("Info: No gradients to compute.");
            return;
        }
        zeroGrad();
        if (rootTensor.getGradient() != null) {
            fillGradientOnes(rootTensor.getGradient());
        }

        ExecutionContext context = ExecutionContext.forwardBackward(runtimeConfig.toBackendRuntimeConfig());
        for (PreparedNodeExecution step : backwardSteps) {
            ComputeEngine.compute(step.node(), step.metadata(), context);
        }
    }

    private void syncRootData(ExecutionMode mode) {
        Tensor actualRoot = forwardOutput.getPrevTensors().get(0);
        if (mode == ExecutionMode.FORWARD_BACKWARD || actualRoot != rootTensor) {
            rootTensor.copyDataFrom(actualRoot);
        }
    }

    private void zeroGrad() {
        for (Tensor tensor : allNodes) {
            if (tensor.getGradient() != null) {
                fillGradientZeros(tensor.getGradient());
            }
        }
    }

    private static void fillGradientOnes(Tensor gradient) {
        switch (gradient.getDataType()) {
            case FLOAT64 -> Arrays.fill(gradient.getFloat64Data(), 1.0);
            case FLOAT32 -> Arrays.fill(gradient.getFloat32Data(), 1.0f);
            case FLOAT16 -> Arrays.fill(gradient.getFloat16Data(), CpuDTypeOps.toHalfBits(1.0f));
            case INT32, BOOL -> throw new UnsupportedOperationException("INT32/BOOL tensors do not support gradient seeding.");
        }
    }

    private static void fillGradientZeros(Tensor gradient) {
        switch (gradient.getDataType()) {
            case FLOAT64 -> Arrays.fill(gradient.getFloat64Data(), 0.0);
            case FLOAT32 -> Arrays.fill(gradient.getFloat32Data(), 0.0f);
            case FLOAT16 -> Arrays.fill(gradient.getFloat16Data(), (short) 0);
            case INT32 -> Arrays.fill(gradient.getInt32Data(), 0);
            case BOOL -> Arrays.fill(gradient.getBoolData(), (byte) 0);
        }
    }
}
