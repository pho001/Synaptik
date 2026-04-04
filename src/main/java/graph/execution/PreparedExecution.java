package graph.execution;

import backend.ComputeEngine;
import backend.kernels.cpu.CpuDTypeOps;
import backend.runtime.ExecutionContext;
import backend.runtime.ExecutionMode;
import config.runtime.RuntimeConfig;
import graph.execution.trace.DispatchTraceMetadata;
import graph.execution.trace.ExecutionStepTrace;
import graph.execution.trace.FusedTraceMetadata;
import graph.execution.trace.LayoutTraceMetadata;
import graph.execution.trace.MatMulTraceMetadata;
import graph.execution.trace.PrepareTrace;
import graph.execution.trace.ReductionTraceMetadata;
import graph.execution.trace.RunTrace;
import graph.execution.trace.StepExecutionMetadata;
import tensor.Tensor;

import java.util.Arrays;
import java.util.LinkedHashMap;
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
    private final PrepareTrace prepareTrace;

    public PreparedExecution(
            RuntimeConfig runtimeConfig,
            boolean supportsBackward,
            List<PreparedNodeExecution> forwardSteps,
            List<PreparedNodeExecution> backwardSteps,
            List<Tensor> allNodes,
            Tensor rootTensor,
            Tensor forwardOutput,
            PrepareTrace prepareTrace
    ) {
        this.runtimeConfig = Objects.requireNonNull(runtimeConfig, "runtimeConfig cannot be null");
        this.supportsBackward = supportsBackward;
        this.forwardSteps = List.copyOf(forwardSteps == null ? List.of() : forwardSteps);
        this.backwardSteps = List.copyOf(backwardSteps == null ? List.of() : backwardSteps);
        this.allNodes = List.copyOf(allNodes == null ? List.of() : allNodes);
        this.rootTensor = Objects.requireNonNull(rootTensor, "rootTensor cannot be null");
        this.forwardOutput = Objects.requireNonNull(forwardOutput, "forwardOutput cannot be null");
        this.prepareTrace = prepareTrace == null ? PrepareTrace.skipped() : prepareTrace;
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

    public PrepareTrace prepareTrace() {
        return prepareTrace;
    }

    public void execute(ExecutionMode mode) {
        executeInternal(mode, false);
    }

    public RunTrace executeTraced(ExecutionMode mode) {
        return executeInternal(mode, true);
    }

    private RunTrace executeInternal(ExecutionMode mode, boolean captureTrace) {
        Objects.requireNonNull(mode, "mode cannot be null");
        if (mode == ExecutionMode.FORWARD_BACKWARD && !supportsBackward) {
            throw new IllegalStateException("Prepared execution does not support backward execution.");
        }

        long runStart = System.nanoTime();
        java.util.ArrayList<ExecutionStepTrace> steps = captureTrace ? new java.util.ArrayList<>() : null;
        ExecutionContext context = new ExecutionContext(runtimeConfig.toBackendRuntimeConfig(), mode);
        executeSteps(forwardSteps, context, captureTrace, steps, 0);

        syncRootData(mode);

        if (mode == ExecutionMode.FORWARD_BACKWARD) {
            zeroGrad();
            if (rootTensor.getGradient() != null) {
                fillGradientOnes(rootTensor.getGradient());
            }
            executeSteps(backwardSteps, context, captureTrace, steps, forwardSteps.size());
        }
        return new RunTrace(mode, System.nanoTime() - runStart, steps == null ? List.of() : steps);
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

    private static void executeSteps(
            List<PreparedNodeExecution> steps,
            ExecutionContext context,
            boolean captureTrace,
            List<ExecutionStepTrace> traces,
            int startIndex
    ) {
        for (int i = 0; i < steps.size(); i++) {
            PreparedNodeExecution step = steps.get(i);
            long t0 = captureTrace ? System.nanoTime() : 0L;
            ComputeEngine.compute(step.node(), step.metadata(), context);
            if (captureTrace) {
                traces.add(toStepTrace(startIndex + i, step, System.nanoTime() - t0));
            }
        }
    }

    private static ExecutionStepTrace toStepTrace(int index, PreparedNodeExecution step, long durationNs) {
        Tensor node = step.node();
        var metadata = step.metadata();
        String opType = node.getOperation() == null ? "LEAF" : node.getOperation().opType().name();
        String kernel = metadata.cpuKernel() == null ? "" : metadata.cpuKernel().getClass().getSimpleName();
        return new ExecutionStepTrace(
                index,
                node.getLabel(),
                opType,
                java.util.Arrays.stream(node.getShapeUnsafe()).boxed().toList(),
                node.getDataType(),
                metadata.backend().name(),
                kernel,
                durationNs,
                buildStepMetadata(step)
        );
    }

    private static StepExecutionMetadata buildStepMetadata(PreparedNodeExecution step) {
        Tensor node = step.node();
        var metadata = step.metadata();
        LinkedHashMap<String, Object> attrs = new LinkedHashMap<>();
        LayoutTraceMetadata layout = new LayoutTraceMetadata(
                node.getStorageOffsetUnsafe(),
                node.isContiguous(),
                metadata.cpuPlan() != null && metadata.cpuPlan().stridedPath(),
                metadata.cpuPlan() == null ? "" : metadata.cpuPlan().targetType().name()
        );
        DispatchTraceMetadata dispatch = null;
        ReductionTraceMetadata reduction = null;
        MatMulTraceMetadata matMul = null;
        FusedTraceMetadata fusedMeta = null;

        if (metadata.cpuPlan() != null) {
            var plan = metadata.cpuPlan();
            if (plan.dispatchHints() != null) {
                dispatch = new DispatchTraceMetadata(
                        plan.dispatchHints().mode().name(),
                        plan.dispatchHints().vectorWidth(),
                        plan.dispatchHints().plannedWorkers(),
                        plan.dispatchHints().scalarChunkSize(),
                        plan.dispatchHints().vectorChunkSize()
                );
            }
            if (plan.reductionHints() != null) {
                reduction = new ReductionTraceMetadata(
                        plan.reductionHints().mode().name(),
                        plan.reductionHints().plannedWorkers(),
                        plan.reductionHints().chunkSize(),
                        plan.reductionHints().vectorWidth(),
                        plan.reductionHints().accuracyMode().name()
                );
            }
            if (plan.matMulHints() != null) {
                matMul = new MatMulTraceMetadata(
                        plan.matMulHints().useBlas(),
                        plan.matMulHints().useBatchedBlas(),
                        plan.matMulHints().parallel(),
                        plan.matMulHints().tileM(),
                        plan.matMulHints().tileN(),
                        plan.matMulHints().tileK(),
                        plan.matMulHints().plannedWorkers(),
                        plan.matMulHints().work()
                );
            }
        }

        if (node.getOperation() instanceof operations.FusedOperation fused) {
            fusedMeta = new FusedTraceMetadata(
                    fused.getPrecisionMode(),
                    fused.isLowCostHint(),
                    fused.getSchedulerSignature(),
                    fused.getDispatchScale(),
                    fused.getPlan().nodeCount(),
                    fused.getPlan().inputCount()
            );
        }

        return new StepExecutionMetadata("node", attrs, layout, dispatch, reduction, matMul, fusedMeta);
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
