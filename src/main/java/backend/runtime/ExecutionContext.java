package backend.runtime;

import config.runtime.RuntimeConfig;
import graph.execution.CompiledNodeExecutionMetadata;
import graph.execution.ExecutionState;
import graph.execution.trace.ConvTraceMetadata;
import tensor.Tensor;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.Objects;

public final class ExecutionContext {
    private final ExecutionMode mode;
    private final boolean useFastExpApprox;
    private final boolean useFastTanhApprox;
    private final Map<Integer, CompiledNodeExecutionMetadata> metadataIndex;
    private final ExecutionState executionState;
    private final Map<Tensor, Object> runtimeStateIndex;
    private final Map<Integer, ConvTraceMetadata> convTraceIndex;

    public ExecutionContext(ExecutionMode mode, boolean useFastExpApprox, boolean useFastTanhApprox) {
        this(mode, useFastExpApprox, useFastTanhApprox, Map.of(), null);
    }

    public ExecutionContext(
            ExecutionMode mode,
            boolean useFastExpApprox,
            boolean useFastTanhApprox,
            Map<Integer, CompiledNodeExecutionMetadata> metadataIndex,
            ExecutionState executionState
    ) {
        this.mode = Objects.requireNonNull(mode, "mode cannot be null");
        this.metadataIndex = Map.copyOf(metadataIndex == null ? Map.of() : metadataIndex);
        this.executionState = executionState;
        this.runtimeStateIndex = Collections.synchronizedMap(new IdentityHashMap<>());
        this.convTraceIndex = Collections.synchronizedMap(new java.util.HashMap<>());
        this.useFastExpApprox = useFastExpApprox;
        this.useFastTanhApprox = useFastTanhApprox;
    }

    public static ExecutionContext fromRuntimeConfig(RuntimeConfig runtimeConfig, ExecutionMode mode) {
        return fromRuntimeConfig(runtimeConfig, mode, Map.of(), null);
    }

    public static ExecutionContext fromRuntimeConfig(
            RuntimeConfig runtimeConfig,
            ExecutionMode mode,
            Map<Integer, CompiledNodeExecutionMetadata> metadataIndex,
            ExecutionState executionState
    ) {
        Objects.requireNonNull(runtimeConfig, "runtimeConfig cannot be null");
        Objects.requireNonNull(mode, "mode cannot be null");
        boolean backwardEnabled = mode == ExecutionMode.FORWARD_BACKWARD;
        return new ExecutionContext(
                mode,
                runtimeConfig.approximation().useFastExp(backwardEnabled),
                runtimeConfig.approximation().useFastTanh(backwardEnabled),
                metadataIndex,
                executionState
        );
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

    public CompiledNodeExecutionMetadata metadataForNodeId(int nodeId) {
        return metadataIndex.get(nodeId);
    }

    public Tensor runtimeTensorForNodeId(int nodeId) {
        if (executionState == null) {
            throw new IllegalStateException("ExecutionContext does not carry per-run ExecutionState.");
        }
        return executionState.runtimeTensorForNodeId(nodeId);
    }

    public backend.kernels.cpu.CpuNodeWorkspace cpuWorkspaceForNodeId(int nodeId) {
        if (executionState == null) {
            return null;
        }
        return executionState.cpuWorkspaceForNodeId(nodeId);
    }

    public Tensor preparedInputTensorFor(int nodeId, int inputIndex) {
        if (executionState == null) {
            throw new IllegalStateException("ExecutionContext does not carry per-run ExecutionState.");
        }
        return executionState.preparedInputTensorFor(nodeId, inputIndex);
    }

    public <T> T runtimeStateFor(Tensor tensor, Class<T> type) {
        Objects.requireNonNull(type, "type cannot be null");
        Object state = runtimeStateIndex.get(tensor);
        return type.isInstance(state) ? type.cast(state) : null;
    }

    public void putRuntimeState(Tensor tensor, Object runtimeState) {
        Objects.requireNonNull(tensor, "tensor cannot be null");
        if (runtimeState == null) {
            runtimeStateIndex.remove(tensor);
            return;
        }
        runtimeStateIndex.put(tensor, runtimeState);
    }

    public void clearRuntimeState(Tensor tensor) {
        if (tensor != null) {
            runtimeStateIndex.remove(tensor);
        }
    }

    public ConvTraceMetadata convTraceForNodeId(int nodeId) {
        return convTraceIndex.get(nodeId);
    }

    public void publishConvTrace(int nodeId, ConvTraceMetadata trace) {
        if (trace == null) {
            convTraceIndex.remove(nodeId);
            return;
        }
        convTraceIndex.put(nodeId, trace);
    }
}
