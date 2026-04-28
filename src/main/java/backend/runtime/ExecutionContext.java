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

/**
 * Per-run runtime context passed to backend kernels.
 *
 * <p>The context exposes execution mode, approximation decisions, prepared metadata, runtime tensor
 * lookup, per-node workspaces, and small backend-specific state maps. It is created for a prepared
 * execution run and should not be treated as a global singleton.</p>
 *
 * <p>The runtime-state and conv-trace maps are synchronized because individual backend helpers can
 * publish state while executing. The graph execution scheduler still owns the higher-level ordering and
 * lifetime of tensors and workspaces.</p>
 */
public final class ExecutionContext {
    private final ExecutionMode mode;
    private final boolean useFastExpApprox;
    private final boolean useFastTanhApprox;
    private final Map<Integer, CompiledNodeExecutionMetadata> metadataIndex;
    private final ExecutionState executionState;
    private final Map<Tensor, Object> runtimeStateIndex;
    private final Map<Integer, ConvTraceMetadata> convTraceIndex;

    /**
     * Creates a standalone context without prepared execution state.
     *
     * @param mode execution mode
     * @param useFastExpApprox whether exp may use fast approximation
     * @param useFastTanhApprox whether tanh may use fast approximation
     */
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

    /**
     * Creates a context from runtime config without prepared metadata or per-run execution state.
     *
     * @param runtimeConfig runtime config used to resolve approximation flags
     * @param mode execution mode
     * @return execution context
     */
    public static ExecutionContext fromRuntimeConfig(RuntimeConfig runtimeConfig, ExecutionMode mode) {
        return fromRuntimeConfig(runtimeConfig, mode, Map.of(), null);
    }

    /**
     * Creates a context from runtime config and prepared execution state.
     *
     * @param runtimeConfig runtime config used to resolve approximation flags
     * @param mode execution mode
     * @param metadataIndex prepared metadata indexed by compiled node id
     * @param executionState per-run tensor/workspace state; may be {@code null} for limited contexts
     * @return execution context
     */
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

    /**
     * @return current execution mode
     */
    public ExecutionMode mode() {
        return mode;
    }

    /**
     * @return {@code true} when the current run includes backward execution
     */
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

    /**
     * Returns the runtime tensor allocated for a compiled node id.
     *
     * @param nodeId compiled node id
     * @return runtime tensor for the node
     * @throws IllegalStateException if this context was created without per-run execution state
     */
    public Tensor runtimeTensorForNodeId(int nodeId) {
        if (executionState == null) {
            throw new IllegalStateException("ExecutionContext does not carry per-run ExecutionState.");
        }
        return executionState.runtimeTensorForNodeId(nodeId);
    }

    public backend.cpu.kernels.CpuNodeWorkspace cpuWorkspaceForNodeId(int nodeId) {
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

    public Integer nodeIdForRuntimeTensor(Tensor tensor) {
        if (executionState == null) {
            return null;
        }
        return executionState.nodeIdForRuntimeTensor(tensor);
    }

    /**
     * Reads backend-specific state attached to a tensor.
     *
     * @param tensor tensor identity used as key
     * @param type expected state type
     * @return state cast to {@code type}, or {@code null} when no state of that type exists
     */
    public <T> T runtimeStateFor(Tensor tensor, Class<T> type) {
        Objects.requireNonNull(type, "type cannot be null");
        Object state = runtimeStateIndex.get(tensor);
        return type.isInstance(state) ? type.cast(state) : null;
    }

    /**
     * Publishes backend-specific state for a tensor.
     *
     * @param tensor tensor identity used as key; must not be {@code null}
     * @param runtimeState state object, or {@code null} to remove the mapping
     */
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

    public void mirrorRuntimeState(Tensor source, Tensor target) {
        Objects.requireNonNull(target, "target cannot be null");
        if (source == null) {
            runtimeStateIndex.remove(target);
            return;
        }
        Object state = runtimeStateIndex.get(source);
        if (state == null) {
            runtimeStateIndex.remove(target);
            return;
        }
        runtimeStateIndex.put(target, state);
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
