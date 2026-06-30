package graph.execution;

import backend.cpu.nativecpu.NativeCpuMemoryPool;
import runtime.contract.ExecutionMode;
import config.runtime.NativeCpuMemoryConfig;
import config.runtime.NativeMemoryPoolPolicy;
import config.runtime.RuntimeConfig;
import graph.model.CompiledNode;
import graph.compile.GraphStructureContract;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.compile.publication.PublicationPlan;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import trace.prepare.PrepareTrace;
import trace.execution.RunTrace;
import graph.compile.planning.memory.MemoryPlan;
import tensor.Tensor;
import training.optimizer.TrainingOptimizer;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Runtime plan produced from {@link graph.compile.CompileArtifacts}.
 *
 * <p>Preparation resolves compile-time nodes to concrete execution operations, CPU kernels, backend metadata, memory
 * binding policy, and ordered forward/backward step lists for one {@link RuntimeConfig}. The prepared object is an
 * immutable description of how to run; each execution creates a fresh {@link ExecutionState}.
 *
 * <p>Running a prepared execution has side effects controlled by {@link PublicationPolicy}. The default execution
 * policy synchronizes output storage back to the source root tensor and publishes compiled gradients after backward
 * execution; lower-publication policies can keep values in the run-scoped execution state for benchmark and device
 * residency diagnostics. Concurrent calls against shared user-visible tensors or shared backend workspaces are not supported.
 */
public final class PreparedExecution implements AutoCloseable {
    private final RuntimeConfig runtimeConfig;
    private final boolean supportsBackward;
    private final List<PreparedExecutionStep> executionSteps;
    private final List<PreparedExecutionStep> forwardSteps;
    private final List<PreparedExecutionStep> backwardSteps;
    private final List<CompiledNode> allNodes;
    private final CompiledTensorDescriptorIndex descriptorIndex;
    private final PublicationPlan publicationPlan;
    private final CompiledNode forwardOutputNode;
    private final MemoryPlan memoryPlan;
    private final PrepareTrace prepareTrace;
    private final Map<Integer, CompiledNodeExecutionMetadata> metadataIndex;
    private final NativeCpuMemoryPool nativeCpuMemoryPool;
    private final AtomicBoolean closed;

    public PreparedExecution(
            RuntimeConfig runtimeConfig,
            boolean supportsBackward,
            List<PreparedExecutionStep> executionSteps,
            List<PreparedExecutionStep> forwardSteps,
            List<PreparedExecutionStep> backwardSteps,
            List<CompiledNode> allNodes,
            CompiledTensorDescriptorIndex descriptorIndex,
            PublicationPlan publicationPlan,
            CompiledNode forwardOutputNode,
            MemoryPlan memoryPlan,
            PrepareTrace prepareTrace
    ) {
        this.runtimeConfig = Objects.requireNonNull(runtimeConfig, "runtimeConfig cannot be null");
        this.supportsBackward = supportsBackward;
        this.executionSteps = List.copyOf(executionSteps == null ? List.of() : executionSteps);
        this.forwardSteps = List.copyOf(forwardSteps == null ? List.of() : forwardSteps);
        this.backwardSteps = List.copyOf(backwardSteps == null ? List.of() : backwardSteps);
        this.allNodes = List.copyOf(allNodes == null ? List.of() : allNodes);
        this.descriptorIndex = Objects.requireNonNull(descriptorIndex, "descriptorIndex cannot be null");
        this.publicationPlan = Objects.requireNonNull(publicationPlan, "publicationPlan cannot be null");
        this.forwardOutputNode = Objects.requireNonNull(forwardOutputNode, "forwardOutputNode cannot be null");
        this.memoryPlan = memoryPlan;
        this.prepareTrace = prepareTrace == null ? PrepareTrace.skipped() : prepareTrace;
        this.metadataIndex = buildMetadataIndex(this.executionSteps);
        this.nativeCpuMemoryPool = createNativeCpuMemoryPool(this.runtimeConfig.nativeCpuMemory());
        this.closed = new AtomicBoolean();
    }

    /**
     * Returns the runtime configuration used to prepare this plan.
     *
     * @return runtime configuration
     */
    public RuntimeConfig runtimeConfig() {
        return runtimeConfig;
    }

    /**
     * Returns whether this plan contains backward work.
     *
     * @return {@code true} when {@link ExecutionMode#FORWARD_BACKWARD} is valid
     */
    public boolean supportsBackward() {
        return supportsBackward;
    }

    /**
     * Returns immutable forward execution steps.
     *
     * @return forward step list
     */
    public List<PreparedExecutionStep> forwardSteps() {
        return forwardSteps;
    }

    /**
     * Returns immutable backward execution steps.
     *
     * @return backward step list, empty for inference-only plans
     */
    public List<PreparedExecutionStep> backwardSteps() {
        return backwardSteps;
    }

    /**
     * Returns the full execution sequence used for forward-backward mode.
     *
     * @return immutable full step list
     */
    public List<PreparedExecutionStep> executionSteps() {
        return executionSteps;
    }

    /**
     * Returns preparation timing and backend-selection metadata.
     *
     * @return prepare trace
     */
    public PrepareTrace prepareTrace() {
        return prepareTrace;
    }

    /**
     * Verifies that this plan was prepared from the expected compiled graph contract.
     *
     * @param expectedRootTensor compiled graph root tensor
     * @param expectedGraphContract compiled graph structure contract
     */
    public void requireCompatibleGraph(Tensor expectedRootTensor, GraphStructureContract expectedGraphContract) {
        if (publicationPlan.rootTensor() != expectedRootTensor || publicationPlan.graphContract() != expectedGraphContract) {
            throw new IllegalArgumentException("Prepared execution was created from a different compiled graph.");
        }
        publicationPlan.graphContract().validateOrThrow(publicationPlan.rootTensor());
    }

    /**
     * Closes resources owned by this prepared plan.
     */
    @Override
    public void close() {
        if (closed.compareAndSet(false, true) && nativeCpuMemoryPool != null) {
            nativeCpuMemoryPool.close();
        }
    }

    /**
     * Executes this prepared plan without collecting per-step trace metadata.
     *
     * @param mode execution mode to run
     * @throws NullPointerException if {@code mode} is {@code null}
     * @throws IllegalStateException if backward execution is requested but this plan has no backward steps
     */
    public void execute(ExecutionMode mode) {
        execute(mode, PublicationPolicy.defaultExecution());
    }

    /**
     * Executes this prepared plan without collecting per-step trace metadata.
     *
     * @param mode execution mode to run
     * @param publicationPolicy values to publish back to user-visible tensors after execution
     * @throws NullPointerException if {@code mode} is {@code null}
     * @throws IllegalStateException if backward execution is requested but this plan has no backward steps
     */
    public void execute(ExecutionMode mode, PublicationPolicy publicationPolicy) {
        executeInternal(mode, false, null, publicationPolicy);
    }

    /**
     * Executes this prepared plan and returns run-level diagnostics.
     *
     * @param mode execution mode to run
     * @return run trace containing duration and per-step metadata
     * @throws NullPointerException if {@code mode} is {@code null}
     * @throws IllegalStateException if backward execution is requested but this plan has no backward steps
     */
    public RunTrace executeTraced(ExecutionMode mode) {
        return executeTraced(mode, PublicationPolicy.defaultExecution());
    }

    /**
     * Executes this prepared plan and returns run-level diagnostics.
     *
     * @param mode execution mode to run
     * @param publicationPolicy values to publish back to user-visible tensors after execution
     * @return run trace containing duration and per-step metadata
     * @throws NullPointerException if {@code mode} is {@code null}
     * @throws IllegalStateException if backward execution is requested but this plan has no backward steps
     */
    public RunTrace executeTraced(ExecutionMode mode, PublicationPolicy publicationPolicy) {
        return executeInternal(mode, true, null, publicationPolicy);
    }

    /**
     * Executes forward/backward and applies an optimizer to trainable parameters without eager public gradient
     * publication.
     *
     * @param optimizer optimizer to apply
     */
    public void executeOptimizerStep(TrainingOptimizer optimizer) {
        executeOptimizerStep(optimizer, PublicationPolicy.defaultOptimizerStep());
    }

    /**
     * Executes forward/backward and applies an optimizer to trainable parameters.
     *
     * @param optimizer optimizer to apply
     * @param publicationPolicy values to publish back to user-visible tensors after execution
     */
    public void executeOptimizerStep(TrainingOptimizer optimizer, PublicationPolicy publicationPolicy) {
        executeInternal(
                ExecutionMode.FORWARD_BACKWARD,
                false,
                Objects.requireNonNull(optimizer, "optimizer cannot be null"),
                publicationPolicy
        );
    }

    /**
     * Executes forward/backward, applies an optimizer to trainable parameters, and returns run diagnostics.
     *
     * @param optimizer optimizer to apply
     * @return run trace
     */
    public RunTrace executeOptimizerStepTraced(TrainingOptimizer optimizer) {
        return executeOptimizerStepTraced(optimizer, PublicationPolicy.defaultOptimizerStep());
    }

    /**
     * Executes forward/backward, applies an optimizer to trainable parameters, and returns run diagnostics.
     *
     * @param optimizer optimizer to apply
     * @param publicationPolicy values to publish back to user-visible tensors after execution
     * @return run trace
     */
    public RunTrace executeOptimizerStepTraced(TrainingOptimizer optimizer, PublicationPolicy publicationPolicy) {
        return executeInternal(
                ExecutionMode.FORWARD_BACKWARD,
                true,
                Objects.requireNonNull(optimizer, "optimizer cannot be null"),
                publicationPolicy
        );
    }

    private RunTrace executeInternal(
            ExecutionMode mode,
            boolean captureTrace,
            TrainingOptimizer optimizer,
            PublicationPolicy publicationPolicy
    ) {
        ensureOpen();
        return new ExecutionRun(
                runtimeConfig,
                supportsBackward,
                executionSteps,
                forwardSteps,
                allNodes,
                descriptorIndex,
                metadataIndex,
                forwardOutputNode,
                publicationPlan,
                memoryPlan,
                nativeCpuMemoryPool,
                mode,
                captureTrace,
                optimizer,
                publicationPolicy
        ).execute();
    }

    /**
     * Convenience wrapper for {@link #execute(ExecutionMode)} in forward-backward mode.
     *
     * <p>If the plan has no backward steps, this method throws an exception instead of silently returning.
     */
    public void backward() {
        if (!supportsBackward) {
            throw new IllegalStateException("Prepared execution does not support backward execution.");
        }
        execute(ExecutionMode.FORWARD_BACKWARD);
    }

    private void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("Prepared execution is closed.");
        }
    }

    private static NativeCpuMemoryPool createNativeCpuMemoryPool(NativeCpuMemoryConfig config) {
        if (config != null && config.poolPolicy() == NativeMemoryPoolPolicy.PER_PREPARED_EXECUTION) {
            return new NativeCpuMemoryPool(config.maxPoolBytes());
        }
        return null;
    }

    private static Map<Integer, CompiledNodeExecutionMetadata> buildMetadataIndex(List<PreparedExecutionStep> executionSteps) {
        Map<Integer, CompiledNodeExecutionMetadata> out = new HashMap<>();
        for (PreparedExecutionStep step : executionSteps) {
            out.put(step.compiledNode().id(), step.metadata());
        }
        return Map.copyOf(out);
    }

}
