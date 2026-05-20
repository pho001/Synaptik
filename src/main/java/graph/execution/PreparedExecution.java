package graph.execution;

import backend.cpu.kernels.CpuDTypeOps;
import backend.cpu.nativecpu.NativeCpuMemoryPool;
import backend.runtime.ExecutionContext;
import backend.runtime.ExecutionMode;
import config.runtime.NativeCpuMemoryConfig;
import config.runtime.NativeMemoryPoolPolicy;
import config.runtime.RuntimeConfig;
import graph.CompiledNode;
import graph.CompiledGradientBinding;
import graph.GradientDTypePolicy;
import graph.compile.GraphStructureContract;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.publication.ExecutionPublisher;
import graph.execution.residency.RuntimeMemoryBinder;
import graph.execution.runner.PreparedExecutionRunner;
import graph.execution.state.ExecutionState;
import graph.execution.trace.ExecutionStepTrace;
import graph.execution.trace.PrepareTrace;
import graph.execution.trace.RunTrace;
import graph.compile.planning.memory.MemoryPlan;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import training.optimizer.OptimizerStepContext;
import training.optimizer.TrainingOptimizer;

import java.util.Arrays;
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
 * residency diagnostics. Concurrent calls against shared source tensors or shared backend workspaces are not supported.
 */
public final class PreparedExecution implements AutoCloseable {
    private final RuntimeConfig runtimeConfig;
    private final boolean supportsBackward;
    private final List<PreparedNodeExecution> executionSteps;
    private final List<PreparedNodeExecution> forwardSteps;
    private final List<PreparedNodeExecution> backwardSteps;
    private final List<CompiledNode> allNodes;
    private final CompiledTensorDescriptorIndex descriptorIndex;
    private final Map<Tensor, CompiledGradientBinding> compiledGradients;
    private final Tensor rootTensor;
    private final GraphStructureContract graphContract;
    private final CompiledNode forwardOutputNode;
    private final CompiledGradientBinding forwardSeedGradient;
    private final MemoryPlan memoryPlan;
    private final PrepareTrace prepareTrace;
    private final Map<Integer, CompiledNodeExecutionMetadata> metadataIndex;
    private final NativeCpuMemoryPool nativeCpuMemoryPool;
    private final AtomicBoolean closed;

    /**
     * Creates a prepared execution from already lowered step metadata.
     *
     * @param runtimeConfig runtime configuration used to select kernels and execution metadata
     * @param supportsBackward whether backward steps are available
     * @param executionSteps full ordered step list used by forward-backward execution
     * @param forwardSteps forward-only step list
     * @param backwardSteps backward-only step list
     * @param allNodes all compiled nodes in graph order
     * @param descriptorIndex immutable tensor descriptor facts for {@code allNodes}
     * @param compiledGradients gradient publication bindings
     * @param rootTensor source root tensor to synchronize after execution
     * @param forwardOutputNode compiled node that holds the forward result
     * @param forwardSeedGradient binding used to seed backward execution
     * @param memoryPlan memory reuse and region binding plan, possibly {@code null}
     * @param prepareTrace preparation diagnostics and timing metadata
     */
    public PreparedExecution(
            RuntimeConfig runtimeConfig,
            boolean supportsBackward,
            List<PreparedNodeExecution> executionSteps,
            List<PreparedNodeExecution> forwardSteps,
            List<PreparedNodeExecution> backwardSteps,
            List<CompiledNode> allNodes,
            CompiledTensorDescriptorIndex descriptorIndex,
            Map<Tensor, CompiledGradientBinding> compiledGradients,
            Tensor rootTensor,
            CompiledNode forwardOutputNode,
            CompiledGradientBinding forwardSeedGradient,
            MemoryPlan memoryPlan,
            PrepareTrace prepareTrace
    ) {
        this(
                runtimeConfig,
                supportsBackward,
                executionSteps,
                forwardSteps,
                backwardSteps,
                allNodes,
                descriptorIndex,
                compiledGradients,
                rootTensor,
                GraphStructureContract.unchecked(),
                forwardOutputNode,
                forwardSeedGradient,
                memoryPlan,
                prepareTrace
        );
    }

    /**
     * Creates a prepared execution from already lowered step metadata and a source graph contract.
     *
     * @param graphContract compile-time user-visible graph structure contract
     */
    public PreparedExecution(
            RuntimeConfig runtimeConfig,
            boolean supportsBackward,
            List<PreparedNodeExecution> executionSteps,
            List<PreparedNodeExecution> forwardSteps,
            List<PreparedNodeExecution> backwardSteps,
            List<CompiledNode> allNodes,
            CompiledTensorDescriptorIndex descriptorIndex,
            Map<Tensor, CompiledGradientBinding> compiledGradients,
            Tensor rootTensor,
            GraphStructureContract graphContract,
            CompiledNode forwardOutputNode,
            CompiledGradientBinding forwardSeedGradient,
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
        this.compiledGradients = Map.copyOf(compiledGradients == null ? Map.of() : compiledGradients);
        this.rootTensor = Objects.requireNonNull(rootTensor, "rootTensor cannot be null");
        this.graphContract = graphContract == null ? GraphStructureContract.unchecked() : graphContract;
        this.forwardOutputNode = Objects.requireNonNull(forwardOutputNode, "forwardOutputNode cannot be null");
        this.forwardSeedGradient = forwardSeedGradient;
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
    public List<PreparedNodeExecution> forwardSteps() {
        return forwardSteps;
    }

    /**
     * Returns immutable backward execution steps.
     *
     * @return backward step list, empty for inference-only plans
     */
    public List<PreparedNodeExecution> backwardSteps() {
        return backwardSteps;
    }

    /**
     * Returns the full execution sequence used for forward-backward mode.
     *
     * @return immutable full step list
     */
    public List<PreparedNodeExecution> executionSteps() {
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
        if (rootTensor != expectedRootTensor || graphContract != expectedGraphContract) {
            throw new IllegalArgumentException("Prepared execution was created from a different compiled graph.");
        }
        graphContract.validateOrThrow(rootTensor);
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
        graphContract.validateOrThrow(rootTensor);
        Objects.requireNonNull(mode, "mode cannot be null");
        PublicationPolicy publication = publicationPolicy == null
                ? (optimizer == null ? PublicationPolicy.defaultExecution() : PublicationPolicy.defaultOptimizerStep())
                : publicationPolicy;
        if (mode == ExecutionMode.FORWARD_BACKWARD && !supportsBackward) {
            throw new IllegalStateException("Prepared execution does not support backward execution.");
        }
        if (optimizer != null && mode != ExecutionMode.FORWARD_BACKWARD) {
            throw new IllegalArgumentException("Optimizer steps require FORWARD_BACKWARD execution.");
        }

        long runStart = System.nanoTime();
        java.util.ArrayList<ExecutionStepTrace> steps = captureTrace ? new java.util.ArrayList<>() : null;
        ExecutionState executionState = ExecutionState.create(allNodes, descriptorIndex, metadataIndex, forwardOutputNode.id());
        executionState.configureNativeCpuMemory(runtimeConfig.nativeCpuMemory(), nativeCpuMemoryPool);
        RuntimeException executionFailure = null;
        Error executionError = null;
        try {
            RuntimeMemoryBinder.bind(memoryPlan, allNodes, descriptorIndex, executionState);
            ExecutionContext context = ExecutionContext.fromRuntimeConfig(runtimeConfig, mode, metadataIndex, executionState);
            OptimizerStepContext optimizerContext = optimizer == null
                    ? null
                    : new OptimizerStepContext(runtimeConfig, context, publication, allNodes, compiledGradients);
            if (optimizer != null) {
                optimizer.beforeExecute(optimizerContext);
            }

            if (mode == ExecutionMode.FORWARD_BACKWARD) {
                seedRootGradient(executionState);
                PreparedExecutionRunner.executeSteps(executionSteps, context, captureTrace, steps, 0);
                if (optimizer == null) {
                    ExecutionPublisher.publishAfterExecution(
                            mode,
                            executionState,
                            publication,
                            rootTensor,
                            allNodes,
                            forwardOutputNode,
                            compiledGradients
                    );
                } else {
                    if (publication.publishesOutputValue() && !publication.publishesAllForwardValues()) {
                        ExecutionPublisher.syncRootData(
                                mode,
                                executionState,
                                rootTensor,
                                allNodes,
                                forwardOutputNode
                        );
                    }
                    optimizer.step(optimizerContext);
                    ExecutionPublisher.publishAfterOptimizerStep(
                            mode,
                            executionState,
                            publication,
                            rootTensor,
                            allNodes,
                            forwardOutputNode,
                            compiledGradients
                    );
                }
            } else {
                PreparedExecutionRunner.executeSteps(forwardSteps, context, captureTrace, steps, 0);
                ExecutionPublisher.publishForwardOnly(
                        mode,
                        executionState,
                        publication,
                        rootTensor,
                        allNodes,
                        forwardOutputNode
                );
            }
            return new RunTrace(
                    mode,
                    System.nanoTime() - runStart,
                    steps == null ? List.of() : steps,
                    executionState.cpuMaterializationTraces(),
                    executionState.hostDeviceTransferTraces(),
                    executionState.nativeCpuMemoryTrace(),
                    optimizerContext == null ? List.of() : optimizerContext.nativeOptimizerTraces()
            );
        } catch (RuntimeException ex) {
            executionFailure = ex;
            throw ex;
        } catch (Error err) {
            executionError = err;
            throw err;
        } finally {
            try {
                executionState.closeResources();
            } catch (RuntimeException closeFailure) {
                if (executionFailure != null) {
                    executionFailure.addSuppressed(closeFailure);
                } else if (executionError != null) {
                    executionError.addSuppressed(closeFailure);
                } else {
                    throw closeFailure;
                }
            }
        }
    }

    /**
     * Convenience wrapper for {@link #execute(ExecutionMode)} in forward-backward mode.
     *
     * <p>If the plan has no backward steps, this method prints an informational message and returns without mutation.
     */
    public void backward() {
        if (!supportsBackward) {
            System.out.println("Info: No gradients to compute.");
            return;
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

    private static Map<Integer, CompiledNodeExecutionMetadata> buildMetadataIndex(List<PreparedNodeExecution> executionSteps) {
        Map<Integer, CompiledNodeExecutionMetadata> out = new HashMap<>();
        for (PreparedNodeExecution step : executionSteps) {
            out.put(step.compiledNode().id(), step.metadata());
        }
        return Map.copyOf(out);
    }

    private void seedRootGradient(ExecutionState executionState) {
        if (!(forwardSeedGradient instanceof CompiledGradientBinding.NodeBinding nodeBinding)) {
            return;
        }
        fillGradientOnes(executionState.runtimeTensorForNodeId(nodeBinding.nodeId()));
    }

    private static void fillGradientOnes(Tensor gradient) {
        GradientDTypePolicy.requireGradientSupported(gradient.getDataType(), "Gradient seeding");
        switch (gradient.getDataType()) {
            case FLOAT64 -> Arrays.fill(TensorInternalAccess.float64Data(gradient), 1.0);
            case FLOAT32 -> Arrays.fill(TensorInternalAccess.float32Data(gradient), 1.0f);
            case BFLOAT16 -> Arrays.fill(TensorInternalAccess.bfloat16Data(gradient), CpuDTypeOps.toBFloat16Bits(1.0f));
            case INT32, INT64, BOOL -> throw GradientDTypePolicy.unsupportedGradientDType(
                    gradient.getDataType(),
                    "Gradient seeding"
            );
        }
    }

}
