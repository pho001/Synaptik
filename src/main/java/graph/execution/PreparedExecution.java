package graph.execution;

import backend.ComputeBackend;
import backend.ComputeEngine;
import backend.accelerator.lowering.GpuCompoundPatternType;
import backend.blas.OpenBlasFfmBridge;
import backend.cpu.fused.plan.FusedOperation;
import backend.cpu.kernels.CpuDTypeOps;
import backend.cpu.kernels.CpuNodeExecutionPlan;
import backend.cpu.kernels.linalg.matmul.exec.PreparedMatMulExecutable;
import backend.cpu.kernels.linalg.matmul.plan.MatMulExecutionRoute;
import backend.cpu.nativecpu.NativeCpuMemoryPool;
import backend.cpu.nativecpu.NativeCpuTraceState;
import backend.cpu.nativecpu.PreparedNativeCpuInputPolicy;
import backend.cpu.nativecpu.PreparedNativeCpuPlan;
import backend.cpu.nativecpu.PreparedNativeCpuRoute;
import backend.lowering.region.CpuNativeRegionPayload;
import backend.lowering.region.RegionRole;
import backend.memory.CpuMaterializationReason;
import backend.runtime.ExecutionContext;
import backend.runtime.ExecutionMode;
import config.runtime.BlasStorageMode;
import config.runtime.CpuStorageProfile;
import config.runtime.NativeCpuMemoryConfig;
import config.runtime.NativeMemoryPoolPolicy;
import config.runtime.RuntimeConfig;
import graph.CompiledNode;
import graph.CompiledGradientBinding;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.execution.trace.ComputeTraceMetadata;
import graph.execution.trace.DispatchTraceMetadata;
import graph.execution.trace.ExecutionStepTrace;
import graph.execution.trace.FusedTraceMetadata;
import graph.execution.trace.LayoutTraceMetadata;
import graph.execution.trace.MatMulTraceMetadata;
import graph.execution.trace.ConvTraceMetadata;
import graph.execution.trace.PrepareTrace;
import graph.execution.trace.ReductionTraceMetadata;
import graph.execution.trace.RunTrace;
import graph.execution.trace.StepExecutionMetadata;
import graph.optimizer.memory.MemoryPlan;
import graph.optimizer.cost.CostComponent;
import graph.optimizer.cost.CostExplanation;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import training.optimizer.OptimizerStepContext;
import training.optimizer.TrainingOptimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Locale;
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
        this.runtimeConfig = Objects.requireNonNull(runtimeConfig, "runtimeConfig cannot be null");
        this.supportsBackward = supportsBackward;
        this.executionSteps = List.copyOf(executionSteps == null ? List.of() : executionSteps);
        this.forwardSteps = List.copyOf(forwardSteps == null ? List.of() : forwardSteps);
        this.backwardSteps = List.copyOf(backwardSteps == null ? List.of() : backwardSteps);
        this.allNodes = List.copyOf(allNodes == null ? List.of() : allNodes);
        this.descriptorIndex = Objects.requireNonNull(descriptorIndex, "descriptorIndex cannot be null");
        this.compiledGradients = Map.copyOf(compiledGradients == null ? Map.of() : compiledGradients);
        this.rootTensor = Objects.requireNonNull(rootTensor, "rootTensor cannot be null");
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
                executeSteps(executionSteps, context, captureTrace, steps, 0);
                if (optimizer == null) {
                    publishAfterExecution(mode, executionState, publication);
                } else {
                    if (publication.publishesOutputValue() && !publication.publishesAllForwardValues()) {
                        syncRootData(mode, executionState);
                    }
                    optimizer.step(optimizerContext);
                    publishAfterOptimizerStep(mode, executionState, publication);
                }
            } else {
                executeSteps(forwardSteps, context, captureTrace, steps, 0);
                publishForwardOnly(mode, executionState, publication);
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
            if (DeviceLayoutViewPropagator.tryPropagate(step, context)) {
                if (captureTrace) {
                    traces.add(toStepTrace(startIndex + i, step, System.nanoTime() - t0, context));
                }
                continue;
            }
            requireCpuReadableInputs(step, context);
            ComputeEngine.compute(step.compiledNode(), step.metadata(), context);
            markResidencyAfterStep(step, context);
            if (captureTrace) {
                traces.add(toStepTrace(startIndex + i, step, System.nanoTime() - t0, context));
            }
        }
    }

    private static void markResidencyAfterStep(PreparedNodeExecution step, ExecutionContext context) {
        int nodeId = step.compiledNode().id();
        if (step.metadata().backend() == ComputeBackend.CPU) {
            var residency = context.residencyForNodeId(nodeId);
            if (residency != null && residency.nativeCurrent()) {
                return;
            }
            context.markCpuCurrent(nodeId, residencyReason(step));
            return;
        }
        var residency = context.residencyForNodeId(nodeId);
        if (residency == null || (!residency.cpuCurrent() && !residency.deviceCurrent())) {
            context.markCpuCurrent(nodeId, residencyReason(step));
        }
    }

    private static void requireCpuReadableInputs(PreparedNodeExecution step, ExecutionContext context) {
        if (step.metadata().backend() != ComputeBackend.CPU) {
            return;
        }
        if (step.metadata().cpuRegionExecutable() != null) {
            return;
        }
        PreparedNativeCpuPlan nativeCpuPlan = step.metadata().cpuPlan() == null
                ? null
                : step.metadata().cpuPlan().nativeCpuPlan();
        if (nativeCpuPlan != null) {
            if (nativeCpuPlan.inputPolicy() == PreparedNativeCpuInputPolicy.ALL_NATIVE) {
                return;
            }
            if (nativeCpuPlan.inputPolicy() == PreparedNativeCpuInputPolicy.CONDITION_CPU_VALUES_NATIVE) {
                List<Integer> inputIds = inputIds(step);
                if (!inputIds.isEmpty()) {
                    context.requireCpuReadable(inputIds.get(0), CpuMaterializationReason.CPU_CONSUMER);
                }
                return;
            }
        }
        List<Integer> inputIds = inputIds(step);
        for (int inputId : inputIds) {
            context.requireCpuReadable(inputId, CpuMaterializationReason.CPU_CONSUMER);
        }
    }

    private static List<Integer> inputIds(PreparedNodeExecution step) {
        return step.metadata().executionInputNodeIds().isEmpty()
                ? step.compiledNode().inputIds()
                : step.metadata().executionInputNodeIds();
    }

    private static String residencyReason(PreparedNodeExecution step) {
        if (step != null && step.metadata().acceleratorExecutable() instanceof backend.metal.exec.PreparedMetalExecutable metal) {
            if (metal.lastExecutionStats().usedCpuFallback()) {
                return "metal cpu fallback wrote CPU array";
            }
            return metal.lastExecutionStats().executionPath() == backend.metal.bridge.MetalMpsBridgeExecutionPath.BUFFER_BINDING
                    ? "metal buffer binding execution wrote device buffer"
                    : "metal bridge copied output to CPU array";
        }
        return "backend wrote CPU array";
    }

    private static ExecutionStepTrace toStepTrace(int index, PreparedNodeExecution step, long durationNs, ExecutionContext context) {
        CompiledNode node = step.compiledNode();
        Tensor semanticNode = step.node();
        var metadata = step.metadata();
        operations.Operation executionOperation = metadata.executionOperation() == null
                ? node.operation()
                : metadata.executionOperation();
        String opType = executionOperation == null ? "LEAF" : executionOperation.opType().name();
        String kernel = metadata.cpuKernel() == null ? "" : metadata.cpuKernel().getClass().getSimpleName();
        return new ExecutionStepTrace(
                index,
                node.label(),
                opType,
                java.util.Arrays.stream(node.shape()).boxed().toList(),
                node.dataType(),
                metadata.backend().name(),
                kernel,
                durationNs,
                buildStepMetadata(node, semanticNode, step, context)
        );
    }

    private static StepExecutionMetadata buildStepMetadata(CompiledNode node, Tensor semanticNode, PreparedNodeExecution step, ExecutionContext context) {
        var metadata = step.metadata();
        LinkedHashMap<String, Object> attrs = new LinkedHashMap<>();
        ComputeTraceMetadata compute = null;
        LayoutTraceMetadata layout = new LayoutTraceMetadata(
                node.storageOffset(),
                node.contiguous(),
                metadata.cpuPlan() != null && metadata.cpuPlan().stridedPath(),
                metadata.cpuPlan() == null ? "" : metadata.cpuPlan().targetType().name()
        );
        DispatchTraceMetadata dispatch = null;
        ReductionTraceMetadata reduction = null;
        MatMulTraceMetadata matMul = null;
        ConvTraceMetadata conv = null;
        FusedTraceMetadata fusedMeta = null;

        if (metadata.cpuPlan() != null) {
            var plan = metadata.cpuPlan();
            compute = new ComputeTraceMetadata(
                    plan.computeContract().computeType().name(),
                    plan.computeContract().storageType().name(),
                    plan.computeContract().computeType().name(),
                    plan.computeContract().backend().name(),
                    plan.computeContract().accumulateType().name()
            );
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
                PreparedMatMulExecutable executable = plan.matMulExecutable();
                MatMulExecutionRoute route = executable == null || executable.lastExecutionRoute() == null
                        ? plan.matMulHints().route()
                        : executable.lastExecutionRoute();
                String blasProvider = matMulBlasProvider(context);
                String blasSymbol = matMulBlasSymbol(node, route, executable, plan);
                String nativeCpuFallbackReason = executable == null ? "" : executable.lastFallbackReason();
                boolean openblasProvider = "OPENBLAS_FFM".equals(blasProvider);
                matMul = new MatMulTraceMetadata(
                        plan.matMulHints().useBlas(),
                        plan.matMulHints().useBatchedBlas(),
                        blasProvider,
                        blasSymbol,
                        route.name(),
                        route.name(),
                        matMulCpuStorageProfile(context),
                        matMulNativeCpuFailurePolicy(context),
                        matMulRequestedCpuStorage(context),
                        matMulActualCpuStorage(route),
                        nativeCpuFallbackReason,
                        openblasProvider && OpenBlasFfmBridge.isFloat32GemmAvailable(),
                        openblasProvider && OpenBlasFfmBridge.isFloat64GemmAvailable(),
                        openblasProvider && OpenBlasFfmBridge.isBFloat16ToFloatGemmAvailable(),
                        openblasProvider && OpenBlasFfmBridge.isBFloat16OutputGemmAvailable(),
                        matMulBf16ContinuationRoute(node, route, blasSymbol),
                        matMulBf16OutputRoute(node, route, blasSymbol),
                        matMulBf16ComputePrecision(node, route, blasSymbol),
                        matMulBf16OutputPrecision(node, route, blasSymbol),
                        matMulCopyInBytes(node, step, context, executable, route),
                        matMulCopyOutBytes(node, executable, route),
                        matMulNativeTempBytes(route),
                        matMulThreadPolicy(context),
                        nativeCpuFallbackReason,
                        plan.matMulHints().parallel(),
                        plan.matMulHints().tileM(),
                        plan.matMulHints().tileN(),
                        plan.matMulHints().tileK(),
                        plan.matMulHints().plannedWorkers(),
                        plan.matMulHints().work(),
                        plan.matMulHints().microKernel().name()
                );
                attrs.put("matMulRoute", matMul.route());
                attrs.put("blasProvider", matMul.blasProvider());
                attrs.put("blasSymbol", matMul.blasSymbol());
                attrs.put("blasRoute", matMul.blasRoute());
                attrs.put("cpuStorageProfile", matMul.cpuStorageProfile());
                attrs.put("nativeCpuFailurePolicy", matMul.nativeCpuFailurePolicy());
                attrs.put("requestedCpuStorage", matMul.requestedCpuStorage());
                attrs.put("actualCpuStorage", matMul.actualCpuStorage());
                attrs.put("nativeCpuFallbackReason", matMul.nativeCpuFallbackReason());
                attrs.put("openblasSgemmAvailable", matMul.openblasSgemmAvailable());
                attrs.put("openblasDgemmAvailable", matMul.openblasDgemmAvailable());
                attrs.put("openblasSbgemmAvailable", matMul.openblasSbgemmAvailable());
                attrs.put("openblasBgemmAvailable", matMul.openblasBgemmAvailable());
                attrs.put("bf16ContinuationRoute", matMul.bf16ContinuationRoute());
                attrs.put("bf16OutputRoute", matMul.bf16OutputRoute());
                attrs.put("bf16ComputePrecision", matMul.bf16ComputePrecision());
                attrs.put("bf16OutputPrecision", matMul.bf16OutputPrecision());
                if ("OPENBLAS_FFM".equals(matMul.blasProvider())) {
                    attrs.put("openblasLookupSource", OpenBlasFfmBridge.lookupSource());
                }
                attrs.put("matMulCopyInBytes", matMul.copyInBytes());
                attrs.put("matMulCopyOutBytes", matMul.copyOutBytes());
                attrs.put("matMulNativeTempBytes", matMul.nativeTempBytes());
                attrs.put("blasThreadPolicy", matMul.threadPolicy());
                if (!matMul.fallbackReason().isBlank()) {
                    attrs.put("matMulFallbackReason", matMul.fallbackReason());
                }
            }
        }

        addNativeCpuRegionRejectionAttrs(attrs, node, metadata, context);

        if (metadata.cpuRegionExecutable() != null) {
            var regionPlan = metadata.cpuRegionExecutable().regionExecutionPlan();
            if (regionPlan != null) {
                addRegionPlanAttrs(attrs, regionPlan);
                attrs.put("nativeCpuRegionId", regionPlan.regionId());
                attrs.put("nativeCpuRegionNodeCount", regionPlan.orderedNodeIds().size());
                attrs.put("nativeCpuRegionInputs", regionPlan.externalInputNodeIds());
                attrs.put("nativeCpuRegionOutputs", regionPlan.boundaryOutputNodeIds());
                attrs.put("nativeCpuRegionRoute", metadata.cpuRegionExecutable().lastRoute());
                attrs.put("nativeCpuRegionDecision", regionPlan.decision().selected() ? "SELECTED" : "REJECTED");
                attrs.put("nativeCpuRegionReason", regionPlan.decision().reason());
                attrs.put("nativeCpuRegionFallbackReason", metadata.cpuRegionExecutable().lastFallbackReason());
                attrs.put("nativeCpuRegionLocalKernelCount", metadata.cpuRegionExecutable().lastRegionLocalKernelCount());
                attrs.put("nativeCpuRegionLocalViewCount", metadata.cpuRegionExecutable().lastRegionLocalViewCount());
                attrs.put("nativeCpuRegionExecutedGroupCount", metadata.cpuRegionExecutable().lastExecutedGroupCount());
                if (regionPlan.backendPayload() instanceof CpuNativeRegionPayload payload) {
                    attrs.put("nativeCpuRegionProviderKind", payload.providerKind());
                    attrs.put("nativeCpuRegionProviderNodes", payload.providerNodeIds());
                    attrs.put("nativeCpuRegionLocalKernelNodes", payload.localKernelNodeIds());
                    attrs.put("nativeCpuRegionViewNodes", regionPlan.nodePlans().stream()
                            .filter(nodePlan -> nodePlan.regionRole() == RegionRole.VIEW_ALIAS)
                            .map(backend.lowering.region.RegionNodePlan::nodeId)
                            .toList());
                    attrs.put("nativeCpuRegionFallbackPlanCount", payload.fallbackPlans().size());
                }
            }
        }

        ConvTraceMetadata trace = context.convTraceForNodeId(node.id());
        if (trace != null) {
            conv = trace;
        }

        operations.Operation executionOperation = metadata.executionOperation() == null
                ? node.operation()
                : metadata.executionOperation();
        if (executionOperation instanceof FusedOperation fused) {
            String executionBackend = step.metadata().fusedExecutable() == null
                    ? ""
                    : step.metadata().fusedExecutable().getClass().getSimpleName();
            fusedMeta = new FusedTraceMetadata(
                    fused.getPrecisionMode(),
                    fused.isLowCostHint(),
                    fused.getDispatchFamily().id(),
                    fused.getSchedulerSignature(),
                    executionBackend,
                    fused.getDispatchScale(),
                    fused.getPlan().nodeCount(),
                    fused.getPlan().inputCount()
            );
        }

        Tensor runtimeTensor = safeRuntimeTensor(context, node.id());
        NativeCpuTraceState nativeCpu = runtimeTensor == null
                ? null
                : context.runtimeStateFor(runtimeTensor, NativeCpuTraceState.class);
        if (nativeCpu != null) {
            attrs.put("cpuStorageProfile", nativeCpu.cpuStorageProfile());
            attrs.put("nativeCpuFailurePolicy", nativeCpu.nativeCpuFailurePolicy());
            attrs.put("requestedCpuStorage", nativeCpu.requestedCpuStorage());
            attrs.put("actualCpuStorage", nativeCpu.actualCpuStorage());
            attrs.put("nativeCpuKernelStatus", nativeCpu.nativeCpuKernelStatus());
            attrs.put("nativeCpuKernelFamily", nativeCpu.nativeCpuKernelFamily());
            attrs.put("nativeCpuFallbackReason", nativeCpu.nativeCpuFallbackReason());
            if (!nativeCpu.storagePrecision().isBlank()) {
                attrs.put("storagePrecision", nativeCpu.storagePrecision());
            }
            if (!nativeCpu.computePrecision().isBlank()) {
                attrs.put("computePrecision", nativeCpu.computePrecision());
            }
        }

        if (metadata.acceleratorExecutable() != null) {
            var decision = metadata.acceleratorExecutable().lastAcceleratorBufferDecision();
            attrs.put("acceleratorBufferMode", decision.mode().name());
            attrs.put("acceleratorBufferBackend", decision.backend().name());
            attrs.put("acceleratorBufferDecision", decision.path().name());
            attrs.put("acceleratorBufferExecutionPath", decision.path().name());
            attrs.put("acceleratorBufferReasonCode", decision.reasonCode().name());
            attrs.put("acceleratorBufferReason", decision.reason());
            attrs.put("acceleratorBufferPreparedInputUsed", decision.preparedInputUsed());
            attrs.put("acceleratorBufferInputCount", decision.inputs().size());
            attrs.put("acceleratorBufferOutputCount", decision.outputs().size());
            attrs.put("cpuMaterializationCount", context.cpuMaterializationTraceCount());
            attrs.put("deviceHandoffCount", deviceHandoffCount(decision));
            var regionPlan = metadata.acceleratorExecutable().regionExecutionPlan();
            if (regionPlan != null) {
                addRegionPlanAttrs(attrs, regionPlan);
            }
            var manifest = metadata.acceleratorExecutable().gpuLoweredRegionManifest();
            if (manifest != null && !manifest.regionId().isBlank()) {
                attrs.put("gpuRegionId", manifest.regionId());
                attrs.put("gpuLoweredRegionId", manifest.regionId());
            }
            if (manifest != null) {
                attrs.put("selectedRegionLength", manifest.selectedRegionLength());
                attrs.put("loweredPrimitiveCount", manifest.loweredPrimitives().size());
                attrs.put("gpuFusedSubpatternCount", manifest.fusedSubpatterns().size());
                attrs.put("gpuFusedSubpatternTypes", manifest.fusedSubpatterns().stream()
                        .map(subpattern -> subpattern.patternType().name())
                        .toList());
            }
            if (manifest != null && !manifest.fusedSubpatterns().isEmpty()) {
                attrs.put("gpuFusedSubpatternOriginalNodeIds", manifest.fusedSubpatterns().stream()
                        .map(subpattern -> subpattern.originalOperationNodeIds())
                        .toList());
                attrs.put("gpuFusedSubpatternLoweredPrimitiveCount", manifest.fusedSubpatterns().stream()
                        .map(subpattern -> subpattern.loweredPrimitiveCount())
                        .toList());
                attrs.put("gpuFusedSubpatternReasons", manifest.fusedSubpatterns().stream()
                        .map(subpattern -> subpattern.reason().name())
                        .toList());
            }
            var compoundSummary = metadata.acceleratorExecutable().compoundSummary();
            if (compoundSummary != null && compoundSummary.patternType() != GpuCompoundPatternType.NONE) {
                attrs.put("gpuCompoundPattern", compoundSummary.patternType().name());
                attrs.put("gpuCompoundSupported", compoundSummary.supported());
                attrs.put("gpuCompoundReason", compoundSummary.reason().name());
                attrs.put("gpuCompoundNodeCount", compoundSummary.orderedNodeIds().size());
                attrs.put("gpuCompoundOrderedNodeIds", compoundSummary.orderedNodeIds());
                attrs.put("gpuCompoundDagNodeTypes", compoundSummary.dagNodeTypes());
                attrs.put("gpuCompoundPostOps", compoundSummary.postOps());
            }
        }
        var layoutTransformDecision = context.layoutTransformDecisionForNodeId(node.id());
        if (layoutTransformDecision != null) {
            attrs.put("gpuLayoutTransformKind", layoutTransformDecision.kind().name());
            attrs.put("gpuLayoutTransformOp", layoutTransformDecision.opType().name());
            attrs.put("gpuLayoutTransformSourceNodeId", layoutTransformDecision.sourceNodeId());
            attrs.put("gpuLayoutTransformTargetNodeId", layoutTransformDecision.targetNodeId());
            attrs.put("gpuLayoutTransformAccepted", layoutTransformDecision.accepted());
            attrs.put("gpuLayoutTransformReasonCode", layoutTransformDecision.reasonCode().name());
            attrs.put("gpuLayoutTransformReason", layoutTransformDecision.reason());
            attrs.put("gpuLayoutTransformSourceLayoutClass", layoutTransformDecision.sourceLayout().layoutClass().name());
            attrs.put("gpuLayoutTransformTargetLayoutClass", layoutTransformDecision.targetLayout().layoutClass().name());
            attrs.put("gpuLayoutTransformBytes", layoutTransformDecision.targetLayout().logicalByteLength());
            attrs.put("gpuLayoutMaterializationCount",
                    isGpuLayoutMaterialization(layoutTransformDecision)
                            ? 1
                            : 0);
            attrs.put("gpuLayoutMaterializationBytes",
                    isGpuLayoutMaterialization(layoutTransformDecision)
                            ? layoutTransformDecision.targetLayout().logicalByteLength()
                            : 0L);
            attrs.putIfAbsent("acceleratorBufferBackend", layoutTransformDecision.backendId());
            attrs.putIfAbsent("acceleratorBufferDecision", layoutTransformDecision.kind().name());
            attrs.putIfAbsent("acceleratorBufferExecutionPath", layoutTransformDecision.accepted()
                    ? layoutTransformDecision.kind().name()
                    : "UNAVAILABLE");
            attrs.putIfAbsent("acceleratorBufferReasonCode", layoutTransformDecision.reasonCode().name());
            attrs.putIfAbsent("acceleratorBufferReason", layoutTransformDecision.reason());
        }

        if (metadata.acceleratorExecutable() instanceof backend.metal.exec.PreparedMetalExecutable metal) {
            var metalStats = metal.lastExecutionStats();
            var route = metal.routeDecision();
            CostExplanation routeCost = route.toCostScore().explain(route.reasonCode().name());
            attrs.put("metalBridgeAvailable", metal.bridge().isAvailable());
            attrs.put("metalBridgeContextAvailable", metal.bridgeContext().available());
            attrs.put("metalBridgeExecutableAvailable", metal.bridgeExecutable().available());
            attrs.put("metalBridgeCacheHit", metal.bridgeExecutable().cacheHit());
            attrs.put("metalSupportsBufferBindings", metal.bridge().supportsBufferBindings());
            attrs.put("metalExecutionRoute", route.selectedRoute().name());
            attrs.put("metalRouteReasonCode", route.reasonCode().name());
            attrs.put("metalRouteRejectedRoutes", route.rejectedRoutes().stream()
                    .map(Enum::name)
                    .toList());
            attrs.put("metalRouteRejectedReasonCodes", route.rejectedReasonCodes().stream()
                    .map(Enum::name)
                    .toList());
            attrs.put("metalRouteRejectedReasons", route.rejectedRouteReasons());
            attrs.put("metalRouteReason", route.detail());
            attrs.put("metalRouteEstimatedCost", route.estimatedRouteCost());
            attrs.put("metalRouteEstimatedCopyCost", route.estimatedCopyCost());
            attrs.put("metalRouteBridgeAvailable", route.bridgeAvailable());
            attrs.put("metalRouteExecutableAvailable", route.executableAvailable());
            attrs.put("metalRouteBufferAbiSupported", route.bufferAbiSupported());
            attrs.put("metalRouteCustomKernelAvailable", route.customKernelAvailable());
            attrs.put("metalRouteNativeCopyCostKnown", route.nativeCopyCostKnown());
            attrs.put("metalRouteCostModel", routeCost.modelName());
            attrs.put("metalRouteCostInputKind", routeCost.inputKind());
            attrs.put("metalRouteCostReason", routeCost.reasonCode());
            attrs.put("metalRouteCostComparison", routeCost.comparison().name());
            attrs.put("metalRouteCostTopContributors", routeCost.topContributors().stream()
                    .map(PreparedExecution::costComponentSummary)
                    .toList());
            attrs.put("metalRouteCostComponents", routeCost.rawComponents().stream()
                    .map(PreparedExecution::costComponentSummary)
                    .toList());
            attrs.put("metalBufferBindingDecision", metal.lastBufferBindingDecision());
            attrs.put("metalOutputBufferWriteProbeSupported", metal.bridge().supportsOutputBufferWriteProbe());
            attrs.put("metalSubgraphNodeCount", metal.plan().nodeIds().size());
            attrs.put("metalSubgraphOps", metal.plan().subgraph().ops().stream().map(op -> op.opType().name()).toList());
            attrs.put("metalEstimatedWork", metal.plan().estimatedWork());
            attrs.put("metalUsedCpuFallback", metalStats.usedCpuFallback());
            attrs.put("metalFallbackReason", metalStats.fallbackReason());
            attrs.put("metalExecutionPath", metalStats.executionPath().name());
            attrs.put("metalExternalInputCount", metalStats.externalInputCount());
            attrs.put("metalOutputCount", metalStats.outputCount());
            attrs.put("metalInputBytes", metalStats.inputBytes());
            attrs.put("metalOutputBytes", metalStats.outputBytes());
            attrs.put("metalJavaToNativeCopyNs", metalStats.javaToNativeCopyNs());
            attrs.put("metalOutputAllocationNs", metalStats.outputAllocationNs());
            attrs.put("metalNativeExecuteNs", metalStats.nativeExecuteNs());
            attrs.put("metalNativeCopyStrategy", metalStats.nativeCopyStrategy().name());
            attrs.put("metalOutputBufferWriteProven", metalStats.outputBufferWriteProven());
            attrs.put("metalOutputBufferWriteStatus", metalStats.outputBufferWriteStatus());
            attrs.put("metalNativeDeviceCopyNs", metalStats.nativeDeviceCopyNs());
            attrs.put("metalNativeToJavaCopyNs", metalStats.nativeToJavaCopyNs());
            attrs.put("metalBridgeTotalNs", metalStats.totalNs());
        }
        if (metadata.acceleratorExecutable() instanceof backend.cuda.exec.PreparedCudaExecutable cuda) {
            var cudaStats = cuda.lastExecutionStats();
            attrs.put("cudaBridgeAvailable", cuda.bridge().isAvailable());
            attrs.put("cudaBridgeContextAvailable", cuda.bridgeContext().available());
            attrs.put("cudaBridgeExecutableAvailable", cuda.bridgeExecutable().available());
            attrs.put("cudaSupportsBufferBindings", cuda.bridge().supportsBufferBindings());
            attrs.put("cudaUsedCpuFallback", cudaStats.usedCpuFallback());
            attrs.put("cudaFallbackReason", cudaStats.fallbackReason());
            attrs.put("cudaExecutionPath", cudaStats.executionPath().name());
            attrs.put("cudaExternalInputCount", cudaStats.externalInputCount());
            attrs.put("cudaOutputCount", cudaStats.outputCount());
            attrs.put("cudaInputBytes", cudaStats.inputBytes());
            attrs.put("cudaOutputBytes", cudaStats.outputBytes());
            attrs.put("cudaJavaToNativeCopyNs", cudaStats.javaToNativeCopyNs());
            attrs.put("cudaNativeExecuteNs", cudaStats.nativeExecuteNs());
            attrs.put("cudaNativeDeviceCopyNs", cudaStats.nativeDeviceCopyNs());
            attrs.put("cudaNativeToJavaCopyNs", cudaStats.nativeToJavaCopyNs());
            attrs.put("cudaBridgeTotalNs", cudaStats.totalNs());
            attrs.put("acceleratorInputBytes", cudaStats.inputBytes());
            attrs.put("acceleratorOutputBytes", cudaStats.outputBytes());
            attrs.put("acceleratorJavaToNativeCopyNs", cudaStats.javaToNativeCopyNs());
            attrs.put("acceleratorNativeToJavaCopyNs", cudaStats.nativeToJavaCopyNs());
            attrs.put("acceleratorNativeDeviceCopyNs", cudaStats.nativeDeviceCopyNs());
        }
        var residency = context.residencyForNodeId(node.id());
        if (residency != null) {
            attrs.put("storageResidency", residency.residency().name());
            attrs.put("storageCpuCurrent", residency.cpuCurrent());
            attrs.put("storageDeviceCurrent", residency.deviceCurrent());
            attrs.put("storageDeviceBackend", residency.deviceBackend());
            attrs.put("storageTransitionReason", residency.lastTransitionReason());
        }
        var deviceBinding = context.deviceBufferBindingForNodeId(node.id());
        if (deviceBinding != null) {
            attrs.put("deviceBufferBackend", deviceBinding.backendId());
            attrs.put("deviceBufferBytes", deviceBinding.logicalByteLength());
            attrs.put("deviceBufferAvailable", deviceBinding.available());
            attrs.put("deviceBuffer", deviceBinding.describe());
        }

        addFallbackSummary(attrs);
        return new StepExecutionMetadata("node", attrs, compute, layout, dispatch, reduction, matMul, conv, fusedMeta);
    }

    private static void addRegionPlanAttrs(
            LinkedHashMap<String, Object> attrs,
            backend.lowering.region.RegionExecutionPlan regionPlan
    ) {
        attrs.put("regionId", regionPlan.regionId());
        attrs.put("regionTarget", regionPlan.target().name());
        attrs.put("loweringFamily", regionPlan.loweringFamily().name());
        attrs.put("anchorNodeId", regionPlan.anchorNodeId());
        attrs.put("orderedNodeIds", regionPlan.orderedNodeIds());
        attrs.put("boundaryOutputNodeIds", regionPlan.boundaryOutputNodeIds());
        attrs.put("regionNodeCount", regionPlan.orderedNodeIds().size());
        attrs.put("regionDecision", regionPlan.decision().selected() ? "SELECTED" : "REJECTED");
        attrs.put("regionReason", regionPlan.decision().reason());
        attrs.put("regionExecutionKindSummary", regionPlan.executionGroups().stream()
                .map(group -> group.executionKind().name())
                .distinct()
                .toList());
        attrs.put("regionStorageContractSummary", regionPlan.executionGroups().stream()
                .map(group -> group.storageContract().name())
                .distinct()
                .toList());
    }

    private static void addNativeCpuRegionRejectionAttrs(
            LinkedHashMap<String, Object> attrs,
            CompiledNode node,
            CompiledNodeExecutionMetadata metadata,
            ExecutionContext context
    ) {
        if (node == null || metadata == null || context == null || context.runtimeConfig() == null) {
            return;
        }
        RuntimeConfig runtimeConfig = context.runtimeConfig();
        if (metadata.backend() != ComputeBackend.CPU
                || metadata.cpuRegionExecutable() != null
                || metadata.acceleratorExecutable() != null
                || runtimeConfig.cpuStorageProfile() == CpuStorageProfile.CPU_ARRAY
                || metadata.cpuPlan() == null) {
            return;
        }
        PreparedNativeCpuPlan nativePlan = metadata.cpuPlan().nativeCpuPlan();
        if (nativePlan != null && nativePlan.route() == PreparedNativeCpuRoute.NATIVE_EXECUTABLE) {
            return;
        }
        String reason = nativeRegionRejectionReason(node, nativePlan, runtimeConfig);
        attrs.put("nativeCpuRegionDecision", "REJECTED");
        attrs.put("nativeCpuRegionReason", reason);
        attrs.put("nativeCpuRegionRoute", "CPU_ARRAY");
        attrs.put("nativeCpuRegionFallbackReason", reason);
        attrs.put("nativeCpuRegionNodeCount", 1);
        attrs.put("nativeCpuRegionInputs", node.inputIds());
        attrs.put("nativeCpuRegionOutputs", List.of(node.id()));
        attrs.put("nativeCpuRegionRejectedNode", node.id());
        attrs.put("nativeCpuRegionRejectedOp", node.operation() == null
                ? "UNKNOWN"
                : node.operation().opType().name());
    }

    private static String nativeRegionRejectionReason(
            CompiledNode node,
            PreparedNativeCpuPlan nativePlan,
            RuntimeConfig runtimeConfig
    ) {
        String opLabel = node == null || node.operation() == null
                ? "unknown"
                : node.operation().opType().name().toLowerCase(Locale.ROOT);
        boolean providerOp = node != null
                && node.operation() != null
                && (node.operation().opType() == operations.Operation.OpType.MATMUL
                || node.operation().opType() == operations.Operation.OpType.LINEAR);
        if (providerOp && runtimeConfig.blas().provider() == backend.blas.BlasProvider.NONE) {
            return "native-cpu-region-provider-unavailable:" + opLabel;
        }
        String planReason = nativePlan == null ? "" : nativePlan.fallbackReason();
        if (runtimeConfig.cpuStorageProfile() == CpuStorageProfile.AUTO
                && (planReason.isBlank() || planReason.startsWith("cpu-storage-profile-not-native"))) {
            return "native-cpu-region-auto-rejected:no-region-selected";
        }
        if (!planReason.isBlank()) {
            return planReason.startsWith("native-cpu-region-")
                    ? planReason
                    : "native-cpu-region-rejected:" + planReason;
        }
        return "native-cpu-region-rejected:no-region-selected";
    }

    private static Tensor safeRuntimeTensor(ExecutionContext context, int nodeId) {
        try {
            return context.runtimeTensorForNodeId(nodeId);
        } catch (RuntimeException ignored) {
            return null;
        }
    }

    private static long matMulCopyInBytes(
            CompiledNode node,
            PreparedNodeExecution step,
            ExecutionContext context,
            PreparedMatMulExecutable executable,
            MatMulExecutionRoute route
    ) {
        if (executable != null && executable.lastCopyInBytes() >= 0L) {
            return executable.lastCopyInBytes();
        }
        if (route != MatMulExecutionRoute.OPENBLAS_ARRAY_COPYING) {
            return route == MatMulExecutionRoute.OPENBLAS_NATIVE_SEGMENT ? 0L : -1L;
        }
        List<Integer> inputIds = step.metadata().executionInputNodeIds().isEmpty()
                ? node.inputIds()
                : step.metadata().executionInputNodeIds();
        long bytes = 0L;
        for (int inputId : inputIds) {
            bytes += logicalByteLength(context.runtimeTensorForNodeId(inputId));
        }
        return bytes;
    }

    private static long matMulCopyOutBytes(
            CompiledNode node,
            PreparedMatMulExecutable executable,
            MatMulExecutionRoute route
    ) {
        if (executable != null && executable.lastCopyOutBytes() >= 0L) {
            return executable.lastCopyOutBytes();
        }
        if (route != MatMulExecutionRoute.OPENBLAS_ARRAY_COPYING) {
            return route == MatMulExecutionRoute.OPENBLAS_NATIVE_SEGMENT ? 0L : -1L;
        }
        return logicalByteLength(node.dataType(), node.shape());
    }

    private static long matMulNativeTempBytes(MatMulExecutionRoute route) {
        return route == MatMulExecutionRoute.OPENBLAS_NATIVE_SEGMENT ? 0L : -1L;
    }

    private static String matMulBlasProvider(ExecutionContext context) {
        if (context.runtimeConfig() == null || context.runtimeConfig().blas() == null) {
            return "";
        }
        return context.runtimeConfig().blas().provider().name();
    }

    private static String matMulThreadPolicy(ExecutionContext context) {
        if (context.runtimeConfig() == null
                || context.runtimeConfig().blas() == null
                || context.runtimeConfig().blas().provider() != backend.blas.BlasProvider.OPENBLAS_FFM) {
            return "";
        }
        return OpenBlasFfmBridge.threadPolicy();
    }

    private static String matMulCpuStorageProfile(ExecutionContext context) {
        return context.runtimeConfig() == null || context.runtimeConfig().cpuStorageProfile() == null
                ? ""
                : context.runtimeConfig().cpuStorageProfile().name();
    }

    private static String matMulNativeCpuFailurePolicy(ExecutionContext context) {
        return context.runtimeConfig() == null || context.runtimeConfig().nativeCpuFailurePolicy() == null
                ? ""
                : context.runtimeConfig().nativeCpuFailurePolicy().name();
    }

    private static String matMulRequestedCpuStorage(ExecutionContext context) {
        if (context.runtimeConfig() == null || context.runtimeConfig().blas() == null) {
            return "";
        }
        CpuStorageProfile profile = context.runtimeConfig().cpuStorageProfile();
        BlasStorageMode mode = switch (profile) {
            case CPU_ARRAY -> BlasStorageMode.CPU_ARRAY;
            case CPU_NATIVE -> BlasStorageMode.CPU_NATIVE;
            case AUTO -> context.runtimeConfig().blas().storageMode();
        };
        return mode.name();
    }

    private static String matMulActualCpuStorage(MatMulExecutionRoute route) {
        return route == MatMulExecutionRoute.OPENBLAS_NATIVE_SEGMENT ? "CPU_NATIVE" : "CPU_ARRAY";
    }

    private static String matMulBlasSymbol(CompiledNode node, MatMulExecutionRoute route) {
        return matMulBlasSymbol(node, route, null);
    }

    private static String matMulBlasSymbol(CompiledNode node, MatMulExecutionRoute route, PreparedMatMulExecutable executable) {
        return matMulBlasSymbol(node, route, executable, null);
    }

    private static String matMulBlasSymbol(
            CompiledNode node,
            MatMulExecutionRoute route,
            PreparedMatMulExecutable executable,
            CpuNodeExecutionPlan plan
    ) {
        if (executable != null && !executable.lastBlasSymbol().isBlank()) {
            return executable.lastBlasSymbol();
        }
        if (route == MatMulExecutionRoute.JAVA_DIRECT) {
            return "";
        }
        if (isBFloat16LinearSbgemmRoute(node, plan)) {
            return "cblas_sbgemm";
        }
        return switch (node.dataType()) {
            case FLOAT32 -> "cblas_sgemm";
            case FLOAT64 -> "cblas_dgemm";
            case BFLOAT16 -> "cblas_bgemm";
            default -> "";
        };
    }

    private static boolean isBFloat16LinearSbgemmRoute(CompiledNode node, CpuNodeExecutionPlan plan) {
        if (node.dataType() != tensor.DataType.BFLOAT16
                || !(node.operation() instanceof operations.linalg.linear linearOp)
                || plan == null
                || plan.matMulHints() == null
                || (!plan.matMulHints().useBlas() && !plan.matMulHints().useBatchedBlas())) {
            return false;
        }
        return OpenBlasFfmBridge.isBFloat16ToFloatGemmAvailable()
                && (plan.publishFloatContinuation() || linearOp.hasBias());
    }

    private static String matMulBf16ContinuationRoute(CompiledNode node, MatMulExecutionRoute route, String blasSymbol) {
        if (node.dataType() != tensor.DataType.BFLOAT16) {
            return "";
        }
        if ("cblas_sbgemm".equals(blasSymbol)) {
            return "SBGEMM";
        }
        if (route == MatMulExecutionRoute.JAVA_DIRECT) {
            return "JAVA";
        }
        if ("cblas_bgemm".equals(blasSymbol)) {
            return "";
        }
        return "UNAVAILABLE";
    }

    private static String matMulBf16OutputRoute(CompiledNode node, MatMulExecutionRoute route, String blasSymbol) {
        if (node.dataType() != tensor.DataType.BFLOAT16) {
            return "";
        }
        if ("cblas_bgemm".equals(blasSymbol)) {
            return "BGEMM";
        }
        if ("cblas_sbgemm".equals(blasSymbol)) {
            return "PROMOTED_F32";
        }
        if (route == MatMulExecutionRoute.JAVA_DIRECT) {
            return "JAVA";
        }
        return "UNAVAILABLE";
    }

    private static String matMulBf16ComputePrecision(CompiledNode node, MatMulExecutionRoute route, String blasSymbol) {
        if (node.dataType() != tensor.DataType.BFLOAT16) {
            return "";
        }
        if ("cblas_bgemm".equals(blasSymbol)) {
            return "BF16_OUTPUT";
        }
        if ("cblas_sbgemm".equals(blasSymbol) || route == MatMulExecutionRoute.JAVA_DIRECT) {
            return "F32_PROMOTED";
        }
        return "UNAVAILABLE";
    }

    private static String matMulBf16OutputPrecision(CompiledNode node, MatMulExecutionRoute route, String blasSymbol) {
        if (node.dataType() != tensor.DataType.BFLOAT16) {
            return "";
        }
        if ("cblas_sbgemm".equals(blasSymbol)) {
            return "F32";
        }
        if ("cblas_bgemm".equals(blasSymbol) || route == MatMulExecutionRoute.JAVA_DIRECT) {
            return "BF16";
        }
        return "UNAVAILABLE";
    }

    private static long logicalByteLength(Tensor tensor) {
        if (tensor == null) {
            return 0L;
        }
        return Math.multiplyExact((long) tensor.getFlatDataSize(), elementBytes(tensor.getDataType()));
    }

    private static long logicalByteLength(tensor.DataType dataType, int[] shape) {
        long elements = 1L;
        for (int dim : shape == null ? new int[0] : shape) {
            elements = Math.multiplyExact(elements, Math.max(0, dim));
        }
        return Math.multiplyExact(elements, elementBytes(dataType));
    }

    private static int elementBytes(tensor.DataType dataType) {
        return switch (dataType) {
            case FLOAT64, INT64 -> Long.BYTES;
            case FLOAT32, INT32 -> Integer.BYTES;
            case BFLOAT16 -> Short.BYTES;
            case BOOL -> Byte.BYTES;
        };
    }

    private static void addFallbackSummary(LinkedHashMap<String, Object> attrs) {
        ArrayList<String> kinds = new ArrayList<>();
        ArrayList<String> reasonCodes = new ArrayList<>();
        ArrayList<String> reasons = new ArrayList<>();

        String acceleratorPath = stringAttr(attrs, "acceleratorBufferExecutionPath");
        if ("CPU_FALLBACK".equals(acceleratorPath)) {
            addFallback(
                    kinds,
                    reasonCodes,
                    reasons,
                    "ACCELERATOR_CPU_FALLBACK",
                    stringAttr(attrs, "acceleratorBufferReasonCode"),
                    stringAttr(attrs, "acceleratorBufferReason")
            );
        } else if ("TENSOR_ARRAY".equals(acceleratorPath)) {
            addFallback(
                    kinds,
                    reasonCodes,
                    reasons,
                    "ACCELERATOR_TENSOR_ARRAY_FALLBACK",
                    stringAttr(attrs, "acceleratorBufferReasonCode"),
                    stringAttr(attrs, "acceleratorBufferReason")
            );
        } else if ("UNAVAILABLE".equals(acceleratorPath)) {
            addFallback(
                    kinds,
                    reasonCodes,
                    reasons,
                    "ACCELERATOR_BUFFER_UNAVAILABLE",
                    stringAttr(attrs, "acceleratorBufferReasonCode"),
                    stringAttr(attrs, "acceleratorBufferReason")
            );
        }

        if (Boolean.TRUE.equals(attrs.get("metalUsedCpuFallback"))) {
            addFallback(
                    kinds,
                    reasonCodes,
                    reasons,
                    "METAL_CPU_FALLBACK",
                    stringAttr(attrs, "metalRouteReasonCode"),
                    firstNonBlank(stringAttr(attrs, "metalFallbackReason"), stringAttr(attrs, "metalRouteReason"))
            );
        }
        if ("TENSOR_ARRAY_COPY".equals(stringAttr(attrs, "metalExecutionPath"))
                || "TENSOR_ARRAY".equals(stringAttr(attrs, "metalExecutionRoute"))) {
            addFallback(
                    kinds,
                    reasonCodes,
                    reasons,
                    "METAL_TENSOR_ARRAY_FALLBACK",
                    stringAttr(attrs, "metalRouteReasonCode"),
                    firstNonBlank(stringAttr(attrs, "metalRouteReason"), stringAttr(attrs, "acceleratorBufferReason"))
            );
        }

        if (Boolean.TRUE.equals(attrs.get("cudaUsedCpuFallback"))) {
            addFallback(
                    kinds,
                    reasonCodes,
                    reasons,
                    "CUDA_CPU_FALLBACK",
                    stringAttr(attrs, "acceleratorBufferReasonCode"),
                    firstNonBlank(stringAttr(attrs, "cudaFallbackReason"), stringAttr(attrs, "acceleratorBufferReason"))
            );
        }
        if ("TENSOR_ARRAY".equals(stringAttr(attrs, "cudaExecutionPath"))) {
            addFallback(
                    kinds,
                    reasonCodes,
                    reasons,
                    "CUDA_TENSOR_ARRAY_FALLBACK",
                    stringAttr(attrs, "acceleratorBufferReasonCode"),
                    firstNonBlank(stringAttr(attrs, "cudaFallbackReason"), stringAttr(attrs, "acceleratorBufferReason"))
            );
        }

        if (!stringAttr(attrs, "matMulFallbackReason").isBlank()) {
            addFallback(
                    kinds,
                    reasonCodes,
                    reasons,
                    "CPU_MATMUL_ROUTE_FALLBACK",
                    stringAttr(attrs, "matMulRoute"),
                    stringAttr(attrs, "matMulFallbackReason")
            );
        }

        if (!kinds.isEmpty()) {
            attrs.put("fallbackOccurred", true);
            attrs.put("fallbackKind", kinds.size() == 1 ? kinds.getFirst() : "MULTIPLE");
            attrs.put("fallbackKinds", List.copyOf(kinds));
            attrs.put("fallbackReasonCode", reasonCodes.size() == 1 ? reasonCodes.getFirst() : String.join(" | ", reasonCodes));
            attrs.put("fallbackReasonCodes", List.copyOf(reasonCodes));
            attrs.put("fallbackReason", reasons.size() == 1 ? reasons.getFirst() : String.join(" | ", reasons));
            attrs.put("fallbackReasons", List.copyOf(reasons));
        }
    }

    private static void addFallback(
            ArrayList<String> kinds,
            ArrayList<String> reasonCodes,
            ArrayList<String> reasons,
            String kind,
            String reasonCode,
            String reason
    ) {
        if (kind == null || kind.isBlank() || kinds.contains(kind)) {
            return;
        }
        kinds.add(kind);
        reasonCodes.add(reasonCode == null || reasonCode.isBlank() ? "UNKNOWN" : reasonCode);
        reasons.add(reason == null || reason.isBlank() ? kind : reason);
    }

    private static String stringAttr(Map<String, Object> attrs, String key) {
        Object value = attrs.get(key);
        return value == null ? "" : String.valueOf(value);
    }

    private static String firstNonBlank(String first, String second) {
        return first == null || first.isBlank() ? (second == null ? "" : second) : first;
    }

    private static String costComponentSummary(CostComponent component) {
        if (component == null) {
            return "";
        }
        return component.name()
                + "=" + String.format(Locale.US, "%.6f", component.value())
                + " " + component.direction().name()
                + " (" + component.reason() + ")";
    }

    private static int deviceHandoffCount(backend.accelerator.buffer.AcceleratorBufferDecision decision) {
        if (decision == null
                || decision.path() != backend.accelerator.buffer.AcceleratorBufferExecutionPath.BUFFER_BINDING) {
            return 0;
        }
        return decision.inputs().size() + decision.outputs().size();
    }

    private static boolean isGpuLayoutMaterialization(
            backend.accelerator.buffer.AcceleratorLayoutTransformDecision decision
    ) {
        if (decision == null || !decision.accepted()) {
            return false;
        }
        return decision.kind() == backend.accelerator.buffer.AcceleratorLayoutTransformKind.DENSE_GPU_MATERIALIZATION
                || decision.kind() == backend.accelerator.buffer.AcceleratorLayoutTransformKind.BROADCAST_GPU_MATERIALIZATION;
    }

    private void publishAfterExecution(
            ExecutionMode mode,
            ExecutionState executionState,
            PublicationPolicy publication
    ) {
        if (publication.publishesAllForwardValues()) {
            publishAllForwardValues(mode, executionState);
        } else if (publication.publishesOutputValue()) {
            syncRootData(mode, executionState);
        }
        if (mode == ExecutionMode.FORWARD_BACKWARD) {
            if (publication.publishesGradients()) {
                publishCompiledGradients(executionState);
            } else {
                clearPublishedGradients();
            }
        }
    }

    private void publishAfterOptimizerStep(
            ExecutionMode mode,
            ExecutionState executionState,
            PublicationPolicy publication
    ) {
        if (publication.publishesAllForwardValues()) {
            publishAllForwardValues(mode, executionState);
        }
        if (publication.publishesGradients()) {
            publishCompiledGradients(executionState);
        } else {
            clearPublishedGradients();
        }
    }

    private void publishForwardOnly(
            ExecutionMode mode,
            ExecutionState executionState,
            PublicationPolicy publication
    ) {
        if (publication.publishesAllForwardValues()) {
            publishAllForwardValues(mode, executionState);
        } else if (publication.publishesOutputValue()) {
            syncRootData(mode, executionState);
        }
    }

    private void publishAllForwardValues(ExecutionMode mode, ExecutionState executionState) {
        syncRootData(mode, executionState);
        Tensor rootPublishTarget = resolveSemanticPublishTarget(rootTensor);
        for (CompiledNode node : allNodes) {
            if (node.backwardNode()) {
                continue;
            }
            Tensor target = node.sourceTensor();
            if (target == null || target == rootTensor || target == rootPublishTarget) {
                continue;
            }
            publishRuntimeTensor(
                    mode,
                    executionState,
                    target,
                    node.id(),
                    CpuMaterializationReason.GRAPH_VALUE_PUBLICATION
            );
            repairSemanticAliasChain(target);
        }
        repairSemanticAliasChain(rootTensor);
    }

    private void syncRootData(ExecutionMode mode, ExecutionState executionState) {
        Integer semanticRootNodeId = nodeIdForSemanticTensor(rootTensor);
        int actualRootNodeId = semanticRootNodeId == null ? resolveForwardRuntimeRootNodeId() : semanticRootNodeId;
        Tensor publishTarget = resolveSemanticPublishTarget(rootTensor);
        Integer publishNodeId = nodeIdForSemanticTensor(publishTarget);
        if (publishNodeId != null) {
            if (publishNodeId != actualRootNodeId
                    && shouldPublishActualRootForAlias(executionState, publishNodeId, actualRootNodeId)) {
                publishRuntimeTensor(
                        mode,
                        executionState,
                        rootTensor,
                        actualRootNodeId,
                        CpuMaterializationReason.GRAPH_OUTPUT
                );
                repairSemanticAliasChain(rootTensor);
                return;
            }
            executionState.requireCpuReadable(publishNodeId, CpuMaterializationReason.GRAPH_OUTPUT);
            Tensor runtimePublished = executionState.runtimeTensorForNodeId(publishNodeId);
            if (publishTarget.getStorage() == runtimePublished.getStorage()) {
                repairSemanticAliasChain(rootTensor);
                return;
            }
            if (mode == ExecutionMode.FORWARD_BACKWARD || runtimePublished != publishTarget) {
                publishTarget.copyDataFrom(runtimePublished);
            }
            repairSemanticAliasChain(rootTensor);
            return;
        }

        publishRuntimeTensor(mode, executionState, rootTensor, actualRootNodeId);
        repairSemanticAliasChain(rootTensor);
    }

    private static boolean shouldPublishActualRootForAlias(
            ExecutionState executionState,
            int publishNodeId,
            int actualRootNodeId
    ) {
        var publishState = executionState.residencyForNodeId(publishNodeId);
        var actualRootState = executionState.residencyForNodeId(actualRootNodeId);
        return !publishState.cpuCurrent()
                && (actualRootState.cpuCurrent() || actualRootState.requiresCpuMaterialization());
    }

    private static void publishRuntimeTensor(
            ExecutionMode mode,
            ExecutionState executionState,
            Tensor publishTarget,
            int nodeId
    ) {
        publishRuntimeTensor(mode, executionState, publishTarget, nodeId, CpuMaterializationReason.GRAPH_OUTPUT);
    }

    private static void publishRuntimeTensor(
            ExecutionMode mode,
            ExecutionState executionState,
            Tensor publishTarget,
            int nodeId,
            CpuMaterializationReason reason
    ) {
        executionState.requireCpuReadable(nodeId, reason);
        Tensor runtimeTensor = executionState.runtimeTensorForNodeId(nodeId);
        if (mode == ExecutionMode.FORWARD_BACKWARD || runtimeTensor != publishTarget) {
            publishTarget.copyDataFrom(runtimeTensor);
        }
    }

    private Tensor resolveSemanticPublishTarget(Tensor tensor) {
        Tensor current = tensor;
        while (isAliasViewOp(current) && current.getPrevTensors() != null && !current.getPrevTensors().isEmpty()) {
            current = current.getPrevTensors().getFirst();
        }
        return current;
    }

    private Integer nodeIdForSemanticTensor(Tensor tensor) {
        if (tensor == null) {
            return null;
        }
        for (CompiledNode node : allNodes) {
            if (node.semanticTensor() == tensor || node.sourceTensor() == tensor) {
                return node.id();
            }
        }
        return null;
    }

    private int resolveForwardRuntimeRootNodeId() {
        if (forwardOutputNode.operation() != null
                && forwardOutputNode.operation().opType() == operations.Operation.OpType.NOOP
                && Tensor.SYSTEM_FORWARD_OUTPUT_LABEL.equals(forwardOutputNode.label())
                && !forwardOutputNode.inputIds().isEmpty()) {
            return forwardOutputNode.inputIds().getFirst();
        }
        return forwardOutputNode.id();
    }

    private boolean isAliasViewOp(Tensor tensor) {
        if (tensor == null || tensor.getOperation() == null) {
            return false;
        }
        if (tensor.getPrevTensors() == null || tensor.getPrevTensors().isEmpty()) {
            return false;
        }
        return switch (tensor.getOperation().opType()) {
            case NOOP, EXPAND, SELECT, SLICE, PERMUTE, EXPAND_DIMS, SQUEEZE -> true;
            case RESHAPE -> tensor.getPrevTensors().getFirst().isContiguous();
            default -> false;
        };
    }

    private void repairSemanticAliasChain(Tensor tensor) {
        if (!isAliasViewOp(tensor) || tensor.getPrevTensors() == null || tensor.getPrevTensors().isEmpty()) {
            return;
        }
        Tensor source = tensor.getPrevTensors().getFirst();
        repairSemanticAliasChain(source);
        TensorInternalAccess.aliasRuntimeFrom(tensor, source);
    }

    private void seedRootGradient(ExecutionState executionState) {
        if (!(forwardSeedGradient instanceof CompiledGradientBinding.NodeBinding nodeBinding)) {
            return;
        }
        fillGradientOnes(executionState.runtimeTensorForNodeId(nodeBinding.nodeId()));
    }

    private void publishCompiledGradients(ExecutionState executionState) {
        for (CompiledNode node : allNodes) {
            if (node.backwardNode()) {
                continue;
            }
            Tensor tensor = node.sourceTensor();
            CompiledGradientBinding binding = compiledGradients.get(tensor);
            if (binding == null) {
                TensorInternalAccess.setGradient(tensor, null);
                continue;
            }
            Tensor published;
            if (binding instanceof CompiledGradientBinding.NodeBinding nodeBinding) {
                executionState.requireCpuReadable(nodeBinding.nodeId(), CpuMaterializationReason.GRADIENT_PUBLICATION);
                published = detachedCopy(executionState.runtimeTensorForNodeId(nodeBinding.nodeId()));
            } else if (binding instanceof CompiledGradientBinding.ConstantBinding constantBinding) {
                published = detachedCopy(constantBinding.template());
            } else {
                throw new IllegalStateException("Unsupported gradient binding type: " + binding.getClass().getName());
            }
            TensorInternalAccess.setGradient(tensor, published);
        }
    }

    private void clearPublishedGradients() {
        for (CompiledNode node : allNodes) {
            if (!node.backwardNode()) {
                TensorInternalAccess.setGradient(node.sourceTensor(), null);
            }
        }
    }

    private static Tensor detachedCopy(Tensor source) {
        Tensor copy = new Tensor(source.getShape(), null, source.getLabel(), source.getDataType());
        copy.copyDataFrom(source);
        return copy;
    }

    private static void fillGradientOnes(Tensor gradient) {
        switch (gradient.getDataType()) {
            case FLOAT64 -> Arrays.fill(gradient.getFloat64Data(), 1.0);
            case FLOAT32 -> Arrays.fill(gradient.getFloat32Data(), 1.0f);
            case BFLOAT16 -> Arrays.fill(gradient.getBFloat16Data(), CpuDTypeOps.toBFloat16Bits(1.0f));
            case INT32, BOOL -> throw new UnsupportedOperationException("INT32/BOOL tensors do not support gradient seeding.");
        }
    }

}
