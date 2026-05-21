package graph.execution;

import backend.cpu.nativecpu.NativeCpuMemoryPool;
import backend.runtime.ExecutionContext;
import backend.runtime.ExecutionMode;
import config.runtime.RuntimeConfig;
import graph.CompiledGradientBinding;
import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.compile.planning.memory.MemoryPlan;
import graph.compile.publication.PublicationPlan;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.publication.ExecutionPublisher;
import graph.execution.residency.RuntimeMemoryBinder;
import graph.execution.runner.PreparedExecutionRunner;
import graph.execution.state.ExecutionState;
import graph.execution.trace.ExecutionStepTrace;
import graph.execution.trace.RunTrace;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.autograd.DifferentiableDTypePolicy;
import tensor.dtype.BFloat16Bits;
import training.optimizer.OptimizerStepContext;
import training.optimizer.TrainingOptimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Owns state and cleanup for one execution of a prepared plan.
 */
final class ExecutionRun {
    private final RuntimeConfig runtimeConfig;
    private final boolean supportsBackward;
    private final List<PreparedExecutionStep> executionSteps;
    private final List<PreparedExecutionStep> forwardSteps;
    private final List<CompiledNode> allNodes;
    private final CompiledTensorDescriptorIndex descriptorIndex;
    private final Map<Integer, CompiledNodeExecutionMetadata> metadataIndex;
    private final CompiledNode forwardOutputNode;
    private final PublicationPlan publicationPlan;
    private final MemoryPlan memoryPlan;
    private final NativeCpuMemoryPool nativeCpuMemoryPool;
    private final ExecutionMode mode;
    private final boolean captureTrace;
    private final TrainingOptimizer optimizer;
    private final PublicationPolicy publication;

    ExecutionRun(
            RuntimeConfig runtimeConfig,
            boolean supportsBackward,
            List<PreparedExecutionStep> executionSteps,
            List<PreparedExecutionStep> forwardSteps,
            List<CompiledNode> allNodes,
            CompiledTensorDescriptorIndex descriptorIndex,
            Map<Integer, CompiledNodeExecutionMetadata> metadataIndex,
            CompiledNode forwardOutputNode,
            PublicationPlan publicationPlan,
            MemoryPlan memoryPlan,
            NativeCpuMemoryPool nativeCpuMemoryPool,
            ExecutionMode mode,
            boolean captureTrace,
            TrainingOptimizer optimizer,
            PublicationPolicy publicationPolicy
    ) {
        this.runtimeConfig = Objects.requireNonNull(runtimeConfig, "runtimeConfig cannot be null");
        this.supportsBackward = supportsBackward;
        this.executionSteps = Objects.requireNonNull(executionSteps, "executionSteps cannot be null");
        this.forwardSteps = Objects.requireNonNull(forwardSteps, "forwardSteps cannot be null");
        this.allNodes = Objects.requireNonNull(allNodes, "allNodes cannot be null");
        this.descriptorIndex = Objects.requireNonNull(descriptorIndex, "descriptorIndex cannot be null");
        this.metadataIndex = Objects.requireNonNull(metadataIndex, "metadataIndex cannot be null");
        this.forwardOutputNode = Objects.requireNonNull(forwardOutputNode, "forwardOutputNode cannot be null");
        this.publicationPlan = Objects.requireNonNull(publicationPlan, "publicationPlan cannot be null");
        this.memoryPlan = memoryPlan;
        this.nativeCpuMemoryPool = nativeCpuMemoryPool;
        this.mode = Objects.requireNonNull(mode, "mode cannot be null");
        this.captureTrace = captureTrace;
        this.optimizer = optimizer;
        this.publication = publicationPolicy == null
                ? (optimizer == null ? PublicationPolicy.defaultExecution() : PublicationPolicy.defaultOptimizerStep())
                : publicationPolicy;
    }

    RunTrace execute() {
        publicationPlan.graphContract().validateOrThrow(publicationPlan.rootTensor());
        if (mode == ExecutionMode.FORWARD_BACKWARD && !supportsBackward) {
            throw new IllegalStateException("Prepared execution does not support backward execution.");
        }
        if (optimizer != null && mode != ExecutionMode.FORWARD_BACKWARD) {
            throw new IllegalArgumentException("Optimizer steps require FORWARD_BACKWARD execution.");
        }

        long runStart = System.nanoTime();
        ArrayList<ExecutionStepTrace> steps = captureTrace ? new ArrayList<>() : null;
        ExecutionState executionState = ExecutionState.create(
                allNodes,
                descriptorIndex,
                metadataIndex,
                forwardOutputNode.id(),
                publicationPlan
        );
        executionState.configureNativeCpuMemory(runtimeConfig.nativeCpuMemory(), nativeCpuMemoryPool);
        RuntimeException executionFailure = null;
        Error executionError = null;
        try {
            RuntimeMemoryBinder.bind(memoryPlan, allNodes, descriptorIndex, executionState);
            ExecutionContext context = ExecutionContext.fromRuntimeConfig(runtimeConfig, mode, metadataIndex, executionState);
            OptimizerStepContext optimizerContext = optimizer == null
                    ? null
                    : new OptimizerStepContext(runtimeConfig, context, publication, allNodes, publicationPlan);
            if (optimizer != null) {
                optimizer.beforeExecute(optimizerContext);
            }

            if (mode == ExecutionMode.FORWARD_BACKWARD) {
                executeForwardBackward(executionState, context, optimizerContext, steps);
            } else {
                executeForwardOnly(executionState, context, steps);
            }
            return trace(mode, runStart, steps, executionState, optimizerContext);
        } catch (RuntimeException ex) {
            executionFailure = ex;
            throw ex;
        } catch (Error err) {
            executionError = err;
            throw err;
        } finally {
            closeExecutionState(executionState, executionFailure, executionError);
        }
    }

    private void executeForwardBackward(
            ExecutionState executionState,
            ExecutionContext context,
            OptimizerStepContext optimizerContext,
            ArrayList<ExecutionStepTrace> steps
    ) {
        seedRootGradient(executionState);
        PreparedExecutionRunner.executeSteps(executionSteps, context, captureTrace, steps, 0);
        if (optimizer == null) {
            ExecutionPublisher.publishAfterExecution(mode, executionState, publication, publicationPlan);
            return;
        }
        if (publication.publishesOutputValue() && !publication.publishesAllForwardValues()) {
            ExecutionPublisher.syncRootData(mode, executionState, publicationPlan);
        }
        optimizer.step(optimizerContext);
        ExecutionPublisher.publishAfterOptimizerStep(mode, executionState, publication, publicationPlan);
    }

    private void executeForwardOnly(
            ExecutionState executionState,
            ExecutionContext context,
            ArrayList<ExecutionStepTrace> steps
    ) {
        PreparedExecutionRunner.executeSteps(forwardSteps, context, captureTrace, steps, 0);
        ExecutionPublisher.publishForwardOnly(mode, executionState, publication, publicationPlan);
    }

    private static RunTrace trace(
            ExecutionMode mode,
            long runStart,
            ArrayList<ExecutionStepTrace> steps,
            ExecutionState executionState,
            OptimizerStepContext optimizerContext
    ) {
        return new RunTrace(
                mode,
                System.nanoTime() - runStart,
                steps == null ? List.of() : steps,
                executionState.cpuMaterializationTraces(),
                executionState.hostDeviceTransferTraces(),
                executionState.nativeCpuMemoryTrace(),
                optimizerContext == null ? List.of() : optimizerContext.nativeOptimizerTraces()
        );
    }

    private static void closeExecutionState(
            ExecutionState executionState,
            RuntimeException executionFailure,
            Error executionError
    ) {
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

    private void seedRootGradient(ExecutionState executionState) {
        if (!(publicationPlan.forwardSeedGradient() instanceof CompiledGradientBinding.NodeBinding nodeBinding)) {
            return;
        }
        fillGradientOnes(executionState.runtimeTensorForNodeId(nodeBinding.nodeId()));
    }

    private static void fillGradientOnes(Tensor gradient) {
        DifferentiableDTypePolicy.requireGradientSupported(gradient.getDataType(), "Gradient seeding");
        switch (gradient.getDataType()) {
            case FLOAT64 -> Arrays.fill(TensorInternalAccess.float64Data(gradient), 1.0);
            case FLOAT32 -> Arrays.fill(TensorInternalAccess.float32Data(gradient), 1.0f);
            case BFLOAT16 -> Arrays.fill(TensorInternalAccess.bfloat16Data(gradient), BFloat16Bits.fromFloat(1.0f));
            case INT32, INT64, BOOL -> throw DifferentiableDTypePolicy.unsupportedGradientDType(
                    gradient.getDataType(),
                    "Gradient seeding"
            );
        }
    }
}
