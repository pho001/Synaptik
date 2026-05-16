package training.optimizer;

import backend.ComputeBackend;
import backend.accelerator.buffer.AcceleratorBufferLayout;
import backend.memory.CpuMaterializationReason;
import backend.memory.DeviceBufferBinding;
import backend.memory.StorageResidency;
import backend.metal.bridge.MetalMpsBridgeContext;
import backend.metal.buffer.MetalBufferAccess;
import backend.metal.buffer.MetalBufferAllocator;
import backend.metal.buffer.MetalBufferBinding;
import backend.metal.buffer.MetalDeviceToCpuMaterializer;
import backend.runtime.ExecutionContext;
import config.runtime.CpuStorageProfile;
import config.runtime.NativeCpuFailurePolicy;
import graph.CompiledGradientBinding;
import graph.CompiledNode;
import graph.execution.PublicationPolicy;
import graph.execution.trace.NativeOptimizerTrace;
import tensor.DataType;
import tensor.Tensor;

import java.util.Collection;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

abstract class AbstractTrainableOptimizer implements TrainingOptimizer {
    private final IdentityHashMap<Tensor, Boolean> explicitParameters;
    private final IdentityHashMap<Tensor, OwnedMetalBinding> metalParameters = new IdentityHashMap<>();

    AbstractTrainableOptimizer(Collection<Tensor> parameters) {
        this.explicitParameters = new IdentityHashMap<>();
        if (parameters != null) {
            for (Tensor parameter : parameters) {
                if (parameter != null) {
                    explicitParameters.put(parameter, Boolean.TRUE);
                }
            }
        }
    }

    @Override
    public void beforeExecute(OptimizerStepContext context) {
        Objects.requireNonNull(context, "context cannot be null");
        ExecutionContext execution = context.executionContext();
        for (TrainableParameterRef ref : selectedParameters(context)) {
            OwnedMetalBinding owned = metalParameters.get(ref.parameterNode().sourceTensor());
            if (owned == null || !owned.binding().available()) {
                continue;
            }
            execution.registerDeviceToCpuMaterializer(
                    ComputeBackend.GPU_METAL.name(),
                    new MetalDeviceToCpuMaterializer(owned.allocator())
            );
            execution.attachDeviceBufferBinding(
                    ref.parameterNode().id(),
                    bindingForNode(ref.parameterNode().id(), owned.binding(), MetalBufferAccess.READ_WRITE),
                    StorageResidency.DEVICE_OWNED,
                    "optimizer-owned Metal parameter"
            );
        }
    }

    @Override
    public final void step(OptimizerStepContext context) {
        Objects.requireNonNull(context, "context cannot be null");
        beforeStep(context);
        for (TrainableParameterRef ref : selectedParameters(context)) {
            if (!(ref.gradientBinding() instanceof CompiledGradientBinding.NodeBinding nodeBinding)) {
                continue;
            }
            int gradientNodeId = nodeBinding.nodeId();
            OptimizerResidencySnapshot before = residencySnapshot(context, ref, gradientNodeId);
            if (tryMetalStep(context, ref, nodeBinding.nodeId())) {
                recordOptimizerTrace(context, ref, gradientNodeId, "GPU_METAL", "", before);
                continue;
            }
            if (nativeCpuStep(context, ref, gradientNodeId)) {
                clearMetalParameter(ref.parameterNode().sourceTensor());
                recordOptimizerTrace(context, ref, gradientNodeId, "CPU_NATIVE", "", before);
                continue;
            }
            String fallbackReason = nativeCpuFallbackReason(context, ref, gradientNodeId);
            if (requiresNativeCpuOptimizer(context)) {
                throw nativeRequiredFailure(context, ref, gradientNodeId, fallbackReason);
            }
            cpuStep(context, ref, gradientNodeId);
            recordOptimizerTrace(context, ref, gradientNodeId, "CPU_ARRAY", fallbackReason, before);
            clearMetalParameter(ref.parameterNode().sourceTensor());
        }
    }

    protected void beforeStep(OptimizerStepContext context) {
    }

    @Override
    public void syncParametersToCpu() {
        for (Map.Entry<Tensor, OwnedMetalBinding> entry : List.copyOf(metalParameters.entrySet())) {
            Tensor parameter = entry.getKey();
            OwnedMetalBinding owned = entry.getValue();
            owned.allocator().readToCpu(owned.binding(), parameter, CpuMaterializationReason.PUBLIC_DATA_ACCESS);
            parameter.markStorageModified();
        }
    }

    @Override
    public void close() {
        for (OwnedMetalBinding owned : List.copyOf(metalParameters.values())) {
            owned.allocator().destroy(owned.binding().handle());
        }
        metalParameters.clear();
    }

    protected abstract boolean metalStep(
            OptimizerStepContext context,
            TrainableParameterRef ref,
            MetalMpsBridgeContext bridgeContext,
            MetalBufferAllocator allocator,
            MetalBufferBinding parameter,
            MetalBufferBinding gradient,
            MetalBufferBinding output
    );

    protected abstract void cpuStep(OptimizerStepContext context, TrainableParameterRef ref, int gradientNodeId);

    protected boolean nativeCpuStep(OptimizerStepContext context, TrainableParameterRef ref, int gradientNodeId) {
        return false;
    }

    protected String nativeCpuFallbackReason(OptimizerStepContext context, TrainableParameterRef ref, int gradientNodeId) {
        return "native-cpu-optimizer-not-implemented";
    }

    protected void recordOptimizerTrace(
            OptimizerStepContext context,
            TrainableParameterRef ref,
            int gradientNodeId,
            String route,
            String fallbackReason
    ) {
        recordOptimizerTrace(context, ref, gradientNodeId, route, fallbackReason, residencySnapshot(context, ref, gradientNodeId));
    }

    protected void recordOptimizerTrace(
            OptimizerStepContext context,
            TrainableParameterRef ref,
            int gradientNodeId,
            String route,
            String fallbackReason,
            OptimizerResidencySnapshot before
    ) {
        OptimizerResidencySnapshot after = residencySnapshot(context, ref, gradientNodeId);
        context.recordNativeOptimizerTrace(new NativeOptimizerTrace(
                optimizerName(),
                route,
                ref.parameterNode().dataType(),
                ref.parameterNode().id(),
                gradientNodeId,
                context.executionContext().runtimeTensorForNodeId(ref.parameterNode().id()).getFlatDataSize(),
                fallbackReason,
                context.publicationPolicy().name(),
                gradientPublication(context.publicationPolicy()),
                optimizerStateStorage(route),
                bf16TrainingPolicy(context, ref, fallbackReason),
                context.runtimeConfig().nativeCpuFailurePolicy().name(),
                before.parameterResidency(),
                after.parameterResidency(),
                before.gradientResidency(),
                after.gradientResidency(),
                publicationSkippedReason(context.publicationPolicy())
        ));
    }

    protected String optimizerStateStorage(String route) {
        return "NONE";
    }

    protected String bf16TrainingPolicy(OptimizerStepContext context, TrainableParameterRef ref, String fallbackReason) {
        if (ref.parameterNode().dataType() != DataType.BFLOAT16) {
            return "";
        }
        return context.runtimeConfig().bfloat16TrainingPolicy().name();
    }

    private static String gradientPublication(PublicationPolicy publicationPolicy) {
        if (publicationPolicy == null) {
            return "SKIPPED";
        }
        if (publicationPolicy.publishesGradients()) {
            return "PUBLISHED";
        }
        return publicationPolicy == PublicationPolicy.NONE ? "NONE" : "SKIPPED";
    }

    private static String publicationSkippedReason(PublicationPolicy publicationPolicy) {
        if (publicationPolicy == null) {
            return "publication-policy-output-only";
        }
        if (publicationPolicy.publishesGradients()) {
            return "";
        }
        return publicationPolicy == PublicationPolicy.NONE
                ? "publication-policy-none"
                : "publication-policy-output-only";
    }

    private static boolean requiresNativeCpuOptimizer(OptimizerStepContext context) {
        return context.runtimeConfig().cpuStorageProfile() == CpuStorageProfile.CPU_NATIVE
                && context.runtimeConfig().nativeCpuFailurePolicy() == NativeCpuFailurePolicy.REQUIRE_NATIVE;
    }

    private IllegalStateException nativeRequiredFailure(
            OptimizerStepContext context,
            TrainableParameterRef ref,
            int gradientNodeId,
            String fallbackReason
    ) {
        String reason = fallbackReason == null || fallbackReason.isBlank()
                ? nativeCpuFallbackReason(context, ref, gradientNodeId)
                : fallbackReason;
        return new IllegalStateException(
                "Native CPU optimizer execution required but " + optimizerName()
                        + " fell back to CPU_ARRAY. parameterNodeId=" + ref.parameterNode().id()
                        + ", gradientNodeId=" + gradientNodeId
                        + ", dtype=" + ref.parameterNode().dataType()
                        + ", reason=" + reason
        );
    }

    private static OptimizerResidencySnapshot residencySnapshot(
            OptimizerStepContext context,
            TrainableParameterRef ref,
            int gradientNodeId
    ) {
        return new OptimizerResidencySnapshot(
                context.executionContext().residencyForNodeId(ref.parameterNode().id()).residency().name(),
                context.executionContext().residencyForNodeId(gradientNodeId).residency().name()
        );
    }

    protected String optimizerName() {
        return getClass().getSimpleName();
    }

    protected List<TrainableParameterRef> selectedParameters(OptimizerStepContext context) {
        return context.trainableParameters().stream()
                .filter(ref -> explicitParameters.isEmpty() || explicitParameters.containsKey(ref.parameterNode().sourceTensor()))
                .toList();
    }

    protected void requireCpuReadable(OptimizerStepContext context, int nodeId) {
        context.executionContext().requireCpuReadable(nodeId, CpuMaterializationReason.OPTIMIZER_STEP);
    }

    private boolean tryMetalStep(OptimizerStepContext context, TrainableParameterRef ref, int gradientNodeId) {
        ExecutionContext execution = context.executionContext();
        MetalBufferAllocator allocator = execution.runtimeService(MetalBufferAllocator.class);
        MetalMpsBridgeContext bridgeContext = execution.runtimeService(MetalMpsBridgeContext.class);
        if (allocator == null || !allocator.available() || bridgeContext == null || !bridgeContext.available()) {
            return false;
        }
        DeviceBufferBinding parameterBinding = execution.deviceBufferBindingForNodeId(ref.parameterNode().id());
        DeviceBufferBinding gradientBinding = execution.deviceBufferBindingForNodeId(gradientNodeId);
        if (!(parameterBinding instanceof MetalBufferBinding metalParameter)
                || !(gradientBinding instanceof MetalBufferBinding metalGradient)
                || !sameShape(ref.parameterNode(), gradientNodeId, context)
                || ref.parameterNode().dataType() != DataType.FLOAT32
                || metalGradient.layout().dataType() != DataType.FLOAT32) {
            return false;
        }
        MetalBufferBinding output;
        try {
            output = allocator.createOutputBinding(
                    ref.parameterNode().id(),
                    AcceleratorBufferLayout.fromTensor(execution.runtimeTensorForNodeId(ref.parameterNode().id()))
            );
            if (!metalStep(context, ref, bridgeContext, allocator, metalParameter, metalGradient, output)) {
                allocator.destroy(output.handle());
                return false;
            }
        } catch (RuntimeException ex) {
            return false;
        }
        replaceMetalParameter(ref.parameterNode(), allocator, output, execution);
        return true;
    }

    private static boolean sameShape(CompiledNode parameterNode, int gradientNodeId, OptimizerStepContext context) {
        Tensor gradient = context.executionContext().runtimeTensorForNodeId(gradientNodeId);
        return java.util.Arrays.equals(parameterNode.shape(), gradient.getShapeUnsafe());
    }

    private void replaceMetalParameter(
            CompiledNode parameterNode,
            MetalBufferAllocator allocator,
            MetalBufferBinding output,
            ExecutionContext execution
    ) {
        Tensor source = parameterNode.sourceTensor();
        OwnedMetalBinding previous = metalParameters.put(source, new OwnedMetalBinding(allocator, output));
        if (previous != null && previous.binding().handle().nativeHandle() != output.handle().nativeHandle()) {
            previous.allocator().destroy(previous.binding().handle());
        }
        execution.attachDeviceBufferBinding(
                parameterNode.id(),
                bindingForNode(parameterNode.id(), output, MetalBufferAccess.READ_WRITE),
                StorageResidency.DEVICE_OWNED,
                "optimizer Metal parameter update"
        );
    }

    private void clearMetalParameter(Tensor parameter) {
        OwnedMetalBinding owned = metalParameters.remove(parameter);
        if (owned != null) {
            owned.allocator().destroy(owned.binding().handle());
        }
    }

    protected static MetalBufferBinding bindingForNode(int nodeId, MetalBufferBinding source, MetalBufferAccess access) {
        return new MetalBufferBinding(nodeId, source.layout(), source.handle(), access);
    }

    private record OwnedMetalBinding(MetalBufferAllocator allocator, MetalBufferBinding binding) {
    }

    protected record OptimizerResidencySnapshot(String parameterResidency, String gradientResidency) {
    }
}
