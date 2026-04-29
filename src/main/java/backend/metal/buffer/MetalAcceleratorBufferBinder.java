package backend.metal.buffer;

import backend.ComputeBackend;
import backend.accelerator.buffer.AcceleratorBufferBindings;
import backend.accelerator.buffer.AcceleratorBufferDecision;
import backend.accelerator.buffer.AcceleratorBufferExecutionPath;
import backend.accelerator.buffer.AcceleratorBufferInputDecision;
import backend.accelerator.buffer.AcceleratorBufferLayout;
import backend.accelerator.buffer.AcceleratorBufferLayoutClass;
import backend.accelerator.buffer.AcceleratorBufferOutputDecision;
import backend.accelerator.buffer.AcceleratorBufferReasonCode;
import backend.accelerator.buffer.AcceleratorBufferRequest;
import backend.accelerator.exec.ResolvedAcceleratorInputs;
import backend.memory.DeviceBufferBinding;
import backend.memory.StorageResidency;
import backend.metal.MetalMpsCapabilities;
import backend.metal.bridge.MetalMpsBridgeContext;
import backend.metal.bridge.MetalMpsGraphBridge;
import backend.runtime.ExecutionContext;
import config.runtime.AcceleratorBufferBindingMode;
import config.runtime.AcceleratorBufferConfig;
import tensor.DataType;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Metal implementation of the common accelerator buffer-binding policy.
 */
public final class MetalAcceleratorBufferBinder {
    private final MetalMpsGraphBridge bridge;
    private final MetalMpsBridgeContext bridgeContext;
    private MetalBufferAllocator allocator;

    public MetalAcceleratorBufferBinder(MetalMpsGraphBridge bridge, MetalMpsBridgeContext bridgeContext) {
        this.bridge = Objects.requireNonNull(bridge, "bridge cannot be null");
        this.bridgeContext = Objects.requireNonNull(bridgeContext, "bridgeContext cannot be null");
    }

    public AcceleratorBufferDecision decide(
            AcceleratorBufferRequest request,
            ResolvedAcceleratorInputs inputs,
            AcceleratorBufferConfig bufferConfig,
            ExecutionContext context
    ) {
        AcceleratorBufferConfig config = bufferConfig == null ? AcceleratorBufferConfig.defaults() : bufferConfig;
        AcceleratorBufferBindingMode mode = config.bindingMode();
        if (mode == AcceleratorBufferBindingMode.OFF) {
            return decision(request, config, AcceleratorBufferExecutionPath.TENSOR_ARRAY, false,
                    AcceleratorBufferReasonCode.BUFFER_BINDINGS_DISABLED, "buffer bindings disabled", List.of(), List.of());
        }
        if (!bridge.supportsBufferBindings()) {
            return decision(request, config, fallbackPath(mode), false,
                    AcceleratorBufferReasonCode.NATIVE_BUFFER_ABI_UNAVAILABLE,
                    "native Metal buffer ABI unavailable: bridge does not support buffer bindings", List.of(), List.of());
        }
        if (request.estimatedWork() < config.minimumEstimatedWork()) {
            return decision(request, config, AcceleratorBufferExecutionPath.TENSOR_ARRAY, false,
                    AcceleratorBufferReasonCode.BELOW_MINIMUM_WORK,
                    "estimated work below buffer minimum", List.of(), List.of());
        }
        MetalBufferAllocator resolvedAllocator = allocator();
        if (!resolvedAllocator.available()) {
            return decision(request, config, fallbackPath(mode), false,
                    AcceleratorBufferReasonCode.BUFFER_ALLOCATOR_UNAVAILABLE,
                    resolvedAllocator.unavailableReason(), List.of(), List.of());
        }

        List<AcceleratorBufferInputDecision> inputDecisions = inputDecisions(request, inputs, config, context);
        AcceleratorBufferInputDecision rejectedInput = inputDecisions.stream()
                .filter(input -> !input.accepted())
                .findFirst()
                .orElse(null);
        if (rejectedInput != null) {
            return decision(request, config, fallbackPath(mode), false,
                    rejectedInput.reasonCode(), rejectedInput.reason(), inputDecisions, List.of());
        }

        List<AcceleratorBufferOutputDecision> outputDecisions = outputDecisions(request, context);
        AcceleratorBufferOutputDecision rejectedOutput = outputDecisions.stream()
                .filter(output -> !output.accepted())
                .findFirst()
                .orElse(null);
        if (rejectedOutput != null) {
            return decision(request, config, fallbackPath(mode), false,
                    rejectedOutput.reasonCode(), rejectedOutput.reason(), inputDecisions, outputDecisions);
        }

        return decision(request, config, AcceleratorBufferExecutionPath.BUFFER_BINDING, true,
                AcceleratorBufferReasonCode.BUFFER_BINDING_AVAILABLE,
                "using native buffer bindings", inputDecisions, outputDecisions);
    }

    public AcceleratorBufferBindings<MetalBufferBinding> resolve(
            AcceleratorBufferRequest request,
            ResolvedAcceleratorInputs inputs,
            AcceleratorBufferDecision decision,
            ExecutionContext context
    ) {
        if (decision == null || decision.path() != AcceleratorBufferExecutionPath.BUFFER_BINDING) {
            return new AcceleratorBufferBindings<>(List.of(), List.of());
        }
        MetalBufferAllocator resolvedAllocator = allocator();
        context.registerDeviceToCpuMaterializer(ComputeBackend.GPU_METAL.name(), new MetalDeviceToCpuMaterializer(resolvedAllocator));
        List<MetalBufferBinding> inputBindings = resolveInputBindings(request, inputs, context, resolvedAllocator);
        List<MetalBufferBinding> outputBindings = resolveOutputBindings(request, context, resolvedAllocator);
        return new AcceleratorBufferBindings<>(inputBindings, outputBindings);
    }

    private List<AcceleratorBufferInputDecision> inputDecisions(
            AcceleratorBufferRequest request,
            ResolvedAcceleratorInputs inputs,
            AcceleratorBufferConfig config,
            ExecutionContext context
    ) {
        List<AcceleratorBufferInputDecision> out = new ArrayList<>(request.externalInputNodeIds().size());
        for (int i = 0; i < request.externalInputNodeIds().size(); i++) {
            int nodeId = request.externalInputNodeIds().get(i);
            boolean prepared = i < inputs.preparedInputUsed().size() && inputs.preparedInputUsed().get(i);
            Tensor tensor = i < inputs.executionExternalInputs().size()
                    ? inputs.executionExternalInputs().get(i)
                    : context.runtimeTensorForNodeId(nodeId);
            AcceleratorBufferLayout layout = layoutForInput(request, inputs, context, i, nodeId);
            DataType expected = i < request.externalInputDataTypes().size() ? request.externalInputDataTypes().get(i) : null;
            String layoutReason = unsupportedBufferLayoutReason(layout);
            if (!layoutReason.isBlank()) {
                out.add(new AcceleratorBufferInputDecision(nodeId, layout, prepared, false,
                        AcceleratorBufferReasonCode.INPUT_LAYOUT_UNSUPPORTED,
                        "external input nodeId=" + nodeId + " input tensor layout unsupported: " + layoutReason));
                continue;
            }
            DeviceBufferBinding existing = context.deviceBufferBindingForNodeId(nodeId);
            if (existing instanceof MetalBufferBinding metalBinding) {
                String reason = incompatibleBindingReason(layout, metalBinding, MetalBufferAccess.READ, expected);
                if (reason.isBlank()) {
                    out.add(new AcceleratorBufferInputDecision(nodeId, layout, prepared, true,
                            AcceleratorBufferReasonCode.BUFFER_BINDING_AVAILABLE, ""));
                    continue;
                }
            } else if (existing != null) {
                out.add(new AcceleratorBufferInputDecision(nodeId, layout, prepared, false,
                        AcceleratorBufferReasonCode.INPUT_BINDING_UNAVAILABLE,
                        "external input nodeId=" + nodeId + " binding is not Metal-compatible: "
                                + existing.getClass().getSimpleName()));
                continue;
            }
            if (prepared && !config.allowPreparedInputMaterialization()) {
                out.add(new AcceleratorBufferInputDecision(nodeId, layout, true, false,
                        AcceleratorBufferReasonCode.PREPARED_INPUT_MATERIALIZATION_DISABLED,
                        "prepared input materialization disabled for nodeId=" + nodeId));
                continue;
            }
            if (!MetalMpsCapabilities.supportsExternalInputDType(tensor.getDataType())) {
                out.add(new AcceleratorBufferInputDecision(nodeId, layout, prepared, false,
                        AcceleratorBufferReasonCode.INPUT_DTYPE_UNSUPPORTED,
                        MetalMpsCapabilities.unsupportedDTypeMessage(tensor.getDataType())));
                continue;
            }
            if (expected != null && tensor.getDataType() != expected) {
                out.add(new AcceleratorBufferInputDecision(nodeId, layout, prepared, false,
                        AcceleratorBufferReasonCode.INPUT_DTYPE_UNSUPPORTED,
                        "external input nodeId=" + nodeId + " tensor dtype " + tensor.getDataType()
                                + " does not match executable dtype " + expected));
                continue;
            }
            var residency = context.residencyForNodeId(nodeId);
            if (!prepared && (residency == null || !residency.cpuCurrent())) {
                out.add(new AcceleratorBufferInputDecision(nodeId, layout, false, false,
                        AcceleratorBufferReasonCode.INPUT_NOT_CPU_CURRENT,
                        "external input nodeId=" + nodeId + " has no Metal binding and CPU storage is not current"));
                continue;
            }
            out.add(new AcceleratorBufferInputDecision(nodeId, layout, prepared, true,
                    AcceleratorBufferReasonCode.BUFFER_BINDING_AVAILABLE, ""));
        }
        return List.copyOf(out);
    }

    private List<AcceleratorBufferOutputDecision> outputDecisions(AcceleratorBufferRequest request, ExecutionContext context) {
        List<AcceleratorBufferOutputDecision> out = new ArrayList<>(request.outputNodeIds().size());
        for (int i = 0; i < request.outputNodeIds().size(); i++) {
            int nodeId = request.outputNodeIds().get(i);
            Tensor tensor = context.runtimeTensorForNodeId(nodeId);
            AcceleratorBufferLayout layout = layoutForOutput(request, context, i, nodeId);
            DataType expected = i < request.outputDataTypes().size() ? request.outputDataTypes().get(i) : null;
            String layoutReason = unsupportedBufferLayoutReason(layout);
            if (!layoutReason.isBlank()) {
                out.add(new AcceleratorBufferOutputDecision(nodeId, layout, false,
                        AcceleratorBufferReasonCode.OUTPUT_LAYOUT_UNSUPPORTED,
                        "output nodeId=" + nodeId + " output tensor layout unsupported: " + layoutReason));
                continue;
            }
            if (!MetalMpsCapabilities.supportsOutputDType(tensor.getDataType())) {
                out.add(new AcceleratorBufferOutputDecision(nodeId, layout, false,
                        AcceleratorBufferReasonCode.OUTPUT_DTYPE_UNSUPPORTED,
                        MetalMpsCapabilities.unsupportedDTypeMessage(tensor.getDataType())));
                continue;
            }
            if (expected != null && tensor.getDataType() != expected) {
                out.add(new AcceleratorBufferOutputDecision(nodeId, layout, false,
                        AcceleratorBufferReasonCode.OUTPUT_DTYPE_UNSUPPORTED,
                        "output nodeId=" + nodeId + " tensor dtype " + tensor.getDataType()
                                + " does not match executable dtype " + expected));
                continue;
            }
            out.add(new AcceleratorBufferOutputDecision(nodeId, layout, true,
                    AcceleratorBufferReasonCode.BUFFER_BINDING_AVAILABLE, ""));
        }
        return List.copyOf(out);
    }

    private List<MetalBufferBinding> resolveInputBindings(
            AcceleratorBufferRequest request,
            ResolvedAcceleratorInputs inputs,
            ExecutionContext context,
            MetalBufferAllocator resolvedAllocator
    ) {
        List<MetalBufferBinding> bindings = new ArrayList<>(request.externalInputNodeIds().size());
        for (int i = 0; i < request.externalInputNodeIds().size(); i++) {
            int nodeId = request.externalInputNodeIds().get(i);
            AcceleratorBufferLayout layout = layoutForInput(request, inputs, context, i, nodeId);
            DataType expected = i < request.externalInputDataTypes().size() ? request.externalInputDataTypes().get(i) : null;
            DeviceBufferBinding existing = context.deviceBufferBindingForNodeId(nodeId);
            if (existing instanceof MetalBufferBinding metalBinding
                    && incompatibleBindingReason(layout, metalBinding, MetalBufferAccess.READ, expected).isBlank()) {
                bindings.add(metalBinding);
                continue;
            }
            Tensor tensor = i < inputs.executionExternalInputs().size()
                    ? inputs.executionExternalInputs().get(i)
                    : context.runtimeTensorForNodeId(nodeId);
            MetalBufferBinding created = expected == DataType.BOOL
                    ? resolvedAllocator.createPredicateInputBinding(nodeId, tensor)
                    : resolvedAllocator.createInputBinding(nodeId, tensor);
            context.registerResource(new MetalBufferResource(resolvedAllocator, created.handle()));
            boolean prepared = i < inputs.preparedInputUsed().size() && inputs.preparedInputUsed().get(i);
            if (!prepared) {
                context.attachDeviceBufferBinding(
                        nodeId,
                        created,
                        StorageResidency.HOST_SHARED_DEVICE_BUFFER,
                        "metal shared input buffer upload"
                );
            }
            bindings.add(created);
        }
        return List.copyOf(bindings);
    }

    private List<MetalBufferBinding> resolveOutputBindings(
            AcceleratorBufferRequest request,
            ExecutionContext context,
            MetalBufferAllocator resolvedAllocator
    ) {
        List<MetalBufferBinding> bindings = new ArrayList<>(request.outputNodeIds().size());
        for (int i = 0; i < request.outputNodeIds().size(); i++) {
            int nodeId = request.outputNodeIds().get(i);
            AcceleratorBufferLayout layout = layoutForOutput(request, context, i, nodeId);
            DataType expected = i < request.outputDataTypes().size() ? request.outputDataTypes().get(i) : null;
            DeviceBufferBinding existing = context.writableDeviceBufferBindingForNodeId(nodeId);
            if (existing instanceof MetalBufferBinding metalBinding
                    && incompatibleBindingReason(layout, metalBinding, MetalBufferAccess.WRITE, expected).isBlank()) {
                bindings.add(metalBinding);
                continue;
            }
            MetalBufferBinding created = resolvedAllocator.createOutputBinding(
                    nodeId,
                    layout
            );
            context.registerResource(new MetalBufferResource(resolvedAllocator, created.handle()));
            context.reserveDeviceBufferBinding(nodeId, created);
            bindings.add(created);
        }
        return List.copyOf(bindings);
    }

    private MetalBufferAllocator allocator() {
        if (allocator == null) {
            allocator = bridge.createBufferAllocator(bridgeContext);
        }
        return allocator;
    }

    private static AcceleratorBufferExecutionPath fallbackPath(AcceleratorBufferBindingMode mode) {
        return mode == AcceleratorBufferBindingMode.REQUIRE
                ? AcceleratorBufferExecutionPath.UNAVAILABLE
                : AcceleratorBufferExecutionPath.TENSOR_ARRAY;
    }

    private static AcceleratorBufferDecision decision(
            AcceleratorBufferRequest request,
            AcceleratorBufferConfig config,
            AcceleratorBufferExecutionPath path,
            boolean allowed,
            AcceleratorBufferReasonCode reasonCode,
            String reason,
            List<AcceleratorBufferInputDecision> inputs,
            List<AcceleratorBufferOutputDecision> outputs
    ) {
        return new AcceleratorBufferDecision(
                request.backend(),
                config.bindingMode(),
                path,
                allowed,
                config.bindingMode() == AcceleratorBufferBindingMode.REQUIRE,
                reasonCode,
                reason,
                inputs,
                outputs
        );
    }

    private static String incompatibleBindingReason(
            AcceleratorBufferLayout expectedLayout,
            MetalBufferBinding metalBinding,
            MetalBufferAccess requiredAccess,
            DataType expectedDataType
    ) {
        if (!metalBinding.available()) {
            return "binding is unavailable: " + metalBinding.describe();
        }
        AcceleratorBufferLayout actualLayout = metalBinding.layout();
        if (expectedLayout != null && actualLayout.dataType() != expectedLayout.dataType()) {
            return "binding dtype " + actualLayout.dataType() + " does not match expected dtype " + expectedLayout.dataType();
        }
        if (expectedDataType != null && actualLayout.dataType() != expectedDataType) {
            return "binding dtype " + actualLayout.dataType() + " does not match executable dtype " + expectedDataType;
        }
        if (expectedLayout != null && !Arrays.equals(actualLayout.shape(), expectedLayout.shape())) {
            return "binding shape " + Arrays.toString(actualLayout.shape())
                    + " does not match expected shape " + Arrays.toString(expectedLayout.shape());
        }
        if (expectedLayout != null && !Arrays.equals(actualLayout.strides(), expectedLayout.strides())) {
            return "binding strides " + Arrays.toString(actualLayout.strides())
                    + " do not match expected strides " + Arrays.toString(expectedLayout.strides());
        }
        if (expectedLayout != null && actualLayout.storageOffset() != expectedLayout.storageOffset()) {
            return "binding storageOffset " + actualLayout.storageOffset()
                    + " does not match expected storageOffset " + expectedLayout.storageOffset();
        }
        if (expectedLayout != null && actualLayout.logicalElementCount() != expectedLayout.logicalElementCount()) {
            return "binding elementCount " + actualLayout.logicalElementCount()
                    + " does not match expected elementCount " + expectedLayout.logicalElementCount();
        }
        if (!accessCompatible(metalBinding.access(), requiredAccess)) {
            return "binding access " + metalBinding.access() + " is incompatible with required " + requiredAccess;
        }
        return "";
    }

    private static boolean accessCompatible(MetalBufferAccess actual, MetalBufferAccess required) {
        if (actual == MetalBufferAccess.READ_WRITE) {
            return true;
        }
        return actual == required;
    }

    private static AcceleratorBufferLayout layoutForInput(
            AcceleratorBufferRequest request,
            ResolvedAcceleratorInputs inputs,
            ExecutionContext context,
            int index,
            int nodeId
    ) {
        if (index < inputs.executionExternalInputs().size()) {
            return AcceleratorBufferLayout.fromTensor(inputs.executionExternalInputs().get(index));
        }
        if (index < request.externalInputLayouts().size()) {
            return request.externalInputLayouts().get(index);
        }
        return AcceleratorBufferLayout.fromTensor(context.runtimeTensorForNodeId(nodeId));
    }

    private static AcceleratorBufferLayout layoutForOutput(
            AcceleratorBufferRequest request,
            ExecutionContext context,
            int index,
            int nodeId
    ) {
        if (index < request.outputLayouts().size()) {
            return request.outputLayouts().get(index);
        }
        return AcceleratorBufferLayout.fromTensor(context.runtimeTensorForNodeId(nodeId));
    }

    private static String unsupportedBufferLayoutReason(AcceleratorBufferLayout layout) {
        if (layout.layoutClass() == AcceleratorBufferLayoutClass.DENSE_CONTIGUOUS) {
            return "";
        }
        return "layoutClass=" + layout.layoutClass() + ", storageOffset=" + layout.storageOffset() + ", strides=" + Arrays.toString(layout.strides());
    }
}
