package backend.cuda.buffer;

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
import backend.cuda.bridge.CudaGraphBridge;
import backend.memory.DeviceBufferBinding;
import backend.memory.StorageResidency;
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
 * CUDA implementation of the shared accelerator buffer-binding preflight policy.
 *
 * <p>Phase 6 validates shared layout and dtype metadata only. It does not allocate
 * CUDA buffers or execute through native buffer handles.</p>
 */
public final class CudaAcceleratorBufferBinder {
    private final CudaGraphBridge bridge;

    public CudaAcceleratorBufferBinder(CudaGraphBridge bridge) {
        this.bridge = Objects.requireNonNull(bridge, "bridge cannot be null");
    }

    public AcceleratorBufferDecision decide(AcceleratorBufferRequest request, AcceleratorBufferConfig bufferConfig) {
        Objects.requireNonNull(request, "request cannot be null");
        AcceleratorBufferConfig config = bufferConfig == null ? AcceleratorBufferConfig.defaults() : bufferConfig;
        AcceleratorBufferBindingMode mode = config.bindingMode();
        if (mode == AcceleratorBufferBindingMode.OFF) {
            return decision(request, config, AcceleratorBufferExecutionPath.TENSOR_ARRAY, false,
                    AcceleratorBufferReasonCode.BUFFER_BINDINGS_DISABLED,
                    "buffer bindings disabled", List.of(), List.of());
        }
        if (!bridge.supportsBufferBindings()) {
            return decision(request, config, fallbackPath(mode), false,
                    AcceleratorBufferReasonCode.NATIVE_BUFFER_ABI_UNAVAILABLE,
                    "native CUDA buffer ABI unavailable: bridge does not support buffer bindings",
                    List.of(),
                    List.of());
        }

        List<AcceleratorBufferInputDecision> inputDecisions = metadataInputDecisions(request);
        AcceleratorBufferInputDecision rejectedInput = inputDecisions.stream()
                .filter(input -> !input.accepted())
                .findFirst()
                .orElse(null);
        if (rejectedInput != null) {
            return decision(request, config, fallbackPath(mode), false,
                    rejectedInput.reasonCode(), rejectedInput.reason(), inputDecisions, List.of());
        }

        List<AcceleratorBufferOutputDecision> outputDecisions = metadataOutputDecisions(request);
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
                "CUDA dense FLOAT32 buffer metadata accepted",
                inputDecisions,
                outputDecisions);
    }

    public AcceleratorBufferDecision decide(
            AcceleratorBufferRequest request,
            ResolvedAcceleratorInputs inputs,
            AcceleratorBufferConfig bufferConfig,
            ExecutionContext context
    ) {
        Objects.requireNonNull(request, "request cannot be null");
        AcceleratorBufferConfig config = bufferConfig == null ? AcceleratorBufferConfig.defaults() : bufferConfig;
        AcceleratorBufferBindingMode mode = config.bindingMode();
        if (mode == AcceleratorBufferBindingMode.OFF) {
            return decision(request, config, AcceleratorBufferExecutionPath.TENSOR_ARRAY, false,
                    AcceleratorBufferReasonCode.BUFFER_BINDINGS_DISABLED,
                    "buffer bindings disabled", List.of(), List.of());
        }
        if (!bridge.supportsBufferBindings()) {
            return decision(request, config, fallbackPath(mode), false,
                    AcceleratorBufferReasonCode.NATIVE_BUFFER_ABI_UNAVAILABLE,
                    "native CUDA buffer ABI unavailable: bridge does not support buffer bindings",
                    List.of(),
                    List.of());
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
                "CUDA dense FLOAT32 buffer metadata accepted",
                inputDecisions,
                outputDecisions);
    }

    public AcceleratorBufferBindings<CudaBufferBinding> resolve(
            AcceleratorBufferRequest request,
            ResolvedAcceleratorInputs inputs,
            AcceleratorBufferDecision decision,
            ExecutionContext context,
            CudaBufferAllocator allocator
    ) {
        if (decision == null || decision.path() != AcceleratorBufferExecutionPath.BUFFER_BINDING) {
            return new AcceleratorBufferBindings<>(List.of(), List.of());
        }
        Objects.requireNonNull(context, "context cannot be null");
        Objects.requireNonNull(allocator, "allocator cannot be null");
        if (!allocator.available()) {
            throw new UnsupportedOperationException(allocator.unavailableReason());
        }
        context.registerDeviceToCpuMaterializer(
                ComputeBackend.GPU_CUDA.name(),
                new CudaDeviceToCpuMaterializer(allocator)
        );
        List<CudaBufferBinding> inputBindings = resolveInputBindings(request, inputs, context, allocator);
        List<CudaBufferBinding> outputBindings = resolveOutputBindings(request, context, allocator);
        return new AcceleratorBufferBindings<>(inputBindings, outputBindings);
    }

    private static List<AcceleratorBufferInputDecision> inputDecisions(
            AcceleratorBufferRequest request,
            ResolvedAcceleratorInputs inputs,
            AcceleratorBufferConfig config,
            ExecutionContext context
    ) {
        List<AcceleratorBufferInputDecision> out = new ArrayList<>(request.externalInputNodeIds().size());
        for (int i = 0; i < request.externalInputNodeIds().size(); i++) {
            int nodeId = request.externalInputNodeIds().get(i);
            boolean prepared = inputs != null
                    && i < inputs.preparedInputUsed().size()
                    && inputs.preparedInputUsed().get(i);
            Tensor tensor = inputTensor(request, inputs, context, i, nodeId);
            AcceleratorBufferLayout layout = layoutForInput(request, inputs, context, i, nodeId);
            DataType expected = i < request.externalInputDataTypes().size()
                    ? request.externalInputDataTypes().get(i)
                    : null;
            DeviceBufferBinding existing = context == null ? null : context.deviceBufferBindingForNodeId(nodeId);
            if (existing instanceof CudaBufferBinding cudaBinding) {
                String reason = incompatibleBindingReason(layout, cudaBinding, CudaBufferAccess.READ, expected);
                if (reason.isBlank()) {
                    out.add(new AcceleratorBufferInputDecision(nodeId, layout, prepared, true,
                            AcceleratorBufferReasonCode.BUFFER_BINDING_AVAILABLE,
                            "CUDA dense FLOAT32 input binding accepted"));
                    continue;
                }
                out.add(new AcceleratorBufferInputDecision(nodeId, layout, prepared, false,
                        AcceleratorBufferReasonCode.INPUT_BINDING_UNAVAILABLE,
                        "external input nodeId=" + nodeId + " CUDA binding is incompatible: " + reason));
                continue;
            } else if (existing != null) {
                out.add(new AcceleratorBufferInputDecision(nodeId, layout, prepared, false,
                        AcceleratorBufferReasonCode.INPUT_BINDING_UNAVAILABLE,
                        "external input nodeId=" + nodeId + " binding is not CUDA-compatible: "
                                + existing.getClass().getSimpleName()));
                continue;
            }
            if (expected != null && expected != DataType.FLOAT32 || layout.dataType() != DataType.FLOAT32) {
                out.add(new AcceleratorBufferInputDecision(nodeId, layout, false, false,
                        AcceleratorBufferReasonCode.INPUT_DTYPE_UNSUPPORTED,
                        "CUDA buffer input nodeId=" + nodeId + " supports only FLOAT32, got " + layout.dataType()));
                continue;
            }
            if (layout.layoutClass() != AcceleratorBufferLayoutClass.DENSE_CONTIGUOUS) {
                out.add(new AcceleratorBufferInputDecision(nodeId, layout, false, false,
                        AcceleratorBufferReasonCode.INPUT_LAYOUT_UNSUPPORTED,
                        "CUDA buffer input nodeId=" + nodeId + " requires DENSE_CONTIGUOUS layout, got "
                                + layout.layoutClass()));
                continue;
            }
            if (prepared && !config.allowPreparedInputMaterialization()) {
                out.add(new AcceleratorBufferInputDecision(nodeId, layout, true, false,
                        AcceleratorBufferReasonCode.PREPARED_INPUT_MATERIALIZATION_DISABLED,
                        "prepared input materialization disabled for nodeId=" + nodeId));
                continue;
            }
            if (tensor == null || tensor.getDataType() != DataType.FLOAT32) {
                out.add(new AcceleratorBufferInputDecision(nodeId, layout, prepared, false,
                        AcceleratorBufferReasonCode.INPUT_DTYPE_UNSUPPORTED,
                        "CUDA buffer input nodeId=" + nodeId + " supports only FLOAT32 runtime tensors"));
                continue;
            }
            if (!prepared && context != null) {
                var residency = context.residencyForNodeId(nodeId);
                if (residency == null || !residency.cpuCurrent()) {
                    out.add(new AcceleratorBufferInputDecision(nodeId, layout, false, false,
                            AcceleratorBufferReasonCode.INPUT_NOT_CPU_CURRENT,
                            "external input nodeId=" + nodeId
                                    + " has no CUDA binding and CPU storage is not current"));
                    continue;
                }
            }
            out.add(new AcceleratorBufferInputDecision(nodeId, layout, prepared, true,
                    AcceleratorBufferReasonCode.BUFFER_BINDING_AVAILABLE,
                    "CUDA dense FLOAT32 input metadata accepted"));
        }
        return List.copyOf(out);
    }

    private static List<AcceleratorBufferInputDecision> metadataInputDecisions(AcceleratorBufferRequest request) {
        List<AcceleratorBufferInputDecision> out = new ArrayList<>(request.externalInputNodeIds().size());
        for (int i = 0; i < request.externalInputNodeIds().size(); i++) {
            int nodeId = request.externalInputNodeIds().get(i);
            AcceleratorBufferLayout layout = request.externalInputLayouts().get(i);
            DataType expected = request.externalInputDataTypes().get(i);
            if (expected != DataType.FLOAT32 || layout.dataType() != DataType.FLOAT32) {
                out.add(new AcceleratorBufferInputDecision(nodeId, layout, false, false,
                        AcceleratorBufferReasonCode.INPUT_DTYPE_UNSUPPORTED,
                        "CUDA buffer input nodeId=" + nodeId + " supports only FLOAT32, got " + layout.dataType()));
                continue;
            }
            if (layout.layoutClass() != AcceleratorBufferLayoutClass.DENSE_CONTIGUOUS) {
                out.add(new AcceleratorBufferInputDecision(nodeId, layout, false, false,
                        AcceleratorBufferReasonCode.INPUT_LAYOUT_UNSUPPORTED,
                        "CUDA buffer input nodeId=" + nodeId + " requires DENSE_CONTIGUOUS layout, got "
                                + layout.layoutClass()));
                continue;
            }
            out.add(new AcceleratorBufferInputDecision(nodeId, layout, false, true,
                    AcceleratorBufferReasonCode.BUFFER_BINDING_AVAILABLE,
                    "CUDA dense FLOAT32 input metadata accepted"));
        }
        return List.copyOf(out);
    }

    private static List<AcceleratorBufferOutputDecision> outputDecisions(
            AcceleratorBufferRequest request,
            ExecutionContext context
    ) {
        List<AcceleratorBufferOutputDecision> out = new ArrayList<>(request.outputNodeIds().size());
        for (int i = 0; i < request.outputNodeIds().size(); i++) {
            int nodeId = request.outputNodeIds().get(i);
            AcceleratorBufferLayout layout = layoutForOutput(request, context, i, nodeId);
            DataType expected = i < request.outputDataTypes().size() ? request.outputDataTypes().get(i) : null;
            DeviceBufferBinding existing = context == null ? null : context.writableDeviceBufferBindingForNodeId(nodeId);
            if (existing instanceof CudaBufferBinding cudaBinding) {
                String reason = incompatibleBindingReason(layout, cudaBinding, CudaBufferAccess.WRITE, expected);
                if (reason.isBlank()) {
                    out.add(new AcceleratorBufferOutputDecision(nodeId, layout, true,
                            AcceleratorBufferReasonCode.BUFFER_BINDING_AVAILABLE,
                            "CUDA dense FLOAT32 output binding accepted"));
                    continue;
                }
            } else if (existing != null) {
                out.add(new AcceleratorBufferOutputDecision(nodeId, layout, false,
                        AcceleratorBufferReasonCode.OUTPUT_BINDING_UNAVAILABLE,
                        "output nodeId=" + nodeId + " binding is not CUDA-compatible: "
                                + existing.getClass().getSimpleName()));
                continue;
            }
            if (expected != null && expected != DataType.FLOAT32 || layout.dataType() != DataType.FLOAT32) {
                out.add(new AcceleratorBufferOutputDecision(nodeId, layout, false,
                        AcceleratorBufferReasonCode.OUTPUT_DTYPE_UNSUPPORTED,
                        "CUDA buffer output nodeId=" + nodeId + " supports only FLOAT32, got " + layout.dataType()));
                continue;
            }
            if (layout.layoutClass() != AcceleratorBufferLayoutClass.DENSE_CONTIGUOUS) {
                out.add(new AcceleratorBufferOutputDecision(nodeId, layout, false,
                        AcceleratorBufferReasonCode.OUTPUT_LAYOUT_UNSUPPORTED,
                        "CUDA buffer output nodeId=" + nodeId + " requires DENSE_CONTIGUOUS layout, got "
                                + layout.layoutClass()));
                continue;
            }
            out.add(new AcceleratorBufferOutputDecision(nodeId, layout, true,
                    AcceleratorBufferReasonCode.BUFFER_BINDING_AVAILABLE,
                    "CUDA dense FLOAT32 output metadata accepted"));
        }
        return List.copyOf(out);
    }

    private static List<AcceleratorBufferOutputDecision> metadataOutputDecisions(AcceleratorBufferRequest request) {
        List<AcceleratorBufferOutputDecision> out = new ArrayList<>(request.outputNodeIds().size());
        for (int i = 0; i < request.outputNodeIds().size(); i++) {
            int nodeId = request.outputNodeIds().get(i);
            AcceleratorBufferLayout layout = request.outputLayouts().get(i);
            DataType expected = request.outputDataTypes().get(i);
            if (expected != DataType.FLOAT32 || layout.dataType() != DataType.FLOAT32) {
                out.add(new AcceleratorBufferOutputDecision(nodeId, layout, false,
                        AcceleratorBufferReasonCode.OUTPUT_DTYPE_UNSUPPORTED,
                        "CUDA buffer output nodeId=" + nodeId + " supports only FLOAT32, got " + layout.dataType()));
                continue;
            }
            if (layout.layoutClass() != AcceleratorBufferLayoutClass.DENSE_CONTIGUOUS) {
                out.add(new AcceleratorBufferOutputDecision(nodeId, layout, false,
                        AcceleratorBufferReasonCode.OUTPUT_LAYOUT_UNSUPPORTED,
                        "CUDA buffer output nodeId=" + nodeId + " requires DENSE_CONTIGUOUS layout, got "
                                + layout.layoutClass()));
                continue;
            }
            out.add(new AcceleratorBufferOutputDecision(nodeId, layout, true,
                    AcceleratorBufferReasonCode.BUFFER_BINDING_AVAILABLE,
                    "CUDA dense FLOAT32 output metadata accepted"));
        }
        return List.copyOf(out);
    }

    private static List<CudaBufferBinding> resolveInputBindings(
            AcceleratorBufferRequest request,
            ResolvedAcceleratorInputs inputs,
            ExecutionContext context,
            CudaBufferAllocator allocator
    ) {
        List<CudaBufferBinding> bindings = new ArrayList<>(request.externalInputNodeIds().size());
        for (int i = 0; i < request.externalInputNodeIds().size(); i++) {
            int nodeId = request.externalInputNodeIds().get(i);
            AcceleratorBufferLayout layout = layoutForInput(request, inputs, context, i, nodeId);
            DataType expected = i < request.externalInputDataTypes().size() ? request.externalInputDataTypes().get(i) : null;
            DeviceBufferBinding existing = context.deviceBufferBindingForNodeId(nodeId);
            if (existing instanceof CudaBufferBinding cudaBinding
                    && incompatibleBindingReason(layout, cudaBinding, CudaBufferAccess.READ, expected).isBlank()) {
                bindings.add(cudaBinding);
                continue;
            }
            Tensor tensor = inputTensor(request, inputs, context, i, nodeId);
            CudaBufferBinding created = allocator.createInputBinding(nodeId, tensor);
            context.registerResource(new CudaBufferResource(allocator, created.handle()));
            boolean prepared = inputs != null
                    && i < inputs.preparedInputUsed().size()
                    && inputs.preparedInputUsed().get(i);
            if (!prepared) {
                context.attachDeviceBufferBinding(
                        nodeId,
                        created,
                        StorageResidency.HOST_SHARED_DEVICE_BUFFER,
                        "cuda shared input buffer upload"
                );
            }
            bindings.add(created);
        }
        return List.copyOf(bindings);
    }

    private static List<CudaBufferBinding> resolveOutputBindings(
            AcceleratorBufferRequest request,
            ExecutionContext context,
            CudaBufferAllocator allocator
    ) {
        List<CudaBufferBinding> bindings = new ArrayList<>(request.outputNodeIds().size());
        for (int i = 0; i < request.outputNodeIds().size(); i++) {
            int nodeId = request.outputNodeIds().get(i);
            AcceleratorBufferLayout layout = layoutForOutput(request, context, i, nodeId);
            DataType expected = i < request.outputDataTypes().size() ? request.outputDataTypes().get(i) : null;
            DeviceBufferBinding existing = context.writableDeviceBufferBindingForNodeId(nodeId);
            if (existing instanceof CudaBufferBinding cudaBinding
                    && incompatibleBindingReason(layout, cudaBinding, CudaBufferAccess.WRITE, expected).isBlank()) {
                bindings.add(cudaBinding);
                continue;
            }
            CudaBufferBinding created = allocator.createOutputBinding(nodeId, layout);
            context.registerResource(new CudaBufferResource(allocator, created.handle()));
            context.reserveDeviceBufferBinding(nodeId, created);
            bindings.add(created);
        }
        return List.copyOf(bindings);
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
                ComputeBackend.GPU_CUDA,
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
            CudaBufferBinding cudaBinding,
            CudaBufferAccess requiredAccess,
            DataType expectedDataType
    ) {
        if (!ComputeBackend.GPU_CUDA.name().equals(cudaBinding.backendId())) {
            return "binding backend " + cudaBinding.backendId() + " is not GPU_CUDA";
        }
        if (!cudaBinding.available()) {
            return "binding is unavailable: " + cudaBinding.describe();
        }
        AcceleratorBufferLayout actualLayout = cudaBinding.layout();
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
        if (!accessCompatible(cudaBinding.access(), requiredAccess)) {
            return "binding access " + cudaBinding.access() + " is incompatible with required " + requiredAccess;
        }
        return "";
    }

    private static boolean accessCompatible(CudaBufferAccess actual, CudaBufferAccess required) {
        if (actual == CudaBufferAccess.READ_WRITE) {
            return true;
        }
        return actual == required;
    }

    private static Tensor inputTensor(
            AcceleratorBufferRequest request,
            ResolvedAcceleratorInputs inputs,
            ExecutionContext context,
            int index,
            int nodeId
    ) {
        if (inputs != null && index < inputs.executionExternalInputs().size()) {
            return inputs.executionExternalInputs().get(index);
        }
        return context == null ? null : context.runtimeTensorForNodeId(nodeId);
    }

    private static AcceleratorBufferLayout layoutForInput(
            AcceleratorBufferRequest request,
            ResolvedAcceleratorInputs inputs,
            ExecutionContext context,
            int index,
            int nodeId
    ) {
        if (inputs != null && index < inputs.executionExternalInputs().size()) {
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
}
