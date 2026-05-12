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
import backend.accelerator.buffer.AcceleratorLayoutTransformDecision;
import backend.accelerator.buffer.AcceleratorLayoutTransformKind;
import backend.accelerator.buffer.AcceleratorLayoutTransformPlanner;
import backend.accelerator.buffer.AcceleratorLayoutTransformRequest;
import backend.accelerator.exec.ResolvedAcceleratorInputs;
import backend.memory.DeviceBufferBinding;
import backend.memory.StorageResidency;
import backend.metal.MetalMpsCapabilities;
import backend.metal.bridge.MetalMpsBridgeContext;
import backend.metal.bridge.MetalMpsGraphBridge;
import backend.runtime.ExecutionContext;
import config.runtime.AcceleratorBufferBindingMode;
import config.runtime.AcceleratorBufferConfig;
import graph.execution.DeviceLayoutMaterializer;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorMetadata;
import tensor.TensorRemap;

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

    /**
     * Initializes and validates the run-independent allocator capability during prepare.
     *
     * <p>This does not allocate any Metal buffers. It only resolves the bridge-scoped allocator object so
     * prepared executables can keep static transport availability out of the hot execute path.</p>
     *
     * @return empty when the allocator is available, otherwise a stable unavailable reason
     */
    public String prepareAllocatorUnavailableReason() {
        MetalBufferAllocator resolvedAllocator = allocator();
        return resolvedAllocator.available() ? "" : resolvedAllocator.unavailableReason();
    }

    /**
     * Registers run-scoped Metal materialization services used by later layout/view steps.
     */
    public void registerRuntimeServices(ExecutionContext context) {
        if (context == null) {
            return;
        }
        MetalBufferAllocator resolvedAllocator = allocator();
        if (!resolvedAllocator.available()) {
            return;
        }
        context.registerRuntimeService(MetalBufferAllocator.class, resolvedAllocator);
        context.registerRuntimeService(MetalMpsBridgeContext.class, bridgeContext);
        context.registerDeviceToCpuMaterializer(
                ComputeBackend.GPU_METAL.name(),
                new MetalDeviceToCpuMaterializer(resolvedAllocator)
        );
        if (bridge.supportsLayoutMaterialization()) {
            context.registerRuntimeService(
                    DeviceLayoutMaterializer.class,
                    new MetalDeviceLayoutMaterializer(bridge, bridgeContext, resolvedAllocator)
            );
        }
    }

    /**
     * Performs only runtime-dependent buffer validation.
     *
     * <p>Callers must run static transport gates during prepare before using this method: buffer mode,
     * native ABI availability, minimum work threshold, bridge availability, executable availability, and
     * static dtype legality. This method intentionally keeps the execute path focused on facts that require
     * the current {@link ExecutionContext}: existing device bindings, runtime residency/currentness, and
     * concrete tensor layouts.</p>
     */
    public AcceleratorBufferDecision validateRuntime(
            AcceleratorBufferRequest request,
            ResolvedAcceleratorInputs inputs,
            AcceleratorBufferConfig bufferConfig,
            ExecutionContext context
    ) {
        AcceleratorBufferConfig config = bufferConfig == null ? AcceleratorBufferConfig.defaults() : bufferConfig;
        List<AcceleratorBufferInputDecision> inputDecisions = inputDecisions(request, inputs, config, context);
        AcceleratorBufferInputDecision rejectedInput = inputDecisions.stream()
                .filter(input -> !input.accepted())
                .findFirst()
                .orElse(null);
        if (rejectedInput != null) {
            return decision(request, config, fallbackPath(config.bindingMode()), false,
                    rejectedInput.reasonCode(), rejectedInput.reason(), inputDecisions, List.of());
        }

        List<AcceleratorBufferOutputDecision> outputDecisions = outputDecisions(request, context);
        AcceleratorBufferOutputDecision rejectedOutput = outputDecisions.stream()
                .filter(output -> !output.accepted())
                .findFirst()
                .orElse(null);
        if (rejectedOutput != null) {
            return decision(request, config, fallbackPath(config.bindingMode()), false,
                    rejectedOutput.reasonCode(), rejectedOutput.reason(), inputDecisions, outputDecisions);
        }

        return decision(request, config, AcceleratorBufferExecutionPath.BUFFER_BINDING, true,
                AcceleratorBufferReasonCode.BUFFER_BINDING_AVAILABLE,
                "using native buffer bindings", inputDecisions, outputDecisions);
    }

    /**
     * Attempts GPU-side dense materialization for non-contiguous external inputs before native buffer execution.
     *
     * <p>MPSGraph buffer execution consumes dense logical tensors. Metadata-only view bindings are safe to propagate
     * between layout nodes, but a compute region boundary must either receive a dense binding or perform an explicit
     * device-side layout legalization. This method performs that repair without making the Java tensor CPU-current.</p>
     */
    public boolean repairRuntimeInputLayouts(
            AcceleratorBufferRequest request,
            ResolvedAcceleratorInputs inputs,
            AcceleratorBufferConfig bufferConfig,
            ExecutionContext context
    ) {
        if (context == null || request == null || inputs == null || !bridge.supportsLayoutMaterialization()) {
            return false;
        }
        AcceleratorBufferConfig config = bufferConfig == null ? AcceleratorBufferConfig.defaults() : bufferConfig;
        registerRuntimeServices(context);
        DeviceLayoutMaterializer materializer = context.runtimeService(DeviceLayoutMaterializer.class);
        if (materializer == null) {
            return false;
        }
        boolean repaired = false;
        for (int i = 0; i < request.externalInputNodeIds().size(); i++) {
            int nodeId = request.externalInputNodeIds().get(i);
            DataType expected = i < request.externalInputDataTypes().size() ? request.externalInputDataTypes().get(i) : null;
            AcceleratorBufferLayout expectedLayout = layoutForInput(request, inputs, context, i, nodeId);
            DeviceBufferBinding existing = context.deviceBufferBindingForNodeId(nodeId);
            if (!(existing instanceof MetalBufferBinding metalBinding) || !metalBinding.available()) {
                if (existing == null
                        && config.allowPreparedInputMaterialization()
                        && canRepairCpuCurrentInputLayout(expectedLayout, expected)
                        && cpuCurrent(context, nodeId)) {
                    Tensor tensor = i < inputs.executionExternalInputs().size()
                            ? inputs.executionExternalInputs().get(i)
                            : context.runtimeTensorForNodeId(nodeId);
                    MetalBufferAllocator resolvedAllocator = allocator();
                    if (!resolvedAllocator.available()) {
                        continue;
                    }
                    Tensor denseTensor = denseCpuTensor(tensor);
                    MetalBufferBinding created = expected == DataType.BOOL
                            ? resolvedAllocator.createPredicateInputBinding(nodeId, denseTensor)
                            : resolvedAllocator.createInputBinding(nodeId, denseTensor);
                    context.registerResource(new MetalBufferResource(resolvedAllocator, created.handle()));
                    context.attachDeviceBufferBinding(
                            nodeId,
                            created,
                            StorageResidency.HOST_SHARED_DEVICE_BUFFER,
                            "metal CPU-current input layout upload repair"
                    );
                    repaired = true;
                }
                continue;
            }
            if (incompatibleInputBindingReason(expectedLayout, metalBinding, expected).isBlank()) {
                continue;
            }
            if (!canRepairInputLayout(expectedLayout, metalBinding.layout(), expected)) {
                continue;
            }
            AcceleratorBufferLayout targetLayout = denseLayoutLike(expectedLayout);
            AcceleratorLayoutTransformRequest transformRequest = new AcceleratorLayoutTransformRequest(
                    ComputeBackend.GPU_METAL.name(),
                    nodeId,
                    nodeId,
                    Operation.OpType.CONTIGUOUS,
                    metalBinding.layout(),
                    targetLayout,
                    metalBinding,
                    request.runsBackwardPass()
            );
            AcceleratorLayoutTransformDecision decision = AcceleratorLayoutTransformPlanner.decide(transformRequest);
            context.publishLayoutTransformDecision(nodeId, decision);
            if (!decision.accepted() || !isGpuMaterialization(decision.kind())) {
                continue;
            }
            DeviceBufferBinding materialized;
            try {
                materialized = materializer.materialize(decision, metalBinding, context);
            } catch (RuntimeException ex) {
                context.publishLayoutTransformDecision(nodeId, AcceleratorLayoutTransformDecision.rejected(
                        transformRequest,
                        materializerFailureReasonCode(ex),
                        "GPU layout transform unsupported: external input layout materializer failed: " + safeMessage(ex)
                ));
                continue;
            }
            if (materialized == null || !materialized.available()) {
                context.publishLayoutTransformDecision(nodeId, AcceleratorLayoutTransformDecision.rejected(
                        transformRequest,
                        AcceleratorBufferReasonCode.GPU_LAYOUT_TRANSFORM_UNSUPPORTED,
                        "GPU layout transform unsupported: external input layout materializer produced no binding"
                ));
                continue;
            }
            context.attachDeviceBufferBinding(
                    nodeId,
                    materialized,
                    StorageResidency.DEVICE_OWNED,
                    "metal external input layout repair"
            );
            repaired = true;
        }
        return repaired;
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
        registerRuntimeServices(context);
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
            DeviceBufferBinding existing = context.deviceBufferBindingForNodeId(nodeId);
            if (existing instanceof MetalBufferBinding metalBinding) {
                MetalLayoutPolicy.Decision layoutDecision = MetalLayoutPolicy.existingDeviceInput(layout);
                if (!layoutDecision.accepted()) {
                    out.add(new AcceleratorBufferInputDecision(nodeId, layout, prepared, false,
                            layoutDecision.reasonCode(),
                            "external input nodeId=" + nodeId + " input tensor layout unsupported: "
                                    + layoutDecision.reason()));
                    continue;
                }
                String reason = incompatibleInputBindingReason(layout, metalBinding, expected);
                if (reason.isBlank()) {
                    out.add(new AcceleratorBufferInputDecision(nodeId, layout, prepared, true,
                            AcceleratorBufferReasonCode.BUFFER_BINDING_AVAILABLE, layoutDecision.reason()));
                    continue;
                }
                out.add(new AcceleratorBufferInputDecision(nodeId, layout, prepared, false,
                        inputBindingReasonCode(reason),
                        "external input nodeId=" + nodeId + " input binding unsupported: " + reason));
                continue;
            } else if (existing != null) {
                out.add(new AcceleratorBufferInputDecision(nodeId, layout, prepared, false,
                        AcceleratorBufferReasonCode.INPUT_BINDING_UNAVAILABLE,
                        "external input nodeId=" + nodeId + " binding is not Metal-compatible: "
                        + existing.getClass().getSimpleName()));
                continue;
            }
            MetalLayoutPolicy.Decision layoutDecision = MetalLayoutPolicy.cpuUploadInput(layout);
            if (!layoutDecision.accepted()) {
                out.add(new AcceleratorBufferInputDecision(nodeId, layout, prepared, false,
                        layoutDecision.reasonCode(),
                        "external input nodeId=" + nodeId + " input tensor layout unsupported: "
                                + layoutDecision.reason()));
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
            MetalLayoutPolicy.Decision layoutDecision = MetalLayoutPolicy.output(layout);
            if (!layoutDecision.accepted()) {
                out.add(new AcceleratorBufferOutputDecision(nodeId, layout, false,
                        layoutDecision.reasonCode(),
                        "output nodeId=" + nodeId + " output tensor layout unsupported: "
                                + layoutDecision.reason()));
                continue;
            }
            out.add(new AcceleratorBufferOutputDecision(nodeId, layout, true,
                    AcceleratorBufferReasonCode.BUFFER_BINDING_AVAILABLE, layoutDecision.reason()));
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
                    && incompatibleInputBindingReason(layout, metalBinding, expected).isBlank()) {
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
                    && incompatibleOutputBindingReason(layout, metalBinding, MetalBufferAccess.WRITE, expected).isBlank()) {
                bindings.add(metalBinding);
                continue;
            }
            // Policy-approved DENSE_PHYSICAL_LOGICAL_VIEW outputs allocate dense physical bytes while
            // retaining logical layout metadata for later CPU materialization.
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

    private static String incompatibleInputBindingReason(
            AcceleratorBufferLayout expectedLayout,
            MetalBufferBinding metalBinding,
            DataType expectedDataType
    ) {
        String common = incompatibleCommonBindingReason(expectedLayout, metalBinding, expectedDataType);
        if (!common.isBlank()) {
            return common;
        }
        AcceleratorBufferLayout actualLayout = metalBinding.layout();
        if (denseLegalizedInputMatches(expectedLayout, actualLayout)) {
            return accessCompatible(metalBinding.access(), MetalBufferAccess.READ)
                    ? ""
                    : "binding access " + metalBinding.access() + " is incompatible with required " + MetalBufferAccess.READ;
        }
        String exact = exactLayoutMismatchReason(expectedLayout, actualLayout);
        if (!exact.isBlank()) {
            return exact;
        }
        if (actualLayout.layoutClass() != AcceleratorBufferLayoutClass.DENSE_CONTIGUOUS) {
            return "binding layout " + actualLayout.layoutClass()
                    + " must be materialized to DENSE_CONTIGUOUS before Metal native buffer compute";
        }
        if (!accessCompatible(metalBinding.access(), MetalBufferAccess.READ)) {
            return "binding access " + metalBinding.access() + " is incompatible with required " + MetalBufferAccess.READ;
        }
        return "";
    }

    private static String incompatibleOutputBindingReason(
            AcceleratorBufferLayout expectedLayout,
            MetalBufferBinding metalBinding,
            MetalBufferAccess requiredAccess,
            DataType expectedDataType
    ) {
        String common = incompatibleCommonBindingReason(expectedLayout, metalBinding, expectedDataType);
        if (!common.isBlank()) {
            return common;
        }
        String exact = exactLayoutMismatchReason(expectedLayout, metalBinding.layout());
        if (!exact.isBlank()) {
            return exact;
        }
        if (!accessCompatible(metalBinding.access(), requiredAccess)) {
            return "binding access " + metalBinding.access() + " is incompatible with required " + requiredAccess;
        }
        return "";
    }

    private static String incompatibleCommonBindingReason(
            AcceleratorBufferLayout expectedLayout,
            MetalBufferBinding metalBinding,
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
        if (expectedLayout != null && actualLayout.logicalElementCount() != expectedLayout.logicalElementCount()) {
            return "binding elementCount " + actualLayout.logicalElementCount()
                    + " does not match expected elementCount " + expectedLayout.logicalElementCount();
        }
        return "";
    }

    private static String exactLayoutMismatchReason(
            AcceleratorBufferLayout expectedLayout,
            AcceleratorBufferLayout actualLayout
    ) {
        if (expectedLayout != null && !Arrays.equals(actualLayout.strides(), expectedLayout.strides())) {
            return "binding strides " + Arrays.toString(actualLayout.strides())
                    + " do not match expected strides " + Arrays.toString(expectedLayout.strides());
        }
        if (expectedLayout != null && actualLayout.storageOffset() != expectedLayout.storageOffset()) {
            return "binding storageOffset " + actualLayout.storageOffset()
                    + " does not match expected storageOffset " + expectedLayout.storageOffset();
        }
        return "";
    }

    private static boolean denseLegalizedInputMatches(AcceleratorBufferLayout expectedLayout, AcceleratorBufferLayout actualLayout) {
        if (expectedLayout == null || actualLayout == null) {
            return false;
        }
        return actualLayout.layoutClass() == AcceleratorBufferLayoutClass.DENSE_CONTIGUOUS
                && actualLayout.storageOffset() == 0
                && expectedLayout.dataType() == actualLayout.dataType()
                && Arrays.equals(expectedLayout.shape(), actualLayout.shape())
                && expectedLayout.logicalElementCount() == actualLayout.logicalElementCount()
                && Arrays.equals(actualLayout.strides(), TensorMetadata.computeStrides(actualLayout.shape()));
    }

    private static boolean canRepairInputLayout(
            AcceleratorBufferLayout expectedLayout,
            AcceleratorBufferLayout sourceLayout,
            DataType expectedDataType
    ) {
        if (expectedLayout == null || sourceLayout == null) {
            return false;
        }
        if (expectedDataType != null && sourceLayout.dataType() != expectedDataType) {
            return false;
        }
        if (expectedLayout.dataType() != DataType.FLOAT32 || sourceLayout.dataType() != DataType.FLOAT32) {
            return false;
        }
        if (!Arrays.equals(expectedLayout.shape(), sourceLayout.shape())
                || expectedLayout.logicalElementCount() != sourceLayout.logicalElementCount()) {
            return false;
        }
        if (sourceLayout.layoutClass() == AcceleratorBufferLayoutClass.DENSE_CONTIGUOUS
                && sourceLayout.storageOffset() == 0) {
            return false;
        }
        return sourceLayout.layoutClass() != AcceleratorBufferLayoutClass.UNSUPPORTED;
    }

    private static boolean canRepairCpuCurrentInputLayout(
            AcceleratorBufferLayout expectedLayout,
            DataType expectedDataType
    ) {
        if (expectedLayout == null) {
            return false;
        }
        if (expectedDataType != null && expectedLayout.dataType() != expectedDataType) {
            return false;
        }
        if (expectedLayout.layoutClass() == AcceleratorBufferLayoutClass.DENSE_CONTIGUOUS
                || expectedLayout.layoutClass() == AcceleratorBufferLayoutClass.UNSUPPORTED) {
            return false;
        }
        return expectedLayout.dataType() == DataType.FLOAT32
                || expectedLayout.dataType() == DataType.BFLOAT16
                || expectedLayout.dataType() == DataType.BOOL;
    }

    private static boolean cpuCurrent(ExecutionContext context, int nodeId) {
        var residency = context.residencyForNodeId(nodeId);
        return residency != null && residency.cpuCurrent();
    }

    private static Tensor denseCpuTensor(Tensor tensor) {
        Objects.requireNonNull(tensor, "tensor cannot be null");
        int[] shape = tensor.getShape();
        int elements = tensor.getFlatDataSize();
        Tensor dense = switch (tensor.getDataType()) {
            case FLOAT32 -> new Tensor(new float[elements], shape, null, tensor.getLabel() + "_metal_dense_input", DataType.FLOAT32);
            case BFLOAT16 -> new Tensor(new short[elements], shape, null, tensor.getLabel() + "_metal_dense_input", DataType.BFLOAT16);
            case BOOL -> new Tensor(new byte[elements], shape, null, tensor.getLabel() + "_metal_dense_input", DataType.BOOL);
            default -> throw new UnsupportedOperationException("Metal CPU-current layout upload repair supports FLOAT32/BFLOAT16/BOOL only; got "
                    + tensor.getDataType());
        };
        TensorRemap.applyTrusted(tensor, dense, null, Integer.MAX_VALUE);
        return dense;
    }

    private static AcceleratorBufferLayout denseLayoutLike(AcceleratorBufferLayout layout) {
        int[] shape = layout.shape();
        return AcceleratorBufferLayout.of(
                layout.dataType(),
                shape,
                TensorMetadata.computeStrides(shape),
                0,
                layout.logicalElementCount()
        );
    }

    private static boolean isGpuMaterialization(AcceleratorLayoutTransformKind kind) {
        return kind == AcceleratorLayoutTransformKind.DENSE_GPU_MATERIALIZATION
                || kind == AcceleratorLayoutTransformKind.BROADCAST_GPU_MATERIALIZATION;
    }

    private static AcceleratorBufferReasonCode materializerFailureReasonCode(RuntimeException ex) {
        String message = safeMessage(ex);
        if (message.contains(AcceleratorBufferReasonCode.NATIVE_LAYOUT_DTYPE_UNSUPPORTED.name())) {
            return AcceleratorBufferReasonCode.NATIVE_LAYOUT_DTYPE_UNSUPPORTED;
        }
        if (message.contains(AcceleratorBufferReasonCode.NATIVE_LAYOUT_METADATA_UNSUPPORTED.name())) {
            return AcceleratorBufferReasonCode.NATIVE_LAYOUT_METADATA_UNSUPPORTED;
        }
        return AcceleratorBufferReasonCode.GPU_LAYOUT_TRANSFORM_UNSUPPORTED;
    }

    private static AcceleratorBufferReasonCode inputBindingReasonCode(String reason) {
        if (reason.contains("dtype")) {
            return AcceleratorBufferReasonCode.INPUT_DTYPE_UNSUPPORTED;
        }
        if (reason.contains("access") || reason.contains("unavailable")) {
            return AcceleratorBufferReasonCode.INPUT_BINDING_UNAVAILABLE;
        }
        return AcceleratorBufferReasonCode.INPUT_LAYOUT_UNSUPPORTED;
    }

    private static String safeMessage(RuntimeException ex) {
        String message = ex.getMessage();
        return message == null || message.isBlank() ? ex.getClass().getSimpleName() : message;
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

}
