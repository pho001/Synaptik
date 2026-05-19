package graph.execution.device;

import backend.ComputeBackend;
import backend.accelerator.buffer.AcceleratorBufferAccessMode;
import backend.accelerator.buffer.AcceleratorBufferLayout;
import backend.accelerator.buffer.AcceleratorBufferReasonCode;
import backend.accelerator.buffer.AcceleratorLayoutTransformDecision;
import backend.accelerator.buffer.AcceleratorLayoutTransformKind;
import backend.accelerator.buffer.AcceleratorLayoutTransformPlanner;
import backend.accelerator.buffer.AcceleratorLayoutTransformRequest;
import backend.memory.DeviceBufferBinding;
import backend.memory.StorageResidency;
import backend.runtime.ExecutionContext;
import config.runtime.AcceleratorBufferBindingMode;
import config.runtime.RuntimeConfig;
import graph.CompiledNode;
import graph.execution.PreparedNodeExecution;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

/**
 * Propagates metadata-only accelerator layout views before CPU materialization.
 */
public final class DeviceLayoutViewPropagator {
    private DeviceLayoutViewPropagator() {
    }

    public static boolean tryPropagate(PreparedNodeExecution step, ExecutionContext context) {
        if (step == null || context == null || step.compiledNode() == null) {
            return false;
        }
        Operation operation = step.executionOperation();
        if (operation == null || !isLayoutTransformCandidate(operation.opType())) {
            return false;
        }
        Integer sourceNodeId = firstInputNodeId(step);
        if (sourceNodeId == null) {
            return false;
        }

        CompiledNode targetNode = step.compiledNode();
        DeviceBufferBinding sourceBinding = context.deviceBufferBindingForNodeId(sourceNodeId);
        String backendId = sourceBinding == null ? backendIdFromTarget(targetNode) : sourceBinding.backendId();
        boolean required = isRequired(context.runtimeConfig(), backendId);
        if (backendId == null || backendId.isBlank()) {
            return false;
        }

        AcceleratorLayoutTransformRequest request = new AcceleratorLayoutTransformRequest(
                backendId,
                sourceNodeId,
                targetNode.id(),
                operation.opType(),
                sourceLayout(sourceNodeId, sourceBinding, context),
                targetLayout(targetNode),
                sourceBinding,
                context.runsBackwardPass()
        );
        AcceleratorLayoutTransformDecision decision = AcceleratorLayoutTransformPlanner.decide(request);
        context.publishLayoutTransformDecision(targetNode.id(), decision);
        if (!decision.accepted()) {
            failIfRequired(required, backendId, decision);
            return false;
        }
        if (isGpuMaterialization(decision.kind())) {
            DeviceLayoutMaterializer materializer = context.runtimeService(DeviceLayoutMaterializer.class);
            if (materializer == null) {
                AcceleratorLayoutTransformDecision rejected = AcceleratorLayoutTransformDecision.rejected(
                        request,
                        AcceleratorBufferReasonCode.GPU_LAYOUT_TRANSFORM_UNSUPPORTED,
                        "GPU layout transform unsupported: no layout materializer registered"
                );
                context.publishLayoutTransformDecision(targetNode.id(), rejected);
                failIfRequired(required, backendId, rejected);
                return false;
            }
            DeviceBufferBinding materialized;
            try {
                materialized = materializer.materialize(decision, sourceBinding, context);
            } catch (RuntimeException ex) {
                AcceleratorLayoutTransformDecision rejected = AcceleratorLayoutTransformDecision.rejected(
                        request,
                        materializerFailureReasonCode(ex),
                        "GPU layout transform unsupported: layout materializer failed: " + safeMessage(ex)
                );
                context.publishLayoutTransformDecision(targetNode.id(), rejected);
                failIfRequired(required, backendId, rejected);
                return false;
            }
            if (materialized == null || !materialized.available()) {
                AcceleratorLayoutTransformDecision rejected = AcceleratorLayoutTransformDecision.rejected(
                        request,
                        AcceleratorBufferReasonCode.GPU_LAYOUT_TRANSFORM_UNSUPPORTED,
                        "GPU layout transform unsupported: layout materializer produced no binding"
                );
                context.publishLayoutTransformDecision(targetNode.id(), rejected);
                failIfRequired(required, backendId, rejected);
                return false;
            }
            context.attachDeviceBufferBinding(
                    targetNode.id(),
                    materialized,
                    StorageResidency.DEVICE_OWNED,
                    materializationReason(backendId)
            );
            return true;
        }
        if (decision.kind() != AcceleratorLayoutTransformKind.METADATA_ONLY_VIEW) {
            return false;
        }

        DeviceBufferBinding targetBinding = viewBinding(targetNode.id(), decision.targetLayout(), sourceBinding, operation.opType());
        if (targetBinding == null) {
            failIfRequired(required, backendId, AcceleratorLayoutTransformDecision.rejected(
                    request,
                    AcceleratorBufferReasonCode.GPU_LAYOUT_TRANSFORM_UNSUPPORTED,
                    "GPU layout transform unsupported for source binding type "
                            + (sourceBinding == null ? "null" : sourceBinding.getClass().getName())
            ));
            return false;
        }
        context.attachDeviceBufferBinding(
                targetNode.id(),
                targetBinding,
                metadataOnlyViewResidency(sourceNodeId, context),
                "device layout view propagation"
        );
        return true;
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

    private static StorageResidency metadataOnlyViewResidency(int sourceNodeId, ExecutionContext context) {
        var sourceResidency = context.residencyForNodeId(sourceNodeId);
        if (sourceResidency != null && sourceResidency.cpuCurrent()) {
            return StorageResidency.HOST_SHARED_DEVICE_BUFFER;
        }
        return StorageResidency.DEVICE_OWNED;
    }

    private static Integer firstInputNodeId(PreparedNodeExecution step) {
        List<Integer> inputIds = step.metadata().executionInputNodeIds().isEmpty()
                ? step.compiledNode().inputIds()
                : step.metadata().executionInputNodeIds();
        return inputIds.isEmpty() ? null : inputIds.getFirst();
    }

    private static AcceleratorBufferLayout sourceLayout(
            int sourceNodeId,
            DeviceBufferBinding sourceBinding,
            ExecutionContext context
    ) {
        if (sourceBinding != null) {
            return sourceBinding.layout();
        }
        Tensor source = context.runtimeTensorForNodeId(sourceNodeId);
        return AcceleratorBufferLayout.fromTensor(source);
    }

    private static AcceleratorBufferLayout targetLayout(CompiledNode targetNode) {
        return AcceleratorBufferLayout.of(
                targetNode.dataType(),
                targetNode.shape(),
                targetNode.strides(),
                targetNode.storageOffset(),
                targetNode.flatDataSize()
        );
    }

    private static DeviceBufferBinding viewBinding(
            int targetNodeId,
            AcceleratorBufferLayout targetLayout,
            DeviceBufferBinding sourceBinding,
            Operation.OpType opType
    ) {
        if (sourceBinding == null) {
            return null;
        }
        AcceleratorBufferAccessMode access = opType == Operation.OpType.EXPAND
                ? AcceleratorBufferAccessMode.READ
                : AcceleratorBufferAccessMode.READ_WRITE;
        return sourceBinding.viewOf(targetNodeId, targetLayout, access);
    }

    private static String backendIdFromTarget(CompiledNode targetNode) {
        ComputeBackend backend = targetNode.backend();
        if (backend == ComputeBackend.GPU_METAL || backend == ComputeBackend.GPU_CUDA) {
            return backend.name();
        }
        return "";
    }

    private static boolean isRequired(RuntimeConfig runtimeConfig, String backendId) {
        if (runtimeConfig == null || backendId == null || backendId.isBlank()) {
            return false;
        }
        if (ComputeBackend.GPU_METAL.name().equals(backendId)) {
            return runtimeConfig.accelerator().metal().buffer().bindingMode() == AcceleratorBufferBindingMode.REQUIRE;
        }
        if (ComputeBackend.GPU_CUDA.name().equals(backendId)) {
            return runtimeConfig.accelerator().cuda().buffer().bindingMode() == AcceleratorBufferBindingMode.REQUIRE;
        }
        return false;
    }

    private static String materializationReason(String backendId) {
        if (ComputeBackend.GPU_METAL.name().equals(backendId)) {
            return "metal gpu layout materialization";
        }
        if (ComputeBackend.GPU_CUDA.name().equals(backendId)) {
            return "cuda gpu layout materialization";
        }
        return "gpu layout materialization";
    }

    private static void failIfRequired(boolean required, String backendId, AcceleratorLayoutTransformDecision decision) {
        if (!required) {
            return;
        }
        throw new IllegalStateException("Accelerator buffer path is required for " + backendId
                + " but unavailable: " + decision.reasonCode().name()
                + " - " + decision.reason());
    }

    private static String safeMessage(RuntimeException ex) {
        String message = ex.getMessage();
        return message == null || message.isBlank() ? ex.getClass().getSimpleName() : message;
    }

    private static boolean isLayoutTransformCandidate(Operation.OpType opType) {
        return switch (opType) {
            case NOOP, SELECT, PERMUTE, EXPAND, EXPAND_DIMS, SQUEEZE, RESHAPE, CONTIGUOUS -> true;
            default -> false;
        };
    }
}
