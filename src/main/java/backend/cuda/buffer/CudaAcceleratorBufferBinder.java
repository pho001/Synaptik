package backend.cuda.buffer;

import backend.ComputeBackend;
import backend.accelerator.buffer.AcceleratorBufferDecision;
import backend.accelerator.buffer.AcceleratorBufferExecutionPath;
import backend.accelerator.buffer.AcceleratorBufferInputDecision;
import backend.accelerator.buffer.AcceleratorBufferLayout;
import backend.accelerator.buffer.AcceleratorBufferLayoutClass;
import backend.accelerator.buffer.AcceleratorBufferOutputDecision;
import backend.accelerator.buffer.AcceleratorBufferReasonCode;
import backend.accelerator.buffer.AcceleratorBufferRequest;
import backend.cuda.bridge.CudaGraphBridge;
import config.runtime.AcceleratorBufferBindingMode;
import config.runtime.AcceleratorBufferConfig;
import tensor.DataType;

import java.util.ArrayList;
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

        List<AcceleratorBufferInputDecision> inputDecisions = inputDecisions(request);
        AcceleratorBufferInputDecision rejectedInput = inputDecisions.stream()
                .filter(input -> !input.accepted())
                .findFirst()
                .orElse(null);
        if (rejectedInput != null) {
            return decision(request, config, fallbackPath(mode), false,
                    rejectedInput.reasonCode(), rejectedInput.reason(), inputDecisions, List.of());
        }

        List<AcceleratorBufferOutputDecision> outputDecisions = outputDecisions(request);
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

    private static List<AcceleratorBufferInputDecision> inputDecisions(AcceleratorBufferRequest request) {
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

    private static List<AcceleratorBufferOutputDecision> outputDecisions(AcceleratorBufferRequest request) {
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
}
