package backend.accelerator.buffer;

import backend.ComputeBackend;
import config.runtime.AcceleratorBufferBindingMode;

import java.util.List;
import java.util.Objects;

/**
 * Backend-neutral decision describing whether an accelerator executable used or skipped native buffers.
 *
 * @param backend accelerator backend
 * @param mode configured buffer binding mode
 * @param path selected execution path
 * @param allowed whether buffer execution is allowed by the decision
 * @param required whether buffer execution was required
 * @param reasonCode stable reason code
 * @param reason human-readable detail
 * @param inputs per-input diagnostics
 * @param outputs per-output diagnostics
 */
public record AcceleratorBufferDecision(
        ComputeBackend backend,
        AcceleratorBufferBindingMode mode,
        AcceleratorBufferExecutionPath path,
        boolean allowed,
        boolean required,
        AcceleratorBufferReasonCode reasonCode,
        String reason,
        List<AcceleratorBufferInputDecision> inputs,
        List<AcceleratorBufferOutputDecision> outputs
) {
    public AcceleratorBufferDecision {
        Objects.requireNonNull(backend, "backend cannot be null");
        mode = mode == null ? AcceleratorBufferBindingMode.AUTO : mode;
        path = path == null ? AcceleratorBufferExecutionPath.UNAVAILABLE : path;
        reasonCode = reasonCode == null ? AcceleratorBufferReasonCode.NOT_EVALUATED : reasonCode;
        reason = reason == null ? "" : reason;
        inputs = List.copyOf(inputs == null ? List.of() : inputs);
        outputs = List.copyOf(outputs == null ? List.of() : outputs);
    }

    public static AcceleratorBufferDecision notEvaluated(ComputeBackend backend) {
        return new AcceleratorBufferDecision(
                backend,
                AcceleratorBufferBindingMode.AUTO,
                AcceleratorBufferExecutionPath.UNAVAILABLE,
                false,
                false,
                AcceleratorBufferReasonCode.NOT_EVALUATED,
                "not evaluated yet",
                List.of(),
                List.of()
        );
    }

    public boolean preparedInputUsed() {
        return inputs.stream().anyMatch(AcceleratorBufferInputDecision::preparedInputUsed);
    }
}
