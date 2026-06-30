package backend.metal.bridge;

import runtime.device.buffer.AcceleratorBufferLayoutClass;
import backend.metal.buffer.MetalBufferAccess;
import backend.metal.buffer.MetalBufferBinding;
import backend.metal.kernel.MetalCustomKernelBridge;
import backend.metal.kernel.MetalCustomKernelCandidate;
import backend.metal.kernel.MetalCustomKernelCapabilities;
import backend.metal.kernel.MetalCustomKernelExecutable;
import backend.metal.lowering.MetalPartitionPlan;
import backend.metal.exec.MetalRouteReasonCode;
import tensor.DataType;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;
import java.util.List;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * FFM-backed custom Metal kernel bridge for scoped native kernels.
 */
public final class MetalMpsFfmCustomKernelBridge implements MetalCustomKernelBridge {
    private static final State STATE = init();

    @Override
    public MetalCustomKernelCapabilities capabilities() {
        if (!STATE.available) {
            return MetalCustomKernelCapabilities.unavailable(STATE.reason);
        }
        return new MetalCustomKernelCapabilities(
                true,
                MetalRouteReasonCode.CUSTOM_KERNEL_SELECTED,
                ""
        );
    }

    @Override
    public MetalCustomKernelExecutable compile(MetalPartitionPlan plan) {
        MetalCustomKernelCandidate candidate = MetalCustomKernelCandidate.evaluate(plan);
        if (!candidate.supported()) {
            return MetalCustomKernelExecutable.unavailable(candidate.reason());
        }
        if (!STATE.available) {
            return MetalCustomKernelExecutable.unavailable(STATE.reason);
        }
        return new MetalCustomKernelExecutable(
                true,
                candidate.kernelId(),
                candidate.primitiveIds(),
                MetalRouteReasonCode.CUSTOM_KERNEL_SELECTED,
                ""
        );
    }

    @Override
    public MetalMpsBridgeExecutionStats executeBuffers(
            MetalMpsBridgeContext context,
            MetalCustomKernelExecutable executable,
            List<MetalBufferBinding> externalInputs,
            List<MetalBufferBinding> outputs
    ) {
        if (!STATE.available) {
            throw new UnsupportedOperationException(STATE.reason);
        }
        if (context == null || !context.available()) {
            throw new UnsupportedOperationException(context == null ? "Missing Metal bridge context." : context.reason());
        }
        if (executable == null || !executable.available()) {
            throw new UnsupportedOperationException(executable == null ? "Missing custom Metal executable." : executable.reason());
        }
        if (!MetalCustomKernelCandidate.RELU_F32_KERNEL_ID.equals(executable.kernelId())) {
            throw new UnsupportedOperationException("Unsupported custom Metal kernel id: " + executable.kernelId());
        }
        List<MetalBufferBinding> inputs = externalInputs == null ? List.of() : List.copyOf(externalInputs);
        List<MetalBufferBinding> out = outputs == null ? List.of() : List.copyOf(outputs);
        if (inputs.size() != 1 || out.size() != 1) {
            throw new UnsupportedOperationException("Custom Metal RELU kernel requires exactly one input and one output.");
        }
        MetalBufferBinding input = inputs.getFirst();
        MetalBufferBinding output = out.getFirst();
        validateDenseF32(input, "input", true);
        validateDenseF32(output, "output", false);
        if (input.layout().logicalElementCount() != output.layout().logicalElementCount()) {
            throw new UnsupportedOperationException("Custom Metal RELU kernel input/output element counts differ.");
        }

        long totalStart = System.nanoTime();
        long nativeStart = totalStart;
        int status;
        try {
            status = (int) STATE.reluF32BufferFn.invokeExact(
                    context.handle(),
                    input.handle().nativeHandle(),
                    output.handle().nativeHandle(),
                    output.layout().logicalElementCount(),
                    input.logicalByteLength(),
                    output.logicalByteLength()
            );
        } catch (Throwable t) {
            throw new UnsupportedOperationException("Custom Metal RELU kernel failed: " + safeMessage(t), t);
        }
        long nativeExecuteNs = System.nanoTime() - nativeStart;
        if (status != 0) {
            throw new UnsupportedOperationException("Custom Metal RELU kernel returned non-zero status: " + status);
        }
        return new MetalMpsBridgeExecutionStats(
                false,
                "",
                MetalMpsBridgeExecutionPath.CUSTOM_KERNEL,
                MetalNativeCopyStrategy.TRUE_OUTPUT_BUFFER_WRITE,
                inputs.size(),
                out.size(),
                input.logicalByteLength(),
                output.logicalByteLength(),
                0L,
                0L,
                nativeExecuteNs,
                0L,
                0L,
                System.nanoTime() - totalStart
        );
    }

    private static void validateDenseF32(MetalBufferBinding binding, String role, boolean input) {
        if (binding == null || !binding.available()) {
            throw new UnsupportedOperationException("Custom Metal " + role + " binding is unavailable.");
        }
        if (binding.layout().dataType() != DataType.FLOAT32) {
            throw new UnsupportedOperationException("Custom Metal " + role + " requires FLOAT32, got "
                    + binding.layout().dataType() + ".");
        }
        if (binding.layout().layoutClass() != AcceleratorBufferLayoutClass.DENSE_CONTIGUOUS
                || binding.layout().storageOffset() != 0) {
            throw new UnsupportedOperationException("Custom Metal " + role + " requires dense contiguous layout, got "
                    + binding.layout().describe() + ".");
        }
        if (input && binding.access() != MetalBufferAccess.READ && binding.access() != MetalBufferAccess.READ_WRITE) {
            throw new UnsupportedOperationException("Custom Metal input binding is not readable.");
        }
        if (!input && binding.access() != MetalBufferAccess.WRITE && binding.access() != MetalBufferAccess.READ_WRITE) {
            throw new UnsupportedOperationException("Custom Metal output binding is not writable.");
        }
        if (binding.handle().byteLength() < binding.logicalByteLength()) {
            throw new UnsupportedOperationException("Custom Metal " + role + " buffer is smaller than logical payload.");
        }
    }

    private static State init() {
        try {
            Arena arena = Arena.ofShared();
            SymbolLookup lookup = resolveLookup(arena);
            Linker linker = Linker.nativeLinker();
            MethodHandle availableFn = optionalHandle(
                    linker,
                    lookup,
                    "synaptik_apple_mps_available",
                    FunctionDescriptor.of(JAVA_INT)
            );
            MethodHandle unavailableReasonFn = optionalHandle(
                    linker,
                    lookup,
                    "synaptik_apple_mps_unavailable_reason",
                    FunctionDescriptor.of(ADDRESS)
            );
            MethodHandle reluF32BufferFn = optionalHandle(
                    linker,
                    lookup,
                    "synaptik_apple_mps_custom_relu_f32_buffer",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_LONG, JAVA_LONG, JAVA_LONG)
            );
            if (availableFn == null) {
                return State.unavailable(arena, "Metal MPS availability symbol unavailable.");
            }
            int available = (int) availableFn.invokeExact();
            if (available == 0) {
                return State.unavailable(arena, unavailableReason(unavailableReasonFn));
            }
            if (reluF32BufferFn == null) {
                return State.unavailable(arena, "custom Metal RELU kernel symbol unavailable.");
            }
            return new State(true, "", arena, reluF32BufferFn);
        } catch (Throwable t) {
            return State.unavailable(null, t.getClass().getSimpleName() + ": " + safeMessage(t));
        }
    }

    private static SymbolLookup resolveLookup(Arena arena) {
        return MetalNativeLibraryResolver.resolveLookup(arena);
    }

    private static MethodHandle optionalHandle(
            Linker linker,
            SymbolLookup lookup,
            String symbol,
            FunctionDescriptor descriptor
    ) {
        MemorySegment segment = lookup.find(symbol).orElse(null);
        return segment == null ? null : linker.downcallHandle(segment, descriptor);
    }

    private static String unavailableReason(MethodHandle unavailableReasonFn) {
        if (unavailableReasonFn == null) {
            return "Metal MPS unavailable.";
        }
        try {
            MemorySegment ptr = (MemorySegment) unavailableReasonFn.invokeExact();
            if (ptr == null || ptr.equals(MemorySegment.NULL)) {
                return "Metal MPS unavailable.";
            }
            return ptr.reinterpret(Long.MAX_VALUE).getString(0, StandardCharsets.UTF_8);
        } catch (Throwable ignored) {
            return "Metal MPS unavailable.";
        }
    }

    private static String safeMessage(Throwable t) {
        String message = t.getMessage();
        return message == null || message.isBlank() ? "<no-message>" : message;
    }

    private record State(
            boolean available,
            String reason,
            @SuppressWarnings("unused") Arena arenaRef,
            MethodHandle reluF32BufferFn
    ) {
        private static State unavailable(Arena arenaRef, String reason) {
            return new State(false, reason == null || reason.isBlank()
                    ? "custom Metal RELU kernel unavailable"
                    : reason, arenaRef, null);
        }
    }
}
