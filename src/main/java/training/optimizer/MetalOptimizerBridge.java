package training.optimizer;

import backend.accelerator.buffer.AcceleratorBufferLayoutClass;
import backend.metal.bridge.MetalMpsBridgeContext;
import backend.metal.bridge.MetalNativeLibraryResolver;
import backend.metal.buffer.MetalBufferAccess;
import backend.metal.buffer.MetalBufferBinding;
import tensor.DataType;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * FFM bridge for optimizer-owned Metal parameter update kernels.
 */
final class MetalOptimizerBridge {
    private static final MetalOptimizerBridge INSTANCE = new MetalOptimizerBridge();
    private final State state = init();

    static MetalOptimizerBridge get() {
        return INSTANCE;
    }

    boolean available() {
        return state.available;
    }

    String unavailableReason() {
        return state.reason;
    }

    void sgdF32(
            MetalMpsBridgeContext context,
            MetalBufferBinding parameter,
            MetalBufferBinding gradient,
            MetalBufferBinding output,
            float learningRate
    ) {
        requireAvailable();
        validateContext(context);
        validateDenseF32(parameter, "parameter", true);
        validateDenseF32(gradient, "gradient", true);
        validateDenseF32(output, "output", false);
        long elements = commonElementCount(parameter, gradient, output);
        try {
            int status = (int) state.sgdF32Fn.invokeExact(
                    context.handle(),
                    parameter.handle().nativeHandle(),
                    gradient.handle().nativeHandle(),
                    output.handle().nativeHandle(),
                    learningRate,
                    elements,
                    parameter.logicalByteLength(),
                    gradient.logicalByteLength(),
                    output.logicalByteLength()
            );
            if (status != 0) {
                throw new UnsupportedOperationException("Metal SGD optimizer kernel returned non-zero status: " + status);
            }
        } catch (Throwable t) {
            throw new UnsupportedOperationException("Metal SGD optimizer kernel failed: " + safeMessage(t), t);
        }
    }

    void adamF32(
            MetalMpsBridgeContext context,
            MetalBufferBinding parameter,
            MetalBufferBinding gradient,
            MetalBufferBinding firstMoment,
            MetalBufferBinding secondMoment,
            MetalBufferBinding output,
            float learningRate,
            float beta1,
            float beta2,
            float epsilon,
            int step
    ) {
        requireAvailable();
        validateContext(context);
        validateDenseF32(parameter, "parameter", true);
        validateDenseF32(gradient, "gradient", true);
        validateDenseF32(firstMoment, "first moment", false);
        validateDenseF32(secondMoment, "second moment", false);
        validateDenseF32(output, "output", false);
        long elements = commonElementCount(parameter, gradient, firstMoment, secondMoment, output);
        try {
            int status = (int) state.adamF32Fn.invokeExact(
                    context.handle(),
                    parameter.handle().nativeHandle(),
                    gradient.handle().nativeHandle(),
                    firstMoment.handle().nativeHandle(),
                    secondMoment.handle().nativeHandle(),
                    output.handle().nativeHandle(),
                    learningRate,
                    beta1,
                    beta2,
                    epsilon,
                    step,
                    elements,
                    parameter.logicalByteLength(),
                    gradient.logicalByteLength(),
                    firstMoment.logicalByteLength(),
                    secondMoment.logicalByteLength(),
                    output.logicalByteLength()
            );
            if (status != 0) {
                throw new UnsupportedOperationException("Metal Adam optimizer kernel returned non-zero status: " + status);
            }
        } catch (Throwable t) {
            throw new UnsupportedOperationException("Metal Adam optimizer kernel failed: " + safeMessage(t), t);
        }
    }

    private void requireAvailable() {
        if (!state.available) {
            throw new UnsupportedOperationException(state.reason);
        }
    }

    private static void validateContext(MetalMpsBridgeContext context) {
        if (context == null || !context.available()) {
            throw new UnsupportedOperationException(context == null ? "Missing Metal bridge context." : context.reason());
        }
    }

    private static void validateDenseF32(MetalBufferBinding binding, String role, boolean readOnly) {
        if (binding == null || !binding.available()) {
            throw new UnsupportedOperationException("Metal optimizer " + role + " binding is unavailable.");
        }
        if (binding.layout().dataType() != DataType.FLOAT32) {
            throw new UnsupportedOperationException("Metal optimizer " + role + " requires FLOAT32, got "
                    + binding.layout().dataType() + ".");
        }
        if (binding.layout().layoutClass() != AcceleratorBufferLayoutClass.DENSE_CONTIGUOUS
                || binding.layout().storageOffset() != 0) {
            throw new UnsupportedOperationException("Metal optimizer " + role + " requires dense contiguous layout, got "
                    + binding.layout().describe() + ".");
        }
        if (readOnly && binding.access() != MetalBufferAccess.READ && binding.access() != MetalBufferAccess.READ_WRITE) {
            throw new UnsupportedOperationException("Metal optimizer " + role + " binding is not readable.");
        }
        if (!readOnly && binding.access() != MetalBufferAccess.WRITE && binding.access() != MetalBufferAccess.READ_WRITE) {
            throw new UnsupportedOperationException("Metal optimizer " + role + " binding is not writable.");
        }
    }

    private static long commonElementCount(MetalBufferBinding first, MetalBufferBinding... rest) {
        long elements = first.layout().logicalElementCount();
        for (MetalBufferBinding binding : rest) {
            if (binding.layout().logicalElementCount() != elements) {
                throw new UnsupportedOperationException("Metal optimizer bindings have mismatched element counts.");
            }
        }
        return elements;
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
            MethodHandle sgdF32Fn = optionalHandle(
                    linker,
                    lookup,
                    "synaptik_apple_mps_optimizer_sgd_f32_buffer",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, JAVA_FLOAT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_LONG)
            );
            MethodHandle adamF32Fn = optionalHandle(
                    linker,
                    lookup,
                    "synaptik_apple_mps_optimizer_adam_f32_buffer",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS,
                            JAVA_FLOAT, JAVA_FLOAT, JAVA_FLOAT, JAVA_FLOAT, JAVA_INT, JAVA_LONG,
                            JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_LONG)
            );
            if (availableFn == null) {
                return State.unavailable(arena, "Metal MPS availability symbol unavailable.");
            }
            int available = (int) availableFn.invokeExact();
            if (available == 0) {
                return State.unavailable(arena, unavailableReason(unavailableReasonFn));
            }
            if (sgdF32Fn == null || adamF32Fn == null) {
                return State.unavailable(arena, "Metal optimizer kernel symbols unavailable.");
            }
            return new State(true, "", arena, sgdF32Fn, adamF32Fn);
        } catch (Throwable t) {
            return State.unavailable(null, t.getClass().getSimpleName() + ": " + safeMessage(t));
        }
    }

    private static SymbolLookup resolveLookup(Arena arena) {
        return MetalNativeLibraryResolver.resolveLookup(arena);
    }

    private static MethodHandle optionalHandle(Linker linker, SymbolLookup lookup, String symbol, FunctionDescriptor descriptor) {
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
            MethodHandle sgdF32Fn,
            MethodHandle adamF32Fn
    ) {
        private static State unavailable(Arena arenaRef, String reason) {
            return new State(false, reason == null || reason.isBlank()
                    ? "Metal optimizer kernels unavailable"
                    : reason, arenaRef, null, null);
        }
    }
}
