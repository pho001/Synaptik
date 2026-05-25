package backend.cpu.nativecpu;

import tensor.dtype.TensorDTypeOps;
import tensor.storage.NativeBFloat16Storage;
import tensor.storage.NativeFloat32Storage;

import java.lang.foreign.MemorySegment;
import java.util.Objects;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

/**
 * Dense contiguous BF16 native CPU copy and conversion kernels.
 */
final class NativeBFloat16Kernels {
    private NativeBFloat16Kernels() {
    }

    static void copy(NativeBFloat16Storage input, NativeBFloat16Storage output, int size) {
        Objects.requireNonNull(input, "input cannot be null");
        Objects.requireNonNull(output, "output cannot be null");
        validateSize(input.getSize(), size, "input");
        validateSize(output.getSize(), size, "output");
        MemorySegment.copy(input.segment(), JAVA_SHORT, 0L, output.segment(), JAVA_SHORT, 0L, size);
        output.markModified();
    }

    static void toFloat32(NativeBFloat16Storage input, NativeFloat32Storage output, int size) {
        Objects.requireNonNull(input, "input cannot be null");
        Objects.requireNonNull(output, "output cannot be null");
        validateSize(input.getSize(), size, "input");
        validateSize(output.getSize(), size, "output");

        var inputSegment = input.segment();
        var outputSegment = output.segment();
        for (int i = 0; i < size; i++) {
            short bits = inputSegment.get(JAVA_SHORT, (long) i * Short.BYTES);
            outputSegment.set(JAVA_FLOAT, (long) i * Float.BYTES, TensorDTypeOps.fromBFloat16Bits(bits));
        }
        output.markModified();
    }

    static void fromFloat32(NativeFloat32Storage input, NativeBFloat16Storage output, int size) {
        Objects.requireNonNull(input, "input cannot be null");
        Objects.requireNonNull(output, "output cannot be null");
        validateSize(input.getSize(), size, "input");
        validateSize(output.getSize(), size, "output");

        var inputSegment = input.segment();
        var outputSegment = output.segment();
        for (int i = 0; i < size; i++) {
            float value = inputSegment.get(JAVA_FLOAT, (long) i * Float.BYTES);
            outputSegment.set(JAVA_SHORT, (long) i * Short.BYTES, TensorDTypeOps.toBFloat16Bits(value));
        }
        output.markModified();
    }

    private static void validateSize(int storageSize, int requestedSize, String label) {
        if (requestedSize < 0) {
            throw new IllegalArgumentException("size cannot be negative: " + requestedSize);
        }
        if (requestedSize > storageSize) {
            throw new IllegalArgumentException(label + " native storage too small. requested="
                    + requestedSize + ", storageSize=" + storageSize);
        }
    }
}
