package backend.cpu.nativecpu;

import runtime.memory.nativecpu.NativeCpuStorageFactory;

import tensor.dtype.TensorDTypeOps;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.storage.NativeBFloat16Storage;
import tensor.storage.NativeFloat32Storage;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NativeBFloat16KernelsTest {
    @Test
    void copyPreservesRawBfloat16Bits() {
        short[] bits = new short[]{
                (short) 0x0000,
                (short) 0x8000,
                (short) 0x3f80,
                (short) 0x7f80,
                (short) 0xff80,
                (short) 0x7fc1,
                (short) 0x0001,
                (short) 0x8001
        };
        NativeBFloat16Storage input = bf16(bits.length, "bf16-copy-in");
        NativeBFloat16Storage output = bf16(bits.length, "bf16-copy-out");

        try {
            write(input, bits);
            NativeBFloat16Kernels.copy(input, output, bits.length);

            assertArrayEquals(bits, read(output, bits.length));
        } finally {
            input.close();
            output.close();
        }
    }

    @Test
    void fromFloat32UsesCanonicalBfloat16RoundToNearestEven() {
        int[] floatBits = new int[]{
                0x3f807fff,
                0x3f808000,
                0x3f808001,
                0x3f818000,
                0x7fa12345
        };
        short[] expected = new short[]{
                (short) 0x3f80,
                (short) 0x3f80,
                (short) 0x3f81,
                (short) 0x3f82,
                (short) 0x7fc0
        };
        NativeFloat32Storage input = f32(floatBits.length, "bf16-from-f32-in");
        NativeBFloat16Storage output = bf16(floatBits.length, "bf16-from-f32-out");

        try {
            for (int i = 0; i < floatBits.length; i++) {
                input.setFloat32At(i, Float.intBitsToFloat(floatBits[i]));
            }
            NativeBFloat16Kernels.fromFloat32(input, output, floatBits.length);

            assertArrayEquals(expected, read(output, expected.length));
        } finally {
            input.close();
            output.close();
        }
    }

    @Test
    void toFloat32ReconstructsBfloat16BitsExactly() {
        short[] bits = new short[]{
                (short) 0x3f80,
                (short) 0x8000,
                (short) 0x7f80,
                (short) 0xff80,
                (short) 0x0001,
                (short) 0x7fc1
        };
        NativeBFloat16Storage input = bf16(bits.length, "bf16-to-f32-in");
        NativeFloat32Storage output = f32(bits.length, "bf16-to-f32-out");

        try {
            write(input, bits);
            NativeBFloat16Kernels.toFloat32(input, output, bits.length);

            for (int i = 0; i < bits.length; i++) {
                assertEquals(
                        Float.floatToRawIntBits(TensorDTypeOps.fromBFloat16Bits(bits[i])),
                        Float.floatToRawIntBits(output.getFloat32At(i))
                );
            }
        } finally {
            input.close();
            output.close();
        }
    }

    @Test
    void kernelsRejectSizePastStorageBounds() {
        NativeBFloat16Storage input = bf16(1, "bf16-size-in");
        NativeBFloat16Storage output = bf16(1, "bf16-size-out");

        try {
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class,
                    () -> NativeBFloat16Kernels.copy(input, output, 2)
            );
            assertTrue(failure.getMessage().contains("storage too small"));
        } finally {
            input.close();
            output.close();
        }
    }

    private static NativeBFloat16Storage bf16(int elements, String label) {
        return (NativeBFloat16Storage) new NativeCpuStorageFactory().allocate(DataType.BFLOAT16, elements, label);
    }

    private static NativeFloat32Storage f32(int elements, String label) {
        return (NativeFloat32Storage) new NativeCpuStorageFactory().allocate(DataType.FLOAT32, elements, label);
    }

    private static void write(NativeBFloat16Storage storage, short[] bits) {
        for (int i = 0; i < bits.length; i++) {
            storage.setBFloat16BitsAt(i, bits[i]);
        }
    }

    private static short[] read(NativeBFloat16Storage storage, int size) {
        short[] bits = new short[size];
        for (int i = 0; i < size; i++) {
            bits[i] = storage.getBFloat16BitsAt(i);
        }
        return bits;
    }
}
