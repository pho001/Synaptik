package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Independent primitive Java design oracle for ordinary low folds. The dense rank-one loop
 * uses int induction; the general axis and image loops use long geometry and canonical logical
 * contributor order. Only coordinate arrays allocated at entry are used, with no boxed values,
 * callbacks, dispatch, or allocation inside output/contributor loops. Segment variants use the
 * same direct checked native-order unaligned short accesses required by generated kernels.
 * Exact decode uses JDK bit/half conversion; final BF16 rounding uses integer
 * round-to-nearest, ties-to-even (RNE) arithmetic only at the final store. Callers supply
 * already-admitted geometry and sufficient disjoint spans; these test oracles do not validate
 * preparation or own the carriers. Their arithmetic core is the design/review reference.
 * The general loops retain all-input matching, not an optimal inverse gather. Image loops hoist
 * q/kernel decoding across positions whereas the retained generated generic fallback does not;
 * that known geometry-loop difference is not evidence of whole-loop or performance parity.
 */
final class CpuLowFoldCleanJavaOracle {
    /** Prevents instances of this stateless test-only oracle. */
    private CpuLowFoldCleanJavaOracle() {}

    /**
     * Dense array specialization with canonical logical input scanning and one final store.
     * Uses exact BFLOAT16 decode, sequential FLOAT32 addition and one final low conversion.
     * @param input non-null borrowed carrier of exact stored low values; not modified
     * @param output non-null distinct writable carrier; only the owned range is written
     * @param inputBase nonnegative input offset in short elements
     * @param outputBase nonnegative output offset in short elements
     * @param inputCount nonnegative logical input count, consistent with admitted geometry
     * @param window positive final-window extent
     * @param step positive window-position step
     * @param start inclusive flattened output ordinal within the admitted domain
     * @param end exclusive flattened output ordinal; at least start and within the admitted domain
     */
    static void axisBfloatInt(short[] input, short[] output, int inputBase, int outputBase,
            int inputCount, int window, int step, int start, int end) {
        for (int cell = start; cell < end; cell++) {
            float sum = +0.0f;
            for (int ordinal = 0, within = 0, target = 0; ordinal < inputCount; ordinal++, within++, target++) {
                if (target == cell) sum += Float.intBitsToFloat(input[inputBase + ordinal] << 16);
                if (within + 1 == window) { within = -1; target += step - window; }
            }
            int bits = Float.floatToRawIntBits(sum);
            int raw = (bits & 0x7fffffff) > 0x7f800000 ? 0x7fc0
                    : (bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16;
            output[outputBase + cell] = (short) raw;
        }
    }

    /**
     * General long-address axis specialization; logical duplicates remain separate contributions.
     * Uses exact BFLOAT16 decode, sequential FLOAT32 addition and one final low conversion.
     * @param input non-null borrowed carrier of exact stored low values; not modified
     * @param output non-null distinct writable carrier; only the owned range is written
     * @param inputBase nonnegative input offset in short elements
     * @param outputBase nonnegative output offset in short elements
     * @param inputExtents non-null admitted input extents; read only
     * @param inputStrides non-null admitted nonnegative input element strides, including repeated reads
     * @param outputExtents non-null admitted output extents; read only
     * @param outputStrides non-null admitted injective output element strides; read only
     * @param axis selected zero-based output axis; the input window dimension is last
     * @param step positive window-position step
     * @param start inclusive flattened output ordinal within the admitted domain
     * @param end exclusive flattened output ordinal; at least start and within the admitted domain
     */
    static void axisBfloatArrays(short[] input, short[] output,
            long inputBase, long outputBase, long[] inputExtents, long[] inputStrides,
            long[] outputExtents, long[] outputStrides, int axis, long step, long start, long end) {
        long inputCount = 1;
        for (long extent : inputExtents) inputCount *= extent;
        long[] coordinate = new long[inputExtents.length];
        long[] target = new long[outputExtents.length];
        for (long cell = start; cell < end; cell++) {
            long remainder = cell, outAddress = outputBase;
            for (int a = target.length - 1; a >= 0; a--) {
                target[a] = remainder % outputExtents[a]; remainder /= outputExtents[a];
                outAddress += target[a] * outputStrides[a];
            }
            for (int a = 0; a < coordinate.length; a++) coordinate[a] = 0;
            float sum = +0.0f;
            for (long ordinal = 0; ordinal < inputCount; ordinal++) {
                boolean match = true;
                for (int a = 0; a < target.length; a++) {
                    long mapped = a == axis ? coordinate[a] * step + coordinate[coordinate.length - 1] : coordinate[a];
                    if (mapped != target[a]) match = false;
                }
                if (match) {
                    long address = inputBase;
                    for (int a = 0; a < coordinate.length; a++) address += coordinate[a] * inputStrides[a];
                    sum += Float.intBitsToFloat(input[(int) address] << 16);
                }
                for (int a = coordinate.length - 1; a >= 0; a--) {
                    if (++coordinate[a] < inputExtents[a]) break;
                    coordinate[a] = 0;
                }
            }
            int bits = Float.floatToRawIntBits(sum);
            int raw = (bits & 0x7fffffff) > 0x7f800000 ? 0x7fc0
                    : (bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16;
            output[(int) outAddress] = (short) raw;
        }
    }

    /**
     * General image specialization scans logical [N,Q,P], omitting padding and ceil tails.
     * Uses exact BFLOAT16 decode, sequential FLOAT32 addition and one final low conversion.
     * @param input non-null borrowed carrier of exact stored low values; not modified
     * @param output non-null distinct writable carrier; only the owned range is written
     * @param inputBase nonnegative input offset in short elements
     * @param outputBase nonnegative output offset in short elements
     * @param inputExtents non-null admitted input extents; read only
     * @param inputStrides non-null admitted nonnegative input element strides, including repeated reads
     * @param outputExtents non-null admitted output extents; read only
     * @param outputStrides non-null admitted injective output element strides; read only
     * @param kh positive kernel height
     * @param kw positive kernel width
     * @param sh positive height stride
     * @param sw positive width stride
     * @param ph nonnegative symmetric height padding
     * @param pw nonnegative symmetric width padding
     * @param dh positive height dilation
     * @param dw positive width dilation
     * @param oh nonnegative admitted grid height; retained for explicit geometry correspondence
     * @param ow nonnegative admitted grid width; positive whenever input positions are visited
     * @param start inclusive flattened output ordinal within the admitted domain
     * @param end exclusive flattened output ordinal; at least start and within the admitted domain
     */
    static void imageBfloatArrays(short[] input, short[] output,
            long inputBase, long outputBase, long[] inputExtents, long[] inputStrides,
            long[] outputExtents, long[] outputStrides, long kh, long kw, long sh, long sw,
            long ph, long pw, long dh, long dw, long oh, long ow, long start, long end) {
        for (long cell = start; cell < end; cell++) {
            long x = cell % outputExtents[3], remainder = cell / outputExtents[3];
            long y = remainder % outputExtents[2]; remainder /= outputExtents[2];
            long c = remainder % outputExtents[1], n = remainder / outputExtents[1];
            long outAddress = outputBase + n * outputStrides[0] + c * outputStrides[1]
                    + y * outputStrides[2] + x * outputStrides[3];
            float sum = +0.0f;
            for (long batch = 0; batch < inputExtents[0]; batch++) {
                for (long q = 0; q < inputExtents[1]; q++) {
                    long channel = q / (kh * kw), k = q % (kh * kw), ky = k / kw, kx = k % kw;
                    for (long p = 0; p < inputExtents[2]; p++) {
                        long mappedY = p / ow * sh - ph + ky * dh;
                        long mappedX = p % ow * sw - pw + kx * dw;
                        if (batch == n && channel == c && mappedY == y && mappedX == x) {
                            long address = inputBase + batch * inputStrides[0] + q * inputStrides[1] + p * inputStrides[2];
                            sum += Float.intBitsToFloat(input[(int) address] << 16);
                        }
                    }
                }
            }
            int bits = Float.floatToRawIntBits(sum);
            int raw = (bits & 0x7fffffff) > 0x7f800000 ? 0x7fc0
                    : (bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16;
            output[(int) outAddress] = (short) raw;
        }
    }

    /**
     * General long-address axis specialization; logical duplicates remain separate contributions.
     * Uses exact BFLOAT16 decode, sequential FLOAT32 addition and one final low conversion.
     * @param input non-null borrowed carrier of exact stored low values; not modified
     * @param output non-null distinct writable carrier; only the owned range is written
     * @param inputBase nonnegative input offset in short elements
     * @param outputBase nonnegative output offset in short elements
     * @param inputExtents non-null admitted input extents; read only
     * @param inputStrides non-null admitted nonnegative input element strides, including repeated reads
     * @param outputExtents non-null admitted output extents; read only
     * @param outputStrides non-null admitted injective output element strides; read only
     * @param axis selected zero-based output axis; the input window dimension is last
     * @param step positive window-position step
     * @param start inclusive flattened output ordinal within the admitted domain
     * @param end exclusive flattened output ordinal; at least start and within the admitted domain
     */
    static void axisBfloatSegments(MemorySegment input, MemorySegment output,
            long inputBase, long outputBase, long[] inputExtents, long[] inputStrides,
            long[] outputExtents, long[] outputStrides, int axis, long step, long start, long end) {
        long inputCount = 1;
        for (long extent : inputExtents) inputCount *= extent;
        long[] coordinate = new long[inputExtents.length];
        long[] target = new long[outputExtents.length];
        for (long cell = start; cell < end; cell++) {
            long remainder = cell, outAddress = outputBase;
            for (int a = target.length - 1; a >= 0; a--) {
                target[a] = remainder % outputExtents[a]; remainder /= outputExtents[a];
                outAddress += target[a] * outputStrides[a];
            }
            for (int a = 0; a < coordinate.length; a++) coordinate[a] = 0;
            float sum = +0.0f;
            for (long ordinal = 0; ordinal < inputCount; ordinal++) {
                boolean match = true;
                for (int a = 0; a < target.length; a++) {
                    long mapped = a == axis ? coordinate[a] * step + coordinate[coordinate.length - 1] : coordinate[a];
                    if (mapped != target[a]) match = false;
                }
                if (match) {
                    long address = inputBase;
                    for (int a = 0; a < coordinate.length; a++) address += coordinate[a] * inputStrides[a];
                    sum += Float.intBitsToFloat(input.get(ValueLayout.JAVA_SHORT_UNALIGNED, address * 2) << 16);
                }
                for (int a = coordinate.length - 1; a >= 0; a--) {
                    if (++coordinate[a] < inputExtents[a]) break;
                    coordinate[a] = 0;
                }
            }
            int bits = Float.floatToRawIntBits(sum);
            int raw = (bits & 0x7fffffff) > 0x7f800000 ? 0x7fc0
                    : (bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16;
            output.set(ValueLayout.JAVA_SHORT_UNALIGNED, outAddress * 2, (short) raw);
        }
    }

    /**
     * General image specialization scans logical [N,Q,P], omitting padding and ceil tails.
     * Uses exact BFLOAT16 decode, sequential FLOAT32 addition and one final low conversion.
     * @param input non-null borrowed carrier of exact stored low values; not modified
     * @param output non-null distinct writable carrier; only the owned range is written
     * @param inputBase nonnegative input offset in short elements
     * @param outputBase nonnegative output offset in short elements
     * @param inputExtents non-null admitted input extents; read only
     * @param inputStrides non-null admitted nonnegative input element strides, including repeated reads
     * @param outputExtents non-null admitted output extents; read only
     * @param outputStrides non-null admitted injective output element strides; read only
     * @param kh positive kernel height
     * @param kw positive kernel width
     * @param sh positive height stride
     * @param sw positive width stride
     * @param ph nonnegative symmetric height padding
     * @param pw nonnegative symmetric width padding
     * @param dh positive height dilation
     * @param dw positive width dilation
     * @param oh nonnegative admitted grid height; retained for explicit geometry correspondence
     * @param ow nonnegative admitted grid width; positive whenever input positions are visited
     * @param start inclusive flattened output ordinal within the admitted domain
     * @param end exclusive flattened output ordinal; at least start and within the admitted domain
     */
    static void imageBfloatSegments(MemorySegment input, MemorySegment output,
            long inputBase, long outputBase, long[] inputExtents, long[] inputStrides,
            long[] outputExtents, long[] outputStrides, long kh, long kw, long sh, long sw,
            long ph, long pw, long dh, long dw, long oh, long ow, long start, long end) {
        for (long cell = start; cell < end; cell++) {
            long x = cell % outputExtents[3], remainder = cell / outputExtents[3];
            long y = remainder % outputExtents[2]; remainder /= outputExtents[2];
            long c = remainder % outputExtents[1], n = remainder / outputExtents[1];
            long outAddress = outputBase + n * outputStrides[0] + c * outputStrides[1]
                    + y * outputStrides[2] + x * outputStrides[3];
            float sum = +0.0f;
            for (long batch = 0; batch < inputExtents[0]; batch++) {
                for (long q = 0; q < inputExtents[1]; q++) {
                    long channel = q / (kh * kw), k = q % (kh * kw), ky = k / kw, kx = k % kw;
                    for (long p = 0; p < inputExtents[2]; p++) {
                        long mappedY = p / ow * sh - ph + ky * dh;
                        long mappedX = p % ow * sw - pw + kx * dw;
                        if (batch == n && channel == c && mappedY == y && mappedX == x) {
                            long address = inputBase + batch * inputStrides[0] + q * inputStrides[1] + p * inputStrides[2];
                            sum += Float.intBitsToFloat(input.get(ValueLayout.JAVA_SHORT_UNALIGNED, address * 2) << 16);
                        }
                    }
                }
            }
            int bits = Float.floatToRawIntBits(sum);
            int raw = (bits & 0x7fffffff) > 0x7f800000 ? 0x7fc0
                    : (bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16;
            output.set(ValueLayout.JAVA_SHORT_UNALIGNED, outAddress * 2, (short) raw);
        }
    }

    /**
     * Dense array specialization with canonical logical input scanning and one final store.
     * Uses exact FLOAT16 decode, sequential FLOAT32 addition and one final low conversion.
     * @param input non-null borrowed carrier of exact stored low values; not modified
     * @param output non-null distinct writable carrier; only the owned range is written
     * @param inputBase nonnegative input offset in short elements
     * @param outputBase nonnegative output offset in short elements
     * @param inputCount nonnegative logical input count, consistent with admitted geometry
     * @param window positive final-window extent
     * @param step positive window-position step
     * @param start inclusive flattened output ordinal within the admitted domain
     * @param end exclusive flattened output ordinal; at least start and within the admitted domain
     */
    static void axisHalfInt(short[] input, short[] output, int inputBase, int outputBase,
            int inputCount, int window, int step, int start, int end) {
        for (int cell = start; cell < end; cell++) {
            float sum = +0.0f;
            for (int ordinal = 0, within = 0, target = 0; ordinal < inputCount; ordinal++, within++, target++) {
                if (target == cell) sum += Float.float16ToFloat(input[inputBase + ordinal]);
                if (within + 1 == window) { within = -1; target += step - window; }
            }
            float value = sum;
            output[outputBase + cell] = Float.floatToFloat16(value);
        }
    }

    /**
     * General long-address axis specialization; logical duplicates remain separate contributions.
     * Uses exact FLOAT16 decode, sequential FLOAT32 addition and one final low conversion.
     * @param input non-null borrowed carrier of exact stored low values; not modified
     * @param output non-null distinct writable carrier; only the owned range is written
     * @param inputBase nonnegative input offset in short elements
     * @param outputBase nonnegative output offset in short elements
     * @param inputExtents non-null admitted input extents; read only
     * @param inputStrides non-null admitted nonnegative input element strides, including repeated reads
     * @param outputExtents non-null admitted output extents; read only
     * @param outputStrides non-null admitted injective output element strides; read only
     * @param axis selected zero-based output axis; the input window dimension is last
     * @param step positive window-position step
     * @param start inclusive flattened output ordinal within the admitted domain
     * @param end exclusive flattened output ordinal; at least start and within the admitted domain
     */
    static void axisHalfArrays(short[] input, short[] output,
            long inputBase, long outputBase, long[] inputExtents, long[] inputStrides,
            long[] outputExtents, long[] outputStrides, int axis, long step, long start, long end) {
        long inputCount = 1;
        for (long extent : inputExtents) inputCount *= extent;
        long[] coordinate = new long[inputExtents.length];
        long[] target = new long[outputExtents.length];
        for (long cell = start; cell < end; cell++) {
            long remainder = cell, outAddress = outputBase;
            for (int a = target.length - 1; a >= 0; a--) {
                target[a] = remainder % outputExtents[a]; remainder /= outputExtents[a];
                outAddress += target[a] * outputStrides[a];
            }
            for (int a = 0; a < coordinate.length; a++) coordinate[a] = 0;
            float sum = +0.0f;
            for (long ordinal = 0; ordinal < inputCount; ordinal++) {
                boolean match = true;
                for (int a = 0; a < target.length; a++) {
                    long mapped = a == axis ? coordinate[a] * step + coordinate[coordinate.length - 1] : coordinate[a];
                    if (mapped != target[a]) match = false;
                }
                if (match) {
                    long address = inputBase;
                    for (int a = 0; a < coordinate.length; a++) address += coordinate[a] * inputStrides[a];
                    sum += Float.float16ToFloat(input[(int) address]);
                }
                for (int a = coordinate.length - 1; a >= 0; a--) {
                    if (++coordinate[a] < inputExtents[a]) break;
                    coordinate[a] = 0;
                }
            }
            float value = sum;
            output[(int) outAddress] = Float.floatToFloat16(value);
        }
    }

    /**
     * General image specialization scans logical [N,Q,P], omitting padding and ceil tails.
     * Uses exact FLOAT16 decode, sequential FLOAT32 addition and one final low conversion.
     * @param input non-null borrowed carrier of exact stored low values; not modified
     * @param output non-null distinct writable carrier; only the owned range is written
     * @param inputBase nonnegative input offset in short elements
     * @param outputBase nonnegative output offset in short elements
     * @param inputExtents non-null admitted input extents; read only
     * @param inputStrides non-null admitted nonnegative input element strides, including repeated reads
     * @param outputExtents non-null admitted output extents; read only
     * @param outputStrides non-null admitted injective output element strides; read only
     * @param kh positive kernel height
     * @param kw positive kernel width
     * @param sh positive height stride
     * @param sw positive width stride
     * @param ph nonnegative symmetric height padding
     * @param pw nonnegative symmetric width padding
     * @param dh positive height dilation
     * @param dw positive width dilation
     * @param oh nonnegative admitted grid height; retained for explicit geometry correspondence
     * @param ow nonnegative admitted grid width; positive whenever input positions are visited
     * @param start inclusive flattened output ordinal within the admitted domain
     * @param end exclusive flattened output ordinal; at least start and within the admitted domain
     */
    static void imageHalfArrays(short[] input, short[] output,
            long inputBase, long outputBase, long[] inputExtents, long[] inputStrides,
            long[] outputExtents, long[] outputStrides, long kh, long kw, long sh, long sw,
            long ph, long pw, long dh, long dw, long oh, long ow, long start, long end) {
        for (long cell = start; cell < end; cell++) {
            long x = cell % outputExtents[3], remainder = cell / outputExtents[3];
            long y = remainder % outputExtents[2]; remainder /= outputExtents[2];
            long c = remainder % outputExtents[1], n = remainder / outputExtents[1];
            long outAddress = outputBase + n * outputStrides[0] + c * outputStrides[1]
                    + y * outputStrides[2] + x * outputStrides[3];
            float sum = +0.0f;
            for (long batch = 0; batch < inputExtents[0]; batch++) {
                for (long q = 0; q < inputExtents[1]; q++) {
                    long channel = q / (kh * kw), k = q % (kh * kw), ky = k / kw, kx = k % kw;
                    for (long p = 0; p < inputExtents[2]; p++) {
                        long mappedY = p / ow * sh - ph + ky * dh;
                        long mappedX = p % ow * sw - pw + kx * dw;
                        if (batch == n && channel == c && mappedY == y && mappedX == x) {
                            long address = inputBase + batch * inputStrides[0] + q * inputStrides[1] + p * inputStrides[2];
                            sum += Float.float16ToFloat(input[(int) address]);
                        }
                    }
                }
            }
            float value = sum;
            output[(int) outAddress] = Float.floatToFloat16(value);
        }
    }

    /**
     * General long-address axis specialization; logical duplicates remain separate contributions.
     * Uses exact FLOAT16 decode, sequential FLOAT32 addition and one final low conversion.
     * @param input non-null borrowed carrier of exact stored low values; not modified
     * @param output non-null distinct writable carrier; only the owned range is written
     * @param inputBase nonnegative input offset in short elements
     * @param outputBase nonnegative output offset in short elements
     * @param inputExtents non-null admitted input extents; read only
     * @param inputStrides non-null admitted nonnegative input element strides, including repeated reads
     * @param outputExtents non-null admitted output extents; read only
     * @param outputStrides non-null admitted injective output element strides; read only
     * @param axis selected zero-based output axis; the input window dimension is last
     * @param step positive window-position step
     * @param start inclusive flattened output ordinal within the admitted domain
     * @param end exclusive flattened output ordinal; at least start and within the admitted domain
     */
    static void axisHalfSegments(MemorySegment input, MemorySegment output,
            long inputBase, long outputBase, long[] inputExtents, long[] inputStrides,
            long[] outputExtents, long[] outputStrides, int axis, long step, long start, long end) {
        long inputCount = 1;
        for (long extent : inputExtents) inputCount *= extent;
        long[] coordinate = new long[inputExtents.length];
        long[] target = new long[outputExtents.length];
        for (long cell = start; cell < end; cell++) {
            long remainder = cell, outAddress = outputBase;
            for (int a = target.length - 1; a >= 0; a--) {
                target[a] = remainder % outputExtents[a]; remainder /= outputExtents[a];
                outAddress += target[a] * outputStrides[a];
            }
            for (int a = 0; a < coordinate.length; a++) coordinate[a] = 0;
            float sum = +0.0f;
            for (long ordinal = 0; ordinal < inputCount; ordinal++) {
                boolean match = true;
                for (int a = 0; a < target.length; a++) {
                    long mapped = a == axis ? coordinate[a] * step + coordinate[coordinate.length - 1] : coordinate[a];
                    if (mapped != target[a]) match = false;
                }
                if (match) {
                    long address = inputBase;
                    for (int a = 0; a < coordinate.length; a++) address += coordinate[a] * inputStrides[a];
                    sum += Float.float16ToFloat(input.get(ValueLayout.JAVA_SHORT_UNALIGNED, address * 2));
                }
                for (int a = coordinate.length - 1; a >= 0; a--) {
                    if (++coordinate[a] < inputExtents[a]) break;
                    coordinate[a] = 0;
                }
            }
            float value = sum;
            output.set(ValueLayout.JAVA_SHORT_UNALIGNED, outAddress * 2, Float.floatToFloat16(value));
        }
    }

    /**
     * General image specialization scans logical [N,Q,P], omitting padding and ceil tails.
     * Uses exact FLOAT16 decode, sequential FLOAT32 addition and one final low conversion.
     * @param input non-null borrowed carrier of exact stored low values; not modified
     * @param output non-null distinct writable carrier; only the owned range is written
     * @param inputBase nonnegative input offset in short elements
     * @param outputBase nonnegative output offset in short elements
     * @param inputExtents non-null admitted input extents; read only
     * @param inputStrides non-null admitted nonnegative input element strides, including repeated reads
     * @param outputExtents non-null admitted output extents; read only
     * @param outputStrides non-null admitted injective output element strides; read only
     * @param kh positive kernel height
     * @param kw positive kernel width
     * @param sh positive height stride
     * @param sw positive width stride
     * @param ph nonnegative symmetric height padding
     * @param pw nonnegative symmetric width padding
     * @param dh positive height dilation
     * @param dw positive width dilation
     * @param oh nonnegative admitted grid height; retained for explicit geometry correspondence
     * @param ow nonnegative admitted grid width; positive whenever input positions are visited
     * @param start inclusive flattened output ordinal within the admitted domain
     * @param end exclusive flattened output ordinal; at least start and within the admitted domain
     */
    static void imageHalfSegments(MemorySegment input, MemorySegment output,
            long inputBase, long outputBase, long[] inputExtents, long[] inputStrides,
            long[] outputExtents, long[] outputStrides, long kh, long kw, long sh, long sw,
            long ph, long pw, long dh, long dw, long oh, long ow, long start, long end) {
        for (long cell = start; cell < end; cell++) {
            long x = cell % outputExtents[3], remainder = cell / outputExtents[3];
            long y = remainder % outputExtents[2]; remainder /= outputExtents[2];
            long c = remainder % outputExtents[1], n = remainder / outputExtents[1];
            long outAddress = outputBase + n * outputStrides[0] + c * outputStrides[1]
                    + y * outputStrides[2] + x * outputStrides[3];
            float sum = +0.0f;
            for (long batch = 0; batch < inputExtents[0]; batch++) {
                for (long q = 0; q < inputExtents[1]; q++) {
                    long channel = q / (kh * kw), k = q % (kh * kw), ky = k / kw, kx = k % kw;
                    for (long p = 0; p < inputExtents[2]; p++) {
                        long mappedY = p / ow * sh - ph + ky * dh;
                        long mappedX = p % ow * sw - pw + kx * dw;
                        if (batch == n && channel == c && mappedY == y && mappedX == x) {
                            long address = inputBase + batch * inputStrides[0] + q * inputStrides[1] + p * inputStrides[2];
                            sum += Float.float16ToFloat(input.get(ValueLayout.JAVA_SHORT_UNALIGNED, address * 2));
                        }
                    }
                }
            }
            float value = sum;
            output.set(ValueLayout.JAVA_SHORT_UNALIGNED, outAddress * 2, Float.floatToFloat16(value));
        }
    }
}

