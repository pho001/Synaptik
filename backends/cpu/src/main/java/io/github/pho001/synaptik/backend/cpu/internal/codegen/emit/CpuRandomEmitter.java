package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRandomIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.CodeBuilder;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDescs;
import java.lang.constant.MethodTypeDesc;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/** Emits exact allocation-free explicit-state initializer and dropout generated bridges. */
public final class CpuRandomEmitter {
    private static final ClassDesc OWNER = ClassDesc.of(CpuRandomEmitter.class.getName());
    private static final DataType[] TYPES = DataType.values();

    /** Creates a stateless emitter with no generator instance or run state. */
    public CpuRandomEmitter() { }

    /**
     * Emits one exact one- or five-boundary scalar random bridge.
     *
     * @param code non-null generated method body
     * @param specialization non-null scalar specialization
     * @param ir non-null encoded random IR containing baked initializer/probability bits
     * @throws NullPointerException if a required reference is null
     * @throws IllegalArgumentException if specialization or encoded identity is inconsistent
     */
    public void emit(CodeBuilder code, CpuKernelSpecialization specialization, CpuKernelIr ir) {
        Parsed parsed = parse(ir.familyIdentity());
        int count = specialization.carrierPattern().size();
        if (count != (parsed.family == CpuRandomIr.Family.INITIAL_STATE ? 1 : 5)
                || specialization.executionStrategy().compute()
                    != io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionStrategy.Compute.SCALAR
                || specialization.scratchParameter()) {
            throw new IllegalArgumentException("random specialization facts disagree");
        }
        for (int index = 0; index < 5; index++) {
            if (index < count) code.aload(index); else code.aconst_null();
        }
        int geometry = count;
        code.aload(geometry).lload(geometry + 1).lload(geometry + 3)
                .loadConstant(parsed.family.ordinal()).loadConstant(parsed.key)
                .loadConstant(parsed.counter).loadConstant(parsed.probabilityBits)
                .invokestatic(OWNER, "execute", MethodTypeDesc.of(ConstantDescs.CD_void,
                        ConstantDescs.CD_Object, ConstantDescs.CD_Object,
                        ConstantDescs.CD_Object, ConstantDescs.CD_Object,
                        ConstantDescs.CD_Object, ConstantDescs.CD_long.arrayType(),
                        ConstantDescs.CD_long, ConstantDescs.CD_long, ConstantDescs.CD_int,
                        ConstantDescs.CD_long, ConstantDescs.CD_long, ConstantDescs.CD_long));
    }

    /**
     * Applies the exact CPU-private SplitMix64 counter mapping.
     *
     * @param key raw unsigned key word
     * @param counter raw unsigned counter word
     * @param logicalIndex logical draw offset, added modulo {@code 2^64}
     * @return exact mapped raw word
     */
    public static long word(long key, long counter, long logicalIndex) {
        return wordFromOffset(keyOffset(key), counter, logicalIndex);
    }

    private static long keyOffset(long key) { return mix64(key + CpuRandomIr.KEY_BIAS); }

    private static long wordFromOffset(long keyOffset, long counter, long logicalIndex) {
        return mix64(counter + logicalIndex + keyOffset);
    }

    /**
     * Converts one mapped word to the exact top-53-bit binary64 uniform value.
     *
     * @param word raw mapped word
     * @return exact value in {@code [0,1)}
     */
    public static double uniform53(long word) { return (word >>> 11) * 0x1.0p-53; }

    private static long mix64(long value) {
        value = (value ^ (value >>> 30)) * CpuRandomIr.MIX_MULTIPLIER_1;
        value = (value ^ (value >>> 27)) * CpuRandomIr.MIX_MULTIPLIER_2;
        return value ^ (value >>> 31);
    }

    /**
     * Executes the generated entry bridge for one state prologue or dropout element range.
     *
     * <p>This method is public only for verified hidden-class linkage. The generated caller
     * supplies already-validated carriers and packed geometry. A {@code [0,0)} range writes the
     * initializer or next state exactly once; a non-empty dropout range writes only its output
     * and canonical BOOL-mask elements. It retains no generator state and does not own or close
     * any carrier.</p>
     *
     * @param value initializer output carrier or read-only dropout value carrier
     * @param state read-only dropout state carrier, or null for initialization
     * @param output writable dropout value carrier, or null for initialization
     * @param mask writable dropout BOOL carrier, or null for initialization
     * @param nextState writable dropout next-state carrier, or null for initialization
     * @param geometry non-null packed boundary geometry produced by the matching lowering
     * @param start inclusive logical dropout ordinal, or zero for the state prologue
     * @param end exclusive logical dropout ordinal, or zero for the state prologue
     * @param familyOrdinal ordinal of the baked {@link CpuRandomIr.Family}
     * @param keyBits baked raw initializer key bits; ignored for dropout
     * @param counterBits baked raw initializer counter bits; ignored for dropout
     * @param probabilityBits baked raw binary64 dropout probability bits; ignored for initialization
     * @throws NullPointerException if a required carrier or {@code geometry} is null
     * @throws IllegalArgumentException if an initializer is invoked for element work or packed
     *     state geometry is inconsistent
     * @throws ArithmeticException if a heap address does not fit an array index
     * @throws IndexOutOfBoundsException if packed geometry addresses outside a supplied carrier
     */
    public static void execute(Object value, Object state, Object output, Object mask,
            Object nextState, long[] geometry, long start, long end, int familyOrdinal,
            long keyBits, long counterBits, long probabilityBits) {
        boolean initializer = familyOrdinal == CpuRandomIr.Family.INITIAL_STATE.ordinal();
        if (start == 0 && end == 0) {
            if (initializer) {
                writeLong(value, stateAddress(geometry, 0, 0), keyBits);
                writeLong(value, stateAddress(geometry, 0, 1), counterBits);
            } else {
                long key = readLong(state, stateAddress(geometry, 1, 0));
                long counter = readLong(state, stateAddress(geometry, 1, 1));
                writeLong(nextState, stateAddress(geometry, 4, 0), key);
                writeLong(nextState, stateAddress(geometry, 4, 1), counter + geometry[2]);
            }
            return;
        }
        if (familyOrdinal != CpuRandomIr.Family.DROPOUT.ordinal()) throw new IllegalArgumentException(
                "initializer has no element range");
        DataType type = TYPES[(int) geometry[1]];
        double probability = Double.longBitsToDouble(probabilityBits);
        double denominator = 1.0d - probability;
        long key = readLong(state, stateAddress(geometry, 1, 0));
        long counter = readLong(state, stateAddress(geometry, 1, 1));
        long keyOffset = keyOffset(key);
        for (long logical = start; logical < end; logical++) {
            boolean keep = uniform53(wordFromOffset(keyOffset, counter, logical)) >= probability;
            writeByte(mask, logicalAddress(geometry, 3, logical), keep ? (byte) 1 : (byte) 0);
            long inputAddress = logicalAddress(geometry, 0, logical);
            long outputAddress = logicalAddress(geometry, 2, logical);
            if (type == DataType.FLOAT64) {
                double result = keep ? readDouble(value, inputAddress) / denominator : 0.0d;
                writeDouble(output, outputAddress, result);
            } else {
                float result = keep ? (float) (((double) readFloat(value, inputAddress)) / denominator)
                        : 0.0f;
                writeFloat(output, outputAddress, result);
            }
        }
    }

    private static long logicalAddress(long[] geometry, int boundary, long logical) {
        int position = layoutPosition(geometry, boundary);
        int rank = (int) geometry[position++]; long address = geometry[position++];
        int extents = position, strides = position + rank;
        long remainder = logical;
        for (int axis = rank - 1; axis >= 0; axis--) {
            long extent = geometry[extents + axis];
            long coordinate = extent == 0 ? 0 : remainder % extent;
            if (extent != 0) remainder /= extent;
            address += coordinate * geometry[strides + axis];
        }
        return address;
    }

    private static long stateAddress(long[] geometry, int boundary, int word) {
        int position = layoutPosition(geometry, boundary);
        int rank = (int) geometry[position++]; long address = geometry[position++];
        if (rank != 1 || geometry[position] != 2) throw new IllegalArgumentException(
                "random state layout must have shape [2]");
        return address + word * geometry[position + rank];
    }

    private static int layoutPosition(long[] geometry, int boundary) {
        int position = 4;
        for (int current = 0; current < boundary; current++) {
            int rank = (int) geometry[position]; position += 2 + 2 * rank;
        }
        return position;
    }

    private static double readDouble(Object carrier, long address) {
        return carrier instanceof double[] array ? array[Math.toIntExact(address)]
                : ((MemorySegment) carrier).get(ValueLayout.JAVA_DOUBLE, address * Double.BYTES);
    }
    private static void writeDouble(Object carrier, long address, double value) {
        if (carrier instanceof double[] array) array[Math.toIntExact(address)] = value;
        else ((MemorySegment) carrier).set(ValueLayout.JAVA_DOUBLE, address * Double.BYTES, value);
    }
    private static float readFloat(Object carrier, long address) {
        return carrier instanceof float[] array ? array[Math.toIntExact(address)]
                : ((MemorySegment) carrier).get(ValueLayout.JAVA_FLOAT, address * Float.BYTES);
    }
    private static void writeFloat(Object carrier, long address, float value) {
        if (carrier instanceof float[] array) array[Math.toIntExact(address)] = value;
        else ((MemorySegment) carrier).set(ValueLayout.JAVA_FLOAT, address * Float.BYTES, value);
    }
    private static long readLong(Object carrier, long address) {
        return carrier instanceof long[] array ? array[Math.toIntExact(address)]
                : ((MemorySegment) carrier).get(ValueLayout.JAVA_LONG, address * Long.BYTES);
    }
    private static void writeLong(Object carrier, long address, long value) {
        if (carrier instanceof long[] array) array[Math.toIntExact(address)] = value;
        else ((MemorySegment) carrier).set(ValueLayout.JAVA_LONG, address * Long.BYTES, value);
    }
    private static void writeByte(Object carrier, long address, byte value) {
        if (carrier instanceof byte[] array) array[Math.toIntExact(address)] = value;
        else ((MemorySegment) carrier).set(ValueLayout.JAVA_BYTE, address, value);
    }

    private static Parsed parse(String identity) {
        if (!identity.startsWith("random:")) throw new IllegalArgumentException(
                "encoded random family identity is invalid");
        String[] parts = identity.split(":");
        CpuRandomIr.Family family = CpuRandomIr.Family.valueOf(parts[1]);
        long key = 0, counter = 0, probability = 0;
        for (String part : parts) {
            if (part.startsWith("key=")) key = Long.parseUnsignedLong(part.substring(4), 16);
            else if (part.startsWith("counter=")) counter = Long.parseUnsignedLong(part.substring(8), 16);
            else if (part.startsWith("probability=")) probability = Long.parseUnsignedLong(part.substring(12), 16);
        }
        return new Parsed(family, key, counter, probability);
    }

    private record Parsed(CpuRandomIr.Family family, long key, long counter,
            long probabilityBits) { }
}
