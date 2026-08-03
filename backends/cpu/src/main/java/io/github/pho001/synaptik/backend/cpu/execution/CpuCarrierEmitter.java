package io.github.pho001.synaptik.backend.cpu.execution;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.MethodTypeDesc;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * Emits exact already-specialized primitive-array and {@link MemorySegment} access sequences.
 * It consumes the direct signature fixed by the specialization, preserves declared read/write
 * access, and performs no storage discovery, copying, allocation, or ownership transfer.
 */
final class CpuCarrierEmitter {
    private static final ClassDesc MEMORY_SEGMENT = ClassDesc.of("java.lang.foreign.MemorySegment");
    private static final ClassDesc VALUE_LAYOUT = ClassDesc.of("java.lang.foreign.ValueLayout");
    private static final ClassDesc BYTE_ORDER = ClassDesc.of("java.nio.ByteOrder");
    private final CodeBuilder code;
    private final CpuKernelSpecialization specialization;

    /**
     * Creates a direct-access helper for one generated method.
     * @param code non-null generated-method builder
     * @param specialization non-null complete specialization
     * @throws NullPointerException if either argument is {@code null}
     */
    CpuCarrierEmitter(CodeBuilder code, CpuKernelSpecialization specialization) {
        this.code = Objects.requireNonNull(code, "code");
        this.specialization = Objects.requireNonNull(specialization, "specialization");
    }
    /** Returns the current entry-body builder.
     * @return the exact non-null code builder supplied at construction */ CodeBuilder code() { return code; }
    /** Returns the specialization governing emitted access.
     * @return the exact non-null immutable specialization supplied at construction */ CpuKernelSpecialization specialization() { return specialization; }

    /**
     * Emits a scalar load, leaving the primitive value on the operand stack.
     * @param argumentIndex valid zero-based specialization argument index
     * @param elementIndexLocal local slot containing a non-negative long element index
     * @throws IndexOutOfBoundsException if {@code argumentIndex} is invalid
     * @throws IllegalArgumentException if the argument is write-only
     */
    void emitScalarLoad(int argumentIndex, int elementIndexLocal) {
        var argument = argument(argumentIndex);
        requireReadable(argument);
        if (argument.carrier() == CpuKernelSpecialization.Carrier.MEMORY_SEGMENT) {
            code.aload(parameterSlot(argumentIndex)); emitLayout(argument.dataType());
            emitSegmentByteOffset(argument, elementIndexLocal);
            code.invokeinterface(MEMORY_SEGMENT, "get", MethodTypeDesc.of(
                    classDesc(argument.dataType()), layoutClass(argument.dataType()),
                    TypeKind.LONG.upperBound()));
        } else {
            code.aload(parameterSlot(argumentIndex)); emitArrayElementIndex(argumentIndex,
                    argument, elementIndexLocal); code.arrayLoad(typeKind(argument.dataType()));
        }
    }

    /**
     * Emits a scalar store, consuming one compatible primitive value from the operand stack.
     * @param argumentIndex valid zero-based specialization argument index
     * @param elementIndexLocal local slot containing a non-negative long element index
     * @throws IndexOutOfBoundsException if {@code argumentIndex} is invalid
     * @throws IllegalArgumentException if the argument is read-only
     */
    void emitScalarStore(int argumentIndex, int elementIndexLocal) {
        var argument = argument(argumentIndex); requireWritable(argument);
        TypeKind kind = typeKind(argument.dataType()); int valueLocal = code.allocateLocal(kind);
        code.storeLocal(kind, valueLocal);
        if (argument.carrier() == CpuKernelSpecialization.Carrier.MEMORY_SEGMENT) {
            code.aload(parameterSlot(argumentIndex)); emitLayout(argument.dataType());
            emitSegmentByteOffset(argument, elementIndexLocal); code.loadLocal(kind, valueLocal);
            code.invokeinterface(MEMORY_SEGMENT, "set", MethodTypeDesc.of(
                    TypeKind.VOID.upperBound(), layoutClass(argument.dataType()),
                    TypeKind.LONG.upperBound(), classDesc(argument.dataType())));
        } else {
            code.aload(parameterSlot(argumentIndex)); emitArrayElementIndex(argumentIndex,
                    argument, elementIndexLocal); code.loadLocal(kind, valueLocal);
            code.arrayStore(kind);
        }
    }

    /**
     * Emits an unmasked full-species vector load, leaving the concrete vector on the stack.
     * @param argumentIndex valid readable argument matching the selected vector lane type
     * @param elementIndexLocal local slot containing a long element index
     * @throws IndexOutOfBoundsException if {@code argumentIndex} is invalid
     * @throws IllegalArgumentException if the argument is write-only or its data type does not
     *     match the selected vector lane type
     */
    void emitVectorLoad(int argumentIndex, int elementIndexLocal) {
        var argument = vectorArgument(argumentIndex, false);
        var vector = new CpuVectorEmitter(code, specialization); vector.loadSpecies();
        ClassDesc vectorClass = vector.vectorClass();
        if (argument.carrier() == CpuKernelSpecialization.Carrier.MEMORY_SEGMENT) {
            code.aload(parameterSlot(argumentIndex)); emitSegmentByteOffset(argument, elementIndexLocal);
            emitByteOrder(); code.invokestatic(vectorClass, "fromMemorySegment", MethodTypeDesc.of(
                    vectorClass, vector.speciesClass(), MEMORY_SEGMENT,
                    TypeKind.LONG.upperBound(), BYTE_ORDER));
        } else {
            code.aload(parameterSlot(argumentIndex)); emitArrayElementIndex(argumentIndex,
                    argument, elementIndexLocal);
            code.invokestatic(vectorClass, "fromArray", MethodTypeDesc.of(vectorClass,
                    vector.speciesClass(), arrayDesc(argument.carrier()),
                    TypeKind.INT.upperBound()));
        }
    }

    /**
     * Emits an unmasked full-species vector store, consuming the concrete vector from the stack.
     * @param argumentIndex valid writable argument matching the selected vector lane type
     * @param elementIndexLocal local slot containing a long element index
     * @throws IndexOutOfBoundsException if {@code argumentIndex} is invalid
     * @throws IllegalArgumentException if the argument is read-only or its data type does not
     *     match the selected vector lane type
     */
    void emitVectorStore(int argumentIndex, int elementIndexLocal) {
        var argument = vectorArgument(argumentIndex, true);
        var vector = new CpuVectorEmitter(code, specialization); int local = code.allocateLocal(TypeKind.REFERENCE);
        code.astore(local); code.aload(local);
        if (argument.carrier() == CpuKernelSpecialization.Carrier.MEMORY_SEGMENT) {
            code.aload(parameterSlot(argumentIndex)); emitSegmentByteOffset(argument, elementIndexLocal);
            emitByteOrder(); code.invokevirtual(vector.vectorClass(), "intoMemorySegment",
                    MethodTypeDesc.of(TypeKind.VOID.upperBound(), MEMORY_SEGMENT,
                            TypeKind.LONG.upperBound(), BYTE_ORDER));
        } else {
            code.aload(parameterSlot(argumentIndex)); emitArrayElementIndex(argumentIndex,
                    argument, elementIndexLocal); code.invokevirtual(vector.vectorClass(), "intoArray",
                    MethodTypeDesc.of(TypeKind.VOID.upperBound(),
                            arrayDesc(argument.carrier()), TypeKind.INT.upperBound()));
        }
    }

    private CpuKernelSpecialization.Argument argument(int index) {
        return specialization.arguments().get(index);
    }
    private CpuKernelSpecialization.Argument vectorArgument(int index, boolean write) {
        var argument = argument(index); if (write) requireWritable(argument); else requireReadable(argument);
        if (argument.dataType() != specialization.vectorShape().orElseThrow().laneType()) {
            throw new IllegalArgumentException("arguments[" + index + "] dataType does not match vector lane type");
        }
        return argument;
    }
    private static void requireReadable(CpuKernelSpecialization.Argument argument) {
        if (argument.access() == PreparedExecutable.BufferAccess.WRITE_ONLY)
            throw new IllegalArgumentException("write-only argument cannot be loaded");
    }
    private static void requireWritable(CpuKernelSpecialization.Argument argument) {
        if (argument.access() == PreparedExecutable.BufferAccess.READ_ONLY)
            throw new IllegalArgumentException("read-only argument cannot be stored");
    }
    private int parameterSlot(int argumentIndex) {
        int slot = 0;
        for (int index = 0; index < argumentIndex; index++) {
            var argument = specialization.arguments().get(index); slot++;
            if (argument.carrier() != CpuKernelSpecialization.Carrier.MEMORY_SEGMENT
                    && !argument.byteOffsetBaked()) slot += 2;
        }
        return slot;
    }
    private void emitArrayElementIndex(int argumentIndex, CpuKernelSpecialization.Argument argument,
            int elementIndexLocal) {
        if (argument.byteOffsetBaked()) code.loadConstant(argument.bakedByteOffset() / width(argument.dataType()));
        else { code.lload(parameterSlot(argumentIndex) + 1); code.loadConstant((long) width(argument.dataType()));
            code.ldiv(); }
        code.lload(elementIndexLocal); code.ladd(); code.l2i();
    }
    private void emitSegmentByteOffset(CpuKernelSpecialization.Argument argument, int indexLocal) {
        code.loadConstant(argument.bakedByteOffset()); code.lload(indexLocal);
        code.loadConstant((long) width(argument.dataType())); code.lmul(); code.ladd();
    }
    private void emitLayout(DataType type) {
        ClassDesc layout = layoutClass(type); code.getstatic(VALUE_LAYOUT, layoutField(type), layout);
        emitByteOrder(); code.invokeinterface(VALUE_LAYOUT, "withOrder",
                MethodTypeDesc.of(VALUE_LAYOUT, BYTE_ORDER)); code.checkcast(layout);
    }
    private void emitByteOrder() { code.getstatic(BYTE_ORDER,
            specialization.byteOrder() == ByteOrder.LITTLE_ENDIAN ? "LITTLE_ENDIAN" : "BIG_ENDIAN",
            BYTE_ORDER); }
    private static String layoutField(DataType type) { return switch (type) {
        case FLOAT64 -> "JAVA_DOUBLE_UNALIGNED"; case FLOAT32 -> "JAVA_FLOAT_UNALIGNED";
        case BFLOAT16 -> "JAVA_SHORT_UNALIGNED"; case INT32 -> "JAVA_INT_UNALIGNED";
        case INT64 -> "JAVA_LONG_UNALIGNED"; case BOOL -> "JAVA_BYTE"; }; }
    private static ClassDesc layoutClass(DataType type) { return ClassDesc.of(
            "java.lang.foreign.ValueLayout$" + switch (type) {
                case FLOAT64 -> "OfDouble"; case FLOAT32 -> "OfFloat"; case BFLOAT16 -> "OfShort";
                case INT32 -> "OfInt"; case INT64 -> "OfLong"; case BOOL -> "OfByte"; }); }
    private static TypeKind typeKind(DataType type) { return switch (type) {
        case FLOAT64 -> TypeKind.DOUBLE; case FLOAT32 -> TypeKind.FLOAT;
        case BFLOAT16 -> TypeKind.SHORT; case INT32 -> TypeKind.INT;
        case INT64 -> TypeKind.LONG; case BOOL -> TypeKind.BYTE; }; }
    private static ClassDesc classDesc(DataType type) { return switch (type) {
        case FLOAT64 -> java.lang.constant.ConstantDescs.CD_double;
        case FLOAT32 -> java.lang.constant.ConstantDescs.CD_float;
        case BFLOAT16 -> java.lang.constant.ConstantDescs.CD_short;
        case INT32 -> java.lang.constant.ConstantDescs.CD_int;
        case INT64 -> java.lang.constant.ConstantDescs.CD_long;
        case BOOL -> java.lang.constant.ConstantDescs.CD_byte;
    }; }
    private static int width(DataType type) { return type.byteWidth(); }
    private static ClassDesc arrayDesc(CpuKernelSpecialization.Carrier carrier) { return switch (carrier) {
        case DOUBLE_ARRAY -> ClassDesc.ofDescriptor("[D"); case FLOAT_ARRAY -> ClassDesc.ofDescriptor("[F");
        case SHORT_ARRAY -> ClassDesc.ofDescriptor("[S"); case INT_ARRAY -> ClassDesc.ofDescriptor("[I");
        case LONG_ARRAY -> ClassDesc.ofDescriptor("[J"); case BYTE_ARRAY -> ClassDesc.ofDescriptor("[B");
        case MEMORY_SEGMENT -> throw new IllegalArgumentException("segment is not an array"); }; }
}
