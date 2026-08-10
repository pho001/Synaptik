package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.MethodTypeDesc;
import io.github.pho001.synaptik.model.datatype.DataType;

/** Package-private direct typed primitive-array or native-order segment access emitter. */
final class CpuCarrierEmitter {
    private static final ClassDesc SEGMENT = ClassDesc.of("java.lang.foreign.MemorySegment");
    private static final ClassDesc VALUE_LAYOUT = ClassDesc.of("java.lang.foreign.ValueLayout");
    private static final ClassDesc DOUBLE_LAYOUT = ClassDesc.of("java.lang.foreign.ValueLayout$OfDouble");
    private static final ClassDesc FLOAT_LAYOUT = ClassDesc.of("java.lang.foreign.ValueLayout$OfFloat");
    private static final ClassDesc INT_LAYOUT = ClassDesc.of("java.lang.foreign.ValueLayout$OfInt");
    private static final ClassDesc LONG_LAYOUT = ClassDesc.of("java.lang.foreign.ValueLayout$OfLong");
    private static final ClassDesc BYTE_LAYOUT = ClassDesc.of("java.lang.foreign.ValueLayout$OfByte");
    private static final ClassDesc BYTE_ORDER = ClassDesc.of("java.nio.ByteOrder");
    private static final ClassDesc DOUBLE_VECTOR = ClassDesc.of("jdk.incubator.vector.DoubleVector");
    private static final ClassDesc FLOAT_VECTOR = ClassDesc.of("jdk.incubator.vector.FloatVector");
    private static final ClassDesc VECTOR_SPECIES = ClassDesc.of("jdk.incubator.vector.VectorSpecies");
    private final CodeBuilder code;
    /**
     * Creates an emitter bound to one non-null generated method body.
     *
     * @param code non-null Class-File API code builder retained for generation only
     */
    CpuCarrierEmitter(CodeBuilder code) { this.code = code; }

    /**
     * Emits one scalar load of the supplied type from a generation-time-selected carrier form.
     *
     * @param type non-null exact logical data type to load
     * @param access non-null selected direct carrier form
     * @param parameterSlot local-variable slot holding the exact carrier
     * @param addressLocal local-variable slot holding an element address
     */
    void load(DataType type, CarrierAccess access, int parameterSlot, int addressLocal) {
        if (access != CarrierAccess.MEMORY_SEGMENT) {
            code.aload(parameterSlot).lload(addressLocal).l2i();
            switch (type) {
                case FLOAT64 -> code.daload(); case FLOAT32 -> code.faload();
                case INT32 -> code.iaload(); case INT64 -> code.laload(); case BOOL -> code.baload();
                default -> throw new IllegalArgumentException("unsupported carrier data type");
            }
            return;
        }
        code.aload(parameterSlot); layout(type);
        code.lload(addressLocal).loadConstant((long) type.byteWidth()).lmul();
        code.invokeinterface(SEGMENT, "get", MethodTypeDesc.of(primitive(type), layoutClass(type),
                TypeKind.LONG.upperBound()));
    }

    /**
     * Emits one scalar store of the supplied type to a generation-time-selected carrier form.
     *
     * @param type non-null exact logical data type to store
     * @param access non-null selected direct carrier form
     * @param parameterSlot local-variable slot holding the exact carrier
     * @param addressLocal local-variable slot holding an element address
     * @param valueLocal local-variable slot holding the value to store
     */
    void store(DataType type, CarrierAccess access, int parameterSlot, int addressLocal, int valueLocal) {
        if (access != CarrierAccess.MEMORY_SEGMENT) {
            code.aload(parameterSlot).lload(addressLocal).l2i(); loadLocal(type, valueLocal);
            switch (type) {
                case FLOAT64 -> code.dastore(); case FLOAT32 -> code.fastore();
                case INT32 -> code.iastore(); case INT64 -> code.lastore(); case BOOL -> code.bastore();
                default -> throw new IllegalArgumentException("unsupported carrier data type");
            }
            return;
        }
        code.aload(parameterSlot); layout(type);
        code.lload(addressLocal).loadConstant((long) type.byteWidth()).lmul();
        loadLocal(type, valueLocal);
        code.invokeinterface(SEGMENT, "set", MethodTypeDesc.of(TypeKind.VOID.upperBound(),
                layoutClass(type), TypeKind.LONG.upperBound(), primitive(type)));
    }

    /**
     * Emits one unmasked preferred-species floating vector load or scalar broadcast.
     *
     * @param type exact vector lane type, either FLOAT32 or FLOAT64
     * @param access non-null generation-time-selected direct carrier form
     * @param parameterSlot local-variable slot holding the exact carrier
     * @param addressLocal local-variable slot holding the element address
     * @param broadcast whether to load one scalar and broadcast it to every lane
     */
    void vectorLoad(DataType type, CarrierAccess access, int parameterSlot, int addressLocal,
            boolean broadcast) {
        ClassDesc vector = vectorClass(type);
        code.getstatic(vector, "SPECIES_PREFERRED", VECTOR_SPECIES);
        if (broadcast) {
            load(type, access, parameterSlot, addressLocal);
            code.invokestatic(vector, "broadcast", MethodTypeDesc.of(vector,
                    VECTOR_SPECIES, primitive(type)));
            return;
        }
        code.aload(parameterSlot);
        if (access == arrayCarrier(type)) {
            code.lload(addressLocal).l2i();
            code.invokestatic(vector, "fromArray", MethodTypeDesc.of(vector,
                    VECTOR_SPECIES, primitive(type).arrayType(),
                    TypeKind.INT.upperBound()));
        } else if (access == CarrierAccess.MEMORY_SEGMENT) {
            code.lload(addressLocal).loadConstant((long) type.byteWidth()).lmul();
            code.invokestatic(BYTE_ORDER, "nativeOrder", MethodTypeDesc.of(BYTE_ORDER));
            code.invokestatic(vector, "fromMemorySegment", MethodTypeDesc.of(vector,
                    VECTOR_SPECIES, SEGMENT, TypeKind.LONG.upperBound(), BYTE_ORDER));
        } else throw new IllegalArgumentException("carrier does not match vector data type");
    }

    /**
     * Emits one unmasked preferred-species floating vector store.
     *
     * @param type exact vector lane type, either FLOAT32 or FLOAT64
     * @param access non-null generation-time-selected direct carrier form
     * @param parameterSlot local-variable slot holding the writable exact carrier
     * @param addressLocal local-variable slot holding the element address
     * @param valueLocal local-variable slot holding the non-null typed vector
     */
    void vectorStore(DataType type, CarrierAccess access, int parameterSlot, int addressLocal,
            int valueLocal) {
        ClassDesc vector = vectorClass(type);
        code.aload(valueLocal).aload(parameterSlot);
        if (access == arrayCarrier(type)) {
            code.lload(addressLocal).l2i();
            code.invokevirtual(vector, "intoArray", MethodTypeDesc.of(
                    TypeKind.VOID.upperBound(), primitive(type).arrayType(),
                    TypeKind.INT.upperBound()));
        } else if (access == CarrierAccess.MEMORY_SEGMENT) {
            code.lload(addressLocal).loadConstant((long) type.byteWidth()).lmul();
            code.invokestatic(BYTE_ORDER, "nativeOrder", MethodTypeDesc.of(BYTE_ORDER));
            code.invokevirtual(vector, "intoMemorySegment", MethodTypeDesc.of(
                    TypeKind.VOID.upperBound(), SEGMENT, TypeKind.LONG.upperBound(), BYTE_ORDER));
        } else throw new IllegalArgumentException("carrier does not match vector data type");
    }

    private void layout(DataType type) {
        String field = switch (type) {
            case FLOAT64 -> "JAVA_DOUBLE_UNALIGNED"; case FLOAT32 -> "JAVA_FLOAT_UNALIGNED";
            case INT32 -> "JAVA_INT_UNALIGNED"; case INT64 -> "JAVA_LONG_UNALIGNED";
            case BOOL -> "JAVA_BYTE"; default -> throw new IllegalArgumentException("unsupported type");
        };
        ClassDesc layout = layoutClass(type);
        code.getstatic(VALUE_LAYOUT, field, layout);
        if (type == DataType.BOOL) return;
        code.invokestatic(BYTE_ORDER, "nativeOrder", MethodTypeDesc.of(BYTE_ORDER));
        code.invokeinterface(VALUE_LAYOUT, "withOrder", MethodTypeDesc.of(VALUE_LAYOUT, BYTE_ORDER));
        code.checkcast(layout);
    }

    private void loadLocal(DataType type, int local) {
        switch (type) {
            case FLOAT64 -> code.dload(local); case FLOAT32 -> code.fload(local);
            case INT32, BOOL -> code.iload(local); case INT64 -> code.lload(local);
            default -> throw new IllegalArgumentException("unsupported type");
        }
    }

    private static ClassDesc layoutClass(DataType type) {
        return switch (type) {
            case FLOAT64 -> DOUBLE_LAYOUT; case FLOAT32 -> FLOAT_LAYOUT; case INT32 -> INT_LAYOUT;
            case INT64 -> LONG_LAYOUT; case BOOL -> BYTE_LAYOUT;
            default -> throw new IllegalArgumentException("unsupported type");
        };
    }

    private static java.lang.constant.ClassDesc primitive(DataType type) {
        return switch (type) {
            case FLOAT64 -> java.lang.constant.ConstantDescs.CD_double;
            case FLOAT32 -> java.lang.constant.ConstantDescs.CD_float;
            case INT32 -> java.lang.constant.ConstantDescs.CD_int;
            case INT64 -> java.lang.constant.ConstantDescs.CD_long;
            case BOOL -> java.lang.constant.ConstantDescs.CD_byte;
            default -> throw new IllegalArgumentException("unsupported type");
        };
    }

    private static ClassDesc vectorClass(DataType type) {
        return switch (type) {
            case FLOAT64 -> DOUBLE_VECTOR;
            case FLOAT32 -> FLOAT_VECTOR;
            default -> throw new IllegalArgumentException("unsupported vector data type");
        };
    }

    private static CarrierAccess arrayCarrier(DataType type) {
        return switch (type) {
            case FLOAT64 -> CarrierAccess.DOUBLE_ARRAY;
            case FLOAT32 -> CarrierAccess.FLOAT_ARRAY;
            default -> throw new IllegalArgumentException("unsupported vector data type");
        };
    }
}
