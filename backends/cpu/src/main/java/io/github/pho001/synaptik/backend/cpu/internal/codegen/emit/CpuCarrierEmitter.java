package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.MethodTypeDesc;
import java.util.List;
import io.github.pho001.synaptik.model.datatype.DataType;

/**
 * Emits direct typed primitive-array or native-order segment loads and stores for one generated
 * CPU entry.
 *
 * <p>The selected lane type determines the exact scalar carrier and preferred Vector API class.
 * Floating comparison/classification BOOL values may instead remain {@code VectorMask} locals;
 * an external scalar/all-zero BOOL condition is converted once to the matching all-true or
 * all-false mask. Materialized BOOL boundaries retain canonical byte {@code 0}/{@code 1}
 * storage. Carrier choice and access geometry are already validated before emission.</p>
 */
final class CpuCarrierEmitter {
    private static final int LAYOUT_LOCAL_COUNT = 6;
    private static final ClassDesc SEGMENT = ClassDesc.of("java.lang.foreign.MemorySegment");
    private static final ClassDesc VALUE_LAYOUT = ClassDesc.of("java.lang.foreign.ValueLayout");
    private static final ClassDesc DOUBLE_LAYOUT = ClassDesc.of("java.lang.foreign.ValueLayout$OfDouble");
    private static final ClassDesc FLOAT_LAYOUT = ClassDesc.of("java.lang.foreign.ValueLayout$OfFloat");
    private static final ClassDesc SHORT_LAYOUT = ClassDesc.of("java.lang.foreign.ValueLayout$OfShort");
    private static final ClassDesc INT_LAYOUT = ClassDesc.of("java.lang.foreign.ValueLayout$OfInt");
    private static final ClassDesc LONG_LAYOUT = ClassDesc.of("java.lang.foreign.ValueLayout$OfLong");
    private static final ClassDesc BYTE_LAYOUT = ClassDesc.of("java.lang.foreign.ValueLayout$OfByte");
    private static final ClassDesc BYTE_ORDER = ClassDesc.of("java.nio.ByteOrder");
    private static final ClassDesc DOUBLE_VECTOR = ClassDesc.of("jdk.incubator.vector.DoubleVector");
    private static final ClassDesc FLOAT_VECTOR = ClassDesc.of("jdk.incubator.vector.FloatVector");
    private static final ClassDesc INT_VECTOR = ClassDesc.of("jdk.incubator.vector.IntVector");
    private static final ClassDesc LONG_VECTOR = ClassDesc.of("jdk.incubator.vector.LongVector");
    private static final ClassDesc BYTE_VECTOR = ClassDesc.of("jdk.incubator.vector.ByteVector");
    private static final ClassDesc VECTOR_SPECIES = ClassDesc.of("jdk.incubator.vector.VectorSpecies");
    private static final ClassDesc VECTOR_MASK = ClassDesc.of("jdk.incubator.vector.VectorMask");
    private final CodeBuilder code;
    private final int layoutLocalBase;
    private int speciesLocal = -1;
    /**
     * Creates an emitter bound to one non-null generated method body.
     *
     * @param code non-null Class-File API code builder retained for generation only
     */
    CpuCarrierEmitter(CodeBuilder code) {
        this.code = code;
        this.layoutLocalBase = firstGeneratedLocal(code);
    }

    /**
     * Reserves the fixed typed-layout local block and initializes every layout required by a
     * segment boundary once at generated-entry invocation setup.
     *
     * <p>The fixed block lets independently focused family emitters construct carrier emitters
     * without sharing mutable generation state. Array-only specializations reserve the same local
     * indexes but emit no layout construction or native-order lookup.</p>
     *
     * @param code non-null Class-File API code builder positioned at the start of the entry body
     * @param boundaryTypes non-null ordered exact boundary data types
     * @param carrierPattern non-null ordered exact carrier forms matching {@code boundaryTypes}
     * @throws IllegalArgumentException if the boundary lists disagree or the layout block is not
     *     the first generated-local allocation
     */
    static void prepareSegmentLayouts(CodeBuilder code, List<DataType> boundaryTypes,
            List<CarrierAccess> carrierPattern) {
        if (boundaryTypes.size() != carrierPattern.size()) {
            throw new IllegalArgumentException("boundary types and carriers must have equal size");
        }
        int base = firstGeneratedLocal(code);
        for (int offset = 0; offset < LAYOUT_LOCAL_COUNT; offset++) {
            if (code.allocateLocal(TypeKind.REFERENCE) != base + offset) {
                throw new IllegalArgumentException(
                        "segment-layout locals must be the first generated locals");
            }
        }
        boolean[] required = new boolean[LAYOUT_LOCAL_COUNT];
        for (int boundary = 0; boundary < boundaryTypes.size(); boundary++) {
            if (carrierPattern.get(boundary) == CarrierAccess.MEMORY_SEGMENT) {
                required[layoutLocalOffset(boundaryTypes.get(boundary))] = true;
            }
        }
        for (DataType type : DataType.values()) {
            int offset = layoutLocalOffset(type);
            if (required[offset]) emitLayout(code, type).astore(base + offset);
        }
    }

    /**
     * Emits and retains one preferred-species local for all vector accesses in this method.
     * @param type non-null exact admitted vector lane type
     */
    void prepareVectorSpecies(DataType type) {
        if (speciesLocal >= 0) return;
        speciesLocal = code.allocateLocal(TypeKind.REFERENCE);
        code.getstatic(vectorClass(type), "SPECIES_PREFERRED", VECTOR_SPECIES).astore(speciesLocal);
    }

    /**
     * Emits one scalar load of the supplied type from a generation-time-selected carrier form.
     *
     * @param type non-null exact logical data type to load
     * @param access non-null selected direct carrier form
     * @param parameterSlot local-variable slot holding the exact carrier
     * @param addressLocal local-variable slot holding an element address
     */
    void load(DataType type, CarrierAccess access, int parameterSlot, int addressLocal) {
        load(type, access, parameterSlot, addressLocal, false);
    }

    /**
     * Emits one scalar load while honoring the selected element-address local width.
     *
     * @param type non-null exact logical data type to load
     * @param access non-null selected direct carrier form
     * @param parameterSlot local-variable slot holding the exact carrier
     * @param addressLocal local-variable slot holding either a proved Java array index when
     *     {@code intAddress} is {@code true}, or a general element address when it is
     *     {@code false}
     * @param intAddress whether heap-array access can use {@code addressLocal} directly as an
     *     {@code int}; current {@link CarrierAccess#MEMORY_SEGMENT} callers keep this
     *     {@code false} and supply a {@code long} element address
     * @throws IllegalArgumentException if the selected data type cannot be loaded through this
     *     carrier form
     */
    void load(DataType type, CarrierAccess access, int parameterSlot, int addressLocal,
            boolean intAddress) {
        if (access != CarrierAccess.MEMORY_SEGMENT) {
            code.aload(parameterSlot);
            if (intAddress) code.iload(addressLocal); else code.lload(addressLocal).l2i();
            switch (type) {
                case FLOAT64 -> code.daload(); case FLOAT32 -> code.faload();
                case BFLOAT16 -> code.saload(); case INT32 -> code.iaload();
                case INT64 -> code.laload(); case BOOL -> code.baload();
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
        store(type, access, parameterSlot, addressLocal, valueLocal, false);
    }

    /**
     * Emits one scalar store while honoring the selected element-address local width.
     *
     * @param type non-null exact logical data type to store
     * @param access non-null selected direct carrier form
     * @param parameterSlot local-variable slot holding the exact carrier
     * @param addressLocal local-variable slot holding either a proved Java array index when
     *     {@code intAddress} is {@code true}, or a general element address when it is
     *     {@code false}
     * @param valueLocal local-variable slot holding the value to store
     * @param intAddress whether heap-array access can use {@code addressLocal} directly as an
     *     {@code int}; current {@link CarrierAccess#MEMORY_SEGMENT} callers keep this
     *     {@code false} and supply a {@code long} element address
     * @throws IllegalArgumentException if the selected data type cannot be stored through this
     *     carrier form
     */
    void store(DataType type, CarrierAccess access, int parameterSlot, int addressLocal,
            int valueLocal, boolean intAddress) {
        if (access != CarrierAccess.MEMORY_SEGMENT) {
            code.aload(parameterSlot);
            if (intAddress) code.iload(addressLocal); else code.lload(addressLocal).l2i();
            loadLocal(type, valueLocal);
            switch (type) {
                case FLOAT64 -> code.dastore(); case FLOAT32 -> code.fastore();
                case BFLOAT16 -> code.sastore(); case INT32 -> code.iastore();
                case INT64 -> code.lastore(); case BOOL -> code.bastore();
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
     * Emits one unmasked preferred-species typed vector load or scalar broadcast.
     *
     * @param type exact admitted vector lane type
     * @param access non-null generation-time-selected direct carrier form
     * @param parameterSlot local-variable slot holding the exact carrier
     * @param addressLocal local-variable slot holding the element address
     * @param broadcast whether to load one scalar and broadcast it to every lane
     */
    void vectorLoad(DataType type, CarrierAccess access, int parameterSlot, int addressLocal,
            boolean broadcast) {
        vectorLoad(type, access, parameterSlot, addressLocal, broadcast, false);
    }

    /**
     * Emits one unmasked preferred-species typed vector load or scalar broadcast using the
     * selected element-address local width.
     *
     * @param type non-null exact admitted vector lane type
     * @param access non-null generation-time-selected direct carrier form
     * @param parameterSlot local-variable slot holding the exact carrier
     * @param addressLocal local-variable slot holding either a proved Java array index when
     *     {@code intAddress} is {@code true}, or a general element address when it is
     *     {@code false}
     * @param broadcast whether to load one scalar and broadcast it to every lane
     * @param intAddress whether heap-array access can use {@code addressLocal} directly as an
     *     {@code int}; current {@link CarrierAccess#MEMORY_SEGMENT} callers keep this
     *     {@code false} and supply a {@code long} element address
     * @throws IllegalArgumentException if the selected carrier does not match the vector lane type
     */
    void vectorLoad(DataType type, CarrierAccess access, int parameterSlot, int addressLocal,
            boolean broadcast, boolean intAddress) {
        ClassDesc vector = vectorClass(type);
        if (speciesLocal < 0) prepareVectorSpecies(type);
        code.aload(speciesLocal);
        if (broadcast) {
            load(type, access, parameterSlot, addressLocal, intAddress);
            code.invokestatic(vector, "broadcast", MethodTypeDesc.of(vector,
                    VECTOR_SPECIES, primitive(type)));
            return;
        }
        code.aload(parameterSlot);
        if (access == arrayCarrier(type)) {
            if (intAddress) code.iload(addressLocal); else code.lload(addressLocal).l2i();
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
     * Emits one unmasked preferred-species typed vector store.
     *
     * @param type exact admitted vector lane type
     * @param access non-null generation-time-selected direct carrier form
     * @param parameterSlot local-variable slot holding the writable exact carrier
     * @param addressLocal local-variable slot holding the element address
     * @param valueLocal local-variable slot holding the non-null typed vector
     */
    void vectorStore(DataType type, CarrierAccess access, int parameterSlot, int addressLocal,
            int valueLocal) {
        vectorStore(type, access, parameterSlot, addressLocal, valueLocal, false);
    }

    /**
     * Emits one unmasked preferred-species typed vector store using the selected element-address
     * local width.
     *
     * @param type non-null exact admitted vector lane type
     * @param access non-null generation-time-selected direct carrier form
     * @param parameterSlot local-variable slot holding the writable exact carrier
     * @param addressLocal local-variable slot holding either a proved Java array index when
     *     {@code intAddress} is {@code true}, or a general element address when it is
     *     {@code false}
     * @param valueLocal local-variable slot holding the non-null typed vector
     * @param intAddress whether heap-array access can use {@code addressLocal} directly as an
     *     {@code int}; current {@link CarrierAccess#MEMORY_SEGMENT} callers keep this
     *     {@code false} and supply a {@code long} element address
     * @throws IllegalArgumentException if the selected carrier does not match the vector lane type
     */
    void vectorStore(DataType type, CarrierAccess access, int parameterSlot, int addressLocal,
            int valueLocal, boolean intAddress) {
        ClassDesc vector = vectorClass(type);
        code.aload(valueLocal).aload(parameterSlot);
        if (access == arrayCarrier(type)) {
            if (intAddress) code.iload(addressLocal); else code.lload(addressLocal).l2i();
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

    /**
     * Emits an allocation-free canonical scalar BOOL load broadcast as a typed numeric mask.
     *
     * @param laneType exact FLOAT32 or FLOAT64 mask lane type
     * @param access exact BOOL boundary carrier form
     * @param parameterSlot local-variable slot holding the carrier
     * @param addressLocal local-variable slot holding the scalar element address
     */
    void scalarBoolMaskLoad(DataType laneType, CarrierAccess access, int parameterSlot,
            int addressLocal) {
        scalarBoolMaskLoad(laneType, access, parameterSlot, addressLocal, false);
    }

    /**
     * Emits an allocation-free canonical scalar BOOL load broadcast as a typed numeric mask using
     * the selected element-address local width.
     *
     * @param laneType non-null exact FLOAT32 or FLOAT64 mask lane type
     * @param access exact BOOL boundary carrier form
     * @param parameterSlot local-variable slot holding the carrier
     * @param addressLocal local-variable slot holding either a proved Java array index when
     *     {@code intAddress} is {@code true}, or a general element address when it is
     *     {@code false}
     * @param intAddress whether heap-array access can use {@code addressLocal} directly as an
     *     {@code int}; current {@link CarrierAccess#MEMORY_SEGMENT} callers keep this
     *     {@code false} and supply a {@code long} element address
     * @throws IllegalArgumentException if {@code laneType} or {@code access} is outside the
     *     supported floating-mask form
     */
    void scalarBoolMaskLoad(DataType laneType, CarrierAccess access, int parameterSlot,
            int addressLocal, boolean intAddress) {
        ClassDesc vector = vectorClass(laneType);
        if (speciesLocal < 0) prepareVectorSpecies(laneType);
        code.aload(speciesLocal);
        load(DataType.BOOL, access, parameterSlot, addressLocal, intAddress);
        code.i2l().lneg();
        code.invokestatic(VECTOR_MASK, "fromLong", MethodTypeDesc.of(VECTOR_MASK,
                VECTOR_SPECIES, TypeKind.LONG.upperBound()));
    }

    private void layout(DataType type) {
        code.aload(layoutLocalBase + layoutLocalOffset(type));
    }

    private static CodeBuilder emitLayout(CodeBuilder code, DataType type) {
        String field = switch (type) {
            case FLOAT64 -> "JAVA_DOUBLE_UNALIGNED"; case FLOAT32 -> "JAVA_FLOAT_UNALIGNED";
            case BFLOAT16 -> "JAVA_SHORT_UNALIGNED";
            case INT32 -> "JAVA_INT_UNALIGNED"; case INT64 -> "JAVA_LONG_UNALIGNED";
            case BOOL -> "JAVA_BYTE"; default -> throw new IllegalArgumentException("unsupported type");
        };
        ClassDesc layout = layoutClass(type);
        code.getstatic(VALUE_LAYOUT, field, layout);
        if (type == DataType.BOOL) return code;
        code.invokestatic(BYTE_ORDER, "nativeOrder", MethodTypeDesc.of(BYTE_ORDER));
        code.invokeinterface(VALUE_LAYOUT, "withOrder", MethodTypeDesc.of(VALUE_LAYOUT, BYTE_ORDER));
        return code.checkcast(layout);
    }

    private static int firstGeneratedLocal(CodeBuilder code) {
        int parameter = 0;
        int lastSlot = -1;
        while (true) {
            try {
                lastSlot = code.parameterSlot(parameter++);
            } catch (IndexOutOfBoundsException exhausted) {
                break;
            }
        }
        // Every generated entry ends with the primitive long end bound.
        if (lastSlot < 0) throw new IllegalArgumentException(
                "generated entries must declare primitive range parameters");
        return lastSlot + TypeKind.LONG.slotSize();
    }

    private static int layoutLocalOffset(DataType type) {
        return switch (type) {
            case FLOAT64 -> 0;
            case FLOAT32 -> 1;
            case BFLOAT16 -> 2;
            case INT32 -> 3;
            case INT64 -> 4;
            case BOOL -> 5;
        };
    }

    private void loadLocal(DataType type, int local) {
        switch (type) {
            case FLOAT64 -> code.dload(local); case FLOAT32 -> code.fload(local);
            case BFLOAT16, INT32, BOOL -> code.iload(local); case INT64 -> code.lload(local);
            default -> throw new IllegalArgumentException("unsupported type");
        }
    }

    private static ClassDesc layoutClass(DataType type) {
        return switch (type) {
            case FLOAT64 -> DOUBLE_LAYOUT; case FLOAT32 -> FLOAT_LAYOUT;
            case BFLOAT16 -> SHORT_LAYOUT; case INT32 -> INT_LAYOUT;
            case INT64 -> LONG_LAYOUT; case BOOL -> BYTE_LAYOUT;
            default -> throw new IllegalArgumentException("unsupported type");
        };
    }

    private static java.lang.constant.ClassDesc primitive(DataType type) {
        return switch (type) {
            case FLOAT64 -> java.lang.constant.ConstantDescs.CD_double;
            case FLOAT32 -> java.lang.constant.ConstantDescs.CD_float;
            case BFLOAT16 -> java.lang.constant.ConstantDescs.CD_short;
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
            case INT32 -> INT_VECTOR;
            case INT64 -> LONG_VECTOR;
            case BOOL -> BYTE_VECTOR;
            default -> throw new IllegalArgumentException("unsupported vector data type");
        };
    }

    private static CarrierAccess arrayCarrier(DataType type) {
        return switch (type) {
            case FLOAT64 -> CarrierAccess.DOUBLE_ARRAY;
            case FLOAT32 -> CarrierAccess.FLOAT_ARRAY;
            case INT32 -> CarrierAccess.INT_ARRAY;
            case INT64 -> CarrierAccess.LONG_ARRAY;
            case BOOL -> CarrierAccess.BYTE_ARRAY;
            default -> throw new IllegalArgumentException("unsupported vector data type");
        };
    }
}
