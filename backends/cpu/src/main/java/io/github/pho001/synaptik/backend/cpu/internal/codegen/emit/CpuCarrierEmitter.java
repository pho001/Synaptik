package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.MethodTypeDesc;

/** Package-private direct FLOAT64 heap-array or segment access emitter. */
final class CpuCarrierEmitter {
    private static final ClassDesc SEGMENT = ClassDesc.of("java.lang.foreign.MemorySegment");
    private static final ClassDesc VALUE_LAYOUT = ClassDesc.of("java.lang.foreign.ValueLayout");
    private static final ClassDesc DOUBLE_LAYOUT = ClassDesc.of("java.lang.foreign.ValueLayout$OfDouble");
    private static final ClassDesc BYTE_ORDER = ClassDesc.of("java.nio.ByteOrder");
    private static final ClassDesc DOUBLE_VECTOR = ClassDesc.of("jdk.incubator.vector.DoubleVector");
    private static final ClassDesc VECTOR_SPECIES = ClassDesc.of("jdk.incubator.vector.VectorSpecies");
    private final CodeBuilder code;
    /**
     * Creates an emitter bound to one non-null generated method body.
     *
     * @param code non-null Class-File API code builder retained for generation only
     */
    CpuCarrierEmitter(CodeBuilder code) { this.code = code; }

    /**
     * Emits one FLOAT64 load from a generation-time-selected carrier form.
     *
     * @param access non-null selected direct carrier form
     * @param parameterSlot local-variable slot holding the exact carrier
     * @param addressLocal local-variable slot holding an element address
     */
    void load(CarrierAccess access, int parameterSlot, int addressLocal) {
        if (access == CarrierAccess.DOUBLE_ARRAY) {
            code.aload(parameterSlot).lload(addressLocal).l2i().daload();
        } else {
            code.aload(parameterSlot); layout();
            code.lload(addressLocal).loadConstant((long) Double.BYTES).lmul();
            code.invokeinterface(SEGMENT, "get", MethodTypeDesc.of(
                    java.lang.constant.ConstantDescs.CD_double, DOUBLE_LAYOUT,
                    TypeKind.LONG.upperBound()));
        }
    }

    /**
     * Emits one FLOAT64 store to a generation-time-selected carrier form.
     *
     * @param access non-null selected direct carrier form
     * @param parameterSlot local-variable slot holding the exact carrier
     * @param addressLocal local-variable slot holding an element address
     * @param valueLocal local-variable slot holding the value to store
     */
    void store(CarrierAccess access, int parameterSlot, int addressLocal, int valueLocal) {
        if (access == CarrierAccess.DOUBLE_ARRAY) {
            code.aload(parameterSlot).lload(addressLocal).l2i().dload(valueLocal).dastore();
        } else {
            code.aload(parameterSlot); layout();
            code.lload(addressLocal).loadConstant((long) Double.BYTES).lmul();
            code.dload(valueLocal);
            code.invokeinterface(SEGMENT, "set", MethodTypeDesc.of(TypeKind.VOID.upperBound(),
                    DOUBLE_LAYOUT, TypeKind.LONG.upperBound(),
                    java.lang.constant.ConstantDescs.CD_double));
        }
    }

    /**
     * Emits one unmasked preferred-species FLOAT64 vector load or scalar broadcast.
     *
     * @param access non-null generation-time-selected direct carrier form
     * @param parameterSlot local-variable slot holding the exact carrier
     * @param addressLocal local-variable slot holding the element address
     * @param broadcast whether to load one scalar and broadcast it to every lane
     */
    void vectorLoad(CarrierAccess access, int parameterSlot, int addressLocal, boolean broadcast) {
        code.getstatic(DOUBLE_VECTOR, "SPECIES_PREFERRED", VECTOR_SPECIES);
        if (broadcast) {
            load(access, parameterSlot, addressLocal);
            code.invokestatic(DOUBLE_VECTOR, "broadcast", MethodTypeDesc.of(DOUBLE_VECTOR,
                    VECTOR_SPECIES, TypeKind.DOUBLE.upperBound()));
            return;
        }
        code.aload(parameterSlot);
        if (access == CarrierAccess.DOUBLE_ARRAY) {
            code.lload(addressLocal).l2i();
            code.invokestatic(DOUBLE_VECTOR, "fromArray", MethodTypeDesc.of(DOUBLE_VECTOR,
                    VECTOR_SPECIES, java.lang.constant.ConstantDescs.CD_double.arrayType(),
                    TypeKind.INT.upperBound()));
        } else {
            code.lload(addressLocal).loadConstant((long) Double.BYTES).lmul();
            code.invokestatic(BYTE_ORDER, "nativeOrder", MethodTypeDesc.of(BYTE_ORDER));
            code.invokestatic(DOUBLE_VECTOR, "fromMemorySegment", MethodTypeDesc.of(DOUBLE_VECTOR,
                    VECTOR_SPECIES, SEGMENT, TypeKind.LONG.upperBound(), BYTE_ORDER));
        }
    }

    /**
     * Emits one unmasked preferred-species FLOAT64 vector store.
     *
     * @param access non-null generation-time-selected direct carrier form
     * @param parameterSlot local-variable slot holding the writable exact carrier
     * @param addressLocal local-variable slot holding the element address
     * @param valueLocal local-variable slot holding the non-null {@code DoubleVector}
     */
    void vectorStore(CarrierAccess access, int parameterSlot, int addressLocal, int valueLocal) {
        code.aload(valueLocal).aload(parameterSlot);
        if (access == CarrierAccess.DOUBLE_ARRAY) {
            code.lload(addressLocal).l2i();
            code.invokevirtual(DOUBLE_VECTOR, "intoArray", MethodTypeDesc.of(
                    TypeKind.VOID.upperBound(), java.lang.constant.ConstantDescs.CD_double.arrayType(),
                    TypeKind.INT.upperBound()));
        } else {
            code.lload(addressLocal).loadConstant((long) Double.BYTES).lmul();
            code.invokestatic(BYTE_ORDER, "nativeOrder", MethodTypeDesc.of(BYTE_ORDER));
            code.invokevirtual(DOUBLE_VECTOR, "intoMemorySegment", MethodTypeDesc.of(
                    TypeKind.VOID.upperBound(), SEGMENT, TypeKind.LONG.upperBound(), BYTE_ORDER));
        }
    }

    private void layout() {
        code.getstatic(VALUE_LAYOUT, "JAVA_DOUBLE_UNALIGNED", DOUBLE_LAYOUT);
        code.invokestatic(BYTE_ORDER, "nativeOrder", MethodTypeDesc.of(BYTE_ORDER));
        code.invokeinterface(VALUE_LAYOUT, "withOrder", MethodTypeDesc.of(VALUE_LAYOUT, BYTE_ORDER));
        code.checkcast(DOUBLE_LAYOUT);
    }
}
