package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import java.lang.classfile.CodeBuilder;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.MethodTypeDesc;

/** Package-private direct canonical-dense FLOAT64 segment access emitter. */
final class CpuCarrierEmitter {
    private static final ClassDesc SEGMENT = ClassDesc.of("java.lang.foreign.MemorySegment");
    private static final ClassDesc VALUE_LAYOUT = ClassDesc.of("java.lang.foreign.ValueLayout");
    private static final ClassDesc DOUBLE_LAYOUT =
            ClassDesc.of("java.lang.foreign.ValueLayout$OfDouble");
    private static final ClassDesc BYTE_ORDER = ClassDesc.of("java.nio.ByteOrder");
    private final CodeBuilder code;
    /** Creates an emitter over one non-null method-code builder. */
    CpuCarrierEmitter(CodeBuilder code) { this.code = code; }

    /** Emits a native-order FLOAT64 load from a segment parameter and long index local. */
    void load(int parameterSlot, int indexLocal) {
        code.aload(parameterSlot);
        layout();
        offset(indexLocal);
        code.invokeinterface(SEGMENT, "get", MethodTypeDesc.of(
                java.lang.constant.ConstantDescs.CD_double, DOUBLE_LAYOUT,
                TypeKind.LONG.upperBound()));
    }

    /** Emits a native-order FLOAT64 store to a segment parameter and long index local. */
    void store(int parameterSlot, int indexLocal, int valueLocal) {
        code.aload(parameterSlot);
        layout();
        offset(indexLocal);
        code.dload(valueLocal);
        code.invokeinterface(SEGMENT, "set", MethodTypeDesc.of(
                TypeKind.VOID.upperBound(), DOUBLE_LAYOUT, TypeKind.LONG.upperBound(),
                java.lang.constant.ConstantDescs.CD_double));
    }

    private void offset(int indexLocal) {
        code.lload(indexLocal).loadConstant((long) Double.BYTES).lmul();
    }
    private void layout() {
        code.getstatic(VALUE_LAYOUT, "JAVA_DOUBLE_UNALIGNED", DOUBLE_LAYOUT);
        code.invokestatic(BYTE_ORDER, "nativeOrder", MethodTypeDesc.of(BYTE_ORDER));
        code.invokeinterface(VALUE_LAYOUT, "withOrder", MethodTypeDesc.of(VALUE_LAYOUT, BYTE_ORDER));
        code.checkcast(DOUBLE_LAYOUT);
    }
}
