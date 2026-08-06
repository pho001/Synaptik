package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.reference.CpuScalarReferenceKernel;
import java.lang.classfile.CodeBuilder;
import java.lang.constant.ClassDesc;
import java.lang.constant.MethodTypeDesc;

/** Package-private exact scalar semantic emitter. */
final class CpuScalarEmitter {
    private static final ClassDesc REFERENCE = ClassDesc.of(
            CpuScalarReferenceKernel.class.getName());
    private final CodeBuilder code;
    /** Creates an emitter over one non-null method-code builder. */
    CpuScalarEmitter(CodeBuilder code) { this.code = code; }
    /** Emits the fixed-order exact-GELU scalar realization for the stack-top value. */
    void gelu() {
        code.invokestatic(REFERENCE, "gelu", MethodTypeDesc.of(
                java.lang.constant.ConstantDescs.CD_double,
                java.lang.constant.ConstantDescs.CD_double));
    }
}
