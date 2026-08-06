package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.util.function.IntConsumer;

/** Package-private universal primitive start/end loop emitter. */
final class CpuLoopEmitter {
    private final CodeBuilder code;
    /** Creates an emitter over one non-null method-code builder. */
    CpuLoopEmitter(CodeBuilder code) { this.code = code; }

    /** Emits the universal half-open primitive-bound loop and delegates its scalar body. */
    void emit(IntConsumer body) {
        int index = code.allocateLocal(TypeKind.LONG);
        code.lload(4).lstore(index);
        var loop = code.newLabel();
        var done = code.newLabel();
        code.labelBinding(loop);
        code.lload(index).lload(6).lcmp().branch(Opcode.IFGE, done);
        body.accept(index);
        code.lload(index).loadConstant(1L).ladd().lstore(index);
        code.branch(Opcode.GOTO, loop);
        code.labelBinding(done);
    }
}
