package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.TypeKind;
import java.lang.classfile.Opcode;

/**
 * Emits one allocation-free scalar represented-bit affine boundary copy.
 *
 * <p>The general generated loop reads cold-composed source/result address pairs for the requested
 * half-open range. When specialization proves two dense heap arrays with unit-stride affine
 * progressions, the generated entry narrows its bounds and first address pair once and advances
 * direct integer array indexes. Both forms transfer primitive payloads without conversion and do
 * not inspect a Shape, layout, operation, route, or carrier kind at runtime.</p>
 */
final class CpuAffineCopyEmitter {
    /** Creates one stateless affine copy emitter. */
    CpuAffineCopyEmitter() {
    }

    /**
     * Emits the already-validated two-boundary copy body.
     *
     * @param code non-null Class-File method body builder
     * @param specialization non-null scalar specialization with exactly two compatible carriers
     * @param ir non-null instruction-free encoded affine copy form
     */
    void emit(CodeBuilder code, CpuKernelSpecialization specialization, CpuKernelIr ir) {
        if (specialization.loopAddressing(ir)
                == CpuKernelSpecialization.LoopAddressing.DENSE_HEAP_ARRAY_INT) {
            emitDenseArrayInt(code, specialization, ir);
            return;
        }
        DataType type = ir.values().getFirst().dataType();
        int value = code.allocateLocal(switch (type) {
            case FLOAT64 -> TypeKind.DOUBLE;
            case FLOAT32 -> TypeKind.FLOAT;
            case BFLOAT16, INT32, BOOL -> TypeKind.INT;
            case INT64 -> TypeKind.LONG;
        });
        var carriers = new CpuCarrierEmitter(code);
        int index = code.allocateLocal(TypeKind.LONG);
        int sourceAddress = code.allocateLocal(TypeKind.LONG);
        int resultAddress = code.allocateLocal(TypeKind.LONG);
        code.lload(3).lstore(index);
        var done = code.newLabel();
        var loop = code.newLabel();
        code.labelBinding(loop);
        code.lload(index).lload(5).lcmp().branch(Opcode.IFGE, done);
        code.aload(2).lload(index).loadConstant(2L).lmul().l2i().laload().lstore(sourceAddress);
        code.aload(2).lload(index).loadConstant(2L).lmul().loadConstant(1L).ladd()
                .l2i().laload().lstore(resultAddress);
            carriers.load(type, specialization.carrierPattern().get(0), 0, sourceAddress);
            switch (type) {
                case FLOAT64 -> code.dstore(value);
                case FLOAT32 -> code.fstore(value);
                case BFLOAT16, INT32, BOOL -> code.istore(value);
                case INT64 -> code.lstore(value);
            }
            carriers.store(type, specialization.carrierPattern().get(1), 1,
                    resultAddress, value);
        code.lload(index).loadConstant(1L).ladd().lstore(index);
        code.branch(Opcode.GOTO, loop);
        code.labelBinding(done);
    }

    private static void emitDenseArrayInt(CodeBuilder code,
            CpuKernelSpecialization specialization, CpuKernelIr ir) {
        DataType type = ir.values().getFirst().dataType();
        int value = code.allocateLocal(switch (type) {
            case FLOAT64 -> TypeKind.DOUBLE;
            case FLOAT32 -> TypeKind.FLOAT;
            case BFLOAT16, INT32, BOOL -> TypeKind.INT;
            case INT64 -> TypeKind.LONG;
        });
        int index = code.allocateLocal(TypeKind.INT);
        int end = code.allocateLocal(TypeKind.INT);
        int sourceAddress = code.allocateLocal(TypeKind.INT);
        int resultAddress = code.allocateLocal(TypeKind.INT);
        code.lload(3).l2i().istore(index);
        code.lload(5).l2i().istore(end);
        code.aload(2).iload(index).loadConstant(2).imul().laload().l2i().istore(sourceAddress);
        code.aload(2).iload(index).loadConstant(2).imul().loadConstant(1).iadd()
                .laload().l2i().istore(resultAddress);
        var carriers = new CpuCarrierEmitter(code);
        var done = code.newLabel();
        var loop = code.newLabel();
        code.labelBinding(loop);
        code.iload(index).iload(end).branch(Opcode.IF_ICMPGE, done);
        carriers.load(type, specialization.carrierPattern().get(0), 0, sourceAddress, true);
        switch (type) {
            case FLOAT64 -> code.dstore(value);
            case FLOAT32 -> code.fstore(value);
            case BFLOAT16, INT32, BOOL -> code.istore(value);
            case INT64 -> code.lstore(value);
        }
        carriers.store(type, specialization.carrierPattern().get(1), 1, resultAddress, value, true);
        code.iinc(index, 1).iinc(sourceAddress, 1).iinc(resultAddress, 1)
                .branch(Opcode.GOTO, loop);
        code.labelBinding(done);
    }
}
