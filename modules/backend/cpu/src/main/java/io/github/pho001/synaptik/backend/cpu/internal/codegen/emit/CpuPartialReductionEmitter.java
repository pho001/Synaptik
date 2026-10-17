package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPartialReductionIr;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDescs;
import java.lang.constant.MethodTypeDesc;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Emits the two direct primitive-array bodies of one admitted modular partial reduction.
 *
 * <p>The partial method owns precisely one {@code long[]} state index and folds its supplied
 * half-open primitive input range into that index.  The combine method owns output publication:
 * it reads the baked number of states for each requested cell in increasing partial ordinal and
 * writes that cell exactly once.  Kind, represented width, and partial count are generation-time
 * facts, leaving neither a semantic switch nor a carrier lookup in either hot loop.</p>
 */
final class CpuPartialReductionEmitter {
    private static final ClassDesc SEGMENT = ClassDesc.of(MemorySegment.class.getName());
    private static final ClassDesc VALUE_LAYOUT = ClassDesc.of(ValueLayout.class.getName());
    private static final ClassDesc INT_LAYOUT = ClassDesc.of("java.lang.foreign.ValueLayout$OfInt");
    private static final ClassDesc LONG_LAYOUT = ClassDesc.of("java.lang.foreign.ValueLayout$OfLong");
    /**
     * Creates a stateless emitter whose only inputs are prepared partial identity facts.
     *
     * <p>It retains no arrays, state workspace, route decision, or worker ownership between
     * generated methods.</p>
     */
    CpuPartialReductionEmitter() { }

    /**
     * Emits one direct primitive-array interval fold and one state-slot write.
     *
     * @param code non-null method builder receiving the partial-body loop
     * @param ir non-null baked type, operation, and partial-count identity
     */
    void emitPartial(CodeBuilder code, CpuPartialReductionIr ir) {
        int index = code.allocateLocal(TypeKind.INT);
        int accumulator = code.allocateLocal(ir.dataType()
                == io.github.pho001.synaptik.model.datatype.DataType.INT32
                ? TypeKind.INT : TypeKind.LONG);
        code.iload(1).istore(index);
        if (ir.dataType() == io.github.pho001.synaptik.model.datatype.DataType.INT32) {
            code.loadConstant(ir.kind() == CpuPartialReductionIr.Kind.SUM ? 0 : 1)
                    .istore(accumulator);
        } else {
            code.loadConstant(ir.kind() == CpuPartialReductionIr.Kind.SUM ? 0L : 1L)
                    .lstore(accumulator);
        }
        var loop = code.newLabel(); var done = code.newLabel();
        code.labelBinding(loop).iload(index).iload(2).branch(Opcode.IF_ICMPGE, done);
        if (ir.dataType() == io.github.pho001.synaptik.model.datatype.DataType.INT32) {
            code.iload(accumulator).aload(0).iload(index).iaload();
            if (ir.kind() == CpuPartialReductionIr.Kind.SUM) code.iadd(); else code.imul();
            code.istore(accumulator);
        } else {
            code.lload(accumulator).aload(0).iload(index).laload();
            if (ir.kind() == CpuPartialReductionIr.Kind.SUM) code.ladd(); else code.lmul();
            code.lstore(accumulator);
        }
        code.iinc(index, 1).branch(Opcode.GOTO, loop).labelBinding(done);
        code.aload(3).getstatic(VALUE_LAYOUT, ir.dataType()
                == io.github.pho001.synaptik.model.datatype.DataType.INT32 ? "JAVA_INT" : "JAVA_LONG",
                ir.dataType() == io.github.pho001.synaptik.model.datatype.DataType.INT32
                        ? INT_LAYOUT : LONG_LAYOUT).lload(4);
        if (ir.dataType() == io.github.pho001.synaptik.model.datatype.DataType.INT32) {
            code.iload(accumulator).invokeinterface(SEGMENT, "set", MethodTypeDesc.of(
                    ConstantDescs.CD_void, INT_LAYOUT, ConstantDescs.CD_long, ConstantDescs.CD_int));
        } else {
            code.lload(accumulator).invokeinterface(SEGMENT, "set", MethodTypeDesc.of(
                    ConstantDescs.CD_void, LONG_LAYOUT, ConstantDescs.CD_long, ConstantDescs.CD_long));
        }
        code.return_();
    }

    /**
     * Emits ascending-partial state consumption and contiguous output-cell publication.
     *
     * @param code non-null method builder receiving the ordered combine loops
     * @param ir non-null baked type, operation, and partial-count identity
     */
    void emitCombine(CodeBuilder code, CpuPartialReductionIr ir) {
        int cell = code.allocateLocal(TypeKind.INT);
        int state = code.allocateLocal(TypeKind.INT);
        int partial = code.allocateLocal(TypeKind.INT);
        int accumulator = code.allocateLocal(ir.dataType()
                == io.github.pho001.synaptik.model.datatype.DataType.INT32
                ? TypeKind.INT : TypeKind.LONG);
        code.iload(1).istore(cell);
        var cells = code.newLabel(); var done = code.newLabel();
        var partials = code.newLabel(); var write = code.newLabel();
        code.labelBinding(cells).iload(cell).iload(2).branch(Opcode.IF_ICMPGE, done);
        code.iload(cell).loadConstant(ir.partialCount()).imul().istore(state);
        code.loadConstant(0).istore(partial);
        if (ir.dataType() == io.github.pho001.synaptik.model.datatype.DataType.INT32) {
            code.loadConstant(ir.kind() == CpuPartialReductionIr.Kind.SUM ? 0 : 1)
                    .istore(accumulator);
        } else {
            code.loadConstant(ir.kind() == CpuPartialReductionIr.Kind.SUM ? 0L : 1L)
                    .lstore(accumulator);
        }
        code.labelBinding(partials).iload(partial).loadConstant(ir.partialCount())
                .branch(Opcode.IF_ICMPGE, write);
        if (ir.dataType() == io.github.pho001.synaptik.model.datatype.DataType.INT32) {
            code.iload(accumulator).aload(0).getstatic(VALUE_LAYOUT, "JAVA_INT", INT_LAYOUT)
                    .iload(state).i2l().loadConstant(CpuPartialReductionIr.STATE_SLICE_BYTES).lmul()
                    .invokeinterface(SEGMENT, "get", MethodTypeDesc.of(ConstantDescs.CD_int,
                            INT_LAYOUT, ConstantDescs.CD_long));
            if (ir.kind() == CpuPartialReductionIr.Kind.SUM) code.iadd(); else code.imul();
            code.istore(accumulator);
        } else {
            code.lload(accumulator).aload(0).getstatic(VALUE_LAYOUT, "JAVA_LONG", LONG_LAYOUT)
                    .iload(state).i2l().loadConstant(CpuPartialReductionIr.STATE_SLICE_BYTES).lmul()
                    .invokeinterface(SEGMENT, "get", MethodTypeDesc.of(ConstantDescs.CD_long,
                            LONG_LAYOUT, ConstantDescs.CD_long));
            if (ir.kind() == CpuPartialReductionIr.Kind.SUM) code.ladd(); else code.lmul();
            code.lstore(accumulator);
        }
        code.iinc(state, 1).iinc(partial, 1).branch(Opcode.GOTO, partials);
        code.labelBinding(write).aload(3).iload(4).iload(cell).iadd();
        if (ir.dataType() == io.github.pho001.synaptik.model.datatype.DataType.INT32)
            code.iload(accumulator).iastore();
        else code.lload(accumulator).lastore();
        code.iinc(cell, 1).branch(Opcode.GOTO, cells).labelBinding(done).return_();
    }
}
