package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDescs;
import java.lang.constant.MethodTypeDesc;

/**
 * Emits one exact Model CAST conversion boundary as primitive generated instructions.
 *
 * <p>The emitter covers all 36 ordered pairs among FLOAT64, FLOAT32, BFLOAT16, INT64, INT32,
 * and BOOL. Same-type conversions reload represented bits; lossy floating NaNs become the
 * positive canonical target NaN; widening preserves the Model-defined NaN mapping; floating-to-
 * integral conversion truncates and saturates with NaN mapped to zero; INT64-to-INT32 retains low
 * bits; and BOOL results are canonical zero or one. FLOAT64 and integral values round directly to
 * BFLOAT16 without a FLOAT32 intermediate. The emitter runs only while a class is built. It
 * contributes no field, helper call, dispatch object, or allocation to generated element work.</p>
 */
final class CpuCastEmitter {
    private static final ClassDesc FLOAT = ClassDesc.of(Float.class.getName());
    private static final ClassDesc DOUBLE = ClassDesc.of(Double.class.getName());
    private static final ClassDesc LONG = ClassDesc.of(Long.class.getName());
    private final CodeBuilder code;
    /**
     * Creates a generation-time emitter for one generated method body.
     *
     * @param code non-null Class-File API builder; not retained by generated code
     */
    CpuCastEmitter(CodeBuilder code) { this.code = code; }

    /**
     * Leaves one target-typed represented value on the generated operand stack.
     *
     * @param source non-null represented type of the value in {@code local}
     * @param target non-null requested represented result type
     * @param local valid local-variable slot containing a primitive value of {@code source}
     * @throws NullPointerException if {@code source} or {@code target} is {@code null}
     * @throws AssertionError if either type is outside the six current Model data types
     */
    void emit(DataType source, DataType target, int local) {
        if (source == target) { raw(source, local); return; }
        if (target == DataType.BFLOAT16) { bfloat(source, local); return; }
        if (target == DataType.FLOAT64 && (source == DataType.FLOAT32 || source == DataType.BFLOAT16)) {
            widen(source, local); return;
        }
        value(source, local);
        switch (target) {
            case FLOAT64 -> { if (source == DataType.INT64) code.l2d(); else if (source == DataType.INT32 || source == DataType.BOOL) code.i2d(); }
            case FLOAT32 -> { if (source == DataType.FLOAT64) narrowFloat(local); else if (source == DataType.INT64) code.l2f(); else if (source == DataType.INT32 || source == DataType.BOOL) code.i2f(); }
            case INT64 -> { if (source == DataType.FLOAT64) code.d2l(); else if (source == DataType.FLOAT32 || source == DataType.BFLOAT16) code.f2l(); else code.i2l(); }
            case INT32 -> { if (source == DataType.FLOAT64) code.d2i(); else if (source == DataType.FLOAT32 || source == DataType.BFLOAT16) code.f2i(); else if (source == DataType.INT64) code.l2i(); }
            case BOOL -> bool(source);
            default -> throw new AssertionError(target);
        }
    }

    private void narrowFloat(int local) {
        // Re-load raw input for the Model's positive canonical NaN rule.
        code.pop2(); int bits = code.allocateLocal(TypeKind.LONG);
        code.dload(local).invokestatic(DOUBLE, "doubleToRawLongBits", MethodTypeDesc.of(ConstantDescs.CD_long, ConstantDescs.CD_double)).lstore(bits);
        var normal=code.newLabel(); var done=code.newLabel();
        code.lload(bits).loadConstant(0x7ff0000000000000L).land().loadConstant(0x7ff0000000000000L).lcmp().branch(Opcode.IFNE,normal);
        code.lload(bits).loadConstant(0x000fffffffffffffL).land().loadConstant(0L).lcmp().branch(Opcode.IFEQ,normal);
        code.loadConstant(Float.intBitsToFloat(0x7fc00000)).branch(Opcode.GOTO,done);
        code.labelBinding(normal).dload(local).d2f().labelBinding(done);
    }

    private void widen(DataType source,int local) {
        int bits=code.allocateLocal(TypeKind.INT); var normal=code.newLabel(); var done=code.newLabel();
        if(source==DataType.FLOAT32) {
            code.fload(local).invokestatic(FLOAT,"floatToRawIntBits",MethodTypeDesc.of(ConstantDescs.CD_int,ConstantDescs.CD_float)).istore(bits);
            code.iload(bits).loadConstant(0x7f800000).iand().loadConstant(0x7f800000).branch(Opcode.IF_ICMPNE,normal);
            code.iload(bits).loadConstant(0x007fffff).iand().branch(Opcode.IFEQ,normal);
            code.iload(bits).i2l().loadConstant(0x80000000L).land().loadConstant(32).lshl().loadConstant(0x7ff0000000000000L).lor().iload(bits).loadConstant(0x007fffff).iand().i2l().loadConstant(29).lshl().lor().invokestatic(DOUBLE,"longBitsToDouble",MethodTypeDesc.of(ConstantDescs.CD_double,ConstantDescs.CD_long)).branch(Opcode.GOTO,done);
            code.labelBinding(normal).fload(local).f2d().labelBinding(done);
        } else {
            code.iload(local).loadConstant(0xffff).iand().istore(bits);
            code.iload(bits).loadConstant(0x7f80).iand().loadConstant(0x7f80).branch(Opcode.IF_ICMPNE,normal);
            code.iload(bits).loadConstant(0x007f).iand().branch(Opcode.IFEQ,normal);
            code.iload(bits).loadConstant(0x8000).iand().i2l().loadConstant(48).lshl().loadConstant(0x7ff0000000000000L).lor().iload(bits).loadConstant(0x007f).iand().i2l().loadConstant(45).lshl().lor().invokestatic(DOUBLE,"longBitsToDouble",MethodTypeDesc.of(ConstantDescs.CD_double,ConstantDescs.CD_long)).branch(Opcode.GOTO,done);
            code.labelBinding(normal).iload(local).loadConstant(16).ishl().invokestatic(FLOAT,"intBitsToFloat",MethodTypeDesc.of(ConstantDescs.CD_float,ConstantDescs.CD_int)).f2d().labelBinding(done);
        }
    }

    private void bfloat(DataType s,int l) {
        switch(s) {
            case FLOAT32 -> floatBfloat(l);
            case FLOAT64 -> doubleBfloat(l);
            case INT64 -> integerBfloat(l,true);
            case INT32 -> integerBfloat(l,false);
            case BOOL -> {var no=code.newLabel();var done=code.newLabel();code.iload(l).branch(Opcode.IFEQ,no).loadConstant(0x3f80).branch(Opcode.GOTO,done).labelBinding(no).loadConstant(0).labelBinding(done);}
            default -> throw new AssertionError(s);
        }
    }
    private void floatBfloat(int l) { int b=code.allocateLocal(TypeKind.INT);var finite=code.newLabel();var done=code.newLabel();code.fload(l).invokestatic(FLOAT,"floatToRawIntBits",MethodTypeDesc.of(ConstantDescs.CD_int,ConstantDescs.CD_float)).istore(b);code.iload(b).loadConstant(0x7fffffff).iand().loadConstant(0x7f800000).branch(Opcode.IF_ICMPLE,finite).loadConstant(0x7fc0).branch(Opcode.GOTO,done);code.labelBinding(finite).iload(b).loadConstant(0x7fff).iadd().iload(b).loadConstant(16).iushr().loadConstant(1).iand().iadd().loadConstant(16).iushr().labelBinding(done); }
    private void integerBfloat(int l,boolean wide) { int v=code.allocateLocal(TypeKind.LONG),m=code.allocateLocal(TypeKind.LONG),e=code.allocateLocal(TypeKind.INT),r=code.allocateLocal(TypeKind.LONG),sign=code.allocateLocal(TypeKind.INT),out=code.allocateLocal(TypeKind.INT);if(wide)code.lload(l);else code.iload(l).i2l();code.lstore(v);var nz=code.newLabel();var neg=code.newLabel();var mag=code.newLabel();var right=code.newLabel();var ready=code.newLabel();var carry=code.newLabel();var finish=code.newLabel();code.lload(v).loadConstant(0L).lcmp().branch(Opcode.IFNE,nz).loadConstant(0).istore(out).branch(Opcode.GOTO,finish).labelBinding(nz);code.lload(v).loadConstant(0L).lcmp().branch(Opcode.IFLT,neg).loadConstant(0).istore(sign).lload(v).lstore(m).branch(Opcode.GOTO,mag).labelBinding(neg).loadConstant(0x8000).istore(sign).lload(v).lneg().lstore(m).labelBinding(mag);code.loadConstant(63).lload(m).invokestatic(LONG,"numberOfLeadingZeros",MethodTypeDesc.of(ConstantDescs.CD_int,ConstantDescs.CD_long)).isub().istore(e);code.iload(e).loadConstant(7).branch(Opcode.IF_ICMPGT,right).lload(m).loadConstant(7).iload(e).isub().lshl().lstore(r).branch(Opcode.GOTO,ready).labelBinding(right);int sh=code.allocateLocal(TypeKind.INT);code.iload(e).loadConstant(7).isub().istore(sh);round(m,sh,r);code.labelBinding(ready);code.lload(r).loadConstant(0x100L).lcmp().branch(Opcode.IFNE,carry).loadConstant(0x80L).lstore(r).iinc(e,1).labelBinding(carry);code.iload(sign).iload(e).loadConstant(127).iadd().loadConstant(7).ishl().ior().lload(r).l2i().loadConstant(0x7f).iand().ior().istore(out).labelBinding(finish).iload(out); }
    private void round(int value,int shift,int result) { var big=code.newLabel();var inc=code.newLabel();var done=code.newLabel();code.iload(shift).loadConstant(63).branch(Opcode.IF_ICMPGE,big);code.lload(value).iload(shift).lushr().lstore(result);int d=code.allocateLocal(TypeKind.LONG),mid=code.allocateLocal(TypeKind.LONG);code.loadConstant(1L).iload(shift).lshl().loadConstant(1L).lsub().lload(value).land().lstore(d);code.loadConstant(1L).iload(shift).loadConstant(1).isub().lshl().lstore(mid);code.lload(d).lload(mid).lcmp().branch(Opcode.IFGT,inc);code.lload(d).lload(mid).lcmp().branch(Opcode.IFNE,done);code.lload(result).loadConstant(1L).land().loadConstant(0L).lcmp().branch(Opcode.IFNE,inc).branch(Opcode.GOTO,done);code.labelBinding(inc).lload(result).loadConstant(1L).ladd().lstore(result).branch(Opcode.GOTO,done);code.labelBinding(big).loadConstant(0L).lstore(result).labelBinding(done); }
    private void doubleBfloat(int local) {
        // Keep the exact direct binary64-to-BFLOAT16 conversion compact enough for C2 to inline
        // the generated entry.  Every potentially nonzero subnormal has raw exponent 889..896;
        // lower finite values round to signed zero, so they need no general variable-shift body.
        int bits = code.allocateLocal(TypeKind.LONG);
        int sign = code.allocateLocal(TypeKind.INT);
        int rawExponent = code.allocateLocal(TypeKind.INT);
        int shift = code.allocateLocal(TypeKind.INT);
        int output = code.allocateLocal(TypeKind.INT);
        int significand = code.allocateLocal(TypeKind.LONG);
        int rounded = code.allocateLocal(TypeKind.LONG);
        code.dload(local).invokestatic(DOUBLE, "doubleToRawLongBits",
                MethodTypeDesc.of(ConstantDescs.CD_long, ConstantDescs.CD_double)).lstore(bits);
        code.lload(bits).loadConstant(48).lushr().l2i().loadConstant(0x8000).iand().istore(sign);
        code.lload(bits).loadConstant(52).lushr().l2i().loadConstant(0x7ff).iand()
                .istore(rawExponent);

        var special = code.newLabel();
        var overflow = code.newLabel();
        var underflow = code.newLabel();
        var subnormal = code.newLabel();
        var round = code.newLabel();
        var carry = code.newLabel();
        var compose = code.newLabel();
        var nan = code.newLabel();
        var done = code.newLabel();
        code.iload(rawExponent).loadConstant(0x7ff).branch(Opcode.IF_ICMPEQ, special);
        code.iload(rawExponent).loadConstant(1150).branch(Opcode.IF_ICMPGT, overflow);
        code.iload(rawExponent).loadConstant(889).branch(Opcode.IF_ICMPLT, underflow);
        code.iload(rawExponent).loadConstant(897).branch(Opcode.IF_ICMPLT, subnormal);
        code.loadConstant(45).istore(shift).branch(Opcode.GOTO, round);
        code.labelBinding(subnormal).loadConstant(942).iload(rawExponent).isub().istore(shift);

        // RNE without a branchy discarded/midpoint decision:
        // (value + 2^(shift-1) - 1 + retainedParity) >>> shift.
        code.labelBinding(round).loadConstant(1L << 52).lload(bits)
                .loadConstant(0x000f_ffff_ffff_ffffL).land().lor().lstore(significand);
        code.lload(significand)
                .loadConstant(1L).iload(shift).loadConstant(1).isub().lshl().ladd()
                .loadConstant(1L).lsub()
                .lload(significand).iload(shift).lushr().loadConstant(1L).land().ladd()
                .iload(shift).lushr().lstore(rounded);
        code.iload(rawExponent).loadConstant(897).branch(Opcode.IF_ICMPLT, compose);
        code.lload(rounded).loadConstant(0x100L).lcmp().branch(Opcode.IFNE, carry);
        code.loadConstant(0x80L).lstore(rounded).iinc(rawExponent, 1);
        code.labelBinding(carry).iload(rawExponent).loadConstant(1150)
                .branch(Opcode.IF_ICMPGT, overflow);
        code.iload(sign).iload(rawExponent).loadConstant(896).isub().loadConstant(7).ishl()
                .ior().lload(rounded).l2i().loadConstant(0x7f).iand().ior().istore(output)
                .branch(Opcode.GOTO, done);
        code.labelBinding(compose).iload(sign).lload(rounded).l2i().ior().istore(output)
                .branch(Opcode.GOTO, done);
        code.labelBinding(underflow).iload(sign).istore(output).branch(Opcode.GOTO, done);
        code.labelBinding(overflow).iload(sign).loadConstant(0x7f80).ior().istore(output)
                .branch(Opcode.GOTO, done);
        code.labelBinding(special).lload(bits).loadConstant(0x000f_ffff_ffff_ffffL).land()
                .loadConstant(0L).lcmp().branch(Opcode.IFNE, nan)
                .iload(sign).loadConstant(0x7f80).ior().istore(output).branch(Opcode.GOTO, done);
        code.labelBinding(nan).loadConstant(0x7fc0).istore(output);
        code.labelBinding(done).iload(output);
    }
    private void bool(DataType s) {var yes=code.newLabel();var done=code.newLabel();switch(s){case FLOAT64->code.loadConstant(0d).dcmpl().branch(Opcode.IFNE,yes);case FLOAT32,BFLOAT16->code.loadConstant(0f).fcmpl().branch(Opcode.IFNE,yes);case INT64->code.loadConstant(0L).lcmp().branch(Opcode.IFNE,yes);case INT32,BOOL->code.branch(Opcode.IFNE,yes);}code.loadConstant(0).branch(Opcode.GOTO,done).labelBinding(yes).loadConstant(1).labelBinding(done);}
    private void raw(DataType t,int l){switch(t){case FLOAT64->code.dload(l);case FLOAT32->code.fload(l);case BFLOAT16,INT32,BOOL->code.iload(l);case INT64->code.lload(l);}}
    private void value(DataType t,int l){if(t==DataType.BFLOAT16)code.iload(l).loadConstant(16).ishl().invokestatic(FLOAT,"intBitsToFloat",MethodTypeDesc.of(ConstantDescs.CD_float,ConstantDescs.CD_int));else raw(t,l);}
}
