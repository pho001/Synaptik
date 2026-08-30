package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDescs;
import java.lang.constant.MethodTypeDesc;

/**
 * Emits allocation-free scalar channels-first (NCHW) max and fixed-divisor average Pool2d loops.
 *
 * <p>Each generated invocation owns a half-open range of complete logical output cells. Concrete
 * geometry stays in the cold-bound primitive geometry carrier, while the hot body performs direct
 * typed array or segment access, visits each cell's full dilated window, and writes that cell once.
 * The emitter does not allocate window storage, split a window across workers, or call a
 * Synaptik-owned numerical helper.
 */
public final class CpuPool2dEmitter {
    private static final ClassDesc FLOAT = ClassDesc.of(Float.class.getName());
    private static final ClassDesc DOUBLE = ClassDesc.of(Double.class.getName());

    /** Creates a stateless Pool2d Class-File emitter. */
    public CpuPool2dEmitter() {}

    /**
     * Emits the selected family over a half-open range of complete output cells.
     *
     * @param code non-null Class-File method builder
     * @param specialization exact scalar carrier specialization
     * @param ir canonical Pool2d kernel identity
     * @throws NullPointerException if a required reference is {@code null}
     * @throws IllegalArgumentException if identity, boundary, or strategy facts disagree
     */
    public void emit(CodeBuilder code, CpuKernelSpecialization specialization, CpuKernelIr ir) {
        boolean max = ir.familyIdentity().startsWith("pool2d:kind=MAX:");
        boolean average = ir.familyIdentity().startsWith("pool2d:kind=AVERAGE:");
        DataType type = specialization.boundaryDataTypes().getFirst();
        if ((!max && !average)
                || specialization.carrierPattern().size() != 2
                || specialization.boundaryDataTypes().size() != 2
                || specialization.boundaryDataTypes().get(1) != type
                || type != DataType.BFLOAT16 && type != DataType.FLOAT32 && type != DataType.FLOAT64
                || specialization.classIdentitySchema() != 55
                || !ir.familyIdentity().contains(":type=" + type + ":")
                || !ir.familyIdentity().endsWith(":realization=DIRECT_SCALAR")
                || specialization.executionStrategy().compute()
                        != io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan
                                .ExecutionStrategy.Compute.SCALAR)
            throw new IllegalArgumentException("Pool2d generated facts disagree");
        var carriers = new CpuCarrierEmitter(code);
        int geometry = 2, start = 3, end = 5;
        int cell = l(code),
                r = l(code),
                n = l(code),
                c = l(code),
                oh = l(code),
                ow = l(code),
                kh = l(code),
                kw = l(code),
                ih = l(code),
                iw = l(code),
                inAddress = l(code),
                outAddress = l(code);
        code.lload(start).lstore(cell);
        var cells = code.newLabel();
        var done = code.newLabel();
        code.labelBinding(cells).lload(cell).lload(end).lcmp().branch(Opcode.IFGE, done);
        code.lload(cell).lstore(r);
        decode(code, geometry, 13, r, ow);
        decode(code, geometry, 12, r, oh);
        decode(code, geometry, 11, r, c);
        code.lload(r).lstore(n);
        code.lload(oh);
        g(code, geometry, 20).lmul();
        g(code, geometry, 22).lsub().lstore(ih);
        code.lload(ow);
        g(code, geometry, 21).lmul();
        g(code, geometry, 23).lsub().lstore(iw);
        address(code, geometry, 1, 14, n, c, oh, ow, outAddress);
        if (max)
            emitMax(
                    code,
                    carriers,
                    specialization,
                    type,
                    geometry,
                    n,
                    c,
                    kh,
                    kw,
                    ih,
                    iw,
                    inAddress,
                    outAddress);
        else
            emitAverage(
                    code,
                    carriers,
                    specialization,
                    type,
                    geometry,
                    n,
                    c,
                    kh,
                    kw,
                    ih,
                    iw,
                    inAddress,
                    outAddress);
        code.lload(cell)
                .loadConstant(1L)
                .ladd()
                .lstore(cell)
                .branch(Opcode.GOTO, cells)
                .labelBinding(done);
    }

    private static void emitMax(
            CodeBuilder code,
            CpuCarrierEmitter carriers,
            CpuKernelSpecialization s,
            DataType type,
            int geometry,
            int n,
            int c,
            int kh,
            int kw,
            int ih0,
            int iw0,
            int address,
            int output) {
        int found = i(code),
                represented = represented(code, type),
                winner = represented(code, type),
                value = value(code, type),
                best = value(code, type),
                ih = l(code),
                iw = l(code);
        code.loadConstant(0).istore(found);
        negativeInfinity(code, type, winner);
        if (type == DataType.FLOAT64) code.loadConstant(Double.NEGATIVE_INFINITY).dstore(best);
        else code.loadConstant(Float.NEGATIVE_INFINITY).fstore(best);
        code.loadConstant(0L).lstore(kh);
        var hs = code.newLabel();
        var hd = code.newLabel();
        var finishWindow = code.newLabel();
        code.labelBinding(hs).lload(kh);
        g(code, geometry, 18).lcmp().branch(Opcode.IFGE, hd);
        code.lload(ih0).lload(kh);
        g(code, geometry, 24).lmul().ladd().lstore(ih);
        code.loadConstant(0L).lstore(kw);
        var ws = code.newLabel();
        var wd = code.newLabel();
        code.labelBinding(ws).lload(kw);
        g(code, geometry, 19).lcmp().branch(Opcode.IFGE, wd);
        code.lload(iw0).lload(kw);
        g(code, geometry, 25).lmul().ladd().lstore(iw);
        var skip = code.newLabel();
        bounds(code, geometry, ih, iw, skip);
        address(code, geometry, 0, 6, n, c, ih, iw, address);
        carriers.loadFrozen(type, s.carrierPattern().getFirst(), 0, address, false);
        storeRepresented(code, type, represented);
        decode(code, type, represented, value);
        var nonNan = code.newLabel();
        loadValue(code, type, value)
                .invokestatic(
                        type == DataType.FLOAT64 ? DOUBLE : FLOAT,
                        "isNaN",
                        MethodTypeDesc.of(
                                ConstantDescs.CD_boolean,
                                type == DataType.FLOAT64 ? ConstantDescs.CD_double : ConstantDescs.CD_float))
                .branch(Opcode.IFEQ, nonNan);
        copyRepresented(code, type, represented, winner);
        code.loadConstant(1).istore(found);
        code.branch(Opcode.GOTO, finishWindow).labelBinding(nonNan);
        var choose = code.newLabel();
        var keep = code.newLabel();
        code.iload(found).branch(Opcode.IFEQ, choose);
        compareGreater(code, type, value, best, choose);
        var notBothZero = code.newLabel();
        loadValue(code, type, value);
        zero(code, type);
        compareEqual(code, type, notBothZero);
        loadValue(code, type, best);
        zero(code, type);
        compareNotEqual(code, type, keep);
        rawPositive(code, type, represented, choose);
        code.branch(Opcode.GOTO, keep).labelBinding(notBothZero).branch(Opcode.GOTO, keep);
        code.labelBinding(choose);
        copyRepresented(code, type, represented, winner);
        copyValue(code, type, value, best);
        code.loadConstant(1).istore(found);
        code.labelBinding(keep).labelBinding(skip);
        code.lload(kw).loadConstant(1L).ladd().lstore(kw).branch(Opcode.GOTO, ws).labelBinding(wd);
        code.lload(kh)
                .loadConstant(1L)
                .ladd()
                .lstore(kh)
                .branch(Opcode.GOTO, hs)
                .labelBinding(hd)
                .labelBinding(finishWindow);
        var have = code.newLabel();
        code.iload(found).branch(Opcode.IFNE, have);
        negativeInfinity(code, type, winner);
        code.labelBinding(have);
        carriers.storeFrozen(type, s.carrierPattern().get(1), 1, output, winner, false);
    }

    private static void emitAverage(
            CodeBuilder code,
            CpuCarrierEmitter carriers,
            CpuKernelSpecialization s,
            DataType type,
            int geometry,
            int n,
            int c,
            int kh,
            int kw,
            int ih0,
            int iw0,
            int address,
            int output) {
        int represented = represented(code, type),
                value = value(code, type),
                sum = value(code, type),
                ih = l(code),
                iw = l(code),
                allNeg = i(code);
        zero(code, type);
        storeValue(code, type, sum);
        code.loadConstant(1).istore(allNeg);
        code.loadConstant(0L).lstore(kh);
        var hs = code.newLabel();
        var hd = code.newLabel();
        code.labelBinding(hs).lload(kh);
        g(code, geometry, 18).lcmp().branch(Opcode.IFGE, hd);
        code.lload(ih0).lload(kh);
        g(code, geometry, 24).lmul().ladd().lstore(ih);
        code.loadConstant(0L).lstore(kw);
        var ws = code.newLabel();
        var wd = code.newLabel();
        var padding = code.newLabel();
        var next = code.newLabel();
        code.labelBinding(ws).lload(kw);
        g(code, geometry, 19).lcmp().branch(Opcode.IFGE, wd);
        code.lload(iw0).lload(kw);
        g(code, geometry, 25).lmul().ladd().lstore(iw);
        bounds(code, geometry, ih, iw, padding);
        address(code, geometry, 0, 6, n, c, ih, iw, address);
        carriers.loadFrozen(type, s.carrierPattern().getFirst(), 0, address, false);
        storeRepresented(code, type, represented);
        decode(code, type, represented, value);
        loadValue(code, type, sum);
        loadValue(code, type, value);
        if (type == DataType.FLOAT64) code.dadd().dstore(sum);
        else code.fadd().fstore(sum);
        rawNegativeZero(code, type, represented, allNeg);
        code.branch(Opcode.GOTO, next)
                .labelBinding(padding)
                .loadConstant(0)
                .istore(allNeg)
                .labelBinding(next);
        code.lload(kw).loadConstant(1L).ladd().lstore(kw).branch(Opcode.GOTO, ws).labelBinding(wd);
        code.lload(kh).loadConstant(1L).ladd().lstore(kh).branch(Opcode.GOTO, hs).labelBinding(hd);
        loadValue(code, type, sum);
        g(code, geometry, 26);
        if (type == DataType.FLOAT64) code.l2d().ddiv().dstore(sum);
        else code.l2f().fdiv().fstore(sum);
        var store = code.newLabel();
        var positiveZero = code.newLabel();
        loadValue(code, type, sum);
        zero(code, type);
        compareNotEqual(code, type, store);
        code.iload(allNeg).branch(Opcode.IFEQ, positiveZero);
        negativeZero(code, type);
        storeValue(code, type, sum);
        code.branch(Opcode.GOTO, store).labelBinding(positiveZero);
        zero(code, type);
        storeValue(code, type, sum);
        code.labelBinding(store);
        if (type == DataType.BFLOAT16) {
            int result = code.allocateLocal(TypeKind.DOUBLE);
            code.fload(sum).f2d().dstore(result);
            CpuNormEmitter.emitStore(code, carriers, s, type, 1, output, result, false, true);
        } else carriers.storeFrozen(type, s.carrierPattern().get(1), 1, output, sum, false);
    }

    private static void bounds(CodeBuilder c, int g, int h, int w, java.lang.classfile.Label skip) {
        c.lload(h).loadConstant(0L).lcmp().branch(Opcode.IFLT, skip);
        c.lload(w).loadConstant(0L).lcmp().branch(Opcode.IFLT, skip);
        c.lload(h);
        g(c, g, 4).lcmp().branch(Opcode.IFGE, skip);
        c.lload(w);
        g(c, g, 5).lcmp().branch(Opcode.IFGE, skip);
    }

    private static void address(
            CodeBuilder c, int g, int base, int strides, int n, int ch, int h, int w, int target) {
        g(c, g, base);
        c.lload(n);
        g(c, g, strides).lmul().ladd().lload(ch);
        g(c, g, strides + 1).lmul().ladd().lload(h);
        g(c, g, strides + 2).lmul().ladd().lload(w);
        g(c, g, strides + 3).lmul().ladd().lstore(target);
    }

    private static void decode(CodeBuilder c, int g, int extent, int r, int out) {
        c.lload(r);
        g(c, g, extent).lrem().lstore(out);
        c.lload(r);
        g(c, g, extent).ldiv().lstore(r);
    }

    private static CodeBuilder g(CodeBuilder c, int geometry, int index) {
        return CpuNormEmitter.geometry(c, geometry, index);
    }

    private static int l(CodeBuilder c) {
        return c.allocateLocal(TypeKind.LONG);
    }

    private static int i(CodeBuilder c) {
        return c.allocateLocal(TypeKind.INT);
    }

    private static int represented(CodeBuilder c, DataType t) {
        return c.allocateLocal(
                t == DataType.FLOAT64
                        ? TypeKind.DOUBLE
                        : t == DataType.FLOAT32 ? TypeKind.FLOAT : TypeKind.INT);
    }

    private static int value(CodeBuilder c, DataType t) {
        return c.allocateLocal(t == DataType.FLOAT64 ? TypeKind.DOUBLE : TypeKind.FLOAT);
    }

    private static void storeRepresented(CodeBuilder c, DataType t, int x) {
        if (t == DataType.FLOAT64) c.dstore(x);
        else if (t == DataType.FLOAT32) c.fstore(x);
        else c.istore(x);
    }

    private static void copyRepresented(CodeBuilder c, DataType t, int a, int b) {
        if (t == DataType.FLOAT64) c.dload(a).dstore(b);
        else if (t == DataType.FLOAT32) c.fload(a).fstore(b);
        else c.iload(a).istore(b);
    }

    private static void decode(CodeBuilder c, DataType t, int represented, int value) {
        if (t == DataType.FLOAT64) c.dload(represented).dstore(value);
        else if (t == DataType.FLOAT32) c.fload(represented).fstore(value);
        else
            c.iload(represented)
                    .loadConstant(16)
                    .ishl()
                    .invokestatic(
                            FLOAT,
                            "intBitsToFloat",
                            MethodTypeDesc.of(ConstantDescs.CD_float, ConstantDescs.CD_int))
                    .fstore(value);
    }

    private static CodeBuilder loadValue(CodeBuilder c, DataType t, int v) {
        return t == DataType.FLOAT64 ? c.dload(v) : c.fload(v);
    }

    private static void storeValue(CodeBuilder c, DataType t, int v) {
        if (t == DataType.FLOAT64) c.dstore(v);
        else c.fstore(v);
    }

    private static void copyValue(CodeBuilder c, DataType t, int a, int b) {
        loadValue(c, t, a);
        storeValue(c, t, b);
    }

    private static CodeBuilder zero(CodeBuilder c, DataType t) {
        return t == DataType.FLOAT64 ? c.loadConstant(+0.0d) : c.loadConstant(+0.0f);
    }

    private static CodeBuilder negativeZero(CodeBuilder c, DataType t) {
        return t == DataType.FLOAT64 ? c.loadConstant(-0.0d) : c.loadConstant(-0.0f);
    }

    private static void compareGreater(
            CodeBuilder c, DataType t, int a, int b, java.lang.classfile.Label yes) {
        loadValue(c, t, a);
        loadValue(c, t, b);
        if (t == DataType.FLOAT64) c.dcmpl();
        else c.fcmpl();
        c.branch(Opcode.IFGT, yes);
    }

    private static void compareEqual(CodeBuilder c, DataType t, java.lang.classfile.Label no) {
        if (t == DataType.FLOAT64) c.dcmpl();
        else c.fcmpl();
        c.branch(Opcode.IFNE, no);
    }

    private static void compareNotEqual(CodeBuilder c, DataType t, java.lang.classfile.Label yes) {
        if (t == DataType.FLOAT64) c.dcmpl();
        else c.fcmpl();
        c.branch(Opcode.IFNE, yes);
    }

    private static void rawPositive(
            CodeBuilder c, DataType t, int represented, java.lang.classfile.Label yes) {
        if (t == DataType.FLOAT64)
            c.dload(represented)
                    .invokestatic(
                            DOUBLE,
                            "doubleToRawLongBits",
                            MethodTypeDesc.of(ConstantDescs.CD_long, ConstantDescs.CD_double))
                    .loadConstant(0L)
                    .lcmp()
                    .branch(Opcode.IFEQ, yes);
        else if (t == DataType.FLOAT32)
            c.fload(represented)
                    .invokestatic(
                            FLOAT,
                            "floatToRawIntBits",
                            MethodTypeDesc.of(ConstantDescs.CD_int, ConstantDescs.CD_float))
                    .branch(Opcode.IFEQ, yes);
        else c.iload(represented).loadConstant(0xffff).iand().branch(Opcode.IFEQ, yes);
    }

    private static void rawNegativeZero(CodeBuilder c, DataType t, int represented, int flag) {
        var negative = c.newLabel();
        if (t == DataType.FLOAT64)
            c.dload(represented)
                    .invokestatic(
                            DOUBLE,
                            "doubleToRawLongBits",
                            MethodTypeDesc.of(ConstantDescs.CD_long, ConstantDescs.CD_double))
                    .loadConstant(Long.MIN_VALUE)
                    .lcmp()
                    .branch(Opcode.IFEQ, negative);
        else if (t == DataType.FLOAT32)
            c.fload(represented)
                    .invokestatic(
                            FLOAT,
                            "floatToRawIntBits",
                            MethodTypeDesc.of(ConstantDescs.CD_int, ConstantDescs.CD_float))
                    .loadConstant(Integer.MIN_VALUE)
                    .branch(Opcode.IF_ICMPEQ, negative);
        else
            c.iload(represented)
                    .loadConstant(0xffff)
                    .iand()
                    .loadConstant(0x8000)
                    .branch(Opcode.IF_ICMPEQ, negative);
        c.loadConstant(0).istore(flag).labelBinding(negative);
    }

    private static void negativeInfinity(CodeBuilder c, DataType t, int target) {
        if (t == DataType.FLOAT64) c.loadConstant(Double.NEGATIVE_INFINITY).dstore(target);
        else if (t == DataType.FLOAT32) c.loadConstant(Float.NEGATIVE_INFINITY).fstore(target);
        else c.loadConstant(0xff80).istore(target);
    }
}
