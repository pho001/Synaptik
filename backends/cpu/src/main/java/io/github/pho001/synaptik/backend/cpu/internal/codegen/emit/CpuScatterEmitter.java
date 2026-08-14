package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;

import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDescs;
import java.lang.constant.MethodTypeDesc;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Emits allocation-free typed scatter output-coordinate writers directly into generated classes.
 * Bounds and replacement uniqueness are validated by cold-bound execution before these entry
 * targets run. Each invocation scans contributions in logical row-major order and writes each owned
 * output coordinate exactly once. Floating multiplication uses only its declared disjoint
 * primitive-limb scratch slice and rounds the exact abstract product once.
 */
public final class CpuScatterEmitter {
    private static final ClassDesc SEGMENT = ClassDesc.of(MemorySegment.class.getName());
    private static final ClassDesc VALUE_LAYOUT = ClassDesc.of(ValueLayout.class.getName());
    private static final ClassDesc LONG_LAYOUT =
            ClassDesc.of("java.lang.foreign.ValueLayout$OfLong");
    private static final ClassDesc LONG_CLASS = ClassDesc.of(Long.class.getName());
    private static final ClassDesc DOUBLE_CLASS = ClassDesc.of(Double.class.getName());
    private static final ClassDesc FLOAT_CLASS = ClassDesc.of(Float.class.getName());
    private static final ClassDesc MATH_CLASS = ClassDesc.of(Math.class.getName());

    /** Creates a stateless scatter emitter. */
    public CpuScatterEmitter() {}

    /**
     * Emits one carrier-, type-, family-, reduction-, and access-specialized writer for two through
     * four unique boundaries and optional exact-product scratch.
     *
     * @param code non-null generated method body
     * @param specialization non-null matching scalar scatter specialization
     * @param ir non-null instruction-free structural scatter encoding matching the specialization
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if the IR is not a supported structural scatter encoding,
     *     or its boundary cardinality does not match the specialization and scatter contract
     */
    public void emit(CodeBuilder code, CpuKernelSpecialization specialization, CpuKernelIr ir) {
        ScatterEncoding p = ScatterEncoding.parse(ir);
        int count = specialization.carrierPattern().size();
        if (count != p.ranks.length || count < 2 || count > 4)
            throw new IllegalArgumentException(
                    "scatter requires two through four matching unique boundaries");
        int geometrySlot = count + (specialization.scratchParameter() ? 1 : 0);
        boolean ints =
                specialization.loopAddressing(ir)
                        == CpuKernelSpecialization.LoopAddressing.DENSE_HEAP_ARRAY_INT;
        if (!specialization.scratchParameter()
                && ints
                && p.outputRank == 1
                && p.ranks[p.indexBoundary] == 1
                && p.ranks[p.updateBoundary] == 1
                && (p.family.equals("SCATTER_ELEMENTS") || p.family.equals("SCATTER_ADD"))) {
            int geometry = p.ranks.length;
            var general = code.newLabel();
            var complete = code.newLabel();
            for (int boundary = 0; boundary < p.ranks.length; boundary++)
                geometry(code, geometry, p.layouts[boundary] + 1)
                        .loadConstant(0L)
                        .lcmp()
                        .branch(Opcode.IFNE, general);
            emitDenseRankOneZeroBase(code, specialization, p);
            code.branch(Opcode.GOTO, complete).labelBinding(general);
            emitDenseRankOne(code, specialization, p);
            code.labelBinding(complete);
            return;
        }
        emitTyped(
                code,
                specialization,
                p,
                ints,
                geometrySlot,
                specialization.scratchParameter() ? count : -1);
    }

    private static void emitDenseRankOneZeroBase(
            CodeBuilder code, CpuKernelSpecialization s, ScatterEncoding p) {
        int g = p.ranks.length,
                logical = code.allocateLocal(TypeKind.INT),
                end = code.allocateLocal(TypeKind.INT),
                update = code.allocateLocal(TypeKind.INT);
        int updateCount = code.allocateLocal(TypeKind.INT),
                selected = code.allocateLocal(TypeKind.INT);
        int value = code.allocateLocal(localKind(p.dataType)),
                right = code.allocateLocal(localKind(p.dataType));
        code.lload(g + 1).l2i().istore(logical);
        code.lload(g + 3).l2i().istore(end);
        geometry(code, g, p.layouts[p.updateBoundary] + 2).l2i().istore(updateCount);
        var carriers = new CpuCarrierEmitter(code);
        var outer = code.newLabel();
        var done = code.newLabel();
        var inner = code.newLabel();
        var write = code.newLabel();
        code.labelBinding(outer).iload(logical).iload(end).branch(Opcode.IF_ICMPGE, done);
        carriers.load(
                p.dataType, s.carrierPattern().get(p.dataBoundary), p.dataBoundary, logical, true);
        storeValue(code, p.dataType, value);
        code.loadConstant(0)
                .istore(update)
                .labelBinding(inner)
                .iload(update)
                .iload(updateCount)
                .branch(Opcode.IF_ICMPGE, write);
        loadIndex(code, carriers, s, p, update, selected, true);
        var noMatch = code.newLabel();
        code.iload(selected).iload(logical).branch(Opcode.IF_ICMPNE, noMatch);
        carriers.load(
                p.dataType,
                s.carrierPattern().get(p.updateBoundary),
                p.updateBoundary,
                update,
                true);
        storeValue(code, p.dataType, right);
        emitReduction(code, p.dataType, p.reduction, value, right);
        code.labelBinding(noMatch).iinc(update, 1).branch(Opcode.GOTO, inner);
        code.labelBinding(write);
        carriers.store(
                p.dataType,
                s.carrierPattern().get(p.outputBoundary),
                p.outputBoundary,
                logical,
                value,
                true);
        code.iinc(logical, 1).branch(Opcode.GOTO, outer).labelBinding(done);
    }

    private static void emitDenseRankOne(
            CodeBuilder code, CpuKernelSpecialization s, ScatterEncoding p) {
        int g = p.ranks.length,
                logical = code.allocateLocal(TypeKind.INT),
                end = code.allocateLocal(TypeKind.INT);
        int update = code.allocateLocal(TypeKind.INT),
                updateCount = code.allocateLocal(TypeKind.INT);
        int dataAddress = code.allocateLocal(TypeKind.INT),
                indexAddress = code.allocateLocal(TypeKind.INT),
                updateAddress = code.allocateLocal(TypeKind.INT),
                outputAddress = code.allocateLocal(TypeKind.INT);
        int indexBase = code.allocateLocal(TypeKind.INT),
                updateBase = code.allocateLocal(TypeKind.INT);
        int selected = code.allocateLocal(TypeKind.INT),
                value = code.allocateLocal(localKind(p.dataType)),
                right = code.allocateLocal(localKind(p.dataType));
        code.lload(g + 1).l2i().istore(logical);
        code.lload(g + 3).l2i().istore(end);
        geometry(code, g, p.layouts[p.updateBoundary] + 2).l2i().istore(updateCount);
        geometry(code, g, p.layouts[p.dataBoundary] + 1)
                .l2i()
                .iload(logical)
                .iadd()
                .istore(dataAddress);
        geometry(code, g, p.layouts[p.outputBoundary] + 1)
                .l2i()
                .iload(logical)
                .iadd()
                .istore(outputAddress);
        geometry(code, g, p.layouts[p.indexBoundary] + 1).l2i().istore(indexBase);
        geometry(code, g, p.layouts[p.updateBoundary] + 1).l2i().istore(updateBase);
        var carriers = new CpuCarrierEmitter(code);
        var outer = code.newLabel();
        var done = code.newLabel();
        var inner = code.newLabel();
        var write = code.newLabel();
        code.labelBinding(outer).iload(logical).iload(end).branch(Opcode.IF_ICMPGE, done);
        carriers.load(
                p.dataType,
                s.carrierPattern().get(p.dataBoundary),
                p.dataBoundary,
                dataAddress,
                true);
        storeValue(code, p.dataType, value);
        code.loadConstant(0).istore(update);
        code.labelBinding(inner).iload(update).iload(updateCount).branch(Opcode.IF_ICMPGE, write);
        code.iload(indexBase).iload(update).iadd().istore(indexAddress);
        code.iload(updateBase).iload(update).iadd().istore(updateAddress);
        loadIndex(code, carriers, s, p, indexAddress, selected, true);
        var noMatch = code.newLabel();
        code.iload(selected).iload(logical).branch(Opcode.IF_ICMPNE, noMatch);
        carriers.load(
                p.dataType,
                s.carrierPattern().get(p.updateBoundary),
                p.updateBoundary,
                updateAddress,
                true);
        storeValue(code, p.dataType, right);
        emitReduction(code, p.dataType, p.reduction, value, right);
        code.labelBinding(noMatch);
        code.iinc(update, 1).branch(Opcode.GOTO, inner);
        code.labelBinding(write);
        carriers.store(
                p.dataType,
                s.carrierPattern().get(p.outputBoundary),
                p.outputBoundary,
                outputAddress,
                value,
                true);
        code.iinc(logical, 1)
                .iinc(dataAddress, 1)
                .iinc(outputAddress, 1)
                .branch(Opcode.GOTO, outer)
                .labelBinding(done);
    }

    private static void emitTyped(
            CodeBuilder code,
            CpuKernelSpecialization s,
            ScatterEncoding p,
            boolean ints,
            int g,
            int scratch) {
        int coordinate = 16 + p.outputRank, updateCoordinate = coordinate + p.outputRank;
        int logical = code.allocateLocal(ints ? TypeKind.INT : TypeKind.LONG),
                end = code.allocateLocal(ints ? TypeKind.INT : TypeKind.LONG);
        int updateOrdinal = code.allocateLocal(ints ? TypeKind.INT : TypeKind.LONG),
                updateCount = code.allocateLocal(ints ? TypeKind.INT : TypeKind.LONG);
        int outputAddress = code.allocateLocal(ints ? TypeKind.INT : TypeKind.LONG),
                dataAddress = code.allocateLocal(ints ? TypeKind.INT : TypeKind.LONG);
        int updateAddress = code.allocateLocal(ints ? TypeKind.INT : TypeKind.LONG),
                indexAddress = code.allocateLocal(ints ? TypeKind.INT : TypeKind.LONG);
        int selected = code.allocateLocal(ints ? TypeKind.INT : TypeKind.LONG),
                found = code.allocateLocal(TypeKind.INT),
                match = code.allocateLocal(TypeKind.INT);
        int value = code.allocateLocal(localKind(p.dataType)),
                right = code.allocateLocal(localKind(p.dataType));
        ExactProductEmitter product =
                scratch >= 0 ? new ExactProductEmitter(code, p.dataType, scratch, g) : null;
        if (ints) {
            code.lload(g + 1).l2i().istore(logical);
            code.lload(g + 3).l2i().istore(end);
            code.loadConstant(1).istore(updateCount);
        } else {
            code.lload(g + 1).lstore(logical);
            code.lload(g + 3).lstore(end);
            code.loadConstant(1L).lstore(updateCount);
        }
        for (int axis = 0; axis < p.outputRank; axis++)
            code.aload(g)
                    .loadConstant(coordinate + axis)
                    .aload(g)
                    .loadConstant(16 + axis)
                    .laload()
                    .lastore();
        for (int axis = 0; axis < p.ranks[p.updateBoundary]; axis++) {
            loadAddress(code, updateCount, ints);
            geometry(code, g, p.layouts[p.updateBoundary] + 2 + axis);
            if (ints) code.l2i();
            multiply(code, ints);
            storeAddress(code, updateCount, ints);
        }
        var carriers = new CpuCarrierEmitter(code);
        var outer = code.newLabel();
        var done = code.newLabel();
        var updates = code.newLabel();
        code.labelBinding(outer);
        compareEnd(code, logical, end, ints, done);
        address(
                code,
                g,
                p.layouts[p.outputBoundary],
                coordinate,
                p.outputRank,
                outputAddress,
                ints);
        address(code, g, p.layouts[p.dataBoundary], coordinate, p.outputRank, dataAddress, ints);
        carriers.load(
                p.dataType,
                s.carrierPattern().get(p.dataBoundary),
                p.dataBoundary,
                dataAddress,
                ints);
        storeValue(code, p.dataType, value);
        code.loadConstant(0).istore(found);
        for (int axis = 0; axis < p.ranks[p.updateBoundary]; axis++)
            code.aload(g).loadConstant(updateCoordinate + axis).loadConstant(0L).lastore();
        if (ints) code.loadConstant(0).istore(updateOrdinal);
        else code.loadConstant(0L).lstore(updateOrdinal);
        code.labelBinding(updates);
        var contributionsDone = code.newLabel();
        compareEnd(code, updateOrdinal, updateCount, ints, contributionsDone);
        code.loadConstant(1).istore(match);
        emitMatch(
                code,
                carriers,
                s,
                p,
                g,
                coordinate,
                updateCoordinate,
                indexAddress,
                selected,
                match,
                ints);
        var noMatch = code.newLabel();
        code.iload(match).branch(Opcode.IFEQ, noMatch);
        address(
                code,
                g,
                p.layouts[p.updateBoundary],
                updateCoordinate,
                p.ranks[p.updateBoundary],
                updateAddress,
                ints);
        carriers.load(
                p.dataType,
                s.carrierPattern().get(p.updateBoundary),
                p.updateBoundary,
                updateAddress,
                ints);
        storeValue(code, p.dataType, right);
        if (product != null) product.emitFactors(value, right, found);
        else emitReduction(code, p.dataType, p.reduction, value, right);
        code.loadConstant(1).istore(found);
        code.labelBinding(noMatch);
        advancePacked(
                code, g, updateCoordinate, p.layouts[p.updateBoundary], p.ranks[p.updateBoundary]);
        increment(code, updateOrdinal, ints);
        code.branch(Opcode.GOTO, updates).labelBinding(contributionsDone);
        if (product != null) product.emitFinish(value, found);
        carriers.store(
                p.dataType,
                s.carrierPattern().get(p.outputBoundary),
                p.outputBoundary,
                outputAddress,
                value,
                ints);
        advancePacked(code, g, coordinate, p.layouts[p.outputBoundary], p.outputRank);
        increment(code, logical, ints);
        code.branch(Opcode.GOTO, outer).labelBinding(done);
    }

    private static void emitMatch(
            CodeBuilder code,
            CpuCarrierEmitter carriers,
            CpuKernelSpecialization s,
            ScatterEncoding p,
            int g,
            int out,
            int update,
            int indexAddress,
            int selected,
            int match,
            boolean ints) {
        if (p.family.equals("SCATTER_ELEMENTS")) {
            for (int d = 0; d < p.outputRank; d++) {
                var skip = code.newLabel();
                var equal = code.newLabel();
                code.loadConstant(d);
                geometry(code, g, 6).l2i().branch(Opcode.IF_ICMPEQ, skip);
                geometry(code, g, update + d);
                geometry(code, g, out + d).lcmp().branch(Opcode.IFEQ, equal);
                code.loadConstant(0).istore(match);
                code.labelBinding(equal).labelBinding(skip);
            }
            address(
                    code,
                    g,
                    p.layouts[p.indexBoundary],
                    update,
                    p.ranks[p.indexBoundary],
                    indexAddress,
                    ints);
            loadIndex(code, carriers, s, p, indexAddress, selected, ints);
            var equal = code.newLabel();
            loadAddress(code, selected, ints);
            code.aload(g).loadConstant(out);
            geometry(code, g, 6).l2i().iadd().laload();
            if (ints) code.l2i().branch(Opcode.IF_ICMPEQ, equal);
            else code.lcmp().branch(Opcode.IFEQ, equal);
            code.loadConstant(0).istore(match);
            code.labelBinding(equal);
            return;
        }
        if (p.family.equals("SCATTER_ADD")) {
            emitScatterAddMatch(
                    code, carriers, s, p, g, out, update, indexAddress, selected, match, ints);
            return;
        }
        emitScatterNdMatch(
                code, carriers, s, p, g, out, update, indexAddress, selected, match, ints);
    }

    private static void emitScatterAddMatch(
            CodeBuilder code,
            CpuCarrierEmitter carriers,
            CpuKernelSpecialization s,
            ScatterEncoding p,
            int g,
            int out,
            int update,
            int indexAddress,
            int selected,
            int match,
            boolean ints) {
        int indexRank = p.ranks[p.indexBoundary];
        for (int d = 0; d < p.outputRank; d++) {
            var selectedAxis = code.newLabel();
            var afterSide = code.newLabel();
            var suffix = code.newLabel();
            var equal = code.newLabel();
            code.loadConstant(d);
            geometry(code, g, 6).l2i().branch(Opcode.IF_ICMPEQ, selectedAxis);
            code.loadConstant(d);
            geometry(code, g, 6).l2i().branch(Opcode.IF_ICMPGT, suffix);
            geometry(code, g, update + d).branch(Opcode.GOTO, afterSide);
            code.labelBinding(suffix);
            geometry(code, g, update + indexRank + d - 1);
            code.labelBinding(afterSide);
            geometry(code, g, out + d).lcmp().branch(Opcode.IFEQ, equal);
            code.loadConstant(0).istore(match);
            code.labelBinding(equal).labelBinding(selectedAxis);
        }
        base(code, g, p.layouts[p.indexBoundary], indexAddress, ints);
        for (int axis = 0; axis < indexRank; axis++) {
            loadAddress(code, indexAddress, ints);
            code.aload(g).loadConstant(update);
            geometry(code, g, 6).l2i().iadd().loadConstant(axis).iadd().laload();
            if (ints) code.l2i();
            stride(code, g, p.layouts[p.indexBoundary], indexRank, axis, ints);
            multiply(code, ints);
            add(code, ints);
            storeAddress(code, indexAddress, ints);
        }
        loadIndex(code, carriers, s, p, indexAddress, selected, ints);
        var equal = code.newLabel();
        loadAddress(code, selected, ints);
        code.aload(g).loadConstant(out);
        geometry(code, g, 6).l2i().iadd().laload();
        if (ints) code.l2i().branch(Opcode.IF_ICMPEQ, equal);
        else code.lcmp().branch(Opcode.IFEQ, equal);
        code.loadConstant(0).istore(match);
        code.labelBinding(equal);
    }

    private static void emitScatterNdMatch(
            CodeBuilder code,
            CpuCarrierEmitter carriers,
            CpuKernelSpecialization s,
            ScatterEncoding p,
            int g,
            int out,
            int update,
            int indexAddress,
            int selected,
            int match,
            boolean ints) {
        int dataRank = p.outputRank, indexRank = p.ranks[p.indexBoundary];
        for (int d = 0; d < dataRank; d++) {
            var notBatch = code.newLabel();
            var equal = code.newLabel();
            code.loadConstant(d);
            geometry(code, g, 7).l2i().branch(Opcode.IF_ICMPGE, notBatch);
            geometry(code, g, update + d);
            geometry(code, g, out + d).lcmp().branch(Opcode.IFEQ, equal);
            code.loadConstant(0).istore(match);
            code.labelBinding(equal).labelBinding(notBatch);
        }
        for (int k = 0; k < dataRank; k++) {
            var afterTuple = code.newLabel();
            var equal = code.newLabel();
            code.loadConstant(k);
            geometry(code, g, 8).l2i().branch(Opcode.IF_ICMPGE, afterTuple);
            base(code, g, p.layouts[p.indexBoundary], indexAddress, ints);
            for (int d = 0; d < indexRank - 1; d++) {
                loadAddress(code, indexAddress, ints);
                geometry(code, g, update + d);
                if (ints) code.l2i();
                stride(code, g, p.layouts[p.indexBoundary], indexRank, d, ints);
                multiply(code, ints);
                add(code, ints);
                storeAddress(code, indexAddress, ints);
            }
            loadAddress(code, indexAddress, ints);
            if (ints) code.loadConstant(k);
            else code.loadConstant((long) k);
            stride(code, g, p.layouts[p.indexBoundary], indexRank, indexRank - 1, ints);
            multiply(code, ints);
            add(code, ints);
            storeAddress(code, indexAddress, ints);
            loadIndex(code, carriers, s, p, indexAddress, selected, ints);
            loadAddress(code, selected, ints);
            code.aload(g).loadConstant(out);
            geometry(code, g, 7).l2i().iadd().loadConstant(k).iadd().laload();
            if (ints) code.l2i().branch(Opcode.IF_ICMPEQ, equal);
            else code.lcmp().branch(Opcode.IFEQ, equal);
            code.loadConstant(0).istore(match);
            code.labelBinding(equal).labelBinding(afterTuple);
        }
        for (int d = 0; d < dataRank; d++) {
            var beforeSuffix = code.newLabel();
            var equal = code.newLabel();
            code.loadConstant(d);
            geometry(code, g, 7).l2i();
            geometry(code, g, 8).l2i().iadd().branch(Opcode.IF_ICMPLT, beforeSuffix);
            code.aload(g).loadConstant(update + indexRank - 1 + d);
            geometry(code, g, 7).l2i().isub();
            geometry(code, g, 8).l2i().isub().laload();
            geometry(code, g, out + d).lcmp().branch(Opcode.IFEQ, equal);
            code.loadConstant(0).istore(match);
            code.labelBinding(equal).labelBinding(beforeSuffix);
        }
    }

    private static void emitReduction(
            CodeBuilder code, DataType type, ScatterReduction reduction, int left, int right) {
        if (reduction == ScatterReduction.NONE) {
            loadValue(code, type, right);
            storeValue(code, type, left);
            return;
        }
        if (type == DataType.BFLOAT16) {
            emitBfloatReduction(code, reduction, left, right);
            code.istore(left);
            return;
        }
        loadValue(code, type, left);
        loadValue(code, type, right);
        switch (type) {
            case FLOAT64 -> {
                if (reduction == ScatterReduction.ADD) code.dadd();
                else if (reduction == ScatterReduction.MUL) code.dmul();
                else
                    code.invokestatic(
                            ClassDesc.of("java.lang.Math"),
                            reduction == ScatterReduction.MIN ? "min" : "max",
                            MethodTypeDesc.of(
                                    ConstantDescs.CD_double,
                                    ConstantDescs.CD_double,
                                    ConstantDescs.CD_double));
            }
            case FLOAT32 -> {
                if (reduction == ScatterReduction.ADD) code.fadd();
                else if (reduction == ScatterReduction.MUL) code.fmul();
                else
                    code.invokestatic(
                            ClassDesc.of("java.lang.Math"),
                            reduction == ScatterReduction.MIN ? "min" : "max",
                            MethodTypeDesc.of(
                                    ConstantDescs.CD_float,
                                    ConstantDescs.CD_float,
                                    ConstantDescs.CD_float));
            }
            case INT32 -> {
                if (reduction == ScatterReduction.ADD) code.iadd();
                else if (reduction == ScatterReduction.MUL) code.imul();
                else
                    code.invokestatic(
                            ClassDesc.of("java.lang.Math"),
                            reduction == ScatterReduction.MIN ? "min" : "max",
                            MethodTypeDesc.of(
                                    ConstantDescs.CD_int,
                                    ConstantDescs.CD_int,
                                    ConstantDescs.CD_int));
            }
            case INT64 -> {
                if (reduction == ScatterReduction.ADD) code.ladd();
                else if (reduction == ScatterReduction.MUL) code.lmul();
                else
                    code.invokestatic(
                            ClassDesc.of("java.lang.Math"),
                            reduction == ScatterReduction.MIN ? "min" : "max",
                            MethodTypeDesc.of(
                                    ConstantDescs.CD_long,
                                    ConstantDescs.CD_long,
                                    ConstantDescs.CD_long));
            }
            case BFLOAT16 -> throw new AssertionError("handled above");
            case BOOL -> throw new IllegalArgumentException("BOOL reduction is unsupported");
        }
        storeValue(code, type, left);
    }

    private static void emitBfloatReduction(
            CodeBuilder code, ScatterReduction reduction, int left, int right) {
        int leftFloat = code.allocateLocal(TypeKind.FLOAT);
        int rightFloat = code.allocateLocal(TypeKind.FLOAT);
        int resultFloat = code.allocateLocal(TypeKind.FLOAT);
        int resultBits = code.allocateLocal(TypeKind.INT);
        int upperBits = code.allocateLocal(TypeKind.INT);
        int lowerBits = code.allocateLocal(TypeKind.INT);

        emitBfloatToFloat(code, left, leftFloat);
        emitBfloatToFloat(code, right, rightFloat);
        if (reduction == ScatterReduction.ADD || reduction == ScatterReduction.MUL) {
            code.fload(leftFloat).fload(rightFloat);
            if (reduction == ScatterReduction.ADD) {
                code.fadd();
            } else {
                code.fmul();
            }
            code.fstore(resultFloat);
        } else {
            var useCanonicalNan = code.newLabel();
            var reduceFiniteValues = code.newLabel();
            var reductionComplete = code.newLabel();
            code.fload(leftFloat)
                    .invokestatic(
                            FLOAT_CLASS,
                            "isNaN",
                            MethodTypeDesc.of(ConstantDescs.CD_boolean, ConstantDescs.CD_float))
                    .branch(Opcode.IFNE, useCanonicalNan);
            code.fload(rightFloat)
                    .invokestatic(
                            FLOAT_CLASS,
                            "isNaN",
                            MethodTypeDesc.of(ConstantDescs.CD_boolean, ConstantDescs.CD_float))
                    .branch(Opcode.IFEQ, reduceFiniteValues);
            code.labelBinding(useCanonicalNan)
                    .loadConstant(Float.NaN)
                    .fstore(resultFloat)
                    .branch(Opcode.GOTO, reductionComplete)
                    .labelBinding(reduceFiniteValues)
                    .fload(leftFloat)
                    .fload(rightFloat)
                    .invokestatic(
                            ClassDesc.of(Math.class.getName()),
                            reduction == ScatterReduction.MIN ? "min" : "max",
                            MethodTypeDesc.of(
                                    ConstantDescs.CD_float,
                                    ConstantDescs.CD_float,
                                    ConstantDescs.CD_float))
                    .fstore(resultFloat)
                    .labelBinding(reductionComplete);
        }

        code.fload(resultFloat)
                .invokestatic(
                        FLOAT_CLASS,
                        "floatToRawIntBits",
                        MethodTypeDesc.of(ConstantDescs.CD_int, ConstantDescs.CD_float))
                .istore(resultBits);
        var finite = code.newLabel();
        var round = code.newLabel();
        var complete = code.newLabel();
        code.iload(resultBits)
                .loadConstant(0x7f800000)
                .iand()
                .loadConstant(0x7f800000)
                .branch(Opcode.IF_ICMPNE, finite);
        code.iload(resultBits).loadConstant(0x007fffff).iand().branch(Opcode.IFEQ, finite);
        code.iload(resultBits)
                .loadConstant(16)
                .iushr()
                .loadConstant(0x40)
                .ior()
                .branch(Opcode.GOTO, complete);
        code.labelBinding(finite).iload(resultBits).loadConstant(16).iushr().istore(upperBits);
        code.iload(resultBits).loadConstant(0xffff).iand().istore(lowerBits);
        code.iload(lowerBits).loadConstant(0x8000).branch(Opcode.IF_ICMPGT, round);
        var exactTie = code.newLabel();
        code.iload(lowerBits).loadConstant(0x8000).branch(Opcode.IF_ICMPEQ, exactTie);
        code.iload(upperBits).branch(Opcode.GOTO, complete);
        code.labelBinding(exactTie)
                .iload(upperBits)
                .loadConstant(1)
                .iand()
                .branch(Opcode.IFNE, round);
        code.iload(upperBits).branch(Opcode.GOTO, complete);
        code.labelBinding(round).iinc(upperBits, 1).iload(upperBits).labelBinding(complete);
    }

    private static void emitBfloatToFloat(CodeBuilder code, int representedBits, int target) {
        code.iload(representedBits)
                .loadConstant(16)
                .ishl()
                .invokestatic(
                        FLOAT_CLASS,
                        "intBitsToFloat",
                        MethodTypeDesc.of(ConstantDescs.CD_float, ConstantDescs.CD_int))
                .fstore(target);
    }

    /** Emits the complete type-specialized exact product into the generated entry itself. */
    private static final class ExactProductEmitter {
        private static final long SIGN = 1, ZERO = 2, INFINITY = 4, NAN = 8;
        private final CodeBuilder code;
        private final DataType type;
        private final int scratch;
        private final int geometry;
        private final int offset;
        private final int bytes;
        private final int flags;
        private final int bits;
        private final int fraction;
        private final int exponentField;
        private final int significand;
        private final int exponent;
        private final int used;
        private final int limb;
        private final int factor;
        private final int carry;
        private final int word;
        private final int low;
        private final int high;
        private final int sum;
        private final int position;
        private final int top;
        private final int bitLength;
        private final int unbiased;
        private final int shift;
        private final int quotient;
        private final int normal;
        private final int guard;
        private final int sticky;
        private final int bitOffset;
        private final int fullLimbs;
        private final int remainingBits;
        private final int result;

        ExactProductEmitter(CodeBuilder code, DataType type, int scratch, int geometry) {
            if (type != DataType.FLOAT64 && type != DataType.FLOAT32 && type != DataType.BFLOAT16)
                throw new IllegalArgumentException("exact product requires a floating type");
            this.code = code;
            this.type = type;
            this.scratch = scratch;
            this.geometry = geometry;
            offset = code.allocateLocal(TypeKind.LONG);
            bytes = code.allocateLocal(TypeKind.LONG);
            flags = code.allocateLocal(TypeKind.LONG);
            bits = code.allocateLocal(TypeKind.LONG);
            fraction = code.allocateLocal(TypeKind.LONG);
            exponentField = code.allocateLocal(TypeKind.LONG);
            significand = code.allocateLocal(TypeKind.LONG);
            exponent = code.allocateLocal(TypeKind.LONG);
            used = code.allocateLocal(TypeKind.INT);
            limb = code.allocateLocal(TypeKind.INT);
            factor = code.allocateLocal(TypeKind.LONG);
            carry = code.allocateLocal(TypeKind.LONG);
            word = code.allocateLocal(TypeKind.LONG);
            low = code.allocateLocal(TypeKind.LONG);
            high = code.allocateLocal(TypeKind.LONG);
            sum = code.allocateLocal(TypeKind.LONG);
            position = code.allocateLocal(TypeKind.LONG);
            top = code.allocateLocal(TypeKind.LONG);
            bitLength = code.allocateLocal(TypeKind.INT);
            unbiased = code.allocateLocal(TypeKind.LONG);
            shift = code.allocateLocal(TypeKind.LONG);
            quotient = code.allocateLocal(TypeKind.LONG);
            normal = code.allocateLocal(TypeKind.INT);
            guard = code.allocateLocal(TypeKind.INT);
            sticky = code.allocateLocal(TypeKind.INT);
            bitOffset = code.allocateLocal(TypeKind.INT);
            fullLimbs = code.allocateLocal(TypeKind.INT);
            remainingBits = code.allocateLocal(TypeKind.INT);
            result = code.allocateLocal(TypeKind.LONG);
            geometry(code, geometry, 13).lstore(offset);
            geometry(code, geometry, 14).lstore(bytes);
        }

        void emitFactors(int left, int right, int found) {
            var initialized = code.newLabel();
            code.iload(found).branch(Opcode.IFNE, initialized);
            emitReset();
            emitFactor(left);
            code.labelBinding(initialized);
            emitFactor(right);
        }

        private void emitReset() {
            set(0, () -> code.loadConstant(0L));
            set(8, () -> code.loadConstant(0L));
            set(16, () -> code.loadConstant(1L));
            code.lload(offset).loadConstant(24L).ladd().lstore(position);
            var loop = code.newLabel();
            var done = code.newLabel();
            code.labelBinding(loop);
            code.lload(position).lload(offset).lload(bytes).ladd().lcmp().branch(Opcode.IFGE, done);
            setAt(position, () -> code.loadConstant(0L));
            code.lload(position)
                    .loadConstant(8L)
                    .ladd()
                    .lstore(position)
                    .branch(Opcode.GOTO, loop)
                    .labelBinding(done);
            set(24, () -> code.loadConstant(1L));
        }

        private void emitFactor(int value) {
            loadBits(value);
            code.lstore(bits);
            get(0).lstore(flags);
            code.lload(bits).loadConstant(signMask()).land().loadConstant(0L).lcmp();
            var positive = code.newLabel();
            code.branch(Opcode.IFEQ, positive);
            code.lload(flags).loadConstant(SIGN).lxor().lstore(flags).labelBinding(positive);
            code.lload(bits).loadConstant(fractionMask()).land().lstore(fraction);
            code.lload(bits)
                    .loadConstant(fractionBits())
                    .lushr()
                    .loadConstant(exponentMask())
                    .land()
                    .lstore(exponentField);
            var finite = code.newLabel();
            code.lload(exponentField)
                    .loadConstant(exponentMask())
                    .lcmp()
                    .branch(Opcode.IFNE, finite);
            var infinity = code.newLabel();
            code.lload(fraction).loadConstant(0L).lcmp().branch(Opcode.IFEQ, infinity);
            code.lload(flags).loadConstant(NAN).lor().lstore(flags);
            var classified = code.newLabel();
            code.branch(Opcode.GOTO, classified).labelBinding(infinity);
            code.lload(flags)
                    .loadConstant(INFINITY)
                    .lor()
                    .lstore(flags)
                    .branch(Opcode.GOTO, classified)
                    .labelBinding(finite);
            var nonzero = code.newLabel();
            code.lload(exponentField).loadConstant(0L).lcmp().branch(Opcode.IFNE, nonzero);
            code.lload(fraction).loadConstant(0L).lcmp().branch(Opcode.IFNE, nonzero);
            code.lload(flags)
                    .loadConstant(ZERO)
                    .lor()
                    .lstore(flags)
                    .branch(Opcode.GOTO, classified)
                    .labelBinding(nonzero);
            var normalFactor = code.newLabel();
            var factorReady = code.newLabel();
            code.lload(exponentField).loadConstant(0L).lcmp().branch(Opcode.IFNE, normalFactor);
            code.lload(fraction).lstore(significand);
            code.loadConstant((long) (1 - bias() - fractionBits()))
                    .lstore(exponent)
                    .branch(Opcode.GOTO, factorReady)
                    .labelBinding(normalFactor);
            code.loadConstant(1L << fractionBits()).lload(fraction).lor().lstore(significand);
            code.lload(exponentField)
                    .loadConstant((long) (bias() + fractionBits()))
                    .lsub()
                    .lstore(exponent)
                    .labelBinding(factorReady);
            get(8).lload(exponent).ladd().lstore(exponent);
            set(8, () -> code.lload(exponent));
            code.lload(significand).lstore(factor);
            emitMultiply();
            code.labelBinding(classified);
            set(0, () -> code.lload(flags));
        }

        private void emitMultiply() {
            get(16).l2i().istore(used);
            code.loadConstant(0L).lstore(carry);
            code.loadConstant(0).istore(limb);
            var loop = code.newLabel();
            var done = code.newLabel();
            code.labelBinding(loop).iload(limb).iload(used).branch(Opcode.IF_ICMPGE, done);
            limbOffset(limb);
            code.lstore(position);
            getAt(position).lstore(word);
            code.lload(word).lload(factor).lmul().lstore(low);
            code.lload(word)
                    .lload(factor)
                    .invokestatic(
                            MATH_CLASS,
                            "unsignedMultiplyHigh",
                            MethodTypeDesc.of(
                                    ConstantDescs.CD_long,
                                    ConstantDescs.CD_long,
                                    ConstantDescs.CD_long))
                    .lstore(high);
            code.lload(low).lload(carry).ladd().lstore(sum);
            code.lload(sum)
                    .lload(low)
                    .invokestatic(
                            LONG_CLASS,
                            "compareUnsigned",
                            MethodTypeDesc.of(
                                    ConstantDescs.CD_int,
                                    ConstantDescs.CD_long,
                                    ConstantDescs.CD_long));
            var noOverflow = code.newLabel();
            code.branch(Opcode.IFGE, noOverflow);
            code.lload(high).loadConstant(1L).ladd().lstore(high).labelBinding(noOverflow);
            setAt(position, () -> code.lload(sum));
            code.lload(high).lstore(carry);
            code.iinc(limb, 1).branch(Opcode.GOTO, loop).labelBinding(done);
            var noCarry = code.newLabel();
            code.lload(carry).loadConstant(0L).lcmp().branch(Opcode.IFEQ, noCarry);
            limbOffset(used);
            code.lstore(position);
            setAt(position, () -> code.lload(carry));
            code.iload(used).loadConstant(1).iadd().i2l().lstore(exponent);
            set(16, () -> code.lload(exponent));
            code.labelBinding(noCarry);
        }

        void emitFinish(int value, int found) {
            var absent = code.newLabel();
            var store = code.newLabel();
            code.iload(found).branch(Opcode.IFEQ, absent);
            get(0).lstore(flags);
            code.loadConstant(0L).lstore(result);
            code.lload(flags).loadConstant(SIGN).land().loadConstant(0L).lcmp();
            var positive = code.newLabel();
            code.branch(Opcode.IFEQ, positive);
            code.loadConstant(signBit()).lstore(result).labelBinding(positive);
            code.lload(flags).loadConstant(NAN).land().loadConstant(0L).lcmp();
            var checkZeroInfinity = code.newLabel();
            code.branch(Opcode.IFEQ, checkZeroInfinity);
            code.lload(result)
                    .loadConstant(canonicalNan())
                    .lor()
                    .lstore(result)
                    .branch(Opcode.GOTO, store)
                    .labelBinding(checkZeroInfinity);
            code.lload(flags)
                    .loadConstant(ZERO | INFINITY)
                    .land()
                    .loadConstant(ZERO | INFINITY)
                    .lcmp();
            var checkInfinity = code.newLabel();
            code.branch(Opcode.IFNE, checkInfinity);
            code.lload(result)
                    .loadConstant(canonicalNan())
                    .lor()
                    .lstore(result)
                    .branch(Opcode.GOTO, store)
                    .labelBinding(checkInfinity);
            code.lload(flags).loadConstant(INFINITY).land().loadConstant(0L).lcmp();
            var checkZero = code.newLabel();
            code.branch(Opcode.IFEQ, checkZero);
            code.lload(result)
                    .loadConstant(positiveInfinity())
                    .lor()
                    .lstore(result)
                    .branch(Opcode.GOTO, store)
                    .labelBinding(checkZero);
            code.lload(flags)
                    .loadConstant(ZERO)
                    .land()
                    .loadConstant(0L)
                    .lcmp()
                    .branch(Opcode.IFNE, store);
            emitFiniteFinish(store);
            code.labelBinding(store);
            storeResult(value);
            code.labelBinding(absent);
        }

        private void emitFiniteFinish(java.lang.classfile.Label store) {
            get(16).l2i().istore(used);
            limbOffsetMinusOne();
            code.lstore(position);
            getAt(position).lstore(top);
            code.iload(used)
                    .loadConstant(1)
                    .isub()
                    .loadConstant(64)
                    .imul()
                    .loadConstant(64)
                    .lload(top)
                    .invokestatic(
                            LONG_CLASS,
                            "numberOfLeadingZeros",
                            MethodTypeDesc.of(ConstantDescs.CD_int, ConstantDescs.CD_long))
                    .isub()
                    .iadd()
                    .istore(bitLength);
            get(8).iload(bitLength).i2l().ladd().loadConstant(1L).lsub().lstore(unbiased);
            code.lload(unbiased).loadConstant((long) maxExponent()).lcmp();
            var notOverflow = code.newLabel();
            code.branch(Opcode.IFLE, notOverflow);
            code.lload(result)
                    .loadConstant(positiveInfinity())
                    .lor()
                    .lstore(result)
                    .branch(Opcode.GOTO, store)
                    .labelBinding(notOverflow);
            code.lload(unbiased).loadConstant((long) minimumNormal()).lcmp();
            var subnormal = code.newLabel();
            var round = code.newLabel();
            code.branch(Opcode.IFLT, subnormal);
            code.iload(bitLength).loadConstant(precision()).isub().i2l().lstore(shift);
            code.loadConstant(1).istore(normal).branch(Opcode.GOTO, round).labelBinding(subnormal);
            code.loadConstant((long) (minimumNormal() - (precision() - 1)));
            get(8);
            code.lsub().lstore(shift);
            code.loadConstant(0).istore(normal).labelBinding(round);
            emitRounded();
            var finishSubnormal = code.newLabel();
            code.iload(normal).branch(Opcode.IFEQ, finishSubnormal);
            code.lload(quotient).loadConstant(1L << precision()).lcmp();
            var normalWidth = code.newLabel();
            code.branch(Opcode.IFNE, normalWidth);
            code.lload(quotient).loadConstant(1).lushr().lstore(quotient);
            code.lload(unbiased).loadConstant(1L).ladd().lstore(unbiased);
            code.lload(unbiased).loadConstant((long) maxExponent()).lcmp();
            var roundedNormal = code.newLabel();
            code.branch(Opcode.IFLE, roundedNormal);
            code.lload(result)
                    .loadConstant(positiveInfinity())
                    .lor()
                    .lstore(result)
                    .branch(Opcode.GOTO, store)
                    .labelBinding(roundedNormal)
                    .labelBinding(normalWidth);
            code.lload(unbiased).loadConstant((long) bias()).ladd().lstore(exponentField);
            code.lload(result)
                    .lload(exponentField)
                    .loadConstant(fractionBits())
                    .lshl()
                    .lload(quotient)
                    .loadConstant(fractionMask())
                    .land()
                    .lor()
                    .lor()
                    .lstore(result)
                    .branch(Opcode.GOTO, store)
                    .labelBinding(finishSubnormal);
            code.lload(quotient).loadConstant(0L).lcmp().branch(Opcode.IFEQ, store);
            code.lload(quotient).loadConstant(1L << fractionBits()).lcmp();
            var trueSubnormal = code.newLabel();
            code.branch(Opcode.IFLT, trueSubnormal);
            code.lload(result)
                    .loadConstant(1L << fractionBits())
                    .lor()
                    .lstore(result)
                    .branch(Opcode.GOTO, store)
                    .labelBinding(trueSubnormal);
            code.lload(result).lload(quotient).lor().lstore(result).branch(Opcode.GOTO, store);
        }

        private void emitRounded() {
            var positiveShift = code.newLabel();
            var withinProduct = code.newLabel();
            var quotientReady = code.newLabel();
            var afterShiftedLow = code.newLabel();
            code.lload(shift).loadConstant(0L).lcmp().branch(Opcode.IFGT, positiveShift);
            get(24).lload(shift)
                    .lneg()
                    .l2i()
                    .lshl()
                    .lstore(quotient)
                    .branch(Opcode.GOTO, quotientReady)
                    .labelBinding(positiveShift);
            code.lload(shift).iload(bitLength).i2l().lcmp().branch(Opcode.IFLE, withinProduct);
            code.loadConstant(0L)
                    .lstore(quotient)
                    .branch(Opcode.GOTO, quotientReady)
                    .labelBinding(withinProduct);
            code.lload(shift).iload(bitLength).i2l().lcmp();
            var belowLength = code.newLabel();
            code.branch(Opcode.IFLT, belowLength);
            code.loadConstant(0L)
                    .lstore(quotient)
                    .branch(Opcode.GOTO, afterShiftedLow)
                    .labelBinding(belowLength);
            code.lload(shift).loadConstant(6).lushr().l2i().istore(limb);
            code.lload(shift).loadConstant(63L).land().l2i().istore(bitOffset);
            limbOffset(limb);
            code.lstore(position);
            getAt(position).iload(bitOffset).lushr().lstore(quotient);
            code.iload(bitOffset);
            var noNextWord = code.newLabel();
            code.branch(Opcode.IFEQ, noNextWord);
            code.iload(limb)
                    .loadConstant(1)
                    .iadd()
                    .iload(used)
                    .branch(Opcode.IF_ICMPGE, noNextWord);
            code.iinc(limb, 1);
            limbOffset(limb);
            code.lstore(position);
            getAt(position).loadConstant(64).iload(bitOffset).isub().lshl();
            code.lload(quotient)
                    .lor()
                    .lstore(quotient)
                    .labelBinding(noNextWord)
                    .labelBinding(afterShiftedLow)
                    .labelBinding(quotientReady);
            code.loadConstant(0).istore(guard);
            code.loadConstant(0).istore(sticky);
            code.lload(shift).loadConstant(1L).lsub().lstore(position);
            code.lload(position).loadConstant(0L).lcmp();
            var guardDone = code.newLabel();
            code.branch(Opcode.IFLT, guardDone);
            code.lload(position).loadConstant(6).lushr().l2i().istore(limb);
            code.iload(limb).iload(used).branch(Opcode.IF_ICMPGE, guardDone);
            code.lload(position).loadConstant(63L).land().l2i().istore(bitOffset);
            limbOffset(limb);
            code.lstore(position);
            getAt(position)
                    .iload(bitOffset)
                    .lushr()
                    .loadConstant(1L)
                    .land()
                    .l2i()
                    .istore(guard)
                    .labelBinding(guardDone);
            code.lload(shift).loadConstant(1L).lsub().lstore(position);
            code.lload(position).loadConstant(0L).lcmp();
            var stickyDone = code.newLabel();
            code.branch(Opcode.IFLE, stickyDone);
            code.lload(position).loadConstant(6).lushr().l2i().istore(fullLimbs);
            code.lload(position).loadConstant(63L).land().l2i().istore(remainingBits);
            code.loadConstant(0).istore(limb);
            var stickyLoop = code.newLabel();
            var afterFull = code.newLabel();
            code.labelBinding(stickyLoop)
                    .iload(limb)
                    .iload(fullLimbs)
                    .branch(Opcode.IF_ICMPGE, afterFull);
            code.iload(limb).iload(used).branch(Opcode.IF_ICMPGE, afterFull);
            limbOffset(limb);
            code.lstore(position);
            getAt(position).loadConstant(0L).lcmp();
            var next = code.newLabel();
            code.branch(Opcode.IFEQ, next);
            code.loadConstant(1)
                    .istore(sticky)
                    .branch(Opcode.GOTO, stickyDone)
                    .labelBinding(next)
                    .iinc(limb, 1)
                    .branch(Opcode.GOTO, stickyLoop)
                    .labelBinding(afterFull);
            code.iload(remainingBits).branch(Opcode.IFEQ, stickyDone);
            code.iload(fullLimbs).iload(used).branch(Opcode.IF_ICMPGE, stickyDone);
            limbOffset(fullLimbs);
            code.lstore(position);
            getAt(position)
                    .loadConstant(1L)
                    .iload(remainingBits)
                    .lshl()
                    .loadConstant(1L)
                    .lsub()
                    .land()
                    .loadConstant(0L)
                    .lcmp();
            code.branch(Opcode.IFEQ, stickyDone);
            code.loadConstant(1).istore(sticky).labelBinding(stickyDone);
            code.iload(guard);
            var rounded = code.newLabel();
            code.branch(Opcode.IFEQ, rounded);
            code.iload(sticky);
            var roundUp = code.newLabel();
            code.branch(Opcode.IFNE, roundUp);
            code.lload(quotient)
                    .loadConstant(1L)
                    .land()
                    .loadConstant(0L)
                    .lcmp()
                    .branch(Opcode.IFEQ, rounded)
                    .labelBinding(roundUp);
            code.lload(quotient).loadConstant(1L).ladd().lstore(quotient).labelBinding(rounded);
        }

        private void loadBits(int value) {
            switch (type) {
                case FLOAT64 ->
                        code.dload(value)
                                .invokestatic(
                                        DOUBLE_CLASS,
                                        "doubleToRawLongBits",
                                        MethodTypeDesc.of(
                                                ConstantDescs.CD_long, ConstantDescs.CD_double));
                case FLOAT32 ->
                        code.fload(value)
                                .invokestatic(
                                        FLOAT_CLASS,
                                        "floatToRawIntBits",
                                        MethodTypeDesc.of(
                                                ConstantDescs.CD_int, ConstantDescs.CD_float))
                                .i2l()
                                .loadConstant(0xffffffffL)
                                .land();
                case BFLOAT16 -> code.iload(value).i2l().loadConstant(0xffffL).land();
                default -> throw new IllegalArgumentException("not a floating product");
            }
        }

        private void storeResult(int value) {
            switch (type) {
                case FLOAT64 ->
                        code.lload(result)
                                .invokestatic(
                                        DOUBLE_CLASS,
                                        "longBitsToDouble",
                                        MethodTypeDesc.of(
                                                ConstantDescs.CD_double, ConstantDescs.CD_long))
                                .dstore(value);
                case FLOAT32 ->
                        code.lload(result)
                                .l2i()
                                .invokestatic(
                                        FLOAT_CLASS,
                                        "intBitsToFloat",
                                        MethodTypeDesc.of(
                                                ConstantDescs.CD_float, ConstantDescs.CD_int))
                                .fstore(value);
                case BFLOAT16 -> code.lload(result).l2i().istore(value);
                default -> throw new IllegalArgumentException("not a floating product");
            }
        }

        private CodeBuilder get(int delta) {
            segmentOffset(delta);
            return code.invokeinterface(
                    SEGMENT,
                    "get",
                    MethodTypeDesc.of(ConstantDescs.CD_long, LONG_LAYOUT, ConstantDescs.CD_long));
        }

        private CodeBuilder getAt(int address) {
            return code.aload(scratch)
                    .getstatic(VALUE_LAYOUT, "JAVA_LONG", LONG_LAYOUT)
                    .lload(address)
                    .invokeinterface(
                            SEGMENT,
                            "get",
                            MethodTypeDesc.of(
                                    ConstantDescs.CD_long, LONG_LAYOUT, ConstantDescs.CD_long));
        }

        private void set(int delta, Runnable value) {
            segmentOffset(delta);
            value.run();
            code.invokeinterface(
                    SEGMENT,
                    "set",
                    MethodTypeDesc.of(
                            ConstantDescs.CD_void,
                            LONG_LAYOUT,
                            ConstantDescs.CD_long,
                            ConstantDescs.CD_long));
        }

        private void setAt(int address, Runnable value) {
            code.aload(scratch).getstatic(VALUE_LAYOUT, "JAVA_LONG", LONG_LAYOUT).lload(address);
            value.run();
            code.invokeinterface(
                    SEGMENT,
                    "set",
                    MethodTypeDesc.of(
                            ConstantDescs.CD_void,
                            LONG_LAYOUT,
                            ConstantDescs.CD_long,
                            ConstantDescs.CD_long));
        }

        private void segmentOffset(int delta) {
            code.aload(scratch).getstatic(VALUE_LAYOUT, "JAVA_LONG", LONG_LAYOUT).lload(offset);
            if (delta != 0) code.loadConstant((long) delta).ladd();
        }

        private void limbOffset(int index) {
            code.lload(offset)
                    .loadConstant(24L)
                    .ladd()
                    .iload(index)
                    .i2l()
                    .loadConstant(8L)
                    .lmul()
                    .ladd();
        }

        private void limbOffsetMinusOne() {
            code.lload(offset)
                    .loadConstant(24L)
                    .ladd()
                    .iload(used)
                    .loadConstant(1)
                    .isub()
                    .i2l()
                    .loadConstant(8L)
                    .lmul()
                    .ladd();
        }

        private long signMask() {
            return type == DataType.FLOAT64
                    ? 1L << 63
                    : type == DataType.FLOAT32 ? 1L << 31 : 1L << 15;
        }

        private int fractionBits() {
            return type == DataType.FLOAT64 ? 52 : type == DataType.FLOAT32 ? 23 : 7;
        }

        private int bias() {
            return type == DataType.FLOAT64 ? 1023 : 127;
        }

        private int precision() {
            return fractionBits() + 1;
        }

        private int maxExponent() {
            return type == DataType.FLOAT64 ? 1023 : 127;
        }

        private int minimumNormal() {
            return 1 - bias();
        }

        private long fractionMask() {
            return (1L << fractionBits()) - 1;
        }

        private long exponentMask() {
            return type == DataType.FLOAT64 ? 0x7ffL : 0xffL;
        }

        private long signBit() {
            return type == DataType.FLOAT64
                    ? 1L << 63
                    : type == DataType.FLOAT32 ? 1L << 31 : 1L << 15;
        }

        private long canonicalNan() {
            return type == DataType.FLOAT64
                    ? 0x7ff8000000000000L
                    : type == DataType.FLOAT32 ? 0x7fc00000L : 0x7fc0L;
        }

        private long positiveInfinity() {
            return type == DataType.FLOAT64
                    ? 0x7ff0000000000000L
                    : type == DataType.FLOAT32 ? 0x7f800000L : 0x7f80L;
        }
    }

    private static void address(
            CodeBuilder code,
            int g,
            int layout,
            int coordinates,
            int rank,
            int target,
            boolean ints) {
        base(code, g, layout, target, ints);
        for (int axis = 0; axis < rank; axis++) {
            loadAddress(code, target, ints);
            geometry(code, g, coordinates + axis);
            if (ints) code.l2i();
            stride(code, g, layout, rank, axis, ints);
            multiply(code, ints);
            add(code, ints);
            storeAddress(code, target, ints);
        }
    }

    private static void base(CodeBuilder code, int g, int layout, int target, boolean ints) {
        geometry(code, g, layout + 1);
        if (ints) code.l2i().istore(target);
        else code.lstore(target);
    }

    private static void stride(
            CodeBuilder code, int g, int layout, int rank, int axis, boolean ints) {
        geometry(code, g, layout + 2 + rank + axis);
        if (ints) code.l2i();
    }

    private static CodeBuilder geometry(CodeBuilder code, int slot, int index) {
        return code.aload(slot).loadConstant(index).laload();
    }

    private static void advancePacked(
            CodeBuilder code, int g, int coordinates, int layout, int rank) {
        var finished = code.newLabel();
        for (int axis = rank - 1; axis >= 0; axis--) {
            code.aload(g).loadConstant(coordinates + axis);
            geometry(code, g, coordinates + axis).loadConstant(1L).ladd().lastore();
            var carry = code.newLabel();
            geometry(code, g, coordinates + axis);
            geometry(code, g, layout + 2 + axis).lcmp().branch(Opcode.IFGE, carry);
            code.branch(Opcode.GOTO, finished)
                    .labelBinding(carry)
                    .aload(g)
                    .loadConstant(coordinates + axis)
                    .loadConstant(0L)
                    .lastore();
        }
        code.labelBinding(finished);
    }

    private static void loadIndex(
            CodeBuilder code,
            CpuCarrierEmitter carriers,
            CpuKernelSpecialization s,
            ScatterEncoding p,
            int address,
            int target,
            boolean ints) {
        carriers.load(
                p.indexType,
                s.carrierPattern().get(p.indexBoundary),
                p.indexBoundary,
                address,
                ints);
        if (ints) {
            if (p.indexType == DataType.INT64) code.l2i();
            code.istore(target);
        } else {
            if (p.indexType == DataType.INT32) code.i2l();
            code.lstore(target);
        }
    }

    private static void compareEnd(
            CodeBuilder code, int value, int end, boolean ints, java.lang.classfile.Label done) {
        if (ints) code.iload(value).iload(end).branch(Opcode.IF_ICMPGE, done);
        else code.lload(value).lload(end).lcmp().branch(Opcode.IFGE, done);
    }

    private static void increment(CodeBuilder code, int local, boolean ints) {
        if (ints) code.iinc(local, 1);
        else code.lload(local).loadConstant(1L).ladd().lstore(local);
    }

    private static void loadAddress(CodeBuilder code, int local, boolean ints) {
        if (ints) code.iload(local);
        else code.lload(local);
    }

    private static void storeAddress(CodeBuilder code, int local, boolean ints) {
        if (ints) code.istore(local);
        else code.lstore(local);
    }

    private static void multiply(CodeBuilder code, boolean ints) {
        if (ints) code.imul();
        else code.lmul();
    }

    private static void add(CodeBuilder code, boolean ints) {
        if (ints) code.iadd();
        else code.ladd();
    }

    private static void loadValue(CodeBuilder code, DataType type, int local) {
        switch (type) {
            case FLOAT64 -> code.dload(local);
            case FLOAT32 -> code.fload(local);
            case INT64 -> code.lload(local);
            case BFLOAT16, INT32, BOOL -> code.iload(local);
        }
    }

    private static void storeValue(CodeBuilder code, DataType type, int local) {
        switch (type) {
            case FLOAT64 -> code.dstore(local);
            case FLOAT32 -> code.fstore(local);
            case INT64 -> code.lstore(local);
            case BFLOAT16, INT32, BOOL -> code.istore(local);
        }
    }

    private static TypeKind localKind(DataType type) {
        return switch (type) {
            case FLOAT64 -> TypeKind.DOUBLE;
            case FLOAT32 -> TypeKind.FLOAT;
            case INT64 -> TypeKind.LONG;
            case BFLOAT16, INT32, BOOL -> TypeKind.INT;
        };
    }

    private record ScatterEncoding(
            String family,
            ScatterReduction reduction,
            int[] boundaryMap,
            int[] ranks,
            int[] layouts,
            int outputRank,
            int dataBoundary,
            int indexBoundary,
            int updateBoundary,
            int outputBoundary,
            DataType dataType,
            DataType indexType) {
        static ScatterEncoding parse(CpuKernelIr ir) {
            String identity = ir.familyIdentity();
            if (!identity.startsWith("scatter:"))
                throw new IllegalArgumentException("unsupported scatter identity");
            int familyEnd = identity.indexOf(':', 8);
            String family = identity.substring(8, familyEnd);
            int reductionStart = identity.indexOf("reduction=") + 10;
            ScatterReduction reduction =
                    ScatterReduction.valueOf(
                            identity.substring(
                                    reductionStart, identity.indexOf(':', reductionStart)));
            String mapText =
                    identity.substring(identity.indexOf("map=") + 4, identity.indexOf(":scratch="));
            int[] boundaryMap =
                    java.util.Arrays.stream(mapText.split(","))
                            .mapToInt(Integer::parseInt)
                            .toArray();
            int count = ir.values().size(),
                    outputRank = ir.values().getLast().accessPlan().iterationRank();
            int[] ranks = new int[count], layouts = new int[count];
            int position =
                    16
                            + 2 * outputRank
                            + ir.values().get(boundaryMap[2]).accessPlan().iterationRank();
            for (int i = 0; i < count; i++) {
                ranks[i] = ir.values().get(i).accessPlan().iterationRank();
                layouts[i] = position;
                position += 2 + 2 * ranks[i];
            }
            int output = count - 1,
                    data = boundaryMap[0],
                    indices = boundaryMap[1],
                    updates = boundaryMap[2];
            return new ScatterEncoding(
                    family,
                    reduction,
                    boundaryMap,
                    ranks,
                    layouts,
                    outputRank,
                    data,
                    indices,
                    updates,
                    output,
                    ir.values().get(data).dataType(),
                    ir.values().get(indices).dataType());
        }
    }
}
