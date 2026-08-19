package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Label;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDescs;
import java.lang.constant.MethodTypeDesc;

/**
 * Emits carrier-, represented-type-, family-, direction-, output-, and access-specialized stable
 * ordering bodies directly into generated CPU classes.
 *
 * <p>Each invocation initializes and stably bottom-up merges logical-axis indices in the two
 * assigned INT64 scratch regions. Generated comparison keeps every floating NaN after non-NaNs,
 * preserves directional signed-zero order and left-on-equality stability, and writes exact
 * represented values and zero-based logical indices. All merge, address, comparison, and output
 * state is primitive and invocation-local; the emitter retains no generated or run state.</p>
 */
public final class CpuOrderingEmitter {
    private static final ClassDesc SEGMENT = ClassDesc.of("java.lang.foreign.MemorySegment");
    private static final ClassDesc VALUE_LAYOUT = ClassDesc.of("java.lang.foreign.ValueLayout");
    private static final ClassDesc LONG_LAYOUT =
            ClassDesc.of("java.lang.foreign.ValueLayout$OfLong");
    private static final ClassDesc DOUBLE_LAYOUT =
            ClassDesc.of("java.lang.foreign.ValueLayout$OfDouble");
    private static final ClassDesc FLOAT_LAYOUT =
            ClassDesc.of("java.lang.foreign.ValueLayout$OfFloat");
    private static final ClassDesc SHORT_LAYOUT =
            ClassDesc.of("java.lang.foreign.ValueLayout$OfShort");
    private static final ClassDesc INT_LAYOUT =
            ClassDesc.of("java.lang.foreign.ValueLayout$OfInt");
    private static final ClassDesc BYTE_LAYOUT =
            ClassDesc.of("java.lang.foreign.ValueLayout$OfByte");
    private static final ClassDesc FLOAT_CLASS = ClassDesc.of("java.lang.Float");

    /** Creates a stateless ordering emitter that retains no carrier or scratch state. */
    public CpuOrderingEmitter() { }

    /**
     * Emits one complete typed stable-ordering body.
     *
     * @param code non-null generated method body
     * @param specialization non-null exact scalar carrier and scratch specialization
     * @param ir non-null instruction-free structural ordering encoding
     * @throws NullPointerException if a required argument is null
     * @throws IllegalArgumentException if the specialization or IR is not an admitted ordering
     *     form
     */
    public void emit(CodeBuilder code, CpuKernelSpecialization specialization, CpuKernelIr ir) {
        Encoding p = Encoding.parse(ir);
        int count = specialization.carrierPattern().size();
        if (count != p.boundaryCount || !specialization.scratchParameter()) {
            throw new IllegalArgumentException("ordering requires exact boundaries and scratch");
        }
        boolean ints = specialization.loopAddressing(ir)
                == CpuKernelSpecialization.LoopAddressing.DENSE_HEAP_ARRAY_INT;
        emitBody(code, specialization, p, ints);
    }

    private static void emitBody(CodeBuilder code, CpuKernelSpecialization s, Encoding p,
            boolean ints) {
        int scratchParameter = p.boundaryCount;
        int geometryParameter = scratchParameter + 1;
        int startParameter = geometryParameter + 1;
        int endParameter = startParameter + 2;
        TypeKind indexKind = ints ? TypeKind.INT : TypeKind.LONG;
        int axis = code.allocateLocal(TypeKind.INT);
        int slice = code.allocateLocal(indexKind), end = code.allocateLocal(indexKind);
        int axisExtent = code.allocateLocal(indexKind), outputCount = code.allocateLocal(indexKind);
        int i = code.allocateLocal(indexKind), width = code.allocateLocal(indexKind);
        int left = code.allocateLocal(indexKind), middle = code.allocateLocal(indexKind);
        int right = code.allocateLocal(indexKind), a = code.allocateLocal(indexKind);
        int b = code.allocateLocal(indexKind), out = code.allocateLocal(indexKind);
        int position = code.allocateLocal(indexKind);
        int leftIndex = code.allocateLocal(TypeKind.LONG), rightIndex = code.allocateLocal(TypeKind.LONG);
        int selectedIndex = code.allocateLocal(TypeKind.LONG);
        int source = code.allocateLocal(TypeKind.LONG), target = code.allocateLocal(TypeKind.LONG);
        int swap = code.allocateLocal(TypeKind.LONG), scratchBase = code.allocateLocal(TypeKind.LONG);
        int regionBytes = code.allocateLocal(TypeKind.LONG);
        LayoutLocals[] layouts = new LayoutLocals[p.boundaryCount];
        for (int boundary = 0; boundary < p.boundaryCount; boundary++) {
            layouts[boundary] = loadLayout(code, geometryParameter, p.layoutOffsets[boundary],
                    p.ranks[boundary], ints);
        }
        int inputAddress = code.allocateLocal(indexKind);
        int firstOutputAddress = code.allocateLocal(indexKind);
        int secondOutputAddress = p.topK ? code.allocateLocal(indexKind) : -1;
        int remainder = code.allocateLocal(indexKind), coordinate = code.allocateLocal(indexKind);
        int inner = code.allocateLocal(indexKind), outer = code.allocateLocal(indexKind);
        int within = code.allocateLocal(indexKind);
        int leftValue = code.allocateLocal(localKind(p.type));
        int rightValue = code.allocateLocal(localKind(p.type));

        geometry(code, geometryParameter, 2).l2i().istore(axis);
        for (LayoutLocals layout : layouts) {
            selectAxisLocal(code, layout.extents, axis, layout.axisExtent, ints);
            selectAxisLocal(code, layout.strides, axis, layout.axisStride, ints);
        }
        if (ints) {
            code.lload(startParameter).l2i().istore(slice);
            code.lload(endParameter).l2i().istore(end);
            selectAxisValue(code, geometryParameter, p.layoutOffsets[0] + 2, p.ranks[0], axis,
                    true, axisExtent);
            if (p.topK) geometry(code, geometryParameter, 3).l2i().istore(outputCount);
            else copy(code, axisExtent, outputCount, true);
        } else {
            code.lload(startParameter).lstore(slice);
            code.lload(endParameter).lstore(end);
            selectAxisValue(code, geometryParameter, p.layoutOffsets[0] + 2, p.ranks[0], axis,
                    false, axisExtent);
            if (p.topK) geometry(code, geometryParameter, 3).lstore(outputCount);
            else copy(code, axisExtent, outputCount, false);
        }
        code.aload(geometryParameter).loadConstant(8).laload()
                .aload(geometryParameter).loadConstant(7).laload().lmul().lstore(scratchBase);
        load(code, axisExtent, ints);
        if (ints) {
            code.i2l();
        }
        code.loadConstant(8L).lmul().lstore(regionBytes);
        prepareDenseMapping(code, layouts[0], axis, inner, ints);

        var sliceLoop = code.newLabel();
        var complete = code.newLabel();
        code.labelBinding(sliceLoop);
        compare(code, slice, end, ints, Opcode.IF_ICMPGE, Opcode.IFGE, complete);
        zero(code, i, ints);
        var initialize = code.newLabel();
        var initialized = code.newLabel();
        code.labelBinding(initialize);
        compare(code, i, axisExtent, ints, Opcode.IF_ICMPGE, Opcode.IFGE, initialized);
        load(code, i, ints);
        if (ints) {
            code.i2l();
        }
        code.lstore(selectedIndex);
        scratchSetAt(code, scratchParameter, scratchBase, i, ints, selectedIndex);
        increment(code, i, ints);
        code.branch(Opcode.GOTO, initialize).labelBinding(initialized);
        code.lload(scratchBase).lstore(source);
        code.lload(scratchBase).lload(regionBytes).ladd().lstore(target);
        one(code, width, ints);
        var pass = code.newLabel();
        var merged = code.newLabel();
        code.labelBinding(pass);
        compare(code, width, axisExtent, ints, Opcode.IF_ICMPGE, Opcode.IFGE, merged);
        zero(code, left, ints);
        var group = code.newLabel();
        var passDone = code.newLabel();
        code.labelBinding(group);
        compare(code, left, axisExtent, ints, Opcode.IF_ICMPGE, Opcode.IFGE, passDone);
        minimum(code, left, width, axisExtent, middle, ints, false);
        minimum(code, left, width, axisExtent, right, ints, true);
        copy(code, left, a, ints);
        copy(code, middle, b, ints);
        copy(code, left, out, ints);
        var merge = code.newLabel();
        var copyLeft = code.newLabel();
        var copyRight = code.newLabel();
        var groupDone = code.newLabel();
        code.labelBinding(merge);
        compare(code, a, middle, ints, Opcode.IF_ICMPGE, Opcode.IFGE, copyRight);
        compare(code, b, right, ints, Opcode.IF_ICMPGE, Opcode.IFGE, copyLeft);
        scratchGetAt(code, scratchParameter, source, a, ints, leftIndex);
        scratchGetAt(code, scratchParameter, source, b, ints, rightIndex);
        emitAddress(code, layouts[0], axis, slice, leftIndex, true, inputAddress, remainder, coordinate,
                inner, outer, within, ints);
        loadCarrier(code, p.type, s.carrierPattern().getFirst(), 0, inputAddress, ints);
        storeValue(code, p.type, leftValue);
        emitAddress(code, layouts[0], axis, slice, rightIndex, true, inputAddress, remainder, coordinate,
                inner, outer, within, ints);
        loadCarrier(code, p.type, s.carrierPattern().getFirst(), 0, inputAddress, ints);
        storeValue(code, p.type, rightValue);
        var chooseLeft = code.newLabel();
        var chooseRight = code.newLabel();
        var advanceMerge = code.newLabel();
        emitComparison(code, p.type, p.descending, leftValue, rightValue, chooseLeft, chooseRight);
        code.labelBinding(chooseLeft);
        scratchSetAt(code, scratchParameter, target, out, ints, leftIndex);
        increment(code, a, ints);
        code.branch(Opcode.GOTO, advanceMerge);
        code.labelBinding(chooseRight);
        scratchSetAt(code, scratchParameter, target, out, ints, rightIndex);
        increment(code, b, ints);
        code.labelBinding(advanceMerge);
        increment(code, out, ints);
        code.branch(Opcode.GOTO, merge);
        code.labelBinding(copyLeft);
        var copyLeftLoop = code.newLabel();
        code.labelBinding(copyLeftLoop);
        compare(code, a, middle, ints, Opcode.IF_ICMPGE, Opcode.IFGE, groupDone);
        scratchGetAt(code, scratchParameter, source, a, ints, leftIndex);
        scratchSetAt(code, scratchParameter, target, out, ints, leftIndex);
        increment(code, a, ints);
        increment(code, out, ints);
        code.branch(Opcode.GOTO, copyLeftLoop);
        code.labelBinding(copyRight);
        var copyRightLoop = code.newLabel();
        code.labelBinding(copyRightLoop);
        compare(code, b, right, ints, Opcode.IF_ICMPGE, Opcode.IFGE, groupDone);
        scratchGetAt(code, scratchParameter, source, b, ints, rightIndex);
        scratchSetAt(code, scratchParameter, target, out, ints, rightIndex);
        increment(code, b, ints);
        increment(code, out, ints);
        code.branch(Opcode.GOTO, copyRightLoop);
        code.labelBinding(groupDone);
        load(code, width, ints);
        constantTwo(code, ints);
        multiply(code, ints);
        load(code, left, ints);
        swapAdd(code, ints);
        store(code, left, ints);
        code.branch(Opcode.GOTO, group).labelBinding(passDone);
        code.lload(source).lstore(swap);
        code.lload(target).lstore(source);
        code.lload(swap).lstore(target);
        if (ints) {
            var doubleWidth = code.newLabel();
            var widthReady = code.newLabel();
            code.iload(width).iload(axisExtent).loadConstant(2).idiv()
                    .branch(Opcode.IF_ICMPLE, doubleWidth);
            code.iload(axisExtent).istore(width).branch(Opcode.GOTO, widthReady)
                    .labelBinding(doubleWidth).iload(width).loadConstant(2).imul().istore(width)
                    .labelBinding(widthReady);
        } else {
            code.lload(width).loadConstant(2L).lmul().lstore(width);
        }
        code.branch(Opcode.GOTO, pass);
        code.labelBinding(merged);
        if (p.topK && !p.sorted) emitIndexOrder(code, scratchParameter, source, outputCount, ints,
                i, a, selectedIndex, leftIndex);
        zero(code, position, ints);
        var output = code.newLabel();
        var sliceDone = code.newLabel();
        code.labelBinding(output);
        compare(code, position, outputCount, ints, Opcode.IF_ICMPGE, Opcode.IFGE, sliceDone);
        scratchGetAt(code, scratchParameter, source, position, ints, selectedIndex);
        if (p.argsort) {
            emitAddress(code, layouts[1], axis, slice, position, false, firstOutputAddress, remainder,
                    coordinate, inner, outer, within, ints);
            storeIndex(code, s.carrierPattern().get(1), 1, firstOutputAddress, selectedIndex, ints);
        } else {
            emitAddress(code, layouts[0], axis, slice, selectedIndex, true, inputAddress, remainder,
                    coordinate, inner, outer, within, ints);
            loadCarrier(code, p.type, s.carrierPattern().getFirst(), 0, inputAddress, ints);
            storeValue(code, p.type, leftValue);
            emitAddress(code, layouts[1], axis, slice, position, false, firstOutputAddress, remainder,
                    coordinate, inner, outer, within, ints);
            storeCarrier(code, p.type, s.carrierPattern().get(1), 1, firstOutputAddress,
                    leftValue, ints);
            if (p.topK) {
                emitAddress(code, layouts[2], axis, slice, position, false, secondOutputAddress, remainder,
                        coordinate, inner, outer, within, ints);
                storeIndex(code, s.carrierPattern().get(2), 2, secondOutputAddress, selectedIndex,
                        ints);
            }
        }
        increment(code, position, ints);
        code.branch(Opcode.GOTO, output);
        code.labelBinding(sliceDone);
        increment(code, slice, ints);
        code.branch(Opcode.GOTO, sliceLoop).labelBinding(complete);
    }

    private static void emitIndexOrder(CodeBuilder code, int scratchParameter, int source,
            int count, boolean ints, int i, int j, int value, int previous) {
        one(code, i, ints);
        var outer = code.newLabel();
        var done = code.newLabel();
        code.labelBinding(outer);
        compare(code, i, count, ints, Opcode.IF_ICMPGE, Opcode.IFGE, done);
        scratchGetAt(code, scratchParameter, source, i, ints, value);
        copy(code, i, j, ints);
        var inner = code.newLabel();
        var ordered = code.newLabel();
        var insert = code.newLabel();
        code.labelBinding(inner);
        zeroCompare(code, j, ints, insert);
        decrement(code, j, ints);
        scratchGetAt(code, scratchParameter, source, j, ints, previous);
        code.lload(previous).lload(value).lcmp().branch(Opcode.IFLE, ordered);
        increment(code, j, ints);
        scratchSetAt(code, scratchParameter, source, j, ints, previous);
        decrement(code, j, ints);
        code.branch(Opcode.GOTO, inner);
        code.labelBinding(ordered);
        increment(code, j, ints);
        code.labelBinding(insert);
        scratchSetAt(code, scratchParameter, source, j, ints, value);
        increment(code, i, ints);
        code.branch(Opcode.GOTO, outer).labelBinding(done);
    }

    private static void emitComparison(CodeBuilder code, DataType type, boolean descending,
            int left, int right, Label chooseLeft, Label chooseRight) {
        if (type == DataType.FLOAT32 || type == DataType.BFLOAT16) {
            if (type == DataType.BFLOAT16) {
                code.iload(left).loadConstant(16).ishl().invokestatic(FLOAT_CLASS, "intBitsToFloat",
                        MethodTypeDesc.of(ConstantDescs.CD_float, ConstantDescs.CD_int)).fstore(left);
                code.iload(right).loadConstant(16).ishl().invokestatic(FLOAT_CLASS, "intBitsToFloat",
                        MethodTypeDesc.of(ConstantDescs.CD_float, ConstantDescs.CD_int)).fstore(right);
            }
            Label leftFinite = code.newLabel();
            Label rightNan = code.newLabel();
            Label equal = code.newLabel();
            code.fload(left).fload(left).fcmpl().branch(Opcode.IFEQ, leftFinite);
            code.fload(right).fload(right).fcmpl().branch(Opcode.IFNE, chooseLeft);
            code.branch(Opcode.GOTO, chooseRight).labelBinding(leftFinite);
            code.fload(right).fload(right).fcmpl().branch(Opcode.IFNE, rightNan);
            code.fload(left).fload(right).fcmpg()
                    .branch(descending ? Opcode.IFGT : Opcode.IFLT, chooseLeft);
            code.fload(left).fload(right).fcmpg()
                    .branch(descending ? Opcode.IFLT : Opcode.IFGT, chooseRight);
            code.branch(Opcode.GOTO, equal).labelBinding(rightNan)
                    .branch(Opcode.GOTO, chooseLeft);
            code.labelBinding(equal);
            code.fload(left).loadConstant(0.0f).fcmpg().branch(Opcode.IFNE, chooseLeft);
            code.loadConstant(1.0f).fload(left).fdiv();
            code.loadConstant(1.0f).fload(right).fdiv();
            code.fcmpg().branch(descending ? Opcode.IFGE : Opcode.IFLE, chooseLeft);
            code.branch(Opcode.GOTO, chooseRight);
            return;
        }
        if (type == DataType.FLOAT64) {
            Label leftFinite = code.newLabel();
            Label rightNan = code.newLabel();
            Label equal = code.newLabel();
            code.dload(left).dload(left).dcmpl().branch(Opcode.IFEQ, leftFinite);
            code.dload(right).dload(right).dcmpl().branch(Opcode.IFNE, chooseLeft);
            code.branch(Opcode.GOTO, chooseRight).labelBinding(leftFinite);
            code.dload(right).dload(right).dcmpl().branch(Opcode.IFNE, rightNan);
            code.dload(left).dload(right).dcmpg()
                    .branch(descending ? Opcode.IFGT : Opcode.IFLT, chooseLeft);
            code.dload(left).dload(right).dcmpg()
                    .branch(descending ? Opcode.IFLT : Opcode.IFGT, chooseRight);
            code.branch(Opcode.GOTO, equal).labelBinding(rightNan)
                    .branch(Opcode.GOTO, chooseLeft);
            code.labelBinding(equal);
            code.dload(left).loadConstant(0.0d).dcmpg().branch(Opcode.IFNE, chooseLeft);
            code.loadConstant(1.0d).dload(left).ddiv();
            code.loadConstant(1.0d).dload(right).ddiv();
            code.dcmpg().branch(descending ? Opcode.IFGE : Opcode.IFLE, chooseLeft);
            code.branch(Opcode.GOTO, chooseRight);
            return;
        }
        loadValue(code, type, left);
        loadValue(code, type, right);
        if (type == DataType.INT64) {
            code.lcmp().branch(descending ? Opcode.IFGE : Opcode.IFLE, chooseLeft);
        } else {
            code.branch(descending ? Opcode.IF_ICMPGE : Opcode.IF_ICMPLE, chooseLeft);
        }
        code.branch(Opcode.GOTO, chooseRight);
    }

    private static void emitAddress(CodeBuilder code, LayoutLocals layout, int axis, int slice,
            int axisIndex, boolean axisIndexLong, int target, int remainder, int coordinate, int inner, int outer,
            int within, boolean ints) {
        if (ints) {
            code.iload(slice).iload(inner).idiv().istore(outer);
            code.iload(slice).iload(outer).iload(inner).imul().isub().istore(within);
            code.iload(layout.base).iload(outer).iload(layout.axisExtent).iload(inner).imul().imul()
                    .iload(within).iadd();
            if (axisIndexLong) {
                code.lload(axisIndex).l2i();
            } else {
                code.iload(axisIndex);
            }
            code.iload(inner).imul().iadd().iadd().istore(target);
            return;
        }
        code.lload(layout.base).lstore(target);
        code.lload(slice).lstore(remainder);
        for (int current = layout.extents.length - 1; current >= 0; current--) {
            var skip = code.newLabel();
            code.iload(axis).loadConstant(current).branch(Opcode.IF_ICMPEQ, skip);
            code.lload(remainder).lload(layout.extents[current]).lrem().lstore(coordinate);
            code.lload(remainder).lload(layout.extents[current]).ldiv().lstore(remainder);
            code.lload(target).lload(coordinate).lload(layout.strides[current]).lmul().ladd()
                    .lstore(target);
            code.labelBinding(skip);
        }
        code.lload(target);
        code.lload(axisIndex);
        code.lload(layout.axisStride).lmul().ladd().lstore(target);
    }

    private static void prepareDenseMapping(CodeBuilder code, LayoutLocals layout, int axis,
            int inner, boolean ints) {
        if (ints) {
            code.loadConstant(1).istore(inner);
            for (int current = 0; current < layout.extents.length; current++) {
                var skip = code.newLabel();
                code.loadConstant(current).iload(axis).branch(Opcode.IF_ICMPLE, skip);
                code.iload(inner).iload(layout.extents[current]).imul().istore(inner);
                code.labelBinding(skip);
            }
        }
    }

    private static LayoutLocals loadLayout(CodeBuilder code, int geometry, int offset, int rank,
            boolean ints) {
        TypeKind indexKind = ints ? TypeKind.INT : TypeKind.LONG;
        int base = code.allocateLocal(indexKind);
        geometry(code, geometry, offset + 1);
        if (ints) {
            code.l2i().istore(base);
        } else {
            code.lstore(base);
        }

        int[] extents = new int[rank];
        int[] strides = new int[rank];
        int product = code.allocateLocal(indexKind);
        int axisExtent = code.allocateLocal(indexKind);
        int axisStride = code.allocateLocal(indexKind);
        if (ints) {
            code.loadConstant(1).istore(product);
        } else {
            code.loadConstant(1L).lstore(product);
        }
        for (int index = 0; index < rank; index++) {
            extents[index] = code.allocateLocal(indexKind);
            geometry(code, geometry, offset + 2 + index);
            if (ints) {
                code.l2i().istore(extents[index]);
                code.iload(product).iload(extents[index]).imul().istore(product);
            } else {
                code.lstore(extents[index]);
            }
        }
        for (int index = 0; index < rank; index++) {
            strides[index] = code.allocateLocal(indexKind);
            geometry(code, geometry, offset + 2 + rank + index);
            if (ints) {
                code.l2i().istore(strides[index]);
            } else {
                code.lstore(strides[index]);
            }
        }
        return new LayoutLocals(base, extents, strides, product, axisExtent, axisStride);
    }

    private static void selectAxisValue(CodeBuilder code, int geometry, int first, int rank,
            int axis, boolean ints, int target) {
        zero(code, target, ints);
        Label done = code.newLabel();
        for (int index = 0; index < rank; index++) {
            Label next = code.newLabel();
            code.iload(axis).loadConstant(index).branch(Opcode.IF_ICMPNE, next);
            geometry(code, geometry, first + index);
            if (ints) {
                code.l2i().istore(target);
            } else {
                code.lstore(target);
            }
            code.branch(Opcode.GOTO, done).labelBinding(next);
        }
        code.labelBinding(done);
    }

    private static void selectAxisLocal(CodeBuilder code, int[] values, int axis, int target,
            boolean ints) {
        zero(code, target, ints);
        Label done = code.newLabel();
        for (int index = 0; index < values.length; index++) {
            Label next = code.newLabel();
            code.iload(axis).loadConstant(index).branch(Opcode.IF_ICMPNE, next);
            load(code, values[index], ints);
            store(code, target, ints);
            code.branch(Opcode.GOTO, done).labelBinding(next);
        }
        code.labelBinding(done);
    }

    private static void minimum(CodeBuilder code, int left, int width, int extent, int target,
            boolean ints, boolean twice) {
        load(code, left, ints);
        load(code, width, ints);
        if (twice) {
            constantTwo(code, ints);
            multiply(code, ints);
        }
        add(code, ints);
        store(code, target, ints);
        Label withinExtent = code.newLabel();
        compare(code, target, extent, ints, Opcode.IF_ICMPLE, Opcode.IFLE, withinExtent);
        copy(code, extent, target, ints);
        code.labelBinding(withinExtent);
    }

    private static void scratchGetAt(CodeBuilder code, int scratch, int base, int index,
            boolean ints, int target) {
        code.aload(scratch).getstatic(VALUE_LAYOUT, "JAVA_LONG", LONG_LAYOUT).lload(base);
        load(code, index, ints);
        if (ints) {
            code.i2l();
        }
        code.loadConstant(8L).lmul().ladd()
                .invokeinterface(SEGMENT, "get", MethodTypeDesc.of(ConstantDescs.CD_long,
                        LONG_LAYOUT, ConstantDescs.CD_long))
                .lstore(target);
    }

    private static void scratchSetAt(CodeBuilder code, int scratch, int base, int index,
            boolean ints, int value) {
        code.aload(scratch).getstatic(VALUE_LAYOUT, "JAVA_LONG", LONG_LAYOUT).lload(base);
        load(code, index, ints);
        if (ints) {
            code.i2l();
        }
        code.loadConstant(8L).lmul().ladd().lload(value)
                .invokeinterface(SEGMENT, "set", MethodTypeDesc.of(ConstantDescs.CD_void,
                        LONG_LAYOUT, ConstantDescs.CD_long, ConstantDescs.CD_long));
    }

    private static void storeIndex(CodeBuilder code, CarrierAccess access, int parameter,
            int address, int value, boolean ints) {
        storeCarrier(code, DataType.INT64, access, parameter, address, value, ints);
    }

    private static void loadCarrier(CodeBuilder code, DataType type, CarrierAccess access,
            int parameter, int address, boolean ints) {
        if (access != CarrierAccess.MEMORY_SEGMENT) {
            new CpuCarrierEmitter(code).load(type, access, parameter, address, ints);
            return;
        }
        code.aload(parameter).getstatic(VALUE_LAYOUT, layoutField(type), layoutClass(type));
        byteOffset(code, type, address, ints);
        code.invokeinterface(SEGMENT, "get", MethodTypeDesc.of(primitive(type), layoutClass(type),
                ConstantDescs.CD_long));
    }

    private static void storeCarrier(CodeBuilder code, DataType type, CarrierAccess access,
            int parameter, int address, int value, boolean ints) {
        if (access != CarrierAccess.MEMORY_SEGMENT) {
            new CpuCarrierEmitter(code).store(type, access, parameter, address, value, ints);
            return;
        }
        code.aload(parameter).getstatic(VALUE_LAYOUT, layoutField(type), layoutClass(type));
        byteOffset(code, type, address, ints);
        loadValue(code, type, value);
        code.invokeinterface(SEGMENT, "set", MethodTypeDesc.of(ConstantDescs.CD_void,
                layoutClass(type), ConstantDescs.CD_long, primitive(type)));
    }

    private static void byteOffset(CodeBuilder code, DataType type, int address, boolean ints) {
        load(code, address, ints);
        if (ints) {
            code.i2l();
        }
        code.loadConstant((long) type.byteWidth()).lmul();
    }

    private static String layoutField(DataType type) {
        return switch (type) {
            case FLOAT64 -> "JAVA_DOUBLE_UNALIGNED";
            case FLOAT32 -> "JAVA_FLOAT_UNALIGNED";
            case BFLOAT16 -> "JAVA_SHORT_UNALIGNED";
            case INT32 -> "JAVA_INT_UNALIGNED";
            case INT64 -> "JAVA_LONG_UNALIGNED";
            case BOOL -> "JAVA_BYTE";
        };
    }

    private static ClassDesc layoutClass(DataType type) {
        return switch (type) {
            case FLOAT64 -> DOUBLE_LAYOUT;
            case FLOAT32 -> FLOAT_LAYOUT;
            case BFLOAT16 -> SHORT_LAYOUT;
            case INT32 -> INT_LAYOUT;
            case INT64 -> LONG_LAYOUT;
            case BOOL -> BYTE_LAYOUT;
        };
    }

    private static ClassDesc primitive(DataType type) {
        return switch (type) {
            case FLOAT64 -> ConstantDescs.CD_double;
            case FLOAT32 -> ConstantDescs.CD_float;
            case BFLOAT16 -> ConstantDescs.CD_short;
            case INT32 -> ConstantDescs.CD_int;
            case INT64 -> ConstantDescs.CD_long;
            case BOOL -> ConstantDescs.CD_byte;
        };
    }

    private static CodeBuilder geometry(CodeBuilder code, int parameter, int index) {
        return code.aload(parameter).loadConstant(index).laload();
    }

    private static void compare(CodeBuilder code, int left, int right, boolean ints,
            Opcode integerOpcode, Opcode longOpcode, Label target) {
        load(code, left, ints);
        load(code, right, ints);
        if (ints) {
            code.branch(integerOpcode, target);
        } else {
            code.lcmp().branch(longOpcode, target);
        }
    }

    private static void load(CodeBuilder code, int local, boolean ints) {
        if (ints) {
            code.iload(local);
        } else {
            code.lload(local);
        }
    }

    private static void store(CodeBuilder code, int local, boolean ints) {
        if (ints) {
            code.istore(local);
        } else {
            code.lstore(local);
        }
    }

    private static void copy(CodeBuilder code, int source, int target, boolean ints) {
        load(code, source, ints);
        store(code, target, ints);
    }

    private static void zero(CodeBuilder code, int local, boolean ints) {
        if (ints) {
            code.loadConstant(0).istore(local);
        } else {
            code.loadConstant(0L).lstore(local);
        }
    }

    private static void one(CodeBuilder code, int local, boolean ints) {
        if (ints) {
            code.loadConstant(1).istore(local);
        } else {
            code.loadConstant(1L).lstore(local);
        }
    }

    private static void increment(CodeBuilder code, int local, boolean ints) {
        if (ints) {
            code.iinc(local, 1);
        } else {
            code.lload(local).loadConstant(1L).ladd().lstore(local);
        }
    }

    private static void decrement(CodeBuilder code, int local, boolean ints) {
        if (ints) {
            code.iinc(local, -1);
        } else {
            code.lload(local).loadConstant(1L).lsub().lstore(local);
        }
    }

    private static void zeroCompare(CodeBuilder code, int local, boolean ints, Label target) {
        load(code, local, ints);
        if (ints) {
            code.branch(Opcode.IFLE, target);
        } else {
            code.loadConstant(0L).lcmp().branch(Opcode.IFLE, target);
        }
    }

    private static void constantTwo(CodeBuilder code, boolean ints) {
        if (ints) {
            code.loadConstant(2);
        } else {
            code.loadConstant(2L);
        }
    }

    private static void multiply(CodeBuilder code, boolean ints) {
        if (ints) {
            code.imul();
        } else {
            code.lmul();
        }
    }

    private static void add(CodeBuilder code, boolean ints) {
        if (ints) {
            code.iadd();
        } else {
            code.ladd();
        }
    }

    private static void swapAdd(CodeBuilder code, boolean ints) {
        if (ints) {
            code.iadd();
        } else {
            code.ladd();
        }
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

    private record LayoutLocals(
            int base,
            int[] extents,
            int[] strides,
            int extentsProduct,
            int axisExtent,
            int axisStride) { }

    private record Encoding(
            DataType type,
            boolean descending,
            boolean sorted,
            boolean argsort,
            boolean topK,
            int boundaryCount,
            int[] ranks,
            int[] layoutOffsets) {
        static Encoding parse(CpuKernelIr ir) {
            String identity = ir.familyIdentity();
            if (!identity.startsWith("ordering:") || !ir.instructions().isEmpty()) {
                throw new IllegalArgumentException("unsupported ordering encoding");
            }
            boolean argsort = identity.startsWith("ordering:ARGSORT:");
            boolean topK = identity.startsWith("ordering:TOP_K:");
            boolean descending = identity.contains(":descending=true:");
            boolean sorted = identity.contains(":sorted=true:");
            int boundaryCount = ir.values().size();
            if (boundaryCount != (topK ? 3 : 2)) {
                throw new IllegalArgumentException("ordering boundary count disagrees");
            }
            int[] ranks = new int[boundaryCount];
            int[] offsets = new int[boundaryCount];
            int offset = 11;
            for (int index = 0; index < boundaryCount; index++) {
                ranks[index] = ir.values().get(index).accessPlan().iterationRank();
                offsets[index] = offset;
                offset += 2 + 2 * ranks[index];
            }
            return new Encoding(ir.values().getFirst().dataType(), descending, sorted, argsort,
                    topK, boundaryCount, ranks, offsets);
        }
    }
}
