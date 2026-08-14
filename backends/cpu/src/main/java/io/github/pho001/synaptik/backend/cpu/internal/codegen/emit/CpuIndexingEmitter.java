package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;

/**
 * Emits carrier-, type-, family-, and access-specialized indexing loops into generated classes.
 * Proved dense heap-array forms use integer loop and address state; other admitted layouts and
 * segment or mixed carriers retain typed long-address traversal. Bound execution owns complete
 * logical-domain index validation and invokes these writers only after that validation succeeds,
 * so emitted code performs mapping and represented-value writes without repeating bounds checks.
 * This emitter owns neither validation, worker submission, nor Runtime policy.
 */
public final class CpuIndexingEmitter {
    /** Creates a stateless indexing emitter. */
    public CpuIndexingEmitter() { }

    /**
     * Emits one direct writer for an arbitrary half-open output range. The specialization and IR
     * determine the exact GATHER, GATHER_ELEMENTS, GATHER_ND, or ONE_HOT body and its typed
     * carrier accesses; concrete geometry remains an invocation argument.
     *
     * @param code non-null Class-File method body receiving typed entry arguments
     * @param specialization non-null scalar indexing specialization whose carrier pattern and
     *        access category match {@code ir}
     * @param ir non-null instruction-free structural indexing encoding for one admitted family
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if the indexing encoding is malformed
     */
    public void emit(CodeBuilder code, CpuKernelSpecialization specialization, CpuKernelIr ir) {
        Parsed parsed = Parsed.parse(ir);
        boolean dense = specialization.loopAddressing(ir)
                == CpuKernelSpecialization.LoopAddressing.DENSE_HEAP_ARRAY_INT;
        if (dense && parsed.family.equals("ONE_HOT")) {
            emitDenseOneHot(code, specialization, parsed);
            return;
        }
        if (dense && parsed.family.equals("GATHER_ELEMENTS")) {
            int geometry = parsed.boundaryRanks.length;
            var general = code.newLabel();
            var done = code.newLabel();
            geometry(code, geometry, 3).loadConstant(parsed.outputRank - 1L).lcmp()
                    .branch(Opcode.IFNE, general);
            emitDenseFinalAxisGatherElements(code, specialization, parsed);
            code.branch(Opcode.GOTO, done).labelBinding(general);
            emitMapped(code, specialization, parsed, true);
            code.labelBinding(done);
            return;
        }
        emitMapped(code, specialization, parsed, dense);
    }

    private static void emitDenseOneHot(CodeBuilder code,
            CpuKernelSpecialization specialization, Parsed p) {
        int geometry = p.boundaryRanks.length;
        int coordinate = 11 + p.map.length;
        int indexLayout = p.layouts[p.indexBoundary];
        int outputLayout = p.layouts[p.outputBoundary];
        int logical = code.allocateLocal(TypeKind.INT);
        int end = code.allocateLocal(TypeKind.INT);
        int indexAddress = code.allocateLocal(TypeKind.INT);
        int outputAddress = code.allocateLocal(TypeKind.INT);
        int classification = code.allocateLocal(TypeKind.INT);
        int depth = code.allocateLocal(TypeKind.INT);
        int selected = code.allocateLocal(TypeKind.INT);
        int value = code.allocateLocal(TypeKind.INT);
        code.lload(geometry + 1).l2i().istore(logical);
        code.lload(geometry + 3).l2i().istore(end);
        geometry(code, geometry, indexLayout + 1).l2i().istore(indexAddress);
        for (int axis = 0; axis < p.boundaryRanks[p.indexBoundary]; axis++) {
            code.iload(indexAddress);
            geometry(code, geometry, coordinate + axis).l2i();
            geometry(code, geometry, indexLayout + 2
                    + p.boundaryRanks[p.indexBoundary] + axis).l2i();
            code.imul().iadd().istore(indexAddress);
        }
        geometry(code, geometry, outputLayout + 1).l2i();
        code.iload(logical).iadd().istore(outputAddress);
        geometry(code, geometry, coordinate + p.outputRank - 1).l2i().istore(classification);
        geometry(code, geometry, outputLayout + 2 + p.outputRank - 1).l2i().istore(depth);
        var carriers = new CpuCarrierEmitter(code);
        var loop = code.newLabel();
        var falseValue = code.newLabel();
        var valueReady = code.newLabel();
        var next = code.newLabel();
        var done = code.newLabel();
        code.labelBinding(loop).iload(logical).iload(end).branch(Opcode.IF_ICMPGE, done);
        loadIndex(code, carriers, specialization, p, indexAddress, selected, true);
        code.iload(selected).iload(classification).branch(Opcode.IF_ICMPNE, falseValue);
        code.loadConstant(1).istore(value).branch(Opcode.GOTO, valueReady);
        code.labelBinding(falseValue).loadConstant(0).istore(value).labelBinding(valueReady);
        carriers.store(DataType.BOOL, specialization.carrierPattern().get(p.outputBoundary),
                p.outputBoundary, outputAddress, value, true);
        code.iinc(logical, 1).iinc(outputAddress, 1).iinc(classification, 1);
        code.iload(classification).iload(depth).branch(Opcode.IF_ICMPLT, next);
        code.loadConstant(0).istore(classification).iinc(indexAddress, 1);
        code.labelBinding(next).branch(Opcode.GOTO, loop).labelBinding(done);
    }

    private static void emitDenseFinalAxisGatherElements(CodeBuilder code,
            CpuKernelSpecialization specialization, Parsed p) {
        int geometry = p.boundaryRanks.length;
        int coordinate = 11 + p.map.length;
        int dataLayout = p.layouts[p.dataBoundary];
        int indexLayout = p.layouts[p.indexBoundary];
        int outputLayout = p.layouts[p.outputBoundary];
        int rank = p.outputRank;
        int[] coordinates = new int[rank];
        for (int axis = 0; axis < rank; axis++) {
            coordinates[axis] = code.allocateLocal(TypeKind.INT);
            geometry(code, geometry, coordinate + axis).l2i().istore(coordinates[axis]);
        }
        int logical = code.allocateLocal(TypeKind.INT);
        int end = code.allocateLocal(TypeKind.INT);
        int indexAddress = code.allocateLocal(TypeKind.INT);
        int outputAddress = code.allocateLocal(TypeKind.INT);
        int rowBase = code.allocateLocal(TypeKind.INT);
        int sourceAddress = code.allocateLocal(TypeKind.INT);
        int selected = code.allocateLocal(TypeKind.INT);
        int value = code.allocateLocal(localKind(p.dataType));
        int rowEnd = code.allocateLocal(TypeKind.INT);
        int axisExtent = code.allocateLocal(TypeKind.INT);
        code.lload(geometry + 1).l2i().istore(logical);
        code.lload(geometry + 3).l2i().istore(end);
        geometry(code, geometry, indexLayout + 1).l2i();
        code.iload(logical).iadd().istore(indexAddress);
        geometry(code, geometry, outputLayout + 1).l2i();
        code.iload(logical).iadd().istore(outputAddress);
        geometry(code, geometry, dataLayout + 1).l2i().istore(rowBase);
        geometry(code, geometry, outputLayout + 2 + rank - 1).l2i().istore(axisExtent);
        for (int axis = 0; axis < rank - 1; axis++) {
            code.iload(rowBase).iload(coordinates[axis]);
            geometry(code, geometry, dataLayout + 2 + rank + axis).l2i();
            code.imul().iadd().istore(rowBase);
        }
        var carriers = new CpuCarrierEmitter(code);
        var rows = code.newLabel();
        var rowBoundReady = code.newLabel();
        var loop = code.newLabel();
        var done = code.newLabel();
        code.labelBinding(rows).iload(logical).iload(end).branch(Opcode.IF_ICMPGE, done);
        code.iload(logical).iload(axisExtent).iload(coordinates[rank - 1]).isub().iadd()
                .istore(rowEnd);
        code.iload(rowEnd).iload(end).branch(Opcode.IF_ICMPLE, rowBoundReady);
        code.iload(end).istore(rowEnd);
        code.labelBinding(rowBoundReady).labelBinding(loop);
        loadIndex(code, carriers, specialization, p, indexAddress, selected, true);
        code.iload(rowBase).iload(selected);
        geometry(code, geometry, dataLayout + 2 + 2 * rank - 1).l2i();
        code.imul().iadd().istore(sourceAddress);
        carriers.load(p.dataType, specialization.carrierPattern().get(p.dataBoundary),
                p.dataBoundary, sourceAddress, true);
        store(code, p.dataType, value);
        carriers.store(p.dataType, specialization.carrierPattern().get(p.outputBoundary),
                p.outputBoundary, outputAddress, value, true);
        code.iinc(logical, 1).iinc(indexAddress, 1).iinc(outputAddress, 1)
                .iinc(coordinates[rank - 1], 1);
        code.iload(logical).iload(rowEnd).branch(Opcode.IF_ICMPLT, loop);
        code.iload(logical).iload(end).branch(Opcode.IF_ICMPGE, done);
        code.loadConstant(0).istore(coordinates[rank - 1]);
        advanceDenseRows(code, geometry, coordinates, rowBase, dataLayout, outputLayout);
        code.branch(Opcode.GOTO, rows).labelBinding(done);
    }

    private static void advanceDensePrefix(CodeBuilder code, int geometry, int[] coordinates,
            int rowBase, int dataLayout, int outputLayout) {
        var finished = code.newLabel();
        for (int axis = coordinates.length - 1; axis >= 0; axis--) {
            code.iinc(coordinates[axis], 1);
            var carry = code.newLabel();
            code.iload(coordinates[axis]);
            geometry(code, geometry, outputLayout + 2 + axis).l2i();
            code.branch(Opcode.IF_ICMPGE, carry);
            if (axis < coordinates.length - 1) {
                code.iload(rowBase);
                geometry(code, geometry, dataLayout + 2 + coordinates.length + axis).l2i();
                code.iadd().istore(rowBase);
            }
            code.branch(Opcode.GOTO, finished).labelBinding(carry);
            code.loadConstant(0).istore(coordinates[axis]);
            if (axis < coordinates.length - 1) {
                code.iload(rowBase);
                geometry(code, geometry, outputLayout + 2 + axis).l2i();
                code.loadConstant(1).isub();
                geometry(code, geometry, dataLayout + 2 + coordinates.length + axis).l2i();
                code.imul().isub().istore(rowBase);
            }
        }
        code.labelBinding(finished);
    }

    private static void advanceDenseRows(CodeBuilder code, int geometry, int[] coordinates,
            int rowBase, int dataLayout, int outputLayout) {
        var finished = code.newLabel();
        for (int axis = coordinates.length - 2; axis >= 0; axis--) {
            code.iinc(coordinates[axis], 1);
            var carry = code.newLabel();
            code.iload(coordinates[axis]);
            geometry(code, geometry, outputLayout + 2 + axis).l2i();
            code.branch(Opcode.IF_ICMPGE, carry);
            code.iload(rowBase);
            geometry(code, geometry, dataLayout + 2 + coordinates.length + axis).l2i();
            code.iadd().istore(rowBase).branch(Opcode.GOTO, finished);
            code.labelBinding(carry).loadConstant(0).istore(coordinates[axis]);
            code.iload(rowBase);
            geometry(code, geometry, outputLayout + 2 + axis).l2i().loadConstant(1).isub();
            geometry(code, geometry, dataLayout + 2 + coordinates.length + axis).l2i();
            code.imul().isub().istore(rowBase);
        }
        code.labelBinding(finished);
    }

    private static void emitMapped(CodeBuilder code, CpuKernelSpecialization specialization,
            Parsed p, boolean ints) {
        int geometry = p.boundaryRanks.length;
        int coordinatePosition = 11 + p.map.length;
        TypeKind addressKind = ints ? TypeKind.INT : TypeKind.LONG;
        int[] coordinates = new int[p.outputRank];
        for (int axis = 0; axis < p.outputRank; axis++) {
            coordinates[axis] = code.allocateLocal(addressKind);
        }
        int logical = code.allocateLocal(addressKind);
        int end = code.allocateLocal(addressKind);
        if (ints) {
            code.lload(geometry + 1).l2i().istore(logical);
            code.lload(geometry + 3).l2i().istore(end);
        } else {
            code.lload(geometry + 1).lstore(logical);
            code.lload(geometry + 3).lstore(end);
        }
        int outputAddress = code.allocateLocal(addressKind);
        int sourceAddress = code.allocateLocal(addressKind);
        int indexAddress = code.allocateLocal(addressKind);
        int selected = code.allocateLocal(addressKind);
        int familyValue = code.allocateLocal(localKind(p.family.equals("ONE_HOT")
                ? DataType.BOOL : p.dataType));
        int axisLocal = code.allocateLocal(TypeKind.INT);
        geometry(code, geometry, 3).l2i().istore(axisLocal);
        int batchLocal = code.allocateLocal(TypeKind.INT);
        int tupleLocal = code.allocateLocal(TypeKind.INT);
        geometry(code, geometry, 4).l2i().istore(batchLocal);
        geometry(code, geometry, 5).l2i().istore(tupleLocal);
        var carriers = new CpuCarrierEmitter(code);
        var loop = code.newLabel();
        var done = code.newLabel();
        code.labelBinding(loop);
        if (ints) code.iload(logical).iload(end).branch(Opcode.IF_ICMPGE, done);
        else code.lload(logical).lload(end).lcmp().branch(Opcode.IFGE, done);
        for (int current = 0; current < p.outputRank; current++) {
            geometry(code, geometry, coordinatePosition + current);
            if (ints) code.l2i().istore(coordinates[current]);
            else code.lstore(coordinates[current]);
        }
        addressInto(code, geometry, p.layouts[p.outputBoundary], coordinates, 0,
                p.outputRank, outputAddress, ints);
        switch (p.family) {
            case "GATHER" -> emitGather(code, carriers, specialization, p, geometry, coordinates,
                    axisLocal, indexAddress, selected, sourceAddress, familyValue, ints);
            case "GATHER_ELEMENTS" -> emitGatherElements(code, carriers, specialization, p,
                    geometry, coordinates, axisLocal, indexAddress, selected, sourceAddress,
                    familyValue, ints);
            case "GATHER_ND" -> emitGatherNd(code, carriers, specialization, p, geometry,
                    coordinatePosition, coordinates, batchLocal, tupleLocal, indexAddress,
                    selected, sourceAddress, familyValue, ints);
            case "ONE_HOT" -> emitOneHot(code, carriers, specialization, p, geometry, coordinates,
                    indexAddress, selected, familyValue, ints);
            default -> throw new IllegalArgumentException("unsupported indexing family");
        }
        DataType outputType = p.family.equals("ONE_HOT") ? DataType.BOOL : p.dataType;
        carriers.store(outputType, specialization.carrierPattern().get(p.outputBoundary),
                p.outputBoundary, outputAddress, familyValue, ints);
        advancePacked(code, geometry, coordinatePosition, p.layouts[p.outputBoundary],
                p.outputRank);
        if (ints) code.iinc(logical, 1);
        else code.lload(logical).loadConstant(1L).ladd().lstore(logical);
        code.branch(Opcode.GOTO, loop).labelBinding(done);
    }

    private static void emitGather(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, Parsed p, int geometry, int[] coordinates,
            int axisLocal, int indexAddress, int selected, int sourceAddress, int value,
            boolean ints) {
        addressFromDynamicSlice(code, geometry, p.layouts[p.indexBoundary], coordinates,
                axisLocal, p.boundaryRanks[p.indexBoundary], indexAddress, ints);
        loadIndex(code, carriers, specialization, p, indexAddress, selected, ints);
        base(code, geometry, p.layouts[p.dataBoundary], sourceAddress, ints);
        int dataRank = p.boundaryRanks[p.dataBoundary];
        int indexRank = p.boundaryRanks[p.indexBoundary];
        for (int axis = 0; axis < dataRank; axis++) {
            int coordinate = code.allocateLocal(ints ? TypeKind.INT : TypeKind.LONG);
            var before = code.newLabel();
            var selectedAxis = code.newLabel();
            var ready = code.newLabel();
            code.loadConstant(axis).iload(axisLocal).branch(Opcode.IF_ICMPLT, before);
            code.loadConstant(axis).iload(axisLocal).branch(Opcode.IF_ICMPEQ, selectedAxis);
            int shifted = axis - 1 + indexRank;
            if (shifted >= 0 && shifted < coordinates.length) {
                loadCoordinate(code, coordinates[shifted], ints);
            } else if (ints) code.loadConstant(0);
            else code.loadConstant(0L);
            storeAddress(code, coordinate, ints); code.branch(Opcode.GOTO, ready);
            code.labelBinding(before);
            if (axis < coordinates.length) loadCoordinate(code, coordinates[axis], ints);
            else if (ints) code.loadConstant(0);
            else code.loadConstant(0L);
            storeAddress(code, coordinate, ints); code.branch(Opcode.GOTO, ready);
            code.labelBinding(selectedAxis); loadAddress(code, selected, ints);
            storeAddress(code, coordinate, ints); code.labelBinding(ready);
            addCoordinate(code, geometry, p.layouts[p.dataBoundary], dataRank, sourceAddress,
                    coordinate, axis, ints);
        }
        move(code, carriers, specialization, p, sourceAddress, value, ints);
    }

    private static void emitGatherElements(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, Parsed p, int geometry, int[] coordinates,
            int axisLocal, int indexAddress, int selected, int sourceAddress, int value,
            boolean ints) {
        addressInto(code, geometry, p.layouts[p.indexBoundary], coordinates, 0,
                p.boundaryRanks[p.indexBoundary], indexAddress, ints);
        loadIndex(code, carriers, specialization, p, indexAddress, selected, ints);
        base(code, geometry, p.layouts[p.dataBoundary], sourceAddress, ints);
        int rank = p.boundaryRanks[p.dataBoundary];
        for (int axis = 0; axis < rank; axis++) {
            int coordinate = code.allocateLocal(ints ? TypeKind.INT : TypeKind.LONG);
            var ordinary = code.newLabel();
            var ready = code.newLabel();
            code.loadConstant(axis).iload(axisLocal).branch(Opcode.IF_ICMPNE, ordinary);
            loadAddress(code, selected, ints); storeAddress(code, coordinate, ints);
            code.branch(Opcode.GOTO, ready).labelBinding(ordinary);
            if (axis < coordinates.length) loadCoordinate(code, coordinates[axis], ints);
            else if (ints) code.loadConstant(0);
            else code.loadConstant(0L);
            storeAddress(code, coordinate, ints);
            code.labelBinding(ready);
            addCoordinate(code, geometry, p.layouts[p.dataBoundary], rank, sourceAddress,
                    coordinate, axis, ints);
        }
        move(code, carriers, specialization, p, sourceAddress, value, ints);
    }

    private static void emitGatherNd(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, Parsed p, int geometry, int coordinatePosition,
            int[] coordinates, int batch, int tuple, int indexAddress, int selected,
            int sourceAddress, int value, boolean ints) {
        base(code, geometry, p.layouts[p.dataBoundary], sourceAddress, ints);
        int dataRank = p.boundaryRanks[p.dataBoundary];
        int indexRank = p.boundaryRanks[p.indexBoundary];
        for (int axis = 0; axis < dataRank; axis++) {
            var indexed = code.newLabel();
            var suffix = code.newLabel();
            var ready = code.newLabel();
            int coordinate = code.allocateLocal(ints ? TypeKind.INT : TypeKind.LONG);
            code.loadConstant(axis).iload(batch).branch(Opcode.IF_ICMPGE, indexed);
            if (axis < coordinates.length) loadCoordinate(code, coordinates[axis], ints);
            else if (ints) code.loadConstant(0);
            else code.loadConstant(0L);
            storeAddress(code, coordinate, ints);
            code.branch(Opcode.GOTO, ready).labelBinding(indexed);
            code.loadConstant(axis).iload(batch).iload(tuple).iadd()
                    .branch(Opcode.IF_ICMPGE, suffix);
            base(code, geometry, p.layouts[p.indexBoundary], indexAddress, ints);
            for (int prefix = 0; prefix < indexRank - 1; prefix++) addCoordinate(code, geometry,
                    p.layouts[p.indexBoundary], indexRank, indexAddress, coordinates[prefix],
                    prefix, ints);
            int component = code.allocateLocal(TypeKind.INT);
            code.loadConstant(axis).iload(batch).isub().istore(component);
            loadAddress(code, indexAddress, ints);
            if (ints) code.iload(component); else code.iload(component).i2l();
            stride(code, geometry, p.layouts[p.indexBoundary], indexRank, indexRank - 1, ints);
            multiply(code, ints); add(code, ints); storeAddress(code, indexAddress, ints);
            loadIndex(code, carriers, specialization, p, indexAddress, selected, ints);
            loadAddress(code, selected, ints); storeAddress(code, coordinate, ints);
            code.branch(Opcode.GOTO, ready).labelBinding(suffix);
            int suffixCoordinate = code.allocateLocal(TypeKind.INT);
            code.loadConstant(coordinatePosition + indexRank - 1 + axis).iload(batch).isub()
                    .iload(tuple).isub().istore(suffixCoordinate);
            code.aload(geometry).iload(suffixCoordinate).laload();
            if (ints) code.l2i();
            storeAddress(code, coordinate, ints); code.labelBinding(ready);
            addCoordinate(code, geometry, p.layouts[p.dataBoundary], dataRank, sourceAddress,
                    coordinate, axis, ints);
        }
        move(code, carriers, specialization, p, sourceAddress, value, ints);
    }

    private static void emitOneHot(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, Parsed p, int geometry, int[] coordinates,
            int indexAddress, int selected, int value, boolean ints) {
        addressInto(code, geometry, p.layouts[p.indexBoundary], coordinates, 0,
                p.boundaryRanks[p.indexBoundary], indexAddress, ints);
        loadIndex(code, carriers, specialization, p, indexAddress, selected, ints);
        var falseValue = code.newLabel();
        var ready = code.newLabel();
        loadAddress(code, selected, ints);
        loadCoordinate(code, coordinates[p.outputRank - 1], ints);
        if (ints) code.branch(Opcode.IF_ICMPNE, falseValue);
        else code.lcmp().branch(Opcode.IFNE, falseValue);
        code.loadConstant(1).istore(value).branch(Opcode.GOTO, ready);
        code.labelBinding(falseValue).loadConstant(0).istore(value).labelBinding(ready);
    }

    private static void addressFromDynamicSlice(CodeBuilder code, int geometry, int layout,
            int[] coordinates, int start, int rank, int target, boolean ints) {
        base(code, geometry, layout, target, ints);
        for (int axis = 0; axis < rank; axis++) {
            int coordinate = code.allocateLocal(ints ? TypeKind.INT : TypeKind.LONG);
            int coordinateIndex = code.allocateLocal(TypeKind.INT);
            code.loadConstant(axis).iload(start).iadd().istore(coordinateIndex);
            code.aload(geometry).loadConstant(11).aload(geometry).loadConstant(2).laload().l2i()
                    .iadd().iload(coordinateIndex).iadd().laload();
            if (ints) code.l2i();
            storeAddress(code, coordinate, ints);
            addCoordinate(code, geometry, layout, rank, target, coordinate, axis, ints);
        }
    }

    private static int allocateAddress(CodeBuilder code, int geometry, int layout,
            int[] coordinates, int start, int rank, boolean ints) {
        int result = code.allocateLocal(ints ? TypeKind.INT : TypeKind.LONG);
        addressInto(code, geometry, layout, coordinates, start, rank, result, ints);
        return result;
    }

    private static void addressInto(CodeBuilder code, int geometry, int layout,
            int[] coordinates, int start, int rank, int target, boolean ints) {
        base(code, geometry, layout, target, ints);
        for (int axis = 0; axis < rank; axis++) addCoordinate(code, geometry, layout, rank,
                target, coordinates[start + axis], axis, ints);
    }

    private static void addCoordinate(CodeBuilder code, int geometry, int layout, int rank,
            int target, int coordinate, int axis, boolean ints) {
        loadAddress(code, target, ints); loadCoordinate(code, coordinate, ints);
        stride(code, geometry, layout, rank, axis, ints);
        multiply(code, ints); add(code, ints); storeAddress(code, target, ints);
    }

    private static void advance(CodeBuilder code, int geometry, int[] coordinates,
            int outputAddress, int layout, boolean ints) {
        var finished = code.newLabel();
        for (int axis = coordinates.length - 1; axis >= 0; axis--) {
            if (ints) code.iinc(coordinates[axis], 1);
            else code.lload(coordinates[axis]).loadConstant(1L).ladd().lstore(coordinates[axis]);
            var carry = code.newLabel();
            loadCoordinate(code, coordinates[axis], ints); geometry(code, geometry, layout + 2 + axis);
            if (ints) code.l2i().branch(Opcode.IF_ICMPGE, carry);
            else code.lcmp().branch(Opcode.IFGE, carry);
            loadAddress(code, outputAddress, ints);
            stride(code, geometry, layout, coordinates.length, axis, ints);
            add(code, ints); storeAddress(code, outputAddress, ints);
            code.branch(Opcode.GOTO, finished).labelBinding(carry);
            if (ints) code.loadConstant(0).istore(coordinates[axis]);
            else code.loadConstant(0L).lstore(coordinates[axis]);
            loadAddress(code, outputAddress, ints);
            geometry(code, geometry, layout + 2 + axis); if (ints) code.l2i();
            if (ints) code.loadConstant(1).isub();
            else code.loadConstant(1L).lsub();
            stride(code, geometry, layout, coordinates.length, axis, ints);
            multiply(code, ints); subtract(code, ints); storeAddress(code, outputAddress, ints);
        }
        code.labelBinding(finished);
    }

    private static void advancePacked(CodeBuilder code, int geometry, int coordinates,
            int outputLayout, int rank) {
        var finished = code.newLabel();
        for (int axis = rank - 1; axis >= 0; axis--) {
            code.aload(geometry).loadConstant(coordinates + axis);
            geometry(code, geometry, coordinates + axis).loadConstant(1L).ladd().lastore();
            var carry = code.newLabel();
            geometry(code, geometry, coordinates + axis);
            geometry(code, geometry, outputLayout + 2 + axis).lcmp()
                    .branch(Opcode.IFGE, carry);
            code.branch(Opcode.GOTO, finished).labelBinding(carry);
            code.aload(geometry).loadConstant(coordinates + axis).loadConstant(0L).lastore();
        }
        code.labelBinding(finished);
    }

    private static void loadIndex(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, Parsed p, int address, int target,
            boolean ints) {
        carriers.load(p.indexType, specialization.carrierPattern().get(p.indexBoundary),
                p.indexBoundary, address, ints);
        if (ints) {
            if (p.indexType == DataType.INT64) code.l2i();
            code.istore(target);
        } else {
            if (p.indexType == DataType.INT32) code.i2l();
            code.lstore(target);
        }
    }

    private static void move(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, Parsed p, int address, int value,
            boolean ints) {
        carriers.load(p.dataType, specialization.carrierPattern().get(p.dataBoundary),
                p.dataBoundary, address, ints);
        store(code, p.dataType, value);
    }

    private static void base(CodeBuilder code, int geometry, int layout, int target, boolean ints) {
        geometry(code, geometry, layout + 1);
        if (ints) code.l2i().istore(target); else code.lstore(target);
    }

    private static void stride(CodeBuilder code, int geometry, int layout, int rank, int axis,
            boolean ints) {
        geometry(code, geometry, layout + 2 + rank + axis);
        if (ints) code.l2i();
    }

    private static CodeBuilder geometry(CodeBuilder code, int slot, int index) {
        return code.aload(slot).loadConstant(index).laload();
    }

    private static void loadAddress(CodeBuilder code, int local, boolean ints) {
        if (ints) code.iload(local); else code.lload(local);
    }
    private static void loadCoordinate(CodeBuilder code, int local, boolean ints) {
        loadAddress(code, local, ints);
    }
    private static void storeAddress(CodeBuilder code, int local, boolean ints) {
        if (ints) code.istore(local); else code.lstore(local);
    }
    private static void multiply(CodeBuilder code, boolean ints) {
        if (ints) code.imul(); else code.lmul();
    }
    private static void add(CodeBuilder code, boolean ints) {
        if (ints) code.iadd(); else code.ladd();
    }
    private static void subtract(CodeBuilder code, boolean ints) {
        if (ints) code.isub(); else code.lsub();
    }
    private static void store(CodeBuilder code, DataType type, int local) {
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

    private record Parsed(String family, int[] map, int[] boundaryRanks, int[] layouts,
            int outputRank, int dataBoundary, int indexBoundary, int outputBoundary,
            DataType dataType, DataType indexType) {
        static Parsed parse(CpuKernelIr ir) {
            String identity = ir.familyIdentity();
            if (!identity.startsWith("indexing:")) throw new IllegalArgumentException(
                    "unsupported indexing identity");
            int familyEnd = identity.indexOf(':', 9);
            String family = identity.substring(9, familyEnd);
            String mapText = identity.substring(identity.indexOf("map=") + 4);
            int[] map = java.util.Arrays.stream(mapText.split(","))
                    .mapToInt(Integer::parseInt).toArray();
            int count = ir.values().size();
            int[] ranks = new int[count];
            int[] layouts = new int[count];
            int outputRank = ir.values().getLast().accessPlan().iterationRank();
            int position = 11 + map.length + outputRank;
            for (int index = 0; index < count; index++) {
                ranks[index] = ir.values().get(index).accessPlan().iterationRank();
                layouts[index] = position;
                position += 2 + 2 * ranks[index];
            }
            int data = family.equals("ONE_HOT") ? -1 : map[0];
            int indices = family.equals("ONE_HOT") ? map[0] : map[1];
            int output = count - 1;
            DataType dataType = family.equals("ONE_HOT") ? DataType.BOOL
                    : ir.values().get(data).dataType();
            return new Parsed(family, map, ranks, layouts, outputRank, data, indices, output,
                    dataType, ir.values().get(indices).dataType());
        }
    }
}
