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
 * Emits allocation-free typed scatter writers directly into generated classes.
 *
 * <p>Cold-bound execution validates index bounds and replacement uniqueness before an emitted entry
 * can run. For replacement, addition, extrema, and integral multiplication, each invocation first
 * copies the base values in its owned half-open output range and then visits the complete logical
 * update domain once in row-major order, applying only targets in that range. Disjoint ranges
 * therefore preserve deterministic update order without atomics or cross-range state.</p>
 *
 * <p>Floating multiplication deliberately retains output-owned target grouping. One range reuses
 * its single declared primitive-limb scratch slice for each owned output, accumulates that target's
 * complete factor group, and rounds the exact abstract product once. This safe split avoids either
 * intermediate rounding or output-proportional exact state while preserving the established
 * resource shape.</p>
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

    /** Creates a stateless generation-time scatter emitter. */
    public CpuScatterEmitter() {}

    /**
     * Emits one carrier-, type-, family-, reduction-, and access-specialized writer for two through
     * four unique boundaries and optional exact-product scratch. The supplied builder is mutated;
     * this method does not execute scatter work or validate runtime values.
     *
     * @param code generated method body to mutate; not {@code null}
     * @param specialization matching scalar scatter specialization; not {@code null}
     * @param ir instruction-free structural scatter encoding matching the specialization; not
     *     {@code null}
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
        if (specialization.scratchParameter()) {
            if (ints
                    && p.family.equals("SCATTER_ELEMENTS")
                    && p.outputRank == 1
                    && p.ranks[p.indexBoundary] == 1
                    && p.ranks[p.updateBoundary] == 1) {
                emitDenseRankOneGroupedProduct(code, specialization, p, geometrySlot, count);
            } else {
                emitTypedGroupedProduct(code, specialization, p, ints, geometrySlot, count);
            }
        } else if (p.family.equals("SCATTER_ELEMENTS")
                && p.outputRank == 2
                && p.ranks[p.indexBoundary] == 2
                && p.ranks[p.updateBoundary] == 2) {
            emitRankTwoElementsCopyThenUpdate(code, specialization, p, ints, geometrySlot);
        } else {
            emitTypedCopyThenUpdate(code, specialization, p, ints, geometrySlot);
        }
    }

    private static void emitDenseRankOneZeroBase(
            CodeBuilder code, CpuKernelSpecialization s, ScatterEncoding p) {
        int g = p.ranks.length,
                start = code.allocateLocal(TypeKind.INT),
                logical = code.allocateLocal(TypeKind.INT),
                end = code.allocateLocal(TypeKind.INT),
                update = code.allocateLocal(TypeKind.INT);
        int updateCount = code.allocateLocal(TypeKind.INT),
                selected = code.allocateLocal(TypeKind.INT);
        int value = code.allocateLocal(localKind(p.dataType)),
                right = code.allocateLocal(localKind(p.dataType));
        code.lload(g + 1).l2i().istore(start);
        code.iload(start).istore(logical);
        code.lload(g + 3).l2i().istore(end);
        geometry(code, g, p.layouts[p.updateBoundary] + 2).l2i().istore(updateCount);
        var carriers = new CpuCarrierEmitter(code);
        var copy = code.newLabel();
        var copied = code.newLabel();
        code.labelBinding(copy).iload(logical).iload(end).branch(Opcode.IF_ICMPGE, copied);
        carriers.load(
                p.dataType, s.carrierPattern().get(p.dataBoundary), p.dataBoundary, logical, true);
        storeValue(code, p.dataType, value);
        carriers.store(
                p.dataType,
                s.carrierPattern().get(p.outputBoundary),
                p.outputBoundary,
                logical,
                value,
                true);
        code.iinc(logical, 1).branch(Opcode.GOTO, copy).labelBinding(copied);
        code.loadConstant(0).istore(update);
        var updates = code.newLabel();
        var done = code.newLabel();
        code.labelBinding(updates)
                .iload(update)
                .iload(updateCount)
                .branch(Opcode.IF_ICMPGE, done);
        loadIndex(code, carriers, s, p, update, selected, true);
        var skip = code.newLabel();
        code.iload(selected).iload(start).branch(Opcode.IF_ICMPLT, skip);
        code.iload(selected).iload(end).branch(Opcode.IF_ICMPGE, skip);
        carriers.load(
                p.dataType,
                s.carrierPattern().get(p.outputBoundary),
                p.outputBoundary,
                selected,
                true);
        storeValue(code, p.dataType, value);
        carriers.load(
                p.dataType,
                s.carrierPattern().get(p.updateBoundary),
                p.updateBoundary,
                update,
                true);
        storeValue(code, p.dataType, right);
        emitReduction(code, p.dataType, p.reduction, value, right);
        carriers.store(
                p.dataType,
                s.carrierPattern().get(p.outputBoundary),
                p.outputBoundary,
                selected,
                value,
                true);
        code.labelBinding(skip).iinc(update, 1).branch(Opcode.GOTO, updates).labelBinding(done);
    }

    private static void emitDenseRankOne(
            CodeBuilder code, CpuKernelSpecialization s, ScatterEncoding p) {
        int g = p.ranks.length,
                start = code.allocateLocal(TypeKind.INT),
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
        code.lload(g + 1).l2i().istore(start);
        code.iload(start).istore(logical);
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
        var copy = code.newLabel();
        var copied = code.newLabel();
        code.labelBinding(copy).iload(logical).iload(end).branch(Opcode.IF_ICMPGE, copied);
        carriers.load(
                p.dataType,
                s.carrierPattern().get(p.dataBoundary),
                p.dataBoundary,
                dataAddress,
                true);
        storeValue(code, p.dataType, value);
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
                .branch(Opcode.GOTO, copy)
                .labelBinding(copied);
        code.loadConstant(0).istore(update);
        var updates = code.newLabel();
        var done = code.newLabel();
        code.labelBinding(updates)
                .iload(update)
                .iload(updateCount)
                .branch(Opcode.IF_ICMPGE, done);
        code.iload(indexBase).iload(update).iadd().istore(indexAddress);
        code.iload(updateBase).iload(update).iadd().istore(updateAddress);
        loadIndex(code, carriers, s, p, indexAddress, selected, true);
        var skip = code.newLabel();
        code.iload(selected).iload(start).branch(Opcode.IF_ICMPLT, skip);
        code.iload(selected).iload(end).branch(Opcode.IF_ICMPGE, skip);
        geometry(code, g, p.layouts[p.outputBoundary] + 1)
                .l2i()
                .iload(selected)
                .iadd()
                .istore(outputAddress);
        carriers.load(
                p.dataType,
                s.carrierPattern().get(p.outputBoundary),
                p.outputBoundary,
                outputAddress,
                true);
        storeValue(code, p.dataType, value);
        carriers.load(
                p.dataType,
                s.carrierPattern().get(p.updateBoundary),
                p.updateBoundary,
                updateAddress,
                true);
        storeValue(code, p.dataType, right);
        emitReduction(code, p.dataType, p.reduction, value, right);
        carriers.store(
                p.dataType,
                s.carrierPattern().get(p.outputBoundary),
                p.outputBoundary,
                outputAddress,
                value,
                true);
        code.labelBinding(skip).iinc(update, 1).branch(Opcode.GOTO, updates).labelBinding(done);
    }

    private static void emitDenseRankOneGroupedProduct(
            CodeBuilder code,
            CpuKernelSpecialization s,
            ScatterEncoding p,
            int g,
            int scratch) {
        int logical = code.allocateLocal(TypeKind.INT), end = code.allocateLocal(TypeKind.INT);
        int update = code.allocateLocal(TypeKind.INT), updateCount = code.allocateLocal(TypeKind.INT);
        int dataAddress = code.allocateLocal(TypeKind.INT), dataStride = code.allocateLocal(TypeKind.INT);
        int indexAddress = code.allocateLocal(TypeKind.INT), indexBase = code.allocateLocal(TypeKind.INT);
        int indexStride = code.allocateLocal(TypeKind.INT), updateAddress = code.allocateLocal(TypeKind.INT);
        int updateBase = code.allocateLocal(TypeKind.INT), updateStride = code.allocateLocal(TypeKind.INT);
        int outputAddress = code.allocateLocal(TypeKind.INT), outputStride = code.allocateLocal(TypeKind.INT);
        int selected = code.allocateLocal(TypeKind.INT), found = code.allocateLocal(TypeKind.INT);
        int value = code.allocateLocal(localKind(p.dataType));
        int right = code.allocateLocal(localKind(p.dataType));
        var carriers = new CpuCarrierEmitter(code);
        var product = new CpuExactProductEmitter(code, p.dataType, scratch, g, 13, 14, true);

        code.lload(g + 1).l2i().istore(logical);
        code.lload(g + 3).l2i().istore(end);
        loadGeometryAddress(code, g, p.layouts[p.updateBoundary] + 2, updateCount, true);
        loadGeometryAddress(code, g, p.layouts[p.dataBoundary] + 3, dataStride, true);
        loadGeometryAddress(code, g, p.layouts[p.indexBoundary] + 1, indexBase, true);
        loadGeometryAddress(code, g, p.layouts[p.indexBoundary] + 3, indexStride, true);
        loadGeometryAddress(code, g, p.layouts[p.updateBoundary] + 1, updateBase, true);
        loadGeometryAddress(code, g, p.layouts[p.updateBoundary] + 3, updateStride, true);
        loadGeometryAddress(code, g, p.layouts[p.outputBoundary] + 3, outputStride, true);
        geometry(code, g, p.layouts[p.dataBoundary] + 1)
                .l2i()
                .iload(logical)
                .iload(dataStride)
                .imul()
                .iadd()
                .istore(dataAddress);
        geometry(code, g, p.layouts[p.outputBoundary] + 1)
                .l2i()
                .iload(logical)
                .iload(outputStride)
                .imul()
                .iadd()
                .istore(outputAddress);

        var outer = code.newLabel();
        var done = code.newLabel();
        compareEnd(code, logical, end, true, done);
        product.emitScatterReset();
        code.labelBinding(outer);
        compareEnd(code, logical, end, true, done);
        carriers.load(
                p.dataType,
                s.carrierPattern().get(p.dataBoundary),
                p.dataBoundary,
                dataAddress,
                true);
        storeValue(code, p.dataType, value);
        code.loadConstant(0).istore(found);
        code.loadConstant(0).istore(update);
        code.iload(indexBase).istore(indexAddress);
        code.iload(updateBase).istore(updateAddress);
        var updates = code.newLabel();
        var contributionsDone = code.newLabel();
        code.labelBinding(updates)
                .iload(update)
                .iload(updateCount)
                .branch(Opcode.IF_ICMPGE, contributionsDone);
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
        product.emitScatterFactors(value, right, found);
        code.loadConstant(1).istore(found);
        code.labelBinding(noMatch)
                .iinc(update, 1)
                .iload(indexAddress)
                .iload(indexStride)
                .iadd()
                .istore(indexAddress)
                .iload(updateAddress)
                .iload(updateStride)
                .iadd()
                .istore(updateAddress)
                .branch(Opcode.GOTO, updates)
                .labelBinding(contributionsDone);
        product.emitFinish(value, found);
        carriers.store(
                p.dataType,
                s.carrierPattern().get(p.outputBoundary),
                p.outputBoundary,
                outputAddress,
                value,
                true);
        code.iinc(logical, 1)
                .iload(dataAddress)
                .iload(dataStride)
                .iadd()
                .istore(dataAddress)
                .iload(outputAddress)
                .iload(outputStride)
                .iadd()
                .istore(outputAddress)
                .branch(Opcode.GOTO, outer)
                .labelBinding(done);
    }

    private static void emitRankTwoElementsCopyThenUpdate(
            CodeBuilder code,
            CpuKernelSpecialization s,
            ScatterEncoding p,
            boolean ints,
            int g) {
        TypeKind addressKind = ints ? TypeKind.INT : TypeKind.LONG;
        int start = code.allocateLocal(addressKind), logical = code.allocateLocal(addressKind);
        int end = code.allocateLocal(addressKind), row = code.allocateLocal(addressKind);
        int column = code.allocateLocal(addressKind), updateRow = code.allocateLocal(addressKind);
        int updateColumn = code.allocateLocal(addressKind);
        int outputExtent = code.allocateLocal(addressKind), updateRows = code.allocateLocal(addressKind);
        int updateColumns = code.allocateLocal(addressKind);
        int dataBase = code.allocateLocal(addressKind), dataStride0 = code.allocateLocal(addressKind);
        int dataStride1 = code.allocateLocal(addressKind), outputBase = code.allocateLocal(addressKind);
        int outputStride0 = code.allocateLocal(addressKind), outputStride1 = code.allocateLocal(addressKind);
        int indexBase = code.allocateLocal(addressKind), indexStride0 = code.allocateLocal(addressKind);
        int indexStride1 = code.allocateLocal(addressKind), updateBase = code.allocateLocal(addressKind);
        int updateStride0 = code.allocateLocal(addressKind), updateStride1 = code.allocateLocal(addressKind);
        int dataAddress = code.allocateLocal(addressKind), outputAddress = code.allocateLocal(addressKind);
        int indexAddress = code.allocateLocal(addressKind), updateAddress = code.allocateLocal(addressKind);
        int selected = code.allocateLocal(addressKind), targetOrdinal = code.allocateLocal(addressKind);
        int value = code.allocateLocal(localKind(p.dataType)), right = code.allocateLocal(localKind(p.dataType));
        boolean segmentValues =
                s.carrierPattern().get(p.dataBoundary)
                                == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT
                        || s.carrierPattern().get(p.updateBoundary)
                                == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT
                        || s.carrierPattern().get(p.outputBoundary)
                                == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT;
        int valueLayout = segmentValues ? code.allocateLocal(TypeKind.REFERENCE) : -1;
        if (segmentValues) emitPreparedSegmentLayout(code, p.dataType, valueLayout);
        if (ints) {
            code.lload(g + 1).l2i().istore(start);
            code.iload(start).istore(logical);
            code.lload(g + 3).l2i().istore(end);
        } else {
            code.lload(g + 1).lstore(start);
            code.lload(start).lstore(logical);
            code.lload(g + 3).lstore(end);
        }
        loadGeometryAddress(code, g, 18, row, ints);
        loadGeometryAddress(code, g, 19, column, ints);
        loadLayoutFacts(code, g, p.layouts[p.dataBoundary], dataBase, dataStride0, dataStride1, ints);
        loadLayoutFacts(code, g, p.layouts[p.outputBoundary], outputBase, outputStride0, outputStride1, ints);
        loadGeometryAddress(code, g, p.layouts[p.outputBoundary] + 3, outputExtent, ints);
        emitRankTwoAddress(code, dataBase, row, dataStride0, column, dataStride1, dataAddress, ints);
        emitRankTwoAddress(code, outputBase, row, outputStride0, column, outputStride1, outputAddress, ints);
        var carriers = new CpuCarrierEmitter(code);
        var copy = code.newLabel();
        var copied = code.newLabel();
        code.labelBinding(copy);
        compareEnd(code, logical, end, ints, copied);
        scatterLoad(
                code,
                carriers,
                p.dataType,
                s.carrierPattern().get(p.dataBoundary),
                p.dataBoundary,
                dataAddress,
                ints,
                valueLayout);
        storeValue(code, p.dataType, value);
        scatterStore(
                code,
                carriers,
                p.dataType,
                s.carrierPattern().get(p.outputBoundary),
                p.outputBoundary,
                outputAddress,
                value,
                ints,
                valueLayout);
        increment(code, logical, ints);
        increment(code, column, ints);
        var copyCarry = code.newLabel();
        loadAddress(code, column, ints); loadAddress(code, outputExtent, ints);
        if (ints) code.branch(Opcode.IF_ICMPGE, copyCarry);
        else code.lcmp().branch(Opcode.IFGE, copyCarry);
        addLocal(code, dataAddress, dataStride1, ints);
        addLocal(code, outputAddress, outputStride1, ints);
        code.branch(Opcode.GOTO, copy).labelBinding(copyCarry);
        if (ints) code.loadConstant(0).istore(column); else code.loadConstant(0L).lstore(column);
        increment(code, row, ints);
        emitRankTwoAddress(code, dataBase, row, dataStride0, column, dataStride1, dataAddress, ints);
        emitRankTwoAddress(code, outputBase, row, outputStride0, column, outputStride1, outputAddress, ints);
        code.branch(Opcode.GOTO, copy).labelBinding(copied);
        loadLayoutFacts(code, g, p.layouts[p.indexBoundary], indexBase, indexStride0, indexStride1, ints);
        loadLayoutFacts(code, g, p.layouts[p.updateBoundary], updateBase, updateStride0, updateStride1, ints);
        loadGeometryAddress(code, g, p.layouts[p.updateBoundary] + 2, updateRows, ints);
        loadGeometryAddress(code, g, p.layouts[p.updateBoundary] + 3, updateColumns, ints);
        if (ints) {
            code.loadConstant(0).istore(updateRow).loadConstant(0).istore(updateColumn);
            code.iload(indexBase).istore(indexAddress).iload(updateBase).istore(updateAddress);
        } else {
            code.loadConstant(0L).lstore(updateRow).loadConstant(0L).lstore(updateColumn);
            code.lload(indexBase).lstore(indexAddress).lload(updateBase).lstore(updateAddress);
        }
        var updates = code.newLabel();
        var done = code.newLabel();
        code.labelBinding(updates);
        loadAddress(code, updateRow, ints); loadAddress(code, updateRows, ints);
        if (ints) code.branch(Opcode.IF_ICMPGE, done); else code.lcmp().branch(Opcode.IFGE, done);
        loadIndex(code, carriers, s, p, indexAddress, selected, ints);
        var axisOne = code.newLabel();
        var targetReady = code.newLabel();
        geometry(code, g, 6).loadConstant(1L).lcmp().branch(Opcode.IFEQ, axisOne);
        loadAddress(code, selected, ints); loadAddress(code, outputExtent, ints); multiply(code, ints);
        loadAddress(code, updateColumn, ints); add(code, ints); storeAddress(code, targetOrdinal, ints);
        emitRankTwoAddress(code, outputBase, selected, outputStride0, updateColumn, outputStride1, outputAddress, ints);
        code.branch(Opcode.GOTO, targetReady).labelBinding(axisOne);
        loadAddress(code, updateRow, ints); loadAddress(code, outputExtent, ints); multiply(code, ints);
        loadAddress(code, selected, ints); add(code, ints); storeAddress(code, targetOrdinal, ints);
        emitRankTwoAddress(code, outputBase, updateRow, outputStride0, selected, outputStride1, outputAddress, ints);
        code.labelBinding(targetReady);
        var skip = code.newLabel();
        loadAddress(code, targetOrdinal, ints); loadAddress(code, start, ints);
        if (ints) code.branch(Opcode.IF_ICMPLT, skip); else code.lcmp().branch(Opcode.IFLT, skip);
        loadAddress(code, targetOrdinal, ints); loadAddress(code, end, ints);
        if (ints) code.branch(Opcode.IF_ICMPGE, skip); else code.lcmp().branch(Opcode.IFGE, skip);
        scatterLoad(
                code,
                carriers,
                p.dataType,
                s.carrierPattern().get(p.outputBoundary),
                p.outputBoundary,
                outputAddress,
                ints,
                valueLayout);
        storeValue(code, p.dataType, value);
        scatterLoad(
                code,
                carriers,
                p.dataType,
                s.carrierPattern().get(p.updateBoundary),
                p.updateBoundary,
                updateAddress,
                ints,
                valueLayout);
        storeValue(code, p.dataType, right);
        emitReduction(code, p.dataType, p.reduction, value, right);
        scatterStore(
                code,
                carriers,
                p.dataType,
                s.carrierPattern().get(p.outputBoundary),
                p.outputBoundary,
                outputAddress,
                value,
                ints,
                valueLayout);
        code.labelBinding(skip);
        increment(code, updateColumn, ints);
        var updateCarry = code.newLabel();
        loadAddress(code, updateColumn, ints); loadAddress(code, updateColumns, ints);
        if (ints) code.branch(Opcode.IF_ICMPGE, updateCarry); else code.lcmp().branch(Opcode.IFGE, updateCarry);
        addLocal(code, indexAddress, indexStride1, ints);
        addLocal(code, updateAddress, updateStride1, ints);
        code.branch(Opcode.GOTO, updates).labelBinding(updateCarry);
        if (ints) code.loadConstant(0).istore(updateColumn); else code.loadConstant(0L).lstore(updateColumn);
        increment(code, updateRow, ints);
        emitRankTwoAddress(code, indexBase, updateRow, indexStride0, updateColumn, indexStride1, indexAddress, ints);
        emitRankTwoAddress(code, updateBase, updateRow, updateStride0, updateColumn, updateStride1, updateAddress, ints);
        code.branch(Opcode.GOTO, updates).labelBinding(done);
    }

    private static void loadLayoutFacts(CodeBuilder code, int g, int layout, int base, int stride0,
            int stride1, boolean ints) {
        loadGeometryAddress(code, g, layout + 1, base, ints);
        loadGeometryAddress(code, g, layout + 4, stride0, ints);
        loadGeometryAddress(code, g, layout + 5, stride1, ints);
    }

    private static void loadGeometryAddress(CodeBuilder code, int g, int index, int local,
            boolean ints) {
        geometry(code, g, index);
        if (ints) code.l2i().istore(local); else code.lstore(local);
    }

    private static void emitRankTwoAddress(CodeBuilder code, int base, int row, int stride0,
            int column, int stride1, int target, boolean ints) {
        loadAddress(code, base, ints);
        loadAddress(code, row, ints); loadAddress(code, stride0, ints); multiply(code, ints); add(code, ints);
        loadAddress(code, column, ints); loadAddress(code, stride1, ints); multiply(code, ints); add(code, ints);
        storeAddress(code, target, ints);
    }

    private static void addLocal(CodeBuilder code, int target, int increment, boolean ints) {
        loadAddress(code, target, ints); loadAddress(code, increment, ints); add(code, ints);
        storeAddress(code, target, ints);
    }

    private static void emitPreparedSegmentLayout(CodeBuilder code, DataType type, int local) {
        String field =
                switch (type) {
                    case FLOAT64 -> "JAVA_DOUBLE_UNALIGNED";
                    case FLOAT32 -> "JAVA_FLOAT_UNALIGNED";
                    case BFLOAT16 -> "JAVA_SHORT_UNALIGNED";
                    case INT32 -> "JAVA_INT_UNALIGNED";
                    case INT64 -> "JAVA_LONG_UNALIGNED";
                    case BOOL -> "JAVA_BYTE";
                };
        ClassDesc layout = scatterLayoutClass(type);
        code.getstatic(VALUE_LAYOUT, field, layout);
        code.astore(local);
    }

    private static void scatterLoad(
            CodeBuilder code,
            CpuCarrierEmitter carriers,
            DataType type,
            CpuKernelSpecialization.CarrierAccess access,
            int parameter,
            int address,
            boolean ints,
            int layout) {
        if (access != CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT) {
            carriers.load(type, access, parameter, address, ints);
            return;
        }
        code.aload(parameter)
                .aload(layout)
                .lload(address)
                .loadConstant((long) type.byteWidth())
                .lmul()
                .invokeinterface(
                        SEGMENT,
                        "get",
                        MethodTypeDesc.of(
                                scatterPrimitive(type), scatterLayoutClass(type), ConstantDescs.CD_long));
    }

    private static void scatterStore(
            CodeBuilder code,
            CpuCarrierEmitter carriers,
            DataType type,
            CpuKernelSpecialization.CarrierAccess access,
            int parameter,
            int address,
            int value,
            boolean ints,
            int layout) {
        if (access != CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT) {
            carriers.store(type, access, parameter, address, value, ints);
            return;
        }
        code.aload(parameter)
                .aload(layout)
                .lload(address)
                .loadConstant((long) type.byteWidth())
                .lmul();
        loadValue(code, type, value);
        code.invokeinterface(
                        SEGMENT,
                        "set",
                MethodTypeDesc.of(
                        ConstantDescs.CD_void,
                        scatterLayoutClass(type),
                        ConstantDescs.CD_long,
                        scatterPrimitive(type)));
    }

    private static ClassDesc scatterLayoutClass(DataType type) {
        return switch (type) {
            case FLOAT64 -> DOUBLE_LAYOUT;
            case FLOAT32 -> FLOAT_LAYOUT;
            case BFLOAT16 -> SHORT_LAYOUT;
            case INT32 -> INT_LAYOUT;
            case INT64 -> LONG_LAYOUT;
            case BOOL -> BYTE_LAYOUT;
        };
    }

    private static ClassDesc scatterPrimitive(DataType type) {
        return switch (type) {
            case FLOAT64 -> ConstantDescs.CD_double;
            case FLOAT32 -> ConstantDescs.CD_float;
            case BFLOAT16 -> ConstantDescs.CD_short;
            case INT32 -> ConstantDescs.CD_int;
            case INT64 -> ConstantDescs.CD_long;
            case BOOL -> ConstantDescs.CD_byte;
        };
    }

    private static void emitTypedCopyThenUpdate(
            CodeBuilder code,
            CpuKernelSpecialization s,
            ScatterEncoding p,
            boolean ints,
            int g) {
        int targetCoordinate = 16;
        int copyCoordinate = targetCoordinate + p.outputRank;
        int updateCoordinate = copyCoordinate + p.outputRank;
        int start = code.allocateLocal(ints ? TypeKind.INT : TypeKind.LONG);
        int logical = code.allocateLocal(ints ? TypeKind.INT : TypeKind.LONG);
        int end = code.allocateLocal(ints ? TypeKind.INT : TypeKind.LONG);
        int updateOrdinal = code.allocateLocal(ints ? TypeKind.INT : TypeKind.LONG);
        int updateCount = code.allocateLocal(ints ? TypeKind.INT : TypeKind.LONG);
        int targetOrdinal = code.allocateLocal(ints ? TypeKind.INT : TypeKind.LONG);
        int outputAddress = code.allocateLocal(ints ? TypeKind.INT : TypeKind.LONG);
        int dataAddress = code.allocateLocal(ints ? TypeKind.INT : TypeKind.LONG);
        int updateAddress = code.allocateLocal(ints ? TypeKind.INT : TypeKind.LONG);
        int indexAddress = code.allocateLocal(ints ? TypeKind.INT : TypeKind.LONG);
        int selected = code.allocateLocal(ints ? TypeKind.INT : TypeKind.LONG);
        int value = code.allocateLocal(localKind(p.dataType));
        int right = code.allocateLocal(localKind(p.dataType));
        if (ints) {
            code.lload(g + 1).l2i().istore(start);
            code.iload(start).istore(logical);
            code.lload(g + 3).l2i().istore(end);
            code.loadConstant(1).istore(updateCount);
        } else {
            code.lload(g + 1).lstore(start);
            code.lload(start).lstore(logical);
            code.lload(g + 3).lstore(end);
            code.loadConstant(1L).lstore(updateCount);
        }
        for (int axis = 0; axis < p.ranks[p.updateBoundary]; axis++) {
            loadAddress(code, updateCount, ints);
            geometry(code, g, p.layouts[p.updateBoundary] + 2 + axis);
            if (ints) code.l2i();
            multiply(code, ints);
            storeAddress(code, updateCount, ints);
        }
        var carriers = new CpuCarrierEmitter(code);
        var copy = code.newLabel();
        var copied = code.newLabel();
        code.labelBinding(copy);
        compareEnd(code, logical, end, ints, copied);
        address(
                code,
                g,
                p.layouts[p.outputBoundary],
                copyCoordinate,
                p.outputRank,
                outputAddress,
                ints);
        address(
                code,
                g,
                p.layouts[p.dataBoundary],
                copyCoordinate,
                p.outputRank,
                dataAddress,
                ints);
        carriers.load(
                p.dataType,
                s.carrierPattern().get(p.dataBoundary),
                p.dataBoundary,
                dataAddress,
                ints);
        storeValue(code, p.dataType, value);
        carriers.store(
                p.dataType,
                s.carrierPattern().get(p.outputBoundary),
                p.outputBoundary,
                outputAddress,
                value,
                ints);
        advancePacked(code, g, copyCoordinate, p.layouts[p.outputBoundary], p.outputRank);
        increment(code, logical, ints);
        code.branch(Opcode.GOTO, copy).labelBinding(copied);
        for (int axis = 0; axis < p.ranks[p.updateBoundary]; axis++) {
            code.aload(g).loadConstant(updateCoordinate + axis).loadConstant(0L).lastore();
        }
        if (ints) code.loadConstant(0).istore(updateOrdinal);
        else code.loadConstant(0L).lstore(updateOrdinal);
        var updates = code.newLabel();
        var done = code.newLabel();
        code.labelBinding(updates);
        compareEnd(code, updateOrdinal, updateCount, ints, done);
        emitTargetCoordinates(
                code,
                carriers,
                s,
                p,
                g,
                targetCoordinate,
                updateCoordinate,
                indexAddress,
                selected,
                ints);
        if (ints) code.loadConstant(0).istore(targetOrdinal);
        else code.loadConstant(0L).lstore(targetOrdinal);
        for (int axis = 0; axis < p.outputRank; axis++) {
            loadAddress(code, targetOrdinal, ints);
            geometry(code, g, p.layouts[p.outputBoundary] + 2 + axis);
            if (ints) code.l2i();
            multiply(code, ints);
            geometry(code, g, targetCoordinate + axis);
            if (ints) code.l2i();
            add(code, ints);
            storeAddress(code, targetOrdinal, ints);
        }
        var skip = code.newLabel();
        loadAddress(code, targetOrdinal, ints);
        loadAddress(code, start, ints);
        if (ints) code.branch(Opcode.IF_ICMPLT, skip);
        else code.lcmp().branch(Opcode.IFLT, skip);
        loadAddress(code, targetOrdinal, ints);
        loadAddress(code, end, ints);
        if (ints) code.branch(Opcode.IF_ICMPGE, skip);
        else code.lcmp().branch(Opcode.IFGE, skip);
        address(
                code,
                g,
                p.layouts[p.outputBoundary],
                targetCoordinate,
                p.outputRank,
                outputAddress,
                ints);
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
                s.carrierPattern().get(p.outputBoundary),
                p.outputBoundary,
                outputAddress,
                ints);
        storeValue(code, p.dataType, value);
        carriers.load(
                p.dataType,
                s.carrierPattern().get(p.updateBoundary),
                p.updateBoundary,
                updateAddress,
                ints);
        storeValue(code, p.dataType, right);
        emitReduction(code, p.dataType, p.reduction, value, right);
        carriers.store(
                p.dataType,
                s.carrierPattern().get(p.outputBoundary),
                p.outputBoundary,
                outputAddress,
                value,
                ints);
        code.labelBinding(skip);
        advancePacked(
                code, g, updateCoordinate, p.layouts[p.updateBoundary], p.ranks[p.updateBoundary]);
        increment(code, updateOrdinal, ints);
        code.branch(Opcode.GOTO, updates).labelBinding(done);
    }

    private static void emitTargetCoordinates(
            CodeBuilder code,
            CpuCarrierEmitter carriers,
            CpuKernelSpecialization s,
            ScatterEncoding p,
            int g,
            int target,
            int update,
            int indexAddress,
            int selected,
            boolean ints) {
        if (p.family.equals("SCATTER_ELEMENTS")) {
            address(
                    code,
                    g,
                    p.layouts[p.indexBoundary],
                    update,
                    p.ranks[p.indexBoundary],
                    indexAddress,
                    ints);
            loadIndex(code, carriers, s, p, indexAddress, selected, ints);
            for (int axis = 0; axis < p.outputRank; axis++) {
                var selectedAxis = code.newLabel();
                var complete = code.newLabel();
                code.aload(g)
                        .loadConstant(target + axis)
                        .loadConstant(axis)
                        .aload(g)
                        .loadConstant(6)
                        .laload()
                        .l2i()
                        .branch(Opcode.IF_ICMPEQ, selectedAxis);
                geometry(code, g, update + axis).branch(Opcode.GOTO, complete);
                code.labelBinding(selectedAxis);
                loadAddress(code, selected, ints);
                if (ints) code.i2l();
                code.labelBinding(complete).lastore();
            }
            return;
        }
        if (p.family.equals("SCATTER_ADD")) {
            int indexRank = p.ranks[p.indexBoundary];
            base(code, g, p.layouts[p.indexBoundary], indexAddress, ints);
            for (int axis = 0; axis < indexRank; axis++) {
                loadAddress(code, indexAddress, ints);
                code.aload(g)
                        .loadConstant(update)
                        .aload(g)
                        .loadConstant(6)
                        .laload()
                        .l2i()
                        .iadd()
                        .loadConstant(axis)
                        .iadd()
                        .laload();
                if (ints) code.l2i();
                stride(code, g, p.layouts[p.indexBoundary], indexRank, axis, ints);
                multiply(code, ints);
                add(code, ints);
                storeAddress(code, indexAddress, ints);
            }
            loadIndex(code, carriers, s, p, indexAddress, selected, ints);
            for (int axis = 0; axis < p.outputRank; axis++) {
                var selectedAxis = code.newLabel();
                var suffix = code.newLabel();
                var complete = code.newLabel();
                code.aload(g)
                        .loadConstant(target + axis)
                        .loadConstant(axis)
                        .aload(g)
                        .loadConstant(6)
                        .laload()
                        .l2i()
                        .branch(Opcode.IF_ICMPEQ, selectedAxis);
                code.loadConstant(axis)
                        .aload(g)
                        .loadConstant(6)
                        .laload()
                        .l2i()
                        .branch(Opcode.IF_ICMPGT, suffix);
                geometry(code, g, update + axis).branch(Opcode.GOTO, complete);
                code.labelBinding(suffix);
                geometry(code, g, update + indexRank + axis - 1).branch(Opcode.GOTO, complete);
                code.labelBinding(selectedAxis);
                loadAddress(code, selected, ints);
                if (ints) code.i2l();
                code.labelBinding(complete).lastore();
            }
            return;
        }
        emitScatterNdTargetCoordinates(
                code, carriers, s, p, g, target, update, indexAddress, selected, ints);
    }

    private static void emitScatterNdTargetCoordinates(
            CodeBuilder code,
            CpuCarrierEmitter carriers,
            CpuKernelSpecialization s,
            ScatterEncoding p,
            int g,
            int target,
            int update,
            int indexAddress,
            int selected,
            boolean ints) {
        int indexRank = p.ranks[p.indexBoundary];
        for (int axis = 0; axis < p.outputRank; axis++) {
            var tuple = code.newLabel();
            var suffix = code.newLabel();
            var complete = code.newLabel();
            code.aload(g)
                    .loadConstant(target + axis)
                    .loadConstant(axis)
                    .aload(g)
                    .loadConstant(7)
                    .laload()
                    .l2i()
                    .branch(Opcode.IF_ICMPGE, tuple);
            geometry(code, g, update + axis).branch(Opcode.GOTO, complete);
            code.labelBinding(tuple)
                    .loadConstant(axis)
                    .aload(g)
                    .loadConstant(7)
                    .laload()
                    .l2i()
                    .aload(g)
                    .loadConstant(8)
                    .laload()
                    .l2i()
                    .iadd()
                    .branch(Opcode.IF_ICMPGE, suffix);
            base(code, g, p.layouts[p.indexBoundary], indexAddress, ints);
            for (int prefix = 0; prefix < indexRank - 1; prefix++) {
                loadAddress(code, indexAddress, ints);
                geometry(code, g, update + prefix);
                if (ints) code.l2i();
                stride(code, g, p.layouts[p.indexBoundary], indexRank, prefix, ints);
                multiply(code, ints);
                add(code, ints);
                storeAddress(code, indexAddress, ints);
            }
            loadAddress(code, indexAddress, ints);
            if (ints) {
                code.loadConstant(axis).aload(g).loadConstant(7).laload().l2i().isub();
            } else {
                code.loadConstant((long) axis).aload(g).loadConstant(7).laload().lsub();
            }
            stride(code, g, p.layouts[p.indexBoundary], indexRank, indexRank - 1, ints);
            multiply(code, ints);
            add(code, ints);
            storeAddress(code, indexAddress, ints);
            loadIndex(code, carriers, s, p, indexAddress, selected, ints);
            loadAddress(code, selected, ints);
            if (ints) code.i2l();
            code.branch(Opcode.GOTO, complete).labelBinding(suffix);
            code.aload(g)
                    .loadConstant(update + indexRank - 1 + axis)
                    .aload(g)
                    .loadConstant(7)
                    .laload()
                    .l2i()
                    .isub()
                    .aload(g)
                    .loadConstant(8)
                    .laload()
                    .l2i()
                    .isub()
                    .laload();
            code.labelBinding(complete).lastore();
        }
    }

    private static void emitTypedGroupedProduct(
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
        CpuExactProductEmitter product =
                scratch >= 0
                        ? new CpuExactProductEmitter(code, p.dataType, scratch, g, 13, 14, true)
                        : null;
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
        compareEnd(code, logical, end, ints, done);
        if (product != null) product.emitReset();
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
