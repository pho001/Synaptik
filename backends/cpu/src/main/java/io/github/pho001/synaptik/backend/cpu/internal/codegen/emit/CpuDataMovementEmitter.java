package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.MethodTypeDesc;
import java.util.Arrays;

/**
 * Emits one direct allocation-free scalar body for static represented-bit movement.
 *
 * <p>The encoded structural IR chooses PAD, TILE, CONCAT, STACK, UNFOLD_AXIS, UNFOLD2D, or
 * SLICE_UPDATE during class generation. Compact primitive geometry supplies range-start
 * coordinates, carrier bases, rank-specific strides, and family facts at invocation time. For
 * slice update, per-axis target and update-ordinal cursors select the update only at positions in
 * the finite signed sequences and otherwise select the base. Proved dense heap-array forms and
 * proved bounded PAD, CONCAT, UNFOLD_AXIS, and UNFOLD2D forms narrow and hoist geometry into
 * invocation-local integer state independently of carrier kind. Their repeated bodies advance
 * primitive address or coordinate cursors, while geometry that cannot be bounded retains the
 * typed long-address loop. Emitted hot loops contain no Model interpretation, reflection, map
 * lookup, per-element allocation, division, or modulo.</p>
 */
final class CpuDataMovementEmitter {
    /** Creates one stateless family emitter. */
    CpuDataMovementEmitter() {
    }

    /**
     * Emits the family-specialized loop represented by an encoded movement IR.
     *
     * @param code non-null method body receiving generated instructions
     * @param specialization non-null exact carrier and represented-type specialization
     * @param ir non-null instruction-free structural movement encoding
     * @throws IllegalArgumentException if the structural family identity is not a supported
     *     movement encoding
     */
    void emit(CodeBuilder code, CpuKernelSpecialization specialization, CpuKernelIr ir) {
        Parsed parsed = Parsed.parse(ir.familyIdentity());
        if (specialization.loopAddressing(ir)
                == CpuKernelSpecialization.LoopAddressing.DENSE_HEAP_ARRAY_INT) {
            emitDenseArrayInt(code, specialization, ir, parsed);
            return;
        }
        if (parsed.family.equals("PAD") || parsed.family.equals("CONCAT")
                || parsed.family.equals("UNFOLD_AXIS") || parsed.family.equals("UNFOLD2D")) {
            emitBoundedTargetOrGeneralLong(code, specialization, ir, parsed);
            return;
        }
        emitGeneralLong(code, specialization, ir, parsed);
    }

    private static void emitGeneralLong(CodeBuilder code,
            CpuKernelSpecialization specialization, CpuKernelIr ir, Parsed parsed) {
        DataType type = ir.values().getFirst().dataType();
        int rank = parsed.rank;
        int uniqueInputs = ir.values().size() - 1;
        int inputRank = ir.values().getFirst().accessPlan().iterationRank();
        int geometrySlot = uniqueInputs + 1;
        int startSlot = geometrySlot + 1;
        int endSlot = startSlot + 2;
        int[] coordinates = new int[rank];
        for (int axis = 0; axis < rank; axis++) {
            coordinates[axis] = code.allocateLocal(TypeKind.LONG);
            geometry(code, geometrySlot, rank + axis).lstore(coordinates[axis]);
        }
        int outputAddress = code.allocateLocal(TypeKind.LONG);
        geometry(code, geometrySlot, 2 * rank).lstore(outputAddress);
        int logical = code.allocateLocal(TypeKind.LONG);
        code.lload(startSlot).lstore(logical);
        int sourceAddress = code.allocateLocal(TypeKind.LONG);
        int value = code.allocateLocal(switch (type) {
            case FLOAT64 -> TypeKind.DOUBLE;
            case FLOAT32 -> TypeKind.FLOAT;
            case BFLOAT16, INT32, BOOL -> TypeKind.INT;
            case INT64 -> TypeKind.LONG;
        });
        int inputBases = 3 * rank + 1;
        int inputStrides = inputBases + uniqueInputs;
        int[] strideOffsets = new int[uniqueInputs];
        int variant = inputStrides;
        for (int input = 0; input < uniqueInputs; input++) {
            strideOffsets[input] = variant;
            variant += ir.values().get(input).accessPlan().iterationRank();
        }
        int[] tileCoordinates = null;
        int[] sliceTargets = null;
        int[] sliceOrdinals = null;
        if (parsed.family.equals("TILE")) {
            tileCoordinates = new int[rank];
            for (int axis = 0; axis < rank; axis++) {
                tileCoordinates[axis] = code.allocateLocal(TypeKind.LONG);
                geometry(code, geometrySlot, variant + rank + axis)
                        .lstore(tileCoordinates[axis]);
            }
        }
        if (parsed.family.equals("SLICE_UPDATE")) {
            sliceTargets = new int[rank];
            sliceOrdinals = new int[rank];
            for (int axis = 0; axis < rank; axis++) {
                sliceTargets[axis] = code.allocateLocal(TypeKind.LONG);
                geometry(code, geometrySlot, variant + 4 * rank + axis).lstore(sliceTargets[axis]);
                sliceOrdinals[axis] = code.allocateLocal(TypeKind.LONG);
                geometry(code, geometrySlot, variant + 5 * rank + axis).lstore(sliceOrdinals[axis]);
            }
        }
        var carriers = new CpuCarrierEmitter(code);
        var done = code.newLabel();
        var loop = code.newLabel();
        code.labelBinding(loop);
        code.lload(logical).lload(endSlot).lcmp().branch(Opcode.IFGE, done);
        switch (parsed.family) {
            case "PAD" -> emitPad(code, carriers, specialization, type, parsed.bits,
                    geometrySlot, coordinates, sourceAddress, value, inputBases,
                    strideOffsets[0], variant);
            case "TILE" -> emitTile(code, carriers, specialization, type, geometrySlot,
                    tileCoordinates, sourceAddress, value, inputBases, inputStrides);
            case "CONCAT" -> emitConcat(code, carriers, specialization, type, parsed,
                    geometrySlot, coordinates, sourceAddress, value, inputBases,
                    strideOffsets, variant, inputRank);
            case "STACK" -> emitStack(code, carriers, specialization, type, parsed,
                    geometrySlot, coordinates, sourceAddress, value, inputBases,
                    strideOffsets, variant, inputRank);
            case "UNFOLD_AXIS" -> emitUnfoldAxis(code, carriers, specialization, type,
                    geometrySlot, coordinates, sourceAddress, value, inputBases,
                    strideOffsets[0], variant, inputRank);
            case "UNFOLD2D" -> emitUnfold2d(code, carriers, specialization, type, parsed.bits,
                    geometrySlot, coordinates, sourceAddress, value, inputBases,
                    strideOffsets[0], variant);
            case "SLICE_UPDATE" -> emitSliceUpdate(code, carriers, specialization, type, parsed,
                    geometrySlot, coordinates, sliceTargets, sliceOrdinals, sourceAddress, value,
                    inputBases, strideOffsets);
            default -> throw new IllegalArgumentException("unsupported movement family");
        }
        carriers.store(type, specialization.carrierPattern().get(uniqueInputs), uniqueInputs,
                outputAddress, value);
        if (tileCoordinates != null) {
            emitTileAdvance(code, geometrySlot, coordinates, outputAddress, tileCoordinates,
                    variant, 2 * rank + 1);
        } else if (sliceTargets == null) {
            emitAdvance(code, geometrySlot, coordinates, outputAddress, 0, 2 * rank + 1);
        } else {
            emitSliceAdvance(code, geometrySlot, coordinates, outputAddress, sliceTargets,
                    sliceOrdinals, variant, 2 * rank + 1);
        }
        code.lload(logical).loadConstant(1L).ladd().lstore(logical);
        code.branch(Opcode.GOTO, loop);
        code.labelBinding(done);
    }

    private static void emitBoundedTargetOrGeneralLong(CodeBuilder code,
            CpuKernelSpecialization specialization, CpuKernelIr ir, Parsed parsed) {
        int uniqueInputs = ir.values().size() - 1;
        int rank = parsed.rank;
        int geometrySlot = uniqueInputs + 1;
        int startSlot = geometrySlot + 1;
        int endSlot = startSlot + 2;
        int inputStrideCount = 0;
        for (int input = 0; input < uniqueInputs; input++) {
            inputStrideCount += ir.values().get(input).accessPlan().iterationRank();
        }
        int variantSize = switch (parsed.family) {
            case "PAD" -> 2 * rank;
            case "CONCAT" -> 2 + parsed.mapping.length;
            case "UNFOLD_AXIS" -> 3;
            case "UNFOLD2D" -> 18;
            default -> throw new IllegalArgumentException("unsupported bounded movement family");
        };
        int geometryLength = 3 * rank + 1 + uniqueInputs + inputStrideCount + variantSize;
        var fallback = code.newLabel();
        var done = code.newLabel();
        if (parsed.family.equals("CONCAT") && rank == 2
                && ir.values().getFirst().accessPlan().iterationRank() == 2
                && ir.values().getFirst().dataType() == DataType.INT32
                && specialization.carrierPattern().get(uniqueInputs) == CarrierAccess.INT_ARRAY) {
            var bounded = code.newLabel();
            emitEarlyFullDenseIntConcat(code, specialization, parsed, geometrySlot, startSlot,
                    endSlot, uniqueInputs, bounded);
            code.branch(Opcode.GOTO, done);
            code.labelBinding(bounded);
        }
        emitBoundedGeometryCheck(code, geometrySlot, startSlot, endSlot, geometryLength, fallback);
        emitBoundedTarget(code, specialization, ir, parsed, geometryLength);
        code.branch(Opcode.GOTO, done);
        code.labelBinding(fallback);
        emitGeneralLong(code, specialization, ir, parsed);
        code.labelBinding(done);
    }

    private static void emitEarlyFullDenseIntConcat(CodeBuilder code,
            CpuKernelSpecialization specialization, Parsed parsed, int geometrySlot,
            int startSlot, int endSlot, int uniqueInputs, java.lang.classfile.Label rejected) {
        int inputBases = 7;
        int inputStrides = inputBases + uniqueInputs;
        int variant = inputStrides + 2 * uniqueInputs;
        code.lload(startSlot).loadConstant(0L).lcmp().branch(Opcode.IFNE, rejected);
        code.lload(endSlot).loadConstant(0L).lcmp().branch(Opcode.IFLT, rejected);
        code.lload(endSlot).loadConstant((long) Integer.MAX_VALUE).lcmp()
                .branch(Opcode.IFGT, rejected);
        geometry(code, geometrySlot, 0).loadConstant(0L).lcmp()
                .branch(Opcode.IFLT, rejected);
        geometry(code, geometrySlot, 1).loadConstant(0L).lcmp()
                .branch(Opcode.IFLT, rejected);
        geometry(code, geometrySlot, 0);
        geometry(code, geometrySlot, 1);
        code.lmul().lload(endSlot).lcmp().branch(Opcode.IFNE, rejected);
        geometry(code, geometrySlot, 4).loadConstant(0L).lcmp()
                .branch(Opcode.IFNE, rejected);
        geometry(code, geometrySlot, 5);
        geometry(code, geometrySlot, 1);
        code.lcmp().branch(Opcode.IFNE, rejected);
        geometry(code, geometrySlot, 6).loadConstant(1L).lcmp()
                .branch(Opcode.IFNE, rejected);
        geometry(code, geometrySlot, variant).loadConstant(0L).lcmp()
                .branch(Opcode.IFNE, rejected);
        for (int boundary = 0; boundary < uniqueInputs; boundary++) {
            geometry(code, geometrySlot, inputBases + boundary).loadConstant(0L).lcmp()
                    .branch(Opcode.IFNE, rejected);
            geometry(code, geometrySlot, inputStrides + 2 * boundary);
            geometry(code, geometrySlot, 1);
            code.lcmp().branch(Opcode.IFNE, rejected);
            geometry(code, geometrySlot, inputStrides + 2 * boundary + 1)
                    .loadConstant(1L).lcmp().branch(Opcode.IFNE, rejected);
        }
        geometry(code, geometrySlot, variant + 1).loadConstant(0L).lcmp()
                .branch(Opcode.IFNE, rejected);
        int partRows = code.allocateLocal(TypeKind.LONG);
        geometry(code, geometrySlot, variant + 2);
        geometry(code, geometrySlot, variant + 1);
        code.lsub().lstore(partRows);
        code.lload(partRows).loadConstant(0L).lcmp().branch(Opcode.IFLT, rejected);
        for (int occurrence = 0; occurrence < parsed.mapping.length; occurrence++) {
            geometry(code, geometrySlot, variant + 1 + occurrence)
                    .lload(partRows).loadConstant((long) occurrence).lmul().lcmp()
                    .branch(Opcode.IFNE, rejected);
        }
        geometry(code, geometrySlot, 0);
        code.lload(partRows).loadConstant((long) parsed.mapping.length).lmul().lcmp()
                .branch(Opcode.IFNE, rejected);
        int count = code.allocateLocal(TypeKind.INT);
        code.lload(partRows);
        geometry(code, geometrySlot, 1);
        code.lmul().l2i().istore(count);
        int ordinal = code.allocateLocal(TypeKind.INT);
        int[] outputOffsets = new int[parsed.mapping.length];
        for (int occurrence = 0; occurrence < parsed.mapping.length; occurrence++) {
            outputOffsets[occurrence] = code.allocateLocal(TypeKind.INT);
            code.iload(count).loadConstant(occurrence).imul()
                    .istore(outputOffsets[occurrence]);
        }
        code.loadConstant(0).istore(ordinal);
        var loop = code.newLabel();
        var finished = code.newLabel();
        code.labelBinding(loop);
        code.iload(ordinal).iload(count).branch(Opcode.IF_ICMPGE, finished);
        emitFullDenseIntConcatElement(code, specialization, parsed, uniqueInputs, ordinal,
                outputOffsets);
        code.iinc(ordinal, 1).branch(Opcode.GOTO, loop);
        code.labelBinding(finished);
    }

    private static void emitBoundedGeometryCheck(CodeBuilder code, int geometrySlot,
            int startSlot, int endSlot, int geometryLength, java.lang.classfile.Label fallback) {
        for (int index = 0; index < geometryLength; index++) {
            geometry(code, geometrySlot, index).loadConstant(0L).lcmp()
                    .branch(Opcode.IFLT, fallback);
            geometry(code, geometrySlot, index).loadConstant((long) Integer.MAX_VALUE).lcmp()
                    .branch(Opcode.IFGT, fallback);
        }
        code.lload(startSlot).loadConstant(0L).lcmp().branch(Opcode.IFLT, fallback);
        code.lload(startSlot).loadConstant((long) Integer.MAX_VALUE).lcmp()
                .branch(Opcode.IFGT, fallback);
        code.lload(endSlot).loadConstant(0L).lcmp().branch(Opcode.IFLT, fallback);
        code.lload(endSlot).loadConstant((long) Integer.MAX_VALUE).lcmp()
                .branch(Opcode.IFGT, fallback);
    }

    private static void emitBoundedTarget(CodeBuilder code,
            CpuKernelSpecialization specialization, CpuKernelIr ir, Parsed parsed,
            int geometryLength) {
        DataType type = ir.values().getFirst().dataType();
        int rank = parsed.rank;
        int uniqueInputs = ir.values().size() - 1;
        int inputRank = ir.values().getFirst().accessPlan().iterationRank();
        int geometrySlot = uniqueInputs + 1;
        int startSlot = geometrySlot + 1;
        int endSlot = startSlot + 2;
        int[] geometry = new int[geometryLength];
        for (int index = 0; index < geometry.length; index++) {
            geometry[index] = code.allocateLocal(TypeKind.INT);
            code.aload(geometrySlot).loadConstant(index).laload().l2i().istore(geometry[index]);
        }
        int[] coordinates = new int[rank];
        for (int axis = 0; axis < rank; axis++) {
            coordinates[axis] = code.allocateLocal(TypeKind.INT);
            code.iload(geometry[rank + axis]).istore(coordinates[axis]);
        }
        int outputAddress = code.allocateLocal(TypeKind.LONG);
        code.iload(geometry[2 * rank]).i2l().lstore(outputAddress);
        int logical = code.allocateLocal(TypeKind.INT);
        int end = code.allocateLocal(TypeKind.INT);
        code.lload(startSlot).l2i().istore(logical);
        code.lload(endSlot).l2i().istore(end);
        int sourceAddress = code.allocateLocal(TypeKind.LONG);
        code.loadConstant(0L).lstore(sourceAddress);
        int value = code.allocateLocal(switch (type) {
            case FLOAT64 -> TypeKind.DOUBLE;
            case FLOAT32 -> TypeKind.FLOAT;
            case BFLOAT16, INT32, BOOL -> TypeKind.INT;
            case INT64 -> TypeKind.LONG;
        });
        int inputBases = 3 * rank + 1;
        int inputStrides = inputBases + uniqueInputs;
        int[] strideOffsets = new int[uniqueInputs];
        int variant = inputStrides;
        for (int input = 0; input < uniqueInputs; input++) {
            strideOffsets[input] = variant;
            variant += ir.values().get(input).accessPlan().iterationRank();
        }
        switch (parsed.family) {
            case "PAD" -> emitBoundedPad(code, specialization, type, parsed.bits,
                    geometry, coordinates, logical, end, outputAddress, sourceAddress, value,
                    inputBases, strideOffsets[0], variant, uniqueInputs);
            case "CONCAT" -> emitBoundedConcat(code, specialization, type, parsed,
                    geometry, coordinates, logical, end, outputAddress, sourceAddress, value,
                    inputBases, strideOffsets, variant, inputRank, uniqueInputs);
            case "UNFOLD_AXIS" -> emitBoundedUnfoldAxis(code, specialization, type,
                    geometry, coordinates, logical, end, outputAddress, sourceAddress, value,
                    inputBases, strideOffsets[0], variant, inputRank, uniqueInputs);
            case "UNFOLD2D" -> emitBoundedUnfold2d(code, specialization, type,
                    parsed.bits, geometry, coordinates, logical, end, outputAddress,
                    sourceAddress, value, inputBases, strideOffsets[0], variant, uniqueInputs,
                    ir.values().get(uniqueInputs).accessPlan().regime()
                            == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Regime
                                    .DENSE_LINEAR);
            default -> throw new IllegalArgumentException("unsupported bounded movement family");
        }
    }

    private static void emitBoundedPad(CodeBuilder code, CpuKernelSpecialization specialization,
            DataType type, long bits, int[] geometry,
            int[] coordinates, int logical, int end, int outputAddress, int sourceAddress,
            int value, int inputBases, int inputStrides, int variant, int outputSlot) {
        var ranged = code.newLabel();
        var completed = code.newLabel();
        if (coordinates.length == 2) {
            code.iload(logical).branch(Opcode.IFNE, ranged);
            code.iload(geometry[0]).i2l().iload(geometry[1]).i2l().lmul()
                    .iload(end).i2l().lcmp().branch(Opcode.IFNE, ranged);
            emitFullRankTwoPad(code, specialization, type, bits, geometry, outputAddress,
                    sourceAddress, value, inputBases, inputStrides, variant, outputSlot);
            code.branch(Opcode.GOTO, completed);
            code.labelBinding(ranged);
        }
        int sourceValid = code.allocateLocal(TypeKind.INT);
        code.loadConstant(0).istore(sourceValid);
        var done = code.newLabel();
        var loop = code.newLabel();
        code.labelBinding(loop);
        code.iload(logical).iload(end).branch(Opcode.IF_ICMPGE, done);
        var padding = code.newLabel();
        var loaded = code.newLabel();
        for (int axis = 0; axis < coordinates.length; axis++) {
            code.iload(coordinates[axis]).iload(geometry[variant + axis])
                    .branch(Opcode.IF_ICMPLT, padding);
            code.iload(coordinates[axis]).iload(geometry[variant + axis])
                    .iload(geometry[variant + coordinates.length + axis]).iadd()
                    .branch(Opcode.IF_ICMPGE, padding);
        }
        var ready = code.newLabel();
        code.iload(sourceValid).branch(Opcode.IFNE, ready);
        emitBoundedPadAddress(code, geometry, coordinates, sourceAddress, inputBases,
                inputStrides, variant);
        code.loadConstant(1).istore(sourceValid);
        code.labelBinding(ready);
        emitBoundedLoad(code, type, specialization.carrierPattern().getFirst(), 0, sourceAddress);
        storeValue(code, type, value);
        if (coordinates.length > 0) {
            int inner = coordinates.length - 1;
            code.lload(sourceAddress).iload(geometry[inputStrides + inner]).i2l().ladd()
                    .lstore(sourceAddress);
            var keep = code.newLabel();
            code.iload(coordinates[inner]).loadConstant(1).iadd()
                    .iload(geometry[variant + inner])
                    .iload(geometry[variant + coordinates.length + inner]).iadd()
                    .branch(Opcode.IF_ICMPLT, keep);
            code.loadConstant(0).istore(sourceValid);
            code.labelBinding(keep);
        }
        code.branch(Opcode.GOTO, loaded);
        code.labelBinding(padding);
        code.loadConstant(0).istore(sourceValid);
        loadImmediate(code, type, bits);
        storeValue(code, type, value);
        code.labelBinding(loaded);
        emitBoundedStore(code, type, specialization.carrierPattern().get(outputSlot), outputSlot,
                outputAddress, value);
        emitBoundedOutputAdvance(code, geometry, coordinates, outputAddress);
        code.iinc(logical, 1).branch(Opcode.GOTO, loop);
        code.labelBinding(done);
        code.labelBinding(completed);
    }

    private static void emitFullRankTwoPad(CodeBuilder code,
            CpuKernelSpecialization specialization, DataType type, long bits, int[] geometry,
            int outputAddress, int sourceAddress, int value, int inputBases, int inputStrides,
            int variant, int outputSlot) {
        int y = code.allocateLocal(TypeKind.INT);
        int x = code.allocateLocal(TypeKind.INT);
        code.loadConstant(0).istore(y);
        var outer = code.newLabel();
        var outerDone = code.newLabel();
        code.labelBinding(outer);
        code.iload(y).iload(geometry[0]).branch(Opcode.IF_ICMPGE, outerDone);
        code.loadConstant(0).istore(x);
        var inner = code.newLabel();
        var innerDone = code.newLabel();
        code.labelBinding(inner);
        code.iload(x).iload(geometry[1]).branch(Opcode.IF_ICMPGE, innerDone);
        var padding = code.newLabel();
        var loaded = code.newLabel();
        code.iload(y).iload(geometry[variant]).branch(Opcode.IF_ICMPLT, padding);
        code.iload(y).iload(geometry[variant]).iload(geometry[variant + 2]).iadd()
                .branch(Opcode.IF_ICMPGE, padding);
        code.iload(x).iload(geometry[variant + 1]).branch(Opcode.IF_ICMPLT, padding);
        code.iload(x).iload(geometry[variant + 1]).iload(geometry[variant + 3]).iadd()
                .branch(Opcode.IF_ICMPGE, padding);
        code.iload(geometry[inputBases]).i2l()
                .iload(y).iload(geometry[variant]).isub().i2l()
                .iload(geometry[inputStrides]).i2l().lmul().ladd()
                .iload(x).iload(geometry[variant + 1]).isub().i2l()
                .iload(geometry[inputStrides + 1]).i2l().lmul().ladd()
                .lstore(sourceAddress);
        emitBoundedLoad(code, type, specialization.carrierPattern().getFirst(), 0, sourceAddress);
        storeValue(code, type, value);
        code.branch(Opcode.GOTO, loaded);
        code.labelBinding(padding);
        loadImmediate(code, type, bits);
        storeValue(code, type, value);
        code.labelBinding(loaded);
        emitBoundedStore(code, type, specialization.carrierPattern().get(outputSlot), outputSlot,
                outputAddress, value);
        code.lload(outputAddress).iload(geometry[6]).i2l().ladd().lstore(outputAddress);
        code.iinc(x, 1).branch(Opcode.GOTO, inner);
        code.labelBinding(innerDone);
        code.lload(outputAddress).iload(geometry[5]).i2l()
                .iload(geometry[1]).i2l().iload(geometry[6]).i2l().lmul().lsub()
                .ladd().lstore(outputAddress);
        code.iinc(y, 1).branch(Opcode.GOTO, outer);
        code.labelBinding(outerDone);
    }

    private static void emitBoundedPadAddress(CodeBuilder code, int[] geometry,
            int[] coordinates, int sourceAddress, int inputBases, int inputStrides, int variant) {
        code.iload(geometry[inputBases]).i2l().lstore(sourceAddress);
        for (int axis = 0; axis < coordinates.length; axis++) {
            code.lload(sourceAddress).iload(coordinates[axis])
                    .iload(geometry[variant + axis]).isub().i2l()
                    .iload(geometry[inputStrides + axis]).i2l().lmul().ladd()
                    .lstore(sourceAddress);
        }
    }

    private static void emitBoundedConcat(CodeBuilder code,
            CpuKernelSpecialization specialization, DataType type, Parsed parsed, int[] geometry,
            int[] coordinates, int logical, int end, int outputAddress, int sourceAddress,
            int value, int inputBases, int[] strideOffsets, int variant, int inputRank,
            int outputSlot) {
        var ranged = code.newLabel();
        var completed = code.newLabel();
        if (coordinates.length == 2 && inputRank == 2
                && (type != DataType.INT32
                        || specialization.carrierPattern().get(outputSlot)
                                != CarrierAccess.INT_ARRAY)) {
            code.iload(logical).branch(Opcode.IFNE, ranged);
            code.iload(geometry[0]).i2l().iload(geometry[1]).i2l().lmul()
                    .iload(end).i2l().lcmp().branch(Opcode.IFNE, ranged);
            code.iload(geometry[variant]).branch(Opcode.IFNE, ranged);
            code.iload(geometry[5]).iload(geometry[1])
                    .branch(Opcode.IF_ICMPNE, ranged);
            code.iload(geometry[6]).loadConstant(1).branch(Opcode.IF_ICMPNE, ranged);
            code.iload(geometry[4]).i2l().iload(end).i2l().ladd()
                    .loadConstant(1L << 31).lcmp().branch(Opcode.IFGT, ranged);
            for (int boundary = 0; boundary < strideOffsets.length; boundary++) {
                code.iload(geometry[strideOffsets[boundary]]).iload(geometry[1])
                        .branch(Opcode.IF_ICMPNE, ranged);
                code.iload(geometry[strideOffsets[boundary] + 1]).loadConstant(1)
                        .branch(Opcode.IF_ICMPNE, ranged);
            }
            for (int occurrence = 0; occurrence < parsed.mapping.length; occurrence++) {
                int boundary = parsed.mapping[occurrence];
                code.iload(geometry[inputBases + boundary]).i2l()
                        .iload(geometry[variant + 2 + occurrence]).i2l()
                        .iload(geometry[variant + 1 + occurrence]).i2l().lsub()
                        .iload(geometry[1]).i2l().lmul().ladd()
                        .loadConstant(1L << 31).lcmp().branch(Opcode.IFGT, ranged);
            }
            for (int occurrence = 1; occurrence < parsed.mapping.length; occurrence++) {
                code.iload(geometry[variant + 2 + occurrence])
                        .iload(geometry[variant + 1 + occurrence]).isub()
                        .iload(geometry[variant + 2]).iload(geometry[variant + 1]).isub()
                        .branch(Opcode.IF_ICMPNE, ranged);
            }
            if (type == DataType.INT32
                    && specialization.carrierPattern().get(outputSlot) == CarrierAccess.INT_ARRAY) {
                code.iload(geometry[4]).branch(Opcode.IFNE, ranged);
                code.iload(geometry[variant + 1]).branch(Opcode.IFNE, ranged);
                for (int occurrence = 0; occurrence < parsed.mapping.length; occurrence++) {
                    int boundary = parsed.mapping[occurrence];
                    code.iload(geometry[inputBases + boundary]).branch(Opcode.IFNE, ranged);
                    code.iload(geometry[variant + 1 + occurrence])
                            .iload(geometry[variant + 2]).iload(geometry[variant + 1]).isub()
                            .loadConstant(occurrence).imul()
                            .branch(Opcode.IF_ICMPNE, ranged);
                }
            }
            emitFullDenseRankTwoConcat(code, specialization, type, parsed, geometry,
                    outputAddress, sourceAddress, value, inputBases, variant, outputSlot);
            code.branch(Opcode.GOTO, completed);
            code.labelBinding(ranged);
        }
        int occurrenceLocal = code.allocateLocal(TypeKind.INT);
        int sourceValid = code.allocateLocal(TypeKind.INT);
        code.loadConstant(0).istore(occurrenceLocal).loadConstant(0).istore(sourceValid);
        var done = code.newLabel();
        var loop = code.newLabel();
        code.labelBinding(loop);
        code.iload(logical).iload(end).branch(Opcode.IF_ICMPGE, done);
        var sourceReady = code.newLabel();
        code.iload(sourceValid).branch(Opcode.IFNE, sourceReady);
        emitBoundedConcatAddress(code, parsed, geometry, coordinates, sourceAddress,
                occurrenceLocal, inputBases, strideOffsets, variant, inputRank);
        code.loadConstant(1).istore(sourceValid);
        code.labelBinding(sourceReady);
        var loaded = code.newLabel();
        for (int occurrence = 0; occurrence < parsed.mapping.length; occurrence++) {
            var next = code.newLabel();
            code.iload(occurrenceLocal).loadConstant(occurrence)
                    .branch(Opcode.IF_ICMPNE, next);
            int boundary = parsed.mapping[occurrence];
            emitBoundedLoad(code, type, specialization.carrierPattern().get(boundary), boundary,
                    sourceAddress);
            storeValue(code, type, value);
            if (inputRank > 0) code.lload(sourceAddress)
                    .iload(geometry[strideOffsets[boundary] + inputRank - 1]).i2l().ladd()
                    .lstore(sourceAddress);
            code.branch(Opcode.GOTO, loaded);
            code.labelBinding(next);
        }
        emitMovementFailure(code, "composition coordinate has no segment");
        code.labelBinding(loaded);
        emitBoundedStore(code, type, specialization.carrierPattern().get(outputSlot), outputSlot,
                outputAddress, value);
        if (coordinates.length > 0) {
            int inner = coordinates.length - 1;
            var keep = code.newLabel();
            code.iload(coordinates[inner]).loadConstant(1).iadd()
                    .iload(geometry[inner]).branch(Opcode.IF_ICMPLT, keep);
            code.loadConstant(0).istore(sourceValid);
            code.labelBinding(keep);
            int concatAxis = geometry[variant];
            for (int axis = 0; axis < coordinates.length; axis++) {
                var notAxis = code.newLabel();
                code.iload(concatAxis).loadConstant(axis).branch(Opcode.IF_ICMPNE, notAxis);
                for (int occurrence = 0; occurrence < parsed.mapping.length - 1; occurrence++) {
                    var notOccurrence = code.newLabel();
                    code.iload(occurrenceLocal).loadConstant(occurrence)
                            .branch(Opcode.IF_ICMPNE, notOccurrence);
                    code.iload(coordinates[axis]).loadConstant(1).iadd()
                            .iload(geometry[variant + 2 + occurrence])
                            .branch(Opcode.IF_ICMPNE, notOccurrence);
                    code.loadConstant(0).istore(sourceValid);
                    code.labelBinding(notOccurrence);
                }
                code.labelBinding(notAxis);
            }
        }
        emitBoundedOutputAdvance(code, geometry, coordinates, outputAddress);
        code.iinc(logical, 1).branch(Opcode.GOTO, loop);
        code.labelBinding(done);
        code.labelBinding(completed);
    }

    private static void emitFullDenseRankTwoConcat(CodeBuilder code,
            CpuKernelSpecialization specialization, DataType type, Parsed parsed, int[] geometry,
            int outputAddress, int sourceAddress, int value, int inputBases, int variant,
            int outputSlot) {
        if (type == DataType.INT32
                && specialization.carrierPattern().get(outputSlot) == CarrierAccess.INT_ARRAY) {
            emitFullDenseIntConcat(code, specialization, parsed, geometry, inputBases, variant,
                    outputSlot);
            return;
        }
        int ordinal = code.allocateLocal(TypeKind.INT);
        int count = code.allocateLocal(TypeKind.INT);
        int address = code.allocateLocal(TypeKind.INT);
        int[] boundedSourceBases = new int[parsed.mapping.length];
        int[] boundedOutputBases = new int[parsed.mapping.length];
        for (int occurrence = 0; occurrence < parsed.mapping.length; occurrence++) {
            int boundary = parsed.mapping[occurrence];
            boundedSourceBases[occurrence] = code.allocateLocal(TypeKind.INT);
            boundedOutputBases[occurrence] = code.allocateLocal(TypeKind.INT);
            code.iload(geometry[inputBases + boundary]).istore(boundedSourceBases[occurrence]);
            code.iload(geometry[4]).iload(geometry[variant + 1 + occurrence])
                    .iload(geometry[1]).imul().iadd().istore(boundedOutputBases[occurrence]);
        }
        code.iload(geometry[variant + 2]).iload(geometry[variant + 1]).isub()
                .iload(geometry[1]).imul().istore(count);
        code.loadConstant(0).istore(ordinal);
        var loop = code.newLabel();
        var done = code.newLabel();
        code.labelBinding(loop);
        code.iload(ordinal).iload(count).branch(Opcode.IF_ICMPGE, done);
        for (int occurrence = 0; occurrence < parsed.mapping.length; occurrence++) {
            int boundary = parsed.mapping[occurrence];
            code.iload(boundedSourceBases[occurrence]).iload(ordinal).iadd().istore(address);
            emitBoundedIntLoad(code, type, specialization.carrierPattern().get(boundary), boundary,
                    address);
            storeValue(code, type, value);
            code.iload(boundedOutputBases[occurrence]).iload(ordinal).iadd().istore(address);
            emitBoundedIntStore(code, type, specialization.carrierPattern().get(outputSlot),
                    outputSlot, address, value);
        }
        code.iinc(ordinal, 1).branch(Opcode.GOTO, loop);
        code.labelBinding(done);
    }

    private static void emitFullDenseIntConcat(CodeBuilder code,
            CpuKernelSpecialization specialization, Parsed parsed, int[] geometry,
            int inputBases, int variant, int outputSlot) {
        int ordinal = code.allocateLocal(TypeKind.INT);
        int count = code.allocateLocal(TypeKind.INT);
        code.iload(geometry[variant + 2]).iload(geometry[variant + 1]).isub()
                .iload(geometry[1]).imul().istore(count);
        int[] outputOffsets = new int[parsed.mapping.length];
        for (int occurrence = 0; occurrence < parsed.mapping.length; occurrence++) {
            outputOffsets[occurrence] = code.allocateLocal(TypeKind.INT);
            code.iload(count).loadConstant(occurrence).imul()
                    .istore(outputOffsets[occurrence]);
        }
        code.loadConstant(0).istore(ordinal);
        var loop = code.newLabel();
        var done = code.newLabel();
        code.labelBinding(loop);
        code.iload(ordinal).iload(count).branch(Opcode.IF_ICMPGE, done);
        emitFullDenseIntConcatElement(code, specialization, parsed, outputSlot, ordinal,
                outputOffsets);
        code.iinc(ordinal, 1).branch(Opcode.GOTO, loop);
        code.labelBinding(done);
    }

    private static void emitFullDenseIntConcatElement(CodeBuilder code,
            CpuKernelSpecialization specialization, Parsed parsed, int outputSlot,
            int ordinal, int[] outputOffsets) {
        for (int occurrence = 0; occurrence < parsed.mapping.length; occurrence++) {
            int boundary = parsed.mapping[occurrence];
            code.aload(outputSlot);
            code.iload(outputOffsets[occurrence]).iload(ordinal).iadd();
            code.aload(boundary);
            if (specialization.carrierPattern().get(boundary) == CarrierAccess.MEMORY_SEGMENT) {
                ClassDesc layout = movementLayout(DataType.INT32);
                code.getstatic(ClassDesc.of("java.lang.foreign.ValueLayout"),
                        movementLayoutField(DataType.INT32), layout);
                code.iload(ordinal).i2l()
                        .loadConstant((long) DataType.INT32.byteWidth()).lmul();
                code.invokeinterface(ClassDesc.of("java.lang.foreign.MemorySegment"), "get",
                        MethodTypeDesc.of(TypeKind.INT.upperBound(), layout,
                                TypeKind.LONG.upperBound()));
            } else {
                code.iload(ordinal).iaload();
            }
            code.iastore();
        }
    }

    private static void emitBoundedConcatAddress(CodeBuilder code, Parsed parsed, int[] geometry,
            int[] coordinates, int sourceAddress, int occurrenceLocal, int inputBases,
            int[] strideOffsets, int variant, int inputRank) {
        int selected = code.allocateLocal(TypeKind.INT);
        code.loadConstant(0).istore(selected);
        for (int axis = 0; axis < coordinates.length; axis++) {
            var next = code.newLabel();
            code.iload(geometry[variant]).loadConstant(axis).branch(Opcode.IF_ICMPNE, next);
            code.iload(coordinates[axis]).istore(selected);
            code.labelBinding(next);
        }
        var ready = code.newLabel();
        for (int occurrence = 0; occurrence < parsed.mapping.length; occurrence++) {
            var next = code.newLabel();
            code.iload(selected).iload(geometry[variant + 2 + occurrence])
                    .branch(Opcode.IF_ICMPGE, next);
            int boundary = parsed.mapping[occurrence];
            code.loadConstant(occurrence).istore(occurrenceLocal);
            code.iload(geometry[inputBases + boundary]).i2l().lstore(sourceAddress);
            for (int axis = 0; axis < inputRank; axis++) {
                code.lload(sourceAddress);
                var ordinary = code.newLabel();
                var coordinateReady = code.newLabel();
                code.iload(geometry[variant]).loadConstant(axis)
                        .branch(Opcode.IF_ICMPNE, ordinary);
                code.iload(selected).iload(geometry[variant + 1 + occurrence]).isub()
                        .branch(Opcode.GOTO, coordinateReady);
                code.labelBinding(ordinary);
                code.iload(coordinates[axis]);
                code.labelBinding(coordinateReady);
                code.i2l().iload(geometry[strideOffsets[boundary] + axis]).i2l().lmul()
                        .ladd().lstore(sourceAddress);
            }
            code.branch(Opcode.GOTO, ready);
            code.labelBinding(next);
        }
        emitMovementFailure(code, "composition coordinate has no segment");
        code.labelBinding(ready);
    }

    private static void emitBoundedUnfoldAxis(CodeBuilder code,
            CpuKernelSpecialization specialization, DataType type, int[] geometry,
            int[] coordinates, int logical, int end, int outputAddress, int sourceAddress,
            int value, int inputBases, int inputStrides, int variant, int inputRank,
            int outputSlot) {
        int[] increments = new int[coordinates.length];
        for (int axis = 0; axis < coordinates.length; axis++) {
            increments[axis] = code.allocateLocal(TypeKind.LONG);
            if (axis == coordinates.length - 1) {
                code.loadConstant(0L).lstore(increments[axis]);
                for (int inputAxis = 0; inputAxis < inputRank; inputAxis++) {
                    var next = code.newLabel();
                    code.iload(geometry[variant]).loadConstant(inputAxis)
                            .branch(Opcode.IF_ICMPNE, next);
                    code.iload(geometry[inputStrides + inputAxis]).i2l()
                            .lstore(increments[axis]);
                    code.labelBinding(next);
                }
            } else {
                var ordinary = code.newLabel();
                var ready = code.newLabel();
                code.iload(geometry[variant]).loadConstant(axis)
                        .branch(Opcode.IF_ICMPNE, ordinary);
                code.iload(geometry[variant + 2]).i2l()
                        .iload(geometry[inputStrides + axis]).i2l().lmul()
                        .lstore(increments[axis]).branch(Opcode.GOTO, ready);
                code.labelBinding(ordinary);
                code.iload(geometry[inputStrides + axis]).i2l().lstore(increments[axis]);
                code.labelBinding(ready);
            }
        }
        emitBoundedUnfoldAxisAddress(code, geometry, coordinates, sourceAddress,
                inputBases, inputStrides, variant, inputRank);
        var done = code.newLabel();
        var loop = code.newLabel();
        code.labelBinding(loop);
        code.iload(logical).iload(end).branch(Opcode.IF_ICMPGE, done);
        emitBoundedLoad(code, type, specialization.carrierPattern().getFirst(), 0, sourceAddress);
        storeValue(code, type, value);
        emitBoundedStore(code, type, specialization.carrierPattern().get(outputSlot), outputSlot,
                outputAddress, value);
        emitBoundedDualAdvance(code, geometry, coordinates, outputAddress, sourceAddress,
                increments);
        code.iinc(logical, 1).branch(Opcode.GOTO, loop);
        code.labelBinding(done);
    }

    private static void emitBoundedUnfoldAxisAddress(CodeBuilder code, int[] geometry,
            int[] coordinates, int sourceAddress, int inputBases, int inputStrides,
            int variant, int inputRank) {
        code.iload(geometry[inputBases]).i2l().lstore(sourceAddress);
        for (int axis = 0; axis < inputRank; axis++) {
            code.lload(sourceAddress);
            var ordinary = code.newLabel();
            var ready = code.newLabel();
            code.iload(geometry[variant]).loadConstant(axis).branch(Opcode.IF_ICMPNE, ordinary);
            code.iload(coordinates[axis]).iload(geometry[variant + 2]).imul()
                    .iload(coordinates[coordinates.length - 1]).iadd()
                    .branch(Opcode.GOTO, ready);
            code.labelBinding(ordinary);
            code.iload(coordinates[axis]);
            code.labelBinding(ready);
            code.i2l().iload(geometry[inputStrides + axis]).i2l().lmul().ladd()
                    .lstore(sourceAddress);
        }
    }

    private static void emitBoundedUnfold2d(CodeBuilder code,
            CpuKernelSpecialization specialization, DataType type, long bits, int[] geometry,
            int[] coordinates, int logical, int end, int outputAddress, int sourceAddress,
            int value, int inputBases, int inputStrides, int variant, int outputSlot,
            boolean denseOutput) {
        var ranged = code.newLabel();
        var completed = code.newLabel();
        if (denseOutput) {
            code.iload(logical).branch(Opcode.IFNE, ranged);
            code.iload(geometry[0]).i2l().iload(geometry[1]).i2l().lmul()
                    .iload(geometry[2]).i2l().lmul().iload(end).i2l().lcmp()
                    .branch(Opcode.IFNE, ranged);
            for (int index = 0; index <= 12; index++) {
                code.iload(geometry[variant + index]).loadConstant(32767)
                        .branch(Opcode.IF_ICMPGT, ranged);
            }
            code.iload(geometry[inputStrides + 3]).loadConstant(1)
                    .branch(Opcode.IF_ICMPNE, ranged);
            code.iload(geometry[inputStrides + 2]).iload(geometry[variant + 2])
                    .branch(Opcode.IF_ICMPNE, ranged);
            code.iload(geometry[inputStrides + 1]).i2l()
                    .iload(geometry[variant + 1]).i2l()
                    .iload(geometry[variant + 2]).i2l().lmul().lcmp()
                    .branch(Opcode.IFNE, ranged);
            code.iload(geometry[inputStrides]).i2l()
                    .iload(geometry[variant]).i2l()
                    .iload(geometry[variant + 1]).i2l().lmul()
                    .iload(geometry[variant + 2]).i2l().lmul().lcmp()
                    .branch(Opcode.IFNE, ranged);
            emitFullDenseUnfold2d(code, specialization, type, bits, geometry, outputAddress,
                    sourceAddress, value, inputBases, inputStrides, variant, outputSlot);
            code.branch(Opcode.GOTO, completed);
            code.labelBinding(ranged);
        }
        int channel = code.allocateLocal(TypeKind.INT);
        int kh = code.allocateLocal(TypeKind.INT);
        int kw = code.allocateLocal(TypeKind.INT);
        int oh = code.allocateLocal(TypeKind.INT);
        int ow = code.allocateLocal(TypeKind.INT);
        code.iload(geometry[variant + 13]).istore(channel);
        code.iload(geometry[variant + 14]).istore(kh);
        code.iload(geometry[variant + 15]).istore(kw);
        code.iload(geometry[variant + 16]).istore(oh);
        code.iload(geometry[variant + 17]).istore(ow);
        int ih = code.allocateLocal(TypeKind.INT);
        int iw = code.allocateLocal(TypeKind.INT);
        int sourceValid = code.allocateLocal(TypeKind.INT);
        code.loadConstant(0).istore(sourceValid);
        emitBoundedUnfold2dSpatial(code, geometry, kh, kw, oh, ow, ih, iw, variant);
        var done = code.newLabel();
        var loop = code.newLabel();
        code.labelBinding(loop);
        code.iload(logical).iload(end).branch(Opcode.IF_ICMPGE, done);
        var padding = code.newLabel();
        var loaded = code.newLabel();
        code.iload(ih).branch(Opcode.IFLT, padding);
        code.iload(ih).iload(geometry[variant + 1]).branch(Opcode.IF_ICMPGE, padding);
        code.iload(iw).branch(Opcode.IFLT, padding);
        code.iload(iw).iload(geometry[variant + 2]).branch(Opcode.IF_ICMPGE, padding);
        var ready = code.newLabel();
        code.iload(sourceValid).branch(Opcode.IFNE, ready);
        emitBoundedUnfold2dAddress(code, geometry, coordinates, sourceAddress, inputBases,
                inputStrides, channel, ih, iw);
        code.loadConstant(1).istore(sourceValid);
        code.labelBinding(ready);
        emitBoundedLoad(code, type, specialization.carrierPattern().getFirst(), 0, sourceAddress);
        storeValue(code, type, value);
        code.lload(sourceAddress).iload(geometry[inputStrides + 3]).i2l()
                .iload(geometry[variant + 6]).i2l().lmul().ladd().lstore(sourceAddress);
        code.branch(Opcode.GOTO, loaded);
        code.labelBinding(padding);
        code.loadConstant(0).istore(sourceValid);
        loadImmediate(code, type, bits);
        storeValue(code, type, value);
        code.labelBinding(loaded);
        emitBoundedStore(code, type, specialization.carrierPattern().get(outputSlot), outputSlot,
                outputAddress, value);
        emitBoundedUnfold2dAdvance(code, geometry, coordinates, outputAddress, channel, kh, kw,
                oh, ow, ih, iw, sourceValid, variant, denseOutput);
        code.iinc(logical, 1).branch(Opcode.GOTO, loop);
        code.labelBinding(done);
        code.labelBinding(completed);
    }

    private static void emitFullDenseUnfold2d(CodeBuilder code,
            CpuKernelSpecialization specialization, DataType type, long bits, int[] geometry,
            int outputAddress, int sourceAddress, int value, int inputBases, int inputStrides,
            int variant, int outputSlot) {
        int batch = code.allocateLocal(TypeKind.INT);
        int channel = code.allocateLocal(TypeKind.INT);
        int kh = code.allocateLocal(TypeKind.INT);
        int kw = code.allocateLocal(TypeKind.INT);
        int oh = code.allocateLocal(TypeKind.INT);
        int ow = code.allocateLocal(TypeKind.INT);
        int ih = code.allocateLocal(TypeKind.INT);
        int iw = code.allocateLocal(TypeKind.INT);
        code.loadConstant(0).istore(batch).loadConstant(0).istore(channel)
                .loadConstant(0).istore(kh).loadConstant(0).istore(kw)
                .loadConstant(0).istore(oh).loadConstant(0).istore(ow);
        var loop = code.newLabel();
        var done = code.newLabel();
        code.labelBinding(loop);
        code.iload(batch).iload(geometry[0]).branch(Opcode.IF_ICMPGE, done);
        code.iload(oh).iload(geometry[variant + 5]).imul()
                .iload(geometry[variant + 7]).isub().iload(kh)
                .iload(geometry[variant + 9]).imul().iadd().istore(ih);
        code.iload(ow).iload(geometry[variant + 6]).imul()
                .iload(geometry[variant + 8]).isub().iload(kw)
                .iload(geometry[variant + 10]).imul().iadd().istore(iw);
        var padding = code.newLabel();
        var loaded = code.newLabel();
        code.iload(ih).branch(Opcode.IFLT, padding);
        code.iload(ih).iload(geometry[variant + 1]).branch(Opcode.IF_ICMPGE, padding);
        code.iload(iw).branch(Opcode.IFLT, padding);
        code.iload(iw).iload(geometry[variant + 2]).branch(Opcode.IF_ICMPGE, padding);
        code.iload(geometry[inputBases]).i2l()
                .iload(batch).i2l().iload(geometry[inputStrides]).i2l().lmul().ladd()
                .iload(channel).i2l().iload(geometry[inputStrides + 1]).i2l().lmul().ladd()
                .iload(ih).i2l().iload(geometry[inputStrides + 2]).i2l().lmul().ladd()
                .iload(iw).i2l().ladd().lstore(sourceAddress);
        emitBoundedLoad(code, type, specialization.carrierPattern().getFirst(), 0, sourceAddress);
        storeValue(code, type, value);
        code.branch(Opcode.GOTO, loaded);
        code.labelBinding(padding);
        loadImmediate(code, type, bits);
        storeValue(code, type, value);
        code.labelBinding(loaded);
        emitBoundedStore(code, type, specialization.carrierPattern().get(outputSlot), outputSlot,
                outputAddress, value);
        code.lload(outputAddress).loadConstant(1L).ladd().lstore(outputAddress);
        code.iinc(ow, 1).iload(ow).iload(geometry[variant + 12])
                .branch(Opcode.IF_ICMPLT, loop);
        code.loadConstant(0).istore(ow).iinc(oh, 1).iload(oh)
                .iload(geometry[variant + 11]).branch(Opcode.IF_ICMPLT, loop);
        code.loadConstant(0).istore(oh).iinc(kw, 1).iload(kw)
                .iload(geometry[variant + 4]).branch(Opcode.IF_ICMPLT, loop);
        code.loadConstant(0).istore(kw).iinc(kh, 1).iload(kh)
                .iload(geometry[variant + 3]).branch(Opcode.IF_ICMPLT, loop);
        code.loadConstant(0).istore(kh).iinc(channel, 1).iload(channel)
                .iload(geometry[variant]).branch(Opcode.IF_ICMPLT, loop);
        code.loadConstant(0).istore(channel).iinc(batch, 1).branch(Opcode.GOTO, loop);
        code.labelBinding(done);
    }

    private static void emitBoundedUnfold2dSpatial(CodeBuilder code, int[] geometry,
            int kh, int kw, int oh, int ow, int ih, int iw, int variant) {
        code.iload(oh).iload(geometry[variant + 5]).imul()
                .iload(geometry[variant + 7]).isub().iload(kh)
                .iload(geometry[variant + 9]).imul().iadd().istore(ih);
        code.iload(ow).iload(geometry[variant + 6]).imul()
                .iload(geometry[variant + 8]).isub().iload(kw)
                .iload(geometry[variant + 10]).imul().iadd().istore(iw);
    }

    private static void emitBoundedUnfold2dAddress(CodeBuilder code, int[] geometry,
            int[] coordinates, int sourceAddress, int inputBases, int inputStrides,
            int channel, int ih, int iw) {
        code.iload(geometry[inputBases]).i2l().lstore(sourceAddress);
        int[] terms = {coordinates[0], channel, ih, iw};
        for (int axis = 0; axis < terms.length; axis++) {
            code.lload(sourceAddress).iload(terms[axis]).i2l()
                    .iload(geometry[inputStrides + axis]).i2l().lmul().ladd()
                    .lstore(sourceAddress);
        }
    }

    private static void emitBoundedUnfold2dAdvance(CodeBuilder code, int[] geometry,
            int[] coordinates, int outputAddress, int channel, int kh, int kw, int oh, int ow,
            int ih, int iw, int sourceValid, int variant, boolean denseOutput) {
        if (denseOutput) {
            code.lload(outputAddress).loadConstant(1L).ladd().lstore(outputAddress);
        } else {
            emitBoundedOutputAdvance(code, geometry, coordinates, outputAddress);
        }
        code.iinc(ow, 1).iload(iw).iload(geometry[variant + 6]).iadd().istore(iw);
        var done = code.newLabel();
        code.iload(ow).iload(geometry[variant + 12]).branch(Opcode.IF_ICMPLT, done);
        code.loadConstant(0).istore(ow).iinc(oh, 1).loadConstant(0).istore(sourceValid);
        var ohReady = code.newLabel();
        code.iload(oh).iload(geometry[variant + 11]).branch(Opcode.IF_ICMPLT, ohReady);
        code.loadConstant(0).istore(oh).iinc(kw, 1);
        var kwReady = code.newLabel();
        code.iload(kw).iload(geometry[variant + 4]).branch(Opcode.IF_ICMPLT, kwReady);
        code.loadConstant(0).istore(kw).iinc(kh, 1);
        var khReady = code.newLabel();
        code.iload(kh).iload(geometry[variant + 3]).branch(Opcode.IF_ICMPLT, khReady);
        code.loadConstant(0).istore(kh).iinc(channel, 1);
        var channelReady = code.newLabel();
        code.iload(channel).iload(geometry[variant]).branch(Opcode.IF_ICMPLT, channelReady);
        code.loadConstant(0).istore(channel);
        if (denseOutput) code.iinc(coordinates[0], 1);
        code.labelBinding(channelReady);
        code.labelBinding(khReady);
        code.labelBinding(kwReady);
        code.labelBinding(ohReady);
        emitBoundedUnfold2dSpatial(code, geometry, kh, kw, oh, ow, ih, iw, variant);
        code.labelBinding(done);
    }

    private static void emitBoundedLoad(CodeBuilder code, DataType type, CarrierAccess access,
            int parameterSlot, int addressLocal) {
        code.aload(parameterSlot);
        if (access != CarrierAccess.MEMORY_SEGMENT) {
            code.lload(addressLocal).l2i();
            switch (type) {
                case FLOAT64 -> code.daload();
                case FLOAT32 -> code.faload();
                case BFLOAT16 -> code.saload();
                case INT32 -> code.iaload();
                case INT64 -> code.laload();
                case BOOL -> code.baload();
            }
            return;
        }
        ClassDesc layout = movementLayout(type);
        code.getstatic(ClassDesc.of("java.lang.foreign.ValueLayout"), movementLayoutField(type),
                layout);
        code.lload(addressLocal).loadConstant((long) type.byteWidth()).lmul();
        code.invokeinterface(ClassDesc.of("java.lang.foreign.MemorySegment"), "get",
                MethodTypeDesc.of(movementPrimitive(type), layout,
                        TypeKind.LONG.upperBound()));
    }

    private static void emitBoundedIntLoad(CodeBuilder code, DataType type, CarrierAccess access,
            int parameterSlot, int addressLocal) {
        code.aload(parameterSlot);
        if (access != CarrierAccess.MEMORY_SEGMENT) {
            code.iload(addressLocal);
            switch (type) {
                case FLOAT64 -> code.daload();
                case FLOAT32 -> code.faload();
                case BFLOAT16 -> code.saload();
                case INT32 -> code.iaload();
                case INT64 -> code.laload();
                case BOOL -> code.baload();
            }
            return;
        }
        ClassDesc layout = movementLayout(type);
        code.getstatic(ClassDesc.of("java.lang.foreign.ValueLayout"), movementLayoutField(type),
                layout);
        code.iload(addressLocal).i2l().loadConstant((long) type.byteWidth()).lmul();
        code.invokeinterface(ClassDesc.of("java.lang.foreign.MemorySegment"), "get",
                MethodTypeDesc.of(movementPrimitive(type), layout,
                        TypeKind.LONG.upperBound()));
    }

    private static void emitBoundedStore(CodeBuilder code, DataType type, CarrierAccess access,
            int parameterSlot, int addressLocal, int valueLocal) {
        code.aload(parameterSlot);
        if (access != CarrierAccess.MEMORY_SEGMENT) {
            code.lload(addressLocal).l2i();
            loadMovementLocal(code, type, valueLocal);
            switch (type) {
                case FLOAT64 -> code.dastore();
                case FLOAT32 -> code.fastore();
                case BFLOAT16 -> code.sastore();
                case INT32 -> code.iastore();
                case INT64 -> code.lastore();
                case BOOL -> code.bastore();
            }
            return;
        }
        ClassDesc layout = movementLayout(type);
        code.getstatic(ClassDesc.of("java.lang.foreign.ValueLayout"), movementLayoutField(type),
                layout);
        code.lload(addressLocal).loadConstant((long) type.byteWidth()).lmul();
        loadMovementLocal(code, type, valueLocal);
        code.invokeinterface(ClassDesc.of("java.lang.foreign.MemorySegment"), "set",
                MethodTypeDesc.of(TypeKind.VOID.upperBound(), layout,
                        TypeKind.LONG.upperBound(), movementPrimitive(type)));
    }

    private static void emitBoundedIntStore(CodeBuilder code, DataType type, CarrierAccess access,
            int parameterSlot, int addressLocal, int valueLocal) {
        code.aload(parameterSlot);
        if (access != CarrierAccess.MEMORY_SEGMENT) {
            code.iload(addressLocal);
            loadMovementLocal(code, type, valueLocal);
            switch (type) {
                case FLOAT64 -> code.dastore();
                case FLOAT32 -> code.fastore();
                case BFLOAT16 -> code.sastore();
                case INT32 -> code.iastore();
                case INT64 -> code.lastore();
                case BOOL -> code.bastore();
            }
            return;
        }
        ClassDesc layout = movementLayout(type);
        code.getstatic(ClassDesc.of("java.lang.foreign.ValueLayout"), movementLayoutField(type),
                layout);
        code.iload(addressLocal).i2l().loadConstant((long) type.byteWidth()).lmul();
        loadMovementLocal(code, type, valueLocal);
        code.invokeinterface(ClassDesc.of("java.lang.foreign.MemorySegment"), "set",
                MethodTypeDesc.of(TypeKind.VOID.upperBound(), layout,
                        TypeKind.LONG.upperBound(), movementPrimitive(type)));
    }

    private static String movementLayoutField(DataType type) {
        return switch (type) {
            case FLOAT64 -> "JAVA_DOUBLE_UNALIGNED";
            case FLOAT32 -> "JAVA_FLOAT_UNALIGNED";
            case BFLOAT16 -> "JAVA_SHORT_UNALIGNED";
            case INT32 -> "JAVA_INT_UNALIGNED";
            case INT64 -> "JAVA_LONG_UNALIGNED";
            case BOOL -> "JAVA_BYTE";
        };
    }

    private static ClassDesc movementLayout(DataType type) {
        return ClassDesc.of("java.lang.foreign.ValueLayout$Of" + switch (type) {
            case FLOAT64 -> "Double";
            case FLOAT32 -> "Float";
            case BFLOAT16 -> "Short";
            case INT32 -> "Int";
            case INT64 -> "Long";
            case BOOL -> "Byte";
        });
    }

    private static ClassDesc movementPrimitive(DataType type) {
        return switch (type) {
            case FLOAT64 -> TypeKind.DOUBLE.upperBound();
            case FLOAT32 -> TypeKind.FLOAT.upperBound();
            case BFLOAT16 -> TypeKind.SHORT.upperBound();
            case INT32 -> TypeKind.INT.upperBound();
            case INT64 -> TypeKind.LONG.upperBound();
            case BOOL -> TypeKind.BYTE.upperBound();
        };
    }

    private static void loadMovementLocal(CodeBuilder code, DataType type, int local) {
        switch (type) {
            case FLOAT64 -> code.dload(local);
            case FLOAT32 -> code.fload(local);
            case BFLOAT16 -> code.iload(local).i2s();
            case INT32 -> code.iload(local);
            case INT64 -> code.lload(local);
            case BOOL -> code.iload(local).i2b();
        }
    }

    private static void emitBoundedOutputAdvance(CodeBuilder code, int[] geometry,
            int[] coordinates, int outputAddress) {
        var finished = code.newLabel();
        int strides = 2 * coordinates.length + 1;
        for (int axis = coordinates.length - 1; axis >= 0; axis--) {
            code.iinc(coordinates[axis], 1);
            code.lload(outputAddress).iload(geometry[strides + axis]).i2l().ladd()
                    .lstore(outputAddress);
            code.iload(coordinates[axis]).iload(geometry[axis])
                    .branch(Opcode.IF_ICMPLT, finished);
            code.loadConstant(0).istore(coordinates[axis]);
            code.lload(outputAddress).iload(geometry[axis]).i2l()
                    .iload(geometry[strides + axis]).i2l().lmul().lsub()
                    .lstore(outputAddress);
        }
        code.labelBinding(finished);
    }

    private static void emitBoundedDualAdvance(CodeBuilder code, int[] geometry,
            int[] coordinates, int outputAddress, int sourceAddress, int[] increments) {
        var finished = code.newLabel();
        int strides = 2 * coordinates.length + 1;
        for (int axis = coordinates.length - 1; axis >= 0; axis--) {
            code.iinc(coordinates[axis], 1);
            code.lload(outputAddress).iload(geometry[strides + axis]).i2l().ladd()
                    .lstore(outputAddress);
            code.lload(sourceAddress).lload(increments[axis]).ladd().lstore(sourceAddress);
            code.iload(coordinates[axis]).iload(geometry[axis])
                    .branch(Opcode.IF_ICMPLT, finished);
            code.loadConstant(0).istore(coordinates[axis]);
            code.lload(outputAddress).iload(geometry[axis]).i2l()
                    .iload(geometry[strides + axis]).i2l().lmul().lsub()
                    .lstore(outputAddress);
            code.lload(sourceAddress).iload(geometry[axis]).i2l()
                    .lload(increments[axis]).lmul().lsub().lstore(sourceAddress);
        }
        code.labelBinding(finished);
    }

    private static void emitDenseArrayInt(CodeBuilder code,
            CpuKernelSpecialization specialization, CpuKernelIr ir, Parsed parsed) {
        DataType type = ir.values().getFirst().dataType();
        int rank = parsed.rank;
        int uniqueInputs = ir.values().size() - 1;
        int inputRank = ir.values().getFirst().accessPlan().iterationRank();
        int geometrySlot = uniqueInputs + 1;
        int startSlot = geometrySlot + 1;
        int endSlot = startSlot + 2;
        int inputBases = 3 * rank + 1;
        int inputStrides = inputBases + uniqueInputs;
        int[] strideOffsets = new int[uniqueInputs];
        int variant = inputStrides;
        for (int input = 0; input < uniqueInputs; input++) {
            strideOffsets[input] = variant;
            variant += ir.values().get(input).accessPlan().iterationRank();
        }
        int variantSize = switch (parsed.family) {
            case "PAD", "TILE" -> 2 * rank;
            case "CONCAT" -> 2 + parsed.mapping.length;
            case "STACK" -> 1;
            case "UNFOLD_AXIS" -> 3;
            case "UNFOLD2D" -> 18;
            case "SLICE_UPDATE" -> 6 * rank;
            default -> throw new IllegalArgumentException("unsupported movement family");
        };
        int[] geometry = new int[variant + variantSize];
        for (int index = 0; index < geometry.length; index++) {
            geometry[index] = code.allocateLocal(TypeKind.INT);
            code.aload(geometrySlot).loadConstant(index).laload().l2i().istore(geometry[index]);
        }
        int[] coordinates = new int[rank];
        for (int axis = 0; axis < rank; axis++) {
            coordinates[axis] = code.allocateLocal(TypeKind.INT);
            code.iload(geometry[rank + axis]).istore(coordinates[axis]);
        }
        int outputAddress = code.allocateLocal(TypeKind.INT);
        code.iload(geometry[2 * rank]).istore(outputAddress);
        int logical = code.allocateLocal(TypeKind.INT);
        int end = code.allocateLocal(TypeKind.INT);
        code.lload(startSlot).l2i().istore(logical);
        code.lload(endSlot).l2i().istore(end);
        int sourceAddress = code.allocateLocal(TypeKind.INT);
        int value = code.allocateLocal(switch (type) {
            case FLOAT64 -> TypeKind.DOUBLE;
            case FLOAT32 -> TypeKind.FLOAT;
            case BFLOAT16, INT32, BOOL -> TypeKind.INT;
            case INT64 -> TypeKind.LONG;
        });
        int[] tileCoordinates = null;
        int[] sliceTargets = null;
        int[] sliceOrdinals = null;
        if (parsed.family.equals("TILE")) {
            tileCoordinates = new int[rank];
            for (int axis = 0; axis < rank; axis++) {
                tileCoordinates[axis] = code.allocateLocal(TypeKind.INT);
                code.iload(geometry[variant + rank + axis]).istore(tileCoordinates[axis]);
            }
            code.iload(geometry[inputBases]).istore(sourceAddress);
            for (int axis = 0; axis < rank; axis++) code.iload(sourceAddress)
                    .iload(tileCoordinates[axis]).iload(geometry[inputStrides + axis])
                    .imul().iadd().istore(sourceAddress);
        } else if (parsed.family.equals("SLICE_UPDATE")) {
            sliceTargets = new int[rank];
            sliceOrdinals = new int[rank];
            for (int axis = 0; axis < rank; axis++) {
                sliceTargets[axis] = code.allocateLocal(TypeKind.INT);
                sliceOrdinals[axis] = code.allocateLocal(TypeKind.INT);
                code.iload(geometry[variant + 4 * rank + axis]).istore(sliceTargets[axis]);
                code.iload(geometry[variant + 5 * rank + axis]).istore(sliceOrdinals[axis]);
            }
        }
        var carriers = new CpuCarrierEmitter(code);
        var done = code.newLabel();
        var loop = code.newLabel();
        code.labelBinding(loop);
        code.iload(logical).iload(end).branch(Opcode.IF_ICMPGE, done);
        switch (parsed.family) {
            case "PAD" -> emitDensePad(code, carriers, specialization, type, parsed.bits,
                    geometry, coordinates, sourceAddress, value, inputBases,
                    strideOffsets[0], variant);
            case "TILE" -> emitDenseTile(code, carriers, specialization, type, geometry,
                    tileCoordinates, sourceAddress, value, inputBases, inputStrides);
            case "CONCAT" -> emitDenseConcat(code, carriers, specialization, type, parsed,
                    geometry, coordinates, sourceAddress, value, inputBases,
                    strideOffsets, variant, inputRank);
            case "STACK" -> emitDenseStack(code, carriers, specialization, type, parsed,
                    geometry, coordinates, sourceAddress, value, inputBases,
                    strideOffsets, variant, inputRank);
            case "UNFOLD_AXIS" -> emitDenseUnfoldAxis(code, carriers, specialization, type,
                    geometry, coordinates, sourceAddress, value, inputBases,
                    strideOffsets[0], variant, inputRank);
            case "UNFOLD2D" -> emitDenseUnfold2d(code, carriers, specialization, type, parsed.bits,
                    geometry, coordinates, sourceAddress, value, inputBases,
                    strideOffsets[0], variant);
            case "SLICE_UPDATE" -> {
                if (rank == 1) emitDenseRankOneSliceUpdate(code, carriers, specialization, type,
                        parsed, geometry, logical, sliceTargets[0], sliceOrdinals[0],
                        sourceAddress, value, inputBases, strideOffsets, variant);
                else emitDenseSliceUpdate(code, carriers, specialization, type, parsed, geometry,
                        coordinates, sliceTargets, sliceOrdinals, sourceAddress, value, inputBases,
                        strideOffsets);
            }
            default -> throw new IllegalArgumentException("unsupported movement family");
        }
        carriers.store(type, specialization.carrierPattern().get(uniqueInputs), uniqueInputs,
                outputAddress, value, true);
        if (tileCoordinates != null) {
            emitDenseTileAdvance(code, geometry, coordinates, tileCoordinates, sourceAddress,
                    inputStrides, variant);
        } else if (sliceTargets == null) {
            emitDenseAdvanceCoordinates(code, geometry, coordinates, 0);
        } else if (rank != 1) {
            emitDenseSliceAdvance(code, geometry, coordinates, sliceTargets, sliceOrdinals, variant);
        }
        code.iinc(outputAddress, 1).iinc(logical, 1).branch(Opcode.GOTO, loop);
        code.labelBinding(done);
    }

    private static void emitDensePad(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType type, long bits, int[] geometry,
            int[] coordinates, int sourceAddress, int value, int inputBases,
            int inputStrides, int variant) {
        var padding = code.newLabel();
        var loaded = code.newLabel();
        for (int axis = 0; axis < coordinates.length; axis++) {
            code.iload(coordinates[axis]).iload(geometry[variant + axis])
                    .branch(Opcode.IF_ICMPLT, padding);
            code.iload(coordinates[axis]).iload(geometry[variant + axis]).isub()
                    .iload(geometry[variant + coordinates.length + axis])
                    .branch(Opcode.IF_ICMPGE, padding);
        }
        code.iload(geometry[inputBases]).istore(sourceAddress);
        for (int axis = 0; axis < coordinates.length; axis++) code.iload(sourceAddress)
                .iload(coordinates[axis]).iload(geometry[variant + axis]).isub()
                .iload(geometry[inputStrides + axis]).imul().iadd().istore(sourceAddress);
        carriers.load(type, specialization.carrierPattern().getFirst(), 0, sourceAddress, true);
        storeValue(code, type, value);
        code.branch(Opcode.GOTO, loaded);
        code.labelBinding(padding);
        loadImmediate(code, type, bits);
        storeValue(code, type, value);
        code.labelBinding(loaded);
    }

    private static void emitDenseTile(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType type, int[] geometry,
            int[] coordinates, int sourceAddress, int value, int inputBases, int inputStrides) {
        carriers.load(type, specialization.carrierPattern().getFirst(), 0, sourceAddress, true);
        storeValue(code, type, value);
    }

    private static void emitDenseTileAdvance(CodeBuilder code, int[] geometry,
            int[] coordinates, int[] tileCoordinates, int sourceAddress, int inputStrides,
            int variant) {
        var finished = code.newLabel();
        for (int axis = coordinates.length - 1; axis >= 0; axis--) {
            code.iinc(tileCoordinates[axis], 1).iload(sourceAddress)
                    .iload(geometry[inputStrides + axis]).iadd().istore(sourceAddress);
            var sourceReady = code.newLabel();
            code.iload(tileCoordinates[axis]).iload(geometry[variant + axis])
                    .branch(Opcode.IF_ICMPLT, sourceReady);
            code.loadConstant(0).istore(tileCoordinates[axis]);
            code.iload(sourceAddress).iload(geometry[variant + axis])
                    .iload(geometry[inputStrides + axis]).imul().isub().istore(sourceAddress);
            code.labelBinding(sourceReady);
            code.iinc(coordinates[axis], 1).iload(coordinates[axis]).iload(geometry[axis])
                    .branch(Opcode.IF_ICMPLT, finished);
            code.loadConstant(0).istore(coordinates[axis]);
        }
        code.labelBinding(finished);
    }

    private static void emitDenseConcat(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType type, Parsed parsed, int[] geometry,
            int[] coordinates, int sourceAddress, int value, int inputBases,
            int[] strideOffsets, int variant, int inputRank) {
        int selected = emitDenseSelectedCoordinate(code, geometry[variant], coordinates);
        var loaded = code.newLabel();
        for (int occurrence = 0; occurrence < parsed.mapping.length; occurrence++) {
            var next = code.newLabel();
            code.iload(selected).iload(geometry[variant + 2 + occurrence])
                    .branch(Opcode.IF_ICMPGE, next);
            int boundary = parsed.mapping[occurrence];
            code.iload(geometry[inputBases + boundary]).istore(sourceAddress);
            for (int axis = 0; axis < inputRank; axis++) {
                int coordinate = emitDenseCoordinateExcept(code, geometry[variant], coordinates,
                        axis, selected, geometry[variant + 1 + occurrence]);
                code.iload(sourceAddress).iload(coordinate)
                        .iload(geometry[strideOffsets[boundary] + axis]).imul().iadd()
                        .istore(sourceAddress);
            }
            carriers.load(type, specialization.carrierPattern().get(boundary), boundary,
                    sourceAddress, true);
            storeValue(code, type, value);
            code.branch(Opcode.GOTO, loaded);
            code.labelBinding(next);
        }
        emitMovementFailure(code, "composition coordinate has no segment");
        code.labelBinding(loaded);
    }

    private static void emitDenseStack(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType type, Parsed parsed, int[] geometry,
            int[] coordinates, int sourceAddress, int value, int inputBases,
            int[] strideOffsets, int variant, int inputRank) {
        int selected = emitDenseSelectedCoordinate(code, geometry[variant], coordinates);
        var loaded = code.newLabel();
        for (int occurrence = 0; occurrence < parsed.mapping.length; occurrence++) {
            var next = code.newLabel();
            code.iload(selected).loadConstant(occurrence).branch(Opcode.IF_ICMPNE, next);
            int boundary = parsed.mapping[occurrence];
            code.iload(geometry[inputBases + boundary]).istore(sourceAddress);
            for (int axis = 0; axis < inputRank; axis++) {
                int coordinate = code.allocateLocal(TypeKind.INT);
                var shifted = code.newLabel();
                var coordinateReady = code.newLabel();
                code.iload(geometry[variant]).loadConstant(axis)
                        .branch(Opcode.IF_ICMPLE, shifted);
                code.iload(coordinates[axis]).istore(coordinate)
                        .branch(Opcode.GOTO, coordinateReady);
                code.labelBinding(shifted);
                code.iload(coordinates[axis + 1]).istore(coordinate);
                code.labelBinding(coordinateReady);
                code.iload(sourceAddress).iload(coordinate)
                        .iload(geometry[strideOffsets[boundary] + axis]).imul().iadd()
                        .istore(sourceAddress);
            }
            carriers.load(type, specialization.carrierPattern().get(boundary), boundary,
                    sourceAddress, true);
            storeValue(code, type, value);
            code.branch(Opcode.GOTO, loaded);
            code.labelBinding(next);
        }
        emitMovementFailure(code, "stack coordinate has no input");
        code.labelBinding(loaded);
    }

    private static void emitDenseUnfoldAxis(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType type, int[] geometry,
            int[] coordinates, int sourceAddress, int value, int inputBases,
            int inputStrides, int variant, int inputRank) {
        code.iload(geometry[inputBases]).istore(sourceAddress);
        for (int axis = 0; axis < inputRank; axis++) {
            int sourceCoordinate = code.allocateLocal(TypeKind.INT);
            var selected = code.newLabel();
            var calculated = code.newLabel();
            code.iload(geometry[variant]).loadConstant(axis)
                    .branch(Opcode.IF_ICMPEQ, selected);
            code.iload(coordinates[axis]).istore(sourceCoordinate)
                    .branch(Opcode.GOTO, calculated);
            code.labelBinding(selected);
            code.iload(coordinates[axis]).iload(geometry[variant + 2]).imul()
                    .iload(coordinates[coordinates.length - 1]).iadd().istore(sourceCoordinate);
            code.labelBinding(calculated);
            code.iload(sourceAddress).iload(sourceCoordinate)
                    .iload(geometry[inputStrides + axis]).imul().iadd().istore(sourceAddress);
        }
        carriers.load(type, specialization.carrierPattern().getFirst(), 0, sourceAddress, true);
        storeValue(code, type, value);
    }

    private static void emitDenseUnfold2d(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType type, long bits, int[] geometry,
            int[] coordinates, int sourceAddress, int value, int inputBases,
            int inputStrides, int variant) {
        int channel = variant + 13, kh = variant + 14, kw = variant + 15;
        int oh = variant + 16, ow = variant + 17;
        int ih = code.allocateLocal(TypeKind.INT);
        int iw = code.allocateLocal(TypeKind.INT);
        code.iload(geometry[oh]).iload(geometry[variant + 5]).imul()
                .iload(geometry[variant + 7]).isub().iload(geometry[kh])
                .iload(geometry[variant + 9]).imul().iadd().istore(ih);
        code.iload(geometry[ow]).iload(geometry[variant + 6]).imul()
                .iload(geometry[variant + 8]).isub().iload(geometry[kw])
                .iload(geometry[variant + 10]).imul().iadd().istore(iw);
        var padding = code.newLabel();
        var loaded = code.newLabel();
        code.iload(ih).branch(Opcode.IFLT, padding);
        code.iload(ih).iload(geometry[variant + 1]).branch(Opcode.IF_ICMPGE, padding);
        code.iload(iw).branch(Opcode.IFLT, padding);
        code.iload(iw).iload(geometry[variant + 2]).branch(Opcode.IF_ICMPGE, padding);
        code.iload(geometry[inputBases]).istore(sourceAddress);
        emitDenseAddressTerm(code, geometry, sourceAddress, coordinates[0], inputStrides);
        code.iload(sourceAddress).iload(geometry[channel]).iload(geometry[inputStrides + 1])
                .imul().iadd().istore(sourceAddress);
        emitDenseAddressTerm(code, geometry, sourceAddress, ih, inputStrides + 2);
        emitDenseAddressTerm(code, geometry, sourceAddress, iw, inputStrides + 3);
        carriers.load(type, specialization.carrierPattern().getFirst(), 0, sourceAddress, true);
        storeValue(code, type, value);
        code.branch(Opcode.GOTO, loaded);
        code.labelBinding(padding);
        loadImmediate(code, type, bits);
        storeValue(code, type, value);
        code.labelBinding(loaded);
        emitDenseUnfold2dAdvance(code, geometry, ow, variant + 12, oh, variant + 11,
                kw, variant + 4, kh, variant + 3, channel, variant);
    }

    private static void emitDenseSliceUpdate(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType type, Parsed parsed, int[] geometry,
            int[] coordinates, int[] targets, int[] ordinals, int sourceAddress, int value,
            int inputBases, int[] strideOffsets) {
        var base = code.newLabel();
        var loaded = code.newLabel();
        for (int axis = 0; axis < coordinates.length; axis++) code.iload(coordinates[axis])
                .iload(targets[axis]).branch(Opcode.IF_ICMPNE, base);
        int updateBoundary = parsed.mapping[1];
        code.iload(geometry[inputBases + updateBoundary]).istore(sourceAddress);
        for (int axis = 0; axis < coordinates.length; axis++) code.iload(sourceAddress)
                .iload(ordinals[axis]).iload(geometry[strideOffsets[updateBoundary] + axis])
                .imul().iadd().istore(sourceAddress);
        carriers.load(type, specialization.carrierPattern().get(updateBoundary), updateBoundary,
                sourceAddress, true);
        storeValue(code, type, value);
        code.branch(Opcode.GOTO, loaded);
        code.labelBinding(base);
        int baseBoundary = parsed.mapping[0];
        code.iload(geometry[inputBases + baseBoundary]).istore(sourceAddress);
        for (int axis = 0; axis < coordinates.length; axis++) code.iload(sourceAddress)
                .iload(coordinates[axis]).iload(geometry[strideOffsets[baseBoundary] + axis])
                .imul().iadd().istore(sourceAddress);
        carriers.load(type, specialization.carrierPattern().get(baseBoundary), baseBoundary,
                sourceAddress, true);
        storeValue(code, type, value);
        code.labelBinding(loaded);
    }

    private static void emitDenseRankOneSliceUpdate(CodeBuilder code,
            CpuCarrierEmitter carriers, CpuKernelSpecialization specialization, DataType type,
            Parsed parsed, int[] geometry, int coordinate, int target, int ordinal,
            int sourceAddress, int value, int inputBases, int[] strideOffsets, int variant) {
        var base = code.newLabel();
        var loaded = code.newLabel();
        code.iload(coordinate).iload(target).branch(Opcode.IF_ICMPNE, base);
        int updateBoundary = parsed.mapping[1];
        code.iload(geometry[inputBases + updateBoundary]).iload(ordinal).iadd()
                .istore(sourceAddress);
        carriers.load(type, specialization.carrierPattern().get(updateBoundary), updateBoundary,
                sourceAddress, true);
        storeValue(code, type, value);
        var negative = code.newLabel();
        var exhausted = code.newLabel();
        code.iload(geometry[variant + 3]).branch(Opcode.IFLT, negative);
        code.iinc(ordinal, 1).iload(ordinal).iload(geometry[variant + 2])
                .branch(Opcode.IF_ICMPGE, exhausted);
        code.iload(target).iload(geometry[variant + 3]).iadd().istore(target)
                .branch(Opcode.GOTO, loaded);
        code.labelBinding(negative);
        code.iinc(ordinal, -1).iload(ordinal).branch(Opcode.IFLT, exhausted);
        code.iload(target).iload(geometry[variant + 3]).isub().istore(target)
                .branch(Opcode.GOTO, loaded);
        code.labelBinding(exhausted);
        code.loadConstant(-1).istore(target).loadConstant(-1).istore(ordinal)
                .branch(Opcode.GOTO, loaded);
        code.labelBinding(base);
        int baseBoundary = parsed.mapping[0];
        code.iload(geometry[inputBases + baseBoundary]).iload(coordinate).iadd()
                .istore(sourceAddress);
        carriers.load(type, specialization.carrierPattern().get(baseBoundary), baseBoundary,
                sourceAddress, true);
        storeValue(code, type, value);
        code.labelBinding(loaded);
    }

    private static void emitDenseAdvanceCoordinates(CodeBuilder code, int[] geometry,
            int[] coordinates, int extentsBase) {
        var finished = code.newLabel();
        for (int axis = coordinates.length - 1; axis >= 0; axis--) {
            code.iinc(coordinates[axis], 1).iload(coordinates[axis])
                    .iload(geometry[extentsBase + axis]).branch(Opcode.IF_ICMPLT, finished);
            code.loadConstant(0).istore(coordinates[axis]);
        }
        code.labelBinding(finished);
    }

    private static void emitDenseSliceAdvance(CodeBuilder code, int[] geometry,
            int[] coordinates, int[] targets, int[] ordinals, int variant) {
        int rank = coordinates.length;
        var finished = code.newLabel();
        for (int axis = rank - 1; axis >= 0; axis--) {
            int old = code.allocateLocal(TypeKind.INT);
            code.iload(coordinates[axis]).istore(old).iinc(coordinates[axis], 1);
            var wrapped = code.newLabel();
            code.iload(coordinates[axis]).iload(geometry[axis])
                    .branch(Opcode.IF_ICMPGE, wrapped);
            emitDenseSliceCursor(code, geometry, old, targets[axis], ordinals[axis],
                    variant + 2 * rank + axis, variant + 3 * rank + axis);
            code.branch(Opcode.GOTO, finished);
            code.labelBinding(wrapped);
            code.loadConstant(0).istore(coordinates[axis]);
            code.iload(geometry[variant + axis]).istore(targets[axis]);
            code.iload(geometry[variant + rank + axis]).istore(ordinals[axis]);
        }
        code.labelBinding(finished);
    }

    private static void emitDenseSliceCursor(CodeBuilder code, int[] geometry, int old,
            int target, int ordinal, int lengthIndex, int stepIndex) {
        var done = code.newLabel();
        code.iload(old).iload(target).branch(Opcode.IF_ICMPNE, done);
        var negative = code.newLabel();
        var exhausted = code.newLabel();
        code.iload(geometry[stepIndex]).branch(Opcode.IFLT, negative);
        code.iinc(ordinal, 1).iload(ordinal).iload(geometry[lengthIndex])
                .branch(Opcode.IF_ICMPGE, exhausted);
        code.iload(target).iload(geometry[stepIndex]).iadd().istore(target)
                .branch(Opcode.GOTO, done);
        code.labelBinding(negative);
        code.iinc(ordinal, -1).iload(ordinal).branch(Opcode.IFLT, exhausted);
        code.iload(target).iload(geometry[stepIndex]).isub().istore(target)
                .branch(Opcode.GOTO, done);
        code.labelBinding(exhausted);
        code.loadConstant(-1).istore(target).loadConstant(-1).istore(ordinal);
        code.labelBinding(done);
    }

    private static int emitDenseSelectedCoordinate(CodeBuilder code, int axis,
            int[] coordinates) {
        int result = code.allocateLocal(TypeKind.INT);
        code.loadConstant(0).istore(result);
        for (int index = 0; index < coordinates.length; index++) {
            var next = code.newLabel();
            code.iload(axis).loadConstant(index).branch(Opcode.IF_ICMPNE, next);
            code.iload(coordinates[index]).istore(result);
            code.labelBinding(next);
        }
        return result;
    }

    private static int emitDenseCoordinateExcept(CodeBuilder code, int axis, int[] coordinates,
            int sourceAxis, int selected, int prefix) {
        int result = code.allocateLocal(TypeKind.INT);
        var ordinary = code.newLabel();
        var done = code.newLabel();
        code.iload(axis).loadConstant(sourceAxis).branch(Opcode.IF_ICMPNE, ordinary);
        code.iload(selected).iload(prefix).isub().istore(result)
                .branch(Opcode.GOTO, done);
        code.labelBinding(ordinary);
        code.iload(coordinates[sourceAxis]).istore(result);
        code.labelBinding(done);
        return result;
    }

    private static void emitDenseAddressTerm(CodeBuilder code, int[] geometry, int address,
            int coordinate, int strideIndex) {
        code.iload(address).iload(coordinate).iload(geometry[strideIndex]).imul().iadd()
                .istore(address);
    }

    private static void emitDenseUnfold2dAdvance(CodeBuilder code, int[] geometry,
            int ow, int outWidth, int oh, int outHeight, int kw, int kernelWidth,
            int kh, int kernelHeight, int channel, int channels) {
        var done = code.newLabel();
        int next = code.allocateLocal(TypeKind.INT);
        code.iload(geometry[ow]).loadConstant(1).iadd().istore(next);
        var carryOh = code.newLabel();
        code.iload(next).iload(geometry[outWidth]).branch(Opcode.IF_ICMPGE, carryOh);
        code.iload(next).istore(geometry[ow]).branch(Opcode.GOTO, done);
        code.labelBinding(carryOh);
        code.loadConstant(0).istore(geometry[ow]);
        code.iload(geometry[oh]).loadConstant(1).iadd().istore(next);
        var carryKw = code.newLabel();
        code.iload(next).iload(geometry[outHeight]).branch(Opcode.IF_ICMPGE, carryKw);
        code.iload(next).istore(geometry[oh]).branch(Opcode.GOTO, done);
        code.labelBinding(carryKw);
        code.loadConstant(0).istore(geometry[oh]);
        code.iload(geometry[kw]).loadConstant(1).iadd().istore(next);
        var carryKh = code.newLabel();
        code.iload(next).iload(geometry[kernelWidth]).branch(Opcode.IF_ICMPGE, carryKh);
        code.iload(next).istore(geometry[kw]).branch(Opcode.GOTO, done);
        code.labelBinding(carryKh);
        code.loadConstant(0).istore(geometry[kw]);
        code.iload(geometry[kh]).loadConstant(1).iadd().istore(next);
        var carryChannel = code.newLabel();
        code.iload(next).iload(geometry[kernelHeight]).branch(Opcode.IF_ICMPGE, carryChannel);
        code.iload(next).istore(geometry[kh]).branch(Opcode.GOTO, done);
        code.labelBinding(carryChannel);
        code.loadConstant(0).istore(geometry[kh]);
        code.iload(geometry[channel]).loadConstant(1).iadd().istore(next);
        var reset = code.newLabel();
        code.iload(next).iload(geometry[channels]).branch(Opcode.IF_ICMPGE, reset);
        code.iload(next).istore(geometry[channel]).branch(Opcode.GOTO, done);
        code.labelBinding(reset);
        code.loadConstant(0).istore(geometry[channel]);
        code.labelBinding(done);
    }

    private static void emitMovementFailure(CodeBuilder code, String message) {
        code.new_(ClassDesc.of("java.lang.IllegalStateException")).dup().loadConstant(message)
                .invokespecial(ClassDesc.of("java.lang.IllegalStateException"), "<init>",
                        MethodTypeDesc.ofDescriptor("(Ljava/lang/String;)V")).athrow();
    }

    private static void emitSliceUpdate(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType type, Parsed parsed, int geometrySlot,
            int[] coordinates, int[] targets, int[] ordinals, int sourceAddress, int value,
            int inputBases, int[] strideOffsets) {
        var base = code.newLabel();
        var loaded = code.newLabel();
        for (int axis = 0; axis < coordinates.length; axis++) {
            code.lload(coordinates[axis]).lload(targets[axis]).lcmp()
                    .branch(Opcode.IFNE, base);
        }
        int updateBoundary = parsed.mapping[1];
        geometry(code, geometrySlot, inputBases + updateBoundary).lstore(sourceAddress);
        for (int axis = 0; axis < coordinates.length; axis++) {
            code.lload(sourceAddress).lload(ordinals[axis]).aload(geometrySlot)
                    .loadConstant(strideOffsets[updateBoundary] + axis).laload().lmul().ladd()
                    .lstore(sourceAddress);
        }
        carriers.load(type, specialization.carrierPattern().get(updateBoundary), updateBoundary,
                sourceAddress);
        storeValue(code, type, value);
        code.branch(Opcode.GOTO, loaded);
        code.labelBinding(base);
        int baseBoundary = parsed.mapping[0];
        geometry(code, geometrySlot, inputBases + baseBoundary).lstore(sourceAddress);
        for (int axis = 0; axis < coordinates.length; axis++) {
            code.lload(sourceAddress).lload(coordinates[axis]).aload(geometrySlot)
                    .loadConstant(strideOffsets[baseBoundary] + axis).laload().lmul().ladd()
                    .lstore(sourceAddress);
        }
        carriers.load(type, specialization.carrierPattern().get(baseBoundary), baseBoundary,
                sourceAddress);
        storeValue(code, type, value);
        code.labelBinding(loaded);
    }

    private static void emitSliceAdvance(CodeBuilder code, int geometrySlot, int[] coordinates,
            int address, int[] targets, int[] ordinals, int variant, int stridesBase) {
        int rank = coordinates.length;
        var finished = code.newLabel();
        for (int axis = rank - 1; axis >= 0; axis--) {
            int old = code.allocateLocal(TypeKind.LONG);
            code.lload(coordinates[axis]).lstore(old);
            code.lload(coordinates[axis]).loadConstant(1L).ladd().lstore(coordinates[axis]);
            code.lload(address).aload(geometrySlot).loadConstant(stridesBase + axis).laload()
                    .ladd().lstore(address);
            var wrapped = code.newLabel();
            code.lload(coordinates[axis]).aload(geometrySlot).loadConstant(axis).laload().lcmp()
                    .branch(Opcode.IFGE, wrapped);
            emitAdvanceSliceCursor(code, geometrySlot, old, targets[axis], ordinals[axis],
                    variant + 2 * rank + axis, variant + 3 * rank + axis);
            code.branch(Opcode.GOTO, finished);
            code.labelBinding(wrapped);
            code.loadConstant(0L).lstore(coordinates[axis]);
            code.lload(address).aload(geometrySlot).loadConstant(axis).laload()
                    .aload(geometrySlot).loadConstant(stridesBase + axis).laload().lmul()
                    .lsub().lstore(address);
            geometry(code, geometrySlot, variant + axis).lstore(targets[axis]);
            geometry(code, geometrySlot, variant + rank + axis).lstore(ordinals[axis]);
        }
        code.labelBinding(finished);
    }

    private static void emitAdvanceSliceCursor(CodeBuilder code, int geometrySlot, int old,
            int target, int ordinal, int lengthIndex, int stepIndex) {
        var done = code.newLabel();
        code.lload(old).lload(target).lcmp().branch(Opcode.IFNE, done);
        var negative = code.newLabel();
        var exhausted = code.newLabel();
        geometry(code, geometrySlot, stepIndex).loadConstant(0L).lcmp()
                .branch(Opcode.IFLT, negative);
        code.lload(ordinal).loadConstant(1L).ladd().lstore(ordinal);
        code.lload(ordinal).aload(geometrySlot).loadConstant(lengthIndex).laload().lcmp()
                .branch(Opcode.IFGE, exhausted);
        code.lload(target).aload(geometrySlot).loadConstant(stepIndex).laload().ladd().lstore(target)
                .branch(Opcode.GOTO, done);
        code.labelBinding(negative);
        code.lload(ordinal).loadConstant(1L).lsub().lstore(ordinal);
        code.lload(ordinal).loadConstant(0L).lcmp().branch(Opcode.IFLT, exhausted);
        code.lload(target).aload(geometrySlot).loadConstant(stepIndex).laload().lsub().lstore(target)
                .branch(Opcode.GOTO, done);
        code.labelBinding(exhausted);
        code.loadConstant(-1L).lstore(target);
        code.loadConstant(-1L).lstore(ordinal);
        code.labelBinding(done);
    }

    private static void emitPad(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType type, long bits, int geometrySlot,
            int[] coordinates, int sourceAddress, int value, int inputBases,
            int inputStrides, int variant) {
        var padding = code.newLabel();
        var loaded = code.newLabel();
        for (int axis = 0; axis < coordinates.length; axis++) {
            code.lload(coordinates[axis]).aload(geometrySlot).loadConstant(variant + axis)
                    .laload().lcmp().branch(Opcode.IFLT, padding);
            code.lload(coordinates[axis]).aload(geometrySlot).loadConstant(variant + axis)
                    .laload().lsub().aload(geometrySlot)
                    .loadConstant(variant + coordinates.length + axis).laload().lcmp()
                    .branch(Opcode.IFGE, padding);
        }
        geometry(code, geometrySlot, inputBases).lstore(sourceAddress);
        for (int axis = 0; axis < coordinates.length; axis++) {
            code.lload(sourceAddress).lload(coordinates[axis])
                    .aload(geometrySlot).loadConstant(variant + axis).laload().lsub()
                    .aload(geometrySlot).loadConstant(inputStrides + axis).laload().lmul()
                    .ladd().lstore(sourceAddress);
        }
        carriers.load(type, specialization.carrierPattern().getFirst(), 0, sourceAddress);
        storeValue(code, type, value);
        code.branch(Opcode.GOTO, loaded);
        code.labelBinding(padding);
        loadImmediate(code, type, bits);
        storeValue(code, type, value);
        code.labelBinding(loaded);
    }

    private static void emitTile(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType type, int geometrySlot,
            int[] tileCoordinates, int sourceAddress, int value, int inputBases,
            int inputStrides) {
        geometry(code, geometrySlot, inputBases).lstore(sourceAddress);
        for (int axis = 0; axis < tileCoordinates.length; axis++) {
            code.lload(sourceAddress).lload(tileCoordinates[axis]).aload(geometrySlot)
                    .loadConstant(inputStrides + axis).laload().lmul().ladd()
                    .lstore(sourceAddress);
        }
        carriers.load(type, specialization.carrierPattern().getFirst(), 0, sourceAddress);
        storeValue(code, type, value);
    }

    private static void emitConcat(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType type, Parsed parsed,
            int geometrySlot, int[] coordinates, int sourceAddress, int value,
            int inputBases, int[] strideOffsets, int variant, int inputRank) {
        int axisCoordinate = selectedCoordinate(code, geometrySlot, variant, coordinates);
        var loaded = code.newLabel();
        for (int occurrence = 0; occurrence < parsed.mapping.length; occurrence++) {
            var next = code.newLabel();
            code.lload(axisCoordinate).aload(geometrySlot)
                    .loadConstant(variant + 2 + occurrence).laload().lcmp()
                    .branch(Opcode.IFGE, next);
            int boundary = parsed.mapping[occurrence];
            geometry(code, geometrySlot, inputBases + boundary).lstore(sourceAddress);
            for (int axis = 0; axis < inputRank; axis++) {
                int coordinate = coordinateExcept(code, geometrySlot, variant,
                        coordinates, axis, axisCoordinate,
                        variant + 1 + occurrence);
                code.lload(sourceAddress).lload(coordinate).aload(geometrySlot)
                        .loadConstant(strideOffsets[boundary] + axis).laload()
                        .lmul().ladd().lstore(sourceAddress);
            }
            carriers.load(type, specialization.carrierPattern().get(boundary), boundary,
                    sourceAddress);
            storeValue(code, type, value);
            code.branch(Opcode.GOTO, loaded);
            code.labelBinding(next);
        }
        code.new_(ClassDesc.of("java.lang.IllegalStateException")).dup()
                .loadConstant("composition coordinate has no segment")
                .invokespecial(ClassDesc.of("java.lang.IllegalStateException"), "<init>",
                        MethodTypeDesc.ofDescriptor("(Ljava/lang/String;)V"))
                .athrow();
        code.labelBinding(loaded);
    }

    private static void emitStack(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType type, Parsed parsed,
            int geometrySlot, int[] coordinates, int sourceAddress, int value,
            int inputBases, int[] strideOffsets, int variant, int inputRank) {
        int occurrenceCoordinate = selectedCoordinate(code, geometrySlot, variant, coordinates);
        var loaded = code.newLabel();
        for (int occurrence = 0; occurrence < parsed.mapping.length; occurrence++) {
            var next = code.newLabel();
            code.lload(occurrenceCoordinate).loadConstant((long) occurrence).lcmp()
                    .branch(Opcode.IFNE, next);
            int boundary = parsed.mapping[occurrence];
            geometry(code, geometrySlot, inputBases + boundary).lstore(sourceAddress);
            for (int inputAxis = 0; inputAxis < inputRank; inputAxis++) {
                int coordinate = stackCoordinate(code, geometrySlot, variant,
                        coordinates, inputAxis);
                code.lload(sourceAddress).lload(coordinate).aload(geometrySlot)
                        .loadConstant(strideOffsets[boundary] + inputAxis).laload()
                        .lmul().ladd().lstore(sourceAddress);
            }
            carriers.load(type, specialization.carrierPattern().get(boundary), boundary,
                    sourceAddress);
            storeValue(code, type, value);
            code.branch(Opcode.GOTO, loaded);
            code.labelBinding(next);
        }
        code.new_(ClassDesc.of("java.lang.IllegalStateException")).dup()
                .loadConstant("stack coordinate has no input")
                .invokespecial(ClassDesc.of("java.lang.IllegalStateException"), "<init>",
                        MethodTypeDesc.ofDescriptor("(Ljava/lang/String;)V"))
                .athrow();
        code.labelBinding(loaded);
    }

    private static void emitUnfoldAxis(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType type, int geometrySlot,
            int[] coordinates, int sourceAddress, int value, int inputBases,
            int inputStrides, int variant, int inputRank) {
        geometry(code, geometrySlot, inputBases).lstore(sourceAddress);
        for (int inputAxis = 0; inputAxis < inputRank; inputAxis++) {
            int sourceCoordinate = code.allocateLocal(TypeKind.LONG);
            var ordinary = code.newLabel();
            var selected = code.newLabel();
            geometry(code, geometrySlot, variant).loadConstant((long) inputAxis).lcmp()
                    .branch(Opcode.IFEQ, selected);
            code.lload(coordinates[inputAxis]).lstore(sourceCoordinate)
                    .branch(Opcode.GOTO, ordinary);
            code.labelBinding(selected);
            code.lload(coordinates[inputAxis]).aload(geometrySlot).loadConstant(variant + 2)
                    .laload().lmul().lload(coordinates[coordinates.length - 1]).ladd()
                    .lstore(sourceCoordinate);
            code.labelBinding(ordinary);
            code.lload(sourceAddress).lload(sourceCoordinate).aload(geometrySlot)
                    .loadConstant(inputStrides + inputAxis).laload().lmul().ladd()
                    .lstore(sourceAddress);
        }
        carriers.load(type, specialization.carrierPattern().getFirst(), 0, sourceAddress);
        storeValue(code, type, value);
    }

    private static void emitUnfold2d(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType type, long bits, int geometrySlot,
            int[] coordinates, int sourceAddress, int value, int inputBases,
            int inputStrides, int variant) {
        int channel = variant + 13, kh = variant + 14, kw = variant + 15;
        int oh = variant + 16, ow = variant + 17;
        int ih = code.allocateLocal(TypeKind.LONG);
        int iw = code.allocateLocal(TypeKind.LONG);
        geometry(code, geometrySlot, oh).aload(geometrySlot).loadConstant(variant + 5).laload()
                .lmul().aload(geometrySlot).loadConstant(variant + 7).laload().lsub()
                .aload(geometrySlot).loadConstant(kh).laload()
                .aload(geometrySlot).loadConstant(variant + 9).laload().lmul().ladd().lstore(ih);
        geometry(code, geometrySlot, ow).aload(geometrySlot).loadConstant(variant + 6).laload()
                .lmul().aload(geometrySlot).loadConstant(variant + 8).laload().lsub()
                .aload(geometrySlot).loadConstant(kw).laload()
                .aload(geometrySlot).loadConstant(variant + 10).laload().lmul().ladd().lstore(iw);
        var padding = code.newLabel();
        var loaded = code.newLabel();
        code.lload(ih).loadConstant(0L).lcmp().branch(Opcode.IFLT, padding);
        code.lload(ih).aload(geometrySlot).loadConstant(variant + 1).laload().lcmp()
                .branch(Opcode.IFGE, padding);
        code.lload(iw).loadConstant(0L).lcmp().branch(Opcode.IFLT, padding);
        code.lload(iw).aload(geometrySlot).loadConstant(variant + 2).laload().lcmp()
                .branch(Opcode.IFGE, padding);
        geometry(code, geometrySlot, inputBases).lstore(sourceAddress);
        addAddressTerm(code, geometrySlot, sourceAddress, coordinates[0], inputStrides);
        addGeometryAddressTerm(code, geometrySlot, sourceAddress, channel, inputStrides + 1);
        addAddressTerm(code, geometrySlot, sourceAddress, ih, inputStrides + 2);
        addAddressTerm(code, geometrySlot, sourceAddress, iw, inputStrides + 3);
        carriers.load(type, specialization.carrierPattern().getFirst(), 0, sourceAddress);
        storeValue(code, type, value);
        code.branch(Opcode.GOTO, loaded);
        code.labelBinding(padding);
        loadImmediate(code, type, bits);
        storeValue(code, type, value);
        code.labelBinding(loaded);
        advanceUnfold2d(code, geometrySlot, ow, variant + 12, oh, variant + 11,
                kw, variant + 4, kh, variant + 3, channel, variant);
    }

    private static void addAddressTerm(CodeBuilder code, int geometrySlot, int address,
            int coordinate, int strideIndex) {
        code.lload(address).lload(coordinate).aload(geometrySlot).loadConstant(strideIndex)
                .laload().lmul().ladd().lstore(address);
    }

    private static void addGeometryAddressTerm(CodeBuilder code, int geometrySlot, int address,
            int coordinateIndex, int strideIndex) {
        code.lload(address).aload(geometrySlot).loadConstant(coordinateIndex).laload()
                .aload(geometrySlot).loadConstant(strideIndex).laload().lmul().ladd()
                .lstore(address);
    }

    private static void advanceUnfold2d(CodeBuilder code, int geometrySlot,
            int ow, int outWidth, int oh, int outHeight, int kw, int kernelWidth,
            int kh, int kernelHeight, int channel, int channels) {
        var done = code.newLabel();
        int next = code.allocateLocal(TypeKind.LONG);
        geometry(code, geometrySlot, ow).loadConstant(1L).ladd().lstore(next);
        var carryOh = code.newLabel();
        code.lload(next).aload(geometrySlot).loadConstant(outWidth).laload().lcmp()
                .branch(Opcode.IFGE, carryOh);
        code.aload(geometrySlot).loadConstant(ow).lload(next).lastore()
                .branch(Opcode.GOTO, done);
        code.labelBinding(carryOh);
        code.aload(geometrySlot).loadConstant(ow).loadConstant(0L).lastore();
        geometry(code, geometrySlot, oh).loadConstant(1L).ladd().lstore(next);
        var carryKw = code.newLabel();
        code.lload(next).aload(geometrySlot).loadConstant(outHeight).laload().lcmp()
                .branch(Opcode.IFGE, carryKw);
        code.aload(geometrySlot).loadConstant(oh).lload(next).lastore()
                .branch(Opcode.GOTO, done);
        code.labelBinding(carryKw);
        code.aload(geometrySlot).loadConstant(oh).loadConstant(0L).lastore();
        geometry(code, geometrySlot, kw).loadConstant(1L).ladd().lstore(next);
        var carryKh = code.newLabel();
        code.lload(next).aload(geometrySlot).loadConstant(kernelWidth).laload().lcmp()
                .branch(Opcode.IFGE, carryKh);
        code.aload(geometrySlot).loadConstant(kw).lload(next).lastore()
                .branch(Opcode.GOTO, done);
        code.labelBinding(carryKh);
        code.aload(geometrySlot).loadConstant(kw).loadConstant(0L).lastore();
        geometry(code, geometrySlot, kh).loadConstant(1L).ladd().lstore(next);
        var carryChannel = code.newLabel();
        code.lload(next).aload(geometrySlot).loadConstant(kernelHeight).laload().lcmp()
                .branch(Opcode.IFGE, carryChannel);
        code.aload(geometrySlot).loadConstant(kh).lload(next).lastore()
                .branch(Opcode.GOTO, done);
        code.labelBinding(carryChannel);
        code.aload(geometrySlot).loadConstant(kh).loadConstant(0L).lastore();
        geometry(code, geometrySlot, channel).loadConstant(1L).ladd().lstore(next);
        var resetChannel = code.newLabel();
        code.lload(next).aload(geometrySlot).loadConstant(channels).laload().lcmp()
                .branch(Opcode.IFGE, resetChannel);
        code.aload(geometrySlot).loadConstant(channel).lload(next).lastore()
                .branch(Opcode.GOTO, done);
        code.labelBinding(resetChannel);
        code.aload(geometrySlot).loadConstant(channel).loadConstant(0L).lastore();
        code.labelBinding(done);
    }

    private static int selectedCoordinate(CodeBuilder code, int geometrySlot, int axisIndex,
            int[] coordinates) {
        int result = code.allocateLocal(TypeKind.LONG);
        code.loadConstant(0L).lstore(result);
        for (int axis = 0; axis < coordinates.length; axis++) {
            var next = code.newLabel();
            geometry(code, geometrySlot, axisIndex).loadConstant((long) axis).lcmp()
                    .branch(Opcode.IFNE, next);
            code.lload(coordinates[axis]).lstore(result);
            code.labelBinding(next);
        }
        return result;
    }

    private static int coordinateExcept(CodeBuilder code, int geometrySlot, int axisIndex,
            int[] coordinates, int sourceAxis, int selected, int prefixIndex) {
        int result = code.allocateLocal(TypeKind.LONG);
        var ordinary = code.newLabel();
        var done = code.newLabel();
        geometry(code, geometrySlot, axisIndex).loadConstant((long) sourceAxis).lcmp()
                .branch(Opcode.IFNE, ordinary);
        code.lload(selected).aload(geometrySlot).loadConstant(prefixIndex).laload().lsub()
                .lstore(result).branch(Opcode.GOTO, done);
        code.labelBinding(ordinary);
        code.lload(coordinates[sourceAxis]).lstore(result);
        code.labelBinding(done);
        return result;
    }

    private static int stackCoordinate(CodeBuilder code, int geometrySlot, int axisIndex,
            int[] coordinates, int inputAxis) {
        int result = code.allocateLocal(TypeKind.LONG);
        var shifted = code.newLabel();
        var done = code.newLabel();
        geometry(code, geometrySlot, axisIndex).loadConstant((long) inputAxis).lcmp()
                .branch(Opcode.IFLE, shifted);
        code.lload(coordinates[inputAxis]).lstore(result).branch(Opcode.GOTO, done);
        code.labelBinding(shifted);
        code.lload(coordinates[inputAxis + 1]).lstore(result);
        code.labelBinding(done);
        return result;
    }

    private static void emitTileAdvance(CodeBuilder code, int geometrySlot,
            int[] coordinates, int outputAddress, int[] tileCoordinates, int inputExtents,
            int outputStrides) {
        var finished = code.newLabel();
        for (int axis = coordinates.length - 1; axis >= 0; axis--) {
            code.lload(tileCoordinates[axis]).loadConstant(1L).ladd()
                    .lstore(tileCoordinates[axis]);
            var sourceReady = code.newLabel();
            code.lload(tileCoordinates[axis]).aload(geometrySlot)
                    .loadConstant(inputExtents + axis).laload().lcmp()
                    .branch(Opcode.IFLT, sourceReady);
            code.loadConstant(0L).lstore(tileCoordinates[axis]);
            code.labelBinding(sourceReady);
            code.lload(coordinates[axis]).loadConstant(1L).ladd().lstore(coordinates[axis]);
            code.lload(outputAddress).aload(geometrySlot)
                    .loadConstant(outputStrides + axis).laload().ladd().lstore(outputAddress);
            code.lload(coordinates[axis]).aload(geometrySlot).loadConstant(axis).laload().lcmp()
                    .branch(Opcode.IFLT, finished);
            code.loadConstant(0L).lstore(coordinates[axis]);
            code.lload(outputAddress).aload(geometrySlot).loadConstant(axis).laload()
                    .aload(geometrySlot).loadConstant(outputStrides + axis).laload().lmul()
                    .lsub().lstore(outputAddress);
        }
        code.labelBinding(finished);
    }

    private static void emitAdvance(CodeBuilder code, int geometrySlot, int[] coordinates,
            int address, int extentsBase, int stridesBase) {
        var finished = code.newLabel();
        for (int axis = coordinates.length - 1; axis >= 0; axis--) {
            code.lload(coordinates[axis]).loadConstant(1L).ladd().lstore(coordinates[axis]);
            code.lload(address).aload(geometrySlot).loadConstant(stridesBase + axis).laload()
                    .ladd().lstore(address);
            code.lload(coordinates[axis]).aload(geometrySlot)
                    .loadConstant(extentsBase + axis).laload().lcmp()
                    .branch(Opcode.IFLT, finished);
            code.loadConstant(0L).lstore(coordinates[axis]);
            code.lload(address).aload(geometrySlot).loadConstant(extentsBase + axis).laload()
                    .aload(geometrySlot).loadConstant(stridesBase + axis).laload().lmul()
                    .lsub().lstore(address);
        }
        code.labelBinding(finished);
    }

    private static CodeBuilder geometry(CodeBuilder code, int slot, int index) {
        return code.aload(slot).loadConstant(index).laload();
    }

    private static void loadImmediate(CodeBuilder code, DataType type, long bits) {
        switch (type) {
            case FLOAT64 -> code.loadConstant(bits).invokestatic(ClassDesc.of("java.lang.Double"),
                    "longBitsToDouble", MethodTypeDesc.ofDescriptor("(J)D"));
            case FLOAT32 -> code.loadConstant((int) bits).invokestatic(ClassDesc.of("java.lang.Float"),
                    "intBitsToFloat", MethodTypeDesc.ofDescriptor("(I)F"));
            case BFLOAT16, INT32, BOOL -> code.loadConstant((int) bits);
            case INT64 -> code.loadConstant(bits);
        }
    }

    private static void storeValue(CodeBuilder code, DataType type, int local) {
        switch (type) {
            case FLOAT64 -> code.dstore(local);
            case FLOAT32 -> code.fstore(local);
            case BFLOAT16, INT32, BOOL -> code.istore(local);
            case INT64 -> code.lstore(local);
        }
    }

    private record Parsed(String family, int rank, int[] mapping, long bits) {
        static Parsed parse(String identity) {
            if (!identity.startsWith("movement:")) {
                throw new IllegalArgumentException("not an encoded movement identity");
            }
            String[] fields = identity.split(":");
            String family = fields[1];
            int rank = Integer.parseInt(fields[2].substring("rank=".length()));
            String map = fields[3].substring("map=".length());
            int[] mapping = map.isEmpty() ? new int[0]
                    : Arrays.stream(map.split(",")).mapToInt(Integer::parseInt).toArray();
            long bits = Long.parseUnsignedLong(fields[4].substring("bits=".length()), 16);
            return new Parsed(family, rank, mapping, bits);
        }
    }
}
