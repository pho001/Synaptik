package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
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
 * the finite signed sequences and otherwise select the base. Proved dense heap-array forms narrow
 * and hoist their geometry into invocation-local integer state; other forms retain long state.
 * Emitted hot loops use carry/reset coordinate state and contain no Model interpretation,
 * reflection, map lookup, per-element allocation, division, or modulo.</p>
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
