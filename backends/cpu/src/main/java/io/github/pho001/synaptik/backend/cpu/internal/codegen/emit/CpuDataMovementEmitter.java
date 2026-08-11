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
 * <p>The encoded structural IR chooses PAD, TILE, CONCAT, STACK, UNFOLD_AXIS, or UNFOLD2D during
 * class generation. Compact primitive geometry supplies range-start coordinates, carrier bases,
 * rank-specific strides, and family facts at invocation time. Emitted hot loops use carry/reset
 * coordinate state and contain no Model interpretation, reflection, map lookup, per-element
 * allocation, division, or modulo.</p>
 */
final class CpuDataMovementEmitter {
    /** Creates one stateless family emitter. */
    CpuDataMovementEmitter() {
    }

    /**
     * Emits the family-specialized loop represented by an encoded movement IR.
     *
     * @param code non-null method body receiving generated instructions
     * @param ir non-null instruction-free structural movement encoding
     * @param specialization non-null exact carrier and represented-type specialization
     * @throws IllegalArgumentException if the structural family identity is not a supported
     *     movement encoding
     */
    void emit(CodeBuilder code, CpuKernelSpecialization specialization, CpuKernelIr ir) {
        Parsed parsed = Parsed.parse(ir.familyIdentity());
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
        if (parsed.family.equals("TILE")) {
            tileCoordinates = new int[rank];
            for (int axis = 0; axis < rank; axis++) {
                tileCoordinates[axis] = code.allocateLocal(TypeKind.LONG);
                geometry(code, geometrySlot, variant + rank + axis)
                        .lstore(tileCoordinates[axis]);
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
            default -> throw new IllegalArgumentException("unsupported movement family");
        }
        carriers.store(type, specialization.carrierPattern().get(uniqueInputs), uniqueInputs,
                outputAddress, value);
        emitAdvance(code, geometrySlot, coordinates, outputAddress, 0, 2 * rank + 1);
        if (tileCoordinates != null) emitAdvance(code, geometrySlot, tileCoordinates,
                sourceAddress, variant, strideOffsets[0]);
        code.lload(logical).loadConstant(1L).ladd().lstore(logical);
        code.branch(Opcode.GOTO, loop);
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
