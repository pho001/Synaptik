package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.MethodTypeDesc;

/**
 * Emits one direct arbitrary-axis batch-normalization inference body.
 *
 * <p>The selected channel or non-channel range form is baked into the class. Each channel loop
 * loads scale, bias, running mean, and running variance and computes one square root before its
 * coordinate loop. Every coordinate then performs subtraction, division, multiplication, and
 * addition at the exact result computation boundary and stores directly. Entry coordinates are
 * decoded once and subsequent addresses advance incrementally; BFLOAT16 and FLOAT32 use real
 * FLOAT32 locals while FLOAT64 uses FLOAT64 locals.</p>
 */
public final class CpuBatchNormInferenceEmitter {
    private static final ClassDesc MATH = ClassDesc.of(Math.class.getName());
    private static final ClassDesc FLOAT_CLASS = ClassDesc.of(Float.class.getName());

    /** Creates a stateless generation-time emitter. */
    public CpuBatchNormInferenceEmitter() { }

    /**
     * Emits one exact selected-form specialization.
     *
     * @param code non-null Class-File method builder
     * @param specialization non-null exact typed carrier specialization without scratch
     * @param ir non-null matching instruction-free batch-inference identity
     * @throws NullPointerException if a required argument or specialization fact is null
     * @throws IllegalArgumentException if family, map, carrier, or scratch facts disagree
     */
    public void emit(CodeBuilder code, CpuKernelSpecialization specialization, CpuKernelIr ir) {
        String identity = ir.familyIdentity();
        boolean channelRange = identity.endsWith(":range=CHANNEL_RANGE");
        boolean nonChannelRange = identity.endsWith(":range=NON_CHANNEL_RANGE");
        int[] map = CpuLayerNormEmitter.map(identity);
        int boundaries = specialization.carrierPattern().size();
        if ((!channelRange && !nonChannelRange) || map.length != 5 || boundaries < 2
                || specialization.boundaryDataTypes().size() != boundaries
                || specialization.scratchParameter()) {
            throw new IllegalArgumentException("batch-normalization specialization disagrees");
        }
        int outputBoundary = boundaries - 1;
        DataType resultType = specialization.boundaryDataTypes().getLast();
        boolean floatComputation = resultType != DataType.FLOAT64;
        int rank = Integer.parseInt(CpuLayerNormEmitter.field(identity, ":rank="));
        int axis = Integer.parseInt(CpuLayerNormEmitter.field(identity, ":axis="));
        int geometry = boundaries, start = geometry + 1, end = start + 2;
        int[] ranks = new int[boundaries], extents = new int[boundaries], strides = new int[boundaries];
        int cursor = 9 + boundaries;
        for (int boundary = 0; boundary < boundaries; boundary++) {
            ranks[boundary] = ir.values().get(boundary).accessPlan().iterationRank();
            extents[boundary] = cursor + 1;
            strides[boundary] = extents[boundary] + ranks[boundary];
            cursor += 1 + 2 * ranks[boundary];
        }
        var carriers = new CpuCarrierEmitter(code);
        int channel = code.allocateLocal(TypeKind.LONG);
        int nonChannel = code.allocateLocal(TypeKind.LONG);
        int prefixCoordinate = code.allocateLocal(TypeKind.LONG);
        int suffixCoordinate = code.allocateLocal(TypeKind.LONG);
        int remaining = code.allocateLocal(TypeKind.LONG);
        int coordinate = code.allocateLocal(TypeKind.LONG);
        int address = code.allocateLocal(TypeKind.LONG);
        int outputAddress = code.allocateLocal(TypeKind.LONG);
        int[] represented = new int[5];
        int[] values = new int[5];
        for (int position = 0; position < 5; position++) {
            represented[position] = CpuLayerNormEmitter.represented(code,
                    specialization.boundaryDataTypes().get(map[position]));
            values[position] = code.allocateLocal(floatComputation ? TypeKind.FLOAT : TypeKind.DOUBLE);
        }
        TypeKind computationKind = floatComputation ? TypeKind.FLOAT : TypeKind.DOUBLE;
        int epsilon = code.allocateLocal(computationKind);
        int radicand = code.allocateLocal(computationKind);
        int denominator = code.allocateLocal(computationKind);
        int centered = code.allocateLocal(computationKind);
        int standardized = code.allocateLocal(computationKind);
        int scaled = code.allocateLocal(computationKind);
        int result = code.allocateLocal(computationKind);
        int encodedStore = resultType == DataType.BFLOAT16
                ? code.allocateLocal(TypeKind.DOUBLE) : -1;
        if (floatComputation) code.loadConstant((float) CpuLayerNormEmitter.epsilon(identity,
                resultType)).fstore(epsilon);
        else code.loadConstant(CpuLayerNormEmitter.epsilon(identity, resultType)).dstore(epsilon);
        boolean denseTensorPair = ir.values().get(map[0]).accessPlan().regime()
                == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Regime.DENSE_LINEAR
                && ir.values().get(outputBoundary).accessPlan().regime()
                == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Regime.DENSE_LINEAR;
        boolean simpleTensorPair = denseTensorPair || rank == 3 && axis == 1;

        if (channelRange) {
            code.lload(start).lstore(channel);
            var channels = code.newLabel(); var complete = code.newLabel();
            code.labelBinding(channels).lload(channel).lload(end).lcmp()
                    .branch(Opcode.IFGE, complete);
            emitChannelValues(code, carriers, specialization, geometry, map, strides,
                    represented, values, channel, address, epsilon, radicand, denominator,
                    resultType, floatComputation);
            code.loadConstant(0L).lstore(nonChannel);
            emitCoordinateLoop(code, carriers, specialization, geometry, rank, axis, map,
                    extents, strides, represented[0], values, channel, nonChannel, remaining,
                    coordinate, address, denominator, centered, standardized, scaled, result,
                    resultType, 6, false, denseTensorPair, simpleTensorPair,
                    prefixCoordinate, suffixCoordinate, outputAddress, floatComputation,
                    encodedStore);
            code.lload(channel).loadConstant(1L).ladd().lstore(channel)
                    .branch(Opcode.GOTO, channels).labelBinding(complete);
            return;
        }

        code.loadConstant(0L).lstore(channel);
        var channels = code.newLabel(); var complete = code.newLabel();
        code.labelBinding(channels).lload(channel); CpuNormEmitter.geometry(code, geometry, 4)
                .lcmp().branch(Opcode.IFGE, complete);
        emitChannelValues(code, carriers, specialization, geometry, map, strides,
                represented, values, channel, address, epsilon, radicand, denominator,
                resultType, floatComputation);
        code.lload(start).lstore(nonChannel);
        emitCoordinateLoop(code, carriers, specialization, geometry, rank, axis, map,
                extents, strides, represented[0], values, channel, nonChannel, remaining,
                coordinate, address, denominator, centered, standardized, scaled, result,
                resultType, -1, true, denseTensorPair, simpleTensorPair,
                prefixCoordinate, suffixCoordinate, outputAddress, floatComputation,
                encodedStore);
        code.lload(channel).loadConstant(1L).ladd().lstore(channel)
                .branch(Opcode.GOTO, channels).labelBinding(complete);
    }

    private static void emitChannelValues(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, int geometry, int[] map, int[] strides,
            int[] represented, int[] values, int channel, int address, int epsilon,
            int radicand, int denominator, DataType resultType, boolean floatComputation) {
        for (int position = 1; position < 5; position++) {
            int boundary = map[position];
            CpuNormEmitter.geometry(code, geometry, 9 + boundary).lload(channel);
            CpuNormEmitter.geometry(code, geometry, strides[boundary]).lmul().ladd()
                    .lstore(address);
            loadValue(code, carriers, specialization, boundary, represented[position],
                    values[position], address, floatComputation);
        }
        arithmetic(code, resultType, values[4], epsilon, Opcode.DADD, radicand,
                floatComputation);
        if (floatComputation) code.fload(radicand).f2d()
                .invokestatic(MATH, "sqrt", CpuNormEmitter.doubleUnary()).d2f()
                .fstore(denominator);
        else code.dload(radicand).invokestatic(MATH, "sqrt", CpuNormEmitter.doubleUnary())
                .dstore(denominator);
    }

    private static void emitCoordinateLoop(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, int geometry, int rank, int axis, int[] map,
            int[] extents, int[] strides, int inputRepresented, int[] values, int channel,
            int nonChannel, int remaining, int coordinate, int address, int denominator,
            int centered, int standardized, int scaled, int result, DataType resultType,
            int endGeometryIndex, boolean useEntryEnd, boolean denseTensorPair,
            boolean simpleTensorPair, int prefixCoordinate, int suffixCoordinate,
            int outputAddress, boolean floatComputation, int encodedStore) {
        int entryEnd = geometry + 3;
        int output = specialization.carrierPattern().size() - 1;
        if (simpleTensorPair) {
            code.lload(nonChannel); CpuNormEmitter.geometry(code, geometry, 5).ldiv()
                    .lstore(prefixCoordinate);
            code.lload(nonChannel); CpuNormEmitter.geometry(code, geometry, 5).lrem()
                    .lstore(suffixCoordinate);
            if (denseTensorPair) {
                emitDenseAddress(code, geometry, 9 + map[0], channel, prefixCoordinate,
                        suffixCoordinate, address);
                emitDenseAddress(code, geometry, 9 + output, channel, prefixCoordinate,
                        suffixCoordinate, outputAddress);
            } else {
                emitRankThreeMiddleAddress(code, geometry, 9 + map[0], strides[map[0]], channel,
                        prefixCoordinate, suffixCoordinate, address);
                emitRankThreeMiddleAddress(code, geometry, 9 + output, strides[output], channel,
                        prefixCoordinate, suffixCoordinate, outputAddress);
            }
        }
        var loop = code.newLabel(); var done = code.newLabel();
        code.labelBinding(loop).lload(nonChannel);
        if (useEntryEnd) code.lload(entryEnd); else CpuNormEmitter.geometry(code, geometry,
                endGeometryIndex);
        code.lcmp().branch(Opcode.IFGE, done);
        if (!simpleTensorPair) emitTensorAddress(code, geometry, rank, axis, extents[map[0]], strides[map[0]],
                    9 + map[0], channel, nonChannel, remaining, coordinate, address);
        loadValue(code, carriers, specialization, map[0], inputRepresented, values[0], address,
                floatComputation);
        arithmetic(code, resultType, values[0], values[3], Opcode.DSUB, centered,
                floatComputation);
        arithmetic(code, resultType, centered, denominator, Opcode.DDIV, standardized,
                floatComputation);
        arithmetic(code, resultType, standardized, values[1], Opcode.DMUL, scaled,
                floatComputation);
        arithmetic(code, resultType, scaled, values[2], Opcode.DADD, result,
                floatComputation);
        if (!simpleTensorPair) emitTensorAddress(code, geometry, rank, axis, extents[output], strides[output],
                    9 + output, channel, nonChannel, remaining, coordinate, address);
        int selectedOutputAddress = simpleTensorPair ? outputAddress : address;
        if (resultType == DataType.FLOAT32) carriers.storeFrozen(resultType,
                specialization.carrierPattern().get(output), output, selectedOutputAddress,
                result, false);
        else {
            if (resultType == DataType.BFLOAT16) code.fload(result).f2d().dstore(encodedStore);
            CpuNormEmitter.emitStore(code, carriers, specialization, resultType, output,
                    selectedOutputAddress, resultType == DataType.BFLOAT16 ? encodedStore : result,
                    false, true);
        }
        code.lload(nonChannel).loadConstant(1L).ladd().lstore(nonChannel);
        if (simpleTensorPair) {
            code.lload(address);
            if (denseTensorPair) code.loadConstant(1L);
            else CpuNormEmitter.geometry(code, geometry, strides[map[0]] + 2);
            code.ladd().lstore(address);
            code.lload(outputAddress);
            if (denseTensorPair) code.loadConstant(1L);
            else CpuNormEmitter.geometry(code, geometry, strides[output] + 2);
            code.ladd().lstore(outputAddress);
            code.lload(suffixCoordinate).loadConstant(1L).ladd().lstore(suffixCoordinate);
            var noWrap = code.newLabel();
            code.lload(suffixCoordinate); CpuNormEmitter.geometry(code, geometry, 5).lcmp()
                    .branch(Opcode.IFLT, noWrap);
            code.loadConstant(0L).lstore(suffixCoordinate);
            code.lload(prefixCoordinate).loadConstant(1L).ladd().lstore(prefixCoordinate);
            code.lload(address);
            if (denseTensorPair) {
                CpuNormEmitter.geometry(code, geometry, 4).loadConstant(1L).lsub();
                CpuNormEmitter.geometry(code, geometry, 5).lmul();
            } else {
                CpuNormEmitter.geometry(code, geometry, strides[map[0]]);
                CpuNormEmitter.geometry(code, geometry, 5);
                CpuNormEmitter.geometry(code, geometry, strides[map[0]] + 2).lmul().lsub();
            }
            code.ladd().lstore(address);
            code.lload(outputAddress);
            if (denseTensorPair) {
                CpuNormEmitter.geometry(code, geometry, 4).loadConstant(1L).lsub();
                CpuNormEmitter.geometry(code, geometry, 5).lmul();
            } else {
                CpuNormEmitter.geometry(code, geometry, strides[output]);
                CpuNormEmitter.geometry(code, geometry, 5);
                CpuNormEmitter.geometry(code, geometry, strides[output] + 2).lmul().lsub();
            }
            code.ladd().lstore(outputAddress);
            code.labelBinding(noWrap);
        }
        code.branch(Opcode.GOTO, loop).labelBinding(done);
    }

    private static void loadValue(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, int boundary, int represented, int target,
            int address, boolean floatComputation) {
        if (!floatComputation) {
            CpuLayerNormEmitter.load(code, carriers, specialization, boundary, represented,
                    target, address);
            return;
        }
        DataType type = specialization.boundaryDataTypes().get(boundary);
        carriers.loadFrozen(type, specialization.carrierPattern().get(boundary), boundary,
                address, false);
        if (type == DataType.FLOAT32) code.fstore(target);
        else code.loadConstant(16).ishl().invokestatic(FLOAT_CLASS, "intBitsToFloat",
                MethodTypeDesc.of(TypeKind.FLOAT.upperBound(), TypeKind.INT.upperBound()))
                .fstore(target);
    }

    private static void arithmetic(CodeBuilder code, DataType resultType, int left, int right,
            Opcode operation, int target, boolean floatComputation) {
        if (!floatComputation) {
            CpuLayerNormEmitter.arithmetic(code, resultType, left, right, operation, target);
            return;
        }
        code.fload(left).fload(right);
        switch (operation) {
            case DADD -> code.fadd(); case DSUB -> code.fsub(); case DMUL -> code.fmul();
            case DDIV -> code.fdiv(); default -> throw new IllegalArgumentException("operation");
        }
        code.fstore(target);
    }

    private static void emitDenseAddress(CodeBuilder code, int geometry, int baseIndex,
            int channel, int prefixCoordinate, int suffixCoordinate, int address) {
        CpuNormEmitter.geometry(code, geometry, baseIndex).lload(prefixCoordinate);
        CpuNormEmitter.geometry(code, geometry, 4).lmul().lload(channel).ladd();
        CpuNormEmitter.geometry(code, geometry, 5).lmul().lload(suffixCoordinate).ladd()
                .ladd().lstore(address);
    }

    private static void emitRankThreeMiddleAddress(CodeBuilder code, int geometry, int baseIndex,
            int strides, int channel, int prefixCoordinate, int suffixCoordinate, int address) {
        CpuNormEmitter.geometry(code, geometry, baseIndex).lload(prefixCoordinate);
        CpuNormEmitter.geometry(code, geometry, strides).lmul().ladd().lload(channel);
        CpuNormEmitter.geometry(code, geometry, strides + 1).lmul().ladd()
                .lload(suffixCoordinate);
        CpuNormEmitter.geometry(code, geometry, strides + 2).lmul().ladd().lstore(address);
    }

    private static void emitTensorAddress(CodeBuilder code, int geometry, int rank, int axis,
            int extents, int strides, int baseIndex, int channel, int nonChannel, int remaining,
            int coordinate, int address) {
        CpuNormEmitter.geometry(code, geometry, baseIndex).lload(channel);
        CpuNormEmitter.geometry(code, geometry, strides + axis).lmul().ladd().lstore(address);
        code.lload(nonChannel).lstore(remaining);
        for (int logicalAxis = rank - 1; logicalAxis >= 0; logicalAxis--) {
            if (logicalAxis == axis) continue;
            code.lload(remaining); CpuNormEmitter.geometry(code, geometry, extents + logicalAxis)
                    .lrem().lstore(coordinate);
            code.lload(remaining); CpuNormEmitter.geometry(code, geometry, extents + logicalAxis)
                    .ldiv().lstore(remaining);
            code.lload(address).lload(coordinate); CpuNormEmitter.geometry(code, geometry,
                    strides + logicalAxis).lmul().ladd().lstore(address);
        }
    }
}
