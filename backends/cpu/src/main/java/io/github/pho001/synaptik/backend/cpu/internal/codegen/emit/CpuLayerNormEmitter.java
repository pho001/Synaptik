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
 * Emits one direct three-pass generated Layer-normalization body.
 *
 * <p>Generation freezes the form, result type, exact epsilon, normalized geometry, access plans,
 * and concrete carriers. The emitted entry obtains the exact mean from its assigned per-range
 * state, forms population variance with compensated primitive sums, and performs one final store
 * per logical result position. This emitter owns no run-time semantic dispatch or workspace.</p>
 */
public final class CpuLayerNormEmitter {
    private static final ClassDesc DOUBLE = ClassDesc.of(Double.class.getName());
    private static final ClassDesc MATH = ClassDesc.of(Math.class.getName());
    /** Creates a stateless generation-time emitter. */
    public CpuLayerNormEmitter() { }
    /**
     * Emits one exact structural specialization into the supplied method body.
     *
     * @param code non-null Class-File method builder to mutate
     * @param specialization non-null exact carrier and optional Layer scratch specialization
     * @param ir non-null instruction-free trailing-Layer kernel identity
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if family, form, boundary map, scratch, or specialization
     *     facts disagree
     */
    public void emit(CodeBuilder code, CpuKernelSpecialization specialization, CpuKernelIr ir) {
        String identity = ir.familyIdentity();
        boolean affine = identity.startsWith("trailing-normalization:LAYER:form=LAYER_AFFINE:");
        if (!affine && !identity.startsWith("trailing-normalization:LAYER:form=LAYER:"))
            throw new IllegalArgumentException("Layer-Norm emitter requires a Layer identity");
        int[] map = map(identity);
        if (map.length != (affine ? 3 : 1))
            throw new IllegalArgumentException("Layer-Norm semantic boundary map disagrees");
        int boundaries = specialization.carrierPattern().size();
        boolean empty = Long.parseLong(field(identity, ":slice=")) == 0;
        if (boundaries <= 1 || specialization.boundaryDataTypes().size() != boundaries
                || specialization.scratchParameter() == empty)
            throw new IllegalArgumentException("Layer-Norm specialization disagrees");
        if (empty) return;
        DataType resultType = specialization.boundaryDataTypes().getLast();
        int normalizedRank = Integer.parseInt(field(identity, ":rank="));
        int scratch = boundaries, geometry = boundaries + 1;
        int start = geometry + 1, end = start + 2;
        int[] rank = new int[boundaries], extent = new int[boundaries], stride = new int[boundaries];
        int cursor = 11 + boundaries;
        for (int boundary = 0; boundary < boundaries; boundary++) {
            rank[boundary] = ir.values().get(boundary).accessPlan().iterationRank();
            extent[boundary] = cursor + 1; stride[boundary] = extent[boundary] + rank[boundary];
            cursor += 1 + 2 * rank[boundary];
        }
        int normalizedExtents = extent[0] + rank[0] - normalizedRank;
        int[] hotStride = stride.clone();
        for (int boundary = 0; boundary < boundaries; boundary++)
            if (ir.values().get(boundary).accessPlan().contiguousSuffix() >= normalizedRank)
                hotStride[boundary] = -1;
        var carriers = new CpuCarrierEmitter(code);
        int slice = code.allocateLocal(TypeKind.LONG), ordinal = code.allocateLocal(TypeKind.LONG);
        int remaining = code.allocateLocal(TypeKind.LONG), coordinate = code.allocateLocal(TypeKind.LONG);
        int address = code.allocateLocal(TypeKind.LONG);
        int[] bases = new int[boundaries];
        for (int i = 0; i < boundaries; i++) bases[i] = code.allocateLocal(TypeKind.LONG);
        int inputBoundary = map[0];
        int representedInput = represented(code, specialization.boundaryDataTypes().get(inputBoundary));
        int value = code.allocateLocal(TypeKind.DOUBLE);
        DataType exactType = resultType == DataType.FLOAT64 ? DataType.FLOAT64 : DataType.FLOAT32;
        int meanRepresented = code.allocateLocal(exactType == DataType.FLOAT64
                ? TypeKind.DOUBLE : TypeKind.FLOAT);
        int mean = code.allocateLocal(TypeKind.DOUBLE), deviation = code.allocateLocal(TypeKind.DOUBLE);
        int deviationSum = code.allocateLocal(TypeKind.DOUBLE);
        int deviationCompensation = code.allocateLocal(TypeKind.DOUBLE);
        int square = code.allocateLocal(TypeKind.DOUBLE), squareSum = code.allocateLocal(TypeKind.DOUBLE);
        int squareCompensation = code.allocateLocal(TypeKind.DOUBLE);
        int temporary = code.allocateLocal(TypeKind.DOUBLE), variance = code.allocateLocal(TypeKind.DOUBLE);
        int root = code.allocateLocal(TypeKind.DOUBLE), result = code.allocateLocal(TypeKind.DOUBLE);
        int nan = code.allocateLocal(TypeKind.INT), infinity = code.allocateLocal(TypeKind.INT);
        int seen = code.allocateLocal(TypeKind.INT), constant = code.allocateLocal(TypeKind.INT);
        int first = code.allocateLocal(TypeKind.DOUBLE);
        long domain = Long.parseLong(field(identity, ":domain="));
        CpuExactSumEmitter exact = new CpuExactSumEmitter(code, exactType, true, scratch, geometry,
                Integer.parseInt(field(identity, ":limbs=")));

        code.lload(start).lstore(slice);
        var slices = code.newLabel(); var done = code.newLabel();
        code.labelBinding(slices).lload(slice).lload(end).lcmp().branch(Opcode.IFGE, done);
        emitSliceBases(code, geometry, slice, normalizedRank, rank, extent, stride, bases,
                remaining, coordinate);
        code.loadConstant(0).istore(nan).loadConstant(0).istore(infinity)
                .loadConstant(0).istore(seen).loadConstant(1).istore(constant)
                .loadConstant(0.0).dstore(first);
        exact.emitReset();
        emitLoop(code, geometry, normalizedRank, rank[inputBoundary], normalizedExtents, hotStride[inputBoundary],
                bases[inputBoundary], ordinal, remaining, coordinate, address, () -> {
                    load(code, carriers, specialization, inputBoundary, representedInput, value, address);
                    CpuNormEmitter.classify(code, value, nan, -1, infinity);
                    var notNan = code.newLabel(); var classified = code.newLabel();
                    code.dload(value).invokestatic(DOUBLE, "isNaN", MethodTypeDesc.of(
                            TypeKind.BOOLEAN.upperBound(), TypeKind.DOUBLE.upperBound()))
                            .branch(Opcode.IFEQ, notNan).branch(Opcode.GOTO, classified)
                            .labelBinding(notNan);
                    var compare = code.newLabel(); var retained = code.newLabel();
                    code.iload(seen).branch(Opcode.IFNE, compare).dload(value).dstore(first)
                            .loadConstant(1).istore(seen).branch(Opcode.GOTO, retained)
                            .labelBinding(compare).dload(value).dload(first).dcmpl()
                            .branch(Opcode.IFEQ, retained).loadConstant(0).istore(constant)
                            .labelBinding(retained).labelBinding(classified);
                    if (exactType == DataType.FLOAT64) code.dload(value).dstore(meanRepresented);
                    else code.dload(value).d2f().fstore(meanRepresented);
                    exact.emitFactor(meanRepresented);
                });
        exact.emitFinish(meanRepresented);
        CpuNormEmitter.decodeRepresented(code, exactType, meanRepresented, mean);
        code.loadConstant(0.0).dstore(deviationSum).loadConstant(0.0)
                .dstore(deviationCompensation).loadConstant(0.0).dstore(squareSum)
                .loadConstant(0.0).dstore(squareCompensation);
        var skipSecondPass = code.newLabel();
        code.iload(nan).iload(infinity).ior().iload(constant).ior()
                .branch(Opcode.IFNE, skipSecondPass);
        emitLoop(code, geometry, normalizedRank, rank[inputBoundary], normalizedExtents, hotStride[inputBoundary],
                bases[inputBoundary], ordinal, remaining, coordinate, address, () -> {
                    load(code, carriers, specialization, inputBoundary, representedInput, value, address);
                    code.dload(value).dload(mean).dsub().dstore(deviation);
                    code.dload(deviation).dstore(temporary);
                    CpuNormEmitter.kahan(code, temporary, deviationSum, deviationCompensation, square);
                    code.dload(deviation).dload(deviation).dmul().dstore(square);
                    CpuNormEmitter.kahan(code, square, squareSum, squareCompensation, temporary);
                });
        code.labelBinding(skipSecondPass);
        code.dload(squareSum).dload(deviationSum).dload(deviationSum).dmul()
                .loadConstant((double) domain).ddiv().dsub().dstore(variance);
        var nonnegative = code.newLabel();
        code.dload(variance).loadConstant(0.0).dcmpl().branch(Opcode.IFGE, nonnegative)
                .loadConstant(0.0).dstore(variance).labelBinding(nonnegative);
        code.dload(variance).loadConstant((double) domain).ddiv()
                .loadConstant(epsilon(identity, resultType)).dadd().invokestatic(MATH, "sqrt",
                        CpuNormEmitter.doubleUnary()).dstore(root);

        int scaleRepresented = affine ? represented(code,
                specialization.boundaryDataTypes().get(map[1])) : -1;
        int biasRepresented = affine ? represented(code,
                specialization.boundaryDataTypes().get(map[2])) : -1;
        int scaleValue = affine ? code.allocateLocal(TypeKind.DOUBLE) : -1;
        int biasValue = affine ? code.allocateLocal(TypeKind.DOUBLE) : -1;
        emitLoop(code, geometry, normalizedRank, rank[inputBoundary], normalizedExtents, hotStride[inputBoundary],
                bases[inputBoundary], ordinal, remaining, coordinate, address, () -> {
                    load(code, carriers, specialization, inputBoundary, representedInput, value, address);
                    var ordinary = code.newLabel(); var standardized = code.newLabel();
                    code.iload(nan).iload(infinity).ior().branch(Opcode.IFEQ, ordinary)
                            .loadConstant(Double.longBitsToDouble(0x7ff8000000000000L)).dstore(result)
                            .branch(Opcode.GOTO, standardized).labelBinding(ordinary);
                    var nonconstant = code.newLabel();
                    code.iload(constant).branch(Opcode.IFEQ, nonconstant)
                            .loadConstant(0.0).dstore(result).branch(Opcode.GOTO, standardized)
                            .labelBinding(nonconstant);
                    arithmetic(code, resultType, value, mean, Opcode.DSUB, result);
                    arithmetic(code, resultType, result, root, Opcode.DDIV, result);
                    code.labelBinding(standardized);
                    if (affine) {
                        emitAddress(code, geometry, normalizedRank, rank[map[1]], normalizedExtents,
                                hotStride[map[1]], bases[map[1]], ordinal, remaining, coordinate, address);
                        load(code, carriers, specialization, map[1], scaleRepresented, scaleValue, address);
                        arithmetic(code, resultType, result, scaleValue, Opcode.DMUL, result);
                        emitAddress(code, geometry, normalizedRank, rank[map[2]], normalizedExtents,
                                hotStride[map[2]], bases[map[2]], ordinal, remaining, coordinate, address);
                        load(code, carriers, specialization, map[2], biasRepresented, biasValue, address);
                        arithmetic(code, resultType, result, biasValue, Opcode.DADD, result);
                    }
                    emitAddress(code, geometry, normalizedRank, rank[boundaries - 1], normalizedExtents,
                            hotStride[boundaries - 1], bases[boundaries - 1], ordinal, remaining,
                            coordinate, address);
                    CpuNormEmitter.emitStore(code, carriers, specialization, resultType,
                            boundaries - 1, address, result, false, true);
                });
        code.lload(slice).loadConstant(1L).ladd().lstore(slice)
                .branch(Opcode.GOTO, slices).labelBinding(done);
    }

    /**
     * Emits leading-slice decoding and derives each boundary's base address.
     *
     * @param code non-null method builder to mutate
     * @param geometry local containing packed primitive geometry
     * @param slice local containing the leading-slice ordinal
     * @param normalizedRank positive trailing normalized rank
     * @param ranks per-boundary logical ranks
     * @param extents per-boundary packed extent offsets
     * @param strides per-boundary packed stride offsets
     * @param bases per-boundary address locals to initialize
     * @param remaining reusable long quotient local
     * @param coordinate reusable long coordinate local
     */
    static void emitSliceBases(CodeBuilder code, int geometry, int slice, int normalizedRank,
            int[] ranks, int[] extents, int[] strides, int[] bases, int remaining, int coordinate) {
        for (int boundary = 0; boundary < ranks.length; boundary++)
            CpuNormEmitter.geometry(code, geometry, 11 + boundary).lstore(bases[boundary]);
        code.lload(slice).lstore(remaining);
        for (int axis = ranks[0] - normalizedRank - 1; axis >= 0; axis--) {
            code.lload(remaining); CpuNormEmitter.geometry(code, geometry, extents[0] + axis)
                    .lrem().lstore(coordinate);
            code.lload(remaining); CpuNormEmitter.geometry(code, geometry, extents[0] + axis)
                    .ldiv().lstore(remaining);
            for (int boundary = 0; boundary < ranks.length; boundary++) {
                if (ranks[boundary] <= normalizedRank) continue;
                code.lload(bases[boundary]).lload(coordinate); CpuNormEmitter.geometry(code,
                        geometry, strides[boundary] + axis).lmul().ladd().lstore(bases[boundary]);
            }
        }
    }

    /**
     * Emits one canonical complete-normalized-slice loop.
     *
     * @param code non-null method builder to mutate
     * @param geometry local containing packed primitive geometry
     * @param normalizedRank positive trailing normalized rank
     * @param rank logical rank of the traversed boundary
     * @param normalizedExtents packed normalized-extent offset
     * @param strides packed stride offset, or {@code -1} for proved contiguous traversal
     * @param base local containing the boundary's leading-slice base
     * @param ordinal reusable normalized ordinal local
     * @param remaining reusable long quotient local
     * @param coordinate reusable long coordinate local
     * @param address target address local
     * @param body non-null emission callback invoked once while building the loop body
     */
    static void emitLoop(CodeBuilder code, int geometry, int normalizedRank, int rank,
            int normalizedExtents, int strides, int base, int ordinal, int remaining,
            int coordinate, int address, Runnable body) {
        code.loadConstant(0L).lstore(ordinal);
        var loop = code.newLabel(); var finish = code.newLabel();
        code.labelBinding(loop).lload(ordinal); CpuNormEmitter.geometry(code, geometry, 2).lcmp()
                .branch(Opcode.IFGE, finish);
        emitAddress(code, geometry, normalizedRank, rank, normalizedExtents, strides, base,
                ordinal, remaining, coordinate, address);
        body.run();
        code.lload(ordinal).loadConstant(1L).ladd().lstore(ordinal)
                .branch(Opcode.GOTO, loop).labelBinding(finish);
    }

    /**
     * Emits address derivation for one normalized ordinal.
     *
     * @param code non-null method builder to mutate
     * @param geometry local containing packed primitive geometry
     * @param normalizedRank positive trailing normalized rank
     * @param rank logical boundary rank
     * @param normalizedExtents packed normalized-extent offset
     * @param strides packed stride offset, or {@code -1} for proved contiguous traversal
     * @param base local containing the boundary base
     * @param ordinal local containing the normalized ordinal
     * @param remaining reusable long quotient local
     * @param coordinate reusable long coordinate local
     * @param address target address local
     */
    static void emitAddress(CodeBuilder code, int geometry, int normalizedRank, int rank,
            int normalizedExtents, int strides, int base, int ordinal, int remaining,
            int coordinate, int address) {
        if (strides < 0) {
            code.lload(base).lload(ordinal).ladd().lstore(address);
            return;
        }
        code.lload(base).lstore(address).lload(ordinal).lstore(remaining);
        for (int suffix = normalizedRank - 1; suffix >= 0; suffix--) {
            int axis = rank - normalizedRank + suffix;
            code.lload(remaining); CpuNormEmitter.geometry(code, geometry,
                    normalizedExtents + suffix).lrem().lstore(coordinate);
            code.lload(remaining); CpuNormEmitter.geometry(code, geometry,
                    normalizedExtents + suffix).ldiv().lstore(remaining);
            code.lload(address).lload(coordinate); CpuNormEmitter.geometry(code, geometry,
                    strides + axis).lmul().ladd().lstore(address);
        }
    }

    /**
     * Allocates a local matching one represented floating boundary type.
     *
     * @param code non-null method builder to mutate
     * @param type BFLOAT16, FLOAT32, or FLOAT64 represented type
     * @return allocated local index
     */
    static int represented(CodeBuilder code, DataType type) {
        return code.allocateLocal(type == DataType.FLOAT64 ? TypeKind.DOUBLE
                : type == DataType.FLOAT32 ? TypeKind.FLOAT : TypeKind.INT);
    }

    /**
     * Emits one frozen-layout typed load and decodes it to a binary64 working local.
     *
     * @param code non-null method builder to mutate
     * @param carriers non-null carrier emitter bound to {@code code}
     * @param specialization non-null exact carrier/type specialization
     * @param boundary zero-based input boundary
     * @param represented local for the represented primitive value
     * @param value binary64 working local to receive the decoded value
     * @param address long element-address local
     */
    static void load(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, int boundary, int represented, int value,
            int address) {
        DataType type = specialization.boundaryDataTypes().get(boundary);
        carriers.loadFrozen(type, specialization.carrierPattern().get(boundary), boundary,
                address, false);
        if (type == DataType.FLOAT64) code.dstore(represented);
        else if (type == DataType.FLOAT32) code.fstore(represented);
        else code.istore(represented);
        CpuNormEmitter.decodeRepresented(code, type, represented, value);
    }

    /**
     * Emits one result-format arithmetic boundary without fusing adjacent operations.
     *
     * @param code non-null method builder to mutate
     * @param type FLOAT64 for binary64 arithmetic, otherwise FLOAT32 arithmetic
     * @param left binary64 local containing the left operand
     * @param right binary64 local containing the right operand
     * @param operation one of {@code DADD}, {@code DSUB}, {@code DMUL}, or {@code DDIV}
     * @param target binary64 local to receive the represented result
     * @throws IllegalArgumentException if {@code operation} is unsupported
     */
    static void arithmetic(CodeBuilder code, DataType type, int left, int right, Opcode operation,
            int target) {
        if (type == DataType.FLOAT64) {
            code.dload(left).dload(right);
            switch (operation) { case DADD -> code.dadd(); case DSUB -> code.dsub();
                case DMUL -> code.dmul(); case DDIV -> code.ddiv();
                default -> throw new IllegalArgumentException("operation"); }
            code.dstore(target); return;
        }
        Opcode floatOperation = switch (operation) {
            case DADD -> Opcode.FADD; case DSUB -> Opcode.FSUB; case DMUL -> Opcode.FMUL;
            case DDIV -> Opcode.FDIV; default -> throw new IllegalArgumentException("operation");
        };
        code.dload(left).d2f().dload(right).d2f();
        switch (floatOperation) { case FADD -> code.fadd(); case FSUB -> code.fsub();
            case FMUL -> code.fmul(); case FDIV -> code.fdiv();
            default -> throw new IllegalArgumentException("operation"); }
        code.f2d().dstore(target);
    }

    /**
     * Decodes the immutable semantic-position boundary map.
     *
     * @param identity non-null canonical trailing-normalization identity
     * @return a new boundary-map array
     */
    static int[] map(String identity) {
        int begin = identity.indexOf(":map=[") + 6, end = identity.indexOf(']', begin);
        String body = identity.substring(begin, end); if (body.isEmpty()) return new int[0];
        String[] entries = body.split(", "); int[] result = new int[entries.length];
        for (int i = 0; i < entries.length; i++) result[i] = Integer.parseInt(entries[i]);
        return result;
    }

    /**
     * Extracts one colon-delimited field from canonical structural identity.
     *
     * @param identity non-null canonical identity
     * @param marker non-null field marker including its leading colon and equals sign
     * @return the field text; never {@code null}
     */
    static String field(String identity, String marker) {
        int begin = identity.indexOf(marker) + marker.length(), end = identity.indexOf(':', begin);
        return identity.substring(begin, end < 0 ? identity.length() : end);
    }

    /**
     * Decodes exact typed epsilon bits from structural identity.
     *
     * @param identity non-null canonical trailing-normalization identity
     * @param type exact BFLOAT16, FLOAT32, or FLOAT64 result type
     * @return positive finite epsilon widened exactly to binary64
     * @throws IllegalArgumentException if {@code type} is not a supported floating type
     */
    static double epsilon(String identity, DataType type) {
        long bits = Long.parseUnsignedLong(field(identity, ":epsilon="));
        return switch (type) {
            case FLOAT64 -> Double.longBitsToDouble(bits);
            case FLOAT32 -> Float.intBitsToFloat((int) bits);
            case BFLOAT16 -> Float.intBitsToFloat((int) bits << 16);
            default -> throw new IllegalArgumentException("normalization result must be floating");
        };
    }
}
