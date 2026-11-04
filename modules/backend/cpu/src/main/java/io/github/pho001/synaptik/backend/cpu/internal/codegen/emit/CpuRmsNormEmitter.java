package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;

/**
 * Emits one direct two-pass generated RMS-normalization body with declared square sites.
 *
 * <p>Generation freezes the form, result type, exact epsilon, normalized geometry, access plans,
 * and concrete carriers. The first pass sums working-precision squares, retaining overflow,
 * then applies count division, epsilon addition and square root; the second normalizes and stores.
 * RMS execution declares no workspace and performs no run-time semantic dispatch.</p>
 * <p>Controlled ordinary emission guards normalization slices by input provenance. Affine-introduced
 * ROW activity independently guards each result coordinate before scale loads or publication.
 * Uncontrolled emission adds no activity instructions.</p>
 */
public final class CpuRmsNormEmitter {
    /** Creates a stateless generation-time emitter. */
    public CpuRmsNormEmitter() { }
    /**
     * Emits one exact structural specialization into the supplied method body.
     *
     * @param code non-null Class-File method builder to mutate
     * @param specialization non-null exact zero-scratch carrier specialization
     * @param ir non-null instruction-free trailing-RMS kernel identity
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if family, form, boundary map, scratch, or specialization
     *     facts disagree
     */
    public void emit(CodeBuilder code, CpuKernelSpecialization specialization, CpuKernelIr ir) {
        String identity = ir.familyIdentity();
        boolean scaled = identity.startsWith("trailing-normalization:RMS:form=RMS_SCALED:");
        if (!scaled && !identity.startsWith("trailing-normalization:RMS:form=RMS:"))
            throw new IllegalArgumentException("RMS-Norm emitter requires an RMS identity");
        int[] map = CpuLayerNormEmitter.map(identity);
        if (map.length != (scaled ? 2 : 1))
            throw new IllegalArgumentException("RMS-Norm semantic boundary map disagrees");
        int boundaries = specialization.carrierPattern().size();
        if (boundaries <= 1 || specialization.boundaryDataTypes().size() != boundaries
                || specialization.scratchParameter())
            throw new IllegalArgumentException("RMS-Norm specialization disagrees");
        DataType resultType = specialization.boundaryDataTypes().getLast();
        int normalizedRank = Integer.parseInt(CpuLayerNormEmitter.field(identity, ":rank="));
        int geometry = boundaries, start = geometry + 1, end = start + 2;
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
        int representedInput = CpuLayerNormEmitter.represented(code,
                specialization.boundaryDataTypes().get(inputBoundary));
        int value = code.allocateLocal(TypeKind.DOUBLE);
        int compensation = code.allocateLocal(TypeKind.DOUBLE), squares = code.allocateLocal(TypeKind.DOUBLE);
        int temporary = code.allocateLocal(TypeKind.DOUBLE);
        int root = code.allocateLocal(TypeKind.DOUBLE), result = code.allocateLocal(TypeKind.DOUBLE);
        int nan = code.allocateLocal(TypeKind.INT), infinity = code.allocateLocal(TypeKind.INT);
        long domain = Long.parseLong(CpuLayerNormEmitter.field(identity, ":domain="));

        code.lload(start).lstore(slice);
        var slices = code.newLabel(); var done = code.newLabel();
        code.labelBinding(slices).lload(slice).lload(end).lcmp().branch(Opcode.IFGE, done);
        var advance = code.newLabel();
        boolean[] omitted = new boolean[rank[inputBoundary]];
        java.util.Arrays.fill(omitted, omitted.length - normalizedRank, omitted.length, true);
        CpuLoopEmitter.emitActiveSliceGuard(code, inputBoundary, slice, TypeKind.LONG,
                geometry, extent[inputBoundary], omitted, advance);
        CpuLayerNormEmitter.emitSliceBases(code, geometry, slice, normalizedRank, rank, extent,
                stride, bases, remaining, coordinate);
        code.loadConstant(0).istore(nan).loadConstant(0).istore(infinity)
                .loadConstant(0.0).dstore(compensation).loadConstant(0.0).dstore(squares);
        CpuLayerNormEmitter.emitLoop(code, geometry, normalizedRank, rank[inputBoundary], normalizedExtents,
                hotStride[inputBoundary], bases[inputBoundary], ordinal, remaining, coordinate, address,
                () -> {
                    CpuLayerNormEmitter.load(code, carriers, specialization, inputBoundary,
                            representedInput, value, address);
                    CpuNormEmitter.classify(code, value, nan, -1, infinity);
                    CpuNormEmitter.arithmetic(code, resultType, value, value, Opcode.DMUL, value);
                    CpuNormEmitter.kahanSquares(code, resultType, value, squares, compensation, temporary);
                });
        int count = code.allocateLocal(TypeKind.LONG);
        code.loadConstant(domain).lstore(count);
        CpuNormEmitter.divideByCount(code, resultType, squares, count);
        code.loadConstant(CpuLayerNormEmitter.epsilon(identity, resultType)).dstore(temporary);
        CpuNormEmitter.arithmetic(code, resultType, squares, temporary, Opcode.DADD, root);
        CpuNormEmitter.unary(code, resultType, "sqrt", root, root);
        var noNan = code.newLabel(); var noInfinity = code.newLabel(); var specialDone = code.newLabel();
        code.iload(nan).branch(Opcode.IFEQ, noNan)
                .loadConstant(Double.longBitsToDouble(0x7ff8000000000000L)).dstore(root)
                .branch(Opcode.GOTO, specialDone).labelBinding(noNan);
        code.iload(infinity).branch(Opcode.IFEQ, noInfinity)
                .loadConstant(Double.POSITIVE_INFINITY).dstore(root)
                .branch(Opcode.GOTO, specialDone).labelBinding(noInfinity).labelBinding(specialDone);

        int scaleRepresented = scaled ? CpuLayerNormEmitter.represented(code,
                specialization.boundaryDataTypes().get(map[1])) : -1;
        int scaleValue = scaled ? code.allocateLocal(TypeKind.DOUBLE) : -1;
        CpuLayerNormEmitter.emitLoop(code, geometry, normalizedRank, rank[inputBoundary], normalizedExtents,
                hotStride[inputBoundary], bases[inputBoundary], ordinal, remaining, coordinate, address,
                () -> {
                    var nextResult = code.newLabel();
                    CpuLayerNormEmitter.emitResultGuard(code, inputBoundary, boundaries - 1, slice,
                            ordinal, domain, nextResult);
                    CpuLayerNormEmitter.load(code, carriers, specialization, inputBoundary,
                            representedInput, value, address);
                    CpuLayerNormEmitter.arithmetic(code, resultType, value, root, Opcode.DDIV, result);
                    if (scaled) {
                        CpuLayerNormEmitter.emitAddress(code, geometry, normalizedRank, rank[map[1]],
                                normalizedExtents, hotStride[map[1]], bases[map[1]], ordinal, remaining,
                                coordinate, address);
                        CpuLayerNormEmitter.load(code, carriers, specialization, map[1],
                                scaleRepresented, scaleValue, address);
                        CpuLayerNormEmitter.arithmetic(code, resultType, result, scaleValue,
                                Opcode.DMUL, result);
                    }
                    CpuLayerNormEmitter.emitAddress(code, geometry, normalizedRank,
                            rank[boundaries - 1], normalizedExtents, hotStride[boundaries - 1],
                            bases[boundaries - 1], ordinal, remaining, coordinate, address);
                    CpuNormEmitter.emitStore(code, carriers, specialization, resultType,
                            boundaries - 1, address, result, false, true);
                    code.labelBinding(nextResult);
                });
        code.labelBinding(advance);
        code.lload(slice).loadConstant(1L).ladd().lstore(slice)
                .branch(Opcode.GOTO, slices).labelBinding(done);
    }
}
