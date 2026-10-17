package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;

/**
 * Emits one direct two-pass generated RMS-normalization body with scaled-square root formation.
 *
 * <p>Generation freezes the form, result type, exact epsilon, normalized geometry, access plans,
 * and concrete carriers. The first emitted pass derives an overflow-resistant uncentered root;
 * the second normalizes, optionally scales, and stores each result once. RMS execution declares
 * no workspace and performs no run-time semantic dispatch.</p>
 */
public final class CpuRmsNormEmitter {
    private static final ClassDesc MATH = ClassDesc.of(Math.class.getName());
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
        int value = code.allocateLocal(TypeKind.DOUBLE), absolute = code.allocateLocal(TypeKind.DOUBLE);
        int scale = code.allocateLocal(TypeKind.DOUBLE), squares = code.allocateLocal(TypeKind.DOUBLE);
        int ratio = code.allocateLocal(TypeKind.DOUBLE), rms = code.allocateLocal(TypeKind.DOUBLE);
        int root = code.allocateLocal(TypeKind.DOUBLE), result = code.allocateLocal(TypeKind.DOUBLE);
        int nan = code.allocateLocal(TypeKind.INT), infinity = code.allocateLocal(TypeKind.INT);
        long domain = Long.parseLong(CpuLayerNormEmitter.field(identity, ":domain="));

        code.lload(start).lstore(slice);
        var slices = code.newLabel(); var done = code.newLabel();
        code.labelBinding(slices).lload(slice).lload(end).lcmp().branch(Opcode.IFGE, done);
        CpuLayerNormEmitter.emitSliceBases(code, geometry, slice, normalizedRank, rank, extent,
                stride, bases, remaining, coordinate);
        code.loadConstant(0).istore(nan).loadConstant(0).istore(infinity)
                .loadConstant(0.0).dstore(scale).loadConstant(0.0).dstore(squares);
        CpuLayerNormEmitter.emitLoop(code, geometry, normalizedRank, rank[inputBoundary], normalizedExtents,
                hotStride[inputBoundary], bases[inputBoundary], ordinal, remaining, coordinate, address,
                () -> {
                    CpuLayerNormEmitter.load(code, carriers, specialization, inputBoundary,
                            representedInput, value, address);
                    CpuNormEmitter.classify(code, value, nan, -1, infinity);
                    code.dload(value).invokestatic(MATH, "abs", CpuNormEmitter.doubleUnary())
                            .dstore(absolute);
                    var zero = code.newLabel(); var smaller = code.newLabel(); var complete = code.newLabel();
                    code.dload(absolute).loadConstant(0.0).dcmpl().branch(Opcode.IFEQ, zero);
                    code.dload(scale).dload(absolute).dcmpl().branch(Opcode.IFGE, smaller);
                    code.dload(scale).dload(absolute).ddiv().dstore(ratio);
                    code.loadConstant(1.0).dload(squares).dload(ratio).dload(ratio).dmul()
                            .dmul().dadd().dstore(squares).dload(absolute).dstore(scale)
                            .branch(Opcode.GOTO, complete).labelBinding(smaller);
                    code.dload(absolute).dload(scale).ddiv().dstore(ratio);
                    code.dload(squares).dload(ratio).dload(ratio).dmul().dadd().dstore(squares)
                            .branch(Opcode.GOTO, complete).labelBinding(zero).labelBinding(complete);
                });
        code.dload(squares).loadConstant((double) domain).ddiv()
                .invokestatic(MATH, "sqrt", CpuNormEmitter.doubleUnary())
                .dload(scale).dmul().dstore(rms);
        code.loadConstant(CpuLayerNormEmitter.epsilon(identity, resultType))
                .invokestatic(MATH, "sqrt", CpuNormEmitter.doubleUnary()).dstore(ratio);
        code.dload(rms).dload(ratio).invokestatic(MATH, "hypot",
                java.lang.constant.MethodTypeDesc.of(TypeKind.DOUBLE.upperBound(),
                        TypeKind.DOUBLE.upperBound(), TypeKind.DOUBLE.upperBound())).dstore(root);
        if (resultType != DataType.FLOAT64) code.dload(root).d2f().f2d().dstore(root);
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
                });
        code.lload(slice).loadConstant(1L).ladd().lstore(slice)
                .branch(Opcode.GOTO, slices).labelBinding(done);
    }
}
