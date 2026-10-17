package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;

/** Emits direct corrected two-pass variance and standard-deviation bodies. */
public final class CpuStatisticalReductionEmitter {
    private static final ClassDesc MATH = ClassDesc.of(Math.class.getName());
    /** Creates a stateless generation-time emitter. */
    public CpuStatisticalReductionEmitter() { }
    /**
     * Emits one direct typed entry body.
     * @param code non-null Class-File method builder to mutate
     * @param specialization non-null exact carrier and scratch specialization
     * @param ir non-null canonical variance or standard-deviation identity
     */
    public void emit(CodeBuilder code, CpuKernelSpecialization specialization, CpuKernelIr ir) {
        boolean standardDeviation = ir.familyIdentity().startsWith(
                "advanced-reduction:STANDARD_DEVIATION:");
        if (!standardDeviation && !ir.familyIdentity().startsWith(
                "advanced-reduction:VARIANCE:") || specialization.carrierPattern().size() != 2
                || specialization.boundaryDataTypes().size() != 2
                || specialization.boundaryDataTypes().getFirst()
                    != specialization.boundaryDataTypes().getLast()
                || !specialization.scratchParameter()) {
            throw new IllegalArgumentException("statistics require matching boundaries and exact-state scratch");
        }
        DataType type = specialization.boundaryDataTypes().getFirst();
        int inputRank = ir.values().getFirst().accessPlan().iterationRank();
        int outputRank = ir.values().getLast().accessPlan().iterationRank();
        boolean keep = ir.familyIdentity().contains(":keep=true:");
        boolean[] selected = CpuNormEmitter.selected(ir.familyIdentity(), inputRank);
        boolean contiguous = CpuNormEmitter.contiguousSelected(selected,
                ir.values().getFirst().accessPlan().contiguousSuffix());
        long correction = CpuNormEmitter.longAfter(ir.familyIdentity(), ":correction=");
        int scratch = 2, geometry = 3, start = 4, end = 6;
        int inputExtents = 11, inputStrides = inputExtents + inputRank;
        int outputExtents = inputStrides + inputRank;
        int outputStrides = outputExtents + outputRank;
        var carriers = new CpuCarrierEmitter(code);
        int cell = code.allocateLocal(TypeKind.LONG);
        int remaining = code.allocateLocal(TypeKind.LONG);
        int coordinate = code.allocateLocal(TypeKind.LONG);
        int inputBase = code.allocateLocal(TypeKind.LONG);
        int outputAddress = code.allocateLocal(TypeKind.LONG);
        int domain = code.allocateLocal(TypeKind.LONG);
        int address = code.allocateLocal(TypeKind.LONG);
        int represented = code.allocateLocal(type == DataType.FLOAT64 ? TypeKind.DOUBLE
                : type == DataType.FLOAT32 ? TypeKind.FLOAT : TypeKind.INT);
        int value = code.allocateLocal(TypeKind.DOUBLE);
        int result = code.allocateLocal(TypeKind.DOUBLE);
        int mean = code.allocateLocal(TypeKind.DOUBLE);
        int deviations = code.allocateLocal(TypeKind.DOUBLE);
        int deviationCompensation = code.allocateLocal(TypeKind.DOUBLE);
        int squares = code.allocateLocal(TypeKind.DOUBLE);
        int squareCompensation = code.allocateLocal(TypeKind.DOUBLE);
        int temporary = code.allocateLocal(TypeKind.DOUBLE);
        int nan = code.allocateLocal(TypeKind.INT);
        int anyInfinity = code.allocateLocal(TypeKind.INT);
        var exact = new CpuExactSumEmitter(code, type, true, scratch, geometry,
                Math.toIntExact(CpuNormEmitter.longAfter(ir.familyIdentity(), ":slice=")
                        / Long.BYTES - 1));

        code.lload(start).lstore(cell);
        var cells = code.newLabel(); var done = code.newLabel();
        code.labelBinding(cells).lload(cell).lload(end).lcmp().branch(Opcode.IFGE, done);
        CpuNormEmitter.geometry(code, geometry, 0).lstore(inputBase);
        CpuNormEmitter.geometry(code, geometry, 1).lstore(outputAddress);
        code.lload(cell).lstore(remaining);
        int[] outputCoordinates = new int[outputRank];
        for (int axis = outputRank - 1; axis >= 0; axis--) {
            int local = code.allocateLocal(TypeKind.LONG); outputCoordinates[axis] = local;
            if (axis == 0) code.lload(remaining).lstore(local);
            else {
                code.lload(remaining); CpuNormEmitter.geometry(code, geometry,
                        outputExtents + axis).lrem().lstore(local);
                code.lload(remaining); CpuNormEmitter.geometry(code, geometry,
                        outputExtents + axis).ldiv().lstore(remaining);
            }
            code.lload(outputAddress).lload(local);
            CpuNormEmitter.geometry(code, geometry, outputStrides + axis).lmul().ladd()
                    .lstore(outputAddress);
        }
        for (int inputAxis = 0, outputAxis = 0; inputAxis < inputRank; inputAxis++) {
            if (selected[inputAxis]) { if (keep) outputAxis++; continue; }
            code.lload(inputBase).lload(outputCoordinates[outputAxis++]);
            CpuNormEmitter.geometry(code, geometry, inputStrides + inputAxis).lmul().ladd()
                    .lstore(inputBase);
        }
        int[] selectedCoordinates = new int[inputRank];
        for (int axis = 0; axis < inputRank; axis++) if (selected[axis])
            selectedCoordinates[axis] = code.allocateLocal(TypeKind.LONG);
        code.loadConstant(0).istore(nan).loadConstant(0).istore(anyInfinity);
        exact.emitReset();
        CpuNormEmitter.emitPass(code, carriers, specialization, type, selected, inputRank, geometry,
                inputExtents, inputStrides, inputBase, domain, remaining, coordinate, address,
                selectedCoordinates, contiguous, represented, value, () -> {
                    CpuNormEmitter.classify(code, value, nan, -1, anyInfinity);
                    exact.emitFactor(represented);
                });
        exact.emitFinish(represented);
        CpuNormEmitter.decodeRepresented(code, type, represented, mean);
        code.loadConstant(0.0).dstore(deviations).loadConstant(0.0)
                .dstore(deviationCompensation).loadConstant(0.0).dstore(squares)
                .loadConstant(0.0).dstore(squareCompensation);
        CpuNormEmitter.emitPass(code, carriers, specialization, type, selected, inputRank, geometry,
                inputExtents, inputStrides, inputBase, domain, remaining, coordinate, address,
                selectedCoordinates, contiguous, represented, value, () -> {
                    code.dload(value).dload(mean).dsub().dstore(value);
                    CpuNormEmitter.kahan(code, value, deviations, deviationCompensation, temporary);
                    code.dload(value).dload(value).dmul().dstore(value);
                    var finiteSquare = code.newLabel(); var squareDone = code.newLabel();
                    code.dload(value).loadConstant(Double.POSITIVE_INFINITY).dcmpl()
                            .branch(Opcode.IFNE, finiteSquare)
                            .loadConstant(Double.POSITIVE_INFINITY).dstore(squares)
                            .loadConstant(0.0).dstore(squareCompensation)
                            .branch(Opcode.GOTO, squareDone).labelBinding(finiteSquare);
                    CpuNormEmitter.kahan(code, value, squares, squareCompensation, temporary);
                    code.labelBinding(squareDone);
                });
        var finiteSquares = code.newLabel(); var statisticDone = code.newLabel();
        code.dload(squares).loadConstant(Double.POSITIVE_INFINITY).dcmpl()
                .branch(Opcode.IFNE, finiteSquares).loadConstant(Double.POSITIVE_INFINITY)
                .dstore(result).branch(Opcode.GOTO, statisticDone).labelBinding(finiteSquares);
        code.dload(squares).dload(deviations).dload(deviations).dmul();
        CpuNormEmitter.geometry(code, geometry, 2).l2d().ddiv().dsub().dstore(result);
        var nonnegative = code.newLabel();
        code.dload(result).loadConstant(0.0).dcmpl().branch(Opcode.IFGE, nonnegative)
                .loadConstant(0.0).dstore(result).labelBinding(nonnegative);
        code.dload(result); CpuNormEmitter.geometry(code, geometry, 2).loadConstant(correction)
                .lsub().l2d().ddiv().dstore(result);
        if (standardDeviation) code.dload(result).invokestatic(MATH, "sqrt",
                CpuNormEmitter.doubleUnary()).dstore(result);
        code.labelBinding(statisticDone);
        var noNan = code.newLabel(); var noInfinity = code.newLabel(); var specialsDone = code.newLabel();
        code.iload(nan).branch(Opcode.IFEQ, noNan)
                .loadConstant(Double.longBitsToDouble(0x7ff8000000000000L)).dstore(result)
                .branch(Opcode.GOTO, specialsDone).labelBinding(noNan)
                .iload(anyInfinity).branch(Opcode.IFEQ, noInfinity)
                .loadConstant(Double.longBitsToDouble(0x7ff8000000000000L)).dstore(result)
                .branch(Opcode.GOTO, specialsDone).labelBinding(noInfinity)
                .labelBinding(specialsDone);
        CpuNormEmitter.emitStore(code, carriers, specialization, type, outputAddress, result);
        code.lload(cell).loadConstant(1L).ladd().lstore(cell)
                .branch(Opcode.GOTO, cells).labelBinding(done);
    }
}
