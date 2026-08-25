package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.MethodTypeDesc;

/** Emits the direct stable two-pass log-sum-exp body. */
public final class CpuLogSumExpEmitter {
    private static final ClassDesc MATH = ClassDesc.of(Math.class.getName());
    /** Creates a stateless generation-time emitter. */
    public CpuLogSumExpEmitter() { }
    /**
     * Emits one direct typed entry body.
     * @param code non-null Class-File method builder to mutate
     * @param specialization non-null exact carrier and entry specialization
     * @param ir non-null canonical advanced log-sum-exp identity
     */
    public void emit(CodeBuilder code, CpuKernelSpecialization specialization, CpuKernelIr ir) {
        if (!ir.familyIdentity().startsWith("advanced-reduction:LOG_SUM_EXP:")
                || specialization.carrierPattern().size() != 2
                || specialization.boundaryDataTypes().size() != 2
                || specialization.boundaryDataTypes().getFirst()
                    != specialization.boundaryDataTypes().getLast()
                || specialization.scratchParameter()) {
            throw new IllegalArgumentException("log-sum-exp requires two matching boundaries without scratch");
        }
        DataType type = specialization.boundaryDataTypes().getFirst();
        int inputRank = ir.values().getFirst().accessPlan().iterationRank();
        int outputRank = ir.values().getLast().accessPlan().iterationRank();
        boolean keep = ir.familyIdentity().contains(":keep=true:");
        boolean[] selected = CpuNormEmitter.selected(ir.familyIdentity(), inputRank);
        boolean contiguous = CpuNormEmitter.contiguousSelected(selected,
                ir.values().getFirst().accessPlan().contiguousSuffix());
        int geometry = 2, start = 3, end = 5;
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
        int maximum = code.allocateLocal(TypeKind.DOUBLE);
        int sum = code.allocateLocal(TypeKind.DOUBLE);
        int compensation = code.allocateLocal(TypeKind.DOUBLE);
        int temporary = code.allocateLocal(TypeKind.DOUBLE);
        int nan = code.allocateLocal(TypeKind.INT);
        int positiveInfinity = code.allocateLocal(TypeKind.INT);
        int anyInfinity = code.allocateLocal(TypeKind.INT);

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
        code.loadConstant(0).istore(nan).loadConstant(0).istore(positiveInfinity)
                .loadConstant(0).istore(anyInfinity)
                .loadConstant(Double.NEGATIVE_INFINITY).dstore(maximum);
        CpuNormEmitter.emitPass(code, carriers, specialization, type, selected, inputRank, geometry,
                inputExtents, inputStrides, inputBase, domain, remaining, coordinate, address,
                selectedCoordinates, contiguous, represented, value, () -> {
                    CpuNormEmitter.classify(code, value, nan, positiveInfinity, anyInfinity);
                    var skip = code.newLabel();
                    code.dload(value).dload(maximum).dcmpl().branch(Opcode.IFLE, skip)
                            .dload(value).dstore(maximum).labelBinding(skip);
                });
        code.loadConstant(0.0).dstore(sum).loadConstant(0.0).dstore(compensation);
        CpuNormEmitter.emitPass(code, carriers, specialization, type, selected, inputRank, geometry,
                inputExtents, inputStrides, inputBase, domain, remaining, coordinate, address,
                selectedCoordinates, contiguous, represented, value, () -> {
                    code.dload(value).dload(maximum).dsub()
                            .invokestatic(MATH, "exp", MethodTypeDesc.of(
                                    TypeKind.DOUBLE.upperBound(), TypeKind.DOUBLE.upperBound()))
                            .dstore(value);
                    CpuNormEmitter.kahan(code, value, sum, compensation, temporary);
                });
        code.dload(maximum).dload(sum).invokestatic(MATH, "log", MethodTypeDesc.of(
                TypeKind.DOUBLE.upperBound(), TypeKind.DOUBLE.upperBound())).dadd().dstore(result);
        var notSingleton = code.newLabel();
        CpuNormEmitter.geometry(code, geometry, 2).loadConstant(1L).lcmp()
                .branch(Opcode.IFNE, notSingleton).dload(maximum).dstore(result)
                .labelBinding(notSingleton);
        var noNan = code.newLabel(); var noPositiveInfinity = code.newLabel();
        var nonempty = code.newLabel(); var specialsDone = code.newLabel();
        code.iload(nan).branch(Opcode.IFEQ, noNan)
                .loadConstant(Double.longBitsToDouble(0x7ff8000000000000L)).dstore(result)
                .branch(Opcode.GOTO, specialsDone).labelBinding(noNan)
                .iload(positiveInfinity).branch(Opcode.IFEQ, noPositiveInfinity)
                .loadConstant(Double.POSITIVE_INFINITY).dstore(result)
                .branch(Opcode.GOTO, specialsDone).labelBinding(noPositiveInfinity);
        CpuNormEmitter.geometry(code, geometry, 2).loadConstant(0L).lcmp()
                .branch(Opcode.IFNE, nonempty).loadConstant(Double.NEGATIVE_INFINITY)
                .dstore(result).branch(Opcode.GOTO, specialsDone).labelBinding(nonempty);
        code.dload(maximum).loadConstant(Double.NEGATIVE_INFINITY).dcmpl()
                .branch(Opcode.IFNE, specialsDone).loadConstant(Double.NEGATIVE_INFINITY)
                .dstore(result).labelBinding(specialsDone);
        CpuNormEmitter.emitStore(code, carriers, specialization, type, outputAddress, result);
        code.lload(cell).loadConstant(1L).ladd().lstore(cell)
                .branch(Opcode.GOTO, cells).labelBinding(done);
    }
}
