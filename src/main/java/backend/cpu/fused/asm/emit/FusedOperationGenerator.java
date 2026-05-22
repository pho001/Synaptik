package backend.cpu.fused.asm.emit;

import backend.cpu.fused.asm.FusedAsmSpecializationKind;

import backend.cpu.fused.asm.FusedGenerationContext;

import backend.cpu.fused.runtime.FusedDTypeOps;

import backend.cpu.fused.ir.FusedExpressionPlan;
import backend.cpu.fused.numeric.FusedComputeKind;
import backend.cpu.fused.numeric.FusedNumericContract;
import backend.cpu.fused.numeric.FusedValueLane;
import org.objectweb.asm.ClassWriter;

/**
 * Internal dispatcher for emitting fused operation bytecode.
 */
public final class FusedOperationGenerator {
    private FusedOperationGenerator() {}

    public static byte[] generate(
            String internalClassName,
            FusedExpressionPlan plan,
            FusedNumericContract numericContract,
            int vectorWidth,
            FusedAsmSpecializationKind specializationKind
    ) {
        int precisionMode = precisionMode(numericContract);
        if (precisionMode != FusedDTypeOps.MODE_F32
                && precisionMode != FusedDTypeOps.MODE_F64
                && precisionMode != FusedDTypeOps.MODE_BF16) {
            throw new IllegalArgumentException("Unsupported fused precision mode=" + precisionMode);
        }

        FusedGenerationContext context = FusedGenerationContext.create(
                internalClassName,
                plan,
                numericContract,
                vectorWidth,
                specializationKind
        );

        ClassWriter cw = FusedClassEmitter.createClass(context);
        FusedConstructorEmitter.emit(cw, context);
        FusedScalarMethodEmitter.emit(cw, context);
        if (specializationKind == FusedAsmSpecializationKind.NONE) {
            FusedVectorMethodEmitter.emit(cw, context);
        } else {
            FusedSpecializedVectorMethodEmitter.emit(cw, context);
        }
        cw.visitEnd();
        return cw.toByteArray();
    }

    private static int precisionMode(FusedNumericContract numericContract) {
        if (numericContract.computeKind() == FusedComputeKind.F64) {
            return FusedDTypeOps.MODE_F64;
        }
        return numericContract.outputValueLane() == FusedValueLane.BF16
                ? FusedDTypeOps.MODE_BF16
                : FusedDTypeOps.MODE_F32;
    }
}
