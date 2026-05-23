package backend.cpu.fused.asm.emit;

import backend.cpu.fused.asm.FusedAsmSpecializationKind;

import backend.cpu.fused.asm.FusedGenerationContext;

import backend.cpu.fused.ir.FusedExpressionPlan;
import backend.cpu.fused.numeric.FusedApproximationContract;
import backend.cpu.fused.numeric.FusedNumericContract;
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
            FusedApproximationContract approximationContract,
            int vectorWidth,
            FusedAsmSpecializationKind specializationKind
    ) {
        FusedGenerationContext context = FusedGenerationContext.create(
                internalClassName,
                plan,
                numericContract,
                approximationContract,
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
}
