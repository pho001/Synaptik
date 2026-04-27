package backend.cpu.fused.codegen;
import org.objectweb.asm.ClassWriter;

public final class FusedOperationGenerator {
    private FusedOperationGenerator() {}

    public static byte[] generate(
            String internalClassName,
            FusedExpressionPlan plan,
            int precisionMode,
            int vectorWidth,
            FusedAsmSpecializationKind specializationKind
    ) {
        if (precisionMode != FusedDTypeOps.MODE_F32
                && precisionMode != FusedDTypeOps.MODE_F64
                && precisionMode != FusedDTypeOps.MODE_BF16) {
            throw new IllegalArgumentException("Unsupported fused precision mode=" + precisionMode);
        }

        FusedGenerationContext context = FusedGenerationContext.create(
                internalClassName,
                plan,
                precisionMode,
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
