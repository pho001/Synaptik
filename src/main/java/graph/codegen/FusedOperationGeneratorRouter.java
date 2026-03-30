package graph.codegen;

import tensor.Tensor;

import java.util.List;

/**
 * Routes fused codegen to dtype-specialized generators.
 */
public final class FusedOperationGeneratorRouter {
    private FusedOperationGeneratorRouter() {}

    public static byte[] generate(
            String internalClassName,
            List<Tensor> cluster,
            Tensor outputTensor,
            List<Tensor> externalInputsInOrder,
            int precisionMode
    ) {
        return switch (precisionMode) {
            case FusedDTypeOps.MODE_F64, FusedDTypeOps.MODE_F32 ->
                    FusedOperationGenerator.generate(internalClassName, cluster, outputTensor, externalInputsInOrder, precisionMode);
            case FusedDTypeOps.MODE_F16 ->
                    HFusedOperationGenerator.generate(internalClassName, cluster, outputTensor, externalInputsInOrder);
            default -> throw new IllegalArgumentException("Unsupported precision mode: " + precisionMode);
        };
    }
}
