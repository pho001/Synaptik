package backend.cpu.kernels.reduction;

import backend.cpu.kernels.CpuDTypeOps;
import backend.cpu.kernels.CpuKernelContext;
import tensor.Tensor;

final class LossReductionExecutor {
    private LossReductionExecutor() {}

    static void executeF64(LossReduction reduction, Tensor a, Tensor b, Tensor node, int classDimension, CpuKernelContext context) {
        validate(reduction, a, b, node, context);
        LossReductionTraversal.validateShapes(a.getShapeUnsafe(), b.getShapeUnsafe(), node.getShapeUnsafe(), classDimension, label(reduction));
        double[] aData = a.getFloat64Data();
        double[] bData = b.getFloat64Data();
        node.getFloat64Data()[node.getStorageOffsetUnsafe()] = LossReductionTraversal.reduceMeanLoss(
                a.getShapeUnsafe(), a.getStridesUnsafe(), a.getStorageOffsetUnsafe(), b.getStridesUnsafe(), b.getStorageOffsetUnsafe(), classDimension, context,
                (baseA, baseB, axisStrideA, axisStrideB, axisSize) ->
                        reduction.computeF64(aData, bData, baseA, baseB, axisStrideA, axisStrideB, axisSize)
        );
    }

    static void executeF32(LossReduction reduction, Tensor a, Tensor b, Tensor node, int classDimension, CpuKernelContext context) {
        validate(reduction, a, b, node, context);
        LossReductionTraversal.validateShapes(a.getShapeUnsafe(), b.getShapeUnsafe(), node.getShapeUnsafe(), classDimension, label(reduction));
        float[] aData = a.getFloat32Data();
        float[] bData = b.getFloat32Data();
        node.getFloat32Data()[node.getStorageOffsetUnsafe()] = (float) LossReductionTraversal.reduceMeanLoss(
                a.getShapeUnsafe(), a.getStridesUnsafe(), a.getStorageOffsetUnsafe(), b.getStridesUnsafe(), b.getStorageOffsetUnsafe(), classDimension, context,
                (baseA, baseB, axisStrideA, axisStrideB, axisSize) ->
                        reduction.computeF32(aData, bData, baseA, baseB, axisStrideA, axisStrideB, axisSize)
        );
    }

    static void executeBF16(LossReduction reduction, Tensor a, Tensor b, Tensor node, int classDimension, CpuKernelContext context) {
        validate(reduction, a, b, node, context);
        LossReductionTraversal.validateShapes(a.getShapeUnsafe(), b.getShapeUnsafe(), node.getShapeUnsafe(), classDimension, label(reduction));
        short[] aData = a.getBFloat16Data();
        short[] bData = b.getBFloat16Data();
        float loss = (float) LossReductionTraversal.reduceMeanLoss(
                a.getShapeUnsafe(), a.getStridesUnsafe(), a.getStorageOffsetUnsafe(), b.getStridesUnsafe(), b.getStorageOffsetUnsafe(), classDimension, context,
                (baseA, baseB, axisStrideA, axisStrideB, axisSize) ->
                        reduction.computeBF16(aData, bData, baseA, baseB, axisStrideA, axisStrideB, axisSize)
        );
        node.getBFloat16Data()[node.getStorageOffsetUnsafe()] = CpuDTypeOps.toBFloat16Bits(loss);
    }

    static void executeF32ToBF16(LossReduction reduction, Tensor a, float[] aData, Tensor b, Tensor node, int classDimension, CpuKernelContext context) {
        validate(reduction, a, b, node, context);
        LossReductionTraversal.validateShapes(a.getShapeUnsafe(), b.getShapeUnsafe(), node.getShapeUnsafe(), classDimension, label(reduction));
        if (aData == null || b.getDataType() != tensor.DataType.BFLOAT16) {
            throw new IllegalArgumentException("Loss F32 continuation requires float logits/logProbs and BF16 targets");
        }
        short[] bData = b.getBFloat16Data();
        float loss = (float) LossReductionTraversal.reduceMeanLoss(
                a.getShapeUnsafe(), a.getStridesUnsafe(), 0, b.getStridesUnsafe(), b.getStorageOffsetUnsafe(), classDimension, context,
                (baseA, baseB, axisStrideA, axisStrideB, axisSize) ->
                        reduction.computeF32ToBF16(aData, bData, baseA, baseB, axisStrideA, axisStrideB, axisSize)
        );
        node.getBFloat16Data()[node.getStorageOffsetUnsafe()] = CpuDTypeOps.toBFloat16Bits(loss);
    }

    private static void validate(LossReduction reduction, Tensor a, Tensor b, Tensor node, CpuKernelContext context) {
        if (reduction == null || a == null || b == null || node == null || context == null) {
            throw new IllegalArgumentException("loss reduction execution arguments cannot be null");
        }
    }

    private static String label(LossReduction reduction) {
        return reduction == LossReduction.NLL ? "NLL loss" : "Cross entropy";
    }
}
