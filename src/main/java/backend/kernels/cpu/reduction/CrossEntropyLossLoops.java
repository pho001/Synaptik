package backend.kernels.cpu.reduction;

import backend.kernels.cpu.CpuDTypeOps;
import backend.kernels.cpu.CpuKernelContext;
import backend.kernels.cpu.CpuThreadPool;
import backend.kernels.cpu.ResolvedReductionHints;
import tensor.Tensor;
import tensor.TensorMetadata;

public final class CrossEntropyLossLoops {
    private CrossEntropyLossLoops() {
    }

    public static void execute(Tensor logits, Tensor targets, Tensor node, int classDimension, CpuKernelContext context) {
        validate(logits, targets, node, classDimension);
        double[] logitsData = logits.getFloat64Data();
        double[] targetData = targets.getFloat64Data();
        node.getFloat64Data()[node.getStorageOffsetUnsafe()] = reduceMeanLoss(
                logits.getShapeUnsafe(),
                logits.getStridesUnsafe(),
                logits.getStorageOffsetUnsafe(),
                targets.getStridesUnsafe(),
                targets.getStorageOffsetUnsafe(),
                classDimension,
                context.reductionHints(),
                group -> computeGroupF64(logitsData, targetData, group.baseA(), group.baseB(), group.axisStrideA(), group.axisStrideB(), group.axisSize())
        );
    }

    public static void executeF32(Tensor logits, Tensor targets, Tensor node, int classDimension, CpuKernelContext context) {
        validate(logits, targets, node, classDimension);
        float[] logitsData = logits.getFloat32Data();
        float[] targetData = targets.getFloat32Data();
        node.getFloat32Data()[node.getStorageOffsetUnsafe()] = (float) reduceMeanLoss(
                logits.getShapeUnsafe(),
                logits.getStridesUnsafe(),
                logits.getStorageOffsetUnsafe(),
                targets.getStridesUnsafe(),
                targets.getStorageOffsetUnsafe(),
                classDimension,
                context.reductionHints(),
                group -> computeGroupF32(logitsData, targetData, group.baseA(), group.baseB(), group.axisStrideA(), group.axisStrideB(), group.axisSize())
        );
    }

    public static void executeBF16(Tensor logits, Tensor targets, Tensor node, int classDimension, CpuKernelContext context) {
        validate(logits, targets, node, classDimension);
        short[] logitsData = logits.getBFloat16Data();
        short[] targetData = targets.getBFloat16Data();
        float loss = (float) reduceMeanLoss(
                logits.getShapeUnsafe(),
                logits.getStridesUnsafe(),
                logits.getStorageOffsetUnsafe(),
                targets.getStridesUnsafe(),
                targets.getStorageOffsetUnsafe(),
                classDimension,
                context.reductionHints(),
                group -> computeGroupF16(logitsData, targetData, group.baseA(), group.baseB(), group.axisStrideA(), group.axisStrideB(), group.axisSize())
        );
        node.getBFloat16Data()[node.getStorageOffsetUnsafe()] = CpuDTypeOps.toBFloat16Bits(loss);
    }

    private static void validate(Tensor logits, Tensor targets, Tensor node, int classDimension) {
        int[] shape = logits.getShapeUnsafe();
        if (shape == null || shape.length == 0) {
            throw new IllegalArgumentException("Cross entropy input shape must not be empty");
        }
        if (classDimension < 0 || classDimension >= shape.length) {
            throw new IllegalArgumentException("Class dimension out of bounds: " + classDimension);
        }
        int[] targetShape = targets.getShapeUnsafe();
        if (targetShape.length != shape.length) {
            throw new IllegalArgumentException("Targets shape rank must match logits rank");
        }
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] != targetShape[i]) {
                throw new IllegalArgumentException("Targets shape must match logits shape");
            }
        }
        int[] outShape = node.getShapeUnsafe();
        if (outShape.length != 1 || outShape[0] != 1) {
            throw new IllegalArgumentException("Cross entropy output must be scalar-shaped [1]");
        }
    }

    private static double computeGroupF64(double[] logitsData, double[] targetData, int baseLogits, int baseTargets, int axisStrideLogits, int axisStrideTargets, int axisSize) {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0, offset = baseLogits; i < axisSize; i++, offset += axisStrideLogits) {
            max = Math.max(max, logitsData[offset]);
        }

        double sumExp = 0.0;
        double weightedLogits = 0.0;
        double targetSum = 0.0;
        for (int i = 0, logitsOffset = baseLogits, targetOffset = baseTargets; i < axisSize; i++, logitsOffset += axisStrideLogits, targetOffset += axisStrideTargets) {
            double target = targetData[targetOffset];
            double logit = logitsData[logitsOffset];
            sumExp += Math.exp(logit - max);
            weightedLogits += target * logit;
            targetSum += target;
        }

        double logSumExp = max + Math.log(sumExp);
        return targetSum * logSumExp - weightedLogits;
    }

    private static double computeGroupF32(float[] logitsData, float[] targetData, int baseLogits, int baseTargets, int axisStrideLogits, int axisStrideTargets, int axisSize) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0, offset = baseLogits; i < axisSize; i++, offset += axisStrideLogits) {
            max = Math.max(max, logitsData[offset]);
        }

        double sumExp = 0.0;
        double weightedLogits = 0.0;
        double targetSum = 0.0;
        for (int i = 0, logitsOffset = baseLogits, targetOffset = baseTargets; i < axisSize; i++, logitsOffset += axisStrideLogits, targetOffset += axisStrideTargets) {
            double target = targetData[targetOffset];
            double logit = logitsData[logitsOffset];
            sumExp += Math.exp(logit - max);
            weightedLogits += target * logit;
            targetSum += target;
        }

        double logSumExp = max + Math.log(sumExp);
        return targetSum * logSumExp - weightedLogits;
    }

    private static double computeGroupF16(short[] logitsData, short[] targetData, int baseLogits, int baseTargets, int axisStrideLogits, int axisStrideTargets, int axisSize) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0, offset = baseLogits; i < axisSize; i++, offset += axisStrideLogits) {
            max = Math.max(max, CpuDTypeOps.fromBFloat16Bits(logitsData[offset]));
        }

        double sumExp = 0.0;
        double weightedLogits = 0.0;
        double targetSum = 0.0;
        for (int i = 0, logitsOffset = baseLogits, targetOffset = baseTargets; i < axisSize; i++, logitsOffset += axisStrideLogits, targetOffset += axisStrideTargets) {
            double target = CpuDTypeOps.fromBFloat16Bits(targetData[targetOffset]);
            double logit = CpuDTypeOps.fromBFloat16Bits(logitsData[logitsOffset]);
            sumExp += Math.exp(logit - max);
            weightedLogits += target * logit;
            targetSum += target;
        }

        double logSumExp = max + Math.log(sumExp);
        return targetSum * logSumExp - weightedLogits;
    }

    private static double reduceMeanLoss(
            int[] shape,
            int[] logitsStrides,
            int logitsBaseOffset,
            int[] targetStrides,
            int targetBaseOffset,
            int classDimension,
            ResolvedReductionHints hints,
            GroupLossComputer computer
    ) {
        int[] reducedShape = reduceShape(shape, classDimension);
        int[] reducedDenseStrides = TensorMetadata.computeStrides(reducedShape);
        int groupCount = logicalSize(reducedShape);
        int axisSize = shape[classDimension];
        int axisStrideLogits = logitsStrides[classDimension];
        int axisStrideTargets = targetStrides[classDimension];

        if (groupCount == 0) {
            return 0.0;
        }

        if (hints != null && hints.parallel() && groupCount > 1) {
            int chunkSize = Math.max(1, hints.chunkSize());
            int chunks = (groupCount + chunkSize - 1) / chunkSize;
            double[] partials = new double[chunks];
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, groupCount);
                double partial = 0.0;
                for (int group = start; group < end; group++) {
                    GroupState state = groupState(group, shape, logitsStrides, logitsBaseOffset, targetStrides, targetBaseOffset, classDimension, reducedDenseStrides, axisSize, axisStrideLogits, axisStrideTargets);
                    partial += computer.compute(state);
                }
                partials[chunk] = partial;
            });
            double total = 0.0;
            for (double partial : partials) {
                total += partial;
            }
            return total / groupCount;
        }

        double total = 0.0;
        for (int group = 0; group < groupCount; group++) {
            GroupState state = groupState(group, shape, logitsStrides, logitsBaseOffset, targetStrides, targetBaseOffset, classDimension, reducedDenseStrides, axisSize, axisStrideLogits, axisStrideTargets);
            total += computer.compute(state);
        }
        return total / groupCount;
    }

    private static GroupState groupState(
            int reducedIndex,
            int[] shape,
            int[] logitsStrides,
            int logitsBaseOffset,
            int[] targetStrides,
            int targetBaseOffset,
            int classDimension,
            int[] reducedDenseStrides,
            int axisSize,
            int axisStrideLogits,
            int axisStrideTargets
        ) {
        int rem = reducedIndex;
        int baseLogits = logitsBaseOffset;
        int baseTargets = targetBaseOffset;
        for (int d = 0, rd = 0; d < shape.length; d++) {
            if (d == classDimension) {
                continue;
            }
            int coord = rem / reducedDenseStrides[rd];
            rem %= reducedDenseStrides[rd];
            baseLogits += coord * logitsStrides[d];
            baseTargets += coord * targetStrides[d];
            rd++;
        }
        return new GroupState(baseLogits, baseTargets, axisStrideLogits, axisStrideTargets, axisSize);
    }

    private static int[] reduceShape(int[] shape, int axis) {
        if (shape.length == 1) {
            return new int[]{1};
        }
        int[] reduced = new int[shape.length - 1];
        for (int i = 0, j = 0; i < shape.length; i++) {
            if (i != axis) {
                reduced[j++] = shape[i];
            }
        }
        return reduced;
    }

    private static int logicalSize(int[] shape) {
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        return size;
    }

    @FunctionalInterface
    private interface GroupLossComputer {
        double compute(GroupState state);
    }

    private record GroupState(int baseA, int baseB, int axisStrideA, int axisStrideB, int axisSize) {
    }
}
