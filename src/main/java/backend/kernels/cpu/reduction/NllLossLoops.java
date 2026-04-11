package backend.kernels.cpu.reduction;

import backend.kernels.cpu.*;

import backend.kernels.cpu.CpuDTypeOps;
import backend.kernels.cpu.CpuKernelContext;
import backend.kernels.cpu.CpuThreadPool;
import backend.kernels.cpu.ResolvedReductionHints;
import tensor.Tensor;
import tensor.TensorMetadata;

public final class NllLossLoops {
    private NllLossLoops() {
    }

    public static void execute(Tensor logProbs, Tensor targets, Tensor node, int classDimension, CpuKernelContext context) {
        validate(logProbs, targets, node, classDimension);
        double[] logData = logProbs.getFloat64Data();
        double[] targetData = targets.getFloat64Data();
        node.getFloat64Data()[node.getStorageOffsetUnsafe()] = reduceMeanLoss(
                logProbs.getShapeUnsafe(),
                logProbs.getStridesUnsafe(),
                logProbs.getStorageOffsetUnsafe(),
                targets.getStridesUnsafe(),
                targets.getStorageOffsetUnsafe(),
                classDimension,
                context.reductionHints(),
                group -> computeGroupF64(logData, targetData, group.baseA(), group.baseB(), group.axisStrideA(), group.axisStrideB(), group.axisSize())
        );
    }

    public static void executeF32(Tensor logProbs, Tensor targets, Tensor node, int classDimension, CpuKernelContext context) {
        validate(logProbs, targets, node, classDimension);
        float[] logData = logProbs.getFloat32Data();
        float[] targetData = targets.getFloat32Data();
        node.getFloat32Data()[node.getStorageOffsetUnsafe()] = (float) reduceMeanLoss(
                logProbs.getShapeUnsafe(),
                logProbs.getStridesUnsafe(),
                logProbs.getStorageOffsetUnsafe(),
                targets.getStridesUnsafe(),
                targets.getStorageOffsetUnsafe(),
                classDimension,
                context.reductionHints(),
                group -> computeGroupF32(logData, targetData, group.baseA(), group.baseB(), group.axisStrideA(), group.axisStrideB(), group.axisSize())
        );
    }

    public static void executeBF16(Tensor logProbs, Tensor targets, Tensor node, int classDimension, CpuKernelContext context) {
        validate(logProbs, targets, node, classDimension);
        short[] logData = logProbs.getBFloat16Data();
        short[] targetData = targets.getBFloat16Data();
        float loss = (float) reduceMeanLoss(
                logProbs.getShapeUnsafe(),
                logProbs.getStridesUnsafe(),
                logProbs.getStorageOffsetUnsafe(),
                targets.getStridesUnsafe(),
                targets.getStorageOffsetUnsafe(),
                classDimension,
                context.reductionHints(),
                group -> computeGroupF16(logData, targetData, group.baseA(), group.baseB(), group.axisStrideA(), group.axisStrideB(), group.axisSize())
        );
        node.getBFloat16Data()[node.getStorageOffsetUnsafe()] = CpuDTypeOps.toBFloat16Bits(loss);
    }

    public static void executeF32ToBF16(Tensor logProbs, float[] logData, Tensor targets, Tensor node, int classDimension, CpuKernelContext context) {
        validate(logProbs, targets, node, classDimension);
        short[] targetData = targets.getBFloat16Data();
        float loss = (float) reduceMeanLoss(
                logProbs.getShapeUnsafe(),
                logProbs.getStridesUnsafe(),
                0,
                targets.getStridesUnsafe(),
                targets.getStorageOffsetUnsafe(),
                classDimension,
                context.reductionHints(),
                group -> computeGroupF32ToBF16(logData, targetData, group.baseA(), group.baseB(), group.axisStrideA(), group.axisStrideB(), group.axisSize())
        );
        node.getBFloat16Data()[node.getStorageOffsetUnsafe()] = CpuDTypeOps.toBFloat16Bits(loss);
    }

    private static void validate(Tensor logProbs, Tensor targets, Tensor node, int classDimension) {
        int[] shape = logProbs.getShapeUnsafe();
        if (shape == null || shape.length == 0) {
            throw new IllegalArgumentException("NLL loss input shape must not be empty");
        }
        if (classDimension < 0 || classDimension >= shape.length) {
            throw new IllegalArgumentException("Class dimension out of bounds: " + classDimension);
        }
        int[] targetShape = targets.getShapeUnsafe();
        if (targetShape.length != shape.length) {
            throw new IllegalArgumentException("Targets shape rank must match logProbs rank");
        }
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] != targetShape[i]) {
                throw new IllegalArgumentException("Targets shape must match logProbs shape");
            }
        }
        int[] outShape = node.getShapeUnsafe();
        if (outShape.length != 1 || outShape[0] != 1) {
            throw new IllegalArgumentException("NLL loss output must be scalar-shaped [1]");
        }
    }

    private static double computeGroupF64(double[] logData, double[] targetData, int baseLog, int baseTarget, int axisStrideLog, int axisStrideTarget, int axisSize) {
        double loss = 0.0;
        for (int i = 0, logOffset = baseLog, targetOffset = baseTarget; i < axisSize; i++, logOffset += axisStrideLog, targetOffset += axisStrideTarget) {
            loss -= targetData[targetOffset] * logData[logOffset];
        }
        return loss;
    }

    private static double computeGroupF32(float[] logData, float[] targetData, int baseLog, int baseTarget, int axisStrideLog, int axisStrideTarget, int axisSize) {
        double loss = 0.0;
        for (int i = 0, logOffset = baseLog, targetOffset = baseTarget; i < axisSize; i++, logOffset += axisStrideLog, targetOffset += axisStrideTarget) {
            loss -= targetData[targetOffset] * logData[logOffset];
        }
        return loss;
    }

    private static double computeGroupF16(short[] logData, short[] targetData, int baseLog, int baseTarget, int axisStrideLog, int axisStrideTarget, int axisSize) {
        double loss = 0.0;
        for (int i = 0, logOffset = baseLog, targetOffset = baseTarget; i < axisSize; i++, logOffset += axisStrideLog, targetOffset += axisStrideTarget) {
            loss -= CpuDTypeOps.fromBFloat16Bits(targetData[targetOffset]) * CpuDTypeOps.fromBFloat16Bits(logData[logOffset]);
        }
        return loss;
    }

    private static double computeGroupF32ToBF16(float[] logData, short[] targetData, int baseLog, int baseTarget, int axisStrideLog, int axisStrideTarget, int axisSize) {
        double loss = 0.0;
        for (int i = 0, logOffset = baseLog, targetOffset = baseTarget; i < axisSize; i++, logOffset += axisStrideLog, targetOffset += axisStrideTarget) {
            loss -= CpuDTypeOps.fromBFloat16Bits(targetData[targetOffset]) * logData[logOffset];
        }
        return loss;
    }

    private static double reduceMeanLoss(
            int[] shape,
            int[] logStrides,
            int logBaseOffset,
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
        int axisStrideLog = logStrides[classDimension];
        int axisStrideTarget = targetStrides[classDimension];

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
                    GroupState state = groupState(group, shape, logStrides, logBaseOffset, targetStrides, targetBaseOffset, classDimension, reducedDenseStrides, axisSize, axisStrideLog, axisStrideTarget);
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
            GroupState state = groupState(group, shape, logStrides, logBaseOffset, targetStrides, targetBaseOffset, classDimension, reducedDenseStrides, axisSize, axisStrideLog, axisStrideTarget);
            total += computer.compute(state);
        }
        return total / groupCount;
    }

    private static GroupState groupState(
            int reducedIndex,
            int[] shape,
            int[] logStrides,
            int logBaseOffset,
            int[] targetStrides,
            int targetBaseOffset,
            int classDimension,
            int[] reducedDenseStrides,
            int axisSize,
            int axisStrideLog,
            int axisStrideTarget
        ) {
        int rem = reducedIndex;
        int baseLog = logBaseOffset;
        int baseTarget = targetBaseOffset;
        for (int d = 0, rd = 0; d < shape.length; d++) {
            if (d == classDimension) {
                continue;
            }
            int coord = rem / reducedDenseStrides[rd];
            rem %= reducedDenseStrides[rd];
            baseLog += coord * logStrides[d];
            baseTarget += coord * targetStrides[d];
            rd++;
        }
        return new GroupState(baseLog, baseTarget, axisStrideLog, axisStrideTarget, axisSize);
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
