package backend.cpu.kernels.reduction;

import backend.cpu.storage.CpuStorageView;

import java.lang.foreign.MemorySegment;

final class BoolReduceLoops {
    private BoolReduceLoops() {}

    static void execute(CpuStorageView input, CpuStorageView output, int dimension, boolean isAll) {
        int[] shape = input.shape();
        ReductionTraversal.validateDimension(shape, dimension);

        if (input.isArray() && output.isArray()) {
            executeArray(input, output, dimension, isAll, shape);
            return;
        }
        executeStorage(input, output, dimension, isAll, shape);
    }

    private static void executeArray(CpuStorageView input, CpuStorageView output, int dimension, boolean isAll, int[] shape) {
        byte[] in = input.requireBoolArray();
        byte[] out = output.requireBoolArray();
        int[] inputStrides = input.strides();
        int inputBaseOffset = input.storageOffset();
        int[] outputShape = output.shape();
        int[] outputStrides = output.strides();
        int outputBaseOffset = output.storageOffset();

        if (dimension == -1) {
            byte value = reduceAllArray(in, shape, inputStrides, inputBaseOffset, input.logicalSize(), isAll);
            int outOffset = ReductionStorageAccess.logicalToOffset(0, outputShape, outputStrides, outputBaseOffset);
            out[outOffset] = value;
            return;
        }
        reduceAxisArray(
                in,
                shape,
                inputStrides,
                inputBaseOffset,
                out,
                outputShape,
                outputStrides,
                outputBaseOffset,
                dimension,
                isAll
        );
    }

    private static void executeStorage(CpuStorageView input, CpuStorageView output, int dimension, boolean isAll, int[] shape) {
        byte[] inArray = ReductionStorageAccess.boolArray(input);
        MemorySegment inSegment = ReductionStorageAccess.boolSegment(input);
        byte[] outArray = ReductionStorageAccess.boolArray(output);
        MemorySegment outSegment = ReductionStorageAccess.boolSegment(output);
        int[] inputStrides = input.strides();
        int inputBaseOffset = input.storageOffset();
        int[] outputShape = output.shape();
        int[] outputStrides = output.strides();
        int outputBaseOffset = output.storageOffset();

        if (dimension == -1) {
            byte value = reduceAll(inArray, inSegment, shape, inputStrides, inputBaseOffset, input.logicalSize(), isAll);
            int outOffset = ReductionStorageAccess.logicalToOffset(0, outputShape, outputStrides, outputBaseOffset);
            ReductionStorageAccess.writeBool(outArray, outSegment, outOffset, value);
            return;
        }
        reduceAxis(
                inArray,
                inSegment,
                shape,
                inputStrides,
                inputBaseOffset,
                outArray,
                outSegment,
                outputShape,
                outputStrides,
                outputBaseOffset,
                dimension,
                isAll
        );
    }

    private static void reduceAxisArray(
            byte[] in,
            int[] inputShape,
            int[] inputStrides,
            int inputBaseOffset,
            byte[] out,
            int[] outputShape,
            int[] outputStrides,
            int outputBaseOffset,
            int dimension,
            boolean isAll
    ) {
        ReductionTraversal.forEachAxisGroup(inputShape, inputStrides, inputBaseOffset, outputShape, dimension, (outIndex, baseOffset, reducedSize, reducedStride) -> {
            boolean acc = in[baseOffset] != 0;
            for (int r = 1; r < reducedSize; r++) {
                boolean value = in[baseOffset + r * reducedStride] != 0;
                acc = isAll ? (acc && value) : (acc || value);
            }
            int outOffset = ReductionStorageAccess.logicalToOffset(outIndex, outputShape, outputStrides, outputBaseOffset);
            out[outOffset] = acc ? (byte) 1 : (byte) 0;
        });
    }

    private static void reduceAxis(
            byte[] inArray,
            MemorySegment inSegment,
            int[] inputShape,
            int[] inputStrides,
            int inputBaseOffset,
            byte[] outArray,
            MemorySegment outSegment,
            int[] outputShape,
            int[] outputStrides,
            int outputBaseOffset,
            int dimension,
            boolean isAll
    ) {
        ReductionTraversal.forEachAxisGroup(inputShape, inputStrides, inputBaseOffset, outputShape, dimension, (outIndex, baseOffset, reducedSize, reducedStride) -> {
            boolean acc = ReductionStorageAccess.readBool(inArray, inSegment, baseOffset) != 0;
            for (int r = 1; r < reducedSize; r++) {
                boolean value = ReductionStorageAccess.readBool(inArray, inSegment, baseOffset + r * reducedStride) != 0;
                acc = isAll ? (acc && value) : (acc || value);
            }
            int outOffset = ReductionStorageAccess.logicalToOffset(outIndex, outputShape, outputStrides, outputBaseOffset);
            ReductionStorageAccess.writeBool(outArray, outSegment, outOffset, acc ? (byte) 1 : (byte) 0);
        });
    }

    private static byte reduceAllArray(
            byte[] in,
            int[] shape,
            int[] strides,
            int baseOffset,
            int logicalSize,
            boolean isAll
    ) {
        int firstOffset = ReductionStorageAccess.logicalToOffset(0, shape, strides, baseOffset);
        boolean acc = in[firstOffset] != 0;
        for (int logical = 1; logical < logicalSize; logical++) {
            int offset = ReductionStorageAccess.logicalToOffset(logical, shape, strides, baseOffset);
            boolean value = in[offset] != 0;
            acc = isAll ? (acc && value) : (acc || value);
        }
        return acc ? (byte) 1 : (byte) 0;
    }

    private static byte reduceAll(
            byte[] inArray,
            MemorySegment inSegment,
            int[] shape,
            int[] strides,
            int baseOffset,
            int logicalSize,
            boolean isAll
    ) {
        int firstOffset = ReductionStorageAccess.logicalToOffset(0, shape, strides, baseOffset);
        boolean acc = ReductionStorageAccess.readBool(inArray, inSegment, firstOffset) != 0;
        for (int logical = 1; logical < logicalSize; logical++) {
            int offset = ReductionStorageAccess.logicalToOffset(logical, shape, strides, baseOffset);
            boolean value = ReductionStorageAccess.readBool(inArray, inSegment, offset) != 0;
            acc = isAll ? (acc && value) : (acc || value);
        }
        return acc ? (byte) 1 : (byte) 0;
    }
}
