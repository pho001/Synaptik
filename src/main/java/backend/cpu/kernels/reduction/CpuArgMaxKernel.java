package backend.cpu.kernels.reduction;

import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import backend.cpu.storage.CpuStorageView;
import operations.reduction.ArgMaxTiePolicy;
import operations.reduction.argMax;
import tensor.DataType;
import tensor.Tensor;
import tensor.dtype.TensorDTypeOps;

import java.lang.foreign.MemorySegment;
import java.util.List;

public final class CpuArgMaxKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        if (!(call.operation() instanceof argMax reduction)) {
            throw new IllegalArgumentException("CpuArgMaxKernel requires argMax operation.");
        }
        requireSingleInput(call.inputTensors(), "ArgMax");
        CpuStorageView input = requireSingleInputView(call, "ArgMax");
        CpuStorageView output = requireOutputView(call, "ArgMax");
        if (input.dtype() == DataType.BOOL) {
            throw new IllegalArgumentException("ArgMax requires numeric input.");
        }
        int axis = reduction.getDimension();
        int[] shape = input.shape();
        if (axis < 0 || axis >= shape.length) {
            throw new IllegalArgumentException("Dimension out of bounds: " + axis);
        }

        boolean lastIndexWins = reduction.tiePolicy() == ArgMaxTiePolicy.LAST_INDEX;
        switch (output.dtype()) {
            case INT32 -> argMaxI32(input, output, axis, lastIndexWins);
            case INT64 -> argMaxI64(input, output, axis, lastIndexWins);
            case FLOAT64, FLOAT32, BFLOAT16, BOOL -> throw new IllegalArgumentException(
                    "ArgMax requires INT32 or INT64 output.");
        }
        return CpuKernelResult.completed();
    }

    private static void argMaxI32(CpuStorageView input, CpuStorageView output, int axis, boolean lastIndexWins) {
        int[] outArray = ReductionStorageAccess.i32Array(output);
        MemorySegment outSegment = ReductionStorageAccess.i32Segment(output);
        int[] inputShape = input.shape();
        int[] inputStrides = input.strides();
        int[] outputShape = output.shape();
        int[] outputStrides = output.strides();
        int groups = output.logicalSize();
        int reducedSize = inputShape[axis];
        int reducedStride = inputStrides[axis];

        switch (input.dtype()) {
            case FLOAT64 -> {
                double[] inArray = ReductionStorageAccess.f64Array(input);
                MemorySegment inSegment = ReductionStorageAccess.f64Segment(input);
                for (int outLogical = 0; outLogical < groups; outLogical++) {
                    int inputBase = axisBaseOffset(outLogical, inputShape, inputStrides, input.storageOffset(), outputShape, axis);
                    int bestIndex = bestIndexF64(inArray, inSegment, inputBase, reducedSize, reducedStride, lastIndexWins);
                    writeBestI32(outArray, outSegment, outputShape, outputStrides, output.storageOffset(), outLogical, bestIndex);
                }
            }
            case FLOAT32 -> {
                float[] inArray = ReductionStorageAccess.f32Array(input);
                MemorySegment inSegment = ReductionStorageAccess.f32Segment(input);
                for (int outLogical = 0; outLogical < groups; outLogical++) {
                    int inputBase = axisBaseOffset(outLogical, inputShape, inputStrides, input.storageOffset(), outputShape, axis);
                    int bestIndex = bestIndexF32(inArray, inSegment, inputBase, reducedSize, reducedStride, lastIndexWins);
                    writeBestI32(outArray, outSegment, outputShape, outputStrides, output.storageOffset(), outLogical, bestIndex);
                }
            }
            case BFLOAT16 -> {
                short[] inArray = ReductionStorageAccess.bf16Array(input);
                MemorySegment inSegment = ReductionStorageAccess.bf16Segment(input);
                for (int outLogical = 0; outLogical < groups; outLogical++) {
                    int inputBase = axisBaseOffset(outLogical, inputShape, inputStrides, input.storageOffset(), outputShape, axis);
                    int bestIndex = bestIndexBF16(inArray, inSegment, inputBase, reducedSize, reducedStride, lastIndexWins);
                    writeBestI32(outArray, outSegment, outputShape, outputStrides, output.storageOffset(), outLogical, bestIndex);
                }
            }
            case INT32 -> {
                int[] inArray = ReductionStorageAccess.i32Array(input);
                MemorySegment inSegment = ReductionStorageAccess.i32Segment(input);
                for (int outLogical = 0; outLogical < groups; outLogical++) {
                    int inputBase = axisBaseOffset(outLogical, inputShape, inputStrides, input.storageOffset(), outputShape, axis);
                    int bestIndex = bestIndexI32(inArray, inSegment, inputBase, reducedSize, reducedStride, lastIndexWins);
                    writeBestI32(outArray, outSegment, outputShape, outputStrides, output.storageOffset(), outLogical, bestIndex);
                }
            }
            case INT64 -> {
                long[] inArray = ReductionStorageAccess.i64Array(input);
                MemorySegment inSegment = ReductionStorageAccess.i64Segment(input);
                for (int outLogical = 0; outLogical < groups; outLogical++) {
                    int inputBase = axisBaseOffset(outLogical, inputShape, inputStrides, input.storageOffset(), outputShape, axis);
                    int bestIndex = bestIndexI64(inArray, inSegment, inputBase, reducedSize, reducedStride, lastIndexWins);
                    writeBestI32(outArray, outSegment, outputShape, outputStrides, output.storageOffset(), outLogical, bestIndex);
                }
            }
            case BOOL -> throw new IllegalArgumentException("ArgMax requires numeric input.");
        }
    }

    private static void argMaxI64(CpuStorageView input, CpuStorageView output, int axis, boolean lastIndexWins) {
        long[] outArray = ReductionStorageAccess.i64Array(output);
        MemorySegment outSegment = ReductionStorageAccess.i64Segment(output);
        int[] inputShape = input.shape();
        int[] inputStrides = input.strides();
        int[] outputShape = output.shape();
        int[] outputStrides = output.strides();
        int groups = output.logicalSize();
        int reducedSize = inputShape[axis];
        int reducedStride = inputStrides[axis];

        switch (input.dtype()) {
            case FLOAT64 -> {
                double[] inArray = ReductionStorageAccess.f64Array(input);
                MemorySegment inSegment = ReductionStorageAccess.f64Segment(input);
                for (int outLogical = 0; outLogical < groups; outLogical++) {
                    int inputBase = axisBaseOffset(outLogical, inputShape, inputStrides, input.storageOffset(), outputShape, axis);
                    int bestIndex = bestIndexF64(inArray, inSegment, inputBase, reducedSize, reducedStride, lastIndexWins);
                    writeBestI64(outArray, outSegment, outputShape, outputStrides, output.storageOffset(), outLogical, bestIndex);
                }
            }
            case FLOAT32 -> {
                float[] inArray = ReductionStorageAccess.f32Array(input);
                MemorySegment inSegment = ReductionStorageAccess.f32Segment(input);
                for (int outLogical = 0; outLogical < groups; outLogical++) {
                    int inputBase = axisBaseOffset(outLogical, inputShape, inputStrides, input.storageOffset(), outputShape, axis);
                    int bestIndex = bestIndexF32(inArray, inSegment, inputBase, reducedSize, reducedStride, lastIndexWins);
                    writeBestI64(outArray, outSegment, outputShape, outputStrides, output.storageOffset(), outLogical, bestIndex);
                }
            }
            case BFLOAT16 -> {
                short[] inArray = ReductionStorageAccess.bf16Array(input);
                MemorySegment inSegment = ReductionStorageAccess.bf16Segment(input);
                for (int outLogical = 0; outLogical < groups; outLogical++) {
                    int inputBase = axisBaseOffset(outLogical, inputShape, inputStrides, input.storageOffset(), outputShape, axis);
                    int bestIndex = bestIndexBF16(inArray, inSegment, inputBase, reducedSize, reducedStride, lastIndexWins);
                    writeBestI64(outArray, outSegment, outputShape, outputStrides, output.storageOffset(), outLogical, bestIndex);
                }
            }
            case INT32 -> {
                int[] inArray = ReductionStorageAccess.i32Array(input);
                MemorySegment inSegment = ReductionStorageAccess.i32Segment(input);
                for (int outLogical = 0; outLogical < groups; outLogical++) {
                    int inputBase = axisBaseOffset(outLogical, inputShape, inputStrides, input.storageOffset(), outputShape, axis);
                    int bestIndex = bestIndexI32(inArray, inSegment, inputBase, reducedSize, reducedStride, lastIndexWins);
                    writeBestI64(outArray, outSegment, outputShape, outputStrides, output.storageOffset(), outLogical, bestIndex);
                }
            }
            case INT64 -> {
                long[] inArray = ReductionStorageAccess.i64Array(input);
                MemorySegment inSegment = ReductionStorageAccess.i64Segment(input);
                for (int outLogical = 0; outLogical < groups; outLogical++) {
                    int inputBase = axisBaseOffset(outLogical, inputShape, inputStrides, input.storageOffset(), outputShape, axis);
                    int bestIndex = bestIndexI64(inArray, inSegment, inputBase, reducedSize, reducedStride, lastIndexWins);
                    writeBestI64(outArray, outSegment, outputShape, outputStrides, output.storageOffset(), outLogical, bestIndex);
                }
            }
            case BOOL -> throw new IllegalArgumentException("ArgMax requires numeric input.");
        }
    }

    private static int bestIndexF64(
            double[] inputArray,
            MemorySegment inputSegment,
            int inputBase,
            int reducedSize,
            int reducedStride,
            boolean lastIndexWins
    ) {
        int bestIndex = 0;
        double bestValue = Double.NEGATIVE_INFINITY;
        boolean seen = false;
        for (int r = 0; r < reducedSize; r++) {
            double value = ReductionStorageAccess.readF64(inputArray, inputSegment, inputBase + r * reducedStride);
            if (isBetter(value, bestValue, seen, lastIndexWins)) {
                seen = true;
                bestValue = value;
                bestIndex = r;
            }
        }
        return bestIndex;
    }

    private static int bestIndexF32(
            float[] inputArray,
            MemorySegment inputSegment,
            int inputBase,
            int reducedSize,
            int reducedStride,
            boolean lastIndexWins
    ) {
        int bestIndex = 0;
        double bestValue = Double.NEGATIVE_INFINITY;
        boolean seen = false;
        for (int r = 0; r < reducedSize; r++) {
            double value = ReductionStorageAccess.readF32(inputArray, inputSegment, inputBase + r * reducedStride);
            if (isBetter(value, bestValue, seen, lastIndexWins)) {
                seen = true;
                bestValue = value;
                bestIndex = r;
            }
        }
        return bestIndex;
    }

    private static int bestIndexBF16(
            short[] inputArray,
            MemorySegment inputSegment,
            int inputBase,
            int reducedSize,
            int reducedStride,
            boolean lastIndexWins
    ) {
        int bestIndex = 0;
        double bestValue = Double.NEGATIVE_INFINITY;
        boolean seen = false;
        for (int r = 0; r < reducedSize; r++) {
            double value = TensorDTypeOps.fromBFloat16Bits(
                    ReductionStorageAccess.readBF16(inputArray, inputSegment, inputBase + r * reducedStride));
            if (isBetter(value, bestValue, seen, lastIndexWins)) {
                seen = true;
                bestValue = value;
                bestIndex = r;
            }
        }
        return bestIndex;
    }

    private static int bestIndexI32(
            int[] inputArray,
            MemorySegment inputSegment,
            int inputBase,
            int reducedSize,
            int reducedStride,
            boolean lastIndexWins
    ) {
        int bestIndex = 0;
        double bestValue = Double.NEGATIVE_INFINITY;
        boolean seen = false;
        for (int r = 0; r < reducedSize; r++) {
            double value = ReductionStorageAccess.readI32(inputArray, inputSegment, inputBase + r * reducedStride);
            if (isBetter(value, bestValue, seen, lastIndexWins)) {
                seen = true;
                bestValue = value;
                bestIndex = r;
            }
        }
        return bestIndex;
    }

    private static int bestIndexI64(
            long[] inputArray,
            MemorySegment inputSegment,
            int inputBase,
            int reducedSize,
            int reducedStride,
            boolean lastIndexWins
    ) {
        int bestIndex = 0;
        double bestValue = Double.NEGATIVE_INFINITY;
        boolean seen = false;
        for (int r = 0; r < reducedSize; r++) {
            double value = ReductionStorageAccess.readI64(inputArray, inputSegment, inputBase + r * reducedStride);
            if (isBetter(value, bestValue, seen, lastIndexWins)) {
                seen = true;
                bestValue = value;
                bestIndex = r;
            }
        }
        return bestIndex;
    }

    private static boolean isBetter(double value, double bestValue, boolean seen, boolean lastIndexWins) {
        return !seen || value > bestValue || (lastIndexWins && Double.compare(value, bestValue) == 0);
    }

    private static void writeBestI32(
            int[] outputArray,
            MemorySegment outputSegment,
            int[] outputShape,
            int[] outputStrides,
            int outputStorageOffset,
            int outputLogical,
            int bestIndex
    ) {
        int outputOffset = ReductionStorageAccess.logicalToOffset(outputLogical, outputShape, outputStrides, outputStorageOffset);
        ReductionStorageAccess.writeI32(outputArray, outputSegment, outputOffset, bestIndex);
    }

    private static void writeBestI64(
            long[] outputArray,
            MemorySegment outputSegment,
            int[] outputShape,
            int[] outputStrides,
            int outputStorageOffset,
            int outputLogical,
            int bestIndex
    ) {
        int outputOffset = ReductionStorageAccess.logicalToOffset(outputLogical, outputShape, outputStrides, outputStorageOffset);
        ReductionStorageAccess.writeI64(outputArray, outputSegment, outputOffset, bestIndex);
    }

    private static int axisBaseOffset(
            int outputLogical,
            int[] inputShape,
            int[] inputStrides,
            int inputStorageOffset,
            int[] outputShape,
            int reducedAxis
    ) {
        int remaining = outputLogical;
        int offset = inputStorageOffset;
        if (outputShape.length == inputShape.length) {
            for (int outDim = outputShape.length - 1; outDim >= 0; outDim--) {
                int coord = remaining % outputShape[outDim];
                remaining /= outputShape[outDim];
                if (outDim != reducedAxis) {
                    offset += coord * inputStrides[outDim];
                }
            }
            return offset;
        }
        for (int outDim = outputShape.length - 1; outDim >= 0; outDim--) {
            int coord = remaining % outputShape[outDim];
            remaining /= outputShape[outDim];
            int inputDim = outDim < reducedAxis ? outDim : outDim + 1;
            offset += coord * inputStrides[inputDim];
        }
        return offset;
    }

    private static Tensor requireSingleInput(List<Tensor> inputs, String label) {
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException(label + " expects exactly one input tensor");
        }
        return inputs.getFirst();
    }

    private static CpuStorageView requireSingleInputView(CpuKernelCall call, String label) {
        if (call.inputs().size() != 1) {
            throw new IllegalArgumentException(label + " expects exactly one input storage view.");
        }
        return call.inputs().getFirst();
    }

    private static CpuStorageView requireOutputView(CpuKernelCall call, String label) {
        if (call.output() == null) {
            throw new IllegalArgumentException(label + " requires an output storage view.");
        }
        return call.output();
    }
}
