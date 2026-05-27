package backend.cpu.kernels.reduction;

import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import backend.cpu.storage.CpuStorageView;
import operations.reduction.reduceProd;
import tensor.DataType;
import tensor.Tensor;
import tensor.dtype.TensorDTypeOps;

import java.lang.foreign.MemorySegment;
import java.util.List;

public final class CpuReduceProdKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        if (!(call.operation() instanceof reduceProd reduction)) {
            throw new IllegalArgumentException("CpuReduceProdKernel requires reduceProd operation.");
        }
        Tensor inputTensor = requireSingleInput(call.inputTensors(), "ReduceProd");
        CpuStorageView input = requireSingleInputView(call, "ReduceProd");
        CpuStorageView output = requireOutputView(call, "ReduceProd");
        if (input.dtype() != output.dtype()) {
            throw new IllegalArgumentException("ReduceProd requires input and output dtypes to match.");
        }

        int dimension = reduction.getDimension();
        int[] inputShape = input.shape();
        if (dimension < -1 || dimension >= inputShape.length) {
            throw new IllegalArgumentException("Dimension out of bounds: " + dimension);
        }
        if (inputTensor.getFlatDataSize() != input.logicalSize()) {
            throw new IllegalArgumentException("ReduceProd input storage view size does not match input tensor.");
        }

        switch (output.dtype()) {
            case FLOAT64 -> reduceF64(input, output, dimension);
            case FLOAT32 -> reduceF32(input, output, dimension);
            case BFLOAT16 -> reduceBF16(input, output, dimension);
            case INT32, INT64, BOOL -> throw new IllegalArgumentException("ReduceProd requires floating output.");
        }
        return CpuKernelResult.completed();
    }

    private static void reduceF64(CpuStorageView input, CpuStorageView output, int dimension) {
        double[] inArray = ReductionStorageAccess.f64Array(input);
        MemorySegment inSegment = ReductionStorageAccess.f64Segment(input);
        double[] outArray = ReductionStorageAccess.f64Array(output);
        MemorySegment outSegment = ReductionStorageAccess.f64Segment(output);
        int[] inputShape = input.shape();
        int[] inputStrides = input.strides();
        int[] outputShape = output.shape();
        int[] outputStrides = output.strides();

        if (dimension == -1) {
            double product = 1.0d;
            for (int logical = 0; logical < input.logicalSize(); logical++) {
                int inputOffset = ReductionStorageAccess.logicalToOffset(logical, inputShape, inputStrides, input.storageOffset());
                product *= ReductionStorageAccess.readF64(inArray, inSegment, inputOffset);
            }
            int outOffset = ReductionStorageAccess.logicalToOffset(0, outputShape, outputStrides, output.storageOffset());
            ReductionStorageAccess.writeF64(outArray, outSegment, outOffset, product);
            return;
        }

        int groups = output.logicalSize();
        int reducedSize = inputShape[dimension];
        int reducedStride = inputStrides[dimension];
        for (int outLogical = 0; outLogical < groups; outLogical++) {
            int inputBase = axisBaseOffset(outLogical, inputShape, inputStrides, input.storageOffset(), outputShape, dimension);
            double product = 1.0d;
            for (int r = 0; r < reducedSize; r++) {
                product *= ReductionStorageAccess.readF64(inArray, inSegment, inputBase + r * reducedStride);
            }
            int outOffset = ReductionStorageAccess.logicalToOffset(outLogical, outputShape, outputStrides, output.storageOffset());
            ReductionStorageAccess.writeF64(outArray, outSegment, outOffset, product);
        }
    }

    private static void reduceF32(CpuStorageView input, CpuStorageView output, int dimension) {
        float[] inArray = ReductionStorageAccess.f32Array(input);
        MemorySegment inSegment = ReductionStorageAccess.f32Segment(input);
        float[] outArray = ReductionStorageAccess.f32Array(output);
        MemorySegment outSegment = ReductionStorageAccess.f32Segment(output);
        int[] inputShape = input.shape();
        int[] inputStrides = input.strides();
        int[] outputShape = output.shape();
        int[] outputStrides = output.strides();

        if (dimension == -1) {
            float product = 1.0f;
            for (int logical = 0; logical < input.logicalSize(); logical++) {
                int inputOffset = ReductionStorageAccess.logicalToOffset(logical, inputShape, inputStrides, input.storageOffset());
                product *= ReductionStorageAccess.readF32(inArray, inSegment, inputOffset);
            }
            int outOffset = ReductionStorageAccess.logicalToOffset(0, outputShape, outputStrides, output.storageOffset());
            ReductionStorageAccess.writeF32(outArray, outSegment, outOffset, product);
            return;
        }

        int groups = output.logicalSize();
        int reducedSize = inputShape[dimension];
        int reducedStride = inputStrides[dimension];
        for (int outLogical = 0; outLogical < groups; outLogical++) {
            int inputBase = axisBaseOffset(outLogical, inputShape, inputStrides, input.storageOffset(), outputShape, dimension);
            float product = 1.0f;
            for (int r = 0; r < reducedSize; r++) {
                product *= ReductionStorageAccess.readF32(inArray, inSegment, inputBase + r * reducedStride);
            }
            int outOffset = ReductionStorageAccess.logicalToOffset(outLogical, outputShape, outputStrides, output.storageOffset());
            ReductionStorageAccess.writeF32(outArray, outSegment, outOffset, product);
        }
    }

    private static void reduceBF16(CpuStorageView input, CpuStorageView output, int dimension) {
        short[] inArray = ReductionStorageAccess.bf16Array(input);
        MemorySegment inSegment = ReductionStorageAccess.bf16Segment(input);
        short[] outArray = ReductionStorageAccess.bf16Array(output);
        MemorySegment outSegment = ReductionStorageAccess.bf16Segment(output);
        int[] inputShape = input.shape();
        int[] inputStrides = input.strides();
        int[] outputShape = output.shape();
        int[] outputStrides = output.strides();

        if (dimension == -1) {
            short product = TensorDTypeOps.toBFloat16Bits(1.0f);
            for (int logical = 0; logical < input.logicalSize(); logical++) {
                int inputOffset = ReductionStorageAccess.logicalToOffset(logical, inputShape, inputStrides, input.storageOffset());
                product = TensorDTypeOps.toBFloat16Bits(
                        TensorDTypeOps.fromBFloat16Bits(product)
                                * TensorDTypeOps.fromBFloat16Bits(ReductionStorageAccess.readBF16(inArray, inSegment, inputOffset))
                );
            }
            int outOffset = ReductionStorageAccess.logicalToOffset(0, outputShape, outputStrides, output.storageOffset());
            ReductionStorageAccess.writeBF16(outArray, outSegment, outOffset, product);
            return;
        }

        int groups = output.logicalSize();
        int reducedSize = inputShape[dimension];
        int reducedStride = inputStrides[dimension];
        for (int outLogical = 0; outLogical < groups; outLogical++) {
            int inputBase = axisBaseOffset(outLogical, inputShape, inputStrides, input.storageOffset(), outputShape, dimension);
            short product = TensorDTypeOps.toBFloat16Bits(1.0f);
            for (int r = 0; r < reducedSize; r++) {
                product = TensorDTypeOps.toBFloat16Bits(
                        TensorDTypeOps.fromBFloat16Bits(product)
                                * TensorDTypeOps.fromBFloat16Bits(ReductionStorageAccess.readBF16(inArray, inSegment, inputBase + r * reducedStride))
                );
            }
            int outOffset = ReductionStorageAccess.logicalToOffset(outLogical, outputShape, outputStrides, output.storageOffset());
            ReductionStorageAccess.writeBF16(outArray, outSegment, outOffset, product);
        }
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
