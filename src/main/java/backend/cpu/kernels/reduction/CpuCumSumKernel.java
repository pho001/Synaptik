package backend.cpu.kernels.reduction;

import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import backend.cpu.storage.CpuStorageView;
import operations.reduction.cumSum;
import tensor.DataType;
import tensor.Tensor;
import tensor.dtype.TensorDTypeOps;

import java.lang.foreign.MemorySegment;
import java.util.List;

public final class CpuCumSumKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        if (!(call.operation() instanceof cumSum scan)) {
            throw new IllegalArgumentException("CpuCumSumKernel requires cumSum operation.");
        }
        requireSingleInput(call.inputTensors(), "CumSum");
        CpuStorageView input = requireSingleInputView(call, "CumSum");
        CpuStorageView output = requireOutputView(call, "CumSum");
        if (input.dtype() == DataType.BOOL || output.dtype() == DataType.BOOL) {
            throw new IllegalArgumentException("CumSum requires floating or integer tensors.");
        }
        if (input.dtype() != output.dtype()) {
            throw new IllegalArgumentException("CumSum requires input and output dtypes to match.");
        }

        int[] shape = input.shape();
        int axis = scan.getAxis();
        if (axis < 0 || axis >= shape.length) {
            throw new IllegalArgumentException("CumSum axis out of bounds: " + axis);
        }

        switch (input.dtype()) {
            case FLOAT64 -> scanF64(input, output, axis, scan.isExclusive(), scan.isReverse());
            case FLOAT32 -> scanF32(input, output, axis, scan.isExclusive(), scan.isReverse());
            case BFLOAT16 -> scanBF16(input, output, axis, scan.isExclusive(), scan.isReverse());
            case INT32 -> scanI32(input, output, axis, scan.isExclusive(), scan.isReverse());
            case INT64 -> scanI64(input, output, axis, scan.isExclusive(), scan.isReverse());
            case BOOL -> throw new IllegalArgumentException("CumSum requires floating or integer tensors.");
        }
        return CpuKernelResult.completed();
    }

    private static void scanF64(CpuStorageView input, CpuStorageView output, int axis, boolean exclusive, boolean reverse) {
        double[] inArray = ReductionStorageAccess.f64Array(input);
        MemorySegment inSegment = ReductionStorageAccess.f64Segment(input);
        double[] outArray = ReductionStorageAccess.f64Array(output);
        MemorySegment outSegment = ReductionStorageAccess.f64Segment(output);
        int[] shape = input.shape();
        int[] inputStrides = input.strides();
        int[] outputShape = output.shape();
        int[] outputStrides = output.strides();
        int axisSize = shape[axis];
        int lineCount = axisSize == 0 ? 0 : input.logicalSize() / axisSize;

        for (int line = 0; line < lineCount; line++) {
            double acc = 0.0d;
            if (reverse) {
                for (int k = axisSize - 1; k >= 0; k--) {
                    int logical = logicalIndex(line, k, axis, shape);
                    int inputOffset = ReductionStorageAccess.logicalToOffset(logical, shape, inputStrides, input.storageOffset());
                    int outputOffset = ReductionStorageAccess.logicalToOffset(logical, outputShape, outputStrides, output.storageOffset());
                    double value = ReductionStorageAccess.readF64(inArray, inSegment, inputOffset);
                    if (exclusive) {
                        ReductionStorageAccess.writeF64(outArray, outSegment, outputOffset, acc);
                        acc += value;
                    } else {
                        acc += value;
                        ReductionStorageAccess.writeF64(outArray, outSegment, outputOffset, acc);
                    }
                }
            } else {
                for (int k = 0; k < axisSize; k++) {
                    int logical = logicalIndex(line, k, axis, shape);
                    int inputOffset = ReductionStorageAccess.logicalToOffset(logical, shape, inputStrides, input.storageOffset());
                    int outputOffset = ReductionStorageAccess.logicalToOffset(logical, outputShape, outputStrides, output.storageOffset());
                    double value = ReductionStorageAccess.readF64(inArray, inSegment, inputOffset);
                    if (exclusive) {
                        ReductionStorageAccess.writeF64(outArray, outSegment, outputOffset, acc);
                        acc += value;
                    } else {
                        acc += value;
                        ReductionStorageAccess.writeF64(outArray, outSegment, outputOffset, acc);
                    }
                }
            }
        }
    }

    private static void scanF32(CpuStorageView input, CpuStorageView output, int axis, boolean exclusive, boolean reverse) {
        float[] inArray = ReductionStorageAccess.f32Array(input);
        MemorySegment inSegment = ReductionStorageAccess.f32Segment(input);
        float[] outArray = ReductionStorageAccess.f32Array(output);
        MemorySegment outSegment = ReductionStorageAccess.f32Segment(output);
        int[] shape = input.shape();
        int[] inputStrides = input.strides();
        int[] outputShape = output.shape();
        int[] outputStrides = output.strides();
        int axisSize = shape[axis];
        int lineCount = axisSize == 0 ? 0 : input.logicalSize() / axisSize;

        for (int line = 0; line < lineCount; line++) {
            double acc = 0.0d;
            if (reverse) {
                for (int k = axisSize - 1; k >= 0; k--) {
                    int logical = logicalIndex(line, k, axis, shape);
                    int inputOffset = ReductionStorageAccess.logicalToOffset(logical, shape, inputStrides, input.storageOffset());
                    int outputOffset = ReductionStorageAccess.logicalToOffset(logical, outputShape, outputStrides, output.storageOffset());
                    float value = ReductionStorageAccess.readF32(inArray, inSegment, inputOffset);
                    if (exclusive) {
                        ReductionStorageAccess.writeF32(outArray, outSegment, outputOffset, (float) acc);
                        acc += value;
                    } else {
                        acc += value;
                        ReductionStorageAccess.writeF32(outArray, outSegment, outputOffset, (float) acc);
                    }
                }
            } else {
                for (int k = 0; k < axisSize; k++) {
                    int logical = logicalIndex(line, k, axis, shape);
                    int inputOffset = ReductionStorageAccess.logicalToOffset(logical, shape, inputStrides, input.storageOffset());
                    int outputOffset = ReductionStorageAccess.logicalToOffset(logical, outputShape, outputStrides, output.storageOffset());
                    float value = ReductionStorageAccess.readF32(inArray, inSegment, inputOffset);
                    if (exclusive) {
                        ReductionStorageAccess.writeF32(outArray, outSegment, outputOffset, (float) acc);
                        acc += value;
                    } else {
                        acc += value;
                        ReductionStorageAccess.writeF32(outArray, outSegment, outputOffset, (float) acc);
                    }
                }
            }
        }
    }

    private static void scanBF16(CpuStorageView input, CpuStorageView output, int axis, boolean exclusive, boolean reverse) {
        short[] inArray = ReductionStorageAccess.bf16Array(input);
        MemorySegment inSegment = ReductionStorageAccess.bf16Segment(input);
        short[] outArray = ReductionStorageAccess.bf16Array(output);
        MemorySegment outSegment = ReductionStorageAccess.bf16Segment(output);
        int[] shape = input.shape();
        int[] inputStrides = input.strides();
        int[] outputShape = output.shape();
        int[] outputStrides = output.strides();
        int axisSize = shape[axis];
        int lineCount = axisSize == 0 ? 0 : input.logicalSize() / axisSize;

        for (int line = 0; line < lineCount; line++) {
            double acc = 0.0d;
            if (reverse) {
                for (int k = axisSize - 1; k >= 0; k--) {
                    int logical = logicalIndex(line, k, axis, shape);
                    int inputOffset = ReductionStorageAccess.logicalToOffset(logical, shape, inputStrides, input.storageOffset());
                    int outputOffset = ReductionStorageAccess.logicalToOffset(logical, outputShape, outputStrides, output.storageOffset());
                    float value = TensorDTypeOps.fromBFloat16Bits(ReductionStorageAccess.readBF16(inArray, inSegment, inputOffset));
                    if (exclusive) {
                        ReductionStorageAccess.writeBF16(outArray, outSegment, outputOffset,
                                TensorDTypeOps.toBFloat16Bits((float) acc));
                        acc += value;
                    } else {
                        acc += value;
                        ReductionStorageAccess.writeBF16(outArray, outSegment, outputOffset,
                                TensorDTypeOps.toBFloat16Bits((float) acc));
                    }
                }
            } else {
                for (int k = 0; k < axisSize; k++) {
                    int logical = logicalIndex(line, k, axis, shape);
                    int inputOffset = ReductionStorageAccess.logicalToOffset(logical, shape, inputStrides, input.storageOffset());
                    int outputOffset = ReductionStorageAccess.logicalToOffset(logical, outputShape, outputStrides, output.storageOffset());
                    float value = TensorDTypeOps.fromBFloat16Bits(ReductionStorageAccess.readBF16(inArray, inSegment, inputOffset));
                    if (exclusive) {
                        ReductionStorageAccess.writeBF16(outArray, outSegment, outputOffset,
                                TensorDTypeOps.toBFloat16Bits((float) acc));
                        acc += value;
                    } else {
                        acc += value;
                        ReductionStorageAccess.writeBF16(outArray, outSegment, outputOffset,
                                TensorDTypeOps.toBFloat16Bits((float) acc));
                    }
                }
            }
        }
    }

    private static void scanI32(CpuStorageView input, CpuStorageView output, int axis, boolean exclusive, boolean reverse) {
        int[] inArray = ReductionStorageAccess.i32Array(input);
        MemorySegment inSegment = ReductionStorageAccess.i32Segment(input);
        int[] outArray = ReductionStorageAccess.i32Array(output);
        MemorySegment outSegment = ReductionStorageAccess.i32Segment(output);
        int[] shape = input.shape();
        int[] inputStrides = input.strides();
        int[] outputShape = output.shape();
        int[] outputStrides = output.strides();
        int axisSize = shape[axis];
        int lineCount = axisSize == 0 ? 0 : input.logicalSize() / axisSize;

        for (int line = 0; line < lineCount; line++) {
            double acc = 0.0d;
            if (reverse) {
                for (int k = axisSize - 1; k >= 0; k--) {
                    int logical = logicalIndex(line, k, axis, shape);
                    int inputOffset = ReductionStorageAccess.logicalToOffset(logical, shape, inputStrides, input.storageOffset());
                    int outputOffset = ReductionStorageAccess.logicalToOffset(logical, outputShape, outputStrides, output.storageOffset());
                    int value = ReductionStorageAccess.readI32(inArray, inSegment, inputOffset);
                    if (exclusive) {
                        ReductionStorageAccess.writeI32(outArray, outSegment, outputOffset, (int) acc);
                        acc += value;
                    } else {
                        acc += value;
                        ReductionStorageAccess.writeI32(outArray, outSegment, outputOffset, (int) acc);
                    }
                }
            } else {
                for (int k = 0; k < axisSize; k++) {
                    int logical = logicalIndex(line, k, axis, shape);
                    int inputOffset = ReductionStorageAccess.logicalToOffset(logical, shape, inputStrides, input.storageOffset());
                    int outputOffset = ReductionStorageAccess.logicalToOffset(logical, outputShape, outputStrides, output.storageOffset());
                    int value = ReductionStorageAccess.readI32(inArray, inSegment, inputOffset);
                    if (exclusive) {
                        ReductionStorageAccess.writeI32(outArray, outSegment, outputOffset, (int) acc);
                        acc += value;
                    } else {
                        acc += value;
                        ReductionStorageAccess.writeI32(outArray, outSegment, outputOffset, (int) acc);
                    }
                }
            }
        }
    }

    private static void scanI64(CpuStorageView input, CpuStorageView output, int axis, boolean exclusive, boolean reverse) {
        long[] inArray = ReductionStorageAccess.i64Array(input);
        MemorySegment inSegment = ReductionStorageAccess.i64Segment(input);
        long[] outArray = ReductionStorageAccess.i64Array(output);
        MemorySegment outSegment = ReductionStorageAccess.i64Segment(output);
        int[] shape = input.shape();
        int[] inputStrides = input.strides();
        int[] outputShape = output.shape();
        int[] outputStrides = output.strides();
        int axisSize = shape[axis];
        int lineCount = axisSize == 0 ? 0 : input.logicalSize() / axisSize;

        for (int line = 0; line < lineCount; line++) {
            long acc = 0L;
            if (reverse) {
                for (int k = axisSize - 1; k >= 0; k--) {
                    int logical = logicalIndex(line, k, axis, shape);
                    int inputOffset = ReductionStorageAccess.logicalToOffset(logical, shape, inputStrides, input.storageOffset());
                    int outputOffset = ReductionStorageAccess.logicalToOffset(logical, outputShape, outputStrides, output.storageOffset());
                    long value = ReductionStorageAccess.readI64(inArray, inSegment, inputOffset);
                    if (exclusive) {
                        ReductionStorageAccess.writeI64(outArray, outSegment, outputOffset, acc);
                        acc += value;
                    } else {
                        acc += value;
                        ReductionStorageAccess.writeI64(outArray, outSegment, outputOffset, acc);
                    }
                }
            } else {
                for (int k = 0; k < axisSize; k++) {
                    int logical = logicalIndex(line, k, axis, shape);
                    int inputOffset = ReductionStorageAccess.logicalToOffset(logical, shape, inputStrides, input.storageOffset());
                    int outputOffset = ReductionStorageAccess.logicalToOffset(logical, outputShape, outputStrides, output.storageOffset());
                    long value = ReductionStorageAccess.readI64(inArray, inSegment, inputOffset);
                    if (exclusive) {
                        ReductionStorageAccess.writeI64(outArray, outSegment, outputOffset, acc);
                        acc += value;
                    } else {
                        acc += value;
                        ReductionStorageAccess.writeI64(outArray, outSegment, outputOffset, acc);
                    }
                }
            }
        }
    }

    private static int logicalIndex(int line, int axisCoord, int axis, int[] shape) {
        int logical = 0;
        int stride = 1;
        int remaining = line;
        for (int dim = shape.length - 1; dim >= 0; dim--) {
            int coord;
            if (dim == axis) {
                coord = axisCoord;
            } else {
                coord = remaining % shape[dim];
                remaining /= shape[dim];
            }
            logical += coord * stride;
            stride *= shape[dim];
        }
        return logical;
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
