package backend.cpu.kernels.reduction;

import tensor.TensorInternalAccess;

import tensor.dtype.TensorDTypeOps;
import backend.cpu.kernels.TypedCpuKernel;
import backend.cpu.execution.CpuKernelContext;
import operations.Operation;
import operations.reduction.cumSum;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorMetadata;

import java.util.List;

public final class CpuCumSumKernel extends TypedCpuKernel {
    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        scan(op, inputs, node);
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        scan(op, inputs, node);
    }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        scan(op, inputs, node);
    }

    @Override
    protected void forwardI32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        scan(op, inputs, node);
    }

    @Override
    protected void forwardI64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        scan(op, inputs, node);
    }

    private static void scan(Operation op, List<Tensor> inputs, Tensor node) {
        if (!(op instanceof cumSum scan)) {
            throw new IllegalArgumentException("CpuCumSumKernel requires cumSum operation.");
        }
        Tensor input = CpuSumKernel.requireSingleInput(inputs, "CumSum");
        if (input.getDataType() == DataType.BOOL || node.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("CumSum requires floating or integer tensors.");
        }
        if (input.getDataType() != node.getDataType()) {
            throw new IllegalArgumentException("CumSum requires input and output dtypes to match.");
        }
        int[] shape = input.getShapeUnsafe();
        int axis = scan.getAxis();
        if (axis < 0 || axis >= shape.length) {
            throw new IllegalArgumentException("CumSum axis out of bounds: " + axis);
        }
        int axisSize = shape[axis];
        int lineCount = axisSize == 0 ? 0 : input.getFlatDataSize() / axisSize;
        int[] denseStrides = TensorMetadata.computeStrides(shape);
        if (input.getDataType() == DataType.INT64) {
            for (int line = 0; line < lineCount; line++) {
                if (scan.isReverse()) {
                    scanReverseLongLine(input, node, line, axis, axisSize, denseStrides, scan.isExclusive());
                } else {
                    scanForwardLongLine(input, node, line, axis, axisSize, denseStrides, scan.isExclusive());
                }
            }
            TensorInternalAccess.markStorageModified(node);
            return;
        }
        for (int line = 0; line < lineCount; line++) {
            if (scan.isReverse()) {
                scanReverseLine(input, node, line, axis, axisSize, denseStrides, scan.isExclusive());
            } else {
                scanForwardLine(input, node, line, axis, axisSize, denseStrides, scan.isExclusive());
            }
        }
        TensorInternalAccess.markStorageModified(node);
    }

    private static void scanForwardLine(
            Tensor input,
            Tensor output,
            int line,
            int axis,
            int axisSize,
            int[] denseStrides,
            boolean exclusive
    ) {
        double acc = 0.0d;
        for (int k = 0; k < axisSize; k++) {
            int logical = logicalIndex(line, k, axis, input.getShapeUnsafe(), denseStrides);
            double value = input.getByFlatIndex(logical);
            if (exclusive) {
                write(output, logical, acc);
                acc += value;
            } else {
                acc += value;
                write(output, logical, acc);
            }
        }
    }

    private static void scanForwardLongLine(
            Tensor input,
            Tensor output,
            int line,
            int axis,
            int axisSize,
            int[] denseStrides,
            boolean exclusive
    ) {
        long acc = 0L;
        for (int k = 0; k < axisSize; k++) {
            int logical = logicalIndex(line, k, axis, input.getShapeUnsafe(), denseStrides);
            long value = input.getInt64ByFlatIndex(logical);
            if (exclusive) {
                writeLong(output, logical, acc);
                acc += value;
            } else {
                acc += value;
                writeLong(output, logical, acc);
            }
        }
    }

    private static void scanReverseLongLine(
            Tensor input,
            Tensor output,
            int line,
            int axis,
            int axisSize,
            int[] denseStrides,
            boolean exclusive
    ) {
        long acc = 0L;
        for (int k = axisSize - 1; k >= 0; k--) {
            int logical = logicalIndex(line, k, axis, input.getShapeUnsafe(), denseStrides);
            long value = input.getInt64ByFlatIndex(logical);
            if (exclusive) {
                writeLong(output, logical, acc);
                acc += value;
            } else {
                acc += value;
                writeLong(output, logical, acc);
            }
        }
    }

    private static void scanReverseLine(
            Tensor input,
            Tensor output,
            int line,
            int axis,
            int axisSize,
            int[] denseStrides,
            boolean exclusive
    ) {
        double acc = 0.0d;
        for (int k = axisSize - 1; k >= 0; k--) {
            int logical = logicalIndex(line, k, axis, input.getShapeUnsafe(), denseStrides);
            double value = input.getByFlatIndex(logical);
            if (exclusive) {
                write(output, logical, acc);
                acc += value;
            } else {
                acc += value;
                write(output, logical, acc);
            }
        }
    }

    private static int logicalIndex(int line, int axisCoord, int axis, int[] shape, int[] denseStrides) {
        int tmp = line;
        int logical = axisCoord * denseStrides[axis];
        for (int d = shape.length - 1; d >= 0; d--) {
            if (d == axis) {
                continue;
            }
            int coord = tmp % shape[d];
            tmp /= shape[d];
            logical += coord * denseStrides[d];
        }
        return logical;
    }

    private static void write(Tensor out, int logical, double value) {
        int offset = out.getStorageOffsetUnsafe() + logical;
        switch (out.getDataType()) {
            case FLOAT64 -> TensorInternalAccess.float64Data(out)[offset] = value;
            case FLOAT32 -> TensorInternalAccess.float32Data(out)[offset] = (float) value;
            case BFLOAT16 -> TensorInternalAccess.bfloat16Data(out)[offset] = TensorDTypeOps.toBFloat16Bits((float) value);
            case INT32 -> TensorInternalAccess.int32Data(out)[offset] = (int) value;
            case INT64 -> TensorInternalAccess.int64Data(out)[offset] = (long) value;
            case BOOL -> throw new IllegalArgumentException("CumSum requires floating or integer output.");
        }
    }

    private static void writeLong(Tensor out, int logical, long value) {
        int offset = out.getStorageOffsetUnsafe() + logical;
        TensorInternalAccess.int64Data(out)[offset] = value;
    }
}
