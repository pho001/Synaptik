package backend.cpu.kernels.layout;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.kernels.TypedCpuKernel;
import operations.Operation;
import operations.layout.unfoldAxis;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.TensorMetadata;
import tensor.dtype.TensorDTypeOps;

import java.util.List;

public final class CpuUnfoldAxisKernel extends TypedCpuKernel {
    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        unfold(op, inputs, node);
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        unfold(op, inputs, node);
    }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        unfold(op, inputs, node);
    }

    @Override
    protected void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        unfold(op, inputs, node);
    }

    @Override
    protected void forwardI32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        unfold(op, inputs, node);
    }

    @Override
    protected void forwardI64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        unfold(op, inputs, node);
    }

    private static void unfold(Operation op, List<Tensor> inputs, Tensor node) {
        if (!(op instanceof unfoldAxis unfoldOp)) {
            throw new IllegalArgumentException("CpuUnfoldAxisKernel requires unfoldAxis operation.");
        }
        Tensor input = requireSingleInput(inputs);
        int[] inputShape = input.getShapeUnsafe();
        int rank = inputShape.length;
        int axis = unfoldOp.getAxis();
        int size = unfoldOp.getSize();
        int step = unfoldOp.getStep();
        int windows = node.getShapeUnsafe()[axis];
        int[] prefixShape = inputShape.clone();
        prefixShape[axis] = windows;
        int[] prefixStrides = TensorMetadata.computeStrides(prefixShape);
        int[] inputDenseStrides = TensorMetadata.computeStrides(inputShape);
        for (int outLogical = 0; outLogical < node.getFlatDataSize(); outLogical++) {
            int windowOffset = outLogical % size;
            int prefixLogical = outLogical / size;
            int tmp = prefixLogical;
            int inputLogical = 0;
            for (int d = 0; d < rank; d++) {
                int coord = tmp / prefixStrides[d];
                tmp %= prefixStrides[d];
                int inputCoord = d == axis ? coord * step + windowOffset : coord;
                inputLogical += inputCoord * inputDenseStrides[d];
            }
            write(node, outLogical, input, inputLogical);
        }
        TensorInternalAccess.markStorageModified(node);
    }

    private static Tensor requireSingleInput(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException("unfold expects exactly one input.");
        }
        return inputs.getFirst();
    }

    private static void write(Tensor out, int outIndex, Tensor input, int inputLogicalIndex) {
        DataType dataType = out.getDataType();
        switch (dataType) {
            case FLOAT64 -> TensorInternalAccess.float64Data(out)[outIndex] = input.getByFlatIndex(inputLogicalIndex);
            case FLOAT32 -> TensorInternalAccess.float32Data(out)[outIndex] = (float) input.getByFlatIndex(inputLogicalIndex);
            case BFLOAT16 -> TensorInternalAccess.bfloat16Data(out)[outIndex] =
                    TensorDTypeOps.toBFloat16Bits((float) input.getByFlatIndex(inputLogicalIndex));
            case INT32 -> TensorInternalAccess.int32Data(out)[outIndex] = (int) input.getByFlatIndex(inputLogicalIndex);
            case INT64 -> TensorInternalAccess.int64Data(out)[outIndex] = input.getInt64ByFlatIndex(inputLogicalIndex);
            case BOOL -> TensorInternalAccess.boolData(out)[outIndex] =
                    input.getByFlatIndex(inputLogicalIndex) == 0.0d ? (byte) 0 : (byte) 1;
        }
    }
}
