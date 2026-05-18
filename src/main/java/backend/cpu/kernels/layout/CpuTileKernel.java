package backend.cpu.kernels.layout;

import tensor.TensorInternalAccess;

import backend.cpu.kernels.CpuDTypeOps;
import backend.cpu.kernels.CpuKernel;
import backend.cpu.kernels.CpuKernelContext;
import operations.Operation;
import operations.layout.tile;
import tensor.Tensor;
import tensor.TensorMetadata;

import java.util.List;

public final class CpuTileKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        tile(op, inputs, node);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        tile(op, inputs, node);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        tile(op, inputs, node);
    }

    @Override
    public void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        tile(op, inputs, node);
    }

    @Override
    public void forwardI32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        tile(op, inputs, node);
    }

    @Override
    public void forwardI64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        tile(op, inputs, node);
    }

    private static void tile(Operation op, List<Tensor> inputs, Tensor node) {
        if (!(op instanceof tile tileOp)) {
            throw new IllegalArgumentException("CpuTileKernel requires tile operation.");
        }
        Tensor input = requireSingleInput(inputs);
        int[] repeats = tileOp.getRepeats();
        int[] outShape = node.getShapeUnsafe();
        int[] outDenseStrides = TensorMetadata.computeStrides(outShape);
        int[] inputShape = input.getShapeUnsafe();
        int[] inputDenseStrides = TensorMetadata.computeStrides(inputShape);
        for (int outLogical = 0; outLogical < node.getFlatDataSize(); outLogical++) {
            int tmp = outLogical;
            int inputLogical = 0;
            for (int d = 0; d < outShape.length; d++) {
                int coord = tmp / outDenseStrides[d];
                tmp %= outDenseStrides[d];
                inputLogical += (coord % inputShape[d]) * inputDenseStrides[d];
            }
            if (node.getDataType() == tensor.DataType.INT64) {
                TensorInternalAccess.int64Data(node)[outLogical] = input.getInt64ByFlatIndex(inputLogical);
                continue;
            }
            write(node, outLogical, input.getByFlatIndex(inputLogical));
        }
        TensorInternalAccess.markStorageModified(node);
    }

    private static Tensor requireSingleInput(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException("tile expects exactly one input.");
        }
        return inputs.getFirst();
    }

    private static void write(Tensor out, int index, double value) {
        switch (out.getDataType()) {
            case FLOAT64 -> TensorInternalAccess.float64Data(out)[index] = value;
            case FLOAT32 -> TensorInternalAccess.float32Data(out)[index] = (float) value;
            case BFLOAT16 -> TensorInternalAccess.bfloat16Data(out)[index] = CpuDTypeOps.toBFloat16Bits((float) value);
            case INT32 -> TensorInternalAccess.int32Data(out)[index] = (int) value;
            case INT64 -> TensorInternalAccess.int64Data(out)[index] = (long) value;
            case BOOL -> TensorInternalAccess.boolData(out)[index] = value == 0.0d ? (byte) 0 : (byte) 1;
        }
    }
}
