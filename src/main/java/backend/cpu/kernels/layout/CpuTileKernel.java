package backend.cpu.kernels.layout;

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
            write(node, outLogical, input.getByFlatIndex(inputLogical));
        }
        node.markStorageModified();
    }

    private static Tensor requireSingleInput(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException("tile expects exactly one input.");
        }
        return inputs.getFirst();
    }

    private static void write(Tensor out, int index, double value) {
        switch (out.getDataType()) {
            case FLOAT64 -> out.getFloat64Data()[index] = value;
            case FLOAT32 -> out.getFloat32Data()[index] = (float) value;
            case BFLOAT16 -> out.getBFloat16Data()[index] = CpuDTypeOps.toBFloat16Bits((float) value);
            case INT32 -> out.getInt32Data()[index] = (int) value;
            case BOOL -> out.getBoolData()[index] = value == 0.0d ? (byte) 0 : (byte) 1;
        }
    }
}
