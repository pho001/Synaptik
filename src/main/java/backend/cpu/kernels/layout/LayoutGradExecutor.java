package backend.cpu.kernels.layout;

import tensor.TensorInternalAccess;

import tensor.dtype.TensorDTypeOps;
import operations.layout.sliceGrad;
import operations.layout.sliceScatterAdd;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorMetadata;

final class LayoutGradExecutor {
    private LayoutGradExecutor() {
    }

    static void sliceGrad(sliceGrad op, Tensor outGrad, Tensor node) {
        sliceScatterAdd(op.getStarts(), op.getAxes(), op.getSteps(), op.getInputShape(), outGrad, node, "sliceGrad");
    }

    static void sliceScatterAdd(sliceScatterAdd op, Tensor updates, Tensor node) {
        sliceScatterAdd(op.getStarts(), op.getAxes(), op.getSteps(), op.getInputShape(), updates, node, "sliceScatterAdd");
    }

    private static void sliceScatterAdd(
            int[] starts,
            int[] axes,
            int[] steps,
            int[] inputShape,
            Tensor updates,
            Tensor node,
            String opName
    ) {
        validateShape(node.getShapeUnsafe(), inputShape, opName + " output shape must match target input shape.");
        validateFloating(node.getDataType(), opName);
        zero(node);

        int[] outGradShape = updates.getShapeUnsafe();
        int[] outGradDense = TensorMetadata.computeStrides(outGradShape);
        int[] inputDense = TensorMetadata.computeStrides(inputShape);
        int total = updates.getFlatDataSize();

        for (int logical = 0; logical < total; logical++) {
            int rem = logical;
            int inputLogical = 0;
            for (int d = 0; d < outGradShape.length; d++) {
                int coord = rem / outGradDense[d];
                rem %= outGradDense[d];
                int inputCoord = coord;
                for (int i = 0; i < axes.length; i++) {
                    if (axes[i] == d) {
                        inputCoord = starts[i] + coord * steps[i];
                        break;
                    }
                }
                inputLogical += inputCoord * inputDense[d];
            }
            add(node, inputLogical, updates.getByFlatIndex(logical));
        }
        TensorInternalAccess.markStorageModified(node);
    }

    private static void validateShape(int[] actual, int[] expected, String message) {
        if (actual.length != expected.length) {
            throw new IllegalArgumentException(message);
        }
        for (int i = 0; i < actual.length; i++) {
            if (actual[i] != expected[i]) {
                throw new IllegalArgumentException(message);
            }
        }
    }

    private static void validateFloating(DataType type, String opName) {
        if (type == DataType.BOOL || type == DataType.INT32) {
            throw new IllegalArgumentException(opName + " requires floating output dtype.");
        }
    }

    private static void zero(Tensor node) {
        switch (node.getDataType()) {
            case FLOAT64 -> java.util.Arrays.fill(TensorInternalAccess.float64Data(node), 0.0d);
            case FLOAT32 -> java.util.Arrays.fill(TensorInternalAccess.float32Data(node), 0.0f);
            case BFLOAT16 -> java.util.Arrays.fill(TensorInternalAccess.bfloat16Data(node), TensorDTypeOps.toBFloat16Bits(0.0f));
            case INT32, BOOL -> throw new IllegalArgumentException("sliceGrad requires floating output dtype.");
        }
    }

    private static void add(Tensor node, int index, double value) {
        switch (node.getDataType()) {
            case FLOAT64 -> TensorInternalAccess.float64Data(node)[index] += value;
            case FLOAT32 -> TensorInternalAccess.float32Data(node)[index] += (float) value;
            case BFLOAT16 -> {
                short[] data = TensorInternalAccess.bfloat16Data(node);
                float acc = TensorDTypeOps.fromBFloat16Bits(data[index]) + (float) value;
                data[index] = TensorDTypeOps.toBFloat16Bits(acc);
            }
            case INT32, BOOL -> throw new IllegalArgumentException("sliceGrad requires floating output dtype.");
        }
    }
}
