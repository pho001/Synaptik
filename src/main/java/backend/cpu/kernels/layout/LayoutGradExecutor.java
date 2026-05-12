package backend.cpu.kernels.layout;

import backend.cpu.kernels.CpuDTypeOps;
import operations.layout.sliceGrad;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorMetadata;

final class LayoutGradExecutor {
    private LayoutGradExecutor() {
    }

    static void sliceGrad(sliceGrad op, Tensor outGrad, Tensor node) {
        int[] inputShape = op.getInputShape();
        validateShape(node.getShapeUnsafe(), inputShape, "sliceGrad output shape must match original input shape.");
        validateFloating(node.getDataType(), "sliceGrad");
        zero(node);

        int[] outGradShape = outGrad.getShapeUnsafe();
        int[] outGradDense = TensorMetadata.computeStrides(outGradShape);
        int[] inputDense = TensorMetadata.computeStrides(inputShape);
        int[] starts = op.getStarts();
        int[] axes = op.getAxes();
        int[] steps = op.getSteps();
        int total = outGrad.getFlatDataSize();

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
            add(node, inputLogical, outGrad.getByFlatIndex(logical));
        }
        node.markStorageModified();
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
            case FLOAT64 -> java.util.Arrays.fill(node.getFloat64Data(), 0.0d);
            case FLOAT32 -> java.util.Arrays.fill(node.getFloat32Data(), 0.0f);
            case BFLOAT16 -> java.util.Arrays.fill(node.getBFloat16Data(), CpuDTypeOps.toBFloat16Bits(0.0f));
            case INT32, BOOL -> throw new IllegalArgumentException("sliceGrad requires floating output dtype.");
        }
    }

    private static void add(Tensor node, int index, double value) {
        switch (node.getDataType()) {
            case FLOAT64 -> node.getFloat64Data()[index] += value;
            case FLOAT32 -> node.getFloat32Data()[index] += (float) value;
            case BFLOAT16 -> {
                short[] data = node.getBFloat16Data();
                float acc = CpuDTypeOps.fromBFloat16Bits(data[index]) + (float) value;
                data[index] = CpuDTypeOps.toBFloat16Bits(acc);
            }
            case INT32, BOOL -> throw new IllegalArgumentException("sliceGrad requires floating output dtype.");
        }
    }
}
