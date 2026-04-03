package backend.kernels.cpu;

import tensor.DataType;
import tensor.Tensor;
import tensor.TensorMetadata;

public final class GatherSupport {
    private GatherSupport() {
    }

    public static void runF64(Tensor input, Tensor indices, Tensor out, int dimension) {
        validateGather(input, indices, out, dimension);
        double[] in = input.getFloat64Data();
        double[] dst = out.getFloat64Data();
        forEachGather(input, indices, out, dimension, (baseIn, baseOut, axisStrideIn, axisStrideOut, axisIndex) ->
                dst[baseOut] = in[baseIn + axisIndex * axisStrideIn]
        );
    }

    public static void runF32(Tensor input, Tensor indices, Tensor out, int dimension) {
        validateGather(input, indices, out, dimension);
        float[] in = input.getFloat32Data();
        float[] dst = out.getFloat32Data();
        forEachGather(input, indices, out, dimension, (baseIn, baseOut, axisStrideIn, axisStrideOut, axisIndex) ->
                dst[baseOut] = in[baseIn + axisIndex * axisStrideIn]
        );
    }

    public static void runF16(Tensor input, Tensor indices, Tensor out, int dimension) {
        validateGather(input, indices, out, dimension);
        short[] in = input.getFloat16Data();
        short[] dst = out.getFloat16Data();
        forEachGather(input, indices, out, dimension, (baseIn, baseOut, axisStrideIn, axisStrideOut, axisIndex) ->
                dst[baseOut] = in[baseIn + axisIndex * axisStrideIn]
        );
    }

    public static void runBOOL(Tensor input, Tensor indices, Tensor out, int dimension) {
        validateGather(input, indices, out, dimension);
        byte[] in = input.getBoolData();
        byte[] dst = out.getBoolData();
        forEachGather(input, indices, out, dimension, (baseIn, baseOut, axisStrideIn, axisStrideOut, axisIndex) ->
                dst[baseOut] = in[baseIn + axisIndex * axisStrideIn]
        );
    }

    public static void scatterF64(Tensor indices, Tensor outGrad, Tensor node, int dimension) {
        validateScatter(indices, outGrad, node, dimension);
        double[] grad = outGrad.getFloat64Data();
        double[] dst = node.getFloat64Data();
        forEachScatter(indices, outGrad, node, dimension, (baseNode, baseGrad, axisStrideNode, axisStrideGrad, axisIndex) ->
                dst[baseNode + axisIndex * axisStrideNode] += grad[baseGrad]
        );
    }

    public static void scatterF32(Tensor indices, Tensor outGrad, Tensor node, int dimension) {
        validateScatter(indices, outGrad, node, dimension);
        float[] grad = outGrad.getFloat32Data();
        float[] dst = node.getFloat32Data();
        forEachScatter(indices, outGrad, node, dimension, (baseNode, baseGrad, axisStrideNode, axisStrideGrad, axisIndex) ->
                dst[baseNode + axisIndex * axisStrideNode] += grad[baseGrad]
        );
    }

    public static void scatterF16(Tensor indices, Tensor outGrad, Tensor node, int dimension) {
        validateScatter(indices, outGrad, node, dimension);
        short[] grad = outGrad.getFloat16Data();
        short[] dst = node.getFloat16Data();
        forEachScatter(indices, outGrad, node, dimension, (baseNode, baseGrad, axisStrideNode, axisStrideGrad, axisIndex) -> {
            float acc = CpuDTypeOps.fromHalfBits(dst[baseNode + axisIndex * axisStrideNode]) + CpuDTypeOps.fromHalfBits(grad[baseGrad]);
            dst[baseNode + axisIndex * axisStrideNode] = CpuDTypeOps.toHalfBits(acc);
        });
    }

    private static void validateGather(Tensor input, Tensor indices, Tensor out, int dimension) {
        int[] inputShape = input.getShapeUnsafe();
        if (dimension < 0 || dimension >= inputShape.length) {
            throw new IllegalArgumentException("Gather dimension out of bounds: " + dimension);
        }
        validateIndexTensor(indices);
        int[] expectedOutShape = reduceShape(inputShape, dimension);
        validateShape(indices.getShapeUnsafe(), expectedOutShape, "Gather indices shape must equal input shape without gathered axis.");
        validateShape(out.getShapeUnsafe(), expectedOutShape, "Gather output shape must equal indices shape.");
        if (input.getDataType() != out.getDataType()) {
            throw new IllegalArgumentException("Gather output dtype must match input dtype.");
        }
    }

    private static void validateScatter(Tensor indices, Tensor outGrad, Tensor node, int dimension) {
        int[] nodeShape = node.getShapeUnsafe();
        if (dimension < 0 || dimension >= nodeShape.length) {
            throw new IllegalArgumentException("GatherGrad dimension out of bounds: " + dimension);
        }
        validateIndexTensor(indices);
        int[] expectedGradShape = reduceShape(nodeShape, dimension);
        validateShape(indices.getShapeUnsafe(), expectedGradShape, "GatherGrad indices shape must equal gradient shape.");
        validateShape(outGrad.getShapeUnsafe(), expectedGradShape, "GatherGrad outGrad shape must equal indices shape.");
        if (outGrad.getDataType() != node.getDataType()) {
            throw new IllegalArgumentException("GatherGrad output dtype must match outGrad dtype.");
        }
    }

    private static void validateIndexTensor(Tensor indices) {
        if (indices.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("Gather indices must be numeric integral values.");
        }
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

    private static int[] reduceShape(int[] shape, int axis) {
        if (shape.length == 1) {
            return new int[]{1};
        }
        int[] reduced = new int[shape.length - 1];
        for (int i = 0, j = 0; i < shape.length; i++) {
            if (i != axis) {
                reduced[j++] = shape[i];
            }
        }
        return reduced;
    }

    private static void forEachGather(Tensor input, Tensor indices, Tensor out, int dimension, GatherConsumer consumer) {
        int[] inputShape = input.getShapeUnsafe();
        int[] inputStrides = input.getStridesUnsafe();
        int[] outShape = out.getShapeUnsafe();
        int[] outStrides = out.getStridesUnsafe();
        int[] reducedDense = TensorMetadata.computeStrides(outShape);
        int total = out.getFlatDataSize();
        int axisSize = inputShape[dimension];
        int axisStrideIn = inputStrides[dimension];

        for (int logical = 0; logical < total; logical++) {
            int rem = logical;
            int baseIn = 0;
            int baseOut = 0;
            for (int d = 0, rd = 0; d < inputShape.length; d++) {
                if (d == dimension) {
                    continue;
                }
                int coord = rem / reducedDense[rd];
                rem %= reducedDense[rd];
                baseIn += coord * inputStrides[d];
                baseOut += coord * outStrides[rd];
                rd++;
            }
            int axisIndex = readAxisIndex(indices, logical, axisSize);
            consumer.accept(baseIn, baseOut, axisStrideIn, 1, axisIndex);
        }
    }

    private static void forEachScatter(Tensor indices, Tensor outGrad, Tensor node, int dimension, ScatterConsumer consumer) {
        int[] nodeShape = node.getShapeUnsafe();
        int[] nodeStrides = node.getStridesUnsafe();
        int[] gradShape = outGrad.getShapeUnsafe();
        int[] gradStrides = outGrad.getStridesUnsafe();
        int[] reducedDense = TensorMetadata.computeStrides(gradShape);
        int total = outGrad.getFlatDataSize();
        int axisSize = nodeShape[dimension];
        int axisStrideNode = nodeStrides[dimension];

        for (int logical = 0; logical < total; logical++) {
            int rem = logical;
            int baseNode = 0;
            int baseGrad = 0;
            for (int d = 0, rd = 0; d < nodeShape.length; d++) {
                if (d == dimension) {
                    continue;
                }
                int coord = rem / reducedDense[rd];
                rem %= reducedDense[rd];
                baseNode += coord * nodeStrides[d];
                baseGrad += coord * gradStrides[rd];
                rd++;
            }
            int axisIndex = readAxisIndex(indices, logical, axisSize);
            consumer.accept(baseNode, baseGrad, axisStrideNode, 1, axisIndex);
        }
    }

    private static int readAxisIndex(Tensor indices, int logicalIndex, int axisSize) {
        double raw = indices.getByFlatIndex(logicalIndex);
        if (!Double.isFinite(raw)) {
            throw new IllegalArgumentException("Gather index must be finite.");
        }
        long integral = Math.round(raw);
        if (Math.abs(raw - integral) > 1e-9) {
            throw new IllegalArgumentException("Gather index must be an integer value. got=" + raw);
        }
        if (integral < 0 || integral >= axisSize) {
            throw new IllegalArgumentException("Gather index out of bounds: " + integral + " for axis size " + axisSize);
        }
        return (int) integral;
    }

    @FunctionalInterface
    private interface GatherConsumer {
        void accept(int baseIn, int baseOut, int axisStrideIn, int axisStrideOut, int axisIndex);
    }

    @FunctionalInterface
    private interface ScatterConsumer {
        void accept(int baseNode, int baseGrad, int axisStrideNode, int axisStrideGrad, int axisIndex);
    }
}
