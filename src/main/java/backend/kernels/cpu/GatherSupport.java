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

    public static void runI32(Tensor input, Tensor indices, Tensor out, int dimension) {
        validateGather(input, indices, out, dimension);
        int[] in = input.getInt32Data();
        int[] dst = out.getInt32Data();
        forEachGather(input, indices, out, dimension, (baseIn, baseOut, axisStrideIn, axisStrideOut, axisIndex) ->
                dst[baseOut] = in[baseIn + axisIndex * axisStrideIn]
        );
    }

    public static void takeAlongAxisF64(Tensor input, Tensor indices, Tensor out, int dimension) {
        validateTakeAlongAxis(input, indices, out, dimension);
        double[] in = input.getFloat64Data();
        double[] dst = out.getFloat64Data();
        forEachTakeAlongAxis(input, indices, out, dimension, (baseIn, outOffset, axisStrideIn, axisIndex) ->
                dst[outOffset] = in[baseIn + axisIndex * axisStrideIn]
        );
    }

    public static void takeAlongAxisF32(Tensor input, Tensor indices, Tensor out, int dimension) {
        validateTakeAlongAxis(input, indices, out, dimension);
        float[] in = input.getFloat32Data();
        float[] dst = out.getFloat32Data();
        forEachTakeAlongAxis(input, indices, out, dimension, (baseIn, outOffset, axisStrideIn, axisIndex) ->
                dst[outOffset] = in[baseIn + axisIndex * axisStrideIn]
        );
    }

    public static void takeAlongAxisF16(Tensor input, Tensor indices, Tensor out, int dimension) {
        validateTakeAlongAxis(input, indices, out, dimension);
        short[] in = input.getFloat16Data();
        short[] dst = out.getFloat16Data();
        forEachTakeAlongAxis(input, indices, out, dimension, (baseIn, outOffset, axisStrideIn, axisIndex) ->
                dst[outOffset] = in[baseIn + axisIndex * axisStrideIn]
        );
    }

    public static void takeAlongAxisBOOL(Tensor input, Tensor indices, Tensor out, int dimension) {
        validateTakeAlongAxis(input, indices, out, dimension);
        byte[] in = input.getBoolData();
        byte[] dst = out.getBoolData();
        forEachTakeAlongAxis(input, indices, out, dimension, (baseIn, outOffset, axisStrideIn, axisIndex) ->
                dst[outOffset] = in[baseIn + axisIndex * axisStrideIn]
        );
    }

    public static void takeAlongAxisI32(Tensor input, Tensor indices, Tensor out, int dimension) {
        validateTakeAlongAxis(input, indices, out, dimension);
        int[] in = input.getInt32Data();
        int[] dst = out.getInt32Data();
        forEachTakeAlongAxis(input, indices, out, dimension, (baseIn, outOffset, axisStrideIn, axisIndex) ->
                dst[outOffset] = in[baseIn + axisIndex * axisStrideIn]
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

    public static void scatterAddF64(Tensor base, Tensor indices, Tensor src, Tensor out, int dimension) {
        validateScatterAdd(base, indices, src, out, dimension);
        out.copyDataFrom(base);
        double[] srcData = src.getFloat64Data();
        double[] dst = out.getFloat64Data();
        forEachScatter(indices, src, out, dimension, (baseNode, baseGrad, axisStrideNode, axisStrideGrad, axisIndex) ->
                dst[baseNode + axisIndex * axisStrideNode] += srcData[baseGrad]
        );
    }

    public static void scatterAddF32(Tensor base, Tensor indices, Tensor src, Tensor out, int dimension) {
        validateScatterAdd(base, indices, src, out, dimension);
        out.copyDataFrom(base);
        float[] srcData = src.getFloat32Data();
        float[] dst = out.getFloat32Data();
        forEachScatter(indices, src, out, dimension, (baseNode, baseGrad, axisStrideNode, axisStrideGrad, axisIndex) ->
                dst[baseNode + axisIndex * axisStrideNode] += srcData[baseGrad]
        );
    }

    public static void scatterAddF16(Tensor base, Tensor indices, Tensor src, Tensor out, int dimension) {
        validateScatterAdd(base, indices, src, out, dimension);
        out.copyDataFrom(base);
        short[] srcData = src.getFloat16Data();
        short[] dst = out.getFloat16Data();
        forEachScatter(indices, src, out, dimension, (baseNode, baseGrad, axisStrideNode, axisStrideGrad, axisIndex) -> {
            float acc = CpuDTypeOps.fromHalfBits(dst[baseNode + axisIndex * axisStrideNode]) + CpuDTypeOps.fromHalfBits(srcData[baseGrad]);
            dst[baseNode + axisIndex * axisStrideNode] = CpuDTypeOps.toHalfBits(acc);
        });
    }

    public static void takeAlongAxisScatterF64(Tensor indices, Tensor outGrad, Tensor node, int dimension) {
        validateTakeAlongAxisScatter(indices, outGrad, node, dimension);
        double[] grad = outGrad.getFloat64Data();
        double[] dst = node.getFloat64Data();
        forEachTakeAlongAxisScatter(indices, outGrad, node, dimension, (baseNode, gradOffset, axisStrideNode, axisIndex) ->
                dst[baseNode + axisIndex * axisStrideNode] += grad[gradOffset]
        );
    }

    public static void takeAlongAxisScatterF32(Tensor indices, Tensor outGrad, Tensor node, int dimension) {
        validateTakeAlongAxisScatter(indices, outGrad, node, dimension);
        float[] grad = outGrad.getFloat32Data();
        float[] dst = node.getFloat32Data();
        forEachTakeAlongAxisScatter(indices, outGrad, node, dimension, (baseNode, gradOffset, axisStrideNode, axisIndex) ->
                dst[baseNode + axisIndex * axisStrideNode] += grad[gradOffset]
        );
    }

    public static void takeAlongAxisScatterF16(Tensor indices, Tensor outGrad, Tensor node, int dimension) {
        validateTakeAlongAxisScatter(indices, outGrad, node, dimension);
        short[] grad = outGrad.getFloat16Data();
        short[] dst = node.getFloat16Data();
        forEachTakeAlongAxisScatter(indices, outGrad, node, dimension, (baseNode, gradOffset, axisStrideNode, axisIndex) -> {
            float acc = CpuDTypeOps.fromHalfBits(dst[baseNode + axisIndex * axisStrideNode]) + CpuDTypeOps.fromHalfBits(grad[gradOffset]);
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

    private static void validateScatterAdd(Tensor base, Tensor indices, Tensor src, Tensor out, int dimension) {
        int[] baseShape = base.getShapeUnsafe();
        if (dimension < 0 || dimension >= baseShape.length) {
            throw new IllegalArgumentException("scatterAdd dimension out of bounds: " + dimension);
        }
        validateIndexTensor(indices);
        int[] expectedSrcShape = reduceShape(baseShape, dimension);
        validateShape(indices.getShapeUnsafe(), expectedSrcShape, "scatterAdd indices shape must equal base shape without scattered axis.");
        validateShape(src.getShapeUnsafe(), expectedSrcShape, "scatterAdd source shape must equal indices shape.");
        validateShape(out.getShapeUnsafe(), baseShape, "scatterAdd output shape must equal base shape.");
        if (base.getDataType() == DataType.BOOL || src.getDataType() == DataType.BOOL || out.getDataType() == DataType.BOOL
                || base.getDataType() == DataType.INT32 || src.getDataType() == DataType.INT32 || out.getDataType() == DataType.INT32) {
            throw new IllegalArgumentException("scatterAdd requires floating numeric tensors.");
        }
        if (base.getDataType() != src.getDataType() || base.getDataType() != out.getDataType()) {
            throw new IllegalArgumentException("scatterAdd requires matching dtypes for base, src and output.");
        }
    }

    private static void validateTakeAlongAxis(Tensor input, Tensor indices, Tensor out, int dimension) {
        int[] inputShape = input.getShapeUnsafe();
        if (dimension < 0 || dimension >= inputShape.length) {
            throw new IllegalArgumentException("takeAlongAxis dimension out of bounds: " + dimension);
        }
        validateIndexTensor(indices);
        int[] indicesShape = indices.getShapeUnsafe();
        if (indicesShape.length != inputShape.length) {
            throw new IllegalArgumentException("takeAlongAxis indices rank must match input rank.");
        }
        for (int i = 0; i < inputShape.length; i++) {
            if (i == dimension) {
                continue;
            }
            if (indicesShape[i] != inputShape[i]) {
                throw new IllegalArgumentException("takeAlongAxis indices must match input shape on all non-axis dimensions.");
            }
        }
        validateShape(out.getShapeUnsafe(), indicesShape, "takeAlongAxis output shape must equal indices shape.");
        if (input.getDataType() != out.getDataType()) {
            throw new IllegalArgumentException("takeAlongAxis output dtype must match input dtype.");
        }
    }

    private static void validateTakeAlongAxisScatter(Tensor indices, Tensor outGrad, Tensor node, int dimension) {
        int[] nodeShape = node.getShapeUnsafe();
        if (dimension < 0 || dimension >= nodeShape.length) {
            throw new IllegalArgumentException("takeAlongAxisGrad dimension out of bounds: " + dimension);
        }
        validateIndexTensor(indices);
        int[] gradShape = outGrad.getShapeUnsafe();
        validateShape(indices.getShapeUnsafe(), gradShape, "takeAlongAxisGrad indices shape must equal outGrad shape.");
        if (gradShape.length != nodeShape.length) {
            throw new IllegalArgumentException("takeAlongAxisGrad outGrad rank must match input rank.");
        }
        for (int i = 0; i < nodeShape.length; i++) {
            if (i == dimension) {
                continue;
            }
            if (gradShape[i] != nodeShape[i]) {
                throw new IllegalArgumentException("takeAlongAxisGrad outGrad shape must match input shape on all non-axis dimensions.");
            }
        }
        if (outGrad.getDataType() != node.getDataType()) {
            throw new IllegalArgumentException("takeAlongAxisGrad output dtype must match outGrad dtype.");
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

    private static void forEachTakeAlongAxis(Tensor input, Tensor indices, Tensor out, int dimension, TakeAlongAxisConsumer consumer) {
        int[] inputShape = input.getShapeUnsafe();
        int[] inputStrides = input.getStridesUnsafe();
        int[] outShape = out.getShapeUnsafe();
        int[] outStrides = out.getStridesUnsafe();
        int[] outDense = TensorMetadata.computeStrides(outShape);
        int total = out.getFlatDataSize();
        int axisSize = inputShape[dimension];
        int axisStrideIn = inputStrides[dimension];

        for (int logical = 0; logical < total; logical++) {
            int rem = logical;
            int baseIn = 0;
            int outOffset = 0;
            for (int d = 0; d < outShape.length; d++) {
                int coord = rem / outDense[d];
                rem %= outDense[d];
                outOffset += coord * outStrides[d];
                if (d != dimension) {
                    baseIn += coord * inputStrides[d];
                }
            }
            int axisIndex = readAxisIndex(indices, logical, axisSize);
            consumer.accept(baseIn, outOffset, axisStrideIn, axisIndex);
        }
    }

    private static void forEachTakeAlongAxisScatter(Tensor indices, Tensor outGrad, Tensor node, int dimension, TakeAlongAxisScatterConsumer consumer) {
        int[] nodeShape = node.getShapeUnsafe();
        int[] nodeStrides = node.getStridesUnsafe();
        int[] gradShape = outGrad.getShapeUnsafe();
        int[] gradStrides = outGrad.getStridesUnsafe();
        int[] gradDense = TensorMetadata.computeStrides(gradShape);
        int total = outGrad.getFlatDataSize();
        int axisSize = nodeShape[dimension];
        int axisStrideNode = nodeStrides[dimension];

        for (int logical = 0; logical < total; logical++) {
            int rem = logical;
            int baseNode = 0;
            int gradOffset = 0;
            for (int d = 0; d < gradShape.length; d++) {
                int coord = rem / gradDense[d];
                rem %= gradDense[d];
                gradOffset += coord * gradStrides[d];
                if (d != dimension) {
                    baseNode += coord * nodeStrides[d];
                }
            }
            int axisIndex = readAxisIndex(indices, logical, axisSize);
            consumer.accept(baseNode, gradOffset, axisStrideNode, axisIndex);
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

    @FunctionalInterface
    private interface TakeAlongAxisConsumer {
        void accept(int baseIn, int outOffset, int axisStrideIn, int axisIndex);
    }

    @FunctionalInterface
    private interface TakeAlongAxisScatterConsumer {
        void accept(int baseNode, int gradOffset, int axisStrideNode, int axisIndex);
    }
}
