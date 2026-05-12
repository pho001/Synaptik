package backend.cpu.kernels.index;

import backend.cpu.kernels.*;
import operations.index.ScatterReduction;

import tensor.DataType;
import tensor.Tensor;
import tensor.TensorMetadata;

final class IndexReadWriteBackend {
    private IndexReadWriteBackend() {
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

    public static void runBF16(Tensor input, Tensor indices, Tensor out, int dimension) {
        validateGather(input, indices, out, dimension);
        short[] in = input.getBFloat16Data();
        short[] dst = out.getBFloat16Data();
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

    public static void gatherAxisF64(Tensor input, Tensor indices, Tensor out, int axis) {
        validateGatherAxis(input, indices, out, axis);
        double[] dst = out.getFloat64Data();
        forEachGatherAxis(input, indices, out, axis, (sourceLogical, outLogical) ->
                dst[outLogical] = input.getByFlatIndex(sourceLogical)
        );
    }

    public static void gatherAxisF32(Tensor input, Tensor indices, Tensor out, int axis) {
        validateGatherAxis(input, indices, out, axis);
        float[] dst = out.getFloat32Data();
        forEachGatherAxis(input, indices, out, axis, (sourceLogical, outLogical) ->
                dst[outLogical] = (float) input.getByFlatIndex(sourceLogical)
        );
    }

    public static void gatherAxisBF16(Tensor input, Tensor indices, Tensor out, int axis) {
        validateGatherAxis(input, indices, out, axis);
        short[] dst = out.getBFloat16Data();
        forEachGatherAxis(input, indices, out, axis, (sourceLogical, outLogical) ->
                dst[outLogical] = CpuDTypeOps.toBFloat16Bits((float) input.getByFlatIndex(sourceLogical))
        );
    }

    public static void gatherAxisBOOL(Tensor input, Tensor indices, Tensor out, int axis) {
        validateGatherAxis(input, indices, out, axis);
        byte[] dst = out.getBoolData();
        forEachGatherAxis(input, indices, out, axis, (sourceLogical, outLogical) ->
                dst[outLogical] = input.getByFlatIndex(sourceLogical) == 0.0d ? (byte) 0 : (byte) 1
        );
    }

    public static void gatherAxisI32(Tensor input, Tensor indices, Tensor out, int axis) {
        validateGatherAxis(input, indices, out, axis);
        int[] dst = out.getInt32Data();
        forEachGatherAxis(input, indices, out, axis, (sourceLogical, outLogical) ->
                dst[outLogical] = (int) input.getByFlatIndex(sourceLogical)
        );
    }

    public static void gatherNdF64(Tensor input, Tensor indices, Tensor out, int batchDims) {
        validateGatherNd(input, indices, out, batchDims);
        double[] in = input.getFloat64Data();
        double[] dst = out.getFloat64Data();
        forEachGatherNd(input, indices, out, batchDims, (sourceOffset, outOffset) ->
                dst[outOffset] = in[sourceOffset]
        );
    }

    public static void gatherNdF32(Tensor input, Tensor indices, Tensor out, int batchDims) {
        validateGatherNd(input, indices, out, batchDims);
        float[] in = input.getFloat32Data();
        float[] dst = out.getFloat32Data();
        forEachGatherNd(input, indices, out, batchDims, (sourceOffset, outOffset) ->
                dst[outOffset] = in[sourceOffset]
        );
    }

    public static void gatherNdBF16(Tensor input, Tensor indices, Tensor out, int batchDims) {
        validateGatherNd(input, indices, out, batchDims);
        short[] in = input.getBFloat16Data();
        short[] dst = out.getBFloat16Data();
        forEachGatherNd(input, indices, out, batchDims, (sourceOffset, outOffset) ->
                dst[outOffset] = in[sourceOffset]
        );
    }

    public static void gatherNdBOOL(Tensor input, Tensor indices, Tensor out, int batchDims) {
        validateGatherNd(input, indices, out, batchDims);
        byte[] in = input.getBoolData();
        byte[] dst = out.getBoolData();
        forEachGatherNd(input, indices, out, batchDims, (sourceOffset, outOffset) ->
                dst[outOffset] = in[sourceOffset]
        );
    }

    public static void gatherNdI32(Tensor input, Tensor indices, Tensor out, int batchDims) {
        validateGatherNd(input, indices, out, batchDims);
        int[] in = input.getInt32Data();
        int[] dst = out.getInt32Data();
        forEachGatherNd(input, indices, out, batchDims, (sourceOffset, outOffset) ->
                dst[outOffset] = in[sourceOffset]
        );
    }

    public static void gatherNdGradF64(Tensor indices, Tensor outGrad, Tensor node, int batchDims) {
        validateGatherNdGrad(indices, outGrad, node, batchDims);
        java.util.Arrays.fill(node.getFloat64Data(), 0.0d);
        double[] grad = outGrad.getFloat64Data();
        double[] dst = node.getFloat64Data();
        forEachGatherNd(node, indices, outGrad, batchDims, (targetOffset, gradOffset) ->
                dst[targetOffset] += grad[gradOffset]
        );
    }

    public static void gatherNdGradF32(Tensor indices, Tensor outGrad, Tensor node, int batchDims) {
        validateGatherNdGrad(indices, outGrad, node, batchDims);
        java.util.Arrays.fill(node.getFloat32Data(), 0.0f);
        float[] grad = outGrad.getFloat32Data();
        float[] dst = node.getFloat32Data();
        forEachGatherNd(node, indices, outGrad, batchDims, (targetOffset, gradOffset) ->
                dst[targetOffset] += grad[gradOffset]
        );
    }

    public static void gatherNdGradBF16(Tensor indices, Tensor outGrad, Tensor node, int batchDims) {
        validateGatherNdGrad(indices, outGrad, node, batchDims);
        java.util.Arrays.fill(node.getBFloat16Data(), CpuDTypeOps.toBFloat16Bits(0.0f));
        short[] grad = outGrad.getBFloat16Data();
        short[] dst = node.getBFloat16Data();
        forEachGatherNd(node, indices, outGrad, batchDims, (targetOffset, gradOffset) -> {
            float acc = CpuDTypeOps.fromBFloat16Bits(dst[targetOffset]) + CpuDTypeOps.fromBFloat16Bits(grad[gradOffset]);
            dst[targetOffset] = CpuDTypeOps.toBFloat16Bits(acc);
        });
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
        short[] in = input.getBFloat16Data();
        short[] dst = out.getBFloat16Data();
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

    public static void scatterBF16(Tensor indices, Tensor outGrad, Tensor node, int dimension) {
        validateScatter(indices, outGrad, node, dimension);
        short[] grad = outGrad.getBFloat16Data();
        short[] dst = node.getBFloat16Data();
        forEachScatter(indices, outGrad, node, dimension, (baseNode, baseGrad, axisStrideNode, axisStrideGrad, axisIndex) -> {
            float acc = CpuDTypeOps.fromBFloat16Bits(dst[baseNode + axisIndex * axisStrideNode]) + CpuDTypeOps.fromBFloat16Bits(grad[baseGrad]);
            dst[baseNode + axisIndex * axisStrideNode] = CpuDTypeOps.toBFloat16Bits(acc);
        });
    }

    public static void gatherAxisGradF64(Tensor indices, Tensor outGrad, Tensor node, int axis) {
        validateGatherAxisGrad(indices, outGrad, node, axis);
        java.util.Arrays.fill(node.getFloat64Data(), 0.0d);
        double[] dst = node.getFloat64Data();
        forEachGatherAxisGrad(indices, outGrad, node, axis, (sourceLogical, outLogical) ->
                dst[sourceLogical] += outGrad.getByFlatIndex(outLogical)
        );
    }

    public static void gatherAxisGradF32(Tensor indices, Tensor outGrad, Tensor node, int axis) {
        validateGatherAxisGrad(indices, outGrad, node, axis);
        java.util.Arrays.fill(node.getFloat32Data(), 0.0f);
        float[] dst = node.getFloat32Data();
        forEachGatherAxisGrad(indices, outGrad, node, axis, (sourceLogical, outLogical) ->
                dst[sourceLogical] += (float) outGrad.getByFlatIndex(outLogical)
        );
    }

    public static void gatherAxisGradBF16(Tensor indices, Tensor outGrad, Tensor node, int axis) {
        validateGatherAxisGrad(indices, outGrad, node, axis);
        java.util.Arrays.fill(node.getBFloat16Data(), CpuDTypeOps.toBFloat16Bits(0.0f));
        short[] dst = node.getBFloat16Data();
        forEachGatherAxisGrad(indices, outGrad, node, axis, (sourceLogical, outLogical) -> {
            float acc = CpuDTypeOps.fromBFloat16Bits(dst[sourceLogical]) + (float) outGrad.getByFlatIndex(outLogical);
            dst[sourceLogical] = CpuDTypeOps.toBFloat16Bits(acc);
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

    public static void scatterAddBF16(Tensor base, Tensor indices, Tensor src, Tensor out, int dimension) {
        validateScatterAdd(base, indices, src, out, dimension);
        out.copyDataFrom(base);
        short[] srcData = src.getBFloat16Data();
        short[] dst = out.getBFloat16Data();
        forEachScatter(indices, src, out, dimension, (baseNode, baseGrad, axisStrideNode, axisStrideGrad, axisIndex) -> {
            float acc = CpuDTypeOps.fromBFloat16Bits(dst[baseNode + axisIndex * axisStrideNode]) + CpuDTypeOps.fromBFloat16Bits(srcData[baseGrad]);
            dst[baseNode + axisIndex * axisStrideNode] = CpuDTypeOps.toBFloat16Bits(acc);
        });
    }

    public static void scatterElementsF64(Tensor data, Tensor indices, Tensor updates, Tensor out, int axis, ScatterReduction reduction) {
        ScatterReduction effectiveReduction = validateScatterElements(data, indices, updates, out, axis, reduction);
        out.copyDataFrom(data);
        double[] updateData = updates.getFloat64Data();
        double[] dst = out.getFloat64Data();
        scatterDuplicateState(out, effectiveReduction, "scatterElements", state ->
                forEachScatterElements(data, indices, updates, out, axis, (updateOffset, targetOffset, targetLogical) -> {
                    state.mark(targetLogical);
                    dst[targetOffset] = reduce(dst[targetOffset], updateData[updateOffset], effectiveReduction);
                }));
    }

    public static void scatterElementsF32(Tensor data, Tensor indices, Tensor updates, Tensor out, int axis, ScatterReduction reduction) {
        ScatterReduction effectiveReduction = validateScatterElements(data, indices, updates, out, axis, reduction);
        out.copyDataFrom(data);
        float[] updateData = updates.getFloat32Data();
        float[] dst = out.getFloat32Data();
        scatterDuplicateState(out, effectiveReduction, "scatterElements", state ->
                forEachScatterElements(data, indices, updates, out, axis, (updateOffset, targetOffset, targetLogical) -> {
                    state.mark(targetLogical);
                    dst[targetOffset] = (float) reduce(dst[targetOffset], updateData[updateOffset], effectiveReduction);
                }));
    }

    public static void scatterElementsBF16(Tensor data, Tensor indices, Tensor updates, Tensor out, int axis, ScatterReduction reduction) {
        ScatterReduction effectiveReduction = validateScatterElements(data, indices, updates, out, axis, reduction);
        out.copyDataFrom(data);
        short[] updateData = updates.getBFloat16Data();
        short[] dst = out.getBFloat16Data();
        scatterDuplicateState(out, effectiveReduction, "scatterElements", state ->
                forEachScatterElements(data, indices, updates, out, axis, (updateOffset, targetOffset, targetLogical) -> {
                    state.mark(targetLogical);
                    float current = CpuDTypeOps.fromBFloat16Bits(dst[targetOffset]);
                    float update = CpuDTypeOps.fromBFloat16Bits(updateData[updateOffset]);
                    dst[targetOffset] = CpuDTypeOps.toBFloat16Bits((float) reduce(current, update, effectiveReduction));
                }));
    }

    public static void scatterElementsBOOL(Tensor data, Tensor indices, Tensor updates, Tensor out, int axis, ScatterReduction reduction) {
        ScatterReduction effectiveReduction = validateScatterElements(data, indices, updates, out, axis, reduction);
        out.copyDataFrom(data);
        byte[] updateData = updates.getBoolData();
        byte[] dst = out.getBoolData();
        scatterDuplicateState(out, effectiveReduction, "scatterElements", state ->
                forEachScatterElements(data, indices, updates, out, axis, (updateOffset, targetOffset, targetLogical) -> {
                    state.mark(targetLogical);
                    dst[targetOffset] = updateData[updateOffset] == 0 ? (byte) 0 : (byte) 1;
                }));
    }

    public static void scatterElementsI32(Tensor data, Tensor indices, Tensor updates, Tensor out, int axis, ScatterReduction reduction) {
        ScatterReduction effectiveReduction = validateScatterElements(data, indices, updates, out, axis, reduction);
        out.copyDataFrom(data);
        int[] updateData = updates.getInt32Data();
        int[] dst = out.getInt32Data();
        scatterDuplicateState(out, effectiveReduction, "scatterElements", state ->
                forEachScatterElements(data, indices, updates, out, axis, (updateOffset, targetOffset, targetLogical) -> {
                    state.mark(targetLogical);
                    dst[targetOffset] = reduceInt(dst[targetOffset], updateData[updateOffset], effectiveReduction);
                }));
    }

    public static void scatterNdF64(Tensor data, Tensor indices, Tensor updates, Tensor out, ScatterReduction reduction) {
        ScatterReduction effectiveReduction = validateScatterNd(data, indices, updates, out, reduction);
        out.copyDataFrom(data);
        double[] updateData = updates.getFloat64Data();
        double[] dst = out.getFloat64Data();
        scatterDuplicateState(out, effectiveReduction, "scatterNd", state ->
                forEachScatterNd(data, indices, updates, out, (updateOffset, targetOffset, targetLogical) -> {
                    state.mark(targetLogical);
                    dst[targetOffset] = reduce(dst[targetOffset], updateData[updateOffset], effectiveReduction);
                }));
    }

    public static void scatterNdF32(Tensor data, Tensor indices, Tensor updates, Tensor out, ScatterReduction reduction) {
        ScatterReduction effectiveReduction = validateScatterNd(data, indices, updates, out, reduction);
        out.copyDataFrom(data);
        float[] updateData = updates.getFloat32Data();
        float[] dst = out.getFloat32Data();
        scatterDuplicateState(out, effectiveReduction, "scatterNd", state ->
                forEachScatterNd(data, indices, updates, out, (updateOffset, targetOffset, targetLogical) -> {
                    state.mark(targetLogical);
                    dst[targetOffset] = (float) reduce(dst[targetOffset], updateData[updateOffset], effectiveReduction);
                }));
    }

    public static void scatterNdBF16(Tensor data, Tensor indices, Tensor updates, Tensor out, ScatterReduction reduction) {
        ScatterReduction effectiveReduction = validateScatterNd(data, indices, updates, out, reduction);
        out.copyDataFrom(data);
        short[] updateData = updates.getBFloat16Data();
        short[] dst = out.getBFloat16Data();
        scatterDuplicateState(out, effectiveReduction, "scatterNd", state ->
                forEachScatterNd(data, indices, updates, out, (updateOffset, targetOffset, targetLogical) -> {
                    state.mark(targetLogical);
                    float current = CpuDTypeOps.fromBFloat16Bits(dst[targetOffset]);
                    float update = CpuDTypeOps.fromBFloat16Bits(updateData[updateOffset]);
                    dst[targetOffset] = CpuDTypeOps.toBFloat16Bits((float) reduce(current, update, effectiveReduction));
                }));
    }

    public static void scatterNdBOOL(Tensor data, Tensor indices, Tensor updates, Tensor out, ScatterReduction reduction) {
        ScatterReduction effectiveReduction = validateScatterNd(data, indices, updates, out, reduction);
        out.copyDataFrom(data);
        byte[] updateData = updates.getBoolData();
        byte[] dst = out.getBoolData();
        scatterDuplicateState(out, effectiveReduction, "scatterNd", state ->
                forEachScatterNd(data, indices, updates, out, (updateOffset, targetOffset, targetLogical) -> {
                    state.mark(targetLogical);
                    dst[targetOffset] = updateData[updateOffset] == 0 ? (byte) 0 : (byte) 1;
                }));
    }

    public static void scatterNdI32(Tensor data, Tensor indices, Tensor updates, Tensor out, ScatterReduction reduction) {
        ScatterReduction effectiveReduction = validateScatterNd(data, indices, updates, out, reduction);
        out.copyDataFrom(data);
        int[] updateData = updates.getInt32Data();
        int[] dst = out.getInt32Data();
        scatterDuplicateState(out, effectiveReduction, "scatterNd", state ->
                forEachScatterNd(data, indices, updates, out, (updateOffset, targetOffset, targetLogical) -> {
                    state.mark(targetLogical);
                    dst[targetOffset] = reduceInt(dst[targetOffset], updateData[updateOffset], effectiveReduction);
                }));
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

    public static void takeAlongAxisScatterBF16(Tensor indices, Tensor outGrad, Tensor node, int dimension) {
        validateTakeAlongAxisScatter(indices, outGrad, node, dimension);
        short[] grad = outGrad.getBFloat16Data();
        short[] dst = node.getBFloat16Data();
        forEachTakeAlongAxisScatter(indices, outGrad, node, dimension, (baseNode, gradOffset, axisStrideNode, axisIndex) -> {
            float acc = CpuDTypeOps.fromBFloat16Bits(dst[baseNode + axisIndex * axisStrideNode]) + CpuDTypeOps.fromBFloat16Bits(grad[gradOffset]);
            dst[baseNode + axisIndex * axisStrideNode] = CpuDTypeOps.toBFloat16Bits(acc);
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

    private static void validateGatherAxis(Tensor input, Tensor indices, Tensor out, int axis) {
        int[] inputShape = input.getShapeUnsafe();
        if (axis < 0 || axis >= inputShape.length) {
            throw new IllegalArgumentException("gatherAxis axis out of bounds: " + axis);
        }
        validateIndexTensor(indices);
        validateShape(out.getShapeUnsafe(), gatherAxisOutputShape(inputShape, indices.getShapeUnsafe(), axis),
                "gatherAxis output shape mismatch.");
        if (input.getDataType() != out.getDataType()) {
            throw new IllegalArgumentException("gatherAxis output dtype must match input dtype.");
        }
    }

    private static void validateGatherAxisGrad(Tensor indices, Tensor outGrad, Tensor node, int axis) {
        int[] nodeShape = node.getShapeUnsafe();
        if (axis < 0 || axis >= nodeShape.length) {
            throw new IllegalArgumentException("gatherAxisGrad axis out of bounds: " + axis);
        }
        validateIndexTensor(indices);
        validateShape(outGrad.getShapeUnsafe(), gatherAxisOutputShape(nodeShape, indices.getShapeUnsafe(), axis),
                "gatherAxisGrad outGrad shape mismatch.");
        if (outGrad.getDataType() != node.getDataType()) {
            throw new IllegalArgumentException("gatherAxisGrad output dtype must match outGrad dtype.");
        }
        if (node.getDataType() == DataType.BOOL || node.getDataType() == DataType.INT32) {
            throw new IllegalArgumentException("gatherAxisGrad requires floating output dtype.");
        }
    }

    private static void validateGatherNd(Tensor input, Tensor indices, Tensor out, int batchDims) {
        validateIndexTensor(indices);
        int[] inputShape = input.getShapeUnsafe();
        int[] indicesShape = indices.getShapeUnsafe();
        validateShape(out.getShapeUnsafe(), gatherNdOutputShape(inputShape, indicesShape, batchDims),
                "gatherNd output shape mismatch.");
        if (input.getDataType() != out.getDataType()) {
            throw new IllegalArgumentException("gatherNd output dtype must match input dtype.");
        }
    }

    private static void validateGatherNdGrad(Tensor indices, Tensor outGrad, Tensor node, int batchDims) {
        validateIndexTensor(indices);
        int[] nodeShape = node.getShapeUnsafe();
        int[] indicesShape = indices.getShapeUnsafe();
        validateShape(outGrad.getShapeUnsafe(), gatherNdOutputShape(nodeShape, indicesShape, batchDims),
                "gatherNdGrad outGrad shape mismatch.");
        if (outGrad.getDataType() != node.getDataType()) {
            throw new IllegalArgumentException("gatherNdGrad output dtype must match outGrad dtype.");
        }
        if (node.getDataType() == DataType.BOOL || node.getDataType() == DataType.INT32) {
            throw new IllegalArgumentException("gatherNdGrad requires floating output dtype.");
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

    private static ScatterReduction validateScatterElements(
            Tensor data,
            Tensor indices,
            Tensor updates,
            Tensor out,
            int axis,
            ScatterReduction reduction
    ) {
        ScatterReduction effectiveReduction = reduction == null ? ScatterReduction.NONE : reduction;
        int[] dataShape = data.getShapeUnsafe();
        if (axis < 0 || axis >= dataShape.length) {
            throw new IllegalArgumentException("scatterElements axis out of bounds: " + axis);
        }
        validateIndexTensor(indices);
        int[] indicesShape = indices.getShapeUnsafe();
        int[] updatesShape = updates.getShapeUnsafe();
        validateShape(out.getShapeUnsafe(), dataShape, "scatterElements output shape must equal data shape.");
        if (indicesShape.length != dataShape.length) {
            throw new IllegalArgumentException("scatterElements indices rank must match data rank.");
        }
        validateShape(updatesShape, indicesShape, "scatterElements updates shape must equal indices shape.");
        for (int i = 0; i < indicesShape.length; i++) {
            if (i != axis && indicesShape[i] != dataShape[i]) {
                throw new IllegalArgumentException("scatterElements indices must match data shape on all non-axis dimensions.");
            }
        }
        if (data.getDataType() != updates.getDataType() || data.getDataType() != out.getDataType()) {
            throw new IllegalArgumentException("scatterElements requires matching dtypes for data, updates and output.");
        }
        if (data.getDataType() == DataType.BOOL && effectiveReduction != ScatterReduction.NONE) {
            throw new IllegalArgumentException("scatterElements BOOL tensors support only NONE reduction.");
        }
        return effectiveReduction;
    }

    private static ScatterReduction validateScatterNd(
            Tensor data,
            Tensor indices,
            Tensor updates,
            Tensor out,
            ScatterReduction reduction
    ) {
        ScatterReduction effectiveReduction = reduction == null ? ScatterReduction.NONE : reduction;
        validateIndexTensor(indices);
        int[] dataShape = data.getShapeUnsafe();
        int[] indicesShape = indices.getShapeUnsafe();
        int[] updatesShape = updates.getShapeUnsafe();
        validateShape(out.getShapeUnsafe(), dataShape, "scatterNd output shape must equal data shape.");
        if (indicesShape.length == 0) {
            throw new IllegalArgumentException("scatterNd indices rank must be at least 1.");
        }
        int tupleRank = indicesShape[indicesShape.length - 1];
        if (tupleRank <= 0 || tupleRank > dataShape.length) {
            throw new IllegalArgumentException("scatterNd final indices dimension must be in [1, data rank].");
        }
        int expectedRank = indicesShape.length - 1 + dataShape.length - tupleRank;
        if (updatesShape.length != expectedRank) {
            if (expectedRank == 0 && updatesShape.length == 1 && updatesShape[0] == 1) {
                if (data.getDataType() != updates.getDataType() || data.getDataType() != out.getDataType()) {
                    throw new IllegalArgumentException("scatterNd requires matching dtypes for data, updates and output.");
                }
                if (data.getDataType() == DataType.BOOL && effectiveReduction != ScatterReduction.NONE) {
                    throw new IllegalArgumentException("scatterNd BOOL tensors support only NONE reduction.");
                }
                return effectiveReduction;
            }
            throw new IllegalArgumentException("scatterNd updates shape must equal indices.shape[:-1] + data.shape[indices.shape[-1]:].");
        }
        int p = 0;
        for (int i = 0; i < indicesShape.length - 1; i++) {
            if (updatesShape[p++] != indicesShape[i]) {
                throw new IllegalArgumentException("scatterNd updates prefix shape must match indices prefix shape.");
            }
        }
        for (int i = tupleRank; i < dataShape.length; i++) {
            if (updatesShape[p++] != dataShape[i]) {
                throw new IllegalArgumentException("scatterNd updates suffix shape must match indexed data slice shape.");
            }
        }
        if (data.getDataType() != updates.getDataType() || data.getDataType() != out.getDataType()) {
            throw new IllegalArgumentException("scatterNd requires matching dtypes for data, updates and output.");
        }
        if (data.getDataType() == DataType.BOOL && effectiveReduction != ScatterReduction.NONE) {
            throw new IllegalArgumentException("scatterNd BOOL tensors support only NONE reduction.");
        }
        return effectiveReduction;
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

    private static int[] gatherAxisOutputShape(int[] dataShape, int[] indicesShape, int axis) {
        int[] out = new int[dataShape.length + indicesShape.length - 1];
        int p = 0;
        for (int i = 0; i < axis; i++) {
            out[p++] = dataShape[i];
        }
        for (int dim : indicesShape) {
            out[p++] = dim;
        }
        for (int i = axis + 1; i < dataShape.length; i++) {
            out[p++] = dataShape[i];
        }
        return out;
    }

    private static int[] gatherNdOutputShape(int[] dataShape, int[] indicesShape, int batchDims) {
        validateGatherNdShape(dataShape, indicesShape, batchDims);
        int tupleRank = indicesShape[indicesShape.length - 1];
        int outputRank = indicesShape.length - 1 + dataShape.length - batchDims - tupleRank;
        if (outputRank == 0) {
            return new int[]{1};
        }
        int[] out = new int[outputRank];
        int p = 0;
        for (int i = 0; i < indicesShape.length - 1; i++) {
            out[p++] = indicesShape[i];
        }
        for (int i = batchDims + tupleRank; i < dataShape.length; i++) {
            out[p++] = dataShape[i];
        }
        return out;
    }

    private static void validateGatherNdShape(int[] dataShape, int[] indicesShape, int batchDims) {
        if (indicesShape.length == 0) {
            throw new IllegalArgumentException("gatherNd indices rank must be at least 1.");
        }
        if (batchDims < 0 || batchDims >= indicesShape.length) {
            throw new IllegalArgumentException("gatherNd batchDims must be in [0, indices rank).");
        }
        if (batchDims > dataShape.length) {
            throw new IllegalArgumentException("gatherNd batchDims cannot exceed data rank.");
        }
        for (int i = 0; i < batchDims; i++) {
            if (indicesShape[i] != dataShape[i]) {
                throw new IllegalArgumentException("gatherNd batch dimensions must match data leading dimensions.");
            }
        }
        int tupleRank = indicesShape[indicesShape.length - 1];
        if (tupleRank <= 0 || batchDims + tupleRank > dataShape.length) {
            throw new IllegalArgumentException("gatherNd final indices dimension must be in [1, data rank - batchDims].");
        }
    }

    private static void forEachGather(Tensor input, Tensor indices, Tensor out, int dimension, GatherConsumer consumer) {
        int[] inputShape = input.getShapeUnsafe();
        int[] inputStrides = input.getStridesUnsafe();
        int[] outShape = out.getShapeUnsafe();
        int[] outStrides = out.getStridesUnsafe();
        int inputBaseOffset = input.getStorageOffsetUnsafe();
        int outBaseOffset = out.getStorageOffsetUnsafe();
        int[] reducedDense = TensorMetadata.computeStrides(outShape);
        int total = out.getFlatDataSize();
        int axisSize = inputShape[dimension];
        int axisStrideIn = inputStrides[dimension];

        for (int logical = 0; logical < total; logical++) {
            int rem = logical;
            int baseIn = inputBaseOffset;
            int baseOut = outBaseOffset;
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
        int nodeBaseOffset = node.getStorageOffsetUnsafe();
        int gradBaseOffset = outGrad.getStorageOffsetUnsafe();
        int[] reducedDense = TensorMetadata.computeStrides(gradShape);
        int total = outGrad.getFlatDataSize();
        int axisSize = nodeShape[dimension];
        int axisStrideNode = nodeStrides[dimension];

        for (int logical = 0; logical < total; logical++) {
            int rem = logical;
            int baseNode = nodeBaseOffset;
            int baseGrad = gradBaseOffset;
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
        int inputBaseOffset = input.getStorageOffsetUnsafe();
        int outBaseOffset = out.getStorageOffsetUnsafe();
        int[] outDense = TensorMetadata.computeStrides(outShape);
        int total = out.getFlatDataSize();
        int axisSize = inputShape[dimension];
        int axisStrideIn = inputStrides[dimension];

        for (int logical = 0; logical < total; logical++) {
            int rem = logical;
            int baseIn = inputBaseOffset;
            int outOffset = outBaseOffset;
            for (int d = 0; d < outShape.length; d++) {
                int coord = rem / outDense[d];
                rem %= outDense[d];
                outOffset += coord * outStrides[d];
                if (d != dimension) {
                    baseIn += coord * inputStrides[d];
                }
            }
            int axisIndex = readAxisIndexAllowNegative(indices, logical, axisSize);
            consumer.accept(baseIn, outOffset, axisStrideIn, axisIndex);
        }
    }

    private static void forEachTakeAlongAxisScatter(Tensor indices, Tensor outGrad, Tensor node, int dimension, TakeAlongAxisScatterConsumer consumer) {
        int[] nodeShape = node.getShapeUnsafe();
        int[] nodeStrides = node.getStridesUnsafe();
        int[] gradShape = outGrad.getShapeUnsafe();
        int[] gradStrides = outGrad.getStridesUnsafe();
        int nodeBaseOffset = node.getStorageOffsetUnsafe();
        int gradBaseOffset = outGrad.getStorageOffsetUnsafe();
        int[] gradDense = TensorMetadata.computeStrides(gradShape);
        int total = outGrad.getFlatDataSize();
        int axisSize = nodeShape[dimension];
        int axisStrideNode = nodeStrides[dimension];

        for (int logical = 0; logical < total; logical++) {
            int rem = logical;
            int baseNode = nodeBaseOffset;
            int gradOffset = gradBaseOffset;
            for (int d = 0; d < gradShape.length; d++) {
                int coord = rem / gradDense[d];
                rem %= gradDense[d];
                gradOffset += coord * gradStrides[d];
                if (d != dimension) {
                    baseNode += coord * nodeStrides[d];
                }
            }
            int axisIndex = readAxisIndexAllowNegative(indices, logical, axisSize);
            consumer.accept(baseNode, gradOffset, axisStrideNode, axisIndex);
        }
    }

    private static void forEachScatterElements(
            Tensor data,
            Tensor indices,
            Tensor updates,
            Tensor out,
            int axis,
            ScatterElementsConsumer consumer
    ) {
        int[] dataShape = data.getShapeUnsafe();
        int[] dataDense = TensorMetadata.computeStrides(dataShape);
        int[] updatesShape = updates.getShapeUnsafe();
        int[] updatesStrides = updates.getStridesUnsafe();
        int[] updatesDense = TensorMetadata.computeStrides(updatesShape);
        int[] outStrides = out.getStridesUnsafe();
        int updatesBaseOffset = updates.getStorageOffsetUnsafe();
        int outBaseOffset = out.getStorageOffsetUnsafe();
        int total = updates.getFlatDataSize();
        int axisSize = dataShape[axis];

        for (int logical = 0; logical < total; logical++) {
            int rem = logical;
            int updateOffset = updatesBaseOffset;
            int targetOffset = outBaseOffset;
            int targetLogical = 0;
            for (int d = 0; d < updatesShape.length; d++) {
                int coord = rem / updatesDense[d];
                rem %= updatesDense[d];
                updateOffset += coord * updatesStrides[d];
                int targetCoord = d == axis ? readAxisIndexAllowNegative(indices, logical, axisSize) : coord;
                targetOffset += targetCoord * outStrides[d];
                targetLogical += targetCoord * dataDense[d];
            }
            consumer.accept(updateOffset, targetOffset, targetLogical);
        }
    }

    private static void forEachScatterNd(
            Tensor data,
            Tensor indices,
            Tensor updates,
            Tensor out,
            ScatterElementsConsumer consumer
    ) {
        int[] dataShape = data.getShapeUnsafe();
        int[] dataDense = TensorMetadata.computeStrides(dataShape);
        int[] indicesShape = indices.getShapeUnsafe();
        int[] indicesDense = TensorMetadata.computeStrides(indicesShape);
        int[] updatesShape = updates.getShapeUnsafe();
        int[] updatesStrides = updates.getStridesUnsafe();
        int[] updatesDense = TensorMetadata.computeStrides(updatesShape);
        int[] outStrides = out.getStridesUnsafe();
        int updatesBaseOffset = updates.getStorageOffsetUnsafe();
        int outBaseOffset = out.getStorageOffsetUnsafe();
        int total = updates.getFlatDataSize();
        int tupleRank = indicesShape[indicesShape.length - 1];
        int prefixRank = indicesShape.length - 1;
        int tupleStride = indicesDense[indicesShape.length - 1];
        int[] coords = new int[updatesShape.length];

        for (int logical = 0; logical < total; logical++) {
            int rem = logical;
            int updateOffset = updatesBaseOffset;
            for (int d = 0; d < updatesShape.length; d++) {
                int coord = rem / updatesDense[d];
                rem %= updatesDense[d];
                coords[d] = coord;
                updateOffset += coord * updatesStrides[d];
            }

            int indexBaseLogical = 0;
            for (int d = 0; d < prefixRank; d++) {
                indexBaseLogical += coords[d] * indicesDense[d];
            }

            int targetOffset = outBaseOffset;
            int targetLogical = 0;
            for (int d = 0; d < tupleRank; d++) {
                int targetCoord = readAxisIndexAllowNegative(indices, indexBaseLogical + d * tupleStride, dataShape[d]);
                targetOffset += targetCoord * outStrides[d];
                targetLogical += targetCoord * dataDense[d];
            }
            for (int d = tupleRank; d < dataShape.length; d++) {
                int updateCoord = coords[prefixRank + d - tupleRank];
                targetOffset += updateCoord * outStrides[d];
                targetLogical += updateCoord * dataDense[d];
            }
            consumer.accept(updateOffset, targetOffset, targetLogical);
        }
    }

    private static void forEachGatherNd(Tensor input, Tensor indices, Tensor out, int batchDims, GatherNdConsumer consumer) {
        int[] inputShape = input.getShapeUnsafe();
        int[] inputStrides = input.getStridesUnsafe();
        int[] indicesShape = indices.getShapeUnsafe();
        int[] indicesDense = TensorMetadata.computeStrides(indicesShape);
        int[] outShape = out.getShapeUnsafe();
        int[] outStrides = out.getStridesUnsafe();
        int[] outDense = TensorMetadata.computeStrides(outShape);
        int inputBaseOffset = input.getStorageOffsetUnsafe();
        int outBaseOffset = out.getStorageOffsetUnsafe();
        int total = out.getFlatDataSize();
        int tupleRank = indicesShape[indicesShape.length - 1];
        int prefixRank = indicesShape.length - 1;
        int tupleStride = indicesDense[indicesShape.length - 1];
        int[] coords = new int[outShape.length];

        for (int logical = 0; logical < total; logical++) {
            int rem = logical;
            int outOffset = outBaseOffset;
            for (int d = 0; d < outShape.length; d++) {
                int coord = rem / outDense[d];
                rem %= outDense[d];
                coords[d] = coord;
                outOffset += coord * outStrides[d];
            }

            int indexBaseLogical = 0;
            for (int d = 0; d < prefixRank; d++) {
                indexBaseLogical += coords[d] * indicesDense[d];
            }

            int sourceOffset = inputBaseOffset;
            for (int d = 0; d < batchDims; d++) {
                sourceOffset += coords[d] * inputStrides[d];
            }
            for (int d = 0; d < tupleRank; d++) {
                int inputDim = batchDims + d;
                int sourceCoord = readAxisIndexAllowNegative(indices, indexBaseLogical + d * tupleStride, inputShape[inputDim]);
                sourceOffset += sourceCoord * inputStrides[inputDim];
            }
            for (int d = batchDims + tupleRank; d < inputShape.length; d++) {
                int suffixCoord = coords[prefixRank + d - batchDims - tupleRank];
                sourceOffset += suffixCoord * inputStrides[d];
            }
            consumer.accept(sourceOffset, outOffset);
        }
    }

    private static void forEachGatherAxis(Tensor input, Tensor indices, Tensor out, int axis, GatherAxisConsumer consumer) {
        int[] inputShape = input.getShapeUnsafe();
        int[] inputDense = TensorMetadata.computeStrides(inputShape);
        int[] indicesShape = indices.getShapeUnsafe();
        int[] indicesDense = TensorMetadata.computeStrides(indicesShape);
        int[] outShape = out.getShapeUnsafe();
        int[] outDense = TensorMetadata.computeStrides(outShape);
        int total = out.getFlatDataSize();
        int axisSize = inputShape[axis];
        int indicesRank = indicesShape.length;

        for (int outLogical = 0; outLogical < total; outLogical++) {
            int sourceLogical = gatherAxisSourceLogical(outLogical, outShape, outDense, inputShape, inputDense,
                    indices, indicesDense, axis, axisSize, indicesRank);
            consumer.accept(sourceLogical, outLogical);
        }
    }

    private static void forEachGatherAxisGrad(Tensor indices, Tensor outGrad, Tensor node, int axis, GatherAxisConsumer consumer) {
        int[] inputShape = node.getShapeUnsafe();
        int[] inputDense = TensorMetadata.computeStrides(inputShape);
        int[] indicesShape = indices.getShapeUnsafe();
        int[] indicesDense = TensorMetadata.computeStrides(indicesShape);
        int[] outShape = outGrad.getShapeUnsafe();
        int[] outDense = TensorMetadata.computeStrides(outShape);
        int total = outGrad.getFlatDataSize();
        int axisSize = inputShape[axis];
        int indicesRank = indicesShape.length;

        for (int outLogical = 0; outLogical < total; outLogical++) {
            int sourceLogical = gatherAxisSourceLogical(outLogical, outShape, outDense, inputShape, inputDense,
                    indices, indicesDense, axis, axisSize, indicesRank);
            consumer.accept(sourceLogical, outLogical);
        }
    }

    private static int gatherAxisSourceLogical(
            int outLogical,
            int[] outShape,
            int[] outDense,
            int[] inputShape,
            int[] inputDense,
            Tensor indices,
            int[] indicesDense,
            int axis,
            int axisSize,
            int indicesRank
    ) {
        int rem = outLogical;
        int sourceLogical = 0;
        int indexLogical = 0;
        for (int d = 0; d < outShape.length; d++) {
            int coord = rem / outDense[d];
            rem %= outDense[d];
            if (d < axis) {
                sourceLogical += coord * inputDense[d];
            } else if (d < axis + indicesRank) {
                indexLogical += coord * indicesDense[d - axis];
            } else {
                int inputDim = d - indicesRank + 1;
                sourceLogical += coord * inputDense[inputDim];
            }
        }
        int axisIndex = readAxisIndexAllowNegative(indices, indexLogical, axisSize);
        sourceLogical += axisIndex * inputDense[axis];
        return sourceLogical;
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

    private static int readAxisIndexAllowNegative(Tensor indices, int logicalIndex, int axisSize) {
        double raw = indices.getByFlatIndex(logicalIndex);
        if (!Double.isFinite(raw)) {
            throw new IllegalArgumentException("Gather index must be finite.");
        }
        long integral = Math.round(raw);
        if (Math.abs(raw - integral) > 1e-9) {
            throw new IllegalArgumentException("Gather index must be an integer value. got=" + raw);
        }
        if (integral < 0) {
            integral += axisSize;
        }
        if (integral < 0 || integral >= axisSize) {
            throw new IllegalArgumentException("Gather index out of bounds: " + raw + " for axis size " + axisSize);
        }
        return (int) integral;
    }

    private static double reduce(double current, double update, ScatterReduction reduction) {
        return switch (reduction) {
            case NONE -> update;
            case ADD -> current + update;
            case MUL -> current * update;
            case MAX -> Math.max(current, update);
            case MIN -> Math.min(current, update);
        };
    }

    private static int reduceInt(int current, int update, ScatterReduction reduction) {
        return switch (reduction) {
            case NONE -> update;
            case ADD -> Math.addExact(current, update);
            case MUL -> Math.multiplyExact(current, update);
            case MAX -> Math.max(current, update);
            case MIN -> Math.min(current, update);
        };
    }

    private static void scatterDuplicateState(Tensor out, ScatterReduction reduction, String operationName, DuplicateStateConsumer consumer) {
        DuplicateState state = reduction == ScatterReduction.NONE
                ? new DuplicateState(new boolean[out.getFlatDataSize()], operationName)
                : DuplicateState.NOOP;
        consumer.accept(state);
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

    @FunctionalInterface
    private interface ScatterElementsConsumer {
        void accept(int updateOffset, int targetOffset, int targetLogical);
    }

    @FunctionalInterface
    private interface GatherNdConsumer {
        void accept(int sourceOffset, int outOffset);
    }

    @FunctionalInterface
    private interface DuplicateStateConsumer {
        void accept(DuplicateState state);
    }

    private static final class DuplicateState {
        private static final DuplicateState NOOP = new DuplicateState(null, "scatter");

        private final boolean[] seen;
        private final String operationName;

        private DuplicateState(boolean[] seen, String operationName) {
            this.seen = seen;
            this.operationName = operationName;
        }

        private void mark(int targetLogical) {
            if (seen == null) {
                return;
            }
            if (seen[targetLogical]) {
                throw new IllegalArgumentException(operationName + " NONE reduction does not allow duplicate target indices.");
            }
            seen[targetLogical] = true;
        }
    }

    @FunctionalInterface
    private interface GatherAxisConsumer {
        void accept(int sourceLogical, int outLogical);
    }
}
