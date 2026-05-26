package backend.cpu.kernels.index;

import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.dtype.TensorDTypeOps;

final class GatherAxisLoops {
    private GatherAxisLoops() {
    }

    static void gatherAxisF64(Tensor input, Tensor indices, Tensor out, int axis) {
        IndexValidation.validateGatherAxis(input, indices, out, axis);
        double[] in = TensorInternalAccess.float64Data(input);
        double[] dst = TensorInternalAccess.float64Data(out);
        GatherAxisPlan plan = GatherAxisPlan.create(input, indices, out, axis);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int outLogical = 0; outLogical < plan.total; outLogical++) {
            plan.computeOffsets(outLogical, indexReader, axis);
            dst[plan.outOffset] = in[plan.sourceOffset];
        }
    }

    static void gatherAxisF32(Tensor input, Tensor indices, Tensor out, int axis) {
        IndexValidation.validateGatherAxis(input, indices, out, axis);
        float[] in = TensorInternalAccess.float32Data(input);
        float[] dst = TensorInternalAccess.float32Data(out);
        GatherAxisPlan plan = GatherAxisPlan.create(input, indices, out, axis);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int outLogical = 0; outLogical < plan.total; outLogical++) {
            plan.computeOffsets(outLogical, indexReader, axis);
            dst[plan.outOffset] = in[plan.sourceOffset];
        }
    }

    static void gatherAxisBF16(Tensor input, Tensor indices, Tensor out, int axis) {
        IndexValidation.validateGatherAxis(input, indices, out, axis);
        short[] in = TensorInternalAccess.bfloat16Data(input);
        short[] dst = TensorInternalAccess.bfloat16Data(out);
        GatherAxisPlan plan = GatherAxisPlan.create(input, indices, out, axis);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int outLogical = 0; outLogical < plan.total; outLogical++) {
            plan.computeOffsets(outLogical, indexReader, axis);
            dst[plan.outOffset] = in[plan.sourceOffset];
        }
    }

    static void gatherAxisBOOL(Tensor input, Tensor indices, Tensor out, int axis) {
        IndexValidation.validateGatherAxis(input, indices, out, axis);
        byte[] in = TensorInternalAccess.boolData(input);
        byte[] dst = TensorInternalAccess.boolData(out);
        GatherAxisPlan plan = GatherAxisPlan.create(input, indices, out, axis);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int outLogical = 0; outLogical < plan.total; outLogical++) {
            plan.computeOffsets(outLogical, indexReader, axis);
            dst[plan.outOffset] = in[plan.sourceOffset];
        }
    }

    static void gatherAxisI32(Tensor input, Tensor indices, Tensor out, int axis) {
        IndexValidation.validateGatherAxis(input, indices, out, axis);
        int[] in = TensorInternalAccess.int32Data(input);
        int[] dst = TensorInternalAccess.int32Data(out);
        GatherAxisPlan plan = GatherAxisPlan.create(input, indices, out, axis);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int outLogical = 0; outLogical < plan.total; outLogical++) {
            plan.computeOffsets(outLogical, indexReader, axis);
            dst[plan.outOffset] = in[plan.sourceOffset];
        }
    }

    static void gatherAxisI64(Tensor input, Tensor indices, Tensor out, int axis) {
        IndexValidation.validateGatherAxis(input, indices, out, axis);
        long[] in = TensorInternalAccess.int64Data(input);
        long[] dst = TensorInternalAccess.int64Data(out);
        GatherAxisPlan plan = GatherAxisPlan.create(input, indices, out, axis);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int outLogical = 0; outLogical < plan.total; outLogical++) {
            plan.computeOffsets(outLogical, indexReader, axis);
            dst[plan.outOffset] = in[plan.sourceOffset];
        }
    }

    static void gatherAxisGradF64(Tensor indices, Tensor outGrad, Tensor node, int axis) {
        IndexValidation.validateGatherAxisGrad(indices, outGrad, node, axis);
        IndexLoopSupport.fillZeroF64(node);
        double[] grad = TensorInternalAccess.float64Data(outGrad);
        double[] dst = TensorInternalAccess.float64Data(node);
        GatherAxisPlan plan = GatherAxisPlan.create(node, indices, outGrad, axis);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int outLogical = 0; outLogical < plan.total; outLogical++) {
            plan.computeOffsets(outLogical, indexReader, axis);
            dst[plan.sourceOffset] += grad[plan.outOffset];
        }
    }

    static void gatherAxisGradF32(Tensor indices, Tensor outGrad, Tensor node, int axis) {
        IndexValidation.validateGatherAxisGrad(indices, outGrad, node, axis);
        IndexLoopSupport.fillZeroF32(node);
        float[] grad = TensorInternalAccess.float32Data(outGrad);
        float[] dst = TensorInternalAccess.float32Data(node);
        GatherAxisPlan plan = GatherAxisPlan.create(node, indices, outGrad, axis);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int outLogical = 0; outLogical < plan.total; outLogical++) {
            plan.computeOffsets(outLogical, indexReader, axis);
            dst[plan.sourceOffset] += grad[plan.outOffset];
        }
    }

    static void gatherAxisGradBF16(Tensor indices, Tensor outGrad, Tensor node, int axis) {
        IndexValidation.validateGatherAxisGrad(indices, outGrad, node, axis);
        IndexLoopSupport.fillZeroBF16(node);
        short[] grad = TensorInternalAccess.bfloat16Data(outGrad);
        short[] dst = TensorInternalAccess.bfloat16Data(node);
        GatherAxisPlan plan = GatherAxisPlan.create(node, indices, outGrad, axis);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int outLogical = 0; outLogical < plan.total; outLogical++) {
            plan.computeOffsets(outLogical, indexReader, axis);
            float acc = TensorDTypeOps.fromBFloat16Bits(dst[plan.sourceOffset])
                    + TensorDTypeOps.fromBFloat16Bits(grad[plan.outOffset]);
            dst[plan.sourceOffset] = TensorDTypeOps.toBFloat16Bits(acc);
        }
    }

    static void scatterAxisAddF64(Tensor data, Tensor indices, Tensor updates, Tensor out, int axis) {
        IndexValidation.validateScatterAxisAdd(data, indices, updates, out, axis);
        out.copyDataFrom(data);
        double[] updateData = TensorInternalAccess.float64Data(updates);
        double[] dst = TensorInternalAccess.float64Data(out);
        GatherAxisPlan plan = GatherAxisPlan.create(out, indices, updates, axis);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int updateLogical = 0; updateLogical < plan.total; updateLogical++) {
            plan.computeOffsets(updateLogical, indexReader, axis);
            dst[plan.sourceOffset] += updateData[plan.outOffset];
        }
    }

    static void scatterAxisAddF32(Tensor data, Tensor indices, Tensor updates, Tensor out, int axis) {
        IndexValidation.validateScatterAxisAdd(data, indices, updates, out, axis);
        out.copyDataFrom(data);
        float[] updateData = TensorInternalAccess.float32Data(updates);
        float[] dst = TensorInternalAccess.float32Data(out);
        GatherAxisPlan plan = GatherAxisPlan.create(out, indices, updates, axis);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int updateLogical = 0; updateLogical < plan.total; updateLogical++) {
            plan.computeOffsets(updateLogical, indexReader, axis);
            dst[plan.sourceOffset] += updateData[plan.outOffset];
        }
    }

    static void scatterAxisAddBF16(Tensor data, Tensor indices, Tensor updates, Tensor out, int axis) {
        IndexValidation.validateScatterAxisAdd(data, indices, updates, out, axis);
        out.copyDataFrom(data);
        short[] updateData = TensorInternalAccess.bfloat16Data(updates);
        short[] dst = TensorInternalAccess.bfloat16Data(out);
        GatherAxisPlan plan = GatherAxisPlan.create(out, indices, updates, axis);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int updateLogical = 0; updateLogical < plan.total; updateLogical++) {
            plan.computeOffsets(updateLogical, indexReader, axis);
            float acc = TensorDTypeOps.fromBFloat16Bits(dst[plan.sourceOffset])
                    + TensorDTypeOps.fromBFloat16Bits(updateData[plan.outOffset]);
            dst[plan.sourceOffset] = TensorDTypeOps.toBFloat16Bits(acc);
        }
    }

    private static final class GatherAxisPlan {
        private final int[] inputShape;
        private final int[] inputStrides;
        private final int[] indicesDense;
        private final int[] outShape;
        private final int[] outDense;
        private final int[] outStrides;
        private final int inputBaseOffset;
        private final int outBaseOffset;
        private final int total;
        private final int axisSize;
        private final int indicesRank;

        private int sourceOffset;
        private int outOffset;

        private GatherAxisPlan(
                int[] inputShape,
                int[] inputStrides,
                int[] indicesDense,
                int[] outShape,
                int[] outDense,
                int[] outStrides,
                int inputBaseOffset,
                int outBaseOffset,
                int total,
                int axisSize,
                int indicesRank
        ) {
            this.inputShape = inputShape;
            this.inputStrides = inputStrides;
            this.indicesDense = indicesDense;
            this.outShape = outShape;
            this.outDense = outDense;
            this.outStrides = outStrides;
            this.inputBaseOffset = inputBaseOffset;
            this.outBaseOffset = outBaseOffset;
            this.total = total;
            this.axisSize = axisSize;
            this.indicesRank = indicesRank;
        }

        static GatherAxisPlan create(Tensor input, Tensor indices, Tensor out, int axis) {
            int[] inputShape = input.getShapeUnsafe();
            int[] indicesShape = indices.getShapeUnsafe();
            int[] outShape = out.getShapeUnsafe();
            return new GatherAxisPlan(
                    inputShape,
                    input.getStridesUnsafe(),
                    IndexLoopSupport.denseStrides(indicesShape),
                    outShape,
                    IndexLoopSupport.denseStrides(outShape),
                    out.getStridesUnsafe(),
                    input.getStorageOffsetUnsafe(),
                    out.getStorageOffsetUnsafe(),
                    out.getFlatDataSize(),
                    inputShape[axis],
                    indicesShape.length);
        }

        void computeOffsets(int outLogical, IndexLoopSupport.IndexReader indexReader, int axis) {
            int rem = outLogical;
            int source = inputBaseOffset;
            int output = outBaseOffset;
            int indexLogical = 0;
            for (int d = 0; d < outShape.length; d++) {
                int coord = rem / outDense[d];
                rem %= outDense[d];
                output += coord * outStrides[d];
                if (d < axis) {
                    source += coord * inputStrides[d];
                } else if (d < axis + indicesRank) {
                    indexLogical += coord * indicesDense[d - axis];
                } else {
                    int inputDim = d - indicesRank + 1;
                    source += coord * inputStrides[inputDim];
                }
            }
            int axisIndex = indexReader.readAxisIndexAllowNegative(indexLogical, axisSize);
            sourceOffset = source + axisIndex * inputStrides[axis];
            outOffset = output;
        }
    }
}
