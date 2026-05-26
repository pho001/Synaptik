package backend.cpu.kernels.index;

import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.dtype.TensorDTypeOps;

final class GatherNdLoops {
    private GatherNdLoops() {
    }

    static void gatherNdF64(Tensor input, Tensor indices, Tensor out, int batchDims) {
        IndexValidation.validateGatherNd(input, indices, out, batchDims);
        double[] in = TensorInternalAccess.float64Data(input);
        double[] dst = TensorInternalAccess.float64Data(out);
        GatherNdPlan plan = GatherNdPlan.create(input, indices, out, batchDims);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indexReader, batchDims);
            dst[plan.outOffset] = in[plan.sourceOffset];
        }
    }

    static void gatherNdF32(Tensor input, Tensor indices, Tensor out, int batchDims) {
        IndexValidation.validateGatherNd(input, indices, out, batchDims);
        float[] in = TensorInternalAccess.float32Data(input);
        float[] dst = TensorInternalAccess.float32Data(out);
        GatherNdPlan plan = GatherNdPlan.create(input, indices, out, batchDims);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indexReader, batchDims);
            dst[plan.outOffset] = in[plan.sourceOffset];
        }
    }

    static void gatherNdBF16(Tensor input, Tensor indices, Tensor out, int batchDims) {
        IndexValidation.validateGatherNd(input, indices, out, batchDims);
        short[] in = TensorInternalAccess.bfloat16Data(input);
        short[] dst = TensorInternalAccess.bfloat16Data(out);
        GatherNdPlan plan = GatherNdPlan.create(input, indices, out, batchDims);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indexReader, batchDims);
            dst[plan.outOffset] = in[plan.sourceOffset];
        }
    }

    static void gatherNdBOOL(Tensor input, Tensor indices, Tensor out, int batchDims) {
        IndexValidation.validateGatherNd(input, indices, out, batchDims);
        byte[] in = TensorInternalAccess.boolData(input);
        byte[] dst = TensorInternalAccess.boolData(out);
        GatherNdPlan plan = GatherNdPlan.create(input, indices, out, batchDims);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indexReader, batchDims);
            dst[plan.outOffset] = in[plan.sourceOffset];
        }
    }

    static void gatherNdI32(Tensor input, Tensor indices, Tensor out, int batchDims) {
        IndexValidation.validateGatherNd(input, indices, out, batchDims);
        int[] in = TensorInternalAccess.int32Data(input);
        int[] dst = TensorInternalAccess.int32Data(out);
        GatherNdPlan plan = GatherNdPlan.create(input, indices, out, batchDims);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indexReader, batchDims);
            dst[plan.outOffset] = in[plan.sourceOffset];
        }
    }

    static void gatherNdI64(Tensor input, Tensor indices, Tensor out, int batchDims) {
        IndexValidation.validateGatherNd(input, indices, out, batchDims);
        long[] in = TensorInternalAccess.int64Data(input);
        long[] dst = TensorInternalAccess.int64Data(out);
        GatherNdPlan plan = GatherNdPlan.create(input, indices, out, batchDims);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indexReader, batchDims);
            dst[plan.outOffset] = in[plan.sourceOffset];
        }
    }

    static void gatherNdGradF64(Tensor indices, Tensor outGrad, Tensor node, int batchDims) {
        IndexValidation.validateGatherNdGrad(indices, outGrad, node, batchDims);
        IndexLoopSupport.fillZeroF64(node);
        double[] grad = TensorInternalAccess.float64Data(outGrad);
        double[] dst = TensorInternalAccess.float64Data(node);
        GatherNdPlan plan = GatherNdPlan.create(node, indices, outGrad, batchDims);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indexReader, batchDims);
            dst[plan.sourceOffset] += grad[plan.outOffset];
        }
    }

    static void gatherNdGradF32(Tensor indices, Tensor outGrad, Tensor node, int batchDims) {
        IndexValidation.validateGatherNdGrad(indices, outGrad, node, batchDims);
        IndexLoopSupport.fillZeroF32(node);
        float[] grad = TensorInternalAccess.float32Data(outGrad);
        float[] dst = TensorInternalAccess.float32Data(node);
        GatherNdPlan plan = GatherNdPlan.create(node, indices, outGrad, batchDims);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indexReader, batchDims);
            dst[plan.sourceOffset] += grad[plan.outOffset];
        }
    }

    static void gatherNdGradBF16(Tensor indices, Tensor outGrad, Tensor node, int batchDims) {
        IndexValidation.validateGatherNdGrad(indices, outGrad, node, batchDims);
        IndexLoopSupport.fillZeroBF16(node);
        short[] grad = TensorInternalAccess.bfloat16Data(outGrad);
        short[] dst = TensorInternalAccess.bfloat16Data(node);
        GatherNdPlan plan = GatherNdPlan.create(node, indices, outGrad, batchDims);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indexReader, batchDims);
            float acc = TensorDTypeOps.fromBFloat16Bits(dst[plan.sourceOffset])
                    + TensorDTypeOps.fromBFloat16Bits(grad[plan.outOffset]);
            dst[plan.sourceOffset] = TensorDTypeOps.toBFloat16Bits(acc);
        }
    }

    private static final class GatherNdPlan {
        private final int[] inputShape;
        private final int[] inputStrides;
        private final int[] indicesDense;
        private final int[] outShape;
        private final int[] outStrides;
        private final int[] outDense;
        private final int inputBaseOffset;
        private final int outBaseOffset;
        private final int total;
        private final int tupleRank;
        private final int prefixRank;
        private final int tupleStride;
        private final int[] coords;

        private int sourceOffset;
        private int outOffset;

        private GatherNdPlan(
                int[] inputShape,
                int[] inputStrides,
                int[] indicesDense,
                int[] outShape,
                int[] outStrides,
                int[] outDense,
                int inputBaseOffset,
                int outBaseOffset,
                int total,
                int tupleRank,
                int prefixRank,
                int tupleStride
        ) {
            this.inputShape = inputShape;
            this.inputStrides = inputStrides;
            this.indicesDense = indicesDense;
            this.outShape = outShape;
            this.outStrides = outStrides;
            this.outDense = outDense;
            this.inputBaseOffset = inputBaseOffset;
            this.outBaseOffset = outBaseOffset;
            this.total = total;
            this.tupleRank = tupleRank;
            this.prefixRank = prefixRank;
            this.tupleStride = tupleStride;
            this.coords = new int[outShape.length];
        }

        static GatherNdPlan create(Tensor input, Tensor indices, Tensor out, int batchDims) {
            int[] indicesShape = indices.getShapeUnsafe();
            int[] indicesDense = IndexLoopSupport.denseStrides(indicesShape);
            int[] outShape = out.getShapeUnsafe();
            return new GatherNdPlan(
                    input.getShapeUnsafe(),
                    input.getStridesUnsafe(),
                    indicesDense,
                    outShape,
                    out.getStridesUnsafe(),
                    IndexLoopSupport.denseStrides(outShape),
                    input.getStorageOffsetUnsafe(),
                    out.getStorageOffsetUnsafe(),
                    out.getFlatDataSize(),
                    indicesShape[indicesShape.length - 1],
                    indicesShape.length - 1,
                    indicesDense[indicesShape.length - 1]);
        }

        void computeOffsets(int logical, IndexLoopSupport.IndexReader indexReader, int batchDims) {
            int rem = logical;
            int output = outBaseOffset;
            for (int d = 0; d < outShape.length; d++) {
                int coord = rem / outDense[d];
                rem %= outDense[d];
                coords[d] = coord;
                output += coord * outStrides[d];
            }

            int indexBaseLogical = 0;
            for (int d = 0; d < prefixRank; d++) {
                indexBaseLogical += coords[d] * indicesDense[d];
            }

            int source = inputBaseOffset;
            for (int d = 0; d < batchDims; d++) {
                source += coords[d] * inputStrides[d];
            }
            for (int d = 0; d < tupleRank; d++) {
                int inputDim = batchDims + d;
                int coord = indexReader.readAxisIndexAllowNegative(
                        indexBaseLogical + d * tupleStride,
                        inputShape[inputDim]);
                source += coord * inputStrides[inputDim];
            }
            for (int d = batchDims + tupleRank; d < inputShape.length; d++) {
                int suffixCoord = coords[prefixRank + d - batchDims - tupleRank];
                source += suffixCoord * inputStrides[d];
            }
            sourceOffset = source;
            outOffset = output;
        }
    }
}
