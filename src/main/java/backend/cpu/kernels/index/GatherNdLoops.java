package backend.cpu.kernels.index;

import backend.cpu.storage.CpuStorageView;
import tensor.DataType;
import tensor.Tensor;

final class GatherNdLoops {
    private GatherNdLoops() {
    }

    static void gatherNdF64(
            Tensor input,
            Tensor indices,
            Tensor out,
            CpuStorageView inputView,
            CpuStorageView indicesView,
            CpuStorageView outView,
            int batchDims
    ) {
        IndexValidation.validateGatherNd(input, indices, out, batchDims);
        IndexLoopSupport.validateReadStorageViews(input, indices, out, inputView, indicesView, outView, DataType.FLOAT64);
        GatherNdPlan plan = GatherNdPlan.create(input, indices, out, batchDims);
        IndexLoopSupport.IndexStoragePlan indexPlan = IndexLoopSupport.indexStoragePlan(indices);
        if (IndexLoopSupport.allArrays(inputView, indicesView, outView)) {
            double[] in = inputView.requireF64Array();
            double[] dst = outView.requireF64Array();
            for (int logical = 0; logical < plan.total; logical++) {
                plan.computeOffsets(logical, indicesView, indexPlan, batchDims);
                dst[plan.outOffset] = in[plan.sourceOffset];
            }
            return;
        }
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indicesView, indexPlan, batchDims);
            IndexLoopSupport.writeF64(outView, plan.outOffset, IndexLoopSupport.readF64(inputView, plan.sourceOffset));
        }
    }

    static void gatherNdF32(
            Tensor input,
            Tensor indices,
            Tensor out,
            CpuStorageView inputView,
            CpuStorageView indicesView,
            CpuStorageView outView,
            int batchDims
    ) {
        IndexValidation.validateGatherNd(input, indices, out, batchDims);
        IndexLoopSupport.validateReadStorageViews(input, indices, out, inputView, indicesView, outView, DataType.FLOAT32);
        GatherNdPlan plan = GatherNdPlan.create(input, indices, out, batchDims);
        IndexLoopSupport.IndexStoragePlan indexPlan = IndexLoopSupport.indexStoragePlan(indices);
        if (IndexLoopSupport.allArrays(inputView, indicesView, outView)) {
            float[] in = inputView.requireF32Array();
            float[] dst = outView.requireF32Array();
            for (int logical = 0; logical < plan.total; logical++) {
                plan.computeOffsets(logical, indicesView, indexPlan, batchDims);
                dst[plan.outOffset] = in[plan.sourceOffset];
            }
            return;
        }
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indicesView, indexPlan, batchDims);
            IndexLoopSupport.writeF32(outView, plan.outOffset, IndexLoopSupport.readF32(inputView, plan.sourceOffset));
        }
    }

    static void gatherNdBF16(
            Tensor input,
            Tensor indices,
            Tensor out,
            CpuStorageView inputView,
            CpuStorageView indicesView,
            CpuStorageView outView,
            int batchDims
    ) {
        IndexValidation.validateGatherNd(input, indices, out, batchDims);
        IndexLoopSupport.validateReadStorageViews(input, indices, out, inputView, indicesView, outView, DataType.BFLOAT16);
        GatherNdPlan plan = GatherNdPlan.create(input, indices, out, batchDims);
        IndexLoopSupport.IndexStoragePlan indexPlan = IndexLoopSupport.indexStoragePlan(indices);
        if (IndexLoopSupport.allArrays(inputView, indicesView, outView)) {
            short[] in = inputView.requireBF16Array();
            short[] dst = outView.requireBF16Array();
            for (int logical = 0; logical < plan.total; logical++) {
                plan.computeOffsets(logical, indicesView, indexPlan, batchDims);
                dst[plan.outOffset] = in[plan.sourceOffset];
            }
            return;
        }
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indicesView, indexPlan, batchDims);
            IndexLoopSupport.writeBF16Bits(outView, plan.outOffset, IndexLoopSupport.readBF16Bits(inputView, plan.sourceOffset));
        }
    }

    static void gatherNdBOOL(
            Tensor input,
            Tensor indices,
            Tensor out,
            CpuStorageView inputView,
            CpuStorageView indicesView,
            CpuStorageView outView,
            int batchDims
    ) {
        IndexValidation.validateGatherNd(input, indices, out, batchDims);
        IndexLoopSupport.validateReadStorageViews(input, indices, out, inputView, indicesView, outView, DataType.BOOL);
        GatherNdPlan plan = GatherNdPlan.create(input, indices, out, batchDims);
        IndexLoopSupport.IndexStoragePlan indexPlan = IndexLoopSupport.indexStoragePlan(indices);
        if (IndexLoopSupport.allArrays(inputView, indicesView, outView)) {
            byte[] in = inputView.requireBoolArray();
            byte[] dst = outView.requireBoolArray();
            for (int logical = 0; logical < plan.total; logical++) {
                plan.computeOffsets(logical, indicesView, indexPlan, batchDims);
                dst[plan.outOffset] = in[plan.sourceOffset];
            }
            return;
        }
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indicesView, indexPlan, batchDims);
            IndexLoopSupport.writeBool(outView, plan.outOffset, IndexLoopSupport.readBool(inputView, plan.sourceOffset));
        }
    }

    static void gatherNdI32(
            Tensor input,
            Tensor indices,
            Tensor out,
            CpuStorageView inputView,
            CpuStorageView indicesView,
            CpuStorageView outView,
            int batchDims
    ) {
        IndexValidation.validateGatherNd(input, indices, out, batchDims);
        IndexLoopSupport.validateReadStorageViews(input, indices, out, inputView, indicesView, outView, DataType.INT32);
        GatherNdPlan plan = GatherNdPlan.create(input, indices, out, batchDims);
        IndexLoopSupport.IndexStoragePlan indexPlan = IndexLoopSupport.indexStoragePlan(indices);
        if (IndexLoopSupport.allArrays(inputView, indicesView, outView)) {
            int[] in = inputView.requireI32Array();
            int[] dst = outView.requireI32Array();
            for (int logical = 0; logical < plan.total; logical++) {
                plan.computeOffsets(logical, indicesView, indexPlan, batchDims);
                dst[plan.outOffset] = in[plan.sourceOffset];
            }
            return;
        }
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indicesView, indexPlan, batchDims);
            IndexLoopSupport.writeI32(outView, plan.outOffset, IndexLoopSupport.readI32(inputView, plan.sourceOffset));
        }
    }

    static void gatherNdI64(
            Tensor input,
            Tensor indices,
            Tensor out,
            CpuStorageView inputView,
            CpuStorageView indicesView,
            CpuStorageView outView,
            int batchDims
    ) {
        IndexValidation.validateGatherNd(input, indices, out, batchDims);
        IndexLoopSupport.validateReadStorageViews(input, indices, out, inputView, indicesView, outView, DataType.INT64);
        GatherNdPlan plan = GatherNdPlan.create(input, indices, out, batchDims);
        IndexLoopSupport.IndexStoragePlan indexPlan = IndexLoopSupport.indexStoragePlan(indices);
        if (IndexLoopSupport.allArrays(inputView, indicesView, outView)) {
            long[] in = inputView.requireI64Array();
            long[] dst = outView.requireI64Array();
            for (int logical = 0; logical < plan.total; logical++) {
                plan.computeOffsets(logical, indicesView, indexPlan, batchDims);
                dst[plan.outOffset] = in[plan.sourceOffset];
            }
            return;
        }
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indicesView, indexPlan, batchDims);
            IndexLoopSupport.writeI64(outView, plan.outOffset, IndexLoopSupport.readI64(inputView, plan.sourceOffset));
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

        void computeOffsets(
                int logical,
                CpuStorageView indices,
                IndexLoopSupport.IndexStoragePlan indexPlan,
                int batchDims
        ) {
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
                int coord = IndexLoopSupport.readAxisIndexAllowNegative(
                        indices,
                        indexPlan,
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
