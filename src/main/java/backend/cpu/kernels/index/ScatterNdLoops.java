package backend.cpu.kernels.index;

import operations.index.ScatterReduction;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.dtype.TensorDTypeOps;

final class ScatterNdLoops {
    private ScatterNdLoops() {
    }

    static void scatterNdF64(Tensor data, Tensor indices, Tensor updates, Tensor out, ScatterReduction reduction, int batchDims) {
        ScatterReduction effectiveReduction = IndexValidation.validateScatterNd(data, indices, updates, out, reduction, batchDims);
        out.copyDataFrom(data);
        double[] updateData = TensorInternalAccess.float64Data(updates);
        double[] dst = TensorInternalAccess.float64Data(out);
        IndexLoopSupport.DuplicateState state = IndexLoopSupport.duplicateState(out, effectiveReduction, "scatterNd");
        ScatterNdPlan plan = ScatterNdPlan.create(data, indices, updates, out);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indexReader, batchDims);
            state.mark(plan.targetLogical);
            dst[plan.targetOffset] = IndexLoopSupport.reduce(dst[plan.targetOffset], updateData[plan.updateOffset], effectiveReduction);
        }
    }

    static void scatterNdF32(Tensor data, Tensor indices, Tensor updates, Tensor out, ScatterReduction reduction, int batchDims) {
        ScatterReduction effectiveReduction = IndexValidation.validateScatterNd(data, indices, updates, out, reduction, batchDims);
        out.copyDataFrom(data);
        float[] updateData = TensorInternalAccess.float32Data(updates);
        float[] dst = TensorInternalAccess.float32Data(out);
        IndexLoopSupport.DuplicateState state = IndexLoopSupport.duplicateState(out, effectiveReduction, "scatterNd");
        ScatterNdPlan plan = ScatterNdPlan.create(data, indices, updates, out);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indexReader, batchDims);
            state.mark(plan.targetLogical);
            dst[plan.targetOffset] = (float) IndexLoopSupport.reduce(dst[plan.targetOffset], updateData[plan.updateOffset], effectiveReduction);
        }
    }

    static void scatterNdBF16(Tensor data, Tensor indices, Tensor updates, Tensor out, ScatterReduction reduction, int batchDims) {
        ScatterReduction effectiveReduction = IndexValidation.validateScatterNd(data, indices, updates, out, reduction, batchDims);
        out.copyDataFrom(data);
        short[] updateData = TensorInternalAccess.bfloat16Data(updates);
        short[] dst = TensorInternalAccess.bfloat16Data(out);
        IndexLoopSupport.DuplicateState state = IndexLoopSupport.duplicateState(out, effectiveReduction, "scatterNd");
        ScatterNdPlan plan = ScatterNdPlan.create(data, indices, updates, out);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indexReader, batchDims);
            state.mark(plan.targetLogical);
            float current = TensorDTypeOps.fromBFloat16Bits(dst[plan.targetOffset]);
            float update = TensorDTypeOps.fromBFloat16Bits(updateData[plan.updateOffset]);
            dst[plan.targetOffset] = TensorDTypeOps.toBFloat16Bits((float) IndexLoopSupport.reduce(current, update, effectiveReduction));
        }
    }

    static void scatterNdBOOL(Tensor data, Tensor indices, Tensor updates, Tensor out, ScatterReduction reduction, int batchDims) {
        ScatterReduction effectiveReduction = IndexValidation.validateScatterNd(data, indices, updates, out, reduction, batchDims);
        out.copyDataFrom(data);
        byte[] updateData = TensorInternalAccess.boolData(updates);
        byte[] dst = TensorInternalAccess.boolData(out);
        IndexLoopSupport.DuplicateState state = IndexLoopSupport.duplicateState(out, effectiveReduction, "scatterNd");
        ScatterNdPlan plan = ScatterNdPlan.create(data, indices, updates, out);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indexReader, batchDims);
            state.mark(plan.targetLogical);
            dst[plan.targetOffset] = updateData[plan.updateOffset] == 0 ? (byte) 0 : (byte) 1;
        }
    }

    static void scatterNdI32(Tensor data, Tensor indices, Tensor updates, Tensor out, ScatterReduction reduction, int batchDims) {
        ScatterReduction effectiveReduction = IndexValidation.validateScatterNd(data, indices, updates, out, reduction, batchDims);
        out.copyDataFrom(data);
        int[] updateData = TensorInternalAccess.int32Data(updates);
        int[] dst = TensorInternalAccess.int32Data(out);
        IndexLoopSupport.DuplicateState state = IndexLoopSupport.duplicateState(out, effectiveReduction, "scatterNd");
        ScatterNdPlan plan = ScatterNdPlan.create(data, indices, updates, out);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indexReader, batchDims);
            state.mark(plan.targetLogical);
            dst[plan.targetOffset] = IndexLoopSupport.reduceInt(dst[plan.targetOffset], updateData[plan.updateOffset], effectiveReduction);
        }
    }

    static void scatterNdI64(Tensor data, Tensor indices, Tensor updates, Tensor out, ScatterReduction reduction, int batchDims) {
        ScatterReduction effectiveReduction = IndexValidation.validateScatterNd(data, indices, updates, out, reduction, batchDims);
        out.copyDataFrom(data);
        long[] updateData = TensorInternalAccess.int64Data(updates);
        long[] dst = TensorInternalAccess.int64Data(out);
        IndexLoopSupport.DuplicateState state = IndexLoopSupport.duplicateState(out, effectiveReduction, "scatterNd");
        ScatterNdPlan plan = ScatterNdPlan.create(data, indices, updates, out);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indexReader, batchDims);
            state.mark(plan.targetLogical);
            dst[plan.targetOffset] = IndexLoopSupport.reduceLong(dst[plan.targetOffset], updateData[plan.updateOffset], effectiveReduction);
        }
    }

    private static final class ScatterNdPlan {
        private final int[] dataShape;
        private final int[] dataDense;
        private final int[] indicesDense;
        private final int[] updatesShape;
        private final int[] updatesStrides;
        private final int[] updatesDense;
        private final int[] outStrides;
        private final int updatesBaseOffset;
        private final int outBaseOffset;
        private final int total;
        private final int tupleRank;
        private final int prefixRank;
        private final int tupleStride;
        private final int[] coords;

        private int updateOffset;
        private int targetOffset;
        private int targetLogical;

        private ScatterNdPlan(
                int[] dataShape,
                int[] dataDense,
                int[] indicesDense,
                int[] updatesShape,
                int[] updatesStrides,
                int[] updatesDense,
                int[] outStrides,
                int updatesBaseOffset,
                int outBaseOffset,
                int total,
                int tupleRank,
                int prefixRank,
                int tupleStride
        ) {
            this.dataShape = dataShape;
            this.dataDense = dataDense;
            this.indicesDense = indicesDense;
            this.updatesShape = updatesShape;
            this.updatesStrides = updatesStrides;
            this.updatesDense = updatesDense;
            this.outStrides = outStrides;
            this.updatesBaseOffset = updatesBaseOffset;
            this.outBaseOffset = outBaseOffset;
            this.total = total;
            this.tupleRank = tupleRank;
            this.prefixRank = prefixRank;
            this.tupleStride = tupleStride;
            this.coords = new int[updatesShape.length];
        }

        static ScatterNdPlan create(Tensor data, Tensor indices, Tensor updates, Tensor out) {
            int[] dataShape = data.getShapeUnsafe();
            int[] indicesShape = indices.getShapeUnsafe();
            int[] indicesDense = IndexLoopSupport.denseStrides(indicesShape);
            int[] updatesShape = updates.getShapeUnsafe();
            return new ScatterNdPlan(
                    dataShape,
                    IndexLoopSupport.denseStrides(dataShape),
                    indicesDense,
                    updatesShape,
                    updates.getStridesUnsafe(),
                    IndexLoopSupport.denseStrides(updatesShape),
                    out.getStridesUnsafe(),
                    updates.getStorageOffsetUnsafe(),
                    out.getStorageOffsetUnsafe(),
                    updates.getFlatDataSize(),
                    indicesShape[indicesShape.length - 1],
                    indicesShape.length - 1,
                    indicesDense[indicesShape.length - 1]);
        }

        void computeOffsets(int logical, IndexLoopSupport.IndexReader indexReader, int batchDims) {
            int rem = logical;
            int update = updatesBaseOffset;
            for (int d = 0; d < updatesShape.length; d++) {
                int coord = rem / updatesDense[d];
                rem %= updatesDense[d];
                coords[d] = coord;
                update += coord * updatesStrides[d];
            }

            int indexBaseLogical = 0;
            for (int d = 0; d < prefixRank; d++) {
                indexBaseLogical += coords[d] * indicesDense[d];
            }

            int target = outBaseOffset;
            int targetFlat = 0;
            for (int d = 0; d < batchDims; d++) {
                int targetCoord = coords[d];
                target += targetCoord * outStrides[d];
                targetFlat += targetCoord * dataDense[d];
            }
            for (int d = 0; d < tupleRank; d++) {
                int dataDim = batchDims + d;
                int targetCoord = indexReader.readAxisIndexAllowNegative(
                        indexBaseLogical + d * tupleStride,
                        dataShape[dataDim]);
                target += targetCoord * outStrides[dataDim];
                targetFlat += targetCoord * dataDense[dataDim];
            }
            for (int d = batchDims + tupleRank; d < dataShape.length; d++) {
                int updateCoord = coords[prefixRank + d - batchDims - tupleRank];
                target += updateCoord * outStrides[d];
                targetFlat += updateCoord * dataDense[d];
            }
            updateOffset = update;
            targetOffset = target;
            targetLogical = targetFlat;
        }
    }
}
