package backend.cpu.kernels.index;

import operations.index.ScatterReduction;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.dtype.TensorDTypeOps;

final class ScatterElementsLoops {
    private ScatterElementsLoops() {
    }

    static void scatterElementsF64(Tensor data, Tensor indices, Tensor updates, Tensor out, int axis, ScatterReduction reduction) {
        ScatterReduction effectiveReduction = IndexValidation.validateScatterElements(data, indices, updates, out, axis, reduction);
        out.copyDataFrom(data);
        double[] updateData = TensorInternalAccess.float64Data(updates);
        double[] dst = TensorInternalAccess.float64Data(out);
        IndexLoopSupport.DuplicateState state = IndexLoopSupport.duplicateState(out, effectiveReduction, "scatterElements");
        ScatterElementsPlan plan = ScatterElementsPlan.create(data, updates, out, axis);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indexReader);
            state.mark(plan.targetLogical);
            dst[plan.targetOffset] = IndexLoopSupport.reduce(dst[plan.targetOffset], updateData[plan.updateOffset], effectiveReduction);
        }
    }

    static void scatterElementsF32(Tensor data, Tensor indices, Tensor updates, Tensor out, int axis, ScatterReduction reduction) {
        ScatterReduction effectiveReduction = IndexValidation.validateScatterElements(data, indices, updates, out, axis, reduction);
        out.copyDataFrom(data);
        float[] updateData = TensorInternalAccess.float32Data(updates);
        float[] dst = TensorInternalAccess.float32Data(out);
        IndexLoopSupport.DuplicateState state = IndexLoopSupport.duplicateState(out, effectiveReduction, "scatterElements");
        ScatterElementsPlan plan = ScatterElementsPlan.create(data, updates, out, axis);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indexReader);
            state.mark(plan.targetLogical);
            dst[plan.targetOffset] = (float) IndexLoopSupport.reduce(dst[plan.targetOffset], updateData[plan.updateOffset], effectiveReduction);
        }
    }

    static void scatterElementsBF16(Tensor data, Tensor indices, Tensor updates, Tensor out, int axis, ScatterReduction reduction) {
        ScatterReduction effectiveReduction = IndexValidation.validateScatterElements(data, indices, updates, out, axis, reduction);
        out.copyDataFrom(data);
        short[] updateData = TensorInternalAccess.bfloat16Data(updates);
        short[] dst = TensorInternalAccess.bfloat16Data(out);
        IndexLoopSupport.DuplicateState state = IndexLoopSupport.duplicateState(out, effectiveReduction, "scatterElements");
        ScatterElementsPlan plan = ScatterElementsPlan.create(data, updates, out, axis);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indexReader);
            state.mark(plan.targetLogical);
            float current = TensorDTypeOps.fromBFloat16Bits(dst[plan.targetOffset]);
            float update = TensorDTypeOps.fromBFloat16Bits(updateData[plan.updateOffset]);
            dst[plan.targetOffset] = TensorDTypeOps.toBFloat16Bits((float) IndexLoopSupport.reduce(current, update, effectiveReduction));
        }
    }

    static void scatterElementsBOOL(Tensor data, Tensor indices, Tensor updates, Tensor out, int axis, ScatterReduction reduction) {
        ScatterReduction effectiveReduction = IndexValidation.validateScatterElements(data, indices, updates, out, axis, reduction);
        out.copyDataFrom(data);
        byte[] updateData = TensorInternalAccess.boolData(updates);
        byte[] dst = TensorInternalAccess.boolData(out);
        IndexLoopSupport.DuplicateState state = IndexLoopSupport.duplicateState(out, effectiveReduction, "scatterElements");
        ScatterElementsPlan plan = ScatterElementsPlan.create(data, updates, out, axis);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indexReader);
            state.mark(plan.targetLogical);
            dst[plan.targetOffset] = updateData[plan.updateOffset] == 0 ? (byte) 0 : (byte) 1;
        }
    }

    static void scatterElementsI32(Tensor data, Tensor indices, Tensor updates, Tensor out, int axis, ScatterReduction reduction) {
        ScatterReduction effectiveReduction = IndexValidation.validateScatterElements(data, indices, updates, out, axis, reduction);
        out.copyDataFrom(data);
        int[] updateData = TensorInternalAccess.int32Data(updates);
        int[] dst = TensorInternalAccess.int32Data(out);
        IndexLoopSupport.DuplicateState state = IndexLoopSupport.duplicateState(out, effectiveReduction, "scatterElements");
        ScatterElementsPlan plan = ScatterElementsPlan.create(data, updates, out, axis);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indexReader);
            state.mark(plan.targetLogical);
            dst[plan.targetOffset] = IndexLoopSupport.reduceInt(dst[plan.targetOffset], updateData[plan.updateOffset], effectiveReduction);
        }
    }

    static void scatterElementsI64(Tensor data, Tensor indices, Tensor updates, Tensor out, int axis, ScatterReduction reduction) {
        ScatterReduction effectiveReduction = IndexValidation.validateScatterElements(data, indices, updates, out, axis, reduction);
        out.copyDataFrom(data);
        long[] updateData = TensorInternalAccess.int64Data(updates);
        long[] dst = TensorInternalAccess.int64Data(out);
        IndexLoopSupport.DuplicateState state = IndexLoopSupport.duplicateState(out, effectiveReduction, "scatterElements");
        ScatterElementsPlan plan = ScatterElementsPlan.create(data, updates, out, axis);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indexReader);
            state.mark(plan.targetLogical);
            dst[plan.targetOffset] = IndexLoopSupport.reduceLong(dst[plan.targetOffset], updateData[plan.updateOffset], effectiveReduction);
        }
    }

    private static final class ScatterElementsPlan {
        private final int[] dataShape;
        private final int[] dataDense;
        private final int[] updatesShape;
        private final int[] updatesStrides;
        private final int[] updatesDense;
        private final int[] outStrides;
        private final int updatesBaseOffset;
        private final int outBaseOffset;
        private final int total;
        private final int axis;
        private final int axisSize;

        private int updateOffset;
        private int targetOffset;
        private int targetLogical;

        private ScatterElementsPlan(
                int[] dataShape,
                int[] dataDense,
                int[] updatesShape,
                int[] updatesStrides,
                int[] updatesDense,
                int[] outStrides,
                int updatesBaseOffset,
                int outBaseOffset,
                int total,
                int axis,
                int axisSize
        ) {
            this.dataShape = dataShape;
            this.dataDense = dataDense;
            this.updatesShape = updatesShape;
            this.updatesStrides = updatesStrides;
            this.updatesDense = updatesDense;
            this.outStrides = outStrides;
            this.updatesBaseOffset = updatesBaseOffset;
            this.outBaseOffset = outBaseOffset;
            this.total = total;
            this.axis = axis;
            this.axisSize = axisSize;
        }

        static ScatterElementsPlan create(Tensor data, Tensor updates, Tensor out, int axis) {
            int[] dataShape = data.getShapeUnsafe();
            int[] updatesShape = updates.getShapeUnsafe();
            return new ScatterElementsPlan(
                    dataShape,
                    IndexLoopSupport.denseStrides(dataShape),
                    updatesShape,
                    updates.getStridesUnsafe(),
                    IndexLoopSupport.denseStrides(updatesShape),
                    out.getStridesUnsafe(),
                    updates.getStorageOffsetUnsafe(),
                    out.getStorageOffsetUnsafe(),
                    updates.getFlatDataSize(),
                    axis,
                    dataShape[axis]);
        }

        void computeOffsets(int logical, IndexLoopSupport.IndexReader indexReader) {
            int rem = logical;
            int update = updatesBaseOffset;
            int target = outBaseOffset;
            int targetFlat = 0;
            for (int d = 0; d < updatesShape.length; d++) {
                int coord = rem / updatesDense[d];
                rem %= updatesDense[d];
                update += coord * updatesStrides[d];
                int targetCoord = d == axis ? indexReader.readAxisIndexAllowNegative(logical, axisSize) : coord;
                target += targetCoord * outStrides[d];
                targetFlat += targetCoord * dataDense[d];
            }
            updateOffset = update;
            targetOffset = target;
            targetLogical = targetFlat;
        }
    }
}
