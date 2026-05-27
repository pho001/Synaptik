package backend.cpu.kernels.index;

import backend.cpu.storage.CpuStorageView;
import operations.index.ScatterReduction;
import tensor.DataType;
import tensor.Tensor;
import tensor.dtype.TensorDTypeOps;

final class ScatterElementsLoops {
    private ScatterElementsLoops() {
    }

    static void scatterElementsF64(
            Tensor data,
            Tensor indices,
            Tensor updates,
            Tensor out,
            CpuStorageView dataView,
            CpuStorageView indicesView,
            CpuStorageView updatesView,
            CpuStorageView outView,
            int axis,
            ScatterReduction reduction
    ) {
        ScatterReduction effectiveReduction = IndexValidation.validateScatterElements(data, indices, updates, out, axis, reduction);
        IndexLoopSupport.validateScatterStorageViews(data, indices, updates, out,
                dataView, indicesView, updatesView, outView, DataType.FLOAT64);
        IndexLoopSupport.copyStorage(data, out, dataView, outView, DataType.FLOAT64);
        IndexLoopSupport.DuplicateState state = IndexLoopSupport.duplicateState(out, effectiveReduction, "scatterElements");
        ScatterElementsPlan plan = ScatterElementsPlan.create(data, updates, out, axis);
        IndexLoopSupport.IndexStoragePlan indexPlan = IndexLoopSupport.indexStoragePlan(indices);
        if (IndexLoopSupport.allArrays(dataView, indicesView, updatesView, outView)) {
            double[] updateData = updatesView.requireF64Array();
            double[] dst = outView.requireF64Array();
            for (int logical = 0; logical < plan.total; logical++) {
                plan.computeOffsets(logical, indicesView, indexPlan);
                state.mark(plan.targetLogical);
                dst[plan.targetOffset] = IndexLoopSupport.reduce(dst[plan.targetOffset], updateData[plan.updateOffset], effectiveReduction);
            }
            return;
        }
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indicesView, indexPlan);
            state.mark(plan.targetLogical);
            IndexLoopSupport.writeF64(outView, plan.targetOffset,
                    IndexLoopSupport.reduce(IndexLoopSupport.readF64(outView, plan.targetOffset),
                            IndexLoopSupport.readF64(updatesView, plan.updateOffset),
                            effectiveReduction));
        }
    }

    static void scatterElementsF32(
            Tensor data,
            Tensor indices,
            Tensor updates,
            Tensor out,
            CpuStorageView dataView,
            CpuStorageView indicesView,
            CpuStorageView updatesView,
            CpuStorageView outView,
            int axis,
            ScatterReduction reduction
    ) {
        ScatterReduction effectiveReduction = IndexValidation.validateScatterElements(data, indices, updates, out, axis, reduction);
        IndexLoopSupport.validateScatterStorageViews(data, indices, updates, out,
                dataView, indicesView, updatesView, outView, DataType.FLOAT32);
        IndexLoopSupport.copyStorage(data, out, dataView, outView, DataType.FLOAT32);
        IndexLoopSupport.DuplicateState state = IndexLoopSupport.duplicateState(out, effectiveReduction, "scatterElements");
        ScatterElementsPlan plan = ScatterElementsPlan.create(data, updates, out, axis);
        IndexLoopSupport.IndexStoragePlan indexPlan = IndexLoopSupport.indexStoragePlan(indices);
        if (IndexLoopSupport.allArrays(dataView, indicesView, updatesView, outView)) {
            float[] updateData = updatesView.requireF32Array();
            float[] dst = outView.requireF32Array();
            for (int logical = 0; logical < plan.total; logical++) {
                plan.computeOffsets(logical, indicesView, indexPlan);
                state.mark(plan.targetLogical);
                dst[plan.targetOffset] = (float) IndexLoopSupport.reduce(dst[plan.targetOffset], updateData[plan.updateOffset], effectiveReduction);
            }
            return;
        }
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indicesView, indexPlan);
            state.mark(plan.targetLogical);
            IndexLoopSupport.writeF32(outView, plan.targetOffset,
                    (float) IndexLoopSupport.reduce(IndexLoopSupport.readF32(outView, plan.targetOffset),
                            IndexLoopSupport.readF32(updatesView, plan.updateOffset),
                            effectiveReduction));
        }
    }

    static void scatterElementsBF16(
            Tensor data,
            Tensor indices,
            Tensor updates,
            Tensor out,
            CpuStorageView dataView,
            CpuStorageView indicesView,
            CpuStorageView updatesView,
            CpuStorageView outView,
            int axis,
            ScatterReduction reduction
    ) {
        ScatterReduction effectiveReduction = IndexValidation.validateScatterElements(data, indices, updates, out, axis, reduction);
        IndexLoopSupport.validateScatterStorageViews(data, indices, updates, out,
                dataView, indicesView, updatesView, outView, DataType.BFLOAT16);
        IndexLoopSupport.copyStorage(data, out, dataView, outView, DataType.BFLOAT16);
        IndexLoopSupport.DuplicateState state = IndexLoopSupport.duplicateState(out, effectiveReduction, "scatterElements");
        ScatterElementsPlan plan = ScatterElementsPlan.create(data, updates, out, axis);
        IndexLoopSupport.IndexStoragePlan indexPlan = IndexLoopSupport.indexStoragePlan(indices);
        if (IndexLoopSupport.allArrays(dataView, indicesView, updatesView, outView)) {
            short[] updateData = updatesView.requireBF16Array();
            short[] dst = outView.requireBF16Array();
            for (int logical = 0; logical < plan.total; logical++) {
                plan.computeOffsets(logical, indicesView, indexPlan);
                state.mark(plan.targetLogical);
                float current = TensorDTypeOps.fromBFloat16Bits(dst[plan.targetOffset]);
                float update = TensorDTypeOps.fromBFloat16Bits(updateData[plan.updateOffset]);
                dst[plan.targetOffset] = TensorDTypeOps.toBFloat16Bits((float) IndexLoopSupport.reduce(current, update, effectiveReduction));
            }
            return;
        }
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indicesView, indexPlan);
            state.mark(plan.targetLogical);
            float current = TensorDTypeOps.fromBFloat16Bits(IndexLoopSupport.readBF16Bits(outView, plan.targetOffset));
            float update = TensorDTypeOps.fromBFloat16Bits(IndexLoopSupport.readBF16Bits(updatesView, plan.updateOffset));
            IndexLoopSupport.writeBF16Bits(outView, plan.targetOffset,
                    TensorDTypeOps.toBFloat16Bits((float) IndexLoopSupport.reduce(current, update, effectiveReduction)));
        }
    }

    static void scatterElementsBOOL(
            Tensor data,
            Tensor indices,
            Tensor updates,
            Tensor out,
            CpuStorageView dataView,
            CpuStorageView indicesView,
            CpuStorageView updatesView,
            CpuStorageView outView,
            int axis,
            ScatterReduction reduction
    ) {
        ScatterReduction effectiveReduction = IndexValidation.validateScatterElements(data, indices, updates, out, axis, reduction);
        IndexLoopSupport.validateScatterStorageViews(data, indices, updates, out,
                dataView, indicesView, updatesView, outView, DataType.BOOL);
        IndexLoopSupport.copyStorage(data, out, dataView, outView, DataType.BOOL);
        IndexLoopSupport.DuplicateState state = IndexLoopSupport.duplicateState(out, effectiveReduction, "scatterElements");
        ScatterElementsPlan plan = ScatterElementsPlan.create(data, updates, out, axis);
        IndexLoopSupport.IndexStoragePlan indexPlan = IndexLoopSupport.indexStoragePlan(indices);
        if (IndexLoopSupport.allArrays(dataView, indicesView, updatesView, outView)) {
            byte[] updateData = updatesView.requireBoolArray();
            byte[] dst = outView.requireBoolArray();
            for (int logical = 0; logical < plan.total; logical++) {
                plan.computeOffsets(logical, indicesView, indexPlan);
                state.mark(plan.targetLogical);
                dst[plan.targetOffset] = updateData[plan.updateOffset] == 0 ? (byte) 0 : (byte) 1;
            }
            return;
        }
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indicesView, indexPlan);
            state.mark(plan.targetLogical);
            byte update = IndexLoopSupport.readBool(updatesView, plan.updateOffset);
            IndexLoopSupport.writeBool(outView, plan.targetOffset, update == 0 ? (byte) 0 : (byte) 1);
        }
    }

    static void scatterElementsI32(
            Tensor data,
            Tensor indices,
            Tensor updates,
            Tensor out,
            CpuStorageView dataView,
            CpuStorageView indicesView,
            CpuStorageView updatesView,
            CpuStorageView outView,
            int axis,
            ScatterReduction reduction
    ) {
        ScatterReduction effectiveReduction = IndexValidation.validateScatterElements(data, indices, updates, out, axis, reduction);
        IndexLoopSupport.validateScatterStorageViews(data, indices, updates, out,
                dataView, indicesView, updatesView, outView, DataType.INT32);
        IndexLoopSupport.copyStorage(data, out, dataView, outView, DataType.INT32);
        IndexLoopSupport.DuplicateState state = IndexLoopSupport.duplicateState(out, effectiveReduction, "scatterElements");
        ScatterElementsPlan plan = ScatterElementsPlan.create(data, updates, out, axis);
        IndexLoopSupport.IndexStoragePlan indexPlan = IndexLoopSupport.indexStoragePlan(indices);
        if (IndexLoopSupport.allArrays(dataView, indicesView, updatesView, outView)) {
            int[] updateData = updatesView.requireI32Array();
            int[] dst = outView.requireI32Array();
            for (int logical = 0; logical < plan.total; logical++) {
                plan.computeOffsets(logical, indicesView, indexPlan);
                state.mark(plan.targetLogical);
                dst[plan.targetOffset] = IndexLoopSupport.reduceInt(dst[plan.targetOffset], updateData[plan.updateOffset], effectiveReduction);
            }
            return;
        }
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indicesView, indexPlan);
            state.mark(plan.targetLogical);
            IndexLoopSupport.writeI32(outView, plan.targetOffset,
                    IndexLoopSupport.reduceInt(IndexLoopSupport.readI32(outView, plan.targetOffset),
                            IndexLoopSupport.readI32(updatesView, plan.updateOffset),
                            effectiveReduction));
        }
    }

    static void scatterElementsI64(
            Tensor data,
            Tensor indices,
            Tensor updates,
            Tensor out,
            CpuStorageView dataView,
            CpuStorageView indicesView,
            CpuStorageView updatesView,
            CpuStorageView outView,
            int axis,
            ScatterReduction reduction
    ) {
        ScatterReduction effectiveReduction = IndexValidation.validateScatterElements(data, indices, updates, out, axis, reduction);
        IndexLoopSupport.validateScatterStorageViews(data, indices, updates, out,
                dataView, indicesView, updatesView, outView, DataType.INT64);
        IndexLoopSupport.copyStorage(data, out, dataView, outView, DataType.INT64);
        IndexLoopSupport.DuplicateState state = IndexLoopSupport.duplicateState(out, effectiveReduction, "scatterElements");
        ScatterElementsPlan plan = ScatterElementsPlan.create(data, updates, out, axis);
        IndexLoopSupport.IndexStoragePlan indexPlan = IndexLoopSupport.indexStoragePlan(indices);
        if (IndexLoopSupport.allArrays(dataView, indicesView, updatesView, outView)) {
            long[] updateData = updatesView.requireI64Array();
            long[] dst = outView.requireI64Array();
            for (int logical = 0; logical < plan.total; logical++) {
                plan.computeOffsets(logical, indicesView, indexPlan);
                state.mark(plan.targetLogical);
                dst[plan.targetOffset] = IndexLoopSupport.reduceLong(dst[plan.targetOffset], updateData[plan.updateOffset], effectiveReduction);
            }
            return;
        }
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indicesView, indexPlan);
            state.mark(plan.targetLogical);
            IndexLoopSupport.writeI64(outView, plan.targetOffset,
                    IndexLoopSupport.reduceLong(IndexLoopSupport.readI64(outView, plan.targetOffset),
                            IndexLoopSupport.readI64(updatesView, plan.updateOffset),
                            effectiveReduction));
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

        void computeOffsets(
                int logical,
                CpuStorageView indices,
                IndexLoopSupport.IndexStoragePlan indexPlan
        ) {
            int rem = logical;
            int update = updatesBaseOffset;
            int target = outBaseOffset;
            int targetFlat = 0;
            for (int d = 0; d < updatesShape.length; d++) {
                int coord = rem / updatesDense[d];
                rem %= updatesDense[d];
                update += coord * updatesStrides[d];
                int targetCoord = d == axis
                        ? IndexLoopSupport.readAxisIndexAllowNegative(indices, indexPlan, logical, axisSize)
                        : coord;
                target += targetCoord * outStrides[d];
                targetFlat += targetCoord * dataDense[d];
            }
            updateOffset = update;
            targetOffset = target;
            targetLogical = targetFlat;
        }
    }
}
