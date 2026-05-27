package backend.cpu.kernels.index;

import backend.cpu.storage.CpuStorageView;
import tensor.DataType;
import tensor.Tensor;
import tensor.dtype.TensorDTypeOps;

final class GatherAxisLoops {
    private GatherAxisLoops() {
    }

    static void gatherAxisF64(
            Tensor input,
            Tensor indices,
            Tensor out,
            CpuStorageView inputView,
            CpuStorageView indicesView,
            CpuStorageView outView,
            int axis
    ) {
        IndexValidation.validateGatherAxis(input, indices, out, axis);
        IndexLoopSupport.validateReadStorageViews(input, indices, out, inputView, indicesView, outView, DataType.FLOAT64);
        GatherAxisPlan plan = GatherAxisPlan.create(input, indices, out, axis);
        IndexLoopSupport.IndexStoragePlan indexPlan = IndexLoopSupport.indexStoragePlan(indicesView);
        if (IndexLoopSupport.allArrays(inputView, indicesView, outView)) {
            double[] in = inputView.requireF64Array();
            double[] dst = outView.requireF64Array();
            for (int outLogical = 0; outLogical < plan.total; outLogical++) {
                plan.computeOffsets(outLogical, indicesView, indexPlan, axis);
                dst[plan.outOffset] = in[plan.sourceOffset];
            }
            return;
        }
        for (int outLogical = 0; outLogical < plan.total; outLogical++) {
            plan.computeOffsets(outLogical, indicesView, indexPlan, axis);
            IndexLoopSupport.writeF64(outView, plan.outOffset, IndexLoopSupport.readF64(inputView, plan.sourceOffset));
        }
    }

    static void gatherAxisF32(
            Tensor input,
            Tensor indices,
            Tensor out,
            CpuStorageView inputView,
            CpuStorageView indicesView,
            CpuStorageView outView,
            int axis
    ) {
        IndexValidation.validateGatherAxis(input, indices, out, axis);
        IndexLoopSupport.validateReadStorageViews(input, indices, out, inputView, indicesView, outView, DataType.FLOAT32);
        GatherAxisPlan plan = GatherAxisPlan.create(input, indices, out, axis);
        IndexLoopSupport.IndexStoragePlan indexPlan = IndexLoopSupport.indexStoragePlan(indicesView);
        if (IndexLoopSupport.allArrays(inputView, indicesView, outView)) {
            float[] in = inputView.requireF32Array();
            float[] dst = outView.requireF32Array();
            for (int outLogical = 0; outLogical < plan.total; outLogical++) {
                plan.computeOffsets(outLogical, indicesView, indexPlan, axis);
                dst[plan.outOffset] = in[plan.sourceOffset];
            }
            return;
        }
        for (int outLogical = 0; outLogical < plan.total; outLogical++) {
            plan.computeOffsets(outLogical, indicesView, indexPlan, axis);
            IndexLoopSupport.writeF32(outView, plan.outOffset, IndexLoopSupport.readF32(inputView, plan.sourceOffset));
        }
    }

    static void gatherAxisBF16(
            Tensor input,
            Tensor indices,
            Tensor out,
            CpuStorageView inputView,
            CpuStorageView indicesView,
            CpuStorageView outView,
            int axis
    ) {
        IndexValidation.validateGatherAxis(input, indices, out, axis);
        IndexLoopSupport.validateReadStorageViews(input, indices, out, inputView, indicesView, outView, DataType.BFLOAT16);
        GatherAxisPlan plan = GatherAxisPlan.create(input, indices, out, axis);
        IndexLoopSupport.IndexStoragePlan indexPlan = IndexLoopSupport.indexStoragePlan(indicesView);
        if (IndexLoopSupport.allArrays(inputView, indicesView, outView)) {
            short[] in = inputView.requireBF16Array();
            short[] dst = outView.requireBF16Array();
            for (int outLogical = 0; outLogical < plan.total; outLogical++) {
                plan.computeOffsets(outLogical, indicesView, indexPlan, axis);
                dst[plan.outOffset] = in[plan.sourceOffset];
            }
            return;
        }
        for (int outLogical = 0; outLogical < plan.total; outLogical++) {
            plan.computeOffsets(outLogical, indicesView, indexPlan, axis);
            IndexLoopSupport.writeBF16Bits(outView, plan.outOffset, IndexLoopSupport.readBF16Bits(inputView, plan.sourceOffset));
        }
    }

    static void gatherAxisBOOL(
            Tensor input,
            Tensor indices,
            Tensor out,
            CpuStorageView inputView,
            CpuStorageView indicesView,
            CpuStorageView outView,
            int axis
    ) {
        IndexValidation.validateGatherAxis(input, indices, out, axis);
        IndexLoopSupport.validateReadStorageViews(input, indices, out, inputView, indicesView, outView, DataType.BOOL);
        GatherAxisPlan plan = GatherAxisPlan.create(input, indices, out, axis);
        IndexLoopSupport.IndexStoragePlan indexPlan = IndexLoopSupport.indexStoragePlan(indicesView);
        if (IndexLoopSupport.allArrays(inputView, indicesView, outView)) {
            byte[] in = inputView.requireBoolArray();
            byte[] dst = outView.requireBoolArray();
            for (int outLogical = 0; outLogical < plan.total; outLogical++) {
                plan.computeOffsets(outLogical, indicesView, indexPlan, axis);
                dst[plan.outOffset] = in[plan.sourceOffset];
            }
            return;
        }
        for (int outLogical = 0; outLogical < plan.total; outLogical++) {
            plan.computeOffsets(outLogical, indicesView, indexPlan, axis);
            IndexLoopSupport.writeBool(outView, plan.outOffset, IndexLoopSupport.readBool(inputView, plan.sourceOffset));
        }
    }

    static void gatherAxisI32(
            Tensor input,
            Tensor indices,
            Tensor out,
            CpuStorageView inputView,
            CpuStorageView indicesView,
            CpuStorageView outView,
            int axis
    ) {
        IndexValidation.validateGatherAxis(input, indices, out, axis);
        IndexLoopSupport.validateReadStorageViews(input, indices, out, inputView, indicesView, outView, DataType.INT32);
        GatherAxisPlan plan = GatherAxisPlan.create(input, indices, out, axis);
        IndexLoopSupport.IndexStoragePlan indexPlan = IndexLoopSupport.indexStoragePlan(indicesView);
        if (IndexLoopSupport.allArrays(inputView, indicesView, outView)) {
            int[] in = inputView.requireI32Array();
            int[] dst = outView.requireI32Array();
            for (int outLogical = 0; outLogical < plan.total; outLogical++) {
                plan.computeOffsets(outLogical, indicesView, indexPlan, axis);
                dst[plan.outOffset] = in[plan.sourceOffset];
            }
            return;
        }
        for (int outLogical = 0; outLogical < plan.total; outLogical++) {
            plan.computeOffsets(outLogical, indicesView, indexPlan, axis);
            IndexLoopSupport.writeI32(outView, plan.outOffset, IndexLoopSupport.readI32(inputView, plan.sourceOffset));
        }
    }

    static void gatherAxisI64(
            Tensor input,
            Tensor indices,
            Tensor out,
            CpuStorageView inputView,
            CpuStorageView indicesView,
            CpuStorageView outView,
            int axis
    ) {
        IndexValidation.validateGatherAxis(input, indices, out, axis);
        IndexLoopSupport.validateReadStorageViews(input, indices, out, inputView, indicesView, outView, DataType.INT64);
        GatherAxisPlan plan = GatherAxisPlan.create(input, indices, out, axis);
        IndexLoopSupport.IndexStoragePlan indexPlan = IndexLoopSupport.indexStoragePlan(indicesView);
        if (IndexLoopSupport.allArrays(inputView, indicesView, outView)) {
            long[] in = inputView.requireI64Array();
            long[] dst = outView.requireI64Array();
            for (int outLogical = 0; outLogical < plan.total; outLogical++) {
                plan.computeOffsets(outLogical, indicesView, indexPlan, axis);
                dst[plan.outOffset] = in[plan.sourceOffset];
            }
            return;
        }
        for (int outLogical = 0; outLogical < plan.total; outLogical++) {
            plan.computeOffsets(outLogical, indicesView, indexPlan, axis);
            IndexLoopSupport.writeI64(outView, plan.outOffset, IndexLoopSupport.readI64(inputView, plan.sourceOffset));
        }
    }

    static void scatterAxisAddF64(
            Tensor data,
            Tensor indices,
            Tensor updates,
            Tensor out,
            CpuStorageView dataView,
            CpuStorageView indicesView,
            CpuStorageView updatesView,
            CpuStorageView outView,
            int axis
    ) {
        IndexValidation.validateScatterAxisAdd(data, indices, updates, out, axis);
        IndexLoopSupport.validateScatterStorageViews(data, indices, updates, out,
                dataView, indicesView, updatesView, outView, DataType.FLOAT64);
        IndexLoopSupport.copyStorage(data, out, dataView, outView, DataType.FLOAT64);
        GatherAxisPlan plan = GatherAxisPlan.create(out, indices, updates, axis);
        IndexLoopSupport.IndexStoragePlan indexPlan = IndexLoopSupport.indexStoragePlan(indicesView);
        if (IndexLoopSupport.allArrays(dataView, indicesView, updatesView, outView)) {
            double[] updateData = updatesView.requireF64Array();
            double[] dst = outView.requireF64Array();
            for (int updateLogical = 0; updateLogical < plan.total; updateLogical++) {
                plan.computeOffsets(updateLogical, indicesView, indexPlan, axis);
                dst[plan.sourceOffset] += updateData[plan.outOffset];
            }
            return;
        }
        for (int updateLogical = 0; updateLogical < plan.total; updateLogical++) {
            plan.computeOffsets(updateLogical, indicesView, indexPlan, axis);
            IndexLoopSupport.writeF64(outView, plan.sourceOffset,
                    IndexLoopSupport.readF64(outView, plan.sourceOffset)
                            + IndexLoopSupport.readF64(updatesView, plan.outOffset));
        }
    }

    static void scatterAxisAddF32(
            Tensor data,
            Tensor indices,
            Tensor updates,
            Tensor out,
            CpuStorageView dataView,
            CpuStorageView indicesView,
            CpuStorageView updatesView,
            CpuStorageView outView,
            int axis
    ) {
        IndexValidation.validateScatterAxisAdd(data, indices, updates, out, axis);
        IndexLoopSupport.validateScatterStorageViews(data, indices, updates, out,
                dataView, indicesView, updatesView, outView, DataType.FLOAT32);
        IndexLoopSupport.copyStorage(data, out, dataView, outView, DataType.FLOAT32);
        GatherAxisPlan plan = GatherAxisPlan.create(out, indices, updates, axis);
        IndexLoopSupport.IndexStoragePlan indexPlan = IndexLoopSupport.indexStoragePlan(indicesView);
        if (IndexLoopSupport.allArrays(dataView, indicesView, updatesView, outView)) {
            float[] updateData = updatesView.requireF32Array();
            float[] dst = outView.requireF32Array();
            for (int updateLogical = 0; updateLogical < plan.total; updateLogical++) {
                plan.computeOffsets(updateLogical, indicesView, indexPlan, axis);
                dst[plan.sourceOffset] += updateData[plan.outOffset];
            }
            return;
        }
        for (int updateLogical = 0; updateLogical < plan.total; updateLogical++) {
            plan.computeOffsets(updateLogical, indicesView, indexPlan, axis);
            IndexLoopSupport.writeF32(outView, plan.sourceOffset,
                    IndexLoopSupport.readF32(outView, plan.sourceOffset)
                            + IndexLoopSupport.readF32(updatesView, plan.outOffset));
        }
    }

    static void scatterAxisAddBF16(
            Tensor data,
            Tensor indices,
            Tensor updates,
            Tensor out,
            CpuStorageView dataView,
            CpuStorageView indicesView,
            CpuStorageView updatesView,
            CpuStorageView outView,
            int axis
    ) {
        IndexValidation.validateScatterAxisAdd(data, indices, updates, out, axis);
        IndexLoopSupport.validateScatterStorageViews(data, indices, updates, out,
                dataView, indicesView, updatesView, outView, DataType.BFLOAT16);
        IndexLoopSupport.copyStorage(data, out, dataView, outView, DataType.BFLOAT16);
        GatherAxisPlan plan = GatherAxisPlan.create(out, indices, updates, axis);
        IndexLoopSupport.IndexStoragePlan indexPlan = IndexLoopSupport.indexStoragePlan(indicesView);
        if (IndexLoopSupport.allArrays(dataView, indicesView, updatesView, outView)) {
            short[] updateData = updatesView.requireBF16Array();
            short[] dst = outView.requireBF16Array();
            for (int updateLogical = 0; updateLogical < plan.total; updateLogical++) {
                plan.computeOffsets(updateLogical, indicesView, indexPlan, axis);
                float acc = TensorDTypeOps.fromBFloat16Bits(dst[plan.sourceOffset])
                        + TensorDTypeOps.fromBFloat16Bits(updateData[plan.outOffset]);
                dst[plan.sourceOffset] = TensorDTypeOps.toBFloat16Bits(acc);
            }
            return;
        }
        for (int updateLogical = 0; updateLogical < plan.total; updateLogical++) {
            plan.computeOffsets(updateLogical, indicesView, indexPlan, axis);
            float acc = TensorDTypeOps.fromBFloat16Bits(IndexLoopSupport.readBF16Bits(outView, plan.sourceOffset))
                    + TensorDTypeOps.fromBFloat16Bits(IndexLoopSupport.readBF16Bits(updatesView, plan.outOffset));
            IndexLoopSupport.writeBF16Bits(outView, plan.sourceOffset, TensorDTypeOps.toBFloat16Bits(acc));
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

        void computeOffsets(
                int outLogical,
                CpuStorageView indices,
                IndexLoopSupport.IndexStoragePlan indexPlan,
                int axis
        ) {
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
            int axisIndex = IndexLoopSupport.readAxisIndexAllowNegative(indices, indexPlan, indexLogical, axisSize);
            sourceOffset = source + axisIndex * inputStrides[axis];
            outOffset = output;
        }
    }
}
