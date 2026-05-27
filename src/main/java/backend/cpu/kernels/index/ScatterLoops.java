package backend.cpu.kernels.index;

import backend.cpu.storage.CpuStorageView;
import tensor.DataType;
import tensor.Tensor;
import tensor.dtype.TensorDTypeOps;

final class ScatterLoops {
    private ScatterLoops() {
    }

    static void scatterAddF64(
            Tensor base,
            Tensor indices,
            Tensor src,
            Tensor out,
            CpuStorageView baseView,
            CpuStorageView indicesView,
            CpuStorageView srcView,
            CpuStorageView outView,
            int dimension
    ) {
        IndexValidation.validateScatterAdd(base, indices, src, out, dimension);
        IndexLoopSupport.validateScatterStorageViews(base, indices, src, out,
                baseView, indicesView, srcView, outView, DataType.FLOAT64);
        IndexLoopSupport.copyStorage(base, out, baseView, outView, DataType.FLOAT64);
        ScatterAddPlan plan = ScatterAddPlan.create(src, out, dimension);
        IndexLoopSupport.IndexStoragePlan indexPlan = IndexLoopSupport.indexStoragePlan(indices);
        if (IndexLoopSupport.allArrays(baseView, indicesView, srcView, outView)) {
            scatterAddF64Array(srcView.requireF64Array(), indicesView, outView.requireF64Array(), indexPlan, plan);
            return;
        }
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indicesView, indexPlan);
            IndexLoopSupport.writeF64(outView, plan.targetOffset,
                    IndexLoopSupport.readF64(outView, plan.targetOffset)
                            + IndexLoopSupport.readF64(srcView, plan.srcOffset));
        }
    }

    static void scatterAddF32(
            Tensor base,
            Tensor indices,
            Tensor src,
            Tensor out,
            CpuStorageView baseView,
            CpuStorageView indicesView,
            CpuStorageView srcView,
            CpuStorageView outView,
            int dimension
    ) {
        IndexValidation.validateScatterAdd(base, indices, src, out, dimension);
        IndexLoopSupport.validateScatterStorageViews(base, indices, src, out,
                baseView, indicesView, srcView, outView, DataType.FLOAT32);
        IndexLoopSupport.copyStorage(base, out, baseView, outView, DataType.FLOAT32);
        ScatterAddPlan plan = ScatterAddPlan.create(src, out, dimension);
        IndexLoopSupport.IndexStoragePlan indexPlan = IndexLoopSupport.indexStoragePlan(indices);
        if (IndexLoopSupport.allArrays(baseView, indicesView, srcView, outView)) {
            scatterAddF32Array(srcView.requireF32Array(), indicesView, outView.requireF32Array(), indexPlan, plan);
            return;
        }
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indicesView, indexPlan);
            IndexLoopSupport.writeF32(outView, plan.targetOffset,
                    IndexLoopSupport.readF32(outView, plan.targetOffset)
                            + IndexLoopSupport.readF32(srcView, plan.srcOffset));
        }
    }

    static void scatterAddBF16(
            Tensor base,
            Tensor indices,
            Tensor src,
            Tensor out,
            CpuStorageView baseView,
            CpuStorageView indicesView,
            CpuStorageView srcView,
            CpuStorageView outView,
            int dimension
    ) {
        IndexValidation.validateScatterAdd(base, indices, src, out, dimension);
        IndexLoopSupport.validateScatterStorageViews(base, indices, src, out,
                baseView, indicesView, srcView, outView, DataType.BFLOAT16);
        IndexLoopSupport.copyStorage(base, out, baseView, outView, DataType.BFLOAT16);
        ScatterAddPlan plan = ScatterAddPlan.create(src, out, dimension);
        IndexLoopSupport.IndexStoragePlan indexPlan = IndexLoopSupport.indexStoragePlan(indices);
        if (IndexLoopSupport.allArrays(baseView, indicesView, srcView, outView)) {
            scatterAddBF16Array(srcView.requireBF16Array(), indicesView, outView.requireBF16Array(), indexPlan, plan);
            return;
        }
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indicesView, indexPlan);
            float acc = TensorDTypeOps.fromBFloat16Bits(IndexLoopSupport.readBF16Bits(outView, plan.targetOffset))
                    + TensorDTypeOps.fromBFloat16Bits(IndexLoopSupport.readBF16Bits(srcView, plan.srcOffset));
            IndexLoopSupport.writeBF16Bits(outView, plan.targetOffset, TensorDTypeOps.toBFloat16Bits(acc));
        }
    }

    private static void scatterAddF64Array(
            double[] srcData,
            CpuStorageView indicesView,
            double[] dst,
            IndexLoopSupport.IndexStoragePlan indexPlan,
            ScatterAddPlan plan
    ) {
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indicesView, indexPlan);
            dst[plan.targetOffset] += srcData[plan.srcOffset];
        }
    }

    private static void scatterAddF32Array(
            float[] srcData,
            CpuStorageView indicesView,
            float[] dst,
            IndexLoopSupport.IndexStoragePlan indexPlan,
            ScatterAddPlan plan
    ) {
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indicesView, indexPlan);
            dst[plan.targetOffset] += srcData[plan.srcOffset];
        }
    }

    private static void scatterAddBF16Array(
            short[] srcData,
            CpuStorageView indicesView,
            short[] dst,
            IndexLoopSupport.IndexStoragePlan indexPlan,
            ScatterAddPlan plan
    ) {
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indicesView, indexPlan);
            float acc = TensorDTypeOps.fromBFloat16Bits(dst[plan.targetOffset])
                    + TensorDTypeOps.fromBFloat16Bits(srcData[plan.srcOffset]);
            dst[plan.targetOffset] = TensorDTypeOps.toBFloat16Bits(acc);
        }
    }

    private static final class ScatterAddPlan {
        private final int[] outShape;
        private final int[] outStrides;
        private final int[] srcStrides;
        private final int[] reducedDense;
        private final int outBaseOffset;
        private final int srcBaseOffset;
        private final int total;
        private final int axisSize;
        private final int axisStrideOut;
        private final int dimension;

        private int targetOffset;
        private int srcOffset;

        private ScatterAddPlan(
                int[] outShape,
                int[] outStrides,
                int[] srcStrides,
                int[] reducedDense,
                int outBaseOffset,
                int srcBaseOffset,
                int total,
                int axisSize,
                int axisStrideOut,
                int dimension
        ) {
            this.outShape = outShape;
            this.outStrides = outStrides;
            this.srcStrides = srcStrides;
            this.reducedDense = reducedDense;
            this.outBaseOffset = outBaseOffset;
            this.srcBaseOffset = srcBaseOffset;
            this.total = total;
            this.axisSize = axisSize;
            this.axisStrideOut = axisStrideOut;
            this.dimension = dimension;
        }

        void computeOffsets(
                int logical,
                CpuStorageView indices,
                IndexLoopSupport.IndexStoragePlan indexPlan
        ) {
            int baseOut = outBaseOffset;
            int src = srcBaseOffset;
            int rem = logical;
            for (int d = 0, rd = 0; d < outShape.length; d++) {
                if (d == dimension) {
                    continue;
                }
                int coord = rem / reducedDense[rd];
                rem %= reducedDense[rd];
                baseOut += coord * outStrides[d];
                src += coord * srcStrides[rd];
                rd++;
            }
            int axisIndex = IndexLoopSupport.readAxisIndex(indices, indexPlan, logical, axisSize);
            targetOffset = baseOut + axisIndex * axisStrideOut;
            srcOffset = src;
        }

        static ScatterAddPlan create(Tensor src, Tensor out, int dimension) {
            int[] outShape = out.getShapeUnsafe();
            return new ScatterAddPlan(
                    outShape,
                    out.getStridesUnsafe(),
                    src.getStridesUnsafe(),
                    IndexLoopSupport.denseStrides(src.getShapeUnsafe()),
                    out.getStorageOffsetUnsafe(),
                    src.getStorageOffsetUnsafe(),
                    src.getFlatDataSize(),
                    outShape[dimension],
                    out.getStridesUnsafe()[dimension],
                    dimension);
        }
    }
}
