package backend.cpu.kernels.index;

import backend.cpu.storage.CpuStorageView;
import tensor.DataType;
import tensor.Tensor;

final class GatherLoops {
    private GatherLoops() {
    }

    static void gatherF64(
            Tensor input,
            Tensor indices,
            Tensor out,
            CpuStorageView inputView,
            CpuStorageView indicesView,
            CpuStorageView outView,
            int dimension
    ) {
        IndexValidation.validateGather(input, indices, out, dimension);
        IndexLoopSupport.validateReadStorageViews(input, indices, out, inputView, indicesView, outView, DataType.FLOAT64);
        GatherPlan plan = GatherPlan.create(input, out, dimension);
        IndexLoopSupport.IndexStoragePlan indexPlan = IndexLoopSupport.indexStoragePlan(indicesView);
        if (IndexLoopSupport.allArrays(inputView, indicesView, outView)) {
            gatherF64Array(inputView.requireF64Array(), indicesView, outView.requireF64Array(), indexPlan, plan);
            return;
        }
        for (int logical = 0; logical < plan.total; logical++) {
            long offsets = plan.baseOffsets(logical);
            int baseIn = GatherPlan.inputOffset(offsets);
            int baseOut = GatherPlan.outputOffset(offsets);
            int axisIndex = IndexLoopSupport.readAxisIndex(indicesView, indexPlan, logical, plan.axisSize);
            IndexLoopSupport.writeF64(outView, baseOut,
                    IndexLoopSupport.readF64(inputView, baseIn + axisIndex * plan.axisStrideIn));
        }
    }

    static void gatherF32(
            Tensor input,
            Tensor indices,
            Tensor out,
            CpuStorageView inputView,
            CpuStorageView indicesView,
            CpuStorageView outView,
            int dimension
    ) {
        IndexValidation.validateGather(input, indices, out, dimension);
        IndexLoopSupport.validateReadStorageViews(input, indices, out, inputView, indicesView, outView, DataType.FLOAT32);
        GatherPlan plan = GatherPlan.create(input, out, dimension);
        IndexLoopSupport.IndexStoragePlan indexPlan = IndexLoopSupport.indexStoragePlan(indicesView);
        if (IndexLoopSupport.allArrays(inputView, indicesView, outView)) {
            gatherF32Array(inputView.requireF32Array(), indicesView, outView.requireF32Array(), indexPlan, plan);
            return;
        }
        for (int logical = 0; logical < plan.total; logical++) {
            long offsets = plan.baseOffsets(logical);
            int baseIn = GatherPlan.inputOffset(offsets);
            int baseOut = GatherPlan.outputOffset(offsets);
            int axisIndex = IndexLoopSupport.readAxisIndex(indicesView, indexPlan, logical, plan.axisSize);
            IndexLoopSupport.writeF32(outView, baseOut,
                    IndexLoopSupport.readF32(inputView, baseIn + axisIndex * plan.axisStrideIn));
        }
    }

    static void gatherBF16(
            Tensor input,
            Tensor indices,
            Tensor out,
            CpuStorageView inputView,
            CpuStorageView indicesView,
            CpuStorageView outView,
            int dimension
    ) {
        IndexValidation.validateGather(input, indices, out, dimension);
        IndexLoopSupport.validateReadStorageViews(input, indices, out, inputView, indicesView, outView, DataType.BFLOAT16);
        GatherPlan plan = GatherPlan.create(input, out, dimension);
        IndexLoopSupport.IndexStoragePlan indexPlan = IndexLoopSupport.indexStoragePlan(indicesView);
        if (IndexLoopSupport.allArrays(inputView, indicesView, outView)) {
            gatherBF16Array(inputView.requireBF16Array(), indicesView, outView.requireBF16Array(), indexPlan, plan);
            return;
        }
        for (int logical = 0; logical < plan.total; logical++) {
            long offsets = plan.baseOffsets(logical);
            int baseIn = GatherPlan.inputOffset(offsets);
            int baseOut = GatherPlan.outputOffset(offsets);
            int axisIndex = IndexLoopSupport.readAxisIndex(indicesView, indexPlan, logical, plan.axisSize);
            IndexLoopSupport.writeBF16Bits(outView, baseOut,
                    IndexLoopSupport.readBF16Bits(inputView, baseIn + axisIndex * plan.axisStrideIn));
        }
    }

    static void gatherBOOL(
            Tensor input,
            Tensor indices,
            Tensor out,
            CpuStorageView inputView,
            CpuStorageView indicesView,
            CpuStorageView outView,
            int dimension
    ) {
        IndexValidation.validateGather(input, indices, out, dimension);
        IndexLoopSupport.validateReadStorageViews(input, indices, out, inputView, indicesView, outView, DataType.BOOL);
        GatherPlan plan = GatherPlan.create(input, out, dimension);
        IndexLoopSupport.IndexStoragePlan indexPlan = IndexLoopSupport.indexStoragePlan(indicesView);
        if (IndexLoopSupport.allArrays(inputView, indicesView, outView)) {
            gatherBOOLArray(inputView.requireBoolArray(), indicesView, outView.requireBoolArray(), indexPlan, plan);
            return;
        }
        for (int logical = 0; logical < plan.total; logical++) {
            long offsets = plan.baseOffsets(logical);
            int baseIn = GatherPlan.inputOffset(offsets);
            int baseOut = GatherPlan.outputOffset(offsets);
            int axisIndex = IndexLoopSupport.readAxisIndex(indicesView, indexPlan, logical, plan.axisSize);
            IndexLoopSupport.writeBool(outView, baseOut,
                    IndexLoopSupport.readBool(inputView, baseIn + axisIndex * plan.axisStrideIn));
        }
    }

    static void gatherI32(
            Tensor input,
            Tensor indices,
            Tensor out,
            CpuStorageView inputView,
            CpuStorageView indicesView,
            CpuStorageView outView,
            int dimension
    ) {
        IndexValidation.validateGather(input, indices, out, dimension);
        IndexLoopSupport.validateReadStorageViews(input, indices, out, inputView, indicesView, outView, DataType.INT32);
        GatherPlan plan = GatherPlan.create(input, out, dimension);
        IndexLoopSupport.IndexStoragePlan indexPlan = IndexLoopSupport.indexStoragePlan(indicesView);
        if (IndexLoopSupport.allArrays(inputView, indicesView, outView)) {
            gatherI32Array(inputView.requireI32Array(), indicesView, outView.requireI32Array(), indexPlan, plan);
            return;
        }
        for (int logical = 0; logical < plan.total; logical++) {
            long offsets = plan.baseOffsets(logical);
            int baseIn = GatherPlan.inputOffset(offsets);
            int baseOut = GatherPlan.outputOffset(offsets);
            int axisIndex = IndexLoopSupport.readAxisIndex(indicesView, indexPlan, logical, plan.axisSize);
            IndexLoopSupport.writeI32(outView, baseOut,
                    IndexLoopSupport.readI32(inputView, baseIn + axisIndex * plan.axisStrideIn));
        }
    }

    static void gatherI64(
            Tensor input,
            Tensor indices,
            Tensor out,
            CpuStorageView inputView,
            CpuStorageView indicesView,
            CpuStorageView outView,
            int dimension
    ) {
        IndexValidation.validateGather(input, indices, out, dimension);
        IndexLoopSupport.validateReadStorageViews(input, indices, out, inputView, indicesView, outView, DataType.INT64);
        GatherPlan plan = GatherPlan.create(input, out, dimension);
        IndexLoopSupport.IndexStoragePlan indexPlan = IndexLoopSupport.indexStoragePlan(indicesView);
        if (IndexLoopSupport.allArrays(inputView, indicesView, outView)) {
            gatherI64Array(inputView.requireI64Array(), indicesView, outView.requireI64Array(), indexPlan, plan);
            return;
        }
        for (int logical = 0; logical < plan.total; logical++) {
            long offsets = plan.baseOffsets(logical);
            int baseIn = GatherPlan.inputOffset(offsets);
            int baseOut = GatherPlan.outputOffset(offsets);
            int axisIndex = IndexLoopSupport.readAxisIndex(indicesView, indexPlan, logical, plan.axisSize);
            IndexLoopSupport.writeI64(outView, baseOut,
                    IndexLoopSupport.readI64(inputView, baseIn + axisIndex * plan.axisStrideIn));
        }
    }

    private static void gatherF64Array(
            double[] in,
            CpuStorageView indicesView,
            double[] dst,
            IndexLoopSupport.IndexStoragePlan indexPlan,
            GatherPlan plan
    ) {
        for (int logical = 0; logical < plan.total; logical++) {
            long offsets = plan.baseOffsets(logical);
            int baseIn = GatherPlan.inputOffset(offsets);
            int axisIndex = IndexLoopSupport.readAxisIndex(indicesView, indexPlan, logical, plan.axisSize);
            dst[GatherPlan.outputOffset(offsets)] = in[baseIn + axisIndex * plan.axisStrideIn];
        }
    }

    private static void gatherF32Array(
            float[] in,
            CpuStorageView indicesView,
            float[] dst,
            IndexLoopSupport.IndexStoragePlan indexPlan,
            GatherPlan plan
    ) {
        for (int logical = 0; logical < plan.total; logical++) {
            long offsets = plan.baseOffsets(logical);
            int baseIn = GatherPlan.inputOffset(offsets);
            int axisIndex = IndexLoopSupport.readAxisIndex(indicesView, indexPlan, logical, plan.axisSize);
            dst[GatherPlan.outputOffset(offsets)] = in[baseIn + axisIndex * plan.axisStrideIn];
        }
    }

    private static void gatherBF16Array(
            short[] in,
            CpuStorageView indicesView,
            short[] dst,
            IndexLoopSupport.IndexStoragePlan indexPlan,
            GatherPlan plan
    ) {
        for (int logical = 0; logical < plan.total; logical++) {
            long offsets = plan.baseOffsets(logical);
            int baseIn = GatherPlan.inputOffset(offsets);
            int axisIndex = IndexLoopSupport.readAxisIndex(indicesView, indexPlan, logical, plan.axisSize);
            dst[GatherPlan.outputOffset(offsets)] = in[baseIn + axisIndex * plan.axisStrideIn];
        }
    }

    private static void gatherBOOLArray(
            byte[] in,
            CpuStorageView indicesView,
            byte[] dst,
            IndexLoopSupport.IndexStoragePlan indexPlan,
            GatherPlan plan
    ) {
        for (int logical = 0; logical < plan.total; logical++) {
            long offsets = plan.baseOffsets(logical);
            int baseIn = GatherPlan.inputOffset(offsets);
            int axisIndex = IndexLoopSupport.readAxisIndex(indicesView, indexPlan, logical, plan.axisSize);
            dst[GatherPlan.outputOffset(offsets)] = in[baseIn + axisIndex * plan.axisStrideIn];
        }
    }

    private static void gatherI32Array(
            int[] in,
            CpuStorageView indicesView,
            int[] dst,
            IndexLoopSupport.IndexStoragePlan indexPlan,
            GatherPlan plan
    ) {
        for (int logical = 0; logical < plan.total; logical++) {
            long offsets = plan.baseOffsets(logical);
            int baseIn = GatherPlan.inputOffset(offsets);
            int axisIndex = IndexLoopSupport.readAxisIndex(indicesView, indexPlan, logical, plan.axisSize);
            dst[GatherPlan.outputOffset(offsets)] = in[baseIn + axisIndex * plan.axisStrideIn];
        }
    }

    private static void gatherI64Array(
            long[] in,
            CpuStorageView indicesView,
            long[] dst,
            IndexLoopSupport.IndexStoragePlan indexPlan,
            GatherPlan plan
    ) {
        for (int logical = 0; logical < plan.total; logical++) {
            long offsets = plan.baseOffsets(logical);
            int baseIn = GatherPlan.inputOffset(offsets);
            int axisIndex = IndexLoopSupport.readAxisIndex(indicesView, indexPlan, logical, plan.axisSize);
            dst[GatherPlan.outputOffset(offsets)] = in[baseIn + axisIndex * plan.axisStrideIn];
        }
    }

    private record GatherPlan(
            int[] inputShape,
            int[] inputStrides,
            int[] outStrides,
            int[] reducedDense,
            int inputBaseOffset,
            int outBaseOffset,
            int total,
            int axisSize,
            int axisStrideIn,
            int dimension
    ) {
        static GatherPlan create(Tensor input, Tensor out, int dimension) {
            int[] inputShape = input.getShapeUnsafe();
            return new GatherPlan(
                    inputShape,
                    input.getStridesUnsafe(),
                    out.getStridesUnsafe(),
                    IndexLoopSupport.denseStrides(out.getShapeUnsafe()),
                    input.getStorageOffsetUnsafe(),
                    out.getStorageOffsetUnsafe(),
                    out.getFlatDataSize(),
                    inputShape[dimension],
                    input.getStridesUnsafe()[dimension],
                    dimension);
        }

        long baseOffsets(int logical) {
            int baseIn = inputBaseOffset;
            int baseOut = outBaseOffset;
            int rem = logical;
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
            return ((long) baseIn << 32) | (baseOut & 0xffffffffL);
        }

        static int inputOffset(long offsets) {
            return (int) (offsets >> 32);
        }

        static int outputOffset(long offsets) {
            return (int) offsets;
        }
    }
}
