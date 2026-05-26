package backend.cpu.kernels.index;

import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.dtype.TensorDTypeOps;

final class ScatterLoops {
    private ScatterLoops() {
    }

    static void scatterAddF64(Tensor base, Tensor indices, Tensor src, Tensor out, int dimension) {
        IndexValidation.validateScatterAdd(base, indices, src, out, dimension);
        out.copyDataFrom(base);
        double[] srcData = TensorInternalAccess.float64Data(src);
        double[] dst = TensorInternalAccess.float64Data(out);
        ScatterAddPlan plan = ScatterAddPlan.create(src, out, dimension);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            int baseOut = plan.outBaseOffset;
            int srcOffset = plan.srcBaseOffset;
            int rem = logical;
            for (int d = 0, rd = 0; d < plan.outShape.length; d++) {
                if (d == dimension) {
                    continue;
                }
                int coord = rem / plan.reducedDense[rd];
                rem %= plan.reducedDense[rd];
                baseOut += coord * plan.outStrides[d];
                srcOffset += coord * plan.srcStrides[rd];
                rd++;
            }
            int axisIndex = indexReader.readAxisIndex(logical, plan.axisSize);
            dst[baseOut + axisIndex * plan.axisStrideOut] += srcData[srcOffset];
        }
    }

    static void scatterAddF32(Tensor base, Tensor indices, Tensor src, Tensor out, int dimension) {
        IndexValidation.validateScatterAdd(base, indices, src, out, dimension);
        out.copyDataFrom(base);
        float[] srcData = TensorInternalAccess.float32Data(src);
        float[] dst = TensorInternalAccess.float32Data(out);
        ScatterAddPlan plan = ScatterAddPlan.create(src, out, dimension);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            int baseOut = plan.outBaseOffset;
            int srcOffset = plan.srcBaseOffset;
            int rem = logical;
            for (int d = 0, rd = 0; d < plan.outShape.length; d++) {
                if (d == dimension) {
                    continue;
                }
                int coord = rem / plan.reducedDense[rd];
                rem %= plan.reducedDense[rd];
                baseOut += coord * plan.outStrides[d];
                srcOffset += coord * plan.srcStrides[rd];
                rd++;
            }
            int axisIndex = indexReader.readAxisIndex(logical, plan.axisSize);
            dst[baseOut + axisIndex * plan.axisStrideOut] += srcData[srcOffset];
        }
    }

    static void scatterAddBF16(Tensor base, Tensor indices, Tensor src, Tensor out, int dimension) {
        IndexValidation.validateScatterAdd(base, indices, src, out, dimension);
        out.copyDataFrom(base);
        short[] srcData = TensorInternalAccess.bfloat16Data(src);
        short[] dst = TensorInternalAccess.bfloat16Data(out);
        ScatterAddPlan plan = ScatterAddPlan.create(src, out, dimension);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            int baseOut = plan.outBaseOffset;
            int srcOffset = plan.srcBaseOffset;
            int rem = logical;
            for (int d = 0, rd = 0; d < plan.outShape.length; d++) {
                if (d == dimension) {
                    continue;
                }
                int coord = rem / plan.reducedDense[rd];
                rem %= plan.reducedDense[rd];
                baseOut += coord * plan.outStrides[d];
                srcOffset += coord * plan.srcStrides[rd];
                rd++;
            }
            int axisIndex = indexReader.readAxisIndex(logical, plan.axisSize);
            int target = baseOut + axisIndex * plan.axisStrideOut;
            float acc = TensorDTypeOps.fromBFloat16Bits(dst[target]) + TensorDTypeOps.fromBFloat16Bits(srcData[srcOffset]);
            dst[target] = TensorDTypeOps.toBFloat16Bits(acc);
        }
    }

    private record ScatterAddPlan(
            int[] outShape,
            int[] outStrides,
            int[] srcStrides,
            int[] reducedDense,
            int outBaseOffset,
            int srcBaseOffset,
            int total,
            int axisSize,
            int axisStrideOut
    ) {
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
                    out.getStridesUnsafe()[dimension]);
        }
    }
}
