package backend.cpu.kernels.index;

import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.dtype.TensorDTypeOps;

final class GatherLoops {
    private GatherLoops() {
    }

    static void gatherF64(Tensor input, Tensor indices, Tensor out, int dimension) {
        IndexValidation.validateGather(input, indices, out, dimension);
        double[] in = TensorInternalAccess.float64Data(input);
        double[] dst = TensorInternalAccess.float64Data(out);
        GatherPlan plan = GatherPlan.create(input, out, dimension);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            int baseIn = plan.inputBaseOffset;
            int baseOut = plan.outBaseOffset;
            int rem = logical;
            for (int d = 0, rd = 0; d < plan.inputShape.length; d++) {
                if (d == dimension) {
                    continue;
                }
                int coord = rem / plan.reducedDense[rd];
                rem %= plan.reducedDense[rd];
                baseIn += coord * plan.inputStrides[d];
                baseOut += coord * plan.outStrides[rd];
                rd++;
            }
            int axisIndex = indexReader.readAxisIndex(logical, plan.axisSize);
            dst[baseOut] = in[baseIn + axisIndex * plan.axisStrideIn];
        }
    }

    static void gatherF32(Tensor input, Tensor indices, Tensor out, int dimension) {
        IndexValidation.validateGather(input, indices, out, dimension);
        float[] in = TensorInternalAccess.float32Data(input);
        float[] dst = TensorInternalAccess.float32Data(out);
        GatherPlan plan = GatherPlan.create(input, out, dimension);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            int baseIn = plan.inputBaseOffset;
            int baseOut = plan.outBaseOffset;
            int rem = logical;
            for (int d = 0, rd = 0; d < plan.inputShape.length; d++) {
                if (d == dimension) {
                    continue;
                }
                int coord = rem / plan.reducedDense[rd];
                rem %= plan.reducedDense[rd];
                baseIn += coord * plan.inputStrides[d];
                baseOut += coord * plan.outStrides[rd];
                rd++;
            }
            int axisIndex = indexReader.readAxisIndex(logical, plan.axisSize);
            dst[baseOut] = in[baseIn + axisIndex * plan.axisStrideIn];
        }
    }

    static void gatherBF16(Tensor input, Tensor indices, Tensor out, int dimension) {
        IndexValidation.validateGather(input, indices, out, dimension);
        short[] in = TensorInternalAccess.bfloat16Data(input);
        short[] dst = TensorInternalAccess.bfloat16Data(out);
        GatherPlan plan = GatherPlan.create(input, out, dimension);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            int baseIn = plan.inputBaseOffset;
            int baseOut = plan.outBaseOffset;
            int rem = logical;
            for (int d = 0, rd = 0; d < plan.inputShape.length; d++) {
                if (d == dimension) {
                    continue;
                }
                int coord = rem / plan.reducedDense[rd];
                rem %= plan.reducedDense[rd];
                baseIn += coord * plan.inputStrides[d];
                baseOut += coord * plan.outStrides[rd];
                rd++;
            }
            int axisIndex = indexReader.readAxisIndex(logical, plan.axisSize);
            dst[baseOut] = in[baseIn + axisIndex * plan.axisStrideIn];
        }
    }

    static void gatherBOOL(Tensor input, Tensor indices, Tensor out, int dimension) {
        IndexValidation.validateGather(input, indices, out, dimension);
        byte[] in = TensorInternalAccess.boolData(input);
        byte[] dst = TensorInternalAccess.boolData(out);
        GatherPlan plan = GatherPlan.create(input, out, dimension);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            int baseIn = plan.inputBaseOffset;
            int baseOut = plan.outBaseOffset;
            int rem = logical;
            for (int d = 0, rd = 0; d < plan.inputShape.length; d++) {
                if (d == dimension) {
                    continue;
                }
                int coord = rem / plan.reducedDense[rd];
                rem %= plan.reducedDense[rd];
                baseIn += coord * plan.inputStrides[d];
                baseOut += coord * plan.outStrides[rd];
                rd++;
            }
            int axisIndex = indexReader.readAxisIndex(logical, plan.axisSize);
            dst[baseOut] = in[baseIn + axisIndex * plan.axisStrideIn];
        }
    }

    static void gatherI32(Tensor input, Tensor indices, Tensor out, int dimension) {
        IndexValidation.validateGather(input, indices, out, dimension);
        int[] in = TensorInternalAccess.int32Data(input);
        int[] dst = TensorInternalAccess.int32Data(out);
        GatherPlan plan = GatherPlan.create(input, out, dimension);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            int baseIn = plan.inputBaseOffset;
            int baseOut = plan.outBaseOffset;
            int rem = logical;
            for (int d = 0, rd = 0; d < plan.inputShape.length; d++) {
                if (d == dimension) {
                    continue;
                }
                int coord = rem / plan.reducedDense[rd];
                rem %= plan.reducedDense[rd];
                baseIn += coord * plan.inputStrides[d];
                baseOut += coord * plan.outStrides[rd];
                rd++;
            }
            int axisIndex = indexReader.readAxisIndex(logical, plan.axisSize);
            dst[baseOut] = in[baseIn + axisIndex * plan.axisStrideIn];
        }
    }

    static void gatherI64(Tensor input, Tensor indices, Tensor out, int dimension) {
        IndexValidation.validateGather(input, indices, out, dimension);
        long[] in = TensorInternalAccess.int64Data(input);
        long[] dst = TensorInternalAccess.int64Data(out);
        GatherPlan plan = GatherPlan.create(input, out, dimension);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            int baseIn = plan.inputBaseOffset;
            int baseOut = plan.outBaseOffset;
            int rem = logical;
            for (int d = 0, rd = 0; d < plan.inputShape.length; d++) {
                if (d == dimension) {
                    continue;
                }
                int coord = rem / plan.reducedDense[rd];
                rem %= plan.reducedDense[rd];
                baseIn += coord * plan.inputStrides[d];
                baseOut += coord * plan.outStrides[rd];
                rd++;
            }
            int axisIndex = indexReader.readAxisIndex(logical, plan.axisSize);
            dst[baseOut] = in[baseIn + axisIndex * plan.axisStrideIn];
        }
    }

    static void gatherGradF64(Tensor indices, Tensor outGrad, Tensor node, int dimension) {
        IndexValidation.validateGatherGrad(indices, outGrad, node, dimension);
        double[] grad = TensorInternalAccess.float64Data(outGrad);
        double[] dst = TensorInternalAccess.float64Data(node);
        ScatterPlan plan = ScatterPlan.create(outGrad, node, dimension);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            int baseNode = plan.nodeBaseOffset;
            int baseGrad = plan.gradBaseOffset;
            int rem = logical;
            for (int d = 0, rd = 0; d < plan.nodeShape.length; d++) {
                if (d == dimension) {
                    continue;
                }
                int coord = rem / plan.reducedDense[rd];
                rem %= plan.reducedDense[rd];
                baseNode += coord * plan.nodeStrides[d];
                baseGrad += coord * plan.gradStrides[rd];
                rd++;
            }
            int axisIndex = indexReader.readAxisIndex(logical, plan.axisSize);
            dst[baseNode + axisIndex * plan.axisStrideNode] += grad[baseGrad];
        }
    }

    static void gatherGradF32(Tensor indices, Tensor outGrad, Tensor node, int dimension) {
        IndexValidation.validateGatherGrad(indices, outGrad, node, dimension);
        float[] grad = TensorInternalAccess.float32Data(outGrad);
        float[] dst = TensorInternalAccess.float32Data(node);
        ScatterPlan plan = ScatterPlan.create(outGrad, node, dimension);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            int baseNode = plan.nodeBaseOffset;
            int baseGrad = plan.gradBaseOffset;
            int rem = logical;
            for (int d = 0, rd = 0; d < plan.nodeShape.length; d++) {
                if (d == dimension) {
                    continue;
                }
                int coord = rem / plan.reducedDense[rd];
                rem %= plan.reducedDense[rd];
                baseNode += coord * plan.nodeStrides[d];
                baseGrad += coord * plan.gradStrides[rd];
                rd++;
            }
            int axisIndex = indexReader.readAxisIndex(logical, plan.axisSize);
            dst[baseNode + axisIndex * plan.axisStrideNode] += grad[baseGrad];
        }
    }

    static void gatherGradBF16(Tensor indices, Tensor outGrad, Tensor node, int dimension) {
        IndexValidation.validateGatherGrad(indices, outGrad, node, dimension);
        short[] grad = TensorInternalAccess.bfloat16Data(outGrad);
        short[] dst = TensorInternalAccess.bfloat16Data(node);
        ScatterPlan plan = ScatterPlan.create(outGrad, node, dimension);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            int baseNode = plan.nodeBaseOffset;
            int baseGrad = plan.gradBaseOffset;
            int rem = logical;
            for (int d = 0, rd = 0; d < plan.nodeShape.length; d++) {
                if (d == dimension) {
                    continue;
                }
                int coord = rem / plan.reducedDense[rd];
                rem %= plan.reducedDense[rd];
                baseNode += coord * plan.nodeStrides[d];
                baseGrad += coord * plan.gradStrides[rd];
                rd++;
            }
            int axisIndex = indexReader.readAxisIndex(logical, plan.axisSize);
            int target = baseNode + axisIndex * plan.axisStrideNode;
            float acc = TensorDTypeOps.fromBFloat16Bits(dst[target]) + TensorDTypeOps.fromBFloat16Bits(grad[baseGrad]);
            dst[target] = TensorDTypeOps.toBFloat16Bits(acc);
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
            int axisStrideIn
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
                    input.getStridesUnsafe()[dimension]);
        }
    }

    private record ScatterPlan(
            int[] nodeShape,
            int[] nodeStrides,
            int[] gradStrides,
            int[] reducedDense,
            int nodeBaseOffset,
            int gradBaseOffset,
            int total,
            int axisSize,
            int axisStrideNode
    ) {
        static ScatterPlan create(Tensor outGrad, Tensor node, int dimension) {
            int[] nodeShape = node.getShapeUnsafe();
            return new ScatterPlan(
                    nodeShape,
                    node.getStridesUnsafe(),
                    outGrad.getStridesUnsafe(),
                    IndexLoopSupport.denseStrides(outGrad.getShapeUnsafe()),
                    node.getStorageOffsetUnsafe(),
                    outGrad.getStorageOffsetUnsafe(),
                    outGrad.getFlatDataSize(),
                    nodeShape[dimension],
                    node.getStridesUnsafe()[dimension]);
        }
    }
}
