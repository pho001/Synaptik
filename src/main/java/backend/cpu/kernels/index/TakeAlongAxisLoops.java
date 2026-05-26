package backend.cpu.kernels.index;

import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.dtype.TensorDTypeOps;

final class TakeAlongAxisLoops {
    private TakeAlongAxisLoops() {
    }

    static void takeAlongAxisF64(Tensor input, Tensor indices, Tensor out, int dimension) {
        IndexValidation.validateTakeAlongAxis(input, indices, out, dimension);
        double[] in = TensorInternalAccess.float64Data(input);
        double[] dst = TensorInternalAccess.float64Data(out);
        TakeAlongAxisPlan plan = TakeAlongAxisPlan.create(input, out, dimension);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indexReader, dimension);
            dst[plan.valueOffset] = in[plan.baseOffset + plan.axisIndex * plan.axisStride];
        }
    }

    static void takeAlongAxisF32(Tensor input, Tensor indices, Tensor out, int dimension) {
        IndexValidation.validateTakeAlongAxis(input, indices, out, dimension);
        float[] in = TensorInternalAccess.float32Data(input);
        float[] dst = TensorInternalAccess.float32Data(out);
        TakeAlongAxisPlan plan = TakeAlongAxisPlan.create(input, out, dimension);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indexReader, dimension);
            dst[plan.valueOffset] = in[plan.baseOffset + plan.axisIndex * plan.axisStride];
        }
    }

    static void takeAlongAxisBF16(Tensor input, Tensor indices, Tensor out, int dimension) {
        IndexValidation.validateTakeAlongAxis(input, indices, out, dimension);
        short[] in = TensorInternalAccess.bfloat16Data(input);
        short[] dst = TensorInternalAccess.bfloat16Data(out);
        TakeAlongAxisPlan plan = TakeAlongAxisPlan.create(input, out, dimension);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indexReader, dimension);
            dst[plan.valueOffset] = in[plan.baseOffset + plan.axisIndex * plan.axisStride];
        }
    }

    static void takeAlongAxisBOOL(Tensor input, Tensor indices, Tensor out, int dimension) {
        IndexValidation.validateTakeAlongAxis(input, indices, out, dimension);
        byte[] in = TensorInternalAccess.boolData(input);
        byte[] dst = TensorInternalAccess.boolData(out);
        TakeAlongAxisPlan plan = TakeAlongAxisPlan.create(input, out, dimension);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indexReader, dimension);
            dst[plan.valueOffset] = in[plan.baseOffset + plan.axisIndex * plan.axisStride];
        }
    }

    static void takeAlongAxisI32(Tensor input, Tensor indices, Tensor out, int dimension) {
        IndexValidation.validateTakeAlongAxis(input, indices, out, dimension);
        int[] in = TensorInternalAccess.int32Data(input);
        int[] dst = TensorInternalAccess.int32Data(out);
        TakeAlongAxisPlan plan = TakeAlongAxisPlan.create(input, out, dimension);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indexReader, dimension);
            dst[plan.valueOffset] = in[plan.baseOffset + plan.axisIndex * plan.axisStride];
        }
    }

    static void takeAlongAxisI64(Tensor input, Tensor indices, Tensor out, int dimension) {
        IndexValidation.validateTakeAlongAxis(input, indices, out, dimension);
        long[] in = TensorInternalAccess.int64Data(input);
        long[] dst = TensorInternalAccess.int64Data(out);
        TakeAlongAxisPlan plan = TakeAlongAxisPlan.create(input, out, dimension);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indexReader, dimension);
            dst[plan.valueOffset] = in[plan.baseOffset + plan.axisIndex * plan.axisStride];
        }
    }

    static void takeAlongAxisGradF64(Tensor indices, Tensor outGrad, Tensor node, int dimension) {
        IndexValidation.validateTakeAlongAxisGrad(indices, outGrad, node, dimension);
        double[] grad = TensorInternalAccess.float64Data(outGrad);
        double[] dst = TensorInternalAccess.float64Data(node);
        TakeAlongAxisPlan plan = TakeAlongAxisPlan.create(node, outGrad, dimension);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indexReader, dimension);
            dst[plan.baseOffset + plan.axisIndex * plan.axisStride] += grad[plan.valueOffset];
        }
    }

    static void takeAlongAxisGradF32(Tensor indices, Tensor outGrad, Tensor node, int dimension) {
        IndexValidation.validateTakeAlongAxisGrad(indices, outGrad, node, dimension);
        float[] grad = TensorInternalAccess.float32Data(outGrad);
        float[] dst = TensorInternalAccess.float32Data(node);
        TakeAlongAxisPlan plan = TakeAlongAxisPlan.create(node, outGrad, dimension);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indexReader, dimension);
            dst[plan.baseOffset + plan.axisIndex * plan.axisStride] += grad[plan.valueOffset];
        }
    }

    static void takeAlongAxisGradBF16(Tensor indices, Tensor outGrad, Tensor node, int dimension) {
        IndexValidation.validateTakeAlongAxisGrad(indices, outGrad, node, dimension);
        short[] grad = TensorInternalAccess.bfloat16Data(outGrad);
        short[] dst = TensorInternalAccess.bfloat16Data(node);
        TakeAlongAxisPlan plan = TakeAlongAxisPlan.create(node, outGrad, dimension);
        IndexLoopSupport.IndexReader indexReader = IndexLoopSupport.indexReader(indices);
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indexReader, dimension);
            int target = plan.baseOffset + plan.axisIndex * plan.axisStride;
            float acc = TensorDTypeOps.fromBFloat16Bits(dst[target]) + TensorDTypeOps.fromBFloat16Bits(grad[plan.valueOffset]);
            dst[target] = TensorDTypeOps.toBFloat16Bits(acc);
        }
    }

    private static final class TakeAlongAxisPlan {
        private final int[] inputShape;
        private final int[] inputStrides;
        private final int[] valueShape;
        private final int[] valueStrides;
        private final int[] valueDense;
        private final int inputBaseOffset;
        private final int valueBaseOffset;
        private final int total;
        private final int axisSize;
        private final int axisStride;

        private int baseOffset;
        private int valueOffset;
        private int axisIndex;

        private TakeAlongAxisPlan(
                int[] inputShape,
                int[] inputStrides,
                int[] valueShape,
                int[] valueStrides,
                int[] valueDense,
                int inputBaseOffset,
                int valueBaseOffset,
                int total,
                int axisSize,
                int axisStride
        ) {
            this.inputShape = inputShape;
            this.inputStrides = inputStrides;
            this.valueShape = valueShape;
            this.valueStrides = valueStrides;
            this.valueDense = valueDense;
            this.inputBaseOffset = inputBaseOffset;
            this.valueBaseOffset = valueBaseOffset;
            this.total = total;
            this.axisSize = axisSize;
            this.axisStride = axisStride;
        }

        static TakeAlongAxisPlan create(Tensor input, Tensor valueTensor, int dimension) {
            int[] inputShape = input.getShapeUnsafe();
            int[] valueShape = valueTensor.getShapeUnsafe();
            return new TakeAlongAxisPlan(
                    inputShape,
                    input.getStridesUnsafe(),
                    valueShape,
                    valueTensor.getStridesUnsafe(),
                    IndexLoopSupport.denseStrides(valueShape),
                    input.getStorageOffsetUnsafe(),
                    valueTensor.getStorageOffsetUnsafe(),
                    valueTensor.getFlatDataSize(),
                    inputShape[dimension],
                    input.getStridesUnsafe()[dimension]);
        }

        void computeOffsets(int logical, IndexLoopSupport.IndexReader indexReader, int dimension) {
            int rem = logical;
            int base = inputBaseOffset;
            int value = valueBaseOffset;
            for (int d = 0; d < valueShape.length; d++) {
                int coord = rem / valueDense[d];
                rem %= valueDense[d];
                value += coord * valueStrides[d];
                if (d != dimension) {
                    base += coord * inputStrides[d];
                }
            }
            baseOffset = base;
            valueOffset = value;
            axisIndex = indexReader.readAxisIndexAllowNegative(logical, axisSize);
        }
    }
}
