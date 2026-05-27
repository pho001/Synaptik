package backend.cpu.kernels.index;

import backend.cpu.storage.CpuStorageView;
import tensor.DataType;
import tensor.Tensor;

final class TakeAlongAxisLoops {
    private TakeAlongAxisLoops() {
    }

    static void takeAlongAxisF64(
            Tensor input,
            Tensor indices,
            Tensor out,
            CpuStorageView inputView,
            CpuStorageView indicesView,
            CpuStorageView outView,
            int dimension
    ) {
        IndexValidation.validateTakeAlongAxis(input, indices, out, dimension);
        IndexLoopSupport.validateReadStorageViews(input, indices, out, inputView, indicesView, outView, DataType.FLOAT64);
        TakeAlongAxisPlan plan = TakeAlongAxisPlan.create(input, out, dimension);
        IndexLoopSupport.IndexStoragePlan indexPlan = IndexLoopSupport.indexStoragePlan(indicesView);
        if (IndexLoopSupport.allArrays(inputView, indicesView, outView)) {
            double[] in = inputView.requireF64Array();
            double[] dst = outView.requireF64Array();
            for (int logical = 0; logical < plan.total; logical++) {
                plan.computeOffsets(logical, indicesView, indexPlan, dimension);
                dst[plan.valueOffset] = in[plan.baseOffset + plan.axisIndex * plan.axisStride];
            }
            return;
        }
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indicesView, indexPlan, dimension);
            IndexLoopSupport.writeF64(outView, plan.valueOffset,
                    IndexLoopSupport.readF64(inputView, plan.baseOffset + plan.axisIndex * plan.axisStride));
        }
    }

    static void takeAlongAxisF32(
            Tensor input,
            Tensor indices,
            Tensor out,
            CpuStorageView inputView,
            CpuStorageView indicesView,
            CpuStorageView outView,
            int dimension
    ) {
        IndexValidation.validateTakeAlongAxis(input, indices, out, dimension);
        IndexLoopSupport.validateReadStorageViews(input, indices, out, inputView, indicesView, outView, DataType.FLOAT32);
        TakeAlongAxisPlan plan = TakeAlongAxisPlan.create(input, out, dimension);
        IndexLoopSupport.IndexStoragePlan indexPlan = IndexLoopSupport.indexStoragePlan(indicesView);
        if (IndexLoopSupport.allArrays(inputView, indicesView, outView)) {
            float[] in = inputView.requireF32Array();
            float[] dst = outView.requireF32Array();
            for (int logical = 0; logical < plan.total; logical++) {
                plan.computeOffsets(logical, indicesView, indexPlan, dimension);
                dst[plan.valueOffset] = in[plan.baseOffset + plan.axisIndex * plan.axisStride];
            }
            return;
        }
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indicesView, indexPlan, dimension);
            IndexLoopSupport.writeF32(outView, plan.valueOffset,
                    IndexLoopSupport.readF32(inputView, plan.baseOffset + plan.axisIndex * plan.axisStride));
        }
    }

    static void takeAlongAxisBF16(
            Tensor input,
            Tensor indices,
            Tensor out,
            CpuStorageView inputView,
            CpuStorageView indicesView,
            CpuStorageView outView,
            int dimension
    ) {
        IndexValidation.validateTakeAlongAxis(input, indices, out, dimension);
        IndexLoopSupport.validateReadStorageViews(input, indices, out, inputView, indicesView, outView, DataType.BFLOAT16);
        TakeAlongAxisPlan plan = TakeAlongAxisPlan.create(input, out, dimension);
        IndexLoopSupport.IndexStoragePlan indexPlan = IndexLoopSupport.indexStoragePlan(indicesView);
        if (IndexLoopSupport.allArrays(inputView, indicesView, outView)) {
            short[] in = inputView.requireBF16Array();
            short[] dst = outView.requireBF16Array();
            for (int logical = 0; logical < plan.total; logical++) {
                plan.computeOffsets(logical, indicesView, indexPlan, dimension);
                dst[plan.valueOffset] = in[plan.baseOffset + plan.axisIndex * plan.axisStride];
            }
            return;
        }
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indicesView, indexPlan, dimension);
            IndexLoopSupport.writeBF16Bits(outView, plan.valueOffset,
                    IndexLoopSupport.readBF16Bits(inputView, plan.baseOffset + plan.axisIndex * plan.axisStride));
        }
    }

    static void takeAlongAxisBOOL(
            Tensor input,
            Tensor indices,
            Tensor out,
            CpuStorageView inputView,
            CpuStorageView indicesView,
            CpuStorageView outView,
            int dimension
    ) {
        IndexValidation.validateTakeAlongAxis(input, indices, out, dimension);
        IndexLoopSupport.validateReadStorageViews(input, indices, out, inputView, indicesView, outView, DataType.BOOL);
        TakeAlongAxisPlan plan = TakeAlongAxisPlan.create(input, out, dimension);
        IndexLoopSupport.IndexStoragePlan indexPlan = IndexLoopSupport.indexStoragePlan(indicesView);
        if (IndexLoopSupport.allArrays(inputView, indicesView, outView)) {
            byte[] in = inputView.requireBoolArray();
            byte[] dst = outView.requireBoolArray();
            for (int logical = 0; logical < plan.total; logical++) {
                plan.computeOffsets(logical, indicesView, indexPlan, dimension);
                dst[plan.valueOffset] = in[plan.baseOffset + plan.axisIndex * plan.axisStride];
            }
            return;
        }
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indicesView, indexPlan, dimension);
            IndexLoopSupport.writeBool(outView, plan.valueOffset,
                    IndexLoopSupport.readBool(inputView, plan.baseOffset + plan.axisIndex * plan.axisStride));
        }
    }

    static void takeAlongAxisI32(
            Tensor input,
            Tensor indices,
            Tensor out,
            CpuStorageView inputView,
            CpuStorageView indicesView,
            CpuStorageView outView,
            int dimension
    ) {
        IndexValidation.validateTakeAlongAxis(input, indices, out, dimension);
        IndexLoopSupport.validateReadStorageViews(input, indices, out, inputView, indicesView, outView, DataType.INT32);
        TakeAlongAxisPlan plan = TakeAlongAxisPlan.create(input, out, dimension);
        IndexLoopSupport.IndexStoragePlan indexPlan = IndexLoopSupport.indexStoragePlan(indicesView);
        if (IndexLoopSupport.allArrays(inputView, indicesView, outView)) {
            int[] in = inputView.requireI32Array();
            int[] dst = outView.requireI32Array();
            for (int logical = 0; logical < plan.total; logical++) {
                plan.computeOffsets(logical, indicesView, indexPlan, dimension);
                dst[plan.valueOffset] = in[plan.baseOffset + plan.axisIndex * plan.axisStride];
            }
            return;
        }
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indicesView, indexPlan, dimension);
            IndexLoopSupport.writeI32(outView, plan.valueOffset,
                    IndexLoopSupport.readI32(inputView, plan.baseOffset + plan.axisIndex * plan.axisStride));
        }
    }

    static void takeAlongAxisI64(
            Tensor input,
            Tensor indices,
            Tensor out,
            CpuStorageView inputView,
            CpuStorageView indicesView,
            CpuStorageView outView,
            int dimension
    ) {
        IndexValidation.validateTakeAlongAxis(input, indices, out, dimension);
        IndexLoopSupport.validateReadStorageViews(input, indices, out, inputView, indicesView, outView, DataType.INT64);
        TakeAlongAxisPlan plan = TakeAlongAxisPlan.create(input, out, dimension);
        IndexLoopSupport.IndexStoragePlan indexPlan = IndexLoopSupport.indexStoragePlan(indicesView);
        if (IndexLoopSupport.allArrays(inputView, indicesView, outView)) {
            long[] in = inputView.requireI64Array();
            long[] dst = outView.requireI64Array();
            for (int logical = 0; logical < plan.total; logical++) {
                plan.computeOffsets(logical, indicesView, indexPlan, dimension);
                dst[plan.valueOffset] = in[plan.baseOffset + plan.axisIndex * plan.axisStride];
            }
            return;
        }
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical, indicesView, indexPlan, dimension);
            IndexLoopSupport.writeI64(outView, plan.valueOffset,
                    IndexLoopSupport.readI64(inputView, plan.baseOffset + plan.axisIndex * plan.axisStride));
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

        void computeOffsets(
                int logical,
                CpuStorageView indices,
                IndexLoopSupport.IndexStoragePlan indexPlan,
                int dimension
        ) {
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
            axisIndex = IndexLoopSupport.readAxisIndexAllowNegative(indices, indexPlan, logical, axisSize);
        }
    }
}
