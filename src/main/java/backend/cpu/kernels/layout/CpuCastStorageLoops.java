package backend.cpu.kernels.layout;

import backend.cpu.storage.CpuStorageView;
import tensor.dtype.TensorDTypeOps;

final class CpuCastStorageLoops {
    private CpuCastStorageLoops() {
    }

    static void cast(CpuStorageView input, CpuStorageView out) {
        if (input.logicalSize() != out.logicalSize()) {
            throw new IllegalArgumentException("cast requires input and output storage views with the same logical size.");
        }
        CastPlan plan = CastPlan.create(input, out);
        if (input.isArray() && out.isArray()) {
            castArrays(input, out, plan);
            return;
        }
        castGeneric(input, out, plan);
    }

    private static void castArrays(CpuStorageView input, CpuStorageView out, CastPlan plan) {
        switch (out.dtype()) {
            case FLOAT64 -> castArrayToF64(input, out.requireF64Array(), plan);
            case FLOAT32 -> castArrayToF32(input, out.requireF32Array(), plan);
            case BFLOAT16 -> castArrayToBF16(input, out.requireBF16Array(), plan);
            case INT32 -> castArrayToI32(input, out.requireI32Array(), plan);
            case INT64 -> castArrayToI64(input, out.requireI64Array(), plan);
            case BOOL -> castArrayToBool(input, out.requireBoolArray(), plan);
        }
    }

    private static void castArrayToF64(CpuStorageView input, double[] out, CastPlan plan) {
        switch (input.dtype()) {
            case FLOAT64 -> {
                double[] in = input.requireF64Array();
                for (int i = 0; i < plan.logicalSize; i++) {
                    out[plan.outputOffset(i)] = in[plan.inputOffset(i)];
                }
            }
            case FLOAT32 -> {
                float[] in = input.requireF32Array();
                for (int i = 0; i < plan.logicalSize; i++) {
                    out[plan.outputOffset(i)] = in[plan.inputOffset(i)];
                }
            }
            case BFLOAT16 -> {
                short[] in = input.requireBF16Array();
                for (int i = 0; i < plan.logicalSize; i++) {
                    out[plan.outputOffset(i)] = TensorDTypeOps.fromBFloat16Bits(in[plan.inputOffset(i)]);
                }
            }
            case INT32 -> {
                int[] in = input.requireI32Array();
                for (int i = 0; i < plan.logicalSize; i++) {
                    out[plan.outputOffset(i)] = in[plan.inputOffset(i)];
                }
            }
            case INT64 -> {
                long[] in = input.requireI64Array();
                for (int i = 0; i < plan.logicalSize; i++) {
                    out[plan.outputOffset(i)] = (double) in[plan.inputOffset(i)];
                }
            }
            case BOOL -> {
                byte[] in = input.requireBoolArray();
                for (int i = 0; i < plan.logicalSize; i++) {
                    out[plan.outputOffset(i)] = in[plan.inputOffset(i)] == 0 ? 0.0d : 1.0d;
                }
            }
        }
    }

    private static void castArrayToF32(CpuStorageView input, float[] out, CastPlan plan) {
        switch (input.dtype()) {
            case FLOAT64 -> {
                double[] in = input.requireF64Array();
                for (int i = 0; i < plan.logicalSize; i++) {
                    out[plan.outputOffset(i)] = (float) in[plan.inputOffset(i)];
                }
            }
            case FLOAT32 -> {
                float[] in = input.requireF32Array();
                for (int i = 0; i < plan.logicalSize; i++) {
                    out[plan.outputOffset(i)] = (float) ((double) in[plan.inputOffset(i)]);
                }
            }
            case BFLOAT16 -> {
                short[] in = input.requireBF16Array();
                for (int i = 0; i < plan.logicalSize; i++) {
                    out[plan.outputOffset(i)] = TensorDTypeOps.fromBFloat16Bits(in[plan.inputOffset(i)]);
                }
            }
            case INT32 -> {
                int[] in = input.requireI32Array();
                for (int i = 0; i < plan.logicalSize; i++) {
                    out[plan.outputOffset(i)] = (float) ((double) in[plan.inputOffset(i)]);
                }
            }
            case INT64 -> {
                long[] in = input.requireI64Array();
                for (int i = 0; i < plan.logicalSize; i++) {
                    out[plan.outputOffset(i)] = (float) ((double) in[plan.inputOffset(i)]);
                }
            }
            case BOOL -> {
                byte[] in = input.requireBoolArray();
                for (int i = 0; i < plan.logicalSize; i++) {
                    out[plan.outputOffset(i)] = in[plan.inputOffset(i)] == 0 ? 0.0f : 1.0f;
                }
            }
        }
    }

    private static void castArrayToBF16(CpuStorageView input, short[] out, CastPlan plan) {
        switch (input.dtype()) {
            case FLOAT64 -> {
                double[] in = input.requireF64Array();
                for (int i = 0; i < plan.logicalSize; i++) {
                    out[plan.outputOffset(i)] = TensorDTypeOps.toBFloat16Bits((float) in[plan.inputOffset(i)]);
                }
            }
            case FLOAT32 -> {
                float[] in = input.requireF32Array();
                for (int i = 0; i < plan.logicalSize; i++) {
                    out[plan.outputOffset(i)] = TensorDTypeOps.toBFloat16Bits((float) ((double) in[plan.inputOffset(i)]));
                }
            }
            case BFLOAT16 -> {
                short[] in = input.requireBF16Array();
                for (int i = 0; i < plan.logicalSize; i++) {
                    float value = TensorDTypeOps.fromBFloat16Bits(in[plan.inputOffset(i)]);
                    out[plan.outputOffset(i)] = TensorDTypeOps.toBFloat16Bits(value);
                }
            }
            case INT32 -> {
                int[] in = input.requireI32Array();
                for (int i = 0; i < plan.logicalSize; i++) {
                    out[plan.outputOffset(i)] = TensorDTypeOps.toBFloat16Bits((float) ((double) in[plan.inputOffset(i)]));
                }
            }
            case INT64 -> {
                long[] in = input.requireI64Array();
                for (int i = 0; i < plan.logicalSize; i++) {
                    out[plan.outputOffset(i)] = TensorDTypeOps.toBFloat16Bits((float) ((double) in[plan.inputOffset(i)]));
                }
            }
            case BOOL -> {
                byte[] in = input.requireBoolArray();
                for (int i = 0; i < plan.logicalSize; i++) {
                    float value = in[plan.inputOffset(i)] == 0 ? 0.0f : 1.0f;
                    out[plan.outputOffset(i)] = TensorDTypeOps.toBFloat16Bits(value);
                }
            }
        }
    }

    private static void castArrayToI32(CpuStorageView input, int[] out, CastPlan plan) {
        switch (input.dtype()) {
            case FLOAT64 -> {
                double[] in = input.requireF64Array();
                for (int i = 0; i < plan.logicalSize; i++) {
                    out[plan.outputOffset(i)] = (int) in[plan.inputOffset(i)];
                }
            }
            case FLOAT32 -> {
                float[] in = input.requireF32Array();
                for (int i = 0; i < plan.logicalSize; i++) {
                    out[plan.outputOffset(i)] = (int) ((double) in[plan.inputOffset(i)]);
                }
            }
            case BFLOAT16 -> {
                short[] in = input.requireBF16Array();
                for (int i = 0; i < plan.logicalSize; i++) {
                    out[plan.outputOffset(i)] = (int) ((double) TensorDTypeOps.fromBFloat16Bits(in[plan.inputOffset(i)]));
                }
            }
            case INT32 -> {
                int[] in = input.requireI32Array();
                for (int i = 0; i < plan.logicalSize; i++) {
                    out[plan.outputOffset(i)] = (int) ((double) in[plan.inputOffset(i)]);
                }
            }
            case INT64 -> {
                long[] in = input.requireI64Array();
                for (int i = 0; i < plan.logicalSize; i++) {
                    out[plan.outputOffset(i)] = (int) ((double) in[plan.inputOffset(i)]);
                }
            }
            case BOOL -> {
                byte[] in = input.requireBoolArray();
                for (int i = 0; i < plan.logicalSize; i++) {
                    out[plan.outputOffset(i)] = in[plan.inputOffset(i)] == 0 ? 0 : 1;
                }
            }
        }
    }

    private static void castArrayToI64(CpuStorageView input, long[] out, CastPlan plan) {
        switch (input.dtype()) {
            case FLOAT64 -> {
                double[] in = input.requireF64Array();
                for (int i = 0; i < plan.logicalSize; i++) {
                    out[plan.outputOffset(i)] = (long) in[plan.inputOffset(i)];
                }
            }
            case FLOAT32 -> {
                float[] in = input.requireF32Array();
                for (int i = 0; i < plan.logicalSize; i++) {
                    out[plan.outputOffset(i)] = (long) ((double) in[plan.inputOffset(i)]);
                }
            }
            case BFLOAT16 -> {
                short[] in = input.requireBF16Array();
                for (int i = 0; i < plan.logicalSize; i++) {
                    out[plan.outputOffset(i)] = (long) ((double) TensorDTypeOps.fromBFloat16Bits(in[plan.inputOffset(i)]));
                }
            }
            case INT32 -> {
                int[] in = input.requireI32Array();
                for (int i = 0; i < plan.logicalSize; i++) {
                    out[plan.outputOffset(i)] = (long) ((double) in[plan.inputOffset(i)]);
                }
            }
            case INT64 -> {
                long[] in = input.requireI64Array();
                for (int i = 0; i < plan.logicalSize; i++) {
                    out[plan.outputOffset(i)] = (long) ((double) in[plan.inputOffset(i)]);
                }
            }
            case BOOL -> {
                byte[] in = input.requireBoolArray();
                for (int i = 0; i < plan.logicalSize; i++) {
                    out[plan.outputOffset(i)] = in[plan.inputOffset(i)] == 0 ? 0L : 1L;
                }
            }
        }
    }

    private static void castArrayToBool(CpuStorageView input, byte[] out, CastPlan plan) {
        switch (input.dtype()) {
            case FLOAT64 -> {
                double[] in = input.requireF64Array();
                for (int i = 0; i < plan.logicalSize; i++) {
                    out[plan.outputOffset(i)] = boolFromDouble(in[plan.inputOffset(i)]);
                }
            }
            case FLOAT32 -> {
                float[] in = input.requireF32Array();
                for (int i = 0; i < plan.logicalSize; i++) {
                    out[plan.outputOffset(i)] = boolFromDouble((double) in[plan.inputOffset(i)]);
                }
            }
            case BFLOAT16 -> {
                short[] in = input.requireBF16Array();
                for (int i = 0; i < plan.logicalSize; i++) {
                    out[plan.outputOffset(i)] = boolFromDouble(
                            (double) TensorDTypeOps.fromBFloat16Bits(in[plan.inputOffset(i)])
                    );
                }
            }
            case INT32 -> {
                int[] in = input.requireI32Array();
                for (int i = 0; i < plan.logicalSize; i++) {
                    out[plan.outputOffset(i)] = boolFromDouble((double) in[plan.inputOffset(i)]);
                }
            }
            case INT64 -> {
                long[] in = input.requireI64Array();
                for (int i = 0; i < plan.logicalSize; i++) {
                    out[plan.outputOffset(i)] = boolFromDouble((double) in[plan.inputOffset(i)]);
                }
            }
            case BOOL -> {
                byte[] in = input.requireBoolArray();
                for (int i = 0; i < plan.logicalSize; i++) {
                    out[plan.outputOffset(i)] = in[plan.inputOffset(i)] == 0 ? (byte) 0 : (byte) 1;
                }
            }
        }
    }

    private static void castGeneric(CpuStorageView input, CpuStorageView out, CastPlan plan) {
        switch (out.dtype()) {
            case FLOAT64 -> {
                for (int i = 0; i < plan.logicalSize; i++) {
                    LayoutStorageSupport.writeF64(out, plan.outputOffset(i), readAsDouble(input, plan.inputOffset(i)));
                }
            }
            case FLOAT32 -> {
                for (int i = 0; i < plan.logicalSize; i++) {
                    LayoutStorageSupport.writeF32(out, plan.outputOffset(i), (float) readAsDouble(input, plan.inputOffset(i)));
                }
            }
            case BFLOAT16 -> {
                for (int i = 0; i < plan.logicalSize; i++) {
                    LayoutStorageSupport.writeBF16(out, plan.outputOffset(i), (float) readAsDouble(input, plan.inputOffset(i)));
                }
            }
            case INT32 -> {
                for (int i = 0; i < plan.logicalSize; i++) {
                    LayoutStorageSupport.writeI32(out, plan.outputOffset(i), (int) readAsDouble(input, plan.inputOffset(i)));
                }
            }
            case INT64 -> {
                for (int i = 0; i < plan.logicalSize; i++) {
                    LayoutStorageSupport.writeI64(out, plan.outputOffset(i), (long) readAsDouble(input, plan.inputOffset(i)));
                }
            }
            case BOOL -> {
                for (int i = 0; i < plan.logicalSize; i++) {
                    LayoutStorageSupport.writeBool(out, plan.outputOffset(i),
                            boolFromDouble(readAsDouble(input, plan.inputOffset(i))));
                }
            }
        }
    }

    private static double readAsDouble(CpuStorageView view, int offset) {
        return switch (view.dtype()) {
            case FLOAT64 -> LayoutStorageSupport.readF64(view, offset);
            case FLOAT32 -> LayoutStorageSupport.readF32(view, offset);
            case BFLOAT16 -> LayoutStorageSupport.readBF16AsF32(view, offset);
            case INT32 -> LayoutStorageSupport.readI32(view, offset);
            case INT64 -> (double) LayoutStorageSupport.readI64(view, offset);
            case BOOL -> LayoutStorageSupport.readBool(view, offset) == 0 ? 0.0d : 1.0d;
        };
    }

    private static byte boolFromDouble(double value) {
        return value == 0.0d ? (byte) 0 : (byte) 1;
    }

    private static final class CastPlan {
        private final int logicalSize;
        private final int[] inputShape;
        private final int[] inputDense;
        private final int[] inputStrides;
        private final int inputBaseOffset;
        private final int[] outputShape;
        private final int[] outputDense;
        private final int[] outputStrides;
        private final int outputBaseOffset;

        private CastPlan(
                int logicalSize,
                int[] inputShape,
                int[] inputDense,
                int[] inputStrides,
                int inputBaseOffset,
                int[] outputShape,
                int[] outputDense,
                int[] outputStrides,
                int outputBaseOffset
        ) {
            this.logicalSize = logicalSize;
            this.inputShape = inputShape;
            this.inputDense = inputDense;
            this.inputStrides = inputStrides;
            this.inputBaseOffset = inputBaseOffset;
            this.outputShape = outputShape;
            this.outputDense = outputDense;
            this.outputStrides = outputStrides;
            this.outputBaseOffset = outputBaseOffset;
        }

        static CastPlan create(CpuStorageView input, CpuStorageView out) {
            int[] inputShape = input.shape();
            int[] outputShape = out.shape();
            return new CastPlan(
                    out.logicalSize(),
                    inputShape,
                    LayoutStorageSupport.denseStrides(inputShape),
                    input.strides(),
                    input.storageOffset(),
                    outputShape,
                    LayoutStorageSupport.denseStrides(outputShape),
                    out.strides(),
                    out.storageOffset()
            );
        }

        int inputOffset(int logical) {
            return LayoutStorageSupport.offsetForLogical(
                    logical,
                    inputShape,
                    inputDense,
                    inputStrides,
                    inputBaseOffset
            );
        }

        int outputOffset(int logical) {
            return LayoutStorageSupport.offsetForLogical(
                    logical,
                    outputShape,
                    outputDense,
                    outputStrides,
                    outputBaseOffset
            );
        }
    }
}
