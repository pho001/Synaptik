package backend.cpu.kernels.layout;

import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import backend.cpu.storage.CpuStorageView;
import operations.Operation;
import operations.layout.pad;

public final class CpuPadKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        if (!(call.operation() instanceof pad padOp)) {
            throw new IllegalArgumentException("CpuPadKernel requires pad operation.");
        }
        LayoutStorageSupport.validateInputViews(1, call.inputs(), "pad");
        CpuStorageView input = call.inputs().getFirst();
        CpuStorageView out = call.output();
        LayoutStorageSupport.validateView(input, out.dtype(), "input");
        LayoutStorageSupport.validateView(out, out.dtype(), "output");
        switch (out.dtype()) {
            case FLOAT64 -> padF64(padOp, input, out);
            case FLOAT32 -> padF32(padOp, input, out);
            case BFLOAT16 -> padBF16(padOp, input, out);
            case INT32 -> padI32(padOp, input, out);
            case INT64 -> padI64(padOp, input, out);
            case BOOL -> padBool(padOp, input, out);
        }
        return CpuKernelResult.completed();
    }

    private static void padF64(pad op, CpuStorageView input, CpuStorageView out) {
        PadPlan plan = PadPlan.create(op, input, out);
        fillF64(out, plan, op.getConstantValue());
        for (int logical = 0; logical < input.logicalSize(); logical++) {
            plan.computeCopyOffsets(logical);
            LayoutStorageSupport.writeF64(out, plan.outOffset,
                    LayoutStorageSupport.readF64(input, plan.inputOffset));
        }
    }

    private static void padF32(pad op, CpuStorageView input, CpuStorageView out) {
        PadPlan plan = PadPlan.create(op, input, out);
        fillF32(out, plan, (float) op.getConstantValue());
        for (int logical = 0; logical < input.logicalSize(); logical++) {
            plan.computeCopyOffsets(logical);
            LayoutStorageSupport.writeF32(out, plan.outOffset,
                    LayoutStorageSupport.readF32(input, plan.inputOffset));
        }
    }

    private static void padBF16(pad op, CpuStorageView input, CpuStorageView out) {
        PadPlan plan = PadPlan.create(op, input, out);
        fillBF16(out, plan, (float) op.getConstantValue());
        for (int logical = 0; logical < input.logicalSize(); logical++) {
            plan.computeCopyOffsets(logical);
            LayoutStorageSupport.writeBF16(out, plan.outOffset,
                    LayoutStorageSupport.readBF16AsF32(input, plan.inputOffset));
        }
    }

    private static void padI32(pad op, CpuStorageView input, CpuStorageView out) {
        PadPlan plan = PadPlan.create(op, input, out);
        fillI32(out, plan, (int) op.getConstantValue());
        for (int logical = 0; logical < input.logicalSize(); logical++) {
            plan.computeCopyOffsets(logical);
            LayoutStorageSupport.writeI32(out, plan.outOffset,
                    LayoutStorageSupport.readI32(input, plan.inputOffset));
        }
    }

    private static void padI64(pad op, CpuStorageView input, CpuStorageView out) {
        PadPlan plan = PadPlan.create(op, input, out);
        fillI64(out, plan, (long) op.getConstantValue());
        for (int logical = 0; logical < input.logicalSize(); logical++) {
            plan.computeCopyOffsets(logical);
            LayoutStorageSupport.writeI64(out, plan.outOffset,
                    LayoutStorageSupport.readI64(input, plan.inputOffset));
        }
    }

    private static void padBool(pad op, CpuStorageView input, CpuStorageView out) {
        PadPlan plan = PadPlan.create(op, input, out);
        fillBool(out, plan, LayoutStorageSupport.boolFromDouble(op.getConstantValue()));
        for (int logical = 0; logical < input.logicalSize(); logical++) {
            plan.computeCopyOffsets(logical);
            LayoutStorageSupport.writeBool(out, plan.outOffset,
                    LayoutStorageSupport.readBool(input, plan.inputOffset));
        }
    }

    private static void fillF64(CpuStorageView out, PadPlan plan, double value) {
        for (int logical = 0; logical < out.logicalSize(); logical++) {
            LayoutStorageSupport.writeF64(out, plan.outOffsetForLogical(logical), value);
        }
    }

    private static void fillF32(CpuStorageView out, PadPlan plan, float value) {
        for (int logical = 0; logical < out.logicalSize(); logical++) {
            LayoutStorageSupport.writeF32(out, plan.outOffsetForLogical(logical), value);
        }
    }

    private static void fillBF16(CpuStorageView out, PadPlan plan, float value) {
        for (int logical = 0; logical < out.logicalSize(); logical++) {
            LayoutStorageSupport.writeBF16(out, plan.outOffsetForLogical(logical), value);
        }
    }

    private static void fillI32(CpuStorageView out, PadPlan plan, int value) {
        for (int logical = 0; logical < out.logicalSize(); logical++) {
            LayoutStorageSupport.writeI32(out, plan.outOffsetForLogical(logical), value);
        }
    }

    private static void fillI64(CpuStorageView out, PadPlan plan, long value) {
        for (int logical = 0; logical < out.logicalSize(); logical++) {
            LayoutStorageSupport.writeI64(out, plan.outOffsetForLogical(logical), value);
        }
    }

    private static void fillBool(CpuStorageView out, PadPlan plan, byte value) {
        for (int logical = 0; logical < out.logicalSize(); logical++) {
            LayoutStorageSupport.writeBool(out, plan.outOffsetForLogical(logical), value);
        }
    }

    private static final class PadPlan {
        private final int[] before;
        private final int[] inputShape;
        private final int[] inputDense;
        private final int[] inputStrides;
        private final int inputBaseOffset;
        private final int[] outShape;
        private final int[] outDense;
        private final int[] outStrides;
        private final int outBaseOffset;
        private int inputOffset;
        private int outOffset;

        private PadPlan(
                int[] before,
                int[] inputShape,
                int[] inputDense,
                int[] inputStrides,
                int inputBaseOffset,
                int[] outShape,
                int[] outDense,
                int[] outStrides,
                int outBaseOffset
        ) {
            this.before = before;
            this.inputShape = inputShape;
            this.inputDense = inputDense;
            this.inputStrides = inputStrides;
            this.inputBaseOffset = inputBaseOffset;
            this.outShape = outShape;
            this.outDense = outDense;
            this.outStrides = outStrides;
            this.outBaseOffset = outBaseOffset;
        }

        static PadPlan create(pad op, CpuStorageView input, CpuStorageView out) {
            int[] inputShape = input.shape();
            int[] outShape = out.shape();
            int[] before = op.getBefore();
            if (before.length != inputShape.length || outShape.length != inputShape.length) {
                throw new IllegalArgumentException("pad rank mismatch.");
            }
            return new PadPlan(
                    before,
                    inputShape,
                    LayoutStorageSupport.denseStrides(inputShape),
                    input.strides(),
                    input.storageOffset(),
                    outShape,
                    LayoutStorageSupport.denseStrides(outShape),
                    out.strides(),
                    out.storageOffset());
        }

        void computeCopyOffsets(int logical) {
            int rem = logical;
            int in = inputBaseOffset;
            int out = outBaseOffset;
            for (int d = 0; d < inputShape.length; d++) {
                int coord = rem / inputDense[d];
                rem %= inputDense[d];
                in += coord * inputStrides[d];
                out += (coord + before[d]) * outStrides[d];
            }
            inputOffset = in;
            outOffset = out;
        }

        int outOffsetForLogical(int logical) {
            return LayoutStorageSupport.offsetForLogical(logical, outShape, outDense, outStrides, outBaseOffset);
        }
    }
}
