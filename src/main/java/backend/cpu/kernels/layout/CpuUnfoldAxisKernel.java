package backend.cpu.kernels.layout;

import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import backend.cpu.storage.CpuStorageView;
import operations.layout.unfoldAxis;

public final class CpuUnfoldAxisKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        if (!(call.operation() instanceof unfoldAxis unfoldOp)) {
            throw new IllegalArgumentException("CpuUnfoldAxisKernel requires unfoldAxis operation.");
        }
        LayoutStorageSupport.validateInputViews(1, call.inputs(), "unfold");
        CpuStorageView input = call.inputs().getFirst();
        CpuStorageView out = call.output();
        LayoutStorageSupport.validateView(input, out.dtype(), "input");
        LayoutStorageSupport.validateView(out, out.dtype(), "output");
        switch (out.dtype()) {
            case FLOAT64 -> unfoldF64(unfoldOp, input, out);
            case FLOAT32 -> unfoldF32(unfoldOp, input, out);
            case BFLOAT16 -> unfoldBF16(unfoldOp, input, out);
            case INT32 -> unfoldI32(unfoldOp, input, out);
            case INT64 -> unfoldI64(unfoldOp, input, out);
            case BOOL -> unfoldBool(unfoldOp, input, out);
        }
        return CpuKernelResult.completed();
    }

    private static void unfoldF64(unfoldAxis op, CpuStorageView input, CpuStorageView out) {
        UnfoldPlan plan = UnfoldPlan.create(op, input, out);
        for (int logical = 0; logical < out.logicalSize(); logical++) {
            plan.computeOffsets(logical);
            LayoutStorageSupport.writeF64(out, plan.outOffset,
                    LayoutStorageSupport.readF64(input, plan.inputOffset));
        }
    }

    private static void unfoldF32(unfoldAxis op, CpuStorageView input, CpuStorageView out) {
        UnfoldPlan plan = UnfoldPlan.create(op, input, out);
        for (int logical = 0; logical < out.logicalSize(); logical++) {
            plan.computeOffsets(logical);
            LayoutStorageSupport.writeF32(out, plan.outOffset,
                    LayoutStorageSupport.readF32(input, plan.inputOffset));
        }
    }

    private static void unfoldBF16(unfoldAxis op, CpuStorageView input, CpuStorageView out) {
        UnfoldPlan plan = UnfoldPlan.create(op, input, out);
        for (int logical = 0; logical < out.logicalSize(); logical++) {
            plan.computeOffsets(logical);
            LayoutStorageSupport.writeBF16(out, plan.outOffset,
                    LayoutStorageSupport.readBF16AsF32(input, plan.inputOffset));
        }
    }

    private static void unfoldI32(unfoldAxis op, CpuStorageView input, CpuStorageView out) {
        UnfoldPlan plan = UnfoldPlan.create(op, input, out);
        for (int logical = 0; logical < out.logicalSize(); logical++) {
            plan.computeOffsets(logical);
            LayoutStorageSupport.writeI32(out, plan.outOffset,
                    LayoutStorageSupport.readI32(input, plan.inputOffset));
        }
    }

    private static void unfoldI64(unfoldAxis op, CpuStorageView input, CpuStorageView out) {
        UnfoldPlan plan = UnfoldPlan.create(op, input, out);
        for (int logical = 0; logical < out.logicalSize(); logical++) {
            plan.computeOffsets(logical);
            LayoutStorageSupport.writeI64(out, plan.outOffset,
                    LayoutStorageSupport.readI64(input, plan.inputOffset));
        }
    }

    private static void unfoldBool(unfoldAxis op, CpuStorageView input, CpuStorageView out) {
        UnfoldPlan plan = UnfoldPlan.create(op, input, out);
        for (int logical = 0; logical < out.logicalSize(); logical++) {
            plan.computeOffsets(logical);
            LayoutStorageSupport.writeBool(out, plan.outOffset,
                    LayoutStorageSupport.readBool(input, plan.inputOffset));
        }
    }

    private static final class UnfoldPlan {
        private final int axis;
        private final int size;
        private final int step;
        private final int rank;
        private final int[] prefixShape;
        private final int[] prefixDense;
        private final int[] inputStrides;
        private final int inputBaseOffset;
        private final int[] outShape;
        private final int[] outDense;
        private final int[] outStrides;
        private final int outBaseOffset;
        private int inputOffset;
        private int outOffset;

        private UnfoldPlan(
                int axis,
                int size,
                int step,
                int rank,
                int[] prefixShape,
                int[] prefixDense,
                int[] inputStrides,
                int inputBaseOffset,
                int[] outShape,
                int[] outDense,
                int[] outStrides,
                int outBaseOffset
        ) {
            this.axis = axis;
            this.size = size;
            this.step = step;
            this.rank = rank;
            this.prefixShape = prefixShape;
            this.prefixDense = prefixDense;
            this.inputStrides = inputStrides;
            this.inputBaseOffset = inputBaseOffset;
            this.outShape = outShape;
            this.outDense = outDense;
            this.outStrides = outStrides;
            this.outBaseOffset = outBaseOffset;
        }

        static UnfoldPlan create(unfoldAxis op, CpuStorageView input, CpuStorageView out) {
            int[] inputShape = input.shape();
            int[] outShape = out.shape();
            int rank = inputShape.length;
            int axis = op.getAxis();
            int size = op.getSize();
            if (axis < 0 || axis >= rank || outShape.length != rank + 1) {
                throw new IllegalArgumentException("unfold axis shape mismatch.");
            }
            int[] prefixShape = inputShape.clone();
            prefixShape[axis] = outShape[axis];
            return new UnfoldPlan(
                    axis,
                    size,
                    op.getStep(),
                    rank,
                    prefixShape,
                    LayoutStorageSupport.denseStrides(prefixShape),
                    input.strides(),
                    input.storageOffset(),
                    outShape,
                    LayoutStorageSupport.denseStrides(outShape),
                    out.strides(),
                    out.storageOffset());
        }

        void computeOffsets(int logical) {
            int windowOffset = logical % size;
            int prefixLogical = logical / size;
            int rem = prefixLogical;
            int in = inputBaseOffset;
            for (int d = 0; d < rank; d++) {
                int coord = rem / prefixDense[d];
                rem %= prefixDense[d];
                int inputCoord = d == axis ? coord * step + windowOffset : coord;
                in += inputCoord * inputStrides[d];
            }
            inputOffset = in;
            outOffset = LayoutStorageSupport.offsetForLogical(logical, outShape, outDense, outStrides, outBaseOffset);
        }
    }
}
