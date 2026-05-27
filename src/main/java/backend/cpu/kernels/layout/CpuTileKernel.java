package backend.cpu.kernels.layout;

import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import backend.cpu.storage.CpuStorageView;
import operations.layout.tile;

public final class CpuTileKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        if (!(call.operation() instanceof tile)) {
            throw new IllegalArgumentException("CpuTileKernel requires tile operation.");
        }
        LayoutStorageSupport.validateInputViews(1, call.inputs(), "tile");
        CpuStorageView input = call.inputs().getFirst();
        CpuStorageView out = call.output();
        LayoutStorageSupport.validateView(input, out.dtype(), "input");
        LayoutStorageSupport.validateView(out, out.dtype(), "output");
        switch (out.dtype()) {
            case FLOAT64 -> tileF64(input, out);
            case FLOAT32 -> tileF32(input, out);
            case BFLOAT16 -> tileBF16(input, out);
            case INT32 -> tileI32(input, out);
            case INT64 -> tileI64(input, out);
            case BOOL -> tileBool(input, out);
        }
        return CpuKernelResult.completed();
    }

    private static void tileF64(CpuStorageView input, CpuStorageView out) {
        TilePlan plan = TilePlan.create(input, out);
        for (int logical = 0; logical < out.logicalSize(); logical++) {
            plan.computeOffsets(logical);
            LayoutStorageSupport.writeF64(out, plan.outOffset,
                    LayoutStorageSupport.readF64(input, plan.inputOffset));
        }
    }

    private static void tileF32(CpuStorageView input, CpuStorageView out) {
        TilePlan plan = TilePlan.create(input, out);
        for (int logical = 0; logical < out.logicalSize(); logical++) {
            plan.computeOffsets(logical);
            LayoutStorageSupport.writeF32(out, plan.outOffset,
                    LayoutStorageSupport.readF32(input, plan.inputOffset));
        }
    }

    private static void tileBF16(CpuStorageView input, CpuStorageView out) {
        TilePlan plan = TilePlan.create(input, out);
        for (int logical = 0; logical < out.logicalSize(); logical++) {
            plan.computeOffsets(logical);
            LayoutStorageSupport.writeBF16(out, plan.outOffset,
                    LayoutStorageSupport.readBF16AsF32(input, plan.inputOffset));
        }
    }

    private static void tileI32(CpuStorageView input, CpuStorageView out) {
        TilePlan plan = TilePlan.create(input, out);
        for (int logical = 0; logical < out.logicalSize(); logical++) {
            plan.computeOffsets(logical);
            LayoutStorageSupport.writeI32(out, plan.outOffset,
                    LayoutStorageSupport.readI32(input, plan.inputOffset));
        }
    }

    private static void tileI64(CpuStorageView input, CpuStorageView out) {
        TilePlan plan = TilePlan.create(input, out);
        for (int logical = 0; logical < out.logicalSize(); logical++) {
            plan.computeOffsets(logical);
            LayoutStorageSupport.writeI64(out, plan.outOffset,
                    LayoutStorageSupport.readI64(input, plan.inputOffset));
        }
    }

    private static void tileBool(CpuStorageView input, CpuStorageView out) {
        TilePlan plan = TilePlan.create(input, out);
        for (int logical = 0; logical < out.logicalSize(); logical++) {
            plan.computeOffsets(logical);
            LayoutStorageSupport.writeBool(out, plan.outOffset,
                    LayoutStorageSupport.readBool(input, plan.inputOffset));
        }
    }

    private static final class TilePlan {
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

        private TilePlan(
                int[] inputShape,
                int[] inputDense,
                int[] inputStrides,
                int inputBaseOffset,
                int[] outShape,
                int[] outDense,
                int[] outStrides,
                int outBaseOffset
        ) {
            this.inputShape = inputShape;
            this.inputDense = inputDense;
            this.inputStrides = inputStrides;
            this.inputBaseOffset = inputBaseOffset;
            this.outShape = outShape;
            this.outDense = outDense;
            this.outStrides = outStrides;
            this.outBaseOffset = outBaseOffset;
        }

        static TilePlan create(CpuStorageView input, CpuStorageView out) {
            int[] inputShape = input.shape();
            int[] outShape = out.shape();
            if (inputShape.length != outShape.length) {
                throw new IllegalArgumentException("tile input and output ranks must match.");
            }
            return new TilePlan(
                    inputShape,
                    LayoutStorageSupport.denseStrides(inputShape),
                    input.strides(),
                    input.storageOffset(),
                    outShape,
                    LayoutStorageSupport.denseStrides(outShape),
                    out.strides(),
                    out.storageOffset());
        }

        void computeOffsets(int logical) {
            int rem = logical;
            int in = inputBaseOffset;
            int out = outBaseOffset;
            for (int d = 0; d < outShape.length; d++) {
                int coord = rem / outDense[d];
                rem %= outDense[d];
                in += (coord % inputShape[d]) * inputStrides[d];
                out += coord * outStrides[d];
            }
            inputOffset = in;
            outOffset = out;
        }
    }
}
