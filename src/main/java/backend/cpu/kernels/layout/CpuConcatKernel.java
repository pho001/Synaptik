package backend.cpu.kernels.layout;

import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import backend.cpu.storage.CpuStorageView;
import operations.Operation;
import operations.layout.concat;
import tensor.DataType;

import java.util.List;

public final class CpuConcatKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        if (!(call.operation() instanceof concat concatOp)) {
            throw new IllegalArgumentException("CpuConcatKernel requires concat operation.");
        }
        if (call.inputs().isEmpty()) {
            throw new IllegalArgumentException("concat expects at least one input storage view.");
        }
        CpuStorageView out = call.output();
        LayoutStorageSupport.validateView(out, out.dtype(), "output");
        for (CpuStorageView input : call.inputs()) {
            LayoutStorageSupport.validateView(input, out.dtype(), "input");
        }
        switch (out.dtype()) {
            case FLOAT64 -> concatF64(concatOp, call.inputs(), out);
            case FLOAT32 -> concatF32(concatOp, call.inputs(), out);
            case BFLOAT16 -> concatBF16(concatOp, call.inputs(), out);
            case INT32 -> concatI32(concatOp, call.inputs(), out);
            case INT64 -> concatI64(concatOp, call.inputs(), out);
            case BOOL -> concatBool(concatOp, call.inputs(), out);
        }
        return CpuKernelResult.completed();
    }

    private static void concatF64(concat op, List<CpuStorageView> inputs, CpuStorageView out) {
        ConcatPlan plan = ConcatPlan.create(op, out);
        int axisOffset = 0;
        for (CpuStorageView input : inputs) {
            copyF64(input, out, plan, axisOffset);
            axisOffset += input.shape()[plan.axis];
        }
    }

    private static void concatF32(concat op, List<CpuStorageView> inputs, CpuStorageView out) {
        ConcatPlan plan = ConcatPlan.create(op, out);
        int axisOffset = 0;
        for (CpuStorageView input : inputs) {
            copyF32(input, out, plan, axisOffset);
            axisOffset += input.shape()[plan.axis];
        }
    }

    private static void concatBF16(concat op, List<CpuStorageView> inputs, CpuStorageView out) {
        ConcatPlan plan = ConcatPlan.create(op, out);
        int axisOffset = 0;
        for (CpuStorageView input : inputs) {
            copyBF16(input, out, plan, axisOffset);
            axisOffset += input.shape()[plan.axis];
        }
    }

    private static void concatI32(concat op, List<CpuStorageView> inputs, CpuStorageView out) {
        ConcatPlan plan = ConcatPlan.create(op, out);
        int axisOffset = 0;
        for (CpuStorageView input : inputs) {
            copyI32(input, out, plan, axisOffset);
            axisOffset += input.shape()[plan.axis];
        }
    }

    private static void concatI64(concat op, List<CpuStorageView> inputs, CpuStorageView out) {
        ConcatPlan plan = ConcatPlan.create(op, out);
        int axisOffset = 0;
        for (CpuStorageView input : inputs) {
            copyI64(input, out, plan, axisOffset);
            axisOffset += input.shape()[plan.axis];
        }
    }

    private static void concatBool(concat op, List<CpuStorageView> inputs, CpuStorageView out) {
        ConcatPlan plan = ConcatPlan.create(op, out);
        int axisOffset = 0;
        for (CpuStorageView input : inputs) {
            copyBool(input, out, plan, axisOffset);
            axisOffset += input.shape()[plan.axis];
        }
    }

    private static void copyF64(CpuStorageView input, CpuStorageView out, ConcatPlan plan, int axisOffset) {
        InputPlan inputPlan = InputPlan.create(input);
        validateInput(input, out.dtype(), plan.axis);
        for (int logical = 0; logical < input.logicalSize(); logical++) {
            inputPlan.compute(logical, plan, axisOffset);
            LayoutStorageSupport.writeF64(out, inputPlan.outOffset,
                    LayoutStorageSupport.readF64(input, inputPlan.inputOffset));
        }
    }

    private static void copyF32(CpuStorageView input, CpuStorageView out, ConcatPlan plan, int axisOffset) {
        InputPlan inputPlan = InputPlan.create(input);
        validateInput(input, out.dtype(), plan.axis);
        for (int logical = 0; logical < input.logicalSize(); logical++) {
            inputPlan.compute(logical, plan, axisOffset);
            LayoutStorageSupport.writeF32(out, inputPlan.outOffset,
                    LayoutStorageSupport.readF32(input, inputPlan.inputOffset));
        }
    }

    private static void copyBF16(CpuStorageView input, CpuStorageView out, ConcatPlan plan, int axisOffset) {
        InputPlan inputPlan = InputPlan.create(input);
        validateInput(input, out.dtype(), plan.axis);
        for (int logical = 0; logical < input.logicalSize(); logical++) {
            inputPlan.compute(logical, plan, axisOffset);
            LayoutStorageSupport.writeBF16(out, inputPlan.outOffset,
                    LayoutStorageSupport.readBF16AsF32(input, inputPlan.inputOffset));
        }
    }

    private static void copyI32(CpuStorageView input, CpuStorageView out, ConcatPlan plan, int axisOffset) {
        InputPlan inputPlan = InputPlan.create(input);
        validateInput(input, out.dtype(), plan.axis);
        for (int logical = 0; logical < input.logicalSize(); logical++) {
            inputPlan.compute(logical, plan, axisOffset);
            LayoutStorageSupport.writeI32(out, inputPlan.outOffset,
                    LayoutStorageSupport.readI32(input, inputPlan.inputOffset));
        }
    }

    private static void copyI64(CpuStorageView input, CpuStorageView out, ConcatPlan plan, int axisOffset) {
        InputPlan inputPlan = InputPlan.create(input);
        validateInput(input, out.dtype(), plan.axis);
        for (int logical = 0; logical < input.logicalSize(); logical++) {
            inputPlan.compute(logical, plan, axisOffset);
            LayoutStorageSupport.writeI64(out, inputPlan.outOffset,
                    LayoutStorageSupport.readI64(input, inputPlan.inputOffset));
        }
    }

    private static void copyBool(CpuStorageView input, CpuStorageView out, ConcatPlan plan, int axisOffset) {
        InputPlan inputPlan = InputPlan.create(input);
        validateInput(input, out.dtype(), plan.axis);
        for (int logical = 0; logical < input.logicalSize(); logical++) {
            inputPlan.compute(logical, plan, axisOffset);
            LayoutStorageSupport.writeBool(out, inputPlan.outOffset,
                    LayoutStorageSupport.readBool(input, inputPlan.inputOffset));
        }
    }

    private static void validateInput(CpuStorageView input, DataType dtype, int axis) {
        LayoutStorageSupport.validateView(input, dtype, "input");
        int[] shape = input.shape();
        if (axis < 0 || axis >= shape.length) {
            throw new IllegalArgumentException("concat axis out of bounds: " + axis);
        }
    }

    private static final class ConcatPlan {
        private final int axis;
        private final int[] outStrides;
        private final int outBaseOffset;

        private ConcatPlan(int axis, int[] outStrides, int outBaseOffset) {
            this.axis = axis;
            this.outStrides = outStrides;
            this.outBaseOffset = outBaseOffset;
        }

        static ConcatPlan create(concat op, CpuStorageView out) {
            int axis = op.getAxis();
            int[] outShape = out.shape();
            if (axis < 0 || axis >= outShape.length) {
                throw new IllegalArgumentException("concat axis out of bounds: " + axis);
            }
            return new ConcatPlan(axis, out.strides(), out.storageOffset());
        }
    }

    private static final class InputPlan {
        private final int[] shape;
        private final int[] dense;
        private final int[] strides;
        private final int baseOffset;
        private int inputOffset;
        private int outOffset;

        private InputPlan(int[] shape, int[] dense, int[] strides, int baseOffset) {
            this.shape = shape;
            this.dense = dense;
            this.strides = strides;
            this.baseOffset = baseOffset;
        }

        static InputPlan create(CpuStorageView input) {
            int[] shape = input.shape();
            return new InputPlan(shape, LayoutStorageSupport.denseStrides(shape), input.strides(), input.storageOffset());
        }

        void compute(int logical, ConcatPlan plan, int axisOffset) {
            int rem = logical;
            int in = baseOffset;
            int out = plan.outBaseOffset;
            for (int d = 0; d < shape.length; d++) {
                int coord = rem / dense[d];
                rem %= dense[d];
                in += coord * strides[d];
                out += (d == plan.axis ? coord + axisOffset : coord) * plan.outStrides[d];
            }
            inputOffset = in;
            outOffset = out;
        }
    }
}
