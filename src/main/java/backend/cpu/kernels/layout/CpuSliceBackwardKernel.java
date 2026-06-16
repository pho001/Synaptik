package backend.cpu.kernels.layout;

import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import backend.cpu.storage.CpuStorageView;
import operations.Operation;
import operations.layout.sliceBackward;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorMetadata;
import tensor.dtype.TensorDTypeOps;

import java.util.List;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

public final class CpuSliceBackwardKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        sliceBackward sliceOp = requireOp(call.operation());
        Tensor updates = requireSingleInput(call.inputTensors());
        if (call.inputs().size() != 1) {
            throw new IllegalArgumentException("sliceBackward expects exactly one input storage view.");
        }
        Tensor node = call.outputTensor();
        CpuStorageView updatesView = call.inputs().getFirst();
        CpuStorageView outView = call.output();
        validateStorageViews(updates, node, updatesView, outView, node.getDataType());
        switch (node.getDataType()) {
            case FLOAT64 -> sliceBackwardF64(sliceOp, updates, node, updatesView, outView);
            case FLOAT32 -> sliceBackwardF32(sliceOp, updates, node, updatesView, outView);
            case BFLOAT16 -> sliceBackwardBF16(sliceOp, updates, node, updatesView, outView);
            case BOOL, INT32, INT64 -> throw new UnsupportedOperationException(
                    "CpuSliceBackwardKernel does not support " + node.getDataType());
        }
        return CpuKernelResult.completed();
    }

    private static sliceBackward requireOp(Operation op) {
        if (!(op instanceof sliceBackward sliceOp)) {
            throw new IllegalArgumentException("CpuSliceBackwardKernel requires sliceBackward operation.");
        }
        return sliceOp;
    }

    private static Tensor requireSingleInput(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException("sliceBackward expects exactly one input.");
        }
        return inputs.getFirst();
    }

    private static void sliceBackwardF64(
            sliceBackward op,
            Tensor updates,
            Tensor node,
            CpuStorageView updatesView,
            CpuStorageView outView
    ) {
        SliceBackwardPlan plan = validateAndCreatePlan(op, updates, node);
        zeroF64(node, outView);
        if (updatesView.isArray() && outView.isArray()) {
            double[] updateData = updatesView.requireF64Array();
            double[] dst = outView.requireF64Array();
            for (int logical = 0; logical < plan.total; logical++) {
                plan.computeOffsets(logical);
                dst[plan.targetOffset] += updateData[plan.updateOffset];
            }
            return;
        }
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical);
            writeF64(outView, plan.targetOffset,
                    readF64(outView, plan.targetOffset) + readF64(updatesView, plan.updateOffset));
        }
    }

    private static void sliceBackwardF32(
            sliceBackward op,
            Tensor updates,
            Tensor node,
            CpuStorageView updatesView,
            CpuStorageView outView
    ) {
        SliceBackwardPlan plan = validateAndCreatePlan(op, updates, node);
        zeroF32(node, outView);
        if (updatesView.isArray() && outView.isArray()) {
            float[] updateData = updatesView.requireF32Array();
            float[] dst = outView.requireF32Array();
            for (int logical = 0; logical < plan.total; logical++) {
                plan.computeOffsets(logical);
                dst[plan.targetOffset] += updateData[plan.updateOffset];
            }
            return;
        }
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical);
            writeF32(outView, plan.targetOffset,
                    readF32(outView, plan.targetOffset) + readF32(updatesView, plan.updateOffset));
        }
    }

    private static void sliceBackwardBF16(
            sliceBackward op,
            Tensor updates,
            Tensor node,
            CpuStorageView updatesView,
            CpuStorageView outView
    ) {
        SliceBackwardPlan plan = validateAndCreatePlan(op, updates, node);
        zeroBF16(node, outView);
        if (updatesView.isArray() && outView.isArray()) {
            short[] updateData = updatesView.requireBF16Array();
            short[] dst = outView.requireBF16Array();
            for (int logical = 0; logical < plan.total; logical++) {
                plan.computeOffsets(logical);
                float acc = TensorDTypeOps.fromBFloat16Bits(dst[plan.targetOffset])
                        + TensorDTypeOps.fromBFloat16Bits(updateData[plan.updateOffset]);
                dst[plan.targetOffset] = TensorDTypeOps.toBFloat16Bits(acc);
            }
            return;
        }
        for (int logical = 0; logical < plan.total; logical++) {
            plan.computeOffsets(logical);
            float acc = TensorDTypeOps.fromBFloat16Bits(readBF16Bits(outView, plan.targetOffset))
                    + TensorDTypeOps.fromBFloat16Bits(readBF16Bits(updatesView, plan.updateOffset));
            writeBF16Bits(outView, plan.targetOffset, TensorDTypeOps.toBFloat16Bits(acc));
        }
    }

    private static SliceBackwardPlan validateAndCreatePlan(sliceBackward op, Tensor updates, Tensor node) {
        int[] inputShape = op.getInputShape();
        validateShape(node.getShapeUnsafe(), inputShape, "sliceBackward output shape must match target input shape.");
        validateFloating(node.getDataType(), "sliceBackward");
        if (updates.getDataType() != node.getDataType()) {
            throw new IllegalArgumentException("sliceBackward requires matching input and output dtypes.");
        }
        return SliceBackwardPlan.create(op, updates, node);
    }

    private static void validateStorageViews(
            Tensor updates,
            Tensor out,
            CpuStorageView updatesView,
            CpuStorageView outView,
            DataType dtype
    ) {
        if (updatesView == null) {
            throw new IllegalArgumentException("updates storage view cannot be null.");
        }
        if (outView == null) {
            throw new IllegalArgumentException("output storage view cannot be null.");
        }
        requireViewMatchesTensor(updates, updatesView, dtype, "updates");
        requireViewMatchesTensor(out, outView, dtype, "output");
    }

    private static void requireViewMatchesTensor(Tensor tensor, CpuStorageView view, DataType expectedDType, String label) {
        if (view.dtype() != expectedDType || view.dtype() != tensor.getDataType()) {
            throw new IllegalStateException(label + " storage dtype mismatch. tensor="
                    + tensor.getDataType() + ", view=" + view.dtype() + ", expected=" + expectedDType);
        }
        if (view.logicalSize() != tensor.getFlatDataSize()) {
            throw new IllegalStateException(label + " storage logical size mismatch. tensor="
                    + tensor.getFlatDataSize() + ", view=" + view.logicalSize());
        }
    }

    private static void validateShape(int[] actual, int[] expected, String message) {
        if (actual.length != expected.length) {
            throw new IllegalArgumentException(message);
        }
        for (int i = 0; i < actual.length; i++) {
            if (actual[i] != expected[i]) {
                throw new IllegalArgumentException(message);
            }
        }
    }

    private static void validateFloating(DataType type, String opName) {
        if (type == DataType.BOOL || type == DataType.INT32 || type == DataType.INT64) {
            throw new IllegalArgumentException(opName + " requires floating output dtype.");
        }
    }

    private static void zeroF64(Tensor out, CpuStorageView outView) {
        int[] shape = out.getShapeUnsafe();
        int[] dense = TensorMetadata.computeStrides(shape);
        int[] strides = out.getStridesUnsafe();
        int baseOffset = out.getStorageOffsetUnsafe();
        int total = out.getFlatDataSize();
        if (outView.isArray()) {
            double[] dst = outView.requireF64Array();
            for (int logical = 0; logical < total; logical++) {
                dst[offsetForLogical(logical, shape, dense, strides, baseOffset)] = 0.0d;
            }
            return;
        }
        for (int logical = 0; logical < total; logical++) {
            writeF64(outView, offsetForLogical(logical, shape, dense, strides, baseOffset), 0.0d);
        }
    }

    private static void zeroF32(Tensor out, CpuStorageView outView) {
        int[] shape = out.getShapeUnsafe();
        int[] dense = TensorMetadata.computeStrides(shape);
        int[] strides = out.getStridesUnsafe();
        int baseOffset = out.getStorageOffsetUnsafe();
        int total = out.getFlatDataSize();
        if (outView.isArray()) {
            float[] dst = outView.requireF32Array();
            for (int logical = 0; logical < total; logical++) {
                dst[offsetForLogical(logical, shape, dense, strides, baseOffset)] = 0.0f;
            }
            return;
        }
        for (int logical = 0; logical < total; logical++) {
            writeF32(outView, offsetForLogical(logical, shape, dense, strides, baseOffset), 0.0f);
        }
    }

    private static void zeroBF16(Tensor out, CpuStorageView outView) {
        int[] shape = out.getShapeUnsafe();
        int[] dense = TensorMetadata.computeStrides(shape);
        int[] strides = out.getStridesUnsafe();
        int baseOffset = out.getStorageOffsetUnsafe();
        int total = out.getFlatDataSize();
        short zero = TensorDTypeOps.toBFloat16Bits(0.0f);
        if (outView.isArray()) {
            short[] dst = outView.requireBF16Array();
            for (int logical = 0; logical < total; logical++) {
                dst[offsetForLogical(logical, shape, dense, strides, baseOffset)] = zero;
            }
            return;
        }
        for (int logical = 0; logical < total; logical++) {
            writeBF16Bits(outView, offsetForLogical(logical, shape, dense, strides, baseOffset), zero);
        }
    }

    private static int offsetForLogical(int logical, int[] shape, int[] dense, int[] strides, int baseOffset) {
        int rem = logical;
        int offset = baseOffset;
        for (int d = 0; d < shape.length; d++) {
            int coord = rem / dense[d];
            rem %= dense[d];
            offset += coord * strides[d];
        }
        return offset;
    }

    private static double readF64(CpuStorageView view, int offset) {
        if (view.isArray()) {
            return view.requireF64Array()[offset];
        }
        return view.requireSegment().get(JAVA_DOUBLE, (long) offset * Double.BYTES);
    }

    private static void writeF64(CpuStorageView view, int offset, double value) {
        if (view.isArray()) {
            view.requireF64Array()[offset] = value;
            return;
        }
        view.requireSegment().set(JAVA_DOUBLE, (long) offset * Double.BYTES, value);
    }

    private static float readF32(CpuStorageView view, int offset) {
        if (view.isArray()) {
            return view.requireF32Array()[offset];
        }
        return view.requireSegment().get(JAVA_FLOAT, (long) offset * Float.BYTES);
    }

    private static void writeF32(CpuStorageView view, int offset, float value) {
        if (view.isArray()) {
            view.requireF32Array()[offset] = value;
            return;
        }
        view.requireSegment().set(JAVA_FLOAT, (long) offset * Float.BYTES, value);
    }

    private static short readBF16Bits(CpuStorageView view, int offset) {
        if (view.isArray()) {
            return view.requireBF16Array()[offset];
        }
        return view.requireSegment().get(JAVA_SHORT, (long) offset * Short.BYTES);
    }

    private static void writeBF16Bits(CpuStorageView view, int offset, short bits) {
        if (view.isArray()) {
            view.requireBF16Array()[offset] = bits;
            return;
        }
        view.requireSegment().set(JAVA_SHORT, (long) offset * Short.BYTES, bits);
    }

    private static final class SliceBackwardPlan {
        private final int[] starts;
        private final int[] axes;
        private final int[] steps;
        private final int[] updateShape;
        private final int[] updateDense;
        private final int[] updateStrides;
        private final int[] targetStrides;
        private final int updateBaseOffset;
        private final int targetBaseOffset;
        private final int total;

        private int updateOffset;
        private int targetOffset;

        private SliceBackwardPlan(
                int[] starts,
                int[] axes,
                int[] steps,
                int[] updateShape,
                int[] updateDense,
                int[] updateStrides,
                int[] targetStrides,
                int updateBaseOffset,
                int targetBaseOffset,
                int total
        ) {
            this.starts = starts;
            this.axes = axes;
            this.steps = steps;
            this.updateShape = updateShape;
            this.updateDense = updateDense;
            this.updateStrides = updateStrides;
            this.targetStrides = targetStrides;
            this.updateBaseOffset = updateBaseOffset;
            this.targetBaseOffset = targetBaseOffset;
            this.total = total;
        }

        static SliceBackwardPlan create(sliceBackward op, Tensor updates, Tensor node) {
            int[] updateShape = updates.getShapeUnsafe();
            return new SliceBackwardPlan(
                    op.getStarts(),
                    op.getAxes(),
                    op.getSteps(),
                    updateShape,
                    TensorMetadata.computeStrides(updateShape),
                    updates.getStridesUnsafe(),
                    node.getStridesUnsafe(),
                    updates.getStorageOffsetUnsafe(),
                    node.getStorageOffsetUnsafe(),
                    updates.getFlatDataSize());
        }

        void computeOffsets(int logical) {
            int rem = logical;
            int update = updateBaseOffset;
            int target = targetBaseOffset;
            for (int d = 0; d < updateShape.length; d++) {
                int coord = rem / updateDense[d];
                rem %= updateDense[d];
                update += coord * updateStrides[d];
                int targetCoord = coord;
                for (int i = 0; i < axes.length; i++) {
                    if (axes[i] == d) {
                        targetCoord = starts[i] + coord * steps[i];
                        break;
                    }
                }
                target += targetCoord * targetStrides[d];
            }
            updateOffset = update;
            targetOffset = target;
        }
    }
}
