package backend.cpu1.kernels.layout.slice;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.kernels.layout.Cpu1LayoutKernelSupport;
import backend.cpu1.prepare.Cpu1PreparedLayoutUnit;
import backend.runtime.ExecutionContext;

import java.util.Arrays;

public final class Cpu1SliceBackwardLayoutLoops {
    private Cpu1SliceBackwardLayoutLoops() {
    }

    public static void sliceBackwardScalar(Cpu1PreparedLayoutUnit unit, ExecutionContext context) {
        Cpu1LayoutKernelSupport support = new Cpu1LayoutKernelSupport(unit, context);
        Cpu1LayoutKernelSupport.LayoutCall call = support.bindMaterializingCall();
        Cpu1TensorView updates = call.inputs().getFirst();
        Cpu1TensorView output = call.output();
        SliceBackwardPlan plan = SliceBackwardPlan.create(unit, updates, output);

        support.fillOutputScalar(output, 0.0d);
        support.launchRange(updates.elementCount(), (start, end) -> {
            for (int logical = start; logical < end; logical++) {
                plan.computeOffsets(logical);
                double value = support.readElement(output, plan.targetOffset)
                        + support.readElement(updates, plan.updateOffset);
                support.writeElement(output, plan.targetOffset, value);
            }
        });
        support.markOutputWritten(call);
    }

    private static final class SliceBackwardPlan {
        private final int[] starts;
        private final int[] steps;
        private final int[] axisToParameter;
        private final int[] updateShape;
        private final int[] updateDenseStrides;
        private final int[] updateStrides;
        private final int[] outputStrides;
        private final int updateBaseOffset;
        private final int outputBaseOffset;

        private int updateOffset;
        private int targetOffset;

        private SliceBackwardPlan(
                int[] starts,
                int[] steps,
                int[] axisToParameter,
                int[] updateShape,
                int[] updateDenseStrides,
                int[] updateStrides,
                int[] outputStrides,
                int updateBaseOffset,
                int outputBaseOffset
        ) {
            this.starts = starts;
            this.steps = steps;
            this.axisToParameter = axisToParameter;
            this.updateShape = updateShape;
            this.updateDenseStrides = updateDenseStrides;
            this.updateStrides = updateStrides;
            this.outputStrides = outputStrides;
            this.updateBaseOffset = updateBaseOffset;
            this.outputBaseOffset = outputBaseOffset;
        }

        static SliceBackwardPlan create(Cpu1PreparedLayoutUnit unit, Cpu1TensorView updates, Cpu1TensorView output) {
            int[] inputShape = unit.sliceInputShape();
            int[] outputShape = output.shape();
            if (!Arrays.equals(inputShape, outputShape)) {
                throw new IllegalArgumentException("cpu1 SLICE_BACKWARD output shape must match inputShape metadata.");
            }
            if (updates.rank() != output.rank()) {
                throw new IllegalArgumentException("cpu1 SLICE_BACKWARD update/output ranks must match.");
            }
            int[] starts = unit.sliceStarts();
            int[] axes = unit.sliceAxes();
            int[] steps = unit.sliceSteps();
            if (starts.length != axes.length || starts.length != steps.length) {
                throw new IllegalArgumentException("cpu1 SLICE_BACKWARD starts/axes/steps lengths must match.");
            }
            int rank = output.rank();
            int[] axisToParameter = new int[rank];
            Arrays.fill(axisToParameter, -1);
            for (int i = 0; i < axes.length; i++) {
                int axis = axes[i];
                if (axis < 0 || axis >= rank) {
                    throw new IllegalArgumentException("cpu1 SLICE_BACKWARD axis out of bounds: " + axis);
                }
                if (axisToParameter[axis] != -1) {
                    throw new IllegalArgumentException("cpu1 SLICE_BACKWARD duplicate axis: " + axis);
                }
                if (steps[i] <= 0) {
                    throw new IllegalArgumentException("cpu1 SLICE_BACKWARD steps must be positive.");
                }
                int lastTarget = starts[i] + Math.max(0, updates.shape(axis) - 1) * steps[i];
                if (starts[i] < 0 || lastTarget >= output.shape(axis)) {
                    throw new IllegalArgumentException("cpu1 SLICE_BACKWARD slice coordinates exceed output shape.");
                }
                axisToParameter[axis] = i;
            }
            for (int dim = 0; dim < rank; dim++) {
                if (axisToParameter[dim] == -1 && updates.shape(dim) != output.shape(dim)) {
                    throw new IllegalArgumentException("cpu1 SLICE_BACKWARD unsliced dimensions must match output shape.");
                }
            }
            return new SliceBackwardPlan(
                    starts,
                    steps,
                    axisToParameter,
                    updates.shape(),
                    Cpu1LayoutKernelSupport.denseStrides(updates.shape()),
                    updates.strides(),
                    output.strides(),
                    updates.storageOffset(),
                    output.storageOffset()
            );
        }

        void computeOffsets(int logical) {
            int remaining = logical;
            int update = updateBaseOffset;
            int target = outputBaseOffset;
            for (int dim = 0; dim < updateShape.length; dim++) {
                int coordinate = remaining / updateDenseStrides[dim];
                remaining %= updateDenseStrides[dim];
                update += coordinate * updateStrides[dim];
                int parameterIndex = axisToParameter[dim];
                int targetCoordinate = parameterIndex < 0
                        ? coordinate
                        : starts[parameterIndex] + coordinate * steps[parameterIndex];
                target += targetCoordinate * outputStrides[dim];
            }
            this.updateOffset = update;
            this.targetOffset = target;
        }
    }
}
