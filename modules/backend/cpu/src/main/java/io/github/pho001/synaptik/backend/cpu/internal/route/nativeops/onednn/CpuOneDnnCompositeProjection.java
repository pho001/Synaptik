package io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormInferenceIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuLossIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPool2dIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPool3dIr;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnKernelSpec.Binary;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnKernelSpec.BinaryAlgorithm;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnKernelSpec.Constant;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnKernelSpec.Pipeline;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnKernelSpec.Pooling;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnKernelSpec.Reduction;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnKernelSpec.ReductionAlgorithm;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnKernelSpec.Step;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnKernelSpec.Tensor;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnKernelSpec.Unary;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnKernelSpec.UnaryAlgorithm;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.loss.LossKind;
import io.github.pho001.synaptik.model.operation.loss.LossReduction;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Cold numeric native recipes for existing common MSE, BatchNorm inference and average-pool units.
 * Each recipe preserves one common unit and its sole output; it neither recognizes a replacement
 * semantic graph nor qualifies an implementation. External inputs remain borrowed unique common
 * boundaries, including when several semantic operand roles refer to the same boundary. Only
 * immutable scalar attribute words are retained as constants; no operand data is retained.
 */
public final class CpuOneDnnCompositeProjection {
    private CpuOneDnnCompositeProjection() {}

    /**
     * Projects the literal F32 arithmetic and exact geometry of supported static common units.
     * MSE mean retains SUM followed by DIV, and BatchNorm retains all six formula sites. Average
     * pooling is representable only with a power-of-two fixed divisor and actual conceptual positive
     * zero padding in every output window. Unsupported families, empty/symbolic domains and native
     * descriptor geometry return empty before native selection; qualification remains separate.
     *
     * @param unit non-null already validated common CPU execution unit; never mutated or retained
     * @return immutable numeric recipe, or empty when the complete unit is not representable
     * @throws NullPointerException if {@code unit} is null
     * @throws IllegalArgumentException if proven common family and boundary facts disagree
     * @throws ArithmeticException if checked descriptor, origin or geometry arithmetic overflows
     */
    public static Optional<CpuOneDnnProjection> from(
            CpuPartitionPreparationPlan.ExecutionUnitPlan unit) {
        Objects.requireNonNull(unit, "unit");
        var portable = unit.portablePlan();
        if (portable == null || unit.outputCount() != 1 || unit.elementCount() == 0
                || portable.specialization().ordinaryBatch()
                || portable.specialization().rowMapReductionIr().isPresent()
                || portable.specialization().materializedSourcePosition() >= 0
                || !unit.carrierPattern().equals(unit.generatedCarrierPattern())
                || portable.specialization().boundaryDataTypes().stream()
                    .anyMatch(type -> type != DataType.FLOAT32)) {
            return Optional.empty();
        }
        if (portable.portableKernelIr() instanceof CpuLossIr loss) return mse(unit, loss);
        if (portable.portableKernelIr() instanceof CpuBatchNormInferenceIr normalization) {
            return batchNorm(unit, normalization);
        }
        if (portable.portableKernelIr() instanceof CpuPool2dIr pool
                && unit.pool2dGeometry().isPresent()) {
            var geometry = unit.pool2dGeometry().orElseThrow();
            if (pool.kind() != geometry.kind() || pool.dataType() != geometry.dataType()) {
                throw new IllegalArgumentException("Pool2d native projection facts disagree");
            }
            if (pool.kind() != CpuPool2dIr.Kind.AVERAGE || pool.dataType() != DataType.FLOAT32) {
                return Optional.empty();
            }
            return pooling(unit, geometry.input().extents(), geometry.input().strides(),
                    geometry.input().offset(), geometry.output().extents(),
                    geometry.output().strides(), geometry.output().offset(),
                    List.of(geometry.kernelHeight(), geometry.kernelWidth()),
                    List.of(geometry.strideHeight(), geometry.strideWidth()),
                    List.of(geometry.dilationHeight(), geometry.dilationWidth()),
                    List.of(geometry.paddingHeight(), geometry.paddingWidth()), geometry.divisor());
        }
        if (portable.portableKernelIr() instanceof CpuPool3dIr pool
                && unit.pool3dGeometry().isPresent()) {
            var geometry = unit.pool3dGeometry().orElseThrow();
            if (pool.kind() != geometry.kind() || pool.dataType() != geometry.dataType()) {
                throw new IllegalArgumentException("Pool3d native projection facts disagree");
            }
            if (pool.kind() != CpuPool3dIr.Kind.AVERAGE || pool.dataType() != DataType.FLOAT32) {
                return Optional.empty();
            }
            return pooling(unit, geometry.input().extents(), geometry.input().strides(),
                    geometry.input().offset(), geometry.output().extents(),
                    geometry.output().strides(), geometry.output().offset(),
                    List.of(geometry.kernelDepth(), geometry.kernelHeight(), geometry.kernelWidth()),
                    List.of(geometry.strideDepth(), geometry.strideHeight(), geometry.strideWidth()),
                    List.of(geometry.dilationDepth(), geometry.dilationHeight(), geometry.dilationWidth()),
                    List.of(geometry.paddingDepth(), geometry.paddingHeight(), geometry.paddingWidth()),
                    geometry.divisor());
        }
        return Optional.empty();
    }

    private static Optional<CpuOneDnnProjection> mse(
            CpuPartitionPreparationPlan.ExecutionUnitPlan unit, CpuLossIr ir) {
        if (ir.kind() != LossKind.MEAN_SQUARED_ERROR || ir.predictionType() != DataType.FLOAT32
                || ir.targetType() != DataType.FLOAT32 || ir.resultType() != DataType.FLOAT32) {
            return Optional.empty();
        }
        var geometry = ir.geometry();
        long[] extents = geometry.extents();
        if (!representable(extents, geometry.predictionStrides())
                || !representable(extents, geometry.targetStrides())) return Optional.empty();
        if (geometry.axis() != -1 || geometry.targetRank() != extents.length
                || geometry.outputRank() != (ir.reduction() == LossReduction.NONE ? extents.length : 0)) {
            throw new IllegalArgumentException("MSE native projection geometry disagrees");
        }
        int unique = ir.boundaryTypes().size() - 1;
        requireBoundaries(unit, unique);
        var inputs = new ArrayList<Tensor>();
        var offsets = new ArrayList<Long>();
        Tensor prediction = tensor(extents, geometry.predictionStrides());
        Tensor target = tensor(extents, geometry.targetStrides());
        for (int boundary = 0; boundary < unique; boundary++) {
            if (ir.roleBoundaryPositions().getFirst() == boundary) {
                inputs.add(prediction);
                offsets.add(bytes(geometry.predictionOffset()));
            } else if (ir.roleBoundaryPositions().getLast() == boundary) {
                inputs.add(target);
                offsets.add(bytes(geometry.targetOffset()));
            } else {
                throw new IllegalArgumentException("MSE native projection has an unused input");
            }
        }
        int predictionSlot = ir.roleBoundaryPositions().getFirst();
        int targetSlot = ir.roleBoundaryPositions().getLast();
        if (predictionSlot == targetSlot && (!prediction.equals(target)
                || geometry.predictionOffset() != geometry.targetOffset())) {
            throw new IllegalArgumentException("repeated MSE native operand geometry disagrees");
        }
        Tensor dense = dense(extents);
        Tensor output = ir.reduction() == LossReduction.NONE
                ? tensor(extents, geometry.outputStrides()) : scalar();
        int deltaSlot = unique + 1;
        var intermediates = new ArrayList<Tensor>();
        intermediates.add(dense);
        var steps = new ArrayList<Step>();
        steps.add(binary(BinaryAlgorithm.SUB, broadcast(prediction), broadcast(target), dense,
                predictionSlot, targetSlot, deltaSlot));
        var constants = new ArrayList<Constant>();
        if (ir.reduction() == LossReduction.NONE) {
            steps.add(binary(BinaryAlgorithm.MUL, dense, dense, output, deltaSlot, deltaSlot, unique));
        } else {
            int squareSlot = unique + 2;
            intermediates.add(dense);
            steps.add(binary(BinaryAlgorithm.MUL, dense, dense, dense,
                    deltaSlot, deltaSlot, squareSlot));
            Tensor reduced = singleton(dense.dimensions().size());
            if (ir.reduction() == LossReduction.SUM) {
                steps.add(new Step(new Reduction(ReductionAlgorithm.SUM, dense, reduced),
                        List.of(squareSlot), unique));
            } else {
                int sumSlot = unique + 3;
                intermediates.add(scalar());
                steps.add(new Step(new Reduction(ReductionAlgorithm.SUM, dense, reduced),
                        List.of(squareSlot), sumSlot));
                long count = 1;
                for (long extent : extents) count = Math.multiplyExact(count, extent);
                constants.add(new Constant(scalar(), List.of(Float.floatToRawIntBits((float) count))));
                int countSlot = unique + 1 + intermediates.size();
                steps.add(binary(BinaryAlgorithm.DIV, scalar(), scalar(), output,
                        sumSlot, countSlot, unique));
            }
        }
        offsets.add(bytes(geometry.outputOffset()));
        return projection(unit, new Pipeline(inputs, output, intermediates, constants, steps), offsets);
    }

    private static Optional<CpuOneDnnProjection> batchNorm(
            CpuPartitionPreparationPlan.ExecutionUnitPlan unit, CpuBatchNormInferenceIr ir) {
        if (ir.resultType() != DataType.FLOAT32
                || ir.inputTypes().stream().anyMatch(type -> type != DataType.FLOAT32)) {
            return Optional.empty();
        }
        var geometry = unit.runtimeFacts().batchNormInferenceGeometry().orElseThrow(() ->
                new IllegalArgumentException("BatchNorm native projection is missing common geometry"));
        if (!ir.inputTypes().equals(geometry.inputTypes()) || ir.resultType() != geometry.resultType()
                || ir.epsilonBits() != geometry.epsilonBits() || ir.channelAxis() != geometry.channelAxis()
                || ir.inputRank() != geometry.output().extents().length
                || !ir.positionToBoundary().equals(geometry.positionToBoundary())) {
            throw new IllegalArgumentException("BatchNorm native projection facts disagree");
        }
        int unique = geometry.inputs().size();
        requireBoundaries(unit, unique);
        var inputs = new ArrayList<Tensor>();
        var offsets = new ArrayList<Long>();
        for (var layout : geometry.inputs()) {
            if (!representable(layout.extents(), layout.strides())) return Optional.empty();
            inputs.add(tensor(layout.extents(), layout.strides()));
            offsets.add(bytes(layout.offset()));
        }
        var layout = geometry.output();
        if (!representable(layout.extents(), layout.strides())) return Optional.empty();
        Tensor output = tensor(layout.extents(), layout.strides());
        Tensor dense = dense(layout.extents());
        Tensor channel = dense(new long[] {geometry.channelCount()});
        var roles = ir.positionToBoundary();
        int x = roles.get(0), scale = roles.get(1), bias = roles.get(2);
        int mean = roles.get(3), variance = roles.get(4);
        int centered = unique + 1, radicand = unique + 2, root = unique + 3;
        int normal = unique + 4, scaled = unique + 5, epsilon = unique + 6;
        Tensor scaleView = channelView(inputs.get(scale), ir.inputRank(), ir.channelAxis());
        Tensor biasView = channelView(inputs.get(bias), ir.inputRank(), ir.channelAxis());
        Tensor meanView = channelView(inputs.get(mean), ir.inputRank(), ir.channelAxis());
        Tensor rootView = channelView(channel, ir.inputRank(), ir.channelAxis());
        List<Step> steps = List.of(
                binary(BinaryAlgorithm.SUB, broadcast(inputs.get(x)), meanView, dense, x, mean, centered),
                binary(BinaryAlgorithm.ADD, broadcast(inputs.get(variance)), scalar(), channel,
                        variance, epsilon, radicand),
                new Step(new Unary(UnaryAlgorithm.SQRT, channel, channel), List.of(radicand), root),
                binary(BinaryAlgorithm.DIV, dense, rootView, dense, centered, root, normal),
                binary(BinaryAlgorithm.MUL, dense, scaleView, dense, normal, scale, scaled),
                binary(BinaryAlgorithm.ADD, dense, biasView, output, scaled, bias, unique));
        offsets.add(bytes(layout.offset()));
        return projection(unit, new Pipeline(inputs, output,
                List.of(dense, channel, channel, dense, dense),
                List.of(new Constant(scalar(), List.of((int) ir.epsilonBits()))), steps), offsets);
    }

    private static Optional<CpuOneDnnProjection> pooling(
            CpuPartitionPreparationPlan.ExecutionUnitPlan unit,
            long[] inputExtents, long[] inputStrides, long inputOffset,
            long[] outputExtents, long[] outputStrides, long outputOffset,
            List<Long> kernel, List<Long> strides, List<Long> dilations,
            List<Long> padding, long divisor) {
        if (!representable(inputExtents, inputStrides)
                || !representable(outputExtents, outputStrides)
                || divisor <= 0 || (divisor & (divisor - 1)) != 0) return Optional.empty();
        boolean everyWindowPadded = false;
        var right = new ArrayList<Long>();
        for (int axis = 0; axis < kernel.size(); axis++) {
            long input = inputExtents[axis + 2], output = outputExtents[axis + 2];
            long stride = strides.get(axis), left = padding.get(axis);
            long span = Math.multiplyExact(kernel.get(axis) - 1, dilations.get(axis));
            long effective = Math.addExact(span, 1);
            // Cartesian windows are all padded iff some spatial axis has no fully in-bounds
            // sampled window. Endpoints suffice even for dilation: samples are monotone.
            long firstFull = left / stride + (left % stride == 0 ? 0 : 1);
            long lastFull = Math.floorDiv(Math.addExact(Math.subtractExact(input - 1, span), left), stride);
            if (firstFull > Math.min(output - 1, lastFull)) everyWindowPadded = true;
            // Preserve the common literal ceil grid with sufficient trailing conceptual zeros.
            // This changes neither the left origin nor the fixed kernel-count divisor.
            long needed = Math.subtractExact(Math.addExact(
                    Math.multiplyExact(output - 1, stride), effective), Math.addExact(input, left));
            long trailing = Math.max(left, needed);
            long numerator = Math.subtractExact(Math.addExact(Math.addExact(input, left), trailing), effective);
            if (numerator < 0 || Math.addExact(numerator / stride, 1) != output) return Optional.empty();
            right.add(trailing);
        }
        if (!everyWindowPadded) return Optional.empty();
        requireBoundaries(unit, 1);
        return projection(unit, new Pooling(tensor(inputExtents, inputStrides),
                tensor(outputExtents, outputStrides), kernel, strides, dilations, padding, right),
                List.of(bytes(inputOffset), bytes(outputOffset)));
    }

    private static Step binary(BinaryAlgorithm algorithm, Tensor left, Tensor right, Tensor output,
            int leftSlot, int rightSlot, int outputSlot) {
        return new Step(new Binary(algorithm, left, right, output),
                List.of(leftSlot, rightSlot), outputSlot);
    }

    private static Tensor channelView(Tensor channel, int rank, int axis) {
        var dimensions = new ArrayList<Long>();
        var strides = new ArrayList<Long>();
        for (int index = 0; index < rank; index++) {
            dimensions.add(index == axis && channel.strides().getFirst() != 0
                    ? channel.dimensions().getFirst() : 1L);
            strides.add(index == axis ? channel.strides().getFirst() : 0L);
        }
        return new Tensor(DataType.FLOAT32, dimensions, strides);
    }

    private static Tensor broadcast(Tensor tensor) {
        var dimensions = new ArrayList<Long>();
        for (int axis = 0; axis < tensor.dimensions().size(); axis++) {
            dimensions.add(tensor.strides().get(axis) == 0 ? 1L : tensor.dimensions().get(axis));
        }
        return new Tensor(DataType.FLOAT32, dimensions, tensor.strides());
    }

    private static Tensor singleton(int rank) {
        return new Tensor(DataType.FLOAT32, java.util.Collections.nCopies(rank, 1L),
                java.util.Collections.nCopies(rank, 1L));
    }

    private static Tensor scalar() { return singleton(1); }

    private static Tensor dense(long[] extents) {
        if (extents.length == 0) return scalar();
        long[] strides = new long[extents.length];
        long stride = 1;
        for (int axis = extents.length - 1; axis >= 0; axis--) {
            strides[axis] = stride;
            stride = Math.multiplyExact(stride, extents[axis]);
        }
        return tensor(extents, strides);
    }

    private static Tensor tensor(long[] dimensions, long[] strides) {
        if (dimensions.length == 0) return scalar();
        return new Tensor(DataType.FLOAT32, Arrays.stream(dimensions).boxed().toList(),
                Arrays.stream(strides).boxed().toList());
    }

    private static boolean representable(long[] dimensions, long[] strides) {
        return dimensions.length <= 12 && dimensions.length == strides.length
                && Arrays.stream(dimensions).allMatch(extent -> extent > 0)
                && Arrays.stream(strides).allMatch(stride -> stride >= 0);
    }

    private static void requireBoundaries(CpuPartitionPreparationPlan.ExecutionUnitPlan unit, int inputs) {
        if (unit.boundaryValues().size() != inputs + 1
                || unit.accessBindings().size() != inputs + 1) {
            throw new IllegalArgumentException("typed composite native boundary count disagrees");
        }
    }

    private static Optional<CpuOneDnnProjection> projection(
            CpuPartitionPreparationPlan.ExecutionUnitPlan unit, CpuOneDnnKernelSpec spec,
            List<Long> offsets) {
        requireBoundaries(unit, spec.inputs().size());
        var positions = new ArrayList<Integer>();
        for (int position = 0; position < spec.inputs().size(); position++) positions.add(position);
        return Optional.of(new CpuOneDnnProjection(spec, offsets, positions, spec.inputs().size()));
    }

    private static long bytes(long elements) { return Math.multiplyExact(elements, Float.BYTES); }
}
