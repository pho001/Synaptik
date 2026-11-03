package io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuConv2dIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuConv3dIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMatmulLowering;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Numeric projection of an existing static F32 contraction and its literal common suffix.
 * This is not numerical qualification or native-route selection. Offsets are in spec-input
 * order followed by the output; they are applied once by the outer boundary binding, never
 * included in a tensor's descriptor-relative byte span.
 *
 * @param spec non-null immutable numeric descriptors, not a qualification credential
 * @param byteOffsets non-null non-negative original byte origins, inputs followed by output; copied
 * @param inputBoundaryPositions non-null common-unit input positions in spec order; copied
 * @param outputBoundaryPosition non-negative common-unit output position, distinct from every input
 */
public record CpuOneDnnProjection(CpuOneDnnKernelSpec spec, List<Long> byteOffsets,
        List<Integer> inputBoundaryPositions, int outputBoundaryPosition) {
    /**
     * Snapshots positional binding without retaining a graph, carrier or native resource.
     *
     * @param spec non-null numeric descriptor recipe
     * @param byteOffsets non-null non-negative input/output byte origins; copied
     * @param inputBoundaryPositions non-null common input positions matching spec order; copied
     * @param outputBoundaryPosition non-negative distinct common output position
     * @throws NullPointerException if a required object or list element is null
     * @throws IllegalArgumentException if boundary positions, origins or operand counts disagree
     */
    public CpuOneDnnProjection {
        Objects.requireNonNull(spec, "spec");
        byteOffsets = List.copyOf(byteOffsets);
        inputBoundaryPositions = List.copyOf(inputBoundaryPositions);
        if (byteOffsets.size() != spec.inputs().size() + 1
                || inputBoundaryPositions.size() != spec.inputs().size()
                || outputBoundaryPosition < 0
                || byteOffsets.stream().anyMatch(offset -> offset < 0)
                || inputBoundaryPositions.stream().anyMatch(position ->
                    position < 0 || position == outputBoundaryPosition)) {
            throw new IllegalArgumentException("oneDNN projection boundaries disagree");
        }
    }

    /**
     * Projects typed MATMUL/Conv2d/Conv3d facts already owned by common CPU lowering.
     * Convolution and literal suffix projection is diagnostic representability, not production
     * qualification. Symbolic recipes, row regions, empty contractions and other represented
     * types stay outside this projection; later qualification and dense-layout checks are mandatory.
     *
     * @param unit non-null already validated common execution unit
     * @return immutable numeric facts, or empty when this unit is not representable here
     * @throws NullPointerException if the unit is null
     * @throws IllegalArgumentException if validated typed common-unit boundaries disagree
     * @throws ArithmeticException if checked origins or descriptor spans overflow
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
        if (portable.specialization().matmulIr().isPresent()
                && unit.matmulGeometry().isPresent()
                && portable.portableKernelIr() instanceof CpuKernelIr) {
            return matmul(unit, portable.specialization().matmulIr().orElseThrow(),
                    unit.matmulGeometry().orElseThrow());
        }
        if (portable.portableKernelIr() instanceof CpuConv2dIr convolution
                && unit.conv2dGeometry().isPresent()) {
            return conv2d(unit, convolution);
        }
        if (portable.portableKernelIr() instanceof CpuConv3dIr convolution
                && unit.conv3dGeometry().isPresent()) {
            return conv3d(unit, convolution);
        }
        return Optional.empty();
    }

    private static Optional<CpuOneDnnProjection> matmul(
            CpuPartitionPreparationPlan.ExecutionUnitPlan unit, CpuMatmulIr ir,
            CpuMatmulLowering.Geometry geometry) {
        if (ir.leftType() != DataType.FLOAT32 || ir.rightType() != DataType.FLOAT32
                || ir.resultType() != DataType.FLOAT32
                || geometry.leftType() != DataType.FLOAT32
                || geometry.rightType() != DataType.FLOAT32
                || geometry.resultType() != DataType.FLOAT32
                || geometry.k() <= 1 || geometry.m() == 0 || geometry.n() == 0
                || geometry.batchCount() == 0 || geometry.outputCount() == 0) {
            return Optional.empty();
        }
        var activation = switch (ir.epilogue().terminal()) {
            case NONE -> CpuOneDnnKernelSpec.Activation.NONE;
            case RELU -> CpuOneDnnKernelSpec.Activation.RELU;
            case TANH -> CpuOneDnnKernelSpec.Activation.TANH;
            case SIGMOID -> CpuOneDnnKernelSpec.Activation.SIGMOID;
            default -> null;
        };
        if (activation == null) return Optional.empty();
        long[] batch = geometry.batchExtents();
        if (batch.length + 2 > 12) return Optional.empty();
        for (long extent : batch) if (extent == 0) return Optional.empty();
        var inputs = new ArrayList<CpuOneDnnKernelSpec.Tensor>();
        inputs.add(matrix(batch, geometry.leftBatchStrides(), geometry.m(), geometry.k(),
                geometry.leftMStride(), geometry.leftKStride(), true));
        inputs.add(matrix(batch, geometry.rightBatchStrides(), geometry.k(), geometry.n(),
                geometry.rightKStride(), geometry.rightNStride(), true));
        var output = matrix(batch, geometry.resultBatchStrides(), geometry.m(), geometry.n(),
                geometry.resultMStride(), geometry.resultNStride(), false);
        var offsets = new ArrayList<Long>();
        offsets.add(bytes(geometry.leftOffset()));
        offsets.add(bytes(geometry.rightOffset()));
        var bias = switch (ir.epilogue().addInputOrder()) {
            case NONE -> CpuOneDnnKernelSpec.Bias.NONE;
            case MATMUL_LEFT -> CpuOneDnnKernelSpec.Bias.EXTERNAL_RIGHT;
            case MATMUL_RIGHT -> CpuOneDnnKernelSpec.Bias.EXTERNAL_LEFT;
        };
        if (ir.epilogue().hasBias()) {
            CpuAccessPlan.Binding binding = unit.accessBindings().get(2);
            if (!representable(binding.extents(), binding.effectiveStrides())) {
                return Optional.empty();
            }
            // Keep the actual rank-one N bias; it is not an intrinsic contraction bias.
            inputs.add(new CpuOneDnnKernelSpec.Tensor(binding.extents(), binding.effectiveStrides()));
            offsets.add(bytes(binding.baseElementOffset()));
        }
        offsets.add(bytes(geometry.resultOffset()));
        return projection(unit, new CpuOneDnnKernelSpec(CpuOneDnnKernelSpec.Operation.MATMUL,
                inputs, output, 1, List.of(), List.of(), List.of(), List.of(), bias, activation), offsets);
    }

    private static CpuOneDnnKernelSpec.Tensor matrix(long[] batch, long[] batchStrides,
            long rows, long columns, long rowStride, long columnStride, boolean broadcast) {
        var dimensions = new ArrayList<Long>();
        var strides = new ArrayList<Long>();
        for (int axis = 0; axis < batch.length; axis++) {
            // A zero effective batch stride is one broadcast operand, not another contributor.
            dimensions.add(broadcast && batchStrides[axis] == 0 ? 1L : batch[axis]);
            strides.add(batchStrides[axis]);
        }
        dimensions.add(rows);
        dimensions.add(columns);
        strides.add(rowStride);
        strides.add(columnStride);
        return new CpuOneDnnKernelSpec.Tensor(dimensions, strides);
    }

    private static Optional<CpuOneDnnProjection> conv2d(
            CpuPartitionPreparationPlan.ExecutionUnitPlan unit, CpuConv2dIr ir) {
        if (ir.resultType() != DataType.FLOAT32
                || ir.inputTypes().stream().anyMatch(type -> type != DataType.FLOAT32)
                || ir.groups() > Integer.MAX_VALUE
                || ir.intrinsicBias() && ir.epilogue() != CpuConv2dIr.Epilogue.NONE) {
            return Optional.empty();
        }
        var boundaries = unit.conv2dGeometry().orElseThrow().boundaries();
        var tensors = new ArrayList<CpuOneDnnKernelSpec.Tensor>();
        var offsets = new ArrayList<Long>();
        for (int position = 0; position < boundaries.size(); position++) {
            var layout = boundaries.get(position);
            var dimensions = longs(layout.extents());
            var strides = longs(layout.strides());
            if (!representable(dimensions, strides)) return Optional.empty();
            if (position == 2 && !ir.intrinsicBias()
                    && ir.epilogue() != CpuConv2dIr.Epilogue.NONE) {
                // Common lowering stores the broadcasted output domain with effective strides.
                // Recover its singleton axes, without turning a width ADD into a channel bias.
                dimensions = broadcastDimensions(dimensions, strides);
            }
            tensors.add(new CpuOneDnnKernelSpec.Tensor(dimensions, strides));
            offsets.add(bytes(layout.offset()));
        }
        var bias = ir.intrinsicBias() ? CpuOneDnnKernelSpec.Bias.INTRINSIC
                : ir.epilogue() == CpuConv2dIr.Epilogue.NONE ? CpuOneDnnKernelSpec.Bias.NONE
                : CpuOneDnnKernelSpec.Bias.EXTERNAL_RIGHT;
        var activation = ir.epilogue() == CpuConv2dIr.Epilogue.ADD_RELU
                ? CpuOneDnnKernelSpec.Activation.RELU : CpuOneDnnKernelSpec.Activation.NONE;
        var padding = List.of(ir.paddingHeight(), ir.paddingWidth());
        return projection(unit, new CpuOneDnnKernelSpec(CpuOneDnnKernelSpec.Operation.CONVOLUTION,
                tensors.subList(0, tensors.size() - 1), tensors.getLast(), (int) ir.groups(),
                List.of(ir.strideHeight(), ir.strideWidth()),
                List.of(ir.dilationHeight(), ir.dilationWidth()), padding, padding, bias, activation), offsets);
    }

    private static Optional<CpuOneDnnProjection> conv3d(
            CpuPartitionPreparationPlan.ExecutionUnitPlan unit, CpuConv3dIr ir) {
        if (ir.resultType() != DataType.FLOAT32
                || ir.inputTypes().stream().anyMatch(type -> type != DataType.FLOAT32)
                || ir.groups() > Integer.MAX_VALUE) {
            return Optional.empty();
        }
        var tensors = new ArrayList<CpuOneDnnKernelSpec.Tensor>();
        var offsets = new ArrayList<Long>();
        for (var layout : unit.conv3dGeometry().orElseThrow().boundaries()) {
            var dimensions = longs(layout.extents());
            var strides = longs(layout.strides());
            if (!representable(dimensions, strides)) return Optional.empty();
            tensors.add(new CpuOneDnnKernelSpec.Tensor(dimensions, strides));
            offsets.add(bytes(layout.offset()));
        }
        var padding = List.of(ir.paddingDepth(), ir.paddingHeight(), ir.paddingWidth());
        return projection(unit, new CpuOneDnnKernelSpec(CpuOneDnnKernelSpec.Operation.CONVOLUTION,
                tensors.subList(0, tensors.size() - 1), tensors.getLast(), (int) ir.groups(),
                List.of(ir.strideDepth(), ir.strideHeight(), ir.strideWidth()),
                List.of(ir.dilationDepth(), ir.dilationHeight(), ir.dilationWidth()), padding, padding,
                ir.intrinsicBias() ? CpuOneDnnKernelSpec.Bias.INTRINSIC : CpuOneDnnKernelSpec.Bias.NONE,
                CpuOneDnnKernelSpec.Activation.NONE), offsets);
    }

    private static Optional<CpuOneDnnProjection> projection(
            CpuPartitionPreparationPlan.ExecutionUnitPlan unit, CpuOneDnnKernelSpec spec,
            List<Long> offsets) {
        if (unit.boundaryValues().size() != spec.inputs().size() + 1
                || unit.accessBindings().size() != unit.boundaryValues().size()) {
            throw new IllegalArgumentException("typed contraction boundary count disagrees");
        }
        var positions = new ArrayList<Integer>();
        for (int position = 0; position < spec.inputs().size(); position++) positions.add(position);
        return Optional.of(new CpuOneDnnProjection(spec, offsets, positions, spec.inputs().size()));
    }

    private static List<Long> broadcastDimensions(List<Long> dimensions, List<Long> strides) {
        var result = new ArrayList<Long>();
        for (int axis = 0; axis < dimensions.size(); axis++) {
            result.add(strides.get(axis) == 0 ? 1L : dimensions.get(axis));
        }
        return result;
    }

    private static boolean representable(List<Long> dimensions, List<Long> strides) {
        return !dimensions.isEmpty() && dimensions.size() <= 12
                && dimensions.size() == strides.size()
                && dimensions.stream().allMatch(extent -> extent > 0)
                && strides.stream().allMatch(stride -> stride >= 0);
    }

    private static List<Long> longs(long[] values) {
        var result = new ArrayList<Long>();
        for (long value : values) result.add(value);
        return result;
    }

    private static long bytes(long elementOffset) {
        return Math.multiplyExact(elementOffset, Float.BYTES);
    }
}
