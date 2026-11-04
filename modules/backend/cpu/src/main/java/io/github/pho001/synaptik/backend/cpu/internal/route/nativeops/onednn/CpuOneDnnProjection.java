package io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAffineCopyIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode;
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
 * Numeric projection of one existing static common CPU unit, retaining every external store.
 * Literal pointwise SSA sites become ordered native steps, not a replacement semantic graph.
 * This is not numerical qualification or native-route selection. Origins are applied once by
 * outer binding and remain outside descriptor-relative spans. Scalar attributes are fixed
 * constant slots, never borrowed inputs.
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
     * Projects supported static F32 ordinary families and literal composite substeps.
     * Unsupported instructions reject the complete unit; dynamic, Scan, empty and multiple-output
     * units stay ordinary. Native implementation qualification remains independently mandatory.
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
                    .anyMatch(type -> type != DataType.FLOAT32 && type != DataType.BOOL)) {
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
        if (portable.portableKernelIr() instanceof CpuKernelIr pointwise) {
            return pointwise(unit, pointwise);
        }
        if (portable.portableKernelIr() instanceof CpuAggregateIr aggregate) {
            return aggregate(unit, aggregate);
        }
        if (portable.portableKernelIr() instanceof CpuAffineCopyIr copy) {
            return movement(unit, copy);
        }
        return CpuOneDnnCompositeProjection.from(unit);
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
            inputs.add(new CpuOneDnnKernelSpec.Tensor(DataType.FLOAT32,
                    binding.extents(), binding.effectiveStrides()));
            offsets.add(bytes(binding.baseElementOffset()));
        }
        offsets.add(bytes(geometry.resultOffset()));
        return projection(unit, new CpuOneDnnKernelSpec.Contraction(CpuOneDnnKernelSpec.Operation.MATMUL,
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
        return new CpuOneDnnKernelSpec.Tensor(DataType.FLOAT32, dimensions, strides);
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
            tensors.add(new CpuOneDnnKernelSpec.Tensor(DataType.FLOAT32, dimensions, strides));
            offsets.add(bytes(layout.offset()));
        }
        var bias = ir.intrinsicBias() ? CpuOneDnnKernelSpec.Bias.INTRINSIC
                : ir.epilogue() == CpuConv2dIr.Epilogue.NONE ? CpuOneDnnKernelSpec.Bias.NONE
                : CpuOneDnnKernelSpec.Bias.EXTERNAL_RIGHT;
        var activation = ir.epilogue() == CpuConv2dIr.Epilogue.ADD_RELU
                ? CpuOneDnnKernelSpec.Activation.RELU : CpuOneDnnKernelSpec.Activation.NONE;
        var padding = List.of(ir.paddingHeight(), ir.paddingWidth());
        return projection(unit, new CpuOneDnnKernelSpec.Contraction(CpuOneDnnKernelSpec.Operation.CONVOLUTION,
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
            tensors.add(new CpuOneDnnKernelSpec.Tensor(DataType.FLOAT32, dimensions, strides));
            offsets.add(bytes(layout.offset()));
        }
        var padding = List.of(ir.paddingDepth(), ir.paddingHeight(), ir.paddingWidth());
        return projection(unit, new CpuOneDnnKernelSpec.Contraction(CpuOneDnnKernelSpec.Operation.CONVOLUTION,
                tensors.subList(0, tensors.size() - 1), tensors.getLast(), (int) ir.groups(),
                List.of(ir.strideDepth(), ir.strideHeight(), ir.strideWidth()),
                List.of(ir.dilationDepth(), ir.dilationHeight(), ir.dilationWidth()), padding, padding,
                ir.intrinsicBias() ? CpuOneDnnKernelSpec.Bias.INTRINSIC : CpuOneDnnKernelSpec.Bias.NONE,
                CpuOneDnnKernelSpec.Activation.NONE), offsets);
    }

    private static Optional<CpuOneDnnProjection> pointwise(
            CpuPartitionPreparationPlan.ExecutionUnitPlan unit, CpuKernelIr ir) {
        if (!ir.familyIdentity().equals("pointwise") || ir.instructions().isEmpty()
                || ir.stores().size() != 1 || ir.stores().getFirst().outputOrdinal() != 0
                || unit.accessBindings().size() != unit.boundaryValues().size()) return Optional.empty();
        int outputValue = ir.stores().getFirst().value();
        int inputCount = unit.boundaryValues().size() - 1;
        var inputs = new ArrayList<CpuOneDnnKernelSpec.Tensor>();
        var intermediates = new ArrayList<CpuOneDnnKernelSpec.Tensor>();
        var tensors = new CpuOneDnnKernelSpec.Tensor[ir.values().size()];
        var slots = new int[ir.values().size()];
        java.util.Arrays.fill(slots, -1);
        int boundary = 0;
        for (var value : ir.values()) {
            if (value.kind() == CpuKernelIr.Value.Kind.INPUT) {
                if (boundary >= inputCount || value.dataType() != DataType.FLOAT32) return Optional.empty();
                var binding = unit.accessBindings().get(boundary);
                if (!valid(binding)) return Optional.empty();
                var tensor = tensor(value.dataType(), binding, true);
                inputs.add(tensor);
                tensors[value.ordinal()] = tensor;
                slots[value.ordinal()] = boundary++;
            } else if (value.ordinal() == outputValue) {
                var binding = unit.accessBindings().getLast();
                if (!valid(binding)) return Optional.empty();
                tensors[value.ordinal()] = tensor(value.dataType(), binding, false);
                slots[value.ordinal()] = inputCount;
            } else {
                if (value.kind() == CpuKernelIr.Value.Kind.OUTPUT) return Optional.empty();
                var dimensions = longs(unit.extents());
                if (!positive(dimensions)) return Optional.empty();
                tensors[value.ordinal()] = dense(value.dataType(), dimensions);
                slots[value.ordinal()] = inputCount + 1 + intermediates.size();
                intermediates.add(tensors[value.ordinal()]);
            }
        }
        if (boundary != inputCount) return Optional.empty();
        var constants = new ArrayList<CpuOneDnnKernelSpec.Constant>();
        var steps = new ArrayList<CpuOneDnnKernelSpec.Step>();
        for (var instruction : ir.instructions()) {
            var operandSlots = new ArrayList<Integer>();
            var operands = new ArrayList<CpuOneDnnKernelSpec.Tensor>();
            for (int input : instruction.inputs()) {
                if (tensors[input].dataType() != DataType.FLOAT32) return Optional.empty();
                operandSlots.add(slots[input]);
                operands.add(tensors[input]);
            }
            var output = tensors[instruction.output()];
            CpuOneDnnKernelSpec atomic;
            var unary = unary(instruction.opcode());
            var binary = binary(instruction.opcode());
            if (unary != null) {
                var input = operands.getFirst();
                var expanded = new CpuOneDnnKernelSpec.Tensor(input.dataType(),
                        output.dimensions(), input.strides());
                atomic = new CpuOneDnnKernelSpec.Unary(unary, expanded, output);
            } else if (binary != null) {
                if (instruction.scalarImmediate() != null) {
                    var immediate = instruction.scalarImmediate();
                    if (immediate.dataType() != DataType.FLOAT32) return Optional.empty();
                    var scalar = dense(DataType.FLOAT32,
                            java.util.Collections.nCopies(output.dimensions().size(), 1L));
                    var constant = new CpuOneDnnKernelSpec.Constant(scalar, List.of((int) immediate.bits()));
                    int constantPosition = constants.indexOf(constant);
                    if (constantPosition < 0) {
                        constantPosition = constants.size();
                        constants.add(constant);
                    }
                    operands.add(scalar);
                    operandSlots.add(inputCount + 1 + intermediates.size() + constantPosition);
                }
                if (instruction.opcode() == CpuPointwiseOpcode.GREATER_THAN
                        || instruction.opcode() == CpuPointwiseOpcode.GREATER_OR_EQUAL) {
                    java.util.Collections.reverse(operands);
                    java.util.Collections.reverse(operandSlots);
                }
                atomic = new CpuOneDnnKernelSpec.Binary(binary, operands.getFirst(), operands.get(1), output);
            } else if (instruction.opcode() == CpuPointwiseOpcode.CAST
                    && operands.getFirst().dataType() == output.dataType()
                    && output.dataType() == DataType.FLOAT32) {
                var input = operands.getFirst();
                atomic = new CpuOneDnnKernelSpec.Reorder(new CpuOneDnnKernelSpec.Tensor(
                        input.dataType(), output.dimensions(), input.strides()), output);
            } else return Optional.empty();
            steps.add(new CpuOneDnnKernelSpec.Step(atomic, operandSlots, slots[instruction.output()]));
        }
        return projection(unit, new CpuOneDnnKernelSpec.Pipeline(inputs, tensors[outputValue],
                intermediates, constants, steps), offsets(unit));
    }

    private static CpuOneDnnKernelSpec.UnaryAlgorithm unary(CpuPointwiseOpcode opcode) {
        return switch (opcode) {
            case ABS -> CpuOneDnnKernelSpec.UnaryAlgorithm.ABS;
            case SQRT -> CpuOneDnnKernelSpec.UnaryAlgorithm.SQRT;
            case TANH -> CpuOneDnnKernelSpec.UnaryAlgorithm.TANH;
            case EXP -> CpuOneDnnKernelSpec.UnaryAlgorithm.EXP;
            default -> null;
        };
    }

    private static CpuOneDnnKernelSpec.BinaryAlgorithm binary(CpuPointwiseOpcode opcode) {
        return switch (opcode) {
            case ADD, SCALAR_ADD -> CpuOneDnnKernelSpec.BinaryAlgorithm.ADD;
            case SUB, SCALAR_SUB -> CpuOneDnnKernelSpec.BinaryAlgorithm.SUB;
            case MUL, SCALAR_MUL -> CpuOneDnnKernelSpec.BinaryAlgorithm.MUL;
            case DIV, SCALAR_DIV -> CpuOneDnnKernelSpec.BinaryAlgorithm.DIV;
            case MIN, SCALAR_MIN -> CpuOneDnnKernelSpec.BinaryAlgorithm.MIN;
            case MAX, SCALAR_MAX -> CpuOneDnnKernelSpec.BinaryAlgorithm.MAX;
            case LESS_THAN, GREATER_THAN -> CpuOneDnnKernelSpec.BinaryAlgorithm.LT;
            case LESS_OR_EQUAL, GREATER_OR_EQUAL -> CpuOneDnnKernelSpec.BinaryAlgorithm.LE;
            case EQUAL -> CpuOneDnnKernelSpec.BinaryAlgorithm.EQ;
            case NOT_EQUAL -> CpuOneDnnKernelSpec.BinaryAlgorithm.NE;
            default -> null;
        };
    }

    private static Optional<CpuOneDnnProjection> aggregate(
            CpuPartitionPreparationPlan.ExecutionUnitPlan unit, CpuAggregateIr ir) {
        if (ir.dataType() != DataType.FLOAT32 || ir.symbolicGeometry()
                || ir.domainCount() == 0 || unit.accessBindings().size() != 2
                || ir.kind() != CpuAggregateIr.Kind.SUM && ir.kind() != CpuAggregateIr.Kind.MEAN
                    && ir.kind() != CpuAggregateIr.Kind.PROD) return Optional.empty();
        var source = unit.accessBindings().getFirst();
        var target = unit.accessBindings().getLast();
        if (!valid(source) || !valid(target)) return Optional.empty();
        var input = tensor(DataType.FLOAT32, source, false);
        if (source.extents().isEmpty()) {
            return projection(unit, new CpuOneDnnKernelSpec.Reorder(input,
                    tensor(DataType.FLOAT32, target, false)), offsets(unit));
        }
        var dimensions = new ArrayList<>(input.dimensions());
        var strides = new ArrayList<Long>();
        var selected = new java.util.HashSet<Integer>();
        for (int axis : ir.selectedAxes()) selected.add(axis);
        int targetAxis = 0;
        int leading = source.extents().size() - target.extents().size();
        for (int axis = 0; axis < dimensions.size(); axis++) {
            if (selected.contains(axis)) {
                dimensions.set(axis, 1L);
                strides.add(0L);
                if (ir.keepDimensions() || ir.form() == CpuAggregateIr.Form.SUM_TO_SHAPE
                        && axis >= leading) targetAxis++;
            } else {
                strides.add(target.effectiveStrides().get(targetAxis++));
            }
        }
        if (targetAxis != target.extents().size()) return Optional.empty();
        var output = new CpuOneDnnKernelSpec.Tensor(DataType.FLOAT32, dimensions, strides);
        if (output.byteSize() != tensor(DataType.FLOAT32, target, false).byteSize()) return Optional.empty();
        var algorithm = ir.kind() == CpuAggregateIr.Kind.PROD
                ? CpuOneDnnKernelSpec.ReductionAlgorithm.PROD : CpuOneDnnKernelSpec.ReductionAlgorithm.SUM;
        if (ir.selectedAxes().length == 0) {
            return projection(unit, new CpuOneDnnKernelSpec.Reorder(input, output), offsets(unit));
        }
        if (ir.kind() != CpuAggregateIr.Kind.MEAN) {
            return projection(unit, new CpuOneDnnKernelSpec.Reduction(algorithm, input, output), offsets(unit));
        }
        var sum = dense(DataType.FLOAT32, dimensions);
        var divisor = dense(DataType.FLOAT32, java.util.Collections.nCopies(dimensions.size(), 1L));
        var constant = new CpuOneDnnKernelSpec.Constant(divisor,
                List.of(Float.floatToRawIntBits((float) ir.domainCount())));
        var pipeline = new CpuOneDnnKernelSpec.Pipeline(List.of(input), output,
                List.of(sum), List.of(constant), List.of(
                    new CpuOneDnnKernelSpec.Step(new CpuOneDnnKernelSpec.Reduction(algorithm, input, sum),
                            List.of(0), 2),
                    new CpuOneDnnKernelSpec.Step(new CpuOneDnnKernelSpec.Binary(
                            CpuOneDnnKernelSpec.BinaryAlgorithm.DIV, sum, divisor, output), List.of(2, 3), 1)));
        return projection(unit, pipeline, offsets(unit));
    }

    private static Optional<CpuOneDnnProjection> movement(
            CpuPartitionPreparationPlan.ExecutionUnitPlan unit, CpuAffineCopyIr ir) {
        if (ir.dataType() != DataType.FLOAT32
                || ir.writeDomain() != CpuAffineCopyIr.WriteDomain.LOGICAL_ELEMENTS
                || unit.accessBindings().size() != 2) return Optional.empty();
        var source = unit.accessBindings().getFirst();
        var output = unit.accessBindings().getLast();
        if (!valid(source) || !valid(output) || !source.extents().equals(output.extents())) return Optional.empty();
        return projection(unit, new CpuOneDnnKernelSpec.Reorder(
                tensor(DataType.FLOAT32, source, false), tensor(DataType.FLOAT32, output, false)), offsets(unit));
    }

    private static List<Long> offsets(CpuPartitionPreparationPlan.ExecutionUnitPlan unit) {
        var types = unit.portablePlan().specialization().boundaryDataTypes();
        var offsets = new ArrayList<Long>();
        for (int position = 0; position < unit.accessBindings().size(); position++) {
            offsets.add(Math.multiplyExact(unit.accessBindings().get(position).baseElementOffset(),
                    types.get(position) == DataType.BOOL ? 1L : Float.BYTES));
        }
        return offsets;
    }

    private static boolean valid(CpuAccessPlan.Binding binding) {
        return binding.start() == 0 && binding.end() == binding.elementCount()
                && (binding.extents().isEmpty() || representable(binding.extents(), binding.effectiveStrides()));
    }

    private static boolean positive(List<Long> dimensions) {
        return dimensions.size() <= 12 && dimensions.stream().allMatch(extent -> extent > 0);
    }

    private static CpuOneDnnKernelSpec.Tensor tensor(DataType type, CpuAccessPlan.Binding binding,
            boolean broadcast) {
        if (binding.extents().isEmpty()) return dense(type, List.of(1L));
        return new CpuOneDnnKernelSpec.Tensor(type, broadcast
                ? broadcastDimensions(binding.extents(), binding.effectiveStrides()) : binding.extents(),
                binding.effectiveStrides());
    }

    private static CpuOneDnnKernelSpec.Tensor dense(DataType type, List<Long> extents) {
        var dimensions = extents.isEmpty() ? List.of(1L) : extents;
        var strides = new ArrayList<>(java.util.Collections.nCopies(dimensions.size(), 1L));
        long stride = 1;
        for (int axis = dimensions.size() - 1; axis >= 0; axis--) {
            strides.set(axis, stride);
            stride = Math.multiplyExact(stride, dimensions.get(axis));
        }
        return new CpuOneDnnKernelSpec.Tensor(type, dimensions, strides);
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
