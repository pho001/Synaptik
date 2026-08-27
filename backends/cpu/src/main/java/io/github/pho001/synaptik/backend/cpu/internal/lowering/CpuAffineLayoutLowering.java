package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAffineCopyIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.index.SelectAttrs;
import io.github.pho001.synaptik.model.operation.layout.*;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.*;

/**
 * Validates and composes the bounded straight-line static affine-view family.
 *
 * <p>Lowering retains graph and logical-memory values while marking eligible unit-private
 * intermediates as CPU-private virtual values. It derives exactly one source boundary, one final
 * result boundary, and validates intermediate and final-result uses from the exact consumer
 * occurrences in the shared partition DAG. Address composition, virtual-value eligibility, and
 * represented-bit copy lowering remain CPU-owned. Zero-stride result layouts use a
 * distinct-address write domain only when every repeated logical coordinate selects the same
 * represented source value.</p>
 */
public final class CpuAffineLayoutLowering {
    private final CpuCapabilityProvider capabilities = new CpuCapabilityProvider();

    /** Creates one stateless affine-layout lowerer. */
    public CpuAffineLayoutLowering() {
    }

    /**
     * Lowers one exact affine chain to one represented-bit boundary copy.
     *
     * @param context non-null complete projected partition context whose shared partition DAG
     *     contains the exact affine-chain nodes and consumer occurrences, with static resolved
     *     layouts and matching logical-memory facts
     * @return one immutable lowered partition with two boundaries and zero or more virtual
     *     intermediates; never {@code null}
     * @throws NullPointerException if required projected state is absent or {@code context} is
     *     {@code null}
     * @throws IllegalArgumentException if the partition is not an eligible one-through-eight
     *     affine chain, an intermediate has an external obligation, or the final write domain is
     *     ambiguous
     * @throws ArithmeticException if exact coordinate, address, span, or count arithmetic
     *     overflows
     */
    public CpuPartitionLowering.LoweredPartition lower(
            PrepareContext<? extends BackendAnalysisInputs> context) {
        List<CompiledNode> nodes = context.partitionDag().nodes();
        Map<ValueId, GraphValue> values = new LinkedHashMap<>();
        context.values().forEach(value -> values.put(value.id(), value));
        Map<ValueId, LogicalMemoryRequirement> memory = new LinkedHashMap<>();
        context.memoryRequirements().forEach(value -> memory.put(value.valueId(), value));
        ValueId source = null;
        ValueId previous = null;
        List<ValueId> virtual = new ArrayList<>();
        List<CpuAffineCopyIr.MappingStep> steps = new ArrayList<>();
        for (int i = 0; i < nodes.size(); i++) {
            CompiledNode node = nodes.get(i);
            if (node.inputs().size() != 1 || node.outputs().size() != 1
                    || i > 0 && !node.inputs().getFirst().equals(previous)) {
                throw new IllegalArgumentException("affine partition must be one connected one-input chain");
            }
            GraphValue input = require(values, node.inputs().getFirst());
            GraphValue output = require(values, node.outputs().getFirst());
            if (!capabilities.supports(new OperationCapabilityQuery(node.operation(),
                    List.of(input.descriptor()), List.of(output.descriptor())))) {
                throw new IllegalArgumentException("partition contains an unsupported CPU affine occurrence");
            }
            if (i == 0) source = node.inputs().getFirst();
            if (i + 1 < nodes.size()) {
                requireVirtual(memory.get(node.outputs().getFirst()), context);
                virtual.add(node.outputs().getFirst());
            }
            steps.add(mappingStep(node, input.descriptor().shape().rank(),
                    output.descriptor().shape().rank()));
            previous = node.outputs().getFirst();
        }
        ValueId result = Objects.requireNonNull(previous);
        validateUses(context, result);
        GraphValue sourceValue = require(values, source);
        GraphValue resultValue = require(values, result);
        LayoutDescriptor sourceLayout = sourceValue.descriptor().layout().orElseThrow();
        LayoutDescriptor resultLayout = resultValue.descriptor().layout().orElseThrow();

        long[] finalExtents = resultValue.descriptor().shape().toLongArray();
        long logicalCount = product(finalExtents);
        LinkedHashMap<Long, Long> writes = new LinkedHashMap<>();
        long[] finalCoordinates = new long[finalExtents.length];
        for (long logical = 0; logical < logicalCount; logical++) {
            long resultAddress = address(resultLayout, finalCoordinates);
            long[] sourceCoordinates = finalCoordinates.clone();
            for (int nodeIndex = nodes.size() - 1; nodeIndex >= 0; nodeIndex--) {
                CompiledNode node = nodes.get(nodeIndex);
                sourceCoordinates = reverse(node, require(values, node.inputs().getFirst()),
                        require(values, node.outputs().getFirst()), sourceCoordinates);
            }
            long sourceAddress = address(sourceLayout, sourceCoordinates);
            Long prior = writes.putIfAbsent(resultAddress, sourceAddress);
            if (prior != null && prior.longValue() != sourceAddress) {
                throw new IllegalArgumentException(
                        "non-injective output coordinates do not select one source value");
            }
            advance(finalCoordinates, finalExtents);
        }
        long[] addressPairs = new long[Math.multiplyExact(writes.size(), 2)];
        int pair = 0;
        for (var entry : writes.entrySet()) {
            addressPairs[pair++] = entry.getValue();
            addressPairs[pair++] = entry.getKey();
        }
        long elementCount = writes.size();
        CpuAccessPlan sourcePlan = tablePlan(CpuAccessPlan.AccessKind.READ, addressPairs, 0);
        CpuAccessPlan resultPlan = tablePlan(CpuAccessPlan.AccessKind.WRITE, addressPairs, 1);
        CpuAffineCopyIr.WriteDomain domain = elementCount == logicalCount
                ? CpuAffineCopyIr.WriteDomain.LOGICAL_ELEMENTS
                : CpuAffineCopyIr.WriteDomain.DISTINCT_ADDRESSES;
        DataType type = resultValue.descriptor().dataType();
        var affine = new CpuAffineCopyIr(type, sourcePlan, resultPlan, steps, domain);

        CpuAccessPlan.Binding sourceBinding = conservativeBinding(sourceValue.descriptor().shape()
                .toLongArray(), sourceLayout, CpuAccessPlan.AccessKind.READ);
        CpuAccessPlan.Binding resultBinding = conservativeBinding(resultValue.descriptor().shape()
                .toLongArray(), resultLayout, CpuAccessPlan.AccessKind.WRITE);
        return new CpuPartitionLowering.LoweredPartition(affine,
                List.of(source, result), List.of(sourceBinding, resultBinding),
                List.of(sourceLayout.referencedElementSpan(), resultLayout.referencedElementSpan()),
                List.of(type, type), virtual, new long[]{elementCount}, elementCount,
                "legal: bounded connected static affine chain", addressPairs);
    }

    private static long[] reverse(CompiledNode node, GraphValue input, GraphValue output,
            long[] coordinates) {
        Object kind = node.operation().kind();
        Object attrs = node.operation().attrs();
        long[] inputExtents = input.descriptor().shape().toLongArray();
        if (kind == ContiguousKind.CONTIGUOUS) return coordinates;
        if (kind == ShapeTransformKind.RESHAPE) {
            long linear = linear(coordinates, output.descriptor().shape().toLongArray());
            return coordinates(linear, inputExtents);
        }
        if (kind == ShapeTransformKind.EXPAND) {
            long[] result = new long[inputExtents.length];
            int offset = coordinates.length - inputExtents.length;
            for (int axis = 0; axis < result.length; axis++) result[axis] = inputExtents[axis] == 1
                    ? 0 : coordinates[axis + offset];
            return result;
        }
        if (kind == AxisTransformKind.PERMUTE) {
            long[] result = new long[coordinates.length];
            List<Integer> axes = ((PermutationAttrs) attrs).axes();
            for (int axis = 0; axis < result.length; axis++) result[axes.get(axis)] = coordinates[axis];
            return result;
        }
        if (kind == AxisTransformKind.EXPAND_DIMS) return remove(coordinates,
                ((AxisTransformAttrs) attrs).axis());
        if (kind == AxisTransformKind.SQUEEZE) return insert(coordinates,
                ((AxisTransformAttrs) attrs).axis(), 0);
        if (kind == io.github.pho001.synaptik.model.operation.index.SelectKind.SELECT) {
            SelectAttrs select = (SelectAttrs) attrs;
            return insert(coordinates, select.axis(), select.index());
        }
        if (kind == SliceKind.SLICE && attrs instanceof SliceAttrs slice) {
            long[] result = coordinates.clone();
            for (int i = 0; i < slice.axes().size(); i++) {
                int axis = slice.axes().get(i);
                result[axis] = Math.addExact(slice.starts().get(i),
                        Math.multiplyExact(result[axis], slice.steps().get(i)));
            }
            return result;
        }
        if (kind == SliceKind.SLICE && attrs instanceof CropToShapeAttrs crop) {
            long[] result = coordinates.clone();
            long[] prefix = crop.prefixShape().toLongArray();
            for (int axis = 0; axis < result.length; axis++) result[axis] = Math.addExact(
                    result[axis], prefix[axis]);
            return result;
        }
        throw new IllegalArgumentException("unsupported affine mapping step");
    }

    private static CpuAffineCopyIr.MappingStep mappingStep(CompiledNode node, int inputRank,
            int outputRank) {
        Object kind = node.operation().kind();
        Object attrs = node.operation().attrs();
        CpuAffineCopyIr.MappingKind mappingKind;
        List<Integer> axes = List.of();
        if (kind == ContiguousKind.CONTIGUOUS) mappingKind = CpuAffineCopyIr.MappingKind.CONTIGUOUS;
        else if (kind == ShapeTransformKind.RESHAPE) mappingKind = CpuAffineCopyIr.MappingKind.RESHAPE;
        else if (kind == ShapeTransformKind.EXPAND) mappingKind = CpuAffineCopyIr.MappingKind.EXPAND;
        else if (kind == AxisTransformKind.PERMUTE) {
            mappingKind = CpuAffineCopyIr.MappingKind.PERMUTE;
            axes = ((PermutationAttrs) attrs).axes();
        } else if (kind == AxisTransformKind.EXPAND_DIMS) {
            mappingKind = CpuAffineCopyIr.MappingKind.EXPAND_DIMS;
            axes = List.of(((AxisTransformAttrs) attrs).axis());
        } else if (kind == AxisTransformKind.SQUEEZE) {
            mappingKind = CpuAffineCopyIr.MappingKind.SQUEEZE;
            axes = List.of(((AxisTransformAttrs) attrs).axis());
        } else if (kind == io.github.pho001.synaptik.model.operation.index.SelectKind.SELECT) {
            mappingKind = CpuAffineCopyIr.MappingKind.SELECT;
            axes = List.of(((SelectAttrs) attrs).axis());
        } else if (attrs instanceof SliceAttrs slice) {
            mappingKind = CpuAffineCopyIr.MappingKind.SLICE;
            axes = slice.axes();
        } else if (attrs instanceof CropToShapeAttrs) {
            mappingKind = CpuAffineCopyIr.MappingKind.CROP_TO_SHAPE;
        } else throw new IllegalArgumentException("unsupported affine mapping step");
        return new CpuAffineCopyIr.MappingStep(mappingKind, inputRank, outputRank, axes);
    }

    private static CpuAccessPlan tablePlan(CpuAccessPlan.AccessKind kind, long[] pairs, int parity) {
        boolean constant = true, dense = true;
        for (int i = parity + 2; i < pairs.length; i += 2) {
            constant &= pairs[i] == pairs[parity];
            dense &= pairs[i] == pairs[i - 2] + 1;
        }
        CpuAccessPlan.Regime regime = constant && kind == CpuAccessPlan.AccessKind.READ
                ? CpuAccessPlan.Regime.SCALAR_ALL_ZERO
                : dense ? CpuAccessPlan.Regime.DENSE_LINEAR : CpuAccessPlan.Regime.GENERAL_ODOMETER;
        CpuAccessPlan.AxisRole role = constant ? CpuAccessPlan.AxisRole.BROADCAST
                : dense ? CpuAccessPlan.AxisRole.CONTIGUOUS : CpuAccessPlan.AxisRole.STRIDED;
        return new CpuAccessPlan(kind, regime, 1, List.of(role), dense ? 1 : 0);
    }

    private static CpuAccessPlan.Binding conservativeBinding(long[] extents,
            LayoutDescriptor layout, CpuAccessPlan.AccessKind kind) {
        long count = product(extents);
        CpuAccessPlan plan = normalizedPlan(extents, layout.strides(), kind);
        return CpuAccessPlan.Binding.create(plan, extents, layout.storageOffset(), layout.strides(),
                count, 0, count, layout.referencedElementSpan());
    }

    private static CpuAccessPlan normalizedPlan(long[] extents, long[] strides,
            CpuAccessPlan.AccessKind kind) {
        int suffix = 0; long expected = 1;
        for (int axis = extents.length - 1; axis >= 0; axis--) {
            if (strides[axis] != expected) break;
            suffix++; expected = Math.multiplyExact(expected, Math.max(1, extents[axis]));
        }
        List<CpuAccessPlan.AxisRole> roles = new ArrayList<>();
        for (int axis = 0; axis < extents.length; axis++) roles.add(strides[axis] == 0
                ? CpuAccessPlan.AxisRole.BROADCAST
                : axis >= extents.length - suffix ? CpuAccessPlan.AxisRole.CONTIGUOUS
                : CpuAccessPlan.AxisRole.STRIDED);
        boolean allZero = Arrays.stream(strides).allMatch(value -> value == 0);
        CpuAccessPlan.Regime regime = allZero && kind == CpuAccessPlan.AccessKind.READ
                ? CpuAccessPlan.Regime.SCALAR_ALL_ZERO
                : suffix == extents.length ? CpuAccessPlan.Regime.DENSE_LINEAR
                : suffix > 0 ? CpuAccessPlan.Regime.BLOCK_OUTER
                : CpuAccessPlan.Regime.GENERAL_ODOMETER;
        return new CpuAccessPlan(kind, regime, extents.length, roles, suffix);
    }

    private static void validateUses(PrepareContext<?> context, ValueId result) {
        if (!context.partitionDag().consumers(result).isEmpty())
            throw new IllegalArgumentException("final affine output must not feed the unit");
        List<CompiledNode> nodes = context.partitionDag().nodes();
        for (int i = 0; i + 1 < nodes.size(); i++) {
            ValueId intermediate = nodes.get(i).outputs().getFirst();
            if (context.partitionDag().consumers(intermediate).size() != 1) {
                throw new IllegalArgumentException("affine intermediate must have one use");
            }
        }
    }
    private static void requireVirtual(LogicalMemoryRequirement requirement, PrepareContext<?> context) {
        if (requirement == null || requirement.graphOutput() || requirement.producerPartition().isEmpty()
                || !requirement.producerPartition().orElseThrow().equals(context.partition())
                || !requirement.consumerPartitions().equals(List.of(context.partition())))
            throw new IllegalArgumentException("internal affine result must remain unit-private");
    }
    private static long address(LayoutDescriptor layout, long[] coordinates) { long result = layout.storageOffset();
        for (int i = 0; i < coordinates.length; i++) result = Math.addExact(result,
                Math.multiplyExact(coordinates[i], layout.stride(i))); return result; }
    private static long linear(long[] coordinates, long[] extents) { long result = 0;
        for (int i = 0; i < extents.length; i++) result = Math.addExact(
                Math.multiplyExact(result, extents[i]), coordinates[i]); return result; }
    private static long[] coordinates(long linear, long[] extents) { long[] result = new long[extents.length];
        for (int i = extents.length - 1; i >= 0; i--) if (extents[i] != 0) {
            result[i] = linear % extents[i]; linear /= extents[i]; } return result; }
    private static long[] remove(long[] source, int removed) { long[] result = new long[source.length - 1];
        for (int i = 0, j = 0; i < source.length; i++) if (i != removed) result[j++] = source[i]; return result; }
    private static long[] insert(long[] source, int inserted, long value) { long[] result = new long[source.length + 1];
        for (int i = 0, j = 0; i < result.length; i++) result[i] = i == inserted ? value : source[j++]; return result; }
    private static void advance(long[] coordinates, long[] extents) {
        for (int i = coordinates.length - 1; i >= 0; i--) { coordinates[i]++;
            if (coordinates[i] < extents[i]) return; coordinates[i] = 0; }
    }
    private static long product(long[] extents) { long result = 1;
        for (long extent : extents) result = Math.multiplyExact(result, extent); return result; }
    private static GraphValue require(Map<ValueId, GraphValue> values, ValueId id) {
        GraphValue value = values.get(id);
        if (value == null) throw new IllegalArgumentException("partition value is not projected: " + id);
        return value;
    }
}
