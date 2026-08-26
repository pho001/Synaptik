package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuConv2dIr;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dKind;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Lowers one exact static grouped NCHW Conv2d occurrence to direct complete-output-cell work.
 *
 * <p>The lowerer validates the occurrence through the CPU capability boundary, snapshots every
 * resolved layout, and retains no scratch, packing, im2col, partial, or combine resource.</p>
 */
public final class CpuConv2dLowering {
    private final CpuCapabilityProvider capabilities = new CpuCapabilityProvider();

    /** Creates a stateless direct-convolution lowerer. */
    public CpuConv2dLowering() { }

    /**
     * Lowers one supported Conv2d-led partition containing the convolution and, when legal, an
     * external ADD or external ADD followed by RELU.
     *
     * @param context non-null one-through-three-node CPU preparation projection
     * @return immutable zero-workspace lowering
     * @throws NullPointerException if {@code context} or a required fact is {@code null}
     * @throws IllegalArgumentException if occurrence, descriptors, layouts, or geometry disagree
     * @throws ArithmeticException if checked count, address, or span arithmetic overflows
     */
    public CpuPartitionLowering.LoweredPartition lower(
            PrepareContext<? extends BackendAnalysisInputs> context) {
        Objects.requireNonNull(context, "context");
        if (context.nodes().isEmpty() || context.nodes().size() > 3) {
            throw new IllegalArgumentException("CPU Conv2d requires one through three bounded nodes");
        }
        var node = context.nodes().getFirst();
        Map<ValueId, GraphValue> values = new LinkedHashMap<>();
        context.values().forEach(value -> values.put(value.id(), value));
        var query = new OperationCapabilityQuery(node.operation(), node.inputs().stream()
                .map(id -> require(values, id).descriptor()).toList(), node.outputs().stream()
                .map(id -> require(values, id).descriptor()).toList());
        if (!capabilities.supports(query) || node.operation().kind() != Conv2dKind.CONV2D
                || !(node.operation().attrs() instanceof Conv2dAttrs attrs)) {
            throw new IllegalArgumentException("partition contains an unsupported CPU Conv2d occurrence");
        }
        ValueId convOutputId = node.outputs().getFirst();
        if (node.inputs().contains(convOutputId)) {
            throw new IllegalArgumentException("Conv2d output must be distinct from every input");
        }
        CpuConv2dIr.Epilogue epilogue = CpuConv2dIr.Epilogue.NONE;
        ValueId externalAddId = null;
        ValueId previous = convOutputId;
        Map<ValueId, LogicalMemoryRequirement> memory = new LinkedHashMap<>();
        context.memoryRequirements().forEach(requirement -> memory.put(requirement.valueId(), requirement));
        if (context.nodes().size() >= 2) {
            var add = context.nodes().get(1);
            if (add.operation().kind() != BinaryArithmeticKind.ADD
                    || add.operation().attrs() != NoOperationAttrs.INSTANCE
                    || add.inputs().size() != 2 || add.outputs().size() != 1
                    || add.inputs().stream().filter(convOutputId::equals).count() != 1) {
                throw new IllegalArgumentException("Conv2d suffix must begin with one direct ADD");
            }
            externalAddId = add.inputs().get(0).equals(convOutputId)
                    ? add.inputs().get(1) : add.inputs().get(0);
            var addQuery = new OperationCapabilityQuery(add.operation(), add.inputs().stream()
                    .map(id -> require(values, id).descriptor()).toList(), add.outputs().stream()
                    .map(id -> require(values, id).descriptor()).toList());
            if (!capabilities.supports(addQuery)
                    || require(values, convOutputId).descriptor().dataType()
                        != require(values, externalAddId).descriptor().dataType()
                    || require(values, convOutputId).descriptor().dataType()
                        == io.github.pho001.synaptik.model.datatype.DataType.BFLOAT16) {
                throw new IllegalArgumentException("Conv2d external ADD is unsupported");
            }
            requirePrivate(memory.get(convOutputId), context, convOutputId);
            previous = add.outputs().getFirst();
            epilogue = CpuConv2dIr.Epilogue.ADD;
        }
        if (context.nodes().size() == 3) {
            var activation = context.nodes().get(2);
            if (activation.operation().kind() != UnaryElementwiseKind.RELU
                    || activation.operation().attrs() != NoOperationAttrs.INSTANCE
                    || !activation.inputs().equals(List.of(previous))
                    || activation.outputs().size() != 1) {
                throw new IllegalArgumentException("Conv2d activation suffix supports only exact RELU");
            }
            var activationQuery = new OperationCapabilityQuery(activation.operation(),
                    activation.inputs().stream().map(id -> require(values, id).descriptor()).toList(),
                    activation.outputs().stream().map(id -> require(values, id).descriptor()).toList());
            if (!capabilities.supports(activationQuery)) {
                throw new IllegalArgumentException("Conv2d RELU suffix is unsupported");
            }
            requirePrivate(memory.get(previous), context, previous);
            previous = activation.outputs().getFirst();
            epilogue = CpuConv2dIr.Epilogue.ADD_RELU;
        }
        ValueId outputId = previous;
        var boundaryIds = new ArrayList<>(node.inputs());
        if (externalAddId != null) boundaryIds.add(externalAddId);
        boundaryIds.add(outputId);
        var bindings = new ArrayList<CpuAccessPlan.Binding>();
        var spans = new ArrayList<Long>();
        var types = new ArrayList<io.github.pho001.synaptik.model.datatype.DataType>();
        var layouts = new ArrayList<Layout>();
        for (int i = 0; i < boundaryIds.size(); i++) {
            GraphValue value = require(values, boundaryIds.get(i));
            Layout layout = layout(value);
            if (externalAddId != null && boundaryIds.get(i).equals(externalAddId)) {
                layout = broadcastLayout(value, require(values, outputId));
            }
            layouts.add(layout);
            bindings.add(binding(layout, i == boundaryIds.size() - 1
                    ? CpuAccessPlan.AccessKind.WRITE : CpuAccessPlan.AccessKind.READ));
            spans.add(value.descriptor().layout().orElseThrow().referencedElementSpan());
            types.add(value.descriptor().dataType());
        }
        List<io.github.pho001.synaptik.model.datatype.DataType> inputTypes = types.subList(
                0, types.size() - 1);
        var ir = new CpuConv2dIr(inputTypes, types.getLast(), attrs.strideHeight(),
                attrs.strideWidth(), attrs.paddingHeight(), attrs.paddingWidth(),
                attrs.dilationHeight(), attrs.dilationWidth(), attrs.groups(), 1,
                node.inputs().size() == 3, epilogue,
                bindings.subList(0, bindings.size() - 1).stream()
                        .map(CpuAccessPlan.Binding::plan).toList(), bindings.getLast().plan());
        long outputCount = count(layouts.getLast().extents());
        return new CpuPartitionLowering.LoweredPartition(ir, boundaryIds, bindings, spans, types,
                context.nodes().subList(0, context.nodes().size() - 1).stream()
                        .map(part -> part.outputs().getFirst()).toList(),
                layouts.getLast().extents(), outputCount,
                epilogue == CpuConv2dIr.Epilogue.NONE
                        ? "legal: one direct static grouped NCHW Conv2d"
                        : "legal: bounded fused Conv2d-led " + epilogue,
                new long[0],
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.of(new Geometry(layouts)));
    }

    private static void requirePrivate(LogicalMemoryRequirement requirement,
            PrepareContext<?> context, ValueId id) {
        if (requirement == null || requirement.graphOutput()
                || requirement.producerPartition().isEmpty()
                || !requirement.producerPartition().orElseThrow().equals(context.partition())
                || !requirement.consumerPartitions().equals(List.of(context.partition()))) {
            throw new IllegalArgumentException("Conv2d fused intermediate must be private: " + id);
        }
    }

    private static Layout broadcastLayout(GraphValue source, GraphValue output) {
        long[] target = output.descriptor().shape().toLongArray();
        long[] shape = source.descriptor().shape().toLongArray();
        long[] original = source.descriptor().layout().orElseThrow().strides();
        long[] strides = new long[target.length];
        int shift = target.length - shape.length;
        if (shift < 0) throw new IllegalArgumentException("Conv2d ADD source rank is too large");
        for (int axis = 0; axis < target.length; axis++) {
            if (axis < shift) strides[axis] = 0;
            else {
                long extent = shape[axis - shift];
                if (extent != target[axis] && extent != 1) {
                    throw new IllegalArgumentException("Conv2d ADD source does not right-broadcast");
                }
                strides[axis] = extent == 1 && target[axis] != 1 ? 0 : original[axis - shift];
            }
        }
        return new Layout(target, source.descriptor().layout().orElseThrow().storageOffset(), strides);
    }

    private static GraphValue require(Map<ValueId, GraphValue> values, ValueId id) {
        GraphValue result = values.get(id);
        if (result == null) throw new IllegalArgumentException("partition value is not projected: " + id);
        return result;
    }

    private static Layout layout(GraphValue value) {
        LayoutDescriptor descriptor = value.descriptor().layout().orElseThrow();
        if (descriptor.storageOffset() < 0
                || Arrays.stream(descriptor.strides()).anyMatch(stride -> stride < 0)) {
            throw new IllegalArgumentException("Conv2d requires non-negative resolved layouts");
        }
        return new Layout(value.descriptor().shape().toLongArray(), descriptor.storageOffset(),
                descriptor.strides());
    }

    private static CpuAccessPlan.Binding binding(Layout layout, CpuAccessPlan.AccessKind kind) {
        int suffix = 0;
        long expected = 1;
        for (int axis = layout.extents().length - 1; axis >= 0; axis--) {
            if (layout.strides()[axis] != expected) break;
            suffix++;
            expected = Math.multiplyExact(expected, Math.max(1, layout.extents()[axis]));
        }
        var roles = new ArrayList<CpuAccessPlan.AxisRole>();
        for (int axis = 0; axis < layout.extents().length; axis++) {
            roles.add(layout.strides()[axis] == 0 ? CpuAccessPlan.AxisRole.BROADCAST
                    : axis >= layout.extents().length - suffix
                    ? CpuAccessPlan.AxisRole.CONTIGUOUS : CpuAccessPlan.AxisRole.STRIDED);
        }
        var plan = new CpuAccessPlan(kind, suffix == layout.extents().length
                ? CpuAccessPlan.Regime.DENSE_LINEAR : CpuAccessPlan.Regime.GENERAL_ODOMETER,
                layout.extents().length, roles, suffix);
        long count = count(layout.extents());
        return CpuAccessPlan.Binding.create(plan, layout.extents(), layout.offset(),
                layout.strides(), count, 0, count, span(layout));
    }

    private static long count(long[] extents) {
        for (long extent : extents) if (extent == 0) return 0;
        long result = 1;
        for (long extent : extents) result = Math.multiplyExact(result, extent);
        return result;
    }

    private static long span(Layout layout) {
        if (count(layout.extents()) == 0) return 0;
        long maximum = layout.offset();
        for (int axis = 0; axis < layout.extents().length; axis++) {
            maximum = Math.addExact(maximum, Math.multiplyExact(layout.extents()[axis] - 1,
                    layout.strides()[axis]));
        }
        return Math.addExact(maximum, 1);
    }

    /**
     * Immutable cold layout geometry packed once per bound invocation.
     *
     * @param boundaries exact input, weight, optional bias, optional external ADD, and output
     *     layouts in generated-entry order; copied defensively
     */
    public record Geometry(List<Layout> boundaries) {
        /**
         * Snapshots exact input, weight, optional bias, and output layouts.
         *
         * @param boundaries exact boundary layouts in generated-entry order; copied defensively
         * @throws NullPointerException if the list or an element is {@code null}
         * @throws IllegalArgumentException if the boundary/rank shape is inconsistent
         */
        public Geometry {
            boundaries = List.copyOf(boundaries);
            if (boundaries.size() < 3 || boundaries.size() > 5
                    || boundaries.get(0).extents().length != 4
                    || boundaries.get(1).extents().length != 4
                    || boundaries.getLast().extents().length != 4
                    || boundaries.subList(2, boundaries.size() - 1).stream()
                    .anyMatch(layout -> layout.extents().length != 1
                            && layout.extents().length != 4)) {
                throw new IllegalArgumentException("Conv2d boundary geometry disagrees");
            }
        }

        /**
         * Packs carrier-relative bases followed by each boundary's extents and strides.
         * @param carrierBases non-null element bases in boundary order
         * @return a new non-null packed geometry array
         * @throws NullPointerException if {@code carrierBases} is {@code null}
         * @throws IllegalArgumentException if base cardinality disagrees
         * @throws ArithmeticException if adding a carrier base and layout offset overflows
         */
        public long[] pack(long[] carrierBases) {
            if (carrierBases.length != boundaries.size()) {
                throw new IllegalArgumentException("Conv2d carrier base count disagrees");
            }
            int length = boundaries.size();
            for (Layout layout : boundaries) length += 2 * layout.extents().length;
            long[] packed = new long[length];
            int cursor = boundaries.size();
            for (int i = 0; i < boundaries.size(); i++) {
                Layout layout = boundaries.get(i);
                packed[i] = Math.addExact(carrierBases[i], layout.offset());
                System.arraycopy(layout.extents(), 0, packed, cursor, layout.extents().length);
                cursor += layout.extents().length;
                System.arraycopy(layout.strides(), 0, packed, cursor, layout.strides().length);
                cursor += layout.strides().length;
            }
            return packed;
        }
    }

    /**
     * Exact immutable logical Shape and non-negative element layout.
     *
     * @param extents non-null logical extents; copied defensively
     * @param offset non-negative element offset within the selected carrier
     * @param strides non-null non-negative element strides; copied defensively
     */
    public record Layout(long[] extents, long offset, long[] strides) {
        /**
         * Snapshots one resolved layout without retaining its descriptor.
         *
         * @param extents non-null logical extents; copied defensively
         * @param offset non-negative element offset within the selected carrier
         * @param strides non-null non-negative element strides; copied defensively
         * @throws NullPointerException if an array is {@code null}
         * @throws IllegalArgumentException if ranks differ or a value is negative
         */
        public Layout {
            extents = extents.clone();
            strides = strides.clone();
            if (extents.length != strides.length || offset < 0
                    || Arrays.stream(extents).anyMatch(value -> value < 0)
                    || Arrays.stream(strides).anyMatch(value -> value < 0)) {
                throw new IllegalArgumentException("Conv2d layout is invalid");
            }
        }
        /** Returns a defensive snapshot of the logical Shape.
         * @return a new non-null extent array
         */
        @Override public long[] extents() { return extents.clone(); }

        /** Returns a defensive snapshot of the element strides.
         * @return a new non-null stride array
         */
        @Override public long[] strides() { return strides.clone(); }
    }
}
