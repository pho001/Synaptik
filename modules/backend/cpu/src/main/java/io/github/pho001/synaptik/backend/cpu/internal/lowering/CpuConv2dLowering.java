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
 * Lowers exact static or one-named-batch grouped NCHW Conv2d to complete-output-cell work.
 *
 * <p>Static occurrences retain the CPU capability boundary. Symbolic occurrences independently
 * check the original semantic descriptors and fixed convolution equations, then retain numeric
 * layout coefficients rather than sampled descriptors. Neither form requires scratch, packing,
 * im2col, partial, or combine resources.</p>
 */
public final class CpuConv2dLowering {
    private final CpuCapabilityProvider capabilities = new CpuCapabilityProvider();

    /** Creates a stateless direct-convolution lowerer. */
    public CpuConv2dLowering() { }

    /**
     * Lowers original symbolic NCHW descriptors without manufacturing a concrete batch graph.
     * @param context complete convolution and optional private ADD/RELU projection
     * @return unresolved numeric convolution recipe with independently checked epilogue
     * @throws IllegalArgumentException if original shapes, types, attributes or private uses disagree
     * @throws ArithmeticException if checked fixed coefficient arithmetic overflows
     */
    public CpuPartitionLowering.LoweredPartition lowerBatch(PrepareContext<? extends BackendAnalysisInputs> context) {
        return lowerBatch(context,CpuPartitionLowering.batchLayouts(context));
    }

    /**
     * Lowers original descriptors using the complete local graph's proved physical mappings.
     * The map is consumed only during lowering; the recipe retains numeric boundary facts only.
     * @param context complete original ordinary unit projection
     * @param layouts non-null full-local value mappings, preserving view strides and offsets
     * @return unresolved numeric family geometry with B-independent generated facts
     * @throws NullPointerException if a required mapping or input is absent
     * @throws IllegalArgumentException if original semantics or numeric mappings disagree
     * @throws ArithmeticException if fixed geometry arithmetic overflows
     */
    public CpuPartitionLowering.LoweredPartition lowerBatch(PrepareContext<? extends BackendAnalysisInputs> context,
            Map<ValueId,CpuAccessPlan.BatchBinding> layouts) {
        var values = batchValues(context);
        if (context.nodes().isEmpty() || context.nodes().size() > 3) throw new IllegalArgumentException("Conv2d node count disagrees");
        var node = context.nodes().getFirst();
        if (node.operation().kind() != Conv2dKind.CONV2D || !(node.operation().attrs() instanceof Conv2dAttrs attrs)
                || node.inputs().size() < 2 || node.inputs().size() > 3 || node.outputs().size() != 1
                || node.inputs().contains(node.outputs().getFirst())) throw new IllegalArgumentException("Conv2d occurrence disagrees");

        var input = require(values,node.inputs().get(0)); var weight = require(values,node.inputs().get(1));
        var convOutput = require(values,node.outputs().getFirst());
        if (input.descriptor().shape().rank() != 4 || weight.descriptor().shape().rank() != 4
                || convOutput.descriptor().shape().rank() != 4)
            throw new IllegalArgumentException("convolution ranks disagree");
        long channels = fixed(input,1), filters = fixed(weight,0);
        if (channels % attrs.groups() != 0 || filters % attrs.groups() != 0
                || Math.multiplyExact(fixed(weight,1),attrs.groups()) != channels)
            throw new IllegalArgumentException("convolution grouped channel equation disagrees");
        var promoted = io.github.pho001.synaptik.model.datatype.DataTypePromotion.promoteFloating(input.descriptor().dataType(),weight.descriptor().dataType());
        boolean gradient = input.descriptor().requiresGrad() || weight.descriptor().requiresGrad();
        if (node.inputs().size() == 3) {
            var bias = require(values,node.inputs().get(2));
            if (bias.descriptor().shape().rank() != 1 || fixed(bias,0) != filters)
                throw new IllegalArgumentException("intrinsic bias shape disagrees");
            promoted = io.github.pho001.synaptik.model.datatype.DataTypePromotion.promoteFloating(promoted,bias.descriptor().dataType());
            gradient |= bias.descriptor().requiresGrad();
        }
        if (convOutput.descriptor().dataType() != promoted || convOutput.descriptor().requiresGrad() != gradient)
            throw new IllegalArgumentException("convolution promoted type or gradient disagrees");
        sameAxis(input,0,convOutput,0);
        if (fixed(convOutput,1) != filters) throw new IllegalArgumentException("convolution output channels disagree");
        long[] strides = new long[]{attrs.strideHeight(),attrs.strideWidth()}, padding = new long[]{attrs.paddingHeight(),attrs.paddingWidth()}, dilation = new long[]{attrs.dilationHeight(),attrs.dilationWidth()};
        for (int axis = 2; axis < 4; axis++) {
            long kernel = fixed(weight,axis);
            if (kernel <= 0) throw new IllegalArgumentException("convolution kernel must be positive");
            long effective = Math.addExact(Math.multiplyExact(kernel-1,dilation[axis-2]),1);
            long padded = Math.addExact(fixed(input,axis),Math.multiplyExact(2,padding[axis-2]));
            if (padded < effective || fixed(convOutput,axis) != Math.addExact((padded-effective)/strides[axis-2],1))
                throw new IllegalArgumentException("convolution original spatial equation disagrees");
        }

        var ids = new ArrayList<ValueId>(node.inputs()); var virtual = new ArrayList<ValueId>();
        ValueId previous = convOutput.id(); ValueId external = null;
        CpuConv2dIr.Epilogue epilogue = CpuConv2dIr.Epilogue.NONE;
        var memory = new LinkedHashMap<ValueId,LogicalMemoryRequirement>();
        context.memoryRequirements().forEach(r -> memory.put(r.valueId(),r));
        if (context.nodes().size() >= 2) {
            var add = context.nodes().get(1);
            if (add.operation().kind() != BinaryArithmeticKind.ADD || add.operation().attrs() != NoOperationAttrs.INSTANCE
                    || add.inputs().size() != 2 || add.outputs().size() != 1
                    || add.inputs().stream().filter(previous::equals).count() != 1)
                throw new IllegalArgumentException("Conv2d suffix must begin with direct ADD");
            external = add.inputs().get(0).equals(previous) ? add.inputs().get(1) : add.inputs().get(0);
            var source = require(values,external); var sum = require(values,add.outputs().getFirst());
            if (source.descriptor().dataType() != promoted || sum.descriptor().dataType() != promoted
                    || !sum.descriptor().shape().equals(convOutput.descriptor().shape())
                    || sum.descriptor().requiresGrad() != (gradient || source.descriptor().requiresGrad())
                    || add.inputs().contains(sum.id())) throw new IllegalArgumentException("Conv2d ADD facts disagree");
            requirePrivate(memory.get(previous),context,previous); virtual.add(previous);
            previous = sum.id(); epilogue = CpuConv2dIr.Epilogue.ADD;
        }
        if (context.nodes().size() == 3) {
            var relu = context.nodes().get(2); var prior = require(values,previous);
            if (relu.operation().kind() != UnaryElementwiseKind.RELU || relu.operation().attrs() != NoOperationAttrs.INSTANCE
                    || !relu.inputs().equals(List.of(previous)) || relu.outputs().size() != 1)
                throw new IllegalArgumentException("Conv2d suffix requires exact RELU");
            var result = require(values,relu.outputs().getFirst());
            if (!result.descriptor().shape().equals(prior.descriptor().shape())
                    || result.descriptor().dataType() != promoted || result.descriptor().requiresGrad() != prior.descriptor().requiresGrad()
                    || result.id().equals(previous)) throw new IllegalArgumentException("Conv2d RELU facts disagree");
            requirePrivate(memory.get(previous),context,previous); virtual.add(previous);
            previous = result.id(); epilogue = CpuConv2dIr.Epilogue.ADD_RELU;
        }
        if (external != null) ids.add(external); ids.add(previous);
        var bindings = new ArrayList<CpuAccessPlan.BatchBinding>();
        for (var id : ids) bindings.add(batchBinding(require(values,id),layouts));
        if (external != null) {
            var source = require(values,external); var target = require(values,previous);
            int shift = target.descriptor().shape().rank() - source.descriptor().shape().rank();
            if (shift < 0) throw new IllegalArgumentException("Conv2d ADD source rank is too large");
            var original = bindings.get(bindings.size()-2); var out = bindings.getLast();
            var sc = new ArrayList<Long>(); var sf = new ArrayList<Long>();
            for (int axis = 0; axis < 4; axis++) {
                int from = axis-shift;
                boolean singleton = from < 0 || original.extentConstants().get(from) == 1 && original.extentBatchFactors().get(from) == 0;
                if (!singleton) sameAxis(source,from,target,axis);
                sc.add(singleton ? 0L : original.strideConstants().get(from));
                sf.add(singleton ? 0L : original.strideBatchFactors().get(from));
            }
            var mapped = CpuPartitionLowering.batchAccess(CpuAccessPlan.AccessKind.READ,out.extentConstants(),out.extentBatchFactors(),sc,sf);
            bindings.set(bindings.size()-2,new CpuAccessPlan.BatchBinding(mapped.plan(),mapped.extentConstants(),mapped.extentBatchFactors(),original.baseElementOffset(),sc,sf));
        }
        bindings.set(bindings.size()-1,writeRecipe(bindings.getLast()));
        var types = ids.stream().map(id -> require(values,id).descriptor().dataType()).toList();
        var ir = new CpuConv2dIr(types.subList(0,types.size()-1),types.getLast(),attrs.strideHeight(),attrs.strideWidth(),
                attrs.paddingHeight(),attrs.paddingWidth(),attrs.dilationHeight(),attrs.dilationWidth(),attrs.groups(),1,
                node.inputs().size()==3,epilogue,bindings.subList(0,bindings.size()-1).stream().map(CpuAccessPlan.BatchBinding::plan).toList(),writePlan(bindings.getLast()));
        return new CpuPartitionLowering.LoweredPartition(new Geometry.BatchGeometry(ir,ids,types,bindings,virtual,new long[0]));
    }

    private static Map<ValueId, GraphValue> batchValues(PrepareContext<?> context) {
        CpuPartitionLowering.requireBatchContext(context);
        var values = new LinkedHashMap<ValueId, GraphValue>();
        context.values().forEach(value -> values.put(value.id(),value));
        return values;
    }

    private static long fixed(GraphValue value, int axis) {
        var dimension = value.descriptor().shape().dimensions().get(axis);
        if (!(dimension instanceof io.github.pho001.synaptik.model.shape.StaticDimension fixed))
            throw new IllegalArgumentException("window and feature dimensions must be fixed");
        return fixed.size();
    }

    private static void sameAxis(GraphValue left, int a, GraphValue right, int b) {
        if (!left.descriptor().shape().dimensions().get(a).equals(
                right.descriptor().shape().dimensions().get(b)))
            throw new IllegalArgumentException("symbolic dimension equation disagrees");
    }

    private static CpuAccessPlan.BatchBinding batchBinding(GraphValue value,
            Map<ValueId,CpuAccessPlan.BatchBinding> layouts) {
        var mapping = Objects.requireNonNull(layouts.get(value.id()), "missing full-local layout");
        var canonical = CpuPartitionLowering.batchCanonical(value.descriptor(),CpuAccessPlan.AccessKind.READ);
        if (!mapping.extentConstants().equals(canonical.extentConstants())
                || !mapping.extentBatchFactors().equals(canonical.extentBatchFactors()))
            throw new IllegalArgumentException("full-local mapping disagrees with original symbolic shape");
        var p = mapping.plan();
        var read = new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,p.regime(),p.iterationRank(),p.axisRoles(),p.contiguousSuffix());
        return new CpuAccessPlan.BatchBinding(read,mapping.extentConstants(),mapping.extentBatchFactors(),
                mapping.baseElementOffset(),mapping.strideConstants(),mapping.strideBatchFactors());
    }

    private static CpuAccessPlan writePlan(CpuAccessPlan.BatchBinding binding) {
        var p = binding.plan();
        return new CpuAccessPlan(CpuAccessPlan.AccessKind.WRITE, p.regime(), p.iterationRank(),
                p.axisRoles(), p.contiguousSuffix());
    }

    private static CpuAccessPlan.BatchBinding writeRecipe(CpuAccessPlan.BatchBinding binding) {
        return new CpuAccessPlan.BatchBinding(writePlan(binding),binding.extentConstants(),
                binding.extentBatchFactors(),binding.baseElementOffset(),binding.strideConstants(),
                binding.strideBatchFactors());
    }

    private static String batchIdentity(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPortableKernelIr ir,
            List<CpuAccessPlan.BatchBinding> bindings, long[] parameters) {
        var key = new StringBuilder(ir.structuralKey());
        for (var b : bindings) key.append('|').append(b.extentConstants()).append(b.extentBatchFactors())
                .append(b.baseElementOffset()).append(b.strideConstants()).append(b.strideBatchFactors());
        return key.append('|').append(Arrays.toString(parameters)).toString();
    }

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
            var activationQuery = new OperationCapabilityQuery(activation.operation(), activation.inputs().stream().map(id -> require(values, id).descriptor()).toList(), activation.outputs().stream().map(id -> require(values, id).descriptor()).toList());
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
         * Unresolved numeric invocation recipe. No representative shape, graph or invocation is retained.
         * @param ir exact B-independent generated facts
         * @param boundaries ordered physical boundaries
         * @param boundaryTypes exact boundary storage types
         * @param bindings affine numeric mappings, including the output mapping
         * @param virtualValues private removed intermediate values
         * @param parameters fixed family geometry, defensively copied

         */
        public record BatchGeometry(CpuConv2dIr ir, List<ValueId> boundaries,
                List<io.github.pho001.synaptik.model.datatype.DataType> boundaryTypes,
                List<CpuAccessPlan.BatchBinding> bindings, List<ValueId> virtualValues,
                long[] parameters) implements CpuPartitionLowering.BatchUnitGeometry {
            /**
             * Snapshots immutable compile facts and checked numeric coefficients.
             * @param ir B-independent generated facts
             * @param boundaries ordered physical boundary identities
             * @param boundaryTypes physical storage types in boundary order
             * @param bindings affine layout coefficients in boundary order
             * @param virtualValues private removed intermediates
             * @param parameters fixed family-specific numeric geometry
             */
            public BatchGeometry {
                Objects.requireNonNull(ir, "ir");
                boundaries = List.copyOf(boundaries); boundaryTypes = List.copyOf(boundaryTypes);
                bindings = List.copyOf(bindings); virtualValues = List.copyOf(virtualValues);
                parameters = parameters.clone();
            }
            /**
             * Returns the fixed Conv2d geometry cells retained for numeric invocation binding.
             * Actual B is resolved separately through the affine boundary mappings.
             * Mutating the result cannot alter this graph-free recipe.
             * @return a new non-null caller-owned copy of the fixed numeric geometry
             */
            @Override public long[] parameters() { return parameters.clone(); }
            /** @return checked affine complete-unit work count, not bytes */
            @Override public io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize iterationCount() { return CpuPartitionLowering.batchElements(bindings.getLast()); }
            /** @return simultaneous direct-scalar workspace bytes */
            @Override public io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize scratchBytes() { return new io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize(0, 0, 1); }
            /** @return structural identity independent of actual B and value identities */
            @Override public String identity() { return batchIdentity(ir, bindings, parameters); }
            /** @return no mutable packed traversal cursors for this ordinary family */
            @Override public List<io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.PackReset> resets() { return List.of(); }
            /**
             * Resolves only numeric geometry on the cold invocation boundary.
             * @param batch non-negative int-representable actual B
             * @return ordinary resolved family geometry with identical generated facts
             * @throws IllegalArgumentException if batch is negative
             * @throws ArithmeticException if geometry exceeds checked host limits
             */
            @Override public CpuPartitionLowering.LoweredPartition bind(long batch) {
                var resolved = new ArrayList<CpuAccessPlan.Binding>();
                for (var binding : bindings) resolved.add(binding.bind(batch));
                var out = resolved.getLast();
                var geometry = new Geometry(resolved.stream().map(b -> new Layout(b.extents().stream().mapToLong(Long::longValue).toArray(), b.baseElementOffset(), b.effectiveStrides().stream().mapToLong(Long::longValue).toArray())).toList());
                return new CpuPartitionLowering.LoweredPartition(ir, boundaries, resolved,
                        resolved.stream().map(CpuAccessPlan.Binding::referencedElementSpan).toList(),
                        boundaryTypes, virtualValues, out.extents().stream().mapToLong(Long::longValue).toArray(), out.elementCount(),
                        "legal: checked symbolic ordinary geometry", new long[0],
                        Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.of(geometry));
            }
        }

        /**
         * One fixed generated assignment of a boundary's element base plus a layout delta.
         * @param cell non-negative destination cell in this family's packed geometry
         * @param boundary boundary position in the pack's carrier-base order
         * @param elementDelta immutable element offset added with checked arithmetic
         */
        public record BaseCell(int cell, int boundary, long elementDelta) { }

        /**
         * One original cursor seed restored before every temporal helper call, including retry.
         * @param cell non-negative destination cell in this family's packed geometry
         * @param value original range-start cursor value
         */
        public record ResetCell(int cell, long value) { }

        /**
         * Cold, immutable facts for emitting fixed numeric assignments, never a runtime program.
         * Scratch range offsets and invariant geometry remain bound once per invocation.
         * @param bases non-null boundary-base assignments, defensively snapshotted
         * @param resets non-null mutable-cursor assignments, defensively snapshotted
         */
        public record TemporalRebind(List<BaseCell> bases, List<ResetCell> resets) {
            /**
             * Snapshots the generation-time assignments; retains no invocation storage.
             * @param bases non-null base facts without null entries
             * @param resets non-null reset facts without null entries
             * @throws NullPointerException if either list or an entry is null
             */
            public TemporalRebind {
                bases = List.copyOf(bases);
                resets = List.copyOf(resets);
            }
        }

        /**
         * Describes temporal base rebinding and restoration of this family's original cursors.
         * Construct these facts only during generation or cold numeric binding; generated code
         * emits the assignments directly, without calling pack, allocating, or iterating facts.
         * Each boundary base is normalized using that boundary's own element width beforehand.
         * @param start non-negative inclusive ordinary helper range start
         * @return immutable owner-defined base locations and original cursor seeds
         * @throws IllegalArgumentException if start is negative
         * @throws ArithmeticException if checked layout or cursor arithmetic overflows
         */
        public TemporalRebind temporalRebind(long start) {
            if (start < 0) throw new IllegalArgumentException("negative temporal range start");
            var bases = new ArrayList<BaseCell>(boundaries.size());
            for (int i = 0; i < boundaries.size(); i++)
                bases.add(new BaseCell(i, i, boundaries.get(i).offset()));
            return new TemporalRebind(bases, List.of());
        }

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
