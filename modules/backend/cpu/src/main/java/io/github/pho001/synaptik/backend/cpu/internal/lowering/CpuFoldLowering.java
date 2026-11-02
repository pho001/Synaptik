package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFoldIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.layout.Fold2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.FoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.WindowTransformKind;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.ContiguousKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Lowers static or one-named-batch FOLD_AXIS/FOLD2D with fixed windows, or proves a bounded
 * private static F32/F64 raw kernel-repetition chain feeding compact storage to FOLD2D.
 * The lowerer revalidates the Model signature, derives compact checked coordinate geometry, and
 * proves a distinct injective output. It allocates no Runtime resource and declares no workspace.
 */
public final class CpuFoldLowering {
    private final CpuCapabilityProvider capabilities = new CpuCapabilityProvider();

    /** Creates a stateless fold lowerer. */
    public CpuFoldLowering() { }

    /**
     * Validates original symbolic overlap-fold equations and records numeric mapping parameters.
     * @param context complete one-node fold projection with fixed folded/window axes
     * @return explicit unresolved numeric fold geometry
     * @throws IllegalArgumentException if original types, ranks or dimension equations disagree
     * @throws ArithmeticException if fixed window geometry overflows
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
        if (context.nodes().size() != 1) throw new IllegalArgumentException("fold requires one node");
        var node = context.nodes().getFirst();
        if (node.inputs().size() != 1 || node.outputs().size() != 1 || node.inputs().contains(node.outputs().getFirst()))
            throw new IllegalArgumentException("fold boundary arity or alias disagrees");
        var input = require(values,node.inputs().getFirst()); var output = require(values,node.outputs().getFirst());
        var type = input.descriptor().dataType();
        if (type != output.descriptor().dataType() || type == DataType.BOOL)
            throw new IllegalArgumentException("fold original types disagree");
        int inRank = input.descriptor().shape().rank(), outRank = output.descriptor().shape().rank();
        CpuFoldIr.Family family; long[] parameters;
        if (node.operation().kind() == WindowTransformKind.FOLD_AXIS && node.operation().attrs() instanceof FoldAxisAttrs a) {
            family = CpuFoldIr.Family.FOLD_AXIS;
            if (inRank < 2 || outRank != inRank-1 || a.axis() < 0 || a.axis() >= outRank
                    || fixed(output,a.axis()) != a.outputSize()) throw new IllegalArgumentException("fold axis rank/target disagrees");
            for (int axis = 0; axis < outRank; axis++) if (axis != a.axis()) sameAxis(input,axis,output,axis);
            long window = fixed(input,inRank-1), windows = fixed(input,a.axis());
            if (window <= 0 || a.outputSize() != 0 && window > a.outputSize())
                throw new IllegalArgumentException("fold window is invalid");
            long expected = a.outputSize() == 0 ? 0 : Math.addExact((a.outputSize()-window)/a.step(),1);
            if (windows != expected) throw new IllegalArgumentException("fold original window-count equation disagrees");
            if (expected != 0) Math.addExact(Math.multiplyExact(expected-1,a.step()),window-1);
            parameters = new long[]{a.axis(),a.outputSize(),a.step(),window};
        } else if (node.operation().kind() == WindowTransformKind.FOLD2D && node.operation().attrs() instanceof Fold2dAttrs a) {
            family = CpuFoldIr.Family.FOLD2D;
            if (inRank != 3 || outRank != 4 || !output.descriptor().shape().equals(a.outputShape())
                    || type != DataType.FLOAT32 && type != DataType.FLOAT64)
                throw new IllegalArgumentException("Fold2d original shape/type disagrees");
            var w = a.window();
            long eh = Math.addExact(Math.multiplyExact(w.dilationHeight(),w.kernelHeight()-1),1);
            long ew = Math.addExact(Math.multiplyExact(w.dilationWidth(),w.kernelWidth()-1),1);
            long oh = windowCount(fixed(output,2),w.paddingHeight(),eh,w.strideHeight(),w.ceilMode());
            long ow = windowCount(fixed(output,3),w.paddingWidth(),ew,w.strideWidth(),w.ceilMode());
            proveTwoDimensionalCoordinates(w,oh,ow); sameAxis(input,0,output,0);
            if (fixed(input,1) != Math.multiplyExact(Math.multiplyExact(fixed(output,1),w.kernelHeight()),w.kernelWidth())
                    || fixed(input,2) != Math.multiplyExact(oh,ow)) throw new IllegalArgumentException("Fold2d original columns equation disagrees");
            parameters = new long[]{w.kernelHeight(),w.kernelWidth(),w.strideHeight(),w.strideWidth(),w.paddingHeight(),w.paddingWidth(),w.dilationHeight(),w.dilationWidth(),oh,ow};
        } else throw new IllegalArgumentException("unsupported fold kind/attributes");
        var in = batchBinding(input,layouts); var out = writeRecipe(batchBinding(output,layouts));
        var ir = new CpuFoldIr(family,type,in.plan(),writePlan(out),CpuFoldIr.CANONICAL_SEQUENTIAL_ADDITION);
        return new CpuPartitionLowering.LoweredPartition(new Geometry.BatchGeometry(ir,List.of(input.id(),output.id()),
                List.of(type,type),List.of(in,out),List.of(),parameters));
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
     * Proves a private same-type kernel repetition chain ending in FOLD2D. The physical source
     * remains a real stored boundary; DIV and all other arithmetic are never absorbed. Complete
     * projected publication/outside-use obligations protect every removed mapping.
     * @param context non-null static unit projection, with full-graph barriers already projected
     * @param <I> exact immutable backend-input projection type
     * @return compact lowering, or empty when any shape, use, layout or arithmetic proof fails
     * @throws NullPointerException if {@code context} is null
     */
    public <I extends BackendAnalysisInputs> Optional<CpuPartitionLowering.LoweredPartition> kernelBroadcast(
            PrepareContext<I> context) {
        var nodes = context.nodes();
        if (context.requiresBatchSize() || nodes.size() < 4 || nodes.size() > 8
                || nodes.getLast().operation().kind() != WindowTransformKind.FOLD2D) return Optional.empty();
        if (context.backendInputs() instanceof io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs cpu
                && (cpu.portableExecution().computePreference()
                    != io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.SCALAR
                    || cpu.portableExecution().configuredMaximumParallelism() != 1)) return Optional.empty();
        try {
            var values = new LinkedHashMap<ValueId, GraphValue>();
            context.values().forEach(v -> values.put(v.id(), v));
            var fold = lower(CpuPartitionDagDecomposer.project(context, List.of(nodes.getLast()), context.backendInputs()));
            var old = fold.foldGeometry().orElseThrow();
            if (old.dataType != DataType.FLOAT32 && old.dataType != DataType.FLOAT64) return Optional.empty();
            var two = (TwoDimensionalGeometry) old.mapping;
            long k = Math.multiplyExact(two.kernelHeight, two.kernelWidth);
            if (k <= 1) return Optional.empty();
            long[] out = old.output.extents;
            long p = Math.multiplyExact(two.outputColumnsHeight, two.outputColumnsWidth);
            long[] compact = {out[0], out[1], p};
            long[] singleton = {out[0], out[1], 1, p};
            long[] repeated = {out[0], out[1], k, p};
            long[] columns = {out[0], Math.multiplyExact(out[1], k), p};
            ValueId source = nodes.getFirst().inputs().getFirst(), previous = source;
            var input = require(values, source);
            Layout physical = layout(input);
            var read = compactBinding(physical);
            if (!Arrays.equals(physical.extents, compact)
                    || read.plan().regime() != CpuAccessPlan.Regime.DENSE_LINEAR) return Optional.empty();
            var memory = new LinkedHashMap<ValueId, io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement>();
            context.memoryRequirements().forEach(r -> memory.put(r.valueId(), r));
            int stage = 0;
            for (int i = 0; i < nodes.size() - 1; i++) {
                var node = nodes.get(i);
                if (node.inputs().size() != 1 || node.outputs().size() != 1
                        || !node.inputs().getFirst().equals(previous)) return Optional.empty();
                var from = require(values, previous); var to = require(values, node.outputs().getFirst());
                if (from.descriptor().dataType() != old.dataType || to.descriptor().dataType() != old.dataType
                        || !from.descriptor().shape().isFullyStatic() || !to.descriptor().shape().isFullyStatic()
                        || !capabilities.supports(new OperationCapabilityQuery(node.operation(),
                            List.of(from.descriptor()), List.of(to.descriptor())))) return Optional.empty();
                for (var value : List.of(from, to)) Math.multiplyExact(
                        binding(layout(value), CpuAccessPlan.AccessKind.READ).referencedElementSpan(), old.dataType.byteWidth());
                long[] shape = to.descriptor().shape().toLongArray();
                Object kind = node.operation().kind();
                if (kind == ContiguousKind.CONTIGUOUS
                        || kind == ShapeTransformKind.RESHAPE && from.descriptor().shape().equals(to.descriptor().shape())) {
                    if (!from.descriptor().shape().equals(to.descriptor().shape())) return Optional.empty();
                } else if (stage == 0 && (kind == AxisTransformKind.EXPAND_DIMS
                        && ((AxisTransformAttrs) node.operation().attrs()).axis() == 2
                        || kind == ShapeTransformKind.RESHAPE) && Arrays.equals(shape, singleton)) stage = 1;
                else if (stage == 1 && kind == ShapeTransformKind.EXPAND && Arrays.equals(shape, repeated)) stage = 2;
                else if (stage == 2 && kind == ShapeTransformKind.RESHAPE && Arrays.equals(shape, columns)
                        && from.descriptor().layout().orElseThrow().isContiguous()) stage = 3;
                else return Optional.empty();
                var obligation = memory.get(to.id()); var uses = context.partitionDag().consumers(to.id());
                if (obligation == null || obligation.graphOutput()
                        || obligation.consumerPartitions().stream().anyMatch(part -> !part.equals(context.partition()))
                        || uses.size() != 1 || uses.getFirst().node() != nodes.get(i + 1)) return Optional.empty();
                previous = to.id();
            }
            if (stage != 3 || !nodes.getLast().inputs().equals(List.of(previous))) return Optional.empty();
            // Prove both extrema of y+pad-ky*dilation and x+pad-kx*dilation before emission.
            Math.addExact(Math.max(0, out[2] - 1), two.paddingHeight);
            Math.addExact(Math.max(0, out[3] - 1), two.paddingWidth);
            Math.multiplyExact(two.kernelHeight - 1, two.dilationHeight);
            Math.multiplyExact(two.kernelWidth - 1, two.dilationWidth);
            var geometry = new Geometry(old.family, old.dataType, physical, old.output, old.mapping, true);
            var ir = new CpuFoldIr(old.family, old.dataType, read.plan(), fold.accessBindings().getLast().plan(),
                    CpuFoldIr.CANONICAL_SEQUENTIAL_ADDITION, true);
            return Optional.of(new CpuPartitionLowering.LoweredPartition(ir,
                    List.of(source, fold.boundaryValues().getLast()), List.of(read, fold.accessBindings().getLast()),
                    List.of(read.referencedElementSpan(), fold.referencedElementSpans().getLast()),
                    List.of(old.dataType, old.dataType), nodes.subList(0, nodes.size() - 1).stream()
                        .map(n -> n.outputs().getFirst()).toList(), out, fold.elementCount(),
                    "legal: private raw kernel repetition; compact stored fold input", new long[0],
                    Optional.empty(), Optional.empty(), Optional.empty(), Optional.of(geometry), Optional.empty()));
        } catch (IllegalArgumentException | ArithmeticException unproved) { return Optional.empty(); }
    }

    /**
     * Lowers one supported current fold occurrence.
     *
     * @param context non-null complete CPU partition projection
     * @return immutable single-unit lowering with compact fold geometry
     * @throws NullPointerException if {@code context} is {@code null}
     * @throws IllegalArgumentException if the occurrence, Shape, layout, or boundary is unsupported
     * @throws ArithmeticException if exact geometry or address arithmetic overflows
     */
    public CpuPartitionLowering.LoweredPartition lower(
            PrepareContext<? extends BackendAnalysisInputs> context) {
        Objects.requireNonNull(context, "context");
        if (context.nodes().size() != 1) {
            throw new IllegalArgumentException("CPU fold partition requires exactly one node");
        }
        var node = context.nodes().getFirst();
        Map<ValueId, GraphValue> values = new LinkedHashMap<>();
        context.values().forEach(value -> values.put(value.id(), value));
        var query = new OperationCapabilityQuery(node.operation(), node.inputs().stream()
                .map(id -> require(values, id).descriptor()).toList(), node.outputs().stream()
        .map(id -> require(values, id).descriptor()).toList());
        if (!capabilities.supports(query)) {
            throw new IllegalArgumentException("partition contains an unsupported CPU fold occurrence");
        }
        ValueId inputId = node.inputs().getFirst();
        ValueId outputId = node.outputs().getFirst();
        if (inputId.equals(outputId)) throw new IllegalArgumentException(
                "fold output must be distinct from its input");
        GraphValue input = require(values, inputId);
        GraphValue output = require(values, outputId);
        Layout inputLayout = layout(input);
        Layout outputLayout = layout(output);
        validateInjective(outputLayout.extents, outputLayout.strides);
        var inputBinding = binding(inputLayout, CpuAccessPlan.AccessKind.READ);
        var outputBinding = binding(outputLayout, CpuAccessPlan.AccessKind.WRITE);
        DataType type = input.descriptor().dataType();
        Object kind = node.operation().kind();
        CpuFoldIr.Family family;
        Mapping mapping;
        if (kind == WindowTransformKind.FOLD_AXIS) {
            FoldAxisAttrs attrs = (FoldAxisAttrs) node.operation().attrs();
            family = CpuFoldIr.Family.FOLD_AXIS;
            long[] in = inputLayout.extents;
            mapping = new AxisGeometry(attrs.axis(), attrs.outputSize(), attrs.step(),
                    in[in.length - 1]);
        } else if (kind == WindowTransformKind.FOLD2D) {
            Fold2dAttrs attrs = (Fold2dAttrs) node.operation().attrs();
            var window = attrs.window();
            long[] out = outputLayout.extents;
            long effectiveHeight = Math.addExact(
                    Math.multiplyExact(window.dilationHeight(), window.kernelHeight() - 1), 1);
            long effectiveWidth = Math.addExact(
                    Math.multiplyExact(window.dilationWidth(), window.kernelWidth() - 1), 1);
            long oh = windowCount(out[2], window.paddingHeight(), effectiveHeight,
                    window.strideHeight(), window.ceilMode());
            long ow = windowCount(out[3], window.paddingWidth(), effectiveWidth,
                    window.strideWidth(), window.ceilMode());
            proveTwoDimensionalCoordinates(window, oh, ow);
            family = CpuFoldIr.Family.FOLD2D;
            mapping = new TwoDimensionalGeometry(window.kernelHeight(), window.kernelWidth(),
                    window.strideHeight(), window.strideWidth(), window.paddingHeight(),
                    window.paddingWidth(), window.dilationHeight(), window.dilationWidth(), oh, ow);
        } else {
            throw new IllegalArgumentException("unsupported fold family");
        }
        var ir = new CpuFoldIr(family, type, inputBinding.plan(), outputBinding.plan(),
                CpuFoldIr.CANONICAL_SEQUENTIAL_ADDITION);
        var geometry = new Geometry(family, type, inputLayout, outputLayout, mapping);
        long outputCount = elementCount(outputLayout.extents);
        return new CpuPartitionLowering.LoweredPartition(ir, List.of(inputId, outputId),
                List.of(inputBinding, outputBinding),
                List.of(input.descriptor().layout().orElseThrow().referencedElementSpan(),
                        output.descriptor().layout().orElseThrow().referencedElementSpan()),
                List.of(type, type), List.of(), outputLayout.extents, outputCount,
                "legal: one fully static resolved-layout overlap fold occurrence", new long[0],
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.of(geometry),
                Optional.empty());
    }

    private static long windowCount(long size, long padding, long effective, long stride,
            boolean ceil) {
        long numerator = Math.subtractExact(Math.addExact(size, Math.multiplyExact(2, padding)), effective);
        if (numerator < 0) throw new IllegalArgumentException("fold2d effective kernel exceeds padded output");
        long quotient = ceil ? Math.floorDiv(Math.addExact(numerator, stride - 1), stride)
                : Math.floorDiv(numerator, stride);
        return Math.addExact(quotient, 1);
    }

    private static void proveTwoDimensionalCoordinates(
            io.github.pho001.synaptik.model.operation.layout.Window2dAttrs window,
            long outputHeight, long outputWidth) {
        if (outputHeight > 0) Math.addExact(Math.subtractExact(
                Math.multiplyExact(outputHeight - 1, window.strideHeight()),
                window.paddingHeight()),
                Math.multiplyExact(window.kernelHeight() - 1, window.dilationHeight()));
        if (outputWidth > 0) Math.addExact(Math.subtractExact(
                Math.multiplyExact(outputWidth - 1, window.strideWidth()),
                window.paddingWidth()),
                Math.multiplyExact(window.kernelWidth() - 1, window.dilationWidth()));
    }

    private static GraphValue require(Map<ValueId, GraphValue> values, ValueId id) {
        GraphValue value = values.get(id);
        if (value == null) throw new IllegalArgumentException("partition value is not projected: " + id);
        return value;
    }

    private static Layout layout(GraphValue value) {
        long[] extents = value.descriptor().shape().toLongArray();
        LayoutDescriptor layout = value.descriptor().layout().orElseThrow();
        if (layout.storageOffset() < 0 || Arrays.stream(layout.strides()).anyMatch(v -> v < 0)) {
            throw new IllegalArgumentException("fold requires non-negative resolved layouts");
        }
        return new Layout(extents, layout.storageOffset(), layout.strides());
    }

    private static CpuAccessPlan.Binding binding(Layout layout, CpuAccessPlan.AccessKind kind) {
        int suffix = 0; long expected = 1;
        for (int axis = layout.extents.length - 1; axis >= 0; axis--) {
            if (layout.strides[axis] != expected) break;
            suffix++; expected = Math.multiplyExact(expected, Math.max(1, layout.extents[axis]));
        }
        var roles = new ArrayList<CpuAccessPlan.AxisRole>();
        for (int axis = 0; axis < layout.extents.length; axis++) roles.add(layout.strides[axis] == 0
                ? CpuAccessPlan.AxisRole.BROADCAST
                : axis >= layout.extents.length - suffix ? CpuAccessPlan.AxisRole.CONTIGUOUS
                : CpuAccessPlan.AxisRole.STRIDED);
        var plan = new CpuAccessPlan(kind, suffix == layout.extents.length
                ? CpuAccessPlan.Regime.DENSE_LINEAR : CpuAccessPlan.Regime.GENERAL_ODOMETER,
                layout.extents.length, roles, suffix);
        long count = elementCount(layout.extents);
        return CpuAccessPlan.Binding.create(plan, layout.extents, layout.offset, layout.strides,
                count, 0, count, referencedSpan(layout));
    }

    /**
     * Binds only canonical physical compact storage, including existing empty canonical domains.
     * Zero dimensions may give earlier canonical strides zero; no input is read in those domains.
     * This selected-only proof leaves ordinary fold access identity unchanged.
     * @param layout non-null resolved physical rank-three layout with nonnegative offset
     * @return immutable dense read binding for the physical span
     * @throws IllegalArgumentException if strides are not canonical or the rank is not three
     * @throws ArithmeticException if span or logical count arithmetic overflows
     */
    private static CpuAccessPlan.Binding compactBinding(Layout layout) {
        if (layout.extents.length != 3 || !Arrays.equals(layout.strides,
                LayoutDescriptor.contiguous(io.github.pho001.synaptik.model.shape.Shape.of(layout.extents)).strides())) {
            throw new IllegalArgumentException("compact fold source must be canonical dense storage");
        }
        var plan = new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,CpuAccessPlan.Regime.DENSE_LINEAR,
                3,List.of(CpuAccessPlan.AxisRole.CONTIGUOUS,CpuAccessPlan.AxisRole.CONTIGUOUS,CpuAccessPlan.AxisRole.CONTIGUOUS),3);
        long count = elementCount(layout.extents);
        return CpuAccessPlan.Binding.create(plan,layout.extents,layout.offset,layout.strides,count,0,count,referencedSpan(layout));
    }

    private static long referencedSpan(Layout layout) {
        if (elementCount(layout.extents) == 0) return 0;
        long maximum = 0;
        for (int i = 0; i < layout.extents.length; i++) maximum = Math.addExact(maximum,
                Math.multiplyExact(layout.extents[i] - 1, layout.strides[i]));
        return Math.addExact(layout.offset, Math.addExact(maximum, 1));
    }

    private static long elementCount(long[] extents) {
        if (Arrays.stream(extents).anyMatch(v -> v == 0)) return 0;
        long count = 1;
        for (long extent : extents) count = Math.multiplyExact(count, extent);
        return count;
    }

    private static void validateInjective(long[] extents, long[] strides) {
        if (elementCount(extents) == 0) return;
        var axes = new ArrayList<Integer>();
        for (int i = 0; i < extents.length; i++) if (extents[i] > 1) axes.add(i);
        axes.sort(Comparator.comparingLong(i -> strides[i]));
        long covered = 1;
        for (int axis : axes) {
            if (strides[axis] < covered) {
                if (elementCount(extents) > 1_000_000) throw new IllegalArgumentException(
                        "fold output layout is not injective");
                var seen = new HashSet<Long>();
                long[] coordinate = new long[extents.length];
                for (long ordinal = 0; ordinal < elementCount(extents); ordinal++) {
                    long address = 0;
                    for (int i = 0; i < coordinate.length; i++) address = Math.addExact(address,
                            Math.multiplyExact(coordinate[i], strides[i]));
                    if (!seen.add(address)) throw new IllegalArgumentException(
                            "fold output layout is not injective");
                    advance(coordinate, extents);
                }
                return;
            }
            covered = Math.addExact(covered, Math.multiplyExact(extents[axis] - 1, strides[axis]));
        }
    }

    private static void advance(long[] coordinate, long[] extents) {
        for (int i = coordinate.length - 1; i >= 0; i--) {
            if (++coordinate[i] < extents[i]) return;
            coordinate[i] = 0;
        }
    }

    /** Marks one immutable, already validated family-specific fold coordinate mapping. */
    public sealed interface Mapping permits AxisGeometry, TwoDimensionalGeometry { }

    /**
     * Compact general-axis fold geometry.
     * @param axis normalized target axis
     * @param outputSize restored target extent
     * @param step positive window step
     * @param windowSize positive final input extent
     */
    public record AxisGeometry(int axis, long outputSize, long step, long windowSize)
            implements Mapping {
        /**
         * Creates checked cold geometry for one general-axis fold.
         *
         * @param axis normalized non-negative axis in the restored target rank
         * @param outputSize non-negative restored target extent
         * @param step positive distance between consecutive window starts
         * @param windowSize positive final input extent interpreted as window size
         * @throws IllegalArgumentException if an axis or extent is negative, or if a step or
         *     window size is not positive
         */
        public AxisGeometry {
            if (axis < 0 || outputSize < 0 || step <= 0 || windowSize <= 0) {
                throw new IllegalArgumentException("fold-axis geometry is invalid");
            }
        }
    }

    /**
     * Compact canonical NCHW fold geometry.
     * @param kernelHeight positive kernel height
     * @param kernelWidth positive kernel width
     * @param strideHeight positive height stride
     * @param strideWidth positive width stride
     * @param paddingHeight non-negative symmetric height padding
     * @param paddingWidth non-negative symmetric width padding
     * @param dilationHeight positive height dilation
     * @param dilationWidth positive width dilation
     * @param outputColumnsHeight checked column-grid height
     * @param outputColumnsWidth checked column-grid width
     */
    public record TwoDimensionalGeometry(long kernelHeight, long kernelWidth, long strideHeight,
            long strideWidth, long paddingHeight, long paddingWidth, long dilationHeight,
            long dilationWidth, long outputColumnsHeight, long outputColumnsWidth)
            implements Mapping {
        /**
         * Creates cold canonical columns-to-NCHW geometry after Model compatibility and checked
         * arithmetic have been proved by lowering.
         *
         * @param kernelHeight positive kernel height
         * @param kernelWidth positive kernel width
         * @param strideHeight positive height stride
         * @param strideWidth positive width stride
         * @param paddingHeight non-negative symmetric height padding
         * @param paddingWidth non-negative symmetric width padding
         * @param dilationHeight positive height dilation
         * @param dilationWidth positive width dilation
         * @param outputColumnsHeight non-negative checked column-grid height
         * @param outputColumnsWidth non-negative checked column-grid width
         */
        public TwoDimensionalGeometry { }
    }

    /**
     * Static resolved layout retained as cold fold geometry.
     * @param extents non-negative logical extents
     * @param offset non-negative element offset
     * @param strides non-negative element strides
     */
    public record Layout(long[] extents, long offset, long[] strides) {
        /**
         * Snapshots one resolved logical layout used only as cold invocation geometry.
         *
         * @param extents non-null non-negative logical extents
         * @param offset non-negative carrier-relative element offset
         * @param strides non-null non-negative element strides with the same rank as
         *     {@code extents}
         * @throws NullPointerException if either array is {@code null}
         * @throws IllegalArgumentException if ranks differ or any numeric fact is negative
         */
        public Layout {
            extents = extents.clone(); strides = strides.clone();
            if (extents.length != strides.length || offset < 0
                    || Arrays.stream(extents).anyMatch(v -> v < 0)
                    || Arrays.stream(strides).anyMatch(v -> v < 0)) {
                throw new IllegalArgumentException("fold layout facts disagree");
            }
        }
        /**
         * Returns this layout's logical extents without exposing retained array state.
         *
         * @return a new defensive copy of the non-negative logical extents
         */
        @Override public long[] extents() { return extents.clone(); }
        /**
         * Returns this layout's element strides without exposing retained array state.
         *
         * @return a new defensive copy of the non-negative element strides
         */
        @Override public long[] strides() { return strides.clone(); }
    }

    /**
     * Immutable compact fold mapping retained by the prepared recipe.
     * @param family current fold family
     * @param dataType exact represented input/output type
     * @param input resolved input layout
     * @param output resolved output layout
     * @param mapping matching family-specific mapping
     * @param kernelBroadcastInput whether physical input is compact F32/F64 [N,C,P]
     */
    public record Geometry(CpuFoldIr.Family family, DataType dataType, Layout input, Layout output,
            Mapping mapping, boolean kernelBroadcastInput) {

        /**
         * Unresolved numeric invocation recipe. No representative shape, graph or invocation is retained.
         * @param ir exact B-independent generated facts
         * @param boundaries ordered physical boundaries
         * @param boundaryTypes exact boundary storage types
         * @param bindings affine numeric mappings, including the output mapping
         * @param virtualValues private removed intermediate values
         * @param parameters fixed family geometry, defensively copied

         */
        public record BatchGeometry(CpuFoldIr ir, List<ValueId> boundaries,
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
             * Returns fixed overlap-fold geometry cells used during numeric batch binding.
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
                Mapping mapping = ir.family() == CpuFoldIr.Family.FOLD_AXIS
                        ? new AxisGeometry(Math.toIntExact(parameters[0]),parameters[1],parameters[2],parameters[3])
                        : new TwoDimensionalGeometry(parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5],parameters[6],parameters[7],parameters[8],parameters[9]);
                var out = resolved.getLast();
                var geometry = new Geometry(ir.family(),ir.dataType(),new Layout(resolved.getFirst().extents().stream().mapToLong(Long::longValue).toArray(),resolved.getFirst().baseElementOffset(),resolved.getFirst().effectiveStrides().stream().mapToLong(Long::longValue).toArray()),new Layout(out.extents().stream().mapToLong(Long::longValue).toArray(),out.baseElementOffset(),out.effectiveStrides().stream().mapToLong(Long::longValue).toArray()),mapping);
                return new CpuPartitionLowering.LoweredPartition(ir, boundaries, resolved,
                        resolved.stream().map(CpuAccessPlan.Binding::referencedElementSpan).toList(),
                        boundaryTypes, virtualValues, out.extents().stream().mapToLong(Long::longValue).toArray(), out.elementCount(),
                        "legal: checked symbolic ordinary geometry", new long[0],
                        Optional.empty(), Optional.empty(), Optional.empty(), Optional.of(geometry),
                        Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                        Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                        Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                        Optional.empty(), Optional.<CpuPartitionLowering.LoweredPartition.MatmulFacts>empty(), Optional.empty(), Optional.empty(),
                        Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty());
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
            int inputRank = input.extents.length, outputRank = output.extents.length;
            int layout = 8 + inputRank + 2 * outputRank;
            // The ordinary emitter initializes local cursors from immutable packed seeds.
            return new TemporalRebind(List.of(new BaseCell(layout + 1, 0, input.offset),
                    new BaseCell(layout + 3 + 2 * inputRank, 1, output.offset)), List.of());
        }

        /**
         * Creates unchanged ordinary fold geometry.
         * @param family non-null fold family
         * @param dataType non-null represented type
         * @param input non-null physical input layout
         * @param output non-null output layout
         * @param mapping non-null family mapping
         */
        public Geometry(CpuFoldIr.Family family, DataType dataType, Layout input, Layout output, Mapping mapping) {
            this(family, dataType, input, output, mapping, false);
        }
        /**
         * Creates one immutable family-specific fold mapping for a prepared recipe.
         *
         * @param family non-null current fold family
         * @param dataType non-null supported represented input and output type
         * @param input non-null resolved input layout
         * @param output non-null resolved output layout
         * @param mapping non-null family-specific mapping matching the family and ranks
         * @param kernelBroadcastInput compact F32/F64 input mode; never enabled for low formats
         * @throws NullPointerException if a reference is {@code null}
         * @throws IllegalArgumentException if family, type, rank, or mapping facts disagree
         * @throws ArithmeticException if compact geometry's checked inverse or physical byte spans overflow
         */
        public Geometry {
            Objects.requireNonNull(family, "family"); Objects.requireNonNull(dataType, "dataType");
            Objects.requireNonNull(input, "input"); Objects.requireNonNull(output, "output");
            Objects.requireNonNull(mapping, "mapping");
            if (dataType == DataType.BOOL || family == CpuFoldIr.Family.FOLD_AXIS
                    && (!(mapping instanceof AxisGeometry axis)
                        || input.extents.length != output.extents.length + 1
                        || axis.axis >= output.extents.length)
                    || family == CpuFoldIr.Family.FOLD2D
                    && (!(mapping instanceof TwoDimensionalGeometry)
                        || input.extents.length != 3 || output.extents.length != 4
                        || dataType == DataType.INT32 || dataType == DataType.INT64)
                    || kernelBroadcastInput && (family != CpuFoldIr.Family.FOLD2D
                        || dataType != DataType.FLOAT32 && dataType != DataType.FLOAT64
                        || compactBinding(input).plan().regime() != CpuAccessPlan.Regime.DENSE_LINEAR
                        || input.extents[0] != output.extents[0] || input.extents[1] != output.extents[1]
                        || input.extents[2] != Math.multiplyExact(((TwoDimensionalGeometry)mapping).outputColumnsHeight,
                            ((TwoDimensionalGeometry)mapping).outputColumnsWidth))) {
                throw new IllegalArgumentException("fold geometry facts disagree");
            }
            if (kernelBroadcastInput) {
                var two = (TwoDimensionalGeometry) mapping;
                if (two.kernelHeight <= 0 || two.kernelWidth <= 0
                        || Math.multiplyExact(two.kernelHeight, two.kernelWidth) <= 1
                        || two.strideHeight <= 0 || two.strideWidth <= 0
                        || two.paddingHeight < 0 || two.paddingWidth < 0
                        || two.dilationHeight <= 0 || two.dilationWidth <= 0
                        || two.outputColumnsHeight < 0 || two.outputColumnsWidth < 0) {
                    throw new IllegalArgumentException("compact fold window is invalid");
                }
                Math.addExact(Math.max(0, output.extents[2] - 1), two.paddingHeight);
                Math.addExact(Math.max(0, output.extents[3] - 1), two.paddingWidth);
                Math.multiplyExact(two.kernelHeight - 1, two.dilationHeight);
                Math.multiplyExact(two.kernelWidth - 1, two.dilationWidth);
                validateInjective(output.extents, output.strides);
                Math.multiplyExact(referencedSpan(input), dataType.byteWidth());
                Math.multiplyExact(referencedSpan(output), dataType.byteWidth());
            }
        }

        /**
         * Returns the output iteration Shape retained by this geometry.
         *
         * @return a new defensive copy of the prepared output extents
         */
        public long[] outputExtents() { return output.extents(); }

        /**
         * Packs carrier bases, range state, layouts, and mapping into invocation-private storage.
         * @param carrierBases exact input and output element bases
         * @param start inclusive flattened output ordinal
         * @param end exclusive flattened output ordinal
         * @return new primitive packed geometry
         * @throws NullPointerException if {@code carrierBases} is {@code null}
         * @throws IllegalArgumentException if the carrier or range facts disagree
         * @throws ArithmeticException if adding a carrier base to a layout offset overflows
         */
        public long[] pack(long[] carrierBases, long start, long end) {
            if (carrierBases.length != 2 || start < 0 || end < start
                    || end > elementCount(output.extents)) {
                throw new IllegalArgumentException("fold packed geometry facts disagree");
            }
            int inputRank = input.extents.length, outputRank = output.extents.length;
            int mappingSize = family == CpuFoldIr.Family.FOLD_AXIS ? 4 : 10;
            int header = 8;
            int size = header + inputRank + outputRank + outputRank
                    + (2 + 2 * inputRank) + (2 + 2 * outputRank) + mappingSize;
            long[] packed = new long[size]; int x = 0;
            packed[x++] = family.ordinal(); packed[x++] = dataType.ordinal();
            packed[x++] = inputRank; packed[x++] = outputRank;
            packed[x++] = start; packed[x++] = end;
            packed[x++] = elementCount(input.extents); packed[x++] = mappingSize;
            x += inputRank;
            long remainder = start;
            for (int axis = outputRank - 1; axis >= 0; axis--) {
                long extent = output.extents[axis];
                if (extent != 0) { packed[x + axis] = remainder % extent; remainder /= extent; }
            }
            System.arraycopy(packed, x, packed, x + outputRank, outputRank);
            x += 2 * outputRank;
            x = packLayout(packed, x, input, carrierBases[0]);
            x = packLayout(packed, x, output, carrierBases[1]);
            if (mapping instanceof AxisGeometry axis) {
                packed[x++] = axis.axis; packed[x++] = axis.outputSize;
                packed[x++] = axis.step; packed[x] = axis.windowSize;
            } else {
                var two = (TwoDimensionalGeometry) mapping;
                packed[x++] = two.kernelHeight; packed[x++] = two.kernelWidth;
                packed[x++] = two.strideHeight; packed[x++] = two.strideWidth;
                packed[x++] = two.paddingHeight; packed[x++] = two.paddingWidth;
                packed[x++] = two.dilationHeight; packed[x++] = two.dilationWidth;
                packed[x++] = two.outputColumnsHeight; packed[x] = two.outputColumnsWidth;
            }
            return packed;
        }

        private static int packLayout(long[] target, int x, Layout layout, long base) {
            target[x++] = layout.extents.length;
            target[x++] = Math.addExact(base, layout.offset);
            for (long extent : layout.extents) target[x++] = extent;
            for (long stride : layout.strides) target[x++] = stride;
            return x;
        }
    }
}
