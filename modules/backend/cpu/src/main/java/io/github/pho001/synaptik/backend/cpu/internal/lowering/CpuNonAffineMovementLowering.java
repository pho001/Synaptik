package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.PackReset;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuDataMovementIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.layout.CompositionAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.CropToShapeAttrs;
import io.github.pho001.synaptik.model.operation.layout.PadAttrs;
import io.github.pho001.synaptik.model.operation.layout.PadKind;
import io.github.pho001.synaptik.model.operation.layout.SliceAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceKind;
import io.github.pho001.synaptik.model.operation.layout.TensorCompositionKind;
import io.github.pho001.synaptik.model.operation.layout.TileAttrs;
import io.github.pho001.synaptik.model.operation.layout.TileKind;
import io.github.pho001.synaptik.model.operation.layout.Unfold2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.UnfoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.Window2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Window3dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Unfold3dAttrs;
import io.github.pho001.synaptik.model.operation.layout.WindowTransformKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.StaticDimension;
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
 * Lowers one fully static {@code PAD}, {@code TILE}, {@code CONCAT}, {@code STACK},
 * {@code UNFOLD_AXIS}, {@code UNFOLD2D}, {@code UNFOLD3D}, or {@code SLICE_UPDATE}
 * occurrence into one represented-bit movement unit.
 *
 * <p>The lowering preserves semantic input occurrence order while declaring each distinct input
 * value once in first-occurrence order. Exact extents, layout offsets and strides, padding,
 * repeats, composition prefixes, and window facts remain in compact immutable {@link Geometry};
 * no prepared address or selector table is retained per output element. General-axis unfold
 * copies every represented type. NCHW/NCDHW window transforms copy all floating types,
 * including the byte-backed OFP8 formats, and retain exact configured padding bits; direct attributes use represented
 * positive zero. Functional slice update normalizes either signed finite-coordinate or
 * target-relative attributes to rank-sized start, length, and step arrays. Its result-domain
 * mapping has the semantic effect of copying the base and replacing selected positions while
 * leaving both inputs unchanged.</p>
 *
 * <p>Variadic composition keeps checked semantic cardinality in prepared occurrence/address
 * tables rather than JVM argument or generated-arm counts. Each source occurrence selects
 * directly into one output traversal; repeated identities share a read boundary without staging.</p>
 *
 * <p>Canonical single-named-B occurrences use {@link #lowerBatch(PrepareContext)}. Their
 * original shape relationships are proved structurally and their numeric geometry is resolved
 * once at cold binding, without sampling B or rebuilding a semantic graph.</p>
 */
public final class CpuNonAffineMovementLowering {
    /** Creates a stateless movement-family lowerer. */
    public CpuNonAffineMovementLowering() {
    }

    /**
     * Lowers one exact movement occurrence.
     *
     * @param context non-null complete CPU partition projection containing exactly one node
     * @return one immutable movement lowering with unique inputs followed by one output
     * @throws NullPointerException if {@code context} is {@code null}
     * @throws IllegalArgumentException if the occurrence is outside the bounded static matrix
     * @throws ArithmeticException if exact extent, span, or element-count arithmetic overflows
     */
    public CpuPartitionLowering.LoweredPartition lower(
            PrepareContext<? extends BackendAnalysisInputs> context) {
        Objects.requireNonNull(context, "context");
        if (context.nodes().size() != 1) {
            throw new IllegalArgumentException("CPU movement partition requires exactly one node");
        }
        var node = context.nodes().getFirst();
        if (node.outputs().size() != 1) {
            throw new IllegalArgumentException("CPU movement occurrence requires one output");
        }
        Map<ValueId, GraphValue> values = new LinkedHashMap<>();
        context.values().forEach(value -> values.put(value.id(), value));
        GraphValue output = require(values, node.outputs().getFirst());
        requireStaticResolved(output);
        long[] outputExtents = output.descriptor().shape().toLongArray();
        LayoutDescriptor outputLayout = output.descriptor().layout().orElseThrow();
        validateInjective(outputExtents, outputLayout.strides());

        var unique = new LinkedHashMap<ValueId, Integer>();
        var occurrences = new ArrayList<GraphValue>(node.inputs().size());
        var occurrenceMap = new ArrayList<Integer>(node.inputs().size());
        for (ValueId inputId : node.inputs()) {
            GraphValue input = require(values, inputId);
            requireStaticResolved(input);
            if (input.descriptor().dataType() != output.descriptor().dataType()) {
                throw new IllegalArgumentException("movement input and output data types must match");
            }
            occurrences.add(input);
            occurrenceMap.add(unique.computeIfAbsent(inputId, ignored -> unique.size()));
        }
        if (occurrences.isEmpty()) throw new IllegalArgumentException("movement requires inputs");

        Object kind = node.operation().kind();
        Object attrs = node.operation().attrs();
        CpuDataMovementIr.MovementPlan movementPlan;
        Geometry.Variant variant;
        if (kind == SliceKind.SLICE_UPDATE) {
            if (occurrences.size() != 2) {
                throw new IllegalArgumentException("SLICE_UPDATE requires base and update inputs");
            }
            long[] base = occurrences.get(0).descriptor().shape().toLongArray();
            long[] update = occurrences.get(1).descriptor().shape().toLongArray();
            if (!Arrays.equals(base, outputExtents) || base.length != update.length) {
                throw new IllegalArgumentException("SLICE_UPDATE base, update, and output ranks are inconsistent");
            }
            long[] starts = new long[base.length];
            long[] lengths = base.clone();
            long[] steps = new long[base.length];
            Arrays.fill(steps, 1L);
            if (attrs instanceof SliceAttrs slice) {
                boolean[] selected = new boolean[base.length];
                for (int index = 0; index < slice.axes().size(); index++) {
                    int axis = slice.axes().get(index);
                    if (axis >= base.length || selected[axis]
                            || update[axis] != slice.lengths().get(index)) {
                        throw new IllegalArgumentException("SLICE_UPDATE signed geometry is inconsistent");
                    }
                    selected[axis] = true;
                    starts[axis] = slice.starts().get(index);
                    lengths[axis] = slice.lengths().get(index);
                    steps[axis] = slice.steps().get(index);
                    if (lengths[axis] > 0) {
                        long last = Math.addExact(starts[axis],
                                Math.multiplyExact(lengths[axis] - 1, steps[axis]));
                        if (starts[axis] >= base[axis] || last < 0 || last >= base[axis]) {
                            throw new IllegalArgumentException("SLICE_UPDATE coordinate is outside the base");
                        }
                    }
                }
                for (int axis = 0; axis < base.length; axis++) {
                    if (!selected[axis] && update[axis] != base[axis]) {
                        throw new IllegalArgumentException("SLICE_UPDATE unselected extent must match base");
                    }
                }
            } else if (attrs instanceof CropToShapeAttrs crop) {
                if (!crop.targetShape().isFullyStatic() || !crop.prefixShape().isFullyStatic()
                        || !crop.targetShape().equals(occurrences.get(1).descriptor().shape())
                        || crop.prefixShape().rank() != base.length) {
                    throw new IllegalArgumentException("SLICE_UPDATE target-relative geometry is inconsistent");
                }
                starts = crop.prefixShape().toLongArray();
                lengths = update.clone();
                for (int axis = 0; axis < base.length; axis++) {
                    if (Math.addExact(starts[axis], lengths[axis]) > base[axis]) {
                        throw new IllegalArgumentException("SLICE_UPDATE target exceeds base");
                    }
                }
            } else {
                throw new IllegalArgumentException("unsupported SLICE_UPDATE attributes");
            }
            movementPlan = new CpuDataMovementIr.SliceUpdatePlan(outputExtents.length, occurrenceMap);
            variant = new Geometry.SliceUpdate(starts, lengths, steps);
        } else if (kind == PadKind.PAD && attrs instanceof PadAttrs pad) {
            requireUnary(occurrences, "PAD");
            long[] input = occurrences.getFirst().descriptor().shape().toLongArray();
            if (pad.before().size() != input.length || pad.after().size() != input.length
                    || pad.constantValue().dataType() != output.descriptor().dataType()
                    || outputExtents.length != input.length) {
                throw new IllegalArgumentException("PAD rank, constant, or descriptor is inconsistent");
            }
            for (int axis = 0; axis < input.length; axis++) {
                long expected = Math.addExact(pad.before().get(axis),
                        Math.addExact(input[axis], pad.after().get(axis)));
                if (outputExtents[axis] != expected) {
                    throw new IllegalArgumentException("PAD output extent is inconsistent");
                }
            }
            long bits = scalarBits(pad.constantValue());
            movementPlan = new CpuDataMovementIr.PadPlan(outputExtents.length, bits);
            variant = new Geometry.Pad(toArray(pad.before()), input);
        } else if (kind == TileKind.TILE && attrs instanceof TileAttrs tile) {
            requireUnary(occurrences, "TILE");
            long[] input = occurrences.getFirst().descriptor().shape().toLongArray();
            if (tile.repeats().size() != input.length || outputExtents.length != input.length) {
                throw new IllegalArgumentException("TILE rank is inconsistent");
            }
            for (int axis = 0; axis < input.length; axis++) {
                if (outputExtents[axis] != Math.multiplyExact(input[axis], tile.repeats().get(axis))) {
                    throw new IllegalArgumentException("TILE output extent is inconsistent");
                }
            }
            movementPlan = new CpuDataMovementIr.TilePlan(outputExtents.length);
            variant = new Geometry.Tile(input, toArray(tile.repeats()));
        } else if (kind == TensorCompositionKind.CONCAT
                && attrs instanceof CompositionAxisAttrs composition) {
            int axis = composition.axis();
            int rank = occurrences.getFirst().descriptor().shape().rank();
            if (rank == 0 || axis >= rank || outputExtents.length != rank) {
                throw new IllegalArgumentException("CONCAT axis or rank is inconsistent");
            }
            long[] first = occurrences.getFirst().descriptor().shape().toLongArray();
            long[] prefixes = new long[Math.addExact(occurrences.size(), 1)];
            for (int index = 0; index < occurrences.size(); index++) {
                long[] shape = occurrences.get(index).descriptor().shape().toLongArray();
                if (shape.length != rank) throw new IllegalArgumentException("CONCAT ranks must match");
                for (int current = 0; current < rank; current++) {
                    if (current != axis && shape[current] != first[current]) {
                        throw new IllegalArgumentException("CONCAT non-axis extents must match");
                    }
                }
                prefixes[index + 1] = Math.addExact(prefixes[index], shape[axis]);
            }
            for (int current = 0; current < rank; current++) {
                long expected = current == axis ? prefixes[prefixes.length - 1] : first[current];
                if (outputExtents[current] != expected) {
                    throw new IllegalArgumentException("CONCAT output shape is inconsistent");
                }
            }
            movementPlan = new CpuDataMovementIr.ConcatPlan(rank, occurrenceMap);
            variant = new Geometry.Concat(axis, prefixes, occurrenceMap);
        } else if (kind == TensorCompositionKind.STACK
                && attrs instanceof CompositionAxisAttrs composition) {
            long[] input = occurrences.getFirst().descriptor().shape().toLongArray();
            int axis = composition.axis();
            if (axis > input.length || outputExtents.length != input.length + 1) {
                throw new IllegalArgumentException("STACK axis or rank is inconsistent");
            }
            for (GraphValue occurrence : occurrences) {
                if (!Arrays.equals(input, occurrence.descriptor().shape().toLongArray())) {
                    throw new IllegalArgumentException("STACK input shapes must match exactly");
                }
            }
            for (int outAxis = 0, inAxis = 0; outAxis < outputExtents.length; outAxis++) {
                long expected = outAxis == axis ? occurrences.size() : input[inAxis++];
                if (outputExtents[outAxis] != expected) {
                    throw new IllegalArgumentException("STACK output shape is inconsistent");
                }
            }
            movementPlan = new CpuDataMovementIr.StackPlan(outputExtents.length, occurrenceMap);
            variant = new Geometry.Stack(axis, occurrenceMap);
        } else if (kind == WindowTransformKind.UNFOLD_AXIS
                && attrs instanceof UnfoldAxisAttrs unfold) {
            requireUnary(occurrences, "UNFOLD_AXIS");
            long[] input = occurrences.getFirst().descriptor().shape().toLongArray();
            int axis = unfold.axis();
            if (input.length == 0 || axis >= input.length || outputExtents.length != input.length + 1
                    || unfold.size() > input[axis]) {
                throw new IllegalArgumentException("UNFOLD_AXIS rank, axis, or size is inconsistent");
            }
            long positions = Math.addExact((input[axis] - unfold.size()) / unfold.step(), 1);
            for (int outAxis = 0; outAxis < outputExtents.length; outAxis++) {
                long expected = outAxis == input.length ? unfold.size()
                        : outAxis == axis ? positions : input[outAxis];
                if (outputExtents[outAxis] != expected) {
                    throw new IllegalArgumentException("UNFOLD_AXIS output shape is inconsistent");
                }
            }
            Math.addExact(Math.multiplyExact(positions - 1, unfold.step()), unfold.size() - 1);
            movementPlan = new CpuDataMovementIr.UnfoldAxisPlan(outputExtents.length);
            variant = new Geometry.UnfoldAxis(axis, unfold.size(), unfold.step());
        } else if (kind == WindowTransformKind.UNFOLD2D
                && (attrs instanceof Window2dAttrs || attrs instanceof Unfold2dAttrs)) {
            requireUnary(occurrences, "UNFOLD2D");
            DataType type = output.descriptor().dataType();
            if (!type.isFloating()) {
                throw new IllegalArgumentException("UNFOLD2D requires a floating represented type");
            }
            Window2dAttrs window;
            long bits;
            if (attrs instanceof Window2dAttrs direct) {
                window = direct;
                bits = 0;
            } else {
                Unfold2dAttrs configured = (Unfold2dAttrs) attrs;
                window = configured.window();
                if (configured.paddingValue().dataType() != type) {
                    throw new IllegalArgumentException("UNFOLD2D padding type must match the boundary type");
                }
                bits = scalarBits(configured.paddingValue());
            }
            long[] input = occurrences.getFirst().descriptor().shape().toLongArray();
            if (input.length != 4 || outputExtents.length != 3) {
                throw new IllegalArgumentException("UNFOLD2D requires rank-four NCHW input and rank-three output");
            }
            long effectiveH = Math.addExact(Math.multiplyExact(window.dilationHeight(),
                    window.kernelHeight() - 1), 1);
            long effectiveW = Math.addExact(Math.multiplyExact(window.dilationWidth(),
                    window.kernelWidth() - 1), 1);
            long paddedH = Math.addExact(input[2], Math.multiplyExact(2, window.paddingHeight()));
            long paddedW = Math.addExact(input[3], Math.multiplyExact(2, window.paddingWidth()));
            long numeratorH = Math.subtractExact(paddedH, effectiveH);
            long numeratorW = Math.subtractExact(paddedW, effectiveW);
            if (numeratorH < 0 || numeratorW < 0) {
                throw new IllegalArgumentException("UNFOLD2D effective kernel exceeds padded input");
            }
            long outH = outputExtent(numeratorH, window.strideHeight(), window.ceilMode());
            long outW = outputExtent(numeratorW, window.strideWidth(), window.ceilMode());
            long columns = Math.multiplyExact(Math.multiplyExact(input[1], window.kernelHeight()),
                    window.kernelWidth());
            long positions = Math.multiplyExact(outH, outW);
            if (outputExtents[0] != input[0] || outputExtents[1] != columns
                    || outputExtents[2] != positions) {
                throw new IllegalArgumentException("UNFOLD2D output shape is inconsistent");
            }
            movementPlan = new CpuDataMovementIr.Unfold2dPlan(3, bits);
            variant = new Geometry.Unfold2d(input[1], input[2], input[3], window.kernelHeight(),
                    window.kernelWidth(), window.strideHeight(), window.strideWidth(),
                    window.paddingHeight(), window.paddingWidth(), window.dilationHeight(),
                    window.dilationWidth(), outH, outW);
        } else if (kind == WindowTransformKind.UNFOLD3D
                && (attrs instanceof Window3dAttrs || attrs instanceof Unfold3dAttrs)) {
            requireUnary(occurrences, "UNFOLD3D");
            DataType type = output.descriptor().dataType();
            if (!type.isFloating())
                throw new IllegalArgumentException("UNFOLD3D requires a floating represented type");
            Window3dAttrs window;
            long bits = 0;
            if (attrs instanceof Window3dAttrs direct) window = direct;
            else {
                var configured = (Unfold3dAttrs) attrs;
                if (configured.paddingValue().dataType() != type)
                    throw new IllegalArgumentException("UNFOLD3D padding type disagrees");
                window = configured.window(); bits = scalarBits(configured.paddingValue());
            }
            long[] input = occurrences.getFirst().descriptor().shape().toLongArray();
            var spatial = CpuSpatialWindowGeometry.of(input, window);
            spatial.requireColumns(input, outputExtents);
            movementPlan = new CpuDataMovementIr.Unfold3dPlan(3, bits);
            variant = new Geometry.Unfold3d(spatial);
        } else {
            throw new IllegalArgumentException("unsupported CPU movement family");
        }

        var boundaryValues = new ArrayList<ValueId>(unique.size() + 1);
        var bindings = new ArrayList<CpuAccessPlan.Binding>(unique.size() + 1);
        var spans = new ArrayList<Long>(unique.size() + 1);
        var types = new ArrayList<DataType>(unique.size() + 1);
        var inputAccesses = new ArrayList<CpuAccessPlan>(unique.size());
        var inputGeometry = new ArrayList<Geometry.Input>(unique.size());
        for (ValueId id : unique.keySet()) {
            GraphValue value = require(values, id);
            LayoutDescriptor layout = value.descriptor().layout().orElseThrow();
            long[] extents = value.descriptor().shape().toLongArray();
            CpuAccessPlan.Binding binding = binding(extents, layout, CpuAccessPlan.AccessKind.READ);
            boundaryValues.add(id);
            bindings.add(binding);
            spans.add(layout.referencedElementSpan());
            types.add(value.descriptor().dataType());
            inputAccesses.add(binding.plan());
            inputGeometry.add(new Geometry.Input(extents, layout.storageOffset(), layout.strides()));
        }
        CpuAccessPlan.Binding outputBinding = binding(outputExtents, outputLayout,
                CpuAccessPlan.AccessKind.WRITE);
        boundaryValues.add(output.id());
        bindings.add(outputBinding);
        spans.add(outputLayout.referencedElementSpan());
        types.add(output.descriptor().dataType());
        Geometry geometry = new Geometry(outputExtents, outputLayout.storageOffset(),
                outputLayout.strides(), inputGeometry, variant);
        var ir = new CpuDataMovementIr(output.descriptor().dataType(), movementPlan,
                inputAccesses, outputBinding.plan());
        return new CpuPartitionLowering.LoweredPartition(ir, boundaryValues, bindings, spans,
                types, List.of(), outputExtents,
                output.descriptor().shape().knownElementCount().orElseThrow(),
                "legal: one static non-affine movement occurrence", new long[0],
                Optional.of(geometry));
    }

    /**
     * Proves one canonical symbolic movement occurrence and constructs its ordinary structural IR.
     * Boundary recipes retain arbitrary original ranks and positions of the common named axis.
     * @param context non-null complete one-node movement projection
     * @return unresolved family-owned numeric recipe and ordinary movement IR
     * @throws IllegalArgumentException if original shapes, types, attributes or layouts disagree
     * @throws ArithmeticException if fixed coefficient arithmetic overflows
     */
    public CpuPartitionLowering.LoweredPartition lowerBatch(
            PrepareContext<? extends BackendAnalysisInputs> context) {
        return lowerBatch(context, CpuPartitionLowering.batchLayouts(context));
    }

    /**
     * Uses the full lexical body's proved physical mappings, preserving incoming view geometry.
     * @param context non-null original one-node movement projection
     * @param layouts non-null complete local value-to-layout map
     * @return ordinary structural IR and unresolved numeric movement recipe
     * @throws IllegalArgumentException if ordinary descriptors or numeric mappings disagree
     */
    public CpuPartitionLowering.LoweredPartition lowerBatch(
            PrepareContext<? extends BackendAnalysisInputs> context,
            Map<ValueId, CpuAccessPlan.BatchBinding> layouts) {
        var batchDimension = CpuPartitionLowering.requireBatchContext(context);
        if (context.nodes().size() != 1) throw invalidBatch();
        var node = context.nodes().getFirst();
        if (node.outputs().size() != 1 || node.inputs().isEmpty()) throw invalidBatch();
        Map<ValueId, GraphValue> values = new LinkedHashMap<>();
        context.values().forEach(v -> values.put(v.id(), v));
        var unique = new LinkedHashMap<ValueId, Integer>();
        var map = new ArrayList<Integer>();
        for (var id : node.inputs()) map.add(unique.computeIfAbsent(id, ignored -> unique.size()));
        if (unique.containsKey(node.outputs().getFirst())) throw invalidBatch();
        var ids = new ArrayList<>(unique.keySet());
        ids.add(node.outputs().getFirst());
        var type = require(values, ids.getLast()).descriptor().dataType();
        var accesses = new ArrayList<CpuAccessPlan.BatchBinding>();
        for (int b = 0; b < ids.size(); b++) {
            var descriptor = require(values, ids.get(b)).descriptor();
            if (descriptor.dataType() != type) throw invalidBatch();
            var access = layouts.get(ids.get(b));
            if (access == null) throw invalidBatch();
            var source = access.plan();
            var permission = new CpuAccessPlan(b + 1 == ids.size() ? CpuAccessPlan.AccessKind.WRITE
                    : CpuAccessPlan.AccessKind.READ, source.regime(), source.iterationRank(),
                    source.axisRoles(), source.contiguousSuffix());
            accesses.add(new CpuAccessPlan.BatchBinding(permission, access.extentConstants(),
                    access.extentBatchFactors(), access.baseElementOffset(),
                    access.strideConstants(), access.strideBatchFactors()));
        }
        var input = accesses.get(map.getFirst());
        var output = accesses.getLast();
        int rank = output.plan().iterationRank(), inputRank = input.plan().iterationRank();
        Object kind = node.operation().kind(), attrs = node.operation().attrs();
        CpuDataMovementIr.MovementPlan plan;
        long[] facts;
        long[] factors;
        if (kind == PadKind.PAD && attrs instanceof PadAttrs pad) {
            if (map.size() != 1 || rank != inputRank || pad.before().size() != rank
                    || pad.after().size() != rank || pad.constantValue().dataType() != type)
                throw invalidBatch();
            for (int a = 0; a < rank; a++) checkExtent(output, a,
                    Math.addExact(ec(input, a), Math.addExact(pad.before().get(a), pad.after().get(a))),
                    ef(input, a));
            plan = new CpuDataMovementIr.PadPlan(rank, scalarBits(pad.constantValue()));
            facts = toArray(pad.before()); factors = new long[facts.length];
        } else if (kind == TileKind.TILE && attrs instanceof TileAttrs tile) {
            if (map.size() != 1 || rank != inputRank || tile.repeats().size() != rank)
                throw invalidBatch();
            for (int a = 0; a < rank; a++) checkExtent(output, a,
                    Math.multiplyExact(ec(input, a), tile.repeats().get(a)),
                    Math.multiplyExact(ef(input, a), tile.repeats().get(a)));
            plan = new CpuDataMovementIr.TilePlan(rank);
            facts = toArray(tile.repeats()); factors = new long[facts.length];
        } else if (kind == TensorCompositionKind.CONCAT && attrs instanceof CompositionAxisAttrs c) {
            int axis = c.axis();
            if (rank == 0 || rank != inputRank || axis >= rank) throw invalidBatch();
            long total = 0, totalFactor = 0;
            for (int position : map) {
                var occurrence = accesses.get(position);
                if (occurrence.plan().iterationRank() != rank) throw invalidBatch();
                for (int a = 0; a < rank; a++)
                    if (a != axis) checkExtent(occurrence, a, ec(input, a), ef(input, a));
                total = Math.addExact(total, ec(occurrence, axis));
                totalFactor = Math.addExact(totalFactor, ef(occurrence, axis));
            }
            for (int a = 0; a < rank; a++) checkExtent(output, a,
                    a == axis ? total : ec(input, a), a == axis ? totalFactor : ef(input, a));
            plan = new CpuDataMovementIr.ConcatPlan(rank, map);
            facts = new long[] {axis}; factors = new long[1];
        } else if (kind == TensorCompositionKind.STACK && attrs instanceof CompositionAxisAttrs c) {
            int axis = c.axis();
            if (rank != inputRank + 1 || axis > inputRank) throw invalidBatch();
            for (int position : map) {
                var occurrence = accesses.get(position);
                if (!occurrence.extentConstants().equals(input.extentConstants())
                        || !occurrence.extentBatchFactors().equals(input.extentBatchFactors()))
                    throw invalidBatch();
            }
            for (int a = 0; a < rank; a++) checkExtent(output, a,
                    a == axis ? map.size() : ec(input, a < axis ? a : a - 1),
                    a == axis ? 0 : ef(input, a < axis ? a : a - 1));
            plan = new CpuDataMovementIr.StackPlan(rank, map);
            facts = new long[] {axis}; factors = new long[1];
        } else if (kind == WindowTransformKind.UNFOLD_AXIS && attrs instanceof UnfoldAxisAttrs u) {
            int axis = u.axis();
            if (map.size() != 1 || inputRank == 0 || rank != inputRank + 1 || axis >= inputRank)
                throw invalidBatch();
            if (ef(input, axis) == 0 && u.size() > ec(input, axis)
                    || ef(input, axis) % u.step() != 0) throw invalidBatch();
            long positions = Math.addExact(Math.floorDiv(
                    Math.subtractExact(ec(input, axis), u.size()), u.step()), 1);
            long positionFactor = ef(input, axis) / u.step();
            for (int a = 0; a < rank; a++) checkExtent(output, a,
                    a == inputRank ? u.size() : a == axis ? positions : ec(input, a),
                    a == inputRank ? 0 : a == axis ? positionFactor : ef(input, a));
            plan = new CpuDataMovementIr.UnfoldAxisPlan(rank);
            facts = new long[] {axis, u.size(), u.step()}; factors = new long[3];
        } else if (kind == WindowTransformKind.UNFOLD2D
                && (attrs instanceof Window2dAttrs || attrs instanceof Unfold2dAttrs)) {
            if (map.size() != 1 || inputRank != 4 || rank != 3
                    || !type.isFloating()) throw invalidBatch();
            Window2dAttrs window;
            long bits;
            if (attrs instanceof Window2dAttrs direct) { window = direct; bits = 0; }
            else {
                var configured = (Unfold2dAttrs) attrs;
                if (configured.paddingValue().dataType() != type) throw invalidBatch();
                window = configured.window(); bits = scalarBits(configured.paddingValue());
            }
            long h = windowPositions(input, 2, window.kernelHeight(), window.dilationHeight(),
                    window.paddingHeight(), window.strideHeight(), window.ceilMode());
            long w = windowPositions(input, 3, window.kernelWidth(), window.dilationWidth(),
                    window.paddingWidth(), window.strideWidth(), window.ceilMode());
            long hf = ef(input, 2) / window.strideHeight();
            long wf = ef(input, 3) / window.strideWidth();
            if (hf != 0 && wf != 0) throw invalidBatch();
            long kernel = Math.multiplyExact(window.kernelHeight(), window.kernelWidth());
            checkExtent(output, 0, ec(input, 0), ef(input, 0));
            checkExtent(output, 1, Math.multiplyExact(ec(input, 1), kernel),
                    Math.multiplyExact(ef(input, 1), kernel));
            checkExtent(output, 2, Math.multiplyExact(h, w),
                    Math.addExact(Math.multiplyExact(hf, w), Math.multiplyExact(h, wf)));
            plan = new CpuDataMovementIr.Unfold2dPlan(rank, bits);
            facts = new long[] {ec(input, 1), ec(input, 2), ec(input, 3), window.kernelHeight(),
                    window.kernelWidth(), window.strideHeight(), window.strideWidth(),
                    window.paddingHeight(), window.paddingWidth(), window.dilationHeight(),
                    window.dilationWidth(), h, w};
            factors = new long[] {ef(input, 1), ef(input, 2), ef(input, 3), 0, 0, 0, 0, 0, 0, 0, 0, hf, wf};
        } else if (kind == WindowTransformKind.UNFOLD3D
                && (attrs instanceof Window3dAttrs || attrs instanceof Unfold3dAttrs)) {
            if (map.size() != 1 || inputRank != 5 || rank != 3
                    || !type.isFloating()) throw invalidBatch();
            Window3dAttrs window;
            long bits = 0;
            if (attrs instanceof Window3dAttrs direct) window = direct;
            else {
                var configured = (Unfold3dAttrs) attrs;
                if (configured.paddingValue().dataType() != type) throw invalidBatch();
                window = configured.window(); bits = scalarBits(configured.paddingValue());
            }
            long[] image = new long[5];
            for (int a = 1; a < 5; a++) {
                if (ef(input, a) != 0) throw invalidBatch();
                image[a] = ec(input, a);
            }
            var spatial = CpuSpatialWindowGeometry.of(image, window);
            checkExtent(output, 0, ec(input, 0), ef(input, 0));
            checkExtent(output, 1, Math.multiplyExact(image[1], spatial.kernelCount()), 0);
            checkExtent(output, 2, spatial.columnCount(), 0);
            plan = new CpuDataMovementIr.Unfold3dPlan(3, bits);
            facts = spatial.cells(); factors = new long[facts.length];
        } else if (kind == SliceKind.SLICE_UPDATE) {
            if (map.size() != 2 || rank != inputRank
                    || !output.extentConstants().equals(input.extentConstants())
                    || !output.extentBatchFactors().equals(input.extentBatchFactors()))
                throw invalidBatch();
            var update = accesses.get(map.get(1));
            if (update.plan().iterationRank() != rank) throw invalidBatch();
            facts = new long[2 * rank + 1]; factors = new long[2 * rank + 1];
            Arrays.fill(facts, rank, 2 * rank, 1);
            if (attrs instanceof SliceAttrs slice) {
                boolean[] selected = new boolean[rank];
                for (int n = 0; n < slice.axes().size(); n++) {
                    int a = slice.axes().get(n);
                    if (a >= rank || selected[a]) throw invalidBatch();
                    selected[a] = true;
                    checkExtent(update, a, slice.lengths().get(n), 0);
                    facts[a] = slice.starts().get(n); facts[rank + a] = slice.steps().get(n);
                    if (slice.lengths().get(n) > 0) {
                        long last = Math.addExact(facts[a], Math.multiplyExact(
                                slice.lengths().get(n) - 1, facts[rank + a]));
                        if (last < 0 || ef(input, a) == 0
                                && (facts[a] >= ec(input, a) || last >= ec(input, a))) throw invalidBatch();
                    }
                }
                for (int a = 0; a < rank; a++)
                    if (!selected[a]) checkExtent(update, a, ec(input, a), ef(input, a));
            } else if (attrs instanceof CropToShapeAttrs crop) {
                facts[2 * rank] = 1; // Target-relative bounds apply even to an empty update axis.
                if (!crop.targetShape().equals(require(values, node.inputs().get(1)).descriptor().shape())
                        || crop.prefixShape().rank() != rank) throw invalidBatch();
                for (int a = 0; a < rank; a++) {
                    var prefix = crop.prefixShape().dimensions().get(a);
                    if (prefix instanceof StaticDimension fixed) facts[a] = fixed.size();
                    else if (prefix instanceof DynamicDimension dynamic && dynamic.equals(batchDimension))
                        factors[a] = 1;
                    else throw invalidBatch();
                    if (ef(input, a) == 0 && (factors[a] != 0 || ef(update, a) != 0
                            || Math.addExact(facts[a], ec(update, a)) > ec(input, a))) throw invalidBatch();
                    if (Math.addExact(factors[a], ef(update, a)) > ef(input, a)) throw invalidBatch();
                }
            } else throw invalidBatch();
            plan = new CpuDataMovementIr.SliceUpdatePlan(rank, map);
        } else throw invalidBatch();
        var ir = new CpuDataMovementIr(type, plan, accesses.subList(0, accesses.size() - 1)
                .stream().map(CpuAccessPlan.BatchBinding::plan).toList(), output.plan());
        return new CpuPartitionLowering.LoweredPartition(new Geometry.BatchGeometry(ir, ids,
                java.util.Collections.nCopies(ids.size(), type), accesses, facts, factors));
    }

    private static IllegalArgumentException invalidBatch() {
        return new IllegalArgumentException("inconsistent symbolic movement geometry");
    }
    private static long ec(CpuAccessPlan.BatchBinding b, int axis) { return b.extentConstants().get(axis); }
    private static long ef(CpuAccessPlan.BatchBinding b, int axis) { return b.extentBatchFactors().get(axis); }
    private static void checkExtent(CpuAccessPlan.BatchBinding b, int axis, long constant, long factor) {
        if (ec(b, axis) != constant || ef(b, axis) != factor) throw invalidBatch();
    }
    private static long windowPositions(CpuAccessPlan.BatchBinding b, int axis, long kernel,
            long dilation, long padding, long stride, boolean ceil) {
        long effective = Math.addExact(Math.multiplyExact(dilation, kernel - 1), 1);
        long numerator = Math.subtractExact(Math.addExact(ec(b, axis), Math.multiplyExact(2, padding)), effective);
        if (ef(b, axis) % stride != 0 || ef(b, axis) == 0 && numerator < 0) throw invalidBatch();
        long quotient = Math.floorDiv(numerator, stride);
        if (ceil && Math.floorMod(numerator, stride) != 0) quotient = Math.addExact(quotient, 1);
        return Math.addExact(quotient, 1);
    }
    private static BatchByteSize batchCount(CpuAccessPlan.BatchBinding b) {
        return CpuPartitionLowering.batchElements(b);
    }

    private static void requireUnary(List<GraphValue> inputs, String family) {
        if (inputs.size() != 1) throw new IllegalArgumentException(family + " requires one input");
    }

    private static GraphValue require(Map<ValueId, GraphValue> values, ValueId id) {
        GraphValue value = values.get(id);
        if (value == null) throw new IllegalArgumentException("partition value is not projected: " + id);
        return value;
    }

    private static void requireStaticResolved(GraphValue value) {
        if (!value.descriptor().shape().isFullyStatic() || value.descriptor().layout().isEmpty()) {
            throw new IllegalArgumentException("movement requires static Shapes and resolved layouts");
        }
    }

    private static long scalarBits(ScalarValue value) {
        return switch (value.dataType()) {
            case FLOAT64 -> Double.doubleToRawLongBits(value.float64Value());
            case FLOAT32 -> Float.floatToRawIntBits(value.float32Value()) & 0xffff_ffffL;
            case BFLOAT16 -> value.bfloat16Bits() & 0xffffL;
            case FLOAT16 -> value.float16Bits() & 0xffffL;
            case FLOAT8_E4M3 -> Byte.toUnsignedInt(value.float8E4m3Bits());
            case FLOAT8_E5M2 -> Byte.toUnsignedInt(value.float8E5m2Bits());
            case INT32 -> value.int32Value() & 0xffff_ffffL;
            case INT64 -> value.int64Value();
            case BOOL -> value.booleanValue() ? 1L : 0L;
        };
    }

    private static long[] toArray(List<Long> source) {
        return source.stream().mapToLong(Long::longValue).toArray();
    }

    private static long outputExtent(long numerator, long stride, boolean ceilMode) {
        long quotient = numerator / stride;
        if (ceilMode && numerator % stride != 0) quotient = Math.addExact(quotient, 1);
        return Math.addExact(quotient, 1);
    }

    private static CpuAccessPlan.Binding binding(long[] extents, LayoutDescriptor layout,
            CpuAccessPlan.AccessKind kind) {
        long[] strides = layout.strides();
        int suffix = 0;
        long expected = 1;
        for (int axis = extents.length - 1; axis >= 0; axis--) {
            if (strides[axis] != expected) break;
            suffix++;
            expected = Math.multiplyExact(expected, Math.max(1, extents[axis]));
        }
        var roles = new ArrayList<CpuAccessPlan.AxisRole>(extents.length);
        for (int axis = 0; axis < extents.length; axis++) roles.add(strides[axis] == 0
                ? CpuAccessPlan.AxisRole.BROADCAST
                : axis >= extents.length - suffix ? CpuAccessPlan.AxisRole.CONTIGUOUS
                : CpuAccessPlan.AxisRole.STRIDED);
        CpuAccessPlan.Regime regime = suffix == extents.length
                ? CpuAccessPlan.Regime.DENSE_LINEAR : CpuAccessPlan.Regime.GENERAL_ODOMETER;
        CpuAccessPlan plan = new CpuAccessPlan(kind, regime, extents.length, roles, suffix);
        long count = Arrays.stream(extents).anyMatch(extent -> extent == 0) ? 0 : 1;
        if (count != 0) for (long extent : extents) count = Math.multiplyExact(count, extent);
        return CpuAccessPlan.Binding.create(plan, extents, layout.storageOffset(), strides,
                count, 0, count, layout.referencedElementSpan());
    }

    private static void validateInjective(long[] extents, long[] strides) {
        if (!CpuOutputInjectivity.isInjective(extents, strides)) {
            throw new IllegalArgumentException("output layout injectivity is unproved");
        }
    }

    /**
     * Compact immutable cold geometry for one movement occurrence.
     *
     * @param outputExtents exact output extents; copied defensively
     * @param outputOffset output layout storage offset in elements
     * @param outputStrides exact output layout strides in elements; copied defensively
     * @param inputs unique input layout facts in boundary order; copied defensively
     * @param variant family-specific compact mapping facts
     */
    public record Geometry(long[] outputExtents, long outputOffset, long[] outputStrides,
            List<Input> inputs, Variant variant) {

        /**
         * Numeric movement recipe with fixed structural IR and affine variant cells.
         * @param ir immutable ordinary movement IR
         * @param boundaries unique inputs then output
         * @param boundaryTypes actual represented boundary types
         * @param bindings exact affine boundary layouts
         * @param constants numeric variant cell constants, defensively copied
         * @param batchFactors numeric variant cell B coefficients, defensively copied
         */
        public record BatchGeometry(CpuDataMovementIr ir, List<ValueId> boundaries,
                List<DataType> boundaryTypes, List<CpuAccessPlan.BatchBinding> bindings,
                long[] constants, long[] batchFactors) implements CpuPartitionLowering.BatchUnitGeometry {
            /**
             * Snapshots numeric preparation facts without Model descriptors or runtime storage.
             * @param ir non-null structural IR
             * @param boundaries non-null ordered identities
             * @param boundaryTypes non-null represented types
             * @param bindings non-null affine layouts
             * @param constants non-null numeric variant constants
             * @param batchFactors non-null equally sized B coefficients
             */
            public BatchGeometry {
                Objects.requireNonNull(ir, "ir");
                boundaries = List.copyOf(boundaries); boundaryTypes = List.copyOf(boundaryTypes);
                bindings = List.copyOf(bindings);
                constants = constants.clone(); batchFactors = batchFactors.clone();
                if (constants.length != batchFactors.length) throw invalidBatch();
            }
            /**
             * Returns the constant terms of the movement variant's affine numeric cells.
             * Binding evaluates each cell as its constant plus its coefficient times actual B.
             * @return a new non-null caller-owned copy of the variant constants
             */
            @Override public long[] constants() { return constants.clone(); }
            /**
             * Returns actual-B coefficients paired by index with the variant constants.
             * These are coefficients, not values evaluated at a representative batch.
             * @return a new non-null caller-owned copy of the variant coefficients
             */
            @Override public long[] batchFactors() { return batchFactors.clone(); }
            /** @return exact affine output element count */
            @Override public BatchByteSize iterationCount() { return batchCount(bindings.getLast()); }
            /** @return zero numeric helper scratch */
            @Override public BatchByteSize scratchBytes() { return new BatchByteSize(0, 0, 1); }
            /** @return the original five mutable UNFOLD2D pack cells; other cursors are local */
            @Override public List<PackReset> resets() {
                if (!(ir.plan() instanceof CpuDataMovementIr.Unfold2dPlan)) return List.of();
                int first = 3 * bindings.getLast().plan().iterationRank() + bindings.size();
                for (int b = 0; b + 1 < bindings.size(); b++)
                    first += bindings.get(b).plan().iterationRank();
                var cells = new ArrayList<PackReset>(5);
                for (int n = 0; n < 5; n++) cells.add(new PackReset(first + 13 + n, 0));
                return List.copyOf(cells);
            }
            /** @return B-independent numeric recipe identity without local value identities */
            @Override public String identity() {
                return "movement:" + bindings + ":" + Arrays.toString(constants)
                        + ":" + Arrays.toString(batchFactors);
            }
            /**
             * Resolves ordinary geometry once at the cold numeric boundary.
             * @param batch non-negative actual batch extent
             * @return ordinary concrete lowering retaining the exact prepared IR
             * @throws IllegalArgumentException if actual window or slice bounds are invalid
             * @throws ArithmeticException if checked layout or variant arithmetic overflows
             */
            @Override public CpuPartitionLowering.LoweredPartition bind(long batch) {
                var access = bindings.stream().map(b -> b.bind(batch)).toList();
                var inputs = access.subList(0, access.size() - 1).stream().map(b -> new Input(
                        b.extents().stream().mapToLong(Long::longValue).toArray(), b.baseElementOffset(),
                        b.effectiveStrides().stream().mapToLong(Long::longValue).toArray())).toList();
                var out = access.getLast();
                long[] extents = out.extents().stream().mapToLong(Long::longValue).toArray();
                long[] facts = new long[constants.length];
                for (int n = 0; n < facts.length; n++) facts[n] = Math.addExact(constants[n],
                        Math.multiplyExact(batchFactors[n], batch));
                Variant variant = switch (ir.plan()) {
                    case CpuDataMovementIr.PadPlan ignored -> new Pad(facts, inputs.getFirst().extents);
                    case CpuDataMovementIr.TilePlan ignored -> new Tile(inputs.getFirst().extents, facts);
                    case CpuDataMovementIr.ConcatPlan concat -> {
                        int axis = Math.toIntExact(facts[0]);
                        long[] prefixes = new long[Math.addExact(concat.occurrenceToBoundary().size(), 1)];
                        for (int n = 0; n + 1 < prefixes.length; n++)
                            prefixes[n + 1] = Math.addExact(prefixes[n],
                                    inputs.get(concat.occurrenceToBoundary().get(n)).extents[axis]);
                        yield new Concat(axis, prefixes, concat.occurrenceToBoundary());
                    }
                    case CpuDataMovementIr.StackPlan stack -> new Stack(Math.toIntExact(facts[0]),
                            stack.occurrenceToBoundary());
                    case CpuDataMovementIr.UnfoldAxisPlan ignored -> {
                        int axis = Math.toIntExact(facts[0]);
                        if (facts[1] > inputs.getFirst().extents[axis]) throw invalidBatch();
                        yield new UnfoldAxis(axis, facts[1], facts[2]);
                    }
                    case CpuDataMovementIr.Unfold2dPlan ignored -> {
                        long effectiveH = Math.addExact(Math.multiplyExact(facts[9], facts[3] - 1), 1);
                        long effectiveW = Math.addExact(Math.multiplyExact(facts[10], facts[4] - 1), 1);
                        if (Math.addExact(facts[1], Math.multiplyExact(2, facts[7])) < effectiveH
                                || Math.addExact(facts[2], Math.multiplyExact(2, facts[8])) < effectiveW)
                            throw invalidBatch();
                        yield new Unfold2d(facts[0], facts[1], facts[2], facts[3], facts[4],
                                facts[5], facts[6], facts[7], facts[8], facts[9], facts[10], facts[11], facts[12]);
                    }
                    case CpuDataMovementIr.Unfold3dPlan ignored ->
                            new Unfold3d(CpuSpatialWindowGeometry.fromCells(facts));
                    case CpuDataMovementIr.SliceUpdatePlan slice -> {
                        int rank = extents.length;
                        long[] starts = Arrays.copyOf(facts, rank);
                        long[] steps = Arrays.copyOfRange(facts, rank, 2 * rank);
                        long[] lengths = inputs.get(slice.occurrenceToBoundary().get(1)).extents;
                        if (facts[2 * rank] != 0) {
                            for (int a = 0; a < rank; a++)
                                if (Math.addExact(starts[a], lengths[a]) > extents[a]) throw invalidBatch();
                        }
                        for (int a = 0; a < rank; a++) if (lengths[a] > 0) {
                            long last = Math.addExact(starts[a], Math.multiplyExact(lengths[a] - 1, steps[a]));
                            if (starts[a] < 0 || starts[a] >= extents[a] || last < 0 || last >= extents[a])
                                throw invalidBatch();
                        }
                        yield new SliceUpdate(starts, lengths, steps);
                    }
                };
                var geometry = new Geometry(extents, out.baseElementOffset(),
                        out.effectiveStrides().stream().mapToLong(Long::longValue).toArray(), inputs, variant);
                return new CpuPartitionLowering.LoweredPartition(ir, boundaries, access,
                        access.stream().map(CpuAccessPlan.Binding::referencedElementSpan).toList(),
                        boundaryTypes, List.of(), extents, out.elementCount(),
                        "legal: bound symbolic movement", new long[0], Optional.of(geometry));
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
            int rank = outputExtents.length;
            // Cold-only reuse keeps variant cursor vocabulary with the original pack owner.
            long[] seed = pack(new long[inputs.size() + 1], start, start);
            var bases = new ArrayList<BaseCell>(inputs.size() + 1);
            for (int i = 0; i < inputs.size(); i++)
                bases.add(new BaseCell(3 * rank + 1 + i, i, inputs.get(i).offset));
            bases.add(new BaseCell(2 * rank, inputs.size(), seed[2 * rank]));
            var resets = new ArrayList<ResetCell>();
            int variantStart = 3 * rank + 1 + inputs.size();
            for (Input input : inputs) variantStart += input.extents.length;
            // Other variants initialize local cursors from immutable pack seeds on entry.
            // UNFOLD2D alone advances these five cursor cells in the pack itself.
            if (variant instanceof Unfold2d) {
                for (int i = 0; i < 5; i++)
                    resets.add(new ResetCell(variantStart + 13 + i, seed[variantStart + 13 + i]));
            }
            return new TemporalRebind(bases, resets);
        }

        /**
         * One unique input's exact static layout geometry.
         *
         * @param extents exact non-negative static extents; copied defensively
         * @param offset non-negative layout storage offset in elements
         * @param strides exact layout strides in elements; copied defensively
         */
        public record Input(long[] extents, long offset, long[] strides) {
            /** Validates and snapshots one input geometry. */
            public Input {
                extents = extents.clone();
                strides = strides.clone();
                if (offset < 0 || extents.length != strides.length) {
                    throw new IllegalArgumentException("movement input geometry is inconsistent");
                }
            }
            /**
             * Returns isolated input extents.
             * @return a defensive copy of the input extents
             */
            @Override public long[] extents() { return extents.clone(); }
            /**
             * Returns isolated input strides.
             * @return a defensive copy of the input strides
             */
            @Override public long[] strides() { return strides.clone(); }
        }
        /**
         * Closed family-specific cold mapping geometry. Implementations contain checked
         * occurrence facts only and are never exposed to Runtime hot-path semantic inspection.
         */
        public sealed interface Variant permits Pad, Tile, Concat, Stack, UnfoldAxis, Unfold2d,
                Unfold3d, SliceUpdate { }
        /**
         * Constant-padding widths and source extents.
         *
         * @param before non-negative before-width per output axis; copied defensively
         * @param inputExtents exact source extent per output axis; copied defensively
         */
        public record Pad(long[] before, long[] inputExtents) implements Variant {
            /** Snapshots the exact rank-matched arrays. */
            public Pad { before = before.clone(); inputExtents = inputExtents.clone(); }
            /**
             * Returns isolated before-widths.
             * @return a defensive copy
             */
            @Override public long[] before() { return before.clone(); }
            /**
             * Returns isolated input extents.
             * @return a defensive copy
             */
            @Override public long[] inputExtents() { return inputExtents.clone(); }
        }
        /**
         * Complete-pattern tiling extents and repeats.
         *
         * @param inputExtents exact source extent per output axis; copied defensively
         * @param repeats positive complete-pattern repeat count per axis; copied defensively
         */
        public record Tile(long[] inputExtents, long[] repeats) implements Variant {
            /** Snapshots the exact rank-matched arrays. */
            public Tile { inputExtents = inputExtents.clone(); repeats = repeats.clone(); }
            /**
             * Returns isolated input extents.
             * @return a defensive copy
             */
            @Override public long[] inputExtents() { return inputExtents.clone(); }
            /**
             * Returns isolated repeat counts.
             * @return a defensive copy
             */
            @Override public long[] repeats() { return repeats.clone(); }
        }
        /**
         * Concatenation axis and encounter-order segment prefixes.
         *
         * @param axis normalized existing output axis
         * @param prefixes ordered cumulative segment starts, including the final total extent;
         *     copied defensively
         * @param occurrenceToBoundary ordered occurrence bindings, copied defensively
         */
        public record Concat(int axis, long[] prefixes, List<Integer> occurrenceToBoundary) implements Variant {
            /** Snapshots the exact segment prefix array and occurrence bindings. */
            public Concat {
                prefixes = prefixes.clone();
                occurrenceToBoundary = List.copyOf(occurrenceToBoundary);
                if (prefixes.length != Math.addExact(occurrenceToBoundary.size(), 1))
                    throw new IllegalArgumentException("concat prefixes and occurrences disagree");
            }
            /**
             * Returns isolated prefixes.
             * @return a defensive copy
             */
            @Override public long[] prefixes() { return prefixes.clone(); }
        }
        /**
         * Stack insertion axis.
         *
         * @param axis normalized inserted output axis
         * @param occurrenceToBoundary ordered occurrence bindings, copied defensively
         */
        public record Stack(int axis, List<Integer> occurrenceToBoundary) implements Variant {
            /** Snapshots the ordered occurrence bindings. */
            public Stack { occurrenceToBoundary = List.copyOf(occurrenceToBoundary); }
        }
        /**
         * General-axis window facts used with an output odometer.
         * @param axis normalized input axis
         * @param size positive window extent
         * @param step positive distance between window starts
         */
        public record UnfoldAxis(int axis, long size, long step) implements Variant { }
        /**
         * Canonical NCHW window and output-grid facts used to seed generated odometers. Padding
         * values remain structural IR bits because they shape emitted instructions.
         *
         * @param channels input channel extent
         * @param height input height extent
         * @param width input width extent
         * @param kernelHeight positive kernel height
         * @param kernelWidth positive kernel width
         * @param strideHeight positive height stride
         * @param strideWidth positive width stride
         * @param paddingHeight non-negative symmetric height padding
         * @param paddingWidth non-negative symmetric width padding
         * @param dilationHeight positive height dilation
         * @param dilationWidth positive width dilation
         * @param outputHeight positive checked output-grid height
         * @param outputWidth positive checked output-grid width
         */
        public record Unfold2d(long channels, long height, long width,
                long kernelHeight, long kernelWidth, long strideHeight, long strideWidth,
                long paddingHeight, long paddingWidth, long dilationHeight, long dilationWidth,
                long outputHeight, long outputWidth) implements Variant { }
        /**
         * Shared checked spatial window facts for volumetric extraction.
         * @param spatial non-null immutable checked geometry
         */
        public record Unfold3d(CpuSpatialWindowGeometry spatial) implements Variant {
            /** @throws NullPointerException if spatial is null */
            public Unfold3d { Objects.requireNonNull(spatial, "spatial"); }
        }
        /**
         * Normalized finite sequence placement on every result axis.
         * @param starts inclusive logical starts; copied defensively
         * @param lengths finite non-negative sequence lengths; copied defensively
         * @param steps signed non-zero sequence steps; copied defensively
         */
        public record SliceUpdate(long[] starts, long[] lengths, long[] steps) implements Variant {
            /**
             * Validates rank agreement and snapshots all arrays.
             *
             * @param starts non-null inclusive logical starts, one per result axis
             * @param lengths non-null finite non-negative lengths, one per result axis
             * @param steps non-null signed non-zero steps, one per result axis
             * @throws NullPointerException if an array is {@code null}
             * @throws IllegalArgumentException if the array ranks differ
             */
            public SliceUpdate {
                starts = starts.clone(); lengths = lengths.clone(); steps = steps.clone();
                if (starts.length != lengths.length || starts.length != steps.length) {
                    throw new IllegalArgumentException("slice-update geometry ranks must match");
                }
            }
            /**
             * Returns isolated inclusive logical starts.
             *
             * @return a defensive copy with one start per result axis
             */
            @Override public long[] starts() { return starts.clone(); }
            /**
             * Returns isolated finite sequence lengths.
             *
             * @return a defensive copy with one non-negative length per result axis
             */
            @Override public long[] lengths() { return lengths.clone(); }
            /**
             * Returns isolated signed sequence steps.
             *
             * @return a defensive copy with one non-zero step per result axis
             */
            @Override public long[] steps() { return steps.clone(); }
        }

        /** Validates and snapshots the compact occurrence geometry. */
        public Geometry {
            outputExtents = outputExtents.clone();
            outputStrides = outputStrides.clone();
            inputs = List.copyOf(inputs);
            Objects.requireNonNull(variant, "variant");
            if (outputOffset < 0 || outputExtents.length != outputStrides.length
                    || inputs.isEmpty()) {
                throw new IllegalArgumentException("movement geometry is inconsistent");
            }
            List<Integer> occurrences = switch (variant) {
                case Concat concat -> concat.occurrenceToBoundary;
                case Stack stack -> stack.occurrenceToBoundary;
                default -> List.of(0);
            };
            int inputCount = inputs.size();
            if (occurrences.isEmpty() || occurrences.stream().anyMatch(b -> b < 0 || b >= inputCount))
                throw new IllegalArgumentException("movement occurrence boundary is out of range");
        }
        /**
         * Returns isolated output extents.
         * @return a defensive copy of output extents
         */
        @Override public long[] outputExtents() { return outputExtents.clone(); }
        /**
         * Returns isolated output strides.
         * @return a defensive copy of output strides
         */
        @Override public long[] outputStrides() { return outputStrides.clone(); }

        /**
         * Packs one range's primitive start state for the generated entry.
         *
         * @param carrierBases exact unique-input then output carrier-relative element bases
         * @param start inclusive output logical range bound
         * @param end exclusive output logical range bound
         * @return one new compact primitive geometry array
         * @throws NullPointerException if {@code carrierBases} is {@code null}
         * @throws IllegalArgumentException if the carrier count or half-open range is invalid
         * @throws ArithmeticException if range coordinates or addresses overflow
         */
        public long[] pack(long[] carrierBases, long start, long end) {
            Objects.requireNonNull(carrierBases, "carrierBases");
            int rank = outputExtents.length;
            int inputStrideCount = 0;
            for (Input input : inputs) inputStrideCount = Math.addExact(inputStrideCount, input.extents.length);
            int variantSize = switch (variant) {
                case Pad ignored -> 2 * rank;
                case Tile ignored -> 2 * rank;
                case Concat concat -> Math.addExact(1, concat.prefixes.length);
                case Stack ignored -> 1;
                case UnfoldAxis ignored -> 3;
                case Unfold2d ignored -> 18;
                case Unfold3d ignored -> 18;
                case SliceUpdate ignored -> 6 * rank;
            };
            long elementCount = 1;
            for (long extent : outputExtents) elementCount = Math.multiplyExact(elementCount, extent);
            if (carrierBases.length != inputs.size() + 1 || start < 0 || end < start
                    || end > elementCount) {
                throw new IllegalArgumentException("movement range geometry is inconsistent");
            }
            long[] coordinates = coordinates(start, outputExtents);
            List<Integer> tableMap = switch (variant) {
                case Concat concat -> concat.occurrenceToBoundary;
                case Stack stack -> stack.occurrenceToBoundary;
                default -> List.of();
            };
            int tableSize = tableMap.size() > 16 ? tableMap.size() : 0;
            int packedSize = Math.addExact(Math.addExact(Math.addExact(
                    Math.addExact(Math.multiplyExact(3, rank), 1), inputs.size()),
                    inputStrideCount), Math.addExact(variantSize, tableSize));
            long[] packed = new long[packedSize];
            int cursor = 0;
            for (long value : outputExtents) packed[cursor++] = value;
            for (long value : coordinates) packed[cursor++] = value;
            long outputAddress = Math.addExact(carrierBases[inputs.size()], outputOffset);
            for (int axis = 0; axis < rank; axis++) outputAddress = Math.addExact(outputAddress,
                    Math.multiplyExact(coordinates[axis], outputStrides[axis]));
            packed[cursor++] = outputAddress;
            for (long value : outputStrides) packed[cursor++] = value;
            for (int input = 0; input < inputs.size(); input++) {
                packed[cursor++] = Math.addExact(carrierBases[input], inputs.get(input).offset);
            }
            for (Input input : inputs) for (long stride : input.strides) packed[cursor++] = stride;
            switch (variant) {
                case Pad pad -> {
                    for (long value : pad.before) packed[cursor++] = value;
                    for (long value : pad.inputExtents) packed[cursor++] = value;
                }
                case Tile tile -> {
                    for (long value : tile.inputExtents) packed[cursor++] = value;
                    for (int axis = 0; axis < rank; axis++) {
                        packed[cursor++] = tile.inputExtents[axis] == 0
                                ? 0 : coordinates[axis] % tile.inputExtents[axis];
                    }
                }
                case Concat concat -> {
                    packed[cursor++] = concat.axis;
                    for (long value : concat.prefixes) packed[cursor++] = value;
                }
                case Stack stack -> packed[cursor++] = stack.axis;
                case UnfoldAxis unfold -> {
                    packed[cursor++] = unfold.axis;
                    packed[cursor++] = unfold.size;
                    packed[cursor++] = unfold.step;
                }
                case Unfold2d unfold -> {
                    packed[cursor++] = unfold.channels;
                    packed[cursor++] = unfold.height;
                    packed[cursor++] = unfold.width;
                    packed[cursor++] = unfold.kernelHeight;
                    packed[cursor++] = unfold.kernelWidth;
                    packed[cursor++] = unfold.strideHeight;
                    packed[cursor++] = unfold.strideWidth;
                    packed[cursor++] = unfold.paddingHeight;
                    packed[cursor++] = unfold.paddingWidth;
                    packed[cursor++] = unfold.dilationHeight;
                    packed[cursor++] = unfold.dilationWidth;
                    packed[cursor++] = unfold.outputHeight;
                    packed[cursor++] = unfold.outputWidth;
                    long q = coordinates[1], p = coordinates[2];
                    packed[cursor++] = q / Math.multiplyExact(unfold.kernelHeight,
                            unfold.kernelWidth);
                    packed[cursor++] = q / unfold.kernelWidth % unfold.kernelHeight;
                    packed[cursor++] = q % unfold.kernelWidth;
                    packed[cursor++] = p / unfold.outputWidth;
                    packed[cursor++] = p % unfold.outputWidth;
                }
                case Unfold3d unfold -> {
                    for (long cell : unfold.spatial.cells()) packed[cursor++] = cell;
                    long[] image = inputs.getFirst().extents;
                    for (int a = 2; a < 5; a++) packed[cursor++] = image[a];
                }
                case SliceUpdate slice -> {
                    long[] lowTargets = new long[rank];
                    long[] lowOrdinals = new long[rank];
                    long[] currentTargets = new long[rank];
                    long[] currentOrdinals = new long[rank];
                    for (int axis = 0; axis < rank; axis++) {
                        long length = slice.lengths[axis], step = slice.steps[axis];
                        if (length == 0) {
                            lowTargets[axis] = currentTargets[axis] = -1;
                            lowOrdinals[axis] = currentOrdinals[axis] = -1;
                            continue;
                        }
                        long lowOrdinal = step > 0 ? 0 : length - 1;
                        long lowTarget = step > 0 ? slice.starts[axis] : Math.addExact(
                                slice.starts[axis], Math.multiplyExact(length - 1, step));
                        lowTargets[axis] = lowTarget;
                        lowOrdinals[axis] = lowOrdinal;
                        long delta = coordinates[axis] <= lowTarget ? 0
                                : coordinates[axis] - lowTarget;
                        long increment = length == 1 ? 1 : step > 0 ? step : -step;
                        long advances = delta / increment + (delta % increment == 0 ? 0 : 1);
                        if (advances >= length) {
                            currentTargets[axis] = currentOrdinals[axis] = -1;
                        } else {
                            currentTargets[axis] = Math.addExact(lowTarget,
                                    Math.multiplyExact(advances, increment));
                            currentOrdinals[axis] = step > 0 ? advances : length - 1 - advances;
                        }
                    }
                    for (long value : lowTargets) packed[cursor++] = value;
                    for (long value : lowOrdinals) packed[cursor++] = value;
                    for (long value : slice.lengths) packed[cursor++] = value;
                    for (long value : slice.steps) packed[cursor++] = value;
                    for (long value : currentTargets) packed[cursor++] = value;
                    for (long value : currentOrdinals) packed[cursor++] = value;
                }
            }
            if (tableSize != 0) for (int boundary : tableMap) packed[cursor++] = boundary;
            return packed;
        }

        private static long[] coordinates(long logical, long[] extents) {
            long[] result = new long[extents.length];
            for (int axis = extents.length - 1; axis >= 0; axis--) {
                if (extents[axis] != 0) {
                    result[axis] = logical % extents[axis];
                    logical /= extents[axis];
                }
            }
            return result;
        }
    }
}
