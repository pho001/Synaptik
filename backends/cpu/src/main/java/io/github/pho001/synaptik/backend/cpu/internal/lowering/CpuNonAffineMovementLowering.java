package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuDataMovementIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.layout.CompositionAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.PadAttrs;
import io.github.pho001.synaptik.model.operation.layout.PadKind;
import io.github.pho001.synaptik.model.operation.layout.TensorCompositionKind;
import io.github.pho001.synaptik.model.operation.layout.TileAttrs;
import io.github.pho001.synaptik.model.operation.layout.TileKind;
import io.github.pho001.synaptik.model.operation.layout.Unfold2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.UnfoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.Window2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.WindowTransformKind;
import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Lowers one fully static {@code PAD}, {@code TILE}, {@code CONCAT}, {@code STACK},
 * {@code UNFOLD_AXIS}, or {@code UNFOLD2D}
 * occurrence into one represented-bit movement unit.
 *
 * <p>The lowering preserves semantic input occurrence order while declaring each distinct input
 * value once in first-occurrence order. Exact extents, layout offsets and strides, padding,
 * repeats, composition prefixes, and window facts remain in compact immutable {@link Geometry};
 * no prepared address or selector table is retained per output element. General-axis unfold
 * copies all six represented types. NCHW two-dimensional unfold copies only FLOAT64, FLOAT32,
 * and BFLOAT16 and retains exact configured padding bits; direct attributes use represented
 * positive zero.</p>
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
        if (occurrences.isEmpty() || occurrences.size() > 16) {
            throw new IllegalArgumentException("movement requires one through sixteen inputs");
        }

        Object kind = node.operation().kind();
        Object attrs = node.operation().attrs();
        CpuDataMovementIr.MovementPlan movementPlan;
        Geometry.Variant variant;
        if (kind == PadKind.PAD && attrs instanceof PadAttrs pad) {
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
            long[] prefixes = new long[occurrences.size() + 1];
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
            variant = new Geometry.Concat(axis, prefixes);
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
            variant = new Geometry.Stack(axis);
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
            if (type != DataType.FLOAT64 && type != DataType.FLOAT32 && type != DataType.BFLOAT16) {
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
        if (Arrays.stream(extents).anyMatch(extent -> extent == 0)) return;
        for (int axis = 0; axis < extents.length; axis++) {
            if (strides[axis] == 0 && extents[axis] > 1) {
                throw new IllegalArgumentException("movement output layout is not injective");
            }
        }
        long count = 1;
        for (long extent : extents) count = Math.multiplyExact(count, extent);
        if (count <= 1_000_000) {
            var addresses = new HashSet<Long>();
            long[] coordinates = new long[extents.length];
            for (long logical = 0; logical < count; logical++) {
                long address = 0;
                for (int axis = 0; axis < extents.length; axis++) address = Math.addExact(address,
                        Math.multiplyExact(coordinates[axis], strides[axis]));
                if (!addresses.add(address)) {
                    throw new IllegalArgumentException("movement output layout is not injective");
                }
                advanceCoordinates(coordinates, extents);
            }
            return;
        }
        var axes = new ArrayList<Integer>();
        for (int axis = 0; axis < extents.length; axis++) if (extents[axis] > 1) axes.add(axis);
        axes.sort(java.util.Comparator.comparingLong(axis -> strides[axis]));
        long covered = 1;
        for (int axis : axes) {
            if (strides[axis] < covered) {
                throw new IllegalArgumentException("movement output injectivity is ambiguous");
            }
            covered = Math.addExact(covered,
                    Math.multiplyExact(extents[axis] - 1, strides[axis]));
        }
    }

    private static void advanceCoordinates(long[] coordinates, long[] extents) {
        for (int axis = coordinates.length - 1; axis >= 0; axis--) {
            if (++coordinates[axis] < extents[axis]) return;
            coordinates[axis] = 0;
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
        public sealed interface Variant permits Pad, Tile, Concat, Stack, UnfoldAxis, Unfold2d { }
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
         */
        public record Concat(int axis, long[] prefixes) implements Variant {
            /** Snapshots the exact segment prefix array. */
            public Concat { prefixes = prefixes.clone(); }
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
         */
        public record Stack(int axis) implements Variant { }
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

        /** Validates and snapshots the compact occurrence geometry. */
        public Geometry {
            outputExtents = outputExtents.clone();
            outputStrides = outputStrides.clone();
            inputs = List.copyOf(inputs);
            Objects.requireNonNull(variant, "variant");
            if (outputOffset < 0 || outputExtents.length != outputStrides.length
                    || inputs.isEmpty() || inputs.size() > 16) {
                throw new IllegalArgumentException("movement geometry is inconsistent");
            }
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
            int inputStrideCount = inputs.stream().mapToInt(input -> input.extents.length).sum();
            int variantSize = switch (variant) {
                case Pad ignored -> 2 * rank;
                case Tile ignored -> 2 * rank;
                case Concat concat -> 1 + concat.prefixes.length;
                case Stack ignored -> 1;
                case UnfoldAxis ignored -> 3;
                case Unfold2d ignored -> 18;
            };
            long elementCount = 1;
            for (long extent : outputExtents) elementCount = Math.multiplyExact(elementCount, extent);
            if (carrierBases.length != inputs.size() + 1 || start < 0 || end < start
                    || end > elementCount) {
                throw new IllegalArgumentException("movement range geometry is inconsistent");
            }
            long[] coordinates = coordinates(start, outputExtents);
            long[] packed = new long[3 * rank + 1 + inputs.size()
                    + inputStrideCount + variantSize];
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
            }
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
