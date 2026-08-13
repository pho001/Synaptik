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
 * Lowers exactly one fully static resolved-layout FOLD_AXIS or FOLD2D occurrence.
 * The lowerer revalidates the Model signature, derives compact checked coordinate geometry, and
 * proves a distinct injective output. It allocates no Runtime resource and declares no workspace.
 */
public final class CpuFoldLowering {
    private final CpuCapabilityProvider capabilities = new CpuCapabilityProvider();

    /** Creates a stateless fold lowerer. */
    public CpuFoldLowering() { }

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
     */
    public record Geometry(CpuFoldIr.Family family, DataType dataType, Layout input, Layout output,
            Mapping mapping) {
        /**
         * Creates one immutable family-specific fold mapping for a prepared recipe.
         *
         * @param family non-null current fold family
         * @param dataType non-null supported represented input and output type
         * @param input non-null resolved input layout
         * @param output non-null resolved output layout
         * @param mapping non-null family-specific mapping matching the family and ranks
         * @throws NullPointerException if a reference is {@code null}
         * @throws IllegalArgumentException if family, type, rank, or mapping facts disagree
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
                        || dataType == DataType.INT32 || dataType == DataType.INT64)) {
                throw new IllegalArgumentException("fold geometry facts disagree");
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
