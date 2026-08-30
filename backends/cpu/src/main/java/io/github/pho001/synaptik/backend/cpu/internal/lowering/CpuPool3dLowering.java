package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPool3dIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.Pool3dKind;
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

/** Cold checked lowering for direct static NCDHW max and fixed-divisor average Pool3d. */
public final class CpuPool3dLowering {
    private final CpuCapabilityProvider capabilities = new CpuCapabilityProvider();

    /** Creates a stateless Pool3d lowerer. */
    public CpuPool3dLowering() {}

    /**
     * Lowers one supported Pool3d occurrence to complete-output-cell geometry.
     *
     * @param context non-null single-node CPU preparation projection
     * @return immutable direct scalar lowering with no workspace or materialization
     * @throws NullPointerException if {@code context} is null
     * @throws IllegalArgumentException if semantics, descriptors, layouts, or geometry disagree
     * @throws ArithmeticException if checked count, divisor, or address arithmetic overflows
     */
    public CpuPartitionLowering.LoweredPartition lower(
            PrepareContext<? extends BackendAnalysisInputs> context) {
        Objects.requireNonNull(context, "context");
        if (context.nodes().size() != 1)
            throw new IllegalArgumentException("CPU Pool3d requires exactly one occurrence");
        var node = context.nodes().getFirst();
        Map<ValueId, GraphValue> values = new LinkedHashMap<>();
        context.values().forEach(value -> values.put(value.id(), value));
        var query = new OperationCapabilityQuery(node.operation(),
                node.inputs().stream().map(id -> require(values, id).descriptor()).toList(),
                node.outputs().stream().map(id -> require(values, id).descriptor()).toList());
        if (!capabilities.supports(query) || node.inputs().size() != 1
                || node.outputs().size() != 1)
            throw new IllegalArgumentException("unsupported CPU Pool3d occurrence");

        Params p = params(node.operation().kind(), node.operation().attrs());
        GraphValue input = require(values, node.inputs().getFirst());
        GraphValue output = require(values, node.outputs().getFirst());
        Layout in = layout(input), out = layout(output);
        long[] ie = in.extents(), oe = out.extents();
        long od = outputExtent(ie[2], p.kd(), p.pd(), p.dd(), p.sd(), p.ceil());
        long oh = outputExtent(ie[3], p.kh(), p.ph(), p.dh(), p.sh(), p.ceil());
        long ow = outputExtent(ie[4], p.kw(), p.pw(), p.dw(), p.sw(), p.ceil());
        if (!Arrays.equals(oe, new long[] {ie[0], ie[1], od, oh, ow}))
            throw new IllegalArgumentException("Pool3d output shape disagrees");
        long divisor = Math.multiplyExact(Math.multiplyExact(p.kd(), p.kh()), p.kw());
        long count = count(oe);
        var inputBinding = binding(in, CpuAccessPlan.AccessKind.READ);
        var outputBinding = binding(out, CpuAccessPlan.AccessKind.WRITE);
        var ir = new CpuPool3dIr(p.kind(), input.descriptor().dataType(),
                CpuPool3dIr.Realization.DIRECT_SCALAR, inputBinding.plan(), outputBinding.plan());
        var geometry = new Geometry(p.kind(), input.descriptor().dataType(), in, out,
                p.kd(), p.kh(), p.kw(), p.sd(), p.sh(), p.sw(), p.pd(), p.ph(), p.pw(),
                p.dd(), p.dh(), p.dw(), divisor, count);
        return new CpuPartitionLowering.LoweredPartition(ir,
                List.of(node.inputs().getFirst(), node.outputs().getFirst()),
                List.of(inputBinding, outputBinding),
                List.of(input.descriptor().layout().orElseThrow().referencedElementSpan(),
                        output.descriptor().layout().orElseThrow().referencedElementSpan()),
                List.of(input.descriptor().dataType(), output.descriptor().dataType()), List.of(),
                oe, count, "legal: direct static NCDHW " + p.kind() + " Pool3d", new long[0],
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.of(geometry));
    }

    private static Params params(Object kind, Object attrs) {
        if (kind == Pool3dKind.MAX_POOL3D && attrs instanceof MaxPool3dAttrs a)
            return new Params(CpuPool3dIr.Kind.MAX, a.kernelDepth(), a.kernelHeight(),
                    a.kernelWidth(), a.strideDepth(), a.strideHeight(), a.strideWidth(),
                    a.paddingDepth(), a.paddingHeight(), a.paddingWidth(), a.dilationDepth(),
                    a.dilationHeight(), a.dilationWidth(), a.ceilMode());
        if (kind == Pool3dKind.AVERAGE_POOL3D && attrs instanceof AveragePool3dAttrs a)
            return new Params(CpuPool3dIr.Kind.AVERAGE, a.kernelDepth(), a.kernelHeight(),
                    a.kernelWidth(), a.strideDepth(), a.strideHeight(), a.strideWidth(),
                    a.paddingDepth(), a.paddingHeight(), a.paddingWidth(), a.dilationDepth(),
                    a.dilationHeight(), a.dilationWidth(), a.ceilMode());
        throw new IllegalArgumentException("Pool3d kind and attributes disagree");
    }

    private static long outputExtent(long d, long k, long p, long dilation, long stride,
            boolean ceil) {
        long effective = Math.addExact(Math.multiplyExact(dilation, Math.subtractExact(k, 1)), 1);
        long numerator = Math.subtractExact(Math.addExact(d, Math.multiplyExact(2, p)), effective);
        if (numerator < 0)
            throw new IllegalArgumentException("Pool3d padded extent is smaller than effective kernel");
        long q = numerator / stride;
        if (ceil && numerator % stride != 0) q = Math.addExact(q, 1);
        return Math.addExact(q, 1);
    }

    private static GraphValue require(Map<ValueId, GraphValue> values, ValueId id) {
        GraphValue value = values.get(id);
        if (value == null) throw new IllegalArgumentException("Pool3d value is not projected: " + id);
        return value;
    }

    private static Layout layout(GraphValue value) {
        var descriptor = value.descriptor().layout().orElseThrow();
        long[] extents = value.descriptor().shape().toLongArray();
        if (extents.length != 5 || descriptor.storageOffset() < 0
                || Arrays.stream(descriptor.strides()).anyMatch(stride -> stride < 0))
            throw new IllegalArgumentException("Pool3d requires rank-five non-negative resolved layout");
        return new Layout(extents, descriptor.storageOffset(), descriptor.strides());
    }

    private static CpuAccessPlan.Binding binding(Layout layout, CpuAccessPlan.AccessKind kind) {
        long[] e = layout.extents(), s = layout.strides();
        int suffix = 0;
        long expected = 1;
        for (int axis = 4; axis >= 0; axis--) {
            if (s[axis] != expected) break;
            suffix++;
            expected = Math.multiplyExact(expected, Math.max(1, e[axis]));
        }
        var roles = new ArrayList<CpuAccessPlan.AxisRole>();
        for (int axis = 0; axis < 5; axis++) roles.add(s[axis] == 0
                ? CpuAccessPlan.AxisRole.BROADCAST
                : axis >= 5 - suffix ? CpuAccessPlan.AxisRole.CONTIGUOUS
                        : CpuAccessPlan.AxisRole.STRIDED);
        var plan = new CpuAccessPlan(kind,
                suffix == 5 ? CpuAccessPlan.Regime.DENSE_LINEAR
                        : CpuAccessPlan.Regime.GENERAL_ODOMETER,
                5, roles, suffix);
        long count = count(e);
        return CpuAccessPlan.Binding.create(plan, e, layout.offset(), s, count, 0, count,
                span(layout));
    }

    private static long count(long[] extents) {
        for (long extent : extents) if (extent == 0) return 0;
        long result = 1;
        for (long extent : extents) result = Math.multiplyExact(result, extent);
        return result;
    }

    private static long span(Layout layout) {
        if (count(layout.extents()) == 0) return 0;
        long last = layout.offset();
        long[] e = layout.extents(), s = layout.strides();
        for (int axis = 0; axis < 5; axis++)
            last = Math.addExact(last, Math.multiplyExact(e[axis] - 1, s[axis]));
        return Math.addExact(last, 1);
    }

    private record Params(CpuPool3dIr.Kind kind, long kd, long kh, long kw, long sd, long sh,
            long sw, long pd, long ph, long pw, long dd, long dh, long dw, boolean ceil) {}

    /**
     * Immutable logical rank-five layout.
     *
     * @param extents five non-negative logical extents; copied defensively
     * @param offset non-negative carrier-relative element offset
     * @param strides five non-negative element strides; copied defensively
     */
    public record Layout(long[] extents, long offset, long[] strides) {
        /**
         * Snapshots one non-negative resolved NCDHW layout.
         * @param extents five non-negative logical extents
         * @param offset non-negative carrier-relative element offset
         * @param strides five non-negative element strides
         * @throws NullPointerException if an array is null
         * @throws IllegalArgumentException if rank or non-negative constraints disagree
         */
        public Layout {
            extents = extents.clone();
            strides = strides.clone();
            if (extents.length != 5 || strides.length != 5 || offset < 0
                    || Arrays.stream(extents).anyMatch(x -> x < 0)
                    || Arrays.stream(strides).anyMatch(x -> x < 0))
                throw new IllegalArgumentException("invalid Pool3d layout");
        }
        /** Returns a defensive snapshot of the logical extents.
         * @return a fresh five-element extent array
         */
        @Override public long[] extents() { return extents.clone(); }
        /** Returns a defensive snapshot of the element strides.
         * @return a fresh five-element stride array
         */
        @Override public long[] strides() { return strides.clone(); }
    }

    /**
     * Complete immutable rank-five invocation geometry.
     *
     * @param kind max or fixed-divisor-average semantics
     * @param dataType exact floating input/output representation
     * @param input resolved non-negative NCDHW input layout
     * @param output resolved injective NCDHW output layout
     * @param kernelDepth positive depth window size
     * @param kernelHeight positive height window size
     * @param kernelWidth positive width window size
     * @param strideDepth positive depth step
     * @param strideHeight positive height step
     * @param strideWidth positive width step
     * @param paddingDepth non-negative symmetric depth padding
     * @param paddingHeight non-negative symmetric height padding
     * @param paddingWidth non-negative symmetric width padding
     * @param dilationDepth positive depth dilation
     * @param dilationHeight positive height dilation
     * @param dilationWidth positive width dilation
     * @param divisor checked positive kernel-volume divisor
     * @param outputCount checked product of the output extents
     */
    public record Geometry(CpuPool3dIr.Kind kind, DataType dataType, Layout input, Layout output,
            long kernelDepth, long kernelHeight, long kernelWidth, long strideDepth,
            long strideHeight, long strideWidth, long paddingDepth, long paddingHeight,
            long paddingWidth, long dilationDepth, long dilationHeight, long dilationWidth,
            long divisor, long outputCount) {
        /**
         * Validates exact intrinsic geometry and output-cell ownership.
         * @param kind max or fixed-divisor average semantics
         * @param dataType exact floating input/output representation
         * @param input resolved non-negative NCDHW input layout
         * @param output resolved injective NCDHW output layout
         * @param kernelDepth positive depth window size
         * @param kernelHeight positive height window size
         * @param kernelWidth positive width window size
         * @param strideDepth positive depth step
         * @param strideHeight positive height step
         * @param strideWidth positive width step
         * @param paddingDepth non-negative symmetric depth padding
         * @param paddingHeight non-negative symmetric height padding
         * @param paddingWidth non-negative symmetric width padding
         * @param dilationDepth positive depth dilation
         * @param dilationHeight positive height dilation
         * @param dilationWidth positive width dilation
         * @param divisor checked {@code kernelDepth * kernelHeight * kernelWidth}
         * @param outputCount checked product of the output extents
         * @throws NullPointerException if a required reference is null
         * @throws IllegalArgumentException if intrinsic or output-count facts disagree
         * @throws ArithmeticException if divisor or output count validation overflows
         */
        public Geometry {
            Objects.requireNonNull(kind, "kind"); Objects.requireNonNull(dataType, "dataType");
            Objects.requireNonNull(input, "input"); Objects.requireNonNull(output, "output");
            if (dataType != DataType.BFLOAT16 && dataType != DataType.FLOAT32
                    && dataType != DataType.FLOAT64 || kernelDepth <= 0 || kernelHeight <= 0
                    || kernelWidth <= 0 || strideDepth <= 0 || strideHeight <= 0
                    || strideWidth <= 0 || paddingDepth < 0 || paddingHeight < 0
                    || paddingWidth < 0 || dilationDepth <= 0 || dilationHeight <= 0
                    || dilationWidth <= 0
                    || divisor != Math.multiplyExact(Math.multiplyExact(kernelDepth, kernelHeight),
                            kernelWidth)
                    || outputCount != count(output.extents()))
                throw new IllegalArgumentException("Pool3d geometry disagrees");
        }

        /**
         * Packs bases, layouts, and intrinsic geometry for the generated entry.
         * @param inputBase carrier-relative input element base
         * @param outputBase carrier-relative output element base
         * @return fresh compact primitive geometry
         * @throws ArithmeticException if adding either carrier base overflows
         */
        public long[] pack(long inputBase, long outputBase) {
            long[] a = new long[36];
            a[0] = Math.addExact(inputBase, input.offset());
            a[1] = Math.addExact(outputBase, output.offset());
            System.arraycopy(input.extents(), 0, a, 2, 5);
            System.arraycopy(input.strides(), 0, a, 7, 5);
            System.arraycopy(output.extents(), 0, a, 12, 5);
            System.arraycopy(output.strides(), 0, a, 17, 5);
            a[22] = kernelDepth; a[23] = kernelHeight; a[24] = kernelWidth;
            a[25] = strideDepth; a[26] = strideHeight; a[27] = strideWidth;
            a[28] = paddingDepth; a[29] = paddingHeight; a[30] = paddingWidth;
            a[31] = dilationDepth; a[32] = dilationHeight; a[33] = dilationWidth;
            a[34] = divisor; a[35] = outputCount;
            return a;
        }
    }
}
