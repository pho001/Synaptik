package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPool2dIr;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.Pool2dKind;
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
 * Performs cold checked lowering for static or one-named-batch channels-first (NCHW) max
 * and fixed-divisor average Pool2d occurrences, preserving proved physical layout recipes.
 *
 * <p>The lowerer revalidates the literal floor or ceiling output grid, checked addressable
 * layouts, an injective output, and exact boundary types. It produces immutable invocation
 * geometry plus one direct scalar code-shaping identity; it does not materialize a window,
 * allocate workspace, or select a fusion or native route.
 */
public final class CpuPool2dLowering {
    private final CpuCapabilityProvider capabilities = new CpuCapabilityProvider();

    /** Creates a stateless Pool2d lowerer with the current CPU capability contract. */
    public CpuPool2dLowering() {}

    /**
     * Checks the original symbolic occurrence and retains an affine, unsampled numeric recipe.
     * @param context complete one-node projection with one canonical named B and fixed windows
     * @return explicit unresolved Pool2d geometry
     * @throws IllegalArgumentException if original semantics, types or dimension equations disagree
     * @throws ArithmeticException if fixed coefficient arithmetic overflows
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
        if (context.nodes().size() != 1) throw new IllegalArgumentException("Pool2d requires one node");
        var node = context.nodes().getFirst();
        if (node.inputs().size() != 1 || node.outputs().size() != 1 || node.inputs().contains(node.outputs().getFirst()))
            throw new IllegalArgumentException("Pool2d boundary arity or alias disagrees");
        CpuPool2dIr.Kind family;
        long kh, kw, sh, sw, ph, pw, dh, dw; boolean ceil;
        if (node.operation().kind() == Pool2dKind.MAX_POOL2D && node.operation().attrs() instanceof MaxPool2dAttrs a) {
            family = CpuPool2dIr.Kind.MAX;
            kh=a.kernelHeight(); kw=a.kernelWidth(); sh=a.strideHeight(); sw=a.strideWidth();
            ph=a.paddingHeight(); pw=a.paddingWidth(); dh=a.dilationHeight(); dw=a.dilationWidth(); ceil=a.ceilMode();
        } else if (node.operation().kind() == Pool2dKind.AVERAGE_POOL2D && node.operation().attrs() instanceof AveragePool2dAttrs a) {
            family = CpuPool2dIr.Kind.AVERAGE;
            kh=a.kernelHeight(); kw=a.kernelWidth(); sh=a.strideHeight(); sw=a.strideWidth();
            ph=a.paddingHeight(); pw=a.paddingWidth(); dh=a.dilationHeight(); dw=a.dilationWidth(); ceil=a.ceilMode();
        } else throw new IllegalArgumentException("Pool2d kind/attributes disagree");
        var input = require(values, node.inputs().getFirst()); var output = require(values, node.outputs().getFirst());
        if (input.descriptor().shape().rank() != 4 || output.descriptor().shape().rank() != 4
                || input.descriptor().dataType() != output.descriptor().dataType()
                || input.descriptor().dataType() == io.github.pho001.synaptik.model.datatype.DataType.BOOL
                || input.descriptor().dataType() == io.github.pho001.synaptik.model.datatype.DataType.INT32
                || input.descriptor().dataType() == io.github.pho001.synaptik.model.datatype.DataType.INT64
                || input.descriptor().requiresGrad() != output.descriptor().requiresGrad())
            throw new IllegalArgumentException("Pool2d original type/rank/gradient facts disagree");
        sameAxis(input,0,output,0); sameAxis(input,1,output,1);
        if (fixed(output,2) != outputExtent(fixed(input,2),kh,ph,dh,sh,ceil)
                || fixed(output,3) != outputExtent(fixed(input,3),kw,pw,dw,sw,ceil))
            throw new IllegalArgumentException("Pool2d original window equation disagrees");
        var in = batchBinding(input,layouts); var out = writeRecipe(batchBinding(output,layouts));
        var ir = new CpuPool2dIr(family,input.descriptor().dataType(),CpuPool2dIr.Realization.DIRECT_SCALAR,in.plan(),writePlan(out));
        return new CpuPartitionLowering.LoweredPartition(new Geometry.BatchGeometry(ir,
                List.of(input.id(),output.id()),List.of(input.descriptor().dataType(),output.descriptor().dataType()),
                List.of(in,out),List.of(),new long[]{kh,kw,sh,sw,ph,pw,dh,dw,Math.multiplyExact(kh,kw)}));
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
     * Lowers one exact pooling occurrence into complete-output-cell geometry.
     *
     * @param context non-null single-node CPU preparation projection
     * @return immutable direct scalar lowering with no workspace
     * @throws NullPointerException if {@code context} is null
     * @throws IllegalArgumentException if semantics, descriptors, or layouts disagree
     * @throws ArithmeticException if checked geometry arithmetic overflows
     */
    public CpuPartitionLowering.LoweredPartition lower(
            PrepareContext<? extends BackendAnalysisInputs> context) {
        Objects.requireNonNull(context, "context");
        if (context.nodes().size() != 1)
            throw new IllegalArgumentException("CPU Pool2d requires exactly one occurrence");
        var node = context.nodes().getFirst();
        Map<ValueId, GraphValue> values = new LinkedHashMap<>();
        context.values().forEach(value -> values.put(value.id(), value));
        var query =
                new OperationCapabilityQuery(node.operation(), node.inputs().stream().map(id -> require(values, id).descriptor()).toList(), node.outputs().stream().map(id -> require(values, id).descriptor()).toList());
        if (!capabilities.supports(query) || node.inputs().size() != 1 || node.outputs().size() != 1)
            throw new IllegalArgumentException("unsupported CPU Pool2d occurrence");
        Object kind = node.operation().kind();
        Object attrs = node.operation().attrs();
        CpuPool2dIr.Kind family;
        long kh, kw, sh, sw, ph, pw, dh, dw;
        boolean ceil;
        if (kind == Pool2dKind.MAX_POOL2D && attrs instanceof MaxPool2dAttrs a) {
            family = CpuPool2dIr.Kind.MAX;
            kh = a.kernelHeight();
            kw = a.kernelWidth();
            sh = a.strideHeight();
            sw = a.strideWidth();
            ph = a.paddingHeight();
            pw = a.paddingWidth();
            dh = a.dilationHeight();
            dw = a.dilationWidth();
            ceil = a.ceilMode();
        } else if (kind == Pool2dKind.AVERAGE_POOL2D && attrs instanceof AveragePool2dAttrs a) {
            family = CpuPool2dIr.Kind.AVERAGE;
            kh = a.kernelHeight();
            kw = a.kernelWidth();
            sh = a.strideHeight();
            sw = a.strideWidth();
            ph = a.paddingHeight();
            pw = a.paddingWidth();
            dh = a.dilationHeight();
            dw = a.dilationWidth();
            ceil = a.ceilMode();
        } else throw new IllegalArgumentException("Pool2d kind and attributes disagree");
        GraphValue input = require(values, node.inputs().getFirst()),
                output = require(values, node.outputs().getFirst());
        Layout in = layout(input), out = layout(output);
        long[] ie = in.extents(), oe = out.extents();
        long oh = outputExtent(ie[2], kh, ph, dh, sh, ceil),
                ow = outputExtent(ie[3], kw, pw, dw, sw, ceil);
        if (!Arrays.equals(oe, new long[] {ie[0], ie[1], oh, ow}))
            throw new IllegalArgumentException("Pool2d output shape disagrees");
        long divisor = Math.multiplyExact(kh, kw), count = count(oe);
        var inputBinding = binding(in, CpuAccessPlan.AccessKind.READ);
        var outputBinding = binding(out, CpuAccessPlan.AccessKind.WRITE);
        var ir =
                new CpuPool2dIr(
                        family,
                        input.descriptor().dataType(),
                        CpuPool2dIr.Realization.DIRECT_SCALAR,
                        inputBinding.plan(),
                        outputBinding.plan());
        var geometry =
                new Geometry(
                        family,
                        input.descriptor().dataType(),
                        in,
                        out,
                        kh,
                        kw,
                        sh,
                        sw,
                        ph,
                        pw,
                        dh,
                        dw,
                        divisor,
                        count);
        return new CpuPartitionLowering.LoweredPartition(
                ir,
                List.of(node.inputs().getFirst(), node.outputs().getFirst()),
                List.of(inputBinding, outputBinding),
                List.of(
                        input.descriptor().layout().orElseThrow().referencedElementSpan(),
                        output.descriptor().layout().orElseThrow().referencedElementSpan()),
                List.of(input.descriptor().dataType(), output.descriptor().dataType()),
                List.of(),
                oe,
                count,
                "legal: direct static NCHW " + family + " Pool2d",
                new long[0],
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.of(geometry));
    }

    private static long outputExtent(
            long d, long k, long p, long dilation, long stride, boolean ceil) {
        long effective = Math.addExact(Math.multiplyExact(dilation, Math.subtractExact(k, 1)), 1);
        long numerator = Math.subtractExact(Math.addExact(d, Math.multiplyExact(2, p)), effective);
        if (numerator < 0)
            throw new IllegalArgumentException("Pool2d padded extent is smaller than effective kernel");
        long q = numerator / stride;
        if (ceil && numerator % stride != 0) q = Math.addExact(q, 1);
        return Math.addExact(q, 1);
    }

    private static GraphValue require(Map<ValueId, GraphValue> values, ValueId id) {
        var value = values.get(id);
        if (value == null) throw new IllegalArgumentException("Pool2d value is not projected: " + id);
        return value;
    }

    private static Layout layout(GraphValue value) {
        var descriptor = value.descriptor();
        var layout = descriptor.layout().orElseThrow();
        long[] extents = descriptor.shape().toLongArray();
        if (extents.length != 4
                || layout.storageOffset() < 0
                || Arrays.stream(layout.strides()).anyMatch(s -> s < 0))
            throw new IllegalArgumentException("Pool2d requires rank-four non-negative resolved layout");
        return new Layout(extents, layout.storageOffset(), layout.strides());
    }

    private static CpuAccessPlan.Binding binding(Layout layout, CpuAccessPlan.AccessKind kind) {
        long[] e = layout.extents(), s = layout.strides();
        int suffix = 0;
        long expected = 1;
        for (int axis = 3; axis >= 0; axis--) {
            if (s[axis] != expected) break;
            suffix++;
            expected = Math.multiplyExact(expected, Math.max(1, e[axis]));
        }
        var roles = new ArrayList<CpuAccessPlan.AxisRole>();
        for (int axis = 0; axis < 4; axis++)
            roles.add(
                    s[axis] == 0
                            ? CpuAccessPlan.AxisRole.BROADCAST
                            : axis >= 4 - suffix
                                    ? CpuAccessPlan.AxisRole.CONTIGUOUS
                                    : CpuAccessPlan.AxisRole.STRIDED);
        var plan =
                new CpuAccessPlan(
                        kind,
                        suffix == 4 ? CpuAccessPlan.Regime.DENSE_LINEAR : CpuAccessPlan.Regime.GENERAL_ODOMETER,
                        4,
                        roles,
                        suffix);
        long count = count(e);
        return CpuAccessPlan.Binding.create(plan, e, layout.offset(), s, count, 0, count, span(layout));
    }

    private static long count(long[] e) {
        for (long x : e) if (x == 0) return 0;
        long r = 1;
        for (long x : e) r = Math.multiplyExact(r, x);
        return r;
    }

    private static long span(Layout l) {
        if (count(l.extents()) == 0) return 0;
        long last = l.offset();
        long[] e = l.extents(), s = l.strides();
        for (int i = 0; i < e.length; i++)
            last = Math.addExact(last, Math.multiplyExact(e[i] - 1, s[i]));
        return Math.addExact(last, 1);
    }

    /**
     * Immutable logical rank-four layout.
     *
     * @param extents non-null four-element logical NCHW extents; copied defensively
     * @param offset non-negative carrier-relative element offset
     * @param strides non-null four-element non-negative element strides; copied defensively
     */
    public record Layout(long[] extents, long offset, long[] strides) {
        /**
         * Snapshots one non-negative resolved rank-four layout.
         *
         * @param extents non-null four-element logical NCHW extents; copied defensively
         * @param offset non-negative carrier-relative element offset
         * @param strides non-null four-element non-negative element strides; copied defensively
         * @throws NullPointerException if an array is {@code null}
         * @throws IllegalArgumentException if rank, extent, offset, or stride facts are invalid
         */
        public Layout {
            extents = extents.clone();
            strides = strides.clone();
            if (extents.length != 4
                    || strides.length != 4
                    || offset < 0
                    || Arrays.stream(extents).anyMatch(x -> x < 0)
                    || Arrays.stream(strides).anyMatch(x -> x < 0))
                throw new IllegalArgumentException("invalid Pool2d layout");
        }

        /**
         * Returns logical extents without exposing retained mutable state.
         *
         * @return a fresh four-element extent array
         */
        @Override
        public long[] extents() {
            return extents.clone();
        }

        /**
         * Returns element strides without exposing retained mutable state.
         *
         * @return a fresh four-element stride array
         */
        @Override
        public long[] strides() {
            return strides.clone();
        }
    }

    /**
     * Complete immutable invocation geometry; carriers and bases remain invocation facts.
     *
     * @param kind exact pooling numerical family
     * @param dataType supported floating boundary type
     * @param input immutable logical NCHW input layout
     * @param output immutable logical NCHW output layout
     * @param kernelHeight positive kernel height in input coordinates
     * @param kernelWidth positive kernel width in input coordinates
     * @param strideHeight positive output-step height in input coordinates
     * @param strideWidth positive output-step width in input coordinates
     * @param paddingHeight non-negative conceptual zero padding on each height edge
     * @param paddingWidth non-negative conceptual zero padding on each width edge
     * @param dilationHeight positive height spacing between kernel samples
     * @param dilationWidth positive width spacing between kernel samples
     * @param divisor exact fixed {@code kernelHeight * kernelWidth} average divisor
     * @param outputCount exact product of output extents, or zero for an empty output
     */
    public record Geometry(
            CpuPool2dIr.Kind kind,
            io.github.pho001.synaptik.model.datatype.DataType dataType,
            Layout input,
            Layout output,
            long kernelHeight,
            long kernelWidth,
            long strideHeight,
            long strideWidth,
            long paddingHeight,
            long paddingWidth,
            long dilationHeight,
            long dilationWidth,
            long divisor,
            long outputCount) {

        /**
         * Unresolved numeric invocation recipe. No representative shape, graph or invocation is retained.
         * @param ir exact B-independent generated facts
         * @param boundaries ordered physical boundaries
         * @param boundaryTypes exact boundary storage types
         * @param bindings affine numeric mappings, including the output mapping
         * @param virtualValues private removed intermediate values
         * @param parameters fixed family geometry, defensively copied

         */
        public record BatchGeometry(CpuPool2dIr ir, List<ValueId> boundaries,
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
             * Returns the fixed Pool2d geometry cells retained for numeric invocation binding.
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
                var geometry = new Geometry(ir.kind(), ir.dataType(), new Layout(resolved.getFirst().extents().stream().mapToLong(Long::longValue).toArray(), resolved.getFirst().baseElementOffset(), resolved.getFirst().effectiveStrides().stream().mapToLong(Long::longValue).toArray()), new Layout(out.extents().stream().mapToLong(Long::longValue).toArray(), out.baseElementOffset(), out.effectiveStrides().stream().mapToLong(Long::longValue).toArray()), parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], parameters[8], out.elementCount());
                return new CpuPartitionLowering.LoweredPartition(ir, boundaries, resolved,
                        resolved.stream().map(CpuAccessPlan.Binding::referencedElementSpan).toList(),
                        boundaryTypes, virtualValues, out.extents().stream().mapToLong(Long::longValue).toArray(), out.elementCount(),
                        "legal: checked symbolic ordinary geometry", new long[0],
                        Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.of(geometry));
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
            return new TemporalRebind(List.of(new BaseCell(0, 0, input.offset()),
                    new BaseCell(1, 1, output.offset())), List.of());
        }

        /**
         * Validates exact intrinsic geometry and complete output-cell ownership count.
         *
         * @param kind exact pooling numerical family
         * @param dataType supported floating boundary type
         * @param input immutable logical NCHW input layout
         * @param output immutable logical NCHW output layout
         * @param kernelHeight positive kernel height in input coordinates
         * @param kernelWidth positive kernel width in input coordinates
         * @param strideHeight positive output-step height in input coordinates
         * @param strideWidth positive output-step width in input coordinates
         * @param paddingHeight non-negative conceptual zero padding on each height edge
         * @param paddingWidth non-negative conceptual zero padding on each width edge
         * @param dilationHeight positive height spacing between kernel samples
         * @param dilationWidth positive width spacing between kernel samples
         * @param divisor exact fixed {@code kernelHeight * kernelWidth} average divisor
         * @param outputCount exact product of output extents, or zero for an empty output
         * @throws NullPointerException if a required reference is {@code null}
         * @throws IllegalArgumentException if intrinsic or output-count facts disagree
         * @throws ArithmeticException if divisor or output-count validation overflows
         */
        public Geometry {
            Objects.requireNonNull(kind, "kind");
            Objects.requireNonNull(dataType, "dataType");
            Objects.requireNonNull(input, "input");
            Objects.requireNonNull(output, "output");
            if (!dataType.isFloating()
                    || kernelHeight <= 0
                    || kernelWidth <= 0
                    || strideHeight <= 0
                    || strideWidth <= 0
                    || paddingHeight < 0
                    || paddingWidth < 0
                    || dilationHeight <= 0
                    || dilationWidth <= 0
                    || divisor != Math.multiplyExact(kernelHeight, kernelWidth)
                    || outputCount != count(output.extents()))
                throw new IllegalArgumentException("Pool2d geometry disagrees");
        }

        /**
         * Packs bases, both layouts, and intrinsic geometry for the generated entry.
         *
         * @param inputBase carrier-relative input element base
         * @param outputBase carrier-relative output element base
         * @return fresh compact primitive geometry
         * @throws ArithmeticException if adding either carrier base overflows
         */
        public long[] pack(long inputBase, long outputBase) {
            long[] a = new long[28];
            a[0] = Math.addExact(inputBase, input.offset());
            a[1] = Math.addExact(outputBase, output.offset());
            System.arraycopy(input.extents(), 0, a, 2, 4);
            System.arraycopy(input.strides(), 0, a, 6, 4);
            System.arraycopy(output.extents(), 0, a, 10, 4);
            System.arraycopy(output.strides(), 0, a, 14, 4);
            a[18] = kernelHeight;
            a[19] = kernelWidth;
            a[20] = strideHeight;
            a[21] = strideWidth;
            a[22] = paddingHeight;
            a[23] = paddingWidth;
            a[24] = dilationHeight;
            a[25] = dilationWidth;
            a[26] = divisor;
            a[27] = outputCount;
            return a;
        }
    }
}
