package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuTrailingNormalizationIr;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.normalization.*;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.*;

/**
 * Lowers one explicit static trailing Layer or RMS normalization to complete-slice ranges.
 *
 * <p>The lowerer admits only the first-class Model occurrence. It snapshots resolved non-negative
 * layouts, deduplicates repeated logical inputs in first-use order, derives checked leading and
 * normalized counts, and declares Layer-only exact-mean scratch geometry. It does not recognize
 * decomposed graphs or choose run-time carriers, worker ranges, or physical resources.</p>
 * <p>The single-B entry retains arbitrary leading axes and fixed normalized features as
 * numeric coefficients. Binding preserves this family's original geometry and arithmetic.</p>
 */
public final class CpuTrailingNormalizationLowering {
    private final CpuCapabilityProvider capabilities = new CpuCapabilityProvider();
    /** Creates a stateless trailing-normalization lowerer. */
    public CpuTrailingNormalizationLowering() { }

    /**
     * Lowers one supported occurrence and derives all checked cold geometry.
     *
     * @param context non-null projection containing exactly one supported first-class occurrence
     * @return one immutable lowering whose ranges count complete leading slices; never
     *     {@code null}
     * @throws NullPointerException if {@code context} or a required projected fact is null
     * @throws IllegalArgumentException if occurrence cardinality, descriptors, layouts, output
     *     identity, promotion, or normalization geometry is unsupported or inconsistent
     * @throws ArithmeticException if Shape, span, address, or exact-state sizing overflows
     */
    public CpuPartitionLowering.LoweredPartition lower(
            PrepareContext<? extends BackendAnalysisInputs> context) {
        Objects.requireNonNull(context, "context");
        if (context.nodes().size() != 1) throw new IllegalArgumentException(
                "CPU trailing normalization requires exactly one node");
        var node = context.nodes().getFirst();
        Map<ValueId, GraphValue> values = new LinkedHashMap<>();
        context.values().forEach(value -> values.put(value.id(), value));
        var query = new OperationCapabilityQuery(node.operation(), node.inputs().stream()
                .map(id -> require(values, id).descriptor()).toList(), node.outputs().stream()
        .map(id -> require(values, id).descriptor()).toList());
        if (!capabilities.supports(query)) throw new IllegalArgumentException(
                "partition contains an unsupported CPU trailing-normalization occurrence");
        ValueId outputId = node.outputs().getFirst();
        if (node.inputs().contains(outputId)) throw new IllegalArgumentException(
                "trailing-normalization output must be distinct from every input");

        CpuTrailingNormalizationIr.Kind kind;
        CpuTrailingNormalizationIr.Form form;
        io.github.pho001.synaptik.model.shape.Shape normalizedShape;
        ScalarValue epsilon;
        if (node.operation().attrs() instanceof LayerNormAttrs attrs) {
            kind = CpuTrailingNormalizationIr.Kind.LAYER;
            form = CpuTrailingNormalizationIr.Form.LAYER;
            normalizedShape = attrs.normalizedShape(); epsilon = attrs.epsilon();
        } else if (node.operation().attrs() instanceof AffineLayerNormAttrs attrs) {
            kind = CpuTrailingNormalizationIr.Kind.LAYER;
            form = CpuTrailingNormalizationIr.Form.LAYER_AFFINE;
            normalizedShape = attrs.normalizedShape(); epsilon = attrs.epsilon();
        } else {
            RmsNormAttrs attrs = (RmsNormAttrs) node.operation().attrs();
            kind = CpuTrailingNormalizationIr.Kind.RMS;
            form = node.inputs().size() == 1 ? CpuTrailingNormalizationIr.Form.RMS
                    : CpuTrailingNormalizationIr.Form.RMS_SCALED;
            normalizedShape = attrs.normalizedShape(); epsilon = attrs.epsilon();
        }
        GraphValue input = require(values, node.inputs().getFirst());
        GraphValue output = require(values, outputId);
        Layout inputLayout = layout(input), outputLayout = layout(output);
        validateInjective(outputLayout);
        int normalizedRank = normalizedShape.rank();
        long normalizedCount = elementCount(normalizedShape.toLongArray());
        long leadingCount = prefixCount(inputLayout.extents, inputLayout.extents.length - normalizedRank);
        long outputCount = elementCount(outputLayout.extents);
        if (outputCount != Math.multiplyExact(leadingCount, normalizedCount))
            throw new IllegalArgumentException("trailing-normalization Shape counts disagree");

        var uniqueIds = new ArrayList<ValueId>();
        var positionMap = new ArrayList<Integer>();
        for (ValueId id : node.inputs()) {
            int index = uniqueIds.indexOf(id);
            if (index < 0) { index = uniqueIds.size(); uniqueIds.add(id); }
            positionMap.add(index);
        }
        var layouts = new ArrayList<Layout>();
        var bindings = new ArrayList<CpuAccessPlan.Binding>();
        var spans = new ArrayList<Long>();
        var types = new ArrayList<DataType>();
        for (ValueId id : uniqueIds) {
            GraphValue value = require(values, id); Layout layout = layout(value);
            layouts.add(layout); bindings.add(binding(layout, CpuAccessPlan.AccessKind.READ));
            spans.add(value.descriptor().layout().orElseThrow().referencedElementSpan());
            types.add(value.descriptor().dataType());
        }
        var outputBinding = binding(outputLayout, CpuAccessPlan.AccessKind.WRITE);
        bindings.add(outputBinding);
        spans.add(output.descriptor().layout().orElseThrow().referencedElementSpan());
        types.add(output.descriptor().dataType());
        var boundaryIds = new ArrayList<>(uniqueIds); boundaryIds.add(outputId);
        DataType exactType = output.descriptor().dataType() == DataType.BFLOAT16
                || output.descriptor().dataType() == DataType.FLOAT16
                ? DataType.FLOAT32 : output.descriptor().dataType();
        long scratch = kind == CpuTrailingNormalizationIr.Kind.LAYER && outputCount > 0
                ? exactStateSliceBytes(exactType, normalizedCount) : 0;
        int limbs = scratch == 0 ? 0 : Math.toIntExact(scratch / Long.BYTES - 1);
        var semanticTypes = node.inputs().stream().map(id -> require(values, id).descriptor().dataType())
                .toList();
        var inputPlans = bindings.subList(0, uniqueIds.size()).stream()
                .map(CpuAccessPlan.Binding::plan).toList();
        long epsilonBits = epsilonBits(output.descriptor().dataType(), epsilon);
        var ir = new CpuTrailingNormalizationIr(kind, form, semanticTypes,
                output.descriptor().dataType(), epsilonBits, normalizedRank, 1,
                kind == CpuTrailingNormalizationIr.Kind.LAYER ? 3 : 2, normalizedCount,
                limbs, scratch, positionMap, inputPlans, outputBinding.plan());
        var geometry = new Geometry(kind, form, semanticTypes, output.descriptor().dataType(),
                epsilonBits, normalizedRank, normalizedCount, leadingCount,
                positionMap, layouts, outputLayout, scratch);
        return new CpuPartitionLowering.LoweredPartition(ir, boundaryIds, bindings, spans, types,
                List.of(), new long[] {outputCount == 0 ? 0 : leadingCount},
                outputCount == 0 ? 0 : leadingCount,
                "legal: one static trailing normalization", new long[0], Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.of(geometry));
    }

    /**
     * Retains exact trailing normalization facts over arbitrary leading single-B axes.
     * @param context complete one-node projection with original symbolic descriptors
     * @return unresolved per-boundary geometry and fixed normalized-feature arithmetic
     * @throws NullPointerException if the context or a required fact is null
     * @throws IllegalArgumentException if operand, output, attribute or promotion facts disagree
     * @throws ArithmeticException if fixed feature or exact-state arithmetic overflows
     */
    public CpuPartitionLowering.LoweredPartition lowerBatch(
            PrepareContext<? extends BackendAnalysisInputs> context) {
        return lowerBatch(context, CpuPartitionLowering.batchLayouts(context));
    }

    /**
     * Lowers using the complete local graph's already proved physical access mappings.
     * No subset graph is used to reconstruct view strides, origins, or symbolic axis positions.
     * @param context original complete one-node descriptor projection
     * @param layouts complete local value-to-layout recipes; neither retained nor mutated
     * @return unresolved family geometry retaining only immutable numeric access facts
     * @throws NullPointerException if either argument or a required fact is null
     * @throws IllegalArgumentException if a mapping, descriptor or family predicate disagrees
     * @throws ArithmeticException if fixed coefficients exceed supported geometry limits
     */
    public CpuPartitionLowering.LoweredPartition lowerBatch(
            PrepareContext<? extends BackendAnalysisInputs> context,
            Map<ValueId, CpuAccessPlan.BatchBinding> layouts) {
        Objects.requireNonNull(layouts, "layouts");
        Objects.requireNonNull(context, "context");
        CpuPartitionLowering.requireBatchContext(context);
        if (context.nodes().size() != 1) throw new IllegalArgumentException("normalization needs one node");
        var node = context.nodes().getFirst();
        CpuTrailingNormalizationIr.Kind kind; CpuTrailingNormalizationIr.Form form;
        io.github.pho001.synaptik.model.shape.Shape normalized; ScalarValue epsilon; int expected;
        if (node.operation().kind() == LayerNormKind.LAYER_NORM
                && node.operation().attrs() instanceof LayerNormAttrs attrs) {
            kind = CpuTrailingNormalizationIr.Kind.LAYER; form = CpuTrailingNormalizationIr.Form.LAYER;
            normalized = attrs.normalizedShape(); epsilon = attrs.epsilon(); expected = 1;
        } else if (node.operation().kind() == LayerNormKind.LAYER_NORM
                && node.operation().attrs() instanceof AffineLayerNormAttrs attrs) {
            kind = CpuTrailingNormalizationIr.Kind.LAYER; form = CpuTrailingNormalizationIr.Form.LAYER_AFFINE;
            normalized = attrs.normalizedShape(); epsilon = attrs.epsilon(); expected = 3;
        } else if (node.operation().kind() == RmsNormKind.RMS_NORM
                && node.operation().attrs() instanceof RmsNormAttrs attrs
                && node.inputs().size() >= 1 && node.inputs().size() <= 2) {
            kind = CpuTrailingNormalizationIr.Kind.RMS;
            form = node.inputs().size() == 1 ? CpuTrailingNormalizationIr.Form.RMS
                    : CpuTrailingNormalizationIr.Form.RMS_SCALED;
            normalized = attrs.normalizedShape(); epsilon = attrs.epsilon(); expected = node.inputs().size();
        } else throw new IllegalArgumentException("unsupported symbolic trailing normalization");
        if (node.inputs().size() != expected || node.outputs().size() != 1
                || node.inputs().contains(node.outputs().getFirst())
                || !normalized.isFullyStatic() || normalized.rank() <= 0)
            throw new IllegalArgumentException("trailing normalization cardinality or features disagree");
        Map<ValueId, GraphValue> values = new LinkedHashMap<>();
        context.values().forEach(value -> values.put(value.id(), value));
        var operands = node.inputs().stream().map(id -> require(values, id)).toList();
        var output = require(values, node.outputs().getFirst());
        var input = operands.getFirst().descriptor(); var out = output.descriptor();
        int leadingRank = input.shape().rank() - normalized.rank();
        DataType result = out.dataType();
        var ids = new ArrayList<ValueId>(); var map = new ArrayList<Integer>();
        for (var id : node.inputs()) { int b = ids.indexOf(id);
            if (b < 0) { b = ids.size(); ids.add(id); } map.add(b); }
        var bindings = new ArrayList<CpuAccessPlan.BatchBinding>(); var types = new ArrayList<DataType>();
        for (var id : ids) { var value = require(values, id);
            bindings.add(batchBinding(value, CpuAccessPlan.AccessKind.READ, layouts)); types.add(value.descriptor().dataType()); }
        var write = batchBinding(output, CpuAccessPlan.AccessKind.WRITE, layouts);
        bindings.add(write); ids.add(output.id()); types.add(result);
        long normalizedCount = elementCount(normalized.toLongArray());
        var iterations = batchProduct(write, 0, leadingRank);
        if (normalizedCount == 0) iterations = new BatchByteSize(0, 0, 1);
        DataType exact = result == DataType.FLOAT16 || result == DataType.BFLOAT16 ? DataType.FLOAT32 : result;
        long scratch = kind == CpuTrailingNormalizationIr.Kind.LAYER && normalizedCount != 0
                && (iterations.constantBytes() != 0 || iterations.bytesPerBatch() != 0)
                ? exactStateSliceBytes(exact, normalizedCount) : 0;
        int limbs = scratch == 0 ? 0 : Math.toIntExact(scratch / Long.BYTES - 1);
        var ir = new CpuTrailingNormalizationIr(kind, form,
                operands.stream().map(v -> v.descriptor().dataType()).toList(), result,
                epsilonBits(result, epsilon), normalized.rank(), 1,
                kind == CpuTrailingNormalizationIr.Kind.LAYER ? 3 : 2, normalizedCount, limbs, scratch,
                map, bindings.subList(0, bindings.size() - 1).stream().map(CpuAccessPlan.BatchBinding::plan).toList(),
                write.plan());
        return new CpuPartitionLowering.LoweredPartition(new Geometry.BatchGeometry(ir, ids, types,
                bindings, iterations, new BatchByteSize(scratch, 0, 1)));
    }


    private static CpuAccessPlan.BatchBinding batchBinding(GraphValue value,
            CpuAccessPlan.AccessKind kind, Map<ValueId, CpuAccessPlan.BatchBinding> layouts) {
        var source = layouts.get(value.id());
        if (source == null || source.plan().iterationRank() != value.descriptor().shape().rank())
            throw new IllegalArgumentException("missing or inconsistent full-local normalization layout");
        var exact = CpuPartitionLowering.batchAccess(kind, source.extentConstants(),
                source.extentBatchFactors(), source.strideConstants(), source.strideBatchFactors());
        return new CpuAccessPlan.BatchBinding(exact.plan(), exact.extentConstants(),
                exact.extentBatchFactors(), source.baseElementOffset(), exact.strideConstants(),
                exact.strideBatchFactors());
    }

    private static BatchByteSize batchProduct(CpuAccessPlan.BatchBinding binding, int begin, int end) {
        long constant = 1, factor = 0;
        for (int axis = begin; axis < end; axis++) {
            long ec = binding.extentConstants().get(axis), ef = binding.extentBatchFactors().get(axis);
            if (factor != 0 && ef != 0) throw new IllegalArgumentException("non-affine normalization count");
            factor = Math.addExact(Math.multiplyExact(factor, ec), Math.multiplyExact(constant, ef));
            constant = Math.multiplyExact(constant, ec);
        }
        return new BatchByteSize(constant, factor, 1);
    }

    private static Layout boundLayout(CpuAccessPlan.Binding binding) {
        return new Layout(binding.extents().stream().mapToLong(Long::longValue).toArray(),
                binding.baseElementOffset(), binding.effectiveStrides().stream()
                        .mapToLong(Long::longValue).toArray());
    }

    private static long epsilonBits(DataType resultType, ScalarValue epsilon) {
        if (resultType == DataType.FLOAT16 && epsilon.dataType() == DataType.FLOAT32)
            return Float.floatToFloat16(epsilon.float32Value()) & 0xffffL;
        return switch (epsilon.dataType()) {
            case FLOAT64 -> Double.doubleToRawLongBits(epsilon.float64Value());
            case FLOAT32 -> Float.floatToRawIntBits(epsilon.float32Value()) & 0xffff_ffffL;
            case BFLOAT16 -> epsilon.bfloat16Bits() & 0xffffL;
            case FLOAT16 -> epsilon.float16Bits() & 0xffffL;
            default -> throw new IllegalArgumentException(
                    "normalization epsilon must be floating");
        };
    }
    private static long exactStateSliceBytes(DataType type, long count) {
        int emin = type == DataType.FLOAT64 ? -1074 : type == DataType.FLOAT32 ? -149 : -133;
        int emax = type == DataType.FLOAT64 ? 1023 : 127;
        int countBits = count <= 1 ? 0 : 64 - Long.numberOfLeadingZeros(count - 1);
        long bits = Math.addExact((long) emax + 1 - emin, Math.addExact(countBits, 1));
        return Math.addExact(8, Math.multiplyExact(8, Math.addExact(bits, 63) / 64));
    }
    private static GraphValue require(Map<ValueId, GraphValue> values, ValueId id) {
        GraphValue value = values.get(id);
        if (value == null) throw new IllegalArgumentException("partition value is not projected: " + id);
        return value;
    }
    private static Layout layout(GraphValue value) {
        LayoutDescriptor source = value.descriptor().layout().orElseThrow();
        if (source.storageOffset() < 0 || Arrays.stream(source.strides()).anyMatch(v -> v < 0))
            throw new IllegalArgumentException("trailing normalization requires non-negative layouts");
        return new Layout(value.descriptor().shape().toLongArray(), source.storageOffset(), source.strides());
    }
    private static CpuAccessPlan.Binding binding(Layout layout, CpuAccessPlan.AccessKind kind) {
        int suffix = 0; long expected = 1;
        for (int axis = layout.extents.length - 1; axis >= 0; axis--) {
            if (layout.strides[axis] != expected) break;
            suffix++; expected = Math.multiplyExact(expected, Math.max(1, layout.extents[axis]));
        }
        var roles = new ArrayList<CpuAccessPlan.AxisRole>();
        for (int axis = 0; axis < layout.extents.length; axis++) roles.add(layout.strides[axis] == 0
                ? CpuAccessPlan.AxisRole.BROADCAST : axis >= layout.extents.length - suffix
                ? CpuAccessPlan.AxisRole.CONTIGUOUS : CpuAccessPlan.AxisRole.STRIDED);
        var plan = new CpuAccessPlan(kind, suffix == layout.extents.length
                ? CpuAccessPlan.Regime.DENSE_LINEAR : CpuAccessPlan.Regime.GENERAL_ODOMETER,
                layout.extents.length, roles, suffix);
        long count = elementCount(layout.extents);
        return CpuAccessPlan.Binding.create(plan, layout.extents, layout.offset, layout.strides,
                count, 0, count, referencedSpan(layout));
    }
    private static long referencedSpan(Layout layout) {
        if (elementCount(layout.extents) == 0) return 0;
        long max = layout.offset;
        for (int axis = 0; axis < layout.extents.length; axis++) max = Math.addExact(max,
                Math.multiplyExact(layout.extents[axis] - 1, layout.strides[axis]));
        return Math.addExact(max, 1);
    }
    private static long elementCount(long[] extents) {
        for (long extent : extents) if (extent == 0) return 0;
        long result = 1; for (long extent : extents) result = Math.multiplyExact(result, extent);
        return result;
    }
    private static long prefixCount(long[] extents, int end) {
        long result = 1; for (int i = 0; i < end; i++) result = Math.multiplyExact(result, extents[i]);
        return result;
    }
    private static long suffixCount(long[] extents, int rank) {
        long result = 1;
        for (int axis = extents.length - rank; axis < extents.length; axis++)
            result = Math.multiplyExact(result, extents[axis]);
        return result;
    }
    private static boolean supported(DataType type) {
        return type == DataType.BFLOAT16 || type == DataType.FLOAT16
                || type == DataType.FLOAT32 || type == DataType.FLOAT64;
    }
    private static void validateInjective(Layout layout) {
        long count = elementCount(layout.extents); if (count == 0) return;
        // Canonical numeric recipes prove injectivity without enumerating invocation elements.
        long expected = 1; boolean canonical = true;
        for (int axis = layout.extents.length - 1; axis >= 0; axis--) {
            if (layout.strides[axis] != expected) { canonical = false; break; }
            expected = Math.multiplyExact(expected, layout.extents[axis]);
        }
        if (canonical) return;
        if (count <= 1_000_000) {
            var addresses = new HashSet<Long>(); long[] c = new long[layout.extents.length];
            for (long logical = 0; logical < count; logical++) {
                long address = layout.offset;
                for (int axis = 0; axis < c.length; axis++) address = Math.addExact(address,
                        Math.multiplyExact(c[axis], layout.strides[axis]));
                if (!addresses.add(address)) throw new IllegalArgumentException(
                        "trailing-normalization output layout must be injective");
                for (int axis = c.length - 1; axis >= 0; axis--)
                    if (++c[axis] < layout.extents[axis]) break; else c[axis] = 0;
            }
            return;
        }
        var axes = new ArrayList<Integer>();
        for (int axis = 0; axis < layout.extents.length; axis++) if (layout.extents[axis] > 1) axes.add(axis);
        axes.sort(Comparator.comparingLong(axis -> layout.strides[axis])); long covered = 1;
        for (int axis : axes) { if (layout.strides[axis] < covered) throw new IllegalArgumentException(
                "trailing-normalization output layout must be injective");
            covered = Math.addExact(covered, Math.multiplyExact(layout.extents[axis] - 1,
                    layout.strides[axis])); }
    }

    /**
     * Resolved non-negative Shape and element-stride geometry.
     *
     * @param extents non-negative static extents; copied defensively
     * @param offset non-negative element offset from the carrier base
     * @param strides non-negative element strides matching {@code extents}; copied defensively
     */
    public record Layout(long[] extents, long offset, long[] strides) {
        /**
         * Validates and snapshots one layout.
         *
         * @throws NullPointerException if an array is {@code null}
         * @throws IllegalArgumentException if ranks differ or any extent, offset, or stride is
         *     negative
         */
        public Layout { extents = extents.clone(); strides = strides.clone();
            if (extents.length != strides.length || offset < 0
                    || Arrays.stream(extents).anyMatch(v -> v < 0)
                    || Arrays.stream(strides).anyMatch(v -> v < 0))
                throw new IllegalArgumentException("trailing-normalization layout is invalid"); }
        /**
         * Returns a defensive copy of the resolved static extents.
         *
         * @return a new copy of the non-negative static extents
         */
        @Override public long[] extents() { return extents.clone(); }
        /**
         * Returns a defensive copy of the resolved element strides.
         *
         * @return a new copy of the non-negative element strides
         */
        @Override public long[] strides() { return strides.clone(); }
    }

    /**
     * Complete cold multi-boundary trailing-slice geometry retained by preparation and binding.
     *
     * @param kind exact Layer or RMS family
     * @param form exact semantic operand form
     * @param inputTypes semantic input types in occurrence order; copied defensively
     * @param resultType exact ordered-promotion result type
     * @param epsilonBits exact raw result-type epsilon bits
     * @param normalizedRank positive trailing normalized rank
     * @param normalizedCount non-negative elements per normalized slice
     * @param leadingCount non-negative number of complete leading slices
     * @param positionToBoundary semantic-position to unique-input-boundary mapping; copied
     *     defensively
     * @param inputs unique resolved input layouts in first-use order; copied defensively
     * @param output distinct resolved injective output layout
     * @param scratchSliceBytes exact bytes per Layer range, or zero for RMS and empty work
     */
    public record Geometry(CpuTrailingNormalizationIr.Kind kind,
            CpuTrailingNormalizationIr.Form form, List<DataType> inputTypes, DataType resultType,
            long epsilonBits, int normalizedRank, long normalizedCount, long leadingCount,
            List<Integer> positionToBoundary, List<Layout> inputs, Layout output,
            long scratchSliceBytes) {

        /**
         * Numeric single-B geometry, retaining no graph, descriptor, or invocation storage.
         * Coefficients are resolved only at the cold invocation boundary; the compiled IR is
         * identical for every admitted B and each output retains its canonical semantic slot.
         * @param ir structurally qualified family IR, independent of actual B
         * @param boundaries unique reads followed by every ordered semantic output
         * @param boundaryTypes exact represented types in boundary order
         * @param bindings immutable per-boundary affine element geometry
         * @param iterationCount affine complete-range count, in items rather than bytes
         * @param scratchBytes fixed exact-state capacity for one simultaneous range, in bytes
         */
        public record BatchGeometry(CpuTrailingNormalizationIr ir, List<ValueId> boundaries,
                List<DataType> boundaryTypes, List<CpuAccessPlan.BatchBinding> bindings,
                BatchByteSize iterationCount, BatchByteSize scratchBytes)
                implements CpuPartitionLowering.BatchUnitGeometry {
            /** Snapshots the immutable numeric recipe and its structural boundary inventory. */
            public BatchGeometry {
                Objects.requireNonNull(ir, "ir");
                boundaries = List.copyOf(boundaries); boundaryTypes = List.copyOf(boundaryTypes);
                bindings = List.copyOf(bindings);
                Objects.requireNonNull(iterationCount, "iterationCount");
                Objects.requireNonNull(scratchBytes, "scratchBytes");
                if (boundaries.size() != bindings.size() || boundaries.size() != boundaryTypes.size())
                    throw new IllegalArgumentException("normalization boundary recipe disagrees");
            }
            /**
             * Declares that these normalization packs have no mutable ordinary cursor cells.
             * @return an immutable empty reset inventory
             */
            @Override public List<io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.PackReset>
                    resets() { return List.of(); }
            /**
             * Returns deterministic coefficients and family identity, never resolved B.
             * @return immutable specialization recipe identity
             */
            @Override public String identity() {
                return ir.structuralKey() + ":normalization-batch-v1:" + bindings
                        + ":items=" + iterationCount + ":scratch=" + scratchBytes;
            }
            /**
             * Resolves checked numeric geometry without invoking semantic lowering.
             * @param batch non-negative invocation B, checked against access count/address limits
             * @return ordinary concrete family geometry with the identical compiled IR
             * @throws IllegalArgumentException if a positive-axis or training-domain rule fails
             * @throws ArithmeticException if resolved geometry exceeds supported limits
             */
            @Override public CpuPartitionLowering.LoweredPartition bind(long batch) {
                var resolved = bindings.stream().map(binding -> binding.bind(batch)).toList();
                var layouts = resolved.stream().map(CpuTrailingNormalizationLowering::boundLayout).toList();
                int unique = layouts.size() - 1;
                var output = layouts.getLast();
                long leading = prefixCount(output.extents, output.extents.length - ir.normalizedRank());
                long items = iterationCount.byteSize(batch);
                var geometry = new Geometry(ir.kind(), ir.form(), ir.inputTypes(), ir.resultType(),
                        ir.epsilonBits(), ir.normalizedRank(), ir.normalizedCount(), leading,
                        ir.positionToBoundary(), layouts.subList(0, unique), output,
                        scratchBytes.byteSize(batch));
                return new CpuPartitionLowering.LoweredPartition(ir, boundaries, resolved,
                        resolved.stream().map(CpuAccessPlan.Binding::referencedElementSpan).toList(),
                        boundaryTypes, List.of(), new long[] {items}, items,
                        "legal: symbolic trailing normalization", new long[0], Optional.empty(),
                        Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                        Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                        Optional.empty(), Optional.empty(), Optional.empty(), Optional.of(geometry));
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
            var bases = new ArrayList<BaseCell>(inputs.size() + 1);
            for (int i = 0; i < inputs.size(); i++)
                bases.add(new BaseCell(11 + i, i, inputs.get(i).offset));
            bases.add(new BaseCell(11 + inputs.size(), inputs.size(), output.offset));
            return new TemporalRebind(bases, List.of());
        }

        /**
         * Validates and snapshots one geometry.
         *
         * @throws NullPointerException if a required reference or list element is {@code null}
         * @throws IllegalArgumentException if promotion, mapping, Shape, family, form, or workspace
         *     facts disagree
         * @throws ArithmeticException if exact element or workspace geometry overflows
         */
        public Geometry { inputTypes = List.copyOf(inputTypes);
            positionToBoundary = List.copyOf(positionToBoundary); inputs = List.copyOf(inputs);
            int expected = switch (form) { case LAYER, RMS -> 1; case RMS_SCALED -> 2;
                case LAYER_AFFINE -> 3; };
            boolean floatingResult = supported(resultType);
            int unique = positionToBoundary.stream().mapToInt(Integer::intValue).max().orElse(-1) + 1;
            DataType promoted = inputTypes.isEmpty() ? null : inputTypes.getFirst();
            for (int index = 1; index < inputTypes.size(); index++)
                promoted = DataTypePromotion.promoteFloating(promoted, inputTypes.get(index));
            if (normalizedRank <= 0 || normalizedCount < 0 || leadingCount < 0
                    || inputs.isEmpty() || output == null || scratchSliceBytes < 0
                    || inputTypes.size() != expected || positionToBoundary.size() != expected
                    || inputs.size() != unique || positionToBoundary.getFirst() != 0
                    || !floatingResult || inputTypes.stream().anyMatch(type -> !supported(type))
                    || promoted != resultType
                    || (kind == CpuTrailingNormalizationIr.Kind.LAYER)
                        != (form == CpuTrailingNormalizationIr.Form.LAYER
                            || form == CpuTrailingNormalizationIr.Form.LAYER_AFFINE)
                    || kind == CpuTrailingNormalizationIr.Kind.RMS && scratchSliceBytes != 0
                    || kind == CpuTrailingNormalizationIr.Kind.LAYER && normalizedCount > 0
                        && leadingCount > 0 && scratchSliceBytes == 0
                    || output.extents.length < normalizedRank
                    || suffixCount(output.extents, normalizedRank) != normalizedCount
                    || prefixCount(output.extents, output.extents.length - normalizedRank)
                        != leadingCount
                    || elementCount(output.extents) != Math.multiplyExact(
                        normalizedCount, leadingCount))
                throw new IllegalArgumentException("trailing-normalization geometry is invalid");
            int nextBoundary = 0;
            for (int position = 0; position < positionToBoundary.size(); position++) {
                int boundary = positionToBoundary.get(position);
                if (boundary < 0 || boundary >= inputs.size() || boundary > nextBoundary)
                    throw new IllegalArgumentException(
                            "trailing-normalization boundary map is invalid");
                if (boundary == nextBoundary) nextBoundary++;
                int first = positionToBoundary.indexOf(boundary);
                if (inputTypes.get(position) != inputTypes.get(first))
                    throw new IllegalArgumentException(
                            "repeated trailing-normalization boundary type disagrees");
            }
            for (int boundary = 0; boundary < inputs.size(); boundary++) {
                Layout layout = inputs.get(boundary);
                int position = positionToBoundary.indexOf(boundary);
                if (position == 0) {
                    if (!Arrays.equals(layout.extents, output.extents))
                        throw new IllegalArgumentException(
                                "trailing-normalization input Shape disagrees");
                } else {
                    if (layout.extents.length != normalizedRank
                            || !Arrays.equals(layout.extents, Arrays.copyOfRange(output.extents,
                                output.extents.length - normalizedRank, output.extents.length)))
                        throw new IllegalArgumentException(
                                "trailing-normalization parameter Shape disagrees");
                }
            }
            validateInjective(output);
        }
        /**
         * Returns exact workspace bytes for the stated simultaneous complete-slice ranges.
         *
         * @param ranges non-negative number of simultaneous ranges
         * @return exact Layer workspace bytes, or zero for RMS and empty work
         * @throws ArithmeticException if multiplication overflows
         */
        public long workspaceBytes(int ranges) { return Math.multiplyExact(scratchSliceBytes, ranges); }
        /**
         * Packs carrier-relative bases and resolved layouts for one direct generated entry.
         *
         * @param bases carrier-relative element bases for every unique input followed by output;
         *     not retained or mutated
         * @return a new primitive geometry array owned by the caller
         * @throws NullPointerException if {@code bases} is {@code null}
         * @throws IllegalArgumentException if the base count does not match the boundaries
         * @throws ArithmeticException if applying a resolved layout offset overflows
         */
        public long[] pack(long[] bases) {
            if (bases.length != inputs.size() + 1) throw new IllegalArgumentException(
                    "trailing-normalization carrier base count disagrees");
            int size = 11 + bases.length;
            for (Layout layout : inputs) size += 1 + layout.extents.length * 2;
            size += 1 + output.extents.length * 2;
            long[] packed = new long[size]; int p = 0;
            packed[p++] = inputs.size(); packed[p++] = normalizedRank;
            packed[p++] = normalizedCount; packed[p++] = leadingCount;
            packed[p++] = output.extents.length; packed[p++] = scratchSliceBytes;
            packed[p++] = 0; packed[p++] = normalizedCount; packed[p++] = 0;
            packed[p++] = scratchSliceBytes; packed[p++] = 0;
            for (int i = 0; i < inputs.size(); i++) packed[p++] = Math.addExact(bases[i], inputs.get(i).offset);
            packed[p++] = Math.addExact(bases[bases.length - 1], output.offset);
            for (Layout layout : inputs) { packed[p++] = layout.extents.length;
                for (long v : layout.extents) packed[p++] = v;
                for (long v : layout.strides) packed[p++] = v; }
            packed[p++] = output.extents.length;
            for (long v : output.extents) packed[p++] = v;
            for (long v : output.strides) packed[p++] = v;
            return packed;
        }
    }
}
