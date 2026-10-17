package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuLossIr;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.loss.DenseCategoricalCrossEntropyWithLogitsAttrs;
import io.github.pho001.synaptik.model.operation.loss.IndexCategoricalCrossEntropyWithLogitsAttrs;
import io.github.pho001.synaptik.model.operation.loss.LossKind;
import io.github.pho001.synaptik.model.operation.loss.LossReduction;
import io.github.pho001.synaptik.model.operation.loss.MeanSquaredErrorAttrs;
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
 * Cold, fail-closed lowering for one atomic first-class loss occurrence.
 *
 * <p>This owner turns one already projected Model loss node into the CPU-private loss identity,
 * direct boundary list, and invocation geometry consumed by finalization.  It deliberately keeps
 * rank, extents, layouts, strides, carrier bases, and ignore-value bits out of generated-class
 * identity.  In particular, a dense categorical target has the logits rank, whereas an index
 * categorical target has one fewer axis because it has no class coordinate.</p>
 */
public final class CpuLossLowering {
    private final CpuCapabilityProvider capabilities = new CpuCapabilityProvider();
    /** Creates a stateless loss lowerer. */
    public CpuLossLowering() { }

    /**
     * Lowers one supported static loss without recognizing a decomposed topology.
     *
     * @param context exact non-null one-node CPU preparation projection
     * @return immutable unique-boundary direct-loss lowering; never {@code null}
     * @throws NullPointerException if {@code context} is null
     * @throws IllegalArgumentException if the occurrence is not one supported direct static loss
     * @throws ArithmeticException if cold element-count, layout-span, or invocation geometry
     *     arithmetic overflows
     */
    public CpuPartitionLowering.LoweredPartition lower(
            PrepareContext<? extends BackendAnalysisInputs> context) {
        Objects.requireNonNull(context, "context");
        if (context.nodes().size() != 1) throw new IllegalArgumentException("loss must be atomic");
        var node = context.nodes().getFirst();
        if (!(node.operation().kind() instanceof LossKind kind)) throw new IllegalArgumentException("loss kind required");
        Map<ValueId, GraphValue> values = new LinkedHashMap<>(); context.values().forEach(v -> values.put(v.id(), v));
        var query = new OperationCapabilityQuery(node.operation(), node.inputs().stream().map(i -> value(values, i).descriptor()).toList(), node.outputs().stream().map(i -> value(values, i).descriptor()).toList());
        if (!capabilities.supports(query)) throw new IllegalArgumentException("unsupported CPU loss");
        var unique = new ArrayList<ValueId>(); var rolePositions = new ArrayList<Integer>();
        for (ValueId input : node.inputs()) { int p = unique.indexOf(input); if (p < 0) { p = unique.size(); unique.add(input); } rolePositions.add(p); }
        var boundaries = new ArrayList<ValueId>(unique); boundaries.add(node.outputs().getFirst());
        var bindings = new ArrayList<CpuAccessPlan.Binding>(); var plans = new ArrayList<CpuAccessPlan>(); var spans = new ArrayList<Long>();
        for (int i = 0; i < boundaries.size(); i++) { GraphValue v = value(values, boundaries.get(i));
            var binding = binding(v, i == boundaries.size() - 1 ? CpuAccessPlan.AccessKind.WRITE : CpuAccessPlan.AccessKind.READ);
            bindings.add(binding); plans.add(binding.plan()); spans.add(v.descriptor().layout().orElseThrow().referencedElementSpan()); }
        GraphValue prediction = value(values, node.inputs().getFirst()), target = value(values, node.inputs().getLast()), output = value(values, node.outputs().getFirst());
        long[] extents = prediction.descriptor().shape().toLongArray(); int axis = -1; boolean ignore = false;
        long ignoreValue = 0L; LossReduction reduction;
        if (node.operation().attrs() instanceof MeanSquaredErrorAttrs a) reduction = a.reduction();
        else if (node.operation().attrs() instanceof DenseCategoricalCrossEntropyWithLogitsAttrs a) { reduction = a.reduction(); axis = a.axis(); }
        else if (node.operation().attrs() instanceof IndexCategoricalCrossEntropyWithLogitsAttrs a) {
            reduction = a.reduction(); axis = a.axis(); ignore = a.ignoreIndex().isPresent();
            if (ignore) {
                var value = a.ignoreIndex().orElseThrow();
                ignoreValue = value.dataType() == io.github.pho001.synaptik.model.datatype.DataType.INT32
                        ? value.int32Value() : value.int64Value();
            }
        }
        else throw new IllegalArgumentException("loss attributes disagree");
        long domain = kind == LossKind.MEAN_SQUARED_ERROR ? product(extents) : samples(extents, axis);
        int targetRank = kind == LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS
                ? extents.length - 1 : extents.length;
        int outputRank = reduction == LossReduction.NONE
                ? kind == LossKind.MEAN_SQUARED_ERROR ? extents.length : extents.length - 1
                : 0;
        var geometry = new Geometry(extents, axis, targetRank, outputRank,
                prediction.descriptor().layout().orElseThrow().storageOffset(),
                target.descriptor().layout().orElseThrow().storageOffset(),
                output.descriptor().layout().orElseThrow().storageOffset(),
                prediction.descriptor().layout().orElseThrow().strides(),
                target.descriptor().layout().orElseThrow().strides(),
                output.descriptor().layout().orElseThrow().strides(), ignore, ignoreValue);
        var types = boundaries.stream().map(i -> value(values, i).descriptor().dataType()).toList();
        var ir = new CpuLossIr(kind, prediction.descriptor().dataType(), target.descriptor().dataType(), output.descriptor().dataType(), reduction, ignore, rolePositions, types, plans, reduction == LossReduction.NONE ? CpuLossIr.RangeForm.INDEPENDENT_DOMAIN : CpuLossIr.RangeForm.COMPLETE_REDUCTION, geometry);
        // A reduced loss owns its scalar publication even for an empty domain.  In particular,
        // SUM publishes +0 and MEAN publishes NaN; representing an empty reduction as no range
        // would silently skip that required store.
        long work = reduction == LossReduction.NONE ? domain : 1;
        return new CpuPartitionLowering.LoweredPartition(ir, boundaries, bindings, spans, types, List.of(), new long[] {work}, work, "legal: atomic direct static loss", new long[0], Optional.empty());
    }
    private static GraphValue value(Map<ValueId, GraphValue> values, ValueId id) { var value = values.get(id); if (value == null) throw new IllegalArgumentException("loss value is not projected"); return value; }
    private static CpuAccessPlan.Binding binding(GraphValue value, CpuAccessPlan.AccessKind kind) {
        LayoutDescriptor layout = value.descriptor().layout().orElseThrow(); long[] extents = value.descriptor().shape().toLongArray(), strides = layout.strides(); int suffix = 0; long expected = 1;
        for (int i = extents.length - 1; i >= 0 && strides[i] == expected; i--) { suffix++; expected = Math.multiplyExact(expected, Math.max(1, extents[i])); }
        var roles = new ArrayList<CpuAccessPlan.AxisRole>(); for (int i = 0; i < extents.length; i++) roles.add(strides[i] == 0 ? CpuAccessPlan.AxisRole.BROADCAST : i >= extents.length - suffix ? CpuAccessPlan.AxisRole.CONTIGUOUS : CpuAccessPlan.AxisRole.STRIDED);
        var plan = new CpuAccessPlan(kind, suffix == extents.length ? CpuAccessPlan.Regime.DENSE_LINEAR : CpuAccessPlan.Regime.GENERAL_ODOMETER, extents.length, roles, suffix); long count = product(extents);
        return CpuAccessPlan.Binding.create(plan, extents, layout.storageOffset(), strides, count, 0, count, layout.referencedElementSpan());
    }
    private static long product(long[] v) { long result = 1; for (long x : v) { if (x == 0) return 0; result = Math.multiplyExact(result, x); } return result; }
    private static long samples(long[] v, int axis) { long result = 1; for (int i = 0; i < v.length; i++) if (i != axis) { if (v[i] == 0) return 0; result = Math.multiplyExact(result, v[i]); } return result; }

    /**
     * Immutable cold loss geometry; it never contributes to generated-class identity.
     *
     * <p>{@code extents} always describes the prediction/logits tensor.  {@code targetRank}
     * describes the coordinate space used by target strides: it equals logits rank for MSE and
     * dense categorical loss, and logits rank minus one for index categorical loss.
     * {@code outputRank} is independent: it equals logits rank for MSE {@code NONE}, logits rank
     * minus one for categorical {@code NONE}, and zero for reductions.  This distinction
     * prevents an index target from consuming a fabricated class coordinate while preserving
     * dense targets' exact logits-shaped layout.</p>
     *
     * @param extents non-null non-negative prediction/logits extents
     * @param axis {@code -1} for MSE, otherwise normalized non-negative logits class axis
     * @param targetRank target coordinate rank, as described above
     * @param outputRank {@code NONE} output coordinate rank, or zero for scalar reductions
     * @param predictionOffset non-negative prediction/logits element offset in its selected
     *     carrier region
     * @param targetOffset non-negative target element offset in its selected carrier region
     * @param outputOffset non-negative result element offset in its selected carrier region
     * @param predictionStrides non-null non-negative prediction/logits element strides
     * @param targetStrides non-null non-negative target element strides
     * @param outputStrides non-null non-negative {@code NONE} output element strides, or empty
     *     for scalar reductions
     * @param ignorePresent whether a cold exact index-ignore value is present
     * @param ignoreValue cold exact integral ignore value when present
     */
    public record Geometry(long[] extents, int axis, int targetRank, int outputRank,
            long predictionOffset, long targetOffset, long outputOffset,
            long[] predictionStrides, long[] targetStrides,
            long[] outputStrides, boolean ignorePresent, long ignoreValue) {
        /**
         * Returns a rank-zero unavailable marker used only by legacy structural constructors.
         *
         * @return a non-null isolated geometry value that cannot be mistaken for a categorical
         *     class-axis occurrence
         */
        public static Geometry unavailable() {
            return new Geometry(new long[0], -1, 0, 0, 0L, 0L, 0L,
                    new long[0], new long[0], new long[0], false, 0L);
        }
        /**
         * Validates and defensively snapshots direct invocation geometry.
         *
         * @throws NullPointerException if an extent or stride array is {@code null}
         * @throws IllegalArgumentException if ranks, axis, offsets, or non-negative layout facts
         *     cannot represent MSE, dense categorical, or index categorical invocation geometry
         */
        public Geometry {
            extents = Objects.requireNonNull(extents, "extents").clone();
            predictionStrides = Objects.requireNonNull(predictionStrides, "predictionStrides").clone();
            targetStrides = Objects.requireNonNull(targetStrides, "targetStrides").clone();
            outputStrides = Objects.requireNonNull(outputStrides, "outputStrides").clone();
            if (axis < -1 || axis >= extents.length
                    || targetRank < 0 || targetRank > extents.length
                    || outputRank < 0 || outputRank > extents.length
                    || predictionOffset < 0 || targetOffset < 0 || outputOffset < 0
                    || axis < 0 && targetRank != extents.length
                    || axis < 0 && outputRank != extents.length && outputRank != 0
                    || axis >= 0 && targetRank != extents.length
                            && targetRank != extents.length - 1
                    || axis >= 0 && outputRank != extents.length - 1 && outputRank != 0
                    || predictionStrides.length != extents.length
                    || targetStrides.length != targetRank
                    || outputStrides.length != outputRank
                    || Arrays.stream(extents).anyMatch(x -> x < 0)
                    || Arrays.stream(predictionStrides).anyMatch(x -> x < 0)
                    || Arrays.stream(targetStrides).anyMatch(x -> x < 0)
                    || Arrays.stream(outputStrides).anyMatch(x -> x < 0)) {
                throw new IllegalArgumentException("loss geometry is invalid");
            }
        }
        /**
         * Returns a defensive copy of prediction/logits extents.
         *
         * @return a new non-null ordered extent array
         */
        @Override public long[] extents() {
            return extents.clone();
        }

        /**
         * Returns a defensive copy of prediction/logits element strides.
         *
         * @return a new non-null ordered stride array
         */
        @Override public long[] predictionStrides() {
            return predictionStrides.clone();
        }

        /**
         * Returns a defensive copy of target element strides in target coordinate order.
         *
         * @return a new non-null ordered stride array with {@link #targetRank()} entries
         */
        @Override public long[] targetStrides() {
            return targetStrides.clone();
        }

        /**
         * Returns a defensive copy of {@code NONE} result element strides.
         *
         * @return a new non-null ordered stride array, empty for scalar reductions
         */
        @Override public long[] outputStrides() {
            return outputStrides.clone();
        }

        /**
         * Packs cold layout data with represented carrier bases for one generated invocation.
         * The format is logits rank, normalized axis, target rank, output rank, three role
         * bases, ignore presence/value, and static logical-domain count, then logits extents and
         * role strides. The static count is the MSE element count or categorical non-class sample
         * count; it is the {@code MEAN} denominator for MSE and dense categorical loss. Index
         * categorical {@code MEAN} instead uses its run-local non-ignored count. Index target
         * strides deliberately have rank one less than logits, while dense target strides retain
         * logits rank and categorical {@code NONE} output strides have rank one less than logits.
         *
         * @param roleBases prediction/logits, target, and result element bases
         * @return a new primitive-only generated-entry geometry payload
         * @throws NullPointerException if {@code roleBases} is {@code null}
         * @throws IllegalArgumentException if {@code roleBases} does not contain exactly three
         *     semantic role bases
         * @throws ArithmeticException if a carrier base plus its cold layout offset overflows
         */
        public long[] pack(long[] roleBases) {
            Objects.requireNonNull(roleBases, "roleBases");
            if (roleBases.length != 3) throw new IllegalArgumentException("loss role bases required");
            long[] packed = new long[10 + extents.length + predictionStrides.length
                    + targetStrides.length + outputStrides.length];
            packed[0] = extents.length;
            packed[1] = axis;
            packed[2] = targetRank;
            packed[3] = outputRank;
            packed[4] = Math.addExact(roleBases[0], predictionOffset);
            packed[5] = Math.addExact(roleBases[1], targetOffset);
            packed[6] = Math.addExact(roleBases[2], outputOffset);
            packed[7] = ignorePresent ? 1L : 0L;
            packed[8] = ignoreValue;
            packed[9] = domainCount();
            int next = 10;
            System.arraycopy(extents, 0, packed, next, extents.length); next += extents.length;
            System.arraycopy(predictionStrides, 0, packed, next, predictionStrides.length);
            next += predictionStrides.length;
            System.arraycopy(targetStrides, 0, packed, next, targetStrides.length);
            next += targetStrides.length;
            System.arraycopy(outputStrides, 0, packed, next, outputStrides.length);
            return packed;
        }

        private long domainCount() {
            long result = 1L;
            for (int coordinate = 0; coordinate < extents.length; coordinate++) {
                if (coordinate == axis) {
                    continue;
                }
                if (extents[coordinate] == 0L) {
                    return 0L;
                }
                result = Math.multiplyExact(result, extents[coordinate]);
            }
            if (axis < 0) {
                // MSE has no omitted class coordinate, so the loop above already included every
                // extent.  Keep this branch only to make the semantic distinction explicit.
                return result;
            }
            return result;
        }
    }
}
