package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.scan.ScanKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpression;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.ExpressionDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.ScanBody;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.LongFunction;
import java.util.function.LongPredicate;
import java.util.function.LongUnaryOperator;

/**
 * Compiler-owned, storage-free admission recipe for one named symbolic batch dimension.
 * Preparation indexes final input/value positions, typed obligations and logical layout intent.
 * Binding reads descriptors only, never tensors or a graph; intermediate descriptors are resolved
 * only when explicitly requested. Instances are immutable and safe to reuse across concurrent runs.
 * Reshape recipes require compile-time proof of contiguous source intent; no invocation performs
 * layout inference. Scope-local obligations and checked suffix counts are eager even for dead ports.
 * Each numeric check retains its outer/scoped occurrence and semantic subject. Failed binding
 * renders that association and actual B; successful binding performs no diagnostic rendering.
 */
public final class BatchBindingPlan {
    private final List<GraphValue> values;
    private final int[] inputs;
    private final int[] inputPosition;
    private final String symbol;
    private final List<List<LongUnaryOperator>> dimensions;
    private final List<LongFunction<long[]>> strides;
    private final boolean[] views;
    private final long[] offsets;
    private final List<Obligation> obligations;

    private BatchBindingPlan(List<GraphValue> values, int[] inputs, int[] inputPosition,
            String symbol, List<List<LongUnaryOperator>> dimensions,
            List<LongFunction<long[]>> strides, boolean[] views, long[] offsets, List<Obligation> obligations) {
        this.values = values; this.inputs = inputs; this.inputPosition = inputPosition;
        this.symbol = symbol; this.dimensions = dimensions; this.strides = strides;
        this.views = views; this.offsets = offsets; this.obligations = obligations;
    }

    /**
     * Prepares final graph metadata without changing the eight CompileArtifacts components.
     * Scoped Scan predicates and all local declarations are indexed separately for each containing
     * occurrence. All canonical Scan ports, including dead siblings, acquire non-view layout recipes.
     * @param artifacts non-null final immutable compile result
     * @return non-null immutable named-batch admission plan
     * @throws NullPointerException if artifacts is null
     * @throws IllegalArgumentException if no caller axis declares B, another symbol or unknown
     *     occurs, or a logical layout cannot be proved from the final graph
     * @throws ArithmeticException if static metadata arithmetic overflows
     */
    public static BatchBindingPlan prepare(CompileArtifacts artifacts) {
        Objects.requireNonNull(artifacts, "artifacts");
        var graph = artifacts.graph();
        Map<ValueId, Integer> positions = new HashMap<>();
        Map<ValueId, NodeId> owners = new HashMap<>();
        for (var node : graph.nodes()) for (ValueId output : node.outputs()) owners.put(output, node.id());
        for (int i = 0; i < graph.values().size(); i++) positions.put(graph.values().get(i).id(), i);
        int[] inputs = artifacts.constants().bindableInputs().stream()
                .mapToInt(positions::get).toArray();
        String symbol = null;
        for (int input : inputs) for (Dimension d : graph.values().get(input).descriptor().shape().dimensions()) {
            if (d instanceof DynamicDimension named) {
                if (symbol == null) symbol = named.symbol();
                else if (!symbol.equals(named.symbol())) throw new IllegalArgumentException("multiple batch symbols");
            }
        }
        if (symbol == null) throw new IllegalArgumentException("batch symbol must occur on a caller axis");
        int size = graph.values().size();
        int[] inputPosition = new int[size]; java.util.Arrays.fill(inputPosition, -1);
        for (int i = 0; i < inputs.length; i++) inputPosition[inputs[i]] = i;
        var dimensions = new ArrayList<List<LongUnaryOperator>>(size);
        var strides = new ArrayList<LongFunction<long[]>>(size);
        boolean[] views = new boolean[size];
        boolean[] contiguous = new boolean[size];
        long[] offsets = new long[size];
        for (GraphValue value : graph.values()) {
            var ds = new ArrayList<LongUnaryOperator>();
            for (Dimension d : value.descriptor().shape().dimensions()) ds.add(dimension(d, symbol));
            dimensions.add(List.copyOf(ds));
            var layout = value.descriptor().layout();
            if (layout.isPresent()) {
                long[] fixed = layout.orElseThrow().strides();
                strides.add(ignored -> fixed.clone());
            } else {
                List<LongUnaryOperator> extents = dimensions.getLast();
                strides.add(batch -> canonicalStrides(extents, batch));
            }
            views[dimensions.size() - 1] = layout.isPresent() && layout.orElseThrow().isView();
            contiguous[dimensions.size() - 1] = layout.isPresent() ? layout.orElseThrow().isContiguous()
                    : !value.descriptor().shape().isFullyStatic();
            offsets[dimensions.size() - 1] = layout.isPresent() ? layout.orElseThrow().storageOffset() : 0;
        }
        var obligations = new ArrayList<Obligation>();
        for (var node : graph.nodes()) {
            var kind = node.operation().kind();
            if (kind == ScanKind.SCAN && node.scanBody().isPresent()) {
                var body = node.scanBody().orElseThrow();
                // Each visit owns this occurrence's complete lexical inventory. Never index a
                // body obligation by bare ValueId/NodeId, which may equal IDs in another scope.
                Map<ValueId, NodeId> localOwners = new HashMap<>();
                for (var localNode : body.graph().nodes()) for (ValueId output : localNode.outputs()) {
                    localOwners.put(output, localNode.id());
                }
                for (var local : body.graph().values()) {
                    indexValidity(local.descriptor().shape(), symbol, obligations,
                            node.id(), localOwners.get(local.id()), local.id());
                }
                for (var constraint : CapturedGraphInference.inferAndValidate(body.graph()).constraints()) {
                    obligations.add(new Obligation(GraphPredicateProof.bind(constraint.predicate(), symbol),
                            node.id(), constraint.nodeId(), null, constraint.subject()));
                }
                for (var id : node.outputs()) {
                    int out = positions.get(id);
                    views[out] = false;
                    offsets[out] = 0;
                    List<LongUnaryOperator> extents = dimensions.get(out);
                    strides.set(out, batch -> canonicalStrides(extents, batch));
                }
                continue;
            }
            if (node.outputs().size() != 1) {
                if (node.outputs().stream().anyMatch(id -> !graph.values().get(positions.get(id)).descriptor().shape().isFullyStatic())) {
                    throw new IllegalArgumentException("unsupported symbolic multi-output layout");
                }
                continue;
            }
            int out = positions.get(node.outputs().getFirst());
            var descriptor = graph.values().get(out).descriptor();
            if (descriptor.layout().isPresent()) continue;
            if (kind instanceof AxisTransformKind || kind instanceof ShapeTransformKind) {
                int in = positions.get(node.inputs().getFirst());
                LongFunction<long[]> source = strides.get(in);
                List<LongUnaryOperator> sourceDimensions = dimensions.get(in);
                List<LongUnaryOperator> targetDimensions = dimensions.get(out);
                if (kind == AxisTransformKind.PERMUTE) {
                    List<Integer> axes = ((PermutationAttrs) node.operation().attrs()).axes();
                    strides.set(out, batch -> {
                        long[] s = source.apply(batch), result = new long[axes.size()];
                        for (int i = 0; i < result.length; i++) result[i] = s[axes.get(i)];
                        return result;
                    });
                    contiguous[out] = contiguous[in];
                    for (int i = 0; i < axes.size(); i++) contiguous[out] &= axes.get(i) == i;
                } else if (kind == ShapeTransformKind.EXPAND) {
                    strides.set(out, batch -> {
                        long[] s = source.apply(batch), result = new long[targetDimensions.size()];
                        int shift = result.length - s.length;
                        for (int i = shift; i < result.length; i++) {
                            int x = i - shift;
                            result[i] = sourceDimensions.get(x).applyAsLong(batch) == 1
                                    && targetDimensions.get(i).applyAsLong(batch) != 1 ? 0 : s[x];
                        }
                        return result;
                    });
                    contiguous[out] = contiguous[in] && graph.values().get(in).descriptor().shape().equals(descriptor.shape());
                } else if (kind == ShapeTransformKind.RESHAPE) {
                    if (!contiguous[in]) throw new IllegalArgumentException("unproved symbolic reshape source at " + node.id());
                    // A cold structural proof, not layout inference at bind. Element-count
                    // equality remains a separately indexed private semantic obligation.
                    strides.set(out, batch -> canonicalStrides(targetDimensions, batch));
                    contiguous[out] = true;
                } else {
                    int axis = ((AxisTransformAttrs) node.operation().attrs()).axis();
                    boolean insert = kind == AxisTransformKind.EXPAND_DIMS;
                    strides.set(out, batch -> {
                        long[] s = source.apply(batch), result = new long[targetDimensions.size()];
                        for (int i = 0; i < result.length; i++) {
                            result[i] = insert && i == axis ? axis == s.length ? 1
                                    : Math.multiplyExact(s[axis], sourceDimensions.get(axis).applyAsLong(batch))
                                    : s[insert ? i < axis ? i : i - 1 : i < axis ? i : i + 1];
                        }
                        return result;
                    });
                    contiguous[out] = contiguous[in];
                }
                views[out] = true;
                offsets[out] = offsets[in];
            } else if (!ScanBody.hasCanonicalContiguousResults(node.operation())) {
                throw new IllegalArgumentException("unproved symbolic result layout at " + node.id());
            }
        }
        // Evaluate preindexed numeric validity even for non-input/dead declarations. This creates
        // no per-run Shape/descriptor and cannot defer a negative derived extent until publication.
        for (int i = 0; i < dimensions.size(); i++) {
            GraphValue value = graph.values().get(i);
            for (int axis = 0; axis < dimensions.get(i).size(); axis++) {
                var extent = dimensions.get(i).get(axis);
                obligations.add(new Obligation(batch -> extent.applyAsLong(batch) >= 0,
                        owners.get(value.id()), null, value.id(), value.descriptor().shape().dimension(axis)));
            }
        }
        for (var value : graph.values()) indexValidity(value.descriptor().shape(), symbol, obligations,
                owners.get(value.id()), null, value.id());
        for (var constraint : artifacts.diagnostics().constraints()) {
            obligations.add(new Obligation(GraphPredicateProof.bind(constraint.predicate(), symbol),
                    constraint.nodeId(), null, null, constraint.subject()));
        }
        return new BatchBindingPlan(graph.values(), inputs, inputPosition, symbol,
                List.copyOf(dimensions), List.copyOf(strides), views, offsets, List.copyOf(obligations));
    }

    /**
     * Indexes complete checked suffix geometry even for dead or zero-time/zero-batch domains.
     * @param shape exact outer or scoped local declaration
     * @param symbol sole caller-owned name
     * @param obligations preparation-local ordered numeric checks, appended without deduplication
     * @param occurrence owning outer producer, or null for an outer input
     * @param scopedOccurrence local producer, or null for an outer value or local input
     * @param value exact outer or scoped value ID
     * @throws IllegalArgumentException if any extent cannot be bound
     */
    private static void indexValidity(Shape shape, String symbol, List<Obligation> obligations,
            NodeId occurrence, NodeId scopedOccurrence, ValueId value) {
        var count = count(shape, symbol);
        obligations.add(new Obligation(batch -> count.applyAsLong(batch) >= 0,
                occurrence, scopedOccurrence, value, shape));
    }

    /**
     * One cold-indexed check with scope-qualified failure context; no subject is rendered on success.
     * @param predicate checked numeric admission rule
     * @param occurrence owning outer occurrence, or null for an input
     * @param scopedOccurrence owning body occurrence, or null outside a body or for a local input
     * @param value associated value, or null for a semantic predicate
     * @param subject original constraint subject or exact geometry, rendered only on failure
     */
    private record Obligation(LongPredicate predicate, NodeId occurrence, NodeId scopedOccurrence,
            ValueId value, Object subject) {
        /**
         * Checks one actual invocation, preserving the original exception category and cause.
         * @param batch admitted caller-owned nonnegative B
         * @throws IllegalArgumentException if the predicate is false or its extent is invalid
         * @throws ArithmeticException if checked predicate/geometry arithmetic overflows
         */
        void validate(long batch) {
            boolean satisfied;
            try {
                satisfied = predicate.test(batch);
            } catch (IllegalArgumentException failure) {
                throw new IllegalArgumentException(context(batch) + ": " + failure.getMessage(), failure);
            } catch (ArithmeticException failure) {
                var contextual = new ArithmeticException(context(batch) + ": " + failure.getMessage());
                contextual.initCause(failure);
                throw contextual;
            }
            if (!satisfied) throw new IllegalArgumentException(context(batch) + ": predicate is unsatisfied");
        }

        /**
         * Renders only a failed obligation's cold association, never inspecting tensor values.
         * @param batch actual invocation extent
         * @return occurrence-qualified diagnostic including the original subject
         */
        private String context(long batch) {
            return "actualB=" + batch + ", outer=" + occurrence + ", scoped=" + scopedOccurrence
                    + ", value=" + value + ", subject=" + subject;
        }
    }

    /**
     * Admits fully static actual descriptors in final compiled caller-input order.
     * Dynamic inputs require present, zero-offset canonical dense layouts; read views may qualify.
     * Every declared occurrence agrees on B, including otherwise unused input positions.
     * Predicate and checked-geometry failures identify actual B, the owning outer/scoped
     * occurrence and the retained semantic subject without rendering diagnostics on success.
     * @param actualInputs non-null ordered descriptors without nulls; membership is snapshotted
     * @return fresh immutable binding, retaining no tensor or storage
     * @throws NullPointerException if the list or a descriptor is null
     * @throws IllegalArgumentException if count, shape, type, gradient flag, layout or predicate fails
     * @throws ArithmeticException if checked geometry or predicate arithmetic overflows
     */
    public Binding bind(List<TensorDescriptor> actualInputs) {
        List<TensorDescriptor> actual = List.copyOf(Objects.requireNonNull(actualInputs, "actualInputs"));
        if (actual.size() != inputs.length) throw new IllegalArgumentException("actual input count disagrees");
        long batch = -1;
        for (int i = 0; i < inputs.length; i++) {
            var template = values.get(inputs[i]).descriptor(); var supplied = actual.get(i);
            if (!supplied.shape().isFullyStatic() || supplied.shape().rank() != template.shape().rank()
                    || supplied.dataType() != template.dataType() || supplied.requiresGrad() != template.requiresGrad()) {
                throw new IllegalArgumentException("actual input descriptor disagrees at " + i);
            }
            for (int axis = 0; axis < template.shape().rank(); axis++) {
                if (template.shape().dimension(axis) instanceof DynamicDimension d && d.symbol().equals(symbol)) {
                    long candidate = supplied.shape().dimension(axis).staticSize().orElseThrow();
                    if (batch >= 0 && batch != candidate) throw new IllegalArgumentException("batch dimensions disagree");
                    batch = candidate;
                }
            }
        }
        if (batch < 0) throw new IllegalArgumentException("batch size is absent");
        for (int i = 0; i < inputs.length; i++) {
            int value = inputs[i]; var template = values.get(value).descriptor(); var supplied = actual.get(i);
            if (template.shape().isFullyStatic()) {
                if (!template.equals(supplied)) throw new IllegalArgumentException("fixed input descriptor disagrees at " + i);
            } else {
                for (int axis = 0; axis < dimensions.get(value).size(); axis++) {
                    if (dimensions.get(value).get(axis).applyAsLong(batch)
                            != supplied.shape().dimension(axis).staticSize().orElseThrow()) {
                        throw new IllegalArgumentException("actual input axis disagrees at " + i);
                    }
                }
                var layout = supplied.layout().orElseThrow(() -> new IllegalArgumentException("dynamic input layout is absent"));
                if (layout.storageOffset() != 0 || !java.util.Arrays.equals(layout.strides(),
                        LayoutDescriptor.contiguous(supplied.shape()).strides())) {
                    throw new IllegalArgumentException("dynamic input must be canonical dense");
                }
            }
        }
        for (Obligation obligation : obligations) obligation.validate(batch);
        return new Binding(this, batch, actual);
    }

    /** An immutable invocation's semantic binding; descriptors are lazy, not eagerly cached. */
    public static final class Binding {
        private final BatchBindingPlan plan;
        private final long batch;
        private final List<TensorDescriptor> inputs;
        private Binding(BatchBindingPlan plan, long batch, List<TensorDescriptor> inputs) {
            this.plan = plan; this.batch = batch; this.inputs = inputs;
        }
        /** Returns the admitted invocation extent.
         * @return actual non-negative B derived from declared caller axes */
        public long batchSize() { return batch; }
        /** Returns the lazy descriptor index domain.
         * @return final graph value-list size, not the maximum numeric ValueId */
        public int valueCount() { return plan.values.size(); }
        /**
         * Resolves only the requested final value descriptor, without graph traversal/inference.
         * @param valueIndex dense final graph value-list index
         * @return non-null actual descriptor, retaining generated value gradient eligibility
         * @throws IndexOutOfBoundsException if the value index is outside the final list
         * @throws IllegalArgumentException if a bound extent is negative
         * @throws ArithmeticException if requested shape/layout arithmetic overflows
         */
        public TensorDescriptor descriptor(int valueIndex) {
            TensorDescriptor template = plan.values.get(valueIndex).descriptor();
            int input = plan.inputPosition[valueIndex];
            if (input >= 0) return inputs.get(input);
            if (template.shape().isFullyStatic() && template.layout().isPresent()) return template;
            long[] extents = new long[plan.dimensions.get(valueIndex).size()];
            for (int axis = 0; axis < extents.length; axis++) extents[axis] = plan.dimensions.get(valueIndex).get(axis).applyAsLong(batch);
            Shape shape = Shape.of(extents);
            LayoutDescriptor layout = plan.views[valueIndex]
                    ? LayoutDescriptor.of(shape, plan.strides.get(valueIndex).apply(batch), plan.offsets[valueIndex], true)
                    : LayoutDescriptor.contiguous(shape);
            return new TensorDescriptor(template.dataType(), shape, java.util.Optional.of(layout), template.requiresGrad());
        }
    }

    /**
     * Indexes one typed dimension formula once, retaining only checked numeric evaluation.
     * @param dimension non-null fixed, named or exact expression dimension
     * @param symbol non-null sole caller-declared batch symbol
     * @return immutable checked numeric formula
     * @throws IllegalArgumentException if a symbol is foreign, an extent is negative or unknown
     * @throws ArithmeticException if later checked formula evaluation overflows
     */
    static LongUnaryOperator dimension(Dimension dimension, String symbol) {
        if (dimension instanceof StaticDimension s) return ignored -> s.size();
        if (dimension instanceof DynamicDimension d) {
            if (!d.symbol().equals(symbol)) throw new IllegalArgumentException("unknown/output-only batch symbol " + d.symbol());
            return b -> b;
        }
        DimensionExpression expression = ((ExpressionDimension) dimension).expression();
        return switch (expression) {
            case DimensionExpression.Unknown ignored -> throw new IllegalArgumentException("anonymous dimension is not batch-bound");
            case DimensionExpression.LinearCombination linear -> {
                var terms = new ArrayList<LongUnaryOperator>(); var coefficients = new ArrayList<Long>();
                linear.coefficients().forEach((d, coefficient) -> { terms.add(dimension(d, symbol)); coefficients.add(coefficient); });
                yield b -> { long result = linear.offset(); for (int i = 0; i < terms.size(); i++) result = Math.addExact(result,
                        Math.multiplyExact(terms.get(i).applyAsLong(b), coefficients.get(i))); return nonnegative(result); };
            }
            case DimensionExpression.Product product -> {
                var terms = new ArrayList<LongUnaryOperator>(); var powers = new ArrayList<Long>();
                product.factors().forEach((d, power) -> { terms.add(dimension(d, symbol)); powers.add(power); });
                yield b -> { long result = product.coefficient(); for (int i = 0; i < terms.size(); i++) {
                    long base = terms.get(i).applyAsLong(b), power = powers.get(i), factor = 1;
                    while (power != 0) { if ((power & 1) != 0) factor = Math.multiplyExact(factor, base);
                        power >>>= 1; if (power != 0) base = Math.multiplyExact(base, base); }
                    result = Math.multiplyExact(result, factor); } return nonnegative(result); };
            }
            case DimensionExpression.FloorDivision floor -> {
                var value = dimension(floor.dividend(), symbol); yield b -> value.applyAsLong(b) / floor.divisor();
            }
            case DimensionExpression.CeilingDivision ceil -> {
                var value = dimension(ceil.dividend(), symbol); yield b -> {
                    long v = value.applyAsLong(b); return v / ceil.divisor() + (v % ceil.divisor() == 0 ? 0 : 1); };
            }
        };
    }

    /**
     * Enforces numeric validity before descriptors or storage exist.
     * @param extent checked numeric formula result in elements
     * @return unchanged non-negative extent
     * @throws IllegalArgumentException if the extent is negative
     */
    private static long nonnegative(long extent) {
        if (extent < 0) throw new IllegalArgumentException("negative bound extent");
        return extent;
    }

    /**
     * Indexes checked right-to-left cardinality without creating an actual Shape during binding.
     * @param shape non-null symbolic or fixed domain
     * @param symbol non-null sole caller-declared batch symbol
     * @return immutable numeric count formula, checked even for trailing geometry behind zero
     * @throws IllegalArgumentException if a dimension cannot be bound
     * @throws ArithmeticException if subsequent count evaluation overflows
     */
    static LongUnaryOperator count(Shape shape, String symbol) {
        var dimensions = shape.dimensions().stream().map(d -> dimension(d, symbol)).toList();
        return batch -> { long result = 1; for (int i = dimensions.size() - 1; i >= 0; i--)
            result = Math.multiplyExact(result, dimensions.get(i).applyAsLong(batch)); return result; };
    }

    /**
     * Resolves row-major intent only for an explicitly requested descriptor.
     * @param dimensions immutable ordered checked extent formulas
     * @param batch admitted non-negative B
     * @return fresh element-stride array, including empty-domain trailing strides
     * @throws ArithmeticException if a checked suffix product overflows
     */
    private static long[] canonicalStrides(List<LongUnaryOperator> dimensions, long batch) {
        long[] strides = new long[dimensions.size()]; long suffix = 1;
        for (int i = strides.length - 1; i >= 0; i--) {
            strides[i] = suffix; suffix = Math.multiplyExact(suffix, dimensions.get(i).applyAsLong(batch));
        }
        return strides;
    }
}
