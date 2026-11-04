package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.*;
import io.github.pho001.synaptik.model.operation.attention.*;
import io.github.pho001.synaptik.model.operation.convolution.*;
import io.github.pho001.synaptik.model.operation.elementwise.binary.*;
import io.github.pho001.synaptik.model.operation.elementwise.cast.*;
import io.github.pho001.synaptik.model.operation.elementwise.classification.*;
import io.github.pho001.synaptik.model.operation.elementwise.comparison.*;
import io.github.pho001.synaptik.model.operation.elementwise.logical.*;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.*;
import io.github.pho001.synaptik.model.operation.elementwise.selection.*;
import io.github.pho001.synaptik.model.operation.elementwise.unary.*;
import io.github.pho001.synaptik.model.operation.index.*;
import io.github.pho001.synaptik.model.operation.layout.*;
import io.github.pho001.synaptik.model.operation.linalg.*;
import io.github.pho001.synaptik.model.operation.loss.*;
import io.github.pho001.synaptik.model.operation.normalization.*;
import io.github.pho001.synaptik.model.operation.ordering.*;
import io.github.pho001.synaptik.model.operation.pooling.*;
import io.github.pho001.synaptik.model.operation.random.DropoutKind;
import io.github.pho001.synaptik.model.operation.random.GraphRngKind;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentScanKind;
import io.github.pho001.synaptik.model.operation.reduction.*;
import io.github.pho001.synaptik.model.operation.scan.*;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

/**
 * Reusable immutable containment of ordinary Tensor expressions for one scan step.
 * Formals are declared leaves, not callbacks or graph IDs. Lists preserve positions and exact
 * references; all formals are identity-distinct and storage/provenance-free. Shapes are fully
 * static or contain one occurrence of the same named batch dimension per value. Static formals
 * have canonical layouts; symbolic formals have unresolved numeric layouts. No batch size is bound.
 * Known ordinary pure operations are admitted independently of backend or derivative support.
 * Hidden leaves, random/effectful operations and nested scans are excluded.
 *
 * <p>Validation reuses ordinary descriptor-only construction guards and compares every canonical
 * sibling without allocating Tensor identities or constructing a validation expression graph.
 * No scan occurrence, history storage, capture or execution is created here. Mutable host-storage
 * associations are rechecked when this template is bound. Collective lexical escape checking
 * belongs to Compiler, not this local source container.</p>
 *
 * <p>All list snapshots are immutable and retain exact Tensor references. The container does not
 * freeze those Tensors' borrowed host-storage associations. Reusing one body preserves its
 * source identities while each {@link Scan#apply} creates an independent outer occurrence.
 * Valid known-pure FLOAT16 TANH and FLOAT32 RELU expressions may be contained without granting
 * derivative rules or execution support for either scan.</p>
 *
 * <p>Named-shape checks also cover shape-bearing attributes and each family's deferred equality
 * or broadcast relations. A fixed non-singleton extent cannot silently specialize the declared
 * symbol, even when an operation's result shape is static. This is a source-shape proof, not
 * validation of runtime index values or numerical-domain obligations.</p>
 *
 * <p>This container has no lengths formal or row-role representation. Length-aware binding adds
 * explicit per-value coordinate, terminal-contribution and shared-state isolation proofs in
 * {@link Scan#apply}, while retaining the same known PURE families and ordinary per-value dtypes.
 * Explicit casts may change intermediate/emission dtype, but each next carry must still match
 * its own declaration's exact shape and dtype; no implicit conversion is inserted. All-valid
 * containment remains ordinary whole-Tensor algebra, without an invented batch coordinate.
 * Source typing alone grants no consumer or execution capability.</p>
 */
public final class ScanBody {
    private final List<Tensor> carry;
    private final List<Tensor> step;
    private final List<Tensor> invariants;
    private final List<Tensor> nextCarry;
    private final List<Tensor> emissions;
    private final List<Tensor> retainedValues;

    /**
     * Snapshots and validates a complete pure step, including unused canonical siblings.
     * Unused declared formals are allowed. Retained values follow producer postorder, then ascending
     * sibling slots; carry leaves are pre-step residuals, while step/invariant leaves are excluded.
     * Body depth is traversed iteratively; no depth cap or nested-scan expansion is introduced.
     *
     * @param carry non-null ordered non-null carried formals, possibly empty; each is a storage/provenance-free
     *     leaf, distinct by identity, with canonical static or unresolved bounded-symbolic geometry
     * @param step non-null, non-empty ordered non-null sequence-slice formals (axis zero removed),
     *     with the same leaf, named-shape, geometry and distinctness requirements as carry formals
     * @param invariants non-null ordered non-null invariant formals, possibly empty, with the
     *     same leaf, named-shape, geometry and distinctness requirements as carry formals
     * @param nextCarry non-null ordered non-null roots matching carry arity, shapes and dtypes;
     *     repeated identities are allowed and every contained leaf must be a declared formal
     * @param emissions non-null ordered non-null static or bounded-symbolic emission roots, possibly empty and
     *     repeatable, closed over the same declared formals; nextCarry plus emissions must be nonempty
     * @throws NullPointerException if a list or element is null
     * @throws IllegalArgumentException if declarations or roots violate their role contracts,
     *     a body wrapper is noncanonical, a value is host-backed, or contained operations are
     *     unknown, effectful, random, nested scans or invalid under ordinary Model guards
     *     or shapes introduce independent/output-only symbols, derived unknowns, repeated batch
     *     axes, undeclared shape-attribute symbols, or unproved named-coordinate transformations,
     *     family equalities or broadcasts
     * @throws ArithmeticException if static shape/layout arithmetic overflows
     */
    public ScanBody(List<Tensor> carry, List<Tensor> step, List<Tensor> invariants,
            List<Tensor> nextCarry, List<Tensor> emissions) {
        this.carry = List.copyOf(carry);
        this.step = List.copyOf(step);
        this.invariants = List.copyOf(invariants);
        this.nextCarry = List.copyOf(nextCarry);
        this.emissions = List.copyOf(emissions);
        carry = this.carry;
        step = this.step;
        invariants = this.invariants;
        nextCarry = this.nextCarry;
        emissions = this.emissions;
        if (step.isEmpty() || carry.size() != nextCarry.size()
                || nextCarry.isEmpty() && emissions.isEmpty()) {
            throw new IllegalArgumentException("scan requires step formals, matching carry and semantic roots");
        }
        Set<Tensor> formals = identitySet();
        DynamicDimension batch = null;
        // Include unused declarations: the symbol belongs to the template, not its live roots.
        for (Tensor formal : formals()) {
            for (Dimension dimension : formal.descriptor().shape().dimensions()) {
                if (dimension instanceof DynamicDimension named) {
                    if (batch != null && !batch.equals(named)) {
                        throw new IllegalArgumentException("scan body requires one named batch dimension");
                    }
                    batch = named;
                }
            }
        }
        for (Tensor formal : formals()) {
            requireShape(formal, batch);
            if (!formals.add(formal) || formal.provenance().isPresent()
                    || formal.hostStorage().isPresent()
                    || !formal.descriptor().layout().equals(formal.descriptor().shape().isFullyStatic()
                            ? java.util.Optional.of(LayoutDescriptor.contiguous(formal.descriptor().shape()))
                            : java.util.Optional.empty())) {
                throw new IllegalArgumentException("formals must be distinct canonical storage-free leaves");
            }
        }
        for (int i = 0; i < carry.size(); i++) {
            TensorDescriptor before = carry.get(i).descriptor();
            TensorDescriptor after = nextCarry.get(i).descriptor();
            if (before.dataType() != after.dataType() || !before.shape().equals(after.shape())) {
                throw new IllegalArgumentException("next carry shape/type mismatch at " + i);
            }
        }
        List<Tensor> retained = new ArrayList<>();
        Set<Tensor> seenValues = identitySet();
        Set<TensorProducer> seenProducers = identitySet();
        for (Tensor root : this.nextCarry) {
            visit(root, formals, seenValues, seenProducers, retained, batch);
        }
        for (Tensor root : this.emissions) {
            visit(root, formals, seenValues, seenProducers, retained, batch);
        }
        for (Tensor formal : this.carry) {
            if (seenValues.add(formal)) {
                retained.add(formal);
            }
        }
        this.retainedValues = List.copyOf(retained);
        validateStorage();
    }

    /**
     * Exposes prior-state declarations in carry binding order.
     * @return immutable ordered carried formals, including unused positions
     */
    public List<Tensor> carry() { return carry; }
    /**
     * Exposes the declarations bound to one original-time slice from each sequence.
     * @return immutable ordered axis-zero slice formals
     */
    public List<Tensor> step() { return step; }
    /**
     * Exposes step-independent declarations, including unused invariant positions.
     * @return immutable ordered invariant formals
     */
    public List<Tensor> invariants() { return invariants; }
    /**
     * Exposes the roots that replace carried values after each step.
     * @return immutable ordered next-carry roots, with repeated identities preserved
     */
    public List<Tensor> nextCarry() { return nextCarry; }
    /**
     * Exposes per-step roots stacked in original-time coordinates by a scan occurrence.
     * @return immutable ordered emission roots, with repeated identities preserved
     */
    public List<Tensor> emissions() { return emissions; }
    /**
     * Exposes producer-postordered values available for later history demand, not allocated saves.
     * Next-carry roots precede emission roots in traversal; each producer contributes all canonical
     * siblings in ascending output order. Unvisited carry formals are appended in declaration
     * order. Step/invariant leaves are excluded and repeated Tensor identities appear only once.
     * @return identity-unique deterministic latent residual bindings, including previous carry
     */
    public List<Tensor> retainedValues() { return retainedValues; }

    /**
     * Combines role declarations without removing unused positions.
     * @return a declaration-ordered snapshot of carry, step and invariant formals
     */
    List<Tensor> formals() {
        List<Tensor> result = new ArrayList<>(carry);
        result.addAll(step);
        result.addAll(invariants);
        return List.copyOf(result);
    }

    /**
     * Rechecks mutable storage on all declared and contained values before occurrence allocation.
     * @throws IllegalArgumentException if storage was attached since body construction
     */
    void validateStorage() {
        for (Tensor formal : formals()) {
            requireStorageFree(formal);
        }
        for (Tensor retained : retainedValues) {
            requireStorageFree(retained);
        }
    }

    private static <T> Set<T> identitySet() {
        return Collections.newSetFromMap(new IdentityHashMap<>());
    }

    private static void requireStorageFree(Tensor value) {
        if (value.hostStorage().isPresent()) {
            throw new IllegalArgumentException("scan body contains host-backed value");
        }
    }

    /**
     * Validates one complete local shape without substituting a numeric batch size.
     * @param value non-null formal or canonical contained value; must be storage-free
     * @param batch canonical declared symbol, or null for a fully static template
     * @throws IllegalArgumentException if dimensions or symbolic layout are unsupported
     * @throws ArithmeticException if a fully static element count overflows
     */
    private static void requireShape(Tensor value, DynamicDimension batch) {
        requireNamedShape(value.descriptor().shape(), batch);
        if (!value.descriptor().shape().isFullyStatic() && value.descriptor().layout().isPresent()) {
            throw new IllegalArgumentException("symbolic scan body layouts must remain unresolved");
        }
        value.descriptor().shape().knownElementCount();
        requireStorageFree(value);
    }

    /**
     * Checks named-coordinate vocabulary in a value or shape-bearing operation attribute.
     * @param shape non-null shape; static extents remain unchanged
     * @param batch sole symbol derived from all declarations, or null for a static template
     * @throws IllegalArgumentException if an undeclared, derived or repeated named axis appears
     */
    private static void requireNamedShape(Shape shape, DynamicDimension batch) {
        int namedAxes = 0;
        for (Dimension dimension : shape.dimensions()) {
            if (dimension instanceof StaticDimension) continue;
            if (!(dimension instanceof DynamicDimension named) || !named.equals(batch) || ++namedAxes > 1) {
                throw new IllegalArgumentException("scan body requires fixed extents or one declared named batch axis");
            }
        }
    }

    /**
     * Validates contained provenance and all canonical siblings in deterministic producer postorder.
     * Uses explicit entry/completion work so lexical depth does not consume the Java call stack.
     * @param value exact root or operand being visited; must be storage-free
     * @param formals identity set of all declarations, including unused ones
     * @param seenValues identity set used to append each retained value once
     * @param seenProducers identity set of producers whose operands/siblings have been inventoried
     * @param retained mutable declaration-scoped ordered residual inventory being assembled
     * @param batch sole declared canonical named dimension, or null for a static body
     * @throws IllegalArgumentException if containment, shape or ordinary family semantics fail
     * @throws ArithmeticException if static shape/layout or fixed-coordinate arithmetic overflows
     */
    private void visit(Tensor value, Set<Tensor> formals, Set<Tensor> seenValues,
            Set<TensorProducer> seenProducers, List<Tensor> retained, DynamicDimension batch) {
        var pending = new ArrayDeque<BodyVisit>();
        pending.addLast(new BodyVisit(value, false));
        while (!pending.isEmpty()) {
            BodyVisit visit = pending.removeLast();
            Tensor current = visit.value();
            if (!visit.complete()) {
                requireShape(current, batch);
                if (formals.contains(current)) {
                    if (seenValues.add(current) && carry.contains(current)) retained.add(current);
                    continue;
                }
                var provenance = current.provenance().orElseThrow(
                        () -> new IllegalArgumentException("undeclared scan body leaf"));
                TensorProducer producer = provenance.producer();
                if (producer.output(provenance.outputIndex()) != current) {
                    throw new IllegalArgumentException("noncanonical body wrapper");
                }
                if (!seenProducers.add(producer)) continue;
                pending.addLast(new BodyVisit(current, true));
                for (int i = producer.inputs().size() - 1; i >= 0; i--) {
                    pending.addLast(new BodyVisit(producer.inputs().get(i), false));
                }
            } else {
                TensorProducer producer = current.provenance().orElseThrow().producer();
                for (int i = 0; i < producer.outputCount(); i++) requireShape(producer.output(i), batch);
                validateOrdinaryOutputs(producer, batch);
                for (int i = 0; i < producer.outputCount(); i++) {
                    Tensor sibling = producer.output(i);
                    if (seenValues.add(sibling)) retained.add(sibling);
                }
            }
        }
    }

    /**
     * One lexical entry or producer-completion event, retaining the exact source identity.
     * @param value canonical root/operand whose producer is being visited
     * @param complete whether operands have already been visited in declaration order
     */
    private record BodyVisit(Tensor value, boolean complete) {}

    /**
     * Proves declared symbols and ordinary relations whose constructors defer symbolic equality.
     * Whole-Tensor reshape needs canonical count equality; length ROW mapping is checked separately.
     * @param operation known ordinary operation with family-validated canonical descriptors
     * @param inputs checked operand descriptors
     * @param outputs checked complete canonical sibling descriptors
     * @param batch sole declared symbol, or null for a static template; attributes may not add one
     * @throws IllegalArgumentException if a named-coordinate transformation cannot be proved
     * @throws ArithmeticException if a fixed feature product overflows
     */
    private static void validateSymbolicTransforms(Operation operation, List<TensorDescriptor> inputs,
            List<TensorDescriptor> outputs, DynamicDimension batch) {
        for (TensorDescriptor input : inputs) requireNamedShape(input.shape(), batch);
        for (TensorDescriptor output : outputs) requireNamedShape(output.shape(), batch);
        OperationAttrs attrs = operation.attrs();
        // Attributes may hide a symbol even when every operand and result is fully static.
        List<Shape> attributeShapes = switch (attrs) {
            case TargetShapeAttrs a -> List.of(a.targetShape());
            case SumToShapeAttrs a -> List.of(a.targetShape());
            case CropToShapeAttrs a -> List.of(a.targetShape(), a.prefixShape());
            case LayerNormAttrs a -> List.of(a.normalizedShape());
            case AffineLayerNormAttrs a -> List.of(a.normalizedShape());
            case RmsNormAttrs a -> List.of(a.normalizedShape());
            case Fold2dAttrs a -> List.of(a.outputShape());
            case Fold3dAttrs a -> List.of(a.outputShape());
            default -> List.of();
        };
        for (Shape shape : attributeShapes) requireNamedShape(shape, batch);
        if (inputs.stream().allMatch(d -> d.shape().isFullyStatic())
                && outputs.stream().allMatch(d -> d.shape().isFullyStatic())
                && attributeShapes.stream().allMatch(Shape::isFullyStatic)) return;
        OperationKind kind = operation.kind();
        Shape in = inputs.getFirst().shape();
        Shape out = outputs.getFirst().shape();
        validateDeferredFamilyShapes(operation, inputs, outputs);
        if (kind == ShapeTransformKind.RESHAPE) {
            if (!elementCountExpression(in).equals(elementCountExpression(out))) {
                throw new IllegalArgumentException("scan reshape requires proved element-count equality");
            }
        } else if (kind == ShapeTransformKind.EXPAND || attrs instanceof SumToShapeAttrs) {
            Shape source = attrs instanceof SumToShapeAttrs ? out : in;
            Shape target = attrs instanceof SumToShapeAttrs ? in : out;
            int offset = target.rank() - source.rank();
            for (int axis = 0; axis < source.rank(); axis++) {
                Dimension from = source.dimension(axis), to = target.dimension(axis + offset);
                if (!from.equals(to) && !from.equals(new StaticDimension(1))) {
                    throw new IllegalArgumentException("scan requires proved named-axis expansion/reduction");
                }
            }
        } else if (kind == MatmulKind.MATMUL) {
            Shape right = inputs.get(1).shape();
            if (!in.dimension(in.rank() - 1).equals(right.dimension(right.rank() == 1 ? 0 : right.rank() - 2))) {
                throw new IllegalArgumentException("scan matmul requires exact contraction dimensions");
            }
            int leftBatch = Math.max(0, in.rank() - 2), rightBatch = Math.max(0, right.rank() - 2);
            for (int i = 1; i <= Math.min(leftBatch, rightBatch); i++) {
                requireBroadcastPair(in.dimension(leftBatch - i), right.dimension(rightBatch - i));
            }
        } else if (kind == SelectKind.SELECT && in.dimension(((SelectAttrs) attrs).axis()).isDynamic()) {
            throw new IllegalArgumentException("scan cannot select an unbound batch coordinate");
        } else if (attrs instanceof CropToShapeAttrs crop) {
            for (int axis = 0; axis < in.rank(); axis++) {
                Dimension extent = in.dimension(axis), size = crop.targetShape().dimension(axis);
                Dimension prefix = crop.prefixShape().dimension(axis);
                if ((extent.isDynamic() || size.isDynamic() || prefix.isDynamic())
                        && (!extent.equals(size) || !prefix.equals(new StaticDimension(0)))) {
                    throw new IllegalArgumentException("scan crop/update must preserve the complete named coordinate");
                }
            }
        } else if (attrs instanceof SliceAttrs slice) {
            for (int axis : slice.axes()) {
                if (in.dimension(axis).isDynamic()) {
                    throw new IllegalArgumentException("scan cannot slice an unbound batch coordinate");
                }
            }
        }
    }

    /**
     * Uses Model's canonical dimension arithmetic without binding a numerical batch.
     * @param shape admitted static or single-declared-symbol shape
     * @return canonical count, preserving ordinary zero-domain semantics
     * @throws ArithmeticException if a nonzero count coefficient overflows
     */
    private static Dimension elementCountExpression(Shape shape) {
        for (Dimension dimension : shape.dimensions()) {
            if (dimension instanceof StaticDimension fixed && fixed.size() == 0) return fixed;
        }
        Dimension count = new StaticDimension(1);
        for (Dimension dimension : shape.dimensions()) count = DimensionExpressions.multiply(count, dimension);
        return count;
    }

    /**
     * Proves named-coordinate equalities/broadcasts that ordinary family guards defer.
     * Families with exact ordinary structural guards need no duplicate inference here. Numeric
     * domain and value obligations remain ordinary semantics, not a new Scan proof language.
     * @param operation known family-validated operation
     * @param inputs checked operand descriptors
     * @param outputs checked complete canonical sibling descriptors
     * @throws IllegalArgumentException if a deferred shape relation cannot be proved exactly
     * @throws ArithmeticException if a convolution channel product overflows
     */
    private static void validateDeferredFamilyShapes(Operation operation, List<TensorDescriptor> inputs,
            List<TensorDescriptor> outputs) {
        OperationKind kind = operation.kind();
        OperationAttrs attrs = operation.attrs();
        Shape in = inputs.getFirst().shape();
        if (kind instanceof ScaledDotProductAttentionKind) {
            validateAttentionShapes(inputs, outputs);
        } else if (attrs instanceof LayerNormAttrs a) {
            requireExactSuffix(in, a.normalizedShape());
        } else if (attrs instanceof AffineLayerNormAttrs a) {
            requireExactSuffix(in, a.normalizedShape());
        } else if (attrs instanceof RmsNormAttrs a) {
            requireExactSuffix(in, a.normalizedShape());
        } else if (kind instanceof BatchNormKind) {
            int axis = attrs instanceof BatchNormInferenceAttrs a ? a.channelAxis()
                    : ((BatchNormTrainingAttrs) attrs).channelAxis();
            for (int i = 1; i < inputs.size(); i++) {
                requireEqualDimension(in.dimension(axis), inputs.get(i).shape().dimension(0));
            }
        } else if (kind instanceof Conv2dKind || kind instanceof Conv3dKind) {
            Shape weight = inputs.get(1).shape();
            long groups = attrs instanceof Conv2dAttrs a ? a.groups() : ((Conv3dAttrs) attrs).groups();
            requireEqualDimension(in.dimension(1), DimensionExpressions.multiply(weight.dimension(1), groups));
            if (inputs.size() == 3) {
                requireEqualDimension(weight.dimension(0), inputs.get(2).shape().dimension(0));
            }
        } else if (kind instanceof LossKind) {
            Shape target = inputs.get(1).shape();
            if (attrs instanceof IndexCategoricalCrossEntropyWithLogitsAttrs a) {
                for (int axis = 0; axis < target.rank(); axis++) {
                    requireEqualDimension(in.dimension(axis < a.axis() ? axis : axis + 1), target.dimension(axis));
                }
            } else {
                for (int axis = 0; axis < in.rank(); axis++) {
                    requireEqualDimension(in.dimension(axis), target.dimension(axis));
                }
            }
        }
    }

    /**
     * Proves attention's three batch broadcasts, embedding/sequence equalities and mask expansion.
     * @param inputs checked query/key/value and optional mask descriptors
     * @param outputs checked canonical attention siblings
     * @throws IllegalArgumentException if a fixed extent would constrain the declared symbol
     */
    private static void validateAttentionShapes(List<TensorDescriptor> inputs, List<TensorDescriptor> outputs) {
        Shape query = inputs.get(0).shape();
        Shape key = inputs.get(1).shape();
        Shape value = inputs.get(2).shape();
        Shape out = outputs.getFirst().shape();
        requireEqualDimension(query.dimension(-1), key.dimension(-1));
        requireEqualDimension(key.dimension(-2), value.dimension(-2));
        int batchRank = out.rank() - 2;
        for (int axis = 0; axis < batchRank; axis++) {
            Dimension q = batchDimension(query, batchRank, axis);
            Dimension k = batchDimension(key, batchRank, axis);
            Dimension v = batchDimension(value, batchRank, axis);
            requireBroadcastPair(q, k);
            requireBroadcastPair(q, v);
            requireBroadcastPair(k, v);
        }
        if (inputs.size() == 4) {
            Shape mask = inputs.get(3).shape();
            int offset = out.rank() - mask.rank();
            for (int axis = 0; axis < mask.rank(); axis++) {
                int scoreAxis = axis + offset;
                Dimension score = scoreAxis < batchRank ? out.dimension(scoreAxis)
                        : scoreAxis == batchRank ? query.dimension(-2) : key.dimension(-2);
                Dimension from = mask.dimension(axis);
                if (!from.equals(score) && !from.equals(new StaticDimension(1))) {
                    throw new IllegalArgumentException("scan attention mask requires proved named-axis expansion");
                }
            }
        }
    }

    /**
     * Aligns an attention operand's leading batch dimensions, treating absent axes as singleton.
     * @param shape rank-at-least-two attention operand shape
     * @param rank result batch-prefix rank
     * @param axis valid result batch-prefix axis
     * @return exact aligned dimension, or a fixed singleton for an implicit leading axis
     */
    private static Dimension batchDimension(Shape shape, int rank, int axis) {
        int local = axis - (rank - (shape.rank() - 2));
        return local < 0 ? new StaticDimension(1) : shape.dimension(local);
    }

    /**
     * Proves an exact trailing normalization shape, not a binding-dependent equality.
     * @param input family-validated input shape
     * @param suffix normalized attribute shape whose rank does not exceed the input rank
     * @throws IllegalArgumentException if any aligned dimension differs structurally
     */
    private static void requireExactSuffix(Shape input, Shape suffix) {
        int offset = input.rank() - suffix.rank();
        for (int axis = 0; axis < suffix.rank(); axis++) {
            requireEqualDimension(input.dimension(axis + offset), suffix.dimension(axis));
        }
    }

    /**
     * Rejects an equality that would require choosing a numeric value for the named coordinate.
     * @param left expected exact static or named dimension
     * @param right actual exact static or named dimension
     * @throws IllegalArgumentException if the dimensions are not structurally equal
     */
    private static void requireEqualDimension(Dimension left, Dimension right) {
        if (!left.equals(right)) throw new IllegalArgumentException("scan requires exact named-axis equality");
    }

    /**
     * Checks a symbolic broadcast without equating B to a fixed non-singleton extent.
     * @param left first aligned dimension
     * @param right second aligned dimension
     * @throws IllegalArgumentException if equality or a fixed singleton cannot prove broadcast
     */
    private static void requireBroadcastPair(Dimension left, Dimension right) {
        if (!left.equals(right) && !left.equals(new StaticDimension(1)) && !right.equals(new StaticDimension(1))) {
            throw new IllegalArgumentException("scan requires proved named-axis broadcast dimensions");
        }
    }

    /**
     * Locates the sole named axis after complete shape admission.
     * @param shape static or admitted one-symbol shape
     * @return named axis index, or -1 for a static shape
     */
    private static int namedAxis(Shape shape) {
        for (int i = 0; i < shape.rank(); i++) if (shape.dimension(i) instanceof DynamicDimension) return i;
        return -1;
    }


    /**
     * Checks one canonical ordinary PURE occurrence without creating Tensor identities.
     *
     * <p>The existing expression owners supply all type, shape, layout and parameter checks and
     * every canonical sibling. This is not Compiler inference, Scan-coordinate proof,
     * differentiability or backend admission. Unknown/effectful kinds and fixed/general nested
     * scans are excluded. Neither the caller list nor the synchronous completion is retained.</p>
     *
     * @param operation non-null already canonical operation occurrence
     * @param inputs non-null ordered non-null operand descriptors; snapshotted
     * @param outputCount exact complete canonical sibling count, not selected-output demand
     * @return immutable complete ordered output descriptors
     * @throws NullPointerException if the operation, list or an operand is null
     * @throws IllegalArgumentException if cardinality, ordinary semantics or canonical attributes
     *     are invalid, or the kind is outside the known ordinary PURE families
     */
    public static List<TensorDescriptor> describeOrdinary(
            Operation operation, List<TensorDescriptor> inputs, int outputCount) {
        Objects.requireNonNull(operation, "operation");
        inputs = List.copyOf(Objects.requireNonNull(inputs, "inputs"));
        operation.signature().validateOccurrence(inputs.size(), outputCount);
        List<TensorDescriptor> outputs = new ArrayList<>();
        java.util.function.BiConsumer<Operation, List<TensorDescriptor>> check = (validated, descriptors) -> {
            if (!operation.equals(validated) || descriptors.size() != outputCount || !outputs.isEmpty()) {
                throw new IllegalArgumentException("invalid ordinary semantics/canonical sibling descriptors");
            }
            outputs.addAll(descriptors);
        };
        switch (operation.kind()) {
            case UnaryElementwiseKind ignored -> TensorUnaryExpressions.describeChecked(operation, inputs, outputCount, check);
            case BinaryArithmeticKind ignored -> TensorBinaryExpressions.describeChecked(operation, inputs, outputCount, check);
            case BinaryComparisonKind ignored -> TensorComparisonExpressions.describeChecked(operation, inputs, outputCount, check);
            case FloatingClassificationKind ignored -> TensorFloatingClassifications.describeChecked(operation, inputs, outputCount, check);
            case BooleanLogicalKind ignored -> TensorLogicalExpressions.describeChecked(operation, inputs, outputCount, check);
            case WhereSelectionKind ignored -> TensorWhereExpressions.describeChecked(operation, inputs, outputCount, check);
            case CastKind ignored -> TensorCastExpressions.describeChecked(operation, inputs, outputCount, check);
            case ScalarElementwiseKind ignored -> TensorScalarExpressions.describeChecked(operation, inputs, outputCount, check);
            case MatmulKind ignored -> TensorMatmulExpressions.describeChecked(operation, inputs, outputCount, check);
            case ContiguousKind ignored -> TensorContiguousExpressions.describeChecked(operation, inputs, outputCount, check);
            case ShapeTransformKind kind -> {
                if (kind == ShapeTransformKind.RESHAPE)
                    TensorReshapeExpressions.describeChecked(operation, inputs, outputCount, check);
                else TensorExpandExpressions.describeChecked(operation, inputs, outputCount, check);
            }
            case AxisTransformKind kind -> {
                if (kind == AxisTransformKind.PERMUTE)
                    TensorPermutationExpressions.describeChecked(operation, inputs, outputCount, check);
                else TensorRankEditingExpressions.describeChecked(operation, inputs, outputCount, check);
            }
            case TensorCompositionKind ignored -> TensorCompositionExpressions.describeChecked(operation, inputs, outputCount, check);
            case SliceKind kind -> {
                if (kind == SliceKind.SLICE && operation.attrs() instanceof SliceAttrs)
                    TensorSliceExpressions.describeChecked(operation, inputs, outputCount, check);
                else TensorSlicePlacementExpressions.describeChecked(operation, inputs, outputCount, check);
            }
            case SelectKind ignored -> TensorSelectExpressions.describeChecked(operation, inputs, outputCount, check);
            case PadKind ignored -> TensorPadTileExpressions.describeChecked(operation, inputs, outputCount, check);
            case TileKind ignored -> TensorPadTileExpressions.describeChecked(operation, inputs, outputCount, check);
            case AxisGatherKind ignored -> TensorAxisGatherExpressions.describeChecked(operation, inputs, outputCount, check);
            case GatherNdKind ignored -> TensorGatherNdExpressions.describeChecked(operation, inputs, outputCount, check);
            case OneHotKind ignored -> TensorOneHotExpressions.describeChecked(operation, inputs, outputCount, check);
            case AxisScatterKind ignored -> TensorAxisScatterExpressions.describeChecked(operation, inputs, outputCount, check);
            case ScatterNdKind ignored -> TensorScatterNdExpressions.describeChecked(operation, inputs, outputCount, check);
            case CumulativeScanKind ignored -> TensorCumulativeScanExpressions.describeChecked(operation, inputs, outputCount, check);
            case AggregateReductionKind ignored -> describeReduction(operation, inputs, outputCount, check);
            case SoftmaxKind ignored -> TensorSoftmaxExpressions.describeChecked(operation, inputs, outputCount, check);
            case LayerNormKind ignored -> TensorLayerNormExpressions.describeChecked(operation, inputs, outputCount, check);
            case RmsNormKind ignored -> TensorRmsNormExpressions.describeChecked(operation, inputs, outputCount, check);
            case Conv2dKind ignored -> TensorConv2dExpressions.describeChecked(operation, inputs, outputCount, check);
            case Conv3dKind ignored -> TensorConv3dExpressions.describeChecked(operation, inputs, outputCount, check);
            case WindowTransformKind ignored -> TensorWindowExpressions.describeChecked(operation, inputs, outputCount, check);
            case LossKind ignored -> TensorLossExpressions.describeChecked(operation, inputs, outputCount, check);
            case TopKKind ignored -> TensorTopKExpressions.describeChecked(operation, inputs, outputCount, check);
            case OrderingKind ignored -> TensorSortExpressions.describeChecked(operation, inputs, outputCount, check);
            case BatchNormKind kind -> {
                if (kind == BatchNormKind.BATCH_NORM_INFERENCE)
                    TensorBatchNormInferenceExpressions.describeChecked(operation, inputs, outputCount, check);
                else TensorBatchNormTrainingExpressions.describeChecked(operation, inputs, outputCount, check);
            }
            case Pool2dKind kind -> {
                if (kind == Pool2dKind.MAX_POOL2D)
                    TensorMaxPool2dExpressions.describeChecked(operation, inputs, outputCount, check);
                else TensorAveragePool2dExpressions.describeChecked(operation, inputs, outputCount, check);
            }
            case Pool3dKind kind -> {
                if (kind == Pool3dKind.MAX_POOL3D)
                    TensorMaxPool3dExpressions.describeChecked(operation, inputs, outputCount, check);
                else TensorAveragePool3dExpressions.describeChecked(operation, inputs, outputCount, check);
            }
            case ScaledDotProductAttentionKind ignored ->
                    TensorScaledDotProductAttentionExpressions.describeChecked(operation, inputs, outputCount, check);
            default -> throw new IllegalArgumentException(
                    "unknown, effectful or nested scan body operation: " + operation.kind());
        }
        if (outputs.size() != outputCount)
            throw new IllegalArgumentException("ordinary semantics did not complete all canonical siblings");
        return List.copyOf(outputs);
    }

    /**
     * Checks one ordinary Scan declaration in its exact symbolic source scope without Tensor IDs.
     *
     * <p>The three-argument ordinary inference retains its deferred constraints. This overload
     * additionally reuses the source guard for declared symbols and deferred family relations.
     * It proves whole-Tensor reshape count, not length-mode ROW contributor mapping.</p>
     *
     * @param operation non-null canonical ordinary operation
     * @param inputs non-null ordered operand descriptors; snapshotted
     * @param outputCount complete canonical sibling count
     * @param batch exact sole declared symbol, or empty for a static template; non-null
     * @return immutable checked complete canonical output descriptors
     * @throws NullPointerException if an argument or operand is null
     * @throws IllegalArgumentException if ordinary semantics or the scoped symbolic proof fails
     * @throws ArithmeticException if checked dimension arithmetic overflows
     */
    public static List<TensorDescriptor> describeOrdinary(Operation operation,
            List<TensorDescriptor> inputs, int outputCount, Optional<DynamicDimension> batch) {
        inputs = List.copyOf(Objects.requireNonNull(inputs, "inputs"));
        Objects.requireNonNull(batch, "batch");
        List<TensorDescriptor> outputs = describeOrdinary(operation, inputs, outputCount);
        validateSymbolicTransforms(operation, inputs, outputs, batch.orElse(null));
        return outputs;
    }

    /**
     * Completes eligible computed logical layouts after ordinary semantics have been independently
     * validated. This cold metadata operation neither changes ordinary Tensor construction nor
     * performs source admission, independent Shape/type inference, ROW proof or backend selection.
     *
     * <p>Only unresolved fully static results can change. Generic materialized families and the
     * separately qualified static Conv2d/Conv3d results receive canonical non-view geometry.
     * Existing family helpers provide exact affine views; the existing noncontiguous-static
     * RESHAPE materialization decision applies only when no such view is proved. Positive static
     * crop uses its family-owned layout recipe. Validated fully static signed-step and empty
     * slices receive canonical non-view geometry instead of negative strides or an empty view.
     * Named and otherwise unproved extraction layouts remain unresolved. Every resolved layout
     * is preserved.</p>
     *
     * <p>Inputs are the current topologically resolved operands, not replacements for caller or
     * formal descriptors. Types, Shapes, gradient flags and canonical result ordering are
     * unchanged. No Tensor, producer, storage, identifier, graph node or runtime state is created.</p>
     *
     * @param operation non-null already validated ordinary operation and attributes
     * @param inputs non-null ordered current operand descriptors, never mutated or retained
     * @param outputs non-null complete independently inferred canonical result descriptors
     * @return the original output list if unchanged, otherwise an immutable list retaining every
     *     unchanged descriptor reference and replacing only proved logical layouts
     * @throws NullPointerException if an argument or descriptor is null
     * @throws IllegalArgumentException if an exact family view cannot satisfy its ordinary guards
     * @throws ArithmeticException if checked canonical or exact view geometry overflows
     */
    public static List<TensorDescriptor> closeOrdinaryLayouts(Operation operation,
            List<TensorDescriptor> inputs, List<TensorDescriptor> outputs) {
        Objects.requireNonNull(operation, "operation");
        Objects.requireNonNull(inputs, "inputs");
        Objects.requireNonNull(outputs, "outputs");
        for (TensorDescriptor input : inputs) Objects.requireNonNull(input, "input descriptor");
        for (TensorDescriptor output : outputs) Objects.requireNonNull(output, "output descriptor");
        OperationKind kind = operation.kind();
        if (hasCanonicalContiguousResults(operation)
                || kind == Conv2dKind.CONV2D || kind == Conv3dKind.CONV3D) {
            ArrayList<TensorDescriptor> closed = null;
            for (int index = 0; index < outputs.size(); index++) {
                TensorDescriptor output = outputs.get(index);
                if (!output.shape().isFullyStatic() || output.layout().isPresent()) continue;
                if (closed == null) closed = new ArrayList<>(outputs);
                closed.set(index, withClosedLayout(output, LayoutDescriptor.contiguous(output.shape())));
            }
            return closed == null ? outputs : Collections.unmodifiableList(closed);
        }
        boolean affine = kind instanceof ShapeTransformKind || kind instanceof AxisTransformKind;
        boolean extraction = kind == SliceKind.SLICE || kind == SelectKind.SELECT;
        if ((!affine && !extraction) || inputs.size() != 1 || outputs.size() != 1) return outputs;
        TensorDescriptor input = inputs.getFirst(), output = outputs.getFirst();
        if (!output.shape().isFullyStatic() || output.layout().isPresent() || input.layout().isEmpty()) {
            return outputs;
        }
        Optional<LayoutDescriptor> layout;
        if (kind == SliceKind.SLICE && operation.attrs() instanceof CropToShapeAttrs crop) {
            // This is layout completion only. Independent Compiler crop constraints stay with
            // Compiler; ordinary crop construction deliberately keeps its unresolved result.
            layout = TensorSlicePlacementExpressions.cropLayout(input, crop, output);
        } else {
            TensorDescriptor inferred = describeOrdinary(operation, inputs, outputs.size()).getFirst();
            if (inferred.dataType() != output.dataType() || !inferred.shape().equals(output.shape())) return outputs;
            layout = inferred.layout();
            if (layout.isEmpty() && kind == ShapeTransformKind.RESHAPE
                    && input.shape().isFullyStatic() && !input.layout().orElseThrow().isContiguous()) {
                layout = Optional.of(LayoutDescriptor.contiguous(output.shape()));
            }
            if (layout.isEmpty() && kind == SliceKind.SLICE
                    && operation.attrs() instanceof SliceAttrs slice
                    && input.shape().isFullyStatic()
                    && (slice.steps().stream().anyMatch(step -> step < 0)
                        || output.shape().knownElementCount().orElseThrow() == 0)) {
                layout = Optional.of(LayoutDescriptor.contiguous(output.shape()));
            }
        }
        return layout.isEmpty() ? outputs : List.of(withClosedLayout(output, layout.orElseThrow()));
    }

    /**
     * Classifies the exact existing generic materialized-result families independently of
     * geometry. A recognized named result may remain unresolved; unknown families are not
     * qualified. Specialized Conv2d/Conv3d completion deliberately does not widen this fact.
     *
     * <p>This fact presupposes validated operation semantics. Flat ordinary effect/RNG families
     * retain their existing metadata eligibility, but this method grants no PURE/Scan membership,
     * symbolic binding, Shape/type proof, backend support or execution capability.</p>
     *
     * @param operation non-null independently validated ordinary operation
     * @return whether all canonical results belong to the existing generic materialized set
     * @throws NullPointerException if the operation is null
     */
    public static boolean hasCanonicalContiguousResults(Operation operation) {
        OperationKind kind = Objects.requireNonNull(operation, "operation").kind();
        return kind instanceof ScaledDotProductAttentionKind
                || kind instanceof BinaryArithmeticKind
                || kind instanceof CastKind
                || kind instanceof FloatingClassificationKind
                || kind instanceof BinaryComparisonKind
                || kind instanceof BooleanLogicalKind
                || kind instanceof ScalarElementwiseKind
                || kind instanceof WhereSelectionKind
                || kind instanceof UnaryElementwiseKind
                || kind instanceof AxisGatherKind
                || kind instanceof AxisScatterKind
                || kind instanceof GatherNdKind
                || kind instanceof OneHotKind
                || kind instanceof ScatterNdKind
                || kind == ContiguousKind.CONTIGUOUS
                || kind instanceof PadKind
                || kind == SliceKind.SLICE_UPDATE
                || kind instanceof TensorCompositionKind
                || kind instanceof TileKind
                || kind instanceof WindowTransformKind
                || kind instanceof MatmulKind
                || kind instanceof LossKind
                || kind instanceof BatchNormKind
                || kind instanceof LayerNormKind
                || kind instanceof RmsNormKind
                || kind instanceof SoftmaxKind
                || kind instanceof OrderingKind
                || kind instanceof TopKKind
                || kind instanceof Pool2dKind
                || kind instanceof Pool3dKind
                || kind instanceof DropoutKind
                || kind instanceof GraphRngKind
                || kind instanceof RecurrentScanKind
                || kind instanceof AggregateReductionKind
                || kind instanceof CumulativeScanKind;
    }

    /** Replaces only geometry, preserving the independently inferred result's other metadata. */
    private static TensorDescriptor withClosedLayout(TensorDescriptor output, LayoutDescriptor layout) {
        return new TensorDescriptor(output.dataType(), output.shape(), Optional.of(layout), output.requiresGrad());
    }

    /** Validates actual siblings through the same ordinary and context-qualified source guards. */
    private static void validateOrdinaryOutputs(TensorProducer producer, DynamicDimension batch) {
        List<TensorDescriptor> inputs = producer.inputs().stream().map(Tensor::descriptor).toList();
        if (!producer.outputDescriptors().equals(
                describeOrdinary(producer.operation(), inputs, producer.outputCount(), Optional.ofNullable(batch)))) {
            throw new IllegalArgumentException("invalid body semantics/canonical sibling descriptors");
        }
    }

    /** Selects an existing reduction owner; descriptor algorithms remain in that owner. */
    private static void describeReduction(Operation operation, List<TensorDescriptor> inputs,
            int outputCount, java.util.function.BiConsumer<Operation, List<TensorDescriptor>> check) {
        switch (operation.attrs()) {
            case NoOperationAttrs ignored -> TensorReductionExpressions.describeChecked(operation, inputs, outputCount, check);
            case AxisReductionAttrs ignored -> TensorReductionExpressions.describeChecked(operation, inputs, outputCount, check);
            case MultiAxisReductionAttrs ignored -> TensorMultiAxisReductionExpressions.describeChecked(operation, inputs, outputCount, check);
            case StatisticalReductionAttrs ignored -> TensorMultiAxisReductionExpressions.describeChecked(operation, inputs, outputCount, check);
            case ArgExtremaAttrs ignored -> TensorArgExtremaExpressions.describeChecked(operation, inputs, outputCount, check);
            case MaskedReductionAttrs ignored -> TensorMaskedReductionExpressions.describeChecked(operation, inputs, outputCount, check);
            case SumToShapeAttrs ignored -> TensorSumToShapeExpressions.describeChecked(operation, inputs, outputCount, check);
            default -> throw new IllegalArgumentException("unknown reduction attributes");
        }
    }
}
