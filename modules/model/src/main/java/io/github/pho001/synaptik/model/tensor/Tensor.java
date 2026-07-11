package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind;
import io.github.pho001.synaptik.model.operation.elementwise.classification.FloatingClassificationKind;
import io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind;
import io.github.pho001.synaptik.model.operation.elementwise.logical.BooleanLogicalKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.index.AxisGatherKind;
import io.github.pho001.synaptik.model.operation.index.AxisScatterKind;
import io.github.pho001.synaptik.model.operation.index.GatherNdAttrs;
import io.github.pho001.synaptik.model.operation.index.GatherNdKind;
import io.github.pho001.synaptik.model.operation.index.ScatterNdAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterNdKind;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
import io.github.pho001.synaptik.model.operation.index.SelectKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.ContiguousKind;
import io.github.pho001.synaptik.model.operation.layout.PadKind;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.layout.SliceKind;
import io.github.pho001.synaptik.model.operation.layout.TensorCompositionKind;
import io.github.pho001.synaptik.model.operation.layout.TileKind;
import io.github.pho001.synaptik.model.operation.layout.Window2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.WindowTransformKind;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.ArgExtremaTiePolicy;
import io.github.pho001.synaptik.model.operation.scan.CumulativeSumKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Public mutable state for one tensor, identified independently of graph-local values and nodes.
 *
 * <p>The tensor retains one immutable {@link TensorId}, one immutable {@link TensorDescriptor},
 * one normalized optional diagnostic label, and immutable optional {@link TensorProvenance}.
 * Its sole mutable state is an optional borrowed {@link HostTensorStorage} association. The
 * synchronized storage methods make reference replacement and clearing atomic and visible with
 * respect to one another; they do not
 * synchronize access to the underlying memory or prevent its caller-owned scope from closing.</p>
 *
 * <p>Construction remains package-private, and {@link TensorFactory} is the supported public
 * creation surface. The factory assigns identifiers unique among its allocations in the current
 * Java virtual machine, while this class still accepts any validated identifier through its
 * internal construction path and does not independently enforce uniqueness. It uses ordinary
 * object identity for inherited equality and hashing, so equal identifier values do not make two
 * tensor objects equal.</p>
 *
 * <p>Provenance is stable expression-origin metadata, independent of storage replacement,
 * clearing, or later storage death. A present provenance identifies one zero-based output of one
 * exact {@link TensorProducer}; this tensor retains the exact descriptor reference from that
 * producer output position. The producer is pre-capture occurrence identity, not graph-local node
 * or value identity, and does not make a tensor an intermediate-representation node. Binary
 * arithmetic, binary comparison,
 * boolean logical, conditional selection, explicit cast, numeric and boolean aggregate reduction,
 * parameterized scalar, and unary
 * elementwise and matrix-multiplication expression methods create fresh storage-free tensors
 * whose immutable provenance records the requested semantics and exact inputs; they do not execute mathematics, validate
 * numerical domains, create gradient rules, or capture a graph.
 * Binary ADD, SUB, MUL, MIN, and MAX promote same-category floating or signed-integral operands;
 * integral ADD, SUB, and MUL have fixed-width two's-complement modular meaning, while integral
 * MIN and MAX use signed order. DIV and POW remain floating-only. All binary arithmetic methods
 * broadcast shapes and combine gradient eligibility by logical OR; integral descriptors
 * necessarily remain non-differentiable. Binary comparison methods accept the same-category
 * floating or integral domains and broadcasting contract but produce non-differentiable
 * {@code BOOL} descriptors; integral relations use signed order. Boolean
 * logical methods accept only {@code BOOL}: conjunction and disjunction broadcast ordered inputs,
 * while negation retains the exact input shape. Their results are also non-differentiable
 * {@code BOOL} descriptors. Conditional selection accepts one {@code BOOL} condition and two
 * floating branches, promotes the branch type, composes two pairwise broadcasts, and propagates
 * gradient eligibility from the branches only. Cast accepts every current source and target data
 * type, retains the exact input shape, and preserves a true gradient request only across a
 * floating-to-floating conversion. Ordinary SUM, PROD, MIN, and MAX aggregates accept floating or
 * signed-integral input; MEAN remains floating-only. Every result preserves the exact input type
 * and gradient eligibility and reduces either every axis to a scalar or one normalized axis,
 * optionally retaining it with extent one. Integral SUM and PROD mean fixed-width modular
 * arithmetic with reassociation permitted, while integral MIN and MAX use signed order. Their
 * empty-domain identities are zero, one, the input type's maximum, and the input type's minimum,
 * respectively. Full, single-axis, and ordered distinct multi-axis forms share these policies;
 * an empty multi-axis list selects a point domain rather than every axis. Construction records
 * those meanings without aggregating values. Boolean aggregate methods require exact BOOL input
 * and construct non-differentiable BOOL results with full, single-axis, or multi-axis Shape rules;
 * empty ALL is true and empty ANY is false. Masked sum and mean require a BOOL mask whose ordinary right-aligned
 * broadcast is exactly the input Shape, remove one axis, and record exact {@code [input, mask]}
 * provenance without inspecting values. False positions exclude even NaN and infinity before
 * aggregation; an empty selected set means zero for sum and NaN for mean. Floating-only
 * log-sum-exp, corrected variance/standard deviation, and L1/L2 norm methods use ordered distinct
 * axes, preserve exact input metadata, and record their first-class numerical targets without
 * decomposition or evaluation. Statistical construction rejects a statically known domain count
 * at most correction and defers dynamic proof. Arg-min and arg-max
 * accept one non-empty selected axis of a floating or integral input, use an explicit first- or
 * last-logical-index tie policy, and produce a non-differentiable {@code INT64} result. Integral
 * candidates use signed order. Floating candidates prefer NaN, order negative zero below positive
 * zero, and order infinities normally. A statically empty selected axis is rejected, while an
 * unselected zero axis or an unbound selected extent is accepted structurally. Construction does
 * not compare values or select an index. Axis gather methods consume ordered
 * {@code [data, indices]}
 * inputs, require exact INT32 or INT64 indices, and apply canonical axis-replacement or same-rank
 * aligned Shape rules without reading index values or checking their bounds. Gather-ND methods consume
 * the same ordered inputs as coordinate tuples, validate a structurally equal shared batch
 * prefix and statically known positive tuple depth, and derive an indices-prefix-plus-data-suffix
 * Shape without reading index values. Scatter-ND methods consume ordered
 * {@code [data, indices, updates]} inputs as coordinate tuples, require the updates Shape to equal
 * the corresponding Gather-ND result Shape, and retain exact data Shape/type with unresolved
 * layout. Axis scatter methods consume ordered
 * {@code [data, indices, updates]} inputs, require exact INT32 or INT64 indices and matching
 * data/update types, and retain exact data Shape/type with unresolved layout. Scatter-elements
 * permits replacement for every current type and arithmetic reduction for numeric types.
 * Construction reads no index or update values,
 * performs no writes or reductions, and never mutates data. Cumulative sum accepts one axis of a
 * floating or integral input,
 * preserves its shape and type, and records whether the scan is exclusive and/or reverse without
 * reading or accumulating values. Softmax and log-softmax accept one axis of a floating input,
 * preserve its shape, type, and gradient eligibility, and record probability or log-probability
 * normalization semantics without calculating values or selecting a numerical algorithm.
 * Scalar ADD, SUB, MUL, MIN, and MAX plus the one-bound clamp conveniences accept one floating or
 * signed-integral input; scalar DIV, POW, and first-class range CLAMP remain floating-only.
 * Scalar methods retain exact matching typed values in attributes, so they do not promote, and
 * their {@code double} conveniences mean exact FLOAT64. The sixteen parameterless unary methods
 * remain floating-only. The three floating-classification methods also accept one floating input,
 * but produce
 * fixed non-differentiable {@code BOOL} descriptors with the exact input shape and unresolved
 * layout. They record classification semantics without reading or classifying values.
 * Every expression result has a fresh factory identity and no label or storage. Most expression
 * results leave layout unresolved; a contiguous request instead publishes newly resolved
 * canonical dense row-major geometry for a fully static Shape and remains unresolved for a
 * dynamic Shape. Reshape preserves the ordered logical element sequence under a normalized target
 * Shape. It publishes same-offset canonical alias-view geometry only when the input layout is
 * resolved and contiguous and the target is fully static; other reshape geometry remains
 * unresolved without inserting materialization. Expand right-aligns the input with an exact
 * target Shape and permits only equal aligned dimensions, statically known input singletons, and
 * new leading axes. A static target and resolved input layout produce a new same-offset view
 * layout with preserved aligned strides and zero strides for new leading or expanded singleton
 * axes; other expand geometry remains unresolved. Neither transformation attaches storage or
 * chooses materialization. Permute reorders exact Dimension references and, when input layout is
 * resolved, exact strides while preserving the input element offset in new logical view metadata.
 * Rank-two transpose is the same PERMUTE operation with axes {@code [1, 0]}. Neither expression
 * attaches storage or guarantees an executable alias. Expand-dimensions inserts one static
 * singleton at a normalized result position; squeeze removes one selected dimension only when
 * its singleton extent is statically proven. Resolved rank edits insert or remove one stride in
 * new same-offset view metadata, while unresolved geometry stays unresolved. Slice accepts
 * parallel half-open bounds, distinct axes, and positive steps. It preserves rank, normalizes and
 * clamps against selected static dimensions, and derives checked resolved view geometry only for
 * non-empty results with resolved input layout. Empty and unresolved results remain unresolved.
 * Constant padding and per-axis tiling require complete rank-aligned long arrays and derive
 * canonical checked static or symbolic extents; neutral transformations preserve exact Dimension
 * references. Both accept every current data type, leave layout unresolved, and retain the raw
 * padding constant or complete-pattern repeat counts without inspecting or materializing values.
 * Concat joins a non-empty ordered sequence along an existing axis after exact data-type, rank,
 * and non-axis Dimension validation, folding its selected extents through canonical checked
 * symbolic addition. Stack joins exactly same-shaped inputs along one inserted axis. Unstack
 * removes one statically sized axis and returns an immutable ordered list of independently indexed
 * result tensors. Concat and stack results have unresolved layout. Unstack results use scalar
 * select's conditional logical-view layout and deliberately carry no shared producer identity.
 * General-axis unfold replaces one static extent with checked window positions and appends the
 * window size, while fold-axis removes the final window dimension and restores one explicit
 * target extent. Two-dimensional unfold and fold use checked static NCHW window geometry and
 * canonical column Shapes. All four window-transform results have unresolved layout and describe
 * materialization or scatter-add semantics without reading, allocating, or accumulating values.
 * Every result records
 * an exact matching
 * {@link BinaryArithmeticKind}, {@link BinaryComparisonKind}, {@link BooleanLogicalKind},
 * {@link WhereSelectionKind}, {@link CastKind}, {@link AggregateReductionKind},
 * {@link CumulativeSumKind}, {@link SoftmaxKind}, {@link AxisGatherKind}, {@link GatherNdKind},
 * {@link AxisScatterKind}, {@link ScatterNdKind},
 * {@link ContiguousKind}, {@link ShapeTransformKind}, {@link AxisTransformKind},
 * {@link SliceKind}, {@link PadKind},
 * {@link TileKind}, {@link TensorCompositionKind}, {@link WindowTransformKind},
 * {@link ScalarElementwiseKind}, {@link UnaryElementwiseKind}, or
 * {@link FloatingClassificationKind}, or {@link MatmulKind}.
 * Gradient eligibility does not promise that a gradient rule exists.
 * The tensor owns no publication, device, runtime-residency, or prepared-execution state and
 * neither allocates nor closes storage.</p>
 */
public final class Tensor {
    private final TensorId id;
    private final TensorDescriptor descriptor;
    private final Optional<String> label;
    private final Optional<TensorProvenance> provenance;
    private HostTensorStorage hostStorage;

    /**
     * Creates tensor state from stable metadata and an optional borrowed host-storage association.
     *
     * <p>Validation proceeds in parameter order: {@code id}, {@code descriptor}, {@code label},
     * {@code provenance}, and {@code hostStorage} optionals must be non-null; a present label is
     * stripped and must remain non-blank; a present provenance must select this exact descriptor
     * reference; then present storage is checked for matching data type, sufficient capacity when
     * layout geometry is resolved, and point-in-time liveness. A static or dynamic unresolved
     * layout performs no capacity check because this class does not invent row-major geometry.
     * Resolved capacity uses the complete referenced element span, including offset and striding;
     * scalar span is one and zero-sized span is zero.</p>
     *
     * <p>The exact immutable identifier and descriptor references are retained. Label uses
     * optional value semantics and is stored normalized. Provenance also uses optional value
     * semantics, and a present value retains the exact immutable provenance reference for this
     * tensor's lifetime. A present storage reference is borrowed and retained exactly, whether
     * writable or read-only. The caller owns its lifetime and may close its scope immediately
     * after construction; synchronization does not make raw memory access thread-safe or extend
     * JDK scope accessibility.</p>
     *
     * @param id non-null immutable tensor identity reference to retain exactly
     * @param descriptor non-null immutable logical descriptor reference to retain exactly
     * @param label non-null optional diagnostic label; present text is stripped and must contain a
     *     non-whitespace character, while empty represents absence
     * @param provenance non-null value-based optional expression origin to retain; a present
     *     result contains the exact immutable provenance reference, must select the exact supplied
     *     descriptor reference, and remains independent of storage mutation
     * @param hostStorage non-null optional borrowed host storage to retain exactly when present;
     *     read-only storage is accepted
     * @throws NullPointerException if {@code id}, {@code descriptor}, {@code label},
     *     {@code provenance}, or {@code hostStorage} is {@code null}, with the corresponding
     *     parameter name as the message
     * @throws IllegalArgumentException if a present label is blank, with message
     *     {@code label must not be blank}; if a present provenance selects a descriptor reference
     *     other than {@code descriptor}, with message
     *     {@code descriptor must be the exact provenance output descriptor reference}; if storage
     *     data type differs from the descriptor, with message
     *     {@code hostStorage data type must match descriptor data type: expected=<expected>,
     *     actual=<actual>}; or if resolved layout span exceeds storage
     *     capacity, with message {@code hostStorage element capacity is smaller than resolved
     *     layout span: required=<required>, actual=<actual>}
     * @throws IllegalStateException if present storage is not alive at the attachment check, with
     *     message {@code hostStorage must be alive when attached}
     */
    Tensor(
            TensorId id,
            TensorDescriptor descriptor,
            Optional<String> label,
            Optional<TensorProvenance> provenance,
            Optional<HostTensorStorage> hostStorage) {
        this.id = Objects.requireNonNull(id, "id");
        this.descriptor = Objects.requireNonNull(descriptor, "descriptor");
        Objects.requireNonNull(label, "label");
        this.provenance = Objects.requireNonNull(provenance, "provenance");
        Objects.requireNonNull(hostStorage, "hostStorage");
        this.label = normalizeLabel(label);

        if (provenance.isPresent()
                && provenance.orElseThrow().outputDescriptor() != descriptor) {
            throw new IllegalArgumentException(
                    "descriptor must be the exact provenance output descriptor reference");
        }

        if (hostStorage.isPresent()) {
            HostTensorStorage suppliedStorage = hostStorage.orElseThrow();
            validateHostStorage(suppliedStorage);
            this.hostStorage = suppliedStorage;
        }
    }

    /**
     * Returns this tensor's stable identity metadata.
     *
     * @return the exact non-null immutable identifier reference supplied at construction
     */
    public TensorId id() {
        return id;
    }

    /**
     * Returns this tensor's stable logical descriptor.
     *
     * @return the exact non-null immutable descriptor reference supplied at construction
     */
    public TensorDescriptor descriptor() {
        return descriptor;
    }

    /**
     * Returns the normalized immutable diagnostic label value.
     *
     * @return a non-null value-based optional containing stripped non-blank text, or empty when no
     *     label was supplied; optional-container identity is not part of the contract
     */
    public Optional<String> label() {
        return label;
    }

    /**
     * Returns this tensor's immutable optional expression-origin metadata.
     *
     * <p>The optional uses value semantics, so callers must not rely on container identity. A
     * present result contains the exact {@link TensorProvenance} reference supplied at
     * construction. Provenance is final and therefore needs no synchronization; replacing,
     * clearing, mutating, or invalidating host storage cannot change it. The value is origin
     * metadata only, not graph-local identity, graph membership, or an executable node.</p>
     *
     * @return a non-null value-based optional containing the exact immutable provenance reference,
     *     or empty for a leaf tensor
     */
    public Optional<TensorProvenance> provenance() {
        return provenance;
    }

    /**
     * Builds an elementwise expression that adds this left operand to {@code right}.
     *
     * <p>Both operands must belong to the same floating or signed-integral category. Floating
     * promotion is unchanged; integral promotion uses {@code INT32 < INT64} with signed extension
     * into an {@code INT64} operation domain. Their shapes must support locally provable
     * right-aligned broadcasting. The fresh result has unresolved layout, gradient eligibility
     * equal to the
     * logical OR of the operand requests, no label or host storage, and provenance containing
     * {@link BinaryArithmeticKind#ADD}, {@code NoOperationAttrs.INSTANCE}, and ordered inputs
     * {@code [this, right]}. Integral addition is fixed-width two's-complement modular arithmetic
     * in the promoted type. This method constructs semantics only; it does not evaluate values,
     * install a gradient rule, capture a graph, or execute.</p>
     *
     * @param right non-null ordered right addend; it is retained by exact reference in result
     *     provenance and is not mutated
     * @return a non-null fresh derived tensor with promoted type, broadcast shape, unresolved
     *     layout, propagated gradient eligibility, and no storage
     * @throws NullPointerException if {@code right} is null, with message {@code right}
     * @throws IllegalArgumentException if an operand is boolean, the operands cross numeric
     *     categories, or their shapes cannot be broadcast under the local shape contract
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor add(Tensor right) {
        return TensorBinaryExpressions.apply(this, right, BinaryArithmeticKind.ADD);
    }

    /**
     * Builds an elementwise expression that subtracts {@code right} from this left operand.
     *
     * <p>Both operands must belong to the same floating or signed-integral category. Floating
     * promotion is unchanged; integral promotion uses {@code INT32 < INT64} with signed extension
     * into an {@code INT64} operation domain. Their shapes must support locally provable
     * right-aligned broadcasting. The fresh result has unresolved layout, gradient eligibility
     * equal to the
     * logical OR of the operand requests, no label or host storage, and provenance containing
     * {@link BinaryArithmeticKind#SUB}, {@code NoOperationAttrs.INSTANCE}, and ordered inputs
     * {@code [this, right]}. Integral subtraction is fixed-width two's-complement modular
     * arithmetic in the promoted type. This method constructs semantics only; it does not
     * evaluate values, install a gradient rule, capture a graph, or execute.</p>
     *
     * @param right non-null ordered subtrahend; it is retained by exact reference in result
     *     provenance and is not mutated
     * @return a non-null fresh derived tensor with promoted type, broadcast shape, unresolved
     *     layout, propagated gradient eligibility, and no storage
     * @throws NullPointerException if {@code right} is null, with message {@code right}
     * @throws IllegalArgumentException if an operand is boolean, the operands cross numeric
     *     categories, or their shapes cannot be broadcast under the local shape contract
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor sub(Tensor right) {
        return TensorBinaryExpressions.apply(this, right, BinaryArithmeticKind.SUB);
    }

    /**
     * Builds an elementwise expression that multiplies this left operand by {@code right}.
     *
     * <p>Both operands must belong to the same floating or signed-integral category. Floating
     * promotion is unchanged; integral promotion uses {@code INT32 < INT64} with signed extension
     * into an {@code INT64} operation domain. Their shapes must support locally provable
     * right-aligned broadcasting. The fresh result has unresolved layout, gradient eligibility
     * equal to the
     * logical OR of the operand requests, no label or host storage, and provenance containing
     * {@link BinaryArithmeticKind#MUL}, {@code NoOperationAttrs.INSTANCE}, and ordered inputs
     * {@code [this, right]}. Integral multiplication is fixed-width two's-complement modular
     * arithmetic in the promoted type. This method constructs semantics only; it does not
     * evaluate values, install a gradient rule, capture a graph, or execute.</p>
     *
     * @param right non-null ordered right factor; it is retained by exact reference in result
     *     provenance and is not mutated
     * @return a non-null fresh derived tensor with promoted type, broadcast shape, unresolved
     *     layout, propagated gradient eligibility, and no storage
     * @throws NullPointerException if {@code right} is null, with message {@code right}
     * @throws IllegalArgumentException if an operand is boolean, the operands cross numeric
     *     categories, or their shapes cannot be broadcast under the local shape contract
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor mul(Tensor right) {
        return TensorBinaryExpressions.apply(this, right, BinaryArithmeticKind.MUL);
    }

    /**
     * Builds an elementwise expression that divides this left operand by {@code right}.
     *
     * <p>Both operands must have floating data types, which are promoted through the unchanged
     * floating hierarchy, and their shapes must support locally provable right-aligned
     * broadcasting. The fresh result has unresolved layout, gradient eligibility equal to the
     * logical OR of the operand requests, no label or host storage, and provenance containing
     * {@link BinaryArithmeticKind#DIV}, {@code NoOperationAttrs.INSTANCE}, and ordered inputs
     * {@code [this, right]}. This method constructs semantics only; numerical execution and
     * gradient rules are deferred.</p>
     *
     * @param right non-null ordered denominator; it is retained by exact reference in result
     *     provenance and is not mutated
     * @return a non-null fresh derived tensor with promoted type, broadcast shape, unresolved
     *     layout, propagated gradient eligibility, and no storage
     * @throws NullPointerException if {@code right} is null, with message {@code right}
     * @throws IllegalArgumentException if an operand is boolean or integral, the operands cross
     *     numeric categories, or their shapes cannot be broadcast under the local shape contract
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor div(Tensor right) {
        return TensorBinaryExpressions.apply(this, right, BinaryArithmeticKind.DIV);
    }

    /**
     * Builds an elementwise expression that selects the minimum of this left operand and
     * {@code right}.
     *
     * <p>Both operands must belong to the same floating or signed-integral category. Floating
     * promotion is unchanged; integral promotion uses {@code INT32 < INT64} with signed extension
     * into an {@code INT64} operation domain. Their shapes must support locally provable
     * right-aligned broadcasting. The fresh result has unresolved layout, gradient eligibility
     * equal to the
     * logical OR of the operand requests, no label or host storage, and provenance containing
     * {@link BinaryArithmeticKind#MIN}, {@code NoOperationAttrs.INSTANCE}, and ordered inputs
     * {@code [this, right]}. Floating extrema retain their NaN, infinity, and signed-zero policy;
     * integral minimum uses ordinary signed order. This method constructs metadata only and does
     * not compare values, define gradients, capture a graph, or execute.</p>
     *
     * @param right non-null ordered right minimum operand; it is retained by exact reference in
     *     result provenance and is not mutated
     * @return a non-null fresh derived tensor with promoted type, broadcast shape, unresolved
     *     layout, propagated gradient eligibility, and no storage
     * @throws NullPointerException if {@code right} is null, with message {@code right}
     * @throws IllegalArgumentException if an operand is boolean, the operands cross numeric
     *     categories, or their shapes cannot be broadcast under the local shape contract
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor minimum(Tensor right) {
        return TensorBinaryExpressions.apply(this, right, BinaryArithmeticKind.MIN);
    }

    /**
     * Builds an elementwise expression that selects the maximum of this left operand and
     * {@code right}.
     *
     * <p>Both operands must belong to the same floating or signed-integral category. Floating
     * promotion is unchanged; integral promotion uses {@code INT32 < INT64} with signed extension
     * into an {@code INT64} operation domain. Their shapes must support locally provable
     * right-aligned broadcasting. The fresh result has unresolved layout, gradient eligibility
     * equal to the
     * logical OR of the operand requests, no label or host storage, and provenance containing
     * {@link BinaryArithmeticKind#MAX}, {@code NoOperationAttrs.INSTANCE}, and ordered inputs
     * {@code [this, right]}. Floating extrema retain their NaN, infinity, and signed-zero policy;
     * integral maximum uses ordinary signed order. This method constructs metadata only and does
     * not compare values, define gradients, capture a graph, or execute.</p>
     *
     * @param right non-null ordered right maximum operand; it is retained by exact reference in
     *     result provenance and is not mutated
     * @return a non-null fresh derived tensor with promoted type, broadcast shape, unresolved
     *     layout, propagated gradient eligibility, and no storage
     * @throws NullPointerException if {@code right} is null, with message {@code right}
     * @throws IllegalArgumentException if an operand is boolean, the operands cross numeric
     *     categories, or their shapes cannot be broadcast under the local shape contract
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor maximum(Tensor right) {
        return TensorBinaryExpressions.apply(this, right, BinaryArithmeticKind.MAX);
    }

    /**
     * Builds an elementwise expression that raises this left base to the {@code right} exponent.
     *
     * <p>Both operands must have floating data types, which are promoted through the unchanged
     * floating hierarchy, and their shapes must support locally provable right-aligned
     * broadcasting. The fresh result has unresolved layout, gradient eligibility equal to the
     * logical OR of the operand requests, no label or host storage, and provenance containing
     * {@link BinaryArithmeticKind#POW}, {@code NoOperationAttrs.INSTANCE}, and ordered inputs
     * {@code [this, right]}. This method constructs semantics only; numerical execution and
     * gradient rules are deferred.</p>
     *
     * @param right non-null ordered exponent; it is retained by exact reference in result
     *     provenance and is not mutated
     * @return a non-null fresh derived tensor with promoted type, broadcast shape, unresolved
     *     layout, propagated gradient eligibility, and no storage
     * @throws NullPointerException if {@code right} is null, with message {@code right}
     * @throws IllegalArgumentException if an operand is boolean or integral, the operands cross
     *     numeric categories, or their shapes cannot be broadcast under the local shape contract
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor pow(Tensor right) {
        return TensorBinaryExpressions.apply(this, right, BinaryArithmeticKind.POW);
    }

    /**
     * Builds the vector, matrix, or batched matrix product {@code this @ right}.
     *
     * <p>Both operands must have rank at least one and same-category floating or signed-integral
     * types, which promote without an inserted cast. The final left axis contracts with the final
     * right axis for a vector or the penultimate right axis otherwise. A rank-one left operand
     * omits its temporary row axis from the result; a rank-one right operand omits its temporary
     * column axis. Leading batch dimensions broadcast right-aligned under MATMUL's local exact-
     * shape rules.</p>
     *
     * <p>Static contraction mismatches and non-broadcastable static batches fail immediately.
     * Unresolved contraction equality and unresolved-versus-static batch singleton-or-equal
     * requirements are deferred for compiler validation or later binding. Two unequal unresolved
     * batch dimensions fail because no exact result extent can be selected locally.</p>
     *
     * <p>The fresh result has the promoted type, exact derived Shape, unresolved layout,
     * gradient eligibility equal to the logical OR of the operands, no label or storage, and
     * provenance containing {@link MatmulKind#MATMUL}, {@code NoOperationAttrs.INSTANCE}, ordered
     * exact inputs {@code [this, right]}, and output index zero. FLOAT64 results accumulate in
     * FLOAT64; FLOAT32 and BFLOAT16 results accumulate in FLOAT32, with BFLOAT16 conversion at
     * output. Floating reassociation and fused multiply-add are permitted without a bitwise-order
     * guarantee. Signed-integral accumulation is modular in the promoted width. Empty contraction
     * produces positive floating zero or integral zero. This method does not evaluate values,
     * guarantee gradient support, capture or compile a graph, choose a backend, allocate result
     * storage, or execute.</p>
     *
     * @param right non-null ordered right operand retained by exact reference in result provenance
     *     and not mutated
     * @return a non-null fresh, unlabeled, storage-free result with promoted type, exact derived
     *     Shape, unresolved layout, propagated gradient request, and MATMUL provenance
     * @throws NullPointerException if {@code right} is null, with message {@code right}
     * @throws IllegalArgumentException if promotion fails; either operand has rank zero; static
     *     contraction dimensions differ; or batch dimensions cannot produce an exact locally
     *     broadcast result
     * @throws IllegalStateException if Tensor identifier space is exhausted
     */
    public Tensor matmul(Tensor right) {
        return TensorMatmulExpressions.apply(this, right);
    }

    /**
     * Builds an elementwise expression testing whether this left operand is greater than
     * {@code right}.
     *
     * <p>Both operands must belong to the same floating or signed-integral category and promote
     * within that category; mixed categories require an explicit cast. Their shapes must support
     * locally provable right-aligned broadcasting. The
     * promoted type validates the comparison domain but is not stored in the result. The fresh
     * result is {@code BOOL}, has unresolved layout and false gradient eligibility, no label or
     * host storage, and provenance containing {@link BinaryComparisonKind#GREATER_THAN},
     * {@code NoOperationAttrs.INSTANCE}, and ordered inputs {@code [this, right]}. This method
     * constructs semantics only. Integral operands use ordinary signed order after promotion;
     * floating special-value policy, gradients, graph capture, and execution are deferred.</p>
     *
     * @param right non-null ordered right comparison operand; it is retained by exact reference
     *     in result provenance and is not mutated
     * @return a non-null fresh derived {@code BOOL} tensor with broadcast shape, unresolved layout,
     *     false gradient eligibility, and no storage
     * @throws NullPointerException if {@code right} is null, with message {@code right}
     * @throws IllegalArgumentException if an operand is boolean, the operands cross numeric
     *     categories, or their shapes cannot be broadcast under the local shape contract
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor greaterThan(Tensor right) {
        return TensorComparisonExpressions.apply(this, right, BinaryComparisonKind.GREATER_THAN);
    }

    /**
     * Builds an elementwise expression testing whether this left operand is greater than or equal
     * to {@code right}.
     *
     * <p>Both operands must belong to the same floating or signed-integral category and promote
     * within that category; mixed categories require an explicit cast. Their shapes must support
     * locally provable right-aligned broadcasting. The
     * promoted type validates the comparison domain but is not stored in the result. The fresh
     * result is {@code BOOL}, has unresolved layout and false gradient eligibility, no label or
     * host storage, and provenance containing {@link BinaryComparisonKind#GREATER_OR_EQUAL},
     * {@code NoOperationAttrs.INSTANCE}, and ordered inputs {@code [this, right]}. This method
     * constructs semantics only. Integral operands use ordinary signed order after promotion;
     * floating special-value policy, gradients, graph capture, and execution are deferred.</p>
     *
     * @param right non-null ordered right comparison operand; it is retained by exact reference
     *     in result provenance and is not mutated
     * @return a non-null fresh derived {@code BOOL} tensor with broadcast shape, unresolved layout,
     *     false gradient eligibility, and no storage
     * @throws NullPointerException if {@code right} is null, with message {@code right}
     * @throws IllegalArgumentException if an operand is boolean, the operands cross numeric
     *     categories, or their shapes cannot be broadcast under the local shape contract
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor greaterOrEqual(Tensor right) {
        return TensorComparisonExpressions.apply(
                this, right, BinaryComparisonKind.GREATER_OR_EQUAL);
    }

    /**
     * Builds an elementwise expression testing whether this left operand is less than
     * {@code right}.
     *
     * <p>Both operands must belong to the same floating or signed-integral category and promote
     * within that category; mixed categories require an explicit cast. Their shapes must support
     * locally provable right-aligned broadcasting. The
     * promoted type validates the comparison domain but is not stored in the result. The fresh
     * result is {@code BOOL}, has unresolved layout and false gradient eligibility, no label or
     * host storage, and provenance containing {@link BinaryComparisonKind#LESS_THAN},
     * {@code NoOperationAttrs.INSTANCE}, and ordered inputs {@code [this, right]}. This method
     * constructs semantics only. Integral operands use ordinary signed order after promotion;
     * floating special-value policy, gradients, graph capture, and execution are deferred.</p>
     *
     * @param right non-null ordered right comparison operand; it is retained by exact reference
     *     in result provenance and is not mutated
     * @return a non-null fresh derived {@code BOOL} tensor with broadcast shape, unresolved layout,
     *     false gradient eligibility, and no storage
     * @throws NullPointerException if {@code right} is null, with message {@code right}
     * @throws IllegalArgumentException if an operand is boolean, the operands cross numeric
     *     categories, or their shapes cannot be broadcast under the local shape contract
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor lessThan(Tensor right) {
        return TensorComparisonExpressions.apply(this, right, BinaryComparisonKind.LESS_THAN);
    }

    /**
     * Builds an elementwise expression testing whether this left operand is less than or equal to
     * {@code right}.
     *
     * <p>Both operands must belong to the same floating or signed-integral category and promote
     * within that category; mixed categories require an explicit cast. Their shapes must support
     * locally provable right-aligned broadcasting. The
     * promoted type validates the comparison domain but is not stored in the result. The fresh
     * result is {@code BOOL}, has unresolved layout and false gradient eligibility, no label or
     * host storage, and provenance containing {@link BinaryComparisonKind#LESS_OR_EQUAL},
     * {@code NoOperationAttrs.INSTANCE}, and ordered inputs {@code [this, right]}. This method
     * constructs semantics only. Integral operands use ordinary signed order after promotion;
     * floating special-value policy, gradients, graph capture, and execution are deferred.</p>
     *
     * @param right non-null ordered right comparison operand; it is retained by exact reference
     *     in result provenance and is not mutated
     * @return a non-null fresh derived {@code BOOL} tensor with broadcast shape, unresolved layout,
     *     false gradient eligibility, and no storage
     * @throws NullPointerException if {@code right} is null, with message {@code right}
     * @throws IllegalArgumentException if an operand is boolean, the operands cross numeric
     *     categories, or their shapes cannot be broadcast under the local shape contract
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor lessOrEqual(Tensor right) {
        return TensorComparisonExpressions.apply(
                this, right, BinaryComparisonKind.LESS_OR_EQUAL);
    }

    /**
     * Builds an elementwise expression testing whether this left operand compares equal to
     * {@code right}.
     *
     * <p>Both operands must belong to the same floating or signed-integral category and promote
     * within that category; mixed categories require an explicit cast. Their shapes must support
     * locally provable right-aligned broadcasting. The
     * promoted type validates the comparison domain but is not stored in the result. The fresh
     * result is {@code BOOL}, has unresolved layout and false gradient eligibility, no label or
     * host storage, and provenance containing {@link BinaryComparisonKind#EQUAL},
     * {@code NoOperationAttrs.INSTANCE}, and ordered inputs {@code [this, right]}. Operand order is
     * retained even though equality is symmetric. Integral equality is exact after signed
     * promotion; floating tolerance and special-value policy, gradients, graph capture, and
     * execution are deferred.</p>
     *
     * @param right non-null ordered right comparison operand; it is retained by exact reference
     *     in result provenance and is not mutated
     * @return a non-null fresh derived {@code BOOL} tensor with broadcast shape, unresolved layout,
     *     false gradient eligibility, and no storage
     * @throws NullPointerException if {@code right} is null, with message {@code right}
     * @throws IllegalArgumentException if an operand is boolean, the operands cross numeric
     *     categories, or their shapes cannot be broadcast under the local shape contract
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor equalTo(Tensor right) {
        return TensorComparisonExpressions.apply(this, right, BinaryComparisonKind.EQUAL);
    }

    /**
     * Builds an elementwise expression testing whether this left operand compares unequal to
     * {@code right}.
     *
     * <p>Both operands must belong to the same floating or signed-integral category and promote
     * within that category; mixed categories require an explicit cast. Their shapes must support
     * locally provable right-aligned broadcasting. The
     * promoted type validates the comparison domain but is not stored in the result. The fresh
     * result is {@code BOOL}, has unresolved layout and false gradient eligibility, no label or
     * host storage, and provenance containing {@link BinaryComparisonKind#NOT_EQUAL},
     * {@code NoOperationAttrs.INSTANCE}, and ordered inputs {@code [this, right]}. Operand order is
     * retained even though inequality is symmetric. Integral inequality is exact after signed
     * promotion; floating tolerance and special-value policy, gradients, graph capture, and
     * execution are deferred.</p>
     *
     * @param right non-null ordered right comparison operand; it is retained by exact reference
     *     in result provenance and is not mutated
     * @return a non-null fresh derived {@code BOOL} tensor with broadcast shape, unresolved layout,
     *     false gradient eligibility, and no storage
     * @throws NullPointerException if {@code right} is null, with message {@code right}
     * @throws IllegalArgumentException if an operand is boolean, the operands cross numeric
     *     categories, or their shapes cannot be broadcast under the local shape contract
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor notEqualTo(Tensor right) {
        return TensorComparisonExpressions.apply(this, right, BinaryComparisonKind.NOT_EQUAL);
    }

    /**
     * Builds an elementwise boolean conjunction of this ordered left input and {@code right}.
     *
     * <p>Both inputs must have exactly {@code BOOL} data type, and their shapes must support
     * locally provable right-aligned broadcasting. The fresh result is {@code BOOL}, has the
     * broadcast shape, unresolved layout, false gradient eligibility, no label or host storage,
     * and provenance containing {@link BooleanLogicalKind#AND},
     * {@code NoOperationAttrs.INSTANCE}, and exact ordered inputs {@code [this, right]}. Operand
     * order is retained even though conjunction is commutative. Construction does not inspect
     * truth values or storage and does not provide Java-style short-circuiting, simplification,
     * gradient rules, graph capture, or execution.</p>
     *
     * @param right non-null ordered right {@code BOOL} input; it is retained by exact reference in
     *     result provenance and is not mutated
     * @return a non-null fresh derived {@code BOOL} tensor with broadcast shape, unresolved layout,
     *     false gradient eligibility, and no storage
     * @throws NullPointerException if {@code right} is null, with message {@code right}
     * @throws IllegalArgumentException if either input is not {@code BOOL} or their shapes cannot
     *     be broadcast under the local shape contract
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor logicalAnd(Tensor right) {
        return TensorLogicalExpressions.applyBinary(this, right, BooleanLogicalKind.AND);
    }

    /**
     * Builds an elementwise boolean disjunction of this ordered left input and {@code right}.
     *
     * <p>Both inputs must have exactly {@code BOOL} data type, and their shapes must support
     * locally provable right-aligned broadcasting. The fresh result is {@code BOOL}, has the
     * broadcast shape, unresolved layout, false gradient eligibility, no label or host storage,
     * and provenance containing {@link BooleanLogicalKind#OR},
     * {@code NoOperationAttrs.INSTANCE}, and exact ordered inputs {@code [this, right]}. Operand
     * order is retained even though disjunction is commutative. Construction does not inspect
     * truth values or storage and does not provide Java-style short-circuiting, simplification,
     * gradient rules, graph capture, or execution.</p>
     *
     * @param right non-null ordered right {@code BOOL} input; it is retained by exact reference in
     *     result provenance and is not mutated
     * @return a non-null fresh derived {@code BOOL} tensor with broadcast shape, unresolved layout,
     *     false gradient eligibility, and no storage
     * @throws NullPointerException if {@code right} is null, with message {@code right}
     * @throws IllegalArgumentException if either input is not {@code BOOL} or their shapes cannot
     *     be broadcast under the local shape contract
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor logicalOr(Tensor right) {
        return TensorLogicalExpressions.applyBinary(this, right, BooleanLogicalKind.OR);
    }

    /**
     * Builds an elementwise boolean negation of this input.
     *
     * <p>This input must have exactly {@code BOOL} data type. The fresh result is {@code BOOL},
     * retains the exact immutable input-shape reference, has unresolved layout, false gradient
     * eligibility, no label or host storage, and provenance containing
     * {@link BooleanLogicalKind#NOT}, {@code NoOperationAttrs.INSTANCE}, and exactly this input.
     * Construction performs no broadcasting, truth-value or storage inspection, double-negation
     * collapse, gradient rule, graph capture, or execution.</p>
     *
     * @return a non-null fresh derived {@code BOOL} tensor with the exact input shape, unresolved
     *     layout, false gradient eligibility, and no storage
     * @throws IllegalArgumentException if this input's data type is not {@code BOOL}
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor logicalNot() {
        return TensorLogicalExpressions.applyUnary(this, BooleanLogicalKind.NOT);
    }

    /**
     * Builds an elementwise expression that conditionally selects between two ordered branches.
     *
     * <p>{@code condition} must have exact {@code BOOL} data type. {@code ifTrue} and
     * {@code ifFalse} must be floating tensors; their data types are promoted through the shared
     * floating hierarchy. Shape construction first broadcasts the true and false branches, then
     * broadcasts the condition with that common branch shape. The fresh result has the promoted
     * branch data type, unresolved layout, gradient eligibility equal only to the logical OR of
     * the two branch requests, no label or host storage, and provenance containing
     * {@link WhereSelectionKind#WHERE}, {@code NoOperationAttrs.INSTANCE}, and exact ordered inputs
     * {@code [condition, ifTrue, ifFalse]}.</p>
     *
     * <p>This method eagerly constructs expression metadata only. It does not inspect values,
     * choose or evaluate either branch, define evaluation order or gradient routing, create a
     * ternary broadcast plan, or implement scalar-index selection. The supplied tensors and all
     * of their metadata and storage associations remain unchanged.</p>
     *
     * @param condition non-null exact {@code BOOL} condition retained by exact reference as the
     *     first provenance input; it is not mutated and does not contribute gradient eligibility
     * @param ifTrue non-null ordered true branch with floating data type, retained by exact
     *     reference as the second provenance input and not mutated
     * @param ifFalse non-null ordered false branch with floating data type, retained by exact
     *     reference as the third provenance input and not mutated
     * @return a non-null fresh derived tensor with promoted branch type, locally proven three-way
     *     broadcast shape, unresolved layout, branch-only gradient eligibility, and no storage
     * @throws NullPointerException if {@code condition}, {@code ifTrue}, or {@code ifFalse} is
     *     null, checked in that order with the parameter name as the message
     * @throws IllegalArgumentException if the condition is not {@code BOOL}, either branch is not
     *     floating, the two branch shapes cannot be broadcast, or the condition cannot be
     *     broadcast with the common branch shape
     * @throws IllegalStateException if tensor identifier space is exhausted after local model
     *     values have been constructed
     */
    public static Tensor where(Tensor condition, Tensor ifTrue, Tensor ifFalse) {
        return TensorWhereExpressions.apply(condition, ifTrue, ifFalse);
    }

    /**
     * Builds scalar addition using an exact binary64 addend.
     *
     * <p>This convenience constructs {@link ScalarValue#float64(double)} and therefore requires
     * an exact FLOAT64 receiver. It delegates to the typed overload without inferring or narrowing
     * the scalar type.</p>
     *
     * @param value binary64 addend retained with its exact primitive bits, including signed zero,
     *     infinity, and NaN payload bits
     * @return a non-null fresh storage-free expression preserving this tensor's exact type, Shape,
     *     and gradient eligibility with unresolved layout and one-input provenance
     * @throws IllegalArgumentException if this tensor's data type is not FLOAT64
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor add(double value) {
        return add(ScalarValue.float64(value));
    }

    /**
     * Builds scalar addition with an exact value matching this numeric tensor's data type.
     *
     * <p>The result records one {@link ScalarElementwiseKind#ADD} operation with the exact value
     * reference, retains this tensor as its sole provenance input, and has a fresh identity, no
     * label, and no storage. Integral addition has fixed-width two's-complement modular meaning.
     * Construction performs no addition, conversion, simplification, gradient definition, graph
     * capture, or execution.</p>
     *
     * @param value non-null exact typed addend; its data type must match this tensor
     * @return a non-null fresh storage-free expression preserving this tensor's metadata
     * @throws NullPointerException if {@code value} is null, with message {@code value}
     * @throws IllegalArgumentException if this tensor is boolean or the value type differs
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor add(ScalarValue value) {
        return TensorScalarExpressions.applyScalar(this, ScalarElementwiseKind.ADD, value);
    }

    /**
     * Builds scalar subtraction using an exact binary64 subtrahend.
     *
     * <p>This convenience constructs {@link ScalarValue#float64(double)} and therefore requires
     * an exact FLOAT64 receiver. It delegates to the typed overload without inferring or narrowing
     * the scalar type.</p>
     *
     * @param value binary64 subtrahend retained with its exact primitive bits
     * @return a non-null fresh storage-free expression preserving this tensor's exact type, Shape,
     *     and gradient eligibility with unresolved layout and one-input provenance
     * @throws IllegalArgumentException if this tensor's data type is not FLOAT64
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor sub(double value) {
        return sub(ScalarValue.float64(value));
    }

    /**
     * Builds scalar subtraction with an exact value matching this numeric tensor's data type.
     *
     * <p>The receiver is the minuend and {@code value} is the subtrahend. The result records one
     * {@link ScalarElementwiseKind#SUB} operation with the exact value reference and this tensor
     * as its sole input. Integral subtraction has fixed-width two's-complement modular meaning.
     * It does not evaluate, convert, simplify, define gradients, capture a graph, or execute.</p>
     *
     * @param value non-null exact typed subtrahend; its data type must match this tensor
     * @return a non-null fresh storage-free expression preserving this tensor's metadata
     * @throws NullPointerException if {@code value} is null, with message {@code value}
     * @throws IllegalArgumentException if this tensor is boolean or the value type differs
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor sub(ScalarValue value) {
        return TensorScalarExpressions.applyScalar(this, ScalarElementwiseKind.SUB, value);
    }

    /**
     * Builds an elementwise expression that multiplies this tensor by {@code scalar}.
     *
     * <p>This tensor must have exact FLOAT64 data type because this convenience adapts
     * {@code scalar} to {@link ScalarValue#float64(double)} without inference or conversion.
     * The fresh result retains the exact input data type and shape reference, has unresolved
     * layout and unchanged gradient eligibility, no label or storage, and one-input provenance
     * containing {@link ScalarElementwiseKind#MUL}. Multiplication, special-value behavior,
     * canonicalization of zero, one, or minus one, gradients, and execution are deferred.</p>
     *
     * @param scalar binary64 multiplier retained with its exact primitive bits, including signed
     *     zero, infinity, and NaN payload bits
     * @return a non-null fresh derived tensor with preserved type, shape, and gradient eligibility
     * @throws IllegalArgumentException if this tensor's data type is not FLOAT64
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor mul(double scalar) {
        return mul(ScalarValue.float64(scalar));
    }

    /**
     * Builds scalar multiplication with an exact value matching this numeric tensor's data type.
     *
     * <p>The non-null value is retained by exact reference in {@code ScalarValueAttrs}. Floating,
     * INT32, and INT64 receivers require the corresponding exact value type. The fresh
     * result preserves Shape, data type, and gradient eligibility, leaves layout unresolved, has
     * no label or storage, and records one-input {@link ScalarElementwiseKind#MUL} provenance.
     * Integral multiplication has fixed-width two's-complement modular meaning. Construction
     * performs no scalar conversion, value inspection, numerical evaluation,
     * gradient definition, graph capture, or execution.</p>
     *
     * @param scalar non-null exact typed multiplier; its data type must match this tensor
     * @return a non-null fresh storage-free expression preserving this tensor's metadata
     * @throws NullPointerException if {@code scalar} is null, with message {@code scalar}
     * @throws IllegalArgumentException if this tensor is boolean or the value data type does not
     *     equal its data type
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor mul(ScalarValue scalar) {
        return TensorScalarExpressions.applyScalar(this, ScalarElementwiseKind.MUL, scalar);
    }

    /**
     * Builds scalar division using an exact binary64 denominator.
     *
     * <p>This convenience constructs {@link ScalarValue#float64(double)} and therefore requires
     * an exact FLOAT64 receiver. It performs no division or zero check.</p>
     *
     * @param value binary64 denominator retained with its exact primitive bits
     * @return a non-null fresh storage-free expression preserving this tensor's exact type, Shape,
     *     and gradient eligibility with unresolved layout and one-input provenance
     * @throws IllegalArgumentException if this tensor's data type is not FLOAT64
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor div(double value) {
        return div(ScalarValue.float64(value));
    }

    /**
     * Builds scalar division with an exact value matching this floating tensor's data type.
     *
     * <p>The receiver is the numerator and {@code value} is the denominator. The result records
     * one {@link ScalarElementwiseKind#DIV} operation and retains the exact value reference. It
     * does not inspect values, validate a numerical domain, define gradients, or execute.</p>
     *
     * @param value non-null exact typed denominator; its data type must match this tensor
     * @return a non-null fresh storage-free expression preserving this tensor's metadata
     * @throws NullPointerException if {@code value} is null, with message {@code value}
     * @throws IllegalArgumentException if this tensor is not floating or the value type differs
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor div(ScalarValue value) {
        return TensorScalarExpressions.applyScalar(this, ScalarElementwiseKind.DIV, value);
    }

    /**
     * Builds pairwise scalar minimum using an exact binary64 candidate.
     *
     * <p>This convenience constructs {@link ScalarValue#float64(double)} and therefore requires
     * an exact FLOAT64 receiver. It delegates to the typed pairwise operation; it is not a
     * reduction.</p>
     *
     * @param value binary64 minimum candidate retained with its exact primitive bits
     * @return a non-null fresh storage-free expression preserving this tensor's exact type, Shape,
     *     and gradient eligibility with unresolved layout and one-input provenance
     * @throws IllegalArgumentException if this tensor's data type is not FLOAT64
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor minimum(double value) {
        return minimum(ScalarValue.float64(value));
    }

    /**
     * Builds pairwise scalar minimum with an exact value matching this numeric tensor's type.
     *
     * <p>The result records one {@link ScalarElementwiseKind#MIN} operation with the exact value
     * reference. Floating minimum retains its NaN, infinity, and signed-zero policy; integral
     * minimum uses ordinary signed order. This pairwise method does not reduce an axis, inspect
     * values, define gradients, capture a graph, or execute.</p>
     *
     * @param value non-null exact typed minimum candidate; its data type must match this tensor
     * @return a non-null fresh storage-free expression preserving this tensor's metadata
     * @throws NullPointerException if {@code value} is null, with message {@code value}
     * @throws IllegalArgumentException if this tensor is boolean or the value type differs
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor minimum(ScalarValue value) {
        return TensorScalarExpressions.applyScalar(this, ScalarElementwiseKind.MIN, value);
    }

    /**
     * Builds pairwise scalar maximum using an exact binary64 candidate.
     *
     * <p>This convenience constructs {@link ScalarValue#float64(double)} and therefore requires
     * an exact FLOAT64 receiver. It delegates to the typed pairwise operation; it is not a
     * reduction.</p>
     *
     * @param value binary64 maximum candidate retained with its exact primitive bits
     * @return a non-null fresh storage-free expression preserving this tensor's exact type, Shape,
     *     and gradient eligibility with unresolved layout and one-input provenance
     * @throws IllegalArgumentException if this tensor's data type is not FLOAT64
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor maximum(double value) {
        return maximum(ScalarValue.float64(value));
    }

    /**
     * Builds pairwise scalar maximum with an exact value matching this numeric tensor's type.
     *
     * <p>The result records one {@link ScalarElementwiseKind#MAX} operation with the exact value
     * reference. Floating maximum retains its NaN, infinity, and signed-zero policy; integral
     * maximum uses ordinary signed order. This pairwise method does not reduce an axis, inspect
     * values, define gradients, capture a graph, or execute.</p>
     *
     * @param value non-null exact typed maximum candidate; its data type must match this tensor
     * @return a non-null fresh storage-free expression preserving this tensor's metadata
     * @throws NullPointerException if {@code value} is null, with message {@code value}
     * @throws IllegalArgumentException if this tensor is boolean or the value type differs
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor maximum(ScalarValue value) {
        return TensorScalarExpressions.applyScalar(this, ScalarElementwiseKind.MAX, value);
    }

    /**
     * Builds an elementwise expression that raises this tensor to {@code exponent}.
     *
     * <p>This tensor must have exact FLOAT64 data type because this convenience adapts the
     * exponent to {@link ScalarValue#float64(double)} without inference or conversion. The
     * fresh result retains the exact input data type and shape reference, has unresolved layout
     * and unchanged gradient eligibility, no label or storage, and one-input provenance containing
     * {@link ScalarElementwiseKind#POW}. Power evaluation, special-value policy, algebraic
     * rewrites, gradients, and execution are deferred.</p>
     *
     * @param exponent binary64 exponent retained with its exact primitive bits, including signed
     *     zero, infinity, and NaN payload bits
     * @return a non-null fresh derived tensor with preserved type, shape, and gradient eligibility
     * @throws IllegalArgumentException if this tensor's data type is not FLOAT64
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor pow(double exponent) {
        return pow(ScalarValue.float64(exponent));
    }

    /**
     * Builds scalar exponentiation with an exact value matching this floating tensor's data type.
     *
     * <p>The value is retained by exact reference in {@code ScalarValueAttrs}; no conversion or
     * numerical-domain validation occurs. The fresh result preserves Shape, data type, and
     * gradient eligibility, leaves layout unresolved, and records exact one-input
     * {@link ScalarElementwiseKind#POW} provenance without evaluating a power.</p>
     *
     * @param exponent non-null exact typed exponent; its data type must match this tensor
     * @return a non-null fresh storage-free expression preserving this tensor's metadata
     * @throws NullPointerException if {@code exponent} is null, with message {@code exponent}
     * @throws IllegalArgumentException if this tensor is not floating or the value data type does
     *     not equal its data type
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor pow(ScalarValue exponent) {
        return TensorScalarExpressions.applyScalar(this, ScalarElementwiseKind.POW, exponent);
    }

    /**
     * Builds one first-class elementwise expression that clamps this tensor to the inclusive range
     * {@code [minValue, maxValue]}.
     *
     * <p>This tensor must have exact FLOAT64 data type because this convenience adapts both bounds
     * through {@link ScalarValue#float64(double)}. Equal bounds, signed zeros, infinities, and
     * NaNs are representable, while a strict primitive {@code minValue > maxValue} is rejected.
     * The fresh result retains the exact input data type
     * and shape reference, has unresolved layout and unchanged gradient eligibility, no label or
     * storage, and one-input provenance containing a single {@link ScalarElementwiseKind#CLAMP}
     * operation. Its value meaning is {@code minimum(maximum(input, minValue), maxValue)} under
     * the selected NaN-propagating, normally ordered infinity and signed-zero extrema policy.
     * Scalar conversion, range simplification or expansion, gradients, and execution are
     * deferred.</p>
     *
     * @param minValue inclusive binary64 lower bound retained with its exact primitive bits
     * @param maxValue inclusive binary64 upper bound retained with its exact primitive bits
     * @return a non-null fresh derived tensor with preserved type, shape, and gradient eligibility
     * @throws IllegalArgumentException if this tensor's data type is not FLOAT64, or if
     *     {@code minValue > maxValue}
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor clamp(double minValue, double maxValue) {
        return clamp(ScalarValue.float64(minValue), ScalarValue.float64(maxValue));
    }

    /**
     * Builds an inclusive clamp with exact bounds matching this floating tensor's data type.
     *
     * <p>Validation requires non-null bounds, a floating receiver, identical numeric bound types,
     * a non-inverted represented range, and exact bound/receiver data-type equality. The bound
     * references are retained in one {@code ClampRangeAttrs}. The fresh result preserves Shape,
     * data type, and gradient eligibility and records one first-class CLAMP operation. Its value
     * meaning composes maximum with the lower bound and then minimum with the upper bound under
     * the selected extrema semantics, without stored intermediates. It does not convert or inspect
     * values, define gradients, or execute.</p>
     *
     * @param minValue non-null exact typed inclusive lower bound
     * @param maxValue non-null exact typed inclusive upper bound of the same type
     * @return a non-null fresh storage-free expression preserving this tensor's metadata
     * @throws NullPointerException if a bound is null, checked in parameter order with its name
     *     as the message
     * @throws IllegalArgumentException if this tensor is not floating, the bounds have different
     *     types or BOOL type, the range is inverted, or the common bound type does not match this
     *     tensor's data type
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor clamp(ScalarValue minValue, ScalarValue maxValue) {
        return TensorScalarExpressions.applyClamp(this, minValue, maxValue);
    }

    /**
     * Builds an elementwise expression that applies the inclusive lower bound {@code minValue}.
     *
     * <p>This exact-FLOAT64 convenience delegates to {@link #maximum(double)}. The fresh result is
     * therefore one scalar {@link ScalarElementwiseKind#MAX} occurrence, not a distinct clamp kind
     * or a composition. It inherits pairwise maximum's NaN, infinity, and signed-zero meaning and
     * performs no value evaluation, canonicalization, gradient work, or execution.</p>
     *
     * @param minValue inclusive binary64 lower bound retained with its exact primitive bits
     * @return a non-null fresh derived tensor with preserved type, shape, and gradient eligibility
     * @throws IllegalArgumentException if this tensor's data type is not FLOAT64
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor clampMin(double minValue) {
        return maximum(minValue);
    }

    /**
     * Builds an inclusive lower clamp with an exact value matching this numeric tensor's type.
     *
     * <p>This convenience delegates to {@link #maximum(ScalarValue)} and creates exactly one
     * scalar {@link ScalarElementwiseKind#MAX} producer. The exact bound reference, Shape, type,
     * and gradient eligibility are retained; integral bounds use signed maximum order. Layout
     * remains unresolved and no conversion,
     * intermediate expression, value evaluation, gradient rule, or execution is added.</p>
     *
     * @param minValue non-null exact typed inclusive lower bound
     * @return a non-null fresh storage-free expression preserving this tensor's metadata
     * @throws NullPointerException if {@code minValue} is null, with delegated scalar-parameter
     *     message {@code value}
     * @throws IllegalArgumentException if this tensor is boolean or the value data type does not
     *     equal its data type
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor clampMin(ScalarValue minValue) {
        return maximum(minValue);
    }

    /**
     * Builds an elementwise expression that applies the inclusive upper bound {@code maxValue}.
     *
     * <p>This exact-FLOAT64 convenience delegates to {@link #minimum(double)}. The fresh result is
     * therefore one scalar {@link ScalarElementwiseKind#MIN} occurrence, not a distinct clamp kind
     * or a composition. It inherits pairwise minimum's NaN, infinity, and signed-zero meaning and
     * performs no value evaluation, canonicalization, gradient work, or execution.</p>
     *
     * @param maxValue inclusive binary64 upper bound retained with its exact primitive bits
     * @return a non-null fresh derived tensor with preserved type, shape, and gradient eligibility
     * @throws IllegalArgumentException if this tensor's data type is not FLOAT64
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor clampMax(double maxValue) {
        return minimum(maxValue);
    }

    /**
     * Builds an inclusive upper clamp with an exact value matching this numeric tensor's type.
     *
     * <p>This convenience delegates to {@link #minimum(ScalarValue)} and creates exactly one
     * scalar {@link ScalarElementwiseKind#MIN} producer. The exact bound reference, Shape, type,
     * and gradient eligibility are retained; integral bounds use signed minimum order. Layout
     * remains unresolved and no conversion,
     * intermediate expression, value evaluation, gradient rule, or execution is added.</p>
     *
     * @param maxValue non-null exact typed inclusive upper bound
     * @return a non-null fresh storage-free expression preserving this tensor's metadata
     * @throws NullPointerException if {@code maxValue} is null, with delegated scalar-parameter
     *     message {@code value}
     * @throws IllegalArgumentException if this tensor is boolean or the value data type does not
     *     equal its data type
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor clampMax(ScalarValue maxValue) {
        return minimum(maxValue);
    }

    /**
     * Builds an elementwise expression for the absolute magnitude of this tensor.
     *
     * <p>The input must be floating. The fresh result retains the exact data type and shape
     * reference, has unresolved layout and unchanged gradient eligibility, no label or storage,
     * and provenance containing {@link UnaryElementwiseKind#ABS},
     * {@code NoOperationAttrs.INSTANCE}, and exactly this input. Numerical and special-value
     * behavior, gradient rules, execution, and backend support are deferred.</p>
     *
     * @return a non-null fresh derived tensor with preserved type, shape, and gradient eligibility
     * @throws IllegalArgumentException if this tensor's data type is not floating
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor abs() {
        return TensorUnaryExpressions.apply(this, UnaryElementwiseKind.ABS);
    }

    /**
     * Builds an elementwise expression for the additive inverse of this tensor.
     *
     * <p>The input must be floating. The fresh result retains the exact data type and shape
     * reference, has unresolved layout and unchanged gradient eligibility, no label or storage,
     * and provenance containing {@link UnaryElementwiseKind#NEG},
     * {@code NoOperationAttrs.INSTANCE}, and exactly this input. Numerical and special-value
     * behavior, canonicalization, gradient rules, execution, and backend support are deferred.</p>
     *
     * @return a non-null fresh derived tensor with preserved type, shape, and gradient eligibility
     * @throws IllegalArgumentException if this tensor's data type is not floating
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor neg() {
        return TensorUnaryExpressions.apply(this, UnaryElementwiseKind.NEG);
    }

    /**
     * Builds an elementwise expression for the multiplicative reciprocal of this tensor.
     *
     * <p>The input must be floating. The fresh result retains the exact data type and shape
     * reference, has unresolved layout and unchanged gradient eligibility, no label or storage,
     * and provenance containing {@link UnaryElementwiseKind#RECIPROCAL},
     * {@code NoOperationAttrs.INSTANCE}, and exactly this input. Zero handling, numerical domain,
     * special-value policy, canonicalization, gradient rules, execution, and backend support are
     * deferred.</p>
     *
     * @return a non-null fresh derived tensor with preserved type, shape, and gradient eligibility
     * @throws IllegalArgumentException if this tensor's data type is not floating
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor reciprocal() {
        return TensorUnaryExpressions.apply(this, UnaryElementwiseKind.RECIPROCAL);
    }

    /**
     * Builds an elementwise natural-logarithm expression from this tensor.
     *
     * <p>The input must be floating. The fresh result retains the exact data type and shape
     * reference, has unresolved layout and unchanged gradient eligibility, no label or storage,
     * and provenance containing {@link UnaryElementwiseKind#LOG},
     * {@code NoOperationAttrs.INSTANCE}, and exactly this input. Mathematical-domain and
     * special-value handling, accuracy, gradient rules, execution, and backend support are
     * deferred.</p>
     *
     * @return a non-null fresh derived tensor with preserved type, shape, and gradient eligibility
     * @throws IllegalArgumentException if this tensor's data type is not floating
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor log() {
        return TensorUnaryExpressions.apply(this, UnaryElementwiseKind.LOG);
    }

    /**
     * Builds an elementwise natural-logarithm-of-one-plus-input expression from this Tensor.
     *
     * <p>The input must be floating. This is one first-class {@link UnaryElementwiseKind#LOG1P}
     * request rather than stored addition and logarithm operations. The fresh result preserves
     * the exact input type, shape reference, and gradient-eligibility request, leaves layout
     * unresolved, and has no label or storage. The mathematical target preserves signed zero,
     * produces negative infinity at negative one, NaN below negative one or for NaN input, and
     * positive infinity for positive infinity. Model construction does not evaluate those values or select an
     * accuracy policy, gradient rule, execution route, or backend.</p>
     *
     * @return a non-null fresh derived Tensor preserving type, shape, and gradient eligibility
     * @throws IllegalArgumentException if this Tensor's data type is not floating
     * @throws IllegalStateException if Tensor identifier space is exhausted
     */
    public Tensor log1p() {
        return TensorUnaryExpressions.apply(this, UnaryElementwiseKind.LOG1P);
    }

    /**
     * Builds an elementwise natural-exponential expression from this tensor.
     *
     * <p>The input must be floating. The fresh result retains the exact data type and shape
     * reference, has unresolved layout and unchanged gradient eligibility, no label or storage,
     * and provenance containing the portable mathematical request {@link
     * UnaryElementwiseKind#EXP}, {@code NoOperationAttrs.INSTANCE}, and exactly this input. The
     * request selects no algorithm and promises no bitwise result, approximation bound, or
     * backend route. Numerical accuracy, overflow, special values, gradients, execution, and
     * backend support are deferred.</p>
     *
     * @return a non-null fresh derived tensor with preserved type, shape, and gradient eligibility
     * @throws IllegalArgumentException if this tensor's data type is not floating
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor exp() {
        return TensorUnaryExpressions.apply(this, UnaryElementwiseKind.EXP);
    }

    /**
     * Builds an elementwise natural-exponential-minus-one expression from this Tensor.
     *
     * <p>The input must be floating. This is one first-class {@link UnaryElementwiseKind#EXPM1}
     * request rather than stored exponential and subtraction operations. The fresh result
     * preserves the exact input type, shape reference, and gradient-eligibility request, leaves
     * layout unresolved, and has no label or storage. The mathematical target preserves signed
     * zero, maps negative infinity to negative one, maps positive infinity to positive infinity,
     * and produces NaN for NaN input. Model construction does not evaluate those values or select an accuracy policy,
     * gradient rule, execution route, or backend.</p>
     *
     * @return a non-null fresh derived Tensor preserving type, shape, and gradient eligibility
     * @throws IllegalArgumentException if this Tensor's data type is not floating
     * @throws IllegalStateException if Tensor identifier space is exhausted
     */
    public Tensor expm1() {
        return TensorUnaryExpressions.apply(this, UnaryElementwiseKind.EXPM1);
    }

    /**
     * Builds an elementwise Gaussian error-function expression from this tensor.
     *
     * <p>The input must be floating. The fresh result retains the exact data type and shape
     * reference, has unresolved layout and unchanged gradient eligibility, no label or storage,
     * and provenance containing {@link UnaryElementwiseKind#ERF},
     * {@code NoOperationAttrs.INSTANCE}, and exactly this input. Accuracy, special-value behavior,
     * gradient rules, execution, and backend support are deferred.</p>
     *
     * @return a non-null fresh derived tensor with preserved type, shape, and gradient eligibility
     * @throws IllegalArgumentException if this tensor's data type is not floating
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor erf() {
        return TensorUnaryExpressions.apply(this, UnaryElementwiseKind.ERF);
    }

    /**
     * Builds an elementwise principal-square-root expression from this tensor.
     *
     * <p>The input must be floating. The fresh result retains the exact data type and shape
     * reference, has unresolved layout and unchanged gradient eligibility, no label or storage,
     * and provenance containing {@link UnaryElementwiseKind#SQRT},
     * {@code NoOperationAttrs.INSTANCE}, and exactly this input. Mathematical-domain and
     * special-value handling, accuracy, gradient rules, execution, and backend support are
     * deferred.</p>
     *
     * @return a non-null fresh derived tensor with preserved type, shape, and gradient eligibility
     * @throws IllegalArgumentException if this tensor's data type is not floating
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor sqrt() {
        return TensorUnaryExpressions.apply(this, UnaryElementwiseKind.SQRT);
    }

    /**
     * Builds an elementwise reciprocal-square-root expression from this Tensor.
     *
     * <p>The input must be floating. This is one first-class {@link UnaryElementwiseKind#RSQRT}
     * request rather than stored square-root and reciprocal operations. The fresh result preserves
     * the exact input type, shape reference, and gradient-eligibility request, leaves layout
     * unresolved, and has no label or storage. Its mathematical target is {@code 1 / sqrt(x)}:
     * signed zero maps to same-signed infinity, positive infinity maps to positive zero, and
     * negative finite values and negative infinity map to NaN, and NaN maps to NaN. Model
     * construction does not evaluate values or select an accuracy policy, gradient rule,
     * execution route, or backend.</p>
     *
     * @return a non-null fresh derived Tensor preserving type, shape, and gradient eligibility
     * @throws IllegalArgumentException if this Tensor's data type is not floating
     * @throws IllegalStateException if Tensor identifier space is exhausted
     */
    public Tensor rsqrt() {
        return TensorUnaryExpressions.apply(this, UnaryElementwiseKind.RSQRT);
    }

    /**
     * Builds an elementwise floor expression from this tensor.
     *
     * <p>The input must be floating. The fresh result retains the exact data type and shape
     * reference, has unresolved layout and unchanged gradient eligibility, no label or storage,
     * and provenance containing {@link UnaryElementwiseKind#FLOOR},
     * {@code NoOperationAttrs.INSTANCE}, and exactly this input. Representation, special values,
     * gradient policy, execution, and backend support are deferred; preserving
     * {@code requiresGrad} does not define a derivative.</p>
     *
     * @return a non-null fresh derived tensor with preserved type, shape, and gradient eligibility
     * @throws IllegalArgumentException if this tensor's data type is not floating
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor floor() {
        return TensorUnaryExpressions.apply(this, UnaryElementwiseKind.FLOOR);
    }

    /**
     * Builds an elementwise ceiling expression from this tensor.
     *
     * <p>The input must be floating. The fresh result retains the exact data type and shape
     * reference, has unresolved layout and unchanged gradient eligibility, no label or storage,
     * and provenance containing {@link UnaryElementwiseKind#CEIL},
     * {@code NoOperationAttrs.INSTANCE}, and exactly this input. Representation, special values,
     * gradient policy, execution, and backend support are deferred; preserving
     * {@code requiresGrad} does not define a derivative.</p>
     *
     * @return a non-null fresh derived tensor with preserved type, shape, and gradient eligibility
     * @throws IllegalArgumentException if this tensor's data type is not floating
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor ceil() {
        return TensorUnaryExpressions.apply(this, UnaryElementwiseKind.CEIL);
    }

    /**
     * Builds an elementwise numeric sign-classification expression from this tensor.
     *
     * <p>The input must be floating. The fresh result retains the exact data type and shape
     * reference, has unresolved layout and unchanged gradient eligibility, no label or storage,
     * and provenance containing {@link UnaryElementwiseKind#SIGN},
     * {@code NoOperationAttrs.INSTANCE}, and exactly this input. Exact representation, signed-zero
     * and NaN behavior, gradient policy, execution, and backend support are deferred; preserving
     * {@code requiresGrad} does not define a derivative.</p>
     *
     * @return a non-null fresh derived tensor with preserved type, shape, and gradient eligibility
     * @throws IllegalArgumentException if this tensor's data type is not floating
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor sign() {
        return TensorUnaryExpressions.apply(this, UnaryElementwiseKind.SIGN);
    }

    /**
     * Builds an elementwise rectified-linear-unit expression from this tensor.
     *
     * <p>The input must be floating. The fresh result retains the exact data type and shape
     * reference, has unresolved layout and unchanged gradient eligibility, no label or storage,
     * and provenance containing {@link UnaryElementwiseKind#RELU},
     * {@code NoOperationAttrs.INSTANCE}, and exactly this input. Zero and special-value behavior,
     * gradient convention, execution, and backend support are deferred.</p>
     *
     * @return a non-null fresh derived tensor with preserved type, shape, and gradient eligibility
     * @throws IllegalArgumentException if this tensor's data type is not floating
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor relu() {
        return TensorUnaryExpressions.apply(this, UnaryElementwiseKind.RELU);
    }

    /**
     * Builds an elementwise logistic-sigmoid expression from this tensor.
     *
     * <p>The input must be floating. The fresh result retains the exact data type and shape
     * reference, has unresolved layout and unchanged gradient eligibility, no label or storage,
     * and provenance containing {@link UnaryElementwiseKind#SIGMOID},
     * {@code NoOperationAttrs.INSTANCE}, and exactly this input. Numerical stability, accuracy,
     * special values, gradient rules, execution, and backend support are deferred.</p>
     *
     * @return a non-null fresh derived tensor with preserved type, shape, and gradient eligibility
     * @throws IllegalArgumentException if this tensor's data type is not floating
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor sigmoid() {
        return TensorUnaryExpressions.apply(this, UnaryElementwiseKind.SIGMOID);
    }

    /**
     * Builds an elementwise hyperbolic-tangent expression from this tensor.
     *
     * <p>The input must be floating. The fresh result retains the exact data type and shape
     * reference, has unresolved layout and unchanged gradient eligibility, no label or storage,
     * and provenance containing the portable mathematical request {@link
     * UnaryElementwiseKind#TANH}, {@code NoOperationAttrs.INSTANCE}, and exactly this input. The
     * request selects no algorithm and promises no bitwise result, approximation bound, or
     * backend route. Accuracy, special values, gradient rules, execution, and backend support are
     * deferred.</p>
     *
     * @return a non-null fresh derived tensor with preserved type, shape, and gradient eligibility
     * @throws IllegalArgumentException if this tensor's data type is not floating
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor tanh() {
        return TensorUnaryExpressions.apply(this, UnaryElementwiseKind.TANH);
    }

    /**
     * Builds an elementwise finite-value classification from this floating Tensor.
     *
     * <p>The fresh result has BOOL type, the exact input shape reference, unresolved layout,
     * disabled gradient eligibility, no label or storage, and provenance containing {@link
     * FloatingClassificationKind#IS_FINITE}, {@code NoOperationAttrs.INSTANCE}, and exactly this
     * input. The represented result is true for finite normal, subnormal, and signed-zero values,
     * and false for infinities and NaNs. Model construction does not inspect or classify stored
     * values, capture a graph, define gradients, or execute work.</p>
     *
     * @return a non-null fresh, non-gradient BOOL classification Tensor with the same shape
     * @throws IllegalArgumentException if this Tensor's data type is not floating
     * @throws IllegalStateException if Tensor identifier space is exhausted
     */
    public Tensor isFinite() {
        return TensorFloatingClassifications.apply(this, FloatingClassificationKind.IS_FINITE);
    }

    /**
     * Builds an elementwise not-a-number classification from this floating Tensor.
     *
     * <p>The fresh result has BOOL type, the exact input shape reference, unresolved layout,
     * disabled gradient eligibility, no label or storage, and provenance containing {@link
     * FloatingClassificationKind#IS_NAN}, {@code NoOperationAttrs.INSTANCE}, and exactly this
     * input. The represented result is true only for NaN, independent of sign, signaling or quiet
     * encoding, and payload. Model construction does not inspect or classify stored values,
     * capture a graph, define gradients, or execute work.</p>
     *
     * @return a non-null fresh, non-gradient BOOL classification Tensor with the same shape
     * @throws IllegalArgumentException if this Tensor's data type is not floating
     * @throws IllegalStateException if Tensor identifier space is exhausted
     */
    public Tensor isNaN() {
        return TensorFloatingClassifications.apply(this, FloatingClassificationKind.IS_NAN);
    }

    /**
     * Builds an elementwise infinity classification from this floating Tensor.
     *
     * <p>The fresh result has BOOL type, the exact input shape reference, unresolved layout,
     * disabled gradient eligibility, no label or storage, and provenance containing {@link
     * FloatingClassificationKind#IS_INF}, {@code NoOperationAttrs.INSTANCE}, and exactly this
     * input. The represented result is true only for positive or negative infinity and false for
     * finite values and NaNs. Model construction does not inspect or classify stored values,
     * capture a graph, define gradients, or execute work.</p>
     *
     * @return a non-null fresh, non-gradient BOOL classification Tensor with the same shape
     * @throws IllegalArgumentException if this Tensor's data type is not floating
     * @throws IllegalStateException if Tensor identifier space is exhausted
     */
    public Tensor isInf() {
        return TensorFloatingClassifications.apply(this, FloatingClassificationKind.IS_INF);
    }

    /**
     * Builds an explicit elementwise expression that converts this tensor to
     * {@code targetDataType}.
     *
     * <p>Every current source and target data-type pair is representable, including a request for
     * the receiver's existing type. The fresh result retains the receiver descriptor's exact
     * immutable shape reference, leaves layout unresolved, has no label or host storage, and
     * records {@link CastKind#CAST}, a new {@link CastAttrs} containing the exact target, and this
     * tensor as its sole provenance input. Gradient eligibility remains true only when the
     * receiver already requests gradients and both source and target types are floating.</p>
     *
     * <p>A same-type request deliberately creates a fresh explicit expression rather than
     * returning this tensor. Compiler optimization later owns redundant-cast elimination. This
     * method records model semantics only: it does not inspect or convert values, preserve
     * resolved layout, allocate storage, define numerical conversion policy or a cast-back
     * gradient rule, promise backend differentiability, capture a graph, or execute work.</p>
     *
     * @param targetDataType non-null requested result data type; the exact enum reference is
     *     retained in cast attributes
     * @return a non-null fresh derived tensor with the target data type, exact input shape,
     *     unresolved layout, derived gradient eligibility, exact one-input provenance, and no
     *     label or host storage
     * @throws NullPointerException if {@code targetDataType} is null, with message
     *     {@code targetDataType}; this failure consumes no Tensor identity
     * @throws IllegalStateException if tensor identifier space is exhausted after the cast's local
     *     immutable model values have been constructed
     */
    public Tensor cast(DataType targetDataType) {
        return TensorCastExpressions.apply(this, targetDataType);
    }

    /**
     * Builds an expression that sums this tensor over every axis.
     *
     * <p>The input must have a floating or signed-integral data type. The fresh result has the
     * canonical rank-zero scalar shape, retains the exact input data type and
     * gradient-eligibility request, leaves layout unresolved, and has no label or host storage.
     * Integral input is necessarily non-differentiable. Provenance contains
     * {@link AggregateReductionKind#SUM}, {@code NoOperationAttrs.INSTANCE}, and exactly this
     * tensor as the sole ordered producer input at output index zero.</p>
     *
     * <p>For INT32 and INT64 input, addition occurs in the exact result type modulo
     * {@code 2^32} or {@code 2^64}; reassociation is permitted, and an empty domain produces zero.
     * Scalar, static, zero-extent, and dynamic Shapes are accepted structurally. Floating sum
     * follows the documented NaN, infinity, signed-zero, and positive-zero empty-domain policy.
     * This method records semantics only:
     * it does not read storage, sum values, implement an algorithm, create a gradient rule,
     * capture a graph, lower an operation, or execute work.</p>
     *
     * @return a non-null fresh storage-free scalar tensor with unchanged numeric data type and
     *     gradient eligibility, unresolved layout, and exact one-input provenance
     * @throws IllegalArgumentException if this tensor's data type is BOOL, with message
     *     {@code input must have a numeric data type for SUM, but was BOOL}; no Tensor identity is
     *     consumed
     * @throws IllegalStateException if tensor identifier space is exhausted after local immutable
     *     expression metadata has been constructed
     */
    public Tensor sum() {
        return TensorReductionExpressions.applyFull(this, AggregateReductionKind.SUM);
    }

    /**
     * Builds an expression that sums this tensor over one axis and removes that axis.
     *
     * <p>The input must be floating or signed integral. {@code axis} accepts the Shape contract's
     * positive or negative indexing and is normalized exactly once against the input rank. The
     * selected dimension is removed, every unaffected immutable dimension reference is retained in order,
     * and reducing a rank-one tensor produces the canonical rank-zero scalar shape. The fresh
     * result preserves the exact input data type and gradient eligibility, leaves layout
     * unresolved, and has no label or storage. Provenance records
     * {@link AggregateReductionKind#SUM} with normalized single-axis attributes and exactly this
     * input.</p>
     *
     * <p>For integral input, the exact result type uses modular addition with reassociation
     * permitted, and every empty selected-axis slice produces zero. Other zero output axes may
     * make the result itself empty. Dynamic extents are accepted and use the same identity when
     * later bound to zero. Floating sum follows the documented NaN, infinity, signed-zero, and
     * positive-zero empty-slice policy. This method reads no value or
     * storage and provides no algorithm, gradient rule, compiler lowering, backend, or execution
     * behavior.</p>
     *
     * @param axis input axis in the inclusive range {@code [-rank, rank - 1]}; negative values
     *     count from the final axis
     * @return a non-null fresh storage-free tensor whose selected axis is removed, with unchanged
     *     numeric data type and gradient eligibility and unresolved layout
     * @throws IllegalArgumentException if this tensor's data type is BOOL, with message
     *     {@code input must have a numeric data type for SUM, but was BOOL}; this check precedes
     *     axis validation
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the input rank, including
     *     every axis for a scalar input
     * @throws IllegalStateException if tensor identifier space is exhausted after local immutable
     *     expression metadata has been constructed
     */
    public Tensor sum(int axis) {
        return TensorReductionExpressions.applyAxis(
                this, AggregateReductionKind.SUM, axis, false);
    }

    /**
     * Builds an expression that sums this tensor over one axis and optionally retains it.
     *
     * <p>The input must be floating or signed integral. {@code axis} is normalized exactly once
     * using the Shape contract. When {@code keepDimensions} is false, the selected axis is removed; when true, it
     * is replaced by a new static dimension of extent one. Every unaffected immutable dimension
     * reference is retained in order. The fresh result preserves the exact input data type and
     * gradient eligibility, leaves layout unresolved, has no label or storage, and records
     * {@link AggregateReductionKind#SUM}, normalized axis attributes, and this sole input.</p>
     *
     * <p>For integral input, the exact result type uses modular addition with reassociation
     * permitted, and every empty selected-axis slice produces zero independent of retention.
     * Zero and dynamic extents are accepted structurally; a dynamic selected extent later bound
     * to zero uses the same identity. Floating sum follows the documented NaN, infinity,
     * signed-zero, and positive-zero empty-slice policy. This method does not
     * inspect values or storage, choose an algorithm, define a gradient rule, or provide compiler,
     * backend, runtime, or execution behavior.</p>
     *
     * @param axis input axis in the inclusive range {@code [-rank, rank - 1]}; negative values
     *     count from the final axis
     * @param keepDimensions {@code true} to retain the selected axis with extent one, or
     *     {@code false} to remove it
     * @return a non-null fresh storage-free tensor with the requested reduction shape, unchanged
     *     numeric data type and gradient eligibility, and unresolved layout
     * @throws IllegalArgumentException if this tensor's data type is BOOL, with message
     *     {@code input must have a numeric data type for SUM, but was BOOL}; this check precedes
     *     axis validation
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the input rank, including
     *     every axis for a scalar input
     * @throws IllegalStateException if tensor identifier space is exhausted after local immutable
     *     expression metadata has been constructed
     */
    public Tensor sum(int axis, boolean keepDimensions) {
        return TensorReductionExpressions.applyAxis(
                this, AggregateReductionKind.SUM, axis, keepDimensions);
    }

    /**
     * Builds a masked sum expression over one axis and removes that axis.
     *
     * <p>This tensor must be floating and {@code mask} must have exact {@link DataType#BOOL} type.
     * Ordinary right-aligned broadcasting of the mask Shape must produce exactly this tensor's Shape. A
     * compatible broadcast that would add a leading axis or enlarge this input is rejected. If a
     * mask shaped {@code [batch, time]} is intended for the first two axes of input
     * {@code [batch, time, features]}, the caller makes that intent visible first with
     * {@code mask.expandDims(2)}, producing {@code [batch, time, 1]}; this method never guesses
     * alignment or inserts a hidden Shape transformation.</p>
     *
     * <p>The normalized axis is removed, with every unaffected Dimension reference retained. The
     * fresh result preserves this tensor's exact data type and gradient eligibility, leaves layout
     * unresolved, has no label or host storage, and records {@link AggregateReductionKind#SUM},
     * masked attributes, and exact ordered provenance {@code [this, mask]}. A false broadcast mask
     * position excludes its input before aggregation, including NaN and infinity, and a slice
     * selecting no values produces floating zero. Static zero-sized reduction axes have the same
     * empty result; runtime zero-sized or all-false dynamic slices follow this semantic contract.</p>
     *
     * <p>This method does not mutate either tensor, inspect values or storage, materialize a mask,
     * sum values, create a gradient rule, capture a graph, or execute work. Preserved gradient
     * eligibility therefore does not promise differentiation support.</p>
     *
     * @param axis input axis in the inclusive range {@code [-rank, rank - 1]}; negative values
     *     count from the final axis
     * @param mask non-null BOOL tensor whose Shape must use ordinary right-aligned broadcasting to
     *     produce exactly this tensor's Shape; ownership remains with the caller and the exact
     *     reference is retained only in immutable provenance
     * @return a non-null fresh storage-free tensor whose selected axis is removed, with unchanged
     *     floating data type and gradient eligibility, unresolved layout, and exact two-input
     *     provenance
     * @throws NullPointerException if {@code mask} is null, with message {@code mask}; no Tensor
     *     identity is consumed
     * @throws IllegalArgumentException if this tensor is not floating, the mask is not BOOL, the
     *     Shapes contain an incompatible aligned pair, or their broadcast does not equal this
     *     tensor's Shape; these failures consume no Tensor identity
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the input rank, including
     *     every axis for a scalar input; type checks precede axis validation
     * @throws IllegalStateException if tensor identifier space is exhausted after local immutable
     *     expression metadata has been constructed
     */
    public Tensor sum(int axis, Tensor mask) {
        return TensorMaskedReductionExpressions.apply(
                this, mask, AggregateReductionKind.SUM, axis);
    }

    /**
     * Builds an expression that computes the arithmetic mean over every input axis.
     *
     * <p>The input must have a floating data type. The fresh result has canonical rank-zero scalar
     * shape, preserves the exact input data type and gradient eligibility, leaves layout
     * unresolved, and has no label or storage. Its provenance contains
     * {@link AggregateReductionKind#MEAN}, {@code NoOperationAttrs.INSTANCE}, and exactly this
     * tensor. Scalar, static, zero-extent, and dynamic shapes are accepted structurally.</p>
     *
     * <p>Mean is exact sum divided by count: NaN and opposite infinities produce NaN, a sole
     * infinity sign is preserved, an empty domain produces NaN, and zero sign follows SUM. This
     * method reads no values and selects no execution algorithm, gradient, compiler, or backend
     * behavior.</p>
     *
     * @return a non-null fresh storage-free scalar tensor with unchanged floating data type and
     *     gradient eligibility, unresolved layout, and exact one-input provenance
     * @throws IllegalArgumentException if this tensor's data type is not floating, with a message
     *     containing the rejected data type; no Tensor identity is consumed
     * @throws IllegalStateException if tensor identifier space is exhausted after local immutable
     *     expression metadata has been constructed
     */
    public Tensor mean() {
        return TensorReductionExpressions.applyFull(this, AggregateReductionKind.MEAN);
    }

    /**
     * Builds an arithmetic-mean expression over one axis and removes that axis.
     *
     * <p>The floating input's positive or negative {@code axis} is normalized exactly once. The
     * selected dimension is removed, every unaffected dimension reference is retained in order,
     * and rank one reduces to the canonical scalar shape. The fresh result has the exact input
     * type and gradient eligibility, unresolved layout, no label or storage, and provenance with
     * {@link AggregateReductionKind#MEAN}, normalized single-axis attributes, and this input.</p>
     *
     * <p>NaN and opposite infinities produce NaN, a sole infinity sign is preserved, an empty
     * selected axis produces NaN, and zero sign follows SUM. No values are read and no algorithm,
     * gradient, compiler, backend, or execution behavior is selected.</p>
     *
     * @param axis input axis in the inclusive range {@code [-rank, rank - 1]}; negative values
     *     count from the final axis
     * @return a non-null fresh storage-free tensor whose selected axis is removed, with unchanged
     *     floating data type and gradient eligibility and unresolved layout
     * @throws IllegalArgumentException if this tensor's data type is not floating, with a message
     *     containing the rejected data type; this check precedes axis validation
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the input rank, including
     *     every axis for a scalar input
     * @throws IllegalStateException if tensor identifier space is exhausted after local immutable
     *     expression metadata has been constructed
     */
    public Tensor mean(int axis) {
        return TensorReductionExpressions.applyAxis(
                this, AggregateReductionKind.MEAN, axis, false);
    }

    /**
     * Builds an arithmetic-mean expression over one axis and optionally retains it.
     *
     * <p>The floating input's {@code axis} is normalized exactly once. A false
     * {@code keepDimensions} removes the selected axis; true replaces it by a new static extent
     * one while preserving every other dimension reference. The fresh result retains exact input
     * type and gradient eligibility, has unresolved layout and no label or storage, and records
     * {@link AggregateReductionKind#MEAN}, normalized axis attributes, and this sole input.</p>
     *
     * <p>Zero and dynamic extents remain structurally valid. The completed floating mean policy
     * covers NaN, infinities, zero sign, and empty selected axes. No values are read, and no
     * algorithm, gradient, compiler, backend, or execution behavior is selected.</p>
     *
     * @param axis input axis in the inclusive range {@code [-rank, rank - 1]}; negative values
     *     count from the final axis
     * @param keepDimensions {@code true} to retain the selected axis with extent one, or
     *     {@code false} to remove it
     * @return a non-null fresh storage-free tensor with the requested reduction shape, unchanged
     *     floating data type and gradient eligibility, and unresolved layout
     * @throws IllegalArgumentException if this tensor's data type is not floating, with a message
     *     containing the rejected data type; this check precedes axis validation
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the input rank, including
     *     every axis for a scalar input
     * @throws IllegalStateException if tensor identifier space is exhausted after local immutable
     *     expression metadata has been constructed
     */
    public Tensor mean(int axis, boolean keepDimensions) {
        return TensorReductionExpressions.applyAxis(
                this, AggregateReductionKind.MEAN, axis, keepDimensions);
    }

    /**
     * Builds a masked arithmetic-mean expression over one axis and removes that axis.
     *
     * <p>This tensor must be floating and {@code mask} must have exact {@link DataType#BOOL} type.
     * Ordinary right-aligned broadcasting of the mask Shape must produce exactly this tensor's Shape. A
     * compatible broadcast that would add a leading axis or enlarge this input is rejected. For
     * input {@code [batch, time, features]} reduced on axis one, a mask
     * {@code [batch, time]} does not implicitly address the first two axes: the caller first uses
     * {@code mask.expandDims(2)} to produce {@code [batch, time, 1]}. The transformed Tensor is
     * then the exact second producer input.</p>
     *
     * <p>The result removes the normalized axis and retains every unaffected Dimension reference.
     * It is a fresh, unlabeled, storage-free tensor with this input's exact data type and gradient
     * eligibility, unresolved layout, {@link AggregateReductionKind#MEAN}, masked attributes, and
     * ordered provenance {@code [this, mask]}. False positions are excluded; each output divides
     * by its selected true-count. A zero selected-count produces NaN in the existing floating
     * result type, without a promised payload or bit pattern. Static zero-sized reduction axes
     * and runtime zero-sized or all-false dynamic slices follow the same NaN contract. False mask
     * positions exclude even NaN and infinity before aggregation.</p>
     *
     * <p>No values, counts, or storage are read, and neither input is mutated. This method does not
     * materialize alignment, divide values, define a gradient rule, capture a graph, or provide
     * compiler, runtime, backend, or execution behavior.</p>
     *
     * @param axis input axis in the inclusive range {@code [-rank, rank - 1]}; negative values
     *     count from the final axis
     * @param mask non-null BOOL tensor whose Shape must use ordinary right-aligned broadcasting to
     *     produce exactly this tensor's Shape; ownership remains with the caller and the exact
     *     reference is retained only in immutable provenance
     * @return a non-null fresh storage-free tensor whose selected axis is removed, with unchanged
     *     floating data type and gradient eligibility, unresolved layout, and exact two-input
     *     provenance
     * @throws NullPointerException if {@code mask} is null, with message {@code mask}; no Tensor
     *     identity is consumed
     * @throws IllegalArgumentException if this tensor is not floating, the mask is not BOOL, the
     *     Shapes contain an incompatible aligned pair, or their broadcast does not equal this
     *     tensor's Shape; these failures consume no Tensor identity
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the input rank, including
     *     every axis for a scalar input; type checks precede axis validation
     * @throws IllegalStateException if tensor identifier space is exhausted after local immutable
     *     expression metadata has been constructed
     */
    public Tensor mean(int axis, Tensor mask) {
        return TensorMaskedReductionExpressions.apply(
                this, mask, AggregateReductionKind.MEAN, axis);
    }

    /**
     * Builds an expression that multiplies this tensor's values over every axis.
     *
     * <p>The input must have a floating or signed-integral data type. The fresh result has
     * canonical rank-zero scalar shape, preserves the exact input data type and
     * gradient-eligibility request, leaves layout unresolved, and has no label or storage.
     * Integral input is necessarily non-differentiable. Provenance contains
     * {@link AggregateReductionKind#PROD}, {@code NoOperationAttrs.INSTANCE}, and exactly this
     * tensor as the sole ordered producer input at output index zero.</p>
     *
     * <p>For INT32 and INT64 input, multiplication occurs in the exact result type modulo
     * {@code 2^32} or {@code 2^64}; reassociation is permitted, and an empty domain produces one.
     * Scalar, static, zero-extent, and dynamic Shapes are accepted structurally. Floating product
     * propagates NaN, makes zero times infinity NaN, follows sign parity for zero/infinity, and
     * returns positive one for empty. This method reads no value or storage, implements no algorithm,
     * creates no gradient rule, and provides no compiler, backend, runtime, or execution behavior.</p>
     *
     * @return a non-null fresh storage-free scalar tensor with unchanged numeric data type and
     *     gradient eligibility, unresolved layout, and exact one-input provenance
     * @throws IllegalArgumentException if this tensor's data type is BOOL, with message
     *     {@code input must have a numeric data type for PROD, but was BOOL}; no Tensor identity
     *     is consumed
     * @throws IllegalStateException if tensor identifier space is exhausted after local immutable
     *     expression metadata has been constructed
     */
    public Tensor prod() {
        return TensorReductionExpressions.applyFull(this, AggregateReductionKind.PROD);
    }

    /**
     * Builds a product expression over one axis and removes that axis.
     *
     * <p>The numeric input's positive or negative {@code axis} is normalized exactly once. The
     * selected dimension is removed, every unaffected dimension reference is retained in order,
     * and rank one reduces to the canonical scalar shape. The fresh result has exact input type
     * and gradient eligibility, unresolved layout, no label or storage, and provenance with
     * {@link AggregateReductionKind#PROD}, normalized axis attributes, and this input. Preserved
     * eligibility does not imply that a product gradient rule exists.</p>
     *
     * <p>For integral input, the exact result type uses modular multiplication with reassociation
     * permitted, and every empty selected-axis slice produces one. Other zero output axes may
     * make the result itself empty. Dynamic extents are accepted and use the same identity when
     * later bound to zero. Floating product follows the documented NaN, zero-times-infinity,
     * sign-parity, and positive-one empty-slice policy. This method reads no value or
     * storage and provides no algorithm, gradient rule, compiler lowering, backend, or execution
     * behavior.</p>
     *
     * @param axis input axis in the inclusive range {@code [-rank, rank - 1]}; negative values
     *     count from the final axis
     * @return a non-null fresh storage-free tensor whose selected axis is removed, with unchanged
     *     numeric data type and gradient eligibility and unresolved layout
     * @throws IllegalArgumentException if this tensor's data type is BOOL, with message
     *     {@code input must have a numeric data type for PROD, but was BOOL}; this check precedes
     *     axis validation
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the input rank, including
     *     every axis for a scalar input
     * @throws IllegalStateException if tensor identifier space is exhausted after local immutable
     *     expression metadata has been constructed
     */
    public Tensor prod(int axis) {
        return TensorReductionExpressions.applyAxis(
                this, AggregateReductionKind.PROD, axis, false);
    }

    /**
     * Builds a product expression over one axis and optionally retains it.
     *
     * <p>The numeric input's {@code axis} is normalized exactly once. A false
     * {@code keepDimensions} removes the selected axis; true replaces it by a new static extent
     * one while preserving all other dimension references. The fresh result retains exact input
     * type and gradient eligibility, has unresolved layout and no label or storage, and records
     * {@link AggregateReductionKind#PROD}, normalized axis attributes, and this sole input.
     * Preserved eligibility is model metadata and does not install a product gradient rule.</p>
     *
     * <p>For integral input, the exact result type uses modular multiplication with reassociation
     * permitted, and every empty selected-axis slice produces one independent of retention. Zero
     * and dynamic extents remain structurally valid; a dynamic selected extent later bound to
     * zero uses the same identity. Floating product follows the documented NaN,
     * zero-times-infinity, sign-parity, and positive-one empty-slice policy. This method reads no
     * value or storage and provides no algorithm, gradient rule, compiler lowering, backend,
     * runtime, or execution behavior.</p>
     *
     * @param axis input axis in the inclusive range {@code [-rank, rank - 1]}; negative values
     *     count from the final axis
     * @param keepDimensions {@code true} to retain the selected axis with extent one, or
     *     {@code false} to remove it
     * @return a non-null fresh storage-free tensor with the requested reduction shape, unchanged
     *     numeric data type and gradient eligibility, and unresolved layout
     * @throws IllegalArgumentException if this tensor's data type is BOOL, with message
     *     {@code input must have a numeric data type for PROD, but was BOOL}; this check precedes
     *     axis validation
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the input rank, including
     *     every axis for a scalar input
     * @throws IllegalStateException if tensor identifier space is exhausted after local immutable
     *     expression metadata has been constructed
     */
    public Tensor prod(int axis, boolean keepDimensions) {
        return TensorReductionExpressions.applyAxis(
                this, AggregateReductionKind.PROD, axis, keepDimensions);
    }

    /**
     * Builds an expression that selects the minimum over every input axis.
     *
     * <p>The input must have a floating or signed-integral data type. The fresh result has the
     * canonical rank-zero scalar shape, preserves the exact input data type and
     * gradient-eligibility request, leaves layout unresolved, and has no label or host storage.
     * Integral input is necessarily non-differentiable. Provenance contains aggregate
     * {@link AggregateReductionKind#MIN}, {@code NoOperationAttrs.INSTANCE}, and exactly this
     * tensor as the sole ordered producer input at output index zero. This aggregate operation is
     * distinct from the two-input elementwise {@link BinaryArithmeticKind#MIN} operation.</p>
     *
     * <p>Integral values use signed order, and an empty INT32 or INT64 domain produces
     * {@link Integer#MAX_VALUE} or {@link Long#MAX_VALUE}. Scalar, static, zero-extent, and dynamic
     * Shapes are accepted structurally. Floating minimum propagates NaN, orders infinities,
     * selects negative zero, and returns positive infinity for empty. This method does not
     * inspect or compare values, implement an algorithm, create an extrema gradient rule, capture
     * a graph, lower an operation, or execute work.</p>
     *
     * @return a non-null fresh storage-free scalar tensor with unchanged numeric data type and
     *     gradient eligibility, unresolved layout, and exact one-input provenance
     * @throws IllegalArgumentException if this tensor's data type is BOOL, with message
     *     {@code input must have a numeric data type for MIN, but was BOOL}; no Tensor identity is
     *     consumed
     * @throws IllegalStateException if tensor identifier space is exhausted after local immutable
     *     expression metadata has been constructed
     */
    public Tensor min() {
        return TensorReductionExpressions.applyFull(this, AggregateReductionKind.MIN);
    }

    /**
     * Builds a minimum expression over one axis and removes that axis.
     *
     * <p>The numeric input's positive or negative {@code axis} is normalized exactly once. The
     * selected dimension is removed, every unaffected immutable dimension reference is retained
     * in order, and rank one reduces to the canonical rank-zero scalar shape. The fresh result
     * preserves the exact input data type and gradient eligibility, leaves layout unresolved, has
     * no label or storage, and records aggregate {@link AggregateReductionKind#MIN}, normalized
     * single-axis attributes, and exactly this input.</p>
     *
     * <p>Integral values use signed order, and every empty selected-axis slice produces the
     * input type's maximum value. Other zero output axes may make the result itself empty.
     * Dynamic selected extents use the same identity when later bound to zero. Floating minimum
     * uses the documented NaN, infinity, negative-zero, and positive-infinity empty policy. This
     * method reads no value or storage and provides no algorithm,
     * gradient rule, compiler lowering, backend, or execution behavior. It is distinct from
     * {@link #minimum(Tensor)}.</p>
     *
     * @param axis input axis in the inclusive range {@code [-rank, rank - 1]}; negative values
     *     count from the final axis
     * @return a non-null fresh storage-free tensor whose selected axis is removed, with unchanged
     *     numeric data type and gradient eligibility and unresolved layout
     * @throws IllegalArgumentException if this tensor's data type is BOOL, with message
     *     {@code input must have a numeric data type for MIN, but was BOOL}; this check precedes
     *     axis validation
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the input rank, including
     *     every axis for a scalar input
     * @throws IllegalStateException if tensor identifier space is exhausted after local immutable
     *     expression metadata has been constructed
     */
    public Tensor min(int axis) {
        return TensorReductionExpressions.applyAxis(
                this, AggregateReductionKind.MIN, axis, false);
    }

    /**
     * Builds a minimum expression over one axis and optionally retains it.
     *
     * <p>The numeric input's {@code axis} is normalized exactly once. A false
     * {@code keepDimensions} removes the selected axis; true replaces it with a new static extent
     * one while preserving every other immutable dimension reference. The fresh result retains
     * the exact input data type and gradient eligibility, has unresolved layout and no label or
     * storage, and records aggregate {@link AggregateReductionKind#MIN}, normalized axis
     * attributes, and this sole input.</p>
     *
     * <p>Integral values use signed order, and every empty selected-axis slice produces the input
     * type's maximum value independent of retention. Zero and dynamic extents remain structurally
     * valid; a dynamic selected extent later bound to zero uses the same identity. Floating
     * minimum uses the documented NaN, infinity, negative-zero, and positive-infinity empty
     * policy. This method reads no value or storage and provides no algorithm,
     * gradient rule, compiler lowering, backend, runtime, or execution behavior.</p>
     *
     * @param axis input axis in the inclusive range {@code [-rank, rank - 1]}; negative values
     *     count from the final axis
     * @param keepDimensions {@code true} to retain the selected axis with extent one, or
     *     {@code false} to remove it
     * @return a non-null fresh storage-free tensor with the requested reduction shape, unchanged
     *     numeric data type and gradient eligibility, and unresolved layout
     * @throws IllegalArgumentException if this tensor's data type is BOOL, with message
     *     {@code input must have a numeric data type for MIN, but was BOOL}; this check precedes
     *     axis validation
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the input rank, including
     *     every axis for a scalar input
     * @throws IllegalStateException if tensor identifier space is exhausted after local immutable
     *     expression metadata has been constructed
     */
    public Tensor min(int axis, boolean keepDimensions) {
        return TensorReductionExpressions.applyAxis(
                this, AggregateReductionKind.MIN, axis, keepDimensions);
    }

    /**
     * Builds an expression that selects the maximum over every input axis.
     *
     * <p>The input must have a floating or signed-integral data type. The fresh result has the
     * canonical rank-zero scalar shape, preserves the exact input data type and
     * gradient-eligibility request, leaves layout unresolved, and has no label or host storage.
     * Integral input is necessarily non-differentiable. Provenance contains aggregate
     * {@link AggregateReductionKind#MAX}, {@code NoOperationAttrs.INSTANCE}, and exactly this
     * tensor as the sole ordered producer input at output index zero. This aggregate operation is
     * distinct from the two-input elementwise {@link BinaryArithmeticKind#MAX} operation.</p>
     *
     * <p>Integral values use signed order, and an empty INT32 or INT64 domain produces
     * {@link Integer#MIN_VALUE} or {@link Long#MIN_VALUE}. Scalar, static, zero-extent, and dynamic
     * Shapes are accepted structurally. Floating maximum propagates NaN, orders infinities,
     * selects positive zero, and returns negative infinity for empty. This method does not
     * inspect or compare values, implement an algorithm, create an extrema gradient rule, capture
     * a graph, lower an operation, or execute work.</p>
     *
     * @return a non-null fresh storage-free scalar tensor with unchanged numeric data type and
     *     gradient eligibility, unresolved layout, and exact one-input provenance
     * @throws IllegalArgumentException if this tensor's data type is BOOL, with message
     *     {@code input must have a numeric data type for MAX, but was BOOL}; no Tensor identity is
     *     consumed
     * @throws IllegalStateException if tensor identifier space is exhausted after local immutable
     *     expression metadata has been constructed
     */
    public Tensor max() {
        return TensorReductionExpressions.applyFull(this, AggregateReductionKind.MAX);
    }

    /**
     * Builds a maximum expression over one axis and removes that axis.
     *
     * <p>The numeric input's positive or negative {@code axis} is normalized exactly once. The
     * selected dimension is removed, every unaffected immutable dimension reference is retained
     * in order, and rank one reduces to the canonical rank-zero scalar shape. The fresh result
     * preserves the exact input data type and gradient eligibility, leaves layout unresolved, has
     * no label or storage, and records aggregate {@link AggregateReductionKind#MAX}, normalized
     * single-axis attributes, and exactly this input.</p>
     *
     * <p>Integral values use signed order, and every empty selected-axis slice produces the input
     * type's minimum value. Other zero output axes may make the result itself empty. Dynamic
     * selected extents use the same identity when later bound to zero. Floating maximum uses the
     * documented NaN, infinity, positive-zero, and negative-infinity empty policy. This method
     * reads no value or storage and provides no algorithm, gradient rule,
     * compiler lowering, backend, or execution behavior. It is distinct from
     * {@link #maximum(Tensor)}.</p>
     *
     * @param axis input axis in the inclusive range {@code [-rank, rank - 1]}; negative values
     *     count from the final axis
     * @return a non-null fresh storage-free tensor whose selected axis is removed, with unchanged
     *     numeric data type and gradient eligibility and unresolved layout
     * @throws IllegalArgumentException if this tensor's data type is BOOL, with message
     *     {@code input must have a numeric data type for MAX, but was BOOL}; this check precedes
     *     axis validation
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the input rank, including
     *     every axis for a scalar input
     * @throws IllegalStateException if tensor identifier space is exhausted after local immutable
     *     expression metadata has been constructed
     */
    public Tensor max(int axis) {
        return TensorReductionExpressions.applyAxis(
                this, AggregateReductionKind.MAX, axis, false);
    }

    /**
     * Builds a maximum expression over one axis and optionally retains it.
     *
     * <p>The numeric input's {@code axis} is normalized exactly once. A false
     * {@code keepDimensions} removes the selected axis; true replaces it with a new static extent
     * one while preserving every other immutable dimension reference. The fresh result retains
     * the exact input data type and gradient eligibility, has unresolved layout and no label or
     * storage, and records aggregate {@link AggregateReductionKind#MAX}, normalized axis
     * attributes, and this sole input.</p>
     *
     * <p>Integral values use signed order, and every empty selected-axis slice produces the input
     * type's minimum value independent of retention. Zero and dynamic extents remain structurally
     * valid; a dynamic selected extent later bound to zero uses the same identity. Floating
     * maximum uses the documented NaN, infinity, positive-zero, and negative-infinity empty
     * policy. This method reads no value or storage and provides no algorithm,
     * gradient rule, compiler lowering, backend, runtime, or execution behavior.</p>
     *
     * @param axis input axis in the inclusive range {@code [-rank, rank - 1]}; negative values
     *     count from the final axis
     * @param keepDimensions {@code true} to retain the selected axis with extent one, or
     *     {@code false} to remove it
     * @return a non-null fresh storage-free tensor with the requested reduction shape, unchanged
     *     numeric data type and gradient eligibility, and unresolved layout
     * @throws IllegalArgumentException if this tensor's data type is BOOL, with message
     *     {@code input must have a numeric data type for MAX, but was BOOL}; this check precedes
     *     axis validation
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the input rank, including
     *     every axis for a scalar input
     * @throws IllegalStateException if tensor identifier space is exhausted after local immutable
     *     expression metadata has been constructed
     */
    public Tensor max(int axis, boolean keepDimensions) {
        return TensorReductionExpressions.applyAxis(
                this, AggregateReductionKind.MAX, axis, keepDimensions);
    }

    /**
     * Builds a boolean conjunction reduction over every input axis.
     *
     * <p>This tensor must have exact {@link DataType#BOOL} type. The fresh result is a canonical
     * rank-zero BOOL scalar with false gradient eligibility, unresolved layout, no label or host
     * storage, and one-input provenance containing {@link AggregateReductionKind#ALL} with
     * {@code NoOperationAttrs.INSTANCE}. Scalar, static, zero-extent, and dynamic shapes are
     * accepted structurally.</p>
     *
     * <p>An empty domain produces true. This method does not inspect truth values or storage,
     * create a gradient rule, capture a graph, report backend support, or execute work. Aggregate
     * ALL is distinct from the two-input elementwise {@link BooleanLogicalKind#AND} operation.</p>
     *
     * @return a non-null fresh storage-free BOOL scalar with false gradient eligibility,
     *     unresolved layout, and exact one-input provenance
     * @throws IllegalArgumentException if this tensor is not BOOL, with a message containing
     *     {@code ALL} and the rejected type; no Tensor identity is consumed
     * @throws IllegalStateException if tensor identifier space is exhausted after local immutable
     *     expression metadata has been constructed
     */
    public Tensor all() {
        return TensorReductionExpressions.applyFull(this, AggregateReductionKind.ALL);
    }

    /**
     * Builds a boolean conjunction reduction over one axis and removes that axis.
     *
     * <p>The BOOL input's positive or negative {@code axis} is normalized exactly once. The
     * selected dimension is removed, unaffected dimension references are retained, and rank one
     * produces the canonical scalar shape. The fresh BOOL result is non-differentiable, has
     * unresolved layout and no label or storage, and records normalized axis attributes and this
     * sole provenance input.</p>
     *
     * <p>An empty selected-axis slice produces true. This method does not inspect truth values,
     * create a gradient rule, capture a graph, report backend support, or execute work. Aggregate ALL is
     * distinct from the two-input elementwise {@link BooleanLogicalKind#AND} operation.</p>
     *
     * @param axis input axis in {@code [-rank, rank - 1]}; negative values count from the end
     * @return a non-null fresh storage-free BOOL tensor whose selected axis is removed, with false
     *     gradient eligibility, unresolved layout, and exact one-input provenance
     * @throws IllegalArgumentException if this tensor is not BOOL, with a message containing
     *     {@code ALL} and the rejected type; this check precedes axis validation
     * @throws IndexOutOfBoundsException if {@code axis} is invalid, including every scalar axis
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor all(int axis) {
        return TensorReductionExpressions.applyAxis(
                this, AggregateReductionKind.ALL, axis, false);
    }

    /**
     * Builds a boolean conjunction reduction over one axis and optionally retains it.
     *
     * <p>The BOOL input's positive or negative {@code axis} is normalized exactly once. A false
     * {@code keepDimensions} removes it; true replaces it with a new static extent one while
     * retaining every unaffected dimension reference. The fresh result is exact BOOL with false
     * gradient eligibility, unresolved layout, no label or storage, normalized axis attributes,
     * and exact one-input aggregate provenance. Zero and dynamic extents are accepted
     * structurally.</p>
     *
     * <p>An empty selected-axis slice produces true. This method does not inspect truth values,
     * create a gradient rule, capture a graph, report backend support, or execute work. Aggregate ALL is
     * distinct from the two-input elementwise {@link BooleanLogicalKind#AND} operation.</p>
     *
     * @param axis input axis in {@code [-rank, rank - 1]}; negative values count from the end
     * @param keepDimensions {@code true} to retain the selected axis with extent one;
     *     {@code false} to remove it
     * @return a non-null fresh storage-free BOOL tensor with the requested reduction shape, false
     *     gradient eligibility, unresolved layout, and exact one-input provenance
     * @throws IllegalArgumentException if this tensor is not BOOL, with a message containing
     *     {@code ALL} and the rejected type; this check precedes axis validation
     * @throws IndexOutOfBoundsException if {@code axis} is invalid, including every scalar axis
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor all(int axis, boolean keepDimensions) {
        return TensorReductionExpressions.applyAxis(
                this, AggregateReductionKind.ALL, axis, keepDimensions);
    }

    /**
     * Builds a boolean disjunction reduction over every input axis.
     *
     * <p>This tensor must have exact {@link DataType#BOOL} type. The fresh result is a canonical
     * rank-zero BOOL scalar with false gradient eligibility, unresolved layout, no label or host
     * storage, and one-input provenance containing {@link AggregateReductionKind#ANY} with
     * {@code NoOperationAttrs.INSTANCE}. Scalar, static, zero-extent, and dynamic shapes are
     * accepted structurally.</p>
     *
     * <p>An empty domain produces false. This method does not inspect truth values or storage,
     * create a gradient rule, capture a graph, report backend support, or execute work. Aggregate
     * ANY is distinct from the two-input elementwise {@link BooleanLogicalKind#OR} operation.</p>
     *
     * @return a non-null fresh storage-free BOOL scalar with false gradient eligibility,
     *     unresolved layout, and exact one-input provenance
     * @throws IllegalArgumentException if this tensor is not BOOL, with a message containing
     *     {@code ANY} and the rejected type; no Tensor identity is consumed
     * @throws IllegalStateException if tensor identifier space is exhausted after local immutable
     *     expression metadata has been constructed
     */
    public Tensor any() {
        return TensorReductionExpressions.applyFull(this, AggregateReductionKind.ANY);
    }

    /**
     * Builds a boolean disjunction reduction over one axis and removes that axis.
     *
     * <p>The BOOL input's axis is normalized exactly once. The selected dimension is removed,
     * unaffected references are retained, and rank one becomes a scalar. The fresh result is
     * non-differentiable BOOL with unresolved layout, no label or storage, normalized axis
     * attributes, and this sole provenance input.</p>
     *
     * <p>An empty selected-axis slice produces false. This method does not inspect truth values,
     * create a gradient rule, capture a graph, report backend support, or execute work. Aggregate ANY is
     * distinct from the two-input elementwise {@link BooleanLogicalKind#OR} operation.</p>
     *
     * @param axis input axis in {@code [-rank, rank - 1]}; negative values count from the end
     * @return a non-null fresh storage-free BOOL tensor whose selected axis is removed, with false
     *     gradient eligibility, unresolved layout, and exact one-input provenance
     * @throws IllegalArgumentException if this tensor is not BOOL, with a message containing
     *     {@code ANY} and the rejected type; this check precedes axis validation
     * @throws IndexOutOfBoundsException if {@code axis} is invalid, including every scalar axis
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor any(int axis) {
        return TensorReductionExpressions.applyAxis(
                this, AggregateReductionKind.ANY, axis, false);
    }

    /**
     * Builds a boolean disjunction reduction over one axis and optionally retains it.
     *
     * <p>The BOOL input's positive or negative {@code axis} is normalized exactly once. A false
     * {@code keepDimensions} removes it; true replaces it with a new static extent one while
     * retaining every unaffected dimension reference. The fresh result is exact BOOL with false
     * gradient eligibility, unresolved layout, no label or storage, normalized axis attributes,
     * and exact one-input aggregate provenance. Zero and dynamic extents are accepted
     * structurally.</p>
     *
     * <p>An empty selected-axis slice produces false. This method does not inspect truth values,
     * create a gradient rule, capture a graph, report backend support, or execute work. Aggregate ANY is
     * distinct from the two-input elementwise {@link BooleanLogicalKind#OR} operation.</p>
     *
     * @param axis input axis in {@code [-rank, rank - 1]}; negative values count from the end
     * @param keepDimensions {@code true} to retain the selected axis with extent one;
     *     {@code false} to remove it
     * @return a non-null fresh storage-free BOOL tensor with the requested reduction shape, false
     *     gradient eligibility, unresolved layout, and exact one-input provenance
     * @throws IllegalArgumentException if this tensor is not BOOL, with a message containing
     *     {@code ANY} and the rejected type; this check precedes axis validation
     * @throws IndexOutOfBoundsException if {@code axis} is invalid, including every scalar axis
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor any(int axis, boolean keepDimensions) {
        return TensorReductionExpressions.applyAxis(
                this, AggregateReductionKind.ANY, axis, keepDimensions);
    }

    /**
     * Builds a sum over caller-ordered distinct axes and removes those axes.
     *
     * <p>Axes are normalized once in caller order; normalized duplicates are rejected. An empty
     * array selects a point domain and returns the point value, unlike {@link #sum()} full
     * reduction. Floating sum uses the documented NaN, infinity, signed-zero, positive-zero empty,
     * and result-format rounding policy; integral sum retains exact type and modular semantics.</p>
     *
     * @param axes non-null caller-owned positive or negative axes; may be empty and is not retained
     * @return a non-null fresh unlabeled, storage-free result with selected axes removed, exact
     *     type/eligibility, unresolved layout, and one-input output-index-zero provenance
     * @throws NullPointerException if {@code axes} is null, with message {@code axes}
     * @throws IllegalArgumentException if this Tensor is BOOL or normalized axes repeat
     * @throws IndexOutOfBoundsException if an axis is invalid for this Tensor's Shape
     * @throws IllegalStateException if tensor identifier space is exhausted after local validation
     */
    public Tensor sum(int... axes) {
        return TensorMultiAxisReductionExpressions.applyOrdinary(
                this, AggregateReductionKind.SUM, axes, false);
    }

    /**
     * Builds a sum over caller-ordered distinct axes with explicit dimension retention.
     *
     * <p>This has the same type, numerical, normalization, freshness, and provenance contract as
     * {@link #sum(int...)}, but retained selected positions become new extent-one Dimensions.
     * Unselected Dimensions retain exact references. An empty list leaves Shape unchanged while
     * {@code keepDimensions} remains part of attributes identity.</p>
     *
     * @param axes non-null caller-owned positive or negative axes; may be empty and is not retained
     * @param keepDimensions whether selected axes remain with extent one
     * @return a non-null fresh storage-free sum expression with the requested Shape and exact
     *     input type/eligibility
     * @throws NullPointerException if {@code axes} is null
     * @throws IllegalArgumentException if this Tensor is BOOL or normalized axes repeat
     * @throws IndexOutOfBoundsException if an axis is invalid
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor sum(int[] axes, boolean keepDimensions) {
        return TensorMultiAxisReductionExpressions.applyOrdinary(
                this, AggregateReductionKind.SUM, axes, keepDimensions);
    }

    /**
     * Builds a floating arithmetic mean over ordered distinct axes and removes them.
     *
     * <p>The result preserves exact floating type/eligibility. Mean is exact sum divided by count:
     * NaN and opposite infinities produce NaN, a sole infinity sign is preserved, empty domains
     * produce NaN, and zero sign follows SUM. Empty axes select one point and return it.</p>
     *
     * @param axes non-null caller-owned positive or negative axes; may be empty and is not retained
     * @return a non-null fresh storage-free reduced-Shape mean with unresolved layout and exact
     *     one-input output-index-zero provenance
     * @throws NullPointerException if {@code axes} is null
     * @throws IllegalArgumentException if this Tensor is not floating or normalized axes repeat
     * @throws IndexOutOfBoundsException if an axis is invalid
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor mean(int... axes) {
        return TensorMultiAxisReductionExpressions.applyOrdinary(
                this, AggregateReductionKind.MEAN, axes, false);
    }

    /**
     * Builds a floating arithmetic mean with explicit selected-axis retention.
     *
     * <p>Numerical, axis-order, metadata, and failure behavior matches {@link #mean(int...)}.
     * Selected positions are removed or replaced with new extent-one Dimensions; unselected
     * references are retained exactly.</p>
     *
     * @param axes non-null caller-owned positive or negative axes; may be empty and is not retained
     * @param keepDimensions whether selected axes remain with extent one
     * @return a non-null fresh storage-free mean with requested Shape and exact type/eligibility
     * @throws NullPointerException if {@code axes} is null
     * @throws IllegalArgumentException if input is not floating or normalized axes repeat
     * @throws IndexOutOfBoundsException if an axis is invalid
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor mean(int[] axes, boolean keepDimensions) {
        return TensorMultiAxisReductionExpressions.applyOrdinary(
                this, AggregateReductionKind.MEAN, axes, keepDimensions);
    }

    /**
     * Builds a product over ordered distinct axes and removes them.
     *
     * <p>Floating product propagates NaN, makes zero times infinity NaN, follows multiplication
     * parity for zero/infinity sign, and uses positive one for an empty domain. Integral product
     * retains exact type with fixed-width modular semantics. Empty axes return the point value.</p>
     *
     * @param axes non-null caller-owned positive or negative axes; may be empty and is not retained
     * @return a non-null fresh storage-free reduced-Shape product preserving type/eligibility
     * @throws NullPointerException if {@code axes} is null
     * @throws IllegalArgumentException if input is BOOL or normalized axes repeat
     * @throws IndexOutOfBoundsException if an axis is invalid
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor prod(int... axes) {
        return TensorMultiAxisReductionExpressions.applyOrdinary(
                this, AggregateReductionKind.PROD, axes, false);
    }

    /**
     * Builds a product with explicit selected-axis retention.
     *
     * <p>Numerical, ordered-axis, metadata, and failure behavior matches {@link #prod(int...)};
     * retained selected axes become extent one and unselected Dimension references remain exact.</p>
     *
     * @param axes non-null caller-owned positive or negative axes; may be empty and is not retained
     * @param keepDimensions whether selected axes remain with extent one
     * @return a non-null fresh storage-free product with requested Shape and exact metadata
     * @throws NullPointerException if {@code axes} is null
     * @throws IllegalArgumentException if input is BOOL or normalized axes repeat
     * @throws IndexOutOfBoundsException if an axis is invalid
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor prod(int[] axes, boolean keepDimensions) {
        return TensorMultiAxisReductionExpressions.applyOrdinary(
                this, AggregateReductionKind.PROD, axes, keepDimensions);
    }

    /**
     * Builds a minimum over ordered distinct axes and removes them.
     *
     * <p>Floating minimum propagates NaN, orders infinities normally, selects negative zero, and
     * returns positive infinity for an empty domain. Integral minimum uses signed order and its
     * bounded maximum empty identity. Empty axes return the point value.</p>
     *
     * @param axes non-null caller-owned positive or negative axes; may be empty and is not retained
     * @return a non-null fresh storage-free reduced-Shape minimum preserving exact metadata
     * @throws NullPointerException if {@code axes} is null
     * @throws IllegalArgumentException if input is BOOL or normalized axes repeat
     * @throws IndexOutOfBoundsException if an axis is invalid
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor min(int... axes) {
        return TensorMultiAxisReductionExpressions.applyOrdinary(
                this, AggregateReductionKind.MIN, axes, false);
    }

    /**
     * Builds a minimum with explicit selected-axis retention.
     *
     * <p>Numerical, ordered-axis, metadata, and failure behavior matches {@link #min(int...)};
     * selected positions are removed or replaced with extent one.</p>
     *
     * @param axes non-null caller-owned positive or negative axes; may be empty and is not retained
     * @param keepDimensions whether selected axes remain with extent one
     * @return a non-null fresh storage-free minimum with requested Shape and exact metadata
     * @throws NullPointerException if {@code axes} is null
     * @throws IllegalArgumentException if input is BOOL or normalized axes repeat
     * @throws IndexOutOfBoundsException if an axis is invalid
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor min(int[] axes, boolean keepDimensions) {
        return TensorMultiAxisReductionExpressions.applyOrdinary(
                this, AggregateReductionKind.MIN, axes, keepDimensions);
    }

    /**
     * Builds a maximum over ordered distinct axes and removes them.
     *
     * <p>Floating maximum propagates NaN, orders infinities normally, selects positive zero, and
     * returns negative infinity for an empty domain. Integral maximum uses signed order and its
     * bounded minimum empty identity. Empty axes return the point value.</p>
     *
     * @param axes non-null caller-owned positive or negative axes; may be empty and is not retained
     * @return a non-null fresh storage-free reduced-Shape maximum preserving exact metadata
     * @throws NullPointerException if {@code axes} is null
     * @throws IllegalArgumentException if input is BOOL or normalized axes repeat
     * @throws IndexOutOfBoundsException if an axis is invalid
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor max(int... axes) {
        return TensorMultiAxisReductionExpressions.applyOrdinary(
                this, AggregateReductionKind.MAX, axes, false);
    }

    /**
     * Builds a maximum with explicit selected-axis retention.
     *
     * <p>Numerical, ordered-axis, metadata, and failure behavior matches {@link #max(int...)};
     * selected positions are removed or replaced with extent one.</p>
     *
     * @param axes non-null caller-owned positive or negative axes; may be empty and is not retained
     * @param keepDimensions whether selected axes remain with extent one
     * @return a non-null fresh storage-free maximum with requested Shape and exact metadata
     * @throws NullPointerException if {@code axes} is null
     * @throws IllegalArgumentException if input is BOOL or normalized axes repeat
     * @throws IndexOutOfBoundsException if an axis is invalid
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor max(int[] axes, boolean keepDimensions) {
        return TensorMultiAxisReductionExpressions.applyOrdinary(
                this, AggregateReductionKind.MAX, axes, keepDimensions);
    }

    /**
     * Builds BOOL conjunction over ordered distinct axes and removes them.
     *
     * <p>Exact BOOL input/result is required. An empty reduction domain produces true; an empty
     * axis list returns each point value. The fresh result is non-differentiable, unresolved,
     * unlabeled, storage-free, and records exact one-input output-index-zero provenance.</p>
     *
     * @param axes non-null caller-owned positive or negative axes; may be empty and is not retained
     * @return a non-null fresh reduced-Shape BOOL conjunction expression
     * @throws NullPointerException if {@code axes} is null
     * @throws IllegalArgumentException if input is not BOOL or normalized axes repeat
     * @throws IndexOutOfBoundsException if an axis is invalid
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor all(int... axes) {
        return TensorMultiAxisReductionExpressions.applyOrdinary(
                this, AggregateReductionKind.ALL, axes, false);
    }

    /**
     * Builds BOOL conjunction with explicit selected-axis retention.
     *
     * <p>Truth, ordered-axis, metadata, and failure behavior matches {@link #all(int...)};
     * selected positions are removed or replaced with extent one.</p>
     *
     * @param axes non-null caller-owned axes; may be empty and is not retained
     * @param keepDimensions whether selected axes remain with extent one
     * @return a non-null fresh storage-free BOOL conjunction with requested Shape
     * @throws NullPointerException if {@code axes} is null
     * @throws IllegalArgumentException if input is not BOOL or normalized axes repeat
     * @throws IndexOutOfBoundsException if an axis is invalid
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor all(int[] axes, boolean keepDimensions) {
        return TensorMultiAxisReductionExpressions.applyOrdinary(
                this, AggregateReductionKind.ALL, axes, keepDimensions);
    }

    /**
     * Builds BOOL disjunction over ordered distinct axes and removes them.
     *
     * <p>Exact BOOL input/result is required. An empty reduction domain produces false; an empty
     * axis list returns each point value. Result metadata and provenance match {@link #all(int...)}.</p>
     *
     * @param axes non-null caller-owned positive or negative axes; may be empty and is not retained
     * @return a non-null fresh reduced-Shape BOOL disjunction expression
     * @throws NullPointerException if {@code axes} is null
     * @throws IllegalArgumentException if input is not BOOL or normalized axes repeat
     * @throws IndexOutOfBoundsException if an axis is invalid
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor any(int... axes) {
        return TensorMultiAxisReductionExpressions.applyOrdinary(
                this, AggregateReductionKind.ANY, axes, false);
    }

    /**
     * Builds BOOL disjunction with explicit selected-axis retention.
     *
     * <p>Truth, ordered-axis, metadata, and failure behavior matches {@link #any(int...)};
     * selected positions are removed or replaced with extent one.</p>
     *
     * @param axes non-null caller-owned axes; may be empty and is not retained
     * @param keepDimensions whether selected axes remain with extent one
     * @return a non-null fresh storage-free BOOL disjunction with requested Shape
     * @throws NullPointerException if {@code axes} is null
     * @throws IllegalArgumentException if input is not BOOL or normalized axes repeat
     * @throws IndexOutOfBoundsException if an axis is invalid
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor any(int[] axes, boolean keepDimensions) {
        return TensorMultiAxisReductionExpressions.applyOrdinary(
                this, AggregateReductionKind.ANY, axes, keepDimensions);
    }

    /**
     * Builds floating log-sum-exp over ordered distinct axes and removes them.
     *
     * <p>The first-class target is {@code log(sum(exp(x_i)))} without prescribing an algorithm.
     * Empty domains and all-negative-infinity domains produce negative infinity; NaN produces
     * NaN; positive infinity produces positive infinity unless NaN exists; and a point domain
     * returns its value, preserving signed zero. Exact floating type/eligibility are preserved.</p>
     *
     * @param axes non-null caller-owned positive or negative axes; may be empty and is not retained
     * @return a non-null fresh storage-free reduced-Shape log-sum-exp with unresolved layout and
     *     exact one-input output-index-zero provenance
     * @throws NullPointerException if {@code axes} is null
     * @throws IllegalArgumentException if input is not floating or normalized axes repeat
     * @throws IndexOutOfBoundsException if an axis is invalid
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor logSumExp(int... axes) {
        return TensorMultiAxisReductionExpressions.applyAdvanced(
                this, AggregateReductionKind.LOG_SUM_EXP, axes, false);
    }

    /**
     * Builds floating log-sum-exp with explicit selected-axis retention.
     *
     * <p>Numerical, ordered-axis, metadata, and failure behavior matches
     * {@link #logSumExp(int...)}; selected positions are removed or replaced with extent one.</p>
     *
     * @param axes non-null caller-owned axes; may be empty and is not retained
     * @param keepDimensions whether selected axes remain with extent one
     * @return a non-null fresh storage-free log-sum-exp with requested Shape and exact metadata
     * @throws NullPointerException if {@code axes} is null
     * @throws IllegalArgumentException if input is not floating or normalized axes repeat
     * @throws IndexOutOfBoundsException if an axis is invalid
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor logSumExp(int[] axes, boolean keepDimensions) {
        return TensorMultiAxisReductionExpressions.applyAdvanced(
                this, AggregateReductionKind.LOG_SUM_EXP, axes, keepDimensions);
    }

    /**
     * Builds population variance over ordered distinct axes and removes them.
     *
     * <p>This is exactly {@link #variance(int[], boolean, long)} with dimensions removed and
     * correction zero. A point domain, including empty axes, produces positive zero. Exact
     * floating type/eligibility are preserved.</p>
     *
     * @param axes non-null caller-owned positive or negative axes; may be empty and is not retained
     * @return a non-null fresh storage-free population-variance expression
     * @throws NullPointerException if {@code axes} is null
     * @throws IllegalArgumentException if input is not floating or normalized axes repeat
     * @throws IndexOutOfBoundsException if an axis is invalid
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor variance(int... axes) {
        return TensorMultiAxisReductionExpressions.applyStatistical(
                this, AggregateReductionKind.VARIANCE, axes, false, 0);
    }

    /**
     * Builds population variance with explicit selected-axis retention and correction zero.
     *
     * @param axes non-null caller-owned axes; may be empty and is not retained
     * @param keepDimensions whether selected axes remain with extent one
     * @return a non-null fresh storage-free population variance with requested Shape
     * @throws NullPointerException if {@code axes} is null
     * @throws IllegalArgumentException if input is not floating or normalized axes repeat
     * @throws IndexOutOfBoundsException if an axis is invalid
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor variance(int[] axes, boolean keepDimensions) {
        return TensorMultiAxisReductionExpressions.applyStatistical(
                this, AggregateReductionKind.VARIANCE, axes, keepDimensions, 0);
    }

    /**
     * Builds corrected variance over ordered distinct axes.
     *
     * <p>For count {@code N}, correction {@code c}, and exact mean {@code mu}, the target is
     * {@code sum((x_i-mu)^2)/(N-c)} and requires {@code N > c}. A statically invalid denominator
     * fails locally; dynamic proof is deferred. NaN or infinity produces NaN; a valid constant
     * finite domain produces positive zero. Construction preserves exact floating type and
     * eligibility, derives Shape, and records one fresh storage-free one-input occurrence without
     * evaluating values or choosing an algorithm.</p>
     *
     * @param axes non-null caller-owned positive or negative axes; may be empty and is not retained
     * @param keepDimensions whether selected axes remain with extent one
     * @param correction non-negative value subtracted from selected-domain count
     * @return a non-null fresh corrected-variance expression with exact metadata and provenance
     * @throws NullPointerException if {@code axes} is null
     * @throws IllegalArgumentException if input is not floating, correction is negative,
     *     normalized axes repeat, or static count is at most correction
     * @throws IndexOutOfBoundsException if an axis is invalid
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor variance(int[] axes, boolean keepDimensions, long correction) {
        return TensorMultiAxisReductionExpressions.applyStatistical(
                this, AggregateReductionKind.VARIANCE, axes, keepDimensions, correction);
    }

    /**
     * Builds population standard deviation over ordered distinct axes and removes them.
     *
     * <p>This is {@link #standardDeviation(int[], boolean, long)} with dimensions removed and
     * correction zero. A point domain produces positive zero.</p>
     *
     * @param axes non-null caller-owned positive or negative axes; may be empty and is not retained
     * @return a non-null fresh storage-free population-standard-deviation expression
     * @throws NullPointerException if {@code axes} is null
     * @throws IllegalArgumentException if input is not floating or normalized axes repeat
     * @throws IndexOutOfBoundsException if an axis is invalid
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor standardDeviation(int... axes) {
        return TensorMultiAxisReductionExpressions.applyStatistical(
                this, AggregateReductionKind.STANDARD_DEVIATION, axes, false, 0);
    }

    /**
     * Builds population standard deviation with explicit axis retention and correction zero.
     *
     * @param axes non-null caller-owned axes; may be empty and is not retained
     * @param keepDimensions whether selected axes remain with extent one
     * @return a non-null fresh storage-free population standard deviation with requested Shape
     * @throws NullPointerException if {@code axes} is null
     * @throws IllegalArgumentException if input is not floating or normalized axes repeat
     * @throws IndexOutOfBoundsException if an axis is invalid
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor standardDeviation(int[] axes, boolean keepDimensions) {
        return TensorMultiAxisReductionExpressions.applyStatistical(
                this, AggregateReductionKind.STANDARD_DEVIATION, axes, keepDimensions, 0);
    }

    /**
     * Builds corrected standard deviation over ordered distinct axes.
     *
     * <p>The target is the non-negative principal square root of corrected variance and shares
     * its required {@code N > correction}, NaN/infinity, metadata, and deferred-execution policy.
     * A valid zero result is positive zero, never negative zero.</p>
     *
     * @param axes non-null caller-owned positive or negative axes; may be empty and is not retained
     * @param keepDimensions whether selected axes remain with extent one
     * @param correction non-negative value subtracted from selected-domain count
     * @return a non-null fresh corrected-standard-deviation expression with exact metadata
     * @throws NullPointerException if {@code axes} is null
     * @throws IllegalArgumentException if input is not floating, correction is negative,
     *     normalized axes repeat, or static count is at most correction
     * @throws IndexOutOfBoundsException if an axis is invalid
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor standardDeviation(int[] axes, boolean keepDimensions, long correction) {
        return TensorMultiAxisReductionExpressions.applyStatistical(
                this, AggregateReductionKind.STANDARD_DEVIATION, axes, keepDimensions, correction);
    }

    /**
     * Builds floating L1 norm over ordered distinct axes and removes them.
     *
     * <p>The target is {@code sum(abs(x_i))}. Empty domains produce positive zero, point domains
     * produce absolute value, NaN produces NaN, and infinity produces positive infinity unless
     * NaN exists. Finite results are non-negative.</p>
     *
     * @param axes non-null caller-owned positive or negative axes; may be empty and is not retained
     * @return a non-null fresh storage-free L1 norm preserving exact type/eligibility
     * @throws NullPointerException if {@code axes} is null
     * @throws IllegalArgumentException if input is not floating or normalized axes repeat
     * @throws IndexOutOfBoundsException if an axis is invalid
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor l1Norm(int... axes) {
        return TensorMultiAxisReductionExpressions.applyAdvanced(
                this, AggregateReductionKind.L1_NORM, axes, false);
    }

    /**
     * Builds floating L1 norm with explicit selected-axis retention.
     *
     * @param axes non-null caller-owned axes; may be empty and is not retained
     * @param keepDimensions whether selected axes remain with extent one
     * @return a non-null fresh storage-free L1 norm with requested Shape and exact metadata
     * @throws NullPointerException if {@code axes} is null
     * @throws IllegalArgumentException if input is not floating or normalized axes repeat
     * @throws IndexOutOfBoundsException if an axis is invalid
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor l1Norm(int[] axes, boolean keepDimensions) {
        return TensorMultiAxisReductionExpressions.applyAdvanced(
                this, AggregateReductionKind.L1_NORM, axes, keepDimensions);
    }

    /**
     * Builds floating L2 norm over ordered distinct axes and removes them.
     *
     * <p>The target is {@code sqrt(sum(x_i*x_i))}. It shares L1 norm's empty, point, NaN,
     * infinity, and positive-zero policy and identifies one first-class operation rather than a
     * stored decomposition.</p>
     *
     * @param axes non-null caller-owned positive or negative axes; may be empty and is not retained
     * @return a non-null fresh storage-free L2 norm preserving exact type/eligibility
     * @throws NullPointerException if {@code axes} is null
     * @throws IllegalArgumentException if input is not floating or normalized axes repeat
     * @throws IndexOutOfBoundsException if an axis is invalid
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor l2Norm(int... axes) {
        return TensorMultiAxisReductionExpressions.applyAdvanced(
                this, AggregateReductionKind.L2_NORM, axes, false);
    }

    /**
     * Builds floating L2 norm with explicit selected-axis retention.
     *
     * @param axes non-null caller-owned axes; may be empty and is not retained
     * @param keepDimensions whether selected axes remain with extent one
     * @return a non-null fresh storage-free L2 norm with requested Shape and exact metadata
     * @throws NullPointerException if {@code axes} is null
     * @throws IllegalArgumentException if input is not floating or normalized axes repeat
     * @throws IndexOutOfBoundsException if an axis is invalid
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor l2Norm(int[] axes, boolean keepDimensions) {
        return TensorMultiAxisReductionExpressions.applyAdvanced(
                this, AggregateReductionKind.L2_NORM, axes, keepDimensions);
    }

    /**
     * Builds an axis-removing arg-min expression using the first logical index for ties.
     *
     * <p>This tensor must have FLOAT64, FLOAT32, BFLOAT16, INT32, or INT64 type. The caller axis is
     * normalized once and removed; rank-one input produces the canonical scalar Shape, and every
     * unaffected Dimension reference is retained. This convenience delegates directly with
     * {@link ArgExtremaTiePolicy#FIRST_INDEX}, which requests the smallest logical coordinate
     * among equal minimum candidates.</p>
     *
     * <p>The fresh result has fixed INT64 type, false gradient eligibility, unresolved layout, no
     * label or host storage, and one-input {@link AggregateReductionKind#ARG_MIN} provenance at
     * output index zero. Integral candidates use signed order. Floating candidates prefer NaN to
     * non-NaN, treat multiple NaNs as ties, order negative zero below positive zero, and order
     * infinities normally. Construction records this meaning without reading values, selecting an
     * index, creating gradient rules, capturing a graph, lowering, or executing work.</p>
     *
     * @param axis input axis in {@code [-rank, rank - 1]}; negative values count from the final
     *     axis
     * @return a non-null fresh, unlabeled, storage-free INT64 arg-min expression whose selected
     *     axis is removed, with false gradient eligibility, unresolved layout, exact one-input
     *     provenance, and output index zero
     * @throws IllegalArgumentException if this tensor is BOOL, with message {@code input must have
     *     a numeric data type, but was BOOL}, or if the normalized selected extent is statically
     *     zero, with message {@code arg-extrema reduction axis must be non-empty, but axis
     *     <normalizedAxis> has static extent 0}; no Tensor identity is consumed
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for this tensor's Shape,
     *     including every axis for a scalar input; no Tensor identity is consumed
     * @throws IllegalStateException if tensor identifier space is exhausted after valid local
     *     metadata and provenance have been constructed
     */
    public Tensor argMin(int axis) {
        return TensorArgExtremaExpressions.apply(
                this, AggregateReductionKind.ARG_MIN, axis, false,
                ArgExtremaTiePolicy.FIRST_INDEX);
    }

    /**
     * Builds an arg-min expression using the first logical index for ties.
     *
     * <p>The numeric input axis is normalized once. A false {@code keepDimensions} removes it;
     * true replaces it with a new static extent one while retaining every unaffected Dimension
     * reference. This convenience delegates directly with {@link
     * ArgExtremaTiePolicy#FIRST_INDEX}. The selected static extent must be positive. An unselected
     * zero extent or dynamic/expression selected extent remains structurally valid.</p>
     *
     * <p>The fresh result is unlabeled and storage-free, with fixed INT64 type, false gradient
     * eligibility, unresolved layout, shared arg-extrema attributes, exact
     * {@link AggregateReductionKind#ARG_MIN} one-input provenance, and output index zero. Floating
     * and integral ordering is the same as {@link #argMin(int)}. Construction performs no value
     * selection, gradient, compiler, backend, runtime, or execution work.</p>
     *
     * @param axis input axis in {@code [-rank, rank - 1]}; negative values count from the final
     *     axis
     * @param keepDimensions {@code true} to retain the selected axis with extent one, or
     *     {@code false} to remove it
     * @return a non-null fresh, unlabeled, storage-free INT64 arg-min expression with the requested
     *     Shape, false gradient eligibility, unresolved layout, and exact index-zero provenance
     * @throws IllegalArgumentException if this tensor is BOOL, with message {@code input must have
     *     a numeric data type, but was BOOL}, or if the normalized selected extent is statically
     *     zero, with the exact arg-extrema empty-axis message; no Tensor identity is consumed
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for this tensor's Shape,
     *     including every scalar axis; no Tensor identity is consumed
     * @throws IllegalStateException if tensor identifier space is exhausted after valid local
     *     metadata and provenance have been constructed
     */
    public Tensor argMin(int axis, boolean keepDimensions) {
        return TensorArgExtremaExpressions.apply(
                this, AggregateReductionKind.ARG_MIN, axis, keepDimensions,
                ArgExtremaTiePolicy.FIRST_INDEX);
    }

    /**
     * Builds an arg-min expression with an explicit logical-index tie policy.
     *
     * <p>The non-null policy is checked before input eligibility and axis validation and retained
     * by exact enum reference in shared attributes. {@code FIRST_INDEX} requests the smallest
     * logical coordinate and {@code LAST_INDEX} the largest among equal minimum candidates,
     * independent of storage offset, stride, layout, or traversal. Integral candidates use signed
     * order. Floating candidates prefer NaN, treat multiple NaNs as ties, order negative zero
     * below positive zero, and order infinities normally.</p>
     *
     * <p>The axis is normalized once and removed or replaced with a new extent one. A statically
     * empty selected extent is invalid, while an unselected zero extent or unbound selected extent
     * is accepted. The fresh result has fixed INT64/false-gradient metadata, unresolved layout,
     * no label or storage, exact one-input ARG_MIN provenance, and output index zero. Construction
     * neither compares values nor supplies gradient, compiler, backend, runtime, or execution
     * behavior.</p>
     *
     * @param axis input axis in {@code [-rank, rank - 1]}; negative values count from the final
     *     axis
     * @param keepDimensions {@code true} to retain the selected axis with extent one, or
     *     {@code false} to remove it
     * @param tiePolicy non-null explicit first- or last-logical-index policy retained by exact
     *     enum reference
     * @return a non-null fresh, unlabeled, storage-free INT64 arg-min expression with the requested
     *     Shape, false gradient eligibility, unresolved layout, exact attributes and ordered
     *     provenance, and output index zero
     * @throws NullPointerException if {@code tiePolicy} is null, with message {@code tiePolicy};
     *     no Tensor identity is consumed
     * @throws IllegalArgumentException if this tensor is BOOL, with the exact numeric-type
     *     message, or if the normalized selected extent is statically zero, with the exact
     *     arg-extrema empty-axis message; no Tensor identity is consumed
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for this tensor's Shape,
     *     including every scalar axis; no Tensor identity is consumed
     * @throws IllegalStateException if tensor identifier space is exhausted after valid local
     *     metadata and provenance have been constructed
     */
    public Tensor argMin(
            int axis, boolean keepDimensions, ArgExtremaTiePolicy tiePolicy) {
        return TensorArgExtremaExpressions.apply(
                this, AggregateReductionKind.ARG_MIN, axis, keepDimensions, tiePolicy);
    }

    /**
     * Builds an axis-removing arg-max expression using the first logical index for ties.
     *
     * <p>This tensor must have FLOAT64, FLOAT32, BFLOAT16, INT32, or INT64 type. The caller axis is
     * normalized once and removed; rank-one input produces the canonical scalar Shape, and every
     * unaffected Dimension reference is retained. This convenience delegates directly with
     * {@link ArgExtremaTiePolicy#FIRST_INDEX}, which requests the smallest logical coordinate
     * among equal maximum candidates.</p>
     *
     * <p>The fresh result has fixed INT64 type, false gradient eligibility, unresolved layout, no
     * label or host storage, and one-input {@link AggregateReductionKind#ARG_MAX} provenance at
     * output index zero. Integral candidates use signed order. Floating candidates prefer NaN to
     * non-NaN, treat multiple NaNs as ties, order negative zero below positive zero, and order
     * infinities normally. Construction records this meaning without reading values, selecting an
     * index, creating gradient rules, capturing a graph, lowering, or executing work.</p>
     *
     * @param axis input axis in {@code [-rank, rank - 1]}; negative values count from the final
     *     axis
     * @return a non-null fresh, unlabeled, storage-free INT64 arg-max expression whose selected
     *     axis is removed, with false gradient eligibility, unresolved layout, exact one-input
     *     provenance, and output index zero
     * @throws IllegalArgumentException if this tensor is BOOL, with message {@code input must have
     *     a numeric data type, but was BOOL}, or if the normalized selected extent is statically
     *     zero, with the exact arg-extrema empty-axis message; no Tensor identity is consumed
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for this tensor's Shape,
     *     including every scalar axis; no Tensor identity is consumed
     * @throws IllegalStateException if tensor identifier space is exhausted after valid local
     *     metadata and provenance have been constructed
     */
    public Tensor argMax(int axis) {
        return TensorArgExtremaExpressions.apply(
                this, AggregateReductionKind.ARG_MAX, axis, false,
                ArgExtremaTiePolicy.FIRST_INDEX);
    }

    /**
     * Builds an arg-max expression using the first logical index for ties.
     *
     * <p>The numeric input axis is normalized once. A false {@code keepDimensions} removes it;
     * true replaces it with a new static extent one while retaining every unaffected Dimension
     * reference. This convenience delegates directly with {@link
     * ArgExtremaTiePolicy#FIRST_INDEX}. A statically empty selected extent is invalid, while an
     * unselected zero extent or dynamic/expression selected extent is accepted.</p>
     *
     * <p>The fresh result is unlabeled and storage-free, with fixed INT64 type, false gradient
     * eligibility, unresolved layout, shared arg-extrema attributes, exact
     * {@link AggregateReductionKind#ARG_MAX} one-input provenance, and output index zero. Floating
     * and integral ordering is the same as {@link #argMax(int)}. Construction performs no value
     * selection, gradient, compiler, backend, runtime, or execution work.</p>
     *
     * @param axis input axis in {@code [-rank, rank - 1]}; negative values count from the final
     *     axis
     * @param keepDimensions {@code true} to retain the selected axis with extent one, or
     *     {@code false} to remove it
     * @return a non-null fresh, unlabeled, storage-free INT64 arg-max expression with the requested
     *     Shape, false gradient eligibility, unresolved layout, and exact index-zero provenance
     * @throws IllegalArgumentException if this tensor is BOOL, with the exact numeric-type
     *     message, or if the normalized selected extent is statically zero, with the exact
     *     arg-extrema empty-axis message; no Tensor identity is consumed
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for this tensor's Shape,
     *     including every scalar axis; no Tensor identity is consumed
     * @throws IllegalStateException if tensor identifier space is exhausted after valid local
     *     metadata and provenance have been constructed
     */
    public Tensor argMax(int axis, boolean keepDimensions) {
        return TensorArgExtremaExpressions.apply(
                this, AggregateReductionKind.ARG_MAX, axis, keepDimensions,
                ArgExtremaTiePolicy.FIRST_INDEX);
    }

    /**
     * Builds an arg-max expression with an explicit logical-index tie policy.
     *
     * <p>The non-null policy is checked before input eligibility and axis validation and retained
     * by exact enum reference in shared attributes. {@code FIRST_INDEX} requests the smallest
     * logical coordinate and {@code LAST_INDEX} the largest among equal maximum candidates,
     * independent of storage offset, stride, layout, or traversal. Integral candidates use signed
     * order. Floating candidates prefer NaN, treat multiple NaNs as ties, order negative zero
     * below positive zero, and order infinities normally.</p>
     *
     * <p>The axis is normalized once and removed or replaced with a new extent one. A statically
     * empty selected extent is invalid, while an unselected zero extent or unbound selected extent
     * is accepted. The fresh result has fixed INT64/false-gradient metadata, unresolved layout,
     * no label or storage, exact one-input ARG_MAX provenance, and output index zero. Construction
     * neither compares values nor supplies gradient, compiler, backend, runtime, or execution
     * behavior.</p>
     *
     * @param axis input axis in {@code [-rank, rank - 1]}; negative values count from the final
     *     axis
     * @param keepDimensions {@code true} to retain the selected axis with extent one, or
     *     {@code false} to remove it
     * @param tiePolicy non-null explicit first- or last-logical-index policy retained by exact
     *     enum reference
     * @return a non-null fresh, unlabeled, storage-free INT64 arg-max expression with the requested
     *     Shape, false gradient eligibility, unresolved layout, exact attributes and ordered
     *     provenance, and output index zero
     * @throws NullPointerException if {@code tiePolicy} is null, with message {@code tiePolicy};
     *     no Tensor identity is consumed
     * @throws IllegalArgumentException if this tensor is BOOL, with the exact numeric-type
     *     message, or if the normalized selected extent is statically zero, with the exact
     *     arg-extrema empty-axis message; no Tensor identity is consumed
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for this tensor's Shape,
     *     including every scalar axis; no Tensor identity is consumed
     * @throws IllegalStateException if tensor identifier space is exhausted after valid local
     *     metadata and provenance have been constructed
     */
    public Tensor argMax(
            int axis, boolean keepDimensions, ArgExtremaTiePolicy tiePolicy) {
        return TensorArgExtremaExpressions.apply(
                this, AggregateReductionKind.ARG_MAX, axis, keepDimensions, tiePolicy);
    }

    /**
     * Creates a fresh inclusive forward cumulative-sum expression along one axis.
     *
     * <p>A cumulative sum replaces each logical position with the sum of the prefix ending at
     * that position. For input {@code [1, 2, 3]}, inclusive forward mode represents
     * {@code [1, 3, 6]}: the first result includes {@code 1}, the second includes
     * {@code 1 + 2}, and the third includes {@code 1 + 2 + 3}. This overload is exactly
     * equivalent to {@code cumSum(axis, false, false)}.</p>
     *
     * <p>The axis may be positive or negative and is normalized against the input rank. The input
     * must have FLOAT64, FLOAT32, BFLOAT16, INT32, or INT64 data type. Construction retains the
     * exact input Shape, data type, and gradient-eligibility metadata, but leaves result layout
     * unresolved. It returns a fresh unlabeled, storage-free Tensor whose provenance contains
     * this Tensor as its sole input. Construction does not inspect or accumulate values and does
     * not define numerical, gradient, compiler, backend, or execution behavior.</p>
     *
     * @param axis the positive or negative input axis accepted by
     *     {@link io.github.pho001.synaptik.model.shape.Shape#normalizeAxis(int)}
     * @return a non-null fresh inclusive forward cumulative-sum expression with unresolved
     *     layout, no label or storage, and exact one-input provenance
     * @throws IllegalArgumentException if this Tensor has BOOL data type, with message
     *     {@code input must have a numeric data type, but was BOOL}
     * @throws IndexOutOfBoundsException if {@code axis} is outside this Tensor's shape rank,
     *     including every axis for a scalar Tensor
     * @throws IllegalStateException if tensor identifier space is exhausted after local metadata
     *     validation and construction
     */
    public Tensor cumSum(int axis) {
        return TensorCumulativeSumExpressions.apply(this, axis, false, false);
    }

    /**
     * Creates a fresh cumulative-sum expression with explicit inclusion and direction modes.
     *
     * <p>For input {@code [1, 2, 3]}, the four modes represent these results while preserving
     * output position order:</p>
     * <ul>
     *   <li>inclusive forward ({@code false, false}): {@code [1, 3, 6]}, from {@code 1},
     *       {@code 1 + 2}, and {@code 1 + 2 + 3};</li>
     *   <li>exclusive forward ({@code true, false}): {@code [0, 1, 3]}, from the empty prefix,
     *       {@code 1}, and {@code 1 + 2};</li>
     *   <li>inclusive reverse ({@code false, true}): {@code [6, 5, 3]}, from
     *       {@code 1 + 2 + 3}, {@code 2 + 3}, and {@code 3}; and</li>
     *   <li>exclusive reverse ({@code true, true}): {@code [5, 3, 0]}, where the final logical
     *       results come from {@code 2 + 3}, {@code 3}, and the empty reverse prefix.</li>
     * </ul>
     *
     * <p>Exclusive mode omits the current element from its prefix. Reverse mode changes traversal
     * direction, not output Shape or dimension order. The axis may be positive or negative and is
     * normalized against the input rank. FLOAT64, FLOAT32, BFLOAT16, INT32, and INT64 are
     * accepted; BOOL is rejected before axis validation. Construction retains the exact input
     * Shape, data type, and gradient-eligibility metadata in an unresolved-layout descriptor. It
     * creates a fresh unlabeled, storage-free Tensor with exact one-input provenance, without
     * inspecting values, executing a scan, or defining numerical, gradient, compiler, or backend
     * behavior.</p>
     *
     * @param axis the positive or negative input axis accepted by
     *     {@link io.github.pho001.synaptik.model.shape.Shape#normalizeAxis(int)}
     * @param exclusive {@code true} to omit the current position from each traversed prefix, or
     *     {@code false} to include it
     * @param reverse {@code true} to traverse from the axis end toward its beginning while keeping
     *     output positions ordered, or {@code false} to traverse forward
     * @return a non-null fresh cumulative-sum expression retaining the requested mode flags,
     *     unresolved layout, no label or storage, and exact one-input provenance
     * @throws IllegalArgumentException if this Tensor has BOOL data type, with message
     *     {@code input must have a numeric data type, but was BOOL}
     * @throws IndexOutOfBoundsException if {@code axis} is outside this Tensor's shape rank,
     *     including every axis for a scalar Tensor
     * @throws IllegalStateException if tensor identifier space is exhausted after local metadata
     *     validation and construction
     */
    public Tensor cumSum(int axis, boolean exclusive, boolean reverse) {
        return TensorCumulativeSumExpressions.apply(this, axis, exclusive, reverse);
    }

    /**
     * Creates a fresh softmax expression along one logical axis.
     *
     * <p>A normalization slice contains positions that differ only along {@code axis}. For the
     * slice {@code [1, 2, 3]}, ideal softmax probabilities are approximately
     * {@code [0.09003057, 0.24472847, 0.66524096]} and sum to one. The axis may be positive or
     * negative and is normalized against this Tensor's exact Shape.</p>
     *
     * <p>This Tensor must have FLOAT64, FLOAT32, or BFLOAT16 data type. The fresh result retains
     * the exact input Shape, data type, and gradient-eligibility metadata, but has unresolved
     * layout, no label or host storage, and exact one-input provenance. Construction does not
     * inspect values, calculate probabilities, select a finite-precision algorithm, decompose the
     * operation, define a gradient rule, capture a graph, or provide compiler, backend, runtime,
     * or execution behavior.</p>
     *
     * @param axis input axis in the inclusive range {@code [-rank, rank - 1]}; negative values
     *     count from the final axis
     * @return a non-null fresh storage-free softmax expression preserving exact Shape, data type,
     *     and gradient eligibility with normalized-axis metadata and one-input provenance
     * @throws IllegalArgumentException if this Tensor does not have a floating data type, with
     *     message {@code input must have a floating data type, but was <dataType>}; this check
     *     precedes axis validation
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for this Tensor's Shape,
     *     including every axis for a scalar Tensor
     * @throws IllegalStateException if tensor identifier space is exhausted after local metadata
     *     validation and construction
     */
    public Tensor softmax(int axis) {
        return TensorSoftmaxExpressions.apply(this, SoftmaxKind.SOFTMAX, axis);
    }

    /**
     * Creates a fresh log-softmax expression along one logical axis.
     *
     * <p>For each normalization slice, the ideal result is the natural logarithm of the
     * corresponding {@link #softmax(int)} probability. For {@code [1, 2, 3]}, ideal values are
     * approximately {@code [-2.40760596, -1.40760596, -0.40760596]}; exponentiating them yields
     * approximately {@code [0.09003057, 0.24472847, 0.66524096]}, whose sum is one. The axis may
     * be positive or negative and is normalized against this Tensor's exact Shape.</p>
     *
     * <p>This Tensor must have FLOAT64, FLOAT32, or BFLOAT16 data type. The fresh result retains
     * the exact input Shape, data type, and gradient-eligibility metadata, but has unresolved
     * layout, no label or host storage, and exact one-input provenance. Construction does not
     * inspect values, calculate logarithms or probabilities, select a finite-precision algorithm,
     * decompose the operation, define a gradient rule, capture a graph, or provide compiler,
     * backend, runtime, or execution behavior.</p>
     *
     * @param axis input axis in the inclusive range {@code [-rank, rank - 1]}; negative values
     *     count from the final axis
     * @return a non-null fresh storage-free log-softmax expression preserving exact Shape, data
     *     type, and gradient eligibility with normalized-axis metadata and one-input provenance
     * @throws IllegalArgumentException if this Tensor does not have a floating data type, with
     *     message {@code input must have a floating data type, but was <dataType>}; this check
     *     precedes axis validation
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for this Tensor's Shape,
     *     including every axis for a scalar Tensor
     * @throws IllegalStateException if tensor identifier space is exhausted after local metadata
     *     validation and construction
     */
    public Tensor logSoftmax(int axis) {
        return TensorSoftmaxExpressions.apply(this, SoftmaxKind.LOG_SOFTMAX, axis);
    }

    /**
     * Creates a fresh expression requesting canonical dense row-major result geometry.
     *
     * <p>The result preserves this tensor's exact logical Shape, DataType, and gradient-
     * eligibility value. A fully static Shape receives a newly constructed resolved layout with
     * canonical row-major strides, logical storage offset zero, non-view metadata, and its checked
     * referenced element span. A Shape containing a dynamic dimension remains unresolved because
     * numeric geometry cannot yet be calculated.</p>
     *
     * <p>The fresh result has a factory-assigned identity, no label or host storage, and provenance
     * containing {@link ContiguousKind#CONTIGUOUS}, {@code NoOperationAttrs.INSTANCE}, and exactly
     * this tensor as its sole input. Construction does not inspect this tensor's layout, label,
     * provenance, storage, liveness, or values. An already-contiguous input and repeated or nested
     * requests therefore remain distinct expressions until a later compiler proves a legal
     * canonicalization.</p>
     *
     * <p>Resolved result geometry describes the requested logical representation; it does not
     * prove eager allocation, copying, distinct physical storage, runtime residency, or a backend
     * route.</p>
     *
     * @return a non-null fresh derived tensor preserving exact Shape, DataType, and gradient
     *     eligibility, with static canonical layout or dynamic unresolved layout, exact one-input
     *     provenance, and no label or storage
     * @throws ArithmeticException if checked canonical stride or referenced-span arithmetic
     *     overflows for a fully static Shape; no tensor identity is consumed
     * @throws IllegalStateException if tensor identifier space is exhausted after local immutable
     *     expression metadata has been constructed
     */
    public Tensor contiguous() {
        return TensorContiguousExpressions.apply(this);
    }

    /**
     * Creates a fresh reshape expression from ordered numeric target dimensions.
     *
     * <p>The caller-owned array is read but never retained or mutated. Each requested dimension
     * must be non-negative except for at most one exact {@code -1}, which is inferred from this
     * tensor's known element count and the non-zero checked product of the other dimensions. An
     * empty request denotes the rank-zero scalar Shape. Zero extents are valid, but inference is
     * ambiguous and rejected when the other requested dimensions have product zero. A dynamic
     * input cannot supply a numeric inferred extent.</p>
     *
     * <p>The result retains this tensor's exact DataType and gradient eligibility and uses the
     * normalized target Shape. Known input and target element counts must match; equality involving
     * a dynamic Shape is deferred to later compiler validation. When this tensor has resolved
     * contiguous geometry and the target is fully static, the result receives a new view-marked
     * layout with canonical target strides and the input element offset. Otherwise layout remains
     * unresolved: this model method does not insert {@link #contiguous()}, choose materialization,
     * or inspect storage.</p>
     *
     * <p>Every valid call returns a fresh unlabeled, storage-free tensor with exact reshape
     * semantics and this tensor as its sole provenance input, including same-shape, repeated, and
     * nested requests. Resolved view metadata does not attach storage or promise zero-copy
     * execution.</p>
     *
     * @param requestedShape non-null caller-owned target dimensions; values must be non-negative
     *     except for at most one {@code -1}, and an empty array requests scalar Shape
     * @return a non-null fresh reshape expression retaining exact type and gradient eligibility,
     *     normalized target Shape, conditional resolved alias-view geometry, and no label/storage
     * @throws NullPointerException if {@code requestedShape} is null, with message
     *     {@code requestedShape}
     * @throws IllegalArgumentException if a dimension is below {@code -1}, multiple {@code -1}
     *     values occur, inference is unavailable or ambiguous, divisibility fails, or known input
     *     and target element counts differ
     * @throws ArithmeticException if requested-product, element-count, canonical-stride, or
     *     referenced-span arithmetic overflows; no tensor identity is consumed
     * @throws IllegalStateException if tensor identifier space is exhausted after all local
     *     metadata has been constructed
     */
    public Tensor reshape(long... requestedShape) {
        return TensorReshapeExpressions.apply(this, requestedShape);
    }

    /**
     * Creates a fresh reshape expression for an exact normalized target Shape.
     *
     * <p>The exact immutable {@code targetShape} reference is retained in both the result
     * descriptor and target-shape attributes without copying or normalization. Scalar,
     * zero-extent, static, and dynamic Shapes are accepted. When both input and target element
     * counts are known they must be equal; when either count is dynamic, equality is deferred
     * without binding symbols or inventing a numeric extent.</p>
     *
     * <p>The result retains this tensor's exact DataType and gradient eligibility. Resolved
     * contiguous input geometry plus a fully static target produces a new view-marked layout with
     * canonical target strides and the input element offset. Unresolved, strided, broadcast, or
     * dynamic geometry remains unresolved because this method does not force contiguous
     * materialization or choose an executable alias/copy route.</p>
     *
     * <p>Every valid call returns a fresh unlabeled, storage-free tensor with exact reshape
     * semantics and this tensor as its sole provenance input. Resolved view metadata neither
     * attaches host storage nor guarantees zero-copy execution.</p>
     *
     * @param targetShape non-null normalized semantic target Shape retained by exact reference
     * @return a non-null fresh reshape expression retaining exact type, target Shape, and gradient
     *     eligibility with conditional resolved alias-view geometry and no label/storage
     * @throws NullPointerException if {@code targetShape} is null, with message
     *     {@code targetShape}
     * @throws IllegalArgumentException if both Shapes have known unequal element counts
     * @throws ArithmeticException if element-count, canonical-stride, or referenced-span
     *     arithmetic overflows; no tensor identity is consumed
     * @throws IllegalStateException if tensor identifier space is exhausted after all local
     *     metadata has been constructed
     */
    public Tensor reshape(Shape targetShape) {
        return TensorReshapeExpressions.apply(this, targetShape);
    }

    /**
     * Creates a fresh expansion expression from raw non-negative target dimensions.
     *
     * <p>The caller-owned array is read by {@link Shape#of(long...)} and is neither retained nor
     * mutated. Every value, including zero, is a literal static extent; numeric {@code -1} has no
     * inference meaning and is rejected as a negative static dimension. An empty request denotes
     * the canonical scalar Shape.</p>
     *
     * <p>The target rank must be at least the input rank. Axes are aligned from the right. Each
     * aligned input dimension must either equal its target dimension structurally or be a
     * statically known singleton; any additional leading target axes are valid. Consequently,
     * equal dynamic symbols and singleton-to-dynamic expansion are locally provable, while other
     * dynamic combinations are rejected without binding symbols or recording constraints.</p>
     *
     * <p>The fresh result retains this tensor's exact DataType and gradient eligibility and uses
     * the normalized target Shape. For a fully static target and any resolved input layout, it
     * receives a new view-marked layout with the exact input element offset, preserved strides on
     * unchanged aligned axes, and zero strides on new leading or expanded singleton axes. A
     * dynamic target or unresolved input layout leaves result layout unresolved. This logical
     * view metadata neither attaches nor aliases host storage nor proves zero-copy execution.</p>
     *
     * <p>Every successful call returns a distinct unlabeled, storage-free tensor whose provenance
     * records {@link ShapeTransformKind#EXPAND}, the normalized target, and this tensor as its sole
     * input. Same-shape, repeated, scalar, and nested requests remain explicit expressions.</p>
     *
     * @param requestedShape non-null caller-owned literal target dimensions; each value must be
     *     non-negative, zero is valid, and an empty array requests scalar Shape
     * @return a non-null fresh expand expression retaining exact type and gradient eligibility,
     *     normalized target Shape, conditional resolved zero-stride view geometry, and no label
     *     or storage
     * @throws NullPointerException if {@code requestedShape} is null, with message
     *     {@code requestedShape}
     * @throws IllegalArgumentException if a requested extent is negative, target rank is below
     *     input rank, or an aligned dimension pair is not locally compatible
     * @throws ArithmeticException if resolved layout stride or referenced-span arithmetic
     *     overflows; no tensor identity is consumed
     * @throws IllegalStateException if tensor identifier space is exhausted after all local
     *     metadata has been constructed
     */
    public Tensor expand(long... requestedShape) {
        return TensorExpandExpressions.apply(this, requestedShape);
    }

    /**
     * Creates a fresh expansion expression for an exact target Shape.
     *
     * <p>The exact immutable {@code targetShape} reference is retained in both the result
     * descriptor and target-shape attributes. Scalar, zero-extent, static, mixed dynamic, and
     * fully dynamic Shapes are accepted when compatibility is locally provable. The target rank
     * must be at least the input rank; right-aligned dimensions must be structurally equal or the
     * input dimension must be a static singleton, while new leading target axes are unrestricted.
     * Unequal symbols and dynamic-versus-non-singleton combinations are rejected rather than
     * converted into hidden constraints.</p>
     *
     * <p>The result retains this tensor's exact DataType and gradient eligibility. A fully static
     * target plus any resolved input layout produces a new view-marked layout that preserves the
     * input offset and unchanged aligned strides and inserts zero strides for new leading or
     * expanded singleton axes. Dynamic target or unresolved input geometry stays unresolved.
     * Layout metadata does not attach host storage, promise an executable alias, or select
     * materialization.</p>
     *
     * <p>Every valid call returns a distinct unlabeled, storage-free tensor with exact expand
     * semantics and this tensor as its sole provenance input, including identity-like, repeated,
     * scalar, and nested requests.</p>
     *
     * @param targetShape non-null exact semantic target Shape retained by reference
     * @return a non-null fresh expand expression retaining exact type, target Shape, and gradient
     *     eligibility with conditional resolved zero-stride view geometry and no label or storage
     * @throws NullPointerException if {@code targetShape} is null, with message
     *     {@code targetShape}
     * @throws IllegalArgumentException if target rank is below input rank or an aligned dimension
     *     pair is not locally compatible
     * @throws ArithmeticException if resolved layout stride or referenced-span arithmetic
     *     overflows; no tensor identity is consumed
     * @throws IllegalStateException if tensor identifier space is exhausted after all local
     *     metadata has been constructed
     */
    public Tensor expand(Shape targetShape) {
        return TensorExpandExpressions.apply(this, targetShape);
    }

    /**
     * Creates a fresh expression that inserts one singleton axis.
     *
     * <p>For input rank {@code r}, the raw insertion position may be in the inclusive range
     * {@code [-r - 1, r]}. A negative value adds {@code r + 1} once, so rank-two axes {@code -3}
     * and {@code 0} insert at the start, while {@code -1} and {@code 2} insert at the end. Scalar
     * axes {@code -1} and {@code 0} both select its sole insertion position.</p>
     *
     * <p>The result Shape has rank one greater, contains one new static dimension of extent one,
     * and preserves the exact immutable references and order of every input Dimension. For any
     * resolved input layout kind, the result receives one new view-marked layout with the same
     * element offset and exact existing strides. The inserted stride is one at the end; otherwise
     * it is the checked product of the following input stride and extent. Unresolved input layout
     * remains unresolved. This logical view metadata attaches no host storage and promises no
     * physical alias or zero-copy execution.</p>
     *
     * <p>The fresh result retains exact data type and gradient eligibility, has no label or
     * storage, and records {@link AxisTransformKind#EXPAND_DIMS}, one
     * {@link AxisTransformAttrs} with the normalized insertion position, and exactly this tensor
     * as its provenance input. Repeated, nested, and inverse-like requests remain explicit.</p>
     *
     * @param axis raw singleton insertion position relative to the result rank
     * @return a non-null fresh expand-dimensions expression with inserted Shape and conditional
     *     resolved same-offset view geometry, exact type/eligibility/provenance, and no label or
     *     storage
     * @throws IndexOutOfBoundsException if {@code axis} is outside the insertion range, with
     *     message {@code Axis <axis> is outside insertion range for shape rank <rank>}; no tensor
     *     identity is consumed
     * @throws ArithmeticException if inserted-stride multiplication, layout classification, or
     *     referenced-span arithmetic overflows; no tensor identity is consumed
     * @throws IllegalStateException if tensor identifier space is exhausted after all local
     *     immutable metadata has been constructed
     */
    public Tensor expandDims(int axis) {
        return TensorRankEditingExpressions.expandDims(this, axis);
    }

    /**
     * Creates a fresh expression that removes one selected, statically known singleton axis.
     *
     * <p>The axis uses the existing Shape-axis range {@code [-rank, rank - 1]}; a negative value
     * counts once from the end. The selected dimension must be statically known with extent
     * exactly one. Zero, another static extent, and a dynamic dimension are rejected because this
     * local model operation neither guesses a future binding nor records a symbolic singleton
     * constraint. A scalar has no axis and therefore cannot be squeezed.</p>
     *
     * <p>The result Shape has rank one lower and preserves the exact immutable references and
     * order of every unaffected Dimension; squeezing rank one produces the canonical scalar Shape.
     * For any resolved input layout kind, the selected stride is omitted and all other exact
     * strides plus the exact element offset are retained in one new view-marked layout. Unresolved
     * input layout stays unresolved. This logical metadata attaches no storage and proves no
     * physical alias or copy-free execution.</p>
     *
     * <p>The fresh result retains exact data type and gradient eligibility, has no label or
     * storage, and records {@link AxisTransformKind#SQUEEZE}, one {@link AxisTransformAttrs} with
     * the normalized input axis, and exactly this tensor as its provenance input. Repeated,
     * nested, and inverse-like requests are not canonicalized.</p>
     *
     * @param axis positive or negative existing axis whose static singleton dimension is removed
     * @return a non-null fresh squeeze expression with one axis removed and conditional resolved
     *     same-offset view geometry, exact type/eligibility/provenance, and no label or storage
     * @throws IndexOutOfBoundsException if {@code axis} is outside the Shape rank, including every
     *     axis for a scalar, with message {@code Axis <axis> is outside shape rank <rank>}; no
     *     tensor identity is consumed
     * @throws IllegalArgumentException if the normalized selected dimension is not statically
     *     known as one, with message {@code cannot squeeze axis <normalizedAxis> of <shape>:
     *     dimension must be statically known as 1}; no tensor identity is consumed
     * @throws ArithmeticException if result-layout classification or referenced-span arithmetic
     *     overflows; no tensor identity is consumed
     * @throws IllegalStateException if tensor identifier space is exhausted after all local
     *     immutable metadata has been constructed
     */
    public Tensor squeeze(int axis) {
        return TensorRankEditingExpressions.squeeze(this, axis);
    }

    /**
     * Creates a fresh expression that completely reorders this tensor's axes.
     *
     * <p>The caller-owned array must contain exactly one entry per input axis. Entries use
     * output-to-input order: {@code requestedAxes[i]} identifies the input axis placed at output
     * axis {@code i}. A negative entry adds the input rank once, so for rank three
     * {@code [1, -3, 2]} normalizes to {@code [1, 0, 2]}. The normalized entries must contain every
     * input axis exactly once. An empty array is the valid identity permutation for a rank-zero
     * scalar. The array is defensively copied and is never retained or mutated.</p>
     *
     * <p>The result Shape contains the exact immutable input Dimension references in normalized
     * output order. If input layout is resolved, every current layout kind produces one new
     * view-marked layout whose strides are reordered in the same way and whose element offset is
     * unchanged. Unresolved input layout remains unresolved. Resolved view metadata does not
     * attach host storage, establish a physical alias, or promise zero-copy execution.</p>
     *
     * <p>The fresh result retains the exact input DataType and gradient eligibility, has no label
     * or storage, and records {@link AxisTransformKind#PERMUTE}, normalized permutation
     * attributes, and exactly this tensor as its provenance input. Identity, inverse, repeated,
     * and nested requests remain explicit expressions without canonicalization.</p>
     *
     * @param requestedAxes non-null caller-owned complete output-to-input axis permutation;
     *     negative entries count once from the input rank, and an empty array is valid only for a
     *     scalar
     * @return a non-null fresh permute expression retaining exact type and gradient eligibility,
     *     reordered Shape and conditional resolved view geometry, and no label or storage
     * @throws NullPointerException if {@code requestedAxes} is null, with message
     *     {@code requestedAxes}
     * @throws IllegalArgumentException if the axis count differs from input rank, a normalized
     *     axis is outside the rank, or a normalized axis is duplicated; no tensor identity is
     *     consumed
     * @throws ArithmeticException if resolved layout classification or referenced-span arithmetic
     *     overflows; no tensor identity is consumed
     * @throws IllegalStateException if tensor identifier space is exhausted after all local
     *     immutable metadata has been constructed
     */
    public Tensor permute(int... requestedAxes) {
        return TensorPermutationExpressions.apply(this, requestedAxes);
    }

    /**
     * Creates a fresh rank-two transpose expression.
     *
     * <p>This tensor must have rank two. The operation is exactly
     * {@link AxisTransformKind#PERMUTE} with normalized output-to-input axes {@code [1, 0]}; there
     * is no separate transpose semantic kind. The result swaps the exact two Dimension references
     * and, for resolved input geometry, swaps the exact two strides while preserving the element
     * offset in a new view-marked layout. Unresolved geometry stays unresolved.</p>
     *
     * <p>Every successful call returns a fresh unlabeled, storage-free tensor with unchanged data
     * type and gradient eligibility and exact one-input provenance. Logical view metadata neither
     * attaches storage nor promises physical aliasing or execution without a copy.</p>
     *
     * @return a non-null fresh rank-two PERMUTE expression with axes {@code [1, 0]}, swapped Shape
     *     and conditional resolved view geometry, exact metadata/provenance, and no label/storage
     * @throws IllegalStateException if this tensor is not rank two, with message
     *     {@code transpose() requires rank-2 tensor, got rank=<rank>}, or if tensor identifier
     *     space is exhausted after all local immutable metadata has been constructed
     * @throws ArithmeticException if resolved layout classification or referenced-span arithmetic
     *     overflows; no tensor identity is consumed
     */
    public Tensor transpose() {
        return TensorPermutationExpressions.transpose(this);
    }

    /**
     * Creates one fresh directional half-open slice expression from four parallel arrays.
     *
     * <p>Entry {@code i} supplies inclusive raw start {@code starts[i]}, exclusive raw end
     * {@code ends[i]}, positive or negative input axis {@code axes[i]}, and signed non-zero step
     * {@code steps[i]}. A positive step selects while the coordinate is below the end; a negative
     * step selects while it is above the end. The arrays must have equal lengths and are cloned
     * before entry inspection. Empty arrays create an explicit identity {@link SliceKind#SLICE}.</p>
     *
     * <p>Axes are normalized once in caller order and must be distinct. Each selected Dimension
     * must be static. A negative bound adds that extent once, then bounds are clamped by direction:
     * positive bounds use {@code [0, D]}, while a negative-step start uses {@code [0, D - 1]} and
     * its exclusive end uses {@code [-1, D - 1]}. Raw {@code -1} means the relative coordinate
     * {@code D - 1}; it is not the conceptual boundary before coordinate zero. For example, on
     * extent five, {@code (4, -1, -1)} is empty, whereas {@code (4, -6, -1)} selects coordinates
     * {@code [4, 3, 2, 1, 0]}.</p>
     *
     * <p>Checked arithmetic calculates the exact selected length and stores normalized
     * start/length/axis/step sequences without an end sentinel. Empty entries use canonical start
     * and length zero, including a negative-step zero-extent axis. Result rank is unchanged,
     * selected axes become new static lengths, and every unselected Dimension reference is
     * preserved exactly. {@link Long#MIN_VALUE} is a valid step when the finite selection is
     * representable.</p>
     *
     * <p>A resolved non-empty all-positive request produces one checked logical view: each start
     * advances the input offset and each selected original stride is multiplied by its step. An
     * unresolved input, empty result, or any negative step leaves the complete result layout
     * unresolved because the current layout contract forbids negative strides. No storage is
     * attached, and unresolved reverse geometry does not imply a copy or kernel choice.</p>
     *
     * <p>The fresh result retains exact data type and gradient eligibility, has no label or
     * storage, and records one identity-distinct producer with {@link SliceKind#SLICE}, normalized
     * attributes in caller order, exact inputs {@code [this]}, one output descriptor, and
     * provenance output index zero. Validation and checked arithmetic fail before identifier
     * allocation; every success consumes one identifier. Values, gradients, compiler behavior,
     * materialization, backend lowering, and execution are deferred.</p>
     *
     * @param starts non-null caller-owned inclusive raw starts; paired by index and never retained
     * @param ends non-null caller-owned exclusive raw ends; paired by index and never retained
     * @param axes non-null caller-owned positive or negative axes; paired by index and never retained
     * @param steps non-null caller-owned signed non-zero steps; paired by index and never retained
     * @return a non-null fresh storage-free SLICE expression with normalized Shape, conditional
     *     resolved view layout, preserved type/eligibility, and exact one-input provenance
     * @throws NullPointerException if any array is null, with its parameter name as the message
     * @throws IllegalArgumentException if lengths differ, an axis is invalid or duplicated after
     *     normalization, a step is zero, or a selected dimension is dynamic
     * @throws ArithmeticException if checked result element-count, layout-offset, stride,
     *     classification, or span arithmetic overflows; no tensor identity is consumed
     * @throws IllegalStateException if tensor identifier space is exhausted after all local
     *     immutable metadata has been constructed
     */
    public Tensor slice(long[] starts, long[] ends, int[] axes, long[] steps) {
        return TensorSliceExpressions.apply(this, starts, ends, axes, steps);
    }

    /**
     * Creates a fresh single-axis step-one slice expression through the general slice path.
     *
     * <p>This convenience is exactly one {@link SliceKind#SLICE} entry with {@code step = 1}; it
     * has no separate semantic kind. Axis and directional half-open bounds use the same one-time
     * negative normalization, positive-direction clamping, selected-static-dimension requirement,
     * canonical empty state, Shape derivation, conditional view geometry, producer/provenance,
     * and identifier behavior as
     * {@link #slice(long[], long[], int[], long[])}.</p>
     *
     * @param axis positive or negative selected input axis
     * @param fromInclusive raw inclusive bound normalized and clamped against the selected extent
     * @param toExclusive raw exclusive bound normalized and clamped against the selected extent
     * @return a non-null fresh storage-free one-axis SLICE expression with step one
     * @throws IllegalArgumentException if the axis is invalid or its dimension is dynamic
     * @throws ArithmeticException if checked result element-count, layout-offset, stride,
     *     classification, or span arithmetic overflows; no tensor identity is consumed
     * @throws IllegalStateException if tensor identifier space is exhausted after all local
     *     immutable metadata has been constructed
     */
    public Tensor sliceAxis(int axis, long fromInclusive, long toExclusive) {
        return TensorSliceExpressions.applyAxis(this, axis, fromInclusive, toExclusive, 1L);
    }

    /**
     * Creates a fresh single-axis slice expression with an explicit signed non-zero step.
     *
     * <p>This convenience creates exactly one {@link SliceKind#SLICE} occurrence and applies all
     * normalization, empty-state, Shape, layout, metadata, producer, provenance, and identifier
     * rules of {@link #slice(long[], long[], int[], long[])}. It does not invoke another public
     * Tensor method. A negative step leaves layout unresolved even when the input is resolved.</p>
     *
     * @param axis positive or negative selected input axis
     * @param fromInclusive raw inclusive bound normalized and clamped against the selected extent
     * @param toExclusive raw directional exclusive bound normalized and clamped against the
     *     selected extent
     * @param step signed non-zero distance between selected coordinates
     * @return a non-null fresh unlabeled, storage-free one-axis SLICE expression with one producer
     *     and provenance output index zero
     * @throws IllegalArgumentException if the axis is invalid, its dimension is dynamic, or
     *     {@code step} is zero; no tensor identity is consumed
     * @throws ArithmeticException if checked bound, length, result, or positive-step view
     *     arithmetic overflows; no tensor identity is consumed
     * @throws IllegalStateException if tensor identifier space is exhausted during final creation
     */
    public Tensor sliceAxis(
            int axis, long fromInclusive, long toExclusive, long step) {
        return TensorSliceExpressions.applyAxis(this, axis, fromInclusive, toExclusive, step);
    }

    /**
     * Creates one fresh slice expression that reverses the requested axes in caller order.
     *
     * <p>The non-null varargs array is cloned once. Each negative axis adds rank once; the first
     * invalid raw axis or repeated normalized axis fails. Every selected Dimension must be static.
     * Extent {@code D > 0} contributes normalized start {@code D - 1}, length {@code D}, and step
     * {@code -1}; extent zero contributes canonical start and length zero with step {@code -1}.
     * All entries belong to one {@link SliceKind#SLICE} and one producer—there is no FLIP kind or
     * per-axis chain.</p>
     *
     * <p>An empty axis list means identity, not all axes. It is valid for scalar, static, zero-
     * extent, and dynamic Shapes because it selects no Dimension, and it still creates a fresh
     * explicit SLICE occurrence. A non-empty scalar request fails axis validation. Every selected
     * reversal has unresolved layout under the current non-negative-stride layout contract.</p>
     *
     * <p>Every success preserves exact Shape references, data type, and gradient eligibility,
     * remains unlabeled and storage-free, and records exact inputs {@code [this]}, one output
     * descriptor, and provenance output index zero. Validation fails before identifier allocation;
     * every success consumes one identifier.</p>
     *
     * @param axes non-null caller-owned positive or negative axes; cloned and never retained, with
     *     an empty array meaning identity
     * @return a non-null fresh storage-free SLICE expression with unresolved layout when any axis
     *     is selected
     * @throws NullPointerException if {@code axes} is null, with message {@code axes}
     * @throws IllegalArgumentException if an axis is invalid, duplicated after normalization, or
     *     selects a dynamic dimension; no tensor identity is consumed
     * @throws IllegalStateException if tensor identifier space is exhausted during final creation
     */
    public Tensor flip(int... axes) {
        return TensorSliceExpressions.flip(this, axes);
    }

    /**
     * Creates a fresh expression that fixes one scalar coordinate on one input axis and removes
     * that axis.
     *
     * <p>The positive or negative {@code axis} is normalized against this tensor's rank. When the
     * selected extent is static, a negative {@code index} adds that extent once and the normalized
     * coordinate must be in bounds. For example, both {@code select(1, 2)} and
     * {@code select(-2, -1)} on Shape {@code [2, 3, 4]} record normalized axis {@code 1}, index
     * {@code 2}, and produce Shape {@code [2, 4]}. A non-negative index on a dynamic selected
     * extent is retained with upper-bound validation deferred; a negative index cannot be
     * normalized locally and is rejected.</p>
     *
     * <p>The result Shape removes the selected Dimension while retaining every unaffected exact
     * Dimension reference. Selecting the only axis of a rank-one tensor produces the canonical
     * scalar Shape. For resolved input geometry and a non-empty result, the result removes the
     * selected stride and advances the element offset in a new logical view descriptor. Thus
     * contiguous Shape {@code [2, 3, 4]} with strides {@code [12, 4, 1]} selected at axis
     * {@code 1}, index {@code 2} produces strides {@code [12, 1]} and offset {@code 8}.
     * Unresolved input geometry and empty results remain unresolved. View metadata neither
     * attaches storage nor promises a physical alias.</p>
     *
     * <p>The fresh result preserves exact data type and gradient eligibility, has no label or
     * storage, and records {@link SelectKind#SELECT} with normalized attributes and exactly this
     * tensor as its provenance input. Scalar select is distinct from conditional {@code WHERE},
     * repeated unstack selection, half-open {@code SLICE}, and tensor-index gather. This method
     * does not read values, define a gradient rule, capture a graph, choose materialization or a
     * backend, or execute selection.</p>
     *
     * @param axis input axis in {@code [-rank, rank - 1]}; negative values count once from the
     *     final axis
     * @param index scalar coordinate; negative values count once from a static selected extent,
     *     while a dynamic selected extent accepts only a non-negative coordinate with deferred
     *     upper-bound validation
     * @return a non-null fresh storage-free SELECT tensor with one axis removed, preserved type
     *     and gradient eligibility, conditional logical view geometry, and exact provenance
     * @throws IndexOutOfBoundsException if {@code axis} is invalid, including every axis for a
     *     scalar input, or a statically normalized index is outside the selected extent
     * @throws IllegalArgumentException if {@code index} is negative for a dynamic selected extent
     * @throws ArithmeticException if checked result-element-count, view-offset,
     *     layout-classification, or referenced-span arithmetic overflows; no tensor identity is
     *     consumed
     * @throws IllegalStateException if tensor identifier space is exhausted after local immutable
     *     metadata has been constructed
     */
    public Tensor select(int axis, long index) {
        return TensorSelectExpressions.apply(this, axis, index);
    }

    /**
     * Creates a fresh canonical gather expression that replaces one data axis with the
     * complete indices Shape.
     *
     * <p>The ordered logical inputs are {@code [this, indices]}, and indices must be exact INT32
     * or INT64. Data Shape {@code [2, 3, 4]}, axis {@code 1}, and indices Shape {@code [5, 6]}
     * produce {@code [2, 5, 6, 4]}. Scalar indices instead produce {@code [2, 4]}. Every
     * unaffected data Dimension and inserted indices Dimension is retained exactly.</p>
     *
     * <p>The result records {@link AxisGatherKind#GATHER}, preserves data type and gradient
     * eligibility, and has unresolved layout, no label, and no storage. No index values or bounds
     * are inspected, and no gradient,
     * compiler, materialization, backend, or execution behavior is defined.</p>
     *
     * @param indices non-null INT32 or INT64 tensor whose complete Shape replaces the selected
     *     data axis; retained by exact reference in provenance and not mutated
     * @param axis data axis in {@code [-rank, rank - 1]}; negative values count from the final axis
     * @return a non-null fresh storage-free GATHER tensor with inserted indices Shape,
     *     preserved data metadata, unresolved layout, and exact provenance
     * @throws NullPointerException if {@code indices} is null, with message {@code indices}
     * @throws IllegalArgumentException if the indices data type is not INT32 or INT64
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the data rank
     * @throws IllegalStateException if tensor identifier space is exhausted after local metadata
     *     construction
     */
    public Tensor gather(Tensor indices, int axis) {
        return TensorAxisGatherExpressions.gather(this, indices, axis);
    }

    /**
     * Creates a fresh same-rank expression whose indices align with data away from one axis.
     *
     * <p>Indices must be exact INT32 or INT64, have the same rank as data, and have equal
     * Dimensions on every non-selected axis. The selected extents may differ. Data Shape
     * {@code [2, 3, 4]}, indices Shape {@code [2, 7, 4]}, and axis {@code 1} retain the exact
     * indices Shape {@code [2, 7, 4]} as the result. Equal dynamic symbols pass, while different
     * symbols fail on their first non-axis mismatch.</p>
     *
     * <p>The result preserves exact data type and gradient eligibility, has unresolved layout, no
     * label or storage, and records {@link AxisGatherKind#GATHER_ELEMENTS}, normalized attributes,
     * and ordered {@code [this, indices]} provenance. It reads no values, checks no value bounds,
     * and defines no gradient, compiler, materialization, backend, or execution behavior. It is
     * distinct from scalar select, gather-ND, and functional scatter.</p>
     *
     * @param indices non-null same-rank INT32 or INT64 tensor matching data away from the selected
     *     axis; its exact Shape becomes the result Shape and it is not mutated
     * @param axis data axis in {@code [-rank, rank - 1]}; negative values count from the final axis
     * @return a non-null fresh storage-free GATHER_ELEMENTS tensor retaining exact indices Shape,
     *     data metadata, unresolved layout, and exact provenance
     * @throws NullPointerException if {@code indices} is null, with message {@code indices}
     * @throws IllegalArgumentException if index type, rank, or non-axis alignment is invalid
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the data rank
     * @throws IllegalStateException if tensor identifier space is exhausted after local metadata
     *     construction
     */
    public Tensor gatherElements(Tensor indices, int axis) {
        return TensorAxisGatherExpressions.gatherElements(this, indices, axis);
    }

    /**
     * Creates a fresh tuple-index expression with no shared leading batch Dimensions.
     *
     * <p>This convenience is exactly {@code gatherNd(indices, 0)}. The non-null
     * {@code indices} tensor must have exact {@link DataType#INT32} or
     * {@link DataType#INT64} type and rank at least one. Its final Dimension must have a
     * statically known positive extent {@code K} no greater than this tensor's rank. That extent
     * is tuple depth: each tuple addresses the first {@code K} data axes. The result Shape is the
     * indices Shape without its final Dimension followed by the remaining data Dimensions. For
     * data {@code [2, 3, 4]} and indices {@code [5, 2]}, the result is {@code [5, 4]}; data
     * {@code [2, 3]} and indices {@code [2]} produce the canonical scalar Shape {@code []}.</p>
     *
     * <p>The result preserves this tensor's exact data type and gradient eligibility, has
     * unresolved layout, no label or storage, and records {@link GatherNdKind#GATHER_ND},
     * {@code new GatherNdAttrs(0)}, and ordered exact provenance {@code [this, indices]}. No index
     * value or bound is inspected. This method defines no gradient, scatter, compiler capture,
     * materialization, backend support, or execution behavior.</p>
     *
     * @param indices non-null INT32 or INT64 coordinate-tuple tensor; its final static Dimension
     *     supplies tuple depth, and it is retained by exact reference without mutation
     * @return a non-null fresh storage-free GATHER_ND tensor with the derived Shape, preserved
     *     data metadata, unresolved layout, and exact provenance
     * @throws NullPointerException if {@code indices} is null, with message {@code indices}
     * @throws IllegalArgumentException if index type, rank, tuple depth, or data-rank compatibility
     *     is invalid
     * @throws ArithmeticException if checked result-rank arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted after local metadata
     *     construction
     */
    public Tensor gatherNd(Tensor indices) {
        return TensorGatherNdExpressions.gatherNd(this, indices);
    }

    /**
     * Creates a fresh tuple-index expression after shared leading batch Dimensions.
     *
     * <p>The ordered logical inputs are {@code [this, indices]}. Indices must have exact
     * {@link DataType#INT32} or {@link DataType#INT64} type and rank at least one. The non-negative
     * {@code batchDimensions} count must be smaller than both input ranks, and each leading batch
     * Dimension must be structurally equal in increasing axis order. Equal static sizes and equal
     * dynamic symbols pass; broadcasting and symbolic constraint generation are not performed.</p>
     *
     * <p>The final indices Dimension must be static. Its positive extent {@code K} is tuple depth
     * and must not exceed {@code dataRank - batchDimensions}. Each tuple indexes data axes
     * {@code [batchDimensions, batchDimensions + K)}. The result Shape is the indices prefix
     * excluding only its final tuple-depth Dimension, followed by the untouched data suffix. Data
     * {@code [2, 3, 4]}, indices {@code [2, 5, 1]}, and one batch Dimension produce
     * {@code [2, 5, 4]}. Data {@code [N, 3, 4]}, indices {@code [N, M, 1]}, and one batch
     * Dimension produce {@code [N, M, 4]} while retaining the exact {@code N}, {@code M}, and
     * suffix Dimension references.</p>
     *
     * <p>The fresh result preserves this tensor's exact data type and gradient eligibility, has
     * unresolved layout, no label or storage, and records {@link GatherNdKind#GATHER_ND}, exact
     * {@link GatherNdAttrs}, and ordered provenance. Validation never reads index values or checks
     * bounds. This tuple-index operation is distinct from one-axis gather and scatter-ND and
     * defines no gradient, compiler capture, materialization, backend support, or execution.</p>
     *
     * @param indices non-null INT32 or INT64 coordinate-tuple tensor; retained by exact reference
     *     in provenance without mutation or value access
     * @param batchDimensions non-negative number of shared leading data and indices Dimensions;
     *     must be smaller than both ranks
     * @return a non-null fresh storage-free GATHER_ND tensor with derived Shape, preserved data
     *     metadata, unresolved layout, and exact provenance
     * @throws NullPointerException if {@code indices} is null, with message {@code indices}
     * @throws IllegalArgumentException if batch count, index type or rank, shared batch prefix, or
     *     static positive tuple depth is invalid
     * @throws ArithmeticException if checked result-rank arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted after local metadata
     *     construction
     */
    public Tensor gatherNd(Tensor indices, int batchDimensions) {
        return TensorGatherNdExpressions.gatherNd(this, indices, batchDimensions);
    }

    /**
     * Creates a fresh tuple-index scatter expression with replacement and no shared batch prefix.
     *
     * <p>This convenience is exactly
     * {@link #scatterNd(Tensor, Tensor, ScatterReduction, int) scatterNd(indices, updates,
     * ScatterReduction.NONE, 0)}. The ordered logical inputs are
     * {@code [this, indices, updates]}. Indices must use exact {@link DataType#INT32} or
     * {@link DataType#INT64}; updates must use this tensor's exact data type. The final indices
     * Dimension is a statically known positive tuple depth. Updates must have the indices prefix
     * without that final Dimension followed by the unindexed data suffix.</p>
     *
     * <p>The fresh result preserves this tensor's exact Shape and type, combines data/update
     * gradient eligibility by logical OR, leaves layout unresolved, and records exact
     * {@link ScatterNdKind#SCATTER_ND}, {@code new ScatterNdAttrs(0, ScatterReduction.NONE)}, and
     * ordered provenance. Construction reads no values, checks no bounds or duplicate targets,
     * performs no replacement, mutates no input, and defines no gradient, compiler, backend, or
     * execution behavior.</p>
     *
     * @param indices non-null INT32 or INT64 coordinate-tuple tensor retained by exact reference;
     *     its values are never inspected
     * @param updates non-null exact-data-type tensor whose Shape equals the corresponding
     *     Gather-ND result Shape; retained by exact reference and never read or mutated
     * @return a non-null fresh storage-free SCATTER_ND tensor with NONE reduction, zero batch
     *     Dimensions, exact data Shape/type, unresolved layout, and three-input provenance
     * @throws NullPointerException if {@code indices} or {@code updates} is null, checked in order
     * @throws IllegalArgumentException if index type, update type, rank, tuple depth, or updates
     *     Shape is invalid
     * @throws ArithmeticException if checked expected-updates-rank arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted after local metadata
     *     construction
     */
    public Tensor scatterNd(Tensor indices, Tensor updates) {
        return TensorScatterNdExpressions.scatterNd(this, indices, updates);
    }

    /**
     * Creates a fresh tuple-index scatter expression with an explicit zero-batch reduction.
     *
     * <p>This convenience is exactly
     * {@link #scatterNd(Tensor, Tensor, ScatterReduction, int) scatterNd(indices, updates,
     * reduction, 0)}. {@link ScatterReduction#NONE} permits every current data type. Arithmetic
     * reductions permit floating and integral data but reject BOOL. The updates Shape is the
     * indices prefix without tuple depth followed by the unindexed data suffix.</p>
     *
     * <p>The result retains exact data Shape/type and data/update gradient-eligibility OR, has
     * unresolved layout, no label or storage, and exact ordered SCATTER_ND provenance. This method
     * reads no index or update value, detects no duplicate target, performs no write or reduction,
     * and defines no numerical order, gradient, compiler, backend, or execution behavior.</p>
     *
     * @param indices non-null INT32 or INT64 coordinate-tuple tensor retained by exact reference
     *     without value access
     * @param updates non-null exact-data-type tensor with the required tuple-scatter updates Shape;
     *     retained by exact reference and never read or mutated
     * @param reduction non-null replacement or arithmetic combination meaning retained exactly
     * @return a non-null fresh storage-free SCATTER_ND tensor with zero batch Dimensions, the
     *     supplied reduction, exact data metadata, unresolved layout, and ordered provenance
     * @throws NullPointerException if {@code indices}, {@code updates}, or {@code reduction} is
     *     null, checked in order
     * @throws IllegalArgumentException if type, reduction, rank, tuple depth, or Shape validation
     *     fails
     * @throws ArithmeticException if checked expected-updates-rank arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted after local metadata
     *     construction
     */
    public Tensor scatterNd(
            Tensor indices, Tensor updates, ScatterReduction reduction) {
        return TensorScatterNdExpressions.scatterNd(this, indices, updates, reduction);
    }

    /**
     * Creates a fresh functional Scatter-ND expression after shared leading batch Dimensions.
     *
     * <p>For data rank {@code R}, indices rank {@code Q}, batch count {@code B}, and tuple depth
     * {@code K} from the final indices Dimension, the leading {@code B} data and indices
     * Dimensions must be structurally equal. Each tuple indexes data axes {@code [B, B + K)}.
     * Updates must have exact Shape {@code indices[0:Q-1] + data[B+K:R]}. Thus data
     * {@code [2,3,4]} with indices {@code [5,2]} and {@code B=0} requires updates
     * {@code [5,4]}; indices {@code [2,5,1]} and {@code B=1} require updates
     * {@code [2,5,4]}; and data {@code [2,3]} with indices {@code [2]} requires canonical scalar
     * updates {@code []}.</p>
     *
     * <p>The fresh result preserves exact data Shape/type, combines data/update gradient
     * eligibility, leaves layout unresolved, and records exact {@link ScatterNdAttrs} and ordered
     * {@code [this, indices, updates]} provenance. A target is the result coordinate or suffix
     * slice addressed by one tuple; duplicate targets are multiple tuples addressing the same
     * target. Construction does not inspect values, validate bounds or duplicates, apply writes or
     * reductions, mutate data, define numeric order, create gradients, capture a graph, select a
     * backend, or execute work.</p>
     *
     * @param indices non-null INT32 or INT64 coordinate-tuple tensor whose leading batch prefix
     *     structurally matches data and whose final static Dimension supplies tuple depth
     * @param updates non-null exact-data-type tensor with Shape equal to the indices prefix plus
     *     unindexed data suffix; retained without value access or mutation
     * @param reduction non-null explicit replacement or arithmetic combination meaning
     * @param batchDimensions non-negative number of structurally equal leading data and indices
     *     Dimensions; must be smaller than both ranks
     * @return a non-null fresh storage-free SCATTER_ND tensor with exact data metadata, unresolved
     *     layout, supplied attributes, and exact three-input provenance
     * @throws NullPointerException if {@code indices}, {@code updates}, or {@code reduction} is
     *     null, checked in order
     * @throws IllegalArgumentException if index/update type, reduction eligibility, indices rank,
     *     batch fit/prefix, tuple depth, or updates Shape is invalid
     * @throws ArithmeticException if checked expected-updates-rank arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted after local metadata
     *     construction
     */
    public Tensor scatterNd(
            Tensor indices,
            Tensor updates,
            ScatterReduction reduction,
            int batchDimensions) {
        return TensorScatterNdExpressions.scatterNd(
                this, indices, updates, reduction, batchDimensions);
    }

    /**
     * Creates a fresh same-rank scatter-elements expression with replacement semantics.
     *
     * <p>This convenience is exactly
     * {@link #scatterElements(Tensor, Tensor, int, ScatterReduction) scatterElements(indices,
     * updates, axis, ScatterReduction.NONE)}. It accepts every current data type when updates
     * match this tensor exactly. Replacement requires unique target coordinates, but duplicate
     * detection needs index values and therefore remains outside this metadata-only construction.</p>
     *
     * @param indices non-null same-rank INT32 or INT64 tensor matching data away from the axis
     * @param updates non-null exact-data-type tensor with Shape equal to indices
     * @param axis data axis in {@code [-rank, rank - 1]}; negative values count from the final axis
     * @return a non-null fresh SCATTER_ELEMENTS tensor with NONE reduction, exact data metadata,
     *     unresolved layout, and exact three-input provenance
     * @throws NullPointerException if {@code indices} or {@code updates} is null, checked in order
     * @throws IllegalArgumentException if index type, update type, rank, or Shape is invalid
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the data rank
     * @throws IllegalStateException if tensor identifier space is exhausted after local metadata
     *     construction
     */
    public Tensor scatterElements(Tensor indices, Tensor updates, int axis) {
        return TensorAxisScatterExpressions.scatterElements(this, indices, updates, axis);
    }

    /**
     * Creates a fresh same-rank scatter-elements expression with an explicit reduction.
     *
     * <p>The ordered inputs are {@code [this, indices, updates]}. Indices must be exact INT32 or
     * INT64; updates must have this tensor's exact type. Indices and updates ranks and Dimensions
     * must match, and every non-selected indices Dimension must equal data. The selected
     * indices/updates extent may differ from data. For data {@code [2, 3, 4]}, axis {@code 1},
     * and indices/updates {@code [2, 5, 4]}, the result retains data Shape {@code [2, 3, 4]}.</p>
     *
     * <p>{@link ScatterReduction#NONE} accepts every current type. {@link ScatterReduction#ADD},
     * {@link ScatterReduction#MUL}, {@link ScatterReduction#MAX}, and
     * {@link ScatterReduction#MIN} accept floating and integral values and reject BOOL. The
     * reduction defines semantic combination at a target; this method does not inspect values,
     * detect NONE duplicates, apply writes/reductions, define numerical order or gradients,
     * capture a graph, select a backend, or execute work.</p>
     *
     * <p>The fresh result has exact data Shape/type, data/update eligibility OR, unresolved layout,
     * no label or storage, exact {@link AxisScatterKind#SCATTER_ELEMENTS} attributes, and ordered
     * provenance. Data and all inputs remain unchanged.</p>
     *
     * @param indices non-null same-rank INT32 or INT64 tensor matching data away from the axis;
     *     retained by exact reference and never read
     * @param updates non-null exact-data-type tensor with Shape equal to indices; retained by
     *     exact reference and never read or mutated
     * @param axis data axis in {@code [-rank, rank - 1]}; negative values count from the final axis
     * @param reduction non-null explicit replacement or arithmetic combination meaning
     * @return a non-null fresh SCATTER_ELEMENTS tensor with exact data metadata, unresolved layout,
     *     and exact three-input provenance
     * @throws NullPointerException if {@code indices}, {@code updates}, or {@code reduction} is
     *     null, checked in order
     * @throws IllegalArgumentException if index type, update type, BOOL reduction, rank, or Shape
     *     compatibility is invalid
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the data rank
     * @throws IllegalStateException if tensor identifier space is exhausted after local metadata
     *     construction
     */
    public Tensor scatterElements(
            Tensor indices, Tensor updates, int axis, ScatterReduction reduction) {
        return TensorAxisScatterExpressions.scatterElements(
                this, indices, updates, axis, reduction);
    }

    /**
     * Creates a fresh expression that adds constant-filled positions around every input axis.
     *
     * <p>The two caller-owned arrays must be non-null and contain exactly one non-negative width
     * per input axis. They are defensively cloned and are never retained or mutated. At axis
     * {@code i}, the result extent is the checked sum
     * {@code inputExtent + before[i] + after[i]}; for example, Shape {@code [2, 3]} with before
     * widths {@code [1, 0]} and after widths {@code [2, 4]} produces Shape {@code [5, 7]}.
     * Static zero extents and empty arrays for a scalar are valid. Dynamic extents retain the
     * canonical symbolic formula; for example, extent {@code N} with before width {@code 2} and
     * after width {@code 3} becomes {@code N + 5}. Zero widths preserve the exact input Dimension
     * reference.</p>
     *
     * <p>This convenience requires an exact FLOAT64 receiver because it adapts
     * {@code constantValue} through {@link ScalarValue#float64(double)}. The result preserves the
     * input data type and gradient-eligibility value. The constant's exact binary64 bits are
     * retained without conversion or backend interpretation.</p>
     *
     * <p>The fresh result always has unresolved layout, including a zero-width identity request,
     * because constant padding is an output-materialization operation rather than input view
     * geometry. It has no label or storage and records {@link PadKind#PAD}, immutable normalized
     * padding attributes, and exactly this tensor as its sole provenance input. Construction does
     * not inspect or copy values, attach an alias, define a gradient, capture or canonicalize a
     * graph, bind or evaluate symbolic extents, select a backend, map ONNX, materialize storage,
     * or execute padding.</p>
     *
     * @param before non-null caller-owned before widths, exactly one per input axis; every width
     *     must be non-negative, and an empty array is valid for a scalar
     * @param after non-null caller-owned after widths, exactly one per input axis; every width
     *     must be non-negative, and an empty array is valid for a scalar
     * @param constantValue exact raw binary64 padding constant retained without conversion or
     *     validation
     * @return a non-null fresh storage-free PAD expression with checked same-rank Shape,
     *     preserved data type and gradient eligibility, unresolved layout, and one-input
     *     provenance
     * @throws NullPointerException if {@code before} or {@code after} is null, with that parameter
     *     name as the message
     * @throws IllegalArgumentException if this tensor is not FLOAT64, either array length differs
     *     from input rank, or a width is negative
     * @throws ArithmeticException if checked static result-extent or symbolic-offset addition
     *     overflows; no tensor identity is consumed
     * @throws IllegalStateException if tensor identifier space is exhausted after all local
     *     immutable metadata has been constructed
     */
    public Tensor pad(long[] before, long[] after, double constantValue) {
        return pad(before, after, ScalarValue.float64(constantValue));
    }

    /**
     * Creates constant padding with an exact value matching this tensor's data type.
     *
     * <p>All six current data types are accepted when the non-null value type exactly equals this
     * tensor's type. Width validation precedes this equality check, which precedes checked Shape
     * arithmetic. The constant reference is retained in immutable {@code PadAttrs}; no scalar
     * conversion, value materialization, gradient rule, compiler behavior, or execution occurs.
     * The fresh result preserves type and gradient eligibility, derives the same-rank Shape, and
     * leaves layout unresolved even for identity widths.</p>
     *
     * @param before non-null caller-owned before widths, one non-negative value per axis
     * @param after non-null caller-owned after widths, one non-negative value per axis
     * @param constantValue non-null exact typed padding value matching this tensor
     * @return a non-null fresh storage-free PAD expression with checked same-rank Shape
     * @throws NullPointerException if {@code before}, {@code after}, or {@code constantValue} is
     *     null, checked in parameter order with its name as the message
     * @throws IllegalArgumentException if widths are invalid or the value type does not match this
     *     tensor's data type
     * @throws ArithmeticException if checked result-extent arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor pad(long[] before, long[] after, ScalarValue constantValue) {
        return TensorPadTileExpressions.pad(this, before, after, constantValue);
    }

    /**
     * Creates a fresh expression that repeats the complete input pattern along every axis.
     *
     * <p>The caller-owned varargs array must be non-null and contain exactly one strictly positive
     * repeat count per input axis. It is defensively cloned and is never retained or mutated. At
     * axis {@code i}, the result extent is the checked product
     * {@code inputExtent * repeats[i]}; for example, Shape {@code [2, 3]} with repeats
     * {@code [2, 4]} produces Shape {@code [4, 12]}. This is complete-pattern tiling rather than
     * repetition of each scalar into a consecutive run. Static zero extents and empty repeats for
     * a scalar are valid. Dynamic extents retain the canonical symbolic product; for example,
     * extent {@code N} repeated four times becomes {@code 4 * N}. Repeat one preserves the exact
     * input Dimension reference.</p>
     *
     * <p>Every current DataType is accepted, and the result preserves the exact input data type
     * and gradient-eligibility value. The fresh result always has unresolved layout, including a
     * repeat-one identity request, because tiling requires output materialization and cannot be
     * represented as an ordinary input view. It has no label or storage and records
     * {@link TileKind#TILE}, immutable repeat attributes, and exactly this tensor as its sole
     * provenance input.</p>
     *
     * <p>Construction does not inspect or repeat values, attach an alias, define a gradient,
     * capture or canonicalize a graph, bind or evaluate symbolic extents, select a backend, map
     * ONNX, materialize storage, or execute tiling.</p>
     *
     * @param repeats non-null caller-owned complete-pattern repeat counts, exactly one positive
     *     value per input axis; an empty array is valid for a scalar
     * @return a non-null fresh storage-free TILE expression with checked same-rank Shape,
     *     preserved data type and gradient eligibility, unresolved layout, and one-input
     *     provenance
     * @throws NullPointerException if {@code repeats} is null, with message {@code repeats}
     * @throws IllegalArgumentException if the array length differs from input rank or a repeat is
     *     non-positive
     * @throws ArithmeticException if checked static result-extent, symbolic coefficient, or
     *     symbolic offset multiplication overflows; no tensor identity is consumed
     * @throws IllegalStateException if tensor identifier space is exhausted after all local
     *     immutable metadata has been constructed
     */
    public Tensor tile(long... repeats) {
        return TensorPadTileExpressions.tile(this, repeats);
    }

    /**
     * Creates a fresh expression that concatenates an ordered non-empty input sequence.
     *
     * <p>The caller-owned varargs array is cloned once and is never retained. Input references,
     * including repeated references, remain in exact order in result provenance. Every input must
     * have the same exact data type and rank, and dimensions outside the selected existing axis
     * must be structurally equal. Validation rejects a null array, an empty array, and the first
     * null copied element before reading any descriptor. It then normalizes the axis, followed by
     * encounter-order data-type, rank, non-axis Dimension, and selected-extent validation.</p>
     *
     * <p>Selected extents are encounter-order folded through canonical checked symbolic addition.
     * Thus extents {@code N} and {@code M} become {@code N + M}, repeated terms combine, existing
     * linear expressions flatten, and static-zero companions preserve the opposing exact
     * Dimension reference. The formula is retained without binding or evaluation. The fresh
     * result has unresolved layout, gradient eligibility equal to the logical OR of all input
     * requests, no label or storage, and exact {@link TensorCompositionKind#CONCAT} provenance.
     * This method neither reads values nor materializes, executes, groups, or defines gradients
     * for the result.</p>
     *
     * @param axis existing input axis along which ordered extents are joined; for rank {@code r},
     *     the accepted raw range is {@code [-r, r - 1]} and a negative value is normalized once
     * @param inputs non-null caller-owned non-empty ordered input array with no null elements;
     *     the array is defensively snapshotted and the tensors are not mutated
     * @return a non-null fresh storage-free CONCAT tensor with checked same-rank Shape,
     *     unresolved layout, propagated gradient eligibility, and ordered provenance
     * @throws NullPointerException if {@code inputs} or an indexed element is null
     * @throws IllegalArgumentException if no inputs are supplied, data types or ranks differ, or
     *     non-concat dimensions differ
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the input rank
     * @throws ArithmeticException if a checked static selected extent, symbolic coefficient, or
     *     symbolic offset overflows
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public static Tensor concat(int axis, Tensor... inputs) {
        return TensorCompositionExpressions.concat(axis, inputs);
    }

    /**
     * Creates a fresh expression that stacks an ordered non-empty input sequence.
     *
     * <p>The caller-owned varargs array is cloned once and is never retained. All inputs must have
     * the same exact data type and structurally identical Shape. The axis is an insertion position
     * in the result; negative syntax is normalized once against {@code inputRank + 1}. The result
     * inserts one static dimension equal to the input count and preserves the first input's exact
     * Dimension references around it. Validation rejects a null array, an empty array, and the
     * first null copied element before reading any descriptor. It then normalizes the insertion
     * axis before checking each input's data type and Shape in encounter order.</p>
     *
     * <p>The fresh result has unresolved layout, gradient eligibility equal to the logical OR of
     * all input requests, no label or storage, and exact ordered
     * {@link TensorCompositionKind#STACK} provenance. Even a one-input request remains an explicit
     * stack. Construction does not broadcast or promote inputs, inspect values, materialize,
     * execute, capture a graph, or define gradient behavior.</p>
     *
     * @param axis insertion position in the result Shape; for input rank {@code r}, the accepted
     *     raw range is {@code [-(r + 1), r]} and a negative value is normalized once
     * @param inputs non-null caller-owned non-empty ordered input array with no null elements;
     *     the array is defensively snapshotted and the tensors are not mutated
     * @return a non-null fresh storage-free STACK tensor with one inserted count dimension,
     *     unresolved layout, propagated gradient eligibility, and ordered provenance
     * @throws NullPointerException if {@code inputs} or an indexed element is null
     * @throws IllegalArgumentException if no inputs are supplied, data types differ, or Shapes are
     *     not structurally identical
     * @throws IndexOutOfBoundsException if {@code axis} is outside the insertion range
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public static Tensor stack(int axis, Tensor... inputs) {
        return TensorCompositionExpressions.stack(axis, inputs);
    }

    /**
     * Creates independently indexed expressions for every coordinate of one selected axis.
     *
     * <p>The positive or negative existing axis is normalized once and must have a statically
     * known extent no larger than {@link Integer#MAX_VALUE}. Each result removes that axis,
     * preserves every other exact Dimension reference, and retains this tensor's exact data type
     * and gradient-eligibility value. Results occur in increasing source-coordinate order in an
     * immutable list. Each result is an independent scalar {@link SelectKind#SELECT} occurrence
     * at its corresponding coordinate.</p>
     *
     * <p>Every output is fresh, unlabeled, storage-free, and uses scalar select's conditional view
     * layout. The outputs are not grouped under one producer identity. A zero extent
     * returns an empty immutable list without consuming a tensor identifier. Identifier
     * exhaustion during a non-empty request may consume identifiers for earlier attempted
     * outputs, but no partial list is returned. This method does not inspect values, materialize,
     * execute, capture a graph, or define gradient behavior.</p>
     *
     * @param axis existing input axis to remove; for rank {@code r}, the accepted raw range is
     *     {@code [-r, r - 1]} and a negative value is normalized once before extent validation
     * @return a non-null immutable ordered list of fresh scalar-select outputs; empty when the
     *     selected static extent is zero
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for this tensor's rank
     * @throws IllegalArgumentException if the selected dimension is dynamic or its static extent
     *     exceeds {@link Integer#MAX_VALUE}
     * @throws IllegalStateException if tensor identifier space is exhausted while creating an
     *     output; identifiers already consumed during this request are not rolled back
     */
    public List<Tensor> unstack(int axis) {
        return TensorCompositionExpressions.unstack(this, axis);
    }

    /**
     * Creates a fresh expression that materializes sliding windows along one input axis.
     *
     * <p>A window is one consecutive group of {@code size} logical elements, and {@code step} is
     * the positive distance between successive window starts. The positive or negative
     * {@code axis} is normalized against this tensor's rank. Its dimension must be static and at
     * least {@code size}. The selected extent {@code D} is replaced by
     * {@code floor((D - size) / step) + 1}, and a new final dimension of extent {@code size} is
     * appended. For example, Shape {@code [2, 5, 3]}, axis one, size three, and step one produces
     * Shape {@code [2, 3, 3, 3]}: the new axis-one extent counts three window positions, the
     * original final extent remains three, and the appended extent is the window size.</p>
     *
     * <p>Every current data type is accepted. The result preserves all unaffected Dimension
     * references, the exact data type, and gradient eligibility. It is fresh, unlabeled,
     * storage-free, and layout-unresolved, with {@link WindowTransformKind#UNFOLD_AXIS} and this
     * tensor as its sole provenance input. Construction uses checked count arithmetic but does
     * not inspect or materialize values, promise a view, define a gradient rule, capture a graph,
     * or execute work.</p>
     *
     * @param axis raw input axis in {@code [-rank, rank - 1]}; negative values count once from
     *     the final axis
     * @param size positive window extent in logical elements; must fit the selected dimension
     * @param step positive distance between consecutive window starts in logical elements
     * @return a non-null fresh storage-free UNFOLD_AXIS tensor with the derived Shape, unresolved
     *     layout, preserved type and gradient eligibility, and exact one-input provenance
     * @throws IllegalArgumentException if the input rank is zero, {@code size} or {@code step} is
     *     non-positive, the selected dimension is dynamic, or {@code size} exceeds its extent
     * @throws IndexOutOfBoundsException if {@code axis} is outside the input rank
     * @throws ArithmeticException if checked window-count arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor unfold(int axis, long size, long step) {
        return TensorWindowExpressions.unfoldAxis(this, axis, size, step);
    }

    /**
     * Creates canonical two-dimensional image-window columns from a rank-four NCHW tensor.
     *
     * <p>NCHW orders dimensions as batch, channel, height, and width. The channel and spatial
     * dimensions must be static; the exact batch Dimension may remain dynamic. The supplied
     * immutable window defines positive kernel, stride, and dilation values, non-negative
     * symmetric padding on both sides of each spatial dimension, and floor or ceil output-size
     * rounding. Dilation is the spacing between kernel samples, and the effective kernel span is
     * {@code dilation * (kernel - 1) + 1}.</p>
     *
     * <p>For height and width independently, checked arithmetic calculates
     * {@code numerator = input + 2 * padding - effectiveKernel}. The effective kernel must fit the
     * padded dimension. Floor mode uses {@code numerator / stride + 1}; ceil mode increments the
     * quotient when a remainder exists and then adds one. This avoids the overflow-prone
     * {@code numerator + stride - 1} form. Result Shape is
     * {@code [N, C * kernelHeight * kernelWidth, outputHeight * outputWidth]}. For input
     * {@code [1,1,3,3]} and a 2-by-2 unit-stride, zero-padding, unit-dilation floor-mode window,
     * the result is {@code [1,4,4]}: one batch, four channel-kernel positions, and four windows.</p>
     *
     * <p>This im2col expression accepts only floating input. The fresh result preserves exact
     * type and gradient eligibility, leaves layout unresolved, has no label or storage, and
     * records {@link WindowTransformKind#UNFOLD2D} with this tensor as its sole input. It does not
     * sample padding, read values, materialize columns, define gradients, or execute work.</p>
     *
     * @param window non-null immutable symmetric two-dimensional window geometry retained by
     *     exact reference in operation attributes
     * @return a non-null fresh rank-three canonical-column tensor with unresolved layout,
     *     preserved floating type and gradient eligibility, and exact one-input provenance
     * @throws NullPointerException if {@code window} is null, with message {@code window}
     * @throws IllegalArgumentException if input is not rank-four NCHW, is not floating, has a
     *     dynamic channel/height/width dimension, or an effective kernel does not fit
     * @throws ArithmeticException if checked effective-kernel, padding, channel-window, spatial,
     *     or output-count arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor unfold2d(Window2dAttrs window) {
        return TensorWindowExpressions.unfold2d(this, window);
    }

    /**
     * Creates an overlap-accumulating two-dimensional fold into an explicit NCHW Shape.
     *
     * <p>The input must be floating canonical im2col data with rank-three Shape
     * {@code [N, C * kernelHeight * kernelWidth, outputHeight * outputWidth]}. The requested output
     * must have rank four in NCHW order, its exact batch Dimension must equal the column batch,
     * and channel, height, width, column-channel, and column-count dimensions must be static. The
     * same checked effective-kernel and floor/ceil formulas documented by
     * {@link #unfold2d(Window2dAttrs)} determine compatibility.</p>
     *
     * <p>Col2im scatters each column entry to its image coordinate. Overlapping entries are added,
     * uncovered positions are conceptually zero, and no overlap averaging occurs. For compatible
     * 2-by-2 unit-stride columns folded to {@code [1,1,3,3]}, the center receives four
     * contributions while each corner receives one. This expression records the operation only;
     * it neither reads columns nor performs accumulation.</p>
     *
     * <p>The result retains the exact supplied Shape reference, input floating type, and gradient
     * eligibility. It is fresh, unlabeled, storage-free, and layout-unresolved, with
     * {@link WindowTransformKind#FOLD2D} and exact one-input provenance.</p>
     *
     * @param outputShape non-null explicit rank-four NCHW result Shape retained by exact reference
     * @param window non-null immutable symmetric window geometry retained by exact reference
     * @return a non-null fresh FOLD2D tensor with the exact output Shape, unresolved layout,
     *     preserved floating type and gradient eligibility, and exact one-input provenance
     * @throws NullPointerException if {@code outputShape} or {@code window} is null, checked in
     *     that order with the parameter name as message
     * @throws IllegalArgumentException if ranks, data type, batch identity, static dimensions,
     *     channel-window extent, window count, or effective-kernel fit are incompatible
     * @throws ArithmeticException if checked effective-kernel, padding, channel-window, spatial,
     *     or window-count arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor fold2d(Shape outputShape, Window2dAttrs window) {
        return TensorWindowExpressions.fold2d(this, outputShape, window);
    }

    /**
     * Returns a synchronized snapshot of the current borrowed host-storage association.
     *
     * <p>A present result contains the exact attached identity-bearing storage object. Storage is
     * borrowed rather than owned and may be read-only. It is not hidden when its caller-owned
     * scope dies after attachment, and the liveness observation available through the returned
     * object may become stale immediately. The optional is a snapshot of reference state; later
     * replacement or clearing does not mutate it. Synchronization covers only this association,
     * not segment contents, liveness, scope closure, or thread accessibility.</p>
     *
     * @return a non-null optional containing the exact current storage reference, or empty when no
     *     storage is associated
     */
    public synchronized Optional<HostTensorStorage> hostStorage() {
        return Optional.ofNullable(hostStorage);
    }

    /**
     * Validates and atomically replaces the borrowed host-storage association.
     *
     * <p>Validation completes before assignment, so failure preserves the previous exact
     * association. The proposed storage must match descriptor data type. A resolved layout
     * requires capacity at least its referenced element span; unresolved layout skips capacity
     * validation. Read-only storage is accepted, while storage already dead at the point-in-time
     * liveness check is rejected. Successful attachment does not transfer ownership, retain or
     * close a scope, guarantee future liveness, or synchronize underlying memory access. The
     * synchronized method makes validation and reference replacement atomic with respect to the
     * other synchronized storage methods only.</p>
     *
     * @param hostStorage non-null live borrowed storage to retain by exact reference
     * @return a non-null optional containing the exact previous reference, or empty when there was
     *     no previous association; the result is a snapshot
     * @throws NullPointerException if {@code hostStorage} is {@code null}, with message
     *     {@code hostStorage}
     * @throws IllegalArgumentException if data type differs from the descriptor, with message
     *     {@code hostStorage data type must match descriptor data type: expected=<expected>,
     *     actual=<actual>}, or resolved layout span exceeds capacity, with message
     *     {@code hostStorage element capacity is smaller than resolved layout span:
     *     required=<required>, actual=<actual>}
     * @throws IllegalStateException if {@code hostStorage} is not alive at the attachment check,
     *     with message {@code hostStorage must be alive when attached}
     */
    public synchronized Optional<HostTensorStorage> replaceHostStorage(
            HostTensorStorage hostStorage) {
        Objects.requireNonNull(hostStorage, "hostStorage");
        validateHostStorage(hostStorage);
        Optional<HostTensorStorage> previous = Optional.ofNullable(this.hostStorage);
        this.hostStorage = hostStorage;
        return previous;
    }

    /**
     * Atomically clears the borrowed host-storage association.
     *
     * <p>Clearing is valid for live or dead storage and never closes, releases, copies, or mutates
     * the borrowed storage, whether writable or read-only. The returned optional is a snapshot
     * containing the exact previous reference; later association changes do not mutate it.
     * Clearing performs no liveness check, so a caller-owned scope may close before or during the
     * call. Synchronization makes this reference transition atomic with respect to the other two
     * storage methods only; it does not coordinate other tensors that may share the same storage,
     * raw-memory access, scope closure, or thread accessibility.</p>
     *
     * @return a non-null optional containing the exact removed storage reference, or empty when
     *     the tensor was already storage-free
     */
    public synchronized Optional<HostTensorStorage> clearHostStorage() {
        Optional<HostTensorStorage> previous = Optional.ofNullable(hostStorage);
        hostStorage = null;
        return previous;
    }

    /**
     * Returns stable metadata-only diagnostic text.
     *
     * <p>The text contains the tensor identity, descriptor, and normalized label. It deliberately
     * omits provenance, operation and input expansion, storage presence, implementation identity,
     * addresses, contents, liveness, graph state, and runtime facts, so origin and storage do not
     * destabilize it. The format is not serialization.</p>
     *
     * @return non-null stable diagnostic text for this tensor's immutable metadata
     */
    @Override
    public String toString() {
        return "Tensor["
                + "id=" + id
                + ", descriptor=" + descriptor
                + ", label=" + label
                + ']';
    }

    /**
     * Normalizes optional diagnostic text without changing absence semantics.
     *
     * @param label non-null optional whose present text is stripped
     * @return non-null empty optional or an optional containing normalized non-blank text
     * @throws IllegalArgumentException if present text is blank after stripping, with message
     *     {@code label must not be blank}
     */
    private static Optional<String> normalizeLabel(Optional<String> label) {
        if (label.isEmpty()) {
            return Optional.empty();
        }
        String normalized = label.orElseThrow().strip();
        if (normalized.isEmpty()) {
            throw new IllegalArgumentException("label must not be blank");
        }
        return Optional.of(normalized);
    }

    /**
     * Validates a proposed borrowed storage association in deterministic compatibility order.
     *
     * <p>Data type is compared first by enum identity. For resolved geometry, capacity is then
     * compared with the complete referenced element span, covering scalar, zero-sized, offset,
     * strided, and broadcast layouts. Capacity may exceed the span for shared views. Static and
     * dynamic unresolved layouts skip capacity comparison because their physical geometry is
     * unknown. Point-in-time liveness is checked last and cannot guarantee later access.</p>
     *
     * @param hostStorage non-null proposed storage; ownership remains with its caller
     * @throws IllegalArgumentException if data type differs from the descriptor, with message
     *     {@code hostStorage data type must match descriptor data type: expected=<expected>,
     *     actual=<actual>}, or resolved layout span exceeds capacity, with message
     *     {@code hostStorage element capacity is smaller than resolved layout span:
     *     required=<required>, actual=<actual>}
     * @throws IllegalStateException if storage is not alive, with message
     *     {@code hostStorage must be alive when attached}
     */
    private void validateHostStorage(HostTensorStorage hostStorage) {
        if (hostStorage.dataType() != descriptor.dataType()) {
            throw new IllegalArgumentException(
                    "hostStorage data type must match descriptor data type: expected="
                            + descriptor.dataType()
                            + ", actual="
                            + hostStorage.dataType());
        }

        descriptor.layout().ifPresent(layout -> {
            long required = layout.referencedElementSpan();
            long actual = hostStorage.elementCapacity();
            if (actual < required) {
                throw new IllegalArgumentException(
                        "hostStorage element capacity is smaller than resolved layout span: required="
                                + required
                                + ", actual="
                                + actual);
            }
        });

        if (!hostStorage.isAlive()) {
            throw new IllegalStateException("hostStorage must be alive when attached");
        }
    }
}
