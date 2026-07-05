package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind;
import io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind;
import io.github.pho001.synaptik.model.operation.elementwise.logical.BooleanLogicalKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
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
 * clearing, or later storage death. It is not graph-local node or value identity and does not
 * make a tensor an intermediate-representation node. Binary arithmetic, binary comparison,
 * boolean logical, conditional selection, explicit cast, numeric and boolean aggregate reduction,
 * parameterized scalar, and unary
 * elementwise expression methods create fresh storage-free tensors whose immutable provenance
 * records the requested semantics and exact inputs; they do not execute mathematics, validate
 * numerical domains, create gradient rules, or capture a graph.
 * Binary arithmetic methods promote floating operands, broadcast shapes, and combine gradient
 * eligibility by logical OR. Binary comparison methods validate the same floating compatibility
 * and broadcasting contracts but produce non-differentiable {@code BOOL} descriptors. Boolean
 * logical methods accept only {@code BOOL}: conjunction and disjunction broadcast ordered inputs,
 * while negation retains the exact input shape. Their results are also non-differentiable
 * {@code BOOL} descriptors. Conditional selection accepts one {@code BOOL} condition and two
 * floating branches, promotes the branch type, composes two pairwise broadcasts, and propagates
 * gradient eligibility from the branches only. Cast accepts every current source and target data
 * type, retains the exact input shape, and preserves a true gradient request only across a
 * floating-to-floating conversion. Numeric aggregate methods accept one floating input and reduce
 * either every axis to a scalar or one normalized axis, optionally retaining it with extent one;
 * they preserve the exact input type and gradient eligibility without aggregating values. Boolean
 * aggregate methods require exact BOOL input and construct non-differentiable BOOL
 * results with the same full- or single-axis shape rules, without inspecting truth values or
 * defining empty-domain identities.
 * Scalar and unary methods accept one floating input and retain its exact data type, shape
 * reference, and gradient eligibility. Scalar methods retain their exact binary64 parameters in
 * typed attributes.
 * Every expression result leaves layout unresolved, has a fresh factory identity and no label or
 * storage, and records an exact matching
 * {@link BinaryArithmeticKind}, {@link BinaryComparisonKind}, {@link BooleanLogicalKind},
 * {@link WhereSelectionKind}, {@link CastKind}, {@link AggregateReductionKind},
 * {@link ScalarElementwiseKind}, or {@link UnaryElementwiseKind}.
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
     * stripped and must remain non-blank; then present storage is checked for matching data type,
     * sufficient capacity when layout geometry is resolved, and point-in-time liveness. A static
     * or dynamic unresolved layout performs no capacity check because this class does not invent
     * row-major geometry. Resolved capacity uses the complete referenced element span, including
     * offset and striding; scalar span is one and zero-sized span is zero.</p>
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
     *     result contains the exact immutable provenance reference and remains independent of
     *     storage mutation
     * @param hostStorage non-null optional borrowed host storage to retain exactly when present;
     *     read-only storage is accepted
     * @throws NullPointerException if {@code id}, {@code descriptor}, {@code label},
     *     {@code provenance}, or {@code hostStorage} is {@code null}, with the corresponding
     *     parameter name as the message
     * @throws IllegalArgumentException if a present label is blank, with message
     *     {@code label must not be blank}; if storage data type differs from the descriptor, with
     *     message {@code hostStorage data type must match descriptor data type:
     *     expected=<expected>, actual=<actual>}; or if resolved layout span exceeds storage
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
     * <p>Both operands must have floating data types, which are promoted through the model's
     * floating hierarchy, and their shapes must support locally provable right-aligned
     * broadcasting. The fresh result has unresolved layout, gradient eligibility equal to the
     * logical OR of the operand requests, no label or host storage, and provenance containing
     * {@link BinaryArithmeticKind#ADD}, {@code NoOperationAttrs.INSTANCE}, and ordered inputs
     * {@code [this, right]}. This method constructs semantics only; numerical execution and
     * gradient rules are deferred.</p>
     *
     * @param right non-null ordered right addend; it is retained by exact reference in result
     *     provenance and is not mutated
     * @return a non-null fresh derived tensor with promoted type, broadcast shape, unresolved
     *     layout, propagated gradient eligibility, and no storage
     * @throws NullPointerException if {@code right} is null, with message {@code right}
     * @throws IllegalArgumentException if either operand is not floating or their shapes cannot
     *     be broadcast under the local shape contract
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor add(Tensor right) {
        return TensorBinaryExpressions.apply(this, right, BinaryArithmeticKind.ADD);
    }

    /**
     * Builds an elementwise expression that subtracts {@code right} from this left operand.
     *
     * <p>Both operands must have floating data types, which are promoted through the model's
     * floating hierarchy, and their shapes must support locally provable right-aligned
     * broadcasting. The fresh result has unresolved layout, gradient eligibility equal to the
     * logical OR of the operand requests, no label or host storage, and provenance containing
     * {@link BinaryArithmeticKind#SUB}, {@code NoOperationAttrs.INSTANCE}, and ordered inputs
     * {@code [this, right]}. This method constructs semantics only; numerical execution and
     * gradient rules are deferred.</p>
     *
     * @param right non-null ordered subtrahend; it is retained by exact reference in result
     *     provenance and is not mutated
     * @return a non-null fresh derived tensor with promoted type, broadcast shape, unresolved
     *     layout, propagated gradient eligibility, and no storage
     * @throws NullPointerException if {@code right} is null, with message {@code right}
     * @throws IllegalArgumentException if either operand is not floating or their shapes cannot
     *     be broadcast under the local shape contract
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor sub(Tensor right) {
        return TensorBinaryExpressions.apply(this, right, BinaryArithmeticKind.SUB);
    }

    /**
     * Builds an elementwise expression that multiplies this left operand by {@code right}.
     *
     * <p>Both operands must have floating data types, which are promoted through the model's
     * floating hierarchy, and their shapes must support locally provable right-aligned
     * broadcasting. The fresh result has unresolved layout, gradient eligibility equal to the
     * logical OR of the operand requests, no label or host storage, and provenance containing
     * {@link BinaryArithmeticKind#MUL}, {@code NoOperationAttrs.INSTANCE}, and ordered inputs
     * {@code [this, right]}. This method constructs semantics only; numerical execution and
     * gradient rules are deferred.</p>
     *
     * @param right non-null ordered right factor; it is retained by exact reference in result
     *     provenance and is not mutated
     * @return a non-null fresh derived tensor with promoted type, broadcast shape, unresolved
     *     layout, propagated gradient eligibility, and no storage
     * @throws NullPointerException if {@code right} is null, with message {@code right}
     * @throws IllegalArgumentException if either operand is not floating or their shapes cannot
     *     be broadcast under the local shape contract
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor mul(Tensor right) {
        return TensorBinaryExpressions.apply(this, right, BinaryArithmeticKind.MUL);
    }

    /**
     * Builds an elementwise expression that divides this left operand by {@code right}.
     *
     * <p>Both operands must have floating data types, which are promoted through the model's
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
     * @throws IllegalArgumentException if either operand is not floating or their shapes cannot
     *     be broadcast under the local shape contract
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor div(Tensor right) {
        return TensorBinaryExpressions.apply(this, right, BinaryArithmeticKind.DIV);
    }

    /**
     * Builds an elementwise expression that selects the minimum of this left operand and
     * {@code right}.
     *
     * <p>Both operands must have floating data types, which are promoted through the model's
     * floating hierarchy, and their shapes must support locally provable right-aligned
     * broadcasting. The fresh result has unresolved layout, gradient eligibility equal to the
     * logical OR of the operand requests, no label or host storage, and provenance containing
     * {@link BinaryArithmeticKind#MIN}, {@code NoOperationAttrs.INSTANCE}, and ordered inputs
     * {@code [this, right]}. This method constructs semantics only; numerical execution and
     * gradient rules are deferred.</p>
     *
     * @param right non-null ordered right minimum operand; it is retained by exact reference in
     *     result provenance and is not mutated
     * @return a non-null fresh derived tensor with promoted type, broadcast shape, unresolved
     *     layout, propagated gradient eligibility, and no storage
     * @throws NullPointerException if {@code right} is null, with message {@code right}
     * @throws IllegalArgumentException if either operand is not floating or their shapes cannot
     *     be broadcast under the local shape contract
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor min(Tensor right) {
        return TensorBinaryExpressions.apply(this, right, BinaryArithmeticKind.MIN);
    }

    /**
     * Builds an elementwise expression that selects the maximum of this left operand and
     * {@code right}.
     *
     * <p>Both operands must have floating data types, which are promoted through the model's
     * floating hierarchy, and their shapes must support locally provable right-aligned
     * broadcasting. The fresh result has unresolved layout, gradient eligibility equal to the
     * logical OR of the operand requests, no label or host storage, and provenance containing
     * {@link BinaryArithmeticKind#MAX}, {@code NoOperationAttrs.INSTANCE}, and ordered inputs
     * {@code [this, right]}. This method constructs semantics only; numerical execution and
     * gradient rules are deferred.</p>
     *
     * @param right non-null ordered right maximum operand; it is retained by exact reference in
     *     result provenance and is not mutated
     * @return a non-null fresh derived tensor with promoted type, broadcast shape, unresolved
     *     layout, propagated gradient eligibility, and no storage
     * @throws NullPointerException if {@code right} is null, with message {@code right}
     * @throws IllegalArgumentException if either operand is not floating or their shapes cannot
     *     be broadcast under the local shape contract
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor max(Tensor right) {
        return TensorBinaryExpressions.apply(this, right, BinaryArithmeticKind.MAX);
    }

    /**
     * Builds an elementwise expression that raises this left base to the {@code right} exponent.
     *
     * <p>Both operands must have floating data types, which are promoted through the model's
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
     * @throws IllegalArgumentException if either operand is not floating or their shapes cannot
     *     be broadcast under the local shape contract
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor pow(Tensor right) {
        return TensorBinaryExpressions.apply(this, right, BinaryArithmeticKind.POW);
    }

    /**
     * Builds an elementwise expression testing whether this left operand is greater than
     * {@code right}.
     *
     * <p>Both operands must have floating data types compatible through the model promotion
     * hierarchy, and their shapes must support locally provable right-aligned broadcasting. The
     * promoted type validates the comparison domain but is not stored in the result. The fresh
     * result is {@code BOOL}, has unresolved layout and false gradient eligibility, no label or
     * host storage, and provenance containing {@link BinaryComparisonKind#GREATER_THAN},
     * {@code NoOperationAttrs.INSTANCE}, and ordered inputs {@code [this, right]}. This method
     * constructs semantics only; numerical comparison behavior, including special floating
     * values, and gradient rules are deferred.</p>
     *
     * @param right non-null ordered right comparison operand; it is retained by exact reference
     *     in result provenance and is not mutated
     * @return a non-null fresh derived {@code BOOL} tensor with broadcast shape, unresolved layout,
     *     false gradient eligibility, and no storage
     * @throws NullPointerException if {@code right} is null, with message {@code right}
     * @throws IllegalArgumentException if either operand is not floating or their shapes cannot
     *     be broadcast under the local shape contract
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor greaterThan(Tensor right) {
        return TensorComparisonExpressions.apply(this, right, BinaryComparisonKind.GREATER_THAN);
    }

    /**
     * Builds an elementwise expression testing whether this left operand is greater than or equal
     * to {@code right}.
     *
     * <p>Both operands must have floating data types compatible through the model promotion
     * hierarchy, and their shapes must support locally provable right-aligned broadcasting. The
     * promoted type validates the comparison domain but is not stored in the result. The fresh
     * result is {@code BOOL}, has unresolved layout and false gradient eligibility, no label or
     * host storage, and provenance containing {@link BinaryComparisonKind#GREATER_OR_EQUAL},
     * {@code NoOperationAttrs.INSTANCE}, and ordered inputs {@code [this, right]}. This method
     * constructs semantics only; numerical comparison behavior, including special floating
     * values, and gradient rules are deferred.</p>
     *
     * @param right non-null ordered right comparison operand; it is retained by exact reference
     *     in result provenance and is not mutated
     * @return a non-null fresh derived {@code BOOL} tensor with broadcast shape, unresolved layout,
     *     false gradient eligibility, and no storage
     * @throws NullPointerException if {@code right} is null, with message {@code right}
     * @throws IllegalArgumentException if either operand is not floating or their shapes cannot
     *     be broadcast under the local shape contract
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
     * <p>Both operands must have floating data types compatible through the model promotion
     * hierarchy, and their shapes must support locally provable right-aligned broadcasting. The
     * promoted type validates the comparison domain but is not stored in the result. The fresh
     * result is {@code BOOL}, has unresolved layout and false gradient eligibility, no label or
     * host storage, and provenance containing {@link BinaryComparisonKind#LESS_THAN},
     * {@code NoOperationAttrs.INSTANCE}, and ordered inputs {@code [this, right]}. This method
     * constructs semantics only; numerical comparison behavior, including special floating
     * values, and gradient rules are deferred.</p>
     *
     * @param right non-null ordered right comparison operand; it is retained by exact reference
     *     in result provenance and is not mutated
     * @return a non-null fresh derived {@code BOOL} tensor with broadcast shape, unresolved layout,
     *     false gradient eligibility, and no storage
     * @throws NullPointerException if {@code right} is null, with message {@code right}
     * @throws IllegalArgumentException if either operand is not floating or their shapes cannot
     *     be broadcast under the local shape contract
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor lessThan(Tensor right) {
        return TensorComparisonExpressions.apply(this, right, BinaryComparisonKind.LESS_THAN);
    }

    /**
     * Builds an elementwise expression testing whether this left operand is less than or equal to
     * {@code right}.
     *
     * <p>Both operands must have floating data types compatible through the model promotion
     * hierarchy, and their shapes must support locally provable right-aligned broadcasting. The
     * promoted type validates the comparison domain but is not stored in the result. The fresh
     * result is {@code BOOL}, has unresolved layout and false gradient eligibility, no label or
     * host storage, and provenance containing {@link BinaryComparisonKind#LESS_OR_EQUAL},
     * {@code NoOperationAttrs.INSTANCE}, and ordered inputs {@code [this, right]}. This method
     * constructs semantics only; numerical comparison behavior, including special floating
     * values, and gradient rules are deferred.</p>
     *
     * @param right non-null ordered right comparison operand; it is retained by exact reference
     *     in result provenance and is not mutated
     * @return a non-null fresh derived {@code BOOL} tensor with broadcast shape, unresolved layout,
     *     false gradient eligibility, and no storage
     * @throws NullPointerException if {@code right} is null, with message {@code right}
     * @throws IllegalArgumentException if either operand is not floating or their shapes cannot
     *     be broadcast under the local shape contract
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
     * <p>Both operands must have floating data types compatible through the model promotion
     * hierarchy, and their shapes must support locally provable right-aligned broadcasting. The
     * promoted type validates the comparison domain but is not stored in the result. The fresh
     * result is {@code BOOL}, has unresolved layout and false gradient eligibility, no label or
     * host storage, and provenance containing {@link BinaryComparisonKind#EQUAL},
     * {@code NoOperationAttrs.INSTANCE}, and ordered inputs {@code [this, right]}. Operand order is
     * retained even though equality is symmetric. This method constructs semantics only; tolerance,
     * NaN and signed-zero comparison behavior, and gradient rules are deferred.</p>
     *
     * @param right non-null ordered right comparison operand; it is retained by exact reference
     *     in result provenance and is not mutated
     * @return a non-null fresh derived {@code BOOL} tensor with broadcast shape, unresolved layout,
     *     false gradient eligibility, and no storage
     * @throws NullPointerException if {@code right} is null, with message {@code right}
     * @throws IllegalArgumentException if either operand is not floating or their shapes cannot
     *     be broadcast under the local shape contract
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor equalTo(Tensor right) {
        return TensorComparisonExpressions.apply(this, right, BinaryComparisonKind.EQUAL);
    }

    /**
     * Builds an elementwise expression testing whether this left operand compares unequal to
     * {@code right}.
     *
     * <p>Both operands must have floating data types compatible through the model promotion
     * hierarchy, and their shapes must support locally provable right-aligned broadcasting. The
     * promoted type validates the comparison domain but is not stored in the result. The fresh
     * result is {@code BOOL}, has unresolved layout and false gradient eligibility, no label or
     * host storage, and provenance containing {@link BinaryComparisonKind#NOT_EQUAL},
     * {@code NoOperationAttrs.INSTANCE}, and ordered inputs {@code [this, right]}. Operand order is
     * retained even though inequality is symmetric. This method constructs semantics only;
     * tolerance, NaN and signed-zero comparison behavior, and gradient rules are deferred.</p>
     *
     * @param right non-null ordered right comparison operand; it is retained by exact reference
     *     in result provenance and is not mutated
     * @return a non-null fresh derived {@code BOOL} tensor with broadcast shape, unresolved layout,
     *     false gradient eligibility, and no storage
     * @throws NullPointerException if {@code right} is null, with message {@code right}
     * @throws IllegalArgumentException if either operand is not floating or their shapes cannot
     *     be broadcast under the local shape contract
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
     * Builds an elementwise expression that multiplies this tensor by {@code scalar}.
     *
     * <p>This tensor must have a floating data type. The supplied binary64 value is retained
     * exactly in {@code ScalarValueAttrs}, without conversion to the input type or normalization.
     * The fresh result retains the exact input data type and shape reference, has unresolved
     * layout and unchanged gradient eligibility, no label or storage, and one-input provenance
     * containing {@link ScalarElementwiseKind#MUL}. Multiplication, special-value behavior,
     * canonicalization of zero, one, or minus one, gradients, and execution are deferred.</p>
     *
     * @param scalar binary64 multiplier retained with its exact primitive bits, including signed
     *     zero, infinity, and NaN payload bits
     * @return a non-null fresh derived tensor with preserved type, shape, and gradient eligibility
     * @throws IllegalArgumentException if this tensor's data type is not floating
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor mul(double scalar) {
        return TensorScalarExpressions.applyScalar(this, ScalarElementwiseKind.MUL, scalar);
    }

    /**
     * Builds an elementwise expression that raises this tensor to {@code exponent}.
     *
     * <p>This tensor must have a floating data type. The supplied binary64 exponent is retained
     * exactly in {@code ScalarValueAttrs}, without conversion or numerical-domain validation. The
     * fresh result retains the exact input data type and shape reference, has unresolved layout
     * and unchanged gradient eligibility, no label or storage, and one-input provenance containing
     * {@link ScalarElementwiseKind#POW}. Power evaluation, special-value policy, algebraic
     * rewrites, gradients, and execution are deferred.</p>
     *
     * @param exponent binary64 exponent retained with its exact primitive bits, including signed
     *     zero, infinity, and NaN payload bits
     * @return a non-null fresh derived tensor with preserved type, shape, and gradient eligibility
     * @throws IllegalArgumentException if this tensor's data type is not floating
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor pow(double exponent) {
        return TensorScalarExpressions.applyScalar(this, ScalarElementwiseKind.POW, exponent);
    }

    /**
     * Builds one first-class elementwise expression that clamps this tensor to the inclusive range
     * {@code [minValue, maxValue]}.
     *
     * <p>This tensor must have a floating data type, which is validated before range ordering.
     * The exact binary64 bounds are retained in one {@code ClampRangeAttrs}; equal bounds, signed
     * zeros, infinities, and NaNs are representable, while only primitive
     * {@code minValue > maxValue} is rejected. The fresh result retains the exact input data type
     * and shape reference, has unresolved layout and unchanged gradient eligibility, no label or
     * storage, and one-input provenance containing a single {@link ScalarElementwiseKind#CLAMP}
     * operation. Scalar conversion, numerical edge behavior, range simplification or expansion,
     * gradients, and execution are deferred.</p>
     *
     * @param minValue inclusive binary64 lower bound retained with its exact primitive bits
     * @param maxValue inclusive binary64 upper bound retained with its exact primitive bits
     * @return a non-null fresh derived tensor with preserved type, shape, and gradient eligibility
     * @throws IllegalArgumentException if this tensor's data type is not floating, or if
     *     {@code minValue > maxValue}
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor clamp(double minValue, double maxValue) {
        return TensorScalarExpressions.applyClamp(this, minValue, maxValue);
    }

    /**
     * Builds an elementwise expression that applies the inclusive lower bound {@code minValue}.
     *
     * <p>This tensor must have a floating data type. The exact binary64 bound is retained in
     * {@code ScalarValueAttrs}. The fresh result retains the exact input data type and shape
     * reference, has unresolved layout and unchanged gradient eligibility, no label or storage,
     * and one-input provenance containing {@link ScalarElementwiseKind#CLAMP_MIN}. Parameter
     * conversion, special-value behavior, canonicalization, gradients, and execution are
     * deferred.</p>
     *
     * @param minValue inclusive binary64 lower bound retained with its exact primitive bits
     * @return a non-null fresh derived tensor with preserved type, shape, and gradient eligibility
     * @throws IllegalArgumentException if this tensor's data type is not floating
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor clampMin(double minValue) {
        return TensorScalarExpressions.applyScalar(
                this, ScalarElementwiseKind.CLAMP_MIN, minValue);
    }

    /**
     * Builds an elementwise expression that applies the inclusive upper bound {@code maxValue}.
     *
     * <p>This tensor must have a floating data type. The exact binary64 bound is retained in
     * {@code ScalarValueAttrs}. The fresh result retains the exact input data type and shape
     * reference, has unresolved layout and unchanged gradient eligibility, no label or storage,
     * and one-input provenance containing {@link ScalarElementwiseKind#CLAMP_MAX}. Parameter
     * conversion, special-value behavior, canonicalization, gradients, and execution are
     * deferred.</p>
     *
     * @param maxValue inclusive binary64 upper bound retained with its exact primitive bits
     * @return a non-null fresh derived tensor with preserved type, shape, and gradient eligibility
     * @throws IllegalArgumentException if this tensor's data type is not floating
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor clampMax(double maxValue) {
        return TensorScalarExpressions.applyScalar(
                this, ScalarElementwiseKind.CLAMP_MAX, maxValue);
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
     * and provenance containing {@link UnaryElementwiseKind#INV},
     * {@code NoOperationAttrs.INSTANCE}, and exactly this input. Zero handling, numerical domain,
     * canonicalization, gradient rules, execution, and backend support are deferred.</p>
     *
     * @return a non-null fresh derived tensor with preserved type, shape, and gradient eligibility
     * @throws IllegalArgumentException if this tensor's data type is not floating
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor inv() {
        return TensorUnaryExpressions.apply(this, UnaryElementwiseKind.INV);
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
     * Builds an elementwise strict natural-exponential expression from this tensor.
     *
     * <p>The input must be floating. The fresh result retains the exact data type and shape
     * reference, has unresolved layout and unchanged gradient eligibility, no label or storage,
     * and provenance containing the distinct strict request {@link UnaryElementwiseKind#EXP},
     * {@code NoOperationAttrs.INSTANCE}, and exactly this input. Numerical accuracy, overflow,
     * special values, gradients, execution, and backend support are deferred.</p>
     *
     * @return a non-null fresh derived tensor with preserved type, shape, and gradient eligibility
     * @throws IllegalArgumentException if this tensor's data type is not floating
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor exp() {
        return TensorUnaryExpressions.apply(this, UnaryElementwiseKind.EXP);
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
     * Builds an elementwise strict hyperbolic-tangent expression from this tensor.
     *
     * <p>The input must be floating. The fresh result retains the exact data type and shape
     * reference, has unresolved layout and unchanged gradient eligibility, no label or storage,
     * and provenance containing the distinct strict request {@link UnaryElementwiseKind#TANH},
     * {@code NoOperationAttrs.INSTANCE}, and exactly this input. Accuracy, special values,
     * gradient rules, execution, and backend support are deferred.</p>
     *
     * @return a non-null fresh derived tensor with preserved type, shape, and gradient eligibility
     * @throws IllegalArgumentException if this tensor's data type is not floating
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor tanh() {
        return TensorUnaryExpressions.apply(this, UnaryElementwiseKind.TANH);
    }

    /**
     * Builds an explicit fast approximate natural-exponential expression from this tensor.
     *
     * <p>The input must be floating. The fresh result retains the exact data type and shape
     * reference, has unresolved layout and unchanged gradient eligibility, no label or storage,
     * and provenance containing {@link UnaryElementwiseKind#FAST_EXP},
     * {@code NoOperationAttrs.INSTANCE}, and exactly this input. This is distinct from
     * {@link #exp()}; no approximation algorithm, accuracy bound, special-value behavior,
     * gradient rule, execution route, or backend availability is promised here.</p>
     *
     * @return a non-null fresh derived tensor with preserved type, shape, and gradient eligibility
     * @throws IllegalArgumentException if this tensor's data type is not floating
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor fastExp() {
        return TensorUnaryExpressions.apply(this, UnaryElementwiseKind.FAST_EXP);
    }

    /**
     * Builds an explicit fast approximate hyperbolic-tangent expression from this tensor.
     *
     * <p>The input must be floating. The fresh result retains the exact data type and shape
     * reference, has unresolved layout and unchanged gradient eligibility, no label or storage,
     * and provenance containing {@link UnaryElementwiseKind#FAST_TANH},
     * {@code NoOperationAttrs.INSTANCE}, and exactly this input. This is distinct from
     * {@link #tanh()}; no approximation algorithm, accuracy bound, special-value behavior,
     * gradient rule, execution route, or backend availability is promised here.</p>
     *
     * @return a non-null fresh derived tensor with preserved type, shape, and gradient eligibility
     * @throws IllegalArgumentException if this tensor's data type is not floating
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    public Tensor fastTanh() {
        return TensorUnaryExpressions.apply(this, UnaryElementwiseKind.FAST_TANH);
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
     * <p>The input must have a floating data type. The fresh result has the canonical rank-zero
     * scalar shape, retains the exact input data type and gradient-eligibility request, leaves
     * layout unresolved, and has no label or host storage. Its provenance contains
     * {@link AggregateReductionKind#SUM}, {@code NoOperationAttrs.INSTANCE}, and exactly this
     * tensor. Scalar, static, zero-extent, and dynamic shapes are accepted without inspecting an
     * element count or defining an empty-domain value.</p>
     *
     * <p>This method records aggregate semantics only. It does not read or sum values, choose
     * accumulation precision or order, create a gradient rule, capture a graph, or execute work.
     * Gradient eligibility therefore does not promise that differentiation is available.</p>
     *
     * @return a non-null fresh storage-free scalar tensor with unchanged floating data type and
     *     gradient eligibility, unresolved layout, and exact one-input provenance
     * @throws IllegalArgumentException if this tensor's data type is not floating, with a message
     *     containing the rejected data type; no Tensor identity is consumed
     * @throws IllegalStateException if tensor identifier space is exhausted after local immutable
     *     expression metadata has been constructed
     */
    public Tensor sum() {
        return TensorReductionExpressions.applyFull(this, AggregateReductionKind.SUM);
    }

    /**
     * Builds an expression that sums this tensor over one axis and removes that axis.
     *
     * <p>The input must be floating. {@code axis} accepts the Shape contract's positive or
     * negative indexing and is normalized exactly once against the input rank. The selected
     * dimension is removed, every unaffected immutable dimension reference is retained in order,
     * and reducing a rank-one tensor produces the canonical rank-zero scalar shape. The fresh
     * result preserves the exact input data type and gradient eligibility, leaves layout
     * unresolved, and has no label or storage. Provenance records
     * {@link AggregateReductionKind#SUM} with normalized single-axis attributes and exactly this
     * input.</p>
     *
     * <p>No values are read or summed, and no empty-domain, accumulation, gradient, compiler, or
     * execution policy is defined.</p>
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
    public Tensor sum(int axis) {
        return TensorReductionExpressions.applyAxis(
                this, AggregateReductionKind.SUM, axis, false);
    }

    /**
     * Builds an expression that sums this tensor over one axis and optionally retains it.
     *
     * <p>The input must be floating. {@code axis} is normalized exactly once using the Shape
     * contract. When {@code keepDimensions} is false, the selected axis is removed; when true, it
     * is replaced by a new static dimension of extent one. Every unaffected immutable dimension
     * reference is retained in order. The fresh result preserves the exact input data type and
     * gradient eligibility, leaves layout unresolved, has no label or storage, and records
     * {@link AggregateReductionKind#SUM}, normalized axis attributes, and this sole input.</p>
     *
     * <p>Zero and dynamic extents are accepted structurally. This method does not inspect or sum
     * values or define empty-domain, accumulation, gradient, compiler, or execution behavior.</p>
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
    public Tensor sum(int axis, boolean keepDimensions) {
        return TensorReductionExpressions.applyAxis(
                this, AggregateReductionKind.SUM, axis, keepDimensions);
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
     * <p>This method neither reads values nor defines denominator, accumulation, empty-domain,
     * numerical accuracy, gradient, compiler, or execution behavior.</p>
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
     * <p>No values are read and no denominator, empty-domain, accumulation, gradient, compiler,
     * or execution policy is defined.</p>
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
     * <p>Zero and dynamic extents remain structurally valid. No values are read, and denominator,
     * empty-domain, accumulation, gradient, compiler, and execution policies remain deferred.</p>
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
     * Builds an expression that multiplies this tensor's values over every axis.
     *
     * <p>The input must have a floating data type. The fresh result has canonical rank-zero scalar
     * shape, preserves the exact input data type and gradient-eligibility request, leaves layout
     * unresolved, and has no label or storage. Its provenance contains
     * {@link AggregateReductionKind#PROD}, {@code NoOperationAttrs.INSTANCE}, and exactly this
     * tensor. Preserving eligibility records model intent only; this method creates no product
     * gradient rule. Scalar, static, zero-extent, and dynamic shapes are accepted structurally.</p>
     *
     * <p>No values are multiplied, and multiplication order, overflow, empty-domain identity,
     * numerical accuracy, gradients, compiler behavior, and execution remain deferred.</p>
     *
     * @return a non-null fresh storage-free scalar tensor with unchanged floating data type and
     *     gradient eligibility, unresolved layout, and exact one-input provenance
     * @throws IllegalArgumentException if this tensor's data type is not floating, with a message
     *     containing the rejected data type; no Tensor identity is consumed
     * @throws IllegalStateException if tensor identifier space is exhausted after local immutable
     *     expression metadata has been constructed
     */
    public Tensor prod() {
        return TensorReductionExpressions.applyFull(this, AggregateReductionKind.PROD);
    }

    /**
     * Builds a product expression over one axis and removes that axis.
     *
     * <p>The floating input's positive or negative {@code axis} is normalized exactly once. The
     * selected dimension is removed, every unaffected dimension reference is retained in order,
     * and rank one reduces to the canonical scalar shape. The fresh result has exact input type
     * and gradient eligibility, unresolved layout, no label or storage, and provenance with
     * {@link AggregateReductionKind#PROD}, normalized axis attributes, and this input. Preserved
     * eligibility does not imply that a product gradient rule exists.</p>
     *
     * <p>No values are multiplied and no empty-domain, ordering, numerical, gradient, compiler,
     * or execution policy is defined.</p>
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
    public Tensor prod(int axis) {
        return TensorReductionExpressions.applyAxis(
                this, AggregateReductionKind.PROD, axis, false);
    }

    /**
     * Builds a product expression over one axis and optionally retains it.
     *
     * <p>The floating input's {@code axis} is normalized exactly once. A false
     * {@code keepDimensions} removes the selected axis; true replaces it by a new static extent
     * one while preserving all other dimension references. The fresh result retains exact input
     * type and gradient eligibility, has unresolved layout and no label or storage, and records
     * {@link AggregateReductionKind#PROD}, normalized axis attributes, and this sole input.
     * Preserved eligibility is model metadata and does not install a product gradient rule.</p>
     *
     * <p>Zero and dynamic extents remain structurally valid. No values are multiplied, and
     * empty-domain, multiplication-order, numerical, gradient, compiler, and execution behavior
     * remains deferred.</p>
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
    public Tensor prod(int axis, boolean keepDimensions) {
        return TensorReductionExpressions.applyAxis(
                this, AggregateReductionKind.PROD, axis, keepDimensions);
    }

    /**
     * Builds an expression that selects the minimum over every input axis.
     *
     * <p>The input must have a floating data type. The fresh result has the canonical rank-zero
     * scalar shape, preserves the exact input data type and gradient-eligibility request, leaves
     * layout unresolved, and has no label or host storage. Its provenance contains aggregate
     * {@link AggregateReductionKind#MIN}, {@code NoOperationAttrs.INSTANCE}, and exactly this
     * tensor. This aggregate operation is distinct from the two-input elementwise
     * {@link BinaryArithmeticKind#MIN} operation.</p>
     *
     * <p>Scalar, static, zero-extent, and dynamic shapes are accepted structurally. This method
     * does not inspect or compare values, define empty-domain, NaN, signed-zero, or tie behavior,
     * create an extrema gradient rule, capture a graph, or execute work. Preserving gradient
     * eligibility therefore does not promise differentiation or a tie-distribution policy.</p>
     *
     * @return a non-null fresh storage-free scalar tensor with unchanged floating data type and
     *     gradient eligibility, unresolved layout, and exact one-input provenance
     * @throws IllegalArgumentException if this tensor's data type is not floating, with a message
     *     containing the rejected data type; no Tensor identity is consumed
     * @throws IllegalStateException if tensor identifier space is exhausted after local immutable
     *     expression metadata has been constructed
     */
    public Tensor min() {
        return TensorReductionExpressions.applyFull(this, AggregateReductionKind.MIN);
    }

    /**
     * Builds a minimum expression over one axis and removes that axis.
     *
     * <p>The floating input's positive or negative {@code axis} is normalized exactly once. The
     * selected dimension is removed, every unaffected immutable dimension reference is retained
     * in order, and rank one reduces to the canonical rank-zero scalar shape. The fresh result
     * preserves the exact input data type and gradient eligibility, leaves layout unresolved, has
     * no label or storage, and records aggregate {@link AggregateReductionKind#MIN}, normalized
     * single-axis attributes, and exactly this input.</p>
     *
     * <p>No values are inspected or compared. Empty-domain, NaN, signed-zero, tie, gradient,
     * compiler, and execution behavior remain deferred. This one-input aggregate expression is
     * distinct from {@link #min(Tensor)}.</p>
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
    public Tensor min(int axis) {
        return TensorReductionExpressions.applyAxis(
                this, AggregateReductionKind.MIN, axis, false);
    }

    /**
     * Builds a minimum expression over one axis and optionally retains it.
     *
     * <p>The floating input's {@code axis} is normalized exactly once. A false
     * {@code keepDimensions} removes the selected axis; true replaces it with a new static extent
     * one while preserving every other immutable dimension reference. The fresh result retains
     * the exact input data type and gradient eligibility, has unresolved layout and no label or
     * storage, and records aggregate {@link AggregateReductionKind#MIN}, normalized axis
     * attributes, and this sole input.</p>
     *
     * <p>Zero and dynamic extents remain structurally valid. No values are inspected, and
     * empty-domain, NaN, signed-zero, tie, gradient, compiler, and execution behavior remains
     * deferred. Preserved eligibility does not install an extrema gradient rule.</p>
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
    public Tensor min(int axis, boolean keepDimensions) {
        return TensorReductionExpressions.applyAxis(
                this, AggregateReductionKind.MIN, axis, keepDimensions);
    }

    /**
     * Builds an expression that selects the maximum over every input axis.
     *
     * <p>The input must have a floating data type. The fresh result has the canonical rank-zero
     * scalar shape, preserves the exact input data type and gradient-eligibility request, leaves
     * layout unresolved, and has no label or host storage. Its provenance contains aggregate
     * {@link AggregateReductionKind#MAX}, {@code NoOperationAttrs.INSTANCE}, and exactly this
     * tensor. This aggregate operation is distinct from the two-input elementwise
     * {@link BinaryArithmeticKind#MAX} operation.</p>
     *
     * <p>Scalar, static, zero-extent, and dynamic shapes are accepted structurally. This method
     * does not inspect or compare values, define empty-domain, NaN, signed-zero, or tie behavior,
     * create an extrema gradient rule, capture a graph, or execute work. Preserving gradient
     * eligibility therefore does not promise differentiation or a tie-distribution policy.</p>
     *
     * @return a non-null fresh storage-free scalar tensor with unchanged floating data type and
     *     gradient eligibility, unresolved layout, and exact one-input provenance
     * @throws IllegalArgumentException if this tensor's data type is not floating, with a message
     *     containing the rejected data type; no Tensor identity is consumed
     * @throws IllegalStateException if tensor identifier space is exhausted after local immutable
     *     expression metadata has been constructed
     */
    public Tensor max() {
        return TensorReductionExpressions.applyFull(this, AggregateReductionKind.MAX);
    }

    /**
     * Builds a maximum expression over one axis and removes that axis.
     *
     * <p>The floating input's positive or negative {@code axis} is normalized exactly once. The
     * selected dimension is removed, every unaffected immutable dimension reference is retained
     * in order, and rank one reduces to the canonical rank-zero scalar shape. The fresh result
     * preserves the exact input data type and gradient eligibility, leaves layout unresolved, has
     * no label or storage, and records aggregate {@link AggregateReductionKind#MAX}, normalized
     * single-axis attributes, and exactly this input.</p>
     *
     * <p>No values are inspected or compared. Empty-domain, NaN, signed-zero, tie, gradient,
     * compiler, and execution behavior remain deferred. This one-input aggregate expression is
     * distinct from {@link #max(Tensor)}.</p>
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
    public Tensor max(int axis) {
        return TensorReductionExpressions.applyAxis(
                this, AggregateReductionKind.MAX, axis, false);
    }

    /**
     * Builds a maximum expression over one axis and optionally retains it.
     *
     * <p>The floating input's {@code axis} is normalized exactly once. A false
     * {@code keepDimensions} removes the selected axis; true replaces it with a new static extent
     * one while preserving every other immutable dimension reference. The fresh result retains
     * the exact input data type and gradient eligibility, has unresolved layout and no label or
     * storage, and records aggregate {@link AggregateReductionKind#MAX}, normalized axis
     * attributes, and this sole input.</p>
     *
     * <p>Zero and dynamic extents remain structurally valid. No values are inspected, and
     * empty-domain, NaN, signed-zero, tie, gradient, compiler, and execution behavior remains
     * deferred. Preserved eligibility does not install an extrema gradient rule.</p>
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
     * <p>This method does not inspect truth values or storage, define an empty-domain identity,
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
     * <p>This method does not inspect truth values, define an empty-domain identity, create a
     * gradient rule, capture a graph, report backend support, or execute work. Aggregate ALL is
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
     * <p>This method does not inspect truth values, define an empty-domain identity, create a
     * gradient rule, capture a graph, report backend support, or execute work. Aggregate ALL is
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
     * <p>This method does not inspect truth values or storage, define an empty-domain identity,
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
     * <p>This method does not inspect truth values, define an empty-domain identity, create a
     * gradient rule, capture a graph, report backend support, or execute work. Aggregate ANY is
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
     * <p>This method does not inspect truth values, define an empty-domain identity, create a
     * gradient rule, capture a graph, report backend support, or execute work. Aggregate ANY is
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
