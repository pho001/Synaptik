package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.logical.BooleanLogicalKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.ShapeBroadcast;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated storage-free boolean logical tensor expressions.
 *
 * <p>This package-private boundary exposes separate binary and unary entries so family arity is
 * validated before descriptor eligibility. Binary conjunction and disjunction require exact
 * {@code BOOL} inputs and derive one locally provable right-aligned broadcast shape. Unary
 * negation requires exact {@code BOOL} and retains the immutable input shape without broadcasting.
 * Both entries then share fixed unresolved, non-differentiable {@code BOOL} result construction,
 * exact parameterless operation composition, ordered provenance, and one central factory
 * delegation.</p>
 *
 * <p>Construction is eager only for expression metadata. It does not inspect values or storage,
 * execute truth logic, short-circuit, insert casts, canonicalize or collapse expressions,
 * propagate gradient eligibility, define gradient rules, capture a graph, or provide backend or
 * runtime behavior.</p>
 */
final class TensorLogicalExpressions {
    /** Prevents instantiation because logical expression construction is stateless. */
    private TensorLogicalExpressions() {
    }

    /**
     * Creates one fresh derived tensor for ordered binary conjunction or disjunction.
     *
     * <p>Validation and construction occur in this exact order: null-check {@code left},
     * {@code right}, and {@code kind}; reject {@link BooleanLogicalKind#NOT}; validate exact
     * {@code BOOL} data type first for {@code left} and then for {@code right}; invoke
     * {@link ShapeBroadcast#broadcast(Shape, Shape)} exactly once; and invoke the common result
     * constructor exactly once with ordered inputs {@code [left, right]}. Failures before central
     * factory delegation allocate no Tensor identity. Neither input, its metadata, provenance,
     * label, storage association, nor storage contents are mutated.</p>
     *
     * @param left non-null ordered left {@code BOOL} input retained by exact reference in provenance
     * @param right non-null ordered right {@code BOOL} input retained by exact reference in provenance
     * @param kind non-null binary logical kind; must be {@link BooleanLogicalKind#AND} or
     *     {@link BooleanLogicalKind#OR}
     * @return the non-null exact fresh derived tensor returned by the central factory
     * @throws NullPointerException if {@code left}, {@code right}, or {@code kind} is null, checked
     *     in that order with the parameter name as the message
     * @throws IllegalArgumentException if {@code kind} is {@code NOT}, if either input is not
     *     {@code BOOL}, checked left before right, or if the shapes cannot be broadcast under the
     *     local shape contract
     * @throws IllegalStateException if tensor identifier space is exhausted after local model
     *     values have been constructed
     */
    static Tensor applyBinary(Tensor left, Tensor right, BooleanLogicalKind kind) {
        Objects.requireNonNull(left, "left");
        Objects.requireNonNull(right, "right");
        Objects.requireNonNull(kind, "kind");

        if (kind == BooleanLogicalKind.NOT) {
            throw new IllegalArgumentException(
                    "binary logical expression kind must be AND or OR, but was NOT");
        }
        DataType leftDataType = left.descriptor().dataType();
        if (leftDataType != DataType.BOOL) {
            throw new IllegalArgumentException(
                    "left must have BOOL data type, but was " + leftDataType);
        }
        DataType rightDataType = right.descriptor().dataType();
        if (rightDataType != DataType.BOOL) {
            throw new IllegalArgumentException(
                    "right must have BOOL data type, but was " + rightDataType);
        }

        Shape shape = ShapeBroadcast.broadcast(
                left.descriptor().shape(), right.descriptor().shape());
        return create(shape, kind, List.of(left, right));
    }

    /**
     * Creates one fresh derived tensor for unary boolean negation.
     *
     * <p>Validation and construction occur in this exact order: null-check {@code input} and
     * {@code kind}; reject {@link BooleanLogicalKind#AND} or {@link BooleanLogicalKind#OR};
     * validate exact {@code BOOL} data type; retain the exact input {@link Shape} reference without
     * broadcasting or reconstruction; and invoke the common result constructor exactly once with
     * input list {@code [input]}. Failures before central factory delegation allocate no Tensor
     * identity. The input and all of its metadata and storage remain unchanged.</p>
     *
     * @param input non-null {@code BOOL} input retained by exact reference in provenance
     * @param kind non-null unary logical kind; must be {@link BooleanLogicalKind#NOT}
     * @return the non-null exact fresh derived tensor returned by the central factory
     * @throws NullPointerException if {@code input} or {@code kind} is null, checked in that order
     *     with the parameter name as the message
     * @throws IllegalArgumentException if {@code kind} is {@code AND} or {@code OR}, or if the
     *     input is not {@code BOOL}
     * @throws IllegalStateException if tensor identifier space is exhausted after local model
     *     values have been constructed
     */
    static Tensor applyUnary(Tensor input, BooleanLogicalKind kind) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(kind, "kind");

        if (kind != BooleanLogicalKind.NOT) {
            throw new IllegalArgumentException(
                    "unary logical expression kind must be NOT, but was " + kind);
        }
        DataType dataType = input.descriptor().dataType();
        if (dataType != DataType.BOOL) {
            throw new IllegalArgumentException(
                    "input must have BOOL data type, but was " + dataType);
        }

        Shape shape = input.descriptor().shape();
        return create(shape, kind, List.of(input));
    }

    /**
     * Constructs the fixed logical result after entry-specific validation is complete.
     *
     * <p>Construction occurs in this exact order: create one unresolved-layout {@code BOOL}
     * descriptor from the exact supplied shape with false gradient eligibility; create one
     * {@link Operation} from the exact kind and {@link NoOperationAttrs#INSTANCE}; create one
     * {@link TensorProvenance} that snapshots the supplied ordered inputs; and delegate exactly
     * once to {@link TensorFactory#createDerived(TensorDescriptor, Optional, Operation, List)}
     * with no label. The derived-construction seam attaches no storage. This method performs no
     * duplicate arity, data-type, or shape validation.</p>
     *
     * @param shape non-null immutable result shape selected by the validated entry; retained
     *     exactly by the descriptor
     * @param kind non-null validated logical kind retained exactly by the operation
     * @param inputs non-null ordered validated input list whose exact Tensor references are
     *     snapshotted by provenance; the list is not mutated
     * @return the non-null exact fresh, unlabeled, storage-free tensor returned by the central
     *     factory
     * @throws NullPointerException if {@code shape}, {@code kind}, or {@code inputs} is null, or
     *     if {@code inputs} contains a null Tensor reference; downstream model contracts report
     *     the failing component
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    private static Tensor create(
            Shape shape,
            BooleanLogicalKind kind,
            List<Tensor> inputs) {
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.BOOL, shape, Optional.empty(), false);
        Operation operation = new Operation(kind, NoOperationAttrs.INSTANCE);
        return TensorFactory.createDerived(descriptor, Optional.empty(), operation, inputs);
    }
}
