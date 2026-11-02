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
        return applyBinaryChecked(left, right, kind, null);
    }

    /**
     * Shares ordinary semantic guards with descriptor-only scan validation. No Tensor is created
     * when the synchronous, unretained sink is non-null; null preserves ordinary construction.
     * Only the null-sink path consumes IDs or can encounter allocator exhaustion.
     * @param left non-null ordered left {@code BOOL} input retained by exact reference in provenance
     * @param right non-null ordered right {@code BOOL} input retained by exact reference in provenance
     * @param kind non-null binary logical kind; must be {@link BooleanLogicalKind#AND} or
     *     {@link BooleanLogicalKind#OR}
     * @param descriptorCheck optional consumer of validated semantics and ordered descriptors
     * @return null after descriptor-only completion, otherwise the ordinary canonical result
     * @throws NullPointerException if {@code left}, {@code right}, or {@code kind} is null, checked
     *     in that order with the parameter name as the message
     * @throws IllegalArgumentException if the supplied descriptor check rejects metadata, or if {@code kind} is {@code NOT}, if either input is not
     *     {@code BOOL}, checked left before right, or if the shapes cannot be broadcast under the
     *     local shape contract
     * @throws IllegalStateException if tensor identifier space is exhausted after local model
     *     values have been constructed
     */
    static Tensor applyBinaryChecked(Tensor left, Tensor right, BooleanLogicalKind kind,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck) {
        return applyBinaryDescriptors(
                left == null ? null : left.descriptor(),
                right == null ? null : right.descriptor(),
                kind,
                descriptorCheck,
                left,
                right);
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
        return applyUnaryChecked(input, kind, null);
    }

    /**
     * Shares ordinary semantic guards with descriptor-only scan validation. No Tensor is created
     * when the synchronous, unretained sink is non-null; null preserves ordinary construction.
     * Only the null-sink path consumes IDs or can encounter allocator exhaustion.
     * @param input non-null {@code BOOL} input retained by exact reference in provenance
     * @param kind non-null unary logical kind; must be {@link BooleanLogicalKind#NOT}
     * @param descriptorCheck optional consumer of validated semantics and ordered descriptors
     * @return null after descriptor-only completion, otherwise the ordinary canonical result
     * @throws NullPointerException if {@code input} or {@code kind} is null, checked in that order
     *     with the parameter name as the message
     * @throws IllegalArgumentException if the supplied descriptor check rejects metadata, or if {@code kind} is {@code AND} or {@code OR}, or if the
     *     input is not {@code BOOL}
     * @throws IllegalStateException if tensor identifier space is exhausted after local model
     *     values have been constructed
     */
    static Tensor applyUnaryChecked(Tensor input, BooleanLogicalKind kind,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck) {
        return applyUnaryDescriptors(
                input == null ? null : input.descriptor(),
                kind,
                descriptorCheck,
                input);
    }


    /**
     * Shares ordinary semantic guards with descriptor-only scan validation. No Tensor is created
     * when the synchronous, unretained sink is non-null; null preserves ordinary construction.
     * Only the null-sink path consumes IDs or can encounter allocator exhaustion.
     * @param shape non-null immutable result shape selected by the validated entry; retained
     *     exactly by the descriptor
     * @param kind non-null validated logical kind retained exactly by the operation
     * @param inputs ordered real input list, unused when a descriptor sink is supplied; references are
     *     snapshotted by provenance; the list is not mutated
     * @param descriptorCheck optional consumer of validated semantics and ordered descriptors
     * @return null after descriptor-only completion, otherwise the ordinary canonical result
     * @throws NullPointerException if {@code shape}, {@code kind}, or {@code inputs} is null, or
     *     if {@code inputs} contains a null Tensor reference; downstream model contracts report
     *     the failing component
     * @throws IllegalStateException if tensor identifier space is exhausted
     * @throws IllegalArgumentException if the supplied descriptor check rejects metadata
     */
    private static Tensor createChecked(
            Shape shape,
            BooleanLogicalKind kind,
            List<Tensor> inputs,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck) {
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.BOOL, shape, Optional.empty(), false);
        Operation operation = new Operation(kind, NoOperationAttrs.INSTANCE);
        if (descriptorCheck != null) {
            descriptorCheck.accept(operation, List.of(descriptor));
            return null;
        }
        return TensorFactory.createDerived(descriptor, Optional.empty(), operation, inputs);
    }

    /**
     * Completes ordinary checked semantics synchronously without creating Tensor identities.
     * The same descriptor guard core serves real construction; the sink is never retained.
     * @param operation non-null ordinary operation selecting this family and its attributes
     * @param inputs non-null ordered operand descriptors, including repeated occurrences
     * @param outputCount exact canonical sibling count
     * @param sink non-null consumer called exactly once with checked semantics and all outputs
     * @throws NullPointerException if a required reference is null
     * @throws IllegalArgumentException if cardinality or ordinary semantic guards reject the request
     */
    static void describeChecked(Operation operation, List<TensorDescriptor> inputs, int outputCount,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> sink) {
        Objects.requireNonNull(operation, "operation");
        Objects.requireNonNull(inputs, "inputs");
        Objects.requireNonNull(sink, "sink");
        if (inputs.size() != (operation.kind() == BooleanLogicalKind.NOT ? 1 : 2) || outputCount != 1) {
            throw new IllegalArgumentException("invalid ordinary descriptor cardinality");
        }
        for (TensorDescriptor input : inputs) {
            Objects.requireNonNull(input, "input descriptor");
        }
        BooleanLogicalKind kind = (BooleanLogicalKind) operation.kind();
        if (kind == BooleanLogicalKind.NOT) {
            applyUnaryDescriptors(inputs.get(0), kind, sink, null);
        } else {
            applyBinaryDescriptors(inputs.get(0), inputs.get(1), kind, sink, null, null);
        }
    }

    /**
     * Runs the shared descriptor guards; provenance references are used only for real construction.
     * @param left descriptor supplying ordinary operand metadata
     * @param right descriptor supplying ordinary operand metadata
     * @param kind ordinary checked construction parameter
     * @param descriptorCheck synchronous, unretained sink; null selects real construction
     * @param originalLeft exact real operand, or null for descriptor completion
     * @param originalRight exact real operand, or null for descriptor completion
     * @return the real result, or null after synchronous descriptor completion
     */
    private static Tensor applyBinaryDescriptors(
            TensorDescriptor left,
            TensorDescriptor right,
            BooleanLogicalKind kind,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck,
            Tensor originalLeft,
            Tensor originalRight) {
        Objects.requireNonNull(left, "left");
        Objects.requireNonNull(right, "right");
        Objects.requireNonNull(kind, "kind");

        if (kind == BooleanLogicalKind.NOT) {
            throw new IllegalArgumentException(
                    "binary logical expression kind must be AND or OR, but was NOT");
        }
        DataType leftDataType = left.dataType();
        if (leftDataType != DataType.BOOL) {
            throw new IllegalArgumentException(
                    "left must have BOOL data type, but was " + leftDataType);
        }
        DataType rightDataType = right.dataType();
        if (rightDataType != DataType.BOOL) {
            throw new IllegalArgumentException(
                    "right must have BOOL data type, but was " + rightDataType);
        }

        Shape shape = ShapeBroadcast.broadcast(
                left.shape(), right.shape());
        return createChecked(shape, kind, (descriptorCheck == null ? List.of(originalLeft, originalRight) : null), descriptorCheck);
    }

    /**
     * Runs the shared descriptor guards; provenance references are used only for real construction.
     * @param input descriptor supplying ordinary operand metadata
     * @param kind ordinary checked construction parameter
     * @param descriptorCheck synchronous, unretained sink; null selects real construction
     * @param originalInput exact real operand, or null for descriptor completion
     * @return the real result, or null after synchronous descriptor completion
     */
    private static Tensor applyUnaryDescriptors(
            TensorDescriptor input,
            BooleanLogicalKind kind,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck,
            Tensor originalInput) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(kind, "kind");

        if (kind != BooleanLogicalKind.NOT) {
            throw new IllegalArgumentException(
                    "unary logical expression kind must be NOT, but was " + kind);
        }
        DataType dataType = input.dataType();
        if (dataType != DataType.BOOL) {
            throw new IllegalArgumentException(
                    "input must have BOOL data type, but was " + dataType);
        }

        Shape shape = input.shape();
        return createChecked(shape, kind, (descriptorCheck == null ? List.of(originalInput) : null), descriptorCheck);
    }
}
