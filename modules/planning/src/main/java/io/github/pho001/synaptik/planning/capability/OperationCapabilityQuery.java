package io.github.pho001.synaptik.planning.capability;

import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.CompiledScanBody;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.scan.ScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.ScanKind;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Describes one structurally valid operation occurrence for a backend capability question.
 *
 * <p>The operation supplies backend-independent semantics and the ordered descriptor lists
 * describe the occurrence's logical input and output positions. Construction snapshots list
 * membership, retains exact operation and descriptor references, and validates signature counts
 * and attachment applicability, including activity declaration positions against the Scan header.
 * It does not infer backend support from semantics or validate descriptor compatibility,
 * graph closure, backend availability, hard requirements, scoring, preparation, routing,
 * or execution.</p>
 *
 * <p>The optional compiled scan body retains its exact immutable reference, including its local
 * graph, ordered role bindings and demanded outer-port indices. A present body is legal only for
 * {@link ScanKind#SCAN}. Without activity, a missing body or header/body/descriptor mismatch remains
 * representable for provider rejection. Model owns body structural validation.
 * Compiler supplies independently validated descriptors and ordinary-use demand. Optional
 * structural activity is legal only with its exact local body graph; providers still validate
 * header/binding applicability and ordinary support independently. This query does not infer demand.</p>
 *
 * <p>Record equality, hashing, and diagnostic text use all five components.
 * The diagnostic text is not a serialization or dispatch contract.</p>
 *
 * @param operation the non-null immutable backend-independent operation semantics; retained by
 *     exact reference
 * @param inputs the non-null ordered input descriptors; every element must be non-null, and list
 *     membership is copied while descriptor references are retained
 * @param outputs the non-null ordered output descriptors; every element must be non-null, and
 *     list membership is copied while descriptor references are retained
 * @param scanBody the non-null optional immutable compiled body, retained by exact reference;
 *     may be present only for general scan semantics
 * @param scanActivityPlan non-null optional structural presence plan; when present it must name
 *     the exact supplied Scan body graph
 */
public record OperationCapabilityQuery(
        Operation operation,
        List<TensorDescriptor> inputs,
        List<TensorDescriptor> outputs,
        Optional<CompiledScanBody> scanBody,
        Optional<ScanActivityPlan> scanActivityPlan) {
    /**
     * Creates an empty-body question, preserving ordinary and fixed-operation construction.
     * General scans without bodies remain representable for provider rejection.
     *
     * @param operation non-null immutable operation semantics retained exactly
     * @param inputs non-null ordered non-null descriptors whose membership is snapshotted
     * @param outputs non-null ordered non-null descriptors whose membership is snapshotted
     * @throws NullPointerException if a reference or descriptor element is null
     * @throws IllegalArgumentException if a signature occurrence count is invalid
     * @throws IllegalStateException if the operation signature declaration is missing or malformed
     */
    public OperationCapabilityQuery(
            Operation operation, List<TensorDescriptor> inputs, List<TensorDescriptor> outputs) {
        this(operation, inputs, outputs, Optional.empty(), Optional.empty());
    }

    /**
     * Creates an immutable capability question for one structurally valid operation occurrence.
     *
     * <p>Validation checks the operation and list references in declaration order,
     * scans input elements in encounter order, snapshots the inputs, then scans and snapshots the
     * outputs. Only after both snapshots exist does the operation signature validate their
     * occurrence counts. Next the optional body reference is checked for null, then a present
     * body is restricted to {@link ScanKind#SCAN}. Thus all existing validation precedes the
     * body checks. A supplied activity plan must name the exact supplied body graph, with numeric
     * carry, sequence-seed and shared-accumulator positions applicable to the header. Next-presence
     * bindings and distinct ordered temporal ADDs are checked against that body. No descriptor
     * inference or backend qualification is performed.</p>
     *
     * @param operation the non-null immutable operation semantics; retained by exact reference
     * @param inputs the non-null ordered input descriptors; every element must be non-null, list
     *     membership is copied, and descriptor references are retained
     * @param outputs the non-null ordered output descriptors; every element must be non-null, list
     *     membership is copied, and descriptor references are retained
     * @param scanBody non-null optional immutable body, retained exactly, present only for scans
     * @param scanActivityPlan non-null optional activity for the exact attached body graph
     * @throws NullPointerException if {@code operation}, {@code inputs},
     *     {@code outputs}, {@code scanBody}, {@code scanActivityPlan}, or a descriptor element
     *     is {@code null}; null optionals identify their component, and descriptor failures
     *     identify the first null encounter as {@code inputs[index]} or {@code outputs[index]}
     * @throws IllegalArgumentException if the input or output count is outside the operation
     *     signature's inclusive accepted range, a body is present for a non-scan kind, or activity
     *     has invalid exact-body association, declaration positions, next scopes or temporal ADDs
     * @throws IllegalStateException if the operation kind's signature declaration has become
     *     missing or malformed
     */
    public OperationCapabilityQuery {
        operation = Objects.requireNonNull(operation, "operation");
        Objects.requireNonNull(inputs, "inputs");
        Objects.requireNonNull(outputs, "outputs");

        for (int index = 0; index < inputs.size(); index++) {
            Objects.requireNonNull(inputs.get(index), "inputs[" + index + "]");
        }
        inputs = List.copyOf(inputs);

        for (int index = 0; index < outputs.size(); index++) {
            Objects.requireNonNull(outputs.get(index), "outputs[" + index + "]");
        }
        outputs = List.copyOf(outputs);

        operation.signature().validateOccurrence(inputs.size(), outputs.size());
        Objects.requireNonNull(scanBody, "scanBody");
        if (scanBody.isPresent() && operation.kind() != ScanKind.SCAN) {
            throw new IllegalArgumentException("ordinary/fixed operations cannot contain a scan body");
        }
        Objects.requireNonNull(scanActivityPlan, "scanActivityPlan");
        if (scanActivityPlan.isPresent()
                && (scanBody.isEmpty()
                        || scanActivityPlan.orElseThrow().graph() != scanBody.orElseThrow().graph())) {
            throw new IllegalArgumentException("scan activity requires its exact attached body graph");
        }
        if (scanActivityPlan.isPresent()) {
            ScanActivityPlan plan = scanActivityPlan.orElseThrow();
            ScanAttrs attrs = (ScanAttrs) operation.attrs();
            CompiledScanBody body = scanBody.orElseThrow();
            if (body.nextCarry().size() != attrs.carryCount()
                    || body.emissions().size() != attrs.emissionCount()
                    || body.residuals().size() != attrs.residualCount()
                    || inputs.size() != attrs.inputCount() || outputs.size() != attrs.outputCount()
                    || plan.graph().inputs().size()
                            != attrs.carryCount() + attrs.sequenceCount() + attrs.invariantCount()) {
                throw new IllegalArgumentException("activity body/header declaration mismatch");
            }
            for (ScanActivityPlan.Carry carry : plan.carries()) {
                if (carry.numericCarryPosition() >= attrs.carryCount()) {
                    throw new IllegalArgumentException("activity carry must name a numeric carry declaration");
                }
            }
            for (int formal : plan.seedFormals()) {
                if (formal < attrs.carryCount()
                        || formal >= attrs.carryCount() + attrs.sequenceCount()) {
                    throw new IllegalArgumentException("activity seed must name a sequence formal");
                }
            }
            for (ScanActivityPlan.Accumulator accumulator : plan.accumulators()) {
                int position = accumulator.numericSharedCarryPosition();
                if (position >= attrs.carryCount()
                        || (attrs.lengthAware() && position < attrs.rowCarryCount())) {
                    throw new IllegalArgumentException("activity accumulator must name a shared numeric carry");
                }
            }
            validateActivityBindings(body, plan);
        }
    }

    private static void validateActivityBindings(CompiledScanBody body, ScanActivityPlan plan) {
        Map<ValueId, CompiledNode> producers = new HashMap<>();
        for (CompiledNode node : body.graph().nodes()) {
            for (ValueId output : node.outputs()) {
                producers.put(output, node);
            }
        }
        Map<ValueId, List<Integer>> roots = new HashMap<>();
        for (int root = 0; root < plan.carries().size(); root++) {
            roots.put(body.graph().inputs().get(plan.carries().get(root).numericCarryPosition()), List.of(root));
        }
        for (int root = 0; root < plan.seedFormals().size(); root++) {
            roots.put(body.graph().inputs().get(plan.seedFormals().get(root)),
                    List.of(plan.carries().size() + root));
        }
        for (ScanActivityPlan.Carry carry : plan.carries()) {
            ValueId next = body.nextCarry().get(carry.numericCarryPosition());
            CompiledNode producer = producers.get(next);
            List<Integer> expected = producer == null ? roots.getOrDefault(next, List.of())
                    : plan.scopes().get(plan.nodeScopes().get(producer.id()));
            if (!plan.scopes().get(carry.nextScope()).equals(expected)) {
                throw new IllegalArgumentException("activity next scope disagrees with its exact next-carry binding");
            }
        }
        for (ScanActivityPlan.Accumulator accumulator : plan.accumulators()) {
            int position = accumulator.numericSharedCarryPosition();
            CompiledNode add = producers.get(body.nextCarry().get(position));
            if (add == null || add.operation().kind() != BinaryArithmeticKind.ADD
                    || add.inputs().size() != 2 || add.outputs().size() != 1
                    || !add.inputs().getFirst().equals(body.graph().inputs().get(position))
                    || plan.merges().containsKey(add.id())
                    || !plan.scopes().get(plan.nodeScopes().get(add.id()))
                            .equals(plan.scopes().get(accumulator.contributionScope()))) {
                throw new IllegalArgumentException(
                        "activity accumulator must retain its separate ordered old-plus-contribution ADD");
            }
        }
    }

    /**
     * Returns the backend-independent semantics of this operation occurrence.
     *
     * @return the exact non-null {@link Operation} reference supplied at construction
     */
    @Override
    public Operation operation() {
        return operation;
    }

    /**
     * Returns the ordered input descriptors of this operation occurrence.
     *
     * @return the non-null immutable membership snapshot in encounter order; elements are the
     *     exact non-null descriptor references supplied at construction
     */
    @Override
    public List<TensorDescriptor> inputs() {
        return inputs;
    }

    /**
     * Returns the ordered output descriptors of this operation occurrence.
     *
     * @return the non-null immutable membership snapshot in encounter order; elements are the
     *     exact non-null descriptor references supplied at construction
     */
    @Override
    public List<TensorDescriptor> outputs() {
        return outputs;
    }

    /**
     * Returns the independently scoped body and its ordered role/demand facts, when supplied.
     *
     * @return the exact non-null optional supplied at construction, containing the exact immutable
     *     body reference when present; empty does not establish scan support
     */
    @Override
    public Optional<CompiledScanBody> scanBody() {
        return scanBody;
    }

    /**
     * Returns optional structural activity for this exact Scan occurrence's local body.
     *
     * @return non-null immutable optional; empty does not infer always-present execution
     */
    @Override
    public Optional<ScanActivityPlan> scanActivityPlan() {
        return scanActivityPlan;
    }
}
