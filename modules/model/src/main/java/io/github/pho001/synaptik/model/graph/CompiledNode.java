package io.github.pho001.synaptik.model.graph;

import io.github.pho001.synaptik.model.operation.Operation;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import io.github.pho001.synaptik.model.operation.scan.ScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.ScanKind;

/**
 * Describes one immutable computation occurrence in a compiled graph model.
 *
 * <p>A node applies one {@link Operation} semantic description to ordered logical input values and
 * produces one or more ordered logical output values. The node is distinct from the operation,
 * which may describe the same semantics at multiple occurrences, and from the {@link GraphValue}
 * data identified by its input and output IDs. This is compile-time model state: it does not
 * lower or execute the operation, select a backend or kernel, own storage, or enter runtime hot
 * paths.</p>
 *
 * <p>The input and output containers are immutable ordered snapshots created with
 * {@link List#copyOf(java.util.Collection)}. Empty inputs and repeated input IDs are valid because
 * a semantic source may have no inputs and separate positions may read the same value. Outputs are
 * non-empty and unique within this node. The record does not validate graph-wide value existence,
 * ID uniqueness across nodes, producers, topology, or descriptors. The supplied operation already
 * owns kind-to-attributes validation, and this occurrence validates its final input and output
 * counts against the derived operation signature.</p>
 *
 * <p>The bounded general-scan exception retains one {@link CompiledScanBody} and checks its
 * formal/result cardinalities against the scan header, without comparing graph-local IDs across
 * scopes. Local formals count carry + sequence + invariant, excluding the optional lengths actual;
 * this structural acceptance does not grant length-mode compilation or execution.
 * The four-argument constructor supplies an empty attachment for existing ordinary and
 * fixed operations. Neither constructor captures a source body or infers its descriptors.</p>
 *
 * <p>Record-generated equality and hashing use all five components, including list order and
 * repeated input positions. The generated {@link #toString()} exposes diagnostic component values
 * but is not a serialization, parsing, execution-dispatch, or graph-validation format.</p>
 *
 * @param id non-null graph-local identity of this computation occurrence; the exact immutable
 *     reference is retained
 * @param operation non-null immutable backend-independent semantics applied by this occurrence;
 *     the exact reference is retained
 * @param inputs non-null ordered input value IDs; every element must be non-null, and an immutable
 *     snapshot preserves empty and repeated positions without promising list-container identity
 * @param outputs non-null, non-empty ordered output value IDs; every element must be non-null and
 *     unique within this node, and an immutable snapshot is retained without promising
 *     list-container identity
 * @param scanBody non-null optional independently scoped compiled body; present exactly for a
 *     general scan, empty for ordinary and fixed operations
 */
public record CompiledNode(
        NodeId id,
        Operation operation,
        List<ValueId> inputs,
        List<ValueId> outputs,
        Optional<CompiledScanBody> scanBody) {
    /**
     * Compatible ordinary/fixed-node constructor with no contained body.
     * @param id non-null graph-local identity
     * @param operation non-null ordinary or fixed semantics
     * @param inputs non-null ordered input IDs, possibly repeated
     * @param outputs non-null non-empty unique ordered output IDs
     * @throws NullPointerException if a required reference or element is null
     * @throws IllegalArgumentException if occurrence validation fails or a general scan lacks its body
     */
    public CompiledNode(NodeId id, Operation operation, List<ValueId> inputs, List<ValueId> outputs) {
        this(id, operation, inputs, outputs, Optional.empty());
    }
    /**
     * Creates a computation occurrence and snapshots its ordered value relationships.
     *
     * <p>All component references and list elements must be non-null. Inputs are inspected in
     * encounter order and then copied with {@link List#copyOf(java.util.Collection)}; they may be
     * empty and may repeat an ID. Outputs must contain at least one value, are inspected in
     * encounter order for nulls and the first later duplicate, and are then copied with
     * {@code List.copyOf}. Both stored containers have list value semantics, so callers must rely
     * on content and order rather than identity with a supplied or otherwise equal list.</p>
     *
     * <p>Validation is local to this record. In particular, an input may also be an output, IDs
     * need not be resolvable without an owning graph, and another separately constructed node may
     * claim the same output. The owning {@link CompiledGraphModel} validates graph-wide structure,
     * while the operation signature owns only local kind/attributes and occurrence-count
     * compatibility.</p>
     *
     * @param id non-null graph-local node identity to retain exactly
     * @param operation non-null immutable operation semantics to retain exactly
     * @param inputs non-null ordered input IDs to snapshot; elements must be non-null, while empty
     *     and repeated inputs are permitted
     * @param outputs non-null, non-empty ordered output IDs to snapshot; elements must be non-null
     *     and unique within this node
     * @param scanBody non-null optional compiled body, present exactly for general scans
     * @throws NullPointerException if {@code id}, {@code operation}, {@code inputs}, or
     *     {@code outputs} or {@code scanBody} is {@code null}; the message is the corresponding
     *     component name
     * @throws NullPointerException if an input element is {@code null}; the message is
     *     {@code inputs[index]} with its zero-based encounter index
     * @throws NullPointerException if an output element is {@code null}; the message is
     *     {@code outputs[index]} with its zero-based encounter index
     * @throws IllegalArgumentException if {@code outputs} is empty; the message is
     *     {@code outputs must not be empty}
     * @throws IllegalArgumentException if a later output repeats an earlier output ID; the message
     *     is {@code outputs[index] duplicates ValueId[value=n]} with the later zero-based encounter
     *     index and the duplicate ID's diagnostic value
     * @throws IllegalArgumentException if the final input or output count is outside the supplied
     *     operation's accepted signature range; if a general scan lacks a compiled body or its
     *     scalar header disagrees with the outer/formal/role cardinalities; or if ordinary/fixed
     *     semantics receive a compiled-body attachment. No descriptor inference or cross-scope
     *     numeric-ID comparison is performed
     */
    public CompiledNode {
        Objects.requireNonNull(id, "id");
        Objects.requireNonNull(operation, "operation");
        Objects.requireNonNull(inputs, "inputs");
        Objects.requireNonNull(outputs, "outputs");
        Objects.requireNonNull(scanBody, "scanBody");

        for (int index = 0; index < inputs.size(); index++) {
            Objects.requireNonNull(inputs.get(index), "inputs[" + index + "]");
        }
        inputs = List.copyOf(inputs);

        if (outputs.isEmpty()) {
            throw new IllegalArgumentException("outputs must not be empty");
        }
        var observedOutputs = new HashSet<ValueId>();
        for (int index = 0; index < outputs.size(); index++) {
            ValueId output = Objects.requireNonNull(outputs.get(index), "outputs[" + index + "]");
            if (!observedOutputs.add(output)) {
                throw new IllegalArgumentException(
                        "outputs[" + index + "] duplicates " + output);
            }
        }
        outputs = List.copyOf(outputs);

        operation.signature().validateOccurrence(inputs.size(), outputs.size());
        if (operation.kind() == ScanKind.SCAN) {
            CompiledScanBody body = scanBody.orElseThrow(
                    () -> new IllegalArgumentException("scan node requires compiled body"));
            ScanAttrs attrs = (ScanAttrs) operation.attrs();
            if (attrs.inputCount() != inputs.size() || attrs.outputCount() != outputs.size()
                    || body.graph().inputs().size() != Math.addExact(
                            Math.addExact(attrs.carryCount(), attrs.sequenceCount()), attrs.invariantCount())
                    || body.nextCarry().size() != attrs.carryCount()
                    || body.emissions().size() != attrs.emissionCount()
                    || body.residuals().size() != attrs.residualCount()) {
                throw new IllegalArgumentException("scan node header/body cardinality mismatch");
            }
        } else if (scanBody.isPresent()) {
            throw new IllegalArgumentException("ordinary/fixed nodes cannot contain a scan body");
        }
    }

    /**
     * Returns this computation occurrence's identity within its owning graph context.
     *
     * @return the exact non-null immutable {@link NodeId} reference supplied at construction;
     *     allocation and graph-wide uniqueness are not established by this record
     */
    public NodeId id() {
        return id;
    }

    /**
     * Returns the backend-independent semantics applied by this computation occurrence.
     *
     * @return the exact non-null immutable {@link Operation} reference supplied at construction;
     *     the result describes semantics but provides no execution or backend behavior
     */
    public Operation operation() {
        return operation;
    }

    /**
     * Returns the immutable ordered snapshot of logical input identities.
     *
     * <p>The result may be empty and may contain repeated IDs at distinct semantic positions.
     * Its container cannot be mutated, preserves encounter order, and has value semantics; no
     * identity relationship with the caller's original list or another equal list is promised.
     * A mutating list operation throws {@link UnsupportedOperationException}.</p>
     *
     * @return a non-null immutable ordered snapshot of non-null input value IDs
     */
    public List<ValueId> inputs() {
        return inputs;
    }

    /**
     * Returns the immutable ordered snapshot of logical output identities.
     *
     * <p>The result contains at least one ID, contains no duplicate ID within this node, preserves
     * encounter order, and cannot be mutated. Its container has value semantics; no identity
     * relationship with the caller's original list or another equal list is promised. A mutating
     * list operation throws {@link UnsupportedOperationException}.</p>
     *
     * @return a non-null, non-empty immutable ordered snapshot of unique non-null output value IDs
     */
    public List<ValueId> outputs() {
        return outputs;
    }

    /**
     * Returns the independently scoped compiled-body attachment.
     * @return non-null optional local compiled containment, never source Tensors or runtime state
     */
    public Optional<CompiledScanBody> scanBody() {
        return scanBody;
    }
}
