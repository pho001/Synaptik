package io.github.pho001.synaptik.backend.metal;

import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Immutable typed operation table for the version-one Metal MPSGraph executable schema.
 *
 * <p>Each node has one exact operation kind, ordered input value indices, and one output value
 * index. Unary {@code NEG} carries no second input; binary operations retain left then right
 * operand order. The schema has no attribute payload, operation name, generic parameter map, or
 * executable state. Native marshalling uses one fixed four-{@code uint32_t} record per node.</p>
 */
final class MetalMpsGraphProgram {
    /** Exact schema version carried across native ABI version four. */
    static final int SCHEMA_VERSION = 1;
    /** Fixed number of unsigned 32-bit cells in one native node record. */
    static final int NODE_RECORD_WIDTH = 4;
    /** Unsigned {@code UINT32_MAX} sentinel for the absent unary second input. */
    static final int NO_SECOND_INPUT = -1;

    /** Closed operation vocabulary and stable schema-local wire identities. */
    enum NodeKind {
        /** Unary FLOAT32 negation. */
        NEG(1, 1),
        /** Ordered FLOAT32 addition. */
        ADD(2, 2),
        /** Ordered FLOAT32 subtraction. */
        SUB(3, 2),
        /** Ordered FLOAT32 multiplication. */
        MUL(4, 2),
        /** Ordered FLOAT32 division. */
        DIV(5, 2);

        private final int wireIdentity;
        private final int inputCount;

        NodeKind(int wireIdentity, int inputCount) {
            this.wireIdentity = wireIdentity;
            this.inputCount = inputCount;
        }

        /** @return stable positive version-one ABI identity */
        int wireIdentity() {
            return wireIdentity;
        }

        /** @return exact semantic input count, either one or two */
        int inputCount() {
            return inputCount;
        }

        /**
         * Resolves an untrusted version-one wire identity.
         *
         * @param wireIdentity raw unsigned 32-bit identity represented as a Java {@code int}
         * @return matching typed kind, or empty for an unknown identity
         */
        static Optional<NodeKind> fromWireIdentity(int wireIdentity) {
            for (NodeKind kind : values()) {
                if (kind.wireIdentity == wireIdentity) {
                    return Optional.of(kind);
                }
            }
            return Optional.empty();
        }
    }

    /**
     * One fixed-width typed node record.
     *
     * @param kind non-null exact operation kind
     * @param firstInputIndex non-negative ordered first or unary input value index
     * @param secondInputIndex non-negative ordered right input for binary kinds, or exactly
     *     {@link #NO_SECOND_INPUT} for unary {@code NEG}
     * @param outputIndex non-negative unique output value index
     */
    record Node(
            NodeKind kind,
            int firstInputIndex,
            int secondInputIndex,
            int outputIndex) {
        Node {
            Objects.requireNonNull(kind, "kind");
            if (firstInputIndex < 0) {
                throw new IllegalArgumentException("firstInputIndex must be non-negative");
            }
            if (outputIndex < 0) {
                throw new IllegalArgumentException("outputIndex must be non-negative");
            }
            if ((kind.inputCount() == 1 && secondInputIndex != NO_SECOND_INPUT)
                    || (kind.inputCount() == 2 && secondInputIndex < 0)) {
                throw new IllegalArgumentException(
                        "secondInputIndex disagrees with the typed node kind");
            }
        }

        /**
         * Creates one unary NEG record.
         *
         * @param inputIndex non-negative input value index
         * @param outputIndex non-negative output value index
         * @return a new non-null typed unary record
         */
        static Node neg(int inputIndex, int outputIndex) {
            return new Node(NodeKind.NEG, inputIndex, NO_SECOND_INPUT, outputIndex);
        }

        /**
         * Creates one binary record.
         *
         * @param kind non-null binary kind
         * @param leftInputIndex non-negative semantic left operand value index
         * @param rightInputIndex non-negative semantic right operand value index
         * @param outputIndex non-negative output value index
         * @return a new non-null typed binary record
         * @throws IllegalArgumentException if {@code kind} is unary
         */
        static Node binary(
                NodeKind kind,
                int leftInputIndex,
                int rightInputIndex,
                int outputIndex) {
            Objects.requireNonNull(kind, "kind");
            if (kind.inputCount() != 2) {
                throw new IllegalArgumentException("binary node kind must have two inputs");
            }
            return new Node(kind, leftInputIndex, rightInputIndex, outputIndex);
        }
    }

    private final List<Node> nodes;

    /**
     * Snapshots one non-empty topologically ordered typed node table.
     *
     * @param nodes non-null non-empty node sequence without null elements
     * @throws NullPointerException if {@code nodes} or one element is {@code null}
     * @throws IllegalArgumentException if the table is empty
     */
    MetalMpsGraphProgram(List<Node> nodes) {
        this.nodes = List.copyOf(nodes);
        if (this.nodes.isEmpty()) {
            throw new IllegalArgumentException("Metal MPSGraph program must contain a node");
        }
    }

    /** @return immutable topological typed node sequence */
    List<Node> nodes() {
        return nodes;
    }

    /**
     * Encodes the exact fixed-width version-one native record table.
     *
     * @return fresh row-major {@code [kind, first, second, output]} cells
     */
    int[] encodedNodeRecords() {
        int[] encoded = new int[Math.multiplyExact(nodes.size(), NODE_RECORD_WIDTH)];
        for (int index = 0; index < nodes.size(); index++) {
            Node node = nodes.get(index);
            int offset = index * NODE_RECORD_WIDTH;
            encoded[offset] = node.kind().wireIdentity();
            encoded[offset + 1] = node.firstInputIndex();
            encoded[offset + 2] = node.secondInputIndex();
            encoded[offset + 3] = node.outputIndex();
        }
        return encoded;
    }
}
