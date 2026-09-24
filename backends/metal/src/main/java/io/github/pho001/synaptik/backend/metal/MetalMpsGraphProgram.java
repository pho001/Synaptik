package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.Objects;

/**
 * Immutable typed operation table for the version-ten Metal MPSGraph node schema.
 *
 * <p>ABI version four points at fixed 160-byte discriminated records. Each record contains a
 * closed operation identity, exact ordered value indices, one typed attribute discriminator, and
 * bounded target-shape, permutation, normalized-axis, reduction, or one-hot depth state. Reduction
 * records use a typed full/single/multi/sum-to-Shape form, ordered axes (including an empty
 * multi-axis list), exact keep-dimensions state, or the exact scalar-or-positive-rank
 * sum-to-Shape target. GATHER retains wire 16, ONE_HOT retains wire 17, and SCATTER_ELEMENTS
 * appends wire 18 with its third input in the typed auxiliary cell. Every unused scalar is a
 * required zero or {@code UINT32_MAX} sentinel and every unused attribute cell is zero. No
 * operation name, generic integer payload, object graph, map, or executable state crosses the
 * ABI.</p>
 */
final class MetalMpsGraphProgram {
    /** Exact node schema carried across native ABI version four. */
    static final int SCHEMA_VERSION = 10;
    /** Maximum target rank or permutation length. */
    static final int MAX_RANK = 16;
    /** Exact fixed native record size. */
    static final int NODE_RECORD_BYTES = 160;
    /** Unsigned {@code UINT32_MAX} sentinel for an absent second input. */
    static final int NO_SECOND_INPUT = -1;
    /** Unsigned {@code UINT32_MAX} sentinel for an absent axis. */
    static final int NO_AXIS = -1;

    private static final long ATTRIBUTE_VALUES_OFFSET = 32L;

    /** Closed attribute vocabulary and stable schema-local wire identities. */
    enum AttributeKind {
        NONE(0), TARGET_SHAPE(1), PERMUTATION(2), AXIS(3), REDUCTION(4), DEPTH(5);
        private final int wireIdentity;

        AttributeKind(int wireIdentity) {
            this.wireIdentity = wireIdentity;
        }

        int wireIdentity() {
            return wireIdentity;
        }
    }

    /** Typed reduction form stored in the schema's axis discriminator cell. */
    enum ReductionForm {
        FULL(1), SINGLE_AXIS(2), MULTI_AXIS(3), SUM_TO_SHAPE(4);

        private final int wireIdentity;

        ReductionForm(int wireIdentity) {
            this.wireIdentity = wireIdentity;
        }

        int wireIdentity() {
            return wireIdentity;
        }

        static ReductionForm fromWireIdentity(int wireIdentity) {
            for (ReductionForm form : values()) {
                if (form.wireIdentity == wireIdentity) return form;
            }
            throw new IllegalArgumentException("unknown Metal reduction form");
        }
    }

    /** Explicit validated state of one value while walking the ordered program. */
    enum ValueState {
        UNAVAILABLE(0), CANONICAL(1), AFFINE_VIEW(2);

        private final int wireIdentity;

        ValueState(int wireIdentity) {
            this.wireIdentity = wireIdentity;
        }

        int wireIdentity() {
            return wireIdentity;
        }
    }

    /** Closed operation vocabulary and stable schema-local wire identities. */
    enum NodeKind {
        NEG(1, 1, AttributeKind.NONE, ValueState.CANONICAL, false),
        ADD(2, 2, AttributeKind.NONE, ValueState.CANONICAL, false),
        SUB(3, 2, AttributeKind.NONE, ValueState.CANONICAL, false),
        MUL(4, 2, AttributeKind.NONE, ValueState.CANONICAL, false),
        DIV(5, 2, AttributeKind.NONE, ValueState.CANONICAL, false),
        RESHAPE(6, 1, AttributeKind.TARGET_SHAPE, ValueState.AFFINE_VIEW, true),
        EXPAND(7, 1, AttributeKind.TARGET_SHAPE, ValueState.AFFINE_VIEW, true),
        PERMUTE(8, 1, AttributeKind.PERMUTATION, ValueState.AFFINE_VIEW, true),
        EXPAND_DIMS(9, 1, AttributeKind.AXIS, ValueState.AFFINE_VIEW, true),
        SQUEEZE(10, 1, AttributeKind.AXIS, ValueState.AFFINE_VIEW, true),
        CONTIGUOUS(11, 1, AttributeKind.NONE, ValueState.CANONICAL, true),
        ABS(12, 1, AttributeKind.NONE, ValueState.CANONICAL, false),
        SUM(13, 1, AttributeKind.REDUCTION, ValueState.CANONICAL, false),
        MEAN(14, 1, AttributeKind.REDUCTION, ValueState.CANONICAL, false),
        MATMUL(15, 2, AttributeKind.NONE, ValueState.CANONICAL, true),
        GATHER(16, 2, AttributeKind.AXIS, ValueState.CANONICAL, false),
        ONE_HOT(17, 1, AttributeKind.DEPTH, ValueState.CANONICAL, false),
        SCATTER_ELEMENTS(18, 3, AttributeKind.AXIS, ValueState.CANONICAL, false);

        private final int wireIdentity;
        private final int inputCount;
        private final AttributeKind attributeKind;
        private final ValueState outputState;
        private final boolean acceptsAffineView;

        NodeKind(
                int wireIdentity,
                int inputCount,
                AttributeKind attributeKind,
                ValueState outputState,
                boolean acceptsAffineView) {
            this.wireIdentity = wireIdentity;
            this.inputCount = inputCount;
            this.attributeKind = attributeKind;
            this.outputState = outputState;
            this.acceptsAffineView = acceptsAffineView;
        }

        int wireIdentity() {
            return wireIdentity;
        }

        int inputCount() {
            return inputCount;
        }

        AttributeKind attributeKind() {
            return attributeKind;
        }

        ValueState outputState() {
            return outputState;
        }

        boolean isAffine() {
            return outputState == ValueState.AFFINE_VIEW;
        }

        boolean accepts(ValueState inputState) {
            return inputState == ValueState.CANONICAL
                    || (inputState == ValueState.AFFINE_VIEW && acceptsAffineView);
        }
    }

    /** One immutable typed version-ten node record. */
    static final class Node {
        private final NodeKind kind;
        private final int firstInputIndex;
        private final int secondInputIndex;
        private final int outputIndex;
        private final int attributeCount;
        private final int axis;
        private final int auxiliary;
        private final long[] attributeValues;

        private Node(
                NodeKind kind,
                int firstInputIndex,
                int secondInputIndex,
                int outputIndex,
                int attributeCount,
                int axis,
                int auxiliary,
                long[] attributeValues) {
            this.kind = Objects.requireNonNull(kind, "kind");
            if (firstInputIndex < 0) {
                throw new IllegalArgumentException("firstInputIndex must be non-negative");
            }
            if (outputIndex < 0) {
                throw new IllegalArgumentException("outputIndex must be non-negative");
            }
            boolean secondInputPresent = kind.inputCount() >= 2;
            if ((!secondInputPresent && secondInputIndex != NO_SECOND_INPUT)
                    || (secondInputPresent && secondInputIndex < 0)) {
                throw new IllegalArgumentException(
                        "secondInputIndex disagrees with the typed node kind");
            }
            if (kind.inputCount() == 3 && auxiliary < 0) {
                throw new IllegalArgumentException(
                        "auxiliary third input disagrees with the typed node kind");
            }
            Objects.requireNonNull(attributeValues, "attributeValues");
            if (kind.attributeKind() == AttributeKind.NONE) {
                if (attributeCount != 0 || axis != NO_AXIS || auxiliary != 0
                        || attributeValues.length != 0) {
                    throw new IllegalArgumentException("no-attribute node contains attribute state");
                }
            } else if (kind.attributeKind() == AttributeKind.TARGET_SHAPE) {
                if (attributeCount < 1 || attributeCount > MAX_RANK
                        || axis != NO_AXIS || auxiliary != 0
                        || attributeValues.length != attributeCount) {
                    throw new IllegalArgumentException("target-shape node attributes are malformed");
                }
                requirePositiveDimensions(attributeValues, "target-shape");
            } else if (kind.attributeKind() == AttributeKind.PERMUTATION) {
                if (attributeCount < 1 || attributeCount > MAX_RANK
                        || axis != NO_AXIS || auxiliary != 0
                        || attributeValues.length != attributeCount) {
                    throw new IllegalArgumentException("permutation node attributes are malformed");
                }
                validateOrderedAxes(attributeValues, attributeCount, true);
            } else if (kind.attributeKind() == AttributeKind.AXIS) {
                boolean auxiliaryValid = kind == NodeKind.SCATTER_ELEMENTS
                        ? auxiliary >= 0
                        : auxiliary == 0;
                if (attributeCount != 1 || axis < 0 || axis >= MAX_RANK
                        || !auxiliaryValid || attributeValues.length != 0) {
                    throw new IllegalArgumentException("axis node attributes are malformed");
                }
            } else if (kind.attributeKind() == AttributeKind.DEPTH) {
                if (kind != NodeKind.ONE_HOT || attributeCount != 1 || axis != NO_AXIS
                        || auxiliary != 0 || attributeValues.length != 1
                        || attributeValues[0] <= 0L) {
                    throw new IllegalArgumentException("depth node attributes are malformed");
                }
            } else {
                ReductionForm form = ReductionForm.fromWireIdentity(axis);
                if (auxiliary < 0 || auxiliary > 1 || attributeCount < 0
                        || attributeCount > MAX_RANK
                        || attributeValues.length != attributeCount) {
                    throw new IllegalArgumentException("reduction node attributes are malformed");
                }
                switch (form) {
                    case FULL -> {
                        if (attributeCount != 0 || auxiliary != 0) {
                            throw new IllegalArgumentException("full reduction attributes are malformed");
                        }
                    }
                    case SINGLE_AXIS -> {
                        if (attributeCount != 1) {
                            throw new IllegalArgumentException(
                                    "single-axis reduction must carry exactly one axis");
                        }
                        validateOrderedAxes(attributeValues, MAX_RANK, false);
                    }
                    case MULTI_AXIS ->
                        validateOrderedAxes(attributeValues, MAX_RANK, false);
                    case SUM_TO_SHAPE -> {
                        if (kind != NodeKind.SUM || auxiliary != 0) {
                            throw new IllegalArgumentException(
                                    "sum-to-Shape is valid only for SUM without keepDimensions");
                        }
                        requirePositiveDimensions(attributeValues, "sum-to-Shape");
                    }
                }
            }
            this.firstInputIndex = firstInputIndex;
            this.secondInputIndex = secondInputIndex;
            this.outputIndex = outputIndex;
            this.attributeCount = attributeCount;
            this.axis = axis;
            this.auxiliary = auxiliary;
            this.attributeValues = attributeValues.clone();
        }

        private static void requirePositiveDimensions(long[] dimensions, String role) {
            for (long dimension : dimensions) {
                if (dimension <= 0L) {
                    throw new IllegalArgumentException(role + " dimensions must be positive");
                }
            }
        }

        private static void validateOrderedAxes(
                long[] axes, int bound, boolean complete) {
            boolean[] seen = new boolean[bound];
            for (long axis : axes) {
                if (axis < 0L || axis >= bound || seen[(int) axis]) {
                    throw new IllegalArgumentException(
                            "axes must be ordered, distinct, and in range");
                }
                seen[(int) axis] = true;
            }
            if (complete && axes.length != bound) {
                throw new IllegalArgumentException("permutation must be complete");
            }
        }

        static Node neg(int inputIndex, int outputIndex) {
            return noAttributes(NodeKind.NEG, inputIndex, NO_SECOND_INPUT, outputIndex);
        }

        static Node abs(int inputIndex, int outputIndex) {
            return noAttributes(NodeKind.ABS, inputIndex, NO_SECOND_INPUT, outputIndex);
        }

        static Node binary(
                NodeKind kind,
                int leftInputIndex,
                int rightInputIndex,
                int outputIndex) {
            Objects.requireNonNull(kind, "kind");
            if (kind.inputCount() != 2 || kind.attributeKind() != AttributeKind.NONE) {
                throw new IllegalArgumentException("binary node kind must have two inputs");
            }
            return noAttributes(kind, leftInputIndex, rightInputIndex, outputIndex);
        }

        static Node matmul(
                int leftInputIndex, int rightInputIndex, int outputIndex) {
            return noAttributes(
                    NodeKind.MATMUL, leftInputIndex, rightInputIndex, outputIndex);
        }

        static Node contiguous(int inputIndex, int outputIndex) {
            return noAttributes(NodeKind.CONTIGUOUS, inputIndex, NO_SECOND_INPUT, outputIndex);
        }

        static Node targetShape(
                NodeKind kind, int inputIndex, int outputIndex, long[] dimensions) {
            Objects.requireNonNull(kind, "kind");
            if (kind.attributeKind() != AttributeKind.TARGET_SHAPE) {
                throw new IllegalArgumentException("node kind does not use target-shape attributes");
            }
            Objects.requireNonNull(dimensions, "dimensions");
            return new Node(kind, inputIndex, NO_SECOND_INPUT, outputIndex,
                    dimensions.length, NO_AXIS, 0, dimensions);
        }

        static Node permutation(int inputIndex, int outputIndex, List<Integer> axes) {
            Objects.requireNonNull(axes, "axes");
            long[] values = longValues(axes);
            return new Node(NodeKind.PERMUTE, inputIndex, NO_SECOND_INPUT, outputIndex,
                    values.length, NO_AXIS, 0, values);
        }

        static Node axis(NodeKind kind, int inputIndex, int outputIndex, int axis) {
            Objects.requireNonNull(kind, "kind");
            if (kind.attributeKind() != AttributeKind.AXIS || kind.inputCount() != 1) {
                throw new IllegalArgumentException(
                        "node kind does not use unary axis attributes");
            }
            return new Node(kind, inputIndex, NO_SECOND_INPUT, outputIndex,
                    1, axis, 0, new long[0]);
        }

        static Node reduction(
                NodeKind kind,
                int inputIndex,
                int outputIndex,
                ReductionForm form,
                List<Integer> axes,
                boolean keepDimensions) {
            Objects.requireNonNull(kind, "kind");
            Objects.requireNonNull(form, "form");
            Objects.requireNonNull(axes, "axes");
            if (kind != NodeKind.SUM && kind != NodeKind.MEAN) {
                throw new IllegalArgumentException("reduction kind must be SUM or MEAN");
            }
            if (form == ReductionForm.SUM_TO_SHAPE) {
                throw new IllegalArgumentException("sum-to-Shape requires target dimensions");
            }
            long[] values = longValues(axes);
            return new Node(kind, inputIndex, NO_SECOND_INPUT, outputIndex,
                    values.length, form.wireIdentity(), keepDimensions ? 1 : 0, values);
        }

        static Node sumToShape(
                int inputIndex, int outputIndex, long[] targetDimensions) {
            Objects.requireNonNull(targetDimensions, "targetDimensions");
            return new Node(NodeKind.SUM, inputIndex, NO_SECOND_INPUT, outputIndex,
                    targetDimensions.length, ReductionForm.SUM_TO_SHAPE.wireIdentity(), 0,
                    targetDimensions);
        }

        static Node gather(
                int dataInputIndex, int indicesInputIndex, int outputIndex, int axis) {
            return new Node(NodeKind.GATHER, dataInputIndex, indicesInputIndex, outputIndex,
                    1, axis, 0, new long[0]);
        }

        static Node scatterElements(
                int dataInputIndex,
                int indicesInputIndex,
                int updatesInputIndex,
                int outputIndex,
                int axis) {
            return new Node(
                    NodeKind.SCATTER_ELEMENTS,
                    dataInputIndex,
                    indicesInputIndex,
                    outputIndex,
                    1,
                    axis,
                    updatesInputIndex,
                    new long[0]);
        }

        static Node oneHot(int indicesInputIndex, int outputIndex, long depth) {
            return new Node(NodeKind.ONE_HOT, indicesInputIndex, NO_SECOND_INPUT, outputIndex,
                    1, NO_AXIS, 0, new long[] {depth});
        }

        private static long[] longValues(List<Integer> values) {
            long[] result = new long[values.size()];
            for (int index = 0; index < values.size(); index++) {
                result[index] = Objects.requireNonNull(
                        values.get(index), "values[" + index + "]");
            }
            return result;
        }

        private static Node noAttributes(
                NodeKind kind, int firstInputIndex, int secondInputIndex, int outputIndex) {
            return new Node(kind, firstInputIndex, secondInputIndex, outputIndex,
                    0, NO_AXIS, 0, new long[0]);
        }

        NodeKind kind() {
            return kind;
        }

        int firstInputIndex() {
            return firstInputIndex;
        }

        int secondInputIndex() {
            return secondInputIndex;
        }

        int outputIndex() {
            return outputIndex;
        }

        int attributeCount() {
            return attributeCount;
        }

        int axis() {
            return axis;
        }

        int auxiliary() {
            return auxiliary;
        }

        long[] attributeValues() {
            return attributeValues.clone();
        }
    }

    private final List<Node> nodes;

    MetalMpsGraphProgram(List<Node> nodes) {
        this.nodes = List.copyOf(nodes);
        if (this.nodes.isEmpty()) {
            throw new IllegalArgumentException("Metal MPSGraph program must contain a node");
        }
    }

    List<Node> nodes() {
        return nodes;
    }

    /** Returns canonical big-endian bytes for workload compatibility hashing and tests. */
    byte[] encodedNodeRecords() {
        ByteBuffer encoded = ByteBuffer.allocate(
                Math.multiplyExact(nodes.size(), NODE_RECORD_BYTES)).order(ByteOrder.BIG_ENDIAN);
        for (Node node : nodes) {
            putRecord(encoded, node);
        }
        return encoded.array();
    }

    /** Allocates and writes exact native-endian version-ten records for one downcall. */
    MemorySegment encodeNative(Arena arena) {
        Objects.requireNonNull(arena, "arena");
        long bytes = Math.multiplyExact((long) nodes.size(), NODE_RECORD_BYTES);
        MemorySegment encoded = arena.allocate(bytes, Long.BYTES);
        for (int index = 0; index < nodes.size(); index++) {
            writeNativeRecord(encoded, Math.multiplyExact((long) index, NODE_RECORD_BYTES),
                    nodes.get(index));
        }
        return encoded;
    }

    private static void putRecord(ByteBuffer encoded, Node node) {
        encoded.putInt(node.kind().wireIdentity());
        encoded.putInt(node.kind().attributeKind().wireIdentity());
        encoded.putInt(node.firstInputIndex());
        encoded.putInt(node.secondInputIndex());
        encoded.putInt(node.outputIndex());
        encoded.putInt(node.attributeCount());
        encoded.putInt(node.axis());
        encoded.putInt(node.auxiliary());
        long[] values = node.attributeValues();
        for (int index = 0; index < MAX_RANK; index++) {
            encoded.putLong(index < values.length ? values[index] : 0L);
        }
    }

    private static void writeNativeRecord(MemorySegment target, long offset, Node node) {
        target.set(JAVA_INT, offset, node.kind().wireIdentity());
        target.set(JAVA_INT, offset + 4L, node.kind().attributeKind().wireIdentity());
        target.set(JAVA_INT, offset + 8L, node.firstInputIndex());
        target.set(JAVA_INT, offset + 12L, node.secondInputIndex());
        target.set(JAVA_INT, offset + 16L, node.outputIndex());
        target.set(JAVA_INT, offset + 20L, node.attributeCount());
        target.set(JAVA_INT, offset + 24L, node.axis());
        target.set(JAVA_INT, offset + 28L, node.auxiliary());
        long[] values = node.attributeValues();
        for (int index = 0; index < MAX_RANK; index++) {
            target.set(JAVA_LONG, offset + ATTRIBUTE_VALUES_OFFSET + (long) index * Long.BYTES,
                    index < values.length ? values[index] : 0L);
        }
    }
}
