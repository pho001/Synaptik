package io.github.pho001.synaptik.backend.metal;

import java.util.List;

/**
 * Single bounded native Scan grammar carried by GENERAL_SCAN attribute words in schema 22.
 * Offsets and lengths are in 64-bit little-endian words. Every section is contiguous, every
 * record self-sizes, and the final offset equals the complete attribute payload length; unknown
 * versions, trailing words and overlapping sections fail closed.
 * The enclosing ordinary program digest authenticates every byte, including the captured body,
 * structural activity, exact demand and all physical binding recipes. No invocation value occurs
 * here. This is not a second ordinary-operation decoder or a separately exported native ABI.
 */
final class MetalScanEncoding {
    static final long MAGIC = 0x5343414e504c414eL;
    static final int VERSION = 1;
    static final int HEADER_WORDS = 32;
    static final int MAX_VALUES = 4096;
    static final int MAX_UNITS = 512;
    static final int MAX_WORDS = 1 << 20;


    /**
     * Validates raw constructor input through the single bounded Scan grammar.
     * @param words complete raw attribute payload, retained by neither decoder nor execution
     * @throws IllegalArgumentException if any intrinsic grammar, control or resource fact disagrees
     */
    static void validate(long[] words) {
        try {
            MetalScanDecoding.decode(words);
        } catch (ArithmeticException | IndexOutOfBoundsException invalid) {
            throw new IllegalArgumentException("Malformed Metal Scan body/binder image", invalid);
        }
    }
    private MetalScanEncoding() {}

    /**
     * Computes the exact serialized size before qualification or allocation, including dense
     * root-member lists, which are distinct from the expanded bitset workspace representation.
     * @param plan complete immutable body/binder plan
     * @return complete bounded word count, including the fixed header
     * @throws IllegalArgumentException if the complete payload exceeds the wire resource limit
     * @throws ArithmeticException if checked section arithmetic overflows
     */
    static int wordCount(MetalScanPlan plan) {
        long words = HEADER_WORDS;
        for (var value : plan.values()) {
            words = Math.addExact(words, Math.addExact(13L, Math.multiplyExact(2L, value.descriptor().shape().rank())));
        }
        for (var unit : plan.units()) {
            var node = unit.node();
            long references = Math.addExact((long) node.inputCount(), node.outputCount());
            words = Math.addExact(words, Math.addExact(14L, Math.addExact(references, node.attributeWordCount())));
        }
        words = Math.addExact(words, Math.addExact((long) plan.formals().size(),
                Math.addExact((long) plan.roles().size(), plan.demanded().size())));
        if (plan.activity().isPresent()) {
            var activity = plan.activity().orElseThrow();
            words = Math.addExact(words, Math.multiplyExact(3L, activity.carries().size()));
            words = Math.addExact(words, activity.seedFormals().size());
            for (var scope : activity.scopes()) words = Math.addExact(words, Math.addExact(1L, scope.size()));
            words = Math.addExact(words, Math.multiplyExact(2L, activity.accumulators().size()));
        }
        if (words > MAX_WORDS) throw new IllegalArgumentException("Metal Scan binder exceeds bounded attribute size");
        return Math.toIntExact(words);
    }

    /**
     * Encodes immutable local binding/control facts and one copy of each ordinary instruction.
     * @param plan independently qualified complete physical body plan
     * @return fresh exact-sized attribute payload; caller owns and may discard the returned array
     * @throws IllegalArgumentException if the bounded grammar exceeds its resource limit
     * @throws ArithmeticException if checked record geometry overflows
     */
    static long[] encode(MetalScanPlan plan) {
        var words = new Words(wordCount(plan));
        var attrs = plan.header();
        words.set(0, MAGIC); words.set(1, (long) VERSION);
        words.set(3, attrs.timeCount());
        words.set(4, (attrs.reverse() ? 1L : 0L) | (attrs.lengthAware() ? 2L : 0L) | (plan.activity().isPresent() ? 4L : 0L));
        words.set(5, (long) attrs.carryCount()); words.set(6, (long) attrs.sequenceCount());
        words.set(7, (long) attrs.invariantCount()); words.set(8, (long) attrs.emissionCount());
        words.set(9, (long) attrs.residualCount()); words.set(10, (long) attrs.rowCarryCount());
        words.set(11, plan.batch()); words.set(12, (long) plan.values().size());
        words.set(13, (long) plan.units().size()); words.set(14, (long) plan.formals().size());
        words.set(15, (long) plan.roles().size()); words.set(16, (long) plan.demanded().size());
        words.set(30, plan.controlByteOffset()); words.set(31, plan.controlWords());
        words.set(22, plan.workspaceBytes()); words.set(23, (long) words.size());
        for (var value : plan.values()) {
            var descriptor = value.descriptor();
            var layout = descriptor.layout().orElseThrow();
            int rank = descriptor.shape().rank();
            append(words, 13L + 2L * rank, MetalMpsGraphProgram.dataTypeWire(descriptor.dataType()), rank,
                    descriptor.requiresGrad() ? 1 : 0,
                    layout.equals(io.github.pho001.synaptik.model.layout.LayoutDescriptor.contiguous(descriptor.shape())) ? 1 : layout.isView() ? 2 : 3,
                    layout.storageOffset(), layout.referencedElementSpan(), value.byteOffset(), value.byteSize(),
                    value.rowAxis(), value.terminal() ? 1 : 0, value.rowInner(), layout.isView() ? 1 : 0);
            for (var dimension : descriptor.shape().dimensions()) words.add(dimension.staticSize().orElseThrow());
            for (long stride : layout.strides()) words.add(stride);
        }
        words.set(24, (long) words.size());
        for (var unit : plan.units()) {
            var node = unit.node();
            int[] inputs = node.inputs(), outputs = node.outputs();
            long[] attributes = node.attributeWords();
            append(words, Math.addExact(14L, Math.addExact(inputs.length, Math.addExact(outputs.length, attributes.length))),
                    node.kind().wireIdentity(), node.attributeKind().wireIdentity(), inputs.length, outputs.length,
                    attributes.length, unit.scope(), unit.leftScope(), unit.rightScope(), unit.rowAxis(),
                    unit.contraction() ? 1 : 0, unit.accumulator(), unit.scratchOffset(), unit.scratchBytes());
            for (int input : inputs) words.add((long) input);
            for (int output : outputs) words.add((long) output);
            for (long attribute : attributes) words.add(attribute);
        }
        words.set(25, (long) words.size()); appendIndices(words, plan.formals());
        words.set(26, (long) words.size()); appendIndices(words, plan.roles());
        words.set(27, (long) words.size()); appendIndices(words, plan.demanded());
        words.set(28, (long) words.size());
        if (plan.activity().isPresent()) {
            var activity = plan.activity().orElseThrow();
            words.set(17, Math.addExact((long) activity.carries().size(), activity.seedFormals().size()));
            words.set(18, (long) activity.scopes().size()); words.set(19, (long) activity.carries().size());
            words.set(20, (long) activity.seedFormals().size()); words.set(21, (long) activity.accumulators().size());
            for (var carry : activity.carries()) append(words, carry.numericCarryPosition(), carry.initialPresence() ? 1 : 0, carry.nextScope());
            appendIndices(words, activity.seedFormals());
            for (var scope : activity.scopes()) { words.add((long) scope.size()); appendIndices(words, scope); }
            for (var accumulator : activity.accumulators()) append(words, accumulator.numericSharedCarryPosition(), accumulator.contributionScope());
        }
        words.set(2, (long) words.size()); words.set(29, (long) words.size());
        return words.finish();
    }

    private static void append(Words target, long... values) {
        for (long value : values) target.add(value);
    }

    private static void appendIndices(Words target, List<Integer> values) {
        for (int value : values) target.add((long) value);
    }

    /** The sole serializer writes directly into its checked exact-sized primitive payload. */
    private static final class Words {
        private final long[] data;
        private int size = HEADER_WORDS;

        Words(int count) { data = new long[count]; }
        void add(long value) { data[size++] = value; }
        void set(int index, long value) { data[index] = value; }
        int size() { return size; }
        long[] finish() {
            if (size != data.length) throw new AssertionError("Metal Scan encoded-size accounting disagrees");
            return data;
        }
    }
}
