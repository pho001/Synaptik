package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.ArrayList;
import java.util.List;

/**
 * Exact run-owned numerical workspace geometry derived from the authenticated program.
 * No workspace is retained in a native executable or shared across invocations. Only FLOAT32
 * attention weights may double as working storage; low siblings are narrowed publications,
 * independent of the FLOAT32 scores/probabilities consumed by the primary result.
 */
final class MetalProgramWorkspace {
    private MetalProgramWorkspace() {}

    /**
     * Derives one node's exact scratch extent without allocating a device resource.
     *
     * @param node authenticated ordinary or bounded Scan node
     * @param values complete ordered descriptors in the node's index namespace
     * @return checked nonnegative bytes; zero means no numerical workspace binding
     * @throws IllegalArgumentException if the bounded Scan header is absent or malformed
     * @throws ArithmeticException if any declared resource product overflows
     */
    static long scratchBytes(
            MetalMpsGraphProgram.Node node,
            List<MetalMpsGraphProgram.ValueDescriptor> values) {
        if (node.kind() == MetalMpsGraphProgram.NodeKind.SCALED_DOT_PRODUCT_ATTENTION
                && (node.outputCount() == 1
                        || values.get(node.outputIndex()).dataType() != DataType.FLOAT32)) {
            long[] output = values.get(node.outputIndex()).dimensions();
            if (node.outputCount() == 1)
                for (long extent : output) if (extent == 0L) return 0L;
            long[] key = values.get(node.inputs()[1]).dimensions();
            long rows = 1L;
            for (int axis = 0; axis < output.length - 1; axis++)
                rows = Math.multiplyExact(rows, output[axis]);
            return Math.multiplyExact(Math.multiplyExact(rows, key[key.length - 2]), Float.BYTES);
        }
        if (node.kind().wireIdentity() == 116) {
            long[] words = node.attributeWords();
            if (words.length < 32 || words[22] < 0L)
                throw new IllegalArgumentException("Scan workspace header is malformed");
            return words[22];
        }
        return 0L;
    }

    /**
     * Lists numerical workspace bindings in stable node order.
     *
     * @param program exact prepared program
     * @param values complete ordered value descriptors
     * @return new immutable-by-ownership array of strictly positive byte extents
     */
    static long[] scratchBytes(
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values) {
        var result = new ArrayList<Long>();
        for (var node : program.nodes()) {
            long bytes = scratchBytes(node, values);
            if (bytes != 0L) result.add(bytes);
        }
        long[] extents = new long[result.size()];
        for (int index = 0; index < extents.length; index++) extents[index] = result.get(index);
        return extents;
    }

    /**
     * Computes total host-address and numerical scratch bytes in one prepared workspace.
     *
     * @param pointerCount exact host pointer count, including scratch and target aliases
     * @param scratch ordered strictly positive device scratch byte extents
     * @return checked total declared bytes
     */
    static long totalBytes(long pointerCount, long[] scratch) {
        long bytes = Math.multiplyExact(pointerCount, Long.BYTES);
        for (long extent : scratch) bytes = Math.addExact(bytes, extent);
        return bytes;
    }
}
