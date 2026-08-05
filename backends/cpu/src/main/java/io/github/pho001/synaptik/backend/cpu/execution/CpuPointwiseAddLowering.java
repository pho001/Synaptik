package io.github.pho001.synaptik.backend.cpu.execution;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.ValueId;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Objects;

/**
 * Immutable lowering facts for one exact dense pointwise {@code ADD} occurrence.
 *
 * <p>The graph value identities map the node's ordered inputs and output to shared partition
 * declarations. The generated-code fingerprint deliberately includes only the pointwise ADD
 * semantic domain and data type, because value identities and element count do not change the
 * emitted loop body or exact generated signature.</p>
 *
 * @param left exact non-null left input value identity
 * @param right exact non-null right input value identity
 * @param output exact non-null output value identity
 * @param dataType exact supported common data type
 * @param elementCount non-negative exact static logical element count
 * @param fingerprint non-null identity-free fingerprint matching {@code dataType}
 */
record CpuPointwiseAddLowering(
        ValueId left,
        ValueId right,
        ValueId output,
        DataType dataType,
        long elementCount,
        CpuLoweringFingerprint fingerprint) {
    /**
     * Creates one node-local recipe while deriving identity-free generated semantics.
     *
     * @param left exact left input value identity
     * @param right exact right input value identity
     * @param output exact output value identity
     * @param dataType supported exact common data type
     * @param elementCount non-negative exact static logical element count
     * @throws NullPointerException if a reference is null
     * @throws IllegalArgumentException if the type or count is unsupported
     */
    CpuPointwiseAddLowering(
            ValueId left, ValueId right, ValueId output, DataType dataType, long elementCount) {
        this(left, right, output, dataType, elementCount, fingerprint(dataType));
    }

    /**
     * Validates one fully specified lowering, including a caller-supplied fingerprint.
     *
     * @param left exact non-null left input value identity
     * @param right exact non-null right input value identity
     * @param output exact non-null output value identity
     * @param dataType exact supported common data type
     * @param elementCount non-negative exact static logical element count
     * @param fingerprint non-null identity-free fingerprint matching {@code dataType}
     * @throws NullPointerException if a reference is {@code null}, in declaration order
     * @throws IllegalArgumentException if the data type or element count is unsupported, or the
     *     fingerprint does not match the data type
     */
    CpuPointwiseAddLowering {
        Objects.requireNonNull(left, "left");
        Objects.requireNonNull(right, "right");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(fingerprint, "fingerprint");
        if (dataType != DataType.FLOAT64 && dataType != DataType.FLOAT32
                && dataType != DataType.INT32 && dataType != DataType.INT64) {
            throw new IllegalArgumentException("unsupported pointwise ADD data type");
        }
        if (elementCount < 0) throw new IllegalArgumentException(
                "elementCount must be non-negative");
        if (!fingerprint.equals(fingerprint(dataType))) throw new IllegalArgumentException(
                "fingerprint does not match pointwise ADD data type");
    }

    private static CpuLoweringFingerprint fingerprint(DataType dataType) {
        Objects.requireNonNull(dataType, "dataType");
        byte[] domain = "synaptik.cpu.pointwise-add.v1".getBytes(StandardCharsets.US_ASCII);
        byte typeTag = switch (dataType) {
            case FLOAT64 -> 1;
            case FLOAT32 -> 2;
            case INT32 -> 3;
            case INT64 -> 4;
            default -> throw new IllegalArgumentException("unsupported pointwise ADD data type");
        };
        return CpuLoweringFingerprint.of(ByteBuffer.allocate(domain.length + 1)
                .put(domain).put(typeTag).array());
    }
}
