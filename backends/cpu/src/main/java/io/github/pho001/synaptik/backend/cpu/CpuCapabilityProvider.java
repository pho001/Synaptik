package io.github.pho001.synaptik.backend.cpu;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutKind;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.Objects;

/**
 * Reports the executable semantic coverage currently delivered by the CPU backend.
 *
 * <p>The provider has a stable CPU ownership identity and advertises only parameterless
 * {@code ADD} occurrences whose two inputs and one output have the same fully static shape and
 * the same {@code FLOAT64}, {@code FLOAT32}, {@code INT32}, or {@code INT64} data type. Each
 * layout must be unresolved or resolved as canonical dense contiguous, zero-offset, non-view
 * geometry. The corresponding implementation is the scalar, single-thread, native-segment
 * route. Construction performs no platform discovery, registration, allocation, or native
 * loading.</p>
 */
public final class CpuCapabilityProvider implements BackendCapabilityProvider {
    /** Stable Planning ownership identity for the CPU backend. */
    public static final BackendId CPU_BACKEND_ID = new BackendId("cpu");

    /** Creates a stateless, narrowly fail-closed CPU capability provider. */
    public CpuCapabilityProvider() {
    }

    /**
     * Returns the stable CPU ownership identity.
     *
     * @return {@link #CPU_BACKEND_ID} by exact reference; never {@code null}
     */
    @Override
    public BackendId backendId() {
        return CPU_BACKEND_ID;
    }

    /**
     * Reports whether an occurrence belongs to the exact implemented dense {@code ADD} matrix.
     * Unresolved layouts are accepted because CPU preparation selects a canonical materialized
     * representation; a resolved offset, strided, or view layout is rejected.
     *
     * @param query non-null immutable operation occurrence to validate structurally
     * @return {@code true} only for the exact implemented occurrence matrix; otherwise
     *     {@code false}
     * @throws NullPointerException if {@code query} is {@code null}, with message {@code query}
     */
    @Override
    public boolean supports(OperationCapabilityQuery query) {
        Objects.requireNonNull(query, "query");
        if (query.operation().kind() != BinaryArithmeticKind.ADD
                || query.operation().attrs() != NoOperationAttrs.INSTANCE
                || query.inputs().size() != 2
                || query.outputs().size() != 1) {
            return false;
        }
        TensorDescriptor left = query.inputs().get(0);
        TensorDescriptor right = query.inputs().get(1);
        TensorDescriptor output = query.outputs().get(0);
        DataType dataType = left.dataType();
        if (!supportedDataType(dataType)
                || right.dataType() != dataType
                || output.dataType() != dataType
                || !left.shape().isFullyStatic()
                || !left.shape().equals(right.shape())
                || !left.shape().equals(output.shape())) {
            return false;
        }
        return canonicalOrUnresolved(left)
                && canonicalOrUnresolved(right)
                && canonicalOrUnresolved(output);
    }

    private static boolean supportedDataType(DataType dataType) {
        return dataType == DataType.FLOAT64
                || dataType == DataType.FLOAT32
                || dataType == DataType.INT32
                || dataType == DataType.INT64;
    }

    private static boolean canonicalOrUnresolved(TensorDescriptor descriptor) {
        return descriptor.layout().isEmpty() || descriptor.layout().filter(layout ->
                layout.kind() == LayoutKind.DENSE_CONTIGUOUS
                        && layout.storageOffset() == 0
                        && !layout.isView()).isPresent();
    }
}
