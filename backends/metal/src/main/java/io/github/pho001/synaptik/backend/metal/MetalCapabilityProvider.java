package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutKind;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.Objects;

/**
 * Reports the exact operation-occurrence capability of the current Metal backend.
 *
 * <p>This provider is immutable and performs no device discovery, native-library loading,
 * allocation, registration, or caching. Support is limited to positive, fully static, resolved
 * dense-contiguous {@code FLOAT32} unary negation occurrences that the backend can lower as part
 * of any resulting maximal Metal partition.</p>
 */
public final class MetalCapabilityProvider implements BackendCapabilityProvider {
    /**
     * Stable Planning ownership identity for the Metal backend.
     *
     * <p>The immutable value is shared by every provider instance and says nothing about device
     * availability or executable readiness.</p>
     */
    public static final BackendId METAL_BACKEND_ID = new BackendId("metal");

    /**
     * Creates a stateless, immutable, and thread-safe Metal capability provider.
     *
     * <p>Construction performs no native loading, device discovery, registration, allocation,
     * or caching.</p>
     */
    public MetalCapabilityProvider() {}

    /**
     * Returns the stable Metal ownership identity without probing native state.
     *
     * @return the exact shared {@link #METAL_BACKEND_ID} instance; never {@code null}
     */
    @Override
    public BackendId backendId() {
        return METAL_BACKEND_ID;
    }

    /**
     * Reports support only for the exact prepared unary-negation domain.
     *
     * @param query the non-null immutable operation occurrence to classify without probing a
     *     device or native library
     * @return {@code true} exactly for the supported unary-negation descriptor domain
     * @throws NullPointerException if {@code query} is {@code null}, with message {@code query}
     */
    @Override
    public boolean supports(OperationCapabilityQuery query) {
        Objects.requireNonNull(query, "query");
        if (query.operation().kind() != UnaryElementwiseKind.NEG
                || query.operation().attrs() != NoOperationAttrs.INSTANCE
                || query.inputs().size() != 1 || query.outputs().size() != 1) {
            return false;
        }
        TensorDescriptor input = query.inputs().getFirst();
        TensorDescriptor output = query.outputs().getFirst();
        return eligible(input) && eligible(output)
                && input.shape().equals(output.shape())
                && input.requiresGrad() == output.requiresGrad();
    }

    private static boolean eligible(TensorDescriptor descriptor) {
        if (descriptor.dataType() != DataType.FLOAT32
                || !descriptor.shape().isFullyStatic()
                || descriptor.shape().rank() < 1 || descriptor.shape().rank() > 16
                || descriptor.layout().isEmpty()) {
            return false;
        }
        long elements = 1L;
        try {
            for (long dimension : descriptor.shape().toLongArray()) {
                if (dimension <= 0L) return false;
                elements = Math.multiplyExact(elements, dimension);
            }
            Math.multiplyExact(elements, Float.BYTES);
        } catch (ArithmeticException overflow) {
            return false;
        }
        var layout = descriptor.layout().orElseThrow();
        return layout.kind() == LayoutKind.DENSE_CONTIGUOUS
                && !layout.isView() && layout.storageOffset() == 0L;
    }
}
