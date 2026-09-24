package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.shape.ShapeBroadcast;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.List;
import java.util.Objects;

/**
 * Reports the exact operation-occurrence capability of the current Metal backend.
 *
 * <p>This provider is immutable and performs no device discovery, native-library loading,
 * allocation, registration, or caching. Support is limited to unary {@code NEG} and binary
 * {@code ADD}, {@code SUB}, {@code MUL}, and {@code DIV} over positive, fully static, canonical
 * dense-contiguous {@code FLOAT32} descriptors. Binary occurrences use exact right-aligned
 * broadcasting and retain ordered operands. Every input and output in one admitted occurrence
 * has the same gradient-eligibility flag.</p>
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
     * Reports support only for the exact prepared Metal elementwise domain.
     *
     * @param query the non-null immutable operation occurrence to classify without probing a
     *     device or native library
     * @return {@code true} exactly for supported unary NEG or binary arithmetic occurrences
     * @throws NullPointerException if {@code query} is {@code null}, with message {@code query}
     */
    @Override
    public boolean supports(OperationCapabilityQuery query) {
        Objects.requireNonNull(query, "query");
        return supportsOccurrence(query.operation(), query.inputs(), query.outputs());
    }

    /**
     * Validates one projected occurrence against the same exact domain used by Planning.
     *
     * @param operation non-null typed operation
     * @param inputs non-null ordered input descriptors
     * @param outputs non-null ordered output descriptors
     * @return whether the occurrence is supported
     */
    static boolean supportsOccurrence(
            Operation operation,
            List<TensorDescriptor> inputs,
            List<TensorDescriptor> outputs) {
        Objects.requireNonNull(operation, "operation");
        Objects.requireNonNull(inputs, "inputs");
        Objects.requireNonNull(outputs, "outputs");
        if (operation.attrs() != NoOperationAttrs.INSTANCE || outputs.size() != 1) {
            return false;
        }
        TensorDescriptor output = outputs.getFirst();
        try {
            if (operation.kind() == UnaryElementwiseKind.NEG) {
                if (inputs.size() != 1) {
                    return false;
                }
                TensorDescriptor input = inputs.getFirst();
                return eligible(input)
                        && eligible(output)
                        && input.shape().equals(output.shape())
                        && input.requiresGrad() == output.requiresGrad();
            }
            if (!(operation.kind() instanceof BinaryArithmeticKind binary)
                    || (binary != BinaryArithmeticKind.ADD
                            && binary != BinaryArithmeticKind.SUB
                            && binary != BinaryArithmeticKind.MUL
                            && binary != BinaryArithmeticKind.DIV)
                    || inputs.size() != 2) {
                return false;
            }
            TensorDescriptor left = inputs.get(0);
            TensorDescriptor right = inputs.get(1);
            return eligible(left)
                    && eligible(right)
                    && eligible(output)
                    && left.requiresGrad() == right.requiresGrad()
                    && left.requiresGrad() == output.requiresGrad()
                    && ShapeBroadcast.broadcast(left.shape(), right.shape())
                            .equals(output.shape());
        } catch (IllegalArgumentException | ArithmeticException incompatible) {
            return false;
        }
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
            return descriptor.layout().orElseThrow().equals(
                    io.github.pho001.synaptik.model.layout.LayoutDescriptor.contiguous(
                            descriptor.shape()));
        } catch (IllegalArgumentException | ArithmeticException invalid) {
            return false;
        }
    }
}
