package io.github.pho001.synaptik.backend.cpu;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutKind;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.shape.ShapeBroadcast;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.Objects;

/**
 * Reports the executable semantic coverage currently delivered by the CPU backend.
 *
 * <p>The provider has a stable CPU ownership identity and advertises only parameterless
 * {@code FLOAT64} {@code ADD}, exact {@code GELU}, and {@code MUL} occurrences with fully static,
 * right-broadcastable or shape-preserving inputs and resolved layouts. Complete-partition lowering
 * normalizes the exact layout geometry and applies the additional
 * alias, fan-out, publication, and partition-boundary checks before declaring support.</p>
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
     * Reports whether an occurrence belongs to the exact implemented semantic set.
     * Binary results must equal the current right-aligned broadcast result, unary results must
     * preserve shape, and every descriptor must be fully static with a resolved layout.
     *
     * @param query non-null immutable operation occurrence to validate structurally
     * @return {@code true} only for the exact implemented occurrence-local matrix; otherwise
     *     {@code false}; complete-partition eligibility may still be stricter
     * @throws NullPointerException if {@code query} is {@code null}, with message {@code query}
     */
    @Override
    public boolean supports(OperationCapabilityQuery query) {
        Objects.requireNonNull(query, "query");
        if (query.operation().attrs() != NoOperationAttrs.INSTANCE
                || query.outputs().size() != 1) {
            return false;
        }
        boolean unary = query.operation().kind() == UnaryElementwiseKind.GELU;
        boolean binary = query.operation().kind() == BinaryArithmeticKind.ADD
                || query.operation().kind() == BinaryArithmeticKind.MUL;
        if ((!unary && !binary) || query.inputs().size() != (unary ? 1 : 2)) return false;
        TensorDescriptor left = query.inputs().getFirst();
        TensorDescriptor output = query.outputs().get(0);
        DataType dataType = left.dataType();
        if (dataType != DataType.FLOAT64 || output.dataType() != dataType
                || !left.shape().isFullyStatic() || !output.shape().isFullyStatic()) {
            return false;
        }
        if (binary) {
            TensorDescriptor right = query.inputs().get(1);
            if (right.dataType() != dataType || !right.shape().isFullyStatic()) return false;
            try {
                if (!ShapeBroadcast.broadcast(left.shape(), right.shape()).equals(output.shape())) {
                    return false;
                }
            } catch (IllegalArgumentException incompatible) { return false; }
        } else if (!left.shape().equals(output.shape())) return false;
        return query.inputs().stream().allMatch(CpuCapabilityProvider::resolved)
                && resolved(output);
    }

    private static boolean resolved(TensorDescriptor descriptor) {
        return descriptor.layout().isPresent();
    }
}
