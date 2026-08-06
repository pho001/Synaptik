package io.github.pho001.synaptik.backend.cpu;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutKind;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.Objects;

/**
 * Reports the executable semantic coverage currently delivered by the CPU backend.
 *
 * <p>The provider has a stable CPU ownership identity and advertises only parameterless
 * {@code FLOAT64} {@code ADD}, exact {@code GELU}, and {@code MUL} occurrences with fully static,
 * equal, resolved canonical-dense geometry. Complete-partition lowering applies the additional
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
     * Reports whether an occurrence belongs to the exact implemented dense semantic set.
     * Every input and output must already have a resolved canonical-dense, zero-offset, non-view
     * layout; unresolved, offset, strided, and view layouts fail closed.
     *
     * @param query non-null immutable operation occurrence to validate structurally
     * @return {@code true} only for the exact implemented occurrence matrix; otherwise
     *     {@code false}
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
        if (dataType != DataType.FLOAT64
                || output.dataType() != dataType
                || !left.shape().isFullyStatic()
                || !left.shape().equals(output.shape())) {
            return false;
        }
        if (binary && (query.inputs().get(1).dataType() != dataType
                || !query.inputs().get(1).shape().equals(left.shape()))) return false;
        return query.inputs().stream().allMatch(CpuCapabilityProvider::canonicalResolved)
                && canonicalResolved(output);
    }

    private static boolean canonicalResolved(TensorDescriptor descriptor) {
        return descriptor.layout().filter(layout ->
                layout.kind() == LayoutKind.DENSE_CONTIGUOUS
                        && layout.storageOffset() == 0
                        && !layout.isView()).isPresent();
    }
}
