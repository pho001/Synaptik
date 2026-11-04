package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.model.graph.CompiledScanBody;
import io.github.pho001.synaptik.model.operation.scan.ScanAttrs;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.ScanActivityPlan;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Cold physical binding facts for one independently scoped native Scan body.
 *
 * <p>Ordinals are local to this plan; they never identify an outer graph value. The ordinary
 * instruction list is stored once, independent of the time count. Every local writer has its own
 * physical region; external sequence/invariant formals are direct read-only offset bindings.
 * Only demanded outer roles reserve history. Mutable buffers, row-presence bits and time/row
 * coordinates belong to an invocation, not to this immutable prepared plan.</p>
 *
 * @param header exact scalar Scan header
 * @param body exact neutral captured body, retained by reference
 * @param values local values in graph-value order, with completed ordinary physical layouts
 * @param units ordinary units in producer order, never time-expanded
 * @param formals ordered carry, sequence-step and invariant local ordinals
 * @param roles ordered final-carry, emission and residual local ordinals, retaining repetitions
 * @param demanded increasing outer role indices whose results are materialized
 * @param activity exact optional Compiler-generated structural-presence association
 * @param batch fixed length-control extent, or zero in all-valid mode
 * @param workspaceBytes checked non-history invocation scratch size in bytes, including control
 * @param controlByteOffset aligned invocation workspace byte offset of the control word array
 * @param controlWords exact word count, including five bounded failure words and simultaneous-presence staging
 */
record MetalScanPlan(ScanAttrs header, CompiledScanBody body, List<Value> values,
        List<Unit> units, List<Integer> formals, List<Integer> roles, List<Integer> demanded,
        Optional<ScanActivityPlan> activity, long batch, long workspaceBytes,
        long controlByteOffset, long controlWords) {
    MetalScanPlan {
        Objects.requireNonNull(header, "header");
        Objects.requireNonNull(body, "body");
        values = List.copyOf(values);
        units = List.copyOf(units);
        formals = List.copyOf(formals);
        roles = List.copyOf(roles);
        demanded = List.copyOf(demanded);
        Objects.requireNonNull(activity, "activity");
        if (batch < 0 || workspaceBytes < 0 || controlByteOffset < 0 || controlWords < 15) {
            throw new IllegalArgumentException("negative Scan geometry");
        }
    }

    /**
     * A local physical value and its independent original-row coordinate.
     * @param descriptor exact resolved ordinary descriptor
     * @param rowAxis original ROW axis, or -1 for SHARED/all-valid
     * @param terminal whether a row contraction occurs on its producing dependency path
     * @param byteOffset aligned byte offset in invocation-local storage, or -1 for external formals
     * @param byteSize complete physical span in bytes, including an affine view's holes/offset
     * @param rowInner logical element product after the original row axis, zero for SHARED
     */
    record Value(TensorDescriptor descriptor, int rowAxis, boolean terminal, long byteOffset,
            long byteSize, long rowInner) {
        Value {
            Objects.requireNonNull(descriptor, "descriptor");
            if (rowAxis < -1 || rowAxis >= descriptor.shape().rank() || byteOffset < -1
                    || byteSize < 0 || rowInner < 0) {
                throw new IllegalArgumentException("invalid Scan local physical value");
            }
        }
    }

    /**
     * A once-compiled ordinary operation and its control binding, not derivative algebra.
     * @param node ordinary Metal instruction, using local value ordinals
     * @param scope activity scope, or -1 for unconditional structural presence
     * @param leftScope optional ADD's left scope, or -1 for an ordinary operation
     * @param rightScope optional ADD's right scope, or -1 for an ordinary operation
     * @param rowAxis common output row axis, or -1 for SHARED
     * @param contraction whether this unit eliminates the identified row coordinate
     * @param accumulator numeric carry position updated by a separate completed temporal ADD,
     *     or -1 for an ordinary operation
     * @param scratchOffset aligned ordinary-unit workspace byte offset, or -1 for no scratch
     * @param scratchBytes exact ordinary-owner workspace bytes, zero for no scratch
     */
    record Unit(MetalMpsGraphProgram.Node node, int scope, int leftScope, int rightScope,
            int rowAxis, boolean contraction, int accumulator, long scratchOffset, long scratchBytes) {
        Unit {
            Objects.requireNonNull(node, "node");
            if (scope < -1 || leftScope < -1 || rightScope < -1 || rowAxis < -1 || accumulator < -1
                    || scratchOffset < -1 || scratchBytes < 0 || ((scratchOffset < 0) != (scratchBytes == 0))
                    || ((leftScope < 0) != (rightScope < 0))) {
                throw new IllegalArgumentException("invalid Scan unit control");
            }
        }
    }
}
