package io.github.pho001.synaptik.model.operation.layout;

import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Identifies backend-independent meanings for materializing and accumulating sliding windows.
 *
 * <p>{@link #UNFOLD_AXIS} and {@link #FOLD_AXIS} are general-axis operations with no image,
 * padding, or dilation assumption. For example, unfolding conceptual Shape {@code [2, 5, 3]}
 * along axis one with size three and step one produces conceptual Shape {@code [2, 3, 3, 3]}:
 * the selected extent becomes three window positions and the window size is appended as the final
 * axis. Unfold means materialized window semantics and does not promise a storage view.</p>
 *
 * <p>Folding is the scatter-add adjoint. Conceptual windows of Shape {@code [3, 3]} with values
 * {@code [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}, axis zero, output size five, and step one produce
 * conceptual Shape {@code [5]} with values {@code [1, 6, 15, 14, 9]}; overlapping contributions
 * are summed. The explicit output size is required because window count, window size, and step do
 * not identify trailing uncovered positions. Public Tensor construction validates this geometry
 * and records the same backend-independent meaning without adding compiler behavior to the
 * model.</p>
 *
 * <p>{@link #UNFOLD2D} and {@link #FOLD2D} use NCHW (batch, channel, height, width) image geometry.
 * Two-dimensional unfold is im2col: conceptual Shape {@code [1, 1, 3, 3]} with a 2-by-2 kernel,
 * unit stride and dilation, zero symmetric padding, and floor mode produces canonical columns of
 * conceptual Shape {@code [1, 4, 4]}. Fold is overlap-summing col2im into an explicit
 * {@code [1, 1, 3, 3]} output: the center receives four contributions while each corner receives
 * one, with no overlap averaging.</p>
 *
 * <p>The exact kind-to-attributes pairings are UNFOLD_AXIS with {@link UnfoldAxisAttrs}, FOLD_AXIS
 * with {@link FoldAxisAttrs}, UNFOLD2D with either {@link Window2dAttrs} for conceptual
 * positive-zero padding or {@link Unfold2dAttrs} for one exact typed padding value, and FOLD2D
 * with {@link Fold2dAttrs}. Family-owned signatures enforce each exact pairing and declare one
 * input and one output. This enum performs no Tensor construction, Shape calculation, sampling,
 * accumulation, layout or storage selection, gradient construction, graph/compiler work,
 * lowering, backend dispatch, or execution. Public Tensor construction currently exists for all
 * four meanings.</p>
 */
public enum WindowTransformKind implements OperationKind {
    /**
     * Materializes no-padding, no-dilation sliding windows along the normalized axis in
     * {@link UnfoldAxisAttrs}, replacing that extent with window positions and appending window
     * size as the final result axis.
     */
    UNFOLD_AXIS,

    /**
     * Scatter-adds the final input window dimension along the normalized target axis in
     * {@link FoldAxisAttrs}, restoring its explicit output extent and summing overlaps.
     *
     * <p>Public {@code Tensor.foldAxis} construction records this semantic after validating the
     * rank, numeric type, static window geometry, and requested output extent. Gradient
     * construction remains compiler-owned and is not implied by this kind.</p>
     */
    FOLD_AXIS,

    /**
     * Materializes a rank-four NCHW input as canonical rank-three im2col columns parameterized by
     * direct conceptual-zero {@link Window2dAttrs} or explicit-padding {@link Unfold2dAttrs}.
     */
    UNFOLD2D,

    /**
     * Accumulates canonical rank-three columns into the explicit rank-four NCHW result described
     * by {@link Fold2dAttrs}, summing rather than averaging overlapping contributions.
     */
    FOLD2D;

    private static final List<OperationSignature> UNFOLD_AXIS_SIGNATURES =
            List.of(OperationSignature.fixed(UnfoldAxisAttrs.class, 1, 1));
    private static final List<OperationSignature> FOLD_AXIS_SIGNATURES =
            List.of(OperationSignature.fixed(FoldAxisAttrs.class, 1, 1));
    private static final List<OperationSignature> UNFOLD_2D_SIGNATURES = List.of(
            OperationSignature.fixed(Window2dAttrs.class, 1, 1),
            OperationSignature.fixed(Unfold2dAttrs.class, 1, 1));
    private static final List<OperationSignature> FOLD_2D_SIGNATURES =
            List.of(OperationSignature.fixed(Fold2dAttrs.class, 1, 1));

    /**
     * Returns the exact one-input, one-output attributes variant accepted by this window kind.
     *
     * @return the stable immutable signature list selected by this kind
     */
    @Override
    public List<OperationSignature> signatures() {
        return switch (this) {
            case UNFOLD_AXIS -> UNFOLD_AXIS_SIGNATURES;
            case FOLD_AXIS -> FOLD_AXIS_SIGNATURES;
            case UNFOLD2D -> UNFOLD_2D_SIGNATURES;
            case FOLD2D -> FOLD_2D_SIGNATURES;
        };
    }
}
