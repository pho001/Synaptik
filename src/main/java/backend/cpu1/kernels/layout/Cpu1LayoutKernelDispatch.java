package backend.cpu1.kernels.layout;

import backend.cpu1.kernels.layout.concat.Cpu1ConcatLayoutLoops;
import backend.cpu1.kernels.layout.copy.Cpu1CopyLayoutLoops;
import backend.cpu1.kernels.layout.pad.Cpu1PadLayoutLoops;
import backend.cpu1.kernels.layout.tile.Cpu1TileLayoutLoops;
import backend.cpu1.kernels.layout.unfold.Cpu1UnfoldLayoutLoops;

import java.util.Objects;

/**
 * Resolves prepared layout kernel ids to concrete layout loop runners outside the hot path.
 */
public final class Cpu1LayoutKernelDispatch {
    private Cpu1LayoutKernelDispatch() {
    }

    public static Cpu1LayoutKernel runnerFor(Cpu1LayoutKernelId kernelId) {
        Objects.requireNonNull(kernelId, "kernelId cannot be null");
        return switch (kernelId) {
            case NOOP_ALIAS, RESHAPE_ALIAS, EXPAND_ALIAS, SELECT_ALIAS, SLICE_ALIAS, PERMUTE_ALIAS,
                    EXPAND_DIMS_ALIAS, SQUEEZE_ALIAS -> Cpu1AliasLayoutKernel::runAlias;
            case RESHAPE_COPY_LINEARIZED_SCALAR -> Cpu1CopyLayoutLoops::reshapeCopyLinearizedScalar;
            case CONTIGUOUS_COPY_SCALAR -> Cpu1CopyLayoutLoops::contiguousCopyScalar;
            case CONTIGUOUS_COPY_VECTOR -> Cpu1CopyLayoutLoops::contiguousCopyVector;
            case CONTIGUOUS_OFFSET_DENSE_BLOCK_COPY_SCALAR -> Cpu1CopyLayoutLoops::contiguousOffsetDenseBlockScalar;
            case CONTIGUOUS_OFFSET_DENSE_BLOCK_COPY_VECTOR -> Cpu1CopyLayoutLoops::contiguousOffsetDenseBlockVector;
            case CONCAT_COPY_SCALAR -> Cpu1ConcatLayoutLoops::concatScalar;
            case CONCAT_AXIS0_BLOCK_COPY_SCALAR -> Cpu1ConcatLayoutLoops::concatAxis0BlockScalar;
            case CONCAT_AXIS0_BLOCK_COPY_VECTOR -> Cpu1ConcatLayoutLoops::concatAxis0BlockVector;
            case CONCAT_MIDDLE_AXIS_BLOCK_COPY_SCALAR -> Cpu1ConcatLayoutLoops::concatMiddleAxisBlockScalar;
            case CONCAT_MIDDLE_AXIS_BLOCK_COPY_VECTOR -> Cpu1ConcatLayoutLoops::concatMiddleAxisBlockVector;
            case CONCAT_INNER_AXIS_BLOCK_COPY_SCALAR -> Cpu1ConcatLayoutLoops::concatInnerAxisBlockScalar;
            case CONCAT_INNER_AXIS_BLOCK_COPY_VECTOR -> Cpu1ConcatLayoutLoops::concatInnerAxisBlockVector;
            case PAD_COPY_SCALAR -> Cpu1PadLayoutLoops::padScalar;
            case PAD_COPY_VECTOR -> Cpu1PadLayoutLoops::padVector;
            case PAD_DENSE_INNER_BLOCK_COPY_SCALAR -> Cpu1PadLayoutLoops::padDenseInnerBlockScalar;
            case PAD_DENSE_INNER_BLOCK_COPY_VECTOR -> Cpu1PadLayoutLoops::padDenseInnerBlockVector;
            case TILE_COPY_SCALAR -> Cpu1TileLayoutLoops::tileScalar;
            case TILE_AXIS0_BLOCK_COPY_SCALAR -> Cpu1TileLayoutLoops::tileAxis0BlockScalar;
            case TILE_AXIS0_BLOCK_COPY_VECTOR -> Cpu1TileLayoutLoops::tileAxis0BlockVector;
            case TILE_LAST_AXIS_BLOCK_COPY_SCALAR -> Cpu1TileLayoutLoops::tileLastAxisBlockScalar;
            case TILE_LAST_AXIS_BLOCK_COPY_VECTOR -> Cpu1TileLayoutLoops::tileLastAxisBlockVector;
            case TILE_DENSE_BLOCK_REPEAT_SCALAR -> Cpu1TileLayoutLoops::tileDenseBlockRepeatScalar;
            case TILE_DENSE_BLOCK_REPEAT_VECTOR -> Cpu1TileLayoutLoops::tileDenseBlockRepeatVector;
            case TILE_DENSE_MULTI_AXIS_BLOCK_COPY_SCALAR -> Cpu1TileLayoutLoops::tileDenseMultiAxisBlockScalar;
            case TILE_DENSE_MULTI_AXIS_BLOCK_COPY_VECTOR -> Cpu1TileLayoutLoops::tileDenseMultiAxisBlockVector;
            case UNFOLD_AXIS_COPY_SCALAR -> Cpu1UnfoldLayoutLoops::unfoldAxisScalar;
            case UNFOLD_AXIS_LAST_AXIS_BLOCK_COPY_SCALAR -> Cpu1UnfoldLayoutLoops::unfoldAxisLastAxisBlockScalar;
            case UNFOLD_AXIS_LAST_AXIS_BLOCK_COPY_VECTOR -> Cpu1UnfoldLayoutLoops::unfoldAxisLastAxisBlockVector;
            case UNFOLD2D_COPY_SCALAR -> Cpu1UnfoldLayoutLoops::unfold2dScalar;
            case FOLD2D_NON_OVERLAP_DIRECT_SCALAR -> Cpu1UnfoldLayoutLoops::fold2dNonOverlapDirectScalar;
            case FOLD2D_COPY_SCALAR -> Cpu1UnfoldLayoutLoops::fold2dScalar;
        };
    }
}
