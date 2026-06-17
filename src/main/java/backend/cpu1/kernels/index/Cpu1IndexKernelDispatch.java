package backend.cpu1.kernels.index;

import backend.cpu1.kernels.index.gather.Cpu1GatherLoops;
import backend.cpu1.kernels.index.gatheraxis.Cpu1GatherAxisLoops;
import backend.cpu1.kernels.index.gathernd.Cpu1GatherNdLoops;
import backend.cpu1.kernels.index.scatter.Cpu1ScatterLoops;
import backend.cpu1.kernels.index.scatter.Cpu1ScatterWriteLoops;
import backend.cpu1.kernels.index.takealongaxis.Cpu1TakeAlongAxisLoops;

/**
 * Resolves prepared index kernel ids to concrete kernels.
 */
public final class Cpu1IndexKernelDispatch {
    private Cpu1IndexKernelDispatch() {
    }

    public static Cpu1IndexKernel kernelFor(Cpu1IndexKernelId kernelId) {
        if (kernelId == null) {
            throw new IllegalArgumentException("kernelId cannot be null");
        }
        return switch (kernelId) {
            case GATHER_F32_I32_DENSE_ARRAY -> Cpu1GatherLoops::gatherF32I32DenseArray;
            case GATHER_F32_I64_DENSE_ARRAY -> Cpu1GatherLoops::gatherF32I64DenseArray;
            case GATHER_F64_I32_DENSE_ARRAY -> Cpu1GatherLoops::gatherF64I32DenseArray;
            case GATHER_F64_I64_DENSE_ARRAY -> Cpu1GatherLoops::gatherF64I64DenseArray;
            case GATHER_BF16_I32_DENSE_ARRAY -> Cpu1GatherLoops::gatherBf16I32DenseArray;
            case GATHER_BF16_I64_DENSE_ARRAY -> Cpu1GatherLoops::gatherBf16I64DenseArray;
            case GATHER_I32_I32_DENSE_ARRAY -> Cpu1GatherLoops::gatherI32I32DenseArray;
            case GATHER_I32_I64_DENSE_ARRAY -> Cpu1GatherLoops::gatherI32I64DenseArray;
            case GATHER_I64_I32_DENSE_ARRAY -> Cpu1GatherLoops::gatherI64I32DenseArray;
            case GATHER_I64_I64_DENSE_ARRAY -> Cpu1GatherLoops::gatherI64I64DenseArray;
            case GATHER_BOOL_I32_DENSE_ARRAY -> Cpu1GatherLoops::gatherBoolI32DenseArray;
            case GATHER_BOOL_I64_DENSE_ARRAY -> Cpu1GatherLoops::gatherBoolI64DenseArray;
            case GATHER_F32_I32_DENSE_SEGMENT -> Cpu1GatherLoops::gatherF32I32DenseSegment;
            case GATHER_F32_I64_DENSE_SEGMENT -> Cpu1GatherLoops::gatherF32I64DenseSegment;
            case GATHER_F64_I32_DENSE_SEGMENT -> Cpu1GatherLoops::gatherF64I32DenseSegment;
            case GATHER_F64_I64_DENSE_SEGMENT -> Cpu1GatherLoops::gatherF64I64DenseSegment;
            case GATHER_BF16_I32_DENSE_SEGMENT -> Cpu1GatherLoops::gatherBf16I32DenseSegment;
            case GATHER_BF16_I64_DENSE_SEGMENT -> Cpu1GatherLoops::gatherBf16I64DenseSegment;
            case GATHER_I32_I32_DENSE_SEGMENT -> Cpu1GatherLoops::gatherI32I32DenseSegment;
            case GATHER_I32_I64_DENSE_SEGMENT -> Cpu1GatherLoops::gatherI32I64DenseSegment;
            case GATHER_I64_I32_DENSE_SEGMENT -> Cpu1GatherLoops::gatherI64I32DenseSegment;
            case GATHER_I64_I64_DENSE_SEGMENT -> Cpu1GatherLoops::gatherI64I64DenseSegment;
            case GATHER_BOOL_I32_DENSE_SEGMENT -> Cpu1GatherLoops::gatherBoolI32DenseSegment;
            case GATHER_BOOL_I64_DENSE_SEGMENT -> Cpu1GatherLoops::gatherBoolI64DenseSegment;

            case GATHER_ND_F32_I32_DENSE_ARRAY -> Cpu1GatherNdLoops::gatherNdF32I32DenseArray;
            case GATHER_ND_F32_I64_DENSE_ARRAY -> Cpu1GatherNdLoops::gatherNdF32I64DenseArray;
            case GATHER_ND_F64_I32_DENSE_ARRAY -> Cpu1GatherNdLoops::gatherNdF64I32DenseArray;
            case GATHER_ND_F64_I64_DENSE_ARRAY -> Cpu1GatherNdLoops::gatherNdF64I64DenseArray;
            case GATHER_ND_BF16_I32_DENSE_ARRAY -> Cpu1GatherNdLoops::gatherNdBf16I32DenseArray;
            case GATHER_ND_BF16_I64_DENSE_ARRAY -> Cpu1GatherNdLoops::gatherNdBf16I64DenseArray;
            case GATHER_ND_I32_I32_DENSE_ARRAY -> Cpu1GatherNdLoops::gatherNdI32I32DenseArray;
            case GATHER_ND_I32_I64_DENSE_ARRAY -> Cpu1GatherNdLoops::gatherNdI32I64DenseArray;
            case GATHER_ND_I64_I32_DENSE_ARRAY -> Cpu1GatherNdLoops::gatherNdI64I32DenseArray;
            case GATHER_ND_I64_I64_DENSE_ARRAY -> Cpu1GatherNdLoops::gatherNdI64I64DenseArray;
            case GATHER_ND_BOOL_I32_DENSE_ARRAY -> Cpu1GatherNdLoops::gatherNdBoolI32DenseArray;
            case GATHER_ND_BOOL_I64_DENSE_ARRAY -> Cpu1GatherNdLoops::gatherNdBoolI64DenseArray;
            case GATHER_ND_F32_I32_DENSE_SEGMENT -> Cpu1GatherNdLoops::gatherNdF32I32DenseSegment;
            case GATHER_ND_F32_I64_DENSE_SEGMENT -> Cpu1GatherNdLoops::gatherNdF32I64DenseSegment;
            case GATHER_ND_F64_I32_DENSE_SEGMENT -> Cpu1GatherNdLoops::gatherNdF64I32DenseSegment;
            case GATHER_ND_F64_I64_DENSE_SEGMENT -> Cpu1GatherNdLoops::gatherNdF64I64DenseSegment;
            case GATHER_ND_BF16_I32_DENSE_SEGMENT -> Cpu1GatherNdLoops::gatherNdBf16I32DenseSegment;
            case GATHER_ND_BF16_I64_DENSE_SEGMENT -> Cpu1GatherNdLoops::gatherNdBf16I64DenseSegment;
            case GATHER_ND_I32_I32_DENSE_SEGMENT -> Cpu1GatherNdLoops::gatherNdI32I32DenseSegment;
            case GATHER_ND_I32_I64_DENSE_SEGMENT -> Cpu1GatherNdLoops::gatherNdI32I64DenseSegment;
            case GATHER_ND_I64_I32_DENSE_SEGMENT -> Cpu1GatherNdLoops::gatherNdI64I32DenseSegment;
            case GATHER_ND_I64_I64_DENSE_SEGMENT -> Cpu1GatherNdLoops::gatherNdI64I64DenseSegment;
            case GATHER_ND_BOOL_I32_DENSE_SEGMENT -> Cpu1GatherNdLoops::gatherNdBoolI32DenseSegment;
            case GATHER_ND_BOOL_I64_DENSE_SEGMENT -> Cpu1GatherNdLoops::gatherNdBoolI64DenseSegment;

            case GATHER_AXIS_F32_I32_DENSE_ARRAY -> Cpu1GatherAxisLoops::gatherAxisF32I32DenseArray;
            case GATHER_AXIS_F32_I64_DENSE_ARRAY -> Cpu1GatherAxisLoops::gatherAxisF32I64DenseArray;
            case GATHER_AXIS_F64_I32_DENSE_ARRAY -> Cpu1GatherAxisLoops::gatherAxisF64I32DenseArray;
            case GATHER_AXIS_F64_I64_DENSE_ARRAY -> Cpu1GatherAxisLoops::gatherAxisF64I64DenseArray;
            case GATHER_AXIS_BF16_I32_DENSE_ARRAY -> Cpu1GatherAxisLoops::gatherAxisBf16I32DenseArray;
            case GATHER_AXIS_BF16_I64_DENSE_ARRAY -> Cpu1GatherAxisLoops::gatherAxisBf16I64DenseArray;
            case GATHER_AXIS_I32_I32_DENSE_ARRAY -> Cpu1GatherAxisLoops::gatherAxisI32I32DenseArray;
            case GATHER_AXIS_I32_I64_DENSE_ARRAY -> Cpu1GatherAxisLoops::gatherAxisI32I64DenseArray;
            case GATHER_AXIS_I64_I32_DENSE_ARRAY -> Cpu1GatherAxisLoops::gatherAxisI64I32DenseArray;
            case GATHER_AXIS_I64_I64_DENSE_ARRAY -> Cpu1GatherAxisLoops::gatherAxisI64I64DenseArray;
            case GATHER_AXIS_BOOL_I32_DENSE_ARRAY -> Cpu1GatherAxisLoops::gatherAxisBoolI32DenseArray;
            case GATHER_AXIS_BOOL_I64_DENSE_ARRAY -> Cpu1GatherAxisLoops::gatherAxisBoolI64DenseArray;
            case GATHER_AXIS_F32_I32_DENSE_SEGMENT -> Cpu1GatherAxisLoops::gatherAxisF32I32DenseSegment;
            case GATHER_AXIS_F32_I64_DENSE_SEGMENT -> Cpu1GatherAxisLoops::gatherAxisF32I64DenseSegment;
            case GATHER_AXIS_F64_I32_DENSE_SEGMENT -> Cpu1GatherAxisLoops::gatherAxisF64I32DenseSegment;
            case GATHER_AXIS_F64_I64_DENSE_SEGMENT -> Cpu1GatherAxisLoops::gatherAxisF64I64DenseSegment;
            case GATHER_AXIS_BF16_I32_DENSE_SEGMENT -> Cpu1GatherAxisLoops::gatherAxisBf16I32DenseSegment;
            case GATHER_AXIS_BF16_I64_DENSE_SEGMENT -> Cpu1GatherAxisLoops::gatherAxisBf16I64DenseSegment;
            case GATHER_AXIS_I32_I32_DENSE_SEGMENT -> Cpu1GatherAxisLoops::gatherAxisI32I32DenseSegment;
            case GATHER_AXIS_I32_I64_DENSE_SEGMENT -> Cpu1GatherAxisLoops::gatherAxisI32I64DenseSegment;
            case GATHER_AXIS_I64_I32_DENSE_SEGMENT -> Cpu1GatherAxisLoops::gatherAxisI64I32DenseSegment;
            case GATHER_AXIS_I64_I64_DENSE_SEGMENT -> Cpu1GatherAxisLoops::gatherAxisI64I64DenseSegment;
            case GATHER_AXIS_BOOL_I32_DENSE_SEGMENT -> Cpu1GatherAxisLoops::gatherAxisBoolI32DenseSegment;
            case GATHER_AXIS_BOOL_I64_DENSE_SEGMENT -> Cpu1GatherAxisLoops::gatherAxisBoolI64DenseSegment;

            case TAKE_ALONG_AXIS_F32_I32_DENSE_ARRAY -> Cpu1TakeAlongAxisLoops::takeAlongAxisF32I32DenseArray;
            case TAKE_ALONG_AXIS_F32_I64_DENSE_ARRAY -> Cpu1TakeAlongAxisLoops::takeAlongAxisF32I64DenseArray;
            case TAKE_ALONG_AXIS_F64_I32_DENSE_ARRAY -> Cpu1TakeAlongAxisLoops::takeAlongAxisF64I32DenseArray;
            case TAKE_ALONG_AXIS_F64_I64_DENSE_ARRAY -> Cpu1TakeAlongAxisLoops::takeAlongAxisF64I64DenseArray;
            case TAKE_ALONG_AXIS_BF16_I32_DENSE_ARRAY -> Cpu1TakeAlongAxisLoops::takeAlongAxisBf16I32DenseArray;
            case TAKE_ALONG_AXIS_BF16_I64_DENSE_ARRAY -> Cpu1TakeAlongAxisLoops::takeAlongAxisBf16I64DenseArray;
            case TAKE_ALONG_AXIS_I32_I32_DENSE_ARRAY -> Cpu1TakeAlongAxisLoops::takeAlongAxisI32I32DenseArray;
            case TAKE_ALONG_AXIS_I32_I64_DENSE_ARRAY -> Cpu1TakeAlongAxisLoops::takeAlongAxisI32I64DenseArray;
            case TAKE_ALONG_AXIS_I64_I32_DENSE_ARRAY -> Cpu1TakeAlongAxisLoops::takeAlongAxisI64I32DenseArray;
            case TAKE_ALONG_AXIS_I64_I64_DENSE_ARRAY -> Cpu1TakeAlongAxisLoops::takeAlongAxisI64I64DenseArray;
            case TAKE_ALONG_AXIS_BOOL_I32_DENSE_ARRAY -> Cpu1TakeAlongAxisLoops::takeAlongAxisBoolI32DenseArray;
            case TAKE_ALONG_AXIS_BOOL_I64_DENSE_ARRAY -> Cpu1TakeAlongAxisLoops::takeAlongAxisBoolI64DenseArray;
            case TAKE_ALONG_AXIS_F32_I32_DENSE_SEGMENT -> Cpu1TakeAlongAxisLoops::takeAlongAxisF32I32DenseSegment;
            case TAKE_ALONG_AXIS_F32_I64_DENSE_SEGMENT -> Cpu1TakeAlongAxisLoops::takeAlongAxisF32I64DenseSegment;
            case TAKE_ALONG_AXIS_F64_I32_DENSE_SEGMENT -> Cpu1TakeAlongAxisLoops::takeAlongAxisF64I32DenseSegment;
            case TAKE_ALONG_AXIS_F64_I64_DENSE_SEGMENT -> Cpu1TakeAlongAxisLoops::takeAlongAxisF64I64DenseSegment;
            case TAKE_ALONG_AXIS_BF16_I32_DENSE_SEGMENT -> Cpu1TakeAlongAxisLoops::takeAlongAxisBf16I32DenseSegment;
            case TAKE_ALONG_AXIS_BF16_I64_DENSE_SEGMENT -> Cpu1TakeAlongAxisLoops::takeAlongAxisBf16I64DenseSegment;
            case TAKE_ALONG_AXIS_I32_I32_DENSE_SEGMENT -> Cpu1TakeAlongAxisLoops::takeAlongAxisI32I32DenseSegment;
            case TAKE_ALONG_AXIS_I32_I64_DENSE_SEGMENT -> Cpu1TakeAlongAxisLoops::takeAlongAxisI32I64DenseSegment;
            case TAKE_ALONG_AXIS_I64_I32_DENSE_SEGMENT -> Cpu1TakeAlongAxisLoops::takeAlongAxisI64I32DenseSegment;
            case TAKE_ALONG_AXIS_I64_I64_DENSE_SEGMENT -> Cpu1TakeAlongAxisLoops::takeAlongAxisI64I64DenseSegment;
            case TAKE_ALONG_AXIS_BOOL_I32_DENSE_SEGMENT -> Cpu1TakeAlongAxisLoops::takeAlongAxisBoolI32DenseSegment;
            case TAKE_ALONG_AXIS_BOOL_I64_DENSE_SEGMENT -> Cpu1TakeAlongAxisLoops::takeAlongAxisBoolI64DenseSegment;

            case SCATTER_ADD_F32_I32_DENSE_ARRAY -> Cpu1IndexKernel.withUpdates(Cpu1ScatterLoops::scatterAddF32I32DenseArray);
            case SCATTER_ADD_F32_I64_DENSE_ARRAY -> Cpu1IndexKernel.withUpdates(Cpu1ScatterLoops::scatterAddF32I64DenseArray);
            case SCATTER_ADD_F64_I32_DENSE_ARRAY -> Cpu1IndexKernel.withUpdates(Cpu1ScatterLoops::scatterAddF64I32DenseArray);
            case SCATTER_ADD_F64_I64_DENSE_ARRAY -> Cpu1IndexKernel.withUpdates(Cpu1ScatterLoops::scatterAddF64I64DenseArray);
            case SCATTER_ADD_BF16_I32_DENSE_ARRAY -> Cpu1IndexKernel.withUpdates(Cpu1ScatterLoops::scatterAddBf16I32DenseArray);
            case SCATTER_ADD_BF16_I64_DENSE_ARRAY -> Cpu1IndexKernel.withUpdates(Cpu1ScatterLoops::scatterAddBf16I64DenseArray);
            case SCATTER_ADD_F32_I32_DENSE_SEGMENT -> Cpu1IndexKernel.withUpdates(Cpu1ScatterLoops::scatterAddF32I32DenseSegment);
            case SCATTER_ADD_F32_I64_DENSE_SEGMENT -> Cpu1IndexKernel.withUpdates(Cpu1ScatterLoops::scatterAddF32I64DenseSegment);
            case SCATTER_ADD_F64_I32_DENSE_SEGMENT -> Cpu1IndexKernel.withUpdates(Cpu1ScatterLoops::scatterAddF64I32DenseSegment);
            case SCATTER_ADD_F64_I64_DENSE_SEGMENT -> Cpu1IndexKernel.withUpdates(Cpu1ScatterLoops::scatterAddF64I64DenseSegment);
            case SCATTER_ADD_BF16_I32_DENSE_SEGMENT -> Cpu1IndexKernel.withUpdates(Cpu1ScatterLoops::scatterAddBf16I32DenseSegment);
            case SCATTER_ADD_BF16_I64_DENSE_SEGMENT -> Cpu1IndexKernel.withUpdates(Cpu1ScatterLoops::scatterAddBf16I64DenseSegment);

            case SCATTER_AXIS_ADD_F32_I32_DENSE_ARRAY -> Cpu1IndexKernel.withUpdates(Cpu1ScatterLoops::scatterAxisAddF32I32DenseArray);
            case SCATTER_AXIS_ADD_F32_I64_DENSE_ARRAY -> Cpu1IndexKernel.withUpdates(Cpu1ScatterLoops::scatterAxisAddF32I64DenseArray);
            case SCATTER_AXIS_ADD_F64_I32_DENSE_ARRAY -> Cpu1IndexKernel.withUpdates(Cpu1ScatterLoops::scatterAxisAddF64I32DenseArray);
            case SCATTER_AXIS_ADD_F64_I64_DENSE_ARRAY -> Cpu1IndexKernel.withUpdates(Cpu1ScatterLoops::scatterAxisAddF64I64DenseArray);
            case SCATTER_AXIS_ADD_BF16_I32_DENSE_ARRAY -> Cpu1IndexKernel.withUpdates(Cpu1ScatterLoops::scatterAxisAddBf16I32DenseArray);
            case SCATTER_AXIS_ADD_BF16_I64_DENSE_ARRAY -> Cpu1IndexKernel.withUpdates(Cpu1ScatterLoops::scatterAxisAddBf16I64DenseArray);
            case SCATTER_AXIS_ADD_F32_I32_DENSE_SEGMENT -> Cpu1IndexKernel.withUpdates(Cpu1ScatterLoops::scatterAxisAddF32I32DenseSegment);
            case SCATTER_AXIS_ADD_F32_I64_DENSE_SEGMENT -> Cpu1IndexKernel.withUpdates(Cpu1ScatterLoops::scatterAxisAddF32I64DenseSegment);
            case SCATTER_AXIS_ADD_F64_I32_DENSE_SEGMENT -> Cpu1IndexKernel.withUpdates(Cpu1ScatterLoops::scatterAxisAddF64I32DenseSegment);
            case SCATTER_AXIS_ADD_F64_I64_DENSE_SEGMENT -> Cpu1IndexKernel.withUpdates(Cpu1ScatterLoops::scatterAxisAddF64I64DenseSegment);
            case SCATTER_AXIS_ADD_BF16_I32_DENSE_SEGMENT -> Cpu1IndexKernel.withUpdates(Cpu1ScatterLoops::scatterAxisAddBf16I32DenseSegment);
            case SCATTER_AXIS_ADD_BF16_I64_DENSE_SEGMENT -> Cpu1IndexKernel.withUpdates(Cpu1ScatterLoops::scatterAxisAddBf16I64DenseSegment);

            case SCATTER_ELEMENTS_F32_I32_DENSE_ARRAY,
                 SCATTER_ELEMENTS_F32_I64_DENSE_ARRAY,
                 SCATTER_ELEMENTS_F64_I32_DENSE_ARRAY,
                 SCATTER_ELEMENTS_F64_I64_DENSE_ARRAY,
                 SCATTER_ELEMENTS_BF16_I32_DENSE_ARRAY,
                 SCATTER_ELEMENTS_BF16_I64_DENSE_ARRAY,
                 SCATTER_ELEMENTS_I32_I32_DENSE_ARRAY,
                 SCATTER_ELEMENTS_I32_I64_DENSE_ARRAY,
                 SCATTER_ELEMENTS_I64_I32_DENSE_ARRAY,
                 SCATTER_ELEMENTS_I64_I64_DENSE_ARRAY,
                 SCATTER_ELEMENTS_BOOL_I32_DENSE_ARRAY,
                 SCATTER_ELEMENTS_BOOL_I64_DENSE_ARRAY ->
                    Cpu1IndexKernel.withUpdates(Cpu1ScatterWriteLoops::scatterElementsDenseArray);
            case SCATTER_ELEMENTS_F32_I32_DENSE_SEGMENT,
                 SCATTER_ELEMENTS_F32_I64_DENSE_SEGMENT,
                 SCATTER_ELEMENTS_F64_I32_DENSE_SEGMENT,
                 SCATTER_ELEMENTS_F64_I64_DENSE_SEGMENT,
                 SCATTER_ELEMENTS_BF16_I32_DENSE_SEGMENT,
                 SCATTER_ELEMENTS_BF16_I64_DENSE_SEGMENT,
                 SCATTER_ELEMENTS_I32_I32_DENSE_SEGMENT,
                 SCATTER_ELEMENTS_I32_I64_DENSE_SEGMENT,
                 SCATTER_ELEMENTS_I64_I32_DENSE_SEGMENT,
                 SCATTER_ELEMENTS_I64_I64_DENSE_SEGMENT,
                 SCATTER_ELEMENTS_BOOL_I32_DENSE_SEGMENT,
                 SCATTER_ELEMENTS_BOOL_I64_DENSE_SEGMENT ->
                    Cpu1IndexKernel.withUpdates(Cpu1ScatterWriteLoops::scatterElementsDenseSegment);
            case SCATTER_ND_F32_I32_DENSE_ARRAY,
                 SCATTER_ND_F32_I64_DENSE_ARRAY,
                 SCATTER_ND_F64_I32_DENSE_ARRAY,
                 SCATTER_ND_F64_I64_DENSE_ARRAY,
                 SCATTER_ND_BF16_I32_DENSE_ARRAY,
                 SCATTER_ND_BF16_I64_DENSE_ARRAY,
                 SCATTER_ND_I32_I32_DENSE_ARRAY,
                 SCATTER_ND_I32_I64_DENSE_ARRAY,
                 SCATTER_ND_I64_I32_DENSE_ARRAY,
                 SCATTER_ND_I64_I64_DENSE_ARRAY,
                 SCATTER_ND_BOOL_I32_DENSE_ARRAY,
                 SCATTER_ND_BOOL_I64_DENSE_ARRAY ->
                    Cpu1IndexKernel.withUpdates(Cpu1ScatterWriteLoops::scatterNdDenseArray);
            case SCATTER_ND_F32_I32_DENSE_SEGMENT,
                 SCATTER_ND_F32_I64_DENSE_SEGMENT,
                 SCATTER_ND_F64_I32_DENSE_SEGMENT,
                 SCATTER_ND_F64_I64_DENSE_SEGMENT,
                 SCATTER_ND_BF16_I32_DENSE_SEGMENT,
                 SCATTER_ND_BF16_I64_DENSE_SEGMENT,
                 SCATTER_ND_I32_I32_DENSE_SEGMENT,
                 SCATTER_ND_I32_I64_DENSE_SEGMENT,
                 SCATTER_ND_I64_I32_DENSE_SEGMENT,
                 SCATTER_ND_I64_I64_DENSE_SEGMENT,
                 SCATTER_ND_BOOL_I32_DENSE_SEGMENT,
                 SCATTER_ND_BOOL_I64_DENSE_SEGMENT ->
                    Cpu1IndexKernel.withUpdates(Cpu1ScatterWriteLoops::scatterNdDenseSegment);
        };
    }
}
