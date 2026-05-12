package backend.cpu.kernels.index;

import backend.cpu.kernels.CpuKernelContext;
import operations.index.ScatterReduction;
import tensor.Tensor;

final class IndexExecutor {
    private IndexExecutor() {
    }

    static void gatherF64(Tensor input, Tensor indices, Tensor out, int dimension, CpuKernelContext context) {
        IndexReadWriteBackend.runF64(input, indices, out, dimension);
    }

    static void gatherF32(Tensor input, Tensor indices, Tensor out, int dimension, CpuKernelContext context) {
        IndexReadWriteBackend.runF32(input, indices, out, dimension);
    }

    static void gatherBF16(Tensor input, Tensor indices, Tensor out, int dimension, CpuKernelContext context) {
        IndexReadWriteBackend.runBF16(input, indices, out, dimension);
    }

    static void gatherBOOL(Tensor input, Tensor indices, Tensor out, int dimension, CpuKernelContext context) {
        IndexReadWriteBackend.runBOOL(input, indices, out, dimension);
    }

    static void gatherI32(Tensor input, Tensor indices, Tensor out, int dimension, CpuKernelContext context) {
        IndexReadWriteBackend.runI32(input, indices, out, dimension);
    }

    static void gatherAxisF64(Tensor input, Tensor indices, Tensor out, int axis, CpuKernelContext context) {
        IndexReadWriteBackend.gatherAxisF64(input, indices, out, axis);
    }

    static void gatherAxisF32(Tensor input, Tensor indices, Tensor out, int axis, CpuKernelContext context) {
        IndexReadWriteBackend.gatherAxisF32(input, indices, out, axis);
    }

    static void gatherAxisBF16(Tensor input, Tensor indices, Tensor out, int axis, CpuKernelContext context) {
        IndexReadWriteBackend.gatherAxisBF16(input, indices, out, axis);
    }

    static void gatherAxisBOOL(Tensor input, Tensor indices, Tensor out, int axis, CpuKernelContext context) {
        IndexReadWriteBackend.gatherAxisBOOL(input, indices, out, axis);
    }

    static void gatherAxisI32(Tensor input, Tensor indices, Tensor out, int axis, CpuKernelContext context) {
        IndexReadWriteBackend.gatherAxisI32(input, indices, out, axis);
    }

    static void gatherNdF64(Tensor input, Tensor indices, Tensor out, int batchDims, CpuKernelContext context) {
        IndexReadWriteBackend.gatherNdF64(input, indices, out, batchDims);
    }

    static void gatherNdF32(Tensor input, Tensor indices, Tensor out, int batchDims, CpuKernelContext context) {
        IndexReadWriteBackend.gatherNdF32(input, indices, out, batchDims);
    }

    static void gatherNdBF16(Tensor input, Tensor indices, Tensor out, int batchDims, CpuKernelContext context) {
        IndexReadWriteBackend.gatherNdBF16(input, indices, out, batchDims);
    }

    static void gatherNdBOOL(Tensor input, Tensor indices, Tensor out, int batchDims, CpuKernelContext context) {
        IndexReadWriteBackend.gatherNdBOOL(input, indices, out, batchDims);
    }

    static void gatherNdI32(Tensor input, Tensor indices, Tensor out, int batchDims, CpuKernelContext context) {
        IndexReadWriteBackend.gatherNdI32(input, indices, out, batchDims);
    }

    static void gatherNdGradF64(Tensor indices, Tensor outGrad, Tensor node, int batchDims, CpuKernelContext context) {
        IndexReadWriteBackend.gatherNdGradF64(indices, outGrad, node, batchDims);
    }

    static void gatherNdGradF32(Tensor indices, Tensor outGrad, Tensor node, int batchDims, CpuKernelContext context) {
        IndexReadWriteBackend.gatherNdGradF32(indices, outGrad, node, batchDims);
    }

    static void gatherNdGradBF16(Tensor indices, Tensor outGrad, Tensor node, int batchDims, CpuKernelContext context) {
        IndexReadWriteBackend.gatherNdGradBF16(indices, outGrad, node, batchDims);
    }

    static void gatherGradF64(Tensor indices, Tensor outGrad, Tensor node, int dimension, CpuKernelContext context) {
        IndexReadWriteBackend.scatterF64(indices, outGrad, node, dimension);
    }

    static void gatherGradF32(Tensor indices, Tensor outGrad, Tensor node, int dimension, CpuKernelContext context) {
        IndexReadWriteBackend.scatterF32(indices, outGrad, node, dimension);
    }

    static void gatherGradBF16(Tensor indices, Tensor outGrad, Tensor node, int dimension, CpuKernelContext context) {
        IndexReadWriteBackend.scatterBF16(indices, outGrad, node, dimension);
    }

    static void gatherAxisGradF64(Tensor indices, Tensor outGrad, Tensor node, int axis, CpuKernelContext context) {
        IndexReadWriteBackend.gatherAxisGradF64(indices, outGrad, node, axis);
    }

    static void gatherAxisGradF32(Tensor indices, Tensor outGrad, Tensor node, int axis, CpuKernelContext context) {
        IndexReadWriteBackend.gatherAxisGradF32(indices, outGrad, node, axis);
    }

    static void gatherAxisGradBF16(Tensor indices, Tensor outGrad, Tensor node, int axis, CpuKernelContext context) {
        IndexReadWriteBackend.gatherAxisGradBF16(indices, outGrad, node, axis);
    }

    static void takeAlongAxisF64(Tensor input, Tensor indices, Tensor out, int dimension, CpuKernelContext context) {
        IndexReadWriteBackend.takeAlongAxisF64(input, indices, out, dimension);
    }

    static void takeAlongAxisF32(Tensor input, Tensor indices, Tensor out, int dimension, CpuKernelContext context) {
        IndexReadWriteBackend.takeAlongAxisF32(input, indices, out, dimension);
    }

    static void takeAlongAxisBF16(Tensor input, Tensor indices, Tensor out, int dimension, CpuKernelContext context) {
        IndexReadWriteBackend.takeAlongAxisF16(input, indices, out, dimension);
    }

    static void takeAlongAxisBOOL(Tensor input, Tensor indices, Tensor out, int dimension, CpuKernelContext context) {
        IndexReadWriteBackend.takeAlongAxisBOOL(input, indices, out, dimension);
    }

    static void takeAlongAxisI32(Tensor input, Tensor indices, Tensor out, int dimension, CpuKernelContext context) {
        IndexReadWriteBackend.takeAlongAxisI32(input, indices, out, dimension);
    }

    static void takeAlongAxisGradF64(Tensor indices, Tensor outGrad, Tensor node, int dimension, CpuKernelContext context) {
        IndexReadWriteBackend.takeAlongAxisScatterF64(indices, outGrad, node, dimension);
    }

    static void takeAlongAxisGradF32(Tensor indices, Tensor outGrad, Tensor node, int dimension, CpuKernelContext context) {
        IndexReadWriteBackend.takeAlongAxisScatterF32(indices, outGrad, node, dimension);
    }

    static void takeAlongAxisGradBF16(Tensor indices, Tensor outGrad, Tensor node, int dimension, CpuKernelContext context) {
        IndexReadWriteBackend.takeAlongAxisScatterBF16(indices, outGrad, node, dimension);
    }

    static void scatterAddF64(Tensor base, Tensor indices, Tensor src, Tensor out, int dimension, CpuKernelContext context) {
        IndexReadWriteBackend.scatterAddF64(base, indices, src, out, dimension);
    }

    static void scatterAddF32(Tensor base, Tensor indices, Tensor src, Tensor out, int dimension, CpuKernelContext context) {
        IndexReadWriteBackend.scatterAddF32(base, indices, src, out, dimension);
    }

    static void scatterAddBF16(Tensor base, Tensor indices, Tensor src, Tensor out, int dimension, CpuKernelContext context) {
        IndexReadWriteBackend.scatterAddBF16(base, indices, src, out, dimension);
    }

    static void scatterElementsF64(Tensor data, Tensor indices, Tensor updates, Tensor out, int axis, ScatterReduction reduction, CpuKernelContext context) {
        IndexReadWriteBackend.scatterElementsF64(data, indices, updates, out, axis, reduction);
    }

    static void scatterElementsF32(Tensor data, Tensor indices, Tensor updates, Tensor out, int axis, ScatterReduction reduction, CpuKernelContext context) {
        IndexReadWriteBackend.scatterElementsF32(data, indices, updates, out, axis, reduction);
    }

    static void scatterElementsBF16(Tensor data, Tensor indices, Tensor updates, Tensor out, int axis, ScatterReduction reduction, CpuKernelContext context) {
        IndexReadWriteBackend.scatterElementsBF16(data, indices, updates, out, axis, reduction);
    }

    static void scatterElementsBOOL(Tensor data, Tensor indices, Tensor updates, Tensor out, int axis, ScatterReduction reduction, CpuKernelContext context) {
        IndexReadWriteBackend.scatterElementsBOOL(data, indices, updates, out, axis, reduction);
    }

    static void scatterElementsI32(Tensor data, Tensor indices, Tensor updates, Tensor out, int axis, ScatterReduction reduction, CpuKernelContext context) {
        IndexReadWriteBackend.scatterElementsI32(data, indices, updates, out, axis, reduction);
    }

    static void scatterNdF64(Tensor data, Tensor indices, Tensor updates, Tensor out, ScatterReduction reduction, CpuKernelContext context) {
        IndexReadWriteBackend.scatterNdF64(data, indices, updates, out, reduction);
    }

    static void scatterNdF32(Tensor data, Tensor indices, Tensor updates, Tensor out, ScatterReduction reduction, CpuKernelContext context) {
        IndexReadWriteBackend.scatterNdF32(data, indices, updates, out, reduction);
    }

    static void scatterNdBF16(Tensor data, Tensor indices, Tensor updates, Tensor out, ScatterReduction reduction, CpuKernelContext context) {
        IndexReadWriteBackend.scatterNdBF16(data, indices, updates, out, reduction);
    }

    static void scatterNdBOOL(Tensor data, Tensor indices, Tensor updates, Tensor out, ScatterReduction reduction, CpuKernelContext context) {
        IndexReadWriteBackend.scatterNdBOOL(data, indices, updates, out, reduction);
    }

    static void scatterNdI32(Tensor data, Tensor indices, Tensor updates, Tensor out, ScatterReduction reduction, CpuKernelContext context) {
        IndexReadWriteBackend.scatterNdI32(data, indices, updates, out, reduction);
    }
}
