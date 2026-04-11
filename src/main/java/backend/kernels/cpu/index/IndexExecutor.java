package backend.kernels.cpu.index;

import backend.kernels.cpu.CpuKernelContext;
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

    static void gatherGradF64(Tensor indices, Tensor outGrad, Tensor node, int dimension, CpuKernelContext context) {
        IndexReadWriteBackend.scatterF64(indices, outGrad, node, dimension);
    }

    static void gatherGradF32(Tensor indices, Tensor outGrad, Tensor node, int dimension, CpuKernelContext context) {
        IndexReadWriteBackend.scatterF32(indices, outGrad, node, dimension);
    }

    static void gatherGradBF16(Tensor indices, Tensor outGrad, Tensor node, int dimension, CpuKernelContext context) {
        IndexReadWriteBackend.scatterBF16(indices, outGrad, node, dimension);
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
}
