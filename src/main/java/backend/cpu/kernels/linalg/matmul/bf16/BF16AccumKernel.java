package backend.cpu.kernels.linalg.matmul.bf16;

@FunctionalInterface
interface BF16AccumKernel {
    void compute(
            float[] packedA,
            float[] accum,
            float[] packedB,
            int packedBOffset,
            int tileRows,
            int colOffset,
            int panelDepth,
            int totalCols,
            int tileCols
    );
}
