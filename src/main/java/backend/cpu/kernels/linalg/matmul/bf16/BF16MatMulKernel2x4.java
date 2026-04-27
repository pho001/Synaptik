package backend.cpu.kernels.linalg.matmul.bf16;

final class BF16MatMulKernel2x4 {
    private BF16MatMulKernel2x4() {
    }

    static void accumulate(
            float[] packedA,
            float[] accum,
            float[] packedB,
            int packedBOffset,
            int tileRows,
            int colOffset,
            int panelDepth,
            int totalCols,
            int tileCols
    ) {
        int width = BF16MatMulAccumulatorSupport.F32.length();
        int vectorLimit = tileCols - (tileCols % width);
        int blockLimit = tileCols - (tileCols % (width * 4));
        int row = 0;
        for (; row + 1 < tileRows; row += 2) {
            BF16MatMulAccumulatorSupport.computeTwoRowsFourColsF32(
                    packedA, accum, packedB,
                    packedBOffset,
                    0, 0,
                    row,
                    colOffset,
                    0, panelDepth,
                    totalCols, panelDepth,
                    tileCols,
                    blockLimit,
                    vectorLimit,
                    width
            );
        }
        for (; row < tileRows; row++) {
            BF16MatMulAccumulatorSupport.computeSingleRowFourColsF32(
                    packedA, accum, packedB,
                    packedBOffset,
                    0, 0,
                    row,
                    colOffset,
                    0, panelDepth,
                    totalCols, panelDepth,
                    tileCols,
                    blockLimit,
                    vectorLimit,
                    width
            );
        }
    }
}
