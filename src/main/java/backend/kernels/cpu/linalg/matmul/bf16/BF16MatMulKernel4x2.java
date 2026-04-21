package backend.kernels.cpu.linalg.matmul.bf16;

final class BF16MatMulKernel4x2 {
    private BF16MatMulKernel4x2() {
    }

    static void accumulate(
            float[] packedA,
            float[] accum,
            float[] packedB,
            int tileRows,
            int colOffset,
            int panelDepth,
            int totalCols,
            int tileCols
    ) {
        int width = BF16MatMulAccumulatorSupport.F32.length();
        int vectorLimit = tileCols - (tileCols % width);
        int blockLimit = tileCols - (tileCols % (width * 2));
        int row = 0;
        for (; row + 3 < tileRows; row += 4) {
            BF16MatMulAccumulatorSupport.computeFourRowsTwoColsF32(
                    packedA, accum, packedB,
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
            BF16MatMulAccumulatorSupport.computeSingleRowTwoColsF32(
                    packedA, accum, packedB,
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
