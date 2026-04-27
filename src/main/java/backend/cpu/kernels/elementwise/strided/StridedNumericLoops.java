package backend.cpu.kernels.elementwise.strided;

import operations.Operation;

final class StridedNumericLoops {
    private StridedNumericLoops() {
    }

    static void forwardF64(
            Operation op,
            StridedNumericInputs.F64 inputs,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        if (inputs.rank() == 1) {
            forwardRank1F64(op, inputs, useFastExpApprox, useFastTanhApprox);
            return;
        }

        if (inputs.rank() == 2 && StridedRank2Loops.tryForwardF64(
                op,
                inputs.a(),
                inputs.b(),
                inputs.aStrides(),
                inputs.bStrides(),
                inputs.aBaseOffset(),
                inputs.bBaseOffset(),
                inputs.out(),
                inputs.outShape()[0],
                inputs.outShape()[1],
                inputs.outStrides(),
                inputs.outBaseOffset(),
                useFastExpApprox,
                useFastTanhApprox
        )) {
            return;
        }

        forwardGenericF64(op, inputs, useFastExpApprox, useFastTanhApprox);
    }

    static void forwardF32(
            Operation op,
            StridedNumericInputs.F32 inputs,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        if (inputs.rank() == 1) {
            forwardRank1F32(op, inputs, useFastExpApprox, useFastTanhApprox);
            return;
        }

        if (inputs.rank() == 2 && StridedRank2Loops.tryForwardF32(
                op,
                inputs.a(),
                inputs.b(),
                inputs.aStrides(),
                inputs.bStrides(),
                inputs.aBaseOffset(),
                inputs.bBaseOffset(),
                inputs.out(),
                inputs.outShape()[0],
                inputs.outShape()[1],
                inputs.outStrides(),
                inputs.outBaseOffset(),
                useFastExpApprox,
                useFastTanhApprox
        )) {
            return;
        }

        forwardGenericF32(op, inputs, useFastExpApprox, useFastTanhApprox);
    }

    static void forwardBF16(
            Operation op,
            StridedNumericInputs.BF16 inputs,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        if (inputs.rank() == 1) {
            forwardRank1BF16(op, inputs, useFastExpApprox, useFastTanhApprox);
            return;
        }

        if (inputs.rank() == 2 && StridedRank2Loops.tryForwardBF16(
                op,
                inputs.a(),
                inputs.b(),
                inputs.aStrides(),
                inputs.bStrides(),
                inputs.aBaseOffset(),
                inputs.bBaseOffset(),
                inputs.out(),
                inputs.outShape()[0],
                inputs.outShape()[1],
                inputs.outStrides(),
                inputs.outBaseOffset(),
                useFastExpApprox,
                useFastTanhApprox
        )) {
            return;
        }

        forwardGenericBF16(op, inputs, useFastExpApprox, useFastTanhApprox);
    }

    private static void forwardRank1F64(
            Operation op,
            StridedNumericInputs.F64 inputs,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        int strideA = inputs.a() != null ? inputs.aStrides()[0] : 0;
        int strideB = inputs.b() != null ? inputs.bStrides()[0] : 0;
        StridedElementWiseSemantics.F64BinaryOp binary = StridedElementWiseSemantics.resolveF64Binary(op);
        if (binary != null) {
            StridedScalarLoops.rank1F64Binary(
                    binary,
                    inputs.a(),
                    inputs.b(),
                    strideA,
                    strideB,
                    inputs.aBaseOffset(),
                    inputs.bBaseOffset(),
                    inputs.out(),
                    inputs.outStrides()[0],
                    inputs.outBaseOffset(),
                    inputs.logicalSize()
            );
            return;
        }
        StridedElementWiseSemantics.F64UnaryOp unary = StridedElementWiseSemantics.resolveF64Unary(op, useFastExpApprox, useFastTanhApprox);
        if (unary != null) {
            StridedScalarLoops.rank1F64Unary(
                    unary,
                    inputs.a(),
                    strideA,
                    inputs.aBaseOffset(),
                    inputs.out(),
                    inputs.outStrides()[0],
                    inputs.outBaseOffset(),
                    inputs.logicalSize()
            );
            return;
        }
        throw new UnsupportedOperationException("Unsupported strided opType=" + op.opType());
    }

    private static void forwardRank1F32(
            Operation op,
            StridedNumericInputs.F32 inputs,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        int strideA = inputs.a() != null ? inputs.aStrides()[0] : 0;
        int strideB = inputs.b() != null ? inputs.bStrides()[0] : 0;
        StridedElementWiseSemantics.F32BinaryOp binary = StridedElementWiseSemantics.resolveF32Binary(op);
        if (binary != null) {
            StridedScalarLoops.rank1F32Binary(
                    binary,
                    inputs.a(),
                    inputs.b(),
                    strideA,
                    strideB,
                    inputs.aBaseOffset(),
                    inputs.bBaseOffset(),
                    inputs.out(),
                    inputs.outStrides()[0],
                    inputs.outBaseOffset(),
                    inputs.logicalSize()
            );
            return;
        }
        StridedElementWiseSemantics.F32UnaryOp unary = StridedElementWiseSemantics.resolveF32Unary(op, useFastExpApprox, useFastTanhApprox);
        if (unary != null) {
            StridedScalarLoops.rank1F32Unary(
                    unary,
                    inputs.a(),
                    strideA,
                    inputs.aBaseOffset(),
                    inputs.out(),
                    inputs.outStrides()[0],
                    inputs.outBaseOffset(),
                    inputs.logicalSize()
            );
            return;
        }
        throw new UnsupportedOperationException("Unsupported strided opType=" + op.opType());
    }

    private static void forwardRank1BF16(
            Operation op,
            StridedNumericInputs.BF16 inputs,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        int strideA = inputs.a() != null ? inputs.aStrides()[0] : 0;
        int strideB = inputs.b() != null ? inputs.bStrides()[0] : 0;
        StridedElementWiseSemantics.BF16BinaryOp binary = StridedElementWiseSemantics.resolveBF16Binary(op);
        if (binary != null) {
            StridedScalarLoops.rank1BF16Binary(
                    binary,
                    inputs.a(),
                    inputs.b(),
                    strideA,
                    strideB,
                    inputs.aBaseOffset(),
                    inputs.bBaseOffset(),
                    inputs.out(),
                    inputs.outStrides()[0],
                    inputs.outBaseOffset(),
                    inputs.logicalSize()
            );
            return;
        }
        StridedElementWiseSemantics.BF16UnaryOp unary = StridedElementWiseSemantics.resolveBF16Unary(op, useFastExpApprox, useFastTanhApprox);
        if (unary != null) {
            StridedScalarLoops.rank1BF16Unary(
                    unary,
                    inputs.a(),
                    strideA,
                    inputs.aBaseOffset(),
                    inputs.out(),
                    inputs.outStrides()[0],
                    inputs.outBaseOffset(),
                    inputs.logicalSize()
            );
            return;
        }
        throw new UnsupportedOperationException("Unsupported strided opType=" + op.opType());
    }

    private static void forwardGenericF64(
            Operation op,
            StridedNumericInputs.F64 inputs,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        StridedElementWiseSemantics.F64BinaryOp binary = StridedElementWiseSemantics.resolveF64Binary(op);
        if (binary != null) {
            StridedScalarLoops.genericF64Binary(
                    binary,
                    inputs.a(),
                    inputs.b(),
                    inputs.aStrides(),
                    inputs.bStrides(),
                    inputs.aBaseOffset(),
                    inputs.bBaseOffset(),
                    inputs.out(),
                    inputs.outShape(),
                    inputs.outStrides(),
                    inputs.outBaseOffset(),
                    inputs.logicalSize()
            );
            return;
        }
        StridedElementWiseSemantics.F64UnaryOp unary = StridedElementWiseSemantics.resolveF64Unary(op, useFastExpApprox, useFastTanhApprox);
        if (unary != null) {
            StridedScalarLoops.genericF64Unary(
                    unary,
                    inputs.a(),
                    inputs.aStrides(),
                    inputs.aBaseOffset(),
                    inputs.out(),
                    inputs.outShape(),
                    inputs.outStrides(),
                    inputs.outBaseOffset(),
                    inputs.logicalSize()
            );
            return;
        }
        throw new UnsupportedOperationException("Unsupported strided opType=" + op.opType());
    }

    private static void forwardGenericF32(
            Operation op,
            StridedNumericInputs.F32 inputs,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        StridedElementWiseSemantics.F32BinaryOp binary = StridedElementWiseSemantics.resolveF32Binary(op);
        if (binary != null) {
            StridedScalarLoops.genericF32Binary(
                    binary,
                    inputs.a(),
                    inputs.b(),
                    inputs.aStrides(),
                    inputs.bStrides(),
                    inputs.aBaseOffset(),
                    inputs.bBaseOffset(),
                    inputs.out(),
                    inputs.outShape(),
                    inputs.outStrides(),
                    inputs.outBaseOffset(),
                    inputs.logicalSize()
            );
            return;
        }
        StridedElementWiseSemantics.F32UnaryOp unary = StridedElementWiseSemantics.resolveF32Unary(op, useFastExpApprox, useFastTanhApprox);
        if (unary != null) {
            StridedScalarLoops.genericF32Unary(
                    unary,
                    inputs.a(),
                    inputs.aStrides(),
                    inputs.aBaseOffset(),
                    inputs.out(),
                    inputs.outShape(),
                    inputs.outStrides(),
                    inputs.outBaseOffset(),
                    inputs.logicalSize()
            );
            return;
        }
        throw new UnsupportedOperationException("Unsupported strided opType=" + op.opType());
    }

    private static void forwardGenericBF16(
            Operation op,
            StridedNumericInputs.BF16 inputs,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        StridedElementWiseSemantics.BF16BinaryOp binary = StridedElementWiseSemantics.resolveBF16Binary(op);
        if (binary != null) {
            StridedScalarLoops.genericBF16Binary(
                    binary,
                    inputs.a(),
                    inputs.b(),
                    inputs.aStrides(),
                    inputs.bStrides(),
                    inputs.aBaseOffset(),
                    inputs.bBaseOffset(),
                    inputs.out(),
                    inputs.outShape(),
                    inputs.outStrides(),
                    inputs.outBaseOffset(),
                    inputs.logicalSize()
            );
            return;
        }
        StridedElementWiseSemantics.BF16UnaryOp unary = StridedElementWiseSemantics.resolveBF16Unary(op, useFastExpApprox, useFastTanhApprox);
        if (unary != null) {
            StridedScalarLoops.genericBF16Unary(
                    unary,
                    inputs.a(),
                    inputs.aStrides(),
                    inputs.aBaseOffset(),
                    inputs.out(),
                    inputs.outShape(),
                    inputs.outStrides(),
                    inputs.outBaseOffset(),
                    inputs.logicalSize()
            );
            return;
        }
        throw new UnsupportedOperationException("Unsupported strided opType=" + op.opType());
    }
}
