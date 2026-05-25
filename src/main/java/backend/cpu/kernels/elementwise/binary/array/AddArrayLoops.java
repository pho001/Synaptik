package backend.cpu.kernels.elementwise.binary.array;

import backend.cpu.kernels.elementwise.ElementwiseNativeSupport;
import backend.cpu.kernels.elementwise.plan.ResolvedDispatchHints;
import backend.cpu.kernels.storage.CpuStorageBindings;
import backend.cpu.kernels.storage.CpuStorageView;
import tensor.DataType;

public final class AddArrayLoops {
    private AddArrayLoops() {
    }

    public static void runF64(double[] left, double[] right, double[] out, ResolvedDispatchHints hints) {
        AddF64.run(left, right, out, hints);
    }

    public static void runF32(float[] left, float[] right, float[] out, ResolvedDispatchHints hints) {
        AddF32.run(left, right, out, hints);
    }

    public static void runBF16(
            short[] leftStorage,
            short[] rightStorage,
            float[] leftContinuation,
            float[] rightContinuation,
            short[] out,
            ResolvedDispatchHints hints
    ) {
        if (leftContinuation != null && rightContinuation != null) {
            AddBF16.run(leftContinuation, rightContinuation, out, hints);
        } else if (leftContinuation != null) {
            AddBF16.run(leftContinuation, rightStorage, out, hints);
        } else if (rightContinuation != null) {
            AddBF16.run(leftStorage, rightContinuation, out, hints);
        } else {
            AddBF16.run(leftStorage, rightStorage, out, hints);
        }
    }

    static void runDense(
            CpuStorageBindings bindings,
            ResolvedDispatchHints hints,
            float[] leftContinuation,
            float[] rightContinuation
    ) {
        validateDenseAdd(bindings);
        DataType dtype = bindings.output().dtype();
        switch (dtype) {
            case FLOAT64 -> runF64(
                    bindings.input(0).requireF64Array(),
                    bindings.input(1).requireF64Array(),
                    bindings.output().requireF64Array(),
                    hints
            );
            case FLOAT32 -> runF32(
                    bindings.input(0).requireF32Array(),
                    bindings.input(1).requireF32Array(),
                    bindings.output().requireF32Array(),
                    hints
            );
            case BFLOAT16 -> runBF16(
                    bindings.input(0).requireBF16Array(),
                    bindings.input(1).requireBF16Array(),
                    leftContinuation,
                    rightContinuation,
                    bindings.output().requireBF16Array(),
                    hints
            );
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException("ADD array loop does not support dtype: " + dtype);
        }
    }

    private static void validateDenseAdd(CpuStorageBindings bindings) {
        if (bindings.inputs().size() != 2) {
            throw new IllegalArgumentException("ADD array loop requires exactly 2 inputs.");
        }
        CpuStorageView left = bindings.input(0);
        CpuStorageView right = bindings.input(1);
        CpuStorageView output = bindings.output();
        if (left.dtype() != output.dtype() || right.dtype() != output.dtype()) {
            throw new IllegalArgumentException("ADD array loop dtype mismatch.");
        }
        if (left.kind() != output.kind() || right.kind() != output.kind()) {
            throw new IllegalArgumentException("ADD array loop requires matching storage kinds.");
        }
        if (left.logicalSize() != output.logicalSize() || right.logicalSize() != output.logicalSize()) {
            throw new IllegalArgumentException("ADD array loop requires same-shape dense inputs.");
        }
        if (!ElementwiseNativeSupport.isDenseView(left)
                || !ElementwiseNativeSupport.isDenseView(right)
                || !ElementwiseNativeSupport.isDenseView(output)) {
            throw new IllegalArgumentException("ADD array loop requires dense zero-offset views.");
        }
    }
}
