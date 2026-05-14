package backend.cpu.kernels.elementwise;

import backend.cpu.kernels.CpuDTypeOps;
import backend.cpu.kernels.CpuExecutionMode;
import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.CpuThreadPool;
import backend.cpu.kernels.layout.plan.ResolvedBroadcastPlan;
import backend.cpu.kernels.elementwise.plan.ResolvedDispatchHints;
import backend.cpu.kernels.layout.plan.ResolvedWhereBroadcastPlan;
import backend.cpu.kernels.elementwise.binary.BinaryElementwiseKernel;
import backend.cpu.kernels.elementwise.compare.CompareElementwiseKernel;
import backend.cpu.kernels.elementwise.logical.LogicalBinaryElementwiseKernel;
import backend.cpu.kernels.elementwise.logical.LogicalUnaryElementwiseKernel;
import backend.cpu.kernels.elementwise.unary.ScalarUnaryElementwiseKernel;
import backend.cpu.kernels.elementwise.unary.UnaryElementwiseKernel;
import backend.cpu.kernels.elementwise.where.WhereElementwiseKernel;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import tensor.DataType;
import tensor.Tensor;

public final class ElementwiseLoops {
    private static final VectorSpecies<Double> F64 = DoubleVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Float> F32 = FloatVector.SPECIES_PREFERRED;

    private ElementwiseLoops() {}

    public static void runBinary(BinaryElementwiseKernel kernel, Tensor left, Tensor right, Tensor out, CpuKernelContext context) {
        ResolvedBroadcastPlan plan = context.broadcastPlan();
        ResolvedDispatchHints hints = hintsOrDefault(context.dispatchHints(), out.getFlatDataSize());
        switch (out.getDataType()) {
            case FLOAT64 -> runBinaryF64(kernel, left.getFloat64Data(), right.getFloat64Data(), out.getFloat64Data(), plan, hints);
            case FLOAT32 -> runBinaryF32(kernel, left.getFloat32Data(), right.getFloat32Data(), out.getFloat32Data(), plan, hints);
            case BFLOAT16 -> {
                float[] leftContinuation = context.inputFloatContinuation(0, left.getFlatDataSize());
                float[] rightContinuation = context.inputFloatContinuation(1, right.getFlatDataSize());
                runBinaryBF16(
                        kernel,
                        left.getBFloat16Data(),
                        right.getBFloat16Data(),
                        leftContinuation,
                        rightContinuation,
                        out.getBFloat16Data(),
                        plan,
                        hints,
                        context
                );
            }
            case INT32, INT64, BOOL -> throw unsupported(out.getDataType(), "binary elementwise kernel");
        }
    }

    public static void runUnary(UnaryElementwiseKernel kernel, Tensor input, Tensor out, CpuKernelContext context) {
        ResolvedDispatchHints hints = hintsOrDefault(context.dispatchHints(), out.getFlatDataSize());
        switch (out.getDataType()) {
            case FLOAT64 -> runUnaryF64(kernel, input.getFloat64Data(), out.getFloat64Data(), hints);
            case FLOAT32 -> runUnaryF32(kernel, input.getFloat32Data(), out.getFloat32Data(), hints);
            case BFLOAT16 -> {
                float[] continuation = context.inputFloatContinuation(0, out.getFlatDataSize());
                runUnaryBF16(
                        kernel,
                        input.getBFloat16Data(),
                        continuation,
                        out.getBFloat16Data(),
                        hints,
                        context
                );
            }
            case INT32, INT64, BOOL -> throw unsupported(out.getDataType(), "unary elementwise kernel");
        }
    }

    public static void runScalarUnary(
            ScalarUnaryElementwiseKernel kernel,
            double parameterF64,
            float parameterF32,
            Tensor input,
            Tensor out,
            CpuKernelContext context
    ) {
        ResolvedDispatchHints hints = hintsOrDefault(context.dispatchHints(), out.getFlatDataSize());
        switch (out.getDataType()) {
            case FLOAT64 -> runScalarUnaryF64(kernel, input.getFloat64Data(), parameterF64, out.getFloat64Data(), hints);
            case FLOAT32 -> runScalarUnaryF32(kernel, input.getFloat32Data(), parameterF32, out.getFloat32Data(), hints);
            case BFLOAT16 -> {
                float[] continuation = context.inputFloatContinuation(0, out.getFlatDataSize());
                runScalarUnaryBF16(
                        kernel,
                        input.getBFloat16Data(),
                        continuation,
                        parameterF32,
                        out.getBFloat16Data(),
                        hints,
                        context
                );
            }
            case INT32, INT64, BOOL -> throw unsupported(out.getDataType(), "scalar unary elementwise kernel");
        }
    }

    public static void runCompare(CompareElementwiseKernel kernel, Tensor left, Tensor right, Tensor out, CpuKernelContext context) {
        ResolvedBroadcastPlan plan = context.broadcastPlan();
        ResolvedDispatchHints hints = hintsOrDefault(context.dispatchHints(), out.getFlatDataSize());
        switch (left.getDataType()) {
            case FLOAT64 -> runCompareF64(kernel, left.getFloat64Data(), right.getFloat64Data(), out.getBoolData(), plan, hints);
            case FLOAT32 -> runCompareF32(kernel, left.getFloat32Data(), right.getFloat32Data(), out.getBoolData(), plan, hints);
            case BFLOAT16 -> runCompareBF16(kernel, left.getBFloat16Data(), right.getBFloat16Data(), out.getBoolData(), plan, hints);
            case INT32, INT64, BOOL -> throw unsupported(left.getDataType(), "compare elementwise kernel");
        }
    }

    public static void runLogicalBinary(LogicalBinaryElementwiseKernel kernel, Tensor left, Tensor right, Tensor out, CpuKernelContext context) {
        ResolvedBroadcastPlan plan = context.broadcastPlan();
        ResolvedDispatchHints hints = hintsOrDefault(context.dispatchHints(), out.getFlatDataSize());
        byte[] leftData = left.getBoolData();
        byte[] rightData = right.getBoolData();
        byte[] outData = out.getBoolData();
        if (plan != null && !plan.isNoBroadcast()) {
            runBroadcast(outData.length, hints, (start, end) -> scalarBroadcastLogicalBinary(kernel, leftData, rightData, outData, plan, start, end));
            return;
        }
        runDirect(outData.length, hints, false,
                (start, end) -> scalarDirectLogicalBinary(kernel, leftData, rightData, outData, start, end),
                null);
    }

    public static void runLogicalUnary(LogicalUnaryElementwiseKernel kernel, Tensor input, Tensor out, CpuKernelContext context) {
        ResolvedDispatchHints hints = hintsOrDefault(context.dispatchHints(), out.getFlatDataSize());
        runDirect(out.getFlatDataSize(), hints, false,
                (start, end) -> scalarDirectLogicalUnary(kernel, input.getBoolData(), out.getBoolData(), start, end),
                null);
    }

    public static void runWhere(WhereElementwiseKernel kernel, Tensor condition, Tensor ifTrue, Tensor ifFalse, Tensor out, CpuKernelContext context) {
        ResolvedWhereBroadcastPlan plan = context.whereBroadcastPlan();
        ResolvedDispatchHints hints = hintsOrDefault(context.dispatchHints(), out.getFlatDataSize());
        switch (out.getDataType()) {
            case FLOAT64 -> runWhereF64(kernel, condition.getBoolData(), ifTrue.getFloat64Data(), ifFalse.getFloat64Data(), out.getFloat64Data(), plan, hints);
            case FLOAT32 -> runWhereF32(kernel, condition.getBoolData(), ifTrue.getFloat32Data(), ifFalse.getFloat32Data(), out.getFloat32Data(), plan, hints);
            case BFLOAT16 -> runWhereBF16(
                    kernel,
                    condition.getBoolData(),
                    ifTrue.getBFloat16Data(),
                    ifFalse.getBFloat16Data(),
                    context.inputFloatContinuation(1, ifTrue.getFlatDataSize()),
                    context.inputFloatContinuation(2, ifFalse.getFlatDataSize()),
                    out.getBFloat16Data(),
                    plan,
                    hints,
                    context
            );
            case INT32, INT64, BOOL -> throw unsupported(out.getDataType(), "where elementwise kernel");
        }
    }

    static BroadcastCursor initBroadcastCursor(ResolvedBroadcastPlan plan, int start) {
        int[] outStrides = plan.outStrides();
        int rank = outStrides.length;
        int[] coords = initCoords(start, outStrides, rank);
        int[] leftEff = plan.aEffStrides();
        int[] rightEff = plan.bEffStrides();
        return new BroadcastCursor(
                plan.outShape(),
                leftEff,
                rightEff,
                plan.aResets(),
                plan.bResets(),
                coords,
                rank,
                initialIndex(coords, leftEff),
                initialIndex(coords, rightEff)
        );
    }

    static void advanceBroadcastCursor(BroadcastCursor cursor) {
        for (int d = cursor.rank - 1; d >= 0; d--) {
            cursor.coords[d]++;
            cursor.leftIdx += cursor.leftEff[d];
            cursor.rightIdx += cursor.rightEff[d];
            if (cursor.coords[d] < cursor.outShape[d]) {
                return;
            }
            cursor.coords[d] = 0;
            cursor.leftIdx -= cursor.leftResets[d];
            cursor.rightIdx -= cursor.rightResets[d];
        }
    }

    static int[] initCoords(int start, int[] outStrides, int rank) {
        int[] coords = new int[rank];
        int temp = start;
        for (int d = 0; d < rank; d++) {
            coords[d] = temp / outStrides[d];
            temp %= outStrides[d];
        }
        return coords;
    }

    static int initialIndex(int[] coords, int[] effectiveStrides) {
        int index = 0;
        for (int d = 0; d < coords.length; d++) {
            index += coords[d] * effectiveStrides[d];
        }
        return index;
    }

    static int[] nextIndices(
            int[] coords,
            int[] outShape,
            int[] leftEff,
            int[] rightEff,
            int[] leftResets,
            int[] rightResets,
            int rank,
            int leftIndex,
            int rightIndex
    ) {
        int nextLeft = leftIndex;
        int nextRight = rightIndex;
        for (int d = rank - 1; d >= 0; d--) {
            coords[d]++;
            nextLeft += leftEff[d];
            nextRight += rightEff[d];
            if (coords[d] < outShape[d]) {
                break;
            }
            coords[d] = 0;
            nextLeft -= leftResets[d];
            nextRight -= rightResets[d];
        }
        return new int[]{nextLeft, nextRight};
    }

    static int[] nextTernaryIndices(
            int[] coords,
            int[] outShape,
            int[] firstEff,
            int[] secondEff,
            int[] thirdEff,
            int[] firstResets,
            int[] secondResets,
            int[] thirdResets,
            int rank,
            int firstIndex,
            int secondIndex,
            int thirdIndex
    ) {
        int nextFirst = firstIndex;
        int nextSecond = secondIndex;
        int nextThird = thirdIndex;
        for (int d = rank - 1; d >= 0; d--) {
            coords[d]++;
            nextFirst += firstEff[d];
            nextSecond += secondEff[d];
            nextThird += thirdEff[d];
            if (coords[d] < outShape[d]) {
                break;
            }
            coords[d] = 0;
            nextFirst -= firstResets[d];
            nextSecond -= secondResets[d];
            nextThird -= thirdResets[d];
        }
        return new int[]{nextFirst, nextSecond, nextThird};
    }

    private static void runBinaryF64(
            BinaryElementwiseKernel kernel,
            double[] left,
            double[] right,
            double[] out,
            ResolvedBroadcastPlan plan,
            ResolvedDispatchHints hints
    ) {
        if (plan != null && !plan.isNoBroadcast()) {
            runBroadcast(out.length, hints, (start, end) -> scalarBroadcastF64(kernel, left, right, out, plan, start, end));
            return;
        }
        if (kernel.supportsDirectF64()) {
            kernel.runDirectF64(left, right, out, hints);
            return;
        }
        boolean vectorized = hints.vectorized() && kernel.supportsVectorF64();
        runDirect(out.length, hints, vectorized,
                (start, end) -> scalarDirectF64(kernel, left, right, out, start, end),
                (start, end) -> vectorDirectF64(kernel, left, right, out, start, end));
    }

    private static void runBinaryF32(
            BinaryElementwiseKernel kernel,
            float[] left,
            float[] right,
            float[] out,
            ResolvedBroadcastPlan plan,
            ResolvedDispatchHints hints
    ) {
        if (plan != null && !plan.isNoBroadcast()) {
            runBroadcast(out.length, hints, (start, end) -> scalarBroadcastF32(kernel, left, right, out, plan, start, end));
            return;
        }
        if (kernel.supportsDirectF32()) {
            kernel.runDirectF32(left, right, out, hints);
            return;
        }
        boolean vectorized = hints.vectorized() && kernel.supportsVectorF32();
        runDirect(out.length, hints, vectorized,
                (start, end) -> scalarDirectF32(kernel, left, right, out, start, end),
                (start, end) -> vectorDirectF32(kernel, left, right, out, start, end));
    }

    private static void runBinaryBF16(
            BinaryElementwiseKernel kernel,
            short[] leftStorage,
            short[] rightStorage,
            float[] leftContinuation,
            float[] rightContinuation,
            short[] out,
            ResolvedBroadcastPlan plan,
            ResolvedDispatchHints hints,
            CpuKernelContext context
    ) {
        if (plan != null && !plan.isNoBroadcast()) {
            if (canPublishFloatContinuation(context)) {
                float[] outFloat = context.cpuWorkspace().requireFloatWorkspace();
                runBroadcast(out.length, hints, (start, end) ->
                        scalarBroadcastBF16ToF32(kernel, leftStorage, rightStorage, leftContinuation, rightContinuation, outFloat, plan, start, end));
                context.cpuWorkspace().publishFloatContinuation(out.length);
                return;
            }
            runBroadcast(out.length, hints, (start, end) ->
                    scalarBroadcastBF16(kernel, leftStorage, rightStorage, leftContinuation, rightContinuation, out, plan, start, end));
            return;
        }
        if (canPublishFloatContinuation(context)) {
            float[] outFloat = context.cpuWorkspace().requireFloatWorkspace();
            runBinaryBF16ToFloat(kernel, leftStorage, rightStorage, leftContinuation, rightContinuation, outFloat, hints);
            context.cpuWorkspace().publishFloatContinuation(out.length);
            return;
        }
        if (kernel.supportsDirectBF16()) {
            kernel.runDirectBF16(leftStorage, rightStorage, leftContinuation, rightContinuation, out, hints);
            return;
        }
        runDirect(out.length, hints, false,
                (start, end) -> scalarDirectBF16(kernel, leftStorage, rightStorage, leftContinuation, rightContinuation, out, start, end),
                null);
    }

    private static void runUnaryF64(UnaryElementwiseKernel kernel, double[] in, double[] out, ResolvedDispatchHints hints) {
        if (kernel.supportsDirectF64()) {
            kernel.runDirectF64(in, out, hints);
            return;
        }
        boolean vectorized = hints.vectorized() && kernel.supportsVectorF64();
        runDirect(out.length, hints, vectorized,
                (start, end) -> scalarUnaryF64(kernel, in, out, start, end),
                (start, end) -> vectorUnaryF64(kernel, in, out, start, end));
    }

    private static void runUnaryF32(UnaryElementwiseKernel kernel, float[] in, float[] out, ResolvedDispatchHints hints) {
        if (kernel.supportsDirectF32()) {
            kernel.runDirectF32(in, out, hints);
            return;
        }
        boolean vectorized = hints.vectorized() && kernel.supportsVectorF32();
        runDirect(out.length, hints, vectorized,
                (start, end) -> scalarUnaryF32(kernel, in, out, start, end),
                (start, end) -> vectorUnaryF32(kernel, in, out, start, end));
    }

    private static void runUnaryBF16(
            UnaryElementwiseKernel kernel,
            short[] in,
            float[] continuation,
            short[] out,
            ResolvedDispatchHints hints,
            CpuKernelContext context
    ) {
        if (canPublishFloatContinuation(context)) {
            float[] outFloat = context.cpuWorkspace().requireFloatWorkspace();
            runUnaryBF16ToFloat(kernel, in, continuation, outFloat, hints);
            context.cpuWorkspace().publishFloatContinuation(out.length);
            return;
        }
        if (kernel.supportsDirectBF16()) {
            kernel.runDirectBF16(in, continuation, out, hints);
            return;
        }
        runDirect(out.length, hints, false,
                (start, end) -> scalarUnaryBF16(kernel, in, continuation, out, start, end),
                null);
    }

    private static void runScalarUnaryF64(
            ScalarUnaryElementwiseKernel kernel,
            double[] in,
            double parameter,
            double[] out,
            ResolvedDispatchHints hints
    ) {
        if (kernel.supportsDirectF64()) {
            kernel.runDirectF64(in, parameter, out, hints);
            return;
        }
        boolean vectorized = hints.vectorized() && kernel.supportsVectorF64();
        runDirect(out.length, hints, vectorized,
                (start, end) -> scalarUnaryF64(kernel, in, parameter, out, start, end),
                (start, end) -> vectorUnaryF64(kernel, in, parameter, out, start, end));
    }

    private static void runScalarUnaryF32(
            ScalarUnaryElementwiseKernel kernel,
            float[] in,
            float parameter,
            float[] out,
            ResolvedDispatchHints hints
    ) {
        if (kernel.supportsDirectF32()) {
            kernel.runDirectF32(in, parameter, out, hints);
            return;
        }
        boolean vectorized = hints.vectorized() && kernel.supportsVectorF32();
        runDirect(out.length, hints, vectorized,
                (start, end) -> scalarUnaryF32(kernel, in, parameter, out, start, end),
                (start, end) -> vectorUnaryF32(kernel, in, parameter, out, start, end));
    }

    private static void runScalarUnaryBF16(
            ScalarUnaryElementwiseKernel kernel,
            short[] in,
            float[] continuation,
            float parameter,
            short[] out,
            ResolvedDispatchHints hints,
            CpuKernelContext context
    ) {
        if (canPublishFloatContinuation(context)) {
            float[] outFloat = context.cpuWorkspace().requireFloatWorkspace();
            runScalarUnaryBF16ToFloat(kernel, in, continuation, parameter, outFloat, hints);
            context.cpuWorkspace().publishFloatContinuation(out.length);
            return;
        }
        if (kernel.supportsDirectBF16()) {
            kernel.runDirectBF16(in, continuation, parameter, out, hints);
            return;
        }
        runDirect(out.length, hints, false,
                (start, end) -> scalarUnaryBF16(kernel, in, continuation, parameter, out, start, end),
                null);
    }

    private static void runCompareF64(
            CompareElementwiseKernel kernel,
            double[] left,
            double[] right,
            byte[] out,
            ResolvedBroadcastPlan plan,
            ResolvedDispatchHints hints
    ) {
        if (plan != null && !plan.isNoBroadcast()) {
            runBroadcast(out.length, hints, (start, end) -> scalarBroadcastCompareF64(kernel, left, right, out, plan, start, end));
            return;
        }
        runDirect(out.length, hints, false,
                (start, end) -> scalarDirectCompareF64(kernel, left, right, out, start, end),
                null);
    }

    private static void runCompareF32(
            CompareElementwiseKernel kernel,
            float[] left,
            float[] right,
            byte[] out,
            ResolvedBroadcastPlan plan,
            ResolvedDispatchHints hints
    ) {
        if (plan != null && !plan.isNoBroadcast()) {
            runBroadcast(out.length, hints, (start, end) -> scalarBroadcastCompareF32(kernel, left, right, out, plan, start, end));
            return;
        }
        runDirect(out.length, hints, false,
                (start, end) -> scalarDirectCompareF32(kernel, left, right, out, start, end),
                null);
    }

    private static void runCompareBF16(
            CompareElementwiseKernel kernel,
            short[] left,
            short[] right,
            byte[] out,
            ResolvedBroadcastPlan plan,
            ResolvedDispatchHints hints
    ) {
        if (plan != null && !plan.isNoBroadcast()) {
            runBroadcast(out.length, hints, (start, end) -> scalarBroadcastCompareBF16(kernel, left, right, out, plan, start, end));
            return;
        }
        runDirect(out.length, hints, false,
                (start, end) -> scalarDirectCompareBF16(kernel, left, right, out, start, end),
                null);
    }

    private static void runWhereF64(
            WhereElementwiseKernel kernel,
            byte[] condition,
            double[] ifTrue,
            double[] ifFalse,
            double[] out,
            ResolvedWhereBroadcastPlan plan,
            ResolvedDispatchHints hints
    ) {
        if (plan != null && !plan.isNoBroadcast()) {
            runBroadcast(out.length, hints, (start, end) -> scalarBroadcastWhereF64(kernel, condition, ifTrue, ifFalse, out, plan, start, end));
            return;
        }
        runDirect(out.length, hints, false,
                (start, end) -> scalarDirectWhereF64(kernel, condition, ifTrue, ifFalse, out, start, end),
                null);
    }

    private static void runWhereF32(
            WhereElementwiseKernel kernel,
            byte[] condition,
            float[] ifTrue,
            float[] ifFalse,
            float[] out,
            ResolvedWhereBroadcastPlan plan,
            ResolvedDispatchHints hints
    ) {
        if (plan != null && !plan.isNoBroadcast()) {
            runBroadcast(out.length, hints, (start, end) -> scalarBroadcastWhereF32(kernel, condition, ifTrue, ifFalse, out, plan, start, end));
            return;
        }
        runDirect(out.length, hints, false,
                (start, end) -> scalarDirectWhereF32(kernel, condition, ifTrue, ifFalse, out, start, end),
                null);
    }

    private static void runWhereBF16(
            WhereElementwiseKernel kernel,
            byte[] condition,
            short[] ifTrue,
            short[] ifFalse,
            float[] ifTrueContinuation,
            float[] ifFalseContinuation,
            short[] out,
            ResolvedWhereBroadcastPlan plan,
            ResolvedDispatchHints hints,
            CpuKernelContext context
    ) {
        if (plan != null && !plan.isNoBroadcast()) {
            if (canPublishFloatContinuation(context)) {
                float[] outFloat = context.cpuWorkspace().requireFloatWorkspace();
                runBroadcast(out.length, hints, (start, end) ->
                        scalarBroadcastWhereBF16ToF32(kernel, condition, ifTrue, ifFalse, ifTrueContinuation, ifFalseContinuation, outFloat, plan, start, end));
                context.cpuWorkspace().publishFloatContinuation(out.length);
                return;
            }
            runBroadcast(out.length, hints, (start, end) ->
                    scalarBroadcastWhereBF16(kernel, condition, ifTrue, ifFalse, ifTrueContinuation, ifFalseContinuation, out, plan, start, end));
            return;
        }
        if (canPublishFloatContinuation(context)) {
            float[] outFloat = context.cpuWorkspace().requireFloatWorkspace();
            runDirect(out.length, hints, false,
                    (start, end) -> scalarDirectWhereBF16ToF32(kernel, condition, ifTrue, ifFalse, ifTrueContinuation, ifFalseContinuation, outFloat, start, end),
                    null);
            context.cpuWorkspace().publishFloatContinuation(out.length);
            return;
        }
        runDirect(out.length, hints, false,
                (start, end) -> scalarDirectWhereBF16(kernel, condition, ifTrue, ifFalse, ifTrueContinuation, ifFalseContinuation, out, start, end),
                null);
    }

    private static void runDirect(
            int length,
            ResolvedDispatchHints hints,
            boolean vectorized,
            RangeConsumer scalar,
            RangeConsumer vector
    ) {
        if (hints.parallel()) {
            int chunkSize = vectorized ? hints.vectorChunkSize() : hints.scalarChunkSize();
            int chunks = (length + chunkSize - 1) / chunkSize;
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, length);
                if (vectorized) {
                    vector.accept(start, end);
                } else {
                    scalar.accept(start, end);
                }
            });
            return;
        }
        if (vectorized) {
            vector.accept(0, length);
        } else {
            scalar.accept(0, length);
        }
    }

    private static void runBroadcast(int length, ResolvedDispatchHints hints, RangeConsumer scalar) {
        if (hints.parallel()) {
            int chunkSize = hints.scalarChunkSize();
            int chunks = (length + chunkSize - 1) / chunkSize;
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, length);
                scalar.accept(start, end);
            });
            return;
        }
        scalar.accept(0, length);
    }

    private static void scalarDirectF64(BinaryElementwiseKernel kernel, double[] left, double[] right, double[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = kernel.applyF64(left[i], right[i]);
        }
    }

    private static void vectorDirectF64(BinaryElementwiseKernel kernel, double[] left, double[] right, double[] out, int start, int end) {
        int width = F64.length();
        int upper = end - ((end - start) % width);
        int i = start;
        for (; i < upper; i += width) {
            kernel.applyVectorF64(DoubleVector.fromArray(F64, left, i), DoubleVector.fromArray(F64, right, i)).intoArray(out, i);
        }
        scalarDirectF64(kernel, left, right, out, i, end);
    }

    private static void scalarDirectF32(BinaryElementwiseKernel kernel, float[] left, float[] right, float[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = kernel.applyF32(left[i], right[i]);
        }
    }

    private static void vectorDirectF32(BinaryElementwiseKernel kernel, float[] left, float[] right, float[] out, int start, int end) {
        int width = F32.length();
        int upper = end - ((end - start) % width);
        int i = start;
        for (; i < upper; i += width) {
            kernel.applyVectorF32(FloatVector.fromArray(F32, left, i), FloatVector.fromArray(F32, right, i)).intoArray(out, i);
        }
        scalarDirectF32(kernel, left, right, out, i, end);
    }

    private static void scalarDirectBF16(
            BinaryElementwiseKernel kernel,
            short[] leftStorage,
            short[] rightStorage,
            float[] leftContinuation,
            float[] rightContinuation,
            short[] out,
            int start,
            int end
    ) {
        for (int i = start; i < end; i++) {
            out[i] = storeBF16(kernel.applyBF16(loadBF16(leftContinuation, leftStorage, i), loadBF16(rightContinuation, rightStorage, i)));
        }
    }

    private static void scalarDirectBF16ToF32(
            BinaryElementwiseKernel kernel,
            short[] leftStorage,
            short[] rightStorage,
            float[] leftContinuation,
            float[] rightContinuation,
            float[] out,
            int start,
            int end
    ) {
        for (int i = start; i < end; i++) {
            out[i] = kernel.applyBF16(loadBF16(leftContinuation, leftStorage, i), loadBF16(rightContinuation, rightStorage, i));
        }
    }

    private static void scalarBroadcastF64(
            BinaryElementwiseKernel kernel,
            double[] left,
            double[] right,
            double[] out,
            ResolvedBroadcastPlan plan,
            int start,
            int end
    ) {
        BroadcastCursor cursor = initBroadcastCursor(plan, start);
        for (int i = start; i < end; i++) {
            out[i] = kernel.applyF64(left[cursor.leftIdx], right[cursor.rightIdx]);
            if (i + 1 < end) {
                advanceBroadcastCursor(cursor);
            }
        }
    }

    private static void scalarBroadcastF32(
            BinaryElementwiseKernel kernel,
            float[] left,
            float[] right,
            float[] out,
            ResolvedBroadcastPlan plan,
            int start,
            int end
    ) {
        BroadcastCursor cursor = initBroadcastCursor(plan, start);
        for (int i = start; i < end; i++) {
            out[i] = kernel.applyF32(left[cursor.leftIdx], right[cursor.rightIdx]);
            if (i + 1 < end) {
                advanceBroadcastCursor(cursor);
            }
        }
    }

    private static void scalarBroadcastBF16(
            BinaryElementwiseKernel kernel,
            short[] leftStorage,
            short[] rightStorage,
            float[] leftContinuation,
            float[] rightContinuation,
            short[] out,
            ResolvedBroadcastPlan plan,
            int start,
            int end
    ) {
        BroadcastCursor cursor = initBroadcastCursor(plan, start);
        for (int i = start; i < end; i++) {
            out[i] = storeBF16(kernel.applyBF16(
                    loadBF16(leftContinuation, leftStorage, cursor.leftIdx),
                    loadBF16(rightContinuation, rightStorage, cursor.rightIdx)
            ));
            if (i + 1 < end) {
                advanceBroadcastCursor(cursor);
            }
        }
    }

    private static void scalarBroadcastBF16ToF32(
            BinaryElementwiseKernel kernel,
            short[] leftStorage,
            short[] rightStorage,
            float[] leftContinuation,
            float[] rightContinuation,
            float[] out,
            ResolvedBroadcastPlan plan,
            int start,
            int end
    ) {
        BroadcastCursor cursor = initBroadcastCursor(plan, start);
        for (int i = start; i < end; i++) {
            out[i] = kernel.applyBF16(
                    loadBF16(leftContinuation, leftStorage, cursor.leftIdx),
                    loadBF16(rightContinuation, rightStorage, cursor.rightIdx)
            );
            if (i + 1 < end) {
                advanceBroadcastCursor(cursor);
            }
        }
    }

    private static void scalarUnaryF64(UnaryElementwiseKernel kernel, double[] in, double[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = kernel.applyF64(in[i]);
        }
    }

    private static void vectorUnaryF64(UnaryElementwiseKernel kernel, double[] in, double[] out, int start, int end) {
        int width = F64.length();
        int upper = end - ((end - start) % width);
        int i = start;
        for (; i < upper; i += width) {
            kernel.applyVectorF64(DoubleVector.fromArray(F64, in, i)).intoArray(out, i);
        }
        scalarUnaryF64(kernel, in, out, i, end);
    }

    private static void scalarUnaryF32(UnaryElementwiseKernel kernel, float[] in, float[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = kernel.applyF32(in[i]);
        }
    }

    private static void vectorUnaryF32(UnaryElementwiseKernel kernel, float[] in, float[] out, int start, int end) {
        int width = F32.length();
        int upper = end - ((end - start) % width);
        int i = start;
        for (; i < upper; i += width) {
            kernel.applyVectorF32(FloatVector.fromArray(F32, in, i)).intoArray(out, i);
        }
        scalarUnaryF32(kernel, in, out, i, end);
    }

    private static void scalarUnaryBF16(
            UnaryElementwiseKernel kernel,
            short[] in,
            float[] continuation,
            short[] out,
            int start,
            int end
    ) {
        for (int i = start; i < end; i++) {
            out[i] = storeBF16(kernel.applyBF16(loadBF16(continuation, in, i)));
        }
    }

    private static void scalarUnaryBF16ToF32(
            UnaryElementwiseKernel kernel,
            short[] in,
            float[] continuation,
            float[] out,
            int start,
            int end
    ) {
        for (int i = start; i < end; i++) {
            out[i] = kernel.applyBF16(loadBF16(continuation, in, i));
        }
    }

    private static void scalarUnaryF64(
            ScalarUnaryElementwiseKernel kernel,
            double[] in,
            double parameter,
            double[] out,
            int start,
            int end
    ) {
        for (int i = start; i < end; i++) {
            out[i] = kernel.applyF64(in[i], parameter);
        }
    }

    private static void vectorUnaryF64(
            ScalarUnaryElementwiseKernel kernel,
            double[] in,
            double parameter,
            double[] out,
            int start,
            int end
    ) {
        int width = F64.length();
        int upper = end - ((end - start) % width);
        int i = start;
        DoubleVector param = DoubleVector.broadcast(F64, parameter);
        for (; i < upper; i += width) {
            kernel.applyVectorF64(DoubleVector.fromArray(F64, in, i), param).intoArray(out, i);
        }
        scalarUnaryF64(kernel, in, parameter, out, i, end);
    }

    private static void scalarUnaryF32(
            ScalarUnaryElementwiseKernel kernel,
            float[] in,
            float parameter,
            float[] out,
            int start,
            int end
    ) {
        for (int i = start; i < end; i++) {
            out[i] = kernel.applyF32(in[i], parameter);
        }
    }

    private static void vectorUnaryF32(
            ScalarUnaryElementwiseKernel kernel,
            float[] in,
            float parameter,
            float[] out,
            int start,
            int end
    ) {
        int width = F32.length();
        int upper = end - ((end - start) % width);
        int i = start;
        FloatVector param = FloatVector.broadcast(F32, parameter);
        for (; i < upper; i += width) {
            kernel.applyVectorF32(FloatVector.fromArray(F32, in, i), param).intoArray(out, i);
        }
        scalarUnaryF32(kernel, in, parameter, out, i, end);
    }

    private static void scalarUnaryBF16(
            ScalarUnaryElementwiseKernel kernel,
            short[] in,
            float[] continuation,
            float parameter,
            short[] out,
            int start,
            int end
    ) {
        for (int i = start; i < end; i++) {
            out[i] = storeBF16(kernel.applyBF16(loadBF16(continuation, in, i), parameter));
        }
    }

    private static void scalarUnaryBF16ToF32(
            ScalarUnaryElementwiseKernel kernel,
            short[] in,
            float[] continuation,
            float parameter,
            float[] out,
            int start,
            int end
    ) {
        for (int i = start; i < end; i++) {
            out[i] = kernel.applyBF16(loadBF16(continuation, in, i), parameter);
        }
    }

    private static void runBinaryBF16ToFloat(
            BinaryElementwiseKernel kernel,
            short[] leftStorage,
            short[] rightStorage,
            float[] leftContinuation,
            float[] rightContinuation,
            float[] out,
            ResolvedDispatchHints hints
    ) {
        if (leftContinuation != null && rightContinuation != null && kernel.supportsDirectF32()) {
            kernel.runDirectF32(leftContinuation, rightContinuation, out, hints);
            return;
        }
        boolean vectorized = leftContinuation != null
                && rightContinuation != null
                && hints.vectorized()
                && kernel.supportsVectorF32();
        runDirect(out.length, hints, vectorized,
                (start, end) -> scalarDirectBF16ToF32(kernel, leftStorage, rightStorage, leftContinuation, rightContinuation, out, start, end),
                (start, end) -> vectorDirectF32(kernel, leftContinuation, rightContinuation, out, start, end));
    }

    private static void runUnaryBF16ToFloat(
            UnaryElementwiseKernel kernel,
            short[] in,
            float[] continuation,
            float[] out,
            ResolvedDispatchHints hints
    ) {
        if (continuation != null && kernel.supportsDirectF32()) {
            kernel.runDirectF32(continuation, out, hints);
            return;
        }
        boolean vectorized = continuation != null && hints.vectorized() && kernel.supportsVectorF32();
        runDirect(out.length, hints, vectorized,
                (start, end) -> scalarUnaryBF16ToF32(kernel, in, continuation, out, start, end),
                (start, end) -> vectorUnaryF32(kernel, continuation, out, start, end));
    }

    private static void runScalarUnaryBF16ToFloat(
            ScalarUnaryElementwiseKernel kernel,
            short[] in,
            float[] continuation,
            float parameter,
            float[] out,
            ResolvedDispatchHints hints
    ) {
        if (continuation != null && kernel.supportsDirectF32()) {
            kernel.runDirectF32(continuation, parameter, out, hints);
            return;
        }
        boolean vectorized = continuation != null && hints.vectorized() && kernel.supportsVectorF32();
        runDirect(out.length, hints, vectorized,
                (start, end) -> scalarUnaryBF16ToF32(kernel, in, continuation, parameter, out, start, end),
                (start, end) -> vectorUnaryF32(kernel, continuation, parameter, out, start, end));
    }

    private static void scalarDirectCompareF64(
            CompareElementwiseKernel kernel,
            double[] left,
            double[] right,
            byte[] out,
            int start,
            int end
    ) {
        for (int i = start; i < end; i++) {
            out[i] = kernel.testF64(left[i], right[i]) ? (byte) 1 : (byte) 0;
        }
    }

    private static void scalarDirectCompareF32(
            CompareElementwiseKernel kernel,
            float[] left,
            float[] right,
            byte[] out,
            int start,
            int end
    ) {
        for (int i = start; i < end; i++) {
            out[i] = kernel.testF32(left[i], right[i]) ? (byte) 1 : (byte) 0;
        }
    }

    private static void scalarDirectCompareBF16(
            CompareElementwiseKernel kernel,
            short[] left,
            short[] right,
            byte[] out,
            int start,
            int end
    ) {
        for (int i = start; i < end; i++) {
            out[i] = kernel.testBF16(loadBF16(null, left, i), loadBF16(null, right, i)) ? (byte) 1 : (byte) 0;
        }
    }

    private static void scalarBroadcastCompareF64(
            CompareElementwiseKernel kernel,
            double[] left,
            double[] right,
            byte[] out,
            ResolvedBroadcastPlan plan,
            int start,
            int end
    ) {
        BroadcastCursor cursor = initBroadcastCursor(plan, start);
        for (int i = start; i < end; i++) {
            out[i] = kernel.testF64(left[cursor.leftIdx], right[cursor.rightIdx]) ? (byte) 1 : (byte) 0;
            if (i + 1 < end) {
                advanceBroadcastCursor(cursor);
            }
        }
    }

    private static void scalarBroadcastCompareF32(
            CompareElementwiseKernel kernel,
            float[] left,
            float[] right,
            byte[] out,
            ResolvedBroadcastPlan plan,
            int start,
            int end
    ) {
        BroadcastCursor cursor = initBroadcastCursor(plan, start);
        for (int i = start; i < end; i++) {
            out[i] = kernel.testF32(left[cursor.leftIdx], right[cursor.rightIdx]) ? (byte) 1 : (byte) 0;
            if (i + 1 < end) {
                advanceBroadcastCursor(cursor);
            }
        }
    }

    private static void scalarBroadcastCompareBF16(
            CompareElementwiseKernel kernel,
            short[] left,
            short[] right,
            byte[] out,
            ResolvedBroadcastPlan plan,
            int start,
            int end
    ) {
        BroadcastCursor cursor = initBroadcastCursor(plan, start);
        for (int i = start; i < end; i++) {
            out[i] = kernel.testBF16(loadBF16(null, left, cursor.leftIdx), loadBF16(null, right, cursor.rightIdx)) ? (byte) 1 : (byte) 0;
            if (i + 1 < end) {
                advanceBroadcastCursor(cursor);
            }
        }
    }

    private static void scalarDirectLogicalBinary(
            LogicalBinaryElementwiseKernel kernel,
            byte[] left,
            byte[] right,
            byte[] out,
            int start,
            int end
    ) {
        for (int i = start; i < end; i++) {
            out[i] = kernel.apply(left[i], right[i]);
        }
    }

    private static void scalarBroadcastLogicalBinary(
            LogicalBinaryElementwiseKernel kernel,
            byte[] left,
            byte[] right,
            byte[] out,
            ResolvedBroadcastPlan plan,
            int start,
            int end
    ) {
        BroadcastCursor cursor = initBroadcastCursor(plan, start);
        for (int i = start; i < end; i++) {
            out[i] = kernel.apply(left[cursor.leftIdx], right[cursor.rightIdx]);
            if (i + 1 < end) {
                advanceBroadcastCursor(cursor);
            }
        }
    }

    private static void scalarDirectLogicalUnary(
            LogicalUnaryElementwiseKernel kernel,
            byte[] in,
            byte[] out,
            int start,
            int end
    ) {
        for (int i = start; i < end; i++) {
            out[i] = kernel.apply(in[i]);
        }
    }

    private static void scalarDirectWhereF64(
            WhereElementwiseKernel kernel,
            byte[] condition,
            double[] ifTrue,
            double[] ifFalse,
            double[] out,
            int start,
            int end
    ) {
        for (int i = start; i < end; i++) {
            out[i] = kernel.applyF64(condition[i], ifTrue[i], ifFalse[i]);
        }
    }

    private static void scalarDirectWhereF32(
            WhereElementwiseKernel kernel,
            byte[] condition,
            float[] ifTrue,
            float[] ifFalse,
            float[] out,
            int start,
            int end
    ) {
        for (int i = start; i < end; i++) {
            out[i] = kernel.applyF32(condition[i], ifTrue[i], ifFalse[i]);
        }
    }

    private static void scalarDirectWhereBF16(
            WhereElementwiseKernel kernel,
            byte[] condition,
            short[] ifTrue,
            short[] ifFalse,
            float[] ifTrueContinuation,
            float[] ifFalseContinuation,
            short[] out,
            int start,
            int end
    ) {
        for (int i = start; i < end; i++) {
            out[i] = storeBF16(kernel.applyBF16(
                    condition[i],
                    loadBF16(ifTrueContinuation, ifTrue, i),
                    loadBF16(ifFalseContinuation, ifFalse, i)
            ));
        }
    }

    private static void scalarDirectWhereBF16ToF32(
            WhereElementwiseKernel kernel,
            byte[] condition,
            short[] ifTrue,
            short[] ifFalse,
            float[] ifTrueContinuation,
            float[] ifFalseContinuation,
            float[] out,
            int start,
            int end
    ) {
        for (int i = start; i < end; i++) {
            out[i] = kernel.applyBF16(
                    condition[i],
                    loadBF16(ifTrueContinuation, ifTrue, i),
                    loadBF16(ifFalseContinuation, ifFalse, i)
            );
        }
    }

    private static void scalarBroadcastWhereF64(
            WhereElementwiseKernel kernel,
            byte[] condition,
            double[] ifTrue,
            double[] ifFalse,
            double[] out,
            ResolvedWhereBroadcastPlan plan,
            int start,
            int end
    ) {
        int[] outStrides = plan.outStrides();
        int[] outShape = plan.outShape();
        int[] condEff = plan.condEffStrides();
        int[] trueEff = plan.trueEffStrides();
        int[] falseEff = plan.falseEffStrides();
        int[] condResets = plan.condResets();
        int[] trueResets = plan.trueResets();
        int[] falseResets = plan.falseResets();
        int rank = outStrides.length;
        int[] coords = initCoords(start, outStrides, rank);
        int condIdx = initialIndex(coords, condEff);
        int trueIdx = initialIndex(coords, trueEff);
        int falseIdx = initialIndex(coords, falseEff);
        for (int i = start; i < end; i++) {
            out[i] = kernel.applyF64(condition[condIdx], ifTrue[trueIdx], ifFalse[falseIdx]);
            if (i + 1 < end) {
                int[] next = nextTernaryIndices(coords, outShape, condEff, trueEff, falseEff, condResets, trueResets, falseResets, rank, condIdx, trueIdx, falseIdx);
                condIdx = next[0];
                trueIdx = next[1];
                falseIdx = next[2];
            }
        }
    }

    private static void scalarBroadcastWhereF32(
            WhereElementwiseKernel kernel,
            byte[] condition,
            float[] ifTrue,
            float[] ifFalse,
            float[] out,
            ResolvedWhereBroadcastPlan plan,
            int start,
            int end
    ) {
        int[] outStrides = plan.outStrides();
        int[] outShape = plan.outShape();
        int[] condEff = plan.condEffStrides();
        int[] trueEff = plan.trueEffStrides();
        int[] falseEff = plan.falseEffStrides();
        int[] condResets = plan.condResets();
        int[] trueResets = plan.trueResets();
        int[] falseResets = plan.falseResets();
        int rank = outStrides.length;
        int[] coords = initCoords(start, outStrides, rank);
        int condIdx = initialIndex(coords, condEff);
        int trueIdx = initialIndex(coords, trueEff);
        int falseIdx = initialIndex(coords, falseEff);
        for (int i = start; i < end; i++) {
            out[i] = kernel.applyF32(condition[condIdx], ifTrue[trueIdx], ifFalse[falseIdx]);
            if (i + 1 < end) {
                int[] next = nextTernaryIndices(coords, outShape, condEff, trueEff, falseEff, condResets, trueResets, falseResets, rank, condIdx, trueIdx, falseIdx);
                condIdx = next[0];
                trueIdx = next[1];
                falseIdx = next[2];
            }
        }
    }

    private static void scalarBroadcastWhereBF16(
            WhereElementwiseKernel kernel,
            byte[] condition,
            short[] ifTrue,
            short[] ifFalse,
            float[] ifTrueContinuation,
            float[] ifFalseContinuation,
            short[] out,
            ResolvedWhereBroadcastPlan plan,
            int start,
            int end
    ) {
        int[] outStrides = plan.outStrides();
        int[] outShape = plan.outShape();
        int[] condEff = plan.condEffStrides();
        int[] trueEff = plan.trueEffStrides();
        int[] falseEff = plan.falseEffStrides();
        int[] condResets = plan.condResets();
        int[] trueResets = plan.trueResets();
        int[] falseResets = plan.falseResets();
        int rank = outStrides.length;
        int[] coords = initCoords(start, outStrides, rank);
        int condIdx = initialIndex(coords, condEff);
        int trueIdx = initialIndex(coords, trueEff);
        int falseIdx = initialIndex(coords, falseEff);
        for (int i = start; i < end; i++) {
            out[i] = storeBF16(kernel.applyBF16(
                    condition[condIdx],
                    loadBF16(ifTrueContinuation, ifTrue, trueIdx),
                    loadBF16(ifFalseContinuation, ifFalse, falseIdx)
            ));
            if (i + 1 < end) {
                int[] next = nextTernaryIndices(coords, outShape, condEff, trueEff, falseEff, condResets, trueResets, falseResets, rank, condIdx, trueIdx, falseIdx);
                condIdx = next[0];
                trueIdx = next[1];
                falseIdx = next[2];
            }
        }
    }

    private static void scalarBroadcastWhereBF16ToF32(
            WhereElementwiseKernel kernel,
            byte[] condition,
            short[] ifTrue,
            short[] ifFalse,
            float[] ifTrueContinuation,
            float[] ifFalseContinuation,
            float[] out,
            ResolvedWhereBroadcastPlan plan,
            int start,
            int end
    ) {
        int[] outStrides = plan.outStrides();
        int[] outShape = plan.outShape();
        int[] condEff = plan.condEffStrides();
        int[] trueEff = plan.trueEffStrides();
        int[] falseEff = plan.falseEffStrides();
        int[] condResets = plan.condResets();
        int[] trueResets = plan.trueResets();
        int[] falseResets = plan.falseResets();
        int rank = outStrides.length;
        int[] coords = initCoords(start, outStrides, rank);
        int condIdx = initialIndex(coords, condEff);
        int trueIdx = initialIndex(coords, trueEff);
        int falseIdx = initialIndex(coords, falseEff);
        for (int i = start; i < end; i++) {
            out[i] = kernel.applyBF16(
                    condition[condIdx],
                    loadBF16(ifTrueContinuation, ifTrue, trueIdx),
                    loadBF16(ifFalseContinuation, ifFalse, falseIdx)
            );
            if (i + 1 < end) {
                int[] next = nextTernaryIndices(coords, outShape, condEff, trueEff, falseEff, condResets, trueResets, falseResets, rank, condIdx, trueIdx, falseIdx);
                condIdx = next[0];
                trueIdx = next[1];
                falseIdx = next[2];
            }
        }
    }

    private static ResolvedDispatchHints hintsOrDefault(ResolvedDispatchHints hints, int length) {
        return hints != null
                ? hints
                : new ResolvedDispatchHints(Math.max(0, length), CpuExecutionMode.SCALAR, Math.max(1, length), Math.max(1, length), 1, 1, false);
    }

    private static float loadBF16(float[] continuation, short[] storage, int index) {
        return continuation != null ? continuation[index] : CpuDTypeOps.fromBFloat16Bits(storage[index]);
    }

    private static boolean canPublishFloatContinuation(CpuKernelContext context) {
        return context != null
                && context.publishFloatContinuation()
                && context.cpuWorkspace() != null;
    }

    private static short storeBF16(float value) {
        return CpuDTypeOps.toBFloat16Bits(value);
    }

    private static UnsupportedOperationException unsupported(DataType dataType, String kind) {
        return new UnsupportedOperationException("Unsupported dtype for " + kind + ": " + dataType);
    }

    @FunctionalInterface
    private interface RangeConsumer {
        void accept(int start, int end);
    }

    static final class BroadcastCursor {
        final int[] outShape;
        final int[] leftEff;
        final int[] rightEff;
        final int[] leftResets;
        final int[] rightResets;
        final int[] coords;
        final int rank;
        int leftIdx;
        int rightIdx;

        private BroadcastCursor(
                int[] outShape,
                int[] leftEff,
                int[] rightEff,
                int[] leftResets,
                int[] rightResets,
                int[] coords,
                int rank,
                int leftIdx,
                int rightIdx
        ) {
            this.outShape = outShape;
            this.leftEff = leftEff;
            this.rightEff = rightEff;
            this.leftResets = leftResets;
            this.rightResets = rightResets;
            this.coords = coords;
            this.rank = rank;
            this.leftIdx = leftIdx;
            this.rightIdx = rightIdx;
        }
    }
}
