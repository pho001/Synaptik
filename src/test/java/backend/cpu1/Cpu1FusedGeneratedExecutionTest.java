package backend.cpu1;

import backend.cpu1.exec.Cpu1FusedKernelArgs;
import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.fused.ir.Cpu1FusedAccessKind;
import backend.cpu1.fused.ir.Cpu1FusedExpressionPlan;
import backend.cpu1.fused.ir.Cpu1FusedInputPlan;
import backend.cpu1.fused.ir.Cpu1FusedNodePlan;
import backend.cpu1.fused.ir.Cpu1FusedScalarParameter;
import backend.cpu1.kernels.Cpu1LayoutKind;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.kernels.fused.codegen.Cpu1FusedCodegenKernel;
import backend.cpu1.kernels.fused.codegen.Cpu1FusedCodegenKernelFactory;
import backend.cpu1.kernels.fused.codegen.Cpu1FusedCodegenLoopKind;
import backend.cpu1.kernels.fused.codegen.Cpu1FusedCodegenPlan;
import backend.cpu1.kernels.fused.codegen.Cpu1FusedCodegenRejectionReason;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.launch.Cpu1SingleThreadLaunch;
import backend.cpu1.prepare.Cpu1PrepareConfig;
import backend.cpu1.prepare.Cpu1PreparedFusedElementwiseUnit;
import backend.cpu1.prepare.dispatch.Cpu1CostClass;
import backend.cpu1.prepare.dispatch.Cpu1FusedDispatchDecision;
import backend.cpu1.storage.Cpu1StorageKind;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class Cpu1FusedGeneratedExecutionTest {
    @Test
    void generatedF32ContiguousComputesMulAddRelu() {
        Tensor left = f32(new float[]{1.0f, -2.0f, 3.0f, -4.0f}, 4);
        Tensor right = f32(new float[]{10.0f, 20.0f, -30.0f, -40.0f}, 4);
        Tensor bias = f32(new float[]{-5.0f, 50.0f, 100.0f, -200.0f}, 4);
        Tensor output = f32(new float[4], 4);
        Cpu1FusedExpressionPlan plan = new Cpu1FusedExpressionPlan(
                List.of(
                        node(0, Operation.OpType.MUL, List.of(0, 1), 3, DataType.FLOAT32),
                        node(1, Operation.OpType.ADD, List.of(3, 2), 4, DataType.FLOAT32),
                        node(2, Operation.OpType.RELU, List.of(4), 5, DataType.FLOAT32)
                ),
                List.of(
                        contiguousInput(0, DataType.FLOAT32, new int[]{4}),
                        contiguousInput(1, DataType.FLOAT32, new int[]{4}),
                        contiguousInput(2, DataType.FLOAT32, new int[]{4})
                ),
                5
        );

        run(plan, DataType.FLOAT32, Cpu1LayoutKind.CONTIGUOUS, Cpu1FusedCodegenLoopKind.CONTIGUOUS_SCALAR,
                List.of(left, right, bias), output);

        assertArrayEquals(new float[]{5.0f, 10.0f, 10.0f, 0.0f}, output.toFloat32ArrayCopy(), 1.0e-6f);
    }

    @Test
    void generatedF32ContiguousComputesPowMinusTwoCanonicalSequence() {
        Tensor input = f32(new float[]{2.0f, -4.0f, 0.5f}, 3);
        Tensor output = f32(new float[3], 3);
        Cpu1FusedExpressionPlan plan = new Cpu1FusedExpressionPlan(
                List.of(
                        node(0, Operation.OpType.MUL, List.of(0, 0), 1, DataType.FLOAT32),
                        node(1, Operation.OpType.INV, List.of(1), 2, DataType.FLOAT32)
                ),
                List.of(contiguousInput(0, DataType.FLOAT32, new int[]{3})),
                2
        );

        run(plan, DataType.FLOAT32, Cpu1LayoutKind.CONTIGUOUS, Cpu1FusedCodegenLoopKind.CONTIGUOUS_SCALAR,
                List.of(input), output);

        assertArrayEquals(new float[]{0.25f, 0.0625f, 4.0f}, output.toFloat32ArrayCopy(), 1.0e-6f);
    }

    @Test
    void generatedF64ContiguousComputesAdd() {
        Tensor left = f64(new double[]{1.5, -2.0, 3.25}, 3);
        Tensor right = f64(new double[]{10.0, 20.5, -30.25}, 3);
        Tensor output = f64(new double[3], 3);
        Cpu1FusedExpressionPlan plan = new Cpu1FusedExpressionPlan(
                List.of(node(0, Operation.OpType.ADD, List.of(0, 1), 2, DataType.FLOAT64)),
                List.of(
                        contiguousInput(0, DataType.FLOAT64, new int[]{3}),
                        contiguousInput(1, DataType.FLOAT64, new int[]{3})
                ),
                2
        );

        run(plan, DataType.FLOAT64, Cpu1LayoutKind.CONTIGUOUS, Cpu1FusedCodegenLoopKind.CONTIGUOUS_SCALAR,
                List.of(left, right), output);

        assertArrayEquals(new double[]{11.5, 18.5, -27.0}, output.toFloat64ArrayCopy(), 1.0e-12);
    }

    @Test
    void generatedStridedScalarComputesBroadcastAddWithoutOffsetHelperCalls() {
        Tensor base = f32(new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, 2, 3);
        Tensor strided = base.permute(1, 0);
        Tensor scalar = f32(new float[]{10.0f}, 1);
        Tensor output = f32(new float[6], 3, 2);
        int[] outShape = new int[]{3, 2};
        Cpu1FusedExpressionPlan plan = new Cpu1FusedExpressionPlan(
                List.of(node(0, Operation.OpType.ADD, List.of(0, 1), 2, DataType.FLOAT32)),
                List.of(
                        input(0, DataType.FLOAT32, new int[]{3, 2}, new int[]{1, 3}, outShape,
                                new int[]{2, 1}, new int[]{1, 3}, Cpu1FusedAccessKind.DIRECT_STRIDED),
                        input(1, DataType.FLOAT32, new int[]{1}, new int[]{1}, outShape,
                                new int[]{2, 1}, new int[]{0, 0}, Cpu1FusedAccessKind.BROADCAST_STRIDED)
                ),
                2
        );

        run(plan, DataType.FLOAT32, Cpu1LayoutKind.STRIDED_RANK2, Cpu1FusedCodegenLoopKind.STRIDED_SCALAR,
                List.of(strided, scalar), output);

        assertArrayEquals(new float[]{11.0f, 14.0f, 12.0f, 15.0f, 13.0f, 16.0f},
                output.toFloat32ArrayCopy(), 1.0e-6f);
    }

    @Test
    void generatedScalarBindingsAreBoundPerInstanceAndTemplateIsReused() {
        Tensor input = f32(new float[]{1.0f, 2.0f, 3.0f}, 3);
        Tensor outputA = f32(new float[3], 3);
        Tensor outputB = f32(new float[3], 3);
        Cpu1FusedExpressionPlan planA = scalarPlan(2.0f);
        Cpu1FusedExpressionPlan planB = scalarPlan(5.0f);
        Cpu1FusedCodegenKernel kernelA = kernel(planA, DataType.FLOAT32,
                Cpu1LayoutKind.CONTIGUOUS, Cpu1FusedCodegenLoopKind.CONTIGUOUS_SCALAR);
        Cpu1FusedCodegenKernel kernelB = kernel(planB, DataType.FLOAT32,
                Cpu1LayoutKind.CONTIGUOUS, Cpu1FusedCodegenLoopKind.CONTIGUOUS_SCALAR);

        run(kernelA, planA, DataType.FLOAT32, Cpu1LayoutKind.CONTIGUOUS, List.of(input), outputA);
        run(kernelB, planB, DataType.FLOAT32, Cpu1LayoutKind.CONTIGUOUS, List.of(input), outputB);

        assertEquals(kernelA.generatedClassName(), kernelB.generatedClassName());
        assertArrayEquals(new float[]{2.0f, 4.0f, 6.0f}, outputA.toFloat32ArrayCopy(), 1.0e-6f);
        assertArrayEquals(new float[]{5.0f, 10.0f, 15.0f}, outputB.toFloat32ArrayCopy(), 1.0e-6f);
    }

    @Test
    void generatedWhereUsesBoolMaskForF32() {
        Tensor mask = bool(new byte[]{1, 0, 1, 0}, 4);
        Tensor trueValues = f32(new float[]{1.0f, 2.0f, 3.0f, 4.0f}, 4);
        Tensor falseValues = f32(new float[]{10.0f, 20.0f, 30.0f, 40.0f}, 4);
        Tensor output = f32(new float[4], 4);
        Cpu1FusedExpressionPlan plan = new Cpu1FusedExpressionPlan(
                List.of(node(0, Operation.OpType.WHERE, List.of(0, 1, 2), 3, DataType.FLOAT32)),
                List.of(
                        contiguousInput(0, DataType.BOOL, new int[]{4}),
                        contiguousInput(1, DataType.FLOAT32, new int[]{4}),
                        contiguousInput(2, DataType.FLOAT32, new int[]{4})
                ),
                3
        );

        run(plan, DataType.FLOAT32, Cpu1LayoutKind.CONTIGUOUS, Cpu1FusedCodegenLoopKind.CONTIGUOUS_SCALAR,
                List.of(mask, trueValues, falseValues), output);

        assertArrayEquals(new float[]{1.0f, 20.0f, 3.0f, 40.0f}, output.toFloat32ArrayCopy(), 1.0e-6f);
    }

    @Test
    void vectorAndMemorySegmentPlansAreRejectedAtPrepareTime() {
        Cpu1FusedExpressionPlan plan = new Cpu1FusedExpressionPlan(
                List.of(node(0, Operation.OpType.RELU, List.of(0), 1, DataType.FLOAT32)),
                List.of(contiguousInput(0, DataType.FLOAT32, new int[]{4})),
                1
        );

        Cpu1FusedCodegenPlan vectorPlan = Cpu1FusedCodegenPlan.from(
                plan,
                DataType.FLOAT32,
                Cpu1LayoutKind.CONTIGUOUS,
                Cpu1StorageKind.JAVA_ARRAY,
                Cpu1FusedCodegenLoopKind.CONTIGUOUS_VECTOR,
                Cpu1PrepareConfig.vectorSingleThread()
        );
        Cpu1FusedCodegenPlan segmentPlan = Cpu1FusedCodegenPlan.from(
                plan,
                DataType.FLOAT32,
                Cpu1LayoutKind.CONTIGUOUS,
                Cpu1StorageKind.MEMORY_SEGMENT,
                Cpu1FusedCodegenLoopKind.CONTIGUOUS_SCALAR,
                Cpu1PrepareConfig.scalarMemorySegmentSingleThread()
        );

        assertEquals(Cpu1FusedCodegenRejectionReason.UNSUPPORTED_LOOP_KIND, vectorPlan.rejectionReason());
        assertEquals(Cpu1FusedCodegenRejectionReason.UNSUPPORTED_STORAGE_KIND, segmentPlan.rejectionReason());
    }

    private static void run(
            Cpu1FusedExpressionPlan plan,
            DataType computeType,
            Cpu1LayoutKind layoutKind,
            Cpu1FusedCodegenLoopKind loopKind,
            List<Tensor> inputs,
            Tensor output
    ) {
        run(kernel(plan, computeType, layoutKind, loopKind), plan, computeType, layoutKind, inputs, output);
    }

    private static void run(
            Cpu1FusedCodegenKernel kernel,
            Cpu1FusedExpressionPlan plan,
            DataType computeType,
            Cpu1LayoutKind layoutKind,
            List<Tensor> inputs,
            Tensor output
    ) {
        Cpu1PreparedFusedElementwiseUnit preparedUnit = prepared(plan, computeType, layoutKind, kernel, output);
        List<Cpu1TensorView> views = new ArrayList<>(inputs.size());
        for (Tensor input : inputs) {
            views.add(Cpu1TensorView.fromTensor(input).broadcastToShape(output.getShape()));
        }
        kernel.computeRange(new Cpu1FusedKernelArgs(preparedUnit, views, Cpu1TensorView.fromTensor(output)),
                0,
                output.getFlatDataSize());
    }

    private static Cpu1FusedCodegenKernel kernel(
            Cpu1FusedExpressionPlan plan,
            DataType computeType,
            Cpu1LayoutKind layoutKind,
            Cpu1FusedCodegenLoopKind loopKind
    ) {
        Cpu1FusedCodegenPlan codegenPlan = Cpu1FusedCodegenPlan.from(
                plan,
                computeType,
                layoutKind,
                Cpu1StorageKind.JAVA_ARRAY,
                loopKind,
                Cpu1PrepareConfig.scalarSingleThread()
        );
        assertEquals(Cpu1FusedCodegenRejectionReason.NONE, codegenPlan.rejectionReason());
        return Cpu1FusedCodegenKernelFactory.prepareKernel(codegenPlan);
    }

    private static Cpu1PreparedFusedElementwiseUnit prepared(
            Cpu1FusedExpressionPlan plan,
            DataType computeType,
            Cpu1LayoutKind layoutKind,
            Cpu1FusedCodegenKernel kernel,
            Tensor output
    ) {
        return new Cpu1PreparedFusedElementwiseUnit(
                "test-fused",
                List.of(100),
                plan.inputs().stream().map(Cpu1FusedInputPlan::nodeId).toList(),
                100,
                computeType,
                output.getFlatDataSize(),
                output.getShape(),
                plan,
                layoutKind,
                Cpu1StorageKind.JAVA_ARRAY,
                new Cpu1SingleThreadLaunch(),
                Cpu1LaunchConfig.singleThread(),
                new Cpu1FusedDispatchDecision(
                        Cpu1CostClass.CHEAP_ELEMENTWISE,
                        Cpu1VectorizationKind.SCALAR,
                        Cpu1LaunchConfig.singleThread(),
                        Cpu1StorageKind.JAVA_ARRAY,
                        1024,
                        1024,
                        1
                ),
                Cpu1FusedCodegenRejectionReason.NONE,
                kernel,
                false,
                false
        );
    }

    private static Cpu1FusedExpressionPlan scalarPlan(float scalar) {
        return new Cpu1FusedExpressionPlan(
                List.of(new Cpu1FusedNodePlan(
                        0,
                        10,
                        Operation.OpType.MUL_SCALAR,
                        List.of(0),
                        1,
                        DataType.FLOAT32,
                        Cpu1FusedScalarParameter.of(scalar, scalar)
                )),
                List.of(contiguousInput(0, DataType.FLOAT32, new int[]{3})),
                1
        );
    }

    private static Cpu1FusedNodePlan node(
            int index,
            Operation.OpType opType,
            List<Integer> inputRefs,
            int outputRef,
            DataType outputType
    ) {
        return new Cpu1FusedNodePlan(index, 10 + index, opType, inputRefs, outputRef, outputType,
                Cpu1FusedScalarParameter.NONE);
    }

    private static Cpu1FusedInputPlan contiguousInput(int ref, DataType dataType, int[] shape) {
        int[] strides = denseStrides(shape);
        return input(ref, dataType, shape, strides, shape, strides, strides, Cpu1FusedAccessKind.DIRECT_CONTIGUOUS);
    }

    private static Cpu1FusedInputPlan input(
            int ref,
            DataType dataType,
            int[] shape,
            int[] strides,
            int[] outputShape,
            int[] outputDenseStrides,
            int[] effectiveStrides,
            Cpu1FusedAccessKind accessKind
    ) {
        return new Cpu1FusedInputPlan(
                ref,
                ref,
                dataType,
                shape,
                strides,
                outputShape,
                outputDenseStrides,
                0,
                effectiveStrides,
                accessKind
        );
    }

    private static int[] denseStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int stride = 1;
        for (int dim = shape.length - 1; dim >= 0; dim--) {
            strides[dim] = stride;
            stride *= shape[dim];
        }
        return strides;
    }

    private static Tensor f32(float[] values, int... shape) {
        return new Tensor(values, shape, null, "f32", DataType.FLOAT32);
    }

    private static Tensor f64(double[] values, int... shape) {
        return new Tensor(values, shape, null, "f64", DataType.FLOAT64);
    }

    private static Tensor bool(byte[] values, int... shape) {
        return new Tensor(values, shape, null, "bool", DataType.BOOL);
    }
}
