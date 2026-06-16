package backend.cpu1;

import backend.cpu.nativecpu.NativeCpuStorageFactory;
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
import backend.cpu1.launch.Cpu1LaunchPolicy;
import backend.cpu1.launch.Cpu1ParallelLaunch;
import backend.cpu1.launch.Cpu1RangeLauncher;
import backend.cpu1.launch.Cpu1SingleThreadLaunch;
import backend.cpu1.prepare.Cpu1PrepareConfig;
import backend.cpu1.prepare.Cpu1PreparedFusedElementwiseUnit;
import backend.cpu1.prepare.dispatch.Cpu1CostClass;
import backend.cpu1.prepare.dispatch.Cpu1FusedDispatchDecision;
import backend.cpu1.storage.Cpu1StorageKind;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.dtype.TensorDTypeOps;
import tensor.storage.NativeTensorStorage;
import utils.FastTranscendentals;
import utils.SpecialFunctions;

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

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
    void generatedF32ScalarComputesMathIntrinsicsAndScalarPow() {
        float[] inputValues = new float[]{0.10f, 0.25f, 0.75f, 1.25f};
        float[] expected = new float[inputValues.length];
        for (int i = 0; i < expected.length; i++) {
            float value = (float) Math.exp(inputValues[i]);
            value = (float) Math.log(value);
            value = (float) Math.tanh(value);
            value = (float) Math.sqrt(value);
            value = 1.0f / (1.0f + (float) Math.exp(-value));
            expected[i] = (float) Math.pow(value, 1.5f);
        }
        Tensor input = f32(inputValues, 4);
        Tensor output = f32(new float[4], 4);
        Cpu1FusedExpressionPlan plan = new Cpu1FusedExpressionPlan(
                List.of(
                        node(0, Operation.OpType.EXP, List.of(0), 1, DataType.FLOAT32),
                        node(1, Operation.OpType.LOG, List.of(1), 2, DataType.FLOAT32),
                        node(2, Operation.OpType.TANH, List.of(2), 3, DataType.FLOAT32),
                        node(3, Operation.OpType.SQRT, List.of(3), 4, DataType.FLOAT32),
                        node(4, Operation.OpType.SIGMOID, List.of(4), 5, DataType.FLOAT32),
                        scalarNode(5, Operation.OpType.POW, List.of(5), 6, DataType.FLOAT32, 1.5f)
                ),
                List.of(contiguousInput(0, DataType.FLOAT32, new int[]{4})),
                6
        );

        run(plan, DataType.FLOAT32, Cpu1LayoutKind.CONTIGUOUS, Cpu1FusedCodegenLoopKind.CONTIGUOUS_SCALAR,
                List.of(input), output);

        assertArrayEquals(expected, output.toFloat32ArrayCopy(), 2.0e-5f);
    }

    @Test
    void generatedF32ScalarHonorsFastApproximationFlagsForExpAndTanh() {
        float[] inputValues = new float[]{-0.75f, -0.25f, 0.25f, 0.75f};
        float[] expected = new float[inputValues.length];
        for (int i = 0; i < expected.length; i++) {
            expected[i] = FastTranscendentals.fastTanhF32(FastTranscendentals.fastExpF32(inputValues[i]));
        }
        Tensor input = f32(inputValues, 4);
        Tensor output = f32(new float[4], 4);
        Cpu1FusedExpressionPlan plan = new Cpu1FusedExpressionPlan(
                List.of(
                        node(0, Operation.OpType.EXP, List.of(0), 1, DataType.FLOAT32),
                        node(1, Operation.OpType.TANH, List.of(1), 2, DataType.FLOAT32)
                ),
                List.of(contiguousInput(0, DataType.FLOAT32, new int[]{4})),
                2
        );
        Cpu1FusedCodegenPlan codegenPlan = Cpu1FusedCodegenPlan.from(
                plan,
                DataType.FLOAT32,
                Cpu1LayoutKind.CONTIGUOUS,
                Cpu1StorageKind.JAVA_ARRAY,
                Cpu1FusedCodegenLoopKind.CONTIGUOUS_SCALAR,
                Cpu1PrepareConfig.scalarSingleThread().withApproximation(true, true)
        );
        assertTrue(codegenPlan.classSignature().canonicalSignature().contains("fastExpF32(F)F"));
        assertTrue(codegenPlan.classSignature().canonicalSignature().contains("fastTanhF32(F)F"));

        run(Cpu1FusedCodegenKernelFactory.prepareKernel(codegenPlan), plan, DataType.FLOAT32,
                Cpu1LayoutKind.CONTIGUOUS, List.of(input), output);

        assertArrayEquals(expected, output.toFloat32ArrayCopy(), 1.0e-6f);
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
    void generatedF64ScalarComputesLogSqrtPowTensorFloorCeilSign() {
        double[] inputValues = new double[]{Math.exp(4.0), Math.exp(9.0), Math.exp(16.0)};
        double[] exponentValues = new double[]{2.0, 3.0, 0.5};
        double[] floorOffsets = new double[]{-4.75, -27.25, -1.25};
        double[] ceilOffsets = new double[]{0.0, 1.0, 0.0};
        double[] expected = new double[inputValues.length];
        for (int i = 0; i < expected.length; i++) {
            double value = Math.log(inputValues[i]);
            value = Math.sqrt(value);
            value = Math.pow(value, exponentValues[i]);
            value = Math.floor(value + floorOffsets[i]);
            value = Math.ceil(value + ceilOffsets[i]);
            expected[i] = value > 0.0d ? 1.0d : (value < 0.0d ? -1.0d : 0.0d);
        }
        Tensor input = f64(inputValues, 3);
        Tensor exponents = f64(exponentValues, 3);
        Tensor floorOffset = f64(floorOffsets, 3);
        Tensor ceilOffset = f64(ceilOffsets, 3);
        Tensor output = f64(new double[3], 3);
        Cpu1FusedExpressionPlan plan = new Cpu1FusedExpressionPlan(
                List.of(
                        node(0, Operation.OpType.LOG, List.of(0), 4, DataType.FLOAT64),
                        node(1, Operation.OpType.SQRT, List.of(4), 5, DataType.FLOAT64),
                        node(2, Operation.OpType.POW_TENSOR, List.of(5, 1), 6, DataType.FLOAT64),
                        node(3, Operation.OpType.ADD, List.of(6, 2), 7, DataType.FLOAT64),
                        node(4, Operation.OpType.FLOOR, List.of(7), 8, DataType.FLOAT64),
                        node(5, Operation.OpType.ADD, List.of(8, 3), 9, DataType.FLOAT64),
                        node(6, Operation.OpType.CEIL, List.of(9), 10, DataType.FLOAT64),
                        node(7, Operation.OpType.SIGN, List.of(10), 11, DataType.FLOAT64)
                ),
                List.of(
                        contiguousInput(0, DataType.FLOAT64, new int[]{3}),
                        contiguousInput(1, DataType.FLOAT64, new int[]{3}),
                        contiguousInput(2, DataType.FLOAT64, new int[]{3}),
                        contiguousInput(3, DataType.FLOAT64, new int[]{3})
                ),
                11
        );

        run(plan, DataType.FLOAT64, Cpu1LayoutKind.CONTIGUOUS, Cpu1FusedCodegenLoopKind.CONTIGUOUS_SCALAR,
                List.of(input, exponents, floorOffset, ceilOffset), output);

        assertArrayEquals(expected, output.toFloat64ArrayCopy(), 1.0e-12);
    }

    @Test
    void generatedBf16ContiguousScalarComputesSupportedSubsetWithRoundtripOutput() {
        float[] leftValues = new float[]{1.25f, -2.5f, 3.75f, -4.5f};
        float[] rightValues = new float[]{2.0f, -3.0f, 0.5f, -1.25f};
        float[] biasValues = new float[]{-1.0f, 0.25f, 1.5f, -10.0f};
        float[] expected = new float[leftValues.length];
        for (int i = 0; i < expected.length; i++) {
            float value = bf16Round(leftValues[i]) * bf16Round(rightValues[i]) + bf16Round(biasValues[i]);
            value = Math.max(value, -2.0f);
            value *= 0.5f;
            value = Math.max(0.0f, value);
            expected[i] = bf16Round(value);
        }
        Tensor left = bf16(leftValues, 4);
        Tensor right = bf16(rightValues, 4);
        Tensor bias = bf16(biasValues, 4);
        Tensor output = bf16(new float[4], 4);
        Cpu1FusedExpressionPlan plan = new Cpu1FusedExpressionPlan(
                List.of(
                        node(0, Operation.OpType.MUL, List.of(0, 1), 3, DataType.BFLOAT16),
                        node(1, Operation.OpType.ADD, List.of(3, 2), 4, DataType.BFLOAT16),
                        scalarNode(2, Operation.OpType.CLAMP_MIN, List.of(4), 5, DataType.BFLOAT16, -2.0f),
                        scalarNode(3, Operation.OpType.MUL_SCALAR, List.of(5), 6, DataType.BFLOAT16, 0.5f),
                        node(4, Operation.OpType.RELU, List.of(6), 7, DataType.BFLOAT16)
                ),
                List.of(
                        contiguousInput(0, DataType.BFLOAT16, new int[]{4}),
                        contiguousInput(1, DataType.BFLOAT16, new int[]{4}),
                        contiguousInput(2, DataType.BFLOAT16, new int[]{4})
                ),
                7
        );

        run(plan, DataType.BFLOAT16, Cpu1LayoutKind.CONTIGUOUS, Cpu1FusedCodegenLoopKind.CONTIGUOUS_SCALAR,
                List.of(left, right, bias), output);

        assertArrayEquals(expected, bf16Values(output), 1.0e-3f);
    }

    @Test
    void generatedBf16ScalarComputesFastIntrinsicsThroughF32Locals() {
        float[] inputValues = new float[]{-0.75f, -0.1f, 0.25f, 0.9f};
        float[] expected = new float[inputValues.length];
        for (int i = 0; i < expected.length; i++) {
            float value = bf16Round(inputValues[i]);
            value = FastTranscendentals.fastExpF32(value);
            value = FastTranscendentals.fastTanhF32(value);
            value = SpecialFunctions.erf(value);
            expected[i] = bf16Round(value);
        }
        Tensor input = bf16(inputValues, 4);
        Tensor output = bf16(new float[4], 4);
        Cpu1FusedExpressionPlan plan = new Cpu1FusedExpressionPlan(
                List.of(
                        node(0, Operation.OpType.FAST_EXP, List.of(0), 1, DataType.BFLOAT16),
                        node(1, Operation.OpType.FAST_TANH, List.of(1), 2, DataType.BFLOAT16),
                        node(2, Operation.OpType.ERF, List.of(2), 3, DataType.BFLOAT16)
                ),
                List.of(contiguousInput(0, DataType.BFLOAT16, new int[]{4})),
                3
        );

        run(plan, DataType.BFLOAT16, Cpu1LayoutKind.CONTIGUOUS, Cpu1FusedCodegenLoopKind.CONTIGUOUS_SCALAR,
                List.of(input), output);

        assertArrayEquals(expected, bf16Values(output), 1.0e-3f);
    }

    @Test
    void generatedBf16StridedScalarComputesBroadcastAdd() {
        Tensor base = bf16(new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, 2, 3);
        Tensor strided = base.permute(1, 0);
        Tensor scalar = bf16(new float[]{10.0f}, 1);
        Tensor output = bf16(new float[6], 3, 2);
        int[] outShape = new int[]{3, 2};
        Cpu1FusedExpressionPlan plan = new Cpu1FusedExpressionPlan(
                List.of(node(0, Operation.OpType.ADD, List.of(0, 1), 2, DataType.BFLOAT16)),
                List.of(
                        input(0, DataType.BFLOAT16, new int[]{3, 2}, new int[]{1, 3}, outShape,
                                new int[]{2, 1}, new int[]{1, 3}, Cpu1FusedAccessKind.DIRECT_STRIDED),
                        input(1, DataType.BFLOAT16, new int[]{1}, new int[]{1}, outShape,
                                new int[]{2, 1}, new int[]{0, 0}, Cpu1FusedAccessKind.BROADCAST_STRIDED)
                ),
                2
        );

        run(plan, DataType.BFLOAT16, Cpu1LayoutKind.STRIDED_RANK2, Cpu1FusedCodegenLoopKind.STRIDED_SCALAR,
                List.of(strided, scalar), output);

        assertArrayEquals(new float[]{11.0f, 14.0f, 12.0f, 15.0f, 13.0f, 16.0f},
                bf16Values(output), 1.0e-3f);
    }

    @Test
    void generatedF32ContiguousVectorComputesSupportedSubsetWithScalarTail() {
        int length = FloatVector.SPECIES_PREFERRED.length() * 2 + 3;
        float[] leftValues = new float[length];
        float[] rightValues = new float[length];
        float[] biasValues = new float[length];
        float[] divisorValues = new float[length];
        float[] minValues = new float[length];
        float[] maxValues = new float[length];
        float[] expected = new float[length];
        for (int i = 0; i < length; i++) {
            leftValues[i] = i + 1.0f;
            rightValues[i] = 0.25f * (i + 2.0f);
            biasValues[i] = 0.5f;
            divisorValues[i] = 2.0f + (i % 3);
            minValues[i] = 0.75f;
            maxValues[i] = 6.0f;
            float value = (leftValues[i] + rightValues[i] - biasValues[i]) * rightValues[i] / divisorValues[i];
            value = Math.min(value, maxValues[i]);
            value = Math.max(value, minValues[i]);
            value = -value;
            value = Math.abs(value);
            value = Math.max(value, 1.0f);
            value = Math.min(value, 5.0f);
            value *= 0.5f;
            value = Math.max(0.0f, value);
            expected[i] = 1.0f / value;
        }
        Tensor left = f32(leftValues, length);
        Tensor right = f32(rightValues, length);
        Tensor bias = f32(biasValues, length);
        Tensor divisor = f32(divisorValues, length);
        Tensor min = f32(minValues, length);
        Tensor max = f32(maxValues, length);
        Tensor output = f32(new float[length], length);
        Cpu1FusedExpressionPlan plan = new Cpu1FusedExpressionPlan(
                List.of(
                        node(0, Operation.OpType.ADD, List.of(0, 1), 6, DataType.FLOAT32),
                        node(1, Operation.OpType.SUB, List.of(6, 2), 7, DataType.FLOAT32),
                        node(2, Operation.OpType.MUL, List.of(7, 1), 8, DataType.FLOAT32),
                        node(3, Operation.OpType.DIV, List.of(8, 3), 9, DataType.FLOAT32),
                        node(4, Operation.OpType.MIN, List.of(9, 5), 10, DataType.FLOAT32),
                        node(5, Operation.OpType.MAX, List.of(10, 4), 11, DataType.FLOAT32),
                        node(6, Operation.OpType.NEG, List.of(11), 12, DataType.FLOAT32),
                        node(7, Operation.OpType.ABS, List.of(12), 13, DataType.FLOAT32),
                        scalarNode(8, Operation.OpType.CLAMP_MIN, List.of(13), 14, DataType.FLOAT32, 1.0f),
                        scalarNode(9, Operation.OpType.CLAMP_MAX, List.of(14), 15, DataType.FLOAT32, 5.0f),
                        scalarNode(10, Operation.OpType.MUL_SCALAR, List.of(15), 16, DataType.FLOAT32, 0.5f),
                        node(11, Operation.OpType.RELU, List.of(16), 17, DataType.FLOAT32),
                        node(12, Operation.OpType.INV, List.of(17), 18, DataType.FLOAT32),
                        node(13, Operation.OpType.NOOP, List.of(18), 19, DataType.FLOAT32)
                ),
                List.of(
                        contiguousInput(0, DataType.FLOAT32, new int[]{length}),
                        contiguousInput(1, DataType.FLOAT32, new int[]{length}),
                        contiguousInput(2, DataType.FLOAT32, new int[]{length}),
                        contiguousInput(3, DataType.FLOAT32, new int[]{length}),
                        contiguousInput(4, DataType.FLOAT32, new int[]{length}),
                        contiguousInput(5, DataType.FLOAT32, new int[]{length})
                ),
                19
        );
        Cpu1FusedCodegenPlan codegenPlan = codegenPlan(
                plan,
                DataType.FLOAT32,
                Cpu1LayoutKind.CONTIGUOUS,
                Cpu1FusedCodegenLoopKind.CONTIGUOUS_VECTOR
        );
        assertEquals(Cpu1FusedCodegenRejectionReason.NONE, codegenPlan.rejectionReason());
        assertTrue(codegenPlan.classSignature().canonicalSignature().contains("loop=CONTIGUOUS_VECTOR"));

        run(Cpu1FusedCodegenKernelFactory.prepareKernel(codegenPlan), plan, DataType.FLOAT32,
                Cpu1LayoutKind.CONTIGUOUS, List.of(left, right, bias, divisor, min, max), output);

        assertArrayEquals(expected, output.toFloat32ArrayCopy(), 1.0e-5f);
    }

    @Test
    void generatedF64ContiguousVectorComputesAddReluWithTail() {
        int length = DoubleVector.SPECIES_PREFERRED.length() + 1;
        double[] leftValues = new double[length];
        double[] rightValues = new double[length];
        double[] expected = new double[length];
        for (int i = 0; i < length; i++) {
            leftValues[i] = i - 2.5;
            rightValues[i] = 1.25 * i;
            expected[i] = Math.max(0.0, leftValues[i] + rightValues[i]);
        }
        Tensor left = f64(leftValues, length);
        Tensor right = f64(rightValues, length);
        Tensor output = f64(new double[length], length);
        Cpu1FusedExpressionPlan plan = new Cpu1FusedExpressionPlan(
                List.of(
                        node(0, Operation.OpType.ADD, List.of(0, 1), 2, DataType.FLOAT64),
                        node(1, Operation.OpType.RELU, List.of(2), 3, DataType.FLOAT64)
                ),
                List.of(
                        contiguousInput(0, DataType.FLOAT64, new int[]{length}),
                        contiguousInput(1, DataType.FLOAT64, new int[]{length})
                ),
                3
        );
        Cpu1FusedCodegenPlan codegenPlan = codegenPlan(
                plan,
                DataType.FLOAT64,
                Cpu1LayoutKind.CONTIGUOUS,
                Cpu1FusedCodegenLoopKind.CONTIGUOUS_VECTOR
        );
        assertEquals(Cpu1FusedCodegenRejectionReason.NONE, codegenPlan.rejectionReason());

        run(Cpu1FusedCodegenKernelFactory.prepareKernel(codegenPlan), plan, DataType.FLOAT64,
                Cpu1LayoutKind.CONTIGUOUS, List.of(left, right), output);

        assertArrayEquals(expected, output.toFloat64ArrayCopy(), 1.0e-12);
    }

    @Test
    void generatedF32ContiguousVectorRunsThroughParallelLaunchPolicy() {
        int chunkSize = FloatVector.SPECIES_PREFERRED.length() + 3;
        int length = chunkSize * 5 + 2;
        float[] leftValues = new float[length];
        float[] rightValues = new float[length];
        float[] biasValues = new float[length];
        float[] expected = new float[length];
        for (int i = 0; i < length; i++) {
            leftValues[i] = (i % 17) - 8.0f;
            rightValues[i] = 0.25f * (i % 11);
            biasValues[i] = (i % 5) - 2.0f;
            expected[i] = Math.max(0.0f, (leftValues[i] + rightValues[i]) * biasValues[i]);
        }
        Tensor left = f32(leftValues, length);
        Tensor right = f32(rightValues, length);
        Tensor bias = f32(biasValues, length);
        Tensor output = f32(new float[length], length);
        Cpu1FusedExpressionPlan plan = new Cpu1FusedExpressionPlan(
                List.of(
                        node(0, Operation.OpType.ADD, List.of(0, 1), 3, DataType.FLOAT32),
                        node(1, Operation.OpType.MUL, List.of(3, 2), 4, DataType.FLOAT32),
                        node(2, Operation.OpType.RELU, List.of(4), 5, DataType.FLOAT32)
                ),
                List.of(
                        contiguousInput(0, DataType.FLOAT32, new int[]{length}),
                        contiguousInput(1, DataType.FLOAT32, new int[]{length}),
                        contiguousInput(2, DataType.FLOAT32, new int[]{length})
                ),
                5
        );
        Cpu1FusedCodegenPlan codegenPlan = codegenPlan(
                plan,
                DataType.FLOAT32,
                Cpu1LayoutKind.CONTIGUOUS,
                Cpu1FusedCodegenLoopKind.CONTIGUOUS_VECTOR
        );
        assertEquals(Cpu1FusedCodegenRejectionReason.NONE, codegenPlan.rejectionReason());
        Cpu1FusedCodegenKernel kernel = Cpu1FusedCodegenKernelFactory.prepareKernel(codegenPlan);
        Cpu1LaunchConfig launchConfig = Cpu1LaunchConfig.parallel(4, chunkSize);
        Cpu1PreparedFusedElementwiseUnit preparedUnit = prepared(
                plan,
                DataType.FLOAT32,
                Cpu1LayoutKind.CONTIGUOUS,
                Cpu1StorageKind.JAVA_ARRAY,
                kernel,
                output,
                launchConfig,
                Cpu1VectorizationKind.VECTOR,
                FloatVector.SPECIES_PREFERRED.length()
        );
        assertTrue(preparedUnit.launchPolicy() instanceof Cpu1ParallelLaunch);

        Cpu1FusedKernelArgs args = new Cpu1FusedKernelArgs(
                preparedUnit,
                List.of(
                        Cpu1TensorView.fromTensor(left),
                        Cpu1TensorView.fromTensor(right),
                        Cpu1TensorView.fromTensor(bias)
                ),
                Cpu1TensorView.fromTensor(output)
        );
        List<int[]> ranges = Collections.synchronizedList(new ArrayList<>());
        preparedUnit.launchPolicy().launch(
                args.elementCount(),
                (startInclusive, endExclusive) -> {
                    ranges.add(new int[]{startInclusive, endExclusive});
                    preparedUnit.generatedKernel().computeRange(args, startInclusive, endExclusive);
                }
        );

        assertArrayEquals(expected, output.toFloat32ArrayCopy(), 1.0e-5f);
        assertEquals(Cpu1RangeLauncher.slotCount(length, launchConfig), ranges.size());
        assertTrue(ranges.size() > 1);
        assertContiguousChunkRanges(ranges, length, chunkSize);
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
    void generatedF32MemorySegmentContiguousComputesMulAddRelu() {
        Tensor left = f32(new float[4], 4);
        Tensor right = f32(new float[4], 4);
        Tensor bias = f32(new float[4], 4);
        Tensor output = f32(new float[4], 4);
        NativeTensorStorage leftStorage = nativeF32(1.0f, -2.0f, 3.0f, -4.0f);
        NativeTensorStorage rightStorage = nativeF32(10.0f, 20.0f, -30.0f, -40.0f);
        NativeTensorStorage biasStorage = nativeF32(-5.0f, 50.0f, 100.0f, -200.0f);
        NativeTensorStorage outputStorage = nativeStorage(DataType.FLOAT32, 4, "f32-segment-output");
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

        runNative(plan, DataType.FLOAT32, Cpu1LayoutKind.CONTIGUOUS, Cpu1FusedCodegenLoopKind.CONTIGUOUS_SCALAR,
                List.of(
                        nativeInput(left, leftStorage),
                        nativeInput(right, rightStorage),
                        nativeInput(bias, biasStorage)
                ),
                output,
                outputStorage);

        assertArrayEquals(new float[]{5.0f, 10.0f, 10.0f, 0.0f}, readNativeF32(outputStorage, 4), 1.0e-6f);
    }

    @Test
    void generatedF32MemorySegmentScalarComputesIntrinsic() {
        Tensor input = f32(new float[4], 4);
        Tensor output = f32(new float[4], 4);
        NativeTensorStorage inputStorage = nativeF32(-0.5f, 0.0f, 0.75f, 1.25f);
        NativeTensorStorage outputStorage = nativeStorage(DataType.FLOAT32, 4, "f32-segment-exp-output");
        Cpu1FusedExpressionPlan plan = new Cpu1FusedExpressionPlan(
                List.of(node(0, Operation.OpType.EXP, List.of(0), 1, DataType.FLOAT32)),
                List.of(contiguousInput(0, DataType.FLOAT32, new int[]{4})),
                1
        );

        runNative(plan, DataType.FLOAT32, Cpu1LayoutKind.CONTIGUOUS, Cpu1FusedCodegenLoopKind.CONTIGUOUS_SCALAR,
                List.of(nativeInput(input, inputStorage)),
                output,
                outputStorage);

        assertArrayEquals(new float[]{
                        (float) Math.exp(-0.5f),
                        1.0f,
                        (float) Math.exp(0.75f),
                        (float) Math.exp(1.25f)
                },
                readNativeF32(outputStorage, 4),
                1.0e-6f);
    }

    @Test
    void generatedF64MemorySegmentContiguousComputesAddRelu() {
        Tensor left = f64(new double[3], 3);
        Tensor right = f64(new double[3], 3);
        Tensor output = f64(new double[3], 3);
        NativeTensorStorage leftStorage = nativeF64(1.5, -2.0, 3.25);
        NativeTensorStorage rightStorage = nativeF64(10.0, 20.5, -30.25);
        NativeTensorStorage outputStorage = nativeStorage(DataType.FLOAT64, 3, "f64-segment-output");
        Cpu1FusedExpressionPlan plan = new Cpu1FusedExpressionPlan(
                List.of(
                        node(0, Operation.OpType.ADD, List.of(0, 1), 2, DataType.FLOAT64),
                        node(1, Operation.OpType.RELU, List.of(2), 3, DataType.FLOAT64)
                ),
                List.of(
                        contiguousInput(0, DataType.FLOAT64, new int[]{3}),
                        contiguousInput(1, DataType.FLOAT64, new int[]{3})
                ),
                3
        );

        runNative(plan, DataType.FLOAT64, Cpu1LayoutKind.CONTIGUOUS, Cpu1FusedCodegenLoopKind.CONTIGUOUS_SCALAR,
                List.of(
                        nativeInput(left, leftStorage),
                        nativeInput(right, rightStorage)
                ),
                output,
                outputStorage);

        assertArrayEquals(new double[]{11.5, 18.5, 0.0}, readNativeF64(outputStorage, 3), 1.0e-12);
    }

    @Test
    void generatedF32MemorySegmentStridedComputesBroadcastAdd() {
        Tensor base = f32(new float[]{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 2, 3);
        Tensor strided = base.permute(1, 0);
        Tensor scalar = f32(new float[1], 1);
        Tensor output = f32(new float[6], 3, 2);
        NativeTensorStorage baseStorage = nativeF32(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f);
        NativeTensorStorage scalarStorage = nativeF32(10.0f);
        NativeTensorStorage outputStorage = nativeStorage(DataType.FLOAT32, 6, "f32-segment-strided-output");
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

        runNative(plan, DataType.FLOAT32, Cpu1LayoutKind.STRIDED_RANK2, Cpu1FusedCodegenLoopKind.STRIDED_SCALAR,
                List.of(
                        nativeInput(strided, baseStorage),
                        nativeInput(scalar, scalarStorage)
                ),
                output,
                outputStorage);

        assertArrayEquals(new float[]{11.0f, 14.0f, 12.0f, 15.0f, 13.0f, 16.0f},
                readNativeF32(outputStorage, 6), 1.0e-6f);
    }

    @Test
    void memorySegmentVectorPlanIsRejectedAtPrepareTime() {
        Cpu1FusedExpressionPlan plan = new Cpu1FusedExpressionPlan(
                List.of(node(0, Operation.OpType.RELU, List.of(0), 1, DataType.FLOAT32)),
                List.of(contiguousInput(0, DataType.FLOAT32, new int[]{4})),
                1
        );

        Cpu1FusedCodegenPlan segmentPlan = Cpu1FusedCodegenPlan.from(
                plan,
                DataType.FLOAT32,
                Cpu1LayoutKind.CONTIGUOUS,
                Cpu1StorageKind.MEMORY_SEGMENT,
                Cpu1FusedCodegenLoopKind.CONTIGUOUS_VECTOR,
                Cpu1PrepareConfig.vectorMemorySegmentSingleThread()
        );

        assertEquals(Cpu1FusedCodegenRejectionReason.UNSUPPORTED_SEGMENT_VECTOR, segmentPlan.rejectionReason());
    }

    @Test
    void bf16VectorPlanIsRejectedAtPrepareTime() {
        Cpu1FusedExpressionPlan plan = new Cpu1FusedExpressionPlan(
                List.of(node(0, Operation.OpType.RELU, List.of(0), 1, DataType.BFLOAT16)),
                List.of(contiguousInput(0, DataType.BFLOAT16, new int[]{4})),
                1
        );

        Cpu1FusedCodegenPlan vectorPlan = Cpu1FusedCodegenPlan.from(
                plan,
                DataType.BFLOAT16,
                Cpu1LayoutKind.CONTIGUOUS,
                Cpu1StorageKind.JAVA_ARRAY,
                Cpu1FusedCodegenLoopKind.CONTIGUOUS_VECTOR,
                Cpu1PrepareConfig.vectorSingleThread()
        );

        assertEquals(Cpu1FusedCodegenRejectionReason.UNSUPPORTED_BF16_VECTOR, vectorPlan.rejectionReason());
    }

    @Test
    void bf16MemorySegmentPlanIsRejectedAtPrepareTime() {
        Cpu1FusedExpressionPlan plan = new Cpu1FusedExpressionPlan(
                List.of(node(0, Operation.OpType.RELU, List.of(0), 1, DataType.BFLOAT16)),
                List.of(contiguousInput(0, DataType.BFLOAT16, new int[]{4})),
                1
        );

        Cpu1FusedCodegenPlan segmentPlan = Cpu1FusedCodegenPlan.from(
                plan,
                DataType.BFLOAT16,
                Cpu1LayoutKind.CONTIGUOUS,
                Cpu1StorageKind.MEMORY_SEGMENT,
                Cpu1FusedCodegenLoopKind.CONTIGUOUS_SCALAR,
                Cpu1PrepareConfig.scalarMemorySegmentSingleThread()
        );

        assertEquals(Cpu1FusedCodegenRejectionReason.UNSUPPORTED_BF16_SEGMENT, segmentPlan.rejectionReason());
    }

    @Test
    void vectorWhereIsExplicitlyRejectedAtPrepareTime() {
        Cpu1FusedExpressionPlan plan = new Cpu1FusedExpressionPlan(
                List.of(node(0, Operation.OpType.WHERE, List.of(0, 1, 2), 3, DataType.FLOAT32)),
                List.of(
                        contiguousInput(0, DataType.BOOL, new int[]{4}),
                        contiguousInput(1, DataType.FLOAT32, new int[]{4}),
                        contiguousInput(2, DataType.FLOAT32, new int[]{4})
                ),
                3
        );

        Cpu1FusedCodegenPlan vectorPlan = Cpu1FusedCodegenPlan.from(
                plan,
                DataType.FLOAT32,
                Cpu1LayoutKind.CONTIGUOUS,
                Cpu1StorageKind.JAVA_ARRAY,
                Cpu1FusedCodegenLoopKind.CONTIGUOUS_VECTOR,
                Cpu1PrepareConfig.vectorSingleThread()
        );

        assertEquals(Cpu1FusedCodegenRejectionReason.UNSUPPORTED_VECTOR_OPERATION, vectorPlan.rejectionReason());
    }

    @Test
    void vectorIntrinsicIsExplicitlyRejectedAtPrepareTime() {
        Cpu1FusedExpressionPlan plan = new Cpu1FusedExpressionPlan(
                List.of(node(0, Operation.OpType.EXP, List.of(0), 1, DataType.FLOAT32)),
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

        assertEquals(Cpu1FusedCodegenRejectionReason.UNSUPPORTED_VECTOR_OPERATION, vectorPlan.rejectionReason());
    }

    @Test
    void powWithoutScalarBindingIsRejectedAtPrepareTime() {
        Cpu1FusedExpressionPlan plan = new Cpu1FusedExpressionPlan(
                List.of(node(0, Operation.OpType.POW, List.of(0), 1, DataType.FLOAT32)),
                List.of(contiguousInput(0, DataType.FLOAT32, new int[]{4})),
                1
        );

        Cpu1FusedCodegenPlan codegenPlan = Cpu1FusedCodegenPlan.from(
                plan,
                DataType.FLOAT32,
                Cpu1LayoutKind.CONTIGUOUS,
                Cpu1StorageKind.JAVA_ARRAY,
                Cpu1FusedCodegenLoopKind.CONTIGUOUS_SCALAR,
                Cpu1PrepareConfig.scalarSingleThread()
        );

        assertEquals(Cpu1FusedCodegenRejectionReason.UNSUPPORTED_SCALAR_BINDING, codegenPlan.rejectionReason());
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

    private static void runNative(
            Cpu1FusedExpressionPlan plan,
            DataType computeType,
            Cpu1LayoutKind layoutKind,
            Cpu1FusedCodegenLoopKind loopKind,
            List<NativeInput> inputs,
            Tensor output,
            NativeTensorStorage outputStorage
    ) {
        Cpu1FusedCodegenPlan codegenPlan = Cpu1FusedCodegenPlan.from(
                plan,
                computeType,
                layoutKind,
                Cpu1StorageKind.MEMORY_SEGMENT,
                loopKind,
                Cpu1PrepareConfig.scalarMemorySegmentSingleThread()
        );
        assertEquals(Cpu1FusedCodegenRejectionReason.NONE, codegenPlan.rejectionReason());
        Cpu1FusedCodegenKernel kernel = Cpu1FusedCodegenKernelFactory.prepareKernel(codegenPlan);
        Cpu1PreparedFusedElementwiseUnit preparedUnit = prepared(
                plan,
                computeType,
                layoutKind,
                Cpu1StorageKind.MEMORY_SEGMENT,
                kernel,
                output
        );
        List<Cpu1TensorView> views = new ArrayList<>(inputs.size());
        for (NativeInput input : inputs) {
            views.add(Cpu1TensorView.fromNativeStorage(input.tensor(), input.storage())
                    .broadcastToShape(output.getShape()));
        }
        kernel.computeRange(new Cpu1FusedKernelArgs(
                        preparedUnit,
                        views,
                        Cpu1TensorView.fromNativeStorage(output, outputStorage)
                ),
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

    private static Cpu1FusedCodegenPlan codegenPlan(
            Cpu1FusedExpressionPlan plan,
            DataType computeType,
            Cpu1LayoutKind layoutKind,
            Cpu1FusedCodegenLoopKind loopKind
    ) {
        return Cpu1FusedCodegenPlan.from(
                plan,
                computeType,
                layoutKind,
                Cpu1StorageKind.JAVA_ARRAY,
                loopKind,
                loopKind == Cpu1FusedCodegenLoopKind.CONTIGUOUS_VECTOR
                        ? Cpu1PrepareConfig.vectorSingleThread()
                        : Cpu1PrepareConfig.scalarSingleThread()
        );
    }

    private static Cpu1PreparedFusedElementwiseUnit prepared(
            Cpu1FusedExpressionPlan plan,
            DataType computeType,
            Cpu1LayoutKind layoutKind,
            Cpu1FusedCodegenKernel kernel,
            Tensor output
    ) {
        return prepared(plan, computeType, layoutKind, Cpu1StorageKind.JAVA_ARRAY, kernel, output);
    }

    private static Cpu1PreparedFusedElementwiseUnit prepared(
            Cpu1FusedExpressionPlan plan,
            DataType computeType,
            Cpu1LayoutKind layoutKind,
            Cpu1StorageKind storageKind,
            Cpu1FusedCodegenKernel kernel,
            Tensor output
    ) {
        return prepared(
                plan,
                computeType,
                layoutKind,
                storageKind,
                kernel,
                output,
                Cpu1LaunchConfig.singleThread(),
                Cpu1VectorizationKind.SCALAR,
                1
        );
    }

    private static Cpu1PreparedFusedElementwiseUnit prepared(
            Cpu1FusedExpressionPlan plan,
            DataType computeType,
            Cpu1LayoutKind layoutKind,
            Cpu1StorageKind storageKind,
            Cpu1FusedCodegenKernel kernel,
            Tensor output,
            Cpu1LaunchConfig launchConfig,
            Cpu1VectorizationKind vectorizationKind,
            int vectorWidth
    ) {
        Cpu1LaunchPolicy launchPolicy = launchConfig.workerCount() == 1
                ? new Cpu1SingleThreadLaunch(launchConfig)
                : new Cpu1ParallelLaunch(launchConfig);
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
                storageKind,
                launchPolicy,
                launchConfig,
                new Cpu1FusedDispatchDecision(
                        Cpu1CostClass.CHEAP_ELEMENTWISE,
                        vectorizationKind,
                        launchConfig,
                        storageKind,
                        1024,
                        1024,
                        vectorWidth
                ),
                Cpu1FusedCodegenRejectionReason.NONE,
                kernel,
                false,
                false
        );
    }

    private static void assertContiguousChunkRanges(List<int[]> ranges, int elementCount, int chunkSize) {
        List<int[]> sorted = new ArrayList<>(ranges);
        sorted.sort((left, right) -> Integer.compare(left[0], right[0]));
        int expectedStart = 0;
        for (int[] range : sorted) {
            assertEquals(expectedStart, range[0]);
            assertTrue(range[1] > range[0]);
            assertTrue(range[1] - range[0] <= chunkSize);
            expectedStart = range[1];
        }
        assertEquals(elementCount, expectedStart);
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

    private static Cpu1FusedNodePlan scalarNode(
            int index,
            Operation.OpType opType,
            List<Integer> inputRefs,
            int outputRef,
            DataType outputType,
            float scalar
    ) {
        return new Cpu1FusedNodePlan(index, 10 + index, opType, inputRefs, outputRef, outputType,
                Cpu1FusedScalarParameter.of(scalar, scalar));
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

    private static NativeInput nativeInput(Tensor tensor, NativeTensorStorage storage) {
        return new NativeInput(tensor, storage);
    }

    private static NativeTensorStorage nativeF32(float... values) {
        NativeTensorStorage storage = nativeStorage(DataType.FLOAT32, values.length, "f32-segment-input");
        MemorySegment segment = storage.segment();
        for (int i = 0; i < values.length; i++) {
            segment.set(JAVA_FLOAT, (long) i * Float.BYTES, values[i]);
        }
        storage.markModified();
        return storage;
    }

    private static NativeTensorStorage nativeF64(double... values) {
        NativeTensorStorage storage = nativeStorage(DataType.FLOAT64, values.length, "f64-segment-input");
        MemorySegment segment = storage.segment();
        for (int i = 0; i < values.length; i++) {
            segment.set(JAVA_DOUBLE, (long) i * Double.BYTES, values[i]);
        }
        storage.markModified();
        return storage;
    }

    private static NativeTensorStorage nativeStorage(DataType dataType, int elements, String label) {
        return new NativeCpuStorageFactory().allocate(dataType, elements, label);
    }

    private static float[] readNativeF32(NativeTensorStorage storage, int length) {
        float[] values = new float[length];
        MemorySegment segment = storage.segment();
        for (int i = 0; i < values.length; i++) {
            values[i] = segment.get(JAVA_FLOAT, (long) i * Float.BYTES);
        }
        return values;
    }

    private static double[] readNativeF64(NativeTensorStorage storage, int length) {
        double[] values = new double[length];
        MemorySegment segment = storage.segment();
        for (int i = 0; i < values.length; i++) {
            values[i] = segment.get(JAVA_DOUBLE, (long) i * Double.BYTES);
        }
        return values;
    }

    private static Tensor f32(float[] values, int... shape) {
        return new Tensor(values, shape, null, "f32", DataType.FLOAT32);
    }

    private static Tensor f64(double[] values, int... shape) {
        return new Tensor(values, shape, null, "f64", DataType.FLOAT64);
    }

    private static Tensor bf16(float[] values, int... shape) {
        short[] bits = new short[values.length];
        for (int i = 0; i < values.length; i++) {
            bits[i] = TensorDTypeOps.toBFloat16Bits(values[i]);
        }
        return new Tensor(bits, shape, null, "bf16", DataType.BFLOAT16);
    }

    private static float[] bf16Values(Tensor tensor) {
        short[] bits = tensor.toBFloat16BitsArrayCopy();
        float[] values = new float[bits.length];
        for (int i = 0; i < bits.length; i++) {
            values[i] = TensorDTypeOps.fromBFloat16Bits(bits[i]);
        }
        return values;
    }

    private static float bf16Round(float value) {
        return TensorDTypeOps.fromBFloat16Bits(TensorDTypeOps.toBFloat16Bits(value));
    }

    private static Tensor bool(byte[] values, int... shape) {
        return new Tensor(values, shape, null, "bool", DataType.BOOL);
    }

    private record NativeInput(Tensor tensor, NativeTensorStorage storage) {
        private NativeInput {
            if (tensor == null) {
                throw new IllegalArgumentException("tensor cannot be null");
            }
            if (storage == null) {
                throw new IllegalArgumentException("storage cannot be null");
            }
        }
    }
}
