package backend.cpu.nativecpu;

import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;

import java.util.EnumSet;
import java.util.Map;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NativeCpuKernelFactsTest {
    @Test
    void matmulIsCurrentlyTheOnlyNativeComputeProviderFact() {
        NativeCpuKernelFact f32 = NativeCpuKernelFacts.factFor(Operation.OpType.MATMUL, DataType.FLOAT32);
        NativeCpuKernelFact f64 = NativeCpuKernelFacts.factFor(Operation.OpType.MATMUL, DataType.FLOAT64);
        NativeCpuKernelFact bf16 = NativeCpuKernelFacts.factFor(Operation.OpType.MATMUL, DataType.BFLOAT16);

        assertEquals(NativeCpuKernelPerformanceStatus.LIBRARY_PROVIDER, f32.status());
        assertEquals(NativeCpuKernelPerformanceStatus.LIBRARY_PROVIDER, f64.status());
        assertEquals(NativeCpuKernelPerformanceStatus.LIBRARY_PROVIDER, bf16.status());
        assertEquals(NativeCpuKernelFamily.OPENBLAS_NATIVE_SEGMENT, f32.family());
        assertEquals("requires-openblas-native-segment", bf16.reason());
        assertTrue(f32.nativeComputeEligible());
        assertTrue(bf16.preservesNativeStorage());
    }

    @Test
    void softmaxLikeOpsHaveStableNativeCpuArrayFallbackReason() {
        NativeCpuKernelFact softmax = NativeCpuKernelFacts.factFor(Operation.OpType.SOFTMAX, DataType.FLOAT32);
        NativeCpuKernelFact logSoftmax = NativeCpuKernelFacts.factFor(Operation.OpType.LOG_SOFTMAX, DataType.FLOAT64);
        NativeCpuKernelFact bf16Softmax = NativeCpuKernelFacts.factFor(Operation.OpType.SOFTMAX, DataType.BFLOAT16);

        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_UNSUPPORTED, softmax.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_UNSUPPORTED, logSoftmax.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_UNSUPPORTED, bf16Softmax.status());
        assertEquals(NativeCpuKernelFamily.ARRAY_ONLY, softmax.family());
        assertEquals("native-softmax-scalar-loop-slower-than-array", softmax.reason());
        assertEquals("native-softmax-scalar-loop-slower-than-array", logSoftmax.reason());
        assertEquals("native-softmax-scalar-loop-slower-than-array", bf16Softmax.reason());
        assertFalse(softmax.nativeComputeEligible());
        assertFalse(logSoftmax.preservesNativeStorage());
    }

    @Test
    void argMaxHasStableNativeCpuIndexOutputFallbackReason() {
        NativeCpuKernelFact argMax = NativeCpuKernelFacts.factFor(Operation.OpType.ARGMAX, DataType.INT64);
        NativeCpuKernelFact f32MatrixRow = NativeCpuKernelFacts.factFor(Operation.OpType.ARGMAX, DataType.FLOAT32);

        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_UNSUPPORTED, argMax.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_UNSUPPORTED, f32MatrixRow.status());
        assertEquals(NativeCpuKernelFamily.ARRAY_ONLY, argMax.family());
        assertEquals("native-argmax-index-output-unsupported", argMax.reason());
        assertEquals("native-argmax-index-output-unsupported", f32MatrixRow.reason());
        assertFalse(argMax.nativeComputeEligible());
        assertFalse(argMax.preservesNativeStorage());
    }

    @Test
    void f32DenseElementwiseMvpOpsHaveNativeSegmentScalarFacts() {
        NativeCpuKernelFact add = NativeCpuKernelFacts.factFor(Operation.OpType.ADD, DataType.FLOAT32);
        NativeCpuKernelFact sub = NativeCpuKernelFacts.factFor(Operation.OpType.SUB, DataType.FLOAT32);
        NativeCpuKernelFact mul = NativeCpuKernelFacts.factFor(Operation.OpType.MUL, DataType.FLOAT32);
        NativeCpuKernelFact div = NativeCpuKernelFacts.factFor(Operation.OpType.DIV, DataType.FLOAT32);
        NativeCpuKernelFact min = NativeCpuKernelFacts.factFor(Operation.OpType.MIN, DataType.FLOAT32);
        NativeCpuKernelFact max = NativeCpuKernelFacts.factFor(Operation.OpType.MAX, DataType.FLOAT32);
        NativeCpuKernelFact powTensor = NativeCpuKernelFacts.factFor(Operation.OpType.POW_TENSOR, DataType.FLOAT32);
        NativeCpuKernelFact mulScalar = NativeCpuKernelFacts.factFor(Operation.OpType.MUL_SCALAR, DataType.FLOAT32);
        NativeCpuKernelFact neg = NativeCpuKernelFacts.factFor(Operation.OpType.NEG, DataType.FLOAT32);
        NativeCpuKernelFact relu = NativeCpuKernelFacts.factFor(Operation.OpType.RELU, DataType.FLOAT32);
        NativeCpuKernelFact clampMin = NativeCpuKernelFacts.factFor(Operation.OpType.CLAMP_MIN, DataType.FLOAT32);
        NativeCpuKernelFact clampMax = NativeCpuKernelFacts.factFor(Operation.OpType.CLAMP_MAX, DataType.FLOAT32);
        NativeCpuKernelFact log = NativeCpuKernelFacts.factFor(Operation.OpType.LOG, DataType.FLOAT32);
        NativeCpuKernelFact exp = NativeCpuKernelFacts.factFor(Operation.OpType.EXP, DataType.FLOAT32);
        NativeCpuKernelFact fastExp = NativeCpuKernelFacts.factFor(Operation.OpType.FAST_EXP, DataType.FLOAT32);
        NativeCpuKernelFact sqrt = NativeCpuKernelFacts.factFor(Operation.OpType.SQRT, DataType.FLOAT32);
        NativeCpuKernelFact abs = NativeCpuKernelFacts.factFor(Operation.OpType.ABS, DataType.FLOAT32);
        NativeCpuKernelFact floor = NativeCpuKernelFacts.factFor(Operation.OpType.FLOOR, DataType.FLOAT32);
        NativeCpuKernelFact ceil = NativeCpuKernelFacts.factFor(Operation.OpType.CEIL, DataType.FLOAT32);
        NativeCpuKernelFact sign = NativeCpuKernelFacts.factFor(Operation.OpType.SIGN, DataType.FLOAT32);
        NativeCpuKernelFact pow = NativeCpuKernelFacts.factFor(Operation.OpType.POW, DataType.FLOAT32);
        NativeCpuKernelFact tanh = NativeCpuKernelFacts.factFor(Operation.OpType.TANH, DataType.FLOAT32);
        NativeCpuKernelFact fastTanh = NativeCpuKernelFacts.factFor(Operation.OpType.FAST_TANH, DataType.FLOAT32);
        NativeCpuKernelFact sigmoid = NativeCpuKernelFacts.factFor(Operation.OpType.SIGMOID, DataType.FLOAT32);
        NativeCpuKernelFact where = NativeCpuKernelFacts.factFor(Operation.OpType.WHERE, DataType.FLOAT32);

        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, add.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, sub.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, mul.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, div.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, min.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, max.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, powTensor.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, mulScalar.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, neg.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, relu.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, clampMin.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, clampMax.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, log.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, exp.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, fastExp.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, sqrt.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, abs.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, floor.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, ceil.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, sign.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, pow.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, tanh.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, fastTanh.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, sigmoid.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, where.status());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, add.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, sub.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, mul.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, div.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, min.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, max.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, powTensor.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, mulScalar.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, neg.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, log.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, fastExp.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, sqrt.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, floor.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, ceil.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, sign.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, pow.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, sigmoid.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, where.family());
        assertEquals("requires-dense-contiguous-same-shape", add.reason());
        assertEquals("requires-dense-contiguous-same-shape", sub.reason());
        assertEquals("requires-dense-contiguous-same-shape", mul.reason());
        assertEquals("requires-dense-contiguous-same-shape", div.reason());
        assertEquals("requires-dense-contiguous-same-shape", min.reason());
        assertEquals("requires-dense-contiguous-same-shape", max.reason());
        assertEquals("requires-dense-contiguous-same-shape", powTensor.reason());
        assertEquals("requires-dense-contiguous-same-shape", where.reason());
        assertEquals("requires-dense-contiguous", mulScalar.reason());
        assertEquals("requires-dense-contiguous", neg.reason());
        assertEquals("requires-dense-contiguous", relu.reason());
        assertEquals("requires-dense-contiguous", clampMin.reason());
        assertEquals("requires-dense-contiguous", clampMax.reason());
        assertEquals("requires-dense-contiguous", log.reason());
        assertEquals("requires-dense-contiguous", exp.reason());
        assertEquals("requires-dense-contiguous", fastExp.reason());
        assertEquals("requires-dense-contiguous", sqrt.reason());
        assertEquals("requires-dense-contiguous", abs.reason());
        assertEquals("requires-dense-contiguous", floor.reason());
        assertEquals("requires-dense-contiguous", ceil.reason());
        assertEquals("requires-dense-contiguous", sign.reason());
        assertEquals("requires-dense-contiguous", pow.reason());
        assertEquals("requires-dense-contiguous", tanh.reason());
        assertEquals("requires-dense-contiguous", fastTanh.reason());
        assertEquals("requires-dense-contiguous", sigmoid.reason());
        assertTrue(add.nativeComputeEligible());
        assertTrue(sub.nativeComputeEligible());
        assertTrue(mul.nativeComputeEligible());
        assertTrue(div.nativeComputeEligible());
        assertTrue(min.nativeComputeEligible());
        assertTrue(max.nativeComputeEligible());
        assertTrue(powTensor.nativeComputeEligible());
        assertTrue(mulScalar.nativeComputeEligible());
        assertTrue(neg.nativeComputeEligible());
        assertTrue(clampMin.preservesNativeStorage());
        assertTrue(clampMax.preservesNativeStorage());
        assertTrue(log.nativeComputeEligible());
        assertTrue(exp.nativeComputeEligible());
        assertTrue(fastExp.nativeComputeEligible());
        assertTrue(sqrt.nativeComputeEligible());
        assertTrue(abs.nativeComputeEligible());
        assertTrue(floor.nativeComputeEligible());
        assertTrue(ceil.nativeComputeEligible());
        assertTrue(sign.nativeComputeEligible());
        assertTrue(pow.nativeComputeEligible());
        assertTrue(tanh.nativeComputeEligible());
        assertTrue(fastTanh.nativeComputeEligible());
        assertTrue(sigmoid.nativeComputeEligible());
        assertTrue(where.nativeComputeEligible());
        assertTrue(relu.preservesNativeStorage());
        assertTrue(sigmoid.preservesNativeStorage());
    }

    @Test
    void f64DenseArithmeticMvpOpsHaveNativeSegmentScalarFacts() {
        NativeCpuKernelFact add = NativeCpuKernelFacts.factFor(Operation.OpType.ADD, DataType.FLOAT64);
        NativeCpuKernelFact sub = NativeCpuKernelFacts.factFor(Operation.OpType.SUB, DataType.FLOAT64);
        NativeCpuKernelFact mul = NativeCpuKernelFacts.factFor(Operation.OpType.MUL, DataType.FLOAT64);
        NativeCpuKernelFact div = NativeCpuKernelFacts.factFor(Operation.OpType.DIV, DataType.FLOAT64);
        NativeCpuKernelFact min = NativeCpuKernelFacts.factFor(Operation.OpType.MIN, DataType.FLOAT64);
        NativeCpuKernelFact max = NativeCpuKernelFacts.factFor(Operation.OpType.MAX, DataType.FLOAT64);
        NativeCpuKernelFact powTensor = NativeCpuKernelFacts.factFor(Operation.OpType.POW_TENSOR, DataType.FLOAT64);
        NativeCpuKernelFact mulScalar = NativeCpuKernelFacts.factFor(Operation.OpType.MUL_SCALAR, DataType.FLOAT64);
        NativeCpuKernelFact neg = NativeCpuKernelFacts.factFor(Operation.OpType.NEG, DataType.FLOAT64);
        NativeCpuKernelFact relu = NativeCpuKernelFacts.factFor(Operation.OpType.RELU, DataType.FLOAT64);
        NativeCpuKernelFact clampMin = NativeCpuKernelFacts.factFor(Operation.OpType.CLAMP_MIN, DataType.FLOAT64);
        NativeCpuKernelFact clampMax = NativeCpuKernelFacts.factFor(Operation.OpType.CLAMP_MAX, DataType.FLOAT64);
        NativeCpuKernelFact where = NativeCpuKernelFacts.factFor(Operation.OpType.WHERE, DataType.FLOAT64);
        NativeCpuKernelFact log = NativeCpuKernelFacts.factFor(Operation.OpType.LOG, DataType.FLOAT64);
        NativeCpuKernelFact exp = NativeCpuKernelFacts.factFor(Operation.OpType.EXP, DataType.FLOAT64);
        NativeCpuKernelFact sqrt = NativeCpuKernelFacts.factFor(Operation.OpType.SQRT, DataType.FLOAT64);
        NativeCpuKernelFact abs = NativeCpuKernelFacts.factFor(Operation.OpType.ABS, DataType.FLOAT64);
        NativeCpuKernelFact floor = NativeCpuKernelFacts.factFor(Operation.OpType.FLOOR, DataType.FLOAT64);
        NativeCpuKernelFact ceil = NativeCpuKernelFacts.factFor(Operation.OpType.CEIL, DataType.FLOAT64);
        NativeCpuKernelFact sign = NativeCpuKernelFacts.factFor(Operation.OpType.SIGN, DataType.FLOAT64);
        NativeCpuKernelFact pow = NativeCpuKernelFacts.factFor(Operation.OpType.POW, DataType.FLOAT64);
        NativeCpuKernelFact tanh = NativeCpuKernelFacts.factFor(Operation.OpType.TANH, DataType.FLOAT64);
        NativeCpuKernelFact sigmoid = NativeCpuKernelFacts.factFor(Operation.OpType.SIGMOID, DataType.FLOAT64);
        NativeCpuKernelFact inv = NativeCpuKernelFacts.factFor(Operation.OpType.INV, DataType.FLOAT64);

        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, add.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, sub.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, mul.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, div.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, min.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, max.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, powTensor.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, mulScalar.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, neg.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, relu.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, clampMin.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, clampMax.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, where.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, log.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, sigmoid.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, floor.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, ceil.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, sign.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, pow.status());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, add.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, min.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, max.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, powTensor.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, neg.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, where.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, floor.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, ceil.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, sign.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, sigmoid.family());
        assertEquals("requires-dense-contiguous-same-shape", div.reason());
        assertEquals("requires-dense-contiguous-same-shape", min.reason());
        assertEquals("requires-dense-contiguous-same-shape", max.reason());
        assertEquals("requires-dense-contiguous-same-shape", powTensor.reason());
        assertEquals("requires-dense-contiguous-same-shape", where.reason());
        assertEquals("requires-dense-contiguous", mulScalar.reason());
        assertEquals("requires-dense-contiguous", relu.reason());
        assertEquals("requires-dense-contiguous", clampMin.reason());
        assertEquals("requires-dense-contiguous", clampMax.reason());
        assertEquals("requires-dense-contiguous", log.reason());
        assertEquals("requires-dense-contiguous", exp.reason());
        assertEquals("requires-dense-contiguous", floor.reason());
        assertEquals("requires-dense-contiguous", ceil.reason());
        assertEquals("requires-dense-contiguous", sign.reason());
        assertEquals("requires-dense-contiguous", sigmoid.reason());
        assertEquals("requires-dense-contiguous", pow.reason());
        assertTrue(add.nativeComputeEligible());
        assertTrue(min.nativeComputeEligible());
        assertTrue(max.nativeComputeEligible());
        assertTrue(powTensor.nativeComputeEligible());
        assertTrue(neg.preservesNativeStorage());
        assertTrue(relu.nativeComputeEligible());
        assertTrue(clampMin.preservesNativeStorage());
        assertTrue(clampMax.preservesNativeStorage());
        assertTrue(where.preservesNativeStorage());
        assertTrue(log.nativeComputeEligible());
        assertTrue(exp.nativeComputeEligible());
        assertTrue(sqrt.nativeComputeEligible());
        assertTrue(abs.nativeComputeEligible());
        assertTrue(floor.nativeComputeEligible());
        assertTrue(ceil.nativeComputeEligible());
        assertTrue(sign.nativeComputeEligible());
        assertTrue(tanh.nativeComputeEligible());
        assertTrue(sigmoid.preservesNativeStorage());
        assertTrue(pow.nativeComputeEligible());
        assertTrue(inv.preservesNativeStorage());
    }

    @Test
    void otherNonBlasElementwiseOpsRemainUnsupportedUntilSegmentKernelsExist() {
        NativeCpuKernelFact erf = NativeCpuKernelFacts.factFor(Operation.OpType.ERF, DataType.FLOAT32);

        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_UNSUPPORTED, erf.status());
        assertFalse(erf.preservesNativeStorage());
    }

    @Test
    void f32ReductionMvpOpsHaveNativeSegmentScalarFacts() {
        NativeCpuKernelFact sum = NativeCpuKernelFacts.factFor(Operation.OpType.SUM, DataType.FLOAT32);
        NativeCpuKernelFact mean = NativeCpuKernelFacts.factFor(Operation.OpType.MEAN, DataType.FLOAT32);
        NativeCpuKernelFact reduceMin = NativeCpuKernelFacts.factFor(Operation.OpType.REDUCE_MIN, DataType.FLOAT32);
        NativeCpuKernelFact reduceMax = NativeCpuKernelFacts.factFor(Operation.OpType.REDUCE_MAX, DataType.FLOAT32);

        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, sum.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, mean.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, reduceMin.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, reduceMax.status());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, sum.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, mean.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, reduceMin.family());
        assertEquals("requires-dense-contiguous-reduction", sum.reason());
        assertEquals("requires-dense-contiguous-reduction", mean.reason());
        assertEquals("requires-dense-contiguous-reduction", reduceMin.reason());
        assertTrue(sum.nativeComputeEligible());
        assertTrue(mean.preservesNativeStorage());
        assertTrue(reduceMin.nativeComputeEligible());
        assertTrue(reduceMax.preservesNativeStorage());
    }

    @Test
    void f64ReductionMvpOpsHaveNativeSegmentScalarFacts() {
        NativeCpuKernelFact sum = NativeCpuKernelFacts.factFor(Operation.OpType.SUM, DataType.FLOAT64);
        NativeCpuKernelFact mean = NativeCpuKernelFacts.factFor(Operation.OpType.MEAN, DataType.FLOAT64);
        NativeCpuKernelFact reduceMin = NativeCpuKernelFacts.factFor(Operation.OpType.REDUCE_MIN, DataType.FLOAT64);
        NativeCpuKernelFact reduceMax = NativeCpuKernelFacts.factFor(Operation.OpType.REDUCE_MAX, DataType.FLOAT64);

        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, sum.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, mean.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, reduceMin.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, reduceMax.status());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, sum.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, mean.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, reduceMax.family());
        assertEquals("requires-dense-contiguous-reduction", sum.reason());
        assertEquals("requires-dense-contiguous-reduction", mean.reason());
        assertEquals("requires-dense-contiguous-reduction", reduceMax.reason());
        assertTrue(sum.nativeComputeEligible());
        assertTrue(mean.preservesNativeStorage());
        assertTrue(reduceMin.nativeComputeEligible());
        assertTrue(reduceMax.nativeComputeEligible());
    }

    @Test
    void boolMaskOpsHaveNativeSegmentScalarFactsWithoutPublicBoolResidency() {
        NativeCpuKernelFact gt = NativeCpuKernelFacts.factFor(Operation.OpType.GT, DataType.BOOL);
        NativeCpuKernelFact ge = NativeCpuKernelFacts.factFor(Operation.OpType.GE, DataType.BOOL);
        NativeCpuKernelFact lt = NativeCpuKernelFacts.factFor(Operation.OpType.LT, DataType.BOOL);
        NativeCpuKernelFact le = NativeCpuKernelFacts.factFor(Operation.OpType.LE, DataType.BOOL);
        NativeCpuKernelFact eq = NativeCpuKernelFacts.factFor(Operation.OpType.EQ, DataType.BOOL);
        NativeCpuKernelFact ne = NativeCpuKernelFacts.factFor(Operation.OpType.NE, DataType.BOOL);
        NativeCpuKernelFact logicalAnd = NativeCpuKernelFacts.factFor(Operation.OpType.LOGICAL_AND, DataType.BOOL);
        NativeCpuKernelFact logicalNot = NativeCpuKernelFacts.factFor(Operation.OpType.LOGICAL_NOT, DataType.BOOL);
        NativeCpuKernelFact reduceAll = NativeCpuKernelFacts.factFor(Operation.OpType.REDUCE_ALL, DataType.BOOL);
        NativeCpuKernelFact reduceAny = NativeCpuKernelFacts.factFor(Operation.OpType.REDUCE_ANY, DataType.BOOL);
        NativeCpuKernelFact add = NativeCpuKernelFacts.factFor(Operation.OpType.ADD, DataType.BOOL);

        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, gt.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, ge.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, lt.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, le.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, eq.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, ne.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, logicalAnd.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, logicalNot.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, reduceAll.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, reduceAny.status());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, gt.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, logicalAnd.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, reduceAll.family());
        assertEquals("requires-dense-contiguous-compare", gt.reason());
        assertEquals("requires-dense-contiguous-same-shape", logicalAnd.reason());
        assertEquals("requires-dense-contiguous", logicalNot.reason());
        assertEquals("requires-dense-contiguous-reduction", reduceAll.reason());
        assertTrue(gt.nativeComputeEligible());
        assertTrue(logicalAnd.nativeComputeEligible());
        assertTrue(logicalNot.nativeComputeEligible());
        assertTrue(reduceAny.nativeComputeEligible());
        assertFalse(gt.preservesNativeStorage());
        assertFalse(logicalAnd.preservesNativeStorage());
        assertFalse(reduceAll.preservesNativeStorage());
        assertEquals(NativeCpuKernelPerformanceStatus.ARRAY_ONLY, add.status());
        assertFalse(add.nativeComputeEligible());
    }

    @Test
    void f32Bf16CastMvpOpsHaveNativeSegmentScalarFacts() {
        NativeCpuKernelFact f32 = NativeCpuKernelFacts.factFor(Operation.OpType.CAST, DataType.FLOAT32);
        NativeCpuKernelFact bf16 = NativeCpuKernelFacts.factFor(Operation.OpType.CAST, DataType.BFLOAT16);
        NativeCpuKernelFact f64 = NativeCpuKernelFacts.factFor(Operation.OpType.CAST, DataType.FLOAT64);

        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, f32.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, bf16.status());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, f32.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, bf16.family());
        assertEquals("requires-dense-contiguous-cast", f32.reason());
        assertEquals("requires-dense-contiguous-cast", bf16.reason());
        assertTrue(f32.nativeComputeEligible());
        assertTrue(bf16.preservesNativeStorage());
        assertEquals("native-kernel-unsupported:cast", f64.reason());
        assertFalse(f64.nativeComputeEligible());
    }

    @Test
    void bf16PromotedElementwiseMvpOpsHaveNativeSegmentScalarFacts() {
        NativeCpuKernelFact add = NativeCpuKernelFacts.factFor(Operation.OpType.ADD, DataType.BFLOAT16);
        NativeCpuKernelFact sub = NativeCpuKernelFacts.factFor(Operation.OpType.SUB, DataType.BFLOAT16);
        NativeCpuKernelFact mul = NativeCpuKernelFacts.factFor(Operation.OpType.MUL, DataType.BFLOAT16);
        NativeCpuKernelFact div = NativeCpuKernelFacts.factFor(Operation.OpType.DIV, DataType.BFLOAT16);
        NativeCpuKernelFact min = NativeCpuKernelFacts.factFor(Operation.OpType.MIN, DataType.BFLOAT16);
        NativeCpuKernelFact max = NativeCpuKernelFacts.factFor(Operation.OpType.MAX, DataType.BFLOAT16);
        NativeCpuKernelFact mulScalar = NativeCpuKernelFacts.factFor(Operation.OpType.MUL_SCALAR, DataType.BFLOAT16);
        NativeCpuKernelFact neg = NativeCpuKernelFacts.factFor(Operation.OpType.NEG, DataType.BFLOAT16);
        NativeCpuKernelFact relu = NativeCpuKernelFacts.factFor(Operation.OpType.RELU, DataType.BFLOAT16);
        NativeCpuKernelFact clampMin = NativeCpuKernelFacts.factFor(Operation.OpType.CLAMP_MIN, DataType.BFLOAT16);
        NativeCpuKernelFact clampMax = NativeCpuKernelFacts.factFor(Operation.OpType.CLAMP_MAX, DataType.BFLOAT16);
        NativeCpuKernelFact abs = NativeCpuKernelFacts.factFor(Operation.OpType.ABS, DataType.BFLOAT16);
        NativeCpuKernelFact log = NativeCpuKernelFacts.factFor(Operation.OpType.LOG, DataType.BFLOAT16);
        NativeCpuKernelFact floor = NativeCpuKernelFacts.factFor(Operation.OpType.FLOOR, DataType.BFLOAT16);
        NativeCpuKernelFact ceil = NativeCpuKernelFacts.factFor(Operation.OpType.CEIL, DataType.BFLOAT16);
        NativeCpuKernelFact sign = NativeCpuKernelFacts.factFor(Operation.OpType.SIGN, DataType.BFLOAT16);
        NativeCpuKernelFact pow = NativeCpuKernelFacts.factFor(Operation.OpType.POW, DataType.BFLOAT16);
        NativeCpuKernelFact powTensor = NativeCpuKernelFacts.factFor(Operation.OpType.POW_TENSOR, DataType.BFLOAT16);
        NativeCpuKernelFact sum = NativeCpuKernelFacts.factFor(Operation.OpType.SUM, DataType.BFLOAT16);
        NativeCpuKernelFact mean = NativeCpuKernelFacts.factFor(Operation.OpType.MEAN, DataType.BFLOAT16);
        NativeCpuKernelFact reduceMin = NativeCpuKernelFacts.factFor(Operation.OpType.REDUCE_MIN, DataType.BFLOAT16);
        NativeCpuKernelFact reduceMax = NativeCpuKernelFacts.factFor(Operation.OpType.REDUCE_MAX, DataType.BFLOAT16);
        NativeCpuKernelFact where = NativeCpuKernelFacts.factFor(Operation.OpType.WHERE, DataType.BFLOAT16);

        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, add.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, sub.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, mul.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, div.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, min.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, max.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, mulScalar.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, neg.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, relu.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, clampMin.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, clampMax.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, abs.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, sum.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, mean.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, where.status());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, add.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, min.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, max.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, abs.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, sum.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, mean.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, where.family());
        assertEquals("requires-dense-contiguous-same-shape", add.reason());
        assertEquals("requires-dense-contiguous-same-shape", sub.reason());
        assertEquals("requires-dense-contiguous-same-shape", mul.reason());
        assertEquals("requires-dense-contiguous-same-shape", div.reason());
        assertEquals("requires-dense-contiguous-same-shape", min.reason());
        assertEquals("requires-dense-contiguous-same-shape", max.reason());
        assertEquals("requires-dense-contiguous", mulScalar.reason());
        assertEquals("requires-dense-contiguous", neg.reason());
        assertEquals("requires-dense-contiguous", relu.reason());
        assertEquals("requires-dense-contiguous", clampMin.reason());
        assertEquals("requires-dense-contiguous", clampMax.reason());
        assertEquals("requires-dense-contiguous", abs.reason());
        assertEquals("requires-dense-contiguous-reduction", sum.reason());
        assertEquals("requires-dense-contiguous-reduction", mean.reason());
        assertEquals("requires-dense-contiguous-same-shape", where.reason());
        assertTrue(add.nativeComputeEligible());
        assertTrue(min.nativeComputeEligible());
        assertTrue(max.nativeComputeEligible());
        assertTrue(clampMin.preservesNativeStorage());
        assertTrue(clampMax.preservesNativeStorage());
        assertTrue(abs.preservesNativeStorage());
        assertTrue(sum.nativeComputeEligible());
        assertTrue(mean.preservesNativeStorage());
        assertEquals("native-kernel-unsupported:log", log.reason());
        assertEquals("native-kernel-unsupported:floor", floor.reason());
        assertEquals("native-kernel-unsupported:ceil", ceil.reason());
        assertEquals("native-kernel-unsupported:sign", sign.reason());
        assertEquals("native-kernel-unsupported:pow", pow.reason());
        assertEquals("native-kernel-unsupported:pow_tensor", powTensor.reason());
        assertEquals("native-bf16-reduce-minmax-output-policy-unsupported", reduceMin.reason());
        assertEquals("native-bf16-reduce-minmax-output-policy-unsupported", reduceMax.reason());
        assertFalse(log.nativeComputeEligible());
        assertFalse(floor.nativeComputeEligible());
        assertFalse(ceil.nativeComputeEligible());
        assertFalse(sign.nativeComputeEligible());
        assertFalse(pow.nativeComputeEligible());
        assertFalse(powTensor.nativeComputeEligible());
        assertFalse(reduceMin.nativeComputeEligible());
        assertFalse(reduceMax.nativeComputeEligible());
        assertTrue(where.nativeComputeEligible());
        assertTrue(where.preservesNativeStorage());
    }

    @Test
    void metadataOnlyViewsCanPreserveNativeStorageWithoutClaimingCompute() {
        NativeCpuKernelFact noop = NativeCpuKernelFacts.factFor(Operation.OpType.NOOP, DataType.FLOAT64);
        NativeCpuKernelFact reshape = NativeCpuKernelFacts.factFor(Operation.OpType.RESHAPE, DataType.FLOAT32);
        NativeCpuKernelFact permute = NativeCpuKernelFacts.factFor(Operation.OpType.PERMUTE, DataType.FLOAT32);
        NativeCpuKernelFact expand = NativeCpuKernelFacts.factFor(Operation.OpType.EXPAND, DataType.FLOAT32);
        NativeCpuKernelFact select = NativeCpuKernelFacts.factFor(Operation.OpType.SELECT, DataType.FLOAT32);
        NativeCpuKernelFact slice = NativeCpuKernelFacts.factFor(Operation.OpType.SLICE, DataType.FLOAT32);
        NativeCpuKernelFact expandDims = NativeCpuKernelFacts.factFor(Operation.OpType.EXPAND_DIMS, DataType.FLOAT32);
        NativeCpuKernelFact squeeze = NativeCpuKernelFacts.factFor(Operation.OpType.SQUEEZE, DataType.BFLOAT16);

        assertEquals(NativeCpuKernelPerformanceStatus.VIEW_ONLY, noop.status());
        assertEquals(NativeCpuKernelFamily.VIEW_ONLY, noop.family());
        assertEquals(NativeCpuKernelPerformanceStatus.VIEW_ONLY, reshape.status());
        assertEquals(NativeCpuKernelFamily.VIEW_ONLY, reshape.family());
        assertEquals(NativeCpuKernelPerformanceStatus.VIEW_ONLY, permute.status());
        assertEquals(NativeCpuKernelFamily.VIEW_ONLY, permute.family());
        assertEquals(NativeCpuKernelPerformanceStatus.VIEW_ONLY, expand.status());
        assertEquals(NativeCpuKernelFamily.VIEW_ONLY, expand.family());
        assertEquals(NativeCpuKernelPerformanceStatus.VIEW_ONLY, select.status());
        assertEquals(NativeCpuKernelFamily.VIEW_ONLY, select.family());
        assertEquals(NativeCpuKernelPerformanceStatus.VIEW_ONLY, slice.status());
        assertEquals(NativeCpuKernelFamily.VIEW_ONLY, slice.family());
        assertEquals(NativeCpuKernelPerformanceStatus.VIEW_ONLY, expandDims.status());
        assertEquals(NativeCpuKernelFamily.VIEW_ONLY, expandDims.family());
        assertTrue(reshape.preservesNativeStorage());
        assertTrue(permute.preservesNativeStorage());
        assertTrue(expand.preservesNativeStorage());
        assertTrue(select.preservesNativeStorage());
        assertTrue(slice.preservesNativeStorage());
        assertTrue(expandDims.preservesNativeStorage());
        assertFalse(reshape.nativeComputeEligible());
        assertEquals("metadata-only-native-view", squeeze.reason());
    }

    @Test
    void contiguousMaterializationHasNativeCopyFact() {
        NativeCpuKernelFact f32 = NativeCpuKernelFacts.factFor(Operation.OpType.CONTIGUOUS, DataType.FLOAT32);
        NativeCpuKernelFact f64 = NativeCpuKernelFacts.factFor(Operation.OpType.CONTIGUOUS, DataType.FLOAT64);
        NativeCpuKernelFact bf16 = NativeCpuKernelFacts.factFor(Operation.OpType.CONTIGUOUS, DataType.BFLOAT16);
        NativeCpuKernelFact bool = NativeCpuKernelFacts.factFor(Operation.OpType.CONTIGUOUS, DataType.BOOL);

        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, f32.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, f64.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, bf16.status());
        assertEquals(NativeCpuKernelFamily.NATIVE_MICROKERNEL, f32.family());
        assertEquals(NativeCpuKernelFamily.NATIVE_MICROKERNEL, bf16.family());
        assertEquals("requires-dense-contiguous-copy", f32.reason());
        assertEquals("requires-dense-contiguous-copy", bf16.reason());
        assertTrue(f32.nativeComputeEligible());
        assertTrue(bf16.preservesNativeStorage());
        assertEquals(NativeCpuKernelPerformanceStatus.ARRAY_ONLY, bool.status());
        assertFalse(bool.nativeComputeEligible());
    }

    @Test
    void materializingLayoutOpsAreNotNativeViewOnlyFactsYet() {
        NativeCpuKernelFact concat = NativeCpuKernelFacts.factFor(Operation.OpType.CONCAT, DataType.FLOAT32);
        NativeCpuKernelFact pad = NativeCpuKernelFacts.factFor(Operation.OpType.PAD, DataType.FLOAT32);
        NativeCpuKernelFact tile = NativeCpuKernelFacts.factFor(Operation.OpType.TILE, DataType.FLOAT32);

        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_UNSUPPORTED, concat.status());
        assertEquals(NativeCpuKernelFamily.ARRAY_ONLY, concat.family());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_UNSUPPORTED, pad.status());
        assertEquals(NativeCpuKernelFamily.ARRAY_ONLY, pad.family());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_UNSUPPORTED, tile.status());
        assertEquals(NativeCpuKernelFamily.ARRAY_ONLY, tile.family());
        assertFalse(concat.preservesNativeStorage());
        assertFalse(pad.preservesNativeStorage());
        assertFalse(tile.preservesNativeStorage());
    }

    @Test
    void unsupportedNativeStorageDtypesAreArrayOnlyFacts() {
        NativeCpuKernelFact boolAdd = NativeCpuKernelFacts.factFor(Operation.OpType.ADD, DataType.BOOL);
        NativeCpuKernelFact intMatmul = NativeCpuKernelFacts.factFor(Operation.OpType.MATMUL, DataType.INT32);

        assertEquals(NativeCpuKernelPerformanceStatus.ARRAY_ONLY, boolAdd.status());
        assertEquals(NativeCpuKernelFamily.ARRAY_ONLY, intMatmul.family());
        assertEquals("native-storage-dtype-unsupported:bool", boolAdd.reason());
        assertFalse(intMatmul.preservesNativeStorage());
    }

    @Test
    void everyOperationHasOneStableFactPerDtype() {
        Map<Operation.OpType, NativeCpuKernelFact> facts = NativeCpuKernelFacts.factsFor(DataType.FLOAT32).stream()
                .collect(Collectors.toMap(NativeCpuKernelFact::opType, fact -> fact));

        assertEquals(Operation.OpType.values().length, facts.size());
        assertTrue(facts.containsKey(Operation.OpType.ADD));
        assertTrue(facts.containsKey(Operation.OpType.MATMUL));
        assertTrue(facts.containsKey(Operation.OpType.UNKNOWN));
        assertEquals("native-kernel-unknown-op", facts.get(Operation.OpType.UNKNOWN).reason());
    }

    @Test
    void segmentScalarFactsRemainCorrectnessFallbacksUntilFastFamiliesExist() {
        EnumSet<NativeCpuKernelFamily> fastFamilies = EnumSet.of(
                NativeCpuKernelFamily.VECTOR_API,
                NativeCpuKernelFamily.GENERATED_DIRECT,
                NativeCpuKernelFamily.NATIVE_MICROKERNEL
        );

        for (DataType dataType : DataType.values()) {
            for (Operation.OpType opType : Operation.OpType.values()) {
                NativeCpuKernelFact fact = NativeCpuKernelFacts.factFor(opType, dataType);

                if (fact.family() == NativeCpuKernelFamily.SEGMENT_SCALAR) {
                    assertEquals(
                            NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW,
                            fact.status(),
                            dataType + " " + opType + " segment scalar must stay out of AUTO fast eligibility"
                    );
                }
                if (fact.status() == NativeCpuKernelPerformanceStatus.NATIVE_FAST) {
                    assertTrue(
                            fastFamilies.contains(fact.family()),
                            dataType + " " + opType + " cannot claim NATIVE_FAST from " + fact.family()
                    );
                }
            }
        }
    }
}
