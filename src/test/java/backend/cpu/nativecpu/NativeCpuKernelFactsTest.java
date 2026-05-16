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
    void f32DenseElementwiseMvpOpsHaveNativeSegmentScalarFacts() {
        NativeCpuKernelFact add = NativeCpuKernelFacts.factFor(Operation.OpType.ADD, DataType.FLOAT32);
        NativeCpuKernelFact sub = NativeCpuKernelFacts.factFor(Operation.OpType.SUB, DataType.FLOAT32);
        NativeCpuKernelFact mul = NativeCpuKernelFacts.factFor(Operation.OpType.MUL, DataType.FLOAT32);
        NativeCpuKernelFact div = NativeCpuKernelFacts.factFor(Operation.OpType.DIV, DataType.FLOAT32);
        NativeCpuKernelFact mulScalar = NativeCpuKernelFacts.factFor(Operation.OpType.MUL_SCALAR, DataType.FLOAT32);
        NativeCpuKernelFact neg = NativeCpuKernelFacts.factFor(Operation.OpType.NEG, DataType.FLOAT32);
        NativeCpuKernelFact relu = NativeCpuKernelFacts.factFor(Operation.OpType.RELU, DataType.FLOAT32);
        NativeCpuKernelFact log = NativeCpuKernelFacts.factFor(Operation.OpType.LOG, DataType.FLOAT32);
        NativeCpuKernelFact exp = NativeCpuKernelFacts.factFor(Operation.OpType.EXP, DataType.FLOAT32);
        NativeCpuKernelFact fastExp = NativeCpuKernelFacts.factFor(Operation.OpType.FAST_EXP, DataType.FLOAT32);
        NativeCpuKernelFact sqrt = NativeCpuKernelFacts.factFor(Operation.OpType.SQRT, DataType.FLOAT32);
        NativeCpuKernelFact abs = NativeCpuKernelFacts.factFor(Operation.OpType.ABS, DataType.FLOAT32);
        NativeCpuKernelFact tanh = NativeCpuKernelFacts.factFor(Operation.OpType.TANH, DataType.FLOAT32);
        NativeCpuKernelFact fastTanh = NativeCpuKernelFacts.factFor(Operation.OpType.FAST_TANH, DataType.FLOAT32);
        NativeCpuKernelFact sigmoid = NativeCpuKernelFacts.factFor(Operation.OpType.SIGMOID, DataType.FLOAT32);
        NativeCpuKernelFact where = NativeCpuKernelFacts.factFor(Operation.OpType.WHERE, DataType.FLOAT32);

        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, add.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, sub.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, mul.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, div.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, mulScalar.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, neg.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, relu.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, log.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, exp.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, fastExp.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, sqrt.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, abs.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, tanh.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, fastTanh.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, sigmoid.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, where.status());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, add.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, sub.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, mul.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, div.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, mulScalar.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, neg.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, log.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, fastExp.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, sqrt.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, sigmoid.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, where.family());
        assertEquals("requires-dense-contiguous-same-shape", add.reason());
        assertEquals("requires-dense-contiguous-same-shape", sub.reason());
        assertEquals("requires-dense-contiguous-same-shape", mul.reason());
        assertEquals("requires-dense-contiguous-same-shape", div.reason());
        assertEquals("requires-dense-contiguous-same-shape", where.reason());
        assertEquals("requires-dense-contiguous", mulScalar.reason());
        assertEquals("requires-dense-contiguous", neg.reason());
        assertEquals("requires-dense-contiguous", relu.reason());
        assertEquals("requires-dense-contiguous", log.reason());
        assertEquals("requires-dense-contiguous", exp.reason());
        assertEquals("requires-dense-contiguous", fastExp.reason());
        assertEquals("requires-dense-contiguous", sqrt.reason());
        assertEquals("requires-dense-contiguous", abs.reason());
        assertEquals("requires-dense-contiguous", tanh.reason());
        assertEquals("requires-dense-contiguous", fastTanh.reason());
        assertEquals("requires-dense-contiguous", sigmoid.reason());
        assertTrue(add.nativeComputeEligible());
        assertTrue(sub.nativeComputeEligible());
        assertTrue(mul.nativeComputeEligible());
        assertTrue(div.nativeComputeEligible());
        assertTrue(mulScalar.nativeComputeEligible());
        assertTrue(neg.nativeComputeEligible());
        assertTrue(log.nativeComputeEligible());
        assertTrue(exp.nativeComputeEligible());
        assertTrue(fastExp.nativeComputeEligible());
        assertTrue(sqrt.nativeComputeEligible());
        assertTrue(abs.nativeComputeEligible());
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
        NativeCpuKernelFact mulScalar = NativeCpuKernelFacts.factFor(Operation.OpType.MUL_SCALAR, DataType.FLOAT64);
        NativeCpuKernelFact neg = NativeCpuKernelFacts.factFor(Operation.OpType.NEG, DataType.FLOAT64);
        NativeCpuKernelFact relu = NativeCpuKernelFacts.factFor(Operation.OpType.RELU, DataType.FLOAT64);
        NativeCpuKernelFact where = NativeCpuKernelFacts.factFor(Operation.OpType.WHERE, DataType.FLOAT64);
        NativeCpuKernelFact log = NativeCpuKernelFacts.factFor(Operation.OpType.LOG, DataType.FLOAT64);
        NativeCpuKernelFact exp = NativeCpuKernelFacts.factFor(Operation.OpType.EXP, DataType.FLOAT64);
        NativeCpuKernelFact sigmoid = NativeCpuKernelFacts.factFor(Operation.OpType.SIGMOID, DataType.FLOAT64);

        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, add.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, sub.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, mul.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, div.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, mulScalar.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, neg.status());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, add.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, neg.family());
        assertEquals("requires-dense-contiguous-same-shape", div.reason());
        assertEquals("requires-dense-contiguous", mulScalar.reason());
        assertTrue(add.nativeComputeEligible());
        assertTrue(neg.preservesNativeStorage());
        assertEquals("native-kernel-unsupported:relu", relu.reason());
        assertEquals("native-kernel-unsupported:where", where.reason());
        assertEquals("native-kernel-unsupported:log", log.reason());
        assertEquals("native-kernel-unsupported:exp", exp.reason());
        assertEquals("native-kernel-unsupported:sigmoid", sigmoid.reason());
        assertFalse(relu.nativeComputeEligible());
        assertFalse(where.preservesNativeStorage());
        assertFalse(log.nativeComputeEligible());
        assertFalse(exp.nativeComputeEligible());
        assertFalse(sigmoid.preservesNativeStorage());
    }

    @Test
    void otherNonBlasElementwiseOpsRemainUnsupportedUntilSegmentKernelsExist() {
        NativeCpuKernelFact reduceMin = NativeCpuKernelFacts.factFor(Operation.OpType.REDUCE_MIN, DataType.FLOAT32);
        NativeCpuKernelFact erf = NativeCpuKernelFacts.factFor(Operation.OpType.ERF, DataType.FLOAT32);

        assertEquals("native-kernel-unsupported:reduce_min", reduceMin.reason());
        assertFalse(reduceMin.nativeComputeEligible());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_UNSUPPORTED, erf.status());
        assertFalse(erf.preservesNativeStorage());
    }

    @Test
    void f32AllReductionMvpOpsHaveNativeSegmentScalarFacts() {
        NativeCpuKernelFact sum = NativeCpuKernelFacts.factFor(Operation.OpType.SUM, DataType.FLOAT32);
        NativeCpuKernelFact mean = NativeCpuKernelFacts.factFor(Operation.OpType.MEAN, DataType.FLOAT32);

        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, sum.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, mean.status());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, sum.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, mean.family());
        assertEquals("requires-dense-contiguous-reduction", sum.reason());
        assertEquals("requires-dense-contiguous-reduction", mean.reason());
        assertTrue(sum.nativeComputeEligible());
        assertTrue(mean.preservesNativeStorage());
    }

    @Test
    void f64ReductionMvpOpsHaveNativeSegmentScalarFacts() {
        NativeCpuKernelFact sum = NativeCpuKernelFacts.factFor(Operation.OpType.SUM, DataType.FLOAT64);
        NativeCpuKernelFact mean = NativeCpuKernelFacts.factFor(Operation.OpType.MEAN, DataType.FLOAT64);
        NativeCpuKernelFact reduceMax = NativeCpuKernelFacts.factFor(Operation.OpType.REDUCE_MAX, DataType.FLOAT64);

        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, sum.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, mean.status());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, sum.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, mean.family());
        assertEquals("requires-dense-contiguous-reduction", sum.reason());
        assertEquals("requires-dense-contiguous-reduction", mean.reason());
        assertTrue(sum.nativeComputeEligible());
        assertTrue(mean.preservesNativeStorage());
        assertEquals("native-kernel-unsupported:reduce_max", reduceMax.reason());
        assertFalse(reduceMax.nativeComputeEligible());
    }

    @Test
    void boolCompareMvpOpsHaveNativeSegmentScalarFactsWithoutNativeBoolResidency() {
        NativeCpuKernelFact gt = NativeCpuKernelFacts.factFor(Operation.OpType.GT, DataType.BOOL);
        NativeCpuKernelFact ge = NativeCpuKernelFacts.factFor(Operation.OpType.GE, DataType.BOOL);
        NativeCpuKernelFact lt = NativeCpuKernelFacts.factFor(Operation.OpType.LT, DataType.BOOL);
        NativeCpuKernelFact le = NativeCpuKernelFacts.factFor(Operation.OpType.LE, DataType.BOOL);
        NativeCpuKernelFact eq = NativeCpuKernelFacts.factFor(Operation.OpType.EQ, DataType.BOOL);
        NativeCpuKernelFact ne = NativeCpuKernelFacts.factFor(Operation.OpType.NE, DataType.BOOL);
        NativeCpuKernelFact add = NativeCpuKernelFacts.factFor(Operation.OpType.ADD, DataType.BOOL);

        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, gt.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, ge.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, lt.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, le.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, eq.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, ne.status());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, gt.family());
        assertEquals("requires-dense-contiguous-compare", gt.reason());
        assertTrue(gt.nativeComputeEligible());
        assertFalse(gt.preservesNativeStorage());
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
        NativeCpuKernelFact mulScalar = NativeCpuKernelFacts.factFor(Operation.OpType.MUL_SCALAR, DataType.BFLOAT16);
        NativeCpuKernelFact neg = NativeCpuKernelFacts.factFor(Operation.OpType.NEG, DataType.BFLOAT16);
        NativeCpuKernelFact relu = NativeCpuKernelFacts.factFor(Operation.OpType.RELU, DataType.BFLOAT16);
        NativeCpuKernelFact abs = NativeCpuKernelFacts.factFor(Operation.OpType.ABS, DataType.BFLOAT16);
        NativeCpuKernelFact log = NativeCpuKernelFacts.factFor(Operation.OpType.LOG, DataType.BFLOAT16);
        NativeCpuKernelFact sum = NativeCpuKernelFacts.factFor(Operation.OpType.SUM, DataType.BFLOAT16);
        NativeCpuKernelFact where = NativeCpuKernelFacts.factFor(Operation.OpType.WHERE, DataType.BFLOAT16);

        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, add.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, sub.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, mul.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, div.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, mulScalar.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, neg.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, relu.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, abs.status());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, add.family());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, abs.family());
        assertEquals("requires-dense-contiguous-same-shape", add.reason());
        assertEquals("requires-dense-contiguous-same-shape", sub.reason());
        assertEquals("requires-dense-contiguous-same-shape", mul.reason());
        assertEquals("requires-dense-contiguous-same-shape", div.reason());
        assertEquals("requires-dense-contiguous", mulScalar.reason());
        assertEquals("requires-dense-contiguous", neg.reason());
        assertEquals("requires-dense-contiguous", relu.reason());
        assertEquals("requires-dense-contiguous", abs.reason());
        assertTrue(add.nativeComputeEligible());
        assertTrue(abs.preservesNativeStorage());
        assertEquals("native-kernel-unsupported:log", log.reason());
        assertEquals("native-kernel-unsupported:sum", sum.reason());
        assertEquals("native-kernel-unsupported:where", where.reason());
        assertFalse(log.nativeComputeEligible());
        assertFalse(sum.nativeComputeEligible());
        assertFalse(where.preservesNativeStorage());
    }

    @Test
    void metadataOnlyViewsCanPreserveNativeStorageWithoutClaimingCompute() {
        NativeCpuKernelFact noop = NativeCpuKernelFacts.factFor(Operation.OpType.NOOP, DataType.FLOAT64);
        NativeCpuKernelFact reshape = NativeCpuKernelFacts.factFor(Operation.OpType.RESHAPE, DataType.FLOAT32);
        NativeCpuKernelFact expandDims = NativeCpuKernelFacts.factFor(Operation.OpType.EXPAND_DIMS, DataType.FLOAT32);
        NativeCpuKernelFact squeeze = NativeCpuKernelFacts.factFor(Operation.OpType.SQUEEZE, DataType.BFLOAT16);

        assertEquals(NativeCpuKernelPerformanceStatus.VIEW_ONLY, noop.status());
        assertEquals(NativeCpuKernelFamily.VIEW_ONLY, noop.family());
        assertEquals(NativeCpuKernelPerformanceStatus.VIEW_ONLY, reshape.status());
        assertEquals(NativeCpuKernelFamily.VIEW_ONLY, reshape.family());
        assertEquals(NativeCpuKernelPerformanceStatus.VIEW_ONLY, expandDims.status());
        assertEquals(NativeCpuKernelFamily.VIEW_ONLY, expandDims.family());
        assertTrue(reshape.preservesNativeStorage());
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
    void nonShapeOnlyViewsAreNotNativeViewOnlyFactsYet() {
        NativeCpuKernelFact select = NativeCpuKernelFacts.factFor(Operation.OpType.SELECT, DataType.FLOAT32);
        NativeCpuKernelFact slice = NativeCpuKernelFacts.factFor(Operation.OpType.SLICE, DataType.FLOAT32);
        NativeCpuKernelFact permute = NativeCpuKernelFacts.factFor(Operation.OpType.PERMUTE, DataType.FLOAT32);
        NativeCpuKernelFact expand = NativeCpuKernelFacts.factFor(Operation.OpType.EXPAND, DataType.FLOAT32);

        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_UNSUPPORTED, select.status());
        assertEquals(NativeCpuKernelFamily.ARRAY_ONLY, select.family());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_UNSUPPORTED, slice.status());
        assertEquals(NativeCpuKernelFamily.ARRAY_ONLY, slice.family());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_UNSUPPORTED, permute.status());
        assertEquals(NativeCpuKernelFamily.ARRAY_ONLY, permute.family());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_UNSUPPORTED, expand.status());
        assertEquals(NativeCpuKernelFamily.ARRAY_ONLY, expand.family());
        assertFalse(select.preservesNativeStorage());
        assertFalse(slice.preservesNativeStorage());
        assertFalse(permute.preservesNativeStorage());
        assertFalse(expand.preservesNativeStorage());
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
