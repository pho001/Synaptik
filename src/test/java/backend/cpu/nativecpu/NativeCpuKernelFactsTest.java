package backend.cpu.nativecpu;

import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;

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
    void f32ReluAndSameShapeAddHaveNativeSegmentScalarFacts() {
        NativeCpuKernelFact add = NativeCpuKernelFacts.factFor(Operation.OpType.ADD, DataType.FLOAT32);
        NativeCpuKernelFact relu = NativeCpuKernelFacts.factFor(Operation.OpType.RELU, DataType.FLOAT32);

        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, add.status());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, relu.status());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, add.family());
        assertEquals("requires-dense-contiguous-same-shape", add.reason());
        assertEquals("requires-dense-contiguous", relu.reason());
        assertTrue(add.nativeComputeEligible());
        assertTrue(relu.preservesNativeStorage());
    }

    @Test
    void otherNonBlasElementwiseOpsRemainUnsupportedUntilSegmentKernelsExist() {
        NativeCpuKernelFact mean = NativeCpuKernelFacts.factFor(Operation.OpType.MEAN, DataType.FLOAT32);
        NativeCpuKernelFact log = NativeCpuKernelFacts.factFor(Operation.OpType.LOG, DataType.FLOAT32);

        assertEquals("native-kernel-unsupported:mean", mean.reason());
        assertFalse(mean.nativeComputeEligible());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_UNSUPPORTED, log.status());
        assertFalse(log.preservesNativeStorage());
    }

    @Test
    void metadataOnlyViewsCanPreserveNativeStorageWithoutClaimingCompute() {
        NativeCpuKernelFact reshape = NativeCpuKernelFacts.factFor(Operation.OpType.RESHAPE, DataType.FLOAT32);
        NativeCpuKernelFact squeeze = NativeCpuKernelFacts.factFor(Operation.OpType.SQUEEZE, DataType.BFLOAT16);

        assertEquals(NativeCpuKernelPerformanceStatus.VIEW_ONLY, reshape.status());
        assertEquals(NativeCpuKernelFamily.VIEW_ONLY, reshape.family());
        assertTrue(reshape.preservesNativeStorage());
        assertFalse(reshape.nativeComputeEligible());
        assertEquals("metadata-only-native-view", squeeze.reason());
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
}
