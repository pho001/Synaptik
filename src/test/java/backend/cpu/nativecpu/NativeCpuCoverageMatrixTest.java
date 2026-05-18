package backend.cpu.nativecpu;

import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;

import java.util.Map;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NativeCpuCoverageMatrixTest {
    @Test
    void matrixContainsOneEntryForEveryOperationAndDtype() {
        assertEquals(
                Operation.OpType.values().length * DataType.values().length,
                NativeCpuCoverageMatrix.entries().size()
        );

        Map<String, NativeCpuCoverageEntry> byKey = NativeCpuCoverageMatrix.entries().stream()
                .collect(Collectors.toMap(
                        entry -> entry.opType().name() + ":" + entry.dataType().name(),
                        entry -> entry
                ));
        assertEquals(NativeCpuCoverageMatrix.entries().size(), byKey.size());
        assertTrue(byKey.containsKey(Operation.OpType.MATMUL.name() + ":" + DataType.FLOAT32.name()));
        assertTrue(byKey.containsKey(Operation.OpType.UNKNOWN.name() + ":" + DataType.BFLOAT16.name()));
    }

    @Test
    void matrixRowsAreDerivedFromKernelFacts() {
        for (NativeCpuCoverageEntry entry : NativeCpuCoverageMatrix.entries()) {
            NativeCpuKernelFact fact = NativeCpuKernelFacts.factFor(entry.opType(), entry.dataType());

            assertEquals(fact.status(), entry.status(), entry.toString());
            assertEquals(fact.family(), entry.family(), entry.toString());
            assertEquals(fact.nativeComputeEligible() || fact.status() == NativeCpuKernelPerformanceStatus.VIEW_ONLY,
                    entry.nativeSupported(), entry.toString());
            assertEquals(fact.preservesNativeStorage(), entry.preservesNativeStorage(), entry.toString());
            if (entry.nativeSupported()) {
                assertEquals("", entry.fallbackReason(), entry.toString());
            } else {
                assertEquals(fact.reason(), entry.fallbackReason(), entry.toString());
            }
        }
    }

    @Test
    void providerElementwiseReductionCastAndContiguousRowsExposeExpectedCoverage() {
        NativeCpuCoverageEntry matmul = NativeCpuCoverageMatrix.entryFor(Operation.OpType.MATMUL, DataType.FLOAT32);
        NativeCpuCoverageEntry add = NativeCpuCoverageMatrix.entryFor(Operation.OpType.ADD, DataType.FLOAT32);
        NativeCpuCoverageEntry min = NativeCpuCoverageMatrix.entryFor(Operation.OpType.MIN, DataType.FLOAT32);
        NativeCpuCoverageEntry floor = NativeCpuCoverageMatrix.entryFor(Operation.OpType.FLOOR, DataType.FLOAT32);
        NativeCpuCoverageEntry pow = NativeCpuCoverageMatrix.entryFor(Operation.OpType.POW, DataType.FLOAT32);
        NativeCpuCoverageEntry powTensor = NativeCpuCoverageMatrix.entryFor(Operation.OpType.POW_TENSOR, DataType.FLOAT64);
        NativeCpuCoverageEntry clampMin = NativeCpuCoverageMatrix.entryFor(Operation.OpType.CLAMP_MIN, DataType.BFLOAT16);
        NativeCpuCoverageEntry bf16Relu = NativeCpuCoverageMatrix.entryFor(Operation.OpType.RELU, DataType.BFLOAT16);
        NativeCpuCoverageEntry sum = NativeCpuCoverageMatrix.entryFor(Operation.OpType.SUM, DataType.FLOAT64);
        NativeCpuCoverageEntry cast = NativeCpuCoverageMatrix.entryFor(Operation.OpType.CAST, DataType.BFLOAT16);
        NativeCpuCoverageEntry contiguous = NativeCpuCoverageMatrix.entryFor(Operation.OpType.CONTIGUOUS, DataType.FLOAT32);

        assertEquals(NativeCpuKernelPerformanceStatus.LIBRARY_PROVIDER, matmul.status());
        assertEquals(NativeCpuKernelFamily.OPENBLAS_NATIVE_SEGMENT, matmul.family());
        assertEquals(NativeCpuCoverageLayoutScope.DENSE_CONTIGUOUS, matmul.layoutScope());
        assertTrue(matmul.nativeSupported());
        assertTrue(matmul.autoFastEligible());

        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, add.status());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, add.family());
        assertEquals(NativeCpuCoverageLayoutScope.DENSE_CONTIGUOUS, add.layoutScope());
        assertTrue(add.nativeSupported());
        assertFalse(add.autoFastEligible());

        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, min.status());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, min.family());
        assertEquals(NativeCpuCoverageLayoutScope.DENSE_CONTIGUOUS, min.layoutScope());
        assertTrue(min.nativeSupported());
        assertFalse(min.autoFastEligible());

        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, pow.status());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, pow.family());
        assertEquals(NativeCpuCoverageLayoutScope.DENSE_CONTIGUOUS, pow.layoutScope());
        assertTrue(pow.nativeSupported());
        assertFalse(pow.autoFastEligible());

        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, floor.status());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, floor.family());
        assertTrue(floor.nativeSupported());
        assertFalse(floor.autoFastEligible());

        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, powTensor.status());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, powTensor.family());
        assertTrue(powTensor.nativeSupported());
        assertFalse(powTensor.autoFastEligible());

        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, clampMin.status());
        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, clampMin.family());
        assertTrue(clampMin.preservesNativeStorage());

        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, bf16Relu.family());
        assertEquals(NativeCpuCoverageLayoutScope.DENSE_CONTIGUOUS, bf16Relu.layoutScope());
        assertTrue(bf16Relu.preservesNativeStorage());

        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, sum.family());
        assertEquals(NativeCpuCoverageLayoutScope.DENSE_CONTIGUOUS, sum.layoutScope());

        assertEquals(NativeCpuKernelFamily.SEGMENT_SCALAR, cast.family());
        assertEquals(NativeCpuCoverageLayoutScope.DENSE_CONTIGUOUS, cast.layoutScope());

        assertEquals(NativeCpuKernelFamily.NATIVE_MICROKERNEL, contiguous.family());
        assertEquals(NativeCpuCoverageLayoutScope.DENSE_CONTIGUOUS, contiguous.layoutScope());
        assertTrue(contiguous.nativeSupported());
        assertFalse(contiguous.autoFastEligible());
    }

    @Test
    void metadataOnlyViewsAreNativeSupportedButNotNativeCompute() {
        NativeCpuCoverageEntry reshape = NativeCpuCoverageMatrix.entryFor(Operation.OpType.RESHAPE, DataType.FLOAT32);
        NativeCpuCoverageEntry permute = NativeCpuCoverageMatrix.entryFor(Operation.OpType.PERMUTE, DataType.FLOAT32);
        NativeCpuCoverageEntry select = NativeCpuCoverageMatrix.entryFor(Operation.OpType.SELECT, DataType.FLOAT32);
        NativeCpuCoverageEntry expand = NativeCpuCoverageMatrix.entryFor(Operation.OpType.EXPAND, DataType.FLOAT32);
        NativeCpuCoverageEntry squeeze = NativeCpuCoverageMatrix.entryFor(Operation.OpType.SQUEEZE, DataType.BFLOAT16);

        assertEquals(NativeCpuKernelPerformanceStatus.VIEW_ONLY, reshape.status());
        assertEquals(NativeCpuKernelFamily.VIEW_ONLY, reshape.family());
        assertEquals(NativeCpuCoverageLayoutScope.VIEW_ONLY, reshape.layoutScope());
        assertTrue(reshape.nativeSupported());
        assertTrue(reshape.preservesNativeStorage());
        assertTrue(reshape.autoFastEligible());
        assertEquals("", reshape.fallbackReason());

        assertEquals(NativeCpuCoverageLayoutScope.VIEW_ONLY, permute.layoutScope());
        assertTrue(permute.nativeSupported());
        assertTrue(permute.preservesNativeStorage());

        assertEquals(NativeCpuCoverageLayoutScope.VIEW_ONLY, select.layoutScope());
        assertTrue(select.nativeSupported());
        assertTrue(select.preservesNativeStorage());

        assertEquals(NativeCpuCoverageLayoutScope.VIEW_ONLY, expand.layoutScope());
        assertTrue(expand.nativeSupported());
        assertTrue(expand.preservesNativeStorage());

        assertEquals(NativeCpuCoverageLayoutScope.VIEW_ONLY, squeeze.layoutScope());
        assertTrue(squeeze.nativeSupported());
    }

    @Test
    void unsupportedLayoutAndDtypeRowsRemainArrayOrStridedUnsupported() {
        NativeCpuCoverageEntry concat = NativeCpuCoverageMatrix.entryFor(Operation.OpType.CONCAT, DataType.FLOAT32);
        NativeCpuCoverageEntry pad = NativeCpuCoverageMatrix.entryFor(Operation.OpType.PAD, DataType.FLOAT32);
        NativeCpuCoverageEntry tile = NativeCpuCoverageMatrix.entryFor(Operation.OpType.TILE, DataType.FLOAT32);
        NativeCpuCoverageEntry boolAdd = NativeCpuCoverageMatrix.entryFor(Operation.OpType.ADD, DataType.BOOL);
        NativeCpuCoverageEntry boolLogicalAnd = NativeCpuCoverageMatrix.entryFor(Operation.OpType.LOGICAL_AND, DataType.BOOL);
        NativeCpuCoverageEntry boolReduceAny = NativeCpuCoverageMatrix.entryFor(Operation.OpType.REDUCE_ANY, DataType.BOOL);
        NativeCpuCoverageEntry intMatmul = NativeCpuCoverageMatrix.entryFor(Operation.OpType.MATMUL, DataType.INT32);

        assertEquals(NativeCpuCoverageLayoutScope.ARRAY_ONLY, concat.layoutScope());
        assertEquals(NativeCpuCoverageLayoutScope.ARRAY_ONLY, pad.layoutScope());
        assertEquals(NativeCpuCoverageLayoutScope.ARRAY_ONLY, tile.layoutScope());
        assertFalse(concat.nativeSupported());
        assertFalse(pad.nativeSupported());
        assertFalse(tile.nativeSupported());
        assertFalse(concat.fallbackReason().isBlank());

        assertEquals(NativeCpuCoverageLayoutScope.ARRAY_ONLY, boolAdd.layoutScope());
        assertEquals(NativeCpuCoverageLayoutScope.DENSE_CONTIGUOUS, boolLogicalAnd.layoutScope());
        assertEquals(NativeCpuCoverageLayoutScope.DENSE_CONTIGUOUS, boolReduceAny.layoutScope());
        assertEquals(NativeCpuCoverageLayoutScope.ARRAY_ONLY, intMatmul.layoutScope());
        assertFalse(boolAdd.nativeSupported());
        assertTrue(boolLogicalAnd.nativeSupported());
        assertTrue(boolReduceAny.nativeSupported());
        assertFalse(intMatmul.nativeSupported());
        assertTrue(boolAdd.fallbackReason().contains("native-storage-dtype-unsupported"));
    }

    @Test
    void correctButSlowNonBlasRowsAreNotAutoFastEligible() {
        for (NativeCpuCoverageEntry entry : NativeCpuCoverageMatrix.entries()) {
            if (entry.status() == NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW) {
                assertFalse(entry.autoFastEligible(), entry.toString());
            }
        }
    }

    @Test
    void entriesForAndIsNativeSupportedUseMatrixRows() {
        assertEquals(
                Operation.OpType.values().length,
                NativeCpuCoverageMatrix.entriesFor(DataType.FLOAT32).size()
        );
        assertTrue(NativeCpuCoverageMatrix.isNativeSupported(Operation.OpType.ADD, DataType.FLOAT32));
        assertTrue(NativeCpuCoverageMatrix.isNativeSupported(Operation.OpType.RESHAPE, DataType.FLOAT32));
        assertFalse(NativeCpuCoverageMatrix.isNativeSupported(Operation.OpType.ERF, DataType.BFLOAT16));
        assertFalse(NativeCpuCoverageMatrix.isNativeSupported(Operation.OpType.MATMUL, DataType.INT32));
    }
}
