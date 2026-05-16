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
        NativeCpuCoverageEntry squeeze = NativeCpuCoverageMatrix.entryFor(Operation.OpType.SQUEEZE, DataType.BFLOAT16);

        assertEquals(NativeCpuKernelPerformanceStatus.VIEW_ONLY, reshape.status());
        assertEquals(NativeCpuKernelFamily.VIEW_ONLY, reshape.family());
        assertEquals(NativeCpuCoverageLayoutScope.VIEW_ONLY, reshape.layoutScope());
        assertTrue(reshape.nativeSupported());
        assertTrue(reshape.preservesNativeStorage());
        assertTrue(reshape.autoFastEligible());
        assertEquals("", reshape.fallbackReason());

        assertEquals(NativeCpuCoverageLayoutScope.VIEW_ONLY, squeeze.layoutScope());
        assertTrue(squeeze.nativeSupported());
    }

    @Test
    void unsupportedLayoutAndDtypeRowsRemainArrayOrStridedUnsupported() {
        NativeCpuCoverageEntry select = NativeCpuCoverageMatrix.entryFor(Operation.OpType.SELECT, DataType.FLOAT32);
        NativeCpuCoverageEntry slice = NativeCpuCoverageMatrix.entryFor(Operation.OpType.SLICE, DataType.FLOAT32);
        NativeCpuCoverageEntry permute = NativeCpuCoverageMatrix.entryFor(Operation.OpType.PERMUTE, DataType.FLOAT32);
        NativeCpuCoverageEntry expand = NativeCpuCoverageMatrix.entryFor(Operation.OpType.EXPAND, DataType.FLOAT32);
        NativeCpuCoverageEntry boolAdd = NativeCpuCoverageMatrix.entryFor(Operation.OpType.ADD, DataType.BOOL);
        NativeCpuCoverageEntry intMatmul = NativeCpuCoverageMatrix.entryFor(Operation.OpType.MATMUL, DataType.INT32);

        assertEquals(NativeCpuCoverageLayoutScope.STRIDED_UNSUPPORTED, select.layoutScope());
        assertEquals(NativeCpuCoverageLayoutScope.STRIDED_UNSUPPORTED, slice.layoutScope());
        assertEquals(NativeCpuCoverageLayoutScope.STRIDED_UNSUPPORTED, permute.layoutScope());
        assertEquals(NativeCpuCoverageLayoutScope.STRIDED_UNSUPPORTED, expand.layoutScope());
        assertFalse(select.nativeSupported());
        assertFalse(slice.nativeSupported());
        assertFalse(permute.nativeSupported());
        assertFalse(expand.nativeSupported());
        assertFalse(select.fallbackReason().isBlank());

        assertEquals(NativeCpuCoverageLayoutScope.ARRAY_ONLY, boolAdd.layoutScope());
        assertEquals(NativeCpuCoverageLayoutScope.ARRAY_ONLY, intMatmul.layoutScope());
        assertFalse(boolAdd.nativeSupported());
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
