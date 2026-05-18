package backend.cpu.nativecpu;

import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;

import java.util.Map;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NativeCpuParityMatrixTest {
    @Test
    void parityMatrixContainsOneExecutableRowForEveryOperationAndDtype() {
        assertEquals(
                Operation.OpType.values().length * DataType.values().length,
                NativeCpuParityMatrix.entries().size()
        );

        Map<String, NativeCpuParityEntry> byKey = NativeCpuParityMatrix.entries().stream()
                .collect(Collectors.toMap(
                        entry -> entry.opType().name() + ":" + entry.dataType().name(),
                        entry -> entry
                ));
        assertEquals(NativeCpuParityMatrix.entries().size(), byKey.size());
        assertTrue(byKey.containsKey(Operation.OpType.ADD.name() + ":" + DataType.FLOAT32.name()));
        assertTrue(byKey.containsKey(Operation.OpType.UNKNOWN.name() + ":" + DataType.BOOL.name()));
    }

    @Test
    void parityRowsStayDerivedFromCoverageRowsAndAutoEligibility() {
        for (NativeCpuParityEntry entry : NativeCpuParityMatrix.entries()) {
            NativeCpuCoverageEntry coverage = NativeCpuCoverageMatrix.entryFor(entry.opType(), entry.dataType());

            assertEquals(coverage.status(), entry.status(), entry.toString());
            assertEquals(coverage.family(), entry.family(), entry.toString());
            assertEquals(
                    coverage.status() == NativeCpuKernelPerformanceStatus.NATIVE_FAST
                            || coverage.status() == NativeCpuKernelPerformanceStatus.LIBRARY_PROVIDER
                            || coverage.status() == NativeCpuKernelPerformanceStatus.VIEW_ONLY,
                    entry.autoEligible(),
                    entry.toString()
            );
            assertEquals(entry.opType() != Operation.OpType.UNKNOWN, entry.logicalOperationDefined(), entry.toString());
            assertFalse(entry.reason().isBlank(), entry.toString());
        }
    }

    @Test
    void autoEligibilityIsExposedAsParityMatrixDecision() {
        assertTrue(NativeCpuParityMatrix.isAutoEligible(Operation.OpType.MATMUL, DataType.FLOAT32));
        assertTrue(NativeCpuParityMatrix.isAutoEligible(Operation.OpType.RESHAPE, DataType.FLOAT32));
        assertFalse(NativeCpuParityMatrix.isAutoEligible(Operation.OpType.RELU, DataType.FLOAT32));
        assertFalse(NativeCpuParityMatrix.isAutoEligible(Operation.OpType.ADD, DataType.BFLOAT16));
        assertFalse(NativeCpuParityMatrix.isAutoEligible(Operation.OpType.MIN, DataType.FLOAT32));
        assertFalse(NativeCpuParityMatrix.isAutoEligible(Operation.OpType.CLAMP_MIN, DataType.BFLOAT16));
        assertFalse(NativeCpuParityMatrix.isAutoEligible(Operation.OpType.FLOOR, DataType.FLOAT32));
        assertFalse(NativeCpuParityMatrix.isAutoEligible(Operation.OpType.CEIL, DataType.FLOAT64));
        assertFalse(NativeCpuParityMatrix.isAutoEligible(Operation.OpType.SIGN, DataType.FLOAT32));
        assertFalse(NativeCpuParityMatrix.isAutoEligible(Operation.OpType.POW, DataType.FLOAT32));
        assertFalse(NativeCpuParityMatrix.isAutoEligible(Operation.OpType.POW_TENSOR, DataType.FLOAT64));
        assertFalse(NativeCpuParityMatrix.isAutoEligible(Operation.OpType.REDUCE_MIN, DataType.FLOAT32));
        assertFalse(NativeCpuParityMatrix.isAutoEligible(Operation.OpType.REDUCE_MAX, DataType.FLOAT64));
        assertFalse(NativeCpuParityMatrix.isAutoEligible(Operation.OpType.ERF, DataType.FLOAT32));
    }

    @Test
    void parityRowsExposeStorageLayoutResidencyAndAutoInvariants() {
        for (NativeCpuParityEntry entry : NativeCpuParityMatrix.entries()) {
            NativeCpuCoverageEntry coverage = NativeCpuCoverageMatrix.entryFor(entry.opType(), entry.dataType());

            assertFalse(entry.resultResidencies().isEmpty(), entry.toString());
            if (entry.logicalOperationDefined()) {
                assertTrue(entry.hasStoragePath(NativeCpuStoragePath.CPU_ARRAY_DENSE), entry.toString());
                assertTrue(entry.hasStoragePath(NativeCpuStoragePath.CPU_ARRAY_STRIDED), entry.toString());
                assertTrue(entry.hasLayoutCapability(NativeCpuLayoutCapability.DENSE), entry.toString());
            } else {
                assertFalse(entry.hasStoragePath(NativeCpuStoragePath.CPU_ARRAY_DENSE), entry.toString());
                assertFalse(entry.hasStoragePath(NativeCpuStoragePath.CPU_ARRAY_STRIDED), entry.toString());
                assertFalse(entry.hasLayoutCapability(NativeCpuLayoutCapability.DENSE), entry.toString());
            }

            assertEquals(coverage.nativeSupported(), hasAnyNativeStoragePath(entry), entry.toString());
            switch (entry.status()) {
                case LIBRARY_PROVIDER -> {
                    assertTrue(entry.autoEligible(), entry.toString());
                    assertTrue(entry.hasStoragePath(NativeCpuStoragePath.CPU_NATIVE_REGION_PROVIDER), entry.toString());
                    assertFalse(entry.hasStoragePath(NativeCpuStoragePath.CPU_NATIVE_SINGLE_DENSE), entry.toString());
                    assertFalse(entry.hasStoragePath(NativeCpuStoragePath.CPU_NATIVE_REGION_DENSE), entry.toString());
                    assertFalse(entry.hasStoragePath(NativeCpuStoragePath.CPU_NATIVE_REGION_STRIDED), entry.toString());
                    assertFalse(entry.hasStoragePath(NativeCpuStoragePath.CPU_NATIVE_REGION_BROADCAST), entry.toString());
                    assertFalse(entry.hasStoragePath(NativeCpuStoragePath.CPU_NATIVE_REGION_VIEW_ALIAS), entry.toString());
                    assertTrue(entry.hasResultResidency(NativeCpuResultResidency.CPU_NATIVE), entry.toString());
                }
                case VIEW_ONLY -> {
                    assertTrue(entry.autoEligible(), entry.toString());
                    assertTrue(entry.hasStoragePath(NativeCpuStoragePath.CPU_NATIVE_REGION_VIEW_ALIAS), entry.toString());
                    assertFalse(entry.hasStoragePath(NativeCpuStoragePath.CPU_NATIVE_REGION_PROVIDER), entry.toString());
                    assertFalse(entry.layoutCapabilities().isEmpty(), entry.toString());
                    assertTrue(entry.hasResultResidency(NativeCpuResultResidency.VIEW_ALIAS), entry.toString());
                }
                case NATIVE_FAST -> assertTrue(entry.autoEligible(), entry.toString());
                case NATIVE_CORRECT_BUT_SLOW -> assertFalse(entry.autoEligible(), entry.toString());
                case ARRAY_ONLY, NATIVE_UNSUPPORTED -> {
                    assertFalse(entry.autoEligible(), entry.toString());
                    assertFalse(hasAnyNativeStoragePath(entry), entry.toString());
                    assertTrue(entry.hasResultResidency(NativeCpuResultResidency.CPU_ARRAY), entry.toString());
                }
            }
        }
    }

    @Test
    void floatElementwiseRowsExposeArrayNativeDenseStridedAndBroadcastSeparately() {
        NativeCpuParityEntry add = NativeCpuParityMatrix.entryFor(Operation.OpType.ADD, DataType.FLOAT32);

        assertTrue(add.logicalOperationDefined());
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, add.status());
        assertFalse(add.autoEligible());
        assertTrue(add.hasStoragePath(NativeCpuStoragePath.CPU_ARRAY_DENSE));
        assertTrue(add.hasStoragePath(NativeCpuStoragePath.CPU_ARRAY_STRIDED));
        assertTrue(add.hasStoragePath(NativeCpuStoragePath.CPU_NATIVE_SINGLE_DENSE));
        assertTrue(add.hasStoragePath(NativeCpuStoragePath.CPU_NATIVE_REGION_DENSE));
        assertTrue(add.hasStoragePath(NativeCpuStoragePath.CPU_NATIVE_REGION_STRIDED));
        assertTrue(add.hasStoragePath(NativeCpuStoragePath.CPU_NATIVE_REGION_BROADCAST));
        assertTrue(add.hasLayoutCapability(NativeCpuLayoutCapability.DENSE));
        assertTrue(add.hasLayoutCapability(NativeCpuLayoutCapability.OFFSET_CONTIGUOUS));
        assertTrue(add.hasLayoutCapability(NativeCpuLayoutCapability.SAME_SHAPE_STRIDED_READ));
        assertTrue(add.hasLayoutCapability(NativeCpuLayoutCapability.ZERO_STRIDE_BROADCAST_READ));
        assertTrue(add.hasLayoutCapability(NativeCpuLayoutCapability.LAST_DIM_BIAS_BROADCAST));
        assertTrue(add.hasResultResidency(NativeCpuResultResidency.CPU_NATIVE));

        NativeCpuParityEntry clampMin = NativeCpuParityMatrix.entryFor(Operation.OpType.CLAMP_MIN, DataType.BFLOAT16);
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, clampMin.status());
        assertFalse(clampMin.autoEligible());
        assertTrue(clampMin.hasStoragePath(NativeCpuStoragePath.CPU_NATIVE_REGION_STRIDED));
        assertFalse(clampMin.hasStoragePath(NativeCpuStoragePath.CPU_NATIVE_REGION_BROADCAST));
        assertTrue(clampMin.hasResultResidency(NativeCpuResultResidency.CPU_NATIVE));

        NativeCpuParityEntry pow = NativeCpuParityMatrix.entryFor(Operation.OpType.POW, DataType.FLOAT32);
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, pow.status());
        assertFalse(pow.autoEligible());
        assertTrue(pow.hasStoragePath(NativeCpuStoragePath.CPU_NATIVE_REGION_STRIDED));
        assertFalse(pow.hasStoragePath(NativeCpuStoragePath.CPU_NATIVE_REGION_BROADCAST));

        NativeCpuParityEntry sign = NativeCpuParityMatrix.entryFor(Operation.OpType.SIGN, DataType.FLOAT64);
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, sign.status());
        assertFalse(sign.autoEligible());
        assertTrue(sign.hasStoragePath(NativeCpuStoragePath.CPU_NATIVE_REGION_STRIDED));
        assertFalse(sign.hasStoragePath(NativeCpuStoragePath.CPU_NATIVE_REGION_BROADCAST));

        NativeCpuParityEntry powTensor = NativeCpuParityMatrix.entryFor(Operation.OpType.POW_TENSOR, DataType.FLOAT64);
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, powTensor.status());
        assertFalse(powTensor.autoEligible());
        assertTrue(powTensor.hasStoragePath(NativeCpuStoragePath.CPU_NATIVE_REGION_BROADCAST));

        NativeCpuParityEntry bf16PowTensor = NativeCpuParityMatrix.entryFor(Operation.OpType.POW_TENSOR, DataType.BFLOAT16);
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_UNSUPPORTED, bf16PowTensor.status());
        assertFalse(bf16PowTensor.autoEligible());
        assertFalse(bf16PowTensor.hasStoragePath(NativeCpuStoragePath.CPU_NATIVE_REGION_DENSE));

        NativeCpuParityEntry reduceMin = NativeCpuParityMatrix.entryFor(Operation.OpType.REDUCE_MIN, DataType.FLOAT32);
        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, reduceMin.status());
        assertFalse(reduceMin.autoEligible());
        assertTrue(reduceMin.hasStoragePath(NativeCpuStoragePath.CPU_NATIVE_REGION_STRIDED));
        assertFalse(reduceMin.hasStoragePath(NativeCpuStoragePath.CPU_NATIVE_REGION_BROADCAST));
        assertTrue(reduceMin.hasResultResidency(NativeCpuResultResidency.CPU_NATIVE));
    }

    @Test
    void providerAndViewRowsHaveDistinctStorageAndResidencyFacts() {
        NativeCpuParityEntry matmul = NativeCpuParityMatrix.entryFor(Operation.OpType.MATMUL, DataType.FLOAT32);
        NativeCpuParityEntry permute = NativeCpuParityMatrix.entryFor(Operation.OpType.PERMUTE, DataType.BFLOAT16);

        assertEquals(NativeCpuKernelPerformanceStatus.LIBRARY_PROVIDER, matmul.status());
        assertTrue(matmul.autoEligible());
        assertTrue(matmul.hasStoragePath(NativeCpuStoragePath.CPU_NATIVE_REGION_PROVIDER));
        assertFalse(matmul.hasStoragePath(NativeCpuStoragePath.CPU_NATIVE_REGION_STRIDED));
        assertTrue(matmul.hasLayoutCapability(NativeCpuLayoutCapability.DENSE));
        assertTrue(matmul.hasResultResidency(NativeCpuResultResidency.CPU_NATIVE));

        assertEquals(NativeCpuKernelPerformanceStatus.VIEW_ONLY, permute.status());
        assertTrue(permute.autoEligible());
        assertTrue(permute.hasStoragePath(NativeCpuStoragePath.CPU_NATIVE_REGION_VIEW_ALIAS));
        assertTrue(permute.hasLayoutCapability(NativeCpuLayoutCapability.TRANSPOSE_PERMUTE_READ_VIEW));
        assertTrue(permute.hasResultResidency(NativeCpuResultResidency.VIEW_ALIAS));
    }

    @Test
    void selectSliceAndExpandRowsExposeNativeViewDescriptors() {
        NativeCpuParityEntry select = NativeCpuParityMatrix.entryFor(Operation.OpType.SELECT, DataType.FLOAT32);
        NativeCpuParityEntry slice = NativeCpuParityMatrix.entryFor(Operation.OpType.SLICE, DataType.FLOAT32);
        NativeCpuParityEntry expand = NativeCpuParityMatrix.entryFor(Operation.OpType.EXPAND, DataType.FLOAT32);

        assertEquals(NativeCpuKernelPerformanceStatus.VIEW_ONLY, select.status());
        assertTrue(select.autoEligible());
        assertTrue(select.hasStoragePath(NativeCpuStoragePath.CPU_ARRAY_DENSE));
        assertTrue(select.hasStoragePath(NativeCpuStoragePath.CPU_ARRAY_STRIDED));
        assertTrue(select.hasStoragePath(NativeCpuStoragePath.CPU_NATIVE_REGION_VIEW_ALIAS));
        assertTrue(select.hasLayoutCapability(NativeCpuLayoutCapability.SELECT_SLICE_OFFSET_VIEW));
        assertTrue(select.hasResultResidency(NativeCpuResultResidency.VIEW_ALIAS));

        assertEquals(NativeCpuKernelPerformanceStatus.VIEW_ONLY, slice.status());
        assertTrue(slice.hasLayoutCapability(NativeCpuLayoutCapability.SELECT_SLICE_OFFSET_VIEW));
        assertTrue(slice.hasResultResidency(NativeCpuResultResidency.VIEW_ALIAS));

        assertEquals(NativeCpuKernelPerformanceStatus.VIEW_ONLY, expand.status());
        assertTrue(expand.hasLayoutCapability(NativeCpuLayoutCapability.ZERO_STRIDE_BROADCAST_READ));
        assertTrue(expand.hasResultResidency(NativeCpuResultResidency.VIEW_ALIAS));
    }

    @Test
    void boolCompareRowsExposeNativeMaskAndCpuArrayBoundaryResidency() {
        NativeCpuParityEntry gt = NativeCpuParityMatrix.entryFor(Operation.OpType.GT, DataType.BOOL);
        NativeCpuParityEntry logicalAnd = NativeCpuParityMatrix.entryFor(Operation.OpType.LOGICAL_AND, DataType.BOOL);
        NativeCpuParityEntry reduceAny = NativeCpuParityMatrix.entryFor(Operation.OpType.REDUCE_ANY, DataType.BOOL);

        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, gt.status());
        assertFalse(gt.autoEligible());
        assertTrue(gt.hasResultResidency(NativeCpuResultResidency.BOOL_MASK_ARRAY));
        assertTrue(gt.hasResultResidency(NativeCpuResultResidency.BOOL_MASK_NATIVE));
        assertTrue(gt.hasStoragePath(NativeCpuStoragePath.CPU_NATIVE_REGION_STRIDED));
        assertTrue(gt.hasStoragePath(NativeCpuStoragePath.CPU_NATIVE_REGION_BROADCAST));

        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, logicalAnd.status());
        assertTrue(logicalAnd.hasResultResidency(NativeCpuResultResidency.BOOL_MASK_ARRAY));
        assertTrue(logicalAnd.hasResultResidency(NativeCpuResultResidency.BOOL_MASK_NATIVE));
        assertTrue(logicalAnd.hasStoragePath(NativeCpuStoragePath.CPU_NATIVE_REGION_BROADCAST));

        assertEquals(NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW, reduceAny.status());
        assertTrue(reduceAny.hasResultResidency(NativeCpuResultResidency.BOOL_MASK_NATIVE));
        assertTrue(reduceAny.hasStoragePath(NativeCpuStoragePath.CPU_NATIVE_REGION_STRIDED));
        assertFalse(reduceAny.hasStoragePath(NativeCpuStoragePath.CPU_NATIVE_REGION_BROADCAST));
    }

    @Test
    void renderCsvIsStableMachineReadableDocumentation() {
        String csv = NativeCpuParityMatrix.renderCsv();

        assertTrue(csv.startsWith("opType,dataType,logicalOperationDefined,status,family,autoEligible,"));
        assertTrue(csv.contains("ADD,FLOAT32,true,NATIVE_CORRECT_BUT_SLOW,SEGMENT_SCALAR,false"));
        assertTrue(csv.contains("MATMUL,FLOAT32,true,LIBRARY_PROVIDER,OPENBLAS_NATIVE_SEGMENT,true"));
        assertEquals(NativeCpuParityMatrix.entries().size() + 1, csv.lines().count());
        csv.lines().forEach(line -> assertEquals(10, line.split(",", -1).length, line));
        assertTrue(csv.contains("CPU_ARRAY_DENSE|CPU_ARRAY_STRIDED|CPU_NATIVE_SINGLE_DENSE"));
    }

    private static boolean hasAnyNativeStoragePath(NativeCpuParityEntry entry) {
        return entry.hasStoragePath(NativeCpuStoragePath.CPU_NATIVE_SINGLE_DENSE)
                || entry.hasStoragePath(NativeCpuStoragePath.CPU_NATIVE_REGION_DENSE)
                || entry.hasStoragePath(NativeCpuStoragePath.CPU_NATIVE_REGION_STRIDED)
                || entry.hasStoragePath(NativeCpuStoragePath.CPU_NATIVE_REGION_BROADCAST)
                || entry.hasStoragePath(NativeCpuStoragePath.CPU_NATIVE_REGION_VIEW_ALIAS)
                || entry.hasStoragePath(NativeCpuStoragePath.CPU_NATIVE_REGION_PROVIDER);
    }
}
