package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import jdk.incubator.vector.FloatVector;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ClampRangeAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import org.junit.jupiter.api.Test;

/** Typed scalar-immediate and clamp execution across admitted layouts and carrier forms. */
class CpuScalarImmediateClampMatrixTest {

    @Test void everyExactArtifactExecutesAgainstItsIndependentTypedCleanJavaLoop() throws Throwable {
        for (var fixture : CpuScalarImmediateClampMatrixOracle.fixtures()) {
            
            Object expected = CpuScalarImmediateClampMatrixOracle.cleanJava(fixture);
            Object actual = CpuScalarImmediateClampMatrixOracle.generated(fixture);
            if (expected instanceof short[] value) assertArrayEquals(value, (short[]) actual, fixture.id());
            else if (expected instanceof float[] value) assertArrayEquals(value, (float[]) actual, fixture.id());
            else if (expected instanceof double[] value) assertArrayEquals(value, (double[]) actual, fixture.id());
            else if (expected instanceof int[] value) assertArrayEquals(value, (int[]) actual, fixture.id());
            else assertArrayEquals((long[]) expected, (long[]) actual, fixture.id());
        }
    }

    @Test void everyCompositionalFormExecutesAgainstItsResolvedLayoutOracle() throws Throwable {
        for (var form : CpuScalarImmediateClampMatrixOracle.forms()) {
            
            assertTypedEquals(CpuScalarImmediateClampMatrixOracle.cleanJava(form),
                    CpuScalarImmediateClampMatrixOracle.generated(form), form.id());
        }
    }

    @Test void reachableSegmentAndOrderedMixedCarriersHaveTheirOwnDescriptorAndTypedExecution() throws Throwable {
        for (var form : CpuScalarImmediateClampMatrixOracle.forms().stream()
                .filter(value -> value.id().startsWith("CARRIER-")).toList()) {
                    
            var fixture = form.fixture();
            var artifact = CpuScalarImmediateClampMatrixOracle.artifact(fixture);
            assertTrue(artifact.descriptor().contains("MemorySegment"), fixture.id());
            assertTypedEquals(CpuScalarImmediateClampMatrixOracle.cleanJava(fixture),
                    CpuScalarImmediateClampMatrixOracle.generated(fixture), fixture.id());
        }
    }

    @Test void sourceCapabilityRowsCloseExactlyAndAdjacentCasesFailBeforeGeneration() {
        var provider = new CpuCapabilityProvider();
        var numeric = List.of(DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64,
                DataType.INT32, DataType.INT64);
        for (var kind : ScalarElementwiseKind.values()) {
            for (var type : io.github.pho001.synaptik.backend.cpu.internal.CpuTestDtypes.currentExecutable()) {
                boolean expected = type != DataType.BOOL
                        && (kind != ScalarElementwiseKind.CLAMP
                            && kind != ScalarElementwiseKind.DIV
                            && kind != ScalarElementwiseKind.POW || type.isFloating());
                var query = query(kind, type, type, type);
                assertEquals(expected, provider.supports(query), kind + " " + type);
            }
        }
        for (var kind : ScalarElementwiseKind.values()) {
            for (var input : numeric) for (var output : numeric) if (input != output) {
                assertFalse(provider.supports(query(kind, input, output, output)),
                        kind + " mixed input/output " + input + '/' + output);
                assertFalse(provider.supports(query(kind, input, input, output)),
                        kind + " mixed immediate " + input + '/' + output);
            }
        }
        for (var type : List.of(DataType.INT32, DataType.INT64)) {
            assertFalse(provider.supports(query(ScalarElementwiseKind.DIV, type, type, type)));
            assertFalse(provider.supports(query(ScalarElementwiseKind.POW, type, type, type)));
            assertFalse(provider.supports(query(ScalarElementwiseKind.CLAMP, type, type, type)));
        }
    }


    @Test void vectorRangeWitnessUsesArbitraryBoundsAndScalarTail() throws Throwable {
        var fixture = base(DataType.FLOAT32);
        int start = FloatVector.SPECIES_PREFERRED.length();
        float[] full = (float[]) CpuScalarImmediateClampMatrixOracle.cleanJava(fixture);
        float[] actual = (float[]) CpuScalarImmediateClampMatrixOracle.generated(fixture, start, 31L);
        float[] expected = new float[32];
        System.arraycopy(full, start, expected, start, 31 - start);
        assertArrayEquals(expected, actual);
    }


    /** Host-sized vector work, scalar tails and untouched output canaries for every access regime. */
    @Test void eachAdmittedRegimeExecutesItsOwnTypedAddressingOracleAcrossCarrierForms() throws Throwable {
        var base = base();
        int columns = FloatVector.SPECIES_PREFERRED.length() + 1;
        int count = 2 * columns;
        Shape matrix = Shape.of(2, columns);
        record Case(String id, LayoutDescriptor layout,
                java.util.function.IntUnaryOperator address, List<CpuKernelSpecialization.CarrierAccess> carriers) { }
        var array = CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY;
        var segment = CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT;
        var cases = List.of(
                new Case("dense", LayoutDescriptor.contiguous(matrix),
                        ordinal -> ordinal, List.of(array, array)),
                new Case("scalar", LayoutDescriptor.of(matrix, new long[] {0, 0}, 0, true),
                        ordinal -> 0, List.of(segment, segment)),
                new Case("last-axis-bias", LayoutDescriptor.of(matrix, new long[] {0, 1}, 0, true),
                        ordinal -> ordinal % columns, List.of(array, segment)),
                new Case("block-outer", LayoutDescriptor.of(matrix, new long[] {columns + 2, 1}, 0, true),
                        ordinal -> (ordinal / columns) * (columns + 2) + ordinal % columns,
                        List.of(segment, array)),
                new Case("general-odometer", LayoutDescriptor.of(matrix, new long[] {2 * columns + 2, 2}, 0, true),
                        ordinal -> (ordinal / columns) * (2 * columns + 2) + (ordinal % columns) * 2,
                        List.of(array, segment)));
        for (var testCase : cases) {
            var fixture = new CpuScalarImmediateClampMatrixOracle.Fixture(base.id() + '-' + testCase.id(),
                    base.operation(), base.type(), base.category(), base.immediate(), base.upper(),
                    testCase.carriers(), base.preference(), base.parallelism());
            var context = CpuScalarImmediateClampMatrixOracle.contextFor(fixture, matrix, testCase.layout(),
                    matrix, LayoutDescriptor.contiguous(matrix));
            var route = new CpuPartitionPreparer().analyze(context).plan().units().getFirst().portablePlan();
            byte[] bytes = new CpuClassFileKernelGenerator().generateClassBytes(route.specialization(), route.kernelIr());
            float[] source = new float[testCase.address().applyAsInt(count - 1) + 1];
            for (int cell = 0; cell < source.length; cell++) source[cell] = (cell - columns) * .25f;
            float[] before = source.clone();
            float[] output = new float[count + 2], expected = new float[count + 2];
            java.util.Arrays.fill(output, -123.25f);
            java.util.Arrays.fill(expected, -123.25f);
            Object input = testCase.carriers().getFirst() == segment ? MemorySegment.ofArray(source) : source;
            Object result = testCase.carriers().getLast() == segment ? MemorySegment.ofArray(output) : output;
            MethodHandle handle = new CpuClassFileKernelGenerator().defineClassBytes(route.specialization(), route.kernelIr(), bytes).entryPoint();
            handle.invokeWithArguments(input, result,
                    CpuScalarImmediateClampMatrixOracle.geometryFor(context, 1L, (long) count - 1),
                    1L, (long) count - 1);
            for (int ordinal = 1; ordinal < count - 1; ordinal++)
                expected[ordinal] = source[testCase.address().applyAsInt(ordinal)] + .5f;
            assertArrayEquals(expected, output, testCase.id());
            assertArrayEquals(before, source, testCase.id() + " immutable input");
        }
    }


    private static CpuScalarImmediateClampMatrixOracle.Fixture base() {
        return base(DataType.FLOAT32);
    }
    private static CpuScalarImmediateClampMatrixOracle.Fixture base(DataType type) {
        return CpuScalarImmediateClampMatrixOracle.fixtures().stream()
                .filter(f -> f.id().equals("ADD-" + type + "-OTHER")).findFirst().orElseThrow();
    }

    private static OperationCapabilityQuery query(ScalarElementwiseKind kind, DataType inputType,
            DataType outputType, DataType immediateType) {
        Shape shape = Shape.of(8);
        var input = new TensorDescriptor(inputType, shape,
                java.util.Optional.of(LayoutDescriptor.contiguous(shape)), false);
        var output = new TensorDescriptor(outputType, shape,
                java.util.Optional.of(LayoutDescriptor.contiguous(shape)), false);
        DataType attributeType = immediateType == DataType.BOOL ? DataType.FLOAT32 : immediateType;
        Operation operation = kind == ScalarElementwiseKind.CLAMP
                ? new Operation(kind, new ClampRangeAttrs(scalar(attributeType, -1), scalar(attributeType, 1)))
                : new Operation(kind, new ScalarValueAttrs(scalar(immediateType, 1)));
        return new OperationCapabilityQuery(operation, List.of(input), List.of(output));
    }

    private static ScalarValue scalar(DataType type, int value) {
        return switch (type) {
            case BFLOAT16 -> ScalarValue.bfloat16(value);
            case FLOAT32 -> ScalarValue.float32(value);
            case FLOAT64 -> ScalarValue.float64(value);
            case INT32 -> ScalarValue.int32(value);
            case INT64 -> ScalarValue.int64(value);
            case BOOL -> ScalarValue.bool(value != 0);
            case FLOAT16 -> throw new IllegalArgumentException("FLOAT16 CPU scalar clamp unsupported");
        };
    }


    private static void assertTypedEquals(Object expected, Object actual, String message) {
        if (expected instanceof short[] value) assertArrayEquals(value, (short[]) actual, message);
        else if (expected instanceof float[] value) assertArrayEquals(value, (float[]) actual, message);
        else if (expected instanceof double[] value) assertArrayEquals(value, (double[]) actual, message);
        else if (expected instanceof int[] value) assertArrayEquals(value, (int[]) actual, message);
        else assertArrayEquals((long[]) expected, (long[]) actual, message);
    }



}
