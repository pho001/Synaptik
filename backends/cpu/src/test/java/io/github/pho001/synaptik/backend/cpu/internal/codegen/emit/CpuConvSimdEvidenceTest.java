package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratorSchema;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuConv2dLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuConv3dLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionStrategy;
import io.github.pho001.synaptik.backend.cpu.internal.reference.CpuConv2dReferenceKernel;
import io.github.pho001.synaptik.backend.cpu.internal.reference.CpuConv3dReferenceKernel;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.io.IOException;
import java.lang.classfile.ClassFile;
import java.lang.classfile.Instruction;
import java.lang.classfile.Opcode;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Optional;
import java.util.stream.StreamSupport;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import org.junit.jupiter.api.Test;

/**
 * Semantic and structural evidence for the production schema-63 output-width Conv form.
 *
 * <p>Every eligible row obtains its schema-63 specialization from {@link CpuPartitionPreparer};
 * an independently prepared scalar route remains the exact semantic and class-file control.
 * Ineligible width geometry and non-dense layouts prove production scalar fallback.</p>
 */
final class CpuConvSimdEvidenceTest {
    private static final String EVIDENCE_ROOT_PROPERTY = "synaptik.cpu.convSimd.evidenceRoot";

    private enum Rank { CONV2D, CONV3D }
    private enum CarrierForm { ARRAYS, SEGMENTS, MIXED }
    private enum Scenario {
        UNIT_INTERIOR_EXACT_WIDTH, UNIT_INTERIOR_TAIL, PADDED_PROLOGUE_INTERIOR_EPILOGUE,
        GROUPED, DEPTHWISE, BATCH_GT_ONE, STRIDE_HEIGHT_OR_DEPTH, DILATION_HEIGHT_OR_DEPTH,
        SCALAR_WIDTH_STRIDE, SCALAR_WIDTH_DILATION, SHORT_WIDTH, NON_DENSE;

        boolean vectorEligible() {
            return switch (this) {
                case SCALAR_WIDTH_STRIDE, SCALAR_WIDTH_DILATION, SHORT_WIDTH, NON_DENSE -> false;
                default -> true;
            };
        }
    }

    private record SemanticRow(Rank rank, DataType type, Scenario scenario, CarrierForm carriers,
            boolean bias, boolean callerParallel) { }
    private record Dossier(Rank rank, DataType type, boolean bias, CarrierForm carriers) { }
    private record Route(CpuKernelSpecialization scalar, CpuKernelIr ir, byte[] scalarBytes,
            CpuKernelSpecialization vector, byte[] vectorBytes) { }

    @Test void stageBSemanticMatrixExecutesExactlyFortyEightRawBitRows() throws Throwable {
        assertEquals(63, CpuGeneratorSchema.CURRENT_VERSION);
        List<SemanticRow> rows = semanticRows();
        assertEquals(48, rows.size());
        assertEquals(48, rows.stream().distinct().count());
        assertEquals(16, rows.stream().filter(row -> row.carriers == CarrierForm.ARRAYS).count());
        assertEquals(16, rows.stream().filter(row -> row.carriers == CarrierForm.SEGMENTS).count());
        assertEquals(16, rows.stream().filter(row -> row.carriers == CarrierForm.MIXED).count());

        for (SemanticRow row : rows) {
            Route route = route(row.rank, row.type, row.scenario, row.carriers, row.bias,
                    row.callerParallel);
            assertEquals(ExecutionStrategy.Compute.SCALAR, route.scalar.executionStrategy().compute(),
                    row.toString());
            assertNotEquals(63, route.scalar.classIdentitySchema(), row.toString());
            if (!row.scenario.vectorEligible()) {
                // The four excluded forms are evidence of the scalar fallback, not permission to
                // manufacture an unsupported schema-63 artifact.
                assertNull(route.vector, row.toString());
                executeScalarRow(row, route);
                continue;
            }
            assertNotNull(route.vector, row.toString());
            assertFalse(route.scalar.structuralKey().equals(route.vector.structuralKey()), row.toString());
            assertFalse(java.util.Arrays.equals(route.scalarBytes, route.vectorBytes), row.toString());
            assertEquals(63, route.vector.classIdentitySchema(), row.toString());
            assertEquals(ExecutionStrategy.VECTOR, route.vector.executionStrategy(), row.toString());
            assertVectorDataflow(route.vectorBytes, row.type, row.carriers, row.toString());
            executeEligibleRow(row, route);
        }
        retainSemanticInventory(rows);
    }

    @Test void stageBClassFileDossiersHaveExactlyTwentyFourStableForms() throws Exception {
        List<Dossier> dossiers = dossiers();
        assertEquals(24, dossiers.size());
        assertEquals(24, dossiers.stream().distinct().count());
        for (Dossier dossier : dossiers) {
            Route route = route(dossier.rank, dossier.type, Scenario.UNIT_INTERIOR_EXACT_WIDTH,
                    dossier.carriers, dossier.bias, false);
            byte[] again = generate(route.vector, route.ir);
            byte[] scalarAgain = generate(route.scalar, route.ir);
            assertArrayEquals(route.vectorBytes, again, dossier.toString());
            assertArrayEquals(route.scalarBytes, scalarAgain, dossier + " scalar deterministic control");
            assertEquals(sha256(route.vectorBytes), sha256(again), dossier.toString());
            assertEquals(sha256(route.scalarBytes), sha256(scalarAgain), dossier + " scalar hash");
            assertEquals(63, route.vector.classIdentitySchema(), dossier.toString());
            assertNotEquals(route.scalar.structuralKey(), route.vector.structuralKey(), dossier.toString());
            assertFalse(java.util.Arrays.equals(route.scalar.compatibilityBytes(),
                    route.vector.classIdentityBytes()), dossier.toString());
            assertVectorDataflow(route.vectorBytes, dossier.type, dossier.carriers, dossier.toString());
            assertScalarControl(route.scalarBytes, dossier.toString());
            if (dossier.rank == Rank.CONV2D && dossier.type == DataType.FLOAT32
                    && !dossier.bias && dossier.carriers == CarrierForm.ARRAYS) {
                assertEquals("61b2a3f334b4284ce399c97e826b3fcee4f6f69a3fad7ac85346d0bc6c7e87c7",
                        sha256(route.scalarBytes), "immutable Stage-A scalar class bytes");
                assertEquals("f59e7af94e680d8d6549e4c7e2fca6399da7168c195fe077ce6fd1d2f523a13a",
                        route.scalar.structuralKey(), "immutable Stage-A scalar identity");
            }
            retain(dossier, route);
        }
        retainStructuralInventory(dossiers);
    }

    private static List<SemanticRow> semanticRows() {
        List<SemanticRow> rows = new ArrayList<>();
        for (Rank rank : Rank.values()) for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (Scenario scenario : Scenario.values()) {
                CarrierForm carriers = CarrierForm.values()[scenario.ordinal() % CarrierForm.values().length];
                boolean bias = (scenario.ordinal() & 1) == 0;
                boolean parallel = scenario == Scenario.BATCH_GT_ONE || scenario == Scenario.GROUPED;
                rows.add(new SemanticRow(rank, type, scenario, carriers, bias, parallel));
            }
        return List.copyOf(rows);
    }

    private static List<Dossier> dossiers() {
        List<Dossier> result = new ArrayList<>();
        for (Rank rank : Rank.values()) for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (boolean bias : List.of(false, true)) for (CarrierForm carriers : CarrierForm.values())
                result.add(new Dossier(rank, type, bias, carriers));
        return List.copyOf(result);
    }

    private static Route route(Rank rank, DataType type, Scenario scenario, CarrierForm form,
            boolean bias, boolean parallel) throws Exception {
        var scalarPortable = new CpuPartitionPreparer().analyze(context(rank, type, scenario, form,
                bias, parallel, CpuPartitionAnalysisInputs.PortableExecutionConfig
                        .ComputePreference.SCALAR))
                .plan().units().getFirst().portablePlan();
        var selectedPortable = new CpuPartitionPreparer().analyze(context(rank, type, scenario,
                form, bias, parallel, CpuPartitionAnalysisInputs.PortableExecutionConfig
                        .ComputePreference.VECTOR_IF_ELIGIBLE))
                .plan().units().getFirst().portablePlan();
        CpuKernelSpecialization scalar = scalarPortable.specialization();
        CpuKernelIr ir = scalarPortable.kernelIr();
        if (!scenario.vectorEligible()) {
            assertEquals(scalar, selectedPortable.specialization());
            return new Route(scalar, ir, generate(scalar, ir), null, null);
        }
        CpuKernelSpecialization vector = selectedPortable.specialization();
        assertEquals(ir, selectedPortable.kernelIr());
        return new Route(scalar, ir, generate(scalar, ir), vector, generate(vector, ir));
    }

    /**
     * Executes scalar and production-vector entries over the same raw-bit stimulus. The
     * generated entry has one range argument, so the parallel matrix fact is represented by two
     * disjoint caller-owned invocations rather than by a made-up parallel schema.
     */
    private static void executeEligibleRow(SemanticRow row, Route route) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            Storage scalar = Storage.create(row, route.scalar, geometry(row, route), arena);
            Storage vector = scalar.copyForOutput(row, route.vector, arena);
            byte[][] immutable = scalar.inputSnapshots();
            MethodHandle scalarHandle = new CpuClassFileKernelGenerator().defineClassBytes(
                    route.scalar, route.scalarBytes).entryPoint();
            MethodHandle vectorHandle = new CpuClassFileKernelGenerator().defineClassBytes(
                    route.vector, route.vectorBytes).entryPoint();
            for (Range range : ranges(row, scalar.outputElements())) {
                invoke(scalarHandle, scalar.arguments(range));
                invoke(vectorHandle, vector.arguments(range));
            }
            assertRawOutputsEqual(row, scalar, vector);
            assertMatchesReference(row, scalar);
            scalar.assertInputsUnchanged(immutable, row.toString());
            vector.assertInputsUnchanged(immutable, row.toString());
        }
    }

    /** Executes each non-dense or width-ineligible row through the selected scalar entry. */
    private static void executeScalarRow(SemanticRow row, Route route) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            Storage scalar = Storage.create(row, route.scalar, geometry(row, route), arena);
            byte[][] immutable = scalar.inputSnapshots();
            MethodHandle handle = new CpuClassFileKernelGenerator().defineClassBytes(
                    route.scalar, route.scalarBytes).entryPoint();
            for (Range range : ranges(row, scalar.outputElements())) invoke(handle, scalar.arguments(range));
            assertMatchesReference(row, scalar);
            scalar.assertOutsideRangesAreSentinels(ranges(row, scalar.outputElements()), row.toString());
            scalar.assertInputsUnchanged(immutable, row.toString());
            assertRawStimulusContainsOrdinaryAndSpecialValues(row.type);
        }
    }

    private static long[] geometry(SemanticRow row, Route route) {
        return row.rank == Rank.CONV2D
                ? routeGeometry2d(row, route)
                : routeGeometry3d(row, route);
    }

    private static long[] routeGeometry2d(SemanticRow row, Route route) {
        var context = context(row.rank, row.type, row.scenario, row.carriers, row.bias,
                row.callerParallel, CpuPartitionAnalysisInputs.PortableExecutionConfig
                        .ComputePreference.SCALAR);
        return new CpuPartitionPreparer().analyze(context).plan().conv2dGeometry().orElseThrow()
                .pack(new long[route.scalar.carrierPattern().size()]);
    }

    private static long[] routeGeometry3d(SemanticRow row, Route route) {
        var context = context(row.rank, row.type, row.scenario, row.carriers, row.bias,
                row.callerParallel, CpuPartitionAnalysisInputs.PortableExecutionConfig
                        .ComputePreference.SCALAR);
        return new CpuPartitionPreparer().analyze(context).plan().units().getFirst()
                .conv3dGeometry().orElseThrow()
                .pack(new long[route.scalar.carrierPattern().size()]);
    }

    private record Range(long start, long end) { }

    private static List<Range> ranges(SemanticRow row, long count) {
        assertTrue(count >= 8, row.toString());
        if (row.scenario == Scenario.UNIT_INTERIOR_EXACT_WIDTH) {
            return List.of(new Range(7, 7), new Range(0, count));
        }
        long start = row.scenario == Scenario.PADDED_PROLOGUE_INTERIOR_EPILOGUE ? 0 : 1;
        long end = count - 1;
        if (!row.callerParallel) return List.of(new Range(start, end));
        long split = start + (end - start) / 2;
        return List.of(new Range(start, split), new Range(split, end));
    }

    private static void invoke(MethodHandle handle, List<Object> arguments) throws Throwable {
        handle.invokeWithArguments(arguments);
    }

    /** Per-invocation direct carrier fixture; offsets remain in the packed geometry. */
    private static final class Storage {
        private static final int FLOAT_SENTINEL = 0x7fa1_23bc;
        private static final long DOUBLE_SENTINEL = 0x7ff4_1234_5678_9abCL;
        private final Rank rank;
        private final DataType type;
        private final CpuKernelSpecialization specialization;
        private final Object[] carriers;
        private final long[] geometry;
        private final long outputElements;

        private Storage(Rank rank, DataType type, CpuKernelSpecialization specialization, Object[] carriers,
                long[] geometry, long outputElements) {
            this.rank = rank;
            this.type = type;
            this.specialization = specialization;
            this.carriers = carriers;
            this.geometry = geometry;
            this.outputElements = outputElements;
        }

        static Storage create(SemanticRow row, CpuKernelSpecialization specialization,
                long[] geometry, Arena arena) {
            int boundaries = specialization.carrierPattern().size();
            Object[] carriers = new Object[boundaries];
            int cursor = boundaries;
            long outputElements = 0;
            for (int boundary = 0; boundary < boundaries; boundary++) {
                int rank = boundary == boundaries - 1 ? (row.rank == Rank.CONV2D ? 4 : 5)
                        : boundary == 2 && row.bias ? 1 : (row.rank == Rank.CONV2D ? 4 : 5);
                long span = geometry[boundary];
                for (int axis = 0; axis < rank; axis++) {
                    long extent = geometry[cursor + axis];
                    long stride = geometry[cursor + rank + axis];
                    span = Math.addExact(span, Math.multiplyExact(extent - 1, stride));
                }
                cursor += rank * 2;
                if (boundary == boundaries - 1) {
                    outputElements = product(geometry, cursor - rank * 2, rank);
                }
                carriers[boundary] = allocate(specialization.carrierPattern().get(boundary),
                        Math.addExact(span, 5), row.type, arena);
                fill(carriers[boundary], row.type, boundary == boundaries - 1);
            }
            return new Storage(row.rank, row.type, specialization, carriers, geometry.clone(), outputElements);
        }

        Storage copyForOutput(SemanticRow row, CpuKernelSpecialization vector, Arena arena) {
            Object[] copy = new Object[carriers.length];
            for (int i = 0; i < copy.length; i++) {
                long elements = elements(carriers[i], type);
                copy[i] = allocate(vector.carrierPattern().get(i), elements, type, arena);
                copy(carriers[i], copy[i]);
            }
            return new Storage(rank, row.type, vector, copy, geometry.clone(), outputElements);
        }

        long outputElements() { return outputElements; }

        List<Object> arguments(Range range) {
            List<Object> result = new ArrayList<>(carriers.length + 3);
            java.util.Collections.addAll(result, carriers);
            result.add(geometry);
            result.add(range.start);
            result.add(range.end);
            return result;
        }

        byte[][] inputSnapshots() {
            byte[][] result = new byte[carriers.length - 1][];
            for (int i = 0; i < result.length; i++) {
                result[i] = bytes(carriers[i]);
            }
            return result;
        }

        void assertInputsUnchanged(byte[][] expected, String name) {
            for (int i = 0; i < expected.length; i++) {
                assertArrayEquals(expected[i], bytes(carriers[i]),
                        name + " immutable input/bias boundary=" + i);
            }
        }

        void assertOutsideRangesAreSentinels(List<Range> ranges, String name) {
            boolean[] written = new boolean[Math.toIntExact(outputElements)];
            for (Range range : ranges) {
                for (long i = range.start; i < range.end; i++) {
                    written[(int) i] = true;
                }
            }
            int boundary = carriers.length - 1;
            long base = geometry[boundary];
            for (int ordinal = 0; ordinal < written.length; ordinal++) {
                if (written[ordinal]) continue;
                long address = outputAddress(ordinal);
                if (type == DataType.FLOAT32) {
                    assertEquals(FLOAT_SENTINEL,
                            Float.floatToRawIntBits(readFloat(carriers[boundary], address)),
                            name + " sentinel " + ordinal);
                } else {
                    assertEquals(DOUBLE_SENTINEL,
                            Double.doubleToRawLongBits(readDouble(carriers[boundary], address)),
                            name + " sentinel " + ordinal);
                }
            }
        }

        private static Object allocate(CarrierAccess access, long elements, DataType type, Arena arena) {
            if (access == CarrierAccess.MEMORY_SEGMENT) return arena.allocate(
                    Math.multiplyExact(elements, type == DataType.FLOAT64 ? Double.BYTES : Float.BYTES),
                    Math.max(Float.BYTES, Double.BYTES));
            return access == CarrierAccess.DOUBLE_ARRAY ? new double[Math.toIntExact(elements)]
                    : new float[Math.toIntExact(elements)];
        }

        private static long elements(Object carrier, DataType type) {
            return carrier instanceof float[] values ? values.length
                    : carrier instanceof double[] values ? values.length
                    : ((MemorySegment) carrier).byteSize() / (type == DataType.FLOAT64 ? Double.BYTES : Float.BYTES);
        }

        private static void fill(Object carrier, DataType type, boolean output) {
            long count = elements(carrier, type);
            for (long index = 0; index < count; index++) {
                if (type == DataType.FLOAT32) writeFloat(carrier, index, output
                        ? Float.intBitsToFloat(FLOAT_SENTINEL) : floatValue(index));
                else writeDouble(carrier, index, output
                        ? Double.longBitsToDouble(DOUBLE_SENTINEL) : doubleValue(index));
            }
        }

        private static float floatValue(long index) {
            return switch ((int) (index % 11)) {
                case 0 -> 1.25f;
                case 1 -> -0.0f;
                case 2 -> Float.MIN_VALUE;
                case 3 -> Float.POSITIVE_INFINITY;
                case 4 -> Float.NEGATIVE_INFINITY;
                case 5 -> Float.intBitsToFloat(0x7fc0_0123);
                case 6 -> Float.MAX_VALUE;
                default -> (index % 19 - 9) * .03125f;
            };
        }

        private static double doubleValue(long index) {
            return switch ((int) (index % 11)) {
                case 0 -> 1.25;
                case 1 -> -0.0;
                case 2 -> Double.MIN_VALUE;
                case 3 -> Double.POSITIVE_INFINITY;
                case 4 -> Double.NEGATIVE_INFINITY;
                case 5 -> Double.longBitsToDouble(0x7ff8_0000_0000_0123L);
                case 6 -> Double.MAX_VALUE;
                default -> (index % 19 - 9) * .03125;
            };
        }

        private static void copy(Object source, Object target) {
            if (source instanceof float[] left && target instanceof float[] right) {
                System.arraycopy(left, 0, right, 0, left.length);
            } else if (source instanceof double[] left && target instanceof double[] right) {
                System.arraycopy(left, 0, right, 0, left.length);
            } else {
                ((MemorySegment) target).copyFrom((MemorySegment) source);
            }
        }

        private static byte[] bytes(Object carrier) {
            if (carrier instanceof float[] values) {
                return MemorySegment.ofArray(values).toArray(ValueLayout.JAVA_BYTE);
            }
            if (carrier instanceof double[] values) {
                return MemorySegment.ofArray(values).toArray(ValueLayout.JAVA_BYTE);
            }
            return ((MemorySegment) carrier).toArray(ValueLayout.JAVA_BYTE);
        }

        private static void writeFloat(Object carrier, long index, float value) {
            if (carrier instanceof float[] values) {
                values[Math.toIntExact(index)] = value;
            } else {
                ((MemorySegment) carrier).set(ValueLayout.JAVA_FLOAT_UNALIGNED
                        .withOrder(ByteOrder.nativeOrder()), index * Float.BYTES, value);
            }
        }
        private static void writeDouble(Object carrier, long index, double value) {
            if (carrier instanceof double[] values) {
                values[Math.toIntExact(index)] = value;
            } else {
                ((MemorySegment) carrier).set(ValueLayout.JAVA_DOUBLE_UNALIGNED
                        .withOrder(ByteOrder.nativeOrder()), index * Double.BYTES, value);
            }
        }
        private static float readFloat(Object carrier, long index) {
            return carrier instanceof float[] values ? values[Math.toIntExact(index)]
                    : ((MemorySegment) carrier).get(ValueLayout.JAVA_FLOAT_UNALIGNED
                            .withOrder(ByteOrder.nativeOrder()), index * Float.BYTES);
        }
        private static double readDouble(Object carrier, long index) {
            return carrier instanceof double[] values ? values[Math.toIntExact(index)]
                    : ((MemorySegment) carrier).get(ValueLayout.JAVA_DOUBLE_UNALIGNED
                            .withOrder(ByteOrder.nativeOrder()), index * Double.BYTES);
        }
        private static long product(long[] geometry, int extentStart, int rank) {
            long result = 1;
            for (int i = 0; i < rank; i++) {
                result = Math.multiplyExact(result, geometry[extentStart + i]);
            }
            return result;
        }

        private double[] representedCarrier(int boundary) {
            long count = elements(carriers[boundary], type);
            double[] result = new double[Math.toIntExact(count)];
            for (int index = 0; index < result.length; index++) {
                result[index] = type == DataType.FLOAT32
                        ? readFloat(carriers[boundary], index)
                        : readDouble(carriers[boundary], index);
            }
            return result;
        }

        private long outputAddress(long ordinal) {
            // Conv ranks are encoded by the output extent count; the preceding boundary count is
            // also the first geometry cursor, so walk the compact layout rather than assuming a
            // contiguous output for the non-dense fallback row.
            int boundaries = carriers.length;
            int outputRank = rank == Rank.CONV2D ? 4 : 5;
            int cursor = boundaries;
            for (int boundary = 0; boundary < boundaries - 1; boundary++) {
                int boundaryRank = boundary == 2 && boundaries == 4 ? 1 : outputRank;
                cursor += boundaryRank * 2;
            }
            long address = geometry[boundaries - 1];
            for (int axis = outputRank - 1; axis >= 0; axis--) {
                long extent = geometry[cursor + axis];
                long coordinate = ordinal % extent;
                ordinal /= extent;
                address = Math.addExact(address, Math.multiplyExact(coordinate,
                        geometry[cursor + outputRank + axis]));
            }
            return address;
        }
    }

    private record ReferenceInputs(double[][] values, long[][] extents, long[] offsets,
            long[][] strides, long[] outputExtents, long groups) { }

    private static void assertMatchesReference(SemanticRow row, Storage actual) {
        ReferenceInputs inputs = referenceInputs(row, actual);
        List<DataType> types = java.util.Collections.nCopies(inputs.values.length, row.type);
        double[] expected = row.rank == Rank.CONV2D
                ? CpuConv2dReferenceKernel.evaluate(types, row.type, inputs.values, inputs.extents,
                        inputs.offsets, inputs.strides, inputs.outputExtents,
                        attrs2d(row.scenario, Math.toIntExact(inputs.groups)))
                : CpuConv3dReferenceKernel.evaluate(types, row.type, inputs.values, inputs.extents,
                        inputs.offsets, inputs.strides, inputs.outputExtents,
                        attrs3d(row.scenario, Math.toIntExact(inputs.groups)));
        Object output = actual.carriers[actual.carriers.length - 1];
        List<Range> ranges = ranges(row, actual.outputElements());
        for (int ordinal = 0; ordinal < expected.length; ordinal++) {
            if (!isWritten(ordinal, ranges)) continue;
            long address = actual.outputAddress(ordinal);
            if (row.type == DataType.FLOAT32) {
                float wanted = (float) expected[ordinal];
                float observed = Storage.readFloat(output, address);
                if (!Float.isNaN(wanted) || !Float.isNaN(observed)) {
                    assertEquals(Float.floatToRawIntBits(wanted), Float.floatToRawIntBits(observed),
                            row + " reference ordinal " + ordinal);
                }
            } else {
                double wanted = expected[ordinal];
                double observed = Storage.readDouble(output, address);
                if (!Double.isNaN(wanted) || !Double.isNaN(observed)) {
                    assertEquals(Double.doubleToRawLongBits(wanted),
                            Double.doubleToRawLongBits(observed),
                            row + " reference ordinal " + ordinal);
                }
            }
        }
    }

    private static boolean isWritten(long ordinal, List<Range> ranges) {
        return ranges.stream().anyMatch(range -> ordinal >= range.start && ordinal < range.end);
    }

    private static ReferenceInputs referenceInputs(SemanticRow row, Storage storage) {
        int boundaries = storage.carriers.length;
        int rank = row.rank == Rank.CONV2D ? 4 : 5;
        int inputCount = boundaries - 1;
        double[][] values = new double[inputCount][];
        long[][] extents = new long[inputCount][];
        long[][] strides = new long[inputCount][];
        long[] offsets = new long[inputCount];
        int cursor = boundaries;
        for (int boundary = 0; boundary < boundaries; boundary++) {
            int boundaryRank = row.bias && boundary == 2 ? 1 : rank;
            long[] boundaryExtents = java.util.Arrays.copyOfRange(storage.geometry, cursor,
                    cursor + boundaryRank);
            long[] boundaryStrides = java.util.Arrays.copyOfRange(storage.geometry,
                    cursor + boundaryRank, cursor + boundaryRank * 2);
            if (boundary < inputCount) {
                values[boundary] = storage.representedCarrier(boundary);
                extents[boundary] = boundaryExtents;
                strides[boundary] = boundaryStrides;
                offsets[boundary] = storage.geometry[boundary];
            }
            cursor += boundaryRank * 2;
            if (boundary == boundaries - 1) {
                long groups = extents[0][1] / extents[1][1];
                return new ReferenceInputs(values, extents, offsets, strides, boundaryExtents,
                        groups);
            }
        }
        throw new AssertionError("missing output boundary");
    }

    private static void assertRawOutputsEqual(SemanticRow row, Storage scalar, Storage vector) {
        Object scalarOutput = scalar.carriers[scalar.carriers.length - 1];
        Object vectorOutput = vector.carriers[vector.carriers.length - 1];
        for (long ordinal = 0; ordinal < scalar.outputElements(); ordinal++) {
            long scalarAddress = scalar.outputAddress(ordinal);
            long vectorAddress = vector.outputAddress(ordinal);
            if (row.type == DataType.FLOAT32) {
                float expected = Storage.readFloat(scalarOutput, scalarAddress);
                float actual = Storage.readFloat(vectorOutput, vectorAddress);
                if (!Float.isNaN(expected) || !Float.isNaN(actual)) {
                    assertEquals(Float.floatToRawIntBits(expected), Float.floatToRawIntBits(actual),
                            row + " output ordinal " + ordinal);
                }
            } else {
                double expected = Storage.readDouble(scalarOutput, scalarAddress);
                double actual = Storage.readDouble(vectorOutput, vectorAddress);
                if (!Double.isNaN(expected) || !Double.isNaN(actual)) {
                    assertEquals(Double.doubleToRawLongBits(expected),
                            Double.doubleToRawLongBits(actual),
                            row + " output ordinal " + ordinal);
                }
            }
        }
        scalar.assertOutsideRangesAreSentinels(ranges(row, scalar.outputElements()), row.toString());
        vector.assertOutsideRangesAreSentinels(ranges(row, vector.outputElements()), row.toString());
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> context(Rank rank, DataType type,
            Scenario scenario, CarrierForm form, boolean bias, boolean parallel,
            CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference preference) {
        int inputChannels = scenario == Scenario.DEPTHWISE ? 4 : 4;
        int groups = scenario == Scenario.GROUPED ? 2 : scenario == Scenario.DEPTHWISE ? 4 : 1;
        int channelsPerGroup = inputChannels / groups;
        List<DataType> types = bias ? List.of(type, type, type) : List.of(type, type);
        List<CarrierAccess> carriers = carrierPattern(type, form, bias);
        int ranges = parallel ? 2 : 1;
        CpuPartitionAnalysisInputs inputs = new CpuPartitionAnalysisInputs(false, carriers,
                new CpuPartitionAnalysisInputs.PortableExecutionConfig(
                        preference,
                        ranges, ranges, 1));
        if (rank == Rank.CONV2D) {
            Shape x = Shape.of(scenario == Scenario.BATCH_GT_ONE ? 2 : 1, inputChannels,
                    scenario == Scenario.SHORT_WIDTH ? 3 : 6,
                    scenario == Scenario.SHORT_WIDTH ? 2 : 18);
            Shape w = Shape.of(inputChannels, channelsPerGroup, 2, scenario == Scenario.SHORT_WIDTH ? 2 : 3);
            Conv2dAttrs attrs = attrs2d(scenario, groups);
            Shape y = output2d(x, w, attrs);
            var base = CpuConv2dLoweringTest.context(types, x, w, y, attrs,
                    scenario == Scenario.NON_DENSE ? nonDense2d(x, w, y, bias) : offsetLayouts2d(x, w, y, bias));
            return withInputs(base, inputs);
        }
        Shape x = Shape.of(scenario == Scenario.BATCH_GT_ONE ? 2 : 1, inputChannels,
                scenario == Scenario.SHORT_WIDTH ? 3 : 5, 6, scenario == Scenario.SHORT_WIDTH ? 2 : 18);
        Shape w = Shape.of(inputChannels, channelsPerGroup, 2, 2, scenario == Scenario.SHORT_WIDTH ? 2 : 3);
        Conv3dAttrs attrs = attrs3d(scenario, groups);
        Shape y = output3d(x, w, attrs);
        var base = CpuConv3dLoweringTest.context(types, x, w, y, attrs,
                scenario == Scenario.NON_DENSE ? nonDense3d(x, w, y, bias) : offsetLayouts3d(x, w, y, bias));
        return withInputs(base, inputs);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> withInputs(
            PrepareContext<CpuPartitionAnalysisInputs> base, CpuPartitionAnalysisInputs inputs) {
        return new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), base.constants(), inputs);
    }

    /** Generates directly for class-file and semantic inspection. */
    private static byte[] generate(CpuKernelSpecialization specialization, CpuKernelIr ir) {
        var generator = new CpuClassFileKernelGenerator();
        return generator.generateClassBytes(specialization, ir);
    }

    /** Records only direct call owners, making forbidden hot-path dependencies reviewable. */
    private static String members(byte[] bytes) {
        return StreamSupport.stream(ClassFile.of().parse(bytes).constantPool().spliterator(), false)
                .filter(MemberRefEntry.class::isInstance).map(MemberRefEntry.class::cast)
                .map(member -> member.owner().asInternalName() + '.' + member.name().stringValue())
                .sorted().collect(java.util.stream.Collectors.joining("\n"));
    }

    private static void assertVectorDataflow(byte[] bytes, DataType type, CarrierForm form,
            String name) {
        assertTrue(ClassFile.of().verify(bytes).isEmpty(), name);
        var model = ClassFile.of().parse(bytes);
        assertAll(name,
                () -> assertTrue(model.fields().isEmpty()),
                () -> assertEquals(1, model.methods().size()),
                () -> assertFalse(model.methods().getFirst().methodTypeSymbol().descriptorString()
                        .contains("Ljava/lang/Object;")));
        var code = model.methods().getFirst().code().orElseThrow();
        String calls = members(bytes);
        String vector = type == DataType.FLOAT32 ? "jdk/incubator/vector/FloatVector"
                : "jdk/incubator/vector/DoubleVector";
        assertAll(name,
                () -> assertTrue(calls.contains(vector + ".broadcast")),
                () -> assertTrue(calls.contains(vector + ".mul")),
                () -> assertTrue(calls.contains(vector + ".add")),
                () -> assertTrue(calls.contains(vector + ".from" + (form == CarrierForm.ARRAYS ? "Array" : "MemorySegment"))
                        || calls.contains(vector + ".fromArray")),
                () -> assertTrue(calls.contains(vector + ".into" + (form == CarrierForm.ARRAYS ? "Array" : "MemorySegment"))
                        || calls.contains(vector + ".intoArray")),
                () -> assertFalse(calls.contains(".fma")),
                () -> assertFalse(calls.contains(".reduce")),
                () -> assertFalse(calls.contains(".lane")),
                () -> assertFalse(calls.contains(".gather")),
                () -> assertFalse(calls.contains(".scatter")),
                () -> assertFalse(calls.contains("VectorMask")),
                () -> assertFalse(calls.contains("synaptik")),
                () -> assertFalse(calls.contains("java/lang/reflect")),
                () -> assertFalse(calls.contains("java/lang/invoke")),
                () -> assertFalse(calls.contains("java/util/")),
                () -> assertFalse(calls.contains("java/lang/String")),
                () -> assertFalse(calls.contains(".valueOf")),
                () -> assertNoForbiddenOpcodes(code, name));
    }

    private static void assertNoForbiddenOpcodes(java.lang.classfile.CodeModel code, String name) {
        for (Instruction instruction : code.elementStream()
                .filter(Instruction.class::isInstance)
                .map(Instruction.class::cast)
                .toList()) {
            Opcode opcode = instruction.opcode();
            assertFalse(opcode == Opcode.NEW
                    || opcode == Opcode.ANEWARRAY
                    || opcode == Opcode.NEWARRAY
                    || opcode == Opcode.MULTIANEWARRAY
                    || opcode == Opcode.INVOKEDYNAMIC
                    || opcode == Opcode.MONITORENTER
                    || opcode == Opcode.MONITOREXIT,
                    name + " forbidden opcode " + opcode);
        }
    }

    private static void assertScalarControl(byte[] bytes, String name) {
        String calls = members(bytes);
        assertAll(name, () -> assertFalse(calls.contains("jdk/incubator/vector")),
                () -> assertFalse(calls.contains("synaptik")));
    }

    private static void assertRawStimulusContainsOrdinaryAndSpecialValues(DataType type) {
        if (type == DataType.FLOAT32) {
            int[] bits = {Float.floatToRawIntBits(1.25f), 0, 0x80000000,
                    Float.floatToRawIntBits(Float.POSITIVE_INFINITY), Float.floatToRawIntBits(Float.NaN),
                    Float.floatToRawIntBits(Float.MIN_VALUE), Float.floatToRawIntBits(Float.MAX_VALUE)};
            assertEquals(7, java.util.Arrays.stream(bits).distinct().count());
        } else {
            long[] bits = {Double.doubleToRawLongBits(1.25), 0L, Long.MIN_VALUE,
                    Double.doubleToRawLongBits(Double.POSITIVE_INFINITY), Double.doubleToRawLongBits(Double.NaN),
                    Double.doubleToRawLongBits(Double.MIN_VALUE), Double.doubleToRawLongBits(Double.MAX_VALUE)};
            assertEquals(7, java.util.Arrays.stream(bits).distinct().count());
        }
    }

    private static void retain(Dossier dossier, Route route) throws IOException {
        String configured = System.getProperty(EVIDENCE_ROOT_PROPERTY);
        if (configured == null || configured.isBlank()) return;
        Path root = Path.of(configured).toAbsolutePath().normalize();
        Path checkout = Path.of("").toAbsolutePath().normalize();
        if (root.startsWith(checkout)) throw new IOException("evidence retention root must be outside checkout");
        Path directory = root.resolve("stage-b-dossiers");
        Files.createDirectories(directory);
        String stem = dossier.rank + "-" + dossier.type + "-bias-" + dossier.bias + "-" + dossier.carriers;
        retainClass(directory, stem + "-scalar", route.scalar, route.scalarBytes);
        retainClass(directory, stem + "-vector", route.vector, route.vectorBytes);
        // javap is deliberately opt-in together with retention: normal focused tests never write
        // generated artifacts into the checkout.
        try {
            for (String classForm : List.of("scalar", "vector")) {
                Process process = new ProcessBuilder("javap", "-c", "-v", "-p",
                        directory.resolve(stem + "-" + classForm + ".class").toString())
                        .redirectErrorStream(true)
                        .redirectOutput(directory.resolve(stem + "-" + classForm + ".javap")
                                .toFile())
                        .start();
                int exitCode = process.waitFor();
                if (exitCode != 0) {
                    throw new IOException("javap failed for " + stem + "-" + classForm
                            + " with exit code " + exitCode);
                }
            }
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw new IOException("interrupted while retaining javap dossier", interrupted);
        }
    }

    private static void retainSemanticInventory(List<SemanticRow> rows) throws IOException {
        Path root = evidenceRoot();
        if (root == null) return;
        StringBuilder csv = new StringBuilder(
                "rank,type,scenario,carriers,bias,orchestration,admission\n");
        for (SemanticRow row : rows) {
            csv.append(row.rank).append(',').append(row.type).append(',')
                    .append(row.scenario).append(',').append(row.carriers).append(',')
                    .append(row.bias).append(',')
                    .append(row.callerParallel ? "PARALLEL" : "SINGLE_THREAD").append(',')
                    .append(row.scenario.vectorEligible() ? "SCHEMA63_VECTOR" : "SCALAR_FALLBACK")
                    .append('\n');
        }
        Path inventory = root.resolve("stage-b-semantic-inventory.csv");
        Files.createDirectories(root);
        Files.writeString(inventory, csv.toString());
        Files.writeString(root.resolve("stage-b-semantic-inventory.sha256"),
                sha256(Files.readAllBytes(inventory)) + "\n");
    }

    private static void retainStructuralInventory(List<Dossier> dossiers) throws IOException {
        Path root = evidenceRoot();
        if (root == null) return;
        Path directory = root.resolve("stage-b-dossiers");
        StringBuilder csv = new StringBuilder("rank,type,bias,carriers,scalar_sha256,vector_sha256\n");
        for (Dossier dossier : dossiers) {
            String stem = dossier.rank + "-" + dossier.type + "-bias-" + dossier.bias + "-"
                    + dossier.carriers;
            csv.append(dossier.rank).append(',').append(dossier.type).append(',')
                    .append(dossier.bias).append(',').append(dossier.carriers).append(',')
                    .append(Files.readString(directory.resolve(stem + "-scalar.sha256")).trim())
                    .append(',')
                    .append(Files.readString(directory.resolve(stem + "-vector.sha256")).trim())
                    .append('\n');
        }
        Files.writeString(directory.resolve("inventory.csv"), csv.toString());
        writeManifest(directory);
    }

    private static Path evidenceRoot() throws IOException {
        String configured = System.getProperty(EVIDENCE_ROOT_PROPERTY);
        if (configured == null || configured.isBlank()) return null;
        Path root = Path.of(configured).toAbsolutePath().normalize();
        Path checkout = Path.of("").toAbsolutePath().normalize();
        if (root.startsWith(checkout)) {
            throw new IOException("evidence retention root must be outside checkout");
        }
        return root;
    }

    private static void writeManifest(Path directory) throws IOException {
        StringBuilder manifest = new StringBuilder();
        try (var paths = Files.walk(directory)) {
            for (Path file : paths.filter(Files::isRegularFile).sorted().toList()) {
                if (file.getFileName().toString().equals("manifest.sha256")) continue;
                manifest.append(sha256(Files.readAllBytes(file))).append("  ")
                        .append(directory.relativize(file)).append('\n');
            }
        }
        Files.writeString(directory.resolve("manifest.sha256"), manifest.toString());
    }

    private static void retainClass(Path directory, String stem, CpuKernelSpecialization specialization,
            byte[] bytes) throws IOException {
        var model = ClassFile.of().parse(bytes);
        Files.write(directory.resolve(stem + ".class"), bytes);
        Files.writeString(directory.resolve(stem + ".sha256"), sha256(bytes) + "\n");
        Files.writeString(directory.resolve(stem + ".identity"), specialization.structuralKey() + "\n");
        Files.writeString(directory.resolve(stem + ".descriptor"), specialization.entryType().descriptorString() + "\n");
        Files.writeString(directory.resolve(stem + ".inventory"), "fields=" + model.fields().size()
                + " methods=" + model.methods().size() + "\n" + members(bytes) + "\n");
    }

    private static String sha256(byte[] bytes) {
        try { return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes)); }
        catch (Exception impossible) { throw new AssertionError(impossible); }
    }

    private static List<CarrierAccess> carrierPattern(DataType type, CarrierForm form, boolean bias) {
        CarrierAccess array = type == DataType.FLOAT32 ? CarrierAccess.FLOAT_ARRAY : CarrierAccess.DOUBLE_ARRAY;
        List<CarrierAccess> result = new ArrayList<>();
        result.add(form == CarrierForm.SEGMENTS ? CarrierAccess.MEMORY_SEGMENT : array);
        result.add(form == CarrierForm.ARRAYS ? array : CarrierAccess.MEMORY_SEGMENT);
        if (bias) result.add(form == CarrierForm.MIXED ? CarrierAccess.MEMORY_SEGMENT
                : form == CarrierForm.SEGMENTS ? CarrierAccess.MEMORY_SEGMENT : array);
        result.add(form == CarrierForm.SEGMENTS ? CarrierAccess.MEMORY_SEGMENT : array);
        return List.copyOf(result);
    }

    private static Conv2dAttrs attrs2d(Scenario s, int groups) {
        return switch (s) {
            case PADDED_PROLOGUE_INTERIOR_EPILOGUE -> new Conv2dAttrs(1, 1, 1, 1, 1, 1, groups);
            case STRIDE_HEIGHT_OR_DEPTH -> new Conv2dAttrs(2, 1, 0, 0, 1, 1, groups);
            case DILATION_HEIGHT_OR_DEPTH -> new Conv2dAttrs(1, 1, 0, 0, 2, 1, groups);
            case SCALAR_WIDTH_STRIDE -> new Conv2dAttrs(1, 2, 0, 0, 1, 1, groups);
            case SCALAR_WIDTH_DILATION -> new Conv2dAttrs(1, 1, 0, 0, 1, 2, groups);
            default -> new Conv2dAttrs(1, 1, 0, 0, 1, 1, groups);
        };
    }

    private static Conv3dAttrs attrs3d(Scenario s, int groups) {
        return switch (s) {
            case PADDED_PROLOGUE_INTERIOR_EPILOGUE -> new Conv3dAttrs(1, 1, 1, 1, 1, 1, 1, 1, 1, groups);
            case STRIDE_HEIGHT_OR_DEPTH -> new Conv3dAttrs(2, 2, 1, 0, 0, 0, 1, 1, 1, groups);
            case DILATION_HEIGHT_OR_DEPTH -> new Conv3dAttrs(2, 2, 1, 0, 0, 0, 2, 2, 1, groups);
            case SCALAR_WIDTH_STRIDE -> new Conv3dAttrs(1, 1, 2, 0, 0, 0, 1, 1, 1, groups);
            case SCALAR_WIDTH_DILATION -> new Conv3dAttrs(1, 1, 1, 0, 0, 0, 1, 1, 2, groups);
            default -> new Conv3dAttrs(1, 1, 1, 0, 0, 0, 1, 1, 1, groups);
        };
    }

    private static Shape output2d(Shape x, Shape w, Conv2dAttrs a) {
        long[] xe = x.toLongArray(), we = w.toLongArray();
        return Shape.of(xe[0], we[0], (xe[2] + 2 * a.paddingHeight() - a.dilationHeight() * (we[2] - 1) - 1) / a.strideHeight() + 1,
                (xe[3] + 2 * a.paddingWidth() - a.dilationWidth() * (we[3] - 1) - 1) / a.strideWidth() + 1);
    }

    private static Shape output3d(Shape x, Shape w, Conv3dAttrs a) {
        long[] xe = x.toLongArray(), we = w.toLongArray();
        return Shape.of(xe[0], we[0], (xe[2] + 2 * a.paddingDepth() - a.dilationDepth() * (we[2] - 1) - 1) / a.strideDepth() + 1,
                (xe[3] + 2 * a.paddingHeight() - a.dilationHeight() * (we[3] - 1) - 1) / a.strideHeight() + 1,
                (xe[4] + 2 * a.paddingWidth() - a.dilationWidth() * (we[4] - 1) - 1) / a.strideWidth() + 1);
    }

    private static List<LayoutDescriptor> offsetLayouts2d(Shape x, Shape w, Shape y, boolean bias) {
        List<LayoutDescriptor> result = new ArrayList<>(List.of(offset(x, 3), offset(w, 5)));
        if (bias) result.add(offset(Shape.of(y.toLongArray()[1]), 2));
        result.add(offset(y, 7));
        return result;
    }

    private static List<LayoutDescriptor> offsetLayouts3d(Shape x, Shape w, Shape y, boolean bias) {
        List<LayoutDescriptor> result = new ArrayList<>(List.of(offset(x, 3), offset(w, 5)));
        if (bias) result.add(offset(Shape.of(y.toLongArray()[1]), 2));
        result.add(offset(y, 7));
        return result;
    }

    private static List<LayoutDescriptor> nonDense2d(Shape x, Shape w, Shape y,
            boolean bias) {
        return nonDense(x, w, y, bias);
    }

    private static List<LayoutDescriptor> nonDense3d(Shape x, Shape w, Shape y,
            boolean bias) {
        return nonDense(x, w, y, bias);
    }

    private static List<LayoutDescriptor> nonDense(Shape x, Shape w, Shape y, boolean bias) {
        List<LayoutDescriptor> result = new ArrayList<>(List.of(nonDense(x, 3), nonDense(w, 5)));
        if (bias) result.add(offset(Shape.of(y.toLongArray()[1]), 2));
        result.add(nonDense(y, 7));
        return result;
    }

    private static LayoutDescriptor offset(Shape shape, long offset) {
        long[] extents = shape.toLongArray();
        long[] strides = new long[extents.length];
        long next = 1;
        for (int i = extents.length - 1; i >= 0; i--) {
            strides[i] = next;
            next = Math.multiplyExact(next, extents[i]);
        }
        return LayoutDescriptor.of(shape, strides, offset, true);
    }

    private static LayoutDescriptor nonDense(Shape shape, long offset) {
        long[] extents = shape.toLongArray();
        long[] strides = new long[extents.length];
        long next = 2;
        for (int i = extents.length - 1; i >= 0; i--) {
            strides[i] = next;
            next = Math.multiplyExact(next, extents[i]);
        }
        return LayoutDescriptor.of(shape, strides, offset, true);
    }
}
