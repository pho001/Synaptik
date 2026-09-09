package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.index.SelectAttrs;
import io.github.pho001.synaptik.model.operation.index.SelectKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.ContiguousKind;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.layout.SliceAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceKind;
import io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.foreign.MemorySegment;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.TreeSet;
import org.junit.jupiter.api.Test;

/** Finite provider-to-preparer affine coverage basis for CPU 0009. */
class CpuAffineGeneratedCoverageFixtureTest {
    private static final List<DataType> TYPES = List.of(DataType.FLOAT64, DataType.FLOAT32,
            DataType.BFLOAT16, DataType.INT64, DataType.INT32, DataType.BOOL);
    private static final List<PortableExecutionConfig> REQUESTS = List.of(
            new PortableExecutionConfig(ComputePreference.SCALAR, 1, 1, 1),
            new PortableExecutionConfig(ComputePreference.VECTOR_IF_ELIGIBLE, 1, 1, 1),
            new PortableExecutionConfig(ComputePreference.SCALAR, 4, 4, 1),
            new PortableExecutionConfig(ComputePreference.VECTOR_IF_ELIGIBLE, 4, 4, 1));

    static List<PrepareContext<CpuPartitionAnalysisInputs>> coverageFixtureContexts() {
        List<PrepareContext<CpuPartitionAnalysisInputs>> rows = new ArrayList<>();
        for (Base base : bases()) for (DataType type : TYPES) for (CarrierMode carrier : CarrierMode.values())
            for (PortableExecutionConfig request : REQUESTS) for (boolean enabled : List.of(false, true))
                rows.add(context(base.operation(), typed(base.input(), type), typed(base.output(), type),
                        carrier.pattern(type), request, enabled));
        return List.copyOf(rows);
    }

    @Test void completeAffineMatrixIsProviderAndPreparerDerived() {
        var rows = coverageFixtureContexts();
        assertEquals(1_536, rows.size());
        for (int index = 0; index < rows.size(); index++) {
            var context = rows.get(index); var node = context.nodes().getFirst();
            assertTrue(new CpuCapabilityProvider().supports(new OperationCapabilityQuery(node.operation(),
                    List.of(context.values().getFirst().descriptor()), List.of(context.values().get(1).descriptor()))),
                    "provider affine row " + index);
            try { assertEquals(1, new CpuPartitionPreparer().analyze(context).plan().units().size()); }
            catch (IllegalArgumentException failure) { throw new AssertionError("preparer affine row " + index, failure); }
        }
    }

    /**
     * Defines and invokes the actual generated entry for every finite affine inventory row.
     *
     * <p>The expected value is deliberately calculated from the operation's coordinate mapping,
     * rather than from prepared address pairs or generated bytes.  Consequently this catches a
     * jointly consistent lowering/code-generation mistake, not merely a missing generated
     * class.  The matrix includes the requested strategy and materialization policy even where
     * those facts normalize to the same entry body.</p>
     */
    @Test void everyAffineInventoryRowDefinesInvokesAndMatchesCoordinateOracle() throws Throwable {
        int invoked = 0;
        Set<String> owners = new TreeSet<>();
        for (Base base : bases()) for (DataType type : TYPES) for (CarrierMode carrier : CarrierMode.values())
            for (PortableExecutionConfig request : REQUESTS) for (boolean enabled : List.of(false, true)) {
                var context = context(base.operation(), typed(base.input(), type), typed(base.output(), type),
                        carrier.pattern(type), request, enabled);
                var analysis = new CpuPartitionPreparer().analyze(context);
                var route = analysis.plan().units().getFirst().portablePlan();
                var generator = new CpuClassFileKernelGenerator();
                var entry = generator.defineClassBytes(route.specialization(),
                        generator.generateClassBytes(route.specialization(), route.kernelIr())).entryPoint();
                Object source = values(type, 32);
                Object output = zeros(type, 32);
                // The type-specific segment conversion is intentionally outside the oracle.
                Object inputArgument = carrier == CarrierMode.SEGMENT || carrier == CarrierMode.SEGMENT_TO_HEAP
                        ? segment(source) : source;
                Object outputArgument = carrier == CarrierMode.SEGMENT || carrier == CarrierMode.HEAP_TO_SEGMENT
                        ? segment(output) : output;
                try {
                    entry.invokeWithArguments(inputArgument, outputArgument, analysis.plan().affineAddressPairs(),
                            0L, analysis.plan().elementCount());
                } catch (Throwable failure) {
                    throw new AssertionError("generated affine invocation failed: " + base.operation().kind()
                            + '/' + type + '/' + carrier + '/' + request + "/materialization=" + enabled, failure);
                }
                assertAffineOracle(type, base, source, output);
                assertTrue(owners.add("affine-matrix:" + invoked), "duplicate affine owner " + invoked);
                invoked++;
            }
        assertEquals(1_536, invoked);
        assertEquals(inventoryOwners(), owners, "missing, duplicate, orphaned, or stale affine owner");
    }

    private static Set<String> inventoryOwners() throws Exception {
        String inventory;
        try (InputStream stream = CpuAffineGeneratedCoverageFixtureTest.class.getResourceAsStream(
                "/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/generated-coverage-inventory.tsv")) {
            assertTrue(stream != null, "generated coverage inventory");
            inventory = new String(stream.readAllBytes(), StandardCharsets.UTF_8);
        }
        assertEquals("527f36de41b64c228dd215c6f3182138b4c70db24c915117c4d7f82743ac41cc",
                java.util.HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256")
                        .digest(inventory.getBytes(StandardCharsets.UTF_8))), "inventory SHA-256");
        Set<String> owners = new TreeSet<>();
        for (String line : inventory.split("\\n")) {
            String[] row = line.split("\\t", -1);
            if (row.length == 27 && row[0].startsWith("affine-matrix:")) {
                assertTrue(owners.add(row[0]), "duplicate affine inventory owner " + row[0]);
            }
        }
        assertEquals(1_536, owners.size(), "affine inventory owner count");
        return owners;
    }

    private static Object values(DataType type, int length) {
        return switch (type) {
            case FLOAT64 -> { double[] result = new double[length]; for (int i = 0; i < length; i++) result[i] = i + .25d; yield result; }
            case FLOAT32 -> { float[] result = new float[length]; for (int i = 0; i < length; i++) result[i] = i + .25f; yield result; }
            case BFLOAT16 -> { short[] result = new short[length]; for (int i = 0; i < length; i++) result[i] = (short) (0x4000 + i); yield result; }
            case INT64 -> { long[] result = new long[length]; for (int i = 0; i < length; i++) result[i] = 10_000L + i; yield result; }
            case INT32 -> { int[] result = new int[length]; for (int i = 0; i < length; i++) result[i] = 10_000 + i; yield result; }
            case BOOL -> { byte[] result = new byte[length]; for (int i = 0; i < length; i++) result[i] = (byte) (i & 1); yield result; }
        };
    }

    private static Object zeros(DataType type, int length) {
        return switch (type) {
            case FLOAT64 -> new double[length]; case FLOAT32 -> new float[length]; case BFLOAT16 -> new short[length];
            case INT64 -> new long[length]; case INT32 -> new int[length]; case BOOL -> new byte[length];
        };
    }

    private static MemorySegment segment(Object array) {
        if (array instanceof double[] value) return MemorySegment.ofArray(value);
        if (array instanceof float[] value) return MemorySegment.ofArray(value);
        if (array instanceof short[] value) return MemorySegment.ofArray(value);
        if (array instanceof long[] value) return MemorySegment.ofArray(value);
        if (array instanceof int[] value) return MemorySegment.ofArray(value);
        return MemorySegment.ofArray((byte[]) array);
    }

    private static void assertAffineOracle(DataType type, Base base, Object source, Object output) {
        long[] outputShape = base.output().shape().dimensions().stream()
                .mapToLong(dimension -> dimension.staticSize().orElseThrow()).toArray();
        long[] outputStrides = base.output().layout().orElseThrow().strides();
        long outputOffset = base.output().layout().orElseThrow().storageOffset();
        for (long linear = 0; linear < base.output().shape().knownElementCount().orElseThrow(); linear++) {
            long[] coordinate = coordinates(linear, outputShape);
            long sourceIndex = sourceIndex(base.operation(), linear, coordinate);
            long targetIndex = outputOffset;
            for (int axis = 0; axis < coordinate.length; axis++) targetIndex += coordinate[axis] * outputStrides[axis];
            assertRawEquals(type, source, (int) sourceIndex, output, (int) targetIndex,
                    base.operation().kind().name() + " logical=" + linear);
        }
    }

    private static long sourceIndex(Operation operation, long linear, long[] out) {
        return switch (operation.kind().name()) {
            case "CONTIGUOUS" -> out[0] + 2 * out[1];
            // Reshape and rank-only transforms retain row-major logical element order.  The
            // result coordinate has a different rank/shape, so deriving an input index from
            // its individual axes would test the fixture's old shape rather than the domain
            // mapping.  The logical linear position is the independent mapping here.
            case "RESHAPE", "EXPAND_DIMS", "SQUEEZE" -> linear;
            case "EXPAND" -> out[0];
            case "PERMUTE" -> out[1] * 3 + out[0];
            case "SELECT" -> out[0] * 3 + 1;
            case "SLICE" -> (out[0] + 1) * 4 + out[1];
            default -> throw new AssertionError("unrecognized affine oracle operation " + operation.kind());
        };
    }

    private static long[] coordinates(long linear, long[] shape) {
        long[] result = new long[shape.length];
        for (int axis = shape.length - 1; axis >= 0; axis--) { result[axis] = linear % shape[axis]; linear /= shape[axis]; }
        return result;
    }

    private static void assertRawEquals(DataType type, Object expected, int expectedIndex, Object actual,
            int actualIndex, String message) {
        switch (type) {
            case FLOAT64 -> assertEquals(Double.doubleToRawLongBits(((double[]) expected)[expectedIndex]), Double.doubleToRawLongBits(((double[]) actual)[actualIndex]), message);
            case FLOAT32 -> assertEquals(Float.floatToRawIntBits(((float[]) expected)[expectedIndex]), Float.floatToRawIntBits(((float[]) actual)[actualIndex]), message);
            case BFLOAT16 -> assertEquals(((short[]) expected)[expectedIndex], ((short[]) actual)[actualIndex], message);
            case INT64 -> assertEquals(((long[]) expected)[expectedIndex], ((long[]) actual)[actualIndex], message);
            case INT32 -> assertEquals(((int[]) expected)[expectedIndex], ((int[]) actual)[actualIndex], message);
            case BOOL -> assertEquals(((byte[]) expected)[expectedIndex], ((byte[]) actual)[actualIndex], message);
        }
    }

    private static List<Base> bases() {
        Shape twoThree = Shape.of(2, 3), threeTwo = Shape.of(3, 2), twoOne = Shape.of(2, 1);
        Shape twoOneThree = Shape.of(2, 1, 3), threeFour = Shape.of(3, 4), twoFour = Shape.of(2, 4);
        return List.of(
                new Base(new Operation(ContiguousKind.CONTIGUOUS, NoOperationAttrs.INSTANCE), d(twoThree, new long[] {1, 2}), dc(twoThree)),
                new Base(new Operation(ShapeTransformKind.RESHAPE, new TargetShapeAttrs(threeTwo)), d(twoThree, new long[] {3, 1}), d(threeTwo, new long[] {2, 1})),
                new Base(new Operation(ShapeTransformKind.EXPAND, new TargetShapeAttrs(twoThree)), dc(twoOne), d(twoThree, new long[] {1, 0})),
                new Base(new Operation(AxisTransformKind.PERMUTE, new PermutationAttrs(List.of(1, 0))), d(twoThree, new long[] {3, 1}), d(threeTwo, new long[] {1, 3})),
                new Base(new Operation(AxisTransformKind.EXPAND_DIMS, new AxisTransformAttrs(1)), d(twoThree, new long[] {3, 1}), d(twoOneThree, new long[] {3, 3, 1})),
                new Base(new Operation(AxisTransformKind.SQUEEZE, new AxisTransformAttrs(1)), d(twoOneThree, new long[] {3, 3, 1}), d(twoThree, new long[] {3, 1})),
                new Base(new Operation(SelectKind.SELECT, new SelectAttrs(1, 1)), dc(Shape.of(3, 3)), d(Shape.of(3), new long[] {3}, 1)),
                new Base(new Operation(SliceKind.SLICE, new SliceAttrs(List.of(1L), List.of(2L), List.of(0), List.of(1L))), dc(threeFour), d(twoFour, new long[] {4, 1}, 4)));
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> context(Operation op, TensorDescriptor input,
            TensorDescriptor output, List<CarrierAccess> carriers, PortableExecutionConfig request, boolean enabled) {
        var node = new CompiledNode(new NodeId(0), op, List.of(new ValueId(0)), List.of(new ValueId(1)));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID, List.of(node.id()));
        var values = List.of(new GraphValue(new ValueId(0), input), new GraphValue(new ValueId(1), output));
        var memory = List.of(new LogicalMemoryRequirement(new ValueId(0), input, Optional.empty(), List.of(partition), false),
                new LogicalMemoryRequirement(new ValueId(1), output, Optional.of(partition), List.of(), true));
        var policy = enabled ? new CpuPartitionAnalysisInputs.MaterializationPolicy(true, 0, 0, 10, 1, 2, 1_000_000, 0, 0)
                : CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED;
        return new PrepareContext<>(partition, List.of(node), values, memory, Map.of(), new CpuPartitionAnalysisInputs(false, carriers, request, policy));
    }
    private static TensorDescriptor typed(TensorDescriptor source, DataType type) { return new TensorDescriptor(type, source.shape(), source.layout(), false); }
    private static TensorDescriptor dc(Shape shape) { return new TensorDescriptor(DataType.FLOAT32, shape, Optional.of(LayoutDescriptor.contiguous(shape)), false); }
    private static TensorDescriptor d(Shape shape, long[] strides) { return d(shape, strides, 0); }
    private static TensorDescriptor d(Shape shape, long[] strides, long offset) { return new TensorDescriptor(DataType.FLOAT32, shape, Optional.of(LayoutDescriptor.of(shape, strides, offset, true)), false); }
    private record Base(Operation operation, TensorDescriptor input, TensorDescriptor output) { }
    private enum CarrierMode {
        HEAP { List<CarrierAccess> pattern(DataType type) { return List.of(heap(type), heap(type)); } },
        SEGMENT { List<CarrierAccess> pattern(DataType type) { return List.of(CarrierAccess.MEMORY_SEGMENT, CarrierAccess.MEMORY_SEGMENT); } },
        HEAP_TO_SEGMENT { List<CarrierAccess> pattern(DataType type) { return List.of(heap(type), CarrierAccess.MEMORY_SEGMENT); } },
        SEGMENT_TO_HEAP { List<CarrierAccess> pattern(DataType type) { return List.of(CarrierAccess.MEMORY_SEGMENT, heap(type)); } };
        abstract List<CarrierAccess> pattern(DataType type);
        private static CarrierAccess heap(DataType type) { return switch (type) {
            case FLOAT64 -> CarrierAccess.DOUBLE_ARRAY; case FLOAT32 -> CarrierAccess.FLOAT_ARRAY; case BFLOAT16 -> CarrierAccess.SHORT_ARRAY;
            case INT64 -> CarrierAccess.LONG_ARRAY; case INT32 -> CarrierAccess.INT_ARRAY; case BOOL -> CarrierAccess.BYTE_ARRAY;
        }; }
    }
}
