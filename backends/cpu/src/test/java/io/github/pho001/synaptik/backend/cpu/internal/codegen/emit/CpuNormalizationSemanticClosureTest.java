package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuTrailingNormalizationIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuSoftmaxLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.backend.cpu.internal.reference.CpuSoftmaxReferenceKernel;
import io.github.pho001.synaptik.backend.cpu.internal.reference.CpuTrailingNormalizationReferenceKernel;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.normalization.AffineLayerNormAttrs;
import io.github.pho001.synaptik.model.operation.normalization.LayerNormAttrs;
import io.github.pho001.synaptik.model.operation.normalization.LayerNormKind;
import io.github.pho001.synaptik.model.operation.normalization.RmsNormAttrs;
import io.github.pho001.synaptik.model.operation.normalization.RmsNormKind;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.TreeSet;
import org.junit.jupiter.api.Test;

/** Directly executes each inventoried softmax and trailing-normalization generated owner. */
class CpuNormalizationSemanticClosureTest {
    private static final List<DataType> FLOATING = List.of(DataType.FLOAT64, DataType.FLOAT32,
            DataType.BFLOAT16);
    private static final Shape INPUT = Shape.of(2, 3);
    private static final Shape NORMALIZED = Shape.of(3);
    private static final CpuPartitionAnalysisInputs.MaterializationPolicy MATERIALIZATION =
            new CpuPartitionAnalysisInputs.MaterializationPolicy(true, 0, 1, 20, 1, 3,
                    1_000_000, 1, 1);

    @Test void everyNormalizationInventoryOwnerDefinesAndInvokesItsActualGeneratedEntryAgainstIndependentOracle()
            throws Throwable {
        Map<String, Set<String>> invoked = new java.util.TreeMap<>();
        for (SoftmaxKind kind : SoftmaxKind.values()) for (DataType type : FLOATING)
            for (int axis : List.of(0, 1)) for (Request request : Request.values()) {
                Fixture fixture = Fixture.softmax(kind, type, axis, request);
                invoke(fixture); add(invoked, fixture.form(), fixture.owner());
            }
        for (DataType input : FLOATING) for (Request request : Request.values()) {
            Fixture layer = Fixture.trailing(CpuTrailingNormalizationIr.Form.LAYER,
                    List.of(input), request);
            Fixture rms = Fixture.trailing(CpuTrailingNormalizationIr.Form.RMS, List.of(input), request);
            invoke(layer); invoke(rms); add(invoked, layer.form(), layer.owner()); add(invoked, rms.form(), rms.owner());
        }
        for (DataType input : FLOATING) for (DataType scale : FLOATING)
            for (Request request : Request.values()) {
                Fixture fixture = Fixture.trailing(CpuTrailingNormalizationIr.Form.RMS_SCALED,
                        List.of(input, scale), request);
                invoke(fixture); add(invoked, fixture.form(), fixture.owner());
            }
        for (DataType input : FLOATING) for (DataType scale : FLOATING) for (DataType bias : FLOATING)
            for (Request request : Request.values()) {
                Fixture fixture = Fixture.trailing(CpuTrailingNormalizationIr.Form.LAYER_AFFINE,
                        List.of(input, scale, bias), request);
                invoke(fixture); add(invoked, fixture.form(), fixture.owner());
            }
        assertCounts(invoked, "SOFTMAX", 30); assertCounts(invoked, "LOG_SOFTMAX", 30);
        assertCounts(invoked, "LAYER", 15); assertCounts(invoked, "LAYER_AFFINE", 135);
        assertCounts(invoked, "RMS", 15); assertCounts(invoked, "RMS_SCALED", 45);
        assertEquals(inventoryOwners(), invoked, "exact inventory owner and current SHA join");
    }

    private static void invoke(Fixture fixture) throws Throwable {
        PrepareContext<CpuPartitionAnalysisInputs> context = configure(fixture.context(), fixture.request());
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan();
        byte[] bytes = new CpuClassFileKernelGenerator().generateClassBytes(route.specialization(), route.kernelIr());
        assertArrayEquals(bytes, new CpuClassFileKernelGenerator().generateClassBytes(
                route.specialization(), route.kernelIr()), fixture + " deterministic bytes");
        var artifact = new CpuClassFileKernelGenerator().defineClassBytes(route.specialization(), bytes);
        assertTrue(plan.materializations().isEmpty(), fixture + " materialization remains unselected");
        List<Storage> storage = storage(context, route.specialization().carrierPattern());
        fillInputs(storage, context);
        long[] beforeInput = storage.getFirst().rawBits();
        Storage output = storage.getLast(); output.fill(-91.0);
        long start = 1; long end = fixture.softmax() ? plan.softmaxGeometry().orElseThrow().sliceCount()
                : plan.trailingNormalizationGeometry().orElseThrow().leadingCount();
        double[] expected = expected(fixture, context, storage);
        List<Object> arguments = new ArrayList<>();
        for (Storage value : storage) arguments.add(value.argument());
        try (Arena arena = Arena.ofConfined()) {
            if (!fixture.softmax()) {
                long scratch = plan.trailingNormalizationGeometry().orElseThrow().scratchSliceBytes();
                if (scratch > 0) arguments.add(arena.allocate(scratch, 8));
            }
            arguments.add(fixture.softmax() ? plan.softmaxGeometry().orElseThrow().pack(new long[2])
                    : plan.trailingNormalizationGeometry().orElseThrow().pack(new long[storage.size()]));
            arguments.add(start); arguments.add(end);
            artifact.entryPoint().invokeWithArguments(arguments);
        }
        assertOutput(fixture, context, output, expected, start, end);
        assertArrayEquals(beforeInput, storage.getFirst().rawBits(), fixture + " input immutability");
    }

    private static double[] expected(Fixture fixture, PrepareContext<CpuPartitionAnalysisInputs> context,
            List<Storage> storage) {
        List<TensorDescriptor> descriptors = context.values().stream().map(GraphValue::descriptor).toList();
        if (fixture.softmax()) {
            TensorDescriptor input = descriptors.getFirst();
            LayoutDescriptor layout = input.layout().orElseThrow();
            return CpuSoftmaxReferenceKernel.evaluate(fixture.softmaxKind(), input.dataType(),
                    storage.getFirst().decoded(), input.shape().toLongArray(), layout.storageOffset(),
                    layout.strides(), fixture.axis());
        }
        int inputs = storage.size() - 1;
        double[][] values = new double[inputs][]; long[][] extents = new long[inputs][];
        long[][] strides = new long[inputs][]; long[] offsets = new long[inputs];
        for (int index = 0; index < inputs; index++) {
            LayoutDescriptor layout = descriptors.get(index).layout().orElseThrow();
            values[index] = storage.get(index).decoded(); extents[index] = descriptors.get(index).shape().toLongArray();
            strides[index] = layout.strides(); offsets[index] = layout.storageOffset();
        }
        DataType result = descriptors.getLast().dataType();
        return CpuTrailingNormalizationReferenceKernel.evaluate(fixture.form().startsWith("LAYER")
                ? CpuTrailingNormalizationIr.Kind.LAYER : CpuTrailingNormalizationIr.Kind.RMS,
                fixture.trailingForm(), fixture.types(), result, epsilon(result), values, extents,
                offsets, strides, 1);
    }

    private static void assertOutput(Fixture fixture, PrepareContext<CpuPartitionAnalysisInputs> context,
            Storage output, double[] expected, long start, long end) {
        TensorDescriptor descriptor = context.values().getLast().descriptor();
        LayoutDescriptor layout = descriptor.layout().orElseThrow();
        long[] shape = descriptor.shape().toLongArray();
        for (long unit = 0; unit < shape[0]; unit++) for (long trailing = 0; trailing < shape[1]; trailing++) {
            long logical = unit * shape[1] + trailing;
            long offset = offset(layout, new long[] {unit, trailing});
            long invocationUnit = fixture.softmax() ? fixture.axis() == 0 ? trailing : unit : unit;
            if (invocationUnit >= start && invocationUnit < end) assertEquals(raw(descriptor.dataType(), expected[(int) logical]),
                    raw(descriptor.dataType(), output.get(offset)), fixture + " oracle at " + logical);
            else assertEquals(raw(descriptor.dataType(), -91.0), raw(descriptor.dataType(), output.get(offset)),
                    fixture + " untouched output at " + logical);
        }
    }

    private static List<Storage> storage(PrepareContext<CpuPartitionAnalysisInputs> context,
            List<CarrierAccess> carriers) {
        List<Storage> result = new ArrayList<>();
        for (int i = 0; i < carriers.size(); i++) result.add(new Storage(
                context.values().get(i).descriptor().dataType(), carriers.get(i), 32));
        return result;
    }

    private static void fillInputs(List<Storage> storage, PrepareContext<CpuPartitionAnalysisInputs> context) {
        for (int input = 0; input < storage.size() - 1; input++) {
            TensorDescriptor descriptor = context.values().get(input).descriptor();
            LayoutDescriptor layout = descriptor.layout().orElseThrow(); long[] shape = descriptor.shape().toLongArray();
            long count = shape.length == 1 ? shape[0] : shape[0] * shape[1];
            for (int ordinal = 0; ordinal < count; ordinal++) {
                long[] coordinate = shape.length == 1 ? new long[] {ordinal}
                        : new long[] {ordinal / shape[1], ordinal % shape[1]};
                storage.get(input).set(offset(layout, coordinate), input == 0
                        ? new double[] {1000.0, 1.0, -1000.0, -3.0, 0.0, 3.0}[ordinal]
                        : input == 1 ? new double[] {2.0, -1.0, 0.5}[ordinal] : new double[] {1.0, 2.0, 3.0}[ordinal]);
            }
        }
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> configure(PrepareContext<CpuPartitionAnalysisInputs> base,
            Request request) {
        List<GraphValue> values = new ArrayList<>(); List<LogicalMemoryRequirement> memory = new ArrayList<>();
        for (int i = 0; i < base.values().size(); i++) {
            TensorDescriptor descriptor = base.values().get(i).descriptor();
            if (request.general() && descriptor.shape().rank() != 0) {
                long[] strides = descriptor.layout().orElseThrow().strides();
                for (int axis = 0; axis < strides.length; axis++) strides[axis] *= 2;
                descriptor = new TensorDescriptor(descriptor.dataType(), descriptor.shape(), Optional.of(
                        LayoutDescriptor.of(descriptor.shape(), strides, 1, true)), descriptor.requiresGrad());
            }
            values.add(new GraphValue(base.values().get(i).id(), descriptor));
            LogicalMemoryRequirement old = base.memoryRequirements().get(i);
            memory.add(new LogicalMemoryRequirement(old.valueId(), descriptor, old.producerPartition(),
                    old.consumerPartitions(), old.graphOutput()));
        }
        List<CarrierAccess> carriers = new ArrayList<>();
        for (int i = 0; i < values.size(); i++) carriers.add(request.segment() || request.mixed() && i % 2 == 1
                ? CarrierAccess.MEMORY_SEGMENT : CpuGeneratedDirectEvidenceClosureTest.heapCarrier(values.get(i).descriptor().dataType()));
        return new PrepareContext<>(base.partition(), base.nodes(), values, memory, base.constants(),
                new CpuPartitionAnalysisInputs(false, carriers, request.execution(), request.materialization()
                        ? MATERIALIZATION : CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED));
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> trailing(CpuTrailingNormalizationIr.Form form,
            List<DataType> types) {
        DataType result = types.getFirst();
        for (int index = 1; index < types.size(); index++) result = DataTypePromotion.promoteFloating(result, types.get(index));
        ScalarValue epsilon = result == DataType.FLOAT64 ? ScalarValue.float64(1e-5)
                : result == DataType.FLOAT32 ? ScalarValue.float32(1e-5f)
                : ScalarValue.bfloat16Bits((short) 0x3728);
        var attrs = form == CpuTrailingNormalizationIr.Form.LAYER ? new LayerNormAttrs(NORMALIZED, epsilon)
                : form == CpuTrailingNormalizationIr.Form.LAYER_AFFINE ? new AffineLayerNormAttrs(NORMALIZED, epsilon)
                : new RmsNormAttrs(NORMALIZED, epsilon);
        var inputDescriptors = new ArrayList<TensorDescriptor>();
        inputDescriptors.add(new TensorDescriptor(types.getFirst(), INPUT, Optional.of(LayoutDescriptor.contiguous(INPUT)), false));
        for (int index = 1; index < types.size(); index++) inputDescriptors.add(new TensorDescriptor(types.get(index), NORMALIZED, Optional.of(LayoutDescriptor.contiguous(NORMALIZED)), false));
        return io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScatterLoweringTest.context(
                new Operation(form == CpuTrailingNormalizationIr.Form.LAYER || form == CpuTrailingNormalizationIr.Form.LAYER_AFFINE
                        ? LayerNormKind.LAYER_NORM : RmsNormKind.RMS_NORM, attrs),
                java.util.stream.IntStream.range(0, types.size()).boxed().toList(), inputDescriptors,
                new TensorDescriptor(result, INPUT, Optional.of(LayoutDescriptor.contiguous(INPUT)), false));
    }

    private static void add(Map<String, Set<String>> values, String form, String owner) {
        assertTrue(values.computeIfAbsent(form, ignored -> new TreeSet<>()).add(owner), "duplicate owner " + owner);
    }
    private static void assertCounts(Map<String, Set<String>> values, String form, int expected) {
        assertEquals(expected, values.getOrDefault(form, Set.of()).size(), form + " exact invoked owners");
    }
    private static Map<String, Set<String>> inventoryOwners() throws Exception {
        String resource = "/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/generated-coverage-inventory.tsv";
        try (var stream = CpuNormalizationSemanticClosureTest.class.getResourceAsStream(resource)) {
            assertTrue(stream != null, resource); Map<String, Set<String>> result = new java.util.TreeMap<>();
            for (String line : new String(stream.readAllBytes(), StandardCharsets.UTF_8).split("\\n", -1)) {
                String[] fields = line.split("\\t", -1);
                if (fields.length == 27 && fields[0].startsWith("specialized:") && Set.of("SOFTMAX", "LOG_SOFTMAX", "LAYER", "LAYER_AFFINE", "RMS", "RMS_SCALED").contains(fields[2]))
                    assertTrue(result.computeIfAbsent(fields[2], ignored -> new TreeSet<>()).add(fields[0]), "duplicate inventory owner " + fields[0]);
            }
            return result;
        }
    }
    private static long offset(LayoutDescriptor layout, long[] coordinate) {
        long result = layout.storageOffset();
        long[] strides = layout.strides();
        for (int axis = 0; axis < coordinate.length; axis++) result += coordinate[axis] * strides[axis];
        return result;
    }

    private static double epsilon(DataType type) {
        return type == DataType.FLOAT64 ? 1e-5 : type == DataType.FLOAT32 ? 1e-5f
                : Float.intBitsToFloat(0x3728 << 16);
    }

    private static long raw(DataType type, double value) {
        return type == DataType.FLOAT64 ? Double.doubleToRawLongBits(value)
                : type == DataType.FLOAT32 ? Integer.toUnsignedLong(Float.floatToRawIntBits((float) value))
                : Integer.toUnsignedLong(Float.floatToRawIntBits((float) value) >>> 16);
    }

    private enum Request {
        HEAP_CONTIGUOUS_SCALAR(false, false, false, false, CpuGeneratedDirectEvidenceClosureTest.scalar(1)),
        SEGMENT_CONTIGUOUS_VECTOR(true, false, false, false, CpuGeneratedDirectEvidenceClosureTest.vector(1)),
        MIXED_GENERAL_PARALLEL_VECTOR(false, true, true, false, CpuGeneratedDirectEvidenceClosureTest.vector(4)),
        HEAP_GENERAL_PARALLEL_SCALAR(false, false, true, false, CpuGeneratedDirectEvidenceClosureTest.scalar(4)),
        HEAP_GENERAL_MATERIALIZATION(false, false, true, true, CpuGeneratedDirectEvidenceClosureTest.scalar(1));
        private final boolean segment, mixed, general, materialization;
        private final CpuPartitionAnalysisInputs.PortableExecutionConfig execution;

        Request(boolean segment, boolean mixed, boolean general, boolean materialization,
                CpuPartitionAnalysisInputs.PortableExecutionConfig execution) {
            this.segment = segment;
            this.mixed = mixed;
            this.general = general;
            this.materialization = materialization;
            this.execution = execution;
        }

        boolean segment() { return segment; }
        boolean mixed() { return mixed; }
        boolean general() { return general; }
        boolean materialization() { return materialization; }
        CpuPartitionAnalysisInputs.PortableExecutionConfig execution() { return execution; }
    }
    private record Fixture(SoftmaxKind softmaxKind, CpuTrailingNormalizationIr.Form trailingForm,
            List<DataType> types, int axis, Request request) {
        static Fixture softmax(SoftmaxKind kind, DataType type, int axis, Request request) {
            return new Fixture(kind, null, List.of(type), axis, request);
        }

        static Fixture trailing(CpuTrailingNormalizationIr.Form form, List<DataType> types,
                Request request) {
            return new Fixture(null, form, List.copyOf(types), -1, request);
        }

        boolean softmax() { return softmaxKind != null; }
        String form() { return softmax() ? softmaxKind.name() : trailingForm.name(); }
        String owner() {
            if (softmax()) return "specialized:softmax/" + softmaxKind + '/' + types.getFirst()
                    + '/' + axis + '/' + request;
            String family = trailingForm == CpuTrailingNormalizationIr.Form.LAYER ? "layer/plain"
                    : trailingForm == CpuTrailingNormalizationIr.Form.LAYER_AFFINE ? "layer/affine"
                    : trailingForm == CpuTrailingNormalizationIr.Form.RMS ? "rms/plain" : "rms/scaled";
            return "specialized:" + family + '/' + String.join("/",
                    types.stream().map(Enum::name).toList()) + '/' + request;
        }

        PrepareContext<CpuPartitionAnalysisInputs> context() {
            return softmax() ? CpuSoftmaxLoweringTest.context(softmaxKind, types.getFirst(), INPUT, axis)
                    : CpuNormalizationSemanticClosureTest.trailing(trailingForm, types);
        }
    }
    private static final class Storage {
        private final DataType type;
        private final CarrierAccess carrier;
        private final Object heap;
        private final MemorySegment segment;

        Storage(DataType type, CarrierAccess carrier, int capacity) {
            this.type = type;
            this.carrier = carrier;
            heap = type == DataType.FLOAT64 ? new double[capacity]
                    : type == DataType.FLOAT32 ? new float[capacity] : new short[capacity];
            segment = type == DataType.FLOAT64 ? MemorySegment.ofArray((double[]) heap)
                    : type == DataType.FLOAT32 ? MemorySegment.ofArray((float[]) heap)
                    : MemorySegment.ofArray((short[]) heap);
        }

        Object argument() { return carrier == CarrierAccess.MEMORY_SEGMENT ? segment : heap; }
        void fill(double value) {
            for (int i = 0; i < java.lang.reflect.Array.getLength(heap); i++) set(i, value);
        }
        void set(long index, double value) {
            if (type == DataType.FLOAT64) ((double[]) heap)[(int) index] = value;
            else if (type == DataType.FLOAT32) ((float[]) heap)[(int) index] = (float) value;
            else ((short[]) heap)[(int) index] = ScalarValue.bfloat16((float) value).bfloat16Bits();
        }
        double get(long index) {
            return type == DataType.FLOAT64 ? ((double[]) heap)[(int) index]
                    : type == DataType.FLOAT32 ? ((float[]) heap)[(int) index]
                    : Float.intBitsToFloat(Short.toUnsignedInt(((short[]) heap)[(int) index]) << 16);
        }
        double[] decoded() {
            double[] result = new double[java.lang.reflect.Array.getLength(heap)];
            for (int i = 0; i < result.length; i++) result[i] = get(i);
            return result;
        }
        long[] rawBits() {
            long[] result = new long[java.lang.reflect.Array.getLength(heap)];
            for (int i = 0; i < result.length; i++) result[i] = raw(type, get(i));
            return result;
        }
    }
}
