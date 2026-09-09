package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuLossIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScatterLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.loss.DenseCategoricalCrossEntropyWithLogitsAttrs;
import io.github.pho001.synaptik.model.operation.loss.IndexCategoricalCrossEntropyWithLogitsAttrs;
import io.github.pho001.synaptik.model.operation.loss.LossKind;
import io.github.pho001.synaptik.model.operation.loss.LossReduction;
import io.github.pho001.synaptik.model.operation.loss.MeanSquaredErrorAttrs;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.foreign.MemorySegment;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Executes each finite loss inventory row against a test-local clean-Java semantic oracle. */
class CpuLossSemanticClosureTest {
    private static final List<DataType> FLOATING = List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16);
    private static final Shape LOGITS = Shape.of(2, 3, 4);
    private static final Shape SAMPLES = Shape.of(2, 4);
    private static final CpuPartitionAnalysisInputs.MaterializationPolicy MATERIALIZATION =
            new CpuPartitionAnalysisInputs.MaterializationPolicy(true, 0, 1, 20, 1, 3, 1_000_000, 1, 1);

    @Test void everyLossInventoryRowDefinesAndInvokesItsActualGeneratedEntryAgainstIndependentOracle() throws Throwable {
        int raw = 0;
        java.util.Set<String> owners = new java.util.TreeSet<>();
        for (LossKind kind : List.of(LossKind.MEAN_SQUARED_ERROR,
                LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS)) {
            for (Fixture fixture : fixtures(kind)) {
                assertTrue(owners.add("specialized:" + fixture.owner()), "duplicate semantic witness " + fixture);
                invokeAndCompare(fixture);
                raw++;
            }
        }
        assertEquals(540, raw, "one exact generated definition/invocation per scoped inventory row");
        assertEquals(inventoryLossOwners(), owners, "complete non-projecting inventory-owner join");
    }

    private static List<Fixture> fixtures(LossKind kind) {
        List<Fixture> result = new ArrayList<>();
        if (kind == LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS) {
            for (DataType logits : FLOATING) for (DataType index : List.of(DataType.INT32, DataType.INT64))
                for (LossReduction reduction : LossReduction.values()) for (boolean ignored : List.of(false, true))
                    for (Request request : Request.values()) result.add(new Fixture(kind, logits, index,
                            reduction, ignored, false, request));
        } else {
            for (DataType left : FLOATING) for (DataType right : FLOATING)
                for (LossReduction reduction : LossReduction.values()) for (Request request : Request.values()) {
                    result.add(new Fixture(kind, left, right, reduction, false, false, request));
                    if (left == right) result.add(new Fixture(kind, left, right, reduction, false, true, request));
                }
        }
        assertEquals(180, result.size(), kind.toString());
        return result;
    }

    private static void invokeAndCompare(Fixture fixture) throws Throwable {
        PrepareContext<CpuPartitionAnalysisInputs> context = configure(base(fixture), fixture.request());
        CpuPartitionPreparationPlan plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan();
        assertEquals(0, plan.materializations().size(), fixture + " materialization remains unselected");
        assertTrue(route.portableKernelIr() instanceof CpuLossIr, fixture.toString());
        CpuLossIr loss = (CpuLossIr) route.portableKernelIr();
        var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(route.specialization(),
                generator.generateClassBytes(route.specialization(), route.kernelIr()));

        List<Storage> storage = new ArrayList<>();
        for (int i = 0; i < loss.boundaryTypes().size(); i++) storage.add(new Storage(
                loss.boundaryTypes().get(i), route.specialization().carrierPattern().get(i)));
        fillInputs(fixture, context, storage, loss);
        Storage output = storage.getLast(); output.fill(-91.25d);
        double[] beforeInputs = storage.subList(0, storage.size() - 1).stream()
                .flatMapToDouble(value -> java.util.Arrays.stream(value.snapshot())).toArray();
        double[] expected = output.snapshot();
        oracle(fixture, context, storage, loss, expected);
        long domain = fixture.reduction() == LossReduction.NONE
                ? (fixture.kind() == LossKind.MEAN_SQUARED_ERROR ? 24 : 8) : 1;
        Object[] arguments = storage.stream().map(Storage::argument).toArray();
        long[] geometry = loss.geometry().pack(new long[] {0, 0, 0});
        if (fixture.request().parallel && domain > 1) {
            long middle = domain / 2;
            entry(artifact, arguments, geometry, 0, middle);
            entry(artifact, arguments, geometry, middle, domain);
        } else entry(artifact, arguments, geometry, 0, domain);
        assertStorage(expected, output, fixture);
        double[] afterInputs = storage.subList(0, storage.size() - 1).stream()
                .flatMapToDouble(value -> java.util.Arrays.stream(value.snapshot())).toArray();
        assertEquals(java.util.Arrays.toString(beforeInputs), java.util.Arrays.toString(afterInputs), fixture + " input mutation");
    }

    private static void entry(CpuGeneratedKernel artifact, Object[] boundaries, long[] geometry,
            long start, long end) throws Throwable {
        Object[] arguments = new Object[boundaries.length + 3];
        System.arraycopy(boundaries, 0, arguments, 0, boundaries.length);
        arguments[boundaries.length] = geometry; arguments[boundaries.length + 1] = start;
        arguments[boundaries.length + 2] = end;
        artifact.entryPoint().invokeWithArguments(arguments);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> base(Fixture f) {
        Shape target = f.kind() == LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS ? SAMPLES : LOGITS;
        Shape output = f.reduction() == LossReduction.NONE
                ? f.kind() == LossKind.MEAN_SQUARED_ERROR ? LOGITS : SAMPLES : Shape.scalar();
        Operation operation = switch (f.kind()) {
            case MEAN_SQUARED_ERROR -> new Operation(f.kind(), new MeanSquaredErrorAttrs(f.reduction()));
            case DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS -> new Operation(f.kind(),
                    new DenseCategoricalCrossEntropyWithLogitsAttrs(1, f.reduction()));
            case INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS -> new Operation(f.kind(),
                    new IndexCategoricalCrossEntropyWithLogitsAttrs(1, f.reduction(), f.ignored()
                            ? Optional.of(f.right() == DataType.INT32 ? ScalarValue.int32(-1) : ScalarValue.int64(-1))
                            : Optional.empty()));
        };
        DataType result = f.kind() == LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS ? f.left()
                : DataTypePromotion.promoteFloating(f.left(), f.right());
        List<TensorDescriptor> inputs = f.alias() ? List.of(descriptor(f.left(), LOGITS))
                : List.of(descriptor(f.left(), LOGITS), descriptor(f.right(), target));
        return CpuScatterLoweringTest.context(operation, f.alias() ? List.of(0, 0) : List.of(0, 1),
                inputs, descriptor(result, output));
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> configure(PrepareContext<CpuPartitionAnalysisInputs> base, Request request) {
        List<GraphValue> values = new ArrayList<>(); List<LogicalMemoryRequirement> memory = new ArrayList<>();
        for (int i = 0; i < base.values().size(); i++) {
            TensorDescriptor descriptor = base.values().get(i).descriptor();
            if (request.general && descriptor.shape().rank() != 0) {
                long[] strides = descriptor.layout().orElseThrow().strides();
                for (int axis = 0; axis < strides.length; axis++) strides[axis] *= 2;
                descriptor = new TensorDescriptor(descriptor.dataType(), descriptor.shape(),
                        Optional.of(LayoutDescriptor.of(descriptor.shape(), strides, 1, true)), false);
            }
            GraphValue value = new GraphValue(base.values().get(i).id(), descriptor); values.add(value);
            var requirement = base.memoryRequirements().get(i);
            memory.add(new LogicalMemoryRequirement(requirement.valueId(), descriptor,
                    requirement.producerPartition(), requirement.consumerPartitions(), requirement.graphOutput()));
        }
        List<CarrierAccess> carriers = new ArrayList<>();
        for (int i = 0; i < values.size(); i++) carriers.add(request.segment || request.mixed && i % 2 == 1
                ? CarrierAccess.MEMORY_SEGMENT : CpuGeneratedDirectEvidenceClosureTest.heapCarrier(values.get(i).descriptor().dataType()));
        return new PrepareContext<>(base.partition(), base.nodes(), values, memory, base.constants(),
                new CpuPartitionAnalysisInputs(false, carriers, request.execution,
                        request.materialization ? MATERIALIZATION : CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED));
    }

    private static void fillInputs(Fixture f, PrepareContext<CpuPartitionAnalysisInputs> context,
            List<Storage> storage, CpuLossIr loss) {
        TensorDescriptor logits = context.values().getFirst().descriptor();
        for (int logical = 0; logical < 24; logical++) storage.get(loss.roleBoundaryPositions().getFirst())
                .set(offset(logits.layout().orElseThrow(), LOGITS, logical), .25 + logical * .125);
        if (f.alias()) return;
        TensorDescriptor target = context.values().get(1).descriptor(); Storage targetStorage = storage.get(loss.roleBoundaryPositions().get(1));
        if (f.kind() == LossKind.MEAN_SQUARED_ERROR) for (int logical = 0; logical < 24; logical++)
            targetStorage.set(offset(target.layout().orElseThrow(), LOGITS, logical), -.4 + logical * .0625);
        else if (f.kind() == LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS) for (int logical = 0; logical < 24; logical++)
            targetStorage.set(offset(target.layout().orElseThrow(), LOGITS, logical), (logical / 4) % 3 == logical % 3 ? 1 : 0);
        else for (int sample = 0; sample < 8; sample++) targetStorage.set(offset(target.layout().orElseThrow(), SAMPLES, sample),
                f.ignored() && sample % 3 == 0 ? -1 : sample % 3);
    }

    /* Independent scalar oracle: it deliberately neither lowers nor calls a production/reference kernel. */
    private static void oracle(Fixture f, PrepareContext<CpuPartitionAnalysisInputs> context, List<Storage> stores,
            CpuLossIr loss, double[] expected) {
        List<LayoutDescriptor> layouts = context.values().stream().map(value -> value.descriptor().layout().orElseThrow()).toList();
        int prediction = loss.roleBoundaryPositions().getFirst(), target = loss.roleBoundaryPositions().get(1), output = stores.size() - 1;
        double total = 0; int count = 0;
        int domain = f.kind() == LossKind.MEAN_SQUARED_ERROR ? 24 : 8;
        for (int item = 0; item < domain; item++) {
            double value;
            if (f.kind() == LossKind.MEAN_SQUARED_ERROR) { double d = narrow(stores.get(output).type, stores.get(prediction).get(offset(layouts.get(0), LOGITS, item))
                    - stores.get(target).get(offset(layouts.get(f.alias() ? 0 : 1), LOGITS, item))); value = narrow(stores.get(output).type, d * d); }
            else { int outer = item / 4, inner = item % 4; double max = Double.NEGATIVE_INFINITY;
                for (int c = 0; c < 3; c++) max = Math.max(max, logit(stores.get(prediction), layouts.get(0), outer, c, inner));
                double sum = 0; for (int c = 0; c < 3; c++) sum = narrow(stores.get(output).type, sum + StrictMath.exp(narrow(stores.get(output).type, logit(stores.get(prediction), layouts.get(0), outer, c, inner) - max)));
                double lse = narrow(stores.get(output).type, max + StrictMath.log(sum));
                if (f.kind() == LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS) { value = 0;
                    LayoutDescriptor targetLayout = layouts.get(f.alias() ? 0 : 1);
                    for (int c = 0; c < 3; c++) { double q = stores.get(target).get(offset(targetLayout, LOGITS, (outer * 3 + c) * 4 + inner));
                        if (q != 0) value = narrow(stores.get(output).type, value + narrow(stores.get(output).type, q * narrow(stores.get(output).type, lse - logit(stores.get(prediction), layouts.get(0), outer, c, inner)))); }
                } else { long index = Math.round(stores.get(target).get(offset(layouts.get(1), SAMPLES, item)));
                    if (f.ignored() && index == -1) value = 0; else value = narrow(stores.get(output).type, lse - logit(stores.get(prediction), layouts.get(0), outer, (int) index, inner)); }
            }
            if (f.reduction() == LossReduction.NONE) expected[(int) offset(layouts.getLast(), f.kind() == LossKind.MEAN_SQUARED_ERROR ? LOGITS : SAMPLES, item)] = value;
            else { total = narrow(stores.get(output).type, total + value); if (!(f.kind() == LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS && f.ignored()
                    && Math.round(stores.get(target).get(offset(layouts.get(1), SAMPLES, item))) == -1)) count++; }
        }
        if (f.reduction() != LossReduction.NONE) expected[(int) layouts.getLast().storageOffset()] =
                f.reduction() == LossReduction.MEAN ? narrow(stores.get(output).type, total / count) : total;
    }

    private static double logit(Storage storage, LayoutDescriptor layout, int outer, int c, int inner) {
        return storage.get(offset(layout, LOGITS, (outer * 3 + c) * 4 + inner));
    }
    private static long offset(LayoutDescriptor layout, Shape shape, int linear) {
        long[] dimensions = shape.toLongArray(), strides = layout.strides(); long value = layout.storageOffset();
        for (int axis = dimensions.length - 1; axis >= 0; axis--) { value += (linear % dimensions[axis]) * strides[axis]; linear /= dimensions[axis]; }
        return value;
    }
    private static TensorDescriptor descriptor(DataType type, Shape shape) { return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), false); }
    private static java.util.Set<String> inventoryLossOwners() throws Exception {
        try (var stream = CpuLossSemanticClosureTest.class.getResourceAsStream(
                "/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/generated-coverage-inventory.tsv")) {
            assertTrue(stream != null, "inventory resource");
            java.util.Set<String> owners = new java.util.TreeSet<>();
            String[] lines = new String(stream.readAllBytes(), StandardCharsets.UTF_8).split("\\n", -1);
            for (int line = 1; line < lines.length - 1; line++) { String[] fields = lines[line].split("\\t", -1);
                if (fields[2].equals("MEAN_SQUARED_ERROR") || fields[2].equals("DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS")
                        || fields[2].equals("INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS")) assertTrue(owners.add(fields[0]), "duplicate inventory owner " + fields[0]); }
            assertEquals(540, owners.size()); return owners;
        }
    }
    private static void assertStorage(double[] expected, Storage actual, Fixture f) {
        for (int i = 0; i < expected.length; i++) assertEquals(quantize(actual.type, expected[i]), actual.get(i),
                actual.type == DataType.BFLOAT16 ? .02 : actual.type == DataType.FLOAT32 ? 2e-5 : 1e-12,
                f + " physical " + i);
    }
    private static double quantize(DataType type, double value) { return switch (type) {
        case BFLOAT16 -> Float.intBitsToFloat((ScalarValue.bfloat16((float) value).bfloat16Bits() & 0xffff) << 16);
        case FLOAT32 -> (float) value;
        default -> value;
    }; }
    private static double narrow(DataType type, double value) { return type == DataType.FLOAT64 ? value : (float) value; }

    private enum Request {
        HEAP_CONTIGUOUS_SCALAR(false, false, false, false, CpuGeneratedDirectEvidenceClosureTest.scalar(1)),
        SEGMENT_CONTIGUOUS_VECTOR(true, false, false, false, CpuGeneratedDirectEvidenceClosureTest.vector(1)),
        MIXED_GENERAL_PARALLEL_VECTOR(false, true, true, false, CpuGeneratedDirectEvidenceClosureTest.vector(4)),
        HEAP_GENERAL_PARALLEL_SCALAR(false, false, true, false, CpuGeneratedDirectEvidenceClosureTest.scalar(4)),
        HEAP_GENERAL_MATERIALIZATION(false, false, true, true, CpuGeneratedDirectEvidenceClosureTest.scalar(1));
        final boolean segment, mixed, general, materialization, parallel; final CpuPartitionAnalysisInputs.PortableExecutionConfig execution;
        Request(boolean segment, boolean mixed, boolean general, boolean materialization, CpuPartitionAnalysisInputs.PortableExecutionConfig execution) {
            this.segment = segment; this.mixed = mixed; this.general = general; this.materialization = materialization;
            this.parallel = execution.availableParallelism() > 1; this.execution = execution;
        }
    }
    private record Fixture(LossKind kind, DataType left, DataType right, LossReduction reduction, boolean ignored, boolean alias, Request request) {
        String owner() { String family = kind == LossKind.MEAN_SQUARED_ERROR ? "mse" : kind == LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS ? "dense" : "index";
            if (kind == LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS) return "loss/" + family + '/' + left + '/' + right + '/' + reduction + "/ignored=" + ignored + '/' + request;
            return alias ? "loss/" + family + '/' + left + '/' + reduction + "/aliased/" + request
                    : "loss/" + family + '/' + left + '/' + right + '/' + reduction + "/ordered/" + request; }
    }

    private static final class Storage {
        final DataType type; final CarrierAccess carrier; final Object heap; final MemorySegment segment;
        Storage(DataType type, CarrierAccess carrier) { this.type = type; this.carrier = carrier; this.heap = switch (type) {
            case FLOAT64 -> new double[64]; case FLOAT32 -> new float[64]; case BFLOAT16 -> new short[64];
            case INT32 -> new int[64]; case INT64 -> new long[64]; default -> throw new AssertionError(type); };
            this.segment = switch (type) { case FLOAT64 -> MemorySegment.ofArray((double[]) heap);
                case FLOAT32 -> MemorySegment.ofArray((float[]) heap); case BFLOAT16 -> MemorySegment.ofArray((short[]) heap);
                case INT32 -> MemorySegment.ofArray((int[]) heap); case INT64 -> MemorySegment.ofArray((long[]) heap);
                default -> throw new AssertionError(type); }; }
        Object argument() { return carrier == CarrierAccess.MEMORY_SEGMENT ? segment : heap; }
        void fill(double value) { for (int i = 0; i < 64; i++) set(i, value); }
        void set(long index, double value) { switch (type) {
            case FLOAT64 -> ((double[]) heap)[(int) index] = value;
            case FLOAT32 -> ((float[]) heap)[(int) index] = (float) value;
            case BFLOAT16 -> ((short[]) heap)[(int) index] = ScalarValue.bfloat16((float) value).bfloat16Bits();
            case INT32 -> ((int[]) heap)[(int) index] = (int) value;
            case INT64 -> ((long[]) heap)[(int) index] = (long) value;
            default -> throw new AssertionError(type); } }
        double get(long index) { return switch (type) {
            case FLOAT64 -> ((double[]) heap)[(int) index]; case FLOAT32 -> ((float[]) heap)[(int) index];
            case BFLOAT16 -> Float.intBitsToFloat((((short[]) heap)[(int) index] & 0xffff) << 16);
            case INT32 -> ((int[]) heap)[(int) index]; case INT64 -> ((long[]) heap)[(int) index]; default -> throw new AssertionError(type); }; }
        double[] snapshot() { double[] copy = new double[64]; for (int i = 0; i < copy.length; i++) copy[i] = get(i); return copy; }
    }
}
