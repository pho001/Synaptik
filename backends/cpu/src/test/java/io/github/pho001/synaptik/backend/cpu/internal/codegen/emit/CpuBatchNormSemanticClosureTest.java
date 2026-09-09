package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormInferenceIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuBatchNormInferenceLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuBatchNormInferenceLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuBatchNormTrainingLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuBatchNormTrainingLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.charset.StandardCharsets;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Executes every finite generated batch-normalization inventory row against a clean Java oracle. */
class CpuBatchNormSemanticClosureTest {
    private static final List<DataType> FLOATING = List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16);
    private static final Shape SHAPE = Shape.of(2, 3, 4);
    private static final CpuPartitionAnalysisInputs.MaterializationPolicy MATERIALIZATION =
            new CpuPartitionAnalysisInputs.MaterializationPolicy(true, 0, 1, 20, 1, 3, 1_000_000, 1, 1);

    @Test void everyBatchNormInferenceInventoryRowDefinesAndInvokesItsActualGeneratedEntryAgainstIndependentOracle()
            throws Throwable {
        Set<String> owners = new TreeSet<>(); int raw = 0;
        for (List<DataType> types : roles()) for (int axis : List.of(0, 1)) for (Request request : Request.values()) {
            Fixture f = new Fixture(false, types, axis, request);
            assertTrue(owners.add(f.owner()), "duplicate inference witness " + f);
            inference(f); raw++;
        }
        assertEquals(2430, raw);
        assertEquals(inventoryOwners("BATCH_NORM_INFERENCE"), owners);
    }

    @Test void everyBatchNormTrainingInventoryRowDefinesAndInvokesItsActualGeneratedEntryAgainstIndependentOracle()
            throws Throwable {
        Set<String> owners = new TreeSet<>(); int raw = 0;
        for (List<DataType> types : roles()) for (int axis : List.of(0, 1)) for (Request request : Request.values()) {
            Fixture f = new Fixture(true, types, axis, request);
            assertTrue(owners.add(f.owner()), "duplicate training witness " + f);
            training(f); raw++;
        }
        assertEquals(2430, raw);
        assertEquals(inventoryOwners("BATCH_NORM_TRAINING"), owners);
    }

    private static void inference(Fixture f) throws Throwable {
        PrepareContext<CpuPartitionAnalysisInputs> context = configure(CpuBatchNormInferenceLoweringTest.context(
                f.types, SHAPE, f.axis, List.of(0, 1, 2, 3, 4)), f.request);
        CpuPartitionPreparationPlan plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        byte[] first = generator.generateClassBytes(route.specialization(), route.kernelIr());
        assertArrayEquals(first, generator.generateClassBytes(route.specialization(), route.kernelIr()), f + " deterministic bytes");
        assertEquals(0, plan.materializations().size(), f + " direct materialization fact");
        List<Storage> stores = stores(route.specialization().boundaryDataTypes(), route.specialization().carrierPattern());
        fillInputs(context, stores, 5); stores.getLast().fill(-91);
        List<double[]> before = snapshots(stores.subList(0, 5));
        double[] expected = stores.getLast().snapshot();
        CpuBatchNormInferenceLowering.Geometry g = plan.batchNormInferenceGeometry().orElseThrow();
        inferenceOracle(g, stores, expected);
        invokeInference(generator.defineClassBytes(route.specialization(), first), stores, g, plan);
        assertStorage(expected, stores.getLast(), f); assertSnapshots(before, stores.subList(0, 5), f);
    }

    private static void training(Fixture f) throws Throwable {
        PrepareContext<CpuPartitionAnalysisInputs> context = configure(CpuBatchNormTrainingLoweringTest.context(
                f.types, SHAPE, f.axis, List.of(0, 1, 2, 3, 4), trainingLayouts(SHAPE, f.axis)), f.request);
        CpuPartitionPreparationPlan plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan(); var generator = new CpuClassFileKernelGenerator();
        byte[] first = generator.generateClassBytes(route.specialization(), route.kernelIr());
        assertArrayEquals(first, generator.generateClassBytes(route.specialization(), route.kernelIr()), f + " deterministic bytes");
        assertEquals(0, plan.materializations().size(), f + " direct materialization fact");
        List<Storage> stores = stores(route.specialization().boundaryDataTypes(), route.specialization().carrierPattern());
        int unique = stores.size() - 5; fillInputs(context, stores, unique); for (int i = unique; i < stores.size(); i++) stores.get(i).fill(-91);
        List<double[]> before = snapshots(stores.subList(0, unique)); double[][] expected = new double[5][];
        for (int i = 0; i < 5; i++) expected[i] = stores.get(unique + i).snapshot();
        CpuBatchNormTrainingLowering.Geometry g = plan.batchNormTrainingGeometry().orElseThrow();
        trainingOracle(g, stores, expected);
        invokeTraining(generator.defineClassBytes(route.specialization(), first), stores, g, plan);
        for (int i = 0; i < 5; i++) assertStorage(expected[i], stores.get(unique + i), f + " output " + i);
        assertSnapshots(before, stores.subList(0, unique), f);
    }

    private static void invokeInference(CpuGeneratedKernel artifact, List<Storage> stores,
            CpuBatchNormInferenceLowering.Geometry g, CpuPartitionPreparationPlan plan) throws Throwable {
        Object[] boundaries = stores.stream().map(Storage::argument).toArray();
        long end = g.rangeForm() == CpuBatchNormInferenceIr.RangeForm.CHANNEL_RANGE ? g.channelCount() : g.nonChannelCount();
        for (long[] range : ranges(end, plan.executionStrategy().orchestration()
                == CpuPartitionPreparationPlan.ExecutionStrategy.Orchestration.PARALLEL ? plan.selectedRangeCount() : 1))
            entry(artifact, boundaries, g.pack(new long[stores.size()]), range[0], range[1]);
    }

    private static void invokeTraining(CpuGeneratedKernel artifact, List<Storage> stores,
            CpuBatchNormTrainingLowering.Geometry g, CpuPartitionPreparationPlan plan) throws Throwable {
        Object[] boundaries = stores.stream().map(Storage::argument).toArray();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment scratch = arena.allocate(g.workspaceBytes(plan.selectedRangeCount()), Long.BYTES);
            int ranges = plan.executionStrategy().orchestration()
                    == CpuPartitionPreparationPlan.ExecutionStrategy.Orchestration.PARALLEL ? plan.selectedRangeCount() : 1;
            int index = 0; for (long[] range : ranges(g.channelCount(), ranges)) {
                long[] geometry = g.pack(new long[stores.size()], index++ * g.scratchSliceBytes());
                Object[] args = new Object[boundaries.length + 4]; System.arraycopy(boundaries, 0, args, 0, boundaries.length);
                args[boundaries.length] = scratch; args[boundaries.length + 1] = geometry;
                args[boundaries.length + 2] = range[0]; args[boundaries.length + 3] = range[1];
                artifact.entryPoint().invokeWithArguments(args);
            }
        }
    }

    private static void entry(CpuGeneratedKernel artifact, Object[] boundaries, long[] geometry, long start, long end) throws Throwable {
        Object[] args = new Object[boundaries.length + 3]; System.arraycopy(boundaries, 0, args, 0, boundaries.length);
        args[boundaries.length] = geometry; args[boundaries.length + 1] = start; args[boundaries.length + 2] = end;
        artifact.entryPoint().invokeWithArguments(args);
    }

    /* Test-local scalar oracle. It deliberately does not call a lowering or production reference kernel. */
    private static void inferenceOracle(CpuBatchNormInferenceLowering.Geometry g, List<Storage> s, double[] expected) {
        DataType result = g.resultType(); int[] map = g.positionToBoundary().stream().mapToInt(Integer::intValue).toArray();
        for (long channel = 0; channel < g.channelCount(); channel++) {
            double scale = s.get(map[1]).get(vector(g.inputs().get(map[1]), channel));
            double bias = s.get(map[2]).get(vector(g.inputs().get(map[2]), channel));
            double mean = s.get(map[3]).get(vector(g.inputs().get(map[3]), channel));
            double variance = s.get(map[4]).get(vector(g.inputs().get(map[4]), channel));
            double denominator = result == DataType.FLOAT64 ? Math.sqrt(variance + epsilon(g.epsilonBits(), result))
                    : (float) Math.sqrt((float) variance + (float) epsilon(g.epsilonBits(), result));
            for (long ordinal = 0; ordinal < g.nonChannelCount(); ordinal++) {
                long input = tensor(g.inputs().get(map[0]), g.channelAxis(), channel, ordinal);
                long output = tensor(g.output(), g.channelAxis(), channel, ordinal);
                double value = op(result, op(result, op(result, s.get(map[0]).get(input), mean, '-'), denominator, '/'), scale, '*');
                expected[(int) output] = op(result, value, bias, '+');
            }
        }
    }

    private static void trainingOracle(CpuBatchNormTrainingLowering.Geometry g, List<Storage> s, double[][] expected) {
        DataType result = g.resultType(); int[] map = g.positionToBoundary().stream().mapToInt(Integer::intValue).toArray(); int unique = g.inputs().size();
        double momentum = epsilon(g.momentumBits(), result), eps = epsilon(g.epsilonBits(), result);
        for (long channel = 0; channel < g.channelCount(); channel++) {
            double sum = 0; for (long n = 0; n < g.reductionCount(); n++) sum += s.get(map[0]).get(tensor(g.inputs().get(map[0]), g.channelAxis(), channel, n));
            double mean = result == DataType.FLOAT64 ? sum / g.reductionCount() : (float) (sum / g.reductionCount());
            double dev = 0, squares = 0;
            for (long n = 0; n < g.reductionCount(); n++) { double d = op(result, s.get(map[0]).get(tensor(g.inputs().get(map[0]), g.channelAxis(), channel, n)), mean, '-'); dev += d; squares += d * d; }
            double numerator = Math.max(0, squares - dev * dev / g.reductionCount());
            double biased = op(result, numerator, g.reductionCount(), '/');
            double unbiased = op(result, numerator, g.reductionCount() - 1, '/');
            double saved = op(result, 1, Math.sqrt(op(result, biased, eps, '+')), '/');
            double scale = s.get(map[1]).get(vector(g.inputs().get(map[1]), channel));
            double bias = s.get(map[2]).get(vector(g.inputs().get(map[2]), channel));
            double oldMean = s.get(map[3]).get(vector(g.inputs().get(map[3]), channel));
            double oldVariance = s.get(map[4]).get(vector(g.inputs().get(map[4]), channel));
            double nextMean = op(result, op(result, op(result, 1, momentum, '-'), oldMean, '*'), op(result, momentum, mean, '*'), '+');
            double nextVariance = op(result, op(result, op(result, 1, momentum, '-'), oldVariance, '*'), op(result, momentum, unbiased, '*'), '+');
            expected[1][(int) vector(g.outputs().get(1), channel)] = nextMean;
            expected[2][(int) vector(g.outputs().get(2), channel)] = nextVariance;
            expected[3][(int) vector(g.outputs().get(3), channel)] = mean;
            expected[4][(int) vector(g.outputs().get(4), channel)] = saved;
            for (long n = 0; n < g.reductionCount(); n++) {
                long input = tensor(g.inputs().get(map[0]), g.channelAxis(), channel, n);
                long output = tensor(g.outputs().getFirst(), g.channelAxis(), channel, n);
                expected[0][(int) output] = op(result, op(result, op(result, op(result, s.get(map[0]).get(input), mean, '-'), saved, '*'), scale, '*'), bias, '+');
            }
        }
    }

    private static double op(DataType type, double left, double right, char operator) {
        if (type == DataType.FLOAT64) return switch (operator) { case '+' -> left + right; case '-' -> left - right; case '*' -> left * right; case '/' -> left / right; default -> throw new AssertionError(); };
        float a = (float) left, b = (float) right; return switch (operator) { case '+' -> a + b; case '-' -> a - b; case '*' -> a * b; case '/' -> a / b; default -> throw new AssertionError(); };
    }
    private static double epsilon(long bits, DataType type) { return switch (type) { case FLOAT64 -> Double.longBitsToDouble(bits); case FLOAT32 -> Float.intBitsToFloat((int) bits); case BFLOAT16 -> Float.intBitsToFloat((int) bits << 16); default -> throw new AssertionError(type); }; }
    private static long vector(Object layout, long channel) { return layout instanceof CpuBatchNormInferenceLowering.Layout l ? l.offset() + channel * l.strides()[0] : ((CpuBatchNormTrainingLowering.Layout) layout).offset() + channel * ((CpuBatchNormTrainingLowering.Layout) layout).strides()[0]; }
    private static long tensor(Object layout, int axis, long channel, long ordinal) {
        long[] extents, strides; long address;
        if (layout instanceof CpuBatchNormInferenceLowering.Layout l) { extents = l.extents(); strides = l.strides(); address = l.offset(); }
        else { var l = (CpuBatchNormTrainingLowering.Layout) layout; extents = l.extents(); strides = l.strides(); address = l.offset(); }
        address += channel * strides[axis]; for (int a = extents.length - 1; a >= 0; a--) if (a != axis) { address += (ordinal % extents[a]) * strides[a]; ordinal /= extents[a]; }
        return address;
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> configure(PrepareContext<CpuPartitionAnalysisInputs> base, Request request) {
        List<GraphValue> values = new ArrayList<>(); List<LogicalMemoryRequirement> memory = new ArrayList<>();
        for (int i = 0; i < base.values().size(); i++) { TensorDescriptor d = base.values().get(i).descriptor();
            if (request.general && d.shape().rank() != 0) { long[] strides = d.layout().orElseThrow().strides(); for (int a = 0; a < strides.length; a++) strides[a] *= 2;
                d = new TensorDescriptor(d.dataType(), d.shape(), Optional.of(LayoutDescriptor.of(d.shape(), strides, 1, true)), false); }
            values.add(new GraphValue(base.values().get(i).id(), d)); var r = base.memoryRequirements().get(i);
            memory.add(new LogicalMemoryRequirement(r.valueId(), d, r.producerPartition(), r.consumerPartitions(), r.graphOutput())); }
        List<CarrierAccess> carriers = new ArrayList<>(); for (int i = 0; i < values.size(); i++) carriers.add(request.segment || request.mixed && i % 2 == 1 ? CarrierAccess.MEMORY_SEGMENT : CpuGeneratedDirectEvidenceClosureTest.heapCarrier(values.get(i).descriptor().dataType()));
        return new PrepareContext<>(base.partition(), base.nodes(), values, memory, base.constants(), new CpuPartitionAnalysisInputs(false, carriers, request.execution, request.materialization ? MATERIALIZATION : CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED));
    }
    private static List<LayoutDescriptor> trainingLayouts(Shape shape, int axis) { Shape v = Shape.of(shape.toLongArray()[axis]); List<LayoutDescriptor> result = new ArrayList<>(); result.add(LayoutDescriptor.contiguous(shape)); for (int i = 1; i < 5; i++) result.add(LayoutDescriptor.contiguous(v)); result.add(LayoutDescriptor.contiguous(shape)); for (int i = 1; i < 5; i++) result.add(LayoutDescriptor.contiguous(v)); return result; }
    private static List<List<DataType>> roles() { List<List<DataType>> result = new ArrayList<>(); for (DataType a : FLOATING) for (DataType b : FLOATING) for (DataType c : FLOATING) for (DataType d : FLOATING) for (DataType e : FLOATING) result.add(List.of(a,b,c,d,e)); return result; }
    private static List<Storage> stores(List<DataType> types, List<CarrierAccess> carriers) { List<Storage> result = new ArrayList<>(); for (int i = 0; i < types.size(); i++) result.add(new Storage(types.get(i), carriers.get(i))); return result; }
    private static void fillInputs(PrepareContext<CpuPartitionAnalysisInputs> context, List<Storage> s, int inputs) { for (int i = 0; i < inputs; i++) { TensorDescriptor d = context.values().get(i).descriptor(); long count = elements(d.shape()); for (long n = 0; n < count; n++) s.get(i).set(offset(d.layout().orElseThrow(), d.shape(), n), i == 0 ? (n % 7 - 3) * .25 : (i == 1 ? 1.25 : i == 2 ? -.5 : i == 3 ? .75 : 2.0)); } }
    private static Set<String> inventoryOwners(String form) throws Exception { try (var stream = CpuBatchNormSemanticClosureTest.class.getResourceAsStream("/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/generated-coverage-inventory.tsv")) { assertNotNull(stream); Set<String> result = new TreeSet<>(); for (String line : new String(stream.readAllBytes(), StandardCharsets.UTF_8).split("\\n")) { String[] fields = line.split("\\t", -1); if (fields.length > 2 && fields[2].equals(form)) assertTrue(result.add(fields[0]), "duplicate inventory owner " + fields[0]); } return result; } }
    private static long[][] ranges(long end, int parts) { int count = (int) Math.min(Math.max(1, parts), Math.max(1, end)); long[][] result = new long[count][2]; for (int i = 0; i < count; i++) { result[i][0] = end * i / count; result[i][1] = end * (i + 1) / count; } return result; }
    private static long offset(LayoutDescriptor l, Shape shape, long linear) { long[] dimensions = shape.toLongArray(), strides = l.strides(); long result = l.storageOffset(); for (int a = dimensions.length - 1; a >= 0; a--) { result += (linear % dimensions[a]) * strides[a]; linear /= dimensions[a]; } return result; }
    private static long elements(Shape shape) { long result = 1; for (long x : shape.toLongArray()) result *= x; return result; }
    private static List<double[]> snapshots(List<Storage> s) { return s.stream().map(Storage::snapshot).toList(); }
    private static void assertSnapshots(List<double[]> before, List<Storage> after, Object label) { for (int i = 0; i < before.size(); i++) assertArrayEquals(before.get(i), after.get(i).snapshot(), label + " input mutation " + i); }
    private static void assertStorage(double[] expected, Storage actual, Object label) { for (int i = 0; i < expected.length; i++) assertEquals(actual.quantize(expected[i]), actual.get(i), 0, label + " physical " + i); }
    private record Fixture(boolean training, List<DataType> types, int axis, Request request) { String owner() { return "specialized:batch-" + (training ? "training" : "inference") + '/' + types + '/' + axis + '/' + request; } }
    private enum Request { HEAP_CONTIGUOUS_SCALAR(false,false,false,false,CpuGeneratedDirectEvidenceClosureTest.scalar(1)), SEGMENT_CONTIGUOUS_VECTOR(true,false,false,false,CpuGeneratedDirectEvidenceClosureTest.vector(1)), MIXED_GENERAL_PARALLEL_VECTOR(false,true,true,false,CpuGeneratedDirectEvidenceClosureTest.vector(4)), HEAP_GENERAL_PARALLEL_SCALAR(false,false,true,false,CpuGeneratedDirectEvidenceClosureTest.scalar(4)), HEAP_GENERAL_MATERIALIZATION(false,false,true,true,CpuGeneratedDirectEvidenceClosureTest.scalar(1)); final boolean segment,mixed,general,materialization; final CpuPartitionAnalysisInputs.PortableExecutionConfig execution; Request(boolean segment,boolean mixed,boolean general,boolean materialization,CpuPartitionAnalysisInputs.PortableExecutionConfig execution) { this.segment=segment;this.mixed=mixed;this.general=general;this.materialization=materialization;this.execution=execution; } }
    private static final class Storage { final DataType type; final CarrierAccess carrier; final Object heap; final MemorySegment segment; Storage(DataType type, CarrierAccess carrier) { this.type=type;this.carrier=carrier; heap=switch(type){case FLOAT64->new double[128];case FLOAT32->new float[128];case BFLOAT16->new short[128];default->throw new AssertionError(type);}; segment=switch(type){case FLOAT64->MemorySegment.ofArray((double[])heap);case FLOAT32->MemorySegment.ofArray((float[])heap);case BFLOAT16->MemorySegment.ofArray((short[])heap);default->throw new AssertionError(type);}; } Object argument(){return carrier==CarrierAccess.MEMORY_SEGMENT?segment:heap;} void fill(double value){for(int i=0;i<128;i++)set(i,value);} void set(long i,double v){switch(type){case FLOAT64->((double[])heap)[(int)i]=v;case FLOAT32->((float[])heap)[(int)i]=(float)v;case BFLOAT16->((short[])heap)[(int)i]=ScalarValue.bfloat16((float)v).bfloat16Bits();default->throw new AssertionError(type);}} double get(long i){return switch(type){case FLOAT64->((double[])heap)[(int)i];case FLOAT32->((float[])heap)[(int)i];case BFLOAT16->Float.intBitsToFloat((((short[])heap)[(int)i]&0xffff)<<16);default->throw new AssertionError(type);};} double quantize(double v){return switch(type){case FLOAT64->v;case FLOAT32->(float)v;case BFLOAT16->Float.intBitsToFloat((ScalarValue.bfloat16((float)v).bfloat16Bits()&0xffff)<<16);default->throw new AssertionError(type);};} double[] snapshot(){double[] r=new double[128];for(int i=0;i<r.length;i++)r[i]=get(i);return r;} }
}
