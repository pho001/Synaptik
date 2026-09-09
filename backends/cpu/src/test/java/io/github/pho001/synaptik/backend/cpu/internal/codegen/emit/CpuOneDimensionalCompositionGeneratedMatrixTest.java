package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuConv2dLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionDagDecomposer;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPool2dLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScatterLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.convolution.Conv1dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool1dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool1dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.Pool2dKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Matrix evidence for the actual Model-owned NCW-to-NCHW one-dimensional compositions. */
class CpuOneDimensionalCompositionGeneratedMatrixTest {
    private static final List<DataType> FLOATING = List.of(DataType.BFLOAT16, DataType.FLOAT32,
            DataType.FLOAT64);
    private static final Map<DataType, List<Request>> ADMITTED_POOL_REQUESTS = Map.of(
            DataType.BFLOAT16, List.of(Request.MIXED_GENERAL_PARALLEL_VECTOR,
                    Request.HEAP_GENERAL_PARALLEL_SCALAR, Request.HEAP_GENERAL_MATERIALIZATION),
            DataType.FLOAT32, List.of(Request.MIXED_GENERAL_PARALLEL_VECTOR,
                    Request.HEAP_GENERAL_PARALLEL_SCALAR, Request.HEAP_GENERAL_MATERIALIZATION),
            DataType.FLOAT64, List.of(Request.MIXED_GENERAL_PARALLEL_VECTOR,
                    Request.HEAP_GENERAL_PARALLEL_SCALAR, Request.HEAP_GENERAL_MATERIALIZATION));

    /** Representative real Model compositions for the independent checkpoint evidence join. */
    static List<PrepareContext<CpuPartitionAnalysisInputs>> coverageFixtureContexts() {
        return List.of(configure(convContext(DataType.FLOAT32, DataType.FLOAT32, null,
                        convCases().getFirst()), Request.MIXED_GENERAL_PARALLEL_VECTOR),
                configure(poolContext(DataType.FLOAT32, poolCases().getFirst()),
                        Request.MIXED_GENERAL_PARALLEL_VECTOR));
    }

    /** Exact Model-provenance Conv1d fixture for semantic-closure execution evidence. */
    static PrepareContext<CpuPartitionAnalysisInputs> semanticConvContext(String geometry,
            DataType input, DataType weight, DataType bias) {
        return convContext(input, weight, bias, convCases().stream()
                .filter(candidate -> candidate.id.equals(geometry)).findFirst().orElseThrow());
    }

    /** Exact Model-provenance Pool1d fixture for direct generated-entry semantic evidence. */
    static PrepareContext<CpuPartitionAnalysisInputs> semanticPoolContext(String geometry,
            DataType type) {
        return poolContext(type, poolCases().stream()
                .filter(candidate -> candidate.id.equals(geometry)).findFirst().orElseThrow());
    }

    /** Applies one exact checked carrier/layout/strategy request to a semantic fixture. */
    static PrepareContext<CpuPartitionAnalysisInputs> semanticConfigure(
            PrepareContext<CpuPartitionAnalysisInputs> context, String request) {
        return configure(context, Request.valueOf(request));
    }

    /** The finite provider-admitted Pool1d composition requests rejected by preparation. */
    static List<CpuGeneratedCoverageEvidenceRegistry.PreparerRejectedFixture> rejectedPoolFixtures() {
        var fixtures = new ArrayList<CpuGeneratedCoverageEvidenceRegistry.PreparerRejectedFixture>();
        for (PoolCase geometry : poolCases()) for (Request request : Request.values()) {
            if (!poolRequestIsAdmitted(DataType.BFLOAT16, request)) {
                fixtures.add(new CpuGeneratedCoverageEvidenceRegistry.PreparerRejectedFixture(
                        "pool1d/" + geometry.id + '/' + DataType.BFLOAT16 + '/' + request,
                        "POOL1D_COMPOSITION", configure(poolContext(DataType.BFLOAT16, geometry), request),
                        "POOL1D_COMPOSITION_AFFINE_BOUNDARY"));
            }
        }
        return List.copyOf(fixtures);
    }

    @Test void everyModelValidConv1dCompositionRoleAndGeometryGeneratesAcrossRequests() {
        int rows = 0, generated = 0;
        for (ConvCase geometry : convCases()) for (DataType input : FLOATING)
            for (DataType weight : FLOATING) {
                generated += verifyConv("conv1d/" + geometry.id + "/plain/" + input + '/' + weight,
                        convContext(input, weight, null, geometry));
                rows++;
                for (DataType bias : FLOATING) {
                    generated += verifyConv("conv1d/" + geometry.id + "/bias/" + input + '/' + weight + '/' + bias,
                            convContext(input, weight, bias, geometry));
                    rows++;
                }
            }
        assertEquals(108, rows, "3 channel forms × 3 geometry mappings × 36 role forms");
        assertEquals(540, generated, "every Conv1d row owns one generated unit for five requests");
    }

    @Test void everyModelValidPool1dCompositionTypeAndGeometryGeneratesAcrossRequests() {
        int rows = 0, generated = 0;
        for (PoolCase geometry : poolCases()) for (DataType type : FLOATING) {
            generated += verifyPool("pool1d/" + geometry.id + '/' + type, poolContext(type, geometry));
            rows++;
        }
        assertEquals(12, rows, "two geometries for each pooling kind × three types");
        assertEquals(36, generated, "twelve rows × three exact general-layout requests");
        assertEquals(8, rejectedPoolFixtures().size(), "four BFLOAT16 geometries × two contiguous requests");
    }

    @Test void malformedPublishedAndNearMatchCompositionsRemainOutsideTheOwnedRoute() {
        var valid = convContext(DataType.FLOAT32, DataType.FLOAT32, null, convCases().getFirst());
        var wrongAxis = replace(valid, 0, new Operation(AxisTransformKind.EXPAND_DIMS,
                new AxisTransformAttrs(1)), false);
        var published = replace(valid, -1, null, true);
        var nearMatch = replace(valid, 2, new Operation(Conv2dKind.CONV2D,
                new Conv2dAttrs(2, 1, 0, 0, 1, 1, 1)), false);
        assertAll(
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuPartitionLowering().lower(wrongAxis)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuPartitionLowering().lower(published)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuPartitionDagDecomposer().decompose(nearMatch,
                                new CpuPartitionLowering())));
    }

    private static int verifyConv(String id, PrepareContext<CpuPartitionAnalysisInputs> base) {
        assertActualConvTopology(base, id);
        int generated = 0;
        for (Request request : Request.values()) {
            var context = configure(base, request);
            assertTrue(new CpuCapabilityProvider().supports(new OperationCapabilityQuery(
                    context.nodes().get(2).operation(), descriptors(context, context.nodes().get(2).inputs()),
                    descriptors(context, context.nodes().get(2).outputs()))), id + "/mapped Conv2d");
            var plan = new CpuPartitionPreparer().analyze(context).plan();
            assertEquals(1, plan.units().size(), id + '/' + request + " must remain one owned unit");
            var unit = plan.units().getFirst();
            assertEquals(context.nodes().get(2).inputs().size() + 1, unit.boundaryValues().size(), id + '/' + request);
            assertEquals(context.nodes().get(2).inputs().size() + 1,
                    unit.portablePlan().specialization().boundaryDataTypes().size(), id + '/' + request);
            assertEquals(plan.executionStrategy(), unit.executionStrategy(), id + '/' + request);
            assertTrue(plan.materializations().isEmpty(), id + '/' + request);
            byte[] bytes = generated(unit.portablePlan());
            CpuGeneratedCoverageEvidenceRegistry.generated("composition:" + id + '/' + request,
                    context, plan);
            var direct = new CpuPartitionPreparer().analyze(withRequest(directConv(context), context)).plan();
            byte[] directBytes = generated(direct.units().getFirst().portablePlan());
            // The synthetic Conv2d route has equivalent semantics but deliberately retains a
            // different composition-specialization identity/body boundary from Conv1d.
            assertNotEquals(direct.units().getFirst().portablePlan().specialization(),
                    unit.portablePlan().specialization(), id + '/' + request);
            assertFalse(java.util.Arrays.equals(directBytes, bytes), id + '/' + request);
            generated++;
        }
        return generated;
    }

    private static int verifyPool(String id, PrepareContext<CpuPartitionAnalysisInputs> base) {
        assertActualPoolTopology(base, id);
        int generated = 0;
        for (Request request : Request.values()) {
            var context = configure(base, request);
            if (!poolRequestIsAdmitted(base.values().getFirst().descriptor().dataType(), request)) {
                assertPoolRejection(id, request, context);
                continue;
            }
            assertTrue(new CpuCapabilityProvider().supports(new OperationCapabilityQuery(
                    context.nodes().get(1).operation(), descriptors(context, context.nodes().get(1).inputs()),
                    descriptors(context, context.nodes().get(1).outputs()))), id + "/mapped Pool2d");
            var plan = assertDoesNotThrow(() -> new CpuPartitionPreparer().analyze(context).plan(),
                    id + '/' + request + " admitted pool request must prepare");
            assertEquals(1, plan.units().size(), id + '/' + request);
            var unit = plan.units().getFirst();
            assertEquals(2, unit.boundaryValues().size(), id + '/' + request);
            assertEquals(2, unit.portablePlan().specialization().boundaryDataTypes().size(), id + '/' + request);
            assertEquals(plan.executionStrategy(), unit.executionStrategy(), id + '/' + request);
            assertTrue(plan.materializations().isEmpty(), id + '/' + request);
            byte[] bytes = generated(unit.portablePlan());
            CpuGeneratedCoverageEvidenceRegistry.generated("composition:" + id + '/' + request,
                    context, plan);
            var direct = new CpuPartitionPreparer().analyze(withRequest(directPool(context), context)).plan();
            byte[] directBytes = generated(direct.units().getFirst().portablePlan());
            assertEquals(direct.units().getFirst().portablePlan().portableKernelIr(), unit.portablePlan().portableKernelIr(), id + '/' + request);
            assertEquals(direct.units().getFirst().portablePlan().specialization(), unit.portablePlan().specialization(), id + '/' + request);
            assertArrayEquals(directBytes, bytes, id + '/' + request);
            generated++;
        }
        return generated;
    }

    private static boolean poolRequestIsAdmitted(DataType type, Request request) {
        // This lookup is the exact checked request/type matrix, not an inference from the direct
        // Pool2d provider capability. BFLOAT16 is explicitly fail-closed for both contiguous
        // requests even though each direct Pool2d semantic query remains provider-supported.
        return ADMITTED_POOL_REQUESTS.get(type).contains(request);
    }

    private static void assertPoolRejection(String id, Request request,
            PrepareContext<CpuPartitionAnalysisInputs> context) {
        var provider = new CpuCapabilityProvider();
        var pool = context.nodes().get(1);
        assertTrue(provider.supports(new OperationCapabilityQuery(pool.operation(),
                descriptors(context, pool.inputs()), descriptors(context, pool.outputs()))),
                id + '/' + request + " direct Pool2d capability is not composition admission");
        var failure = assertThrows(IllegalArgumentException.class,
                () -> new CpuPartitionPreparer().analyze(context), id + '/' + request);
        assertEquals("CPU partition contains an independently unsupported node at ordinal 0",
                failure.getMessage(), id + '/' + request + " preparer rejection category");
        assertEquals("partition contains an unsupported CPU affine occurrence",
                failure.getCause().getMessage(), id + '/' + request + " affine rejection reason");
    }

    private static byte[] generated(CpuPortableRoutePlan route) {
        byte[] bytes = new CpuClassFileKernelGenerator().generateClassBytes(route.specialization(), route.kernelIr());
        CpuGeneratedDirectEvidenceClosureTest.assertClosedSpecializedClass(bytes, route.specialization());
        return bytes;
    }

    private static void assertCapability(PrepareContext<CpuPartitionAnalysisInputs> context, String id) {
        var provider = new CpuCapabilityProvider();
        for (CompiledNode node : context.nodes()) assertTrue(provider.supports(new OperationCapabilityQuery(
                node.operation(), descriptors(context, node.inputs()), descriptors(context, node.outputs()))), id + '/' + node.id());
    }

    private static List<TensorDescriptor> descriptors(PrepareContext<?> context, List<ValueId> ids) {
        return ids.stream().map(id -> context.values().stream().filter(value -> value.id().equals(id))
                .findFirst().orElseThrow().descriptor()).toList();
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> directConv(PrepareContext<CpuPartitionAnalysisInputs> context) {
        var node = context.nodes().get(2); var attrs = (Conv2dAttrs) node.operation().attrs();
        var values = context.values(); var input = values.getFirst().descriptor(); var weight = values.get(1).descriptor();
        var output = values.getLast().descriptor();
        var types = new ArrayList<DataType>(List.of(input.dataType(), weight.dataType()));
        if (node.inputs().size() == 3) types.add(values.get(2).descriptor().dataType());
        return CpuConv2dLoweringTest.context(types, Shape.of(1, input.shape().toLongArray()[1], 1, input.shape().toLongArray()[2]),
                Shape.of(weight.shape().toLongArray()[0], weight.shape().toLongArray()[1], 1, weight.shape().toLongArray()[2]),
                Shape.of(1, output.shape().toLongArray()[1], 1, output.shape().toLongArray()[2]), attrs, null);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> directPool(PrepareContext<CpuPartitionAnalysisInputs> context) {
        var node = context.nodes().get(1); var input = context.values().getFirst().descriptor(); var output = context.values().getLast().descriptor();
        Shape directInput = Shape.of(1, input.shape().toLongArray()[1], 1, input.shape().toLongArray()[2]);
        Shape directOutput = Shape.of(1, output.shape().toLongArray()[1], 1, output.shape().toLongArray()[2]);
        return CpuScatterLoweringTest.context(new Operation((Pool2dKind) node.operation().kind(), node.operation().attrs()), List.of(0),
                List.of(new TensorDescriptor(input.dataType(), directInput, Optional.of(context.values().get(1).descriptor().layout().orElseThrow()), false)),
                new TensorDescriptor(output.dataType(), directOutput, Optional.of(context.values().get(2).descriptor().layout().orElseThrow()), false));
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> withRequest(
            PrepareContext<CpuPartitionAnalysisInputs> direct,
            PrepareContext<CpuPartitionAnalysisInputs> source) {
        return new PrepareContext<>(direct.partition(), direct.nodes(), direct.values(),
                direct.memoryRequirements(), direct.constants(), source.backendInputs());
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> configure(PrepareContext<CpuPartitionAnalysisInputs> base, Request request) {
        var values = new ArrayList<GraphValue>(); var memory = new ArrayList<LogicalMemoryRequirement>();
        for (int i = 0; i < base.values().size(); i++) {
            var old = base.values().get(i); var descriptor = old.descriptor();
            if (request.general) { long[] strides = descriptor.layout().orElseThrow().strides();
                for (int axis = 0; axis < strides.length; axis++) strides[axis] *= 2;
                descriptor = new TensorDescriptor(descriptor.dataType(), descriptor.shape(),
                        Optional.of(LayoutDescriptor.of(descriptor.shape(), strides, 1, true)), descriptor.requiresGrad()); }
            values.add(new GraphValue(old.id(), descriptor)); var requirement = base.memoryRequirements().get(i);
            memory.add(new LogicalMemoryRequirement(requirement.valueId(), descriptor, requirement.producerPartition(), requirement.consumerPartitions(), requirement.graphOutput()));
        }
        var carriers = new ArrayList<CarrierAccess>();
        for (int i = 0; i < values.size(); i++) if (memory.get(i).producerPartition().isEmpty()
                || memory.get(i).graphOutput()) carriers.add(request.segment || request.mixed && i % 2 == 1
                ? CarrierAccess.MEMORY_SEGMENT : CpuGeneratedDirectEvidenceClosureTest.heapCarrier(values.get(i).descriptor().dataType()));
        return new PrepareContext<>(base.partition(), base.nodes(), values, memory, Map.of(), new CpuPartitionAnalysisInputs(false, carriers,
                request.execution, request.materialization ? new CpuPartitionAnalysisInputs.MaterializationPolicy(true, 0, 1, 20, 1, 3, 1_000_000, 1, 1) : CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED));
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> convContext(DataType input, DataType weight, DataType bias, ConvCase c) {
        // Build Model expressions first: their provenance is the source of the exact four nodes below.
        var x = TensorFactory.create(new TensorDescriptor(input, c.input, Optional.empty(), false));
        var w = TensorFactory.create(new TensorDescriptor(weight, c.weight, Optional.empty(), false));
        var y = bias == null ? x.conv1d(w, c.attrs) : x.conv1d(w, TensorFactory.create(new TensorDescriptor(bias, Shape.of(c.weight.toLongArray()[0]), Optional.empty(), false)), c.attrs);
        assertEquals(AxisTransformKind.SQUEEZE, y.provenance().orElseThrow().operation().kind());
        var convolution = y.provenance().orElseThrow().inputs().getFirst().provenance().orElseThrow();
        assertEquals(Conv2dKind.CONV2D, convolution.operation().kind());
        return composition(List.of(convolution.inputs().get(0).provenance().orElseThrow().operation(), convolution.inputs().get(1).provenance().orElseThrow().operation(), convolution.operation(), y.provenance().orElseThrow().operation()),
                java.util.Arrays.asList(input, weight, bias), List.of(c.input, c.weight, y.descriptor().shape()), true);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> poolContext(DataType type, PoolCase c) {
        var x = TensorFactory.create(new TensorDescriptor(type, c.input, Optional.empty(), false));
        var y = c.maximum ? x.maxPool1d(new MaxPool1dAttrs(c.kernel, c.stride, c.padding, c.dilation, c.ceil))
                : x.averagePool1d(new AveragePool1dAttrs(c.kernel, c.stride, c.padding, c.dilation, c.ceil));
        var pool = y.provenance().orElseThrow().inputs().getFirst().provenance().orElseThrow();
        return composition(List.of(pool.inputs().getFirst().provenance().orElseThrow().operation(), pool.operation(), y.provenance().orElseThrow().operation()),
                List.of(type), List.of(c.input, y.descriptor().shape()), false);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> composition(List<Operation> operations, List<DataType> types, List<Shape> external, boolean convolution) {
        Shape input = external.getFirst(); Shape output = external.getLast(); Shape expandedInput = Shape.of(input.toLongArray()[0], input.toLongArray()[1], 1, input.toLongArray()[2]);
        var shapes = new ArrayList<Shape>(); shapes.add(input);
        if (convolution) { Shape weight = external.get(1); shapes.add(weight); if (types.get(2) != null) shapes.add(Shape.of(weight.toLongArray()[0])); shapes.add(expandedInput); shapes.add(Shape.of(weight.toLongArray()[0], weight.toLongArray()[1], 1, weight.toLongArray()[2])); shapes.add(Shape.of(output.toLongArray()[0], output.toLongArray()[1], 1, output.toLongArray()[2])); shapes.add(output); }
        else { shapes.add(expandedInput); shapes.add(Shape.of(output.toLongArray()[0], output.toLongArray()[1], 1, output.toLongArray()[2])); shapes.add(output); }
        var ids = java.util.stream.LongStream.range(0, shapes.size()).mapToObj(ValueId::new).toList();
        var nodes = new ArrayList<CompiledNode>();
        if (convolution) { int bias = types.get(2) == null ? -1 : 2, ei = bias < 0 ? 2 : 3, ew = ei + 1, pooled = ei + 2, out = ei + 3;
            nodes.add(new CompiledNode(new NodeId(0), operations.get(0), List.of(ids.get(0)), List.of(ids.get(ei)))); nodes.add(new CompiledNode(new NodeId(1), operations.get(1), List.of(ids.get(1)), List.of(ids.get(ew))));
            var inputs = new ArrayList<ValueId>(List.of(ids.get(ei), ids.get(ew))); if (bias >= 0) inputs.add(ids.get(bias)); nodes.add(new CompiledNode(new NodeId(2), operations.get(2), inputs, List.of(ids.get(pooled)))); nodes.add(new CompiledNode(new NodeId(3), operations.get(3), List.of(ids.get(pooled)), List.of(ids.get(out))));
        } else { nodes.add(new CompiledNode(new NodeId(0), operations.get(0), List.of(ids.get(0)), List.of(ids.get(1)))); nodes.add(new CompiledNode(new NodeId(1), operations.get(1), List.of(ids.get(1)), List.of(ids.get(2)))); nodes.add(new CompiledNode(new NodeId(2), operations.get(2), List.of(ids.get(2)), List.of(ids.get(3)))); }
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID, nodes.stream().map(CompiledNode::id).toList()); var values = new ArrayList<GraphValue>(); var memory = new ArrayList<LogicalMemoryRequirement>();
        int biasIndex = convolution && types.get(2) != null ? 2 : -1;
        int expandedInputIndex = convolution ? biasIndex < 0 ? 2 : 3 : -1;
        int expandedWeightIndex = expandedInputIndex + 1;
        for (int i = 0; i < shapes.size(); i++) { ValueId id = ids.get(i); DataType type = convolution ? (i == 0 || i == expandedInputIndex ? types.get(0) : i == 1 || i == expandedWeightIndex ? types.get(1) : i == biasIndex ? types.get(2) : yType(types)) : types.getFirst();
            var descriptor = new TensorDescriptor(type, shapes.get(i), Optional.of(LayoutDescriptor.contiguous(shapes.get(i))), false); values.add(new GraphValue(id, descriptor)); boolean produced = nodes.stream().anyMatch(n -> n.outputs().contains(id)); memory.add(new LogicalMemoryRequirement(id, descriptor, produced ? Optional.of(partition) : Optional.empty(), i == shapes.size() - 1 ? List.of() : List.of(partition), i == shapes.size() - 1)); }
        return new PrepareContext<>(partition, nodes, values, memory, Map.of(), CpuPartitionAnalysisInputs.DEFAULT);
    }

    private static DataType yType(List<DataType> types) { DataType result = types.get(0); for (int i = 1; i < types.size(); i++) if (types.get(i) != null) result = io.github.pho001.synaptik.model.datatype.DataTypePromotion.promoteFloating(result, types.get(i)); return result; }
    private static void assertActualConvTopology(PrepareContext<?> c, String id) { assertEquals(List.of(AxisTransformKind.EXPAND_DIMS, AxisTransformKind.EXPAND_DIMS, Conv2dKind.CONV2D, AxisTransformKind.SQUEEZE), c.nodes().stream().map(n -> n.operation().kind()).toList(), id); }
    private static void assertActualPoolTopology(PrepareContext<?> c, String id) { assertEquals(List.of(AxisTransformKind.EXPAND_DIMS, c.nodes().get(1).operation().kind(), AxisTransformKind.SQUEEZE), c.nodes().stream().map(n -> n.operation().kind()).toList(), id); }
    private static PrepareContext<CpuPartitionAnalysisInputs> replace(PrepareContext<CpuPartitionAnalysisInputs> base, int index, Operation operation, boolean publish) { var nodes = new ArrayList<>(base.nodes()); if (index >= 0) { var old = nodes.get(index); nodes.set(index, new CompiledNode(old.id(), operation, old.inputs(), old.outputs())); } var memory = new ArrayList<LogicalMemoryRequirement>(); for (var value : base.memoryRequirements()) memory.add(new LogicalMemoryRequirement(value.valueId(), value.descriptor(), value.producerPartition(), value.consumerPartitions(), publish && value.valueId().equals(new ValueId(2)) || value.graphOutput())); return new PrepareContext<>(base.partition(), nodes, base.values(), memory, Map.of(), base.backendInputs()); }
    private static List<ConvCase> convCases() { return List.of(new ConvCase("ordinary", new Conv1dAttrs(2, 1, 2, 1), Shape.of(1, 2, 9), Shape.of(4, 2, 3)), new ConvCase("grouped", new Conv1dAttrs(1, 1, 1, 2), Shape.of(1, 2, 7), Shape.of(4, 1, 2)), new ConvCase("depthwise", new Conv1dAttrs(2, 2, 1, 2), Shape.of(1, 2, 8), Shape.of(2, 1, 3))); }
    private static List<PoolCase> poolCases() { return List.of(new PoolCase("max-floor", true, 3, 2, 1, 2, false, Shape.of(1, 2, 9)), new PoolCase("max-ceil", true, 2, 3, 2, 1, true, Shape.of(1, 2, 3)), new PoolCase("average-floor", false, 3, 2, 1, 2, false, Shape.of(1, 2, 9)), new PoolCase("average-ceil", false, 2, 3, 2, 1, true, Shape.of(1, 2, 3))); }
    private record ConvCase(String id, Conv1dAttrs attrs, Shape input, Shape weight) { }
    private record PoolCase(String id, boolean maximum, long kernel, long stride, long padding, long dilation, boolean ceil, Shape input) { }
    private enum Request { HEAP_CONTIGUOUS_SCALAR(false, false, false, CpuGeneratedDirectEvidenceClosureTest.scalar(1)), SEGMENT_CONTIGUOUS_VECTOR(true, false, false, CpuGeneratedDirectEvidenceClosureTest.vector(1)), MIXED_GENERAL_PARALLEL_VECTOR(false, true, true, CpuGeneratedDirectEvidenceClosureTest.vector(4)), HEAP_GENERAL_PARALLEL_SCALAR(false, false, true, CpuGeneratedDirectEvidenceClosureTest.scalar(4)), HEAP_GENERAL_MATERIALIZATION(false, false, true, CpuGeneratedDirectEvidenceClosureTest.scalar(1)); final boolean segment, mixed, general, materialization; final CpuPartitionAnalysisInputs.PortableExecutionConfig execution; Request(boolean segment, boolean mixed, boolean general, CpuPartitionAnalysisInputs.PortableExecutionConfig execution) { this.segment = segment; this.mixed = mixed; this.general = general; this.materialization = name().contains("MATERIALIZATION"); this.execution = execution; } }
}
