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

/** Shared Model-owned NCW-to-NCHW fixtures and malformed-composition guards. */
class CpuOneDimensionalCompositionGeneratedMatrixTest {
    private static final List<DataType> FLOATING = List.of(DataType.BFLOAT16, DataType.FLOAT16,
            DataType.FLOAT32, DataType.FLOAT64);
    private static final Map<DataType, List<Request>> ADMITTED_POOL_REQUESTS = Map.of(
            DataType.BFLOAT16, List.of(Request.MIXED_GENERAL_PARALLEL_VECTOR,
                    Request.HEAP_GENERAL_PARALLEL_SCALAR, Request.HEAP_GENERAL_MATERIALIZATION),
            DataType.FLOAT16, List.of(Request.MIXED_GENERAL_PARALLEL_VECTOR,
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

    /**
     * Retains carrier/layout requests outside the currently qualified Pool1d composition route.
     * Direct Pool2d capability does not imply admission of these composed requests.
     * @return rejected source contexts and their semantic forms
     */
    static List<CpuGeneratedCoverageEvidenceRegistry.PreparerRejectedFixture> rejectedPoolFixtures() {
        var fixtures = new ArrayList<CpuGeneratedCoverageEvidenceRegistry.PreparerRejectedFixture>();
        for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT16))
            for (PoolCase geometry : poolCases()) for (Request request : Request.values()) {
                if (!poolRequestIsAdmitted(type, request)) {
                    fixtures.add(new CpuGeneratedCoverageEvidenceRegistry.PreparerRejectedFixture(
                            "pool1d/" + geometry.id + '/' + type + '/' + request,
                            "POOL1D_COMPOSITION", configure(poolContext(type, geometry), request),
                            "POOL1D_COMPOSITION_AFFINE_BOUNDARY"));
                }
            }
        return List.copyOf(fixtures);
    }

    void everyModelValidConv1dCompositionRoleAndGeometryGeneratesAcrossRequests() {
        for (ConvCase geometry : convCases()) for (DataType input : FLOATING)
            for (DataType weight : FLOATING) {
                String plain = "conv1d/" + geometry.id + "/plain/" + input + '/' + weight;
                verifyConv(plain, convContext(input, weight, null, geometry));
                for (DataType bias : FLOATING) {
                    String biased = "conv1d/" + geometry.id + "/bias/" + input + '/' + weight + '/' + bias;
                    verifyConv(biased, convContext(input, weight, bias, geometry));
                }
            }
    }

    void everyModelValidPool1dCompositionTypeAndGeometryGeneratesAcrossRequests() {
        for (PoolCase geometry : poolCases()) for (DataType type : FLOATING) {
            String owner = "pool1d/" + geometry.id + '/' + type;
            verifyPool(owner, poolContext(type, geometry));
        }
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

    private static void verifyConv(String id, PrepareContext<CpuPartitionAnalysisInputs> base) {
        for (Request request : Request.values()) {
            var context = configure(base, request);
            io.github.pho001.synaptik.backend.cpu.CpuGeneratedCoverageUnsupportedBoundaryTest.assertCurrentSupport(
                    new CpuCapabilityProvider(), new OperationCapabilityQuery(context.nodes().get(2).operation(),
                            descriptors(context, context.nodes().get(2).inputs()),
                            descriptors(context, context.nodes().get(2).outputs())), id + "/mapped Conv2d");
            var plan = new CpuPartitionPreparer().analyze(context).plan();
            CpuGeneratedCoverageEvidenceRegistry.generated("composition:" + id + '/' + request, context, plan);
        }
    }

    private static void verifyPool(String id, PrepareContext<CpuPartitionAnalysisInputs> base) {
        for (Request request : Request.values()) {
            var context = configure(base, request);
            if (!poolRequestIsAdmitted(base.values().getFirst().descriptor().dataType(), request)) {
                assertPoolRejection(id, request, context);
                continue;
            }
            io.github.pho001.synaptik.backend.cpu.CpuGeneratedCoverageUnsupportedBoundaryTest.assertCurrentSupport(
                    new CpuCapabilityProvider(), new OperationCapabilityQuery(context.nodes().get(1).operation(),
                            descriptors(context, context.nodes().get(1).inputs()),
                            descriptors(context, context.nodes().get(1).outputs())), id + "/mapped Pool2d");
            var plan = new CpuPartitionPreparer().analyze(context).plan();
            CpuGeneratedCoverageEvidenceRegistry.generated(
                    poolOccurrenceId(id, base.values().getFirst().descriptor().dataType(), request), context, plan);
        }
    }

    /**
     * Keeps the finite composition request identity separate from direct Pool2d admission.
     * @param id original Pool1d source owner
     * @param type original data type for the historical request matrix
     * @param request exact source carrier/layout/strategy request
     * @return original generated-owner or rejected-owner occurrence identity
     */
    private static String poolOccurrenceId(String id, DataType type, Request request) {
        return (poolRequestIsAdmitted(type, request) ? "composition:" : "rejected:composition:")
                + id + '/' + request;
    }

    private static boolean poolRequestIsAdmitted(DataType type, Request request) {
        // Preserve the historical request/type matrix independently of current arithmetic
        // availability. Direct Pool2d capability never implied admission of contiguous
        // Pool1d composition requests.
        return ADMITTED_POOL_REQUESTS.get(type).contains(request);
    }

    private static void assertPoolRejection(String id, Request request,
            PrepareContext<CpuPartitionAnalysisInputs> context) {
        var provider = new CpuCapabilityProvider();
        var pool = context.nodes().get(1);
        io.github.pho001.synaptik.backend.cpu.CpuGeneratedCoverageUnsupportedBoundaryTest.assertCurrentSupport(provider, new OperationCapabilityQuery(pool.operation(), descriptors(context, pool.inputs()), descriptors(context, pool.outputs())), id + '/' + request + " direct Pool2d capability is not composition admission");
        var failure = assertThrows(IllegalArgumentException.class,
                () -> new CpuPartitionPreparer().analyze(context), id + '/' + request);
    }

    

    

    private static List<TensorDescriptor> descriptors(PrepareContext<?> context, List<ValueId> ids) {
        return ids.stream().map(id -> context.values().stream().filter(value -> value.id().equals(id))
                .findFirst().orElseThrow().descriptor()).toList();
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
    
    
    private static PrepareContext<CpuPartitionAnalysisInputs> replace(PrepareContext<CpuPartitionAnalysisInputs> base, int index, Operation operation, boolean publish) { var nodes = new ArrayList<>(base.nodes()); if (index >= 0) { var old = nodes.get(index); nodes.set(index, new CompiledNode(old.id(), operation, old.inputs(), old.outputs())); } var memory = new ArrayList<LogicalMemoryRequirement>(); for (var value : base.memoryRequirements()) memory.add(new LogicalMemoryRequirement(value.valueId(), value.descriptor(), value.producerPartition(), value.consumerPartitions(), publish && value.valueId().equals(new ValueId(2)) || value.graphOutput())); return new PrepareContext<>(base.partition(), nodes, base.values(), memory, Map.of(), base.backendInputs()); }
    private static List<ConvCase> convCases() { return List.of(new ConvCase("ordinary", new Conv1dAttrs(2, 1, 2, 1), Shape.of(1, 2, 9), Shape.of(4, 2, 3)), new ConvCase("grouped", new Conv1dAttrs(1, 1, 1, 2), Shape.of(1, 2, 7), Shape.of(4, 1, 2)), new ConvCase("depthwise", new Conv1dAttrs(2, 2, 1, 2), Shape.of(1, 2, 8), Shape.of(2, 1, 3))); }
    private static List<PoolCase> poolCases() { return List.of(new PoolCase("max-floor", true, 3, 2, 1, 2, false, Shape.of(1, 2, 9)), new PoolCase("max-ceil", true, 2, 3, 2, 1, true, Shape.of(1, 2, 3)), new PoolCase("average-floor", false, 3, 2, 1, 2, false, Shape.of(1, 2, 9)), new PoolCase("average-ceil", false, 2, 3, 2, 1, true, Shape.of(1, 2, 3))); }
    private record ConvCase(String id, Conv1dAttrs attrs, Shape input, Shape weight) { }
    private record PoolCase(String id, boolean maximum, long kernel, long stride, long padding, long dilation, boolean ceil, Shape input) { }
    private enum Request { HEAP_CONTIGUOUS_SCALAR(false, false, false, CpuGeneratedDirectEvidenceClosureTest.scalar(1)), SEGMENT_CONTIGUOUS_VECTOR(true, false, false, CpuGeneratedDirectEvidenceClosureTest.vector(1)), MIXED_GENERAL_PARALLEL_VECTOR(false, true, true, CpuGeneratedDirectEvidenceClosureTest.vector(4)), HEAP_GENERAL_PARALLEL_SCALAR(false, false, true, CpuGeneratedDirectEvidenceClosureTest.scalar(4)), HEAP_GENERAL_MATERIALIZATION(false, false, true, CpuGeneratedDirectEvidenceClosureTest.scalar(1)); final boolean segment, mixed, general, materialization; final CpuPartitionAnalysisInputs.PortableExecutionConfig execution; Request(boolean segment, boolean mixed, boolean general, CpuPartitionAnalysisInputs.PortableExecutionConfig execution) { this.segment = segment; this.mixed = mixed; this.general = general; this.materialization = name().contains("MATERIALIZATION"); this.execution = execution; } }
}
