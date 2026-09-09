package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAggregateLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuIndexingLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuOrderingLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRandomLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScanLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScatterLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.index.AxisGatherKind;
import io.github.pho001.synaptik.model.operation.index.AxisScatterKind;
import io.github.pho001.synaptik.model.operation.index.GatherNdKind;
import io.github.pho001.synaptik.model.operation.index.OneHotKind;
import io.github.pho001.synaptik.model.operation.index.ScatterNdKind;
import io.github.pho001.synaptik.model.operation.ordering.OrderingKind;
import io.github.pho001.synaptik.model.operation.ordering.SortAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKKind;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.TreeMap;
import org.junit.jupiter.api.Test;

/**
 * Exhaustive ordinary non-pointwise generated-route matrix.
 *
 * <p>The base universe is assembled from Model-valid role contracts, not from CPU predicates.
 * A base row is never rotated or sampled: each valid data/index/value/output role combination is
 * expanded before the real provider, preparer, and class-file generator are called.  The request
 * axes are intentionally recorded separately from the selected strategy because vector requests
 * are currently allowed to select scalar ordinary bodies.</p>
 */
class CpuOrdinaryNonPointwiseGeneratedMatrixTest {
    private static final List<DataType> ALL = List.of(DataType.FLOAT64, DataType.FLOAT32,
            DataType.BFLOAT16, DataType.INT64, DataType.INT32, DataType.BOOL);
    private static final List<DataType> NUMERIC = List.of(DataType.FLOAT64, DataType.FLOAT32,
            DataType.BFLOAT16, DataType.INT64, DataType.INT32);
    private static final List<DataType> FLOATING = List.of(DataType.FLOAT64, DataType.FLOAT32,
            DataType.BFLOAT16);
    private static final List<DataType> INDICES = List.of(DataType.INT64, DataType.INT32);

    @Test void everyModelValidOrdinaryBaseRoleTypeAndRuntimeRequestUsesProductionBoundaries()
            throws Exception {
        List<Base> bases = bases();
        assertEquals(bases.size(), bases.stream().map(Base::id).distinct().count(), "base ids");
        assertTrue(bases.size() > 300, "base universe must not be the former 59 rotating rows");
        var selected = new TreeMap<String, Long>();
        int runtimeCandidates = 0;
        for (Base base : bases) {
            for (Request request : requests()) {
                runtimeCandidates++;
                var context = configured(base.context(), request);
                var node = context.nodes().getFirst();
                var query = new OperationCapabilityQuery(node.operation(),
                        CpuGeneratedDirectEvidenceClosureTest.descriptors(context, node.inputs()),
                        CpuGeneratedDirectEvidenceClosureTest.descriptors(context, node.outputs()));
                assertTrue(new CpuCapabilityProvider().supports(query), base.id() + '/' + request.id());
                var plan = new CpuPartitionPreparer().analyze(context).plan();
                var route = plan.units().getFirst().portablePlan();
                var specialization = route.specialization();
                assertEquals(plan.generatedCarrierPattern(), specialization.carrierPattern(), base.id());
                assertEquals(plan.workspaceDeclaration().isPresent(), specialization.scratchParameter(), base.id());
                byte[] first = new CpuClassFileKernelGenerator().generateClassBytes(specialization, route.kernelIr());
                byte[] second = new CpuClassFileKernelGenerator().generateClassBytes(specialization, route.kernelIr());
                assertArrayEquals(first, second, base.id() + '/' + request.id());
                CpuGeneratedDirectEvidenceClosureTest.assertClosedSpecializedClass(first, specialization);
                CpuGeneratedCoverageEvidenceRegistry.generated("ordinary:" + base.id() + '/' + request.id(),
                        context, plan);
                selected.merge(request.id() + "->" + plan.executionStrategy() + ":m="
                        + plan.materializations().size(), 1L, Long::sum);
            }
        }
        assertEquals(bases.size() * requests().size(), runtimeCandidates);
        assertTrue(selected.keySet().stream().anyMatch(key -> key.startsWith("segment-contiguous-vector->scalar")), selected.toString());
        assertTrue(selected.keySet().stream().anyMatch(key -> key.startsWith("mixed-general-parallel-vector->parallel")), selected.toString());
    }

    @Test void explicitInvalidRoleTypeAndShapeClassesAreRejectedByProvider() {
        var provider = new CpuCapabilityProvider();
        var fixtures = rejectedProviderFixtures();
        assertEquals(7, fixtures.size());
        assertEquals(fixtures.size(), fixtures.stream().map(CpuGeneratedCoverageEvidenceRegistry.ProviderRejectedFixture::id)
                .distinct().count());
        for (var fixture : fixtures) {
            assertRejected(provider, fixture.operation(), fixture.inputs(), fixture.outputs(), fixture.id());
        }
    }

    /** The finite provider-rejection boundary set used by this matrix and the checkpoint join. */
    static List<CpuGeneratedCoverageEvidenceRegistry.ProviderRejectedFixture> rejectedProviderFixtures() {
        var result = new ArrayList<CpuGeneratedCoverageEvidenceRegistry.ProviderRejectedFixture>();
        // Model contracts: gather/one-hot indices are INT32/INT64; output for one-hot is BOOL.
        result.add(rejected("gather-index-role", new Operation(AxisGatherKind.GATHER,
                new io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs(0)),
                List.of(desc(DataType.FLOAT32, Shape.of(2, 3)), desc(DataType.FLOAT32, Shape.of(2))),
                List.of(desc(DataType.FLOAT32, Shape.of(2, 3))), "INVALID_INDEX_ROLE"));
        result.add(rejected("one-hot-output-role", new Operation(OneHotKind.ONE_HOT,
                new io.github.pho001.synaptik.model.operation.index.OneHotAttrs(3)),
                List.of(desc(DataType.INT32, Shape.of(2))), List.of(desc(DataType.FLOAT32, Shape.of(2, 3))),
                "INVALID_OUTPUT_ROLE"));
        result.add(rejected("scan-bool", new Operation(CumulativeScanKind.CUM_SUM,
                new io.github.pho001.synaptik.model.operation.scan.CumulativeScanAttrs(0, false, false)),
                List.of(desc(DataType.BOOL, Shape.of(2))), List.of(desc(DataType.BOOL, Shape.of(2))), "BOOL_INAPPLICABLE"));
        result.add(rejected("mean-integral", new Operation(AggregateReductionKind.MEAN, NoOperationAttrs.INSTANCE),
                List.of(desc(DataType.INT32, Shape.of(2))), List.of(desc(DataType.INT32, Shape.scalar())), "INTEGRAL_MEAN_INAPPLICABLE"));
        result.add(rejected("scatter-add-bool", new Operation(AxisScatterKind.SCATTER_ADD,
                new io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs(0)),
                List.of(desc(DataType.BOOL, Shape.of(2)), desc(DataType.INT32, Shape.of(2)), desc(DataType.BOOL, Shape.of(2))),
                List.of(desc(DataType.BOOL, Shape.of(2))), "BOOL_INAPPLICABLE"));
        // Tensor.fold2d accepts only floating canonical columns.  Retargeting a CPU test fixture
        // to INT64 creates a structurally shaped operation query, not a Model-valid occurrence;
        // keep that boundary explicit instead of treating a prior CPU rejection as missing support.
        var fold2d = CpuGeneratedDirectEvidenceClosureTest.ordinaryFixtureSeeds().stream()
                .filter(seed -> seed.id().equals("fold2d")).findFirst().orElseThrow().context();
        var integralFold2d = retarget(fold2d, allRoles(fold2d, DataType.INT64));
        var foldNode = integralFold2d.nodes().getFirst();
        result.add(rejected("fold2d-integral-not-model-valid", foldNode.operation(),
                CpuGeneratedDirectEvidenceClosureTest.descriptors(integralFold2d, foldNode.inputs()),
                CpuGeneratedDirectEvidenceClosureTest.descriptors(integralFold2d, foldNode.outputs()),
                "INTEGRAL_FOLD2D_INAPPLICABLE"));
        // Model permits BFLOAT16 dropout, but the current ordinary CPU route intentionally
        // admits only FLOAT64 and FLOAT32; this is provider inapplicability, not a role error.
        var dropout = CpuGeneratedDirectEvidenceClosureTest.ordinaryFixtureSeeds().stream()
                .filter(seed -> seed.id().equals("dropout-f32")).findFirst().orElseThrow().context();
        var bfloat16Dropout = retarget(dropout,
                List.of(DataType.BFLOAT16, DataType.INT64, DataType.BFLOAT16, DataType.BOOL, DataType.INT64));
        var dropoutNode = bfloat16Dropout.nodes().getFirst();
        result.add(rejected("dropout-bfloat16-provider-inapplicable", dropoutNode.operation(),
                CpuGeneratedDirectEvidenceClosureTest.descriptors(bfloat16Dropout, dropoutNode.inputs()),
                CpuGeneratedDirectEvidenceClosureTest.descriptors(bfloat16Dropout, dropoutNode.outputs()),
                "BFLOAT16_DROPOUT_INAPPLICABLE"));
        return List.copyOf(result);
    }

    private static CpuGeneratedCoverageEvidenceRegistry.ProviderRejectedFixture rejected(String id,
            Operation operation, List<TensorDescriptor> inputs, List<TensorDescriptor> outputs, String reason) {
        return new CpuGeneratedCoverageEvidenceRegistry.ProviderRejectedFixture(id, operation.kind().name(),
                operation, inputs, outputs, reason);
    }

    private static void assertRejected(CpuCapabilityProvider provider, Operation operation,
            List<TensorDescriptor> inputs, List<TensorDescriptor> outputs, String id) {
        assertFalse(provider.supports(new OperationCapabilityQuery(operation, inputs, outputs)), id);
    }

    private static List<Base> bases() {
        var result = new ArrayList<Base>();
        // Movement and fold preserve one element type in every input/output role.
        for (var seed : CpuGeneratedDirectEvidenceClosureTest.ordinaryFixtureSeeds()) {
            var kind = seed.context().nodes().getFirst().operation().kind();
            if (kind instanceof CumulativeScanKind || kind instanceof AggregateReductionKind
                    || kind instanceof AxisScatterKind || kind instanceof ScatterNdKind
                    || kind instanceof OrderingKind || kind instanceof TopKKind || kind instanceof OneHotKind
                    || kind instanceof AxisGatherKind || kind instanceof GatherNdKind) continue;
            if (seed.id().equals("initial-state")) {
                // GraphRng INITIAL_STATE has one fixed INT64 [2] state output and no value role.
                result.add(new Base(seed.id(), seed.context()));
                continue;
            }
            // PAD and UNFOLD2D carry a typed scalar fill attribute.  Their all-type expansion
            // must rebuild that Model attribute, rather than retargeting tensor descriptors and
            // accidentally manufacturing an invalid INT32-fill/FLOAT64-data occurrence.
            List<DataType> types = seed.id().equals("pad") ? List.of(DataType.INT32)
                    : seed.id().equals("unfold2d") ? List.of(DataType.FLOAT64)
                    : seed.id().equals("fold-axis") ? NUMERIC
                    : seed.id().equals("fold2d") ? FLOATING
                    : kind.name().equals("DROPOUT") ? List.of(DataType.FLOAT64, DataType.FLOAT32) : ALL;
            for (DataType type : types)
                result.add(new Base(seed.id() + "/" + type, retarget(seed.context(),
                        seed.id().startsWith("dropout-")
                                ? List.of(type, DataType.INT64, type, DataType.BOOL, DataType.INT64)
                                : allRoles(seed.context(), type))));
        }
        addIndexing(result);
        addScatter(result);
        addScans(result);
        addAggregates(result);
        addOrdering(result);
        return List.copyOf(result);
    }

    private static void addIndexing(List<Base> result) {
        for (DataType value : ALL) for (DataType index : INDICES) {
            result.add(new Base("gather/" + value + '/' + index, retarget(CpuGeneratedDirectEvidenceClosureTest
                    .ordinaryFixtureSeeds().stream().filter(seed -> seed.id().equals("gather")).findFirst().orElseThrow().context(), List.of(value, index, value))));
            result.add(new Base("gather-elements/" + value + '/' + index, retarget(CpuGeneratedDirectEvidenceClosureTest
                    .ordinaryFixtureSeeds().stream().filter(seed -> seed.id().equals("gather-elements")).findFirst().orElseThrow().context(), List.of(value, index, value))));
            result.add(new Base("gather-nd/" + value + '/' + index, retarget(CpuGeneratedDirectEvidenceClosureTest
                    .ordinaryFixtureSeeds().stream().filter(seed -> seed.id().equals("gather-nd")).findFirst().orElseThrow().context(), List.of(value, index, value))));
        }
        for (DataType index : INDICES) result.add(new Base("one-hot/" + index, retarget(CpuGeneratedDirectEvidenceClosureTest
                .ordinaryFixtureSeeds().stream().filter(seed -> seed.id().equals("one-hot")).findFirst().orElseThrow().context(), List.of(index, DataType.BOOL))));
    }

    private static void addScatter(List<Base> result) {
        // Build every Model reduction form from source-owned shapes.  The fixture seed list is
        // deliberately not reused here: it contains representative reductions whose BOOL/numeric
        // eligibility would otherwise be projected to the wrong reduction contract.
        for (io.github.pho001.synaptik.model.operation.index.ScatterReduction reduction : io.github.pho001.synaptik.model.operation.index.ScatterReduction.values())
            for (DataType value : reduction.name().equals("NONE") ? ALL : NUMERIC) for (DataType index : INDICES) {
                var element = CpuGeneratedDirectEvidenceClosureTest.scatter(AxisScatterKind.SCATTER_ELEMENTS,
                        new io.github.pho001.synaptik.model.operation.index.ScatterElementsAttrs(0, reduction));
                result.add(new Base("scatter-elements-" + reduction + '/' + value + '/' + index,
                        retarget(element, List.of(value, index, value, value))));
                var nd = CpuScatterLoweringTest.context(new Operation(ScatterNdKind.SCATTER_ND,
                        new io.github.pho001.synaptik.model.operation.index.ScatterNdAttrs(1, reduction)), List.of(0, 1, 2),
                        List.of(desc(value, Shape.of(2, 3, 4)), desc(index, Shape.of(2, 5, 1)), desc(value, Shape.of(2, 5, 4))), desc(value, Shape.of(2, 3, 4)));
                result.add(new Base("scatter-nd-" + reduction + '/' + value + '/' + index, nd));
            }
    }

    /** Exact ordinary-matrix scatter rows, including each carrier/layout/request realization. */
    static List<ScatterCandidate> scatterCandidates() {
        var bases = new ArrayList<Base>();
        addScatter(bases);
        var candidates = new ArrayList<ScatterCandidate>();
        for (Base base : bases) for (Request request : requests())
            candidates.add(new ScatterCandidate(base.id() + '/' + request.id(), configured(base.context(), request)));
        return List.copyOf(candidates);
    }

    /**
     * Returns the finite aggregate and scan universe that owns the 460 direct semantic rows.
     *
     * <p>This deliberately retains the four requested carrier/layout/strategy contexts instead
     * of projecting a result from a representative artifact.  The closure test reconstructs and
     * invokes each returned context independently.</p>
     */
    static List<AggregateOrScanCandidate> aggregateOrScanCandidates() {
        var candidates = new ArrayList<AggregateOrScanCandidate>();
        var bases = new ArrayList<Base>();
        addScans(bases);
        addAggregates(bases);
        for (Base base : bases) for (Request request : requests())
            candidates.add(new AggregateOrScanCandidate(base.id() + '/' + request.id(),
                    configured(base.context(), request)));
        return List.copyOf(candidates);
    }

    /**
     * Returns every ordinary gather and ordering realization whose generated entry has a
     * dedicated direct semantic oracle.
     *
     * <p>ONE_HOT is deliberately excluded: it has a distinct boolean-only role contract and is
     * covered by its own generated-kernel semantic suite.  The remaining rows are the exact
     * 144 indexing and 192 ordering identities represented by the ordinary matrix.</p>
     */
    static List<IndexingOrOrderingCandidate> indexingOrOrderingCandidates() {
        var bases = new ArrayList<Base>();
        addIndexing(bases);
        addOrdering(bases);
        var candidates = new ArrayList<IndexingOrOrderingCandidate>();
        for (Base base : bases) {
            var kind = base.context().nodes().getFirst().operation().kind();
            if (kind == OneHotKind.ONE_HOT) continue;
            for (Request request : requests())
                candidates.add(new IndexingOrOrderingCandidate(base.id() + '/' + request.id(),
                        configured(base.context(), request)));
        }
        return List.copyOf(candidates);
    }

    /**
     * Returns the exact ordinary random and one-hot owner projection.
     *
     * <p>The projection retains all four request forms for each Model-valid dropout value type,
     * both one-hot index widths, and the fixed initializer state contract.  Its dedicated
     * closure invokes every returned generated entry; this method intentionally contains no
     * support inference or representative projection.</p>
     */
    static List<RandomOrOneHotCandidate> randomOrOneHotCandidates() {
        var candidates = new ArrayList<RandomOrOneHotCandidate>();
        for (Base base : bases()) {
            String form = base.context().nodes().getFirst().operation().kind().name();
            if (!Set.of("DROPOUT", "ONE_HOT", "INITIAL_STATE").contains(form)) continue;
            for (Request request : requests()) candidates.add(new RandomOrOneHotCandidate(
                    "ordinary:" + base.id() + '/' + request.id(), form,
                    configured(base.context(), request)));
        }
        return List.copyOf(candidates);
    }

    /**
     * Returns the exact finite movement/fold projection owned by the direct semantic closure.
     *
     * <p>This is deliberately a projection of the same Model-valid bases and four request
     * realizations used by the ordinary matrix.  It is not a second hand-maintained matrix:
     * adding, removing, or retargeting an ordinary row therefore makes the closure's checked
     * owner join fail rather than silently changing its coverage.</p>
     */
    static List<MovementOrFoldCandidate> movementOrFoldCandidates() {
        var candidates = new ArrayList<MovementOrFoldCandidate>();
        for (Base base : bases()) {
            String form = base.context().nodes().getFirst().operation().kind().name();
            if (!Set.of("CONCAT", "STACK", "TILE", "SLICE_UPDATE", "UNFOLD_AXIS",
                    "FOLD_AXIS", "FOLD2D", "UNFOLD2D", "PAD").contains(form)) continue;
            for (Request request : requests()) {
                candidates.add(new MovementOrFoldCandidate("ordinary:" + base.id() + '/'
                        + request.id(), form, configured(base.context(), request)));
            }
        }
        return List.copyOf(candidates);
    }

    private static void addScans(List<Base> result) {
        for (CumulativeScanKind kind : CumulativeScanKind.values()) for (DataType type : NUMERIC)
            for (boolean exclusive : List.of(false, true)) for (boolean reverse : List.of(false, true))
                result.add(new Base(kind + "/" + type + "/" + exclusive + '/' + reverse,
                        CpuScanLoweringTest.context(kind, type, Shape.of(2, 3), 1, exclusive, reverse)));
    }

    private static void addAggregates(List<Base> result) {
        for (AggregateReductionKind kind : List.of(AggregateReductionKind.MIN, AggregateReductionKind.MAX,
                AggregateReductionKind.SUM, AggregateReductionKind.MEAN, AggregateReductionKind.PROD,
                AggregateReductionKind.ALL, AggregateReductionKind.ANY)) {
            List<DataType> types = kind == AggregateReductionKind.MEAN ? FLOATING
                    : kind == AggregateReductionKind.ALL || kind == AggregateReductionKind.ANY ? List.of(DataType.BOOL) : NUMERIC;
            for (DataType type : types) {
                result.add(new Base("aggregate/" + kind + '/' + type + "/full", CpuAggregateLoweringTest.context(kind, type, Shape.of(4, 8), NoOperationAttrs.INSTANCE, Shape.scalar())));
                result.add(new Base("aggregate/" + kind + '/' + type + "/axis", CpuAggregateLoweringTest.context(kind, type, Shape.of(4, 8), new io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs(1, false), Shape.of(4))));
                result.add(new Base("aggregate/" + kind + '/' + type + "/multi", CpuAggregateLoweringTest.context(kind, type, Shape.of(2, 4, 8), new io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs(List.of(0, 2), false), Shape.of(4))));
            }
        }
    }

    private static void addOrdering(List<Base> result) {
        for (DataType type : ALL) for (boolean descending : List.of(false, true)) {
            result.add(new Base("sort/" + type + '/' + descending, CpuOrderingLoweringTest.context(new Operation(OrderingKind.SORT, new SortAttrs(1, descending)), type, Shape.of(2, 4), Shape.of(2, 4), false)));
            result.add(new Base("argsort/" + type + '/' + descending, CpuOrderingLoweringTest.context(new Operation(OrderingKind.ARGSORT, new SortAttrs(1, descending)), type, Shape.of(2, 4), Shape.of(2, 4), false)));
            for (boolean sorted : List.of(false, true)) result.add(new Base("top-k/" + type + '/' + descending + '/' + sorted,
                    CpuOrderingLoweringTest.context(new Operation(TopKKind.TOP_K, new TopKAttrs(1, 2, descending, sorted)), type, Shape.of(2, 4), Shape.of(2, 2), true)));
        }
    }

    private static List<DataType> allRoles(PrepareContext<CpuPartitionAnalysisInputs> context, DataType type) {
        return java.util.Collections.nCopies(context.values().size(), type);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> retarget(PrepareContext<CpuPartitionAnalysisInputs> base, List<DataType> roles) {
        assertEquals(base.values().size(), roles.size());
        var values = new ArrayList<GraphValue>(); var memory = new ArrayList<LogicalMemoryRequirement>();
        for (int i = 0; i < base.values().size(); i++) {
            var old = base.values().get(i); var descriptor = new TensorDescriptor(roles.get(i), old.descriptor().shape(), old.descriptor().layout(), old.descriptor().requiresGrad());
            values.add(new GraphValue(old.id(), descriptor)); var requirement = base.memoryRequirements().get(i);
            memory.add(new LogicalMemoryRequirement(requirement.valueId(), descriptor, requirement.producerPartition(), requirement.consumerPartitions(), requirement.graphOutput()));
        }
        return new PrepareContext<>(base.partition(), base.nodes(), values, memory, base.constants(), base.backendInputs());
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> configured(PrepareContext<CpuPartitionAnalysisInputs> base, Request request) {
        var values = new ArrayList<GraphValue>(); var memory = new ArrayList<LogicalMemoryRequirement>();
        for (int i = 0; i < base.values().size(); i++) {
            var old = base.values().get(i); var d = old.descriptor();
            if (request.general() && d.shape().rank() != 0) { long[] strides = d.layout().orElseThrow().strides(); for (int a = 0; a < strides.length; a++) strides[a] *= 2; d = new TensorDescriptor(d.dataType(), d.shape(), Optional.of(LayoutDescriptor.of(d.shape(), strides, 1, true)), d.requiresGrad()); }
            values.add(new GraphValue(old.id(), d)); var r = base.memoryRequirements().get(i); memory.add(new LogicalMemoryRequirement(r.valueId(), d, r.producerPartition(), r.consumerPartitions(), r.graphOutput()));
        }
        var carriers = new ArrayList<CpuKernelSpecialization.CarrierAccess>();
        for (int i = 0; i < values.size(); i++) carriers.add(request.segment() ? CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT : request.mixed() && i % 2 == 1 ? CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT : CpuGeneratedDirectEvidenceClosureTest.heapCarrier(values.get(i).descriptor().dataType()));
        var materialization = request.materialization() ? new CpuPartitionAnalysisInputs.MaterializationPolicy(true, 0, 1, 20, 1, 3, 1_000_000, 1, 1) : CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED;
        return new PrepareContext<>(base.partition(), base.nodes(), values, memory, base.constants(), new CpuPartitionAnalysisInputs(false, carriers, request.execution(), materialization));
    }

    private static List<Request> requests() { return List.of(new Request("heap-contiguous-scalar", false, false, false, false, CpuGeneratedDirectEvidenceClosureTest.scalar(1)), new Request("segment-contiguous-vector", true, false, false, false, CpuGeneratedDirectEvidenceClosureTest.vector(1)), new Request("mixed-general-parallel-vector", false, true, true, false, CpuGeneratedDirectEvidenceClosureTest.vector(4)), new Request("heap-general-materialization-candidate", false, false, true, true, CpuGeneratedDirectEvidenceClosureTest.scalar(4))); }
    private static TensorDescriptor desc(DataType type, Shape shape) { return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), false); }
    private record Base(String id, PrepareContext<CpuPartitionAnalysisInputs> context) { }
    private record Request(String id, boolean segment, boolean mixed, boolean general, boolean materialization, CpuPartitionAnalysisInputs.PortableExecutionConfig execution) { }
    record ScatterCandidate(String ownerId, PrepareContext<CpuPartitionAnalysisInputs> context) { }
    record AggregateOrScanCandidate(String ownerId,
            PrepareContext<CpuPartitionAnalysisInputs> context) { }
    record IndexingOrOrderingCandidate(String ownerId,
            PrepareContext<CpuPartitionAnalysisInputs> context) { }
    record RandomOrOneHotCandidate(String ownerId, String operationForm,
            PrepareContext<CpuPartitionAnalysisInputs> context) { }
    record MovementOrFoldCandidate(String ownerId, String operationForm,
            PrepareContext<CpuPartitionAnalysisInputs> context) { }
}
