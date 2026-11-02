package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.compiler.*;
import io.github.pho001.synaptik.config.compile.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.*;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Source-built ordinary layout closure and exact carry-bank liveness regressions. */
public class CpuGeneralScanLoweringTest {
    /** Unchanged qualified recipes do not acquire an empty composite field or namespace. */
    @Test void retainedRecipeKeepsItsSelectiveIdentityIntersection() {
        var ir = CpuGeneralScanLowering.analyze(denseReadFixture(DataType.FLOAT32, 0, 3, false)).ir();
        assertTrue(ir.composition().isEmpty());
        assertFalse(ir.toString().contains(", composition="));
        assertFalse(ir.bakedSpecializationIdentity().contains("ordinary-composition"));
        assertFalse(ir.encodedKernelIr().familyIdentity().startsWith("general-scan-composition"));
    }

    /** A zero-node body binds real typed roots without a fabricated computation or carry. */
    @Test void ordinaryCompositionKeepsEmptyBodyAndPerValueWidths() {
        var a = new ValueId(0); var b = new ValueId(1); var x = new ValueId(2);
        var f64 = tensor(DataType.FLOAT64, false, 2, 3).descriptor();
        var bool = tensor(DataType.BOOL, false, 2, 3).descriptor();
        var i32 = tensor(DataType.INT32, false, 2, 3).descriptor();
        var graph = new CompiledGraphModel(List.of(new GraphValue(a, f64), new GraphValue(b, bool), new GraphValue(x, i32)),
                List.of(), List.of(a, b, x), List.of(a, b, x), Map.of());
        var body = new CompiledScanBody(graph, List.of(a, b), List.of(x), List.of(a, b), List.of(0, 1, 2, 3, 4));
        var operation = new io.github.pho001.synaptik.model.operation.Operation(
                io.github.pho001.synaptik.model.operation.scan.ScanKind.SCAN,
                new io.github.pho001.synaptik.model.operation.scan.ScanAttrs(2, 2, 1, 0, 1, 2, false));
        var query = new OperationCapabilityQuery(operation,
                List.of(f64, bool, tensor(DataType.INT32, false, 2, 2, 3).descriptor()),
                List.of(f64, bool, tensor(DataType.INT32, false, 2, 2, 3).descriptor(),
                        tensor(DataType.FLOAT64, false, 2, 2, 3).descriptor(), tensor(DataType.BOOL, false, 2, 2, 3).descriptor()),
                Optional.of(body), Optional.empty());
        var facts = CpuGeneralScanLowering.analyze(query);
        var physical = facts.ir().composition().orElseThrow();
        assertTrue(physical.units().isEmpty()); assertTrue(facts.ordinaryUnits().isEmpty());
        assertEquals(List.of(48L, 6L, 24L), physical.values().stream().map(v -> v.bytes().constantBytes()).toList());
        assertTrue(physical.values().stream().allMatch(v -> v.rowAxis() == -1), "all-valid has no fabricated ROW");
        assertEquals(List.of(0, 1), facts.ir().nextOrdinals());
        for (var carry : physical.carries()) {
            assertEquals(physical.values().get(carry.formal()).offset(), carry.oldOffset());
            assertTrue(carry.newOffset().constantBytes() >= physical.values().getLast().offset().constantBytes()
                    + physical.values().getLast().bytes().constantBytes());
        }
        assertTrue(facts.ir().arithmetic().values().isEmpty(), "no dummy arithmetic dtype");
    }

    /** Swapped old carry roots receive independent simultaneous-next snapshot storage. */
    @Test void ordinaryCompositionPreservesBothOldVersionsForCarryPermutation() {
        Tensor a = tensor(DataType.FLOAT64, false, 2, 3), b = tensor(DataType.FLOAT64, false, 2, 3);
        Tensor x = tensor(DataType.BOOL, false, 2, 3);
        var body = new ScanBody(List.of(a, b), List.of(x), List.of(), List.of(b, a), List.of(x));
        var scan = Scan.apply(List.of(tensor(DataType.FLOAT64, false, 2, 3), tensor(DataType.FLOAT64, false, 2, 3)),
                List.of(tensor(DataType.BOOL, false, 4, 2, 3)), List.of(), body, false);
        var roots = new ArrayList<Tensor>(scan.finalCarry()); roots.addAll(scan.outputs()); roots.addAll(scan.residuals());
        var facts = CpuGeneralScanLowering.analyze(query(roots)); var physical = facts.ir().composition().orElseThrow();
        assertEquals(facts.ir().formalOrdinals().get(1).intValue(), physical.carries().get(0).next());
        assertEquals(facts.ir().formalOrdinals().get(0).intValue(), physical.carries().get(1).next());
        assertTrue(physical.carries().get(1).newOffset().constantBytes()
                >= physical.carries().get(0).newOffset().constantBytes()
                    + physical.values().get(physical.carries().get(0).formal()).bytes().constantBytes());
        for (int port : facts.ir().demandedPorts()) assertEquals(physical.units().size(),
                physical.values().get(facts.ir().portOrdinals().get(port)).death());
    }

    /** Ordinary multi-unit lowering keeps mixed predicate histories and the exact local scope. */
    @Test void ordinaryCompositionReusesLocalOwnersAndTypedPublicationCuts() {
        Tensor h = tensor(false, 2, 3), x = tensor(false, 2, 3);
        Tensor next = h.relu().add(x), predicate = next.greaterThan(x);
        var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(next), List.of(predicate));
        var scan = Scan.apply(List.of(tensor(false, 2, 3)), List.of(tensor(false, 4, 2, 3)), List.of(), body, false);
        var outputs = new ArrayList<Tensor>(scan.finalCarry()); outputs.addAll(scan.outputs()); outputs.addAll(scan.residuals());
        var query = query(outputs); var facts = CpuGeneralScanLowering.analyze(query);
        var physical = facts.ir().composition().orElseThrow(); var captured = query.scanBody().orElseThrow();
        assertEquals(physical.units().size(), facts.ordinaryUnits().size());
        assertTrue(physical.values().stream().anyMatch(v -> v.descriptor().dataType() == DataType.BOOL));
        var local = CpuGeneralScanCompositionLowering.bodyContext(query,
                io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.DEFAULT);
        assertTrue(local.constants().isEmpty(), "no outer constants copied by coincident ID");
        var projected = new HashMap<ValueId, TensorDescriptor>();
        local.values().forEach(v -> projected.put(v.id(), v.descriptor()));
        for (var value : captured.graph().values()) {
            var descriptor = projected.get(value.id());
            assertNotNull(descriptor, "body-local values cannot resolve through outer IDs");
            assertEquals(value.descriptor().dataType(), descriptor.dataType());
            assertEquals(value.descriptor().shape(), descriptor.shape());
            assertEquals(value.descriptor().requiresGrad(), descriptor.requiresGrad());
        }
        for (var node : local.nodes()) {
            assertTrue(projected.keySet().containsAll(node.inputs()));
            for (var output : node.outputs())
                assertTrue(projected.get(output).layout().isPresent(), "computed static layouts are completed locally");
        }
        for (var requirement : local.memoryRequirements())
            assertEquals(projected.get(requirement.valueId()), requirement.descriptor());
        var published = new HashSet<>(captured.nextCarry()); published.addAll(captured.emissions());
        for (var id : published)
            assertTrue(local.memoryRequirements().stream().anyMatch(r -> r.valueId().equals(id) && r.graphOutput()),
                    "typed carry and emission cuts survive local projection");
        for (var unit : facts.ordinaryUnits())
            assertDoesNotThrow(() -> CpuPartitionLowering.requirePermittedLowPrecisionUse(unit.portableKernelIr()));
        assertTrue(physical.values().stream().allMatch(v -> v.rowAxis() == -1));
    }

    /** Empty-carry passthrough retains an emission's actual dtype and no fictitious state. */
    @Test void ordinaryCompositionAllowsEmptyCarryWithoutInferringBatchFromAState() {
        Tensor x = tensor(DataType.INT32, false, 2, 3);
        var body = new ScanBody(List.of(), List.of(x), List.of(), List.of(), List.of(x));
        var scan = Scan.apply(List.of(), List.of(tensor(DataType.INT32, false, 3, 2, 3)), List.of(), body, false);
        var facts = CpuGeneralScanLowering.analyze(query(scan.outputs()));
        assertEquals(0, facts.ir().carryCount());
        assertTrue(facts.ir().composition().orElseThrow().units().isEmpty());
        assertEquals(List.of(DataType.INT32, DataType.INT32), facts.ir().boundaryTypes());
    }

    /** Explicit lengths remain independent INT64 control with no carried declaration. */
    @Test void ordinaryCompositionAllowsNamedEmptyCarryWithoutSamplingBatch() {
        var batch = new io.github.pho001.synaptik.model.shape.DynamicDimension("Rows");
        var feature = new io.github.pho001.synaptik.model.shape.StaticDimension(3);
        Tensor x = symbolicTensor(DataType.INT32, Shape.ofDimensions(batch, feature));
        var body = new ScanBody(List.of(), List.of(x), List.of(), List.of(), List.of(x));
        var scan = Scan.apply(List.of(), List.of(symbolicTensor(DataType.INT32, Shape.ofDimensions(
                        new io.github.pho001.synaptik.model.shape.StaticDimension(4), batch, feature))), List.of(),
                symbolicTensor(DataType.INT64, Shape.ofDimensions(batch)), 0, List.of(OptionalInt.of(0)), body, true);
        var facts = CpuGeneralScanLowering.analyze(query(scan.outputs()));
        var physical = facts.ir().composition().orElseThrow();
        assertTrue(physical.units().isEmpty()); assertTrue(facts.ir().symbolic()); assertTrue(facts.ir().lengthAware());
        assertEquals(0, facts.ir().carryCount()); assertEquals(1, facts.ir().lengthRead());
        assertEquals(0, physical.values().getFirst().bytes().constantBytes());
        assertEquals(12, physical.values().getFirst().bytes().bytesPerBatch());
        assertEquals(List.of(DataType.INT32, DataType.INT64, DataType.INT32), facts.ir().boundaryTypes());
        assertEquals(Optional.of(batch), physical.batch());
    }

    /** A proved nonleading rank-four ROW history retains its own typed physical descriptor. */
    @Test void ordinaryCompositionKeepsHighRankPredicateHistoryRowProvenance() {
        Tensor x = tensor(DataType.FLOAT64, false, 2, 3, 4, 5);
        Tensor predicate = x.greaterThan(x), transposed = predicate.permute(1, 2, 0, 3);
        Tensor emission = transposed.permute(2, 0, 1, 3);
        var body = new ScanBody(List.of(), List.of(x), List.of(), List.of(), List.of(emission));
        var scan = Scan.apply(List.of(), List.of(tensor(DataType.FLOAT64, false, 3, 2, 3, 4, 5)), List.of(),
                tensor(DataType.INT64, false, 2), 0, List.of(OptionalInt.of(0)), body, false);
        var roots = new ArrayList<Tensor>(scan.outputs()); roots.addAll(scan.residuals());
        var physical = CpuGeneralScanLowering.analyze(query(roots)).ir().composition().orElseThrow();
        assertTrue(physical.values().stream().anyMatch(v -> v.descriptor().dataType() == DataType.BOOL
                && v.descriptor().shape().rank() == 4 && v.rowAxis() == 2));
        assertTrue(physical.boundaryDescriptors().stream().anyMatch(d -> d.dataType() == DataType.BOOL && d.shape().rank() == 5));
    }

    /** The temporary low-arithmetic lock applies before any empty time-domain shortcut. */
    @Test void ordinaryCompositionIndependentlyRejectsNestedLowNumericsAtZeroTime() {
        Tensor h = tensor(DataType.FLOAT16, false, 2, 3), x = tensor(DataType.FLOAT16, false, 2, 3);
        var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h.tanh()), List.of());
        var scan = Scan.apply(List.of(tensor(DataType.FLOAT16, false, 2, 3)),
                List.of(tensor(DataType.FLOAT16, false, 0, 2, 3)), List.of(), body, false);
        assertThrows(IllegalArgumentException.class, () -> CpuGeneralScanCompositionLowering.analyze(query(scan.finalCarry())));
    }

    /**
     * Proves a producer-independent SUM tail may lose only its dead local cut. A demanded
     * exact SUM history or a later cross-cell reduction retains that cut and its lifetime.
     * This compares semantic ownership, not coincident resolved B/feature sizes.
     */
    @Test void combinedSumTailDropsOnlyUnobservedCutAndRetainsDemandAndFanout() {
        for (DataType type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean reverse : List.of(false,true))
            for (int mode = 0; mode < 3; mode++) {
                var b = new io.github.pho001.synaptik.model.shape.DynamicDimension("rows");
                var two = new io.github.pho001.synaptik.model.shape.StaticDimension(2);
                var row = Shape.ofDimensions(b,two); var stack = Shape.ofDimensions(new io.github.pho001.synaptik.model.shape.StaticDimension(3),b,two);
                Tensor h = symbolicTensor(type,row), x = symbolicTensor(type,row), old = tensor(type,false,2), other = tensor(type,false);
                Tensor sum = x.sumToShape(Shape.of(2));
                var carries = new ArrayList<>(List.of(h,old)); var next = new ArrayList<>(List.of(h.add(x),old.add(sum)));
                var initial = new ArrayList<>(List.of(symbolicTensor(type,row),tensor(type,false,2)));
                if (mode == 2) { carries.add(other); next.add(other.add(sum.sum())); initial.add(tensor(type,false)); }
                var body = new ScanBody(carries,List.of(x),List.of(),next,List.of());
                var scan = Scan.apply(initial,List.of(symbolicTensor(type,stack)),List.of(),symbolicTensor(DataType.INT64,Shape.ofDimensions(b)),1,List.of(OptionalInt.of(0)),body,reverse);
                var roots = new ArrayList<>(scan.finalCarry());
                if (mode == 1) roots.add(scan.residuals().get(body.retainedValues().indexOf(sum)));
                var facts = CpuGeneralScanLowering.analyze(query(roots)); var stages = facts.ir().stages().orElseThrow();
                var reduction = stages.order().stream().filter(s -> s.reduction() && s.sumToShape()).findFirst().orElseThrow();
                assertFalse(reduction.epilogue().isEmpty(),"carry ADD tail is independently fusible");
                var cut = stages.cuts().stream().filter(c -> c.value() == reduction.output()).findFirst();
                assertEquals(mode != 0,cut.isPresent(),"only demand or later fan-out keeps SUM cut");
                if (mode == 1) assertTrue(cut.orElseThrow().port() >= facts.ir().carryCount(),"exact demanded SUM history");
                if (mode == 2) assertTrue(cut.orElseThrow().death() > cut.orElseThrow().birth(),"later consumer lifetime retained");
            }
    }

    /**
     * Creates an unresolved canonical symbolic descriptor for combined-only proof fixtures.
     * @param type non-null exact represented/control type
     * @param shape non-null static or one-named-B shape
     * @return distinct non-gradient Tensor with canonical static or unresolved symbolic layout
     */
    private static Tensor symbolicTensor(DataType type,Shape shape) {
        return TensorFactory.create(new TensorDescriptor(type,shape,shape.isFullyStatic()
                ? Optional.of(LayoutDescriptor.contiguous(shape)) : Optional.empty(),false));
    }
    /** Numeric-only admission: two double banks count double cells, not four-byte units. */
    @Test void combinedDoubleWorkspaceAdmitsIntBoundedCellsWithoutAllocatingTheirStorage() {
        var b = new io.github.pho001.synaptik.model.shape.DynamicDimension("B");
        var one = new io.github.pho001.synaptik.model.shape.StaticDimension(1);
        var row = new TensorDescriptor(DataType.FLOAT64,Shape.ofDimensions(b,one),Optional.empty(),false);
        var steps = new TensorDescriptor(DataType.FLOAT64,Shape.ofDimensions(one,b,one),Optional.empty(),false);
        var lengths = new TensorDescriptor(DataType.INT64,Shape.ofDimensions(b),Optional.empty(),false);
        Tensor h = TensorFactory.create(row), x = TensorFactory.create(row), w = tensor(DataType.FLOAT64,false,1,1);
        var scan = Scan.apply(List.of(TensorFactory.create(row)),List.of(TensorFactory.create(steps)),
                List.of(tensor(DataType.FLOAT64,false,1,1)),TensorFactory.create(lengths),1,List.of(OptionalInt.of(0)),
                new ScanBody(List.of(h),List.of(x),List.of(w),List.of(h.matmul(w)),List.of()),false);
        var facts = CpuGeneralScanLowering.analyze(query(scan.finalCarry()));
        var recipe = facts.ir().stages().orElseThrow().workspaceSize(8);
        assertEquals(recipe,facts.geometry().workspaceSize()); assertEquals(8,recipe.elementByteWidth());
        assertEquals(0,recipe.constantBytes()); assertEquals(16,recipe.bytesPerBatch());
        assertEquals(9_600_000_000L,recipe.byteSize(600_000_000L));
        assertEquals(1_200_000_000L,recipe.byteSize(600_000_000L)/8);
        long[] pack = facts.pack(600_000_000L); assertEquals(600_000_000L,pack[pack.length-1]);
        assertThrows(ArithmeticException.class,() -> recipe.byteSize(1_100_000_000L));
        assertThrows(ArithmeticException.class,() -> facts.pack(1_100_000_000L));
    }

    @Test
    void rawSymbolicLengthDescriptorsQualifyIndependentlyWithoutSampleBatch() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (long time : new long[] {0, 3}) for (boolean reverse : List.of(false, true)) {
                var query = symbolicLengthQuery(type,time,reverse); var facts = CpuGeneralScanLowering.analyze(query);
                assertTrue(facts.ir().symbolic()); assertTrue(facts.ir().lengthAware());
                assertEquals(type,facts.ir().dataType());
                for (int batch : new int[]{0,1,7}) {
                    long[] packed = facts.pack(batch); assertEquals(batch,packed[packed.length-1]);
                    assertEquals(batch*8L,facts.batchBindings().get(facts.ir().lengthRead()).bind(batch).referencedElementSpan()*8);
                }
                var inputs = new ArrayList<>(query.inputs()); var l = inputs.getLast();
                inputs.set(inputs.size()-1,new TensorDescriptor(DataType.INT64,
                        Shape.ofDimensions(new io.github.pho001.synaptik.model.shape.DynamicDimension("Other")),Optional.empty(),false));
                assertRejected(new OperationCapabilityQuery(query.operation(), inputs, query.outputs(), query.scanBody(), query.scanActivityPlan()));
                inputs.set(inputs.size()-1,new TensorDescriptor(type,l.shape(),Optional.empty(),true));
                assertRejected(new OperationCapabilityQuery(query.operation(), inputs, query.outputs(), query.scanBody(), query.scanActivityPlan()));
            }
    }

    /**
     * Builds a raw length query without asking Model or Compiler to qualify the combination.
     * @param type homogeneous FLOAT32 or FLOAT64 body type
     * @param time fixed non-negative T, including zero
     * @param reverse whether traversal decreases original time
     * @return non-null unresolved symbolic query with all canonical ports and no demanded history;
     *     construction performs no inference, lowering or allocation of Tensor identities/storage
     * @throws NullPointerException if type is null
     * @throws IllegalArgumentException if time is negative
     */
    public static OperationCapabilityQuery symbolicLengthQuery(DataType type, long time, boolean reverse) {
        var batch = new io.github.pho001.synaptik.model.shape.DynamicDimension("Customers");
        var width = new io.github.pho001.synaptik.model.shape.StaticDimension(2);
        var row = new TensorDescriptor(type, Shape.ofDimensions(batch, width), Optional.empty(), false);
        var sequence = new TensorDescriptor(type, Shape.ofDimensions(
                new io.github.pho001.synaptik.model.shape.StaticDimension(time), batch, width), Optional.empty(), false);
        var lengths = new TensorDescriptor(DataType.INT64, Shape.ofDimensions(batch), Optional.empty(), false);
        var local = new CompiledGraphModel(List.of(new GraphValue(new ValueId(0), row), new GraphValue(new ValueId(1), row)),
                List.of(), List.of(new ValueId(0), new ValueId(1)), List.of(new ValueId(0)), Map.of());
        var body = new CompiledScanBody(local, List.of(new ValueId(0)), List.of(), List.of(new ValueId(0)), List.of(0));
        var operation = new io.github.pho001.synaptik.model.operation.Operation(
                io.github.pho001.synaptik.model.operation.scan.ScanKind.SCAN,
                new io.github.pho001.synaptik.model.operation.scan.ScanAttrs(time, 1, 1, 0, 0, 1, reverse, true, 1, List.of(OptionalInt.of(0))));
        return new OperationCapabilityQuery(operation, List.of(row, sequence, lengths), List.of(row, sequence), Optional.of(body), java.util.Optional.empty());
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void lengthCompleteInventoryAndExplicitRolesRejectEvenWithoutDemandOrWork(DataType type) {
        for (int time : List.of(0,5)) {
            var q = lengthFixture(type,1,time,2,2,false,true); var b = q.scanBody().orElseThrow(); var g = b.graph();
            var nodes = new ArrayList<>(g.nodes()); var first = nodes.getFirst();
            nodes.set(0,new CompiledNode(first.id(),new io.github.pho001.synaptik.model.operation.Operation(
                    io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind.GREATER_THAN,first.operation().attrs()),first.inputs(),first.outputs()));
            var bad = new CompiledGraphModel(g.values(),nodes,g.inputs(),g.outputs(),g.nodePhases());
            assertRejected(new OperationCapabilityQuery(q.operation(), q.inputs(), q.outputs(), Optional.of(
                    new CompiledScanBody(bad,b.nextCarry(),b.emissions(),b.residuals(),List.of())), java.util.Optional.empty()));
            var a = (io.github.pho001.synaptik.model.operation.scan.ScanAttrs)q.operation().attrs();
            var wrongRole = new io.github.pho001.synaptik.model.operation.scan.ScanAttrs(time, a.carryCount(), a.sequenceCount(), a.invariantCount(), a.emissionCount(), a.residualCount(), false, true, 1, List.of(OptionalInt.empty()));
            assertRejected(new OperationCapabilityQuery(new io.github.pho001.synaptik.model.operation.Operation(
                    io.github.pho001.synaptik.model.operation.scan.ScanKind.SCAN,wrongRole), q.inputs(), q.outputs(), q.scanBody(), q.scanActivityPlan()));
            for (DataType low : List.of(DataType.BFLOAT16,DataType.FLOAT16)) {
                var local = g.values().stream().map(v -> new GraphValue(v.id(),new TensorDescriptor(low,v.descriptor().shape(),v.descriptor().layout(),v.descriptor().requiresGrad()))).toList();
                var inputs = q.inputs().stream().map(d -> new TensorDescriptor(d.dataType() == DataType.INT64 ? DataType.INT64 : low,d.shape(),d.layout(),d.requiresGrad())).toList();
                var outputs = q.outputs().stream().map(d -> new TensorDescriptor(low,d.shape(),d.layout(),d.requiresGrad())).toList();
                assertEquals(DataType.INT64,inputs.getLast().dataType());
                assertRejected(new OperationCapabilityQuery(q.operation(), inputs, outputs, Optional.of(new CompiledScanBody(
                        new CompiledGraphModel(local,g.nodes(),g.inputs(),g.outputs(),g.nodePhases()),b.nextCarry(),b.emissions(),b.residuals(),List.of())), java.util.Optional.empty()));
            }
        }
    }
    /**
     * Captures the bounded length fixture without hiding residual demand or ROW provenance.
     * @param type represented body type
     * @param family zero addition, one broadcast multiply, two rank-two product
     * @param time fixed T
     * @param batch fixed B
     * @param features fixed trailing width
     * @param reverse traversal direction
     * @param histories whether every canonical retained site is observed
     * @return complete query with separate INT64 lengths
     */
    public static OperationCapabilityQuery lengthFixture(DataType type,int family,int time,int batch,int features,
            boolean reverse,boolean histories) {
        Tensor h = tensor(type,false,batch,features), x = tensor(type,false,batch,features);
        Tensor w = tensor(type,false,family == 2 ? new long[]{features,features} : new long[]{features});
        Tensor next = h.add(family == 0 ? x : family == 1 ? x.mul(w) : x.matmul(w));
        var body = new ScanBody(List.of(h),List.of(x),List.of(w),List.of(next),List.of(next,h));
        var scan = Scan.apply(List.of(tensor(type,false,batch,features)),List.of(tensor(type,false,time,batch,features)),
                List.of(tensor(type,false,w.descriptor().shape().toLongArray())),tensor(DataType.INT64,false,batch),
                1,List.of(OptionalInt.of(0)),body,reverse);
        var roots = new ArrayList<>(scan.finalCarry()); roots.addAll(scan.outputs()); if (histories) roots.addAll(scan.residuals());
        return query(roots);
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void lengthControlHasNoFormalAndGeometryFreeCanonicalIdentity(DataType type) {
        for (int family = 0; family < 3; family++) {
            var a = CpuGeneralScanLowering.analyze(lengthFixture(type,family,5,3,2,false,true));
            var b = CpuGeneralScanLowering.analyze(lengthFixture(type,family,13,4,2,false,true));
            assertEquals(3,a.ir().formalOrdinals().size()); assertEquals(List.of(0,1,2),a.ir().actualReads());
            assertEquals(4,a.ir().scalarReads().size()); assertEquals(3,a.ir().lengthRead());
            assertEquals(DataType.INT64,a.ir().boundaryTypes().get(3)); assertEquals(3,a.spans().get(3));
            assertEquals(a.ir().structuralKey(),b.ir().structuralKey());
            assertNotEquals(a.ir().bakedSpecializationIdentity(),b.ir().bakedSpecializationIdentity());
            assertTrue(a.ir().bakedSpecializationIdentity().startsWith(family == 2 ? "|scan-length-aware-matmul-v2=" : "|scan-length-aware-v1="));
            assertEquals(List.of(0,0,-1),a.ir().stages().orElseThrow().rowAxes().subList(0,3));
        }
    }
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void denseActualViewsKeepExactProofAcrossEveryExistingFamilyAndRole(DataType type) {
        for (int family = 0; family < 3; family++) {
            var canonical = denseReadFixture(type,family,3,false);
            var views = withActualReadViews(canonical,true);
            assertNotEquals(canonical.inputs(),views.inputs(),"truthful view metadata is retained");
            for (var d : views.inputs()) assertTrue(d.layout().orElseThrow().isView());
            assertEquals(CpuGeneralScanLowering.analyze(canonical),CpuGeneralScanLowering.analyze(views),
                    "IR, accesses, exact ranges/spans, geometry and resources, family="+family);
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void denseReadPermissionKeepsDeadFormalWriteGuardsAndComposesExactReadLayouts(DataType type) {
        for (int time : List.of(0,3)) {
            var q = withActualReadViews(denseReadFixture(type,1,time,false),true);
            assertDoesNotThrow(() -> CpuGeneralScanLowering.analyze(q));
            var b = q.scanBody().orElseThrow(); var g = b.graph();
            // No demand does not excuse validating any declared write or formal.
            var dead = new CompiledScanBody(g,b.nextCarry(),b.emissions(),b.residuals(),List.of());
            for (int write : List.of(0,q.outputs().size()-1)) {
                var outputs = new ArrayList<>(q.outputs()); outputs.set(write,viewFlag(outputs.get(write),true));
                assertRejected(new OperationCapabilityQuery(q.operation(), q.inputs(), outputs, q.scanBody(), q.scanActivityPlan()));
                assertRejected(new OperationCapabilityQuery(q.operation(), q.inputs(), outputs, Optional.of(dead), java.util.Optional.empty()));
            }
            var values = g.values().stream().map(v -> g.inputs().contains(v.id())
                    ? new GraphValue(v.id(),viewFlag(v.descriptor(),true)) : v).toList();
            var badGraph = new CompiledGraphModel(values,g.nodes(),g.inputs(),g.outputs(),g.nodePhases());
            assertRejected(new OperationCapabilityQuery(q.operation(), q.inputs(), q.outputs(), Optional.of(
                    new CompiledScanBody(badGraph,b.nextCarry(),b.emissions(),b.residuals(),List.of())), java.util.Optional.empty()));
            for (int role = 0; role < q.inputs().size(); role++) {
                var d = q.inputs().get(role); var strides = d.layout().orElseThrow().strides();
                var wrong = strides.clone(); wrong[wrong.length-1] = 2;
                for (var layout : List.of(Optional.<LayoutDescriptor>empty(),
                        Optional.of(LayoutDescriptor.of(d.shape(),strides,1,true)),
                        Optional.of(LayoutDescriptor.of(d.shape(),wrong,0,true)))) {
                    var inputs = new ArrayList<>(q.inputs()); inputs.set(role,new TensorDescriptor(type,d.shape(),layout,d.requiresGrad()));
                    var changed = new OperationCapabilityQuery(q.operation(), inputs, q.outputs(), Optional.of(dead), java.util.Optional.empty());
                    if (layout.isEmpty()) assertRejected(changed);
                    else {
                        var physical = CpuGeneralScanLowering.analyze(changed).ir().composition().orElseThrow();
                        assertEquals(layout, physical.boundaryDescriptors().get(role).layout());
                    }
                }
            }
        }
        // A noncanonical singleton stride remains an exact physical read, not a rewritten layout.
        Tensor h = tensor(type,false,1,3), x = tensor(type,false,1,3);
        var body = new ScanBody(List.of(h),List.of(x),List.of(),List.of(h.add(x)),List.of());
        var scan = Scan.apply(List.of(tensor(type,false,1,3)),List.of(tensor(type,false,3,1,3)),List.of(),body,false);
        var q = query(scan.finalCarry()); var inputs = new ArrayList<>(q.inputs()); var d = inputs.getFirst();
        inputs.set(0,new TensorDescriptor(type,d.shape(),Optional.of(LayoutDescriptor.of(d.shape(),new long[] {97,1},0,true)),false));
        var physical = CpuGeneralScanLowering.analyze(new OperationCapabilityQuery(q.operation(), inputs,
                q.outputs(), q.scanBody(), q.scanActivityPlan())).ir().composition().orElseThrow();
        assertEquals(List.of(97L, 1L), physical.boundaryMappings().getFirst().strideConstants());
    }

    /**
     * Checks both public provider admission and the exact lowering failure boundary.
     * @param q complete deliberately invalid occurrence; not executed
     */
    private static void assertRejected(OperationCapabilityQuery q) {
        assertFalse(new io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider().supports(q));
        assertThrows(IllegalArgumentException.class,() -> CpuGeneralScanLowering.analyze(q));
    }

    /**
     * Creates a compact canonical three-role fixture in an existing emitted family.
     * @param type homogeneous represented floating type
     * @param family zero for same-shape pointwise, one for broadcast/SUM, two for MATMUL
     * @param time non-negative static time count
     * @param reverse whether traversal decreases original time
     * @return captured actual query with demanded final carry and emission, never null
     */
    public static OperationCapabilityQuery denseReadFixture(DataType type, int family, int time, boolean reverse) {
        Tensor h = tensor(type,true,2,3), x = tensor(type,true,2,family == 2 ? 2 : 3);
        Tensor w = family == 0 ? tensor(type,true,2,3) : family == 1 ? tensor(type,true,3) : tensor(type,true,2,3);
        Tensor next = h.add(family == 2 ? x.matmul(w) : x.mul(w)).tanh();
        var body = new ScanBody(List.of(h),List.of(x),List.of(w),List.of(next),
                List.of(family == 1 ? next.sum(0,false) : next));
        var scan = Scan.apply(List.of(tensor(type,true,2,3)),List.of(tensor(type,true,time,2,family == 2 ? 2 : 3)),
                List.of(tensor(type,true,w.descriptor().shape().toLongArray())),body,reverse);
        return query(List.of(scan.finalCarry().getFirst(),scan.outputs().getFirst()));
    }

    /**
     * Changes only actual read view flags; source body, strides, offsets and outputs are untouched.
     * All-zero broadcast layouts must remain views, so clearing flags skips that retained case.
     * @param q resolved query whose actual layouts are dense or all-zero scalar reads
     * @param view desired flag for dense reads
     * @return descriptor twin with unchanged addresses and represented types
     */
    public static OperationCapabilityQuery withActualReadViews(OperationCapabilityQuery q, boolean view) {
        return new OperationCapabilityQuery(q.operation(), q.inputs().stream().map(d ->
                Arrays.equals(d.layout().orElseThrow().strides(),LayoutDescriptor.contiguous(d.shape()).strides())
                        ? viewFlag(d,view) : d).toList(), q.outputs(), q.scanBody(), q.scanActivityPlan());
    }

    /**
     * Retains every descriptor fact except the layout's view marker.
     * @param d resolved dense descriptor, never mutated
     * @param view replacement marker
     * @return distinct descriptor retaining exact geometry and gradient eligibility
     */
    private static TensorDescriptor viewFlag(TensorDescriptor d, boolean view) {
        var l = d.layout().orElseThrow();
        return new TensorDescriptor(d.dataType(),d.shape(),Optional.of(LayoutDescriptor.of(d.shape(),l.strides(),l.storageOffset(),view)),d.requiresGrad());
    }

    @Test void float64RejectsIndividualStackedAggregateAndAffineEightByteOverflowBeforeGeneration() {
        for (var q : List.of(
                io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanGeneratedKernelTest.query(0,Long.MAX_VALUE/6,false,List.of()),
                io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanGeneratedKernelTest.query(3,Long.MAX_VALUE/20,false,List.of()),
                io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanGeneratedKernelTest.query(3,Long.MAX_VALUE/30,false,List.of(0)))) {
            assertDoesNotThrow(() -> CpuGeneralScanLowering.analyze(q),"four-byte individual and aggregate geometry fits");
            var typed = withType(q,DataType.FLOAT64);
            assertThrows(ArithmeticException.class,() -> CpuGeneralScanLowering.analyze(typed));
            assertFalse(new io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider().supports(typed));
        }
        Tensor h = tensor(false,0,Long.MAX_VALUE/6), x = tensor(false,0,Long.MAX_VALUE/6);
        var body = new ScanBody(List.of(h),List.of(x),List.of(),List.of(h),List.of(x.permute(1,0).contiguous()));
        var scan = Scan.apply(List.of(tensor(false,0,Long.MAX_VALUE/6)),List.of(tensor(false,3,0,Long.MAX_VALUE/6)),List.of(),body,false);
        var q = query(scan.outputs()); assertDoesNotThrow(() -> CpuGeneralScanLowering.analyze(q));
        assertThrows(ArithmeticException.class,() -> CpuGeneralScanLowering.analyze(withType(q,DataType.FLOAT64)),"even empty affine views must have checked byte coefficients");
    }

    /**
     * Retypes a no-scalar-attribute query without rewriting any canonical or local view layout.
     * @param q complete descriptor-only fixture with dtype-independent attributes
     * @param type exact replacement dtype
     * @return independently scoped body-bearing query retaining all affine witness layouts
     */
    public static OperationCapabilityQuery withType(OperationCapabilityQuery q, DataType type) {
        java.util.function.Function<TensorDescriptor,TensorDescriptor> typed = d -> new TensorDescriptor(type,d.shape(),d.layout(),d.requiresGrad());
        var b = q.scanBody().orElseThrow(); var g = b.graph();
        var graph = new CompiledGraphModel(g.values().stream().map(v -> new GraphValue(v.id(),typed.apply(v.descriptor()))).toList(),g.nodes(),g.inputs(),g.outputs(),g.nodePhases());
        return new OperationCapabilityQuery(q.operation(), q.inputs().stream().map(typed).toList(), q.outputs().stream().map(typed).toList(), Optional.of(new CompiledScanBody(graph,b.nextCarry(),b.emissions(),b.residuals(),b.demandedOutputIndices())), java.util.Optional.empty());
    }
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void matmulRightOperandDependencyIsLiveAndCutsReuseOnlyDisjointInclusiveLifetimes(DataType type) {
        Tensor h = tensor(type,false,2,3), x = tensor(type,false,2,3), w = tensor(type,false,3,3);
        Tensor right = w.add(w), p0 = h.matmul(right), p1 = p0.matmul(w), p2 = p1.matmul(w), p3 = p2.matmul(w);
        var body = new ScanBody(List.of(h),List.of(x),List.of(w),List.of(p3.add(x)),List.of());
        var scan = Scan.apply(List.of(tensor(type,false,2,3)),List.of(tensor(type,false,3,2,3)),List.of(tensor(type,false,3,3)),body,false);
        var plan = CpuGeneralScanLowering.analyze(query(scan.finalCarry())).ir().stages().orElseThrow();
        var matrices = plan.order().stream().filter(s -> s.right() >= 0).toList(); assertEquals(4,matrices.size());
        int rhs = matrices.getFirst().right();
        var cut = plan.cuts().stream().filter(c -> c.value() == rhs).findFirst().orElseThrow();
        assertEquals(plan.order().indexOf(matrices.getFirst()),cut.death());
        assertTrue(plan.order().stream().anyMatch(s -> s.values().contains(rhs)),"right arithmetic not pruned");
        var scratch = plan.cuts().stream().filter(c -> c.port() < 0 && c.carry() < 0).toList();
        assertTrue(scratch.stream().anyMatch(a -> scratch.stream().anyMatch(b -> a != b && a.offset() == b.offset())),"disjoint MATMUL temporaries reuse");
        for (var a : scratch) for (var b : scratch) if (a != b && a.birth() <= b.death() && b.birth() <= a.death())
            assertFalse(a.offset() < b.offset()+b.bytes() && b.offset() < a.offset()+a.bytes(),"inclusive lifetimes do not overlap storage");
    }

    @Test void identityExpandPreservesOrdinaryViewAndZeroStrides() {
        Tensor h = tensor(false, 3), x = tensor(false, 3), scalar = tensor(false);
        var dense = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h.expand(Shape.of(3))), List.of());
        var a = Scan.apply(List.of(tensor(false, 3)), List.of(tensor(false, 4, 3)), List.of(), dense, false);
        assertDoesNotThrow(() -> CpuGeneralScanLowering.analyze(query(List.of(a.finalCarry().getFirst()))));
        var zero = scalar.expand(Shape.of(3)).expand(Shape.of(3));
        var body = new ScanBody(List.of(h), List.of(x), List.of(scalar), List.of(h.add(zero)), List.of());
        var b = Scan.apply(List.of(tensor(false, 3)), List.of(tensor(false, 4, 3)), List.of(tensor(false)), body, false);
        assertDoesNotThrow(() -> CpuGeneralScanLowering.analyze(query(List.of(b.finalCarry().getFirst()))));
    }

    @Test void contiguousConsumesProvedDenseArithmeticDespiteAbsentCapturedLayout() {
        Tensor h = tensor(false, 3), x = tensor(false, 3);
        var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h.add(x).contiguous()), List.of());
        var scan = Scan.apply(List.of(tensor(false, 3)), List.of(tensor(false, 4, 3)), List.of(), body, false);
        assertDoesNotThrow(() -> CpuGeneralScanLowering.analyze(query(List.of(scan.finalCarry().getFirst()))));
    }

    @Test void unusedCarryHasNoBankBytesOrStores() {
        Tensor h = tensor(false, 3), x = tensor(false, 3);
        var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(x.add(x)), List.of(x));
        var scan = Scan.apply(List.of(tensor(false, 3)), List.of(tensor(false, 4, 3)), List.of(), body, false);
        var facts = CpuGeneralScanLowering.analyze(query(scan.outputs()));
        assertEquals(0, facts.geometry().scratchBytes());
        assertEquals(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr.Storage.UNUSED,
                facts.ir().carries().getFirst().storage());
    }

    @Test void tiedNextChoosesAnyDemandedFinalWrapperBeforeAllocatingScratch() {
        Tensor a = tensor(false, 3), b = tensor(false, 3), x = tensor(false, 3);
        Tensor v = a.add(b).add(x);
        var body = new ScanBody(List.of(a, b), List.of(x), List.of(), List.of(v, v), List.of());
        var scan = Scan.apply(List.of(tensor(false, 3), tensor(false, 3)), List.of(tensor(false, 4, 3)), List.of(), body, false);
        var facts = CpuGeneralScanLowering.analyze(query(List.of(scan.finalCarry().get(1))));
        assertEquals(0, facts.geometry().scratchBytes());
        assertEquals(facts.ir().carries().get(0), facts.ir().carries().get(1));
        assertEquals(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr.Storage.OUTPUT,
                facts.ir().carries().getFirst().storage());
    }

    public static Tensor tensor(boolean grad, long... dimensions) {
        return tensor(DataType.FLOAT32, grad, dimensions);
    }

    /**
     * Creates a storage-free canonical typed test formal or actual source leaf.
     * @param type represented dtype, never inferred from a carrier
     * @param grad gradient eligibility
     * @param dimensions static logical extents, borrowed without mutation
     * @return new identity-distinct Tensor with no storage
     */
    public static Tensor tensor(DataType type, boolean grad, long... dimensions) {
        Shape s = Shape.of(dimensions);
        return TensorFactory.create(new TensorDescriptor(type, s, Optional.of(LayoutDescriptor.contiguous(s)), grad));
    }

    public static OperationCapabilityQuery query(List<Tensor> outputs) {
        var backend = io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider.CPU_BACKEND_ID;
        var capture = new io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider() {
            public io.github.pho001.synaptik.backend.contract.BackendId backendId() { return backend; }
            public boolean supports(OperationCapabilityQuery q) { return true; }
        };
        var availability = new io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot(backend,
                Map.of(new io.github.pho001.synaptik.backend.contract.BackendDeviceId(backend, "0"),
                    io.github.pho001.synaptik.backend.contract.DeviceClass.CPU));
        var artifacts = GraphCompilationPort.compile(CompileMode.FORWARD_ONLY, outputs, Optional.empty(),
                GraphOptimizationConfig.disabled(), BackendIntent.unconstrained(), PartitionScoringConfig.neutral(),
                List.of(capture), List.of(availability));
        return queries(artifacts).getFirst();
    }

    /**
     * Queries the actual compiled occurrences without dropping selected-backward activity.
     * @param artifacts complete immutable compilation result
     * @return ordered body-bearing queries with their exact containing-node associations
     */
    public static List<OperationCapabilityQuery> queries(
            io.github.pho001.synaptik.compiler.CompileArtifacts artifacts) {
        var graph = artifacts.graph();
        var values = new HashMap<ValueId, TensorDescriptor>();
        graph.values().forEach(v -> values.put(v.id(), v.descriptor()));
        return graph.nodes().stream().filter(n -> n.scanBody().isPresent())
                .map(n -> new OperationCapabilityQuery(n.operation(),
                        n.inputs().stream().map(values::get).toList(),
                        n.outputs().stream().map(values::get).toList(), n.scanBody(),
                        Optional.ofNullable(artifacts.derivatives().scanActivityByNode().get(n.id()))))
                .toList();
    }
}
