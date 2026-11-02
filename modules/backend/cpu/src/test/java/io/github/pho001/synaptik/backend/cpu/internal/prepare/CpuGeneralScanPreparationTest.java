package io.github.pho001.synaptik.backend.cpu.internal.prepare;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanGeneratedKernelTest;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Ordinary analysis/assignment/finalization fixture with dead canonical ports retained in scope. */
public class CpuGeneralScanPreparationTest {
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=io.github.pho001.synaptik.model.datatype.DataType.class,names={"FLOAT32","FLOAT64"})
    void lengthControlDeclaresEightByteSpanBeforeSlotsAndKeepsActualReadScope(io.github.pho001.synaptik.model.datatype.DataType type) {
        for (var q : CpuGeneralScanGeneratedKernelTest.lengthAutomaticQueries(type)) {
            var analysis = new CpuPartitionPreparer().analyze(context(q));
            var unit = analysis.plan().units().getFirst();
            var facts = unit.runtimeFacts().generalScan().orElseThrow();
            assertTrue(facts.ir().lengthAware()); assertEquals(1,unit.selectedRangeCount());
            var control = analysis.requirements().stream().filter(r -> r instanceof io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement.Buffer b
                    && b.valueId().equals(unit.boundaryValues().get(facts.ir().lengthRead())))
                    .map(io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement.Buffer.class::cast).findFirst().orElseThrow();
            assertEquals(24,control.byteSize()); assertEquals(8,control.byteAlignment());
            assertEquals(io.github.pho001.synaptik.model.datatype.DataType.INT64,unit.portablePlan().specialization().boundaryDataTypes().get(facts.ir().lengthRead()));
            assertEquals(facts.ir().formalOrdinals().size(),facts.ir().actualReads().size());
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=io.github.pho001.synaptik.model.datatype.DataType.class,names={"FLOAT32","FLOAT64"})
    void actualSiblingWorkspaceIsTimeBoundedAndScalesAffinelyWithBatch(io.github.pho001.synaptik.model.datatype.DataType type) {
        var reserved = new HashMap<List<Integer>,Long>();
        for (int time : List.of(5,257)) for (int batch : List.of(3,6,9)) {
            var h = io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLoweringTest.tensor(type,true,batch,8);
            var x = io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLoweringTest.tensor(type,true,batch,8);
            var w = io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLoweringTest.tensor(type,true,8);
            var next = h.add(x.mul(w));
            var initial = io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLoweringTest.tensor(type,true,batch,8);
            var sequence = io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLoweringTest.tensor(type,true,time,batch,8);
            var weight = io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLoweringTest.tensor(type,true,8);
            var lengths = io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLoweringTest.tensor(
                    io.github.pho001.synaptik.model.datatype.DataType.INT64,false,batch);
            var body = new io.github.pho001.synaptik.model.tensor.ScanBody(List.of(h),List.of(x),List.of(w),List.of(next),List.of(next));
            var scan = io.github.pho001.synaptik.model.tensor.Scan.apply(List.of(initial),List.of(sequence),List.of(weight),
                    lengths,1,List.of(OptionalInt.of(0)),body,false);
            var roots = List.of(scan.finalCarry().getFirst(),scan.outputs().getFirst());
            var artifact = io.github.pho001.synaptik.backend.cpu.CpuGeneralScanCapabilityTest.compile(roots,
                    Optional.of(io.github.pho001.synaptik.backend.cpu.CpuGeneralScanCapabilityTest.request(roots,List.of(initial,sequence,weight))),
                    false,new CpuCapabilityProvider());
            long bytes = 0;
            for (var query : io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLoweringTest.queries(artifact)) {
                var analysis = new CpuPartitionPreparer().analyze(context(query));
                bytes += analysis.requirements().stream()
                        .filter(io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement.Workspace.class::isInstance)
                        .map(io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement.Workspace.class::cast)
                        .mapToLong(io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement.Workspace::byteSize).sum();
            }
            reserved.put(List.of(time,batch),bytes);
        }
        for (int batch : List.of(3,6,9))
            assertEquals(reserved.get(List.of(5,batch)),reserved.get(List.of(257,batch)),
                    "declared sibling scratch does not retain time histories");
        long small = reserved.get(List.of(5,3)), medium = reserved.get(List.of(5,6)), large = reserved.get(List.of(5,9));
        assertTrue(small <= medium && medium <= large, "larger row domains do not shrink workspace capacity");
        assertEquals(medium-small,large-medium, "fixed-feature scratch scales at most affinely with batch, not quadratically");
    }
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=io.github.pho001.synaptik.model.datatype.DataType.class,names={"FLOAT32","FLOAT64"})
    void denseReadTwinsDeclareIdenticalResourcesAndAssignedLowerings(io.github.pho001.synaptik.model.datatype.DataType type) {
        for (int family = 0; family < 3; family++) {
            var q = io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLoweringTest.denseReadFixture(type,family,3,false);
            var view = io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLoweringTest.withActualReadViews(q,true);
            var a = new CpuPartitionPreparer().analyze(context(q));
            var b = new CpuPartitionPreparer().analyze(context(view));
            assertEquals(a.requirements(),b.requirements(),"exact pre-assignment byte sizes, alignment and order");
            var u = a.plan().units().getFirst(); var v = b.plan().units().getFirst();
            assertEquals(u.boundaryValues(),v.boundaryValues()); assertEquals(u.outputCount(),v.outputCount());
            assertEquals(u.runtimeFacts().generalScan(),v.runtimeFacts().generalScan());
            assertEquals(u.runtimeFacts().workspaceDeclaration(),v.runtimeFacts().workspaceDeclaration());
            assertEquals(u.portablePlan().specialization(),v.portablePlan().specialization());
            assertEquals(u.portablePlan().kernelIr(),v.portablePlan().kernelIr());
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=io.github.pho001.synaptik.model.datatype.DataType.class,names={"FLOAT32","FLOAT64"})
    void matmulInferenceHasZeroUndemandedHistoryAndOnlyNecessaryPredeclaredBanks(io.github.pho001.synaptik.model.datatype.DataType type) {
        var h = io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLoweringTest.tensor(type,false,2,3);
        var x = io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLoweringTest.tensor(type,false,2,2);
        var w = io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLoweringTest.tensor(type,false,2,3);
        var product = x.matmul(w); var next = h.add(product);
        var body = new io.github.pho001.synaptik.model.tensor.ScanBody(List.of(h),List.of(x),List.of(w),List.of(next),List.of(next));
        var scan = io.github.pho001.synaptik.model.tensor.Scan.apply(
                List.of(io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLoweringTest.tensor(type,false,2,3)),
                List.of(io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLoweringTest.tensor(type,false,3,2,2)),
                List.of(io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLoweringTest.tensor(type,false,2,3)),body,false);
        for (boolean emitted : List.of(false,true)) {
            var roots = emitted ? List.of(scan.finalCarry().getFirst(),scan.outputs().getFirst()) : scan.finalCarry();
            var analysis = new CpuPartitionPreparer().analyze(context(io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLoweringTest.query(roots)));
            var unit = analysis.plan().units().getFirst(); var facts = unit.runtimeFacts().generalScan().orElseThrow();
            assertEquals(emitted ? 0 : 12L*type.byteWidth(),facts.geometry().scratchBytes());
            assertEquals(emitted ? 2 : 1,unit.outputCount());
            var plan = facts.ir().stages().orElseThrow(); assertEquals(1,plan.cuts().size(),"no undemanded product/history cut");
            assertEquals(emitted ? 1 : -1,plan.cuts().getFirst().port());
            assertEquals(emitted ? -1 : 0,plan.cuts().getFirst().carry());
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=io.github.pho001.synaptik.model.datatype.DataType.class,names={"FLOAT32","FLOAT64"})
    void nonOutputBanksExistOnlyWhenALaterTimeReadReallyNeedsThem(io.github.pho001.synaptik.model.datatype.DataType type) {
        for (int t : List.of(0, 1, 3, 257)) for (int n : List.of(0, 2)) {
            var h = io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLoweringTest.tensor(type,false,n);
            var x = io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLoweringTest.tensor(type,false,n);
            var body = new io.github.pho001.synaptik.model.tensor.ScanBody(List.of(h),List.of(x),List.of(),List.of(h.add(x)),List.of(h));
            var scan = io.github.pho001.synaptik.model.tensor.Scan.apply(
                    List.of(io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLoweringTest.tensor(type,false,n)),
                    List.of(io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLoweringTest.tensor(type,false,t,n)),List.of(),body,false);
            var q = io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLoweringTest.query(scan.outputs());
            var analysis = new CpuPartitionPreparer().analyze(context(q));
            var workspace = analysis.plan().units().getFirst().runtimeFacts().workspaceDeclaration();
            assertEquals(t > 1 && n > 0, workspace.isPresent());
            workspace.ifPresent(w -> assertEquals((long)type.byteWidth()*n,w.byteSize()));
        }
    }
    @Test void declaresOnlyDemandedWrappersAndNoFrameBackedScratch() {
        var analysis = new CpuPartitionPreparer().analyze(context(CpuGeneralScanGeneratedKernelTest.query(257, 3, false, List.of(0, 1))));
        var unit = analysis.plan().units().getFirst();
        assertTrue(unit.runtimeFacts().generalScan().isPresent());
        assertTrue(unit.runtimeFacts().recurrentScan().isEmpty());
        assertEquals(4, analysis.requirements().size());
        assertEquals(0, unit.runtimeFacts().generalScan().orElseThrow().geometry().scratchBytes());
        assertEquals(2, unit.outputCount());
        assertEquals(71, unit.portablePlan().specialization().classIdentitySchema());
    }

    public static PrepareContext<CpuPartitionAnalysisInputs> context(OperationCapabilityQuery q) {
        int reads = q.inputs().size();
        var node = new CompiledNode(new NodeId(0), q.operation(),
                java.util.stream.IntStream.range(0, reads).mapToObj(ValueId::new).toList(),
                java.util.stream.IntStream.range(reads, reads + q.outputs().size()).mapToObj(ValueId::new).toList(), q.scanBody());
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID, List.of(node.id()));
        var descriptors = new ArrayList<>(q.inputs()); descriptors.addAll(q.outputs());
        var values = new ArrayList<GraphValue>(); var memory = new ArrayList<LogicalMemoryRequirement>();
        for (int p = 0; p < descriptors.size(); p++) {
            var id = new ValueId(p); var d = descriptors.get(p); boolean output = p >= reads;
            values.add(new GraphValue(id, d));
            memory.add(new LogicalMemoryRequirement(id, d, output ? Optional.of(partition) : Optional.empty(),
                    output ? List.of() : List.of(partition), output && q.scanBody().orElseThrow().demandedOutputIndices().contains(p - reads)));
        }
        return new PrepareContext<>(partition, List.of(node), values, memory, Map.of(),
                new CpuPartitionAnalysisInputs(false, List.of()),
                q.scanActivityPlan().map(plan -> Map.of(node.id(), plan)).orElseGet(Map::of));
    }
}
