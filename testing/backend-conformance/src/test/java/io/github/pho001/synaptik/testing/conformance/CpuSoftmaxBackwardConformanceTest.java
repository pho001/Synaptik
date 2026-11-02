package io.github.pho001.synaptik.testing.conformance;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.*;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuSoftmaxBackwardIr;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuContiguousWorkspace;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.*;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.reduction.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.*;
import io.github.pho001.synaptik.prepare.analysis.*;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.memory.*;
import io.github.pho001.synaptik.runtime.run.*;
import java.lang.foreign.*;
import java.math.BigDecimal;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Prepared derivative primitives have their own domain, full write span and run-private state. */
class CpuSoftmaxBackwardConformanceTest {
    @Test void bothDtypesAndKindsReuseOnePreparedEntryWithIndependentRunState() {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean log : List.of(false,true)) {
            var executable = fixture(type,log);
            assertEquals(3,executable.memoryPlan().buffers().size());
            assertEquals(6L*type.byteWidth(),executable.memoryPlan().buffers().getLast().byteSize());
            try (var cpu = CpuBackendIntegration.open()) {
                var y = storage(type,new double[]{-.5,-1,0,-2,-.25,-.75});
                var g = storage(type,new double[]{1,-2,3,4,-5,6}); var output = storage(type,new double[6]);
                var g2 = storage(type,new double[]{-7,8,-9,10,11,-12}); var output2 = storage(type,new double[6]);
                try (var first = state(executable,cpu,type,y,g,output); var second = state(executable,cpu,type,y,g2,output2)) {
                    var one = executable.bind(first); var two = executable.bind(second);
                    two.execute(); one.execute(); two.execute();
                    check(type,log,y,g,output); check(type,log,y,g2,output2);
                }
            }
        }
    }

    /**
     * Uses independently classified row sums and typed per-element derivative sites.
     * NaN payloads are unspecified; every other result retains its represented bits and sign.
     */
    @Test void nonfiniteBackwardOperandsUsePrimitiveRulesRatherThanForwardFiniteValidator() {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean log : List.of(false,true)) {
            var executable = fixture(type,log);
            try (var cpu = CpuBackendIntegration.open()) {
                var y = storage(type,new double[]{Double.NaN,0,1,Double.POSITIVE_INFINITY,Double.NEGATIVE_INFINITY,-0d});
                var g = storage(type,new double[]{1,2,3,Double.POSITIVE_INFINITY,Double.NEGATIVE_INFINITY,0});
                var out = storage(type,new double[6]);
                try (var state = state(executable,cpu,type,y,g,out)) {
                    assertDoesNotThrow(() -> executable.bind(state).execute());
                    // Softmax's second row contributes [+Inf,+Inf,-0], hence SUM=+Inf.
                    // Log-softmax instead sums g: [1,2,3] -> 6 and [+Inf,-Inf,+0] -> NaN.
                    double[] sums = log ? new double[]{6,Double.NaN}
                            : new double[]{Double.NaN,Double.POSITIVE_INFINITY};
                    for (int i = 0; i < 6; i++) {
                        double expected;
                        if (type == DataType.FLOAT32) {
                            float saved = (float)read(type,y,i), seed = (float)read(type,g,i), sum = (float)sums[i/3];
                            if (log) {
                                float exponential = (float)StrictMath.exp(saved);
                                float product = exponential*sum;
                                expected = seed-product;
                            } else {
                                float difference = seed-sum;
                                expected = saved*difference;
                            }
                        } else {
                            double saved = read(type,y,i), seed = read(type,g,i), sum = sums[i/3];
                            if (log) {
                                double exponential = StrictMath.exp(saved);
                                double product = exponential*sum;
                                expected = seed-product;
                            } else {
                                double difference = seed-sum;
                                expected = saved*difference;
                            }
                        }
                        double actual = read(type,out,i);
                        String position = type+" log="+log+" element="+i;
                        if (Double.isNaN(expected)) assertTrue(Double.isNaN(actual),position);
                        else if (type == DataType.FLOAT32)
                            assertEquals(Float.floatToRawIntBits((float)expected),Float.floatToRawIntBits((float)actual),position);
                        else assertEquals(Double.doubleToRawLongBits(expected),Double.doubleToRawLongBits(actual),position);
                    }
                }
            }
        }
    }

    @Test void overlapReadOnlyUndersizedAndWrongTypeBindingsFailBeforeAnyWrite() {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean log : List.of(false,true)) {
            var executable = fixture(type,log);
            try (var cpu = CpuBackendIntegration.open()) {
                var y = storage(type,new double[]{1,2,3,4,5,6}); var g = storage(type,new double[]{-1,2,-3,4,-5,6});
                for (var out : List.of(y,g,y.asReadOnly(),y.asSlice(0,y.byteSize()-type.byteWidth()))) {
                    try (var state = state(executable,cpu,type,y,g,out)) {
                        assertThrows(IllegalArgumentException.class,() -> executable.bind(state));
                        assertEquals(1,read(type,y,0)); assertEquals(6,read(type,y,5));
                    }
                }
                var wrong = type == DataType.FLOAT32 ? DataType.FLOAT64 : DataType.FLOAT32;
                var wrongOutput = storage(wrong,new double[6]);
                var buffers = List.of(binding(cpu,type,y),binding(cpu,type,g),binding(cpu,wrong,wrongOutput));
                try (var state = new RunState(executable.memoryPlan(),buffers,workspaces(executable))) {
                    assertThrows(IllegalArgumentException.class,() -> executable.bind(state));
                }
            }
        }
    }

    private static PreparedExecutable fixture(DataType type,boolean log) {
        var shape = Shape.of(2,3); var reduced = Shape.of(2,1);
        var nodes = List.of(node(0,log ? UnaryElementwiseKind.EXP : BinaryArithmeticKind.MUL,log ? List.of(0) : List.of(1,0),2),
                new CompiledNode(new NodeId(1),new Operation(AggregateReductionKind.SUM,new AxisReductionAttrs(1,true)),List.of(new ValueId(log ? 1 : 2)),List.of(new ValueId(3))),
                node(2,log ? BinaryArithmeticKind.MUL : BinaryArithmeticKind.SUB,log ? List.of(2,3) : List.of(1,3),4),
                node(3,log ? BinaryArithmeticKind.SUB : BinaryArithmeticKind.MUL,log ? List.of(1,4) : List.of(0,4),5));
        var p = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,nodes.stream().map(CompiledNode::id).toList());
        var values = new ArrayList<GraphValue>();
        for (int i = 0; i < 6; i++) { var s = i == 3 ? reduced : shape;
            values.add(new GraphValue(new ValueId(i),new TensorDescriptor(type,s,Optional.of(LayoutDescriptor.contiguous(s)),false))); }
        var memory = values.stream().map(v -> new LogicalMemoryRequirement(v.id(),v.descriptor(),v.id().value() >= 2 ? Optional.of(p) : Optional.empty(),
                v.id().value() != 5 ? List.of(p) : List.of(),v.id().value() == 5)).toList();
        var analysis = new CpuPartitionPreparer().analyze(new PrepareContext<>(p,nodes,values,memory,Map.of(),CpuPartitionAnalysisInputs.DEFAULT));
        assertInstanceOf(CpuSoftmaxBackwardIr.class,analysis.plan().units().getFirst().portablePlan().portableKernelIr());
        var buffers = new ArrayList<PreparedMemoryPlan.BufferEntry>(); var workspaces = new ArrayList<PreparedMemoryPlan.WorkspaceEntry>();
        var assignments = new ArrayList<PreparationResourceAssignment>();
        for (var r : analysis.requirements()) {
            if (r instanceof PreparationResourceRequirement.Buffer b) {
                var slot = new BufferSlot(buffers.size()); buffers.add(new PreparedMemoryPlan.BufferEntry(slot,b.byteSize(),b.byteAlignment()));
                assignments.add(new PreparationResourceAssignment.Buffer(b,slot,buffers.size()-1,0));
            } else if (r instanceof PreparationResourceRequirement.Workspace w) {
                var slot = new WorkspaceSlot(workspaces.size()); workspaces.add(new PreparedMemoryPlan.WorkspaceEntry(slot,w.byteSize(),w.byteAlignment()));
                assignments.add(new PreparationResourceAssignment.Workspace(w,slot,workspaces.size()-1));
            }
        }
        return new CpuPartitionFinalizer().finalizePartition(new BackendPartitionFinalization<>(analysis,new PreparedMemoryPlan(buffers,workspaces),assignments)).executable();
    }

    private static CompiledNode node(int id,OperationKind kind,List<Integer> in,int out) {
        return new CompiledNode(new NodeId(id),new Operation(kind,NoOperationAttrs.INSTANCE),in.stream().map(ValueId::new).toList(),List.of(new ValueId(out)));
    }
    private static List<BufferRepresentationBinding> binding(CpuBackendIntegration cpu,DataType type,MemorySegment s) {
        return List.of(new BufferRepresentationBinding(cpu.borrow(new MemorySegmentStorage(type,s.byteSize()/type.byteWidth(),s)),RunResourceOwnership.BORROWED));
    }
    private static List<io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation> workspaces(PreparedExecutable e) {
        return e.memoryPlan().workspaces().stream().map(w -> (io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation)
                CpuContiguousWorkspace.allocate(w.byteSize(),w.byteAlignment())).toList();
    }
    private static RunState state(PreparedExecutable e,CpuBackendIntegration cpu,DataType type,MemorySegment y,MemorySegment g,MemorySegment out) {
        return new RunState(e.memoryPlan(),List.of(binding(cpu,type,y),binding(cpu,type,g),binding(cpu,type,out)),workspaces(e));
    }
    private static MemorySegment storage(DataType type,double[] x) {
        if (type == DataType.FLOAT64) return MemorySegment.ofArray(x.clone());
        float[] f = new float[x.length]; for (int i = 0; i < f.length; i++) f[i] = (float)x[i]; return MemorySegment.ofArray(f);
    }
    private static double read(DataType type,MemorySegment s,int i) { return type == DataType.FLOAT32 ? s.getAtIndex(ValueLayout.JAVA_FLOAT,i) : s.getAtIndex(ValueLayout.JAVA_DOUBLE,i); }
    private static double typed(DataType type,double v) { return type == DataType.FLOAT32 ? (float)v : v; }
    private static void check(DataType type,boolean log,MemorySegment y,MemorySegment g,MemorySegment actual) {
        for (int row = 0; row < 2; row++) {
            BigDecimal sum = BigDecimal.ZERO;
            for (int f = 0; f < 3; f++) { int i = row*3+f; sum = sum.add(new BigDecimal(log ? read(type,g,i) : typed(type,read(type,g,i)*read(type,y,i)))); }
            double rounded = type == DataType.FLOAT32 ? sum.floatValue() : sum.doubleValue();
            for (int f = 0; f < 3; f++) {
                int i = row*3+f; double saved = read(type,y,i), seed = read(type,g,i);
                double expected = log ? typed(type,seed-typed(type,typed(type,StrictMath.exp(saved))*rounded))
                        : typed(type,saved*typed(type,seed-rounded));
                if (type == DataType.FLOAT32) assertEquals(Float.floatToRawIntBits((float)expected),Float.floatToRawIntBits((float)read(type,actual,i)));
                else assertEquals(Double.doubleToRawLongBits(expected),Double.doubleToRawLongBits(read(type,actual,i)));
            }
        }
    }
}
