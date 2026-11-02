package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.ir.*;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.*;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.classfile.*;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.lang.foreign.*;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Untimed generated/split/independent-Java numerical and Class-File acceptance. */
public class CpuSoftmaxBackwardGeneratedKernelTest {
    @Test void preparedBindingProtectsFullGradientSpanAndExactScratchBeforeAnyWrite() {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean log : List.of(false,true)) {
            var c = CpuSoftmaxBackwardLoweringTest.context(type,Shape.of(2,4),log,List.of());
            var e = CpuPartitionFinalizerTest.finalizeExecutable(new CpuPartitionPreparer().analyze(c),Optional.empty());
            assertEquals(3,e.memoryPlan().buffers().size());
            assertEquals(8L*type.byteWidth(),e.memoryPlan().buffers().getLast().byteSize());
            var y = segment(array(type,new double[]{1,2,3,4,5,6,7,8})); var g = segment(array(type,8));
            var output = segment(array(type,8)); var w = e.memoryPlan().workspaces().getFirst();
            for (var invalid : List.of(y,g,output.asReadOnly(),output.asSlice(0,output.byteSize()-type.byteWidth()))) {
                try (var scratch = io.github.pho001.synaptik.backend.cpu.internal.memory.CpuContiguousWorkspace.allocate(w.byteSize(),8);
                        var state = boundState(e,type,List.of(y,g,invalid),scratch)) {
                    assertThrows(IllegalArgumentException.class,() -> e.bind(state));
                    assertEquals(type == DataType.FLOAT32 ? 1f : 1d,type == DataType.FLOAT32 ? y.getAtIndex(ValueLayout.JAVA_FLOAT,0) : y.getAtIndex(ValueLayout.JAVA_DOUBLE,0));
                }
            }
            try (var wrong = io.github.pho001.synaptik.backend.cpu.internal.memory.CpuContiguousWorkspace.allocate(w.byteSize()-8,8);
                    var state = boundState(e,type,List.of(y,g,output),wrong)) {
                assertThrows(IllegalArgumentException.class,() -> e.bind(state));
            }
            try (var scratch = io.github.pho001.synaptik.backend.cpu.internal.memory.CpuContiguousWorkspace.allocate(w.byteSize(),8);
                    var state = boundState(e,type,List.of(y,g,scratch.writableSegment().asSlice(0,output.byteSize())),scratch)) {
                assertThrows(IllegalArgumentException.class,() -> e.bind(state));
            }
        }
    }

    private static io.github.pho001.synaptik.runtime.run.RunState boundState(
            io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutable e,DataType type,
            List<MemorySegment> physical,io.github.pho001.synaptik.backend.cpu.internal.memory.CpuContiguousWorkspace workspace) {
        var bindings = physical.stream().map(s -> List.of(new io.github.pho001.synaptik.runtime.run.BufferRepresentationBinding(
                io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBorrowedBuffer.borrow(new io.github.pho001.synaptik.model.storage.MemorySegmentStorage(type,s.byteSize()/type.byteWidth(),s)),
                io.github.pho001.synaptik.runtime.run.RunResourceOwnership.BORROWED))).toList();
        return new io.github.pho001.synaptik.runtime.run.RunState(e.memoryPlan(),bindings,List.of(workspace));
    }

    @Test void broadcastCotangentLoadsOncePerRowAndIdenticalRolesShareEachPassLoad() throws Throwable {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean log : List.of(false,true))
            for (boolean shared : List.of(false,true)) {
                var heap = type == DataType.FLOAT32 ? CarrierAccess.FLOAT_ARRAY : CarrierAccess.DOUBLE_ARRAY;
                var c = CpuSoftmaxBackwardLoweringTest.context(type,Shape.of(2,4),log,List.of());
                var nodes = c.nodes(); var values = new ArrayList<>(c.values());
                if (shared) nodes = nodes.stream().map(n -> new CompiledNode(n.id(),n.operation(),
                        n.inputs().stream().map(v -> v.equals(new ValueId(1)) ? new ValueId(0) : v).toList(),n.outputs())).toList();
                else values.set(1,new GraphValue(new ValueId(1),new TensorDescriptor(type,Shape.of(2,4),
                        Optional.of(LayoutDescriptor.of(Shape.of(2,4),new long[]{1,0},0,true)),false)));
                c = CpuSoftmaxBackwardLoweringTest.rebuild(c,nodes,values,Set.of());
                c = new PrepareContext<>(c.partition(),c.nodes(),c.values(),c.memoryRequirements(),Map.of(),
                        new CpuPartitionAnalysisInputs(false,Collections.nCopies(shared ? 2 : 3,heap)));
                var plan = new CpuPartitionPreparer().analyze(c).plan();
                var route = plan.units().getFirst().portablePlan();
                var model = ClassFile.of().parse(new CpuClassFileKernelGenerator().generateClassBytes(route.specialization(),route.kernelIr()));
                var ops = model.methods().getFirst().code().orElseThrow().elementStream().filter(Instruction.class::isInstance)
                        .map(Instruction.class::cast).map(Instruction::opcode).toList();
                assertEquals(shared ? 2 : log ? 2 : 3,Collections.frequency(ops,type == DataType.FLOAT32 ? Opcode.FALOAD : Opcode.DALOAD));
                assertEquals(1,Collections.frequency(ops,type == DataType.FLOAT32 ? Opcode.FASTORE : Opcode.DASTORE));
                Object y = array(type,new double[]{.25,.5,-.25,1,.125,-.5,.75,1}), g = shared ? y : array(type,new double[]{2,-3});
                Object actual = array(type,8), expected = array(type,8);
                invoke(plan,shared ? List.of(y,actual) : List.of(y,g,actual),new long[shared ? 2 : 3],0,2);
                oracle(plan,y,0,g,0,shared ? 4 : 1,shared ? 1 : 0,expected,0,0,2);
                for (int i = 0; i < 8; i++) assertRaw(type,get(expected,i),get(actual,i));
                // An empty legal range must not perform the hoisted row read.
                Object empty = array(type,0);
                invoke(plan,shared ? List.of(empty,empty) : List.of(empty,empty,empty),new long[shared ? 2 : 3],1,1);
            }
        // Broadcast row-load placement is guarded for both empty logical dimensions as well.
        for (var shape : List.of(Shape.of(0,4),Shape.of(2,0))) {
            var c = CpuSoftmaxBackwardLoweringTest.context(DataType.FLOAT32,shape,true,List.of());
            var values = new ArrayList<>(c.values()); values.set(1,new GraphValue(new ValueId(1),new TensorDescriptor(DataType.FLOAT32,shape,
                    Optional.of(LayoutDescriptor.of(shape,new long[]{0,0},0,true)),false)));
            var plan = new CpuPartitionPreparer().analyze(CpuSoftmaxBackwardLoweringTest.rebuild(c,c.nodes(),values,Set.of())).plan();
            invoke(plan,List.of(MemorySegment.NULL,MemorySegment.NULL,MemorySegment.NULL),new long[3],0,shape.toLongArray()[0]);
        }
    }
    @Test void bothTypesKindsRanksEmptyAndSingletonMatchMaterializedSplitAndCleanJava() throws Throwable {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean log : List.of(false,true))
            for (var shape : List.of(Shape.of(5),Shape.of(2,5),Shape.of(0),Shape.of(0,5),Shape.of(2,0),Shape.of(2,1))) {
                var c = CpuSoftmaxBackwardLoweringTest.context(type,shape,log,List.of());
                int count = Math.toIntExact(shape.knownElementCount().orElseThrow());
                Object y = array(type,count), g = array(type,count), actual = array(type,count), expected = array(type,count);
                for (int i = 0; i < count; i++) { set(y,i,(i%7-3)*.125); set(g,i,(i%5-2)*.75); }
                var plan = new CpuPartitionPreparer().analyze(c).plan();
                int width = Math.toIntExact(shape.toLongArray()[shape.rank()-1]);
                int rows = shape.rank() == 1 ? 1 : Math.toIntExact(shape.toLongArray()[0]);
                invoke(plan,List.of(segment(y),segment(g),segment(actual)),new long[3],0,rows);
                oracle(plan,y,0,g,0,width,1,expected,0,0,rows);
                Object split = split(c,y,g);
                for (int i = 0; i < count; i++) { assertRaw(type,get(expected,i),get(actual,i)); assertRaw(type,get(split,i),get(actual,i)); }
            }
    }

    @Test void specialClassesSignsCancellationOverflowAndPrimitiveRoundingRemainExact() throws Throwable {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean log : List.of(false,true)) {
            double min = type == DataType.FLOAT32 ? Float.MIN_VALUE : Double.MIN_VALUE;
            double max = type == DataType.FLOAT32 ? Float.MAX_VALUE : Double.MAX_VALUE;
            double epsilon = Math.scalb(1d,type == DataType.FLOAT32 ? -23 : -52);
            double[][] saved = {{1+epsilon,1,1,1},{1,1,1,1},{-0d,0d,1,-1},{min,-min,1,1},
                    {1,1,1,1},{Double.NaN,0,1,-1},{Double.POSITIVE_INFINITY,Double.NEGATIVE_INFINITY,0,-0d},
                    {max,max,-max,-max},{-1000,0,1000,-1}};
            double[][] seeds = {{1+epsilon,-(1+2*epsilon),0,0},{1,epsilon/2,-1,epsilon/2},
                    {-0d,-0d,-0d,-0d},{1,1,min,-min},{max,max,-max,-max},{1,2,3,4},
                    {0,1,Double.POSITIVE_INFINITY,Double.NEGATIVE_INFINITY},{2,2,2,2},{1,-2,3,-4}};
            for (int example = 0; example < saved.length; example++) {
                var c = CpuSoftmaxBackwardLoweringTest.context(type,Shape.of(4),log,List.of());
                Object y = array(type,saved[example]), g = array(type,seeds[example]), actual = array(type,4), expected = array(type,4);
                var plan = new CpuPartitionPreparer().analyze(c).plan();
                invoke(plan,List.of(segment(y),segment(g),segment(actual)),new long[3],0,1);
                oracle(plan,y,0,g,0,4,1,expected,0,0,1); var split = split(c,y,g);
                for (int i = 0; i < 4; i++) { assertRaw(type,get(expected,i),get(actual,i)); assertRaw(type,get(split,i),get(actual,i)); }
            }
            // A once-rounded product sum is zero, whereas an unrounded dot product is eps^2.
            if (!log) {
                var c = CpuSoftmaxBackwardLoweringTest.context(type,Shape.of(2),false,List.of());
                var p = new CpuPartitionPreparer().analyze(c).plan();
                Object y = array(type,new double[]{1+epsilon,1}), g = array(type,new double[]{1+epsilon,-(1+2*epsilon)});
                Object out = array(type,2);
                invoke(p,List.of(segment(y),segment(g),segment(out)),new long[3],0,1);
                double expected = type == DataType.FLOAT32 ? (float)(1+epsilon)*(float)(1+epsilon) : (1+epsilon)*(1+epsilon);
                assertRaw(type,expected,get(out,0));
            }
        }
    }

    @Test void allEightIndependentCarrierTuplesOffsetsBroadcastsAndSliceSubranges() throws Throwable {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean log : List.of(false,true))
            for (int mask = 0; mask < 8; mask++) for (int broadcast = 0; broadcast < 3; broadcast++) {
                var tuple = new ArrayList<CarrierAccess>();
                for (int p = 0; p < 3; p++) tuple.add((mask&(1<<p)) != 0 ? CarrierAccess.MEMORY_SEGMENT
                        : type == DataType.FLOAT32 ? CarrierAccess.FLOAT_ARRAY : CarrierAccess.DOUBLE_ARRAY);
                var c = CpuSoftmaxBackwardLoweringTest.context(type,Shape.of(3,4),log,tuple);
                var values = new ArrayList<GraphValue>();
                for (var v : c.values()) {
                    var shape = v.descriptor().shape(); long[] strides = LayoutDescriptor.contiguous(shape).strides();
                    if (v.id().equals(new ValueId(1)) && broadcast != 0) strides = new long[]{broadcast == 1 ? 1 : 0,0};
                    values.add(new GraphValue(v.id(),new TensorDescriptor(type,shape,Optional.of(LayoutDescriptor.of(shape,strides,2,
                            v.id().equals(new ValueId(1)) && broadcast != 0)),false)));
                }
                c = CpuSoftmaxBackwardLoweringTest.rebuild(c,c.nodes(),values,Set.of());
                var plan = new CpuPartitionPreparer().analyze(c).plan();
                Object y = array(type,20), g = array(type,20), actual = array(type,20), expected = array(type,20);
                for (int i = 0; i < 20; i++) { set(y,i,(i%7-3)*.125); set(g,i,(i%5-2)*.5); set(actual,i,97); set(expected,i,97); }
                int gr = broadcast == 0 ? 4 : broadcast == 1 ? 1 : 0, gf = broadcast == 0 ? 1 : 0;
                try (var arena = Arena.ofConfined()) {
                    var physical = new ArrayList<Object>(); var originals = List.of(y,g,actual);
                    for (int i = 0; i < 3; i++) {
                        if (tuple.get(i) == CarrierAccess.MEMORY_SEGMENT) {
                            var from = segment(originals.get(i)); var nativeSegment = arena.allocate(from.byteSize(),type.byteWidth());
                            MemorySegment.copy(from,0,nativeSegment,0,from.byteSize()); physical.add(nativeSegment);
                        } else physical.add(originals.get(i));
                    }
                    // Each call is a legal complete-slice subrange. Sentinels protect row zero.
                    invoke(plan,physical,new long[]{3,3,3},1,2);
                    invoke(plan,physical,new long[]{3,3,3},2,3);
                    invoke(plan,physical,new long[]{3,3,3},3,3);
                    if (physical.getLast() instanceof MemorySegment s) MemorySegment.copy(s,0,segment(actual),0,s.byteSize());
                }
                oracle(plan,y,5,g,5,gr,gf,expected,5,1,3);
                for (int i = 0; i < 20; i++) assertRaw(type,get(expected,i),get(actual,i));
            }
    }

    @Test void generatedClassFilesAreDirectTypedTwoPassBodiesWithNoHiddenRuntimeWork() throws Throwable {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean log : List.of(false,true)) {
            var heap = type == DataType.FLOAT32 ? CarrierAccess.FLOAT_ARRAY : CarrierAccess.DOUBLE_ARRAY;
            var plan = new CpuPartitionPreparer().analyze(CpuSoftmaxBackwardLoweringTest.context(type,Shape.of(32,128),log,Collections.nCopies(3,heap))).plan();
            var route = plan.units().getFirst().portablePlan(); var generator = new CpuClassFileKernelGenerator();
            byte[] bytes = generator.generateClassBytes(route.specialization(),route.kernelIr());
            assertArrayEquals(bytes,generator.generateClassBytes(route.specialization(),route.kernelIr()));
            var model = ClassFile.of().parse(bytes);
            assertEquals(1,model.methods().size()); assertTrue(model.fields().isEmpty()); assertEquals(0,model.constantPool().bootstrapMethodCount());
            for (var entry : model.constantPool()) if (entry instanceof MemberRefEntry member) {
                String owner = member.owner().asInternalName();
                assertTrue(Set.of("java/lang/Float","java/lang/Double","java/lang/Long","java/lang/StrictMath").contains(owner)
                        || owner.startsWith("java/lang/foreign/"),owner);
            }
            var ops = model.methods().getFirst().code().orElseThrow().elementStream().filter(Instruction.class::isInstance)
                    .map(Instruction.class::cast).map(Instruction::opcode).toList();
            assertTrue(ops.stream().noneMatch(o -> o.name().contains("NEW") || o.name().contains("SWITCH") || o == Opcode.INVOKEDYNAMIC));
            assertEquals(log ? 1 : 2,Collections.frequency(ops,type == DataType.FLOAT32 ? Opcode.FMUL : Opcode.DMUL));
            assertEquals(1,Collections.frequency(ops,type == DataType.FLOAT32 ? Opcode.FSUB : Opcode.DSUB));
            assertEquals(0,Collections.frequency(ops,Opcode.FADD));
            assertFalse(ops.contains(Opcode.IDIV)); assertFalse(ops.contains(Opcode.IREM)); assertFalse(ops.contains(Opcode.LDIV));
            // Exactly one output-array store site, solely in the second pass.
            assertEquals(1,Collections.frequency(ops,type == DataType.FLOAT32 ? Opcode.FASTORE : Opcode.DASTORE));
            // Both feature passes and the outer complete-slice loop are top tested.
            assertTrue(Collections.frequency(ops,Opcode.IF_ICMPGE) >= 3);
            String root = System.getenv("SYNAPTIK_CPU0051_STRUCTURE");
            if (root != null) {
                var path = java.nio.file.Path.of(root); java.nio.file.Files.createDirectories(path);
                String stem = (log ? "logsoftmax" : "softmax")+"-"+type;
                java.nio.file.Files.write(path.resolve(stem+".class"),bytes);
                java.nio.file.Files.write(path.resolve(stem+".compatibility"),route.specialization().compatibilityBytes());
                java.nio.file.Files.write(path.resolve(stem+".classidentity"),route.specialization().classIdentityBytes());
            }
            Object y = array(type,4096), g = array(type,4096), actual = array(type,4096), expected = array(type,4096);
            for (int i = 0; i < 4096; i++) { set(y,i,(i%11-5)*.125); set(g,i,(i%7-3)*.5); }
            invoke(plan,List.of(y,g,actual),new long[3],0,32); oracle(plan,y,0,g,0,128,1,expected,0,0,32);
            for (int i = 0; i < 4096; i++) assertRaw(type,get(expected,i),get(actual,i));
        }
    }

    private static void oracle(CpuPartitionPreparationPlan plan,Object y,int yb,Object g,int gb,int gr,int gf,Object out,int ob,int start,int end) {
        var ir = (CpuSoftmaxBackwardIr)plan.units().getFirst().portablePlan().portableKernelIr();
        int width = Math.toIntExact(ir.aggregate().domainCount()), limbs = ir.aggregate().stateLimbCount();
        try (var arena = Arena.ofConfined()) {
            var state = arena.allocate(ir.aggregate().scratchSliceBytes(),8);
            if (y instanceof float[] a) {
                if (ir.logarithmic()) CpuSoftmaxBackwardCleanJavaOracle.f32Log(a,yb,(float[])g,gb,gr,gf,(float[])out,ob,width,start,end,state,limbs);
                else CpuSoftmaxBackwardCleanJavaOracle.f32Soft(a,yb,(float[])g,gb,gr,gf,(float[])out,ob,width,start,end,state,limbs);
            } else {
                if (ir.logarithmic()) CpuSoftmaxBackwardCleanJavaOracle.f64Log((double[])y,yb,(double[])g,gb,gr,gf,(double[])out,ob,width,start,end,state,limbs);
                else CpuSoftmaxBackwardCleanJavaOracle.f64Soft((double[])y,yb,(double[])g,gb,gr,gf,(double[])out,ob,width,start,end,state,limbs);
            }
        }
    }

    static void invoke(CpuPartitionPreparationPlan plan,List<?> physical,long[] bases,long start,long end) throws Throwable {
        var route = plan.units().getFirst().portablePlan(); var generator = new CpuClassFileKernelGenerator();
        var entry = generator.defineClassBytes(route.specialization(), route.kernelIr(), generator.generateClassBytes(route.specialization(), route.kernelIr())).entryPoint();
        try (var arena = Arena.ofConfined()) {
            var args = new ArrayList<Object>(physical);
            if (route.specialization().scratchParameter()) args.add(arena.allocate(plan.aggregateGeometry().orElseThrow().workspaceBytes(1),8));
            args.add(route.specialization().softmaxBackwardIr().isPresent() ? plan.aggregateGeometry().orElseThrow().packFusion(bases)
                    : plan.aggregateGeometry().isPresent() ? plan.aggregateGeometry().orElseThrow().pack(bases) : pointwiseGeometry(plan));
            args.add(start); args.add(end); entry.invokeWithArguments(args);
        }
    }

    private static Object split(PrepareContext<CpuPartitionAnalysisInputs> c,Object y,Object g) throws Throwable {
        var physical = new HashMap<ValueId,Object>(); physical.put(new ValueId(0),y); physical.put(new ValueId(1),g);
        var type = c.values().getFirst().descriptor().dataType();
        for (var n : c.nodes()) {
            var one = new CpuPartitionDagDecomposer().unitContext(c,List.of(n),CpuPartitionAnalysisInputs.DEFAULT);
            var plan = new CpuPartitionPreparer().analyze(one).plan();
            var args = new ArrayList<Object>();
            for (int i = 0; i < plan.boundaryValues().size(); i++) {
                var id = plan.boundaryValues().get(i);
                if (!physical.containsKey(id)) physical.put(id,array(type,Math.toIntExact(plan.bufferDeclarations().get(i).byteSize()/type.byteWidth())));
                args.add(segment(physical.get(id)));
            }
            invoke(plan,args,new long[args.size()],0,plan.elementCount());
        }
        return physical.get(new ValueId(5));
    }

    private static long[] pointwiseGeometry(CpuPartitionPreparationPlan plan) {
        var bindings = plan.accessBindings(); int count = bindings.size(), rank = bindings.getFirst().extents().size();
        long[] p = new long[2*rank+3*count+count*rank];
        for (int a = 0; a < rank; a++) p[a] = bindings.getFirst().extents().get(a);
        for (int i = 0; i < count; i++) {
            var b = bindings.get(i); p[2*rank+i] = b.startAddress();
            for (int a = 0; a < rank; a++) p[2*rank+count+i*rank+a] = b.effectiveStrides().get(a);
            long size = 1; for (int a = rank-b.plan().contiguousSuffix(); a < rank; a++) size *= b.extents().get(a);
            p[2*rank+count+count*rank+count+i] = size;
        }
        return p;
    }
    private static Object array(DataType type,int size) { return type == DataType.FLOAT32 ? new float[size] : new double[size]; }
    private static Object array(DataType type,double[] values) { Object a = array(type,values.length); for (int i = 0; i < values.length; i++) set(a,i,values[i]); return a; }
    private static MemorySegment segment(Object a) { return a instanceof float[] f ? MemorySegment.ofArray(f) : MemorySegment.ofArray((double[])a); }
    private static double get(Object a,int i) { return a instanceof float[] f ? f[i] : ((double[])a)[i]; }
    private static void set(Object a,int i,double x) { if (a instanceof float[] f) f[i] = (float)x; else ((double[])a)[i] = x; }
    private static void assertRaw(DataType type,double expected,double actual) {
        if (Double.isNaN(expected)) assertTrue(Double.isNaN(actual));
        else if (type == DataType.FLOAT32) assertEquals(Float.floatToRawIntBits((float)expected),Float.floatToRawIntBits((float)actual));
        else assertEquals(Double.doubleToRawLongBits(expected),Double.doubleToRawLongBits(actual));
    }
}
