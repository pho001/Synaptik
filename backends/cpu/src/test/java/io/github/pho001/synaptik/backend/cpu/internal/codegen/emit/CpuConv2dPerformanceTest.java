package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuConv2dLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuConv2dLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionUnitPlan;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.convolution.*;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.ByteOrder;
import java.nio.file.*;
import java.time.Instant;
import java.util.*;

/** Generated/direct five-fork performance harness for CPU 0008. */
public final class CpuConv2dPerformanceTest {
    private static final long MIN_BATCH_NANOS = 25_000_000L;
    private static final Path EVIDENCE = Path.of(
            "/private/tmp/synaptik-cpu-0008-retained-evidence-20260826");
    private static final ValueLayout.OfShort SHORT = ValueLayout.JAVA_SHORT_UNALIGNED
            .withOrder(ByteOrder.nativeOrder());
    private static final ValueLayout.OfDouble DOUBLE = ValueLayout.JAVA_DOUBLE_UNALIGNED
            .withOrder(ByteOrder.nativeOrder());
    private static volatile long sink;
    @FunctionalInterface private interface Action { long run() throws Throwable; }
    private record Case(String name, Action generated, Action direct, Runnable verify) { }
    private record Geometry(long[][] extents, long[][] strides, long[] offsets,
            Conv2dAttrs attrs, DataType resultType, boolean bias, boolean add, boolean relu) { }

    private CpuConv2dPerformanceTest() { }

    /** Runs one isolated fork, or aggregates accepted fork zero through four. */
    public static void main(String[] args) throws Throwable {
        if (args.length > 0 && args[0].equals("aggregate")) { aggregate(); return; }
        int fork = args.length == 0 ? 0 : Integer.parseInt(args[0]);
        int attempt = args.length < 2 ? 0 : Integer.parseInt(args[1]);
        StringBuilder report = new StringBuilder();
        try (Arena arena = Arena.ofConfined()) {
            environment(report);
            List<Case> cases = cases(arena);
            int[] repetitions = new int[cases.size() * 2];
            for (int i = 0; i < cases.size(); i++) {
                gate(cases.get(i));
                repetitions[2 * i] = repetitions(cases.get(i).generated);
                repetitions[2 * i + 1] = repetitions(cases.get(i).direct);
            }
            long[][] generated = new long[cases.size()][9], direct = new long[cases.size()][9];
            Random random = new Random(0x0008_20260826L ^ fork * 0x9e3779b97f4a7c15L
                    ^ attempt * 0xd1b54a32d192ed03L);
            for (int round = -5; round < 9; round++) {
                List<Integer> order = new ArrayList<>();
                for (int i = 0; i < cases.size(); i++) order.add(i);
                Collections.shuffle(order, random);
                for (int i : order) {
                    long gt, dt;
                    if (random.nextBoolean()) {
                        gt = time(cases.get(i).generated, repetitions[2 * i]);
                        dt = time(cases.get(i).direct, repetitions[2 * i + 1]);
                    } else {
                        dt = time(cases.get(i).direct, repetitions[2 * i + 1]);
                        gt = time(cases.get(i).generated, repetitions[2 * i]);
                    }
                    if (round >= 0) {
                        generated[i][round] = gt / repetitions[2 * i];
                        direct[i][round] = dt / repetitions[2 * i + 1];
                        gate(cases.get(i));
                    }
                }
            }
            int failures = 0;
            for (int i = 0; i < cases.size(); i++) {
                long gm = median(generated[i]), dm = median(direct[i]);
                double ratio = (double) gm / dm;
                if (!(ratio <= 1.15)) failures++;
                report.append(String.format(Locale.ROOT, "RESULT,%s,%d,%d,%.9f,%s,%s%n",
                        cases.get(i).name, gm, dm, ratio, Arrays.toString(generated[i]),
                        Arrays.toString(direct[i])));
            }
            report.append("SINK,").append(sink).append('\n');
            if (failures != 0) throw new AssertionError("ratio failures " + failures);
        } catch (Throwable failure) {
            report.append("REJECTED,").append(failure.getClass().getName()).append(',')
                    .append(clean(failure.getMessage())).append('\n');
            retain(false, fork, attempt, report.toString());
            System.out.print(report); throw failure;
        }
        retain(true, fork, attempt, report.toString());
        System.out.print(report);
    }

    private static List<Case> cases(Arena arena) throws Throwable {
        var result = new ArrayList<Case>();
        result.add(f32("CONV-DENSE-F32", Shape.of(2,8,24,24), Shape.of(16,8,3,3),
                Shape.of(2,16,22,22), Conv2dAttrs.defaults(), false, false, 1));
        result.add(f64("CONV-GROUPED-F64", Shape.of(1,8,25,25), Shape.of(8,2,3,3),
                Shape.of(1,8,25,25), new Conv2dAttrs(1,1,2,2,2,2,4), true));
        result.add(bf16("CONV-DEPTHWISE-BF16", Shape.of(2,8,24,24), Shape.of(8,1,3,3),
                Shape.of(2,8,24,24), new Conv2dAttrs(1,1,1,1,1,1,8)));
        result.add(mixed(arena));
        result.add(f32("CONV-FUSED-ADD-RELU", Shape.of(2,8,24,24), Shape.of(16,8,3,3),
                Shape.of(2,16,22,22), Conv2dAttrs.defaults(), true, true, 1));
        result.add(f32("CONV-PARALLEL-F32", Shape.of(4,16,28,28), Shape.of(32,16,3,3),
                Shape.of(4,32,26,26), Conv2dAttrs.defaults(), false, false, 4));
        result.add(f32("CONV-SPLIT-PAIR", Shape.of(2,8,24,24), Shape.of(16,8,3,3),
                Shape.of(2,16,22,22), Conv2dAttrs.defaults(), true, true, -1));
        result.add(pointwiseControl());
        return List.copyOf(result);
    }

    private static Case pointwiseControl() throws Throwable {
        Shape shape = Shape.of(262_144);
        TensorDescriptor descriptor = descriptor(shape);
        List<ValueId> ids = List.of(new ValueId(100), new ValueId(101), new ValueId(102));
        var node = new CompiledNode(new NodeId(100), new Operation(BinaryArithmeticKind.ADD,
                NoOperationAttrs.INSTANCE), ids.subList(0, 2), List.of(ids.get(2)));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                List.of(node.id()));
        var values = ids.stream().map(id -> new GraphValue(id, descriptor)).toList();
        var memory = List.of(
                new LogicalMemoryRequirement(ids.get(0), descriptor, Optional.empty(),
                        List.of(partition), false),
                new LogicalMemoryRequirement(ids.get(1), descriptor, Optional.empty(),
                        List.of(partition), false),
                new LogicalMemoryRequirement(ids.get(2), descriptor, Optional.of(partition),
                        List.of(), true));
        var base = new PrepareContext<>(partition, List.of(node), values, memory, Map.of(),
                CpuPartitionAnalysisInputs.DEFAULT);
        var unit = new CpuPartitionPreparer().analyze(withInputs(base,
                Collections.nCopies(3, CarrierAccess.FLOAT_ARRAY), 1)).plan().units().getFirst();
        MethodHandle handle = handle(new CpuClassFileKernelGenerator(), unit);
        long[] geometry = pointwiseGeometry(unit);
        float[] left = new float[262_144], right = new float[262_144];
        float[] generated = new float[262_144], direct = new float[262_144];
        fill(left); fillExternal(right);
        Action generatedAction = () -> { handle.invokeExact(left, right, generated, geometry,
                0L, 262_144L); return checksum(generated); };
        Action directAction = () -> { for (int i=0;i<direct.length;i++)
                direct[i]=left[i]+right[i]; return checksum(direct); };
        return new Case("CONTROL-F32-ADD", generatedAction, directAction,
                () -> { if(!Arrays.equals(generated,direct))throw new AssertionError("control"); });
    }

    private static Case f32(String name, Shape inputShape, Shape weightShape, Shape outputShape,
            Conv2dAttrs attrs, boolean addRelu, boolean publishedIntermediate, int ranges)
            throws Throwable {
        PrepareContext<CpuPartitionAnalysisInputs> context = addRelu
                ? chain(inputShape, weightShape, outputShape, attrs, publishedIntermediate,
                    ranges < 0 ? 1 : ranges)
                : directContext(List.of(DataType.FLOAT32, DataType.FLOAT32), inputShape,
                    weightShape, outputShape, attrs, ranges);
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        int count = Math.toIntExact(Arrays.stream(outputShape.toLongArray()).reduce(1,
                Math::multiplyExact));
        float[] input = new float[Math.toIntExact(Arrays.stream(inputShape.toLongArray())
                .reduce(1, Math::multiplyExact))];
        float[] weight = new float[Math.toIntExact(Arrays.stream(weightShape.toLongArray())
                .reduce(1, Math::multiplyExact))];
        float[] external = addRelu ? new float[count] : null;
        float[] generatedIntermediate = publishedIntermediate ? new float[count] : null;
        float[] directIntermediate = publishedIntermediate ? new float[count] : null;
        float[] generated = new float[count], direct = new float[count];
        fill(input); fill(weight); if (external != null) fillExternal(external);
        var geometry = geometry(plan.units().getFirst(), attrs, false,
                !publishedIntermediate && addRelu, !publishedIntermediate && addRelu);
        var generator = new CpuClassFileKernelGenerator();
        MethodHandle first = handle(generator, plan.units().getFirst());
        Action generatedAction;
        Action directAction;
        if (publishedIntermediate) {
            MethodHandle second = handle(generator, plan.units().get(1));
            long[] suffixGeometry = pointwiseGeometry(plan.units().get(1));
            generatedAction = () -> { first.invokeExact(input, weight, generatedIntermediate,
                    plan.units().getFirst().conv2dGeometry().orElseThrow().pack(new long[3]),
                    0L, (long) count); second.invokeExact(generatedIntermediate, external, generated,
                    suffixGeometry, 0L, (long) count); return checksum(generated); };
            directAction = () -> { conv32(geometry, input, weight, null, null,
                    directIntermediate, 0, count); for (int i=0;i<count;i++)
                    direct[i]=Math.max(+0.0f, directIntermediate[i]+external[i]);
                    return checksum(direct); };
        } else {
            long[] packed = plan.units().getFirst().conv2dGeometry().orElseThrow()
                    .pack(new long[addRelu ? 4 : 3]);
            int selectedRanges = plan.units().getFirst().selectedRangeCount();
            generatedAction = () -> { for (int r=0;r<selectedRanges;r++) {
                    long start=(long)count*r/selectedRanges,end=(long)count*(r+1)/selectedRanges;
                    if(addRelu) first.invokeExact(input,weight,external,generated,packed,start,end);
                    else first.invokeExact(input,weight,generated,packed,start,end); }
                    return checksum(generated); };
            directAction = () -> { for(int r=0;r<selectedRanges;r++) conv32(geometry,input,weight,
                    null,external,direct,(long)count*r/selectedRanges,
                    (long)count*(r+1)/selectedRanges); return checksum(direct); };
        }
        return new Case(name, generatedAction, directAction,
                () -> { if(!Arrays.equals(generated,direct)) throw new AssertionError(name); });
    }

    private static Case f64(String name, Shape inputShape, Shape weightShape, Shape outputShape,
            Conv2dAttrs attrs, boolean bias) throws Throwable {
        var context = directContext(Collections.nCopies(bias ? 3 : 2, DataType.FLOAT64), inputShape,
                weightShape, outputShape, attrs, 1);
        var unit = new CpuPartitionPreparer().analyze(context).plan().units().getFirst();
        MethodHandle handle = handle(new CpuClassFileKernelGenerator(), unit);
        int xc=count(inputShape),wc=count(weightShape),yc=count(outputShape),oc=(int)weightShape.toLongArray()[0];
        double[] x=new double[xc],w=new double[wc],b=bias?new double[oc]:null,go=new double[yc],d=new double[yc];
        fill(x);fill(w);if(b!=null)fill(b);
        long[] packed=unit.conv2dGeometry().orElseThrow().pack(new long[bias?4:3]);
        Geometry g=geometry(unit,attrs,bias,false,false);
        Action generated=()->{if(bias)handle.invokeExact(x,w,b,go,packed,0L,(long)yc);else handle.invokeExact(x,w,go,packed,0L,(long)yc);return checksum(go);};
        Action direct=()->{conv64(g,x,w,b,d,0,yc);return checksum(d);};
        return new Case(name,generated,direct,()->{if(!Arrays.equals(go,d))throw new AssertionError(name);});
    }

    private static Case bf16(String name, Shape inputShape, Shape weightShape, Shape outputShape,
            Conv2dAttrs attrs) throws Throwable {
        var context=directContext(Collections.nCopies(3,DataType.BFLOAT16),inputShape,weightShape,
                outputShape,attrs,1);var unit=new CpuPartitionPreparer().analyze(context).plan().units().getFirst();
        MethodHandle handle=handle(new CpuClassFileKernelGenerator(),unit);int yc=count(outputShape);
        short[]x=new short[count(inputShape)],w=new short[count(weightShape)],b=new short[(int)weightShape.toLongArray()[0]],go=new short[yc],d=new short[yc];fill(x);fill(w);fill(b);
        long[]packed=unit.conv2dGeometry().orElseThrow().pack(new long[4]);Geometry g=geometry(unit,attrs,true,false,false);
        Action generated=()->{handle.invokeExact(x,w,b,go,packed,0L,(long)yc);return checksum(go);};
        Action direct=()->{conv16(g,x,w,b,d,0,yc);return checksum(d);};
        return new Case(name,generated,direct,()->{if(!Arrays.equals(go,d))throw new AssertionError(name);});
    }

    private static Case mixed(Arena arena) throws Throwable {
        Shape xs=Shape.of(1,4,18,19),ws=Shape.of(6,2,3,2),ys=Shape.of(1,6,16,18);
        var layouts=List.of(LayoutDescriptor.of(xs,new long[]{1600,389,20,1},7,true),
                LayoutDescriptor.of(ws,new long[]{31,13,4,2},3,true),
                LayoutDescriptor.of(ys,new long[]{2200,347,20,1},5,true));
        var base=CpuConv2dLoweringTest.context(List.of(DataType.BFLOAT16,DataType.FLOAT64),xs,ws,ys,
                new Conv2dAttrs(1,1,0,0,1,1,2),layouts);
        var context=withInputs(base,List.of(CarrierAccess.MEMORY_SEGMENT,CarrierAccess.DOUBLE_ARRAY,
                CarrierAccess.MEMORY_SEGMENT),1);var unit=new CpuPartitionPreparer().analyze(context).plan().units().getFirst();
        MethodHandle handle=handle(new CpuClassFileKernelGenerator(),unit);var boundaries=unit.conv2dGeometry().orElseThrow().boundaries();
        MemorySegment x=arena.allocate(span(boundaries.get(0))*2L,2),go=arena.allocate(span(boundaries.get(2))*8L,8),d=arena.allocate(span(boundaries.get(2))*8L,8);double[]w=new double[Math.toIntExact(span(boundaries.get(1)))];
        for(long i=0;i<x.byteSize()/2;i++)x.set(SHORT,i*2,bf((float)((i%29-14)*.03125)));fill(w);
        int yc=count(ys);long[]packed=unit.conv2dGeometry().orElseThrow().pack(new long[3]);Geometry g=geometry(unit,new Conv2dAttrs(1,1,0,0,1,1,2),false,false,false);
        Action generated=()->{handle.invokeExact(x,w,go,packed,0L,(long)yc);return checksum64(go);};
        Action direct=()->{convMixed(g,x,w,d,0,yc);return checksum64(d);};
        return new Case("CONV-GENERAL-MIXED",generated,direct,()->{if(go.mismatch(d)!=-1)throw new AssertionError("mixed");});
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> directContext(List<DataType> types,
            Shape x,Shape w,Shape y,Conv2dAttrs attrs,int ranges){var base=CpuConv2dLoweringTest.context(types,x,w,y,attrs,null);List<CarrierAccess>cs=Collections.nCopies(types.size()+1,types.getFirst()==DataType.FLOAT64?CarrierAccess.DOUBLE_ARRAY:types.getFirst()==DataType.BFLOAT16?CarrierAccess.SHORT_ARRAY:CarrierAccess.FLOAT_ARRAY);return withInputs(base,cs,Math.max(1,ranges));}
    private static PrepareContext<CpuPartitionAnalysisInputs> withInputs(PrepareContext<CpuPartitionAnalysisInputs>b,List<CarrierAccess>c,int p){var config=new CpuPartitionAnalysisInputs.PortableExecutionConfig(CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.SCALAR,p,p,1);return new PrepareContext<>(b.partition(),b.nodes(),b.values(),b.memoryRequirements(),b.constants(),new CpuPartitionAnalysisInputs(false,c,config));}

    private static PrepareContext<CpuPartitionAnalysisInputs> chain(Shape xs,Shape ws,Shape ys,
            Conv2dAttrs attrs,boolean publish,int ranges){TensorDescriptor x=descriptor(xs),w=descriptor(ws),t=descriptor(ys);List<ValueId>ids=java.util.stream.LongStream.range(0,6).mapToObj(ValueId::new).toList();var nodes=List.of(new CompiledNode(new NodeId(0),new Operation(Conv2dKind.CONV2D,attrs),ids.subList(0,2),List.of(ids.get(3))),new CompiledNode(new NodeId(1),new Operation(BinaryArithmeticKind.ADD,NoOperationAttrs.INSTANCE),List.of(ids.get(3),ids.get(2)),List.of(ids.get(4))),new CompiledNode(new NodeId(2),new Operation(UnaryElementwiseKind.RELU,NoOperationAttrs.INSTANCE),List.of(ids.get(4)),List.of(ids.get(5))));var partition=new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,nodes.stream().map(CompiledNode::id).toList());var ds=List.of(x,w,t,t,t,t);var values=new ArrayList<GraphValue>();var memory=new ArrayList<LogicalMemoryRequirement>();for(int i=0;i<6;i++){values.add(new GraphValue(ids.get(i),ds.get(i)));boolean produced=i>=3,published=i==5||publish&&i==3;memory.add(new LogicalMemoryRequirement(ids.get(i),ds.get(i),produced?Optional.of(partition):Optional.empty(),published?List.of():List.of(partition),published));}var base=new PrepareContext<>(partition,nodes,values,memory,Map.of(),CpuPartitionAnalysisInputs.DEFAULT);return withInputs(base,Collections.nCopies(publish?5:4,CarrierAccess.FLOAT_ARRAY),ranges);}
    private static TensorDescriptor descriptor(Shape s){return new TensorDescriptor(DataType.FLOAT32,s,Optional.of(LayoutDescriptor.contiguous(s)),false);}

    private static MethodHandle handle(CpuClassFileKernelGenerator g,ExecutionUnitPlan u){byte[]b=g.generateClassBytes(u.portablePlan().specialization(),u.portablePlan().kernelIr());return g.defineClassBytes(u.portablePlan().specialization(),b).entryPoint();}
    private static Geometry geometry(ExecutionUnitPlan u,Conv2dAttrs a,boolean bias,boolean add,boolean relu){var bs=u.conv2dGeometry().orElseThrow().boundaries();long[][]e=new long[bs.size()][],s=new long[bs.size()][];long[]o=new long[bs.size()];for(int i=0;i<bs.size();i++){e[i]=bs.get(i).extents();s[i]=bs.get(i).strides();o[i]=bs.get(i).offset();}return new Geometry(e,s,o,a,u.portablePlan().specialization().boundaryDataTypes().getLast(),bias,add,relu);}
    private static long[] pointwiseGeometry(ExecutionUnitPlan u){int rank=u.accessBindings().getFirst().plan().iterationRank(),n=u.accessBindings().size();long[]g=new long[2*rank+n+n*rank+2*n];var first=u.accessBindings().getFirst();for(int a=0;a<rank;a++){g[a]=first.extents().get(a);g[rank+a]=0;}for(int v=0;v<n;v++){var b=u.accessBindings().get(v);g[2*rank+v]=b.startAddress();for(int a=0;a<rank;a++)g[2*rank+n+v*rank+a]=b.effectiveStrides().get(a);long inner=1;for(int a=rank-b.plan().contiguousSuffix();a<rank;a++)inner*=b.extents().get(a);g[2*rank+n+n*rank+n+v]=inner;}return g;}

    private static void conv32(Geometry g,float[]x,float[]w,float[]b,float[]external,float[]out,long start,long end){long[]xe=g.extents[0],we=g.extents[1],ye=g.extents[g.extents.length-1],xs=g.strides[0],ws=g.strides[1],ys=g.strides[g.strides.length-1];long opg=ye[1]/g.attrs.groups();for(long cell=start;cell<end;cell++){long q=cell,ow=q%ye[3];q/=ye[3];long oh=q%ye[2];q/=ye[2];long oc=q%ye[1],n=q/ye[1],sumIndex=g.offsets[g.offsets.length-1]+n*ys[0]+oc*ys[1]+oh*ys[2]+ow*ys[3];float sum=b==null?0:b[(int)(g.offsets[2]+oc*g.strides[2][0])];long cb=oc/opg*we[1];for(long ic=0;ic<we[1];ic++)for(long kh=0;kh<we[2];kh++)for(long kw=0;kw<we[3];kw++){long ih=oh*g.attrs.strideHeight()-g.attrs.paddingHeight()+kh*g.attrs.dilationHeight(),iw=ow*g.attrs.strideWidth()-g.attrs.paddingWidth()+kw*g.attrs.dilationWidth();float xv=ih<0||ih>=xe[2]||iw<0||iw>=xe[3]?0:x[(int)(g.offsets[0]+n*xs[0]+(cb+ic)*xs[1]+ih*xs[2]+iw*xs[3])];float wv=w[(int)(g.offsets[1]+oc*ws[0]+ic*ws[1]+kh*ws[2]+kw*ws[3])];sum=(float)(sum+xv*wv);}if(external!=null){sum=(float)(sum+external[(int)sumIndex]);if(g.relu)sum=Math.max(+0f,sum);}out[(int)sumIndex]=sum;}}
    private static void conv64(Geometry g,double[]x,double[]w,double[]b,double[]out,long start,long end){long[]xe=g.extents[0],we=g.extents[1],ye=g.extents[g.extents.length-1],xs=g.strides[0],ws=g.strides[1],ys=g.strides[g.strides.length-1];long opg=ye[1]/g.attrs.groups();for(long cell=start;cell<end;cell++){long q=cell,ow=q%ye[3];q/=ye[3];long oh=q%ye[2];q/=ye[2];long oc=q%ye[1],n=q/ye[1],oa=g.offsets[g.offsets.length-1]+n*ys[0]+oc*ys[1]+oh*ys[2]+ow*ys[3];double sum=b==null?0:b[(int)(g.offsets[2]+oc*g.strides[2][0])];long cb=oc/opg*we[1];for(long ic=0;ic<we[1];ic++)for(long kh=0;kh<we[2];kh++)for(long kw=0;kw<we[3];kw++){long ih=oh*g.attrs.strideHeight()-g.attrs.paddingHeight()+kh*g.attrs.dilationHeight(),iw=ow*g.attrs.strideWidth()-g.attrs.paddingWidth()+kw*g.attrs.dilationWidth();double xv=ih<0||ih>=xe[2]||iw<0||iw>=xe[3]?0:x[(int)(g.offsets[0]+n*xs[0]+(cb+ic)*xs[1]+ih*xs[2]+iw*xs[3])];sum+=xv*w[(int)(g.offsets[1]+oc*ws[0]+ic*ws[1]+kh*ws[2]+kw*ws[3])];}out[(int)oa]=sum;}}
    private static void conv16(Geometry g,short[]x,short[]w,short[]b,short[]out,long start,long end){long[]xe=g.extents[0],we=g.extents[1],ye=g.extents[g.extents.length-1],xs=g.strides[0],ws=g.strides[1],ys=g.strides[g.strides.length-1];long opg=ye[1]/g.attrs.groups();for(long cell=start;cell<end;cell++){long q=cell,ow=q%ye[3];q/=ye[3];long oh=q%ye[2];q/=ye[2];long oc=q%ye[1],n=q/ye[1],oa=g.offsets[g.offsets.length-1]+n*ys[0]+oc*ys[1]+oh*ys[2]+ow*ys[3];float sum=decode(b[(int)(g.offsets[2]+oc*g.strides[2][0])]);long cb=oc/opg*we[1];for(long ic=0;ic<we[1];ic++)for(long kh=0;kh<we[2];kh++)for(long kw=0;kw<we[3];kw++){long ih=oh*g.attrs.strideHeight()-g.attrs.paddingHeight()+kh*g.attrs.dilationHeight(),iw=ow*g.attrs.strideWidth()-g.attrs.paddingWidth()+kw*g.attrs.dilationWidth();float xv=ih<0||ih>=xe[2]||iw<0||iw>=xe[3]?0:decode(x[(int)(g.offsets[0]+n*xs[0]+(cb+ic)*xs[1]+ih*xs[2]+iw*xs[3])]);float wv=decode(w[(int)(g.offsets[1]+oc*ws[0]+ic*ws[1]+kh*ws[2]+kw*ws[3])]);sum=(float)(sum+xv*wv);}out[(int)oa]=bf(sum);}}
    private static void convMixed(Geometry g,MemorySegment x,double[]w,MemorySegment out,long start,long end){long[]xe=g.extents[0],we=g.extents[1],ye=g.extents[2],xs=g.strides[0],ws=g.strides[1],ys=g.strides[2];long opg=ye[1]/g.attrs.groups();for(long cell=start;cell<end;cell++){long q=cell,ow=q%ye[3];q/=ye[3];long oh=q%ye[2];q/=ye[2];long oc=q%ye[1],n=q/ye[1],oa=g.offsets[2]+n*ys[0]+oc*ys[1]+oh*ys[2]+ow*ys[3];double sum=0;long cb=oc/opg*we[1];for(long ic=0;ic<we[1];ic++)for(long kh=0;kh<we[2];kh++)for(long kw=0;kw<we[3];kw++){long ih=oh*g.attrs.strideHeight()-g.attrs.paddingHeight()+kh*g.attrs.dilationHeight(),iw=ow*g.attrs.strideWidth()-g.attrs.paddingWidth()+kw*g.attrs.dilationWidth();double xv=ih<0||ih>=xe[2]||iw<0||iw>=xe[3]?0:decode(x.get(SHORT,2*(g.offsets[0]+n*xs[0]+(cb+ic)*xs[1]+ih*xs[2]+iw*xs[3])));sum+=xv*w[(int)(g.offsets[1]+oc*ws[0]+ic*ws[1]+kh*ws[2]+kw*ws[3])];}out.set(DOUBLE,oa*8,sum);}}

    private static int count(Shape s){return Math.toIntExact(Arrays.stream(s.toLongArray()).reduce(1,Math::multiplyExact));}private static long span(CpuConv2dLowering.Layout l){long m=l.offset();for(int i=0;i<l.extents().length;i++)m+=Math.max(0,l.extents()[i]-1)*l.strides()[i];return m+1;}
    private static void fill(float[]x){for(int i=0;i<x.length;i++)x[i]=(i%31-15)*.03125f;}private static void fill(double[]x){for(int i=0;i<x.length;i++)x[i]=(i%31-15)*.03125;}private static void fill(short[]x){for(int i=0;i<x.length;i++)x[i]=bf((i%31-15)*.03125f);}private static void fillExternal(float[]x){for(int i=0;i<x.length;i++)x[i]=(i%13-6)*.0625f;}
    private static short bf(float v){int b=Float.floatToRawIntBits(v);if((b&0x7fffffff)>0x7f800000)return(short)0x7fc0;return(short)((b+0x7fff+(b>>>16&1))>>>16);}private static float decode(short v){return Float.intBitsToFloat(Short.toUnsignedInt(v)<<16);}
    private static long checksum(float[]x){long h=0;for(float v:x)h=Long.rotateLeft(h,1)^Integer.toUnsignedLong(Float.floatToRawIntBits(v));return h;}private static long checksum(double[]x){long h=0;for(double v:x)h=Long.rotateLeft(h,1)^Double.doubleToRawLongBits(v);return h;}private static long checksum(short[]x){long h=0;for(short v:x)h=Long.rotateLeft(h,1)^Short.toUnsignedLong(v);return h;}private static long checksum64(MemorySegment x){long h=0;for(long p=0;p<x.byteSize();p+=8)h=Long.rotateLeft(h,1)^Double.doubleToRawLongBits(x.get(DOUBLE,p));return h;}
    private static void gate(Case c)throws Throwable{long g=c.generated.run(),d=c.direct.run();if(g!=d)throw new AssertionError(c.name+" checksum "+g+"/"+d);c.verify.run();}private static long time(Action a,int n)throws Throwable{long s=System.nanoTime(),v=0;for(int i=0;i<n;i++)v^=a.run();sink^=v;return System.nanoTime()-s;}private static int repetitions(Action a)throws Throwable{int n=1;while(time(a,n)<MIN_BATCH_NANOS)n=Math.multiplyExact(n,2);return n;}private static long median(long[]a){long[]c=a.clone();Arrays.sort(c);return c[4];}
    private static void environment(StringBuilder out){long total=Runtime.getRuntime().totalMemory(),max=Runtime.getRuntime().maxMemory();String java=System.getProperty("java.version");out.append("ENV,java=").append(java).append(",vm=").append(System.getProperty("java.vm.name")).append(",processors=").append(Runtime.getRuntime().availableProcessors()).append(",totalMemory=").append(total).append(",maxMemory=").append(max).append(",byteOrder=").append(ByteOrder.nativeOrder()).append('\n');long low=900L<<20,high=1100L<<20;if(!java.startsWith("26")||total<low||max<low||max>high)throw new AssertionError("requires Java 26 -Xms1g -Xmx1g");}
    private static void retain(boolean accepted,int fork,int attempt,String text)throws Exception{Path dir=EVIDENCE.resolve(accepted?"forks":"rejected-samples");Files.createDirectories(dir);String name=accepted?"fork-"+fork+".csv":"fork-"+fork+"-attempt-"+attempt+"-"+Instant.now().toEpochMilli()+".csv";Files.writeString(dir.resolve(name),text);}private static String clean(String s){return s==null?"":s.replace('\n',' ').replace('\r',' ').replace(',',';');}
    private static void aggregate()throws Exception{List<String>names=null;double[][]ratios=new double[5][8];for(int f=0;f<5;f++){List<String>ns=new ArrayList<>();int i=0;for(String line:Files.readAllLines(EVIDENCE.resolve("forks/fork-"+f+".csv")))if(line.startsWith("RESULT,")){String[]x=line.split(",",6);ns.add(x[1]);ratios[f][i++]=Double.parseDouble(x[4]);}if(i!=8)throw new AssertionError("fork "+f+" count "+i);if(names==null)names=List.copyOf(ns);else if(!names.equals(ns))throw new AssertionError("inventory");}StringBuilder s=new StringBuilder();int failures=0;for(int i=0;i<8;i++){double[]v=new double[5];for(int f=0;f<5;f++)v[f]=ratios[f][i];Arrays.sort(v);if(v[2]>1.15)failures++;s.append(String.format(Locale.ROOT,"AGGREGATE,%s,%.9f,%s%n",names.get(i),v[2],Arrays.toString(v)));}long rejected=Files.exists(EVIDENCE.resolve("rejected-samples"))?Files.list(EVIDENCE.resolve("rejected-samples")).count():0;s.append("COUNTS,accepted=5,rejected=").append(rejected).append('\n');Files.writeString(EVIDENCE.resolve("summary.csv"),s);System.out.print(s);if(failures!=0)throw new AssertionError("aggregate failures "+failures);}
}
