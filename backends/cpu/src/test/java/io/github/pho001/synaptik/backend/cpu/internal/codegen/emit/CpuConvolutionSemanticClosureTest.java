package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuConv2dLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuConv2dLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuConv3dLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuConv3dLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.foreign.MemorySegment;
import java.nio.charset.StandardCharsets;
import java.util.*;
import org.junit.jupiter.api.Test;

/**
 * Executes every finite convolution inventory row against a test-local channels-first oracle.
 * The oracle is intentionally scalar Java and has no dependency on CPU lowering or reference
 * kernels.
 */
class CpuConvolutionSemanticClosureTest {
    private static final List<DataType> FLOATING = List.of(DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64);
    private static final CpuPartitionAnalysisInputs.MaterializationPolicy MATERIALIZATION =
            new CpuPartitionAnalysisInputs.MaterializationPolicy(true, 0, 1, 20, 1, 3, 1_000_000, 1, 1);

    @Test void everyConv1dCompositionInventoryRowDefinesAndInvokesItsActualLoweredEntry() throws Throwable {
        Set<String> owners = new TreeSet<>(); int rows = 0;
        for (String geometry : List.of("ordinary", "grouped", "depthwise")) for (DataType x : FLOATING)
            for (DataType w : FLOATING) for (DataType bias : roles()) for (Request request : Request.values()) {
                Fixture fixture = new Fixture("CONV1D_COMPOSITION", geometry, x, w, bias, request);
                assertTrue(owners.add(fixture.owner()), "duplicate " + fixture);
                execute2d(fixture, CpuOneDimensionalCompositionGeneratedMatrixTest.semanticConfigure(
                        CpuOneDimensionalCompositionGeneratedMatrixTest.semanticConvContext(geometry, x, w, bias), request.name()));
                rows++;
            }
        assertEquals(540, rows); assertEquals(inventoryOwners("CONV1D_COMPOSITION"), owners);
    }

    @Test void everyConv2dInventoryRowDefinesAndInvokesItsActualGeneratedEntry() throws Throwable {
        Set<String> owners = new TreeSet<>(); int rows = 0;
        for (Case2 geometry : cases2()) for (DataType x : FLOATING) for (DataType w : FLOATING)
            for (DataType bias : roles()) for (Request request : Request.values()) {
                Fixture fixture = new Fixture("CONV2D", geometry.id, x, w, bias, request);
                assertTrue(owners.add(fixture.owner()), "duplicate " + fixture);
                execute2d(fixture, configure(CpuConv2dLoweringTest.context(types(x, w, bias), geometry.input,
                        geometry.weight, geometry.output, geometry.attrs, null), request)); rows++;
            }
        assertEquals(540, rows); assertEquals(inventoryOwners("CONV2D"), owners);
    }

    @Test void everyConv3dInventoryRowDefinesAndInvokesItsActualGeneratedEntry() throws Throwable {
        Set<String> owners = new TreeSet<>(); int rows = 0;
        for (Case3 geometry : cases3()) for (DataType x : FLOATING) for (DataType w : FLOATING)
            for (DataType bias : roles()) for (Request request : Request.values()) {
                Fixture fixture = new Fixture("CONV3D", geometry.id, x, w, bias, request);
                assertTrue(owners.add(fixture.owner()), "duplicate " + fixture);
                execute3d(fixture, configure(CpuConv3dLoweringTest.context(types(x, w, bias), geometry.input,
                        geometry.weight, geometry.output, geometry.attrs, null), request)); rows++;
            }
        assertEquals(540, rows); assertEquals(inventoryOwners("CONV3D"), owners);
    }

    private static List<DataType> roles() { return Arrays.asList(null, DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64); }
    private static List<DataType> types(DataType x, DataType w, DataType bias) {
        List<DataType> result = new ArrayList<>(List.of(x, w)); if (bias != null) result.add(bias); return result;
    }

    private static void execute2d(Fixture f, PrepareContext<CpuPartitionAnalysisInputs> context) throws Throwable {
        var unit = new CpuPartitionPreparer().analyze(context).plan().units().getFirst();
        var route = unit.portablePlan(); var g = unit.conv2dGeometry().orElseThrow();
        byte[] bytes = new CpuClassFileKernelGenerator().generateClassBytes(route.specialization(), route.kernelIr());
        assertArrayEquals(bytes, new CpuClassFileKernelGenerator().generateClassBytes(route.specialization(), route.kernelIr()), f + " bytes");
        var entry = new CpuClassFileKernelGenerator().defineClassBytes(route.specialization(), bytes);
        List<Store> stores = stores(route.specialization().boundaryDataTypes(), route.specialization().carrierPattern(), layouts(g));
        populate(stores, layouts(g)); Store output = stores.getLast(); output.fill(-91.25);
        double[] expected = output.snapshot(); Conv2dAttrs attrs = (Conv2dAttrs) context.nodes().get(f.form.equals("CONV1D_COMPOSITION") ? 2 : 0).operation().attrs();
        oracle2d(stores, layouts(g), attrs, expected);
        preserveOutsideWorkerRange(expected, output.snapshot(), layouts(g).getLast());
        invoke(entry, stores, g.pack(new long[stores.size()]), outputDomain(layouts(g).getLast()), f.request.parallel);
        assertOutput(expected, output, f); assertInputsUnchanged(stores, f);
    }

    private static void execute3d(Fixture f, PrepareContext<CpuPartitionAnalysisInputs> context) throws Throwable {
        var unit = new CpuPartitionPreparer().analyze(context).plan().units().getFirst();
        var route = unit.portablePlan(); var g = unit.conv3dGeometry().orElseThrow();
        var generator = new CpuClassFileKernelGenerator(); byte[] bytes = generator.generateClassBytes(route.specialization(), route.kernelIr());
        assertArrayEquals(bytes, generator.generateClassBytes(route.specialization(), route.kernelIr()), f + " bytes");
        var entry = generator.defineClassBytes(route.specialization(), bytes);
        List<Store> stores = stores(route.specialization().boundaryDataTypes(), route.specialization().carrierPattern(), layouts(g));
        populate(stores, layouts(g)); Store output = stores.getLast(); output.fill(-91.25);
        double[] expected = output.snapshot(); Conv3dAttrs attrs = (Conv3dAttrs) context.nodes().getFirst().operation().attrs();
        oracle3d(stores, layouts(g), attrs, expected);
        preserveOutsideWorkerRange(expected, output.snapshot(), layouts(g).getLast());
        invoke(entry, stores, g.pack(new long[stores.size()]), outputDomain(layouts(g).getLast()), f.request.parallel);
        assertOutput(expected, output, f); assertInputsUnchanged(stores, f);
    }

    private static void invoke(CpuGeneratedKernel artifact, List<Store> stores, long[] geometry, long domain, boolean parallel) throws Throwable {
        long start = 1, end = domain - 1; // leaves both physical edge cells as untouched sentinels
        Object[] args = new Object[stores.size() + 3]; for (int i = 0; i < stores.size(); i++) args[i] = stores.get(i).argument();
        args[stores.size()] = geometry;
        if (parallel) { long middle = start + (end - start) / 2; args[stores.size() + 1] = middle; args[stores.size() + 2] = end; artifact.entryPoint().invokeWithArguments(args);
            args[stores.size() + 1] = start; args[stores.size() + 2] = middle; artifact.entryPoint().invokeWithArguments(args); }
        else { args[stores.size() + 1] = start; args[stores.size() + 2] = end; artifact.entryPoint().invokeWithArguments(args); }
    }

    /* Clean channels-first cross-correlation oracle. It only reads the test-local stores. */
    private static void oracle2d(List<Store> s, List<L> l, Conv2dAttrs a, double[] expected) {
        long[] x = l.get(0).e, w = l.get(1).e, y = l.getLast().e; int bias = s.size() == 4 ? 2 : -1;
        for (long n=0;n<y[0];n++) for(long oc=0;oc<y[1];oc++) for(long oh=0;oh<y[2];oh++) for(long ow=0;ow<y[3];ow++) {
            double sum = bias < 0 ? 0 : s.get(bias).get(at(l.get(bias), oc)); long base = (oc/(y[1]/a.groups()))*(x[1]/a.groups());
            for(long ic=0;ic<w[1];ic++) for(long kh=0;kh<w[2];kh++) for(long kw=0;kw<w[3];kw++) { long ih=oh*a.strideHeight()-a.paddingHeight()+kh*a.dilationHeight(), iw=ow*a.strideWidth()-a.paddingWidth()+kw*a.dilationWidth();
                if(ih>=0&&ih<x[2]&&iw>=0&&iw<x[3]) sum=narrow(s.getLast().type, sum+narrow(s.getLast().type, s.get(0).get(at(l.get(0),n,base+ic,ih,iw))*s.get(1).get(at(l.get(1),oc,ic,kh,kw)))); }
            expected[(int)at(l.getLast(),n,oc,oh,ow)] = sum;
        }
    }
    private static void oracle3d(List<Store> s, List<L> l, Conv3dAttrs a, double[] expected) {
        long[] x=l.get(0).e,w=l.get(1).e,y=l.getLast().e; int bias=s.size()==4?2:-1;
        for(long n=0;n<y[0];n++)for(long oc=0;oc<y[1];oc++)for(long od=0;od<y[2];od++)for(long oh=0;oh<y[3];oh++)for(long ow=0;ow<y[4];ow++){ double sum=bias<0?0:s.get(bias).get(at(l.get(bias),oc)); long base=(oc/(y[1]/a.groups()))*(x[1]/a.groups());
            for(long ic=0;ic<w[1];ic++)for(long kd=0;kd<w[2];kd++)for(long kh=0;kh<w[3];kh++)for(long kw=0;kw<w[4];kw++){long id=od*a.strideDepth()-a.paddingDepth()+kd*a.dilationDepth(),ih=oh*a.strideHeight()-a.paddingHeight()+kh*a.dilationHeight(),iw=ow*a.strideWidth()-a.paddingWidth()+kw*a.dilationWidth();if(id>=0&&id<x[2]&&ih>=0&&ih<x[3]&&iw>=0&&iw<x[4])sum=narrow(s.getLast().type,sum+narrow(s.getLast().type,s.get(0).get(at(l.get(0),n,base+ic,id,ih,iw))*s.get(1).get(at(l.get(1),oc,ic,kd,kh,kw))));} expected[(int)at(l.getLast(),n,oc,od,oh,ow)]=sum; }
    }

    private static void populate(List<Store> s, List<L> l) { for(int boundary=0;boundary<s.size()-1;boundary++) for(long i=0;i<elements(l.get(boundary).e);i++) s.get(boundary).set(atLinear(l.get(boundary),i), .125*(boundary+1)+i*.0625); for(Store store:s)store.before=store.snapshot(); }
    private static void preserveOutsideWorkerRange(double[] expected, double[] sentinel, L output) { long domain=elements(output.e); for(long logical=0;logical<domain;logical++) if(logical==0||logical==domain-1) expected[(int)atLinear(output,logical)]=sentinel[(int)atLinear(output,logical)]; }
    private static void assertInputsUnchanged(List<Store> stores, Fixture f) { for(int i=0;i<stores.size()-1;i++) assertArrayEquals(stores.get(i).before, stores.get(i).snapshot(), f+" input "+i); }
    private static void assertOutput(double[] expected, Store actual, Fixture f) { for(int i=0;i<expected.length;i++) assertEquals(q(actual.type,expected[i]),actual.get(i),actual.type==DataType.BFLOAT16?.025:actual.type==DataType.FLOAT32?2e-5:1e-12,f+" physical "+i); }

    private static PrepareContext<CpuPartitionAnalysisInputs> configure(PrepareContext<CpuPartitionAnalysisInputs> base, Request r) { List<GraphValue> values=new ArrayList<>();List<LogicalMemoryRequirement> memory=new ArrayList<>();for(int i=0;i<base.values().size();i++){TensorDescriptor d=base.values().get(i).descriptor();if(r.general){long[] strides=d.layout().orElseThrow().strides();for(int j=0;j<strides.length;j++)strides[j]*=2;d=new TensorDescriptor(d.dataType(),d.shape(),Optional.of(LayoutDescriptor.of(d.shape(),strides,1,true)),false);}values.add(new GraphValue(base.values().get(i).id(),d));var m=base.memoryRequirements().get(i);memory.add(new LogicalMemoryRequirement(m.valueId(),d,m.producerPartition(),m.consumerPartitions(),m.graphOutput()));}List<CarrierAccess> carriers=new ArrayList<>();for(int i=0;i<values.size();i++)carriers.add(r.segment||r.mixed&&i%2==1?CarrierAccess.MEMORY_SEGMENT:CpuGeneratedDirectEvidenceClosureTest.heapCarrier(values.get(i).descriptor().dataType()));return new PrepareContext<>(base.partition(),base.nodes(),values,memory,base.constants(),new CpuPartitionAnalysisInputs(false,carriers,r.execution,r.materialization?MATERIALIZATION:CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED)); }
    private static List<L> layouts(CpuConv2dLowering.Geometry g){return g.boundaries().stream().map(v->new L(v.extents(),v.offset(),v.strides())).toList();}
    private static List<L> layouts(CpuConv3dLowering.Geometry g){return g.boundaries().stream().map(v->new L(v.extents(),v.offset(),v.strides())).toList();}
    private static List<Store> stores(List<DataType> types,List<CarrierAccess> carriers,List<L> layouts){List<Store> r=new ArrayList<>();for(int i=0;i<types.size();i++)r.add(new Store(types.get(i),carriers.get(i),(int)(max(layouts.get(i))+1)));return r;}
    private static long max(L l){long n=l.o;for(int i=0;i<l.e.length;i++)n+=(l.e[i]-1)*l.s[i];return n;} private static long at(L l,long... c){long n=l.o;for(int i=0;i<c.length;i++)n+=c[i]*l.s[i];return n;}private static long atLinear(L l,long linear){long n=l.o;for(int i=l.e.length-1;i>=0;i--){n+=(linear%l.e[i])*l.s[i];linear/=l.e[i];}return n;}private static long elements(long[] e){long n=1;for(long x:e)n*=x;return n;}private static long outputDomain(L l){return elements(l.e);} private static double narrow(DataType t,double v){return t==DataType.FLOAT64?v:(float)v;}private static double q(DataType t,double v){return t==DataType.BFLOAT16?Float.intBitsToFloat((ScalarValue.bfloat16((float)v).bfloat16Bits()&0xffff)<<16):t==DataType.FLOAT32?(float)v:v;}
    private static Set<String> inventoryOwners(String form) throws Exception {try(var in=CpuConvolutionSemanticClosureTest.class.getResourceAsStream("/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/generated-coverage-inventory.tsv")){assertNotNull(in);Set<String> r=new TreeSet<>();for(String line:new String(in.readAllBytes(),StandardCharsets.UTF_8).split("\\n")){String[] f=line.split("\\t",-1);if(f.length==27&&f[2].equals(form))assertTrue(r.add(f[0]),"inventory duplicate "+f[0]);}assertEquals(540,r.size());return r;}}
    private static List<Case2> cases2(){return List.of(new Case2("ordinary",new Conv2dAttrs(1,2,1,0,2,1,1),Shape.of(1,2,7,7),Shape.of(4,2,2,2),Shape.of(1,4,7,3)),new Case2("grouped",new Conv2dAttrs(1,2,1,0,2,1,2),Shape.of(1,2,7,7),Shape.of(4,1,2,2),Shape.of(1,4,7,3)),new Case2("depthwise",new Conv2dAttrs(1,2,1,0,2,1,2),Shape.of(1,2,7,7),Shape.of(2,1,2,2),Shape.of(1,2,7,3)));}
    private static List<Case3> cases3(){return List.of(new Case3("ordinary",new Conv3dAttrs(1,2,1,1,0,1,2,1,1,1),Shape.of(1,2,7,7,7),Shape.of(4,2,2,2,2),Shape.of(1,4,7,3,8)),new Case3("grouped",new Conv3dAttrs(1,2,1,1,0,1,2,1,1,2),Shape.of(1,2,7,7,7),Shape.of(4,1,2,2,2),Shape.of(1,4,7,3,8)),new Case3("depthwise",new Conv3dAttrs(1,2,1,1,0,1,2,1,1,2),Shape.of(1,2,7,7,7),Shape.of(2,1,2,2,2),Shape.of(1,2,7,3,8)));}
    private enum Request {HEAP_CONTIGUOUS_SCALAR(false,false,false,false,CpuGeneratedDirectEvidenceClosureTest.scalar(1)),SEGMENT_CONTIGUOUS_VECTOR(true,false,false,false,CpuGeneratedDirectEvidenceClosureTest.vector(1)),MIXED_GENERAL_PARALLEL_VECTOR(false,true,true,false,CpuGeneratedDirectEvidenceClosureTest.vector(4)),HEAP_GENERAL_PARALLEL_SCALAR(false,false,true,false,CpuGeneratedDirectEvidenceClosureTest.scalar(4)),HEAP_GENERAL_MATERIALIZATION(false,false,true,true,CpuGeneratedDirectEvidenceClosureTest.scalar(1));final boolean segment,mixed,general,materialization,parallel;final CpuPartitionAnalysisInputs.PortableExecutionConfig execution;Request(boolean s,boolean m,boolean g,boolean z,CpuPartitionAnalysisInputs.PortableExecutionConfig e){segment=s;mixed=m;general=g;materialization=z;execution=e;parallel=e.availableParallelism()>1;}}
    private record Fixture(String form,String geometry,DataType x,DataType w,DataType bias,Request request){String owner(){return form.equals("CONV1D_COMPOSITION")?"composition:conv1d/"+geometry+'/'+(bias==null?"plain/":"bias/")+x+'/'+w+(bias==null?"":"/"+bias)+'/'+request:"specialized:"+form.toLowerCase()+"/"+geometry+'/'+(bias==null?"no-bias/":"bias/")+x+'/'+w+(bias==null?"":"/"+bias)+'/'+request;}}
    private record Case2(String id,Conv2dAttrs attrs,Shape input,Shape weight,Shape output){} private record Case3(String id,Conv3dAttrs attrs,Shape input,Shape weight,Shape output){} private record L(long[] e,long o,long[] s){}
    private static final class Store {final DataType type;final CarrierAccess carrier;final Object heap;final MemorySegment segment;double[] before;Store(DataType t,CarrierAccess c,int n){type=t;carrier=c;heap=switch(t){case FLOAT64->new double[n];case FLOAT32->new float[n];case BFLOAT16->new short[n];default->throw new AssertionError(t);};segment=switch(t){case FLOAT64->MemorySegment.ofArray((double[])heap);case FLOAT32->MemorySegment.ofArray((float[])heap);case BFLOAT16->MemorySegment.ofArray((short[])heap);default->throw new AssertionError(t);};}Object argument(){return carrier==CarrierAccess.MEMORY_SEGMENT?segment:heap;}void fill(double v){for(int i=0;i<snapshot().length;i++)set(i,v);}void set(long i,double v){switch(type){case FLOAT64->((double[])heap)[(int)i]=v;case FLOAT32->((float[])heap)[(int)i]=(float)v;case BFLOAT16->((short[])heap)[(int)i]=ScalarValue.bfloat16((float)v).bfloat16Bits();default->throw new AssertionError(type);}}double get(long i){return switch(type){case FLOAT64->((double[])heap)[(int)i];case FLOAT32->((float[])heap)[(int)i];case BFLOAT16->Float.intBitsToFloat((((short[])heap)[(int)i]&0xffff)<<16);default->throw new AssertionError(type);};}double[] snapshot(){int n=type==DataType.FLOAT64?((double[])heap).length:type==DataType.FLOAT32?((float[])heap).length:((short[])heap).length;double[] r=new double[n];for(int i=0;i<n;i++)r[i]=get(i);return r;}}
}
