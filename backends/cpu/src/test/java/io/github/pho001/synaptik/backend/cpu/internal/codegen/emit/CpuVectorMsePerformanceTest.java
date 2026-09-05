package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuLossIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScatterLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.loss.LossKind;
import io.github.pho001.synaptik.model.operation.loss.LossReduction;
import io.github.pho001.synaptik.model.operation.loss.MeanSquaredErrorAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.lang.classfile.ClassFile;
import java.lang.classfile.Instruction;
import java.lang.classfile.Opcode;
import java.lang.classfile.instruction.InvokeInstruction;
import java.lang.reflect.AccessFlag;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/** Opt-in exact 12-row generated-versus-clean-Java Vector MSE performance protocol. */
class CpuVectorMsePerformanceTest {
    private static final String ENABLE = "synaptik.cpu.vectorMse.performance";
    private static final String ROOT = "synaptik.cpu.vectorMse.performanceEvidenceRoot";
    private static final int FORKS = 5, WARMUPS = 5, PAIRS = 9, EXACT_CHUNKS = 8192, OFFSET = 7;
    private static final long MINIMUM_NANOS = 25_000_000L, CALIBRATION_NANOS = 50_000_000L;
    private static final double LIMIT = 1.15d;
    private static final ByteOrder ORDER = ByteOrder.nativeOrder();
    private static final ValueLayout.OfFloat FLOAT = ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ORDER);
    private static final ValueLayout.OfDouble DOUBLE = ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ORDER);
    private static volatile long sink;

    enum Row {
        F32_AAA_EXACT(DataType.FLOAT32,"AAA",false,false,false), F32_SSS_TAIL(DataType.FLOAT32,"SSS",true,true,false),
        F32_ASA_TAIL(DataType.FLOAT32,"ASA",true,false,false), F32_SAS_EXACT(DataType.FLOAT32,"SAS",false,true,false),
        F32_SHARED_AS_TAIL(DataType.FLOAT32,"AS",true,false,true), F32_SHARED_SA_EXACT(DataType.FLOAT32,"SA",false,true,true),
        F64_AAA_EXACT(DataType.FLOAT64,"AAA",false,false,false), F64_SSS_TAIL(DataType.FLOAT64,"SSS",true,true,false),
        F64_ASA_TAIL(DataType.FLOAT64,"ASA",true,false,false), F64_SAS_EXACT(DataType.FLOAT64,"SAS",false,true,false),
        F64_SHARED_AS_TAIL(DataType.FLOAT64,"AS",true,false,true), F64_SHARED_SA_EXACT(DataType.FLOAT64,"SA",false,true,true);
        final DataType type; final String carriers; final boolean tail, parallel, shared;
        Row(DataType type, String carriers, boolean tail, boolean parallel, boolean shared) {
            this.type=type; this.carriers=carriers; this.tail=tail; this.parallel=parallel; this.shared=shared;
        }
        String range() { return tail ? "tail" : "exact"; }
    }

    public static void main(String[] args) throws Throwable {
        if (args.length == 2 && args[0].equals("--fork")) fork(root(), Integer.parseInt(args[1]));
        else parent(root(), FORKS);
    }

    @Test void exactTwelveRowProtocolAndOracleAreExecutable() throws Exception {
        assertEquals(12, Row.values().length); assertEquals(6, Arrays.stream(Row.values()).filter(r -> r.type == DataType.FLOAT32).count());
        assertEquals(6, Arrays.stream(Row.values()).filter(r -> r.type == DataType.FLOAT64).count());
        assertEquals(5, FORKS); assertEquals(5, WARMUPS); assertEquals(9, PAIRS); assertEquals(540, 12 * FORKS * PAIRS);
        assertEquals(25_000_000L, MINIMUM_NANOS); assertEquals(50_000_000L, CALIBRATION_NANOS); assertEquals(1.15d, LIMIT);
        String source = CpuVectorMsePerformanceOracle.source(specs());
        assertTrue(source.contains("SPECIES_PREFERRED") && source.contains("d.mul(d)") && source.contains("while(i<end)"));
        assertFalse(source.contains("Object") || source.contains("reduceLanes") || source.contains("invokeWithArguments"));
        byte[] oracle = CpuVectorMsePerformanceOracle.compile(specs());
        assertTrue(oracle.length > 0); inspectOracle(oracle);
    }

    @Test void generatedAndDirectTypedRowsAgreeBeforeTiming() throws Throwable {
        for (Row row : Row.values()) try (Work work = new Work(row)) { work.verify(); }
    }

    @Test void retainedFiveFreshForkEvidence() throws Exception {
        String mode = System.getProperty(ENABLE, "");
        Assumptions.assumeTrue(mode.equals("true")); parent(root(), FORKS);
    }

    @Test void diagnosticOneFreshForkEvidence() throws Throwable {
        Assumptions.assumeTrue(System.getProperty(ENABLE, "").equals("diagnostic")); parent(root(), 1);
    }

    private static Path root() {
        String value = System.getProperty(ROOT);
        if (value == null || value.isBlank()) throw new IllegalArgumentException(ROOT + " is required");
        return Path.of(value).toAbsolutePath();
    }
    private static Path dir(Path root) { return root.resolve("cpu-vector-mse-performance"); }

    private static void parent(Path root, int forks) throws Exception {
        Path evidence = dir(root); Files.createDirectories(evidence); protocol(evidence, forks);
        Files.copy(source(CpuVectorMsePerformanceTest.class), evidence.resolve("CpuVectorMsePerformanceTest.java"), StandardCopyOption.REPLACE_EXISTING);
        Files.copy(source(CpuVectorMsePerformanceOracle.class), evidence.resolve("CpuVectorMsePerformanceOracle.java"), StandardCopyOption.REPLACE_EXISTING);
        Files.write(evidence.resolve("VectorMsePerformanceOracleGenerated.class"), Oracle.BYTES);
        javap(evidence.resolve("VectorMsePerformanceOracleGenerated.class"), evidence.resolve("oracle-javap.txt"));
        String java = Path.of(System.getProperty("java.home"), "bin", "java").toString(); StringBuilder commands = new StringBuilder();
        for (int f=0; f<forks; f++) {
            List<String> command=List.of(java,"-Xms1g","-Xmx1g","-XX:-TieredCompilation","-Xbatch","--add-modules","jdk.incubator.vector","-cp",System.getProperty("java.class.path"),"-D"+ROOT+'='+root,getClassName(),"--fork",Integer.toString(f));
            commands.append(String.join(" ",command)).append('\n');
            Process process=new ProcessBuilder(command).redirectOutput(evidence.resolve("fork-"+f+".stdout").toFile()).redirectError(evidence.resolve("fork-"+f+".stderr").toFile()).start();
            assertEquals(0,process.waitFor(),"fork "+f); validateFork(evidence,f);
        }
        Files.writeString(evidence.resolve("commands.txt"),commands.toString()); aggregate(evidence,forks); manifest(evidence);
    }

    private static void protocol(Path p, int forks) throws Exception {
        Files.writeString(p.resolve("protocol.txt"),"rows=12\nforks="+forks+"\nwarmup_pairs=5\nretained_symmetric_pairs=9\norders=G-D-D-G,D-G-G-D\nshared_calibration_generated_ns=50000000\nshared_calibration_direct_ns=50000000\nindividual_retained_ns=25000000\nfixed_heap=-Xms1g,-Xmx1g\nc2_only=-XX:-TieredCompilation,-Xbatch\nretry=false\ndiscard=false\nthreshold=1.15\nrow_range_field=exact|tail\n");
        Files.writeString(p.resolve("environment.txt"),"java="+System.getProperty("java.version")+"\nvm="+System.getProperty("java.vm.name")+"\nos="+System.getProperty("os.name")+' '+System.getProperty("os.version")+"\ncpu="+System.getProperty("os.arch")+"\nheap=-Xms1g,-Xmx1g\n");
    }

    private static void fork(Path root, int fork) throws Throwable {
        if (fork != 0 && (fork < 0 || fork >= FORKS)) throw new IllegalArgumentException("fork");
        Path p=dir(root); Files.createDirectories(p); Random random=new Random(0x0008_005EL + fork);
        StringBuilder summary=new StringBuilder("row,range,iterations,fork_median,checksum\n");
        for (Row row:Row.values()) try (Work work=new Work(row)) {
            work.verify(); for(int warmup=0;warmup<WARMUPS;warmup++) {
                work.prepareTimedPair(); pair(work,1,random); work.verifyPostPair();
            }
            int iterations=calibrate(work); double[] ratios=new double[PAIRS];
            StringBuilder raw=new StringBuilder("row,range,pair,iterations,order,g_before_ns,d_after_ns,d_before_ns,g_after_ns,ratio,checksum\n");
            for(int pair=0;pair<PAIRS;pair++) { work.prepareTimedPair(); Measurement sample=pair(work,iterations,random); assertTrue(sample.allAtLeast(MINIMUM_NANOS),row+" retained timing under 25ms"); assertTrue(sample.ratio()<=LIMIT,row+" pair "+pair+"="+sample.ratio()); work.verifyPostPair(); ratios[pair]=sample.ratio(); raw.append(row).append(',').append(row.range()).append(',').append(pair).append(',').append(iterations).append(',').append(sample.order).append(',').append(sample.gb).append(',').append(sample.da).append(',').append(sample.db).append(',').append(sample.ga).append(',').append(sample.ratio()).append(',').append(work.checksum()).append('\n'); }
            Files.writeString(p.resolve("samples-fork-"+fork+"-"+row+".csv"),raw.toString()); Arrays.sort(ratios); assertTrue(ratios[4]<=LIMIT,row+" fork median="+ratios[4]);
            Files.write(p.resolve("generated-"+row+".class"),work.generatedBytes); javap(p.resolve("generated-"+row+".class"),p.resolve("generated-"+row+".javap"));
            summary.append(row).append(',').append(row.range()).append(',').append(iterations).append(',').append(ratios[4]).append(',').append(work.checksum()).append('\n');
        }
        Files.writeString(p.resolve("raw-fork-"+fork+".csv"),summary.toString());
    }

    private static int calibrate(Work work) throws Throwable { int n=1; for (;;) { work.prepareTimedPair(); Measurement m=pair(work,n,new Random(0)); work.verifyPostPair(); if(m.generated()>=CALIBRATION_NANOS && m.direct()>=CALIBRATION_NANOS) return n; n=Math.multiplyExact(n,2); } }
    private static Measurement pair(Work work,int iterations,Random random) throws Throwable { boolean gd=random.nextBoolean(); long gb,da,db,ga; if(gd){gb=time(work,true,iterations);da=time(work,false,iterations);db=time(work,false,iterations);ga=time(work,true,iterations);}else{db=time(work,false,iterations);ga=time(work,true,iterations);gb=time(work,true,iterations);da=time(work,false,iterations);} return new Measurement(gd?"G-D-D-G":"D-G-G-D",gb,da,db,ga); }
    private static long time(Work work,boolean generated,int n) throws Throwable { long start=System.nanoTime(); for(int i=0;i<n;i++) if(generated) work.generated(); else work.direct(); return System.nanoTime()-start; }

    private static final class Work implements AutoCloseable {
        final Row row; final Arena arena=Arena.ofConfined(); final Object first,second,output;
        final long[] geometry; final int count, capacity, offset; final MethodHandle generated,direct;
        final byte[] generatedBytes; final long[] firstSnapshot, secondSnapshot, expectedOutput;
        Work(Row row) throws Throwable { this.row=row; int lanes=lanes(row.type); count=EXACT_CHUNKS*lanes+(row.tail?3:0); offset=row.tail?OFFSET:0; capacity=offset+count+lanes+16; List<CarrierAccess> accesses=accesses(row); first=carrier(row.type,capacity,accesses.getFirst(),arena); second=row.shared?first:carrier(row.type,capacity,accesses.get(1),arena); output=carrier(row.type,capacity,accesses.getLast(),arena); fill(output,row.type,capacity,-91); input(first,row.type,offset,count,0); if(!row.shared) input(second,row.type,offset,count,1); var route=route(row,accesses); generatedBytes=new CpuClassFileKernelGenerator().generateClassBytes(route.specialization(),route.kernelIr()); generated=new CpuClassFileKernelGenerator().defineClassBytes(route.specialization(),generatedBytes).entryPoint(); direct=Oracle.entry(row); long[] bases=new long[]{offset,offset,offset}; geometry=((CpuLossIr)route.portableKernelIr()).geometry().pack(bases); firstSnapshot=allBits(first,row.type,capacity); secondSnapshot=allBits(second,row.type,capacity); expectedOutput=allBits(output,row.type,capacity); for(int i=0;i<count;i++) expectedOutput[offset+i]=mseBits(row.type,bits(first,row.type,offset+i),bits(second,row.type,offset+i)); }
        void generated() throws Throwable { run(generated); sink^=checksum(); }
        void direct() throws Throwable { run(direct); sink^=checksum(); }
        void run(MethodHandle handle) throws Throwable { if (!row.parallel) invoke(handle, 0, count); else { int start=0; for(int worker=0;worker<4;worker++){int end=start+count/4+(worker<count%4?1:0); invoke(handle,start,end); start=end;} } }
        void invoke(MethodHandle handle, long start, long end) throws Throwable { if(row.type==DataType.FLOAT32) invoke32(handle,start,end); else invoke64(handle,start,end); }
        void invoke32(MethodHandle h,long start,long end) throws Throwable { if(row.shared) { if(row.carriers.equals("AS")) h.invokeExact((float[])first,(MemorySegment)output,geometry,start,end); else h.invokeExact((MemorySegment)first,(float[])output,geometry,start,end); } else switch(row.carriers) { case "AAA" -> h.invokeExact((float[])first,(float[])second,(float[])output,geometry,start,end); case "SSS" -> h.invokeExact((MemorySegment)first,(MemorySegment)second,(MemorySegment)output,geometry,start,end); case "ASA" -> h.invokeExact((float[])first,(MemorySegment)second,(float[])output,geometry,start,end); case "SAS" -> h.invokeExact((MemorySegment)first,(float[])second,(MemorySegment)output,geometry,start,end); default -> throw new AssertionError(row); } }
        void invoke64(MethodHandle h,long start,long end) throws Throwable { if(row.shared) { if(row.carriers.equals("AS")) h.invokeExact((double[])first,(MemorySegment)output,geometry,start,end); else h.invokeExact((MemorySegment)first,(double[])output,geometry,start,end); } else switch(row.carriers) { case "AAA" -> h.invokeExact((double[])first,(double[])second,(double[])output,geometry,start,end); case "SSS" -> h.invokeExact((MemorySegment)first,(MemorySegment)second,(MemorySegment)output,geometry,start,end); case "ASA" -> h.invokeExact((double[])first,(MemorySegment)second,(double[])output,geometry,start,end); case "SAS" -> h.invokeExact((MemorySegment)first,(double[])second,(MemorySegment)output,geometry,start,end); default -> throw new AssertionError(row); } }
        void verify() throws Throwable { prepareTimedPair(); generated(); verifyPostPair(); prepareTimedPair(); direct(); verifyPostPair(); }
        void prepareTimedPair() { fill(output,row.type,capacity,-91); assertArrayEquals(firstSnapshot,allBits(first,row.type,capacity),row+" immutable prediction pre"); assertArrayEquals(secondSnapshot,allBits(second,row.type,capacity),row+" immutable target pre"); long sentinel=bitsOf(row.type,-91); for(long bit:allBits(output,row.type,capacity))assertEquals(sentinel,bit,row+" whole output pre"); }
        void verifyPostPair() { assertArrayEquals(firstSnapshot,allBits(first,row.type,capacity),row+" immutable prediction post"); assertArrayEquals(secondSnapshot,allBits(second,row.type,capacity),row+" immutable target post"); assertArrayEquals(expectedOutput,allBits(output,row.type,capacity),row+" whole output post"); }
        long checksum() { long value=0; int base=(int)geometry[6]; for(int i=0;i<count;i+=Math.max(1,count/17)) value=31*value+bits(output,row.type,base+i); return value; }
        @Override public void close(){arena.close();}
    }

    private static final class Oracle { static final byte[] BYTES=CpuVectorMsePerformanceOracle.compile(specs()); static final Class<?> TYPE=define(); static Class<?> define(){try{return MethodHandles.lookup().defineClass(BYTES);}catch(IllegalAccessException e){throw new ExceptionInInitializerError(e);}} static MethodHandle entry(Row r) throws NoSuchMethodException,IllegalAccessException { Class<?> a=r.type==DataType.FLOAT32?float[].class:double[].class, p=r.carriers.charAt(0)=='A'?a:MemorySegment.class, t=r.shared?p:(r.carriers.charAt(1)=='A'?a:MemorySegment.class), o=r.carriers.charAt(r.carriers.length()-1)=='A'?a:MemorySegment.class; MethodType type=r.shared?MethodType.methodType(void.class,p,o,long[].class,long.class,long.class):MethodType.methodType(void.class,p,t,o,long[].class,long.class,long.class); return MethodHandles.lookup().findStatic(TYPE,r.name().toLowerCase(),type); } }

    private static List<CpuVectorMsePerformanceOracle.Spec> specs(){List<CpuVectorMsePerformanceOracle.Spec> list=new ArrayList<>();for(Row r:Row.values())list.add(new CpuVectorMsePerformanceOracle.Spec(r.name().toLowerCase(),r.type==DataType.FLOAT32?CpuVectorMsePerformanceOracle.Floating.F32:CpuVectorMsePerformanceOracle.Floating.F64,carrier(r.carriers.charAt(0)),carrier(r.shared?r.carriers.charAt(0):r.carriers.charAt(1)),carrier(r.carriers.charAt(r.carriers.length()-1)),r.shared));return List.copyOf(list);}
    private static CpuVectorMsePerformanceOracle.Carrier carrier(char c){return c=='A'?CpuVectorMsePerformanceOracle.Carrier.ARRAY:CpuVectorMsePerformanceOracle.Carrier.SEGMENT;}
    private static List<CarrierAccess> accesses(Row r){CarrierAccess a=r.type==DataType.FLOAT32?CarrierAccess.FLOAT_ARRAY:CarrierAccess.DOUBLE_ARRAY;List<CarrierAccess> out=new ArrayList<>();for(int i=0;i<r.carriers.length();i++)out.add(r.carriers.charAt(i)=='A'?a:CarrierAccess.MEMORY_SEGMENT);return out;}
    private static io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan route(Row row,List<CarrierAccess> c){int lanes=lanes(row.type),count=EXACT_CHUNKS*lanes+(row.tail?3:0);Shape shape=Shape.of(1,count);List<Integer> roles=row.shared?List.of(0,0):List.of(0,1);List<TensorDescriptor> inputs=row.shared?List.of(CpuScatterLoweringTest.desc(row.type,shape)):List.of(CpuScatterLoweringTest.desc(row.type,shape),CpuScatterLoweringTest.desc(row.type,shape));PrepareContext<CpuPartitionAnalysisInputs> base=CpuScatterLoweringTest.context(new Operation(LossKind.MEAN_SQUARED_ERROR,new MeanSquaredErrorAttrs(LossReduction.NONE)),roles,inputs,CpuScatterLoweringTest.desc(row.type,shape));var config=new PortableExecutionConfig(ComputePreference.VECTOR_IF_ELIGIBLE,row.parallel?4:1,row.parallel?4:1,1);return new CpuPartitionPreparer().analyze(new PrepareContext<>(base.partition(),base.nodes(),base.values(),base.memoryRequirements(),Map.of(),new CpuPartitionAnalysisInputs(false,c,config))).plan().units().getFirst().portablePlan();}
    private static Object carrier(DataType t,int n,CarrierAccess c,Arena arena){if(c==CarrierAccess.MEMORY_SEGMENT)return arena.allocate((long)n*t.byteWidth(),t.byteWidth());return t==DataType.FLOAT32?new float[n]:new double[n];}
    private static void input(Object x,DataType t,int offset,int count,int role){for(int i=0;i<count;i++)put(x,t,offset+i,((i*17+role*11)%29-14)*.25);}
    private static void fill(Object x,DataType t,int n,double v){for(int i=0;i<n;i++)put(x,t,i,v);}
    private static void put(Object x,DataType t,int i,double v){if(x instanceof float[] a)a[i]=(float)v;else if(x instanceof double[] a)a[i]=v;else if(t==DataType.FLOAT32)((MemorySegment)x).setAtIndex(FLOAT,i,(float)v);else ((MemorySegment)x).setAtIndex(DOUBLE,i,v);}
    private static long bits(Object x,DataType t,int i){if(t==DataType.FLOAT32)return Float.floatToRawIntBits(x instanceof float[] a?a[i]:((MemorySegment)x).getAtIndex(FLOAT,i));return Double.doubleToRawLongBits(x instanceof double[] a?a[i]:((MemorySegment)x).getAtIndex(DOUBLE,i));}
    private static long[] allBits(Object x,DataType t,int count){long[] out=new long[count];for(int i=0;i<count;i++)out[i]=bits(x,t,i);return out;}
    private static long bitsOf(DataType t,double value){return t==DataType.FLOAT32?Float.floatToRawIntBits((float)value):Double.doubleToRawLongBits(value);}
    private static long mseBits(DataType t,long prediction,long target){if(t==DataType.FLOAT32){float d=Float.intBitsToFloat((int)prediction)-Float.intBitsToFloat((int)target);return Float.floatToRawIntBits(d*d);}double d=Double.longBitsToDouble(prediction)-Double.longBitsToDouble(target);return Double.doubleToRawLongBits(d*d);}
    private static int lanes(DataType t){return t==DataType.FLOAT32?FloatVector.SPECIES_PREFERRED.length():DoubleVector.SPECIES_PREFERRED.length();}
    private static void validateFork(Path p,int fork)throws Exception{List<String> lines=Files.readAllLines(p.resolve("raw-fork-"+fork+".csv"));assertEquals(13,lines.size());for(int i=1;i<lines.size();i++){String[] f=lines.get(i).split(",");assertEquals(Row.values()[i-1].range(),f[1]);assertTrue(Double.parseDouble(f[3])<=LIMIT);}}
    private static void aggregate(Path p,int forks)throws Exception{StringBuilder out=new StringBuilder("row,range");for(int f=0;f<forks;f++)out.append(",fork").append(f);out.append(",median,accepted\n");for(int row=0;row<12;row++){double[] v=new double[forks];for(int f=0;f<forks;f++)v[f]=Double.parseDouble(Files.readAllLines(p.resolve("raw-fork-"+f+".csv")).get(row+1).split(",")[3]);double[] s=v.clone();Arrays.sort(s);assertTrue(s[forks/2]<=LIMIT);out.append(Row.values()[row]).append(',').append(Row.values()[row].range());for(double x:v)out.append(',').append(x);out.append(',').append(s[forks/2]).append(",true\n");}Files.writeString(p.resolve("aggregate.csv"),out.toString());}
    private static void javap(Path clazz,Path target)throws Exception{Process process=new ProcessBuilder(Path.of(System.getProperty("java.home"),"bin","javap").toString(),"-c","-v","-p",clazz.toString()).redirectErrorStream(true).start();Files.write(target,process.getInputStream().readAllBytes());assertEquals(0,process.waitFor());}
    private static void inspectOracle(byte[] bytes){assertTrue(ClassFile.of().verify(bytes).isEmpty());var model=ClassFile.of().parse(bytes);assertTrue(model.flags().has(AccessFlag.FINAL));assertTrue(model.fields().isEmpty());assertEquals(13,model.methods().size());for(var row:Row.values()){var method=model.methods().stream().filter(m->m.methodName().stringValue().equals(row.name().toLowerCase())).findFirst().orElseThrow();assertTrue(method.flags().has(AccessFlag.PUBLIC)&&method.flags().has(AccessFlag.STATIC));assertFalse(method.methodTypeSymbol().descriptorString().contains("Ljava/lang/Object;"));int sub=0,mul=0,loads=0,stores=0,branches=0;for(Instruction instruction:method.code().orElseThrow().elementStream().filter(Instruction.class::isInstance).map(Instruction.class::cast).toList()){Opcode opcode=instruction.opcode();assertFalse(opcode.name().startsWith("NEW")||opcode==Opcode.MONITORENTER||opcode==Opcode.MONITOREXIT,row.toString());if(opcode.name().startsWith("IF")||opcode==Opcode.GOTO)branches++;if(instruction instanceof InvokeInstruction call){String owner=call.owner().asInternalName(),name=call.name().stringValue();assertFalse(owner.startsWith("io/github/pho001/synaptik")||owner.startsWith("java/util/")||owner.startsWith("java/lang/reflect")||owner.startsWith("java/lang/invoke"),owner+'.'+name);assertFalse(Set.of("reduceLanes","fma","lane","withLane","toArray").contains(name),name);if(name.equals("fromArray")||name.equals("fromMemorySegment"))loads++;if(name.equals("intoArray")||name.equals("intoMemorySegment"))stores++;if(name.equals("sub"))sub++;if(name.equals("mul"))mul++;}}assertEquals(2,loads,row.toString());assertEquals(1,stores,row.toString());assertEquals(1,sub,row.toString());assertEquals(1,mul,row.toString());assertTrue(branches>=4,row.toString());assertTrue(method.code().orElseThrow().elementStream().filter(Instruction.class::isInstance).map(Instruction.class::cast).anyMatch(i->i.opcode()==(row.type==DataType.FLOAT32?Opcode.FSUB:Opcode.DSUB)));assertTrue(method.code().orElseThrow().elementStream().filter(Instruction.class::isInstance).map(Instruction.class::cast).anyMatch(i->i.opcode()==(row.type==DataType.FLOAT32?Opcode.FMUL:Opcode.DMUL)));}}
    private static void manifest(Path p)throws Exception{MessageDigest digest=MessageDigest.getInstance("SHA-256");StringBuilder out=new StringBuilder();try(var paths=Files.walk(p)){for(Path x:paths.filter(Files::isRegularFile).sorted(Comparator.naturalOrder()).toList())if(!x.getFileName().toString().startsWith("manifest"))out.append(java.util.HexFormat.of().formatHex(digest.digest(Files.readAllBytes(x)))).append("  ").append(p.relativize(x)).append('\n');}Files.writeString(p.resolve("manifest.sha256"),out.toString());}
    private static Path source(Class<?> type){Path cwd=Path.of(System.getProperty("user.dir")).toAbsolutePath();Path a=cwd.resolve("src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/"+type.getSimpleName()+".java");if(Files.isRegularFile(a))return a;Path b=cwd.resolve("backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/"+type.getSimpleName()+".java");if(Files.isRegularFile(b))return b;throw new IllegalStateException("source not found");}
    private static String getClassName(){return CpuVectorMsePerformanceTest.class.getName();}
    private record Measurement(String order,long gb,long da,long db,long ga){long generated(){return gb+ga;}long direct(){return da+db;}boolean allAtLeast(long min){return gb>=min&&da>=min&&db>=min&&ga>=min;}double ratio(){return Math.sqrt(((double)gb/da)*((double)ga/db));}}
}
