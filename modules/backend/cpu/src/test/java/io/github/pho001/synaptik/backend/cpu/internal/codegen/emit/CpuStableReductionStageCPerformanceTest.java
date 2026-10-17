package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.*;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/**
 * Opt-in, append-only Stage-C measurement for the Stage-A-permitted CPU-0008O candidate two.
 * The hot bodies use typed carriers selected before timing; vector blocks are consumed lane zero
 * through lane {@code species - 1}, so this is not a reassociated reduction experiment.
 */
final class CpuStableReductionStageCPerformanceTest {
    private static final String ENABLE="SYNAPTIK_CPU_STABLE_REDUCTION_SPIKE", ROOT="SYNAPTIK_CPU_STABLE_REDUCTION_EVIDENCE_ROOT";
    private static final int FORKS=5, PAIRS=9, WARMUPS=5; private static final long MIN=25_000_000L, CAL=50_000_000L, CAP=720_000_000_000L;
    private static volatile double sink;
    enum Form { SOFTMAX, LOG_SOFTMAX, DENSE_CATEGORICAL, INDEX_CATEGORICAL, ATTENTION }
    enum Kind { F32, F64 }
    record Row(Form form, Kind kind, boolean tail) { String key(){return form+"-"+kind+"-"+(tail?"tail":"exact");} int formId(){return form.ordinal();} }
    interface Body { double run(); }

    public static void main(String[] a) throws Exception { if(a.length==3&&a[0].equals("--fork")) child(Path.of(a[2]),Integer.parseInt(a[1])); else parent(Path.of(root())); }
    @Test void stageCOptInFiveForkMeasurement() throws Exception { Assumptions.assumeTrue("true".equals(System.getenv(ENABLE))); parent(Path.of(root())); }
    @Test void stageCInventoryAndSmokeProtocol() throws Exception {
        assertEquals(20, rows().size()); assertEquals(0, rows().stream().filter(r->false).count());
        for(Row r:rows()) try(Arena a=Arena.ofConfined()){ Prepared p=prepare(r,a); assertEquals(bits(p.v.run()),bits(p.s.run())); assertEquals(bits(p.s.run()),bits(p.d.run())); }
    }
    private static String root(){String v=System.getenv(ROOT); if(v==null||v.isBlank())throw new IllegalArgumentException(ROOT+" explicit evidence root is required"); return v;}
    private static List<Row> rows(){var r=new ArrayList<Row>();for(Form f:Form.values())for(Kind k:Kind.values()){r.add(new Row(f,k,false));r.add(new Row(f,k,true));}return List.copyOf(r);}
    private static void parent(Path root) throws Exception {
        root=root.toAbsolutePath().normalize(); assertEquals(Path.of("/private/tmp/synaptik-cpu-0008o-stage-c-20260906"),root);
        if(Files.exists(root)) throw new IllegalStateException("fresh evidence root required: "+root); Files.createDirectories(root);
        List<Row> rows=rows(); String frozen="stage=CPU-0008O-C\ncandidate=2 VECTOR_MAP_ORDERED_FOLD\nindexed-bodies=0 STOP\nC3=STOP_MODEL_OR_ARCHITECTURE_DECISION\nC4=STOP_MODEL_OR_ARCHITECTURE_DECISION\nrows="+String.join(",",rows.stream().map(Row::key).toList())+"\ncarrier-map=F32 exact=AA tail=AS; F64 exact=SS tail=SA\nprotocol=forks=5,warmups=5,shared-calibration>=50ms,retained=9,randomized-symmetric-four-timings,min-side=25ms\nflags=-Xms1g -Xmx1g -XX:-TieredCompilation -Xbatch\nthresholds=V/S<=0.95,V/D<=1.15\nseed-rule=0x80008O+fork+row+comparison\nwall-cap-ns="+CAP+"\nsource-sha256="+sha(source(CpuStableReductionStageCPerformanceTest.class))+"\ndirect-source-sha256="+sha(source(CpuStableReductionStageCDirectOracle.class))+"\n";
        Files.writeString(root.resolve("frozen-manifest-inputs.txt"),frozen); Files.writeString(root.resolve("environment.txt"),env());
        long started=System.nanoTime();
        for(int f=0;f<FORKS;f++) { List<String> c=List.of(Path.of(System.getProperty("java.home"),"bin","java").toString(),"-Xms1g","-Xmx1g","-XX:-TieredCompilation","-Xbatch","--add-modules","jdk.incubator.vector","-cp",System.getProperty("java.class.path"),CpuStableReductionStageCPerformanceTest.class.getName(),"--fork",Integer.toString(f),root.toString()); Process q=new ProcessBuilder(c).redirectOutput(root.resolve("fork-"+f+".stdout").toFile()).redirectError(root.resolve("fork-"+f+".stderr").toFile()).start(); if(q.waitFor()!=0)throw new AssertionError("fork "+f+" failed"); if(System.nanoTime()-started>CAP)throw new AssertionError("ENVIRONMENT_LIMIT"); }
        auditAndAggregate(root,rows); Files.writeString(root.resolve("status.txt"),"COMPLETE append-only retained capture\n");
    }
    private static void child(Path root,int fork) throws Exception {
        StringBuilder raw=new StringBuilder("fork,row,comparison,pair,order,iterations,a_ns,b_ns,c_ns,d_ns,ratio,checksum\n");
        int ordinal=0; for(Row r:rows()) try(Arena a=Arena.ofConfined()) { Prepared p=prepare(r,a); assertEquals(bits(p.v.run()),bits(p.s.run()),r.key()); assertEquals(bits(p.s.run()),bits(p.d.run()),r.key()); for(int c=0;c<2;c++){Body v=p.v, other=c==0?p.s:p.d; Random random=new Random(0x80008L+fork*1000L+ordinal*10L+c); for(int w=0;w<WARMUPS;w++) symmetric(v,other,1,random.nextBoolean()); int n=calibrate(v,other); for(int pair=0;pair<PAIRS;pair++){boolean first=random.nextBoolean(); long[] t=symmetric(v,other,n,first); long va=first?t[0]:t[1], oa=first?t[1]:t[0], vb=first?t[2]:t[3], ob=first?t[3]:t[2]; assertTrue(va>=MIN&&vb>=MIN&&oa>=MIN&&ob>=MIN,"minimum side"); double ratio=((double)(va+vb))/(oa+ob); raw.append(fork).append(',').append(r.key()).append(',').append(c==0?"VS":"VD").append(',').append(pair).append(',').append(first?"V-O-O-V":"O-V-V-O").append(',').append(n).append(',').append(t[0]).append(',').append(t[1]).append(',').append(t[2]).append(',').append(t[3]).append(',').append(ratio).append(',').append(Double.doubleToRawLongBits(sink)).append('\n'); }} ordinal++; }
        Files.writeString(root.resolve("raw-fork-"+fork+".csv"),raw.toString());
    }
    private static int calibrate(Body a,Body b) { int n=1; for(;;){long x=time(a,n),y=time(b,n);if(x>=CAL&&y>=CAL)return n;if(n>1<<26)throw new AssertionError("calibration");n*=2;} }
    private static long[] symmetric(Body a,Body b,int n,boolean aFirst){return aFirst?new long[]{time(a,n),time(b,n),time(b,n),time(a,n)}:new long[]{time(b,n),time(a,n),time(a,n),time(b,n)};}
    private static long time(Body b,int n){long s=System.nanoTime();double x=0;for(int i=0;i<n;i++)x+=b.run();sink=x;return System.nanoTime()-s;}
    private static Prepared prepare(Row r,Arena a){int base=(r.kind==Kind.F32?FloatVector.SPECIES_PREFERRED.length():DoubleVector.SPECIES_PREFERRED.length());final int n=r.tail?2*base+3:base;final int f=r.formId(); if(r.kind==Kind.F32){float[] in=new float[n],out=new float[n];for(int i=0;i<n;i++)in[i]=(float)(i*.03125-1); if(r.tail){MemorySegment o=a.allocate(n*4L,8);return new Prepared(()->v32as(in,o,0,n,f),()->s32as(in,o,0,n,f),()->CpuStableReductionStageCDirectOracle.f32as(in,o,0,n,f));}return new Prepared(()->v32aa(in,out,0,n,f),()->s32aa(in,out,0,n,f),()->CpuStableReductionStageCDirectOracle.f32aa(in,out,0,n,f));} MemorySegment in=a.allocate(n*8L,8),out=a.allocate(n*8L,8);for(int i=0;i<n;i++)in.set(ValueLayout.JAVA_DOUBLE_UNALIGNED,i*8L,i*.03125-1); if(r.tail){double[] o=new double[n];return new Prepared(()->v64sa(in,o,0,n,f),()->s64sa(in,o,0,n,f),()->CpuStableReductionStageCDirectOracle.f64sa(in,o,0,n,f));}return new Prepared(()->v64ss(in,out,0,n,f),()->s64ss(in,out,0,n,f),()->CpuStableReductionStageCDirectOracle.f64ss(in,out,0,n,f)); }
    record Prepared(Body v,Body s,Body d) { }
    // V uses genuine preferred-species loads and increasing lane folds; S is its generated/test scalar body.
    private static float v32aa(float[] x,float[] o,long s,long e,int f){float m=-Float.MAX_VALUE;int l=FloatVector.SPECIES_PREFERRED.length();long p=s;for(;p+l<=e;p+=l){var v=FloatVector.fromArray(FloatVector.SPECIES_PREFERRED,x,(int)p).lanewise(VectorOperators.ADD,0f);for(int q=0;q<l;q++)m=Math.max(m,v.lane(q));}for(;p<e;p++)m=Math.max(m,x[(int)p]);float z=0;p=s;for(;p+l<=e;p+=l){var v=FloatVector.fromArray(FloatVector.SPECIES_PREFERRED,x,(int)p).lanewise(VectorOperators.ADD,0f);for(int q=0;q<l;q++)z+=(float)StrictMath.exp((double)(v.lane(q)-m));}for(;p<e;p++)z+=(float)StrictMath.exp((double)(x[(int)p]-m));return finish32(x,o,s,e,f,m,z);}
    private static float v32as(float[] x,MemorySegment o,long s,long e,int f){float m=-Float.MAX_VALUE;int l=FloatVector.SPECIES_PREFERRED.length();long p=s;for(;p+l<=e;p+=l){var v=FloatVector.fromArray(FloatVector.SPECIES_PREFERRED,x,(int)p).lanewise(VectorOperators.ADD,0f);for(int q=0;q<l;q++)m=Math.max(m,v.lane(q));}for(;p<e;p++)m=Math.max(m,x[(int)p]);float z=0;p=s;for(;p+l<=e;p+=l){var v=FloatVector.fromArray(FloatVector.SPECIES_PREFERRED,x,(int)p).lanewise(VectorOperators.ADD,0f);for(int q=0;q<l;q++)z+=(float)StrictMath.exp((double)(v.lane(q)-m));}for(;p<e;p++)z+=(float)StrictMath.exp((double)(x[(int)p]-m));float v=f==1?x[(int)s]-m-(float)StrictMath.log((double)z):f>=2?m+(float)StrictMath.log((double)z)-x[(int)Math.min(s+1,e-1)]:z;if(f==0)for(long i=s;i<e;i++)o.set(ValueLayout.JAVA_FLOAT_UNALIGNED,i*4,(float)StrictMath.exp((double)(x[(int)i]-m))/z);else o.set(ValueLayout.JAVA_FLOAT_UNALIGNED,s*4,v);return v;}
    private static float s32aa(float[] x,float[] o,long s,long e,int f){float m=-Float.MAX_VALUE;for(long i=s;i<e;i++)m=Math.max(m,x[(int)i]);float z=0;for(long i=s;i<e;i++)z+=(float)StrictMath.exp((double)(x[(int)i]-m));return finish32(x,o,s,e,f,m,z);} private static float s32as(float[] x,MemorySegment o,long s,long e,int f){float m=-Float.MAX_VALUE;for(long i=s;i<e;i++)m=Math.max(m,x[(int)i]);float z=0;for(long i=s;i<e;i++)z+=(float)StrictMath.exp((double)(x[(int)i]-m));float v=f==1?x[(int)s]-m-(float)StrictMath.log((double)z):f>=2?m+(float)StrictMath.log((double)z)-x[(int)Math.min(s+1,e-1)]:z;if(f==0)for(long i=s;i<e;i++)o.set(ValueLayout.JAVA_FLOAT_UNALIGNED,i*4,(float)StrictMath.exp((double)(x[(int)i]-m))/z);else o.set(ValueLayout.JAVA_FLOAT_UNALIGNED,s*4,v);return v;}
    private static float finish32(float[]x,float[]o,long s,long e,int f,float m,float z){float v=f==1?x[(int)s]-m-(float)StrictMath.log((double)z):f>=2?m+(float)StrictMath.log((double)z)-x[(int)Math.min(s+1,e-1)]:z;if(f==0)for(long i=s;i<e;i++)o[(int)i]=(float)StrictMath.exp((double)(x[(int)i]-m))/z;else o[(int)s]=v;return v;}
    private static double v64ss(MemorySegment x,MemorySegment o,long s,long e,int f){double m=-Double.MAX_VALUE;int l=DoubleVector.SPECIES_PREFERRED.length();long p=s;for(;p+l<=e;p+=l){var v=DoubleVector.fromMemorySegment(DoubleVector.SPECIES_PREFERRED,x,p*8,java.nio.ByteOrder.nativeOrder()).lanewise(VectorOperators.ADD,0d);for(int q=0;q<l;q++)m=Math.max(m,v.lane(q));}for(;p<e;p++)m=Math.max(m,x.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,p*8));double z=0;p=s;for(;p+l<=e;p+=l){var v=DoubleVector.fromMemorySegment(DoubleVector.SPECIES_PREFERRED,x,p*8,java.nio.ByteOrder.nativeOrder()).lanewise(VectorOperators.ADD,0d);for(int q=0;q<l;q++)z+=StrictMath.exp(v.lane(q)-m);}for(;p<e;p++)z+=StrictMath.exp(x.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,p*8)-m);return finish64(x,o,s,e,f,m,z);}
    private static double v64sa(MemorySegment x,double[] o,long s,long e,int f){double m=-Double.MAX_VALUE;int l=DoubleVector.SPECIES_PREFERRED.length();long p=s;for(;p+l<=e;p+=l){var v=DoubleVector.fromMemorySegment(DoubleVector.SPECIES_PREFERRED,x,p*8,java.nio.ByteOrder.nativeOrder()).lanewise(VectorOperators.ADD,0d);for(int q=0;q<l;q++)m=Math.max(m,v.lane(q));}for(;p<e;p++)m=Math.max(m,x.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,p*8));double z=0;p=s;for(;p+l<=e;p+=l){var v=DoubleVector.fromMemorySegment(DoubleVector.SPECIES_PREFERRED,x,p*8,java.nio.ByteOrder.nativeOrder()).lanewise(VectorOperators.ADD,0d);for(int q=0;q<l;q++)z+=StrictMath.exp(v.lane(q)-m);}for(;p<e;p++)z+=StrictMath.exp(x.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,p*8)-m);double v=f==1?x.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,s*8)-m-StrictMath.log(z):f>=2?m+StrictMath.log(z)-x.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,Math.min(s+1,e-1)*8):z;if(f==0)for(long i=s;i<e;i++)o[(int)i]=StrictMath.exp(x.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,i*8)-m)/z;else o[(int)s]=v;return v;}
    private static double s64ss(MemorySegment x,MemorySegment o,long s,long e,int f){double m=-Double.MAX_VALUE;for(long i=s;i<e;i++)m=Math.max(m,x.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,i*8));double z=0;for(long i=s;i<e;i++)z+=StrictMath.exp(x.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,i*8)-m);return finish64(x,o,s,e,f,m,z);} private static double s64sa(MemorySegment x,double[] o,long s,long e,int f){double m=-Double.MAX_VALUE;for(long i=s;i<e;i++)m=Math.max(m,x.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,i*8));double z=0;for(long i=s;i<e;i++)z+=StrictMath.exp(x.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,i*8)-m);double v=f==1?x.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,s*8)-m-StrictMath.log(z):f>=2?m+StrictMath.log(z)-x.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,Math.min(s+1,e-1)*8):z;if(f==0)for(long i=s;i<e;i++)o[(int)i]=StrictMath.exp(x.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,i*8)-m)/z;else o[(int)s]=v;return v;}
    private static double finish64(MemorySegment x,MemorySegment o,long s,long e,int f,double m,double z){double v=f==1?x.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,s*8)-m-StrictMath.log(z):f>=2?m+StrictMath.log(z)-x.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,Math.min(s+1,e-1)*8):z;if(f==0)for(long i=s;i<e;i++)o.set(ValueLayout.JAVA_DOUBLE_UNALIGNED,i*8,StrictMath.exp(x.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,i*8)-m)/z);else o.set(ValueLayout.JAVA_DOUBLE_UNALIGNED,s*8,v);return v;}
    /**
     * Evaluates all retained raw records for a complete five-fork capture.
     *
     * <p>Eligibility requires every pair, every per-fork median, and the median of the five fork
     * medians to satisfy its comparison threshold. A miss is recorded as {@code KEEP_SCALAR}; it
     * is not a harness failure or a reason to weaken the protocol.</p>
     *
     * @param root evidence directory containing one complete CSV for each required fork
     * @param rows fixed Stage-C row inventory whose records are evaluated
     * @throws Exception if the capture is structurally incomplete or cannot be read
     */
    private static void auditAndAggregate(Path root,List<Row> rows) throws Exception {
        List<String> lines = new ArrayList<>();
        long sides = 0;
        long passingPairs = 0;
        long failingPairs = 0;
        long passingForkMedians = 0;
        long failingForkMedians = 0;
        long passingAggregates = 0;
        long failingAggregates = 0;
        for (Row row : rows) for (String comparison : List.of("VS", "VD")) {
            double threshold = comparison.equals("VS") ? .95 : 1.15;
            double[] forkMedians = new double[FORKS];
            boolean allPairsPass = true;
            boolean allForkMediansPass = true;
            for (int fork = 0; fork < FORKS; fork++) {
                var records = Files.readAllLines(root.resolve("raw-fork-" + fork + ".csv")).stream()
                        .skip(1).filter(line -> line.contains("," + row.key() + "," + comparison + ",")).toList();
                assertEquals(PAIRS, records.size(), row.key() + comparison + fork);
                double[] ratios = new double[PAIRS];
                for (int pair = 0; pair < PAIRS; pair++) {
                    String[] fields = records.get(pair).split(",");
                    ratios[pair] = Double.parseDouble(fields[10]);
                    boolean pairPass = ratios[pair] <= threshold;
                    allPairsPass &= pairPass;
                    if (pairPass) passingPairs++; else failingPairs++;
                    for (int timing = 6; timing < 10; timing++) {
                        assertTrue(Long.parseLong(fields[timing]) >= MIN, "minimum side");
                        sides++;
                    }
                }
                Arrays.sort(ratios);
                forkMedians[fork] = ratios[PAIRS / 2];
                boolean medianPass = forkMedians[fork] <= threshold;
                allForkMediansPass &= medianPass;
                if (medianPass) passingForkMedians++; else failingForkMedians++;
            }
            Arrays.sort(forkMedians);
            double aggregate = forkMedians[FORKS / 2];
            boolean aggregatePass = aggregate <= threshold;
            if (aggregatePass) passingAggregates++; else failingAggregates++;
            boolean keepScalar = !(allPairsPass && allForkMediansPass && aggregatePass);
            lines.add(row.key() + "," + comparison + ",pairs-pass=" + allPairsPass
                    + ",fork-medians-pass=" + allForkMediansPass + ",aggregate=" + aggregate
                    + ",aggregate-pass=" + aggregatePass + ",decision="
                    + (keepScalar ? "KEEP_SCALAR" : "ELIGIBLE_FOR_SEPARATE_PRODUCTION_TASK"));
        }
        assertEquals(7200, sides);
        assertEquals(40, lines.size());
        Files.write(root.resolve("aggregates.csv"), lines);
        Files.writeString(root.resolve("audit.txt"), "row-fork-executions=100\nretained-pairs=1800\n"
                + "timed-sides=7200\nfork-medians=200\naggregates=40\npairs-passing=" + passingPairs
                + "\npairs-failing=" + failingPairs + "\nfork-medians-passing=" + passingForkMedians
                + "\nfork-medians-failing=" + failingForkMedians + "\naggregates-passing="
                + passingAggregates + "\naggregates-failing=" + failingAggregates
                + "\nordering=fork,row,comparison,pair\nC3=C4=STOP_MODEL_OR_ARCHITECTURE_DECISION\n");
    }
    private static long bits(double d){return Double.doubleToRawLongBits(d);} private static String source(Class<?> c)throws Exception{String n=c.getSimpleName()+".class";try(InputStream i=c.getResourceAsStream(n)){return java.util.HexFormat.of().formatHex(i.readAllBytes());}} private static String sha(String s)throws Exception{return java.util.HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(s.getBytes(StandardCharsets.UTF_8)));} private static String env(){return "java="+System.getProperty("java.version")+"\nos="+System.getProperty("os.name")+"\narch="+System.getProperty("os.arch")+"\nF32species="+FloatVector.SPECIES_PREFERRED.length()+"\nF64species="+DoubleVector.SPECIES_PREFERRED.length()+"\naffinity/governor=unavailable\n";}
}
