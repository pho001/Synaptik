package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.*;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.ir.*;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.ClassFile;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.file.*;
import java.security.MessageDigest;
import java.time.Instant;
import java.util.*;

/** Fixed ten-row, five-fork generated/direct MATMUL evidence harness for CPU 0008F. */
public final class CpuMatmulPerformanceTest {
    private static final Path ROOT=Path.of(System.getProperty("synaptik.cpu.matmul.evidenceRoot",
            "/private/tmp/synaptik-cpu-0008f-retained-evidence-attempt-11-20260828"));
    private static final long MIN_NANOS=25_000_000L;
    private static volatile long sink;
    @FunctionalInterface private interface Action { long run() throws Throwable; }
    private record Case(String name,Action generated,Action direct,Runnable verify) { }
    private CpuMatmulPerformanceTest() { }

    /** Runs one immutable measured fork, or aggregates exactly forks zero through four. */
    public static void main(String[] args)throws Throwable {
        if(args.length>0&&args[0].equals("aggregate")){aggregate();return;}
        verifyRangePolymorphism();
        if(args.length>0&&args[0].equals("verify")){for(Case value:cases())gate(value);
            System.out.println("VERIFY,rows=10,materializations=2,status=PASS");return;}
        int fork=args.length==0?0:Integer.parseInt(args[0]);
        if(fork<0||fork>=5)throw new IllegalArgumentException("fork must be zero through four");
        StringBuilder report=new StringBuilder();environment(report);
        try {
            List<Case>cases=cases();int[]reps=new int[cases.size()*2];
            for(int i=0;i<cases.size();i++){gate(cases.get(i));reps[2*i]=repetitions(cases.get(i).generated);
                reps[2*i+1]=repetitions(cases.get(i).direct);}
            long[][]g=new long[cases.size()][9],d=new long[cases.size()][9];
            Random random=new Random(0x0008f_20260828L^fork*0x9e3779b97f4a7c15L);
            for(int round=-5;round<9;round++){List<Integer>order=new ArrayList<>();
                for(int i=0;i<cases.size();i++)order.add(i);Collections.shuffle(order,random);
                for(int i:order){long gt,dt;if(random.nextBoolean()){gt=time(cases.get(i).generated,reps[2*i]);dt=time(cases.get(i).direct,reps[2*i+1]);}
                    else{dt=time(cases.get(i).direct,reps[2*i+1]);gt=time(cases.get(i).generated,reps[2*i]);}
                    if(round>=0){g[i][round]=gt/reps[2*i];d[i][round]=dt/reps[2*i+1];gate(cases.get(i));}}}
            int failures=0;for(int i=0;i<cases.size();i++){long gm=median(g[i]),dm=median(d[i]);double ratio=(double)gm/dm;
                if(ratio>1.15)failures++;String kind=cases.get(i).name.startsWith("MATERIALIZE-")
                        ?"MATERIALIZATION":"RESULT";report.append(String.format(Locale.ROOT,kind+",%s,%d,%d,%.9f,%s,%s%n",
                    cases.get(i).name,gm,dm,ratio,Arrays.toString(g[i]),Arrays.toString(d[i])));}
            report.append("SINK,").append(sink).append('\n');retain(fork,report.toString());System.out.print(report);
            if(failures!=0)throw new AssertionError("ratio failures "+failures);
        }catch(Throwable failure){report.append("MEASURED_FAILURE,").append(failure.getClass().getName()).append(',')
                .append(clean(failure.getMessage())).append('\n');retain(fork,report.toString());System.out.print(report);throw failure;}
    }

    private static List<Case> cases()throws Throwable {
        var result=new ArrayList<Case>(List.of(f64("F64-VECTOR-VECTOR-K257",1,257,1,1,CpuMatmulIr.Realization.DIRECT_SCALAR,false,false),
            f32("F32-MATRIX-VECTOR-64X127",64,127,1,1,CpuMatmulIr.Realization.DIRECT_SCALAR,false,false),
            f64("F64-DIRECT-N-VECTOR",2,63,128,1,CpuMatmulIr.Realization.DIRECT_N_VECTOR,false,false),
            i32("I32-DIRECT-N-VECTOR",2,65,96,CpuMatmulIr.Realization.DIRECT_N_VECTOR),
            bf16("BF16-SCALAR-TILED",32,63,48,false),mixed("BF16-F32-SCALAR-TILED",32,63,48),
            f32("F32-N-VECTOR-TILED",32,127,256,1,CpuMatmulIr.Realization.TILED_N_VECTOR_2X2,false,false),
            f64("F64-BATCH-BROADCAST-N-VECTOR-TILED",16,127,256,3,CpuMatmulIr.Realization.TILED_N_VECTOR_2X2,false,false),
            f32("F32-BIAS-FUSED-SCALAR",32,63,48,1,CpuMatmulIr.Realization.DIRECT_SCALAR,true,false),
            f32("F32-TERMINAL-SPLIT-CONTROL",32,63,48,1,CpuMatmulIr.Realization.DIRECT_SCALAR,false,true)));
        result.add(material32());result.add(material64());return List.copyOf(result);
    }

    private static Case material32()throws Throwable{int batch=1,m=32,k=127,n=256;float[]a=new float[m*k],source=new float[n*k],dense=new float[k*n],go=new float[m*n],direct=new float[m*n];fill(a);fill(source);Generated gen=generated("MATERIALIZE-RIGHT-F32",DataType.FLOAT32,DataType.FLOAT32,DataType.FLOAT32,CpuMatmulIr.Realization.TILED_N_VECTOR_2X2,false,batch,m,k,n,List.of());MethodHandle h=bound(gen,a,dense,go);long[]geometry={0,0,0,batch,m,k,n,k,1,1,k,n,1,0,0,0,(long)batch*m*n,1};long work=(long)batch*((m+1)/2)*((n+1)/2);MethodHandle directHandle=boundDirect("directTiledScalarF32",methodType(float[].class,float[].class,float[].class),geometry,a,source,direct);Action ga=()->{for(int p=0;p<k;p++)for(int j=0;j<n;j++)dense[p*n+j]=source[j*k+p];h.invokeExact(0L,gen.work);return checksum(go);};Action da=()->{directHandle.invokeExact(0L,work);return checksum(direct);};return new Case("MATERIALIZE-RIGHT-F32",ga,da,()->{if(!Arrays.equals(go,direct))throw new AssertionError("MATERIALIZE-RIGHT-F32");});}
    private static Case material64()throws Throwable{int batch=3,m=16,k=127,n=256;double[]a=new double[batch*m*k],source=new double[n*k],dense=new double[k*n],go=new double[batch*m*n],direct=new double[batch*m*n];fill(a);fill(source);Generated gen=generated("MATERIALIZE-RIGHT-F64-BATCH",DataType.FLOAT64,DataType.FLOAT64,DataType.FLOAT64,CpuMatmulIr.Realization.TILED_N_VECTOR_2X2,false,batch,m,k,n,List.of());MethodHandle h=bound(gen,a,dense,go);long[]geometry={0,0,0,batch,m,k,n,k,1,1,k,n,1,1,0,0,(long)batch*m*n,1,batch,(long)m*k,0,(long)m*n};long work=(long)batch*((m+1)/2)*((n+1)/2);MethodHandle directHandle=boundDirect("directTiledScalarF64",methodType(double[].class,double[].class,double[].class),geometry,a,source,direct);Action ga=()->{for(int p=0;p<k;p++)for(int j=0;j<n;j++)dense[p*n+j]=source[j*k+p];h.invokeExact(0L,gen.work);return checksum(go);};Action da=()->{directHandle.invokeExact(0L,work);return checksum(direct);};return new Case("MATERIALIZE-RIGHT-F64-BATCH",ga,da,()->{if(!Arrays.equals(go,direct))throw new AssertionError("MATERIALIZE-RIGHT-F64-BATCH");});}

    private static Case f32(String name,int m,int k,int n,int batch,CpuMatmulIr.Realization form,
            boolean bias,boolean splitRelu)throws Throwable{
        int cells=Math.multiplyExact(Math.multiplyExact(batch,m),n);float[]a=new float[batch*m*k],b=new float[k*n],x=bias?new float[n]:null;
        float[]go=new float[cells],direct=new float[cells];fill(a);fill(b);if(x!=null)fill(x);
        Generated gen=generated(name,DataType.FLOAT32,DataType.FLOAT32,DataType.FLOAT32,form,bias,
                batch,m,k,n,List.of(CarrierAccess.FLOAT_ARRAY,CarrierAccess.FLOAT_ARRAY,
                bias?CarrierAccess.FLOAT_ARRAY:CarrierAccess.FLOAT_ARRAY));
        MethodHandle h=bias?bound(gen,a,b,x,go):bound(gen,a,b,go);
        MethodHandle directHandle=bias
                ?boundDirect("directScalarBiasF32",methodType(float[].class,float[].class,
                    float[].class,float[].class),gen.geometry,a,b,x,direct)
                :boundDirect(form==CpuMatmulIr.Realization.TILED_N_VECTOR_2X2
                    ?"directTiledNVectorF32":form==CpuMatmulIr.Realization.DIRECT_N_VECTOR
                    ?"directNVectorF32":"directScalarF32",methodType(float[].class,float[].class,
                    float[].class),gen.geometry,a,b,direct);
        Action ga=()->{h.invokeExact(0L,gen.work);if(splitRelu)relu(go);return checksum(go);};
        Action da=()->{directHandle.invokeExact(0L,gen.work);if(splitRelu)relu(direct);return checksum(direct);};
        return new Case(name,ga,da,()->{if(!Arrays.equals(go,direct))throw new AssertionError(name);});
    }
    private static Case f64(String name,int m,int k,int n,int batch,CpuMatmulIr.Realization form,
            boolean bias,boolean splitRelu)throws Throwable{
        int cells=batch*m*n;double[]a=new double[batch*m*k],b=new double[k*n],x=bias?new double[n]:null,go=new double[cells],direct=new double[cells];fill(a);fill(b);if(x!=null)fill(x);
        Generated gen=generated(name,DataType.FLOAT64,DataType.FLOAT64,DataType.FLOAT64,form,bias,batch,m,k,n,List.of());
        MethodHandle h=bias?bound(gen,a,b,x,go):bound(gen,a,b,go);
        MethodHandle directHandle=bias
                ?boundDirect("directScalarBiasF64",methodType(double[].class,double[].class,
                    double[].class,double[].class),gen.geometry,a,b,x,direct)
                :boundDirect(form==CpuMatmulIr.Realization.TILED_N_VECTOR_2X2
                    ?"directTiledNVectorF64":form==CpuMatmulIr.Realization.DIRECT_N_VECTOR
                    ?"directNVectorF64":"directScalarF64",methodType(double[].class,double[].class,
                    double[].class),gen.geometry,a,b,direct);
        Action ga=()->{h.invokeExact(0L,gen.work);if(splitRelu)relu(go);return checksum(go);};
        Action da=()->{directHandle.invokeExact(0L,gen.work);if(splitRelu)relu(direct);return checksum(direct);};
        return new Case(name,ga,da,()->{if(!Arrays.equals(go,direct))throw new AssertionError(name);});
    }
    private static Case i32(String name,int m,int k,int n,CpuMatmulIr.Realization form)throws Throwable{
        int[]a=new int[m*k],b=new int[k*n],go=new int[m*n],direct=new int[m*n];fill(a);fill(b);
        Generated gen=generated(name,DataType.INT32,DataType.INT32,DataType.INT32,form,false,1,m,k,n,List.of());
        MethodHandle h=bound(gen,a,b,go);
        MethodHandle directHandle=boundDirect("directNVectorI32",
                methodType(int[].class,int[].class,int[].class),gen.geometry,a,b,direct);
        Action ga=()->{h.invokeExact(0L,gen.work);return checksum(go);};
        Action da=()->{directHandle.invokeExact(0L,gen.work);return checksum(direct);};
        return new Case(name,ga,da,()->{if(!Arrays.equals(go,direct))throw new AssertionError(name);});
    }
    private static Case bf16(String name,int m,int k,int n,boolean ignored)throws Throwable{
        short[]a=new short[m*k],b=new short[k*n],go=new short[m*n],direct=new short[m*n];fill(a);fill(b);
        Generated gen=generated(name,DataType.BFLOAT16,DataType.BFLOAT16,DataType.BFLOAT16,CpuMatmulIr.Realization.TILED_SCALAR_2X2,false,1,m,k,n,List.of());
        MethodHandle h=bound(gen,a,b,go);
        MethodHandle directHandle=boundDirect("directTiledBf16",
                methodType(short[].class,short[].class,short[].class),gen.geometry,a,b,direct);
        Action ga=()->{h.invokeExact(0L,gen.work);return checksum(go);};
        Action da=()->{directHandle.invokeExact(0L,gen.work);return checksum(direct);};
        return new Case(name,ga,da,()->{if(!Arrays.equals(go,direct))throw new AssertionError(name);});
    }
    private static Case mixed(String name,int m,int k,int n)throws Throwable{
        short[]a=new short[m*k];float[]b=new float[k*n],go=new float[m*n],direct=new float[m*n];fill(a);fill(b);
        Generated gen=generated(name,DataType.BFLOAT16,DataType.FLOAT32,DataType.FLOAT32,CpuMatmulIr.Realization.TILED_SCALAR_2X2,false,1,m,k,n,List.of());
        MethodHandle h=bound(gen,a,b,go);
        MethodHandle directHandle=boundDirect("directTiledBf16F32",
                methodType(short[].class,float[].class,float[].class),gen.geometry,a,b,direct);
        Action ga=()->{h.invokeExact(0L,gen.work);return checksum(go);};
        Action da=()->{directHandle.invokeExact(0L,gen.work);return checksum(direct);};
        return new Case(name,ga,da,()->{if(!Arrays.equals(go,direct))throw new AssertionError(name);});
    }

    private record Generated(MethodHandle handle,long[]geometry,long work){}
    private static MethodHandle bound(Generated generated,Object...carriers) {
        MethodHandle handle=MethodHandles.insertArguments(generated.handle,0,carriers);
        return MethodHandles.insertArguments(handle,0,(Object)generated.geometry);
    }
    private static MethodType methodType(Class<?>...carriers) {
        Class<?>[] parameters=Arrays.copyOf(carriers,carriers.length+3);
        parameters[carriers.length]=long[].class;parameters[carriers.length+1]=long.class;
        parameters[carriers.length+2]=long.class;
        return MethodType.methodType(void.class,parameters);
    }
    private static MethodHandle boundDirect(String name,MethodType type,long[]geometry,
            Object...carriers)throws ReflectiveOperationException {
        MethodHandle handle=MethodHandles.lookup().findStatic(CpuMatmulPerformanceTest.class,name,type);
        handle=MethodHandles.insertArguments(handle,0,carriers);
        return MethodHandles.insertArguments(handle,0,(Object)geometry);
    }
    private static void verifyRangePolymorphism()throws Throwable {
        int width=jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.length();
        int batch=6,m=3,k=5,n=2*width+3;float[]left=new float[3*m*k],right=new float[k*n];
        fill(left);fill(right);float[]expected=new float[batch*m*n];
        for(int q=0;q<batch;q++)for(int row=0;row<m;row++)for(int column=0;column<n;column++){
            float sum=0;for(int inner=0;inner<k;inner++)sum+=left[((q%3)*m+row)*k+inner]
                    *right[inner*n+column];expected[(q*m+row)*n+column]=sum;}
        long[]geometry={0,0,0,batch,m,k,n,k,1,n,1,n,1,2,0,0,(long)batch*m*n,1,
                2,3,0,(long)m*k,0,0,(long)3*m*n,(long)m*n};
        verifyRangeMethod("directScalarF32",geometry,(long)batch*m*n,left,right,expected);
        verifyRangeMethod("directNVectorF32",geometry,(long)batch*m,left,right,expected);
        verifyRangeMethod("directTiledScalarF32",geometry,
                (long)batch*((m+1)/2)*((n+1)/2),left,right,expected);
        verifyRangeMethod("directTiledNVectorF32",geometry,
                (long)batch*((m+1)/2)*((n+2L*width-1)/(2L*width)),left,right,expected);
        long[]zero=geometry.clone();zero[3]=0;zero[4]=0;zero[16]=0;
        float[]empty={37};MethodHandle emptyHandle=boundDirect("directScalarF32",
                methodType(float[].class,float[].class,float[].class),zero,left,right,empty);
        emptyHandle.invokeExact(0L,0L);if(empty[0]!=37)throw new AssertionError("empty MATMUL range wrote output");
    }
    private static void verifyRangeMethod(String name,long[]geometry,long work,float[]left,
            float[]right,float[]expected)throws Throwable {
        float[]output=new float[expected.length];Arrays.fill(output,Float.NaN);
        MethodHandle handle=boundDirect(name,methodType(float[].class,float[].class,float[].class),
                geometry,left,right,output);handle.invokeExact(0L,0L);
        for(float value:output)if(!Float.isNaN(value))throw new AssertionError(name+" empty range");
        long split=work/2;handle.invokeExact(split,work);handle.invokeExact(0L,split);
        if(!Arrays.equals(expected,output))throw new AssertionError(name+" split range");
    }
    private static Generated generated(String name,DataType lt,DataType rt,DataType out,
            CpuMatmulIr.Realization form,boolean bias,int batch,int m,int k,int n,List<CarrierAccess>unused)throws Exception{
        var read=plan(CpuAccessPlan.AccessKind.READ);var write=plan(CpuAccessPlan.AccessKind.WRITE);
        var biasPlan=new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,CpuAccessPlan.Regime.LAST_AXIS_BIAS,1,List.of(CpuAccessPlan.AxisRole.CONTIGUOUS),1);
        var epi=bias?new CpuMatmulIr.Epilogue(CpuMatmulIr.Epilogue.AddInputOrder.MATMUL_LEFT,CpuMatmulIr.Epilogue.Terminal.NONE,null):CpuMatmulIr.Epilogue.none();
        int bits=(form==CpuMatmulIr.Realization.DIRECT_N_VECTOR||form==CpuMatmulIr.Realization.TILED_N_VECTOR_2X2)?species(out):0;
        var ir=new CpuMatmulIr(lt,rt,out,form,epi,bits,CpuMatmulIr.NumericalForm.SEQUENTIAL,bias?List.of(read,read,biasPlan):List.of(read,read),write);
        List<DataType>types=bias?List.of(lt,rt,out,out):List.of(lt,rt,out);List<CarrierAccess>carriers=types.stream().map(CpuMatmulPerformanceTest::carrier).toList();
        var spec=new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(ir.structuralKey()),CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                bits==0?CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR:CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR,
                types,carriers,bits,-1,List.of(),false,54,Optional.of(ir));
        var generator=new CpuClassFileKernelGenerator();byte[]bytes=generator.generateClassBytes(spec,ir.encodedKernelIr());
        retainGenerated(name,bytes,spec,ir);MethodHandle handle=generator.defineClassBytes(spec,bytes).entryPoint();
        long[]g=batch==1
                ?new long[]{0,0,0,batch,m,k,n,k,1,n,1,n,1,0,0,bias?1:0,(long)batch*m*n,1}
                :new long[]{0,0,0,batch,m,k,n,k,1,n,1,n,1,1,0,bias?1:0,
                    (long)batch*m*n,1,batch,(long)m*k,0,(long)m*n};
        long work=switch(form){case DIRECT_SCALAR->(long)batch*m*n;case DIRECT_N_VECTOR->(long)batch*m;
            case TILED_SCALAR_2X2->(long)batch*((m+1)/2)*((n+1)/2);case TILED_N_VECTOR_2X2->(long)batch*((m+1)/2)*((n+2L*lanes(out)-1)/(2L*lanes(out)));};
        return new Generated(handle,g,work);
    }

    private static void directScalarF32(float[]a,float[]b,float[]o,long[]g,long start,long end){
        int m=(int)g[4],k=(int)g[5],n=(int)g[6],lm=(int)g[7],lk=(int)g[8];
        int rk=(int)g[9],rn=(int)g[10],om=(int)g[11],on=(int)g[12],rank=(int)g[13];
        for(long ordinal=start;ordinal<end;ordinal++){long remaining=ordinal;int column=(int)(remaining%n);
            remaining/=n;int row=(int)(remaining%m);long batch=remaining/m,lb=g[0],rb=g[1],ob=g[2];
            for(int axis=rank-1;axis>=0;axis--){long extent=g[18+axis],coordinate=batch%extent;batch/=extent;
                lb+=coordinate*g[18+rank+axis];rb+=coordinate*g[18+2*rank+axis];ob+=coordinate*g[18+3*rank+axis];}
            int la=(int)(lb+(long)row*lm),ra=(int)(rb+(long)column*rn);float sum=0;
            for(int inner=0;inner<k;inner++){sum+=a[la]*b[ra];la+=lk;ra+=rk;}
            o[(int)(ob+(long)row*om+(long)column*on)]=sum;}}

    private static void directScalarBiasF32(float[]a,float[]b,float[]bias,float[]o,long[]g,
            long start,long end){
        int m=(int)g[4],k=(int)g[5],n=(int)g[6],lm=(int)g[7],lk=(int)g[8];
        int rk=(int)g[9],rn=(int)g[10],om=(int)g[11],on=(int)g[12],rank=(int)g[13];
        int biasBase=(int)g[14],biasStride=(int)g[15];
        for(long ordinal=start;ordinal<end;ordinal++){long remaining=ordinal;int column=(int)(remaining%n);
            remaining/=n;int row=(int)(remaining%m);long batchOrdinal=remaining/m,lb=g[0],rb=g[1],ob=g[2];
            for(int axis=rank-1;axis>=0;axis--){long extent=g[18+axis],coordinate=batchOrdinal%extent;
                batchOrdinal/=extent;lb+=coordinate*g[18+rank+axis];rb+=coordinate*g[18+2*rank+axis];
                ob+=coordinate*g[18+3*rank+axis];}
            int la=(int)(lb+(long)row*lm),ra=(int)(rb+(long)column*rn);float sum=0;
            for(int inner=0;inner<k;inner++){sum+=a[la]*b[ra];la+=lk;ra+=rk;}
            o[(int)(ob+(long)row*om+(long)column*on)]=sum+bias[biasBase+column*biasStride];}}

    private static void directScalarF64(double[]a,double[]b,double[]o,long[]g,long start,long end){
        int m=(int)g[4],k=(int)g[5],n=(int)g[6],lm=(int)g[7],lk=(int)g[8];
        int rk=(int)g[9],rn=(int)g[10],om=(int)g[11],on=(int)g[12],rank=(int)g[13];
        for(long ordinal=start;ordinal<end;ordinal++){long remaining=ordinal;int column=(int)(remaining%n);
            remaining/=n;int row=(int)(remaining%m);long batchOrdinal=remaining/m,lb=g[0],rb=g[1],ob=g[2];
            for(int axis=rank-1;axis>=0;axis--){long extent=g[18+axis],coordinate=batchOrdinal%extent;
                batchOrdinal/=extent;lb+=coordinate*g[18+rank+axis];rb+=coordinate*g[18+2*rank+axis];
                ob+=coordinate*g[18+3*rank+axis];}
            int la=(int)(lb+(long)row*lm),ra=(int)(rb+(long)column*rn);double sum=0;
            for(int inner=0;inner<k;inner++){sum+=a[la]*b[ra];la+=lk;ra+=rk;}
            o[(int)(ob+(long)row*om+(long)column*on)]=sum;}}

    private static void directScalarBiasF64(double[]a,double[]b,double[]bias,double[]o,long[]g,
            long start,long end){
        int m=(int)g[4],k=(int)g[5],n=(int)g[6],lm=(int)g[7],lk=(int)g[8];
        int rk=(int)g[9],rn=(int)g[10],om=(int)g[11],on=(int)g[12],rank=(int)g[13];
        int biasBase=(int)g[14],biasStride=(int)g[15];
        for(long ordinal=start;ordinal<end;ordinal++){long remaining=ordinal;int column=(int)(remaining%n);
            remaining/=n;int row=(int)(remaining%m);long batchOrdinal=remaining/m,lb=g[0],rb=g[1],ob=g[2];
            for(int axis=rank-1;axis>=0;axis--){long extent=g[18+axis],coordinate=batchOrdinal%extent;
                batchOrdinal/=extent;lb+=coordinate*g[18+rank+axis];rb+=coordinate*g[18+2*rank+axis];
                ob+=coordinate*g[18+3*rank+axis];}
            int la=(int)(lb+(long)row*lm),ra=(int)(rb+(long)column*rn);double sum=0;
            for(int inner=0;inner<k;inner++){sum+=a[la]*b[ra];la+=lk;ra+=rk;}
            o[(int)(ob+(long)row*om+(long)column*on)]=sum+bias[biasBase+column*biasStride];}}

    private static void directNVectorF32(float[]a,float[]b,float[]o,long[]g,long start,long end){
        var species=jdk.incubator.vector.FloatVector.SPECIES_PREFERRED;int width=species.length();
        int m=(int)g[4],k=(int)g[5],n=(int)g[6],lm=(int)g[7],lk=(int)g[8];
        int rk=(int)g[9],rn=(int)g[10],om=(int)g[11],on=(int)g[12],rank=(int)g[13];
        for(long ordinal=start;ordinal<end;ordinal++){int row=(int)(ordinal%m);long batchOrdinal=ordinal/m;
            long lb=g[0],rb=g[1],ob=g[2];for(int axis=rank-1;axis>=0;axis--){long extent=g[18+axis];
                long coordinate=batchOrdinal%extent;batchOrdinal/=extent;lb+=coordinate*g[18+rank+axis];
                rb+=coordinate*g[18+2*rank+axis];ob+=coordinate*g[18+3*rank+axis];}
            int leftRow=(int)(lb+(long)row*lm),outputRow=(int)(ob+(long)row*om),column=0;
            for(;column+width<=n;column+=width){var sum=jdk.incubator.vector.FloatVector.zero(species);
                int la=leftRow,ra=(int)(rb+(long)column*rn);for(int inner=0;inner<k;inner++){
                    sum=sum.add(jdk.incubator.vector.FloatVector.fromArray(species,b,ra).mul(a[la]));la+=lk;ra+=rk;}
                sum.intoArray(o,outputRow+column*on);}
            for(;column<n;column++){float sum=0;int la=leftRow,ra=(int)(rb+(long)column*rn);
                for(int inner=0;inner<k;inner++){sum+=a[la]*b[ra];la+=lk;ra+=rk;}
                o[outputRow+column*on]=sum;}}}

    private static void directNVectorF64(double[]a,double[]b,double[]o,long[]g,long start,long end){
        var species=jdk.incubator.vector.DoubleVector.SPECIES_PREFERRED;int width=species.length();
        int m=(int)g[4],k=(int)g[5],n=(int)g[6],lm=(int)g[7],lk=(int)g[8];
        int rk=(int)g[9],rn=(int)g[10],om=(int)g[11],on=(int)g[12],rank=(int)g[13];
        for(long ordinal=start;ordinal<end;ordinal++){int row=(int)(ordinal%m);long batchOrdinal=ordinal/m;
            long lb=g[0],rb=g[1],ob=g[2];for(int axis=rank-1;axis>=0;axis--){long extent=g[18+axis];
                long coordinate=batchOrdinal%extent;batchOrdinal/=extent;lb+=coordinate*g[18+rank+axis];
                rb+=coordinate*g[18+2*rank+axis];ob+=coordinate*g[18+3*rank+axis];}
            int leftRow=(int)(lb+(long)row*lm),outputRow=(int)(ob+(long)row*om),column=0;
            for(;column+width<=n;column+=width){var sum=jdk.incubator.vector.DoubleVector.zero(species);
                int la=leftRow,ra=(int)(rb+(long)column*rn);for(int inner=0;inner<k;inner++){
                    sum=sum.add(jdk.incubator.vector.DoubleVector.fromArray(species,b,ra).mul(a[la]));la+=lk;ra+=rk;}
                sum.intoArray(o,outputRow+column*on);}
            for(;column<n;column++){double sum=0;int la=leftRow,ra=(int)(rb+(long)column*rn);
                for(int inner=0;inner<k;inner++){sum+=a[la]*b[ra];la+=lk;ra+=rk;}
                o[outputRow+column*on]=sum;}}}

    private static void directNVectorI32(int[]a,int[]b,int[]o,long[]g,long start,long end){
        var species=jdk.incubator.vector.IntVector.SPECIES_PREFERRED;int width=species.length();
        int m=(int)g[4],k=(int)g[5],n=(int)g[6],lm=(int)g[7],lk=(int)g[8];
        int rk=(int)g[9],rn=(int)g[10],om=(int)g[11],on=(int)g[12],rank=(int)g[13];
        for(long ordinal=start;ordinal<end;ordinal++){int row=(int)(ordinal%m);long batchOrdinal=ordinal/m;
            long lb=g[0],rb=g[1],ob=g[2];for(int axis=rank-1;axis>=0;axis--){long extent=g[18+axis];
                long coordinate=batchOrdinal%extent;batchOrdinal/=extent;lb+=coordinate*g[18+rank+axis];
                rb+=coordinate*g[18+2*rank+axis];ob+=coordinate*g[18+3*rank+axis];}
            int leftRow=(int)(lb+(long)row*lm),outputRow=(int)(ob+(long)row*om),column=0;
            for(;column+width<=n;column+=width){var sum=jdk.incubator.vector.IntVector.zero(species);
                int la=leftRow,ra=(int)(rb+(long)column*rn);for(int inner=0;inner<k;inner++){
                    sum=sum.add(jdk.incubator.vector.IntVector.fromArray(species,b,ra).mul(a[la]));la+=lk;ra+=rk;}
                sum.intoArray(o,outputRow+column*on);}
            for(;column<n;column++){int sum=0,la=leftRow,ra=(int)(rb+(long)column*rn);
                for(int inner=0;inner<k;inner++){sum+=a[la]*b[ra];la+=lk;ra+=rk;}
                o[outputRow+column*on]=sum;}}}

    private static void directTiledNVectorF32(float[]a,float[]b,float[]o,long[]g,long start,long end){
        var species=jdk.incubator.vector.FloatVector.SPECIES_PREFERRED;int width=species.length();
        int m=(int)g[4],k=(int)g[5],n=(int)g[6],lm=(int)g[7],lk=(int)g[8];
        int rk=(int)g[9],rn=(int)g[10],om=(int)g[11],on=(int)g[12],rank=(int)g[13];
        long mTiles=(m+1L)/2,nTiles=(n+2L*width-1)/(2L*width);
        for(long ordinal=start;ordinal<end;ordinal++){long remaining=ordinal,nt=remaining%nTiles;
            remaining/=nTiles;long mt=remaining%mTiles,batchOrdinal=remaining/mTiles;
            int row=(int)(2*mt),column=(int)(2L*width*nt);long lb=g[0],rb=g[1],ob=g[2];
            for(int axis=rank-1;axis>=0;axis--){long extent=g[18+axis],coordinate=batchOrdinal%extent;
                batchOrdinal/=extent;lb+=coordinate*g[18+rank+axis];rb+=coordinate*g[18+2*rank+axis];
                ob+=coordinate*g[18+3*rank+axis];}
            int left0=(int)(lb+(long)row*lm),left1=left0+lm;
            int output0=(int)(ob+(long)row*om),output1=output0+om;
            boolean row1=row+1<m,full0=column+width<=n,full1=column+2*width<=n;
            var sum00=jdk.incubator.vector.FloatVector.zero(species);var sum01=sum00;
            var sum10=sum00;var sum11=sum00;int la0=left0,la1=left1;
            int ra0=(int)(rb+(long)column*rn),ra1=(int)(rb+(long)(column+width)*rn);
            for(int inner=0;inner<k;inner++){float l0=a[la0],l1=row1?a[la1]:0;
                if(full0){var right0=jdk.incubator.vector.FloatVector.fromArray(species,b,ra0);
                    sum00=sum00.add(right0.mul(l0));if(row1)sum10=sum10.add(right0.mul(l1));}
                if(full1){var right1=jdk.incubator.vector.FloatVector.fromArray(species,b,ra1);
                    sum01=sum01.add(right1.mul(l0));if(row1)sum11=sum11.add(right1.mul(l1));}
                la0+=lk;la1+=lk;ra0+=rk;ra1+=rk;}
            int tail=column;if(full0){sum00.intoArray(o,output0+column*on);
                if(row1)sum10.intoArray(o,output1+column*on);tail=column+width;}
            if(full1){sum01.intoArray(o,output0+(column+width)*on);
                if(row1)sum11.intoArray(o,output1+(column+width)*on);tail=column+2*width;}
            int tileEnd=Math.min(n,column+2*width);
            for(int r=0;r<2&&row+r<m;r++)for(int c=tail;c<tileEnd;c++){float sum=0;
                int la=r==0?left0:left1,ra=(int)(rb+(long)c*rn);for(int inner=0;inner<k;inner++){
                    sum+=a[la]*b[ra];la+=lk;ra+=rk;}o[(r==0?output0:output1)+c*on]=sum;}}}

    private static void directTiledNVectorF64(double[]a,double[]b,double[]o,long[]g,long start,long end){
        var species=jdk.incubator.vector.DoubleVector.SPECIES_PREFERRED;int width=species.length();
        int m=(int)g[4],k=(int)g[5],n=(int)g[6],lm=(int)g[7],lk=(int)g[8];
        int rk=(int)g[9],rn=(int)g[10],om=(int)g[11],on=(int)g[12],rank=(int)g[13];
        long mTiles=(m+1L)/2,nTiles=(n+2L*width-1)/(2L*width);
        for(long ordinal=start;ordinal<end;ordinal++){long remaining=ordinal,nt=remaining%nTiles;
            remaining/=nTiles;long mt=remaining%mTiles,batchOrdinal=remaining/mTiles;
            int row=(int)(2*mt),column=(int)(2L*width*nt);long lb=g[0],rb=g[1],ob=g[2];
            for(int axis=rank-1;axis>=0;axis--){long extent=g[18+axis],coordinate=batchOrdinal%extent;
                batchOrdinal/=extent;lb+=coordinate*g[18+rank+axis];rb+=coordinate*g[18+2*rank+axis];
                ob+=coordinate*g[18+3*rank+axis];}
            int left0=(int)(lb+(long)row*lm),left1=left0+lm;
            int output0=(int)(ob+(long)row*om),output1=output0+om;
            boolean row1=row+1<m,full0=column+width<=n,full1=column+2*width<=n;
            var sum00=jdk.incubator.vector.DoubleVector.zero(species);var sum01=sum00;
            var sum10=sum00;var sum11=sum00;int la0=left0,la1=left1;
            int ra0=(int)(rb+(long)column*rn),ra1=(int)(rb+(long)(column+width)*rn);
            for(int inner=0;inner<k;inner++){double l0=a[la0],l1=row1?a[la1]:0;
                if(full0){var right0=jdk.incubator.vector.DoubleVector.fromArray(species,b,ra0);
                    sum00=sum00.add(right0.mul(l0));if(row1)sum10=sum10.add(right0.mul(l1));}
                if(full1){var right1=jdk.incubator.vector.DoubleVector.fromArray(species,b,ra1);
                    sum01=sum01.add(right1.mul(l0));if(row1)sum11=sum11.add(right1.mul(l1));}
                la0+=lk;la1+=lk;ra0+=rk;ra1+=rk;}
            int tail=column;if(full0){sum00.intoArray(o,output0+column*on);
                if(row1)sum10.intoArray(o,output1+column*on);tail=column+width;}
            if(full1){sum01.intoArray(o,output0+(column+width)*on);
                if(row1)sum11.intoArray(o,output1+(column+width)*on);tail=column+2*width;}
            int tileEnd=Math.min(n,column+2*width);
            for(int r=0;r<2&&row+r<m;r++)for(int c=tail;c<tileEnd;c++){double sum=0;
                int la=r==0?left0:left1,ra=(int)(rb+(long)c*rn);for(int inner=0;inner<k;inner++){
                    sum+=a[la]*b[ra];la+=lk;ra+=rk;}o[(r==0?output0:output1)+c*on]=sum;}}}

    private static void directTiledBf16(short[]a,short[]b,short[]o,long[]g,long start,long end){
        int m=(int)g[4],k=(int)g[5],n=(int)g[6],lm=(int)g[7],lk=(int)g[8];
        int rk=(int)g[9],rn=(int)g[10],om=(int)g[11],on=(int)g[12],rank=(int)g[13];
        long mTiles=(m+1L)/2,nTiles=(n+1L)/2;
        for(long ordinal=start;ordinal<end;ordinal++){long remaining=ordinal,nt=remaining%nTiles;
            remaining/=nTiles;long mt=remaining%mTiles,batchOrdinal=remaining/mTiles;
            int row=(int)(2*mt),column=(int)(2*nt);long lb=g[0],rb=g[1],ob=g[2];
            for(int axis=rank-1;axis>=0;axis--){long extent=g[18+axis],coordinate=batchOrdinal%extent;
                batchOrdinal/=extent;lb+=coordinate*g[18+rank+axis];rb+=coordinate*g[18+2*rank+axis];
                ob+=coordinate*g[18+3*rank+axis];}
            int a0=(int)(lb+(long)row*lm),a1=a0+lm,b0=(int)(rb+(long)column*rn),b1=b0+rn;
            float s00=0,s01=0,s10=0,s11=0;for(int inner=0;inner<k;inner++){
                float l0=BFloat16Bits.toFloat(a[a0]),r0=BFloat16Bits.toFloat(b[b0]);s00+=l0*r0;
                if(column+1<n)s01+=l0*BFloat16Bits.toFloat(b[b1]);if(row+1<m){float l1=BFloat16Bits.toFloat(a[a1]);
                    s10+=l1*r0;if(column+1<n)s11+=l1*BFloat16Bits.toFloat(b[b1]);}
                a0+=lk;a1+=lk;b0+=rk;b1+=rk;}
            int out=(int)(ob+(long)row*om+(long)column*on);o[out]=BFloat16Bits.fromFloat(s00);
            if(column+1<n)o[out+on]=BFloat16Bits.fromFloat(s01);if(row+1<m){out+=om;
                o[out]=BFloat16Bits.fromFloat(s10);if(column+1<n)o[out+on]=BFloat16Bits.fromFloat(s11);}}}

    private static void directTiledBf16F32(short[]a,float[]b,float[]o,long[]g,long start,long end){
        int m=(int)g[4],k=(int)g[5],n=(int)g[6],lm=(int)g[7],lk=(int)g[8];
        int rk=(int)g[9],rn=(int)g[10],om=(int)g[11],on=(int)g[12],rank=(int)g[13];
        long mTiles=(m+1L)/2,nTiles=(n+1L)/2;
        for(long ordinal=start;ordinal<end;ordinal++){long remaining=ordinal,nt=remaining%nTiles;
            remaining/=nTiles;long mt=remaining%mTiles,batchOrdinal=remaining/mTiles;
            int row=(int)(2*mt),column=(int)(2*nt);long lb=g[0],rb=g[1],ob=g[2];
            for(int axis=rank-1;axis>=0;axis--){long extent=g[18+axis],coordinate=batchOrdinal%extent;
                batchOrdinal/=extent;lb+=coordinate*g[18+rank+axis];rb+=coordinate*g[18+2*rank+axis];
                ob+=coordinate*g[18+3*rank+axis];}
            int a0=(int)(lb+(long)row*lm),a1=a0+lm,b0=(int)(rb+(long)column*rn),b1=b0+rn;
            float s00=0,s01=0,s10=0,s11=0;for(int inner=0;inner<k;inner++){
                float l0=BFloat16Bits.toFloat(a[a0]),r0=b[b0];s00+=l0*r0;
                if(column+1<n)s01+=l0*b[b1];if(row+1<m){float l1=BFloat16Bits.toFloat(a[a1]);
                    s10+=l1*r0;if(column+1<n)s11+=l1*b[b1];}a0+=lk;a1+=lk;b0+=rk;b1+=rk;}
            int out=(int)(ob+(long)row*om+(long)column*on);o[out]=s00;if(column+1<n)o[out+on]=s01;
            if(row+1<m){out+=om;o[out]=s10;if(column+1<n)o[out+on]=s11;}}}

    private static void directTiledScalarF32(float[]a,float[]b,float[]o,long[]g,long start,long end){
        int m=(int)g[4],k=(int)g[5],n=(int)g[6],lm=(int)g[7],lk=(int)g[8];
        int rk=(int)g[9],rn=(int)g[10],om=(int)g[11],on=(int)g[12],rank=(int)g[13];
        long mTiles=(m+1L)/2,nTiles=(n+1L)/2;
        for(long ordinal=start;ordinal<end;ordinal++){long remaining=ordinal,nt=remaining%nTiles;
            remaining/=nTiles;long mt=remaining%mTiles,batchOrdinal=remaining/mTiles;
            int row=(int)(2*mt),column=(int)(2*nt);long lb=g[0],rb=g[1],ob=g[2];
            for(int axis=rank-1;axis>=0;axis--){long extent=g[18+axis],coordinate=batchOrdinal%extent;
                batchOrdinal/=extent;lb+=coordinate*g[18+rank+axis];rb+=coordinate*g[18+2*rank+axis];
                ob+=coordinate*g[18+3*rank+axis];}
            int a0=(int)(lb+(long)row*lm),a1=a0+lm,b0=(int)(rb+(long)column*rn),b1=b0+rn;
            float s00=0,s01=0,s10=0,s11=0;for(int inner=0;inner<k;inner++){float l0=a[a0],r0=b[b0];
                s00+=l0*r0;if(column+1<n)s01+=l0*b[b1];if(row+1<m){float l1=a[a1];s10+=l1*r0;
                    if(column+1<n)s11+=l1*b[b1];}a0+=lk;a1+=lk;b0+=rk;b1+=rk;}
            int out=(int)(ob+(long)row*om+(long)column*on);o[out]=s00;if(column+1<n)o[out+on]=s01;
            if(row+1<m){out+=om;o[out]=s10;if(column+1<n)o[out+on]=s11;}}}

    private static void directTiledScalarF64(double[]a,double[]b,double[]o,long[]g,long start,long end){
        int m=(int)g[4],k=(int)g[5],n=(int)g[6],lm=(int)g[7],lk=(int)g[8];
        int rk=(int)g[9],rn=(int)g[10],om=(int)g[11],on=(int)g[12],rank=(int)g[13];
        long mTiles=(m+1L)/2,nTiles=(n+1L)/2;
        for(long ordinal=start;ordinal<end;ordinal++){long remaining=ordinal,nt=remaining%nTiles;
            remaining/=nTiles;long mt=remaining%mTiles,batchOrdinal=remaining/mTiles;
            int row=(int)(2*mt),column=(int)(2*nt);long lb=g[0],rb=g[1],ob=g[2];
            for(int axis=rank-1;axis>=0;axis--){long extent=g[18+axis],coordinate=batchOrdinal%extent;
                batchOrdinal/=extent;lb+=coordinate*g[18+rank+axis];rb+=coordinate*g[18+2*rank+axis];
                ob+=coordinate*g[18+3*rank+axis];}
            int a0=(int)(lb+(long)row*lm),a1=a0+lm,b0=(int)(rb+(long)column*rn),b1=b0+rn;
            double s00=0,s01=0,s10=0,s11=0;for(int inner=0;inner<k;inner++){double l0=a[a0],r0=b[b0];
                s00+=l0*r0;if(column+1<n)s01+=l0*b[b1];if(row+1<m){double l1=a[a1];s10+=l1*r0;
                    if(column+1<n)s11+=l1*b[b1];}a0+=lk;a1+=lk;b0+=rk;b1+=rk;}
            int out=(int)(ob+(long)row*om+(long)column*on);o[out]=s00;if(column+1<n)o[out+on]=s01;
            if(row+1<m){out+=om;o[out]=s10;if(column+1<n)o[out+on]=s11;}}}
    private static void relu(float[]v){for(int i=0;i<v.length;i++)v[i]=Math.max(v[i],+0.0f);}private static void relu(double[]v){for(int i=0;i<v.length;i++)v[i]=Math.max(v[i],+0.0);}

    private static CpuAccessPlan plan(CpuAccessPlan.AccessKind kind){return new CpuAccessPlan(kind,CpuAccessPlan.Regime.DENSE_LINEAR,2,List.of(CpuAccessPlan.AxisRole.CONTIGUOUS,CpuAccessPlan.AxisRole.CONTIGUOUS),2);}
    private static CarrierAccess carrier(DataType t){return switch(t){case BFLOAT16->CarrierAccess.SHORT_ARRAY;case FLOAT32->CarrierAccess.FLOAT_ARRAY;case FLOAT64->CarrierAccess.DOUBLE_ARRAY;case INT32->CarrierAccess.INT_ARRAY;case INT64->CarrierAccess.LONG_ARRAY;case BOOL->throw new AssertionError();};}
    private static int species(DataType t){return switch(t){case FLOAT32,INT32->jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.vectorBitSize();case FLOAT64,INT64->jdk.incubator.vector.DoubleVector.SPECIES_PREFERRED.vectorBitSize();default->0;};}
    private static int lanes(DataType t){return switch(t){case FLOAT32,INT32->jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.length();case FLOAT64,INT64->jdk.incubator.vector.DoubleVector.SPECIES_PREFERRED.length();default->1;};}
    private static void fill(float[]v){for(int i=0;i<v.length;i++)v[i]=(i%17-8)*.03125f;}private static void fill(double[]v){for(int i=0;i<v.length;i++)v[i]=(i%17-8)*.03125;}private static void fill(int[]v){for(int i=0;i<v.length;i++)v[i]=i%17-8;}private static void fill(short[]v){for(int i=0;i<v.length;i++)v[i]=BFloat16Bits.fromFloat((i%17-8)*.03125f);}
    private static long checksum(float[]v){long s=0;for(float x:v)s=31*s+Float.floatToRawIntBits(x);return s;}private static long checksum(double[]v){long s=0;for(double x:v)s=31*s+Double.doubleToRawLongBits(x);return s;}private static long checksum(int[]v){long s=0;for(int x:v)s=31*s+x;return s;}private static long checksum(short[]v){long s=0;for(short x:v)s=31*s+(x&0xffff);return s;}
    private static void gate(Case c)throws Throwable{long g=c.generated.run(),d=c.direct.run();sink^=g^d;c.verify.run();}
    private static int repetitions(Action a)throws Throwable{int r=1;while(r<1<<20&&time(a,r)<MIN_NANOS)r*=2;return r;}
    private static long time(Action a,int reps)throws Throwable{long start=System.nanoTime(),s=0;for(int i=0;i<reps;i++)s^=a.run();sink^=s;return System.nanoTime()-start;}
    private static long median(long[]v){long[]c=v.clone();Arrays.sort(c);return c[c.length/2];}
    private static void environment(StringBuilder s){s.append("ENV,java=").append(System.getProperty("java.version")).append(",vm=").append(System.getProperty("java.vm.name")).append(",os=").append(System.getProperty("os.name")).append(",arch=").append(System.getProperty("os.arch")).append('\n');}
    private static void retainGenerated(String name,byte[]bytes,CpuKernelSpecialization spec,CpuMatmulIr ir)throws Exception{Path d=ROOT.resolve("generated");Files.createDirectories(d);Files.write(d.resolve(name+".class"),bytes);Files.write(d.resolve(name+".compatibility"),spec.compatibilityBytes());Files.writeString(d.resolve(name+".specialization"),spec+"\nir="+ir+"\nsha256="+HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes))+"\n");var model=ClassFile.of().parse(bytes);StringBuilder refs=new StringBuilder();java.util.stream.StreamSupport.stream(model.constantPool().spliterator(),false).filter(MemberRefEntry.class::isInstance).map(MemberRefEntry.class::cast).forEach(e->refs.append(e.owner().asInternalName()).append('.').append(e.name().stringValue()).append(e.type().stringValue()).append('\n'));Files.writeString(d.resolve(name+".members"),refs);Files.writeString(d.resolve(name+".parse"),model.flags()+"\nfields="+model.fields().size()+"\nmethods="+model.methods().size()+"\n");}
    private static void retain(int fork,String text)throws Exception{Path d=ROOT.resolve("forks");Files.createDirectories(d);Files.writeString(d.resolve("fork-"+fork+".csv"),text);}
    private static String clean(String s){return s==null?"":s.replace('\n',' ').replace('\r',' ').replace(',',';');}
    private static void aggregate()throws Exception{List<String>names=null;double[][]ratios=new double[5][12];for(int f=0;f<5;f++){List<String>ns=new ArrayList<>();int i=0;for(String line:Files.readAllLines(ROOT.resolve("forks/fork-"+f+".csv")))if(line.startsWith("RESULT,")||line.startsWith("MATERIALIZATION,")){String[]x=line.split(",",6);ns.add(x[1]);ratios[f][i++]=Double.parseDouble(x[4]);}if(i!=12)throw new AssertionError("fork "+f+" rows "+i);if(names==null)names=List.copyOf(ns);else if(!names.equals(ns))throw new AssertionError("inventory");}StringBuilder s=new StringBuilder();int failures=0;for(int i=0;i<12;i++){double[]v=new double[5];for(int f=0;f<5;f++)v[f]=ratios[f][i];Arrays.sort(v);if(v[2]>1.15)failures++;s.append(String.format(Locale.ROOT,"AGGREGATE,%s,%.9f,%s%n",names.get(i),v[2],Arrays.toString(v)));}Files.writeString(ROOT.resolve("summary.csv"),s);System.out.print(s);if(failures!=0)throw new AssertionError("aggregate failures "+failures);}
}
