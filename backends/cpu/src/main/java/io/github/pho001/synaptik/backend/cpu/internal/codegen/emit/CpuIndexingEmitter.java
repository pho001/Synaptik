package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.CodeBuilder;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDescs;
import java.lang.constant.MethodTypeDesc;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Emits the generated entry bridge and owns direct indexing output loops.
 * Bound CPU execution validates the complete logical index domain before calling these methods;
 * these entry targets therefore perform output mapping and writes only. They allocate no
 * per-element state and neither schedule workers nor repeat bounds validation.
 */
public final class CpuIndexingEmitter {
    private static final ClassDesc OWNER = ClassDesc.of(CpuIndexingEmitter.class.getName());
    private static final DataType[] DATA_TYPES = DataType.values();

    /** Creates a stateless indexing emitter. */
    public CpuIndexingEmitter() { }

    /**
     * Emits one direct call carrying two or three typed boundaries without allocation.
     *
     * @param code non-null Class-File method body receiving the already-typed entry arguments
     * @param specialization non-null scalar indexing specialization with two or three boundaries
     * @throws NullPointerException if either argument is {@code null}
     * @throws IllegalArgumentException if the specialization has any other boundary count
     */
    public void emit(CodeBuilder code, CpuKernelSpecialization specialization) {
        int count = specialization.carrierPattern().size();
        if (count != 2 && count != 3) throw new IllegalArgumentException(
                "indexing requires two or three boundaries");
        for (int i = 0; i < count; i++) code.aload(i);
        code.aload(count).lload(count + 1).lload(count + 3);
        MethodTypeDesc type = count == 2
                ? MethodTypeDesc.of(ConstantDescs.CD_void, ConstantDescs.CD_Object,
                    ConstantDescs.CD_Object, ConstantDescs.CD_long.arrayType(),
                    ConstantDescs.CD_long, ConstantDescs.CD_long)
                : MethodTypeDesc.of(ConstantDescs.CD_void, ConstantDescs.CD_Object,
                    ConstantDescs.CD_Object, ConstantDescs.CD_Object,
                    ConstantDescs.CD_long.arrayType(), ConstantDescs.CD_long,
                    ConstantDescs.CD_long);
        code.invokestatic(OWNER, count == 2 ? "execute2" : "execute3", type);
    }

    /**
     * Writes one validated two-boundary indexing range, currently one-hot output.
     *
     * @param first non-null direct indices carrier
     * @param output non-null writable direct output carrier
     * @param geometry non-null packed cold mapping and range-start state
     * @param start non-negative inclusive output logical ordinal represented by {@code geometry}
     * @param end exclusive output logical ordinal represented by {@code geometry}
     */
    public static void execute2(Object first, Object output, long[] geometry, long start, long end) {
        execute(first, null, output, geometry, start, end);
    }

    /**
     * Writes one validated three-boundary gather-family range.
     *
     * @param first non-null first unique input carrier
     * @param second non-null second unique input carrier
     * @param output non-null writable direct output carrier
     * @param geometry non-null packed cold mapping and range-start state
     * @param start non-negative inclusive output logical ordinal represented by {@code geometry}
     * @param end exclusive output logical ordinal represented by {@code geometry}
     */
    public static void execute3(Object first, Object second, Object output, long[] geometry,
            long start, long end) {
        execute(first, second, output, geometry, start, end);
    }

    private static void execute(Object first, Object second, Object output, long[] p,
            long start, long end) {
        int family=(int)p[0], boundaries=(int)p[1], mapCount=(int)p[2], axis=(int)p[3];
        int batch=(int)p[4], parameter=(int)p[5], outputRank=(int)p[8];
        int indexBoundary=(int)p[9], dataBoundary=(int)p[10];
        int coordinatePosition=11+mapCount;
        int layoutsPosition=coordinatePosition+outputRank;
        int dataLayout=layoutPosition(p,layoutsPosition,dataBoundary);
        int indexLayout=layoutPosition(p,layoutsPosition,indexBoundary);
        int outputLayout=layoutPosition(p,layoutsPosition,boundaries-1);
        int typesPosition=afterLayouts(p,layoutsPosition,boundaries);
        DataType dataType=DATA_TYPES[(int)p[typesPosition+Math.max(0,dataBoundary)]];
        DataType indexType=DATA_TYPES[(int)p[typesPosition+indexBoundary]];
        Object dataCarrier=carrier(first,second,output,dataBoundary,boundaries);
        Object indexCarrier=carrier(first,second,output,indexBoundary,boundaries);
        for (long logical = start; logical < end; logical++) {
            long outputAddress=addressFromOutput(p,outputLayout,coordinatePosition,0,outputRank);
            if (family == 3) {
                long indexAddress=addressFromOutput(p,indexLayout,coordinatePosition,0,
                        (int)p[indexLayout]);
                long selected=readIndex(indexCarrier,indexAddress,indexType);
                writeByte(output,outputAddress,(byte)(selected==p[coordinatePosition+outputRank-1]?1:0));
            } else {
                int dataRank=(int)p[dataLayout], indexRank=(int)p[indexLayout];
                long dataAddress=p[dataLayout+1];
                if(family==0){
                    long indexAddress=addressFromOutput(p,indexLayout,coordinatePosition,axis,indexRank);
                    long selected=readIndex(indexCarrier,indexAddress,indexType);
                    for(int a=0;a<dataRank;a++){
                        long c=a<axis?p[coordinatePosition+a]:a==axis?selected:
                                p[coordinatePosition+a-1+indexRank];
                        dataAddress+=c*p[dataLayout+2+dataRank+a];
                    }
                }else if(family==1){
                    long indexAddress=addressFromOutput(p,indexLayout,coordinatePosition,0,indexRank);
                    long selected=readIndex(indexCarrier,indexAddress,indexType);
                    for(int a=0;a<dataRank;a++)dataAddress+=(a==axis?selected:
                            p[coordinatePosition+a])*p[dataLayout+2+dataRank+a];
                }else{
                    for(int a=0;a<batch;a++)dataAddress+=p[coordinatePosition+a]
                            *p[dataLayout+2+dataRank+a];
                    for(int k=0;k<parameter;k++){
                        long indexAddress=p[indexLayout+1];
                        for(int a=0;a<indexRank-1;a++)indexAddress+=p[coordinatePosition+a]
                                *p[indexLayout+2+indexRank+a];
                        indexAddress+=k*p[indexLayout+2+indexRank+indexRank-1];
                        dataAddress+=readIndex(indexCarrier,indexAddress,indexType)
                                *p[dataLayout+2+dataRank+batch+k];
                    }
                    for(int a=batch+parameter;a<dataRank;a++)dataAddress+=
                            p[coordinatePosition+indexRank-1+a-batch-parameter]
                            *p[dataLayout+2+dataRank+a];
                }
                copy(dataCarrier,dataAddress,output,outputAddress,dataType);
            }
            advanceOutput(p,coordinatePosition,outputLayout,outputRank);
        }
    }

    private static int layoutPosition(long[] p,int start,int boundary){int x=start;for(int b=0;b<boundary;b++){int r=(int)p[x];x+=2+2*r;}return x;}
    private static int afterLayouts(long[] p,int start,int count){int x=start;for(int b=0;b<count;b++){int r=(int)p[x];x+=2+2*r;}return x;}
    private static Object carrier(Object first,Object second,Object output,int boundary,int count){return boundary==count-1?output:boundary==0?first:second;}
    private static long addressFromOutput(long[] p,int layout,int coordinates,int coordinateStart,int rank){long a=p[layout+1];int layoutRank=(int)p[layout];for(int i=0;i<rank;i++)a+=p[coordinates+coordinateStart+i]*p[layout+2+layoutRank+i];return a;}
    private static void advanceOutput(long[] p,int coordinates,int outputLayout,int rank){for(int a=rank-1;a>=0;a--){long next=p[coordinates+a]+1;p[coordinates+a]=next;if(next<p[outputLayout+2+a])return;p[coordinates+a]=0;}}
    private static long readIndex(Object carrier, long address, DataType type) {
        if (type == DataType.INT32) return carrier instanceof int[] a ? a[Math.toIntExact(address)]
                : ((MemorySegment) carrier).get(ValueLayout.JAVA_INT, address * Integer.BYTES);
        return carrier instanceof long[] a ? a[Math.toIntExact(address)]
                : ((MemorySegment) carrier).get(ValueLayout.JAVA_LONG, address * Long.BYTES);
    }
    private static void copy(Object source, long sourceAddress, Object output, long outputAddress,
            DataType type) {
        switch (type) {
            case FLOAT64 -> writeDouble(output, outputAddress, source instanceof double[] a
                    ? a[Math.toIntExact(sourceAddress)] : ((MemorySegment) source).get(
                        ValueLayout.JAVA_DOUBLE, sourceAddress * Double.BYTES));
            case FLOAT32 -> writeFloat(output, outputAddress, source instanceof float[] a
                    ? a[Math.toIntExact(sourceAddress)] : ((MemorySegment) source).get(
                        ValueLayout.JAVA_FLOAT, sourceAddress * Float.BYTES));
            case BFLOAT16 -> writeShort(output, outputAddress, source instanceof short[] a
                    ? a[Math.toIntExact(sourceAddress)] : ((MemorySegment) source).get(
                        ValueLayout.JAVA_SHORT, sourceAddress * Short.BYTES));
            case INT32 -> writeInt(output, outputAddress, source instanceof int[] a
                    ? a[Math.toIntExact(sourceAddress)] : ((MemorySegment) source).get(
                        ValueLayout.JAVA_INT, sourceAddress * Integer.BYTES));
            case INT64 -> writeLong(output, outputAddress, source instanceof long[] a
                    ? a[Math.toIntExact(sourceAddress)] : ((MemorySegment) source).get(
                        ValueLayout.JAVA_LONG, sourceAddress * Long.BYTES));
            case BOOL -> writeByte(output, outputAddress, source instanceof byte[] a
                    ? a[Math.toIntExact(sourceAddress)] : ((MemorySegment) source).get(
                        ValueLayout.JAVA_BYTE, sourceAddress));
        }
    }
    private static void writeDouble(Object o,long a,double v){if(o instanceof double[] x)x[Math.toIntExact(a)]=v;else((MemorySegment)o).set(ValueLayout.JAVA_DOUBLE,a*8,v);}
    private static void writeFloat(Object o,long a,float v){if(o instanceof float[] x)x[Math.toIntExact(a)]=v;else((MemorySegment)o).set(ValueLayout.JAVA_FLOAT,a*4,v);}
    private static void writeShort(Object o,long a,short v){if(o instanceof short[] x)x[Math.toIntExact(a)]=v;else((MemorySegment)o).set(ValueLayout.JAVA_SHORT,a*2,v);}
    private static void writeInt(Object o,long a,int v){if(o instanceof int[] x)x[Math.toIntExact(a)]=v;else((MemorySegment)o).set(ValueLayout.JAVA_INT,a*4,v);}
    private static void writeLong(Object o,long a,long v){if(o instanceof long[] x)x[Math.toIntExact(a)]=v;else((MemorySegment)o).set(ValueLayout.JAVA_LONG,a*8,v);}
    private static void writeByte(Object o,long a,byte v){if(o instanceof byte[] x)x[Math.toIntExact(a)]=v;else((MemorySegment)o).set(ValueLayout.JAVA_BYTE,a,v);}

}
