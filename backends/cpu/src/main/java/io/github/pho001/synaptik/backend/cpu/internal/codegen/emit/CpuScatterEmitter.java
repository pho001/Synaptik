package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
import java.lang.classfile.CodeBuilder;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDescs;
import java.lang.constant.MethodTypeDesc;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;

/**
 * Emits the direct generated scatter bridge and owns allocation-free output-coordinate writers.
 * Bounds and replacement uniqueness are validated by cold-bound execution before these entry
 * targets run. Each invocation scans contributions in logical row-major order and writes each
 * owned output coordinate exactly once. Floating multiplication uses only its declared disjoint
 * primitive-limb scratch slice and rounds the exact abstract product once.
 */
public final class CpuScatterEmitter {
    private static final ClassDesc OWNER = ClassDesc.of(CpuScatterEmitter.class.getName());
    private static final DataType[] TYPES = DataType.values();
    private static final ScatterReduction[] REDUCTIONS = ScatterReduction.values();

    /** Creates a stateless scatter emitter. */
    public CpuScatterEmitter() { }

    /**
     * Emits one direct bridge for two through four unique boundaries and optional scratch.
     *
     * @param code non-null generated method body
     * @param specialization non-null matching scalar scatter specialization
     * @throws IllegalArgumentException if boundary cardinality is outside the scatter contract
     */
    public void emit(CodeBuilder code, CpuKernelSpecialization specialization) {
        int count = specialization.carrierPattern().size();
        if (count < 2 || count > 4) throw new IllegalArgumentException(
                "scatter requires two through four unique boundaries");
        for (int i = 0; i < count; i++) code.aload(i);
        int next = count;
        if (specialization.scratchParameter()) code.aload(next++);
        code.aload(next).lload(next + 1).lload(next + 3);
        var parameters = new java.util.ArrayList<ClassDesc>();
        for (int i = 0; i < count; i++) parameters.add(ConstantDescs.CD_Object);
        if (specialization.scratchParameter()) parameters.add(
                ClassDesc.of(MemorySegment.class.getName()));
        parameters.add(ConstantDescs.CD_long.arrayType());
        parameters.add(ConstantDescs.CD_long); parameters.add(ConstantDescs.CD_long);
        code.invokestatic(OWNER, "execute" + count
                        + (specialization.scratchParameter() ? "Scratch" : ""),
                MethodTypeDesc.of(ConstantDescs.CD_void, parameters));
    }

    /**
     * Executes a validated two-boundary scatter range.
     * @param a first unique input carrier
     * @param output writable output carrier
     * @param geometry packed invocation-private coordinate geometry
     * @param start inclusive output ordinal
     * @param end exclusive output ordinal
     */
    public static void execute2(Object a, Object output, long[] geometry, long start, long end) {
        execute(a, null, null, output, null, geometry, start, end);
    }
    /**
     * Executes a validated three-boundary scatter range.
     * @param a first unique input carrier
     * @param b second unique input carrier
     * @param output writable output carrier
     * @param geometry packed invocation-private coordinate geometry
     * @param start inclusive output ordinal
     * @param end exclusive output ordinal
     */
    public static void execute3(Object a, Object b, Object output, long[] geometry, long start, long end) {
        execute(a, b, null, output, null, geometry, start, end);
    }
    /**
     * Executes a validated four-boundary scatter range.
     * @param a first unique input carrier
     * @param b second unique input carrier
     * @param c third unique input carrier
     * @param output writable output carrier
     * @param geometry packed invocation-private coordinate geometry
     * @param start inclusive output ordinal
     * @param end exclusive output ordinal
     */
    public static void execute4(Object a, Object b, Object c, Object output, long[] geometry,
            long start, long end) {
        execute(a, b, c, output, null, geometry, start, end);
    }
    /**
     * Executes a validated two-boundary floating-product scatter range.
     * @param a first unique input carrier
     * @param output writable output carrier
     * @param scratch declared writable exact-product workspace
     * @param geometry packed invocation-private coordinate and scratch-slice geometry
     * @param start inclusive output ordinal
     * @param end exclusive output ordinal
     */
    public static void execute2Scratch(Object a, Object output, MemorySegment scratch,
            long[] geometry, long start, long end) {
        execute(a, null, null, output, scratch, geometry, start, end);
    }
    /**
     * Executes a validated three-boundary floating-product scatter range.
     * @param a first unique input carrier
     * @param b second unique input carrier
     * @param output writable output carrier
     * @param scratch declared writable exact-product workspace
     * @param geometry packed invocation-private coordinate and scratch-slice geometry
     * @param start inclusive output ordinal
     * @param end exclusive output ordinal
     */
    public static void execute3Scratch(Object a, Object b, Object output, MemorySegment scratch,
            long[] geometry, long start, long end) {
        execute(a, b, null, output, scratch, geometry, start, end);
    }
    /**
     * Executes a validated four-boundary floating-product scatter range.
     * @param a first unique input carrier
     * @param b second unique input carrier
     * @param c third unique input carrier
     * @param output writable output carrier
     * @param scratch declared writable exact-product workspace
     * @param geometry packed invocation-private coordinate and scratch-slice geometry
     * @param start inclusive output ordinal
     * @param end exclusive output ordinal
     */
    public static void execute4Scratch(Object a, Object b, Object c, Object output,
            MemorySegment scratch, long[] geometry, long start, long end) {
        execute(a, b, c, output, scratch, geometry, start, end);
    }

    private static void execute(Object a, Object b, Object c, Object output, MemorySegment scratch,
            long[] p, long start, long end) {
        int family=(int)p[0], boundaries=(int)p[2], dataBoundary=(int)p[3];
        int indexBoundary=(int)p[4], updateBoundary=(int)p[5], axis=(int)p[6];
        int batch=(int)p[7], tuple=(int)p[8], outRank=(int)p[11], updateRank=(int)p[12];
        int outSeed=16, outCoordinates=outSeed+outRank;
        int updateCoordinates=outCoordinates+outRank;
        int layouts=updateCoordinates+updateRank;
        int dataLayout=layoutPosition(p,layouts,dataBoundary);
        int indexLayout=layoutPosition(p,layouts,indexBoundary);
        int updateLayout=layoutPosition(p,layouts,updateBoundary);
        int outputLayout=layoutPosition(p,layouts,boundaries-1);
        int types=afterLayouts(p,layouts,boundaries);
        DataType dataType=TYPES[(int)p[types+dataBoundary]];
        DataType indexType=TYPES[(int)p[types+indexBoundary]];
        ScatterReduction reduction=REDUCTIONS[(int)p[1]];
        Object data=carrier(a,b,c,output,dataBoundary,boundaries);
        Object indices=carrier(a,b,c,output,indexBoundary,boundaries);
        Object updates=carrier(a,b,c,output,updateBoundary,boundaries);
        long updateCount=count(p,updateLayout);
        System.arraycopy(p,outSeed,p,outCoordinates,outRank);
        for(long logical=start;logical<end;logical++){
            long outputAddress=address(p,outputLayout,outCoordinates);
            long dataAddress=address(p,dataLayout,outCoordinates);
            boolean found=false;
            long bits=readBits(data,dataAddress,dataType);
            Arrays.fill(p,updateCoordinates,updateCoordinates+updateRank,0L);
            for(long updateOrdinal=0;updateOrdinal<updateCount;updateOrdinal++){
                if(matches(p,family,axis,batch,tuple,outCoordinates,updateCoordinates,
                        dataLayout,indexLayout,indices,indexType)){
                    long updateAddress=address(p,updateLayout,updateCoordinates);
                    long updateBits=readBits(updates,updateAddress,dataType);
                    if(reduction==ScatterReduction.NONE){bits=updateBits;found=true;break;}
                    if(reduction==ScatterReduction.MUL && floating(dataType)){
                        if(!found){ExactProduct.reset(scratch,p[13],p[14]);ExactProduct.factor(
                                scratch,p[13],bits,dataType);}
                        ExactProduct.factor(scratch,p[13],updateBits,dataType);
                    }else bits=reduce(bits,updateBits,dataType,reduction);
                    found=true;
                }
                advance(p,updateCoordinates,updateLayout,updateRank);
            }
            if(found && reduction==ScatterReduction.MUL && floating(dataType))
                bits=ExactProduct.finish(scratch,p[13],dataType);
            writeBits(output,outputAddress,dataType,bits);
            advance(p,outCoordinates,outputLayout,outRank);
        }
    }

    private static boolean matches(long[] p,int family,int axis,int batch,int tuple,
            int out,int update,int dataLayout,int indexLayout,Object indices,DataType indexType){
        int dataRank=(int)p[dataLayout], indexRank=(int)p[indexLayout];
        if(family==0){
            for(int d=0;d<dataRank;d++)if(d!=axis&&p[update+d]!=p[out+d])return false;
            return readIndex(indices,address(p,indexLayout,update),indexType)==p[out+axis];
        }
        if(family==1){
            for(int d=0;d<axis;d++)if(p[update+d]!=p[out+d])return false;
            for(int d=axis+1;d<dataRank;d++)if(p[update+axis+indexRank+d-axis-1]!=p[out+d])return false;
            return readIndex(indices,addressSlice(p,indexLayout,update+axis,indexRank),indexType)
                    ==p[out+axis];
        }
        for(int d=0;d<batch;d++)if(p[update+d]!=p[out+d])return false;
        for(int k=0;k<tuple;k++){
            long indexAddress=p[indexLayout+1];
            for(int d=0;d<indexRank-1;d++)indexAddress+=p[update+d]*p[indexLayout+2+indexRank+d];
            indexAddress+=k*p[indexLayout+2+indexRank+indexRank-1];
            if(readIndex(indices,indexAddress,indexType)!=p[out+batch+k])return false;
        }
        for(int d=batch+tuple;d<dataRank;d++)if(p[update+indexRank-1+d-batch-tuple]!=p[out+d])return false;
        return true;
    }

    private static long reduce(long left,long right,DataType type,ScatterReduction reduction){
        return switch(type){
            case INT32 -> switch(reduction){case ADD->(int)left+(int)right;case MUL->(int)left*(int)right;
                case MIN->Math.min((int)left,(int)right);case MAX->Math.max((int)left,(int)right);default->right;};
            case INT64 -> switch(reduction){case ADD->left+right;case MUL->left*right;
                case MIN->Math.min(left,right);case MAX->Math.max(left,right);default->right;};
            case FLOAT64 -> Double.doubleToRawLongBits(floatingReduce(
                    Double.longBitsToDouble(left),Double.longBitsToDouble(right),reduction));
            case FLOAT32 -> Integer.toUnsignedLong(Float.floatToRawIntBits((float)floatingReduce(
                    Float.intBitsToFloat((int)left),Float.intBitsToFloat((int)right),reduction)));
            case BFLOAT16 -> Short.toUnsignedLong(floatToBfloat((float)floatingReduce(
                    bfloatToFloat((short)left),bfloatToFloat((short)right),reduction)));
            case BOOL -> right;
        };
    }
    private static double floatingReduce(double a,double b,ScatterReduction r){
        if(r==ScatterReduction.ADD)return a+b;
        if(Double.isNaN(a)||Double.isNaN(b))return Double.NaN;
        if(r==ScatterReduction.MIN){if(a==0&&b==0)return rawNegative(a)||rawNegative(b)?-0.0:0.0;return Math.min(a,b);}
        if(r==ScatterReduction.MAX){if(a==0&&b==0)return !rawNegative(a)||!rawNegative(b)?0.0:-0.0;return Math.max(a,b);}
        return a*b;
    }
    private static boolean rawNegative(double v){return Double.doubleToRawLongBits(v)<0;}
    private static boolean floating(DataType t){return t==DataType.FLOAT64||t==DataType.FLOAT32||t==DataType.BFLOAT16;}
    private static int layoutPosition(long[]p,int start,int boundary){int x=start;for(int b=0;b<boundary;b++){int r=(int)p[x];x+=2+2*r;}return x;}
    private static int afterLayouts(long[]p,int start,int boundaries){return layoutPosition(p,start,boundaries);}
    private static long count(long[]p,int layout){long n=1;int r=(int)p[layout];for(int i=0;i<r;i++){long e=p[layout+2+i];if(e==0)return 0;n*=e;}return n;}
    private static long address(long[]p,int layout,int coordinate){return addressSlice(p,layout,coordinate,(int)p[layout]);}
    private static long addressSlice(long[]p,int layout,int coordinate,int rank){long a=p[layout+1];int r=(int)p[layout];for(int i=0;i<rank;i++)a+=p[coordinate+i]*p[layout+2+r+i];return a;}
    private static void advance(long[]p,int coordinate,int layout,int rank){for(int i=rank-1;i>=0;i--){long next=p[coordinate+i]+1;p[coordinate+i]=next;if(next<p[layout+2+i])return;p[coordinate+i]=0;}}
    private static Object carrier(Object a,Object b,Object c,Object output,int boundary,int count){return boundary==count-1?output:boundary==0?a:boundary==1?b:c;}
    private static long readIndex(Object carrier,long address,DataType type){return type==DataType.INT32?(carrier instanceof int[]x?x[Math.toIntExact(address)]:((MemorySegment)carrier).get(ValueLayout.JAVA_INT,address*4)):(carrier instanceof long[]x?x[Math.toIntExact(address)]:((MemorySegment)carrier).get(ValueLayout.JAVA_LONG,address*8));}
    private static long readBits(Object carrier,long address,DataType type){return switch(type){case FLOAT64->Double.doubleToRawLongBits(carrier instanceof double[]x?x[Math.toIntExact(address)]:((MemorySegment)carrier).get(ValueLayout.JAVA_DOUBLE,address*8));case FLOAT32->Integer.toUnsignedLong(Float.floatToRawIntBits(carrier instanceof float[]x?x[Math.toIntExact(address)]:((MemorySegment)carrier).get(ValueLayout.JAVA_FLOAT,address*4)));case BFLOAT16->Short.toUnsignedLong(carrier instanceof short[]x?x[Math.toIntExact(address)]:((MemorySegment)carrier).get(ValueLayout.JAVA_SHORT,address*2));case INT32->carrier instanceof int[]x?x[Math.toIntExact(address)]:((MemorySegment)carrier).get(ValueLayout.JAVA_INT,address*4);case INT64->carrier instanceof long[]x?x[Math.toIntExact(address)]:((MemorySegment)carrier).get(ValueLayout.JAVA_LONG,address*8);case BOOL->carrier instanceof byte[]x?x[Math.toIntExact(address)]:((MemorySegment)carrier).get(ValueLayout.JAVA_BYTE,address);};}
    private static void writeBits(Object carrier,long address,DataType type,long bits){switch(type){case FLOAT64->{double v=Double.longBitsToDouble(bits);if(carrier instanceof double[]x)x[Math.toIntExact(address)]=v;else((MemorySegment)carrier).set(ValueLayout.JAVA_DOUBLE,address*8,v);}case FLOAT32->{float v=Float.intBitsToFloat((int)bits);if(carrier instanceof float[]x)x[Math.toIntExact(address)]=v;else((MemorySegment)carrier).set(ValueLayout.JAVA_FLOAT,address*4,v);}case BFLOAT16->{short v=(short)bits;if(carrier instanceof short[]x)x[Math.toIntExact(address)]=v;else((MemorySegment)carrier).set(ValueLayout.JAVA_SHORT,address*2,v);}case INT32->{int v=(int)bits;if(carrier instanceof int[]x)x[Math.toIntExact(address)]=v;else((MemorySegment)carrier).set(ValueLayout.JAVA_INT,address*4,v);}case INT64->{if(carrier instanceof long[]x)x[Math.toIntExact(address)]=bits;else((MemorySegment)carrier).set(ValueLayout.JAVA_LONG,address*8,bits);}case BOOL->{byte v=(byte)bits;if(carrier instanceof byte[]x)x[Math.toIntExact(address)]=v;else((MemorySegment)carrier).set(ValueLayout.JAVA_BYTE,address,v);}}}
    private static float bfloatToFloat(short bits){return Float.intBitsToFloat(Short.toUnsignedInt(bits)<<16);}
    private static short floatToBfloat(float value){int bits=Float.floatToRawIntBits(value);if((bits&0x7f800000)==0x7f800000&&(bits&0x7fffff)!=0)return(short)((bits>>>16)|0x40);int upper=bits>>>16;int lower=bits&0xffff;if(lower>0x8000||(lower==0x8000&&(upper&1)!=0))upper++;return(short)upper;}

    /** Fixed-capacity unsigned significand product backed entirely by the declared workspace. */
    private static final class ExactProduct {
        private static final long SIGN=1,ZERO=2,INFINITY=4,NAN=8;
        static void reset(MemorySegment s,long o,long bytes){s.set(ValueLayout.JAVA_LONG,o,0);s.set(ValueLayout.JAVA_LONG,o+8,0);s.set(ValueLayout.JAVA_LONG,o+16,1);for(long p=o+24;p<o+bytes;p+=8)s.set(ValueLayout.JAVA_LONG,p,0);s.set(ValueLayout.JAVA_LONG,o+24,1);}
        static void factor(MemorySegment s,long o,long bits,DataType type){long flags=s.get(ValueLayout.JAVA_LONG,o);long signMask=type==DataType.FLOAT64?1L<<63:type==DataType.FLOAT32?1L<<31:1L<<15;if((bits&signMask)!=0)flags^=SIGN;int fractionBits=type==DataType.FLOAT64?52:type==DataType.FLOAT32?23:7;int exponentBits=type==DataType.FLOAT64?11:8;int bias=type==DataType.FLOAT64?1023:127;long fractionMask=(1L<<fractionBits)-1;long exponentMask=(1L<<exponentBits)-1;long fraction=bits&fractionMask;long exponentField=(bits>>>fractionBits)&exponentMask;if(exponentField==exponentMask){flags|=fraction!=0?NAN:INFINITY;s.set(ValueLayout.JAVA_LONG,o,flags);return;}if(exponentField==0&&fraction==0){flags|=ZERO;s.set(ValueLayout.JAVA_LONG,o,flags);return;}long significand=exponentField==0?fraction:(1L<<fractionBits)|fraction;long exponent=(exponentField==0?1-bias:exponentField-bias)-fractionBits;long sum=Math.addExact(s.get(ValueLayout.JAVA_LONG,o+8),exponent);s.set(ValueLayout.JAVA_LONG,o+8,sum);multiply(s,o,significand);s.set(ValueLayout.JAVA_LONG,o,flags);}
        static void multiply(MemorySegment s,long o,long factor){int used=Math.toIntExact(s.get(ValueLayout.JAVA_LONG,o+16));long carry=0;for(int i=0;i<used;i++){long x=s.get(ValueLayout.JAVA_LONG,o+24+8L*i);long low=x*factor;long high=Math.unsignedMultiplyHigh(x,factor);long sum=low+carry;if(Long.compareUnsigned(sum,low)<0)high++;s.set(ValueLayout.JAVA_LONG,o+24+8L*i,sum);carry=high;}if(carry!=0){s.set(ValueLayout.JAVA_LONG,o+24+8L*used,carry);s.set(ValueLayout.JAVA_LONG,o+16,used+1);}}
        static long finish(MemorySegment s,long o,DataType type){long flags=s.get(ValueLayout.JAVA_LONG,o),sign=flags&SIGN;int total=type==DataType.FLOAT64?64:type==DataType.FLOAT32?32:16;long signBit=sign==0?0:1L<<(total-1);if((flags&NAN)!=0||(flags&(ZERO|INFINITY))==(ZERO|INFINITY))return signBit|canonicalNan(type);if((flags&INFINITY)!=0)return signBit|positiveInfinity(type);if((flags&ZERO)!=0)return signBit;int p=type==DataType.FLOAT64?53:type==DataType.FLOAT32?24:8;int bias=type==DataType.FLOAT64?1023:127;int maxExp=type==DataType.FLOAT64?1023:127;int minNormal=1-bias;int fractionBits=p-1;int used=Math.toIntExact(s.get(ValueLayout.JAVA_LONG,o+16));long top=s.get(ValueLayout.JAVA_LONG,o+24+8L*(used-1));int bitLength=64*(used-1)+(64-Long.numberOfLeadingZeros(top));long exponent=s.get(ValueLayout.JAVA_LONG,o+8);long unbiased=exponent+bitLength-1;if(unbiased>maxExp)return signBit|positiveInfinity(type);if(unbiased>=minNormal){long q=rounded(s,o,bitLength-p);if(q==(1L<<p)){q>>>=1;unbiased++;if(unbiased>maxExp)return signBit|positiveInfinity(type);}long expField=unbiased+bias;long fraction=q&((1L<<fractionBits)-1);return signBit|(expField<<fractionBits)|fraction;}long quantum=minNormal-fractionBits;long shift=quantum-exponent;long q=rounded(s,o,shift);if(q==0)return signBit;if(q>=(1L<<fractionBits))return signBit|(1L<<fractionBits);return signBit|q;}
        static long rounded(MemorySegment s,long o,long shift){int used=Math.toIntExact(s.get(ValueLayout.JAVA_LONG,o+16));int bitLength=64*(used-1)+(64-Long.numberOfLeadingZeros(s.get(ValueLayout.JAVA_LONG,o+24+8L*(used-1))));if(shift<=0)return low(s,o)<<-shift;if(shift>bitLength)return 0;long q=shift>=bitLength?0:shiftedLow(s,o,shift);boolean guard=bit(s,o,shift-1);boolean sticky=anyBelow(s,o,shift-1);return guard&&(sticky||(q&1)!=0)?q+1:q;}
        static long low(MemorySegment s,long o){return s.get(ValueLayout.JAVA_LONG,o+24);}
        static long shiftedLow(MemorySegment s,long o,long shift){int limb=(int)(shift>>>6),bits=(int)(shift&63);long value=s.get(ValueLayout.JAVA_LONG,o+24+8L*limb)>>>bits;if(bits!=0){int used=Math.toIntExact(s.get(ValueLayout.JAVA_LONG,o+16));if(limb+1<used)value|=s.get(ValueLayout.JAVA_LONG,o+24+8L*(limb+1))<<(64-bits);}return value;}
        static boolean bit(MemorySegment s,long o,long bit){if(bit<0)return false;int limb=(int)(bit>>>6);int used=Math.toIntExact(s.get(ValueLayout.JAVA_LONG,o+16));return limb<used&&((s.get(ValueLayout.JAVA_LONG,o+24+8L*limb)>>>((int)bit&63))&1)!=0;}
        static boolean anyBelow(MemorySegment s,long o,long bits){if(bits<=0)return false;int full=(int)(bits>>>6);int used=Math.toIntExact(s.get(ValueLayout.JAVA_LONG,o+16));for(int i=0;i<Math.min(full,used);i++)if(s.get(ValueLayout.JAVA_LONG,o+24+8L*i)!=0)return true;int remain=(int)(bits&63);return remain!=0&&full<used&&(s.get(ValueLayout.JAVA_LONG,o+24+8L*full)&((1L<<remain)-1))!=0;}
        static long canonicalNan(DataType t){return t==DataType.FLOAT64?0x7ff8000000000000L:t==DataType.FLOAT32?0x7fc00000L:0x7fc0L;}
        static long positiveInfinity(DataType t){return t==DataType.FLOAT64?0x7ff0000000000000L:t==DataType.FLOAT32?0x7f800000L:0x7f80L;}
    }
}
