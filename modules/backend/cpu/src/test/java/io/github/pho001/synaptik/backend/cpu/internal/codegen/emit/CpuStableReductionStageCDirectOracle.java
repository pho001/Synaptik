package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/** Independently compiled clean-Java direct controls for CPU-0008O Stage C. */
final class CpuStableReductionStageCDirectOracle {
    private CpuStableReductionStageCDirectOracle() { }

    static float f32aa(float[] in, float[] out, long start, long end, int form) {
        float max = -Float.MAX_VALUE; for (long i=start;i<end;i++) max=Math.max(max,in[(int)i]);
        float sum=0; for(long i=start;i<end;i++) sum+=(float)StrictMath.exp((double)(in[(int)i]-max));
        float value=form==1 ? in[(int)start]-max-(float)StrictMath.log((double)sum) :
                form>=2 ? max+(float)StrictMath.log((double)sum)-in[(int)Math.min(start+1,end-1)] : sum;
        if(form==0) for(long i=start;i<end;i++) out[(int)i]=(float)StrictMath.exp((double)(in[(int)i]-max))/sum;
        else out[(int)start]=value; return value;
    }
    static float f32as(float[] in, MemorySegment out, long start, long end, int form) {
        float max=-Float.MAX_VALUE; for(long i=start;i<end;i++) max=Math.max(max,in[(int)i]);
        float sum=0; for(long i=start;i<end;i++) sum+=(float)StrictMath.exp((double)(in[(int)i]-max));
        float value=form==1?in[(int)start]-max-(float)StrictMath.log((double)sum):form>=2?max+(float)StrictMath.log((double)sum)-in[(int)Math.min(start+1,end-1)]:sum;
        if(form==0) for(long i=start;i<end;i++) out.set(ValueLayout.JAVA_FLOAT_UNALIGNED,i*4,(float)StrictMath.exp((double)(in[(int)i]-max))/sum); else out.set(ValueLayout.JAVA_FLOAT_UNALIGNED,start*4,value); return value;
    }
    static double f64ss(MemorySegment in, MemorySegment out, long start, long end, int form) {
        double max=-Double.MAX_VALUE; for(long i=start;i<end;i++) max=Math.max(max,in.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,i*8));
        double sum=0; for(long i=start;i<end;i++) sum+=StrictMath.exp(in.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,i*8)-max);
        double value=form==1?in.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,start*8)-max-StrictMath.log(sum):form>=2?max+StrictMath.log(sum)-in.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,Math.min(start+1,end-1)*8):sum;
        if(form==0) for(long i=start;i<end;i++) out.set(ValueLayout.JAVA_DOUBLE_UNALIGNED,i*8,StrictMath.exp(in.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,i*8)-max)/sum); else out.set(ValueLayout.JAVA_DOUBLE_UNALIGNED,start*8,value); return value;
    }
    static double f64sa(MemorySegment in, double[] out, long start, long end, int form) {
        double max=-Double.MAX_VALUE; for(long i=start;i<end;i++) max=Math.max(max,in.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,i*8));
        double sum=0; for(long i=start;i<end;i++) sum+=StrictMath.exp(in.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,i*8)-max);
        double value=form==1?in.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,start*8)-max-StrictMath.log(sum):form>=2?max+StrictMath.log(sum)-in.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,Math.min(start+1,end-1)*8):sum;
        if(form==0) for(long i=start;i<end;i++) out[(int)i]=StrictMath.exp(in.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,i*8)-max)/sum; else out[(int)start]=value; return value;
    }
}
