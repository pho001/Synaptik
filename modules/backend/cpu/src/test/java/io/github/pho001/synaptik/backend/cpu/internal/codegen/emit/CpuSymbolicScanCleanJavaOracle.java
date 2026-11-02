package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Two literal-specialized native FLOAT32 references for the timed symbolic Scan families.
 * Only B is dynamic. Callers select once, supply live same-thread native segments with exact
 * shapes, and isolate every writable segment from inputs and other writes. Both mandatory
 * histories and all five public outputs are written, with zero temporary workspace.
 * This source/Class-File is an algorithm/dataflow oracle, not a claim of identical JIT assembly.
 */
final class CpuSymbolicScanCleanJavaOracle {
    /** Prevents construction of the stateless, test-only reference. */
    private CpuSymbolicScanCleanJavaOracle() {}

    /**
     * Computes T3/F4/H3 TANH forward and the actual reverse stage order, dW before dX.
     * FLOAT32 producing/accumulator sites remain distinct; reverse reads represented saved y,
     * joint final/emission seeds and literal AD constants one/zero, not generator-only inputs.
     * This method has no shape selector, helper, allocation or activation replay.
     *
     * @param b non-negative runtime batch, with all byte products addressable
     * @param initial borrowed read-only initial carry [B,3]
     * @param x borrowed read-only sequence [3,B,4]
     * @param w borrowed read-only weights [4,3]
     * @param fs borrowed read-only final seed [B,3]
     * @param es borrowed read-only emission seed [3,B,3]
     * @param carry separate writable final carry [B,3], fully overwritten
     * @param history separate writable public emission [3,B,3], fully overwritten
     * @param savedHistory separate writable mandatory activation history [3,B,3]
     * @param dh separate writable dInitialCarry [B,3], reused as reverse carry
     * @param dx separate writable dX [3,B,4], fully overwritten
     * @param dw separate writable dW [4,3], fully overwritten even at B=0
     * @throws IllegalStateException if a borrowed segment is closed
     * @throws IndexOutOfBoundsException if storage is smaller than its declared shape
     */
    static void t3f4h3(int b,MemorySegment initial,MemorySegment x,MemorySegment w,
            MemorySegment fs,MemorySegment es,
            MemorySegment carry,MemorySegment history,MemorySegment savedHistory,
            MemorySegment dh,MemorySegment dx,MemorySegment dw) {
        long frame=b*12L, sequenceStride=b*16L;
        for (int time=0;time<3;time++) {
            long timeX=time*sequenceStride, timeY=time*frame;
            MemorySegment previous=time==0?initial:history;
            long previousBase=time==0?0:(time-1)*frame;
            for (int row=0;row<b;row++) for (int col=0;col<3;col++) {
                float sum=0f;
                long xp=timeX+row*16L, wp=col*4L;
                for (int k=0;k<4;k++) {
                    sum+=x.get(ValueLayout.JAVA_FLOAT,xp)*w.get(ValueLayout.JAVA_FLOAT,wp);
                    xp+=4; wp+=12;
                }
                long cell=row*12L+col*4L;
                float input=sum+previous.get(ValueLayout.JAVA_FLOAT,previousBase+cell);
                float y=(float)StrictMath.tanh(input);
                history.set(ValueLayout.JAVA_FLOAT,timeY+cell,y);
                savedHistory.set(ValueLayout.JAVA_FLOAT,timeY+cell,y);
            }
        }
        MemorySegment.copy(history,2*frame,carry,0,frame);
        for (int time=2;time>=0;time--) {
            long timeX=time*sequenceStride, timeY=time*frame;
            MemorySegment oldH=time==2?fs:dh;
            for (int row=0;row<b;row++) for (int col=0;col<3;col++) {
                long cell=row*12L+col*4L;
                float y=savedHistory.get(ValueLayout.JAVA_FLOAT,timeY+cell);
                float incoming=oldH.get(ValueLayout.JAVA_FLOAT,cell)+es.get(ValueLayout.JAVA_FLOAT,timeY+cell);
                float derivative=1f-y*y;
                dh.set(ValueLayout.JAVA_FLOAT,cell,incoming*derivative);
            }
            // Actual reverse contracts transposed X over B, adds old dW, then computes dX.
            for (int k=0;k<4;k++) for (int col=0;col<3;col++) {
                float sum=0f;
                long xp=timeX+k*4L, gp=col*4L;
                for (int row=0;row<b;row++) {
                    sum+=x.get(ValueLayout.JAVA_FLOAT,xp)*dh.get(ValueLayout.JAVA_FLOAT,gp);
                    xp+=16; gp+=12;
                }
                long cell=k*12L+col*4L;
                float old=time==2?0f:dw.get(ValueLayout.JAVA_FLOAT,cell);
                dw.set(ValueLayout.JAVA_FLOAT,cell,old+sum);
            }
            for (int row=0;row<b;row++) for (int k=0;k<4;k++) {
                float sum=0f;
                long gp=row*12L, wp=k*12L;
                for (int col=0;col<3;col++) {
                    sum+=dh.get(ValueLayout.JAVA_FLOAT,gp)*w.get(ValueLayout.JAVA_FLOAT,wp);
                    gp+=4; wp+=4;
                }
                dx.set(ValueLayout.JAVA_FLOAT,timeX+row*16L+k*4L,sum);
            }
        }
    }


    /**
     * Computes T4/F32/H32 TANH forward and the actual reverse stage order, dW before dX.
     * FLOAT32 producing/accumulator sites remain distinct; reverse reads represented saved y,
     * joint final/emission seeds and literal AD constants one/zero, not generator-only inputs.
     * This method has no shape selector, helper, allocation or activation replay.
     *
     * @param b non-negative runtime batch, with all byte products addressable
     * @param initial borrowed read-only initial carry [B,32]
     * @param x borrowed read-only sequence [4,B,32]
     * @param w borrowed read-only weights [32,32]
     * @param fs borrowed read-only final seed [B,32]
     * @param es borrowed read-only emission seed [4,B,32]
     * @param carry separate writable final carry [B,32], fully overwritten
     * @param history separate writable public emission [4,B,32], fully overwritten
     * @param savedHistory separate writable mandatory activation history [4,B,32]
     * @param dh separate writable dInitialCarry [B,32], reused as reverse carry
     * @param dx separate writable dX [4,B,32], fully overwritten
     * @param dw separate writable dW [32,32], fully overwritten even at B=0
     * @throws IllegalStateException if a borrowed segment is closed
     * @throws IndexOutOfBoundsException if storage is smaller than its declared shape
     */
    static void t4f32h32(int b,MemorySegment initial,MemorySegment x,MemorySegment w,
            MemorySegment fs,MemorySegment es,
            MemorySegment carry,MemorySegment history,MemorySegment savedHistory,
            MemorySegment dh,MemorySegment dx,MemorySegment dw) {
        long frame=b*128L, sequenceStride=b*128L;
        for (int time=0;time<4;time++) {
            long timeX=time*sequenceStride, timeY=time*frame;
            MemorySegment previous=time==0?initial:history;
            long previousBase=time==0?0:(time-1)*frame;
            for (int row=0;row<b;row++) for (int col=0;col<32;col++) {
                float sum=0f;
                long xp=timeX+row*128L, wp=col*4L;
                for (int k=0;k<32;k++) {
                    sum+=x.get(ValueLayout.JAVA_FLOAT,xp)*w.get(ValueLayout.JAVA_FLOAT,wp);
                    xp+=4; wp+=128;
                }
                long cell=row*128L+col*4L;
                float input=sum+previous.get(ValueLayout.JAVA_FLOAT,previousBase+cell);
                float y=(float)StrictMath.tanh(input);
                history.set(ValueLayout.JAVA_FLOAT,timeY+cell,y);
                savedHistory.set(ValueLayout.JAVA_FLOAT,timeY+cell,y);
            }
        }
        MemorySegment.copy(history,3*frame,carry,0,frame);
        for (int time=3;time>=0;time--) {
            long timeX=time*sequenceStride, timeY=time*frame;
            MemorySegment oldH=time==3?fs:dh;
            for (int row=0;row<b;row++) for (int col=0;col<32;col++) {
                long cell=row*128L+col*4L;
                float y=savedHistory.get(ValueLayout.JAVA_FLOAT,timeY+cell);
                float incoming=oldH.get(ValueLayout.JAVA_FLOAT,cell)+es.get(ValueLayout.JAVA_FLOAT,timeY+cell);
                float derivative=1f-y*y;
                dh.set(ValueLayout.JAVA_FLOAT,cell,incoming*derivative);
            }
            // Actual reverse contracts transposed X over B, adds old dW, then computes dX.
            for (int k=0;k<32;k++) for (int col=0;col<32;col++) {
                float sum=0f;
                long xp=timeX+k*4L, gp=col*4L;
                for (int row=0;row<b;row++) {
                    sum+=x.get(ValueLayout.JAVA_FLOAT,xp)*dh.get(ValueLayout.JAVA_FLOAT,gp);
                    xp+=128; gp+=128;
                }
                long cell=k*128L+col*4L;
                float old=time==3?0f:dw.get(ValueLayout.JAVA_FLOAT,cell);
                dw.set(ValueLayout.JAVA_FLOAT,cell,old+sum);
            }
            for (int row=0;row<b;row++) for (int k=0;k<32;k++) {
                float sum=0f;
                long gp=row*128L, wp=k*128L;
                for (int col=0;col<32;col++) {
                    sum+=dh.get(ValueLayout.JAVA_FLOAT,gp)*w.get(ValueLayout.JAVA_FLOAT,wp);
                    gp+=4; wp+=4;
                }
                dx.set(ValueLayout.JAVA_FLOAT,timeX+row*128L+k*4L,sum);
            }
        }
    }

}
