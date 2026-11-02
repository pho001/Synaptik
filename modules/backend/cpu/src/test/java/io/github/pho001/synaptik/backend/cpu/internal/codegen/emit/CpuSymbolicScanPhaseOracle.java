package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Literal forward/reverse extraction of {@link CpuSymbolicScanCleanJavaOracle#t4f32h32}.
 * Only B is dynamic; FLOAT32 producing sites, contributor order and every mandatory write
 * are unchanged. Borrowed native segments must be live, aligned, accessible on this thread
 * and exactly shaped; writes must be mutually disjoint and isolated from reads. Splitting
 * the Java call can change inlining: this is an algorithm oracle, not identical optimized code.
 */
final class CpuSymbolicScanPhaseOracle {
    /** Prevents construction of the stateless, test-only phase references. */
    private CpuSymbolicScanPhaseOracle() {}

    /**
     * Writes T4/F32/H32 TANH emissions and a separate represented activation history, then
     * copies the final emission to its distinct carry output. No reverse work is performed.
     * Arithmetic and memory operations are extracted unchanged from the joint reference.
     *
     * @param b non-negative runtime batch, with all byte products addressable
     * @param initial borrowed read-only initial carry [B,32]
     * @param x borrowed read-only sequence [4,B,32]
     * @param w borrowed read-only weights [32,32]
     * @param carry separate writable final carry [B,32], fully overwritten
     * @param history separate writable public emission [4,B,32], fully overwritten
     * @param savedHistory separate writable mandatory activation history [4,B,32], fully overwritten
     * @throws NullPointerException if an accessed segment is null
     * @throws IllegalStateException if an accessed segment is closed
     * @throws WrongThreadException if an accessed segment is confined to another thread
     * @throws IllegalArgumentException if an access is misaligned or a write targets read-only storage
     * @throws IndexOutOfBoundsException if storage is smaller than its declared shape
     */
    static void forward(int b,MemorySegment initial,MemorySegment x,MemorySegment w,
            MemorySegment carry,MemorySegment history,MemorySegment savedHistory) {
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
    }

    /**
     * Reads pre-established represented saved y and joint final/emission seeds, overwriting
     * all three gradients. Each time step computes the derivative with literal 1f, then dW
     * before dX; the first dW contribution uses literal 0f. There is no activation replay,
     * initial-carry/public-history read, allocation, or helper call. Repeated calls reset
     * their own gradient carries, including nonempty typed-zero dW at B=0.
     *
     * @param b non-negative runtime batch, with all byte products addressable
     * @param x borrowed read-only sequence [4,B,32]
     * @param w borrowed read-only weights [32,32]
     * @param fs borrowed read-only final seed [B,32]
     * @param es borrowed read-only emission seed [4,B,32]
     * @param savedHistory borrowed read-only represented activation history [4,B,32]
     * @param dh separate writable dInitialCarry [B,32], reused as reverse carry
     * @param dx separate writable dX [4,B,32], fully overwritten
     * @param dw separate writable dW [32,32], fully overwritten even at B=0
     * @throws NullPointerException if an accessed segment is null
     * @throws IllegalStateException if an accessed segment is closed
     * @throws WrongThreadException if an accessed segment is confined to another thread
     * @throws IllegalArgumentException if an access is misaligned or a write targets read-only storage
     * @throws IndexOutOfBoundsException if storage is smaller than its declared shape
     */
    static void reverse(int b,MemorySegment x,MemorySegment w,MemorySegment fs,MemorySegment es,
            MemorySegment savedHistory,MemorySegment dh,MemorySegment dx,MemorySegment dw) {
        long frame=b*128L, sequenceStride=b*128L;
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
