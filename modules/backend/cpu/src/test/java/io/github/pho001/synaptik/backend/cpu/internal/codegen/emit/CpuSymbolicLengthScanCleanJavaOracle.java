package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import java.lang.foreign.MemorySegment;
import static java.lang.foreign.ValueLayout.*;

/**
 * Specialized T4/F3/H2 clean-Java references with runtime B and fixed represented sites.
 * Each typed method has direct active-row/feature/contributor loops and preallocated carriers.
 * A complete row product precedes the recurrent product/epilogue; the old carry is independent
 * of padded histories. Reverse uses exact saved outputs/previous carries, increasing-b sums,
 * one distinct parameter ADD per active time, and O(parameter) contribution state.
 * There is no per-step allocation, callback, graph interpretation, boxing or dtype dispatch.
 */
final class CpuSymbolicLengthScanCleanJavaOracle {
    private CpuSymbolicLengthScanCleanJavaOracle() {}

    /**
     * Executes one traversal with independent raw carry backing and exact active saved sites.
     * @param initial borrowed B-by-2 initial state
     * @param x borrowed [4,B,3] sequence; padding is never read
     * @param w borrowed [3,2] input weights
     * @param u borrowed [2,2] recurrent weights
     * @param bias borrowed [2] bias
     * @param lengths borrowed INT64[B] prevalidated lengths
     * @param result distinct writable [B,2] state
     * @param emissions distinct writable [4,B,2] output
     * @param previous distinct writable previous-carry history at active producing sites
     * @param saved distinct writable represented activation history
     * @param work preallocated product and next-state banks, at least 4*B cells
     * @param b nonnegative invocation batch, int-bounded
     * @param reverse whether the original traversal decreases time
     */
    static void forwardFloat32(MemorySegment initial,MemorySegment x,MemorySegment w,MemorySegment u,
            MemorySegment bias,MemorySegment lengths,MemorySegment result,MemorySegment emissions,
            MemorySegment previous,MemorySegment saved,MemorySegment work,int b,boolean reverse) {
        for (int i = 0; i < 2*b; i++) result.setAtIndex(JAVA_FLOAT,i,initial.getAtIndex(JAVA_FLOAT,i));
        for (int i = 0; i < 8*b; i++) {
            emissions.setAtIndex(JAVA_FLOAT,i,0f); previous.setAtIndex(JAVA_FLOAT,i,0f); saved.setAtIndex(JAVA_FLOAT,i,0f);
        }
        long maximum = 0;
        for (int row = 0; row < b; row++) maximum = Math.max(maximum,lengths.getAtIndex(JAVA_LONG,row));
        for (int step = 0; step < 4; step++) {
            int t = reverse ? 3-step : step;
            if (t >= maximum) continue;
            for (int row = 0; row < b; row++) if (t < lengths.getAtIndex(JAVA_LONG,row)) for (int col = 0; col < 2; col++) {
                previous.setAtIndex(JAVA_FLOAT,(t*b+row)*2+col,result.getAtIndex(JAVA_FLOAT,row*2+col));
                float sum = 0f;
                for (int k = 0; k < 3; k++) sum += x.getAtIndex(JAVA_FLOAT,(t*b+row)*3+k)*w.getAtIndex(JAVA_FLOAT,k*2+col);
                work.setAtIndex(JAVA_FLOAT,row*2+col,sum);
            }
            for (int row = 0; row < b; row++) if (t < lengths.getAtIndex(JAVA_LONG,row)) for (int col = 0; col < 2; col++) {
                float sum = 0f;
                for (int k = 0; k < 2; k++) sum += result.getAtIndex(JAVA_FLOAT,row*2+k)*u.getAtIndex(JAVA_FLOAT,k*2+col);
                float next = (float)Math.tanh((work.getAtIndex(JAVA_FLOAT,row*2+col)+sum)+bias.getAtIndex(JAVA_FLOAT,col));
                work.setAtIndex(JAVA_FLOAT,2*b+row*2+col,next);
                emissions.setAtIndex(JAVA_FLOAT,(t*b+row)*2+col,next); saved.setAtIndex(JAVA_FLOAT,(t*b+row)*2+col,next);
            }
            for (int row = 0; row < b; row++) if (t < lengths.getAtIndex(JAVA_LONG,row)) for (int col = 0; col < 2; col++)
                result.setAtIndex(JAVA_FLOAT,row*2+col,work.getAtIndex(JAVA_FLOAT,2*b+row*2+col));
        }
    }

    /**
     * Computes the independent first-order active-prefix adjoint, never activation replay.
     * @param x borrowed [4,B,3] primal sequence
     * @param w borrowed [3,2] input weights
     * @param u borrowed [2,2] recurrent weights
     * @param lengths borrowed prevalidated INT64[B]
     * @param fs borrowed [B,2] final seed, copied raw for inactive rows
     * @param es borrowed [4,B,2] emission seed, padding never read
     * @param previous exact active previous-carry history
     * @param saved exact represented activation history
     * @param dh writable [B,2] initial-state cotangent
     * @param dx writable [4,B,3] sequence cotangent, inactive cells raw +0
     * @param dw writable fixed [3,2] parameter sum
     * @param du writable fixed [2,2] parameter sum
     * @param db writable fixed [2] bias sum
     * @param work preallocated local-adjoint bank, at least 2*B cells; all old-dh reads precede its update
     * @param b nonnegative invocation batch
     * @param reverse original traversal direction; adjoint visits the opposite direction
     */
    static void reverseFloat32(MemorySegment x,MemorySegment w,MemorySegment u,MemorySegment lengths,
            MemorySegment fs,MemorySegment es,MemorySegment previous,MemorySegment saved,
            MemorySegment dh,MemorySegment dx,MemorySegment dw,MemorySegment du,MemorySegment db,
            MemorySegment work,int b,boolean reverse) {
        for (int i = 0; i < 2*b; i++) dh.setAtIndex(JAVA_FLOAT,i,fs.getAtIndex(JAVA_FLOAT,i));
        for (int i = 0; i < 12*b; i++) dx.setAtIndex(JAVA_FLOAT,i,0f);
        for (int i = 0; i < 6; i++) dw.setAtIndex(JAVA_FLOAT,i,0f);
        for (int i = 0; i < 4; i++) du.setAtIndex(JAVA_FLOAT,i,0f);
        for (int i = 0; i < 2; i++) db.setAtIndex(JAVA_FLOAT,i,0f);
        long maximum = 0;
        for (int row = 0; row < b; row++) maximum = Math.max(maximum,lengths.getAtIndex(JAVA_LONG,row));
        for (int step = 0; step < 4; step++) {
            int t = reverse ? step : 3-step;
            if (t >= maximum) continue;
            for (int row = 0; row < b; row++) if (t < lengths.getAtIndex(JAVA_LONG,row)) for (int col = 0; col < 2; col++) {
                float y = saved.getAtIndex(JAVA_FLOAT,(t*b+row)*2+col);
                float seed = dh.getAtIndex(JAVA_FLOAT,row*2+col)+es.getAtIndex(JAVA_FLOAT,(t*b+row)*2+col);
                float local = seed*((float)1-y*y);
                work.setAtIndex(JAVA_FLOAT,row*2+col,local);
            }
            for (int row = 0; row < b; row++) if (t < lengths.getAtIndex(JAVA_LONG,row)) for (int k = 0; k < 3; k++) {
                float sum = 0f;
                for (int col = 0; col < 2; col++) sum += work.getAtIndex(JAVA_FLOAT,row*2+col)*w.getAtIndex(JAVA_FLOAT,k*2+col);
                dx.setAtIndex(JAVA_FLOAT,(t*b+row)*3+k,sum);
            }
            for (int k = 0; k < 3; k++) for (int col = 0; col < 2; col++) {
                float sum = 0f;
                for (int row = 0; row < b; row++) if (t < lengths.getAtIndex(JAVA_LONG,row))
                    sum += x.getAtIndex(JAVA_FLOAT,(t*b+row)*3+k)*work.getAtIndex(JAVA_FLOAT,row*2+col);
                dw.setAtIndex(JAVA_FLOAT,k*2+col,dw.getAtIndex(JAVA_FLOAT,k*2+col)+sum);
            }
            for (int k = 0; k < 2; k++) for (int col = 0; col < 2; col++) {
                float sum = 0f;
                for (int row = 0; row < b; row++) if (t < lengths.getAtIndex(JAVA_LONG,row))
                    sum += previous.getAtIndex(JAVA_FLOAT,(t*b+row)*2+k)*work.getAtIndex(JAVA_FLOAT,row*2+col);
                du.setAtIndex(JAVA_FLOAT,k*2+col,du.getAtIndex(JAVA_FLOAT,k*2+col)+sum);
            }
            for (int col = 0; col < 2; col++) {
                float sum = 0f;
                for (int row = 0; row < b; row++) if (t < lengths.getAtIndex(JAVA_LONG,row)) sum += work.getAtIndex(JAVA_FLOAT,row*2+col);
                db.setAtIndex(JAVA_FLOAT,col,db.getAtIndex(JAVA_FLOAT,col)+sum);
            }
            for (int row = 0; row < b; row++) if (t < lengths.getAtIndex(JAVA_LONG,row)) for (int k = 0; k < 2; k++) {
                float sum = 0f;
                for (int col = 0; col < 2; col++) sum += work.getAtIndex(JAVA_FLOAT,row*2+col)*u.getAtIndex(JAVA_FLOAT,k*2+col);
                dh.setAtIndex(JAVA_FLOAT,row*2+k,sum);
            }
        }
    }

    /**
     * Executes one traversal with independent raw carry backing and exact active saved sites.
     * @param initial borrowed B-by-2 initial state
     * @param x borrowed [4,B,3] sequence; padding is never read
     * @param w borrowed [3,2] input weights
     * @param u borrowed [2,2] recurrent weights
     * @param bias borrowed [2] bias
     * @param lengths borrowed INT64[B] prevalidated lengths
     * @param result distinct writable [B,2] state
     * @param emissions distinct writable [4,B,2] output
     * @param previous distinct writable previous-carry history at active producing sites
     * @param saved distinct writable represented activation history
     * @param work preallocated product and next-state banks, at least 4*B cells
     * @param b nonnegative invocation batch, int-bounded
     * @param reverse whether the original traversal decreases time
     */
    static void forwardFloat64(MemorySegment initial,MemorySegment x,MemorySegment w,MemorySegment u,
            MemorySegment bias,MemorySegment lengths,MemorySegment result,MemorySegment emissions,
            MemorySegment previous,MemorySegment saved,MemorySegment work,int b,boolean reverse) {
        for (int i = 0; i < 2*b; i++) result.setAtIndex(JAVA_DOUBLE,i,initial.getAtIndex(JAVA_DOUBLE,i));
        for (int i = 0; i < 8*b; i++) {
            emissions.setAtIndex(JAVA_DOUBLE,i,0d); previous.setAtIndex(JAVA_DOUBLE,i,0d); saved.setAtIndex(JAVA_DOUBLE,i,0d);
        }
        long maximum = 0;
        for (int row = 0; row < b; row++) maximum = Math.max(maximum,lengths.getAtIndex(JAVA_LONG,row));
        for (int step = 0; step < 4; step++) {
            int t = reverse ? 3-step : step;
            if (t >= maximum) continue;
            for (int row = 0; row < b; row++) if (t < lengths.getAtIndex(JAVA_LONG,row)) for (int col = 0; col < 2; col++) {
                previous.setAtIndex(JAVA_DOUBLE,(t*b+row)*2+col,result.getAtIndex(JAVA_DOUBLE,row*2+col));
                double sum = 0d;
                for (int k = 0; k < 3; k++) sum += x.getAtIndex(JAVA_DOUBLE,(t*b+row)*3+k)*w.getAtIndex(JAVA_DOUBLE,k*2+col);
                work.setAtIndex(JAVA_DOUBLE,row*2+col,sum);
            }
            for (int row = 0; row < b; row++) if (t < lengths.getAtIndex(JAVA_LONG,row)) for (int col = 0; col < 2; col++) {
                double sum = 0d;
                for (int k = 0; k < 2; k++) sum += result.getAtIndex(JAVA_DOUBLE,row*2+k)*u.getAtIndex(JAVA_DOUBLE,k*2+col);
                double next = (double)Math.tanh((work.getAtIndex(JAVA_DOUBLE,row*2+col)+sum)+bias.getAtIndex(JAVA_DOUBLE,col));
                work.setAtIndex(JAVA_DOUBLE,2*b+row*2+col,next);
                emissions.setAtIndex(JAVA_DOUBLE,(t*b+row)*2+col,next); saved.setAtIndex(JAVA_DOUBLE,(t*b+row)*2+col,next);
            }
            for (int row = 0; row < b; row++) if (t < lengths.getAtIndex(JAVA_LONG,row)) for (int col = 0; col < 2; col++)
                result.setAtIndex(JAVA_DOUBLE,row*2+col,work.getAtIndex(JAVA_DOUBLE,2*b+row*2+col));
        }
    }

    /**
     * Computes the independent first-order active-prefix adjoint, never activation replay.
     * @param x borrowed [4,B,3] primal sequence
     * @param w borrowed [3,2] input weights
     * @param u borrowed [2,2] recurrent weights
     * @param lengths borrowed prevalidated INT64[B]
     * @param fs borrowed [B,2] final seed, copied raw for inactive rows
     * @param es borrowed [4,B,2] emission seed, padding never read
     * @param previous exact active previous-carry history
     * @param saved exact represented activation history
     * @param dh writable [B,2] initial-state cotangent
     * @param dx writable [4,B,3] sequence cotangent, inactive cells raw +0
     * @param dw writable fixed [3,2] parameter sum
     * @param du writable fixed [2,2] parameter sum
     * @param db writable fixed [2] bias sum
     * @param work preallocated local-adjoint bank, at least 2*B cells; all old-dh reads precede its update
     * @param b nonnegative invocation batch
     * @param reverse original traversal direction; adjoint visits the opposite direction
     */
    static void reverseFloat64(MemorySegment x,MemorySegment w,MemorySegment u,MemorySegment lengths,
            MemorySegment fs,MemorySegment es,MemorySegment previous,MemorySegment saved,
            MemorySegment dh,MemorySegment dx,MemorySegment dw,MemorySegment du,MemorySegment db,
            MemorySegment work,int b,boolean reverse) {
        for (int i = 0; i < 2*b; i++) dh.setAtIndex(JAVA_DOUBLE,i,fs.getAtIndex(JAVA_DOUBLE,i));
        for (int i = 0; i < 12*b; i++) dx.setAtIndex(JAVA_DOUBLE,i,0d);
        for (int i = 0; i < 6; i++) dw.setAtIndex(JAVA_DOUBLE,i,0d);
        for (int i = 0; i < 4; i++) du.setAtIndex(JAVA_DOUBLE,i,0d);
        for (int i = 0; i < 2; i++) db.setAtIndex(JAVA_DOUBLE,i,0d);
        long maximum = 0;
        for (int row = 0; row < b; row++) maximum = Math.max(maximum,lengths.getAtIndex(JAVA_LONG,row));
        for (int step = 0; step < 4; step++) {
            int t = reverse ? step : 3-step;
            if (t >= maximum) continue;
            for (int row = 0; row < b; row++) if (t < lengths.getAtIndex(JAVA_LONG,row)) for (int col = 0; col < 2; col++) {
                double y = saved.getAtIndex(JAVA_DOUBLE,(t*b+row)*2+col);
                double seed = dh.getAtIndex(JAVA_DOUBLE,row*2+col)+es.getAtIndex(JAVA_DOUBLE,(t*b+row)*2+col);
                double local = seed*((double)1-y*y);
                work.setAtIndex(JAVA_DOUBLE,row*2+col,local);
            }
            for (int row = 0; row < b; row++) if (t < lengths.getAtIndex(JAVA_LONG,row)) for (int k = 0; k < 3; k++) {
                double sum = 0d;
                for (int col = 0; col < 2; col++) sum += work.getAtIndex(JAVA_DOUBLE,row*2+col)*w.getAtIndex(JAVA_DOUBLE,k*2+col);
                dx.setAtIndex(JAVA_DOUBLE,(t*b+row)*3+k,sum);
            }
            for (int k = 0; k < 3; k++) for (int col = 0; col < 2; col++) {
                double sum = 0d;
                for (int row = 0; row < b; row++) if (t < lengths.getAtIndex(JAVA_LONG,row))
                    sum += x.getAtIndex(JAVA_DOUBLE,(t*b+row)*3+k)*work.getAtIndex(JAVA_DOUBLE,row*2+col);
                dw.setAtIndex(JAVA_DOUBLE,k*2+col,dw.getAtIndex(JAVA_DOUBLE,k*2+col)+sum);
            }
            for (int k = 0; k < 2; k++) for (int col = 0; col < 2; col++) {
                double sum = 0d;
                for (int row = 0; row < b; row++) if (t < lengths.getAtIndex(JAVA_LONG,row))
                    sum += previous.getAtIndex(JAVA_DOUBLE,(t*b+row)*2+k)*work.getAtIndex(JAVA_DOUBLE,row*2+col);
                du.setAtIndex(JAVA_DOUBLE,k*2+col,du.getAtIndex(JAVA_DOUBLE,k*2+col)+sum);
            }
            for (int col = 0; col < 2; col++) {
                double sum = 0d;
                for (int row = 0; row < b; row++) if (t < lengths.getAtIndex(JAVA_LONG,row)) sum += work.getAtIndex(JAVA_DOUBLE,row*2+col);
                db.setAtIndex(JAVA_DOUBLE,col,db.getAtIndex(JAVA_DOUBLE,col)+sum);
            }
            for (int row = 0; row < b; row++) if (t < lengths.getAtIndex(JAVA_LONG,row)) for (int k = 0; k < 2; k++) {
                double sum = 0d;
                for (int col = 0; col < 2; col++) sum += work.getAtIndex(JAVA_DOUBLE,row*2+col)*u.getAtIndex(JAVA_DOUBLE,k*2+col);
                dh.setAtIndex(JAVA_DOUBLE,row*2+k,sum);
            }
        }
    }
}
