package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Clean Java oracle for the CPU0071 static-length T4/B7/F3/H2 specialization. Unlike the
 * combined-v9 reference, contractions retain long K and selected byte cursors, unconditional
 * cursor advancement and separate typed operand locals. Setup constants are outside time.
 * Preallocated carry/product/incoming storage preserves whole-domain stage boundaries,
 * simultaneous old/next lifetimes, original-time histories and exact saved producing sites.
 * This is test-only direct mathematics, not a production fallback or new execution owner.
 */
final class CpuMatmulBindingCleanJavaOracle {
    /** Prevents construction of the stateless specialized oracle. */
    private CpuMatmulBindingCleanJavaOracle() {}

    /**
     * Specialized static-length forward: separate product domain, recurrent/tail domain,
     * original-time histories, alternating old/next carry banks and raw inactive copies.
     * Constant byte strides precede time; only selected carrier/bank bases change per step.
     * @param initial borrowed [7,2] state
     * @param x borrowed [4,7,3] sequence
     * @param w borrowed [3,2] weights
     * @param u borrowed [2,2] recurrent weights
     * @param bias borrowed [2] bias
     * @param lengths borrowed prevalidated [7] INT64 prefix lengths
     * @param result distinct writable [7,2] final carry
     * @param emissions distinct writable [4,7,2] emissions
     * @param previous distinct writable original-time previous-carry history
     * @param saved distinct writable represented TANH history
     * @param work preallocated 42-cell product and two carry banks
     */
    static void forwardFloat32(MemorySegment initial,MemorySegment x,MemorySegment w,MemorySegment u,
            MemorySegment bias,MemorySegment lengths,MemorySegment result,MemorySegment emissions,
            MemorySegment previous,MemorySegment saved,MemorySegment work) {
        var layout = ValueLayout.JAVA_FLOAT_UNALIGNED;
        for (long p = 0; p < 56L*4; p += 4) {
            emissions.set(layout,p,0f); previous.set(layout,p,0f); saved.set(layout,p,0f);
        }
        long maximum = 0;
        for (int row = 0; row < 7; row++) maximum = Math.max(maximum,lengths.getAtIndex(ValueLayout.JAVA_LONG,row));
        for (int t = 0; t < 4; t++) {
            MemorySegment old = t == 0 ? initial : work;
            long oldBase = t == 0 ? 0 : ((t-1)&1)*14L*4, nextBase = (t&1)*14L*4;
            if (t < maximum) {
                for (int row = 0; row < 7; row++) if (t < lengths.getAtIndex(ValueLayout.JAVA_LONG,row))
                    for (int col = 0; col < 2; col++) {
                        long p = (row*2L+col)*4;
                        previous.set(layout,(t*14L)*4+p,old.get(layout,oldBase+p));
                    }
                for (int row = 0; row < 7; row++) if (t < lengths.getAtIndex(ValueLayout.JAVA_LONG,row))
                    for (int col = 0; col < 2; col++) {
                        float sum = 0f;
                        long a = (t*21L+row*3L)*4, b = col*(long)4;
                        for (long k = 0; k < 3; k++) {
                            float left = x.get(layout,a), right = w.get(layout,b);
                            sum = sum+left*right; a += 4; b += 2L*4;
                        }
                        work.set(layout,(28L+row*2L+col)*4,sum);
                    }
                for (int row = 0; row < 7; row++) if (t < lengths.getAtIndex(ValueLayout.JAVA_LONG,row))
                    for (int col = 0; col < 2; col++) {
                        float sum = 0f;
                        long a = oldBase+row*2L*4, b = col*(long)4;
                        for (long k = 0; k < 2; k++) {
                            float left = old.get(layout,a), right = u.get(layout,b);
                            sum = sum+left*right; a += 4; b += 2L*4;
                        }
                        float next = (float)Math.tanh((work.get(layout,(28L+row*2L+col)*4)+sum)+bias.get(layout,col*(long)4));
                        emissions.set(layout,(t*14L+row*2L+col)*4,next);
                    }
                for (int row = 0; row < 7; row++) if (t < lengths.getAtIndex(ValueLayout.JAVA_LONG,row))
                    for (int col = 0; col < 2; col++) {
                        long p = (t*14L+row*2L+col)*4;
                        saved.set(layout,p,emissions.get(layout,p));
                    }
            }
            for (int row = 0; row < 7; row++) for (int col = 0; col < 2; col++) {
                long p = (row*2L+col)*4;
                work.set(layout,nextBase+p,t < lengths.getAtIndex(ValueLayout.JAVA_LONG,row)
                        ? emissions.get(layout,t*14L*4+p) : old.get(layout,oldBase+p));
            }
        }
        for (long p = 0; p < 14L*4; p += 4) result.set(layout,p,work.get(layout,14L*4+p));
    }

    /**
     * Actual-sibling specialized static-length adjoint with incoming scratch, then dH, dW,
     * dU, bias SUM/ADD, and dX domains in the declared stage order. Active K branches precede
     * both operand loads; scalar SUM remains distinct from ordinary exact SUM.
     * @param x borrowed [4,7,3] primal sequence
     * @param w borrowed [3,2] primal weights
     * @param u borrowed [2,2] primal recurrent weights
     * @param lengths borrowed validated [7] INT64 prefix lengths
     * @param fs borrowed [7,2] final seed
     * @param es borrowed [4,7,2] emission seed, padding never read
     * @param previous borrowed exact previous-carry history
     * @param saved borrowed exact represented activation history
     * @param dh distinct writable [7,2] state gradient
     * @param dx distinct writable [4,7,3] sequence gradient
     * @param dw distinct writable [3,2] parameter gradient
     * @param du distinct writable [2,2] parameter gradient
     * @param db distinct writable [2] bias gradient
     * @param work preallocated 16-cell incoming and bias-contribution scratch
     */
    static void reverseFloat32(MemorySegment x,MemorySegment w,MemorySegment u,MemorySegment lengths,
            MemorySegment fs,MemorySegment es,MemorySegment previous,MemorySegment saved,
            MemorySegment dh,MemorySegment dx,MemorySegment dw,MemorySegment du,MemorySegment db,
            MemorySegment work) {
        var layout = ValueLayout.JAVA_FLOAT_UNALIGNED;
        for (long p = 0; p < 14L*4; p += 4) dh.set(layout,p,fs.get(layout,p));
        for (long p = 0; p < 84L*4; p += 4) dx.set(layout,p,0f);
        for (long p = 0; p < 6L*4; p += 4) dw.set(layout,p,0f);
        for (long p = 0; p < 4L*4; p += 4) du.set(layout,p,0f);
        for (long p = 0; p < 2L*4; p += 4) db.set(layout,p,0f);
        long maximum = 0;
        for (int row = 0; row < 7; row++) maximum = Math.max(maximum,lengths.getAtIndex(ValueLayout.JAVA_LONG,row));
        for (int t = 3; t >= 0; t--) {
            if (t >= maximum) continue;
            for (int row = 0; row < 7; row++) if (t < lengths.getAtIndex(ValueLayout.JAVA_LONG,row))
                for (int col = 0; col < 2; col++) {
                    long p = (row*2L+col)*4;
                    float y = saved.get(layout,t*14L*4+p);
                    float incoming = dh.get(layout,p)+es.get(layout,t*14L*4+p);
                    work.set(layout,p,incoming*(1f-y*y));
                }
            for (int row = 0; row < 7; row++) if (t < lengths.getAtIndex(ValueLayout.JAVA_LONG,row))
                for (int col = 0; col < 2; col++) {
                    float sum = 0f; long a = row*2L*4, b = col*2L*4;
                    for (long k = 0; k < 2; k++) {
                        float left = work.get(layout,a), right = u.get(layout,b);
                        sum = sum+left*right; a += 4; b += 4;
                    }
                    dh.set(layout,(row*2L+col)*4,sum);
                }
            for (int feature = 0; feature < 3; feature++) for (int col = 0; col < 2; col++) {
                float sum = 0f; long a = (t*21L+feature)*4, b = col*(long)4;
                for (long row = 0; row < 7; row++) {
                    if (t < lengths.getAtIndex(ValueLayout.JAVA_LONG,row)) {
                        float left = x.get(layout,a), right = work.get(layout,b); sum = sum+left*right;
                    }
                    a += 3L*4; b += 2L*4;
                }
                long p = (feature*2L+col)*4; dw.set(layout,p,dw.get(layout,p)+sum);
            }
            for (int feature = 0; feature < 2; feature++) for (int col = 0; col < 2; col++) {
                float sum = 0f; long a = (t*14L+feature)*4, b = col*(long)4;
                for (long row = 0; row < 7; row++) {
                    if (t < lengths.getAtIndex(ValueLayout.JAVA_LONG,row)) {
                        float left = previous.get(layout,a), right = work.get(layout,b); sum = sum+left*right;
                    }
                    a += 2L*4; b += 2L*4;
                }
                long p = (feature*2L+col)*4; du.set(layout,p,du.get(layout,p)+sum);
            }
            for (int col = 0; col < 2; col++) {
                float sum = 0f;
                for (long row = 0; row < 7; row++) if (t < lengths.getAtIndex(ValueLayout.JAVA_LONG,row))
                    sum += work.get(layout,(row*2+col)*4);
                work.set(layout,(14L+col)*4,sum);
            }
            for (int col = 0; col < 2; col++) {
                long p = col*(long)4; db.set(layout,p,db.get(layout,p)+work.get(layout,(14L+col)*4));
            }
            for (int row = 0; row < 7; row++) if (t < lengths.getAtIndex(ValueLayout.JAVA_LONG,row))
                for (int feature = 0; feature < 3; feature++) {
                    float sum = 0f; long a = row*2L*4, b = feature*2L*4;
                    for (long k = 0; k < 2; k++) {
                        float left = work.get(layout,a), right = w.get(layout,b);
                        sum = sum+left*right; a += 4; b += 4;
                    }
                    dx.set(layout,(t*21L+row*3L+feature)*4,sum);
                }
        }
    }

    /**
     * Specialized static-length forward: separate product domain, recurrent/tail domain,
     * original-time histories, alternating old/next carry banks and raw inactive copies.
     * Constant byte strides precede time; only selected carrier/bank bases change per step.
     * @param initial borrowed [7,2] state
     * @param x borrowed [4,7,3] sequence
     * @param w borrowed [3,2] weights
     * @param u borrowed [2,2] recurrent weights
     * @param bias borrowed [2] bias
     * @param lengths borrowed prevalidated [7] INT64 prefix lengths
     * @param result distinct writable [7,2] final carry
     * @param emissions distinct writable [4,7,2] emissions
     * @param previous distinct writable original-time previous-carry history
     * @param saved distinct writable represented TANH history
     * @param work preallocated 42-cell product and two carry banks
     */
    static void forwardFloat64(MemorySegment initial,MemorySegment x,MemorySegment w,MemorySegment u,
            MemorySegment bias,MemorySegment lengths,MemorySegment result,MemorySegment emissions,
            MemorySegment previous,MemorySegment saved,MemorySegment work) {
        var layout = ValueLayout.JAVA_DOUBLE_UNALIGNED;
        for (long p = 0; p < 56L*8; p += 8) {
            emissions.set(layout,p,0d); previous.set(layout,p,0d); saved.set(layout,p,0d);
        }
        long maximum = 0;
        for (int row = 0; row < 7; row++) maximum = Math.max(maximum,lengths.getAtIndex(ValueLayout.JAVA_LONG,row));
        for (int t = 0; t < 4; t++) {
            MemorySegment old = t == 0 ? initial : work;
            long oldBase = t == 0 ? 0 : ((t-1)&1)*14L*8, nextBase = (t&1)*14L*8;
            if (t < maximum) {
                for (int row = 0; row < 7; row++) if (t < lengths.getAtIndex(ValueLayout.JAVA_LONG,row))
                    for (int col = 0; col < 2; col++) {
                        long p = (row*2L+col)*8;
                        previous.set(layout,(t*14L)*8+p,old.get(layout,oldBase+p));
                    }
                for (int row = 0; row < 7; row++) if (t < lengths.getAtIndex(ValueLayout.JAVA_LONG,row))
                    for (int col = 0; col < 2; col++) {
                        double sum = 0d;
                        long a = (t*21L+row*3L)*8, b = col*(long)8;
                        for (long k = 0; k < 3; k++) {
                            double left = x.get(layout,a), right = w.get(layout,b);
                            sum = sum+left*right; a += 8; b += 2L*8;
                        }
                        work.set(layout,(28L+row*2L+col)*8,sum);
                    }
                for (int row = 0; row < 7; row++) if (t < lengths.getAtIndex(ValueLayout.JAVA_LONG,row))
                    for (int col = 0; col < 2; col++) {
                        double sum = 0d;
                        long a = oldBase+row*2L*8, b = col*(long)8;
                        for (long k = 0; k < 2; k++) {
                            double left = old.get(layout,a), right = u.get(layout,b);
                            sum = sum+left*right; a += 8; b += 2L*8;
                        }
                        double next = Math.tanh((work.get(layout,(28L+row*2L+col)*8)+sum)+bias.get(layout,col*(long)8));
                        emissions.set(layout,(t*14L+row*2L+col)*8,next);
                    }
                for (int row = 0; row < 7; row++) if (t < lengths.getAtIndex(ValueLayout.JAVA_LONG,row))
                    for (int col = 0; col < 2; col++) {
                        long p = (t*14L+row*2L+col)*8;
                        saved.set(layout,p,emissions.get(layout,p));
                    }
            }
            for (int row = 0; row < 7; row++) for (int col = 0; col < 2; col++) {
                long p = (row*2L+col)*8;
                work.set(layout,nextBase+p,t < lengths.getAtIndex(ValueLayout.JAVA_LONG,row)
                        ? emissions.get(layout,t*14L*8+p) : old.get(layout,oldBase+p));
            }
        }
        for (long p = 0; p < 14L*8; p += 8) result.set(layout,p,work.get(layout,14L*8+p));
    }

    /**
     * Actual-sibling specialized static-length adjoint with incoming scratch, then dH, dW,
     * dU, bias SUM/ADD, and dX domains in the declared stage order. Active K branches precede
     * both operand loads; scalar SUM remains distinct from ordinary exact SUM.
     * @param x borrowed [4,7,3] primal sequence
     * @param w borrowed [3,2] primal weights
     * @param u borrowed [2,2] primal recurrent weights
     * @param lengths borrowed validated [7] INT64 prefix lengths
     * @param fs borrowed [7,2] final seed
     * @param es borrowed [4,7,2] emission seed, padding never read
     * @param previous borrowed exact previous-carry history
     * @param saved borrowed exact represented activation history
     * @param dh distinct writable [7,2] state gradient
     * @param dx distinct writable [4,7,3] sequence gradient
     * @param dw distinct writable [3,2] parameter gradient
     * @param du distinct writable [2,2] parameter gradient
     * @param db distinct writable [2] bias gradient
     * @param work preallocated 16-cell incoming and bias-contribution scratch
     */
    static void reverseFloat64(MemorySegment x,MemorySegment w,MemorySegment u,MemorySegment lengths,
            MemorySegment fs,MemorySegment es,MemorySegment previous,MemorySegment saved,
            MemorySegment dh,MemorySegment dx,MemorySegment dw,MemorySegment du,MemorySegment db,
            MemorySegment work) {
        var layout = ValueLayout.JAVA_DOUBLE_UNALIGNED;
        for (long p = 0; p < 14L*8; p += 8) dh.set(layout,p,fs.get(layout,p));
        for (long p = 0; p < 84L*8; p += 8) dx.set(layout,p,0d);
        for (long p = 0; p < 6L*8; p += 8) dw.set(layout,p,0d);
        for (long p = 0; p < 4L*8; p += 8) du.set(layout,p,0d);
        for (long p = 0; p < 2L*8; p += 8) db.set(layout,p,0d);
        long maximum = 0;
        for (int row = 0; row < 7; row++) maximum = Math.max(maximum,lengths.getAtIndex(ValueLayout.JAVA_LONG,row));
        for (int t = 3; t >= 0; t--) {
            if (t >= maximum) continue;
            for (int row = 0; row < 7; row++) if (t < lengths.getAtIndex(ValueLayout.JAVA_LONG,row))
                for (int col = 0; col < 2; col++) {
                    long p = (row*2L+col)*8;
                    double y = saved.get(layout,t*14L*8+p);
                    double incoming = dh.get(layout,p)+es.get(layout,t*14L*8+p);
                    work.set(layout,p,incoming*(1d-y*y));
                }
            for (int row = 0; row < 7; row++) if (t < lengths.getAtIndex(ValueLayout.JAVA_LONG,row))
                for (int col = 0; col < 2; col++) {
                    double sum = 0d; long a = row*2L*8, b = col*2L*8;
                    for (long k = 0; k < 2; k++) {
                        double left = work.get(layout,a), right = u.get(layout,b);
                        sum = sum+left*right; a += 8; b += 8;
                    }
                    dh.set(layout,(row*2L+col)*8,sum);
                }
            for (int feature = 0; feature < 3; feature++) for (int col = 0; col < 2; col++) {
                double sum = 0d; long a = (t*21L+feature)*8, b = col*(long)8;
                for (long row = 0; row < 7; row++) {
                    if (t < lengths.getAtIndex(ValueLayout.JAVA_LONG,row)) {
                        double left = x.get(layout,a), right = work.get(layout,b); sum = sum+left*right;
                    }
                    a += 3L*8; b += 2L*8;
                }
                long p = (feature*2L+col)*8; dw.set(layout,p,dw.get(layout,p)+sum);
            }
            for (int feature = 0; feature < 2; feature++) for (int col = 0; col < 2; col++) {
                double sum = 0d; long a = (t*14L+feature)*8, b = col*(long)8;
                for (long row = 0; row < 7; row++) {
                    if (t < lengths.getAtIndex(ValueLayout.JAVA_LONG,row)) {
                        double left = previous.get(layout,a), right = work.get(layout,b); sum = sum+left*right;
                    }
                    a += 2L*8; b += 2L*8;
                }
                long p = (feature*2L+col)*8; du.set(layout,p,du.get(layout,p)+sum);
            }
            for (int col = 0; col < 2; col++) {
                double sum = 0d;
                for (long row = 0; row < 7; row++) if (t < lengths.getAtIndex(ValueLayout.JAVA_LONG,row))
                    sum += work.get(layout,(row*2+col)*8);
                work.set(layout,(14L+col)*8,sum);
            }
            for (int col = 0; col < 2; col++) {
                long p = col*(long)8; db.set(layout,p,db.get(layout,p)+work.get(layout,(14L+col)*8));
            }
            for (int row = 0; row < 7; row++) if (t < lengths.getAtIndex(ValueLayout.JAVA_LONG,row))
                for (int feature = 0; feature < 3; feature++) {
                    double sum = 0d; long a = row*2L*8, b = feature*2L*8;
                    for (long k = 0; k < 2; k++) {
                        double left = work.get(layout,a), right = w.get(layout,b);
                        sum = sum+left*right; a += 8; b += 8;
                    }
                    dx.set(layout,(t*21L+row*3L+feature)*8,sum);
                }
        }
    }
}
