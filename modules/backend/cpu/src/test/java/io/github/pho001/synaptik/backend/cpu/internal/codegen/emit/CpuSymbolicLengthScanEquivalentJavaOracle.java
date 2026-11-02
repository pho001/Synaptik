package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import java.lang.foreign.MemorySegment;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED;
import static java.lang.foreign.ValueLayout.JAVA_LONG_UNALIGNED;

/**
 * Executable F3/H2 reverse reference with invocation time, batch and actual scalar operands.
 * This is direct typed Java, not generated-source reconstruction or a graph interpreter.
 * Saved forward primals supply every derivative; there is no activation replay. Only the
 * local adjoint crosses row loops, in a caller-owned 2*B float bank. Parameter contributions
 * use scalar accumulators, never a batch-by-parameter bank. All accesses are checked native-order
 * unaligned byte accesses, matching the generated entry rather than the retained literal oracle.
 */
final class CpuSymbolicLengthScanEquivalentJavaOracle {
    /** Prevents construction of the stateless test-only reference. */
    private CpuSymbolicLengthScanEquivalentJavaOracle() {}

    /**
     * Computes the actual first-order reverse algorithm for a prevalidated F32/F3/H2 invocation.
     * Each float expression is a represented producing site: seed ADD, saved-output square,
     * actual-scalar SUB, adjoint MUL, sequential product/ADD contractions and distinct per-time
     * parameter carry ADD. A B=1 bias SUM moves its sole represented contributor raw; B>1
     * starts at positive zero even if only one row is active. Inactive row carry is untouched,
     * inactive dX is raw positive zero, and inactive time performs no body arithmetic or reads.
     * Nonempty parameter outputs initialize from actual scalar splats even at T=0 or B=0.
     *
     * <p>All segments are non-null and borrowed for this call. Inputs are stable and do not
     * overlap any writable output or work bank. Output/work segments are writable, mutually
     * disjoint and owned by the caller. Geometry and lengths have already passed production
     * preflight; this method is a kernel reference, not a replacement admission protocol.
     * Byte products must be representable and row/feature domains int-bounded, as at generated
     * entry. The method has no shared mutable state; separate writable banks permit concurrency.
     *
     * @param x canonical [T,B,3] primal sequence; only active original-time cells are read
     * @param w canonical [3,2] input weights, read only during active body work
     * @param u canonical [2,2] recurrent weights, read only during active body work
     * @param lengths prevalidated INT64[B] lengths in [0,T], unchanged during this call
     * @param fs canonical [B,2] final-carry seed, copied raw before any reverse step
     * @param es canonical [T,B,2] joint emission seed; padding is never read
     * @param previous exact represented [T,B,2] previous-carry history from the forward scan
     * @param saved exact represented [T,B,2] TANH output history, not replayed activations
     * @param initialDW actual one-float splat initializing every [3,2] dW cell
     * @param initialDU actual one-float splat initializing every [2,2] dU cell
     * @param initialDB actual one-float splat initializing every [2] db cell
     * @param derivative actual one-float operand of the derivative SUB, read only if any time is active
     * @param dh distinct canonical writable [B,2] dInitialCarry result
     * @param dx distinct canonical writable [T,B,3] dSequence result
     * @param dw distinct canonical writable [3,2] accumulated input-weight cotangent
     * @param du distinct canonical writable [2,2] accumulated recurrent-weight cotangent
     * @param db distinct canonical writable [2] accumulated bias cotangent
     * @param work distinct writable local-adjoint bank with at least 8*B bytes
     * @param timeCount nonnegative runtime T from this invocation's numeric pack; not a baked literal
     * @param batch nonnegative invocation B within the admitted int-bounded geometry
     * @param sourceReverse whether the original scan decreases time; reverse visits the opposite order
     * @throws IndexOutOfBoundsException if a consumed segment is shorter than its declared span
     * @throws IllegalStateException if a consumed segment's scope is closed
     * @throws IllegalArgumentException if a consumed output or work segment is read-only
     * @throws WrongThreadException if a consumed segment is confined to another thread
     */
    static void reverseFloat32(MemorySegment x, MemorySegment w, MemorySegment u,
            MemorySegment lengths, MemorySegment fs, MemorySegment es,
            MemorySegment previous, MemorySegment saved, MemorySegment initialDW,
            MemorySegment initialDU, MemorySegment initialDB, MemorySegment derivative,
            MemorySegment dh, MemorySegment dx, MemorySegment dw, MemorySegment du,
            MemorySegment db, MemorySegment work, long timeCount, int batch, boolean sourceReverse) {
        long rowBytes = 8L * batch, sequenceBytes = 12L * batch;
        for (long address = 0; address < rowBytes; address += 4)
            dh.set(JAVA_FLOAT_UNALIGNED, address, fs.get(JAVA_FLOAT_UNALIGNED, address));
        for (long address = 0, limit = timeCount * sequenceBytes; address < limit; address += 4)
            dx.set(JAVA_FLOAT_UNALIGNED, address, 0f);
        float startW = initialDW.get(JAVA_FLOAT_UNALIGNED, 0);
        for (long address = 0; address < 24; address += 4) dw.set(JAVA_FLOAT_UNALIGNED, address, startW);
        float startU = initialDU.get(JAVA_FLOAT_UNALIGNED, 0);
        for (long address = 0; address < 16; address += 4) du.set(JAVA_FLOAT_UNALIGNED, address, startU);
        float startB = initialDB.get(JAVA_FLOAT_UNALIGNED, 0);
        for (long address = 0; address < 8; address += 4) db.set(JAVA_FLOAT_UNALIGNED, address, startB);
        if (timeCount == 0) return;
        long maximum = 0;
        for (int row = 0; row < batch; row++)
            maximum = Math.max(maximum, lengths.get(JAVA_LONG_UNALIGNED, 8L * row));
        if (maximum == 0) return;
        float operand = derivative.get(JAVA_FLOAT_UNALIGNED, 0);
        for (long step = 0; step < timeCount; step++) {
            long time = sourceReverse ? step : timeCount - 1 - step;
            if (time >= maximum) continue;
            long rowFrame = time * rowBytes, sequenceFrame = time * sequenceBytes;
            // All old-dh reads precede all next-dh writes, including cross-feature contractions.
            for (int row = 0; row < batch; row++) if (time < lengths.get(JAVA_LONG_UNALIGNED, 8L * row)) {
                long rowBase = 8L * row, historyBase = rowFrame + rowBase;
                for (int col = 0; col < 2; col++) {
                    long feature = 4L * col;
                    float seed = dh.get(JAVA_FLOAT_UNALIGNED, rowBase + feature)
                            + es.get(JAVA_FLOAT_UNALIGNED, historyBase + feature);
                    float y = saved.get(JAVA_FLOAT_UNALIGNED, historyBase + feature);
                    float square = y * y;
                    float slope = operand - square;
                    work.set(JAVA_FLOAT_UNALIGNED, rowBase + feature, seed * slope);
                }
            }
            for (int row = 0; row < batch; row++) if (time < lengths.get(JAVA_LONG_UNALIGNED, 8L * row)) {
                for (int k = 0; k < 3; k++) {
                    float sum = 0f;
                    for (int col = 0; col < 2; col++)
                        sum += work.get(JAVA_FLOAT_UNALIGNED, 8L * row + 4L * col)
                                * w.get(JAVA_FLOAT_UNALIGNED, 8L * k + 4L * col);
                    dx.set(JAVA_FLOAT_UNALIGNED, sequenceFrame + 12L * row + 4L * k, sum);
                }
            }
            for (int k = 0; k < 3; k++) for (int col = 0; col < 2; col++) {
                float contribution = 0f;
                for (int row = 0; row < batch; row++) if (time < lengths.get(JAVA_LONG_UNALIGNED, 8L * row))
                    contribution += x.get(JAVA_FLOAT_UNALIGNED, sequenceFrame + 12L * row + 4L * k)
                            * work.get(JAVA_FLOAT_UNALIGNED, 8L * row + 4L * col);
                long address = 8L * k + 4L * col;
                dw.set(JAVA_FLOAT_UNALIGNED, address, dw.get(JAVA_FLOAT_UNALIGNED, address) + contribution);
            }
            for (int k = 0; k < 2; k++) for (int col = 0; col < 2; col++) {
                float contribution = 0f;
                for (int row = 0; row < batch; row++) if (time < lengths.get(JAVA_LONG_UNALIGNED, 8L * row))
                    contribution += previous.get(JAVA_FLOAT_UNALIGNED, rowFrame + 8L * row + 4L * k)
                            * work.get(JAVA_FLOAT_UNALIGNED, 8L * row + 4L * col);
                long address = 8L * k + 4L * col;
                du.set(JAVA_FLOAT_UNALIGNED, address, du.get(JAVA_FLOAT_UNALIGNED, address) + contribution);
            }
            // Domain-one SUM is raw movement, not +0 plus the sole contributor.
            if (batch == 1) {
                for (int col = 0; col < 2; col++) {
                    long address = 4L * col;
                    float contribution = work.get(JAVA_FLOAT_UNALIGNED, address);
                    db.set(JAVA_FLOAT_UNALIGNED, address, db.get(JAVA_FLOAT_UNALIGNED, address) + contribution);
                }
            } else {
                for (int col = 0; col < 2; col++) {
                    float contribution = 0f;
                    for (int row = 0; row < batch; row++) if (time < lengths.get(JAVA_LONG_UNALIGNED, 8L * row))
                        contribution += work.get(JAVA_FLOAT_UNALIGNED, 8L * row + 4L * col);
                    long address = 4L * col;
                    db.set(JAVA_FLOAT_UNALIGNED, address, db.get(JAVA_FLOAT_UNALIGNED, address) + contribution);
                }
            }
            for (int row = 0; row < batch; row++) if (time < lengths.get(JAVA_LONG_UNALIGNED, 8L * row)) {
                for (int k = 0; k < 2; k++) {
                    float sum = 0f;
                    for (int col = 0; col < 2; col++)
                        sum += work.get(JAVA_FLOAT_UNALIGNED, 8L * row + 4L * col)
                                * u.get(JAVA_FLOAT_UNALIGNED, 8L * k + 4L * col);
                    dh.set(JAVA_FLOAT_UNALIGNED, 8L * row + 4L * k, sum);
                }
            }
        }
    }
}
