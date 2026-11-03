package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Specialized clean-Java algorithm/dataflow oracles for next=emit=tanh(h+x) and its actual
 * first-order reverse, broadcast-parameter reverse of next=emit=h+x*w, and the rank-two matrix
 * forward/actual-reverse pair next=emit=h+x@w. Every carrier is a
 * MemorySegment, including heap inputs. There is no primitive-array fast path, replay, initial
 * staging or per-step allocation. The pointwise cases reuse demanded histories as next-time
 * backing; the broadcast reverse needs simultaneous old/new banks across its reduction.
 * Distinct publications remain distinct in both algorithms.
 * The FLOAT64 methods provide direct-double references with matched represented sites and
 * necessary storage; structural comparison and short timings do not establish JIT-tier or
 * performance parity. These are test oracles, not production scan executors.
 */
public final class CpuGeneralScanCleanJavaOracle {
    /**
     * Specialized length forward B3/F8: raw initial copy, +0 padding, then active MUL/ADD.
     * Segments are non-null, live, accessible and preflighted; writable outputs are disjoint.
     * No replay, parameter bank or per-step allocation is used.
     * @param initial borrowed 24-cell initial state
     * @param sequence borrowed T*24-cell sequence; padded cells never read
     * @param weight borrowed eight-cell parameter
     * @param lengths borrowed INT64[3] values in [0,T]
     * @param result distinct writable final state
     * @param emissions distinct writable T*24-cell output
     * @param geometry borrowed geometry, T at zero, unchanged
     * @param start unused matched entry bound
     * @param end unused matched entry bound
     */
    public static void lengthForwardFloat32B3F8(MemorySegment initial,MemorySegment sequence,
            MemorySegment weight,MemorySegment lengths,MemorySegment result,MemorySegment emissions,
            long[] geometry,long start,long end) {
        long tCount = geometry[0], maximum = 0;
        for (long at = 0; at < 24L*4; at += 4) result.set(ValueLayout.JAVA_FLOAT_UNALIGNED,at,initial.get(ValueLayout.JAVA_FLOAT_UNALIGNED,at));
        for (long at = 0; at < tCount*24L*4; at += 4) emissions.set(ValueLayout.JAVA_FLOAT_UNALIGNED,at,0f);
        for (int b = 0; b < 3; b++) maximum = Math.max(maximum,lengths.get(ValueLayout.JAVA_LONG_UNALIGNED,b*8L));
        for (long t = 0; t < tCount; t++) {
            if (t >= maximum) continue;
            for (int b = 0; b < 3; b++) {
                if (t >= lengths.get(ValueLayout.JAVA_LONG_UNALIGNED,b*8L)) continue;
                for (int j = 0; j < 8; j++) {
                    long at = (b*8L+j)*4, stacked = (t*24+b*8L+j)*4;
                    float product = sequence.get(ValueLayout.JAVA_FLOAT_UNALIGNED,stacked)*weight.get(ValueLayout.JAVA_FLOAT_UNALIGNED,j*4L);
                    float next = result.get(ValueLayout.JAVA_FLOAT_UNALIGNED,at)+product;
                    emissions.set(ValueLayout.JAVA_FLOAT_UNALIGNED,stacked,next);
                    result.set(ValueLayout.JAVA_FLOAT_UNALIGNED,at,next);
                }
            }
        }
    }

    /**
     * Specialized actual reverse B3/F8 with joint seeds and active contributors in increasing b.
     * Each typed contribution sum starts at zero, followed by ONE distinct shared ADD per time.
     * Segments are live/preflighted, outputs disjoint; no B-by-parameter bank or per-step allocation.
     * @param finalSeed borrowed 24-cell final cotangent
     * @param initialParameterSum borrowed scalar initializer for shared state
     * @param sequence borrowed T*24-cell exact primal sequence
     * @param emissionSeed borrowed T*24-cell seeds; padded cells never read
     * @param weight borrowed eight-cell exact parameter
     * @param lengths borrowed INT64[3] admitted lengths
     * @param dInitial distinct writable 24-cell row state
     * @param dWeight distinct writable eight-cell shared state
     * @param dSequence distinct writable T*24-cell output
     * @param geometry borrowed geometry, T at zero, unchanged
     * @param start unused matched entry bound
     * @param end unused matched entry bound
     */
    public static void lengthReverseFloat32B3F8(MemorySegment finalSeed,MemorySegment initialParameterSum,
            MemorySegment sequence,MemorySegment emissionSeed,MemorySegment weight,MemorySegment lengths,
            MemorySegment dInitial,MemorySegment dWeight,MemorySegment dSequence,long[] geometry,long start,long end) {
        long tCount = geometry[0], maximum = 0;
        for (long at = 0; at < 24L*4; at += 4) dInitial.set(ValueLayout.JAVA_FLOAT_UNALIGNED,at,finalSeed.get(ValueLayout.JAVA_FLOAT_UNALIGNED,at));
        for (int j = 0; j < 8; j++) dWeight.set(ValueLayout.JAVA_FLOAT_UNALIGNED,j*4L,initialParameterSum.get(ValueLayout.JAVA_FLOAT_UNALIGNED,0));
        for (long at = 0; at < tCount*24L*4; at += 4) dSequence.set(ValueLayout.JAVA_FLOAT_UNALIGNED,at,0f);
        for (int b = 0; b < 3; b++) maximum = Math.max(maximum,lengths.get(ValueLayout.JAVA_LONG_UNALIGNED,b*8L));
        for (long t = tCount-1; t >= 0; t--) {
            if (t >= maximum) continue;
            for (int j = 0; j < 8; j++) {
                float sum = 0f;
                for (int b = 0; b < 3; b++) {
                    if (t >= lengths.get(ValueLayout.JAVA_LONG_UNALIGNED,b*8L)) continue;
                    long at = (b*8L+j)*4, stacked = (t*24+b*8L+j)*4;
                    float incoming = dInitial.get(ValueLayout.JAVA_FLOAT_UNALIGNED,at)+emissionSeed.get(ValueLayout.JAVA_FLOAT_UNALIGNED,stacked);
                    dInitial.set(ValueLayout.JAVA_FLOAT_UNALIGNED,at,incoming);
                    float contribution = incoming*sequence.get(ValueLayout.JAVA_FLOAT_UNALIGNED,stacked);
                    float dx = incoming*weight.get(ValueLayout.JAVA_FLOAT_UNALIGNED,j*4L);
                    dSequence.set(ValueLayout.JAVA_FLOAT_UNALIGNED,stacked,dx);
                    sum += contribution;
                }
                float next = dWeight.get(ValueLayout.JAVA_FLOAT_UNALIGNED,j*4L)+sum;
                dWeight.set(ValueLayout.JAVA_FLOAT_UNALIGNED,j*4L,next);
            }
        }
    }

    /**
     * Specialized length forward B3/F8: raw initial copy, +0 padding, then active MUL/ADD.
     * Segments are non-null, live, accessible and preflighted; writable outputs are disjoint.
     * No replay, parameter bank or per-step allocation is used.
     * @param initial borrowed 24-cell initial state
     * @param sequence borrowed T*24-cell sequence; padded cells never read
     * @param weight borrowed eight-cell parameter
     * @param lengths borrowed INT64[3] values in [0,T]
     * @param result distinct writable final state
     * @param emissions distinct writable T*24-cell output
     * @param geometry borrowed geometry, T at zero, unchanged
     * @param start unused matched entry bound
     * @param end unused matched entry bound
     */
    public static void lengthForwardFloat64B3F8(MemorySegment initial,MemorySegment sequence,
            MemorySegment weight,MemorySegment lengths,MemorySegment result,MemorySegment emissions,
            long[] geometry,long start,long end) {
        long tCount = geometry[0], maximum = 0;
        for (long at = 0; at < 24L*8; at += 8) result.set(ValueLayout.JAVA_DOUBLE_UNALIGNED,at,initial.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,at));
        for (long at = 0; at < tCount*24L*8; at += 8) emissions.set(ValueLayout.JAVA_DOUBLE_UNALIGNED,at,0d);
        for (int b = 0; b < 3; b++) maximum = Math.max(maximum,lengths.get(ValueLayout.JAVA_LONG_UNALIGNED,b*8L));
        for (long t = 0; t < tCount; t++) {
            if (t >= maximum) continue;
            for (int b = 0; b < 3; b++) {
                if (t >= lengths.get(ValueLayout.JAVA_LONG_UNALIGNED,b*8L)) continue;
                for (int j = 0; j < 8; j++) {
                    long at = (b*8L+j)*8, stacked = (t*24+b*8L+j)*8;
                    double product = sequence.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,stacked)*weight.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,j*8L);
                    double next = result.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,at)+product;
                    emissions.set(ValueLayout.JAVA_DOUBLE_UNALIGNED,stacked,next);
                    result.set(ValueLayout.JAVA_DOUBLE_UNALIGNED,at,next);
                }
            }
        }
    }

    /**
     * Specialized actual reverse B3/F8 with joint seeds and active contributors in increasing b.
     * Each typed contribution sum starts at zero, followed by ONE distinct shared ADD per time.
     * Segments are live/preflighted, outputs disjoint; no B-by-parameter bank or per-step allocation.
     * @param finalSeed borrowed 24-cell final cotangent
     * @param initialParameterSum borrowed scalar initializer for shared state
     * @param sequence borrowed T*24-cell exact primal sequence
     * @param emissionSeed borrowed T*24-cell seeds; padded cells never read
     * @param weight borrowed eight-cell exact parameter
     * @param lengths borrowed INT64[3] admitted lengths
     * @param dInitial distinct writable 24-cell row state
     * @param dWeight distinct writable eight-cell shared state
     * @param dSequence distinct writable T*24-cell output
     * @param geometry borrowed geometry, T at zero, unchanged
     * @param start unused matched entry bound
     * @param end unused matched entry bound
     */
    public static void lengthReverseFloat64B3F8(MemorySegment finalSeed,MemorySegment initialParameterSum,
            MemorySegment sequence,MemorySegment emissionSeed,MemorySegment weight,MemorySegment lengths,
            MemorySegment dInitial,MemorySegment dWeight,MemorySegment dSequence,long[] geometry,long start,long end) {
        long tCount = geometry[0], maximum = 0;
        for (long at = 0; at < 24L*8; at += 8) dInitial.set(ValueLayout.JAVA_DOUBLE_UNALIGNED,at,finalSeed.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,at));
        for (int j = 0; j < 8; j++) dWeight.set(ValueLayout.JAVA_DOUBLE_UNALIGNED,j*8L,initialParameterSum.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,0));
        for (long at = 0; at < tCount*24L*8; at += 8) dSequence.set(ValueLayout.JAVA_DOUBLE_UNALIGNED,at,0d);
        for (int b = 0; b < 3; b++) maximum = Math.max(maximum,lengths.get(ValueLayout.JAVA_LONG_UNALIGNED,b*8L));
        for (long t = tCount-1; t >= 0; t--) {
            if (t >= maximum) continue;
            for (int j = 0; j < 8; j++) {
                double sum = 0d;
                for (int b = 0; b < 3; b++) {
                    if (t >= lengths.get(ValueLayout.JAVA_LONG_UNALIGNED,b*8L)) continue;
                    long at = (b*8L+j)*8, stacked = (t*24+b*8L+j)*8;
                    double incoming = dInitial.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,at)+emissionSeed.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,stacked);
                    dInitial.set(ValueLayout.JAVA_DOUBLE_UNALIGNED,at,incoming);
                    double contribution = incoming*sequence.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,stacked);
                    double dx = incoming*weight.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,j*8L);
                    dSequence.set(ValueLayout.JAVA_DOUBLE_UNALIGNED,stacked,dx);
                    sum += contribution;
                }
                double next = dWeight.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,j*8L)+sum;
                dWeight.set(ValueLayout.JAVA_DOUBLE_UNALIGNED,j*8L,next);
            }
        }
    }
    /**
     * Matched generated-entry reference for {@code next=emit=h+xStep*w[8]} at static B=4/H=8.
     * Time is borrowed as a long from geometry at every invocation; start/end are unused, as in
     * the shaped generated entry. Each multiply precedes its represented add. Emissions back
     * the next step, and final publication copies all 32 cells, including raw initial bits at T=0.
     * All segments are borrowed, non-null, live and accessible to the caller, with disjoint
     * outputs; this reference neither closes them nor validates bindings. Access uses unaligned
     * native-order FP64 layouts, as in the generated entry. T*256 must fit a nonnegative long.
     * No input staging, primitive-array value path, or per-step allocation is used. The generic
     * {@link #broadcastForwardFloat64(MemorySegment, MemorySegment, MemorySegment, MemorySegment,
     * MemorySegment, int, int, int)} reference remains a separate, geometry-parameterized oracle.
     * @param initial borrowed canonical 32-cell FP64 initial carry
     * @param sequence borrowed canonical T*32-cell FP64 sequence
     * @param weight borrowed canonical eight-cell FP64 broadcast parameter
     * @param result distinct writable 32-cell FP64 final publication
     * @param emissions distinct writable T*32-cell FP64 stacked frames and next-step backing
     * @param geometry borrowed non-null array of length at least one, unchanged; nonnegative long T is at index zero
     * @param start unused long entry bound, retained for ABI equivalence
     * @param end unused long entry bound, retained for ABI equivalence
     * @throws NullPointerException if geometry or a segment used by an access is null
     * @throws IndexOutOfBoundsException if geometry has no first element or a segment is too small for an access
     * @throws IllegalArgumentException if an output receiving a write is read-only
     * @throws IllegalStateException if an accessed segment's scope is no longer alive
     * @throws WrongThreadException if a confined segment is accessed outside its owning thread
     */
    public static void broadcastForwardFloat64B4H8(MemorySegment initial, MemorySegment sequence, MemorySegment weight,
            MemorySegment result, MemorySegment emissions, long[] geometry, long start, long end) {
        var layout = ValueLayout.JAVA_DOUBLE_UNALIGNED;
        long time = geometry[0];
        MemorySegment old = initial; long oldBase = 0;
        for (long t = 0; t < time; t++) {
            long frame = t*256L;
            for (int b = 0; b < 4; b++) for (int j = 0; j < 8; j++) {
                long cell = b*64L+j*8L;
                double product = sequence.get(layout,frame+cell)*weight.get(layout,j*8L);
                double next = old.get(layout,oldBase+cell)+product;
                emissions.set(layout,frame+cell,next);
            }
            old = emissions; oldBase = frame;
        }
        for (long p = 0; p < 256L; p += 8) result.set(layout,p,old.get(layout,oldBase+p));
    }

    /**
     * Direct-double forward of {@code next=emit=h+x*w[H]}, retaining each multiply/add site.
     * @param initial borrowed dense B*H initial carry
     * @param sequence borrowed original-time T*B*H steps
     * @param weight borrowed H-cell broadcast parameter
     * @param result distinct dense final publication
     * @param emissions demanded original-time next frames, also next-step backing
     * @param time positive time count
     * @param batch positive B extent
     * @param width positive H extent
     */
    public static void broadcastForwardFloat64(MemorySegment initial, MemorySegment sequence, MemorySegment weight,
            MemorySegment result, MemorySegment emissions, int time, int batch, int width) {
        var layout = ValueLayout.JAVA_DOUBLE_UNALIGNED; long bytes = batch*(long)width*8;
        MemorySegment old = initial; long oldBase = 0;
        for (int t = 0; t < time; t++) {
            long frame = t*bytes;
            for (int b = 0; b < batch; b++) for (int j = 0; j < width; j++) {
                long cell = (b*(long)width+j)*8;
                double product = sequence.get(layout,frame+cell)*weight.get(layout,j*8L);
                double next = old.get(layout,oldBase+cell)+product;
                emissions.set(layout,frame+cell,next);
            }
            old = emissions; oldBase = frame;
        }
        for (long p = 0; p < bytes; p += 8) result.set(layout,p,emissions.get(layout,oldBase+p));
    }
    /**
     * Specialized scalar reference for {@code next=emit=h+x@w}. One F64 dot accumulator feeds
     * the represented ADD directly; the demanded emission is next-step carry backing. No product
     * array, input staging, per-step allocation or graph dispatch is needed.
     * @param initial borrowed dense B*N initial carry
     * @param sequence borrowed original-time T*B*K sequence
     * @param weight borrowed dense K*N parameter
     * @param result distinct B*N final publication
     * @param emissions demanded T*B*N next-value frames
     * @param time positive step count
     * @param rows positive B extent
     * @param contraction positive K extent
     * @param columns positive N extent
     */
    public static void matmulForwardFloat64(MemorySegment initial, MemorySegment sequence, MemorySegment weight,
            MemorySegment result, MemorySegment emissions, int time, int rows, int contraction, int columns) {
        var layout = ValueLayout.JAVA_DOUBLE_UNALIGNED;
        long frame = rows*(long)columns*8, inputFrame = rows*(long)contraction*8;
        MemorySegment old = initial; long oldBase = 0;
        for (int t = 0; t < time; t++) {
            long outBase = t*frame;
            for (int row = 0; row < rows; row++) for (int column = 0; column < columns; column++) {
                double sum = 0;
                long a = t*inputFrame+row*(long)contraction*8, b = column*8L;
                if (contraction == 1) sum = sequence.get(layout,a)*weight.get(layout,b);
                else for (int k = 0; k < contraction; k++) {
                    double left = sequence.get(layout,a), right = weight.get(layout,b);
                    sum = sum+left*right; a += 8; b += columns*8L;
                }
                long cell = (row*(long)columns+column)*8;
                double next = old.get(layout,oldBase+cell)+sum;
                emissions.set(layout,outBase+cell,next);
            }
            old = emissions; oldBase = outBase;
        }
        for (long p = 0; p < frame; p += 8) result.set(layout,p,emissions.get(layout,oldBase+p));
    }

    /**
     * Specialized actual first-order reverse of {@code next=emit=h+x@w}. Incoming cotangents
     * must materialize once because both matrix products read them across coordinates. Separate
     * old/new banks retain the complete carry and K*N parameter sum. Each dot feeds its sole
     * represented result/addition directly; no per-step allocation or temporary product exists.
     * @param finalSeed borrowed B*N final seed
     * @param emissionSeed borrowed original-time T*B*N seeds
     * @param sequence borrowed primal T*B*K sequence
     * @param weight borrowed primal K*N parameter
     * @param zero borrowed single-cell parameter-sum initializer
     * @param initialGradient distinct B*N final carry-gradient publication
     * @param weightGradient distinct K*N parameter-gradient publication
     * @param sequenceGradient demanded T*B*K matrix-gradient frames
     * @param scratch run-owned 2*(B*N+K*N)*8-byte old/new banks
     * @param time positive step count
     * @param rows positive B extent
     * @param contraction positive K extent
     * @param columns positive N extent
     */
    public static void matmulReverseFloat64(MemorySegment finalSeed, MemorySegment emissionSeed, MemorySegment sequence,
            MemorySegment weight, MemorySegment zero, MemorySegment initialGradient, MemorySegment weightGradient,
            MemorySegment sequenceGradient, MemorySegment scratch, int time, int rows, int contraction, int columns) {
        var layout = ValueLayout.JAVA_DOUBLE_UNALIGNED;
        long frame = rows*(long)columns*8, parameter = contraction*(long)columns*8, inputFrame = rows*(long)contraction*8;
        long lastCarry = 0, lastParameter = 0;
        for (int step = 0; step < time; step++) {
            int t = time-1-step;
            MemorySegment old = step == 0 ? finalSeed : scratch, oldParameter = step == 0 ? zero : scratch;
            long oldBase = step == 0 ? 0 : ((step-1)&1)*frame;
            long parameterBase = step == 0 ? 0 : 2*frame+((step-1)&1)*parameter;
            long next = (step&1)*frame, nextParameter = 2*frame+(step&1)*parameter;
            for (long p = 0; p < frame; p += 8) {
                double incoming = old.get(layout,oldBase+p)+emissionSeed.get(layout,t*frame+p);
                scratch.set(layout,next+p,incoming);
            }
            for (int k = 0; k < contraction; k++) for (int n = 0; n < columns; n++) {
                double sum = 0; long a = t*inputFrame+k*8L, b = next+n*8L;
                if (rows == 1) sum = sequence.get(layout,a)*scratch.get(layout,b);
                else for (int row = 0; row < rows; row++) {
                    double left = sequence.get(layout,a), right = scratch.get(layout,b);
                    sum = sum+left*right; a += contraction*8L; b += columns*8L;
                }
                long p = (k*(long)columns+n)*8;
                double accumulated = oldParameter.get(layout,parameterBase+(step == 0 ? 0 : p))+sum;
                scratch.set(layout,nextParameter+p,accumulated);
            }
            for (int row = 0; row < rows; row++) for (int k = 0; k < contraction; k++) {
                double sum = 0; long a = next+row*(long)columns*8, b = k*(long)columns*8;
                if (columns == 1) sum = scratch.get(layout,a)*weight.get(layout,b);
                else for (int n = 0; n < columns; n++) {
                    double left = scratch.get(layout,a), right = weight.get(layout,b);
                    sum = sum+left*right; a += 8; b += 8;
                }
                sequenceGradient.set(layout,t*inputFrame+(row*(long)contraction+k)*8,sum);
            }
            lastCarry = next; lastParameter = nextParameter;
        }
        for (long p = 0; p < frame; p += 8) initialGradient.set(layout,p,scratch.get(layout,lastCarry+p));
        for (long p = 0; p < parameter; p += 8) weightGradient.set(layout,p,scratch.get(layout,lastParameter+p));
    }

    /**
     * Specialized first-order reverse of {@code next=emit=h+x*w}, with vector parameter
     * {@code w[H]} broadcast into {@code [B,H]}. The output-cell/selected-row loop evaluates all
     * represented incoming, product and sequence-gradient sites once, accumulating the vector
     * parameter contribution in one F64 local. Separate old/new banks preserve complete carry
     * versions; no contribution array, reduction scratch, input staging or per-step allocation
     * is required. The producing incoming and parameter-sum sites write directly to next banks.
     *
     * @param finalSeed borrowed dense final-carry seed
     * @param emissionSeed borrowed original-time dense emission seeds
     * @param sequence borrowed original-time primal sequence
     * @param weight borrowed dense H-cell parameter
     * @param zero borrowed one-cell exact typed accumulator initializer
     * @param initialGradient distinct B*H-cell final publication
     * @param weightGradient distinct H-cell final publication
     * @param sequenceGradient demanded stacked gradient emission
     * @param scratch independent run-owned 2*(B*H+H)*8-byte carry banks
     * @param time positive original step count
     * @param batch positive selected row count
     * @param width positive vector width
     */
    public static void broadcastReverseFloat64(MemorySegment finalSeed, MemorySegment emissionSeed,
            MemorySegment sequence, MemorySegment weight, MemorySegment zero, MemorySegment initialGradient,
            MemorySegment weightGradient, MemorySegment sequenceGradient, MemorySegment scratch,
            int time, int batch, int width) {
        var layout = ValueLayout.JAVA_DOUBLE_UNALIGNED;
        long frameBytes = batch * (long)width * 8, vectorBytes = width * 8L;
        long lastCarry = 0, lastParameter = 0;
        for (int step = 0; step < time; step++) {
            int t = time-1-step;
            MemorySegment old = step == 0 ? finalSeed : scratch;
            MemorySegment oldParameter = step == 0 ? zero : scratch;
            long oldBase = step == 0 ? 0 : ((step-1)&1)*frameBytes;
            long oldParameterBase = step == 0 ? 0 : 2*frameBytes+((step-1)&1)*vectorBytes;
            long nextBase = (step&1)*frameBytes, nextParameterBase = 2*frameBytes+(step&1)*vectorBytes;
            long frame = t*frameBytes;
            for (int j = 0; j < width; j++) {
                double sum = 0;
                double parameter = weight.get(layout,j*8L);
                for (int b = 0; b < batch; b++) {
                    long coordinate = (b*(long)width+j)*8;
                    double incoming = old.get(layout,oldBase+coordinate)+emissionSeed.get(layout,frame+coordinate);
                    scratch.set(layout,nextBase+coordinate,incoming);
                    double contribution = incoming*sequence.get(layout,frame+coordinate);
                    double dx = incoming*parameter;
                    sequenceGradient.set(layout,frame+coordinate,dx);
                    sum = sum+contribution;
                }
                double accumulated = oldParameter.get(layout,oldParameterBase+(step == 0 ? 0 : j*8L))+sum;
                scratch.set(layout,nextParameterBase+j*8L,accumulated);
            }
            lastCarry = nextBase; lastParameter = nextParameterBase;
        }
        for (long address = 0; address < frameBytes; address += 8)
            initialGradient.set(layout,address,scratch.get(layout,lastCarry+address));
        for (long address = 0; address < vectorBytes; address += 8)
            weightGradient.set(layout,address,scratch.get(layout,lastParameter+address));
    }

    /**
     * Computes forward FLOAT64 producing sites using a demanded emission as exact carry backing.
     * @param initial borrowed dense initial values
     * @param sequence borrowed dense original-time steps
     * @param result distinct dense final publication
     * @param emissions demanded original-time next values
     * @param saved optional distinct demanded activation residual, null for inference
     * @param time non-negative time count
     * @param elements non-negative coordinates per step
     * @param reverse decreasing original-time traversal
     */
    public static void forwardFloat64(MemorySegment initial, MemorySegment sequence, MemorySegment result,
            MemorySegment emissions, MemorySegment saved, long time, long elements, boolean reverse) {
        var layout = ValueLayout.JAVA_DOUBLE_UNALIGNED;
        long bytes = elements * 8L;
        if (elements == 0) return;
        if (time == 0) {
            for (long address = 0; address < bytes; address += 8)
                result.set(layout, address, initial.get(layout, address));
            return;
        }
        long frame = reverse ? (time - 1) * bytes : 0;
        long advance = reverse ? -bytes : bytes;
        MemorySegment carry = initial;
        long carryBase = 0;
        for (long remaining = time; remaining > 0; remaining--) {
            long carryAddress = carryBase, sequenceAddress = frame, outputAddress = frame;
            for (long i = 0; i < elements; i++) {
                double old = carry.get(layout, carryAddress);
                double x = sequence.get(layout, sequenceAddress);
                double sum = old + x;
                double next = StrictMath.tanh(sum);
                emissions.set(layout, outputAddress, next);
                if (saved != null) saved.set(layout, outputAddress, next);
                carryAddress += 8; sequenceAddress += 8; outputAddress += 8;
            }
            carry = emissions; carryBase = frame; frame += advance;
        }
        for (long address = 0; address < bytes; address += 8)
            result.set(layout, address, emissions.get(layout, carryBase + address));
    }

    /**
     * Computes the opposite traversal from exact saved outputs with both seed classes.
     * @param finalSeed borrowed dense final-carry seed
     * @param emissionSeeds borrowed stacked output seeds
     * @param saved exact original activation history, never recomputed
     * @param initialGradient distinct dense dInitial publication
     * @param sequenceGradient demanded stacked dSequence, also the exact cotangent carry backing
     * @param time non-negative time count
     * @param elements non-negative coordinates per step
     * @param originalReverse whether the original traversal was decreasing
     */
    public static void reverseFloat64(MemorySegment finalSeed, MemorySegment emissionSeeds, MemorySegment saved,
            MemorySegment initialGradient, MemorySegment sequenceGradient, long time, long elements,
            boolean originalReverse) {
        var layout = ValueLayout.JAVA_DOUBLE_UNALIGNED;
        long bytes = elements * 8L;
        if (elements == 0) return;
        if (time == 0) {
            for (long address = 0; address < bytes; address += 8)
                initialGradient.set(layout, address, finalSeed.get(layout, address));
            return;
        }
        long frame = originalReverse ? 0 : (time - 1) * bytes;
        long advance = originalReverse ? bytes : -bytes;
        MemorySegment carry = finalSeed;
        long carryBase = 0;
        for (long remaining = time; remaining > 0; remaining--) {
            long carryAddress = carryBase, address = frame;
            for (long i = 0; i < elements; i++) {
                double old = carry.get(layout, carryAddress);
                double seed = emissionSeeds.get(layout, address);
                double y = saved.get(layout, address);
                double incoming = old + seed;
                double square = y * y;
                double factor = 1d - square;
                double next = incoming * factor;
                sequenceGradient.set(layout, address, next);
                carryAddress += 8; address += 8;
            }
            carry = sequenceGradient; carryBase = frame; frame += advance;
        }
        for (long address = 0; address < bytes; address += 8)
            initialGradient.set(layout, address, sequenceGradient.get(layout, carryBase + address));
    }
    private CpuGeneralScanCleanJavaOracle() {}

    /**
     * Specialized scalar reference for {@code next=emit=h+x@w}. One F32 dot accumulator feeds
     * the represented ADD directly; the demanded emission is next-step carry backing. No product
     * array, input staging, per-step allocation or graph dispatch is needed.
     * @param initial borrowed dense B*N initial carry
     * @param sequence borrowed original-time T*B*K sequence
     * @param weight borrowed dense K*N parameter
     * @param result distinct B*N final publication
     * @param emissions demanded T*B*N next-value frames
     * @param time positive step count
     * @param rows positive B extent
     * @param contraction positive K extent
     * @param columns positive N extent
     */
    public static void matmulForward(MemorySegment initial, MemorySegment sequence, MemorySegment weight,
            MemorySegment result, MemorySegment emissions, int time, int rows, int contraction, int columns) {
        var layout = ValueLayout.JAVA_FLOAT_UNALIGNED;
        long frame = rows*(long)columns*4, inputFrame = rows*(long)contraction*4;
        MemorySegment old = initial; long oldBase = 0;
        for (int t = 0; t < time; t++) {
            long outBase = t*frame;
            for (int row = 0; row < rows; row++) for (int column = 0; column < columns; column++) {
                float sum = 0;
                long a = t*inputFrame+row*(long)contraction*4, b = column*4L;
                if (contraction == 1) sum = sequence.get(layout,a)*weight.get(layout,b);
                else for (int k = 0; k < contraction; k++) {
                    float left = sequence.get(layout,a), right = weight.get(layout,b);
                    sum = sum+left*right; a += 4; b += columns*4L;
                }
                long cell = (row*(long)columns+column)*4;
                float next = old.get(layout,oldBase+cell)+sum;
                emissions.set(layout,outBase+cell,next);
            }
            old = emissions; oldBase = outBase;
        }
        for (long p = 0; p < frame; p += 4) result.set(layout,p,emissions.get(layout,oldBase+p));
    }

    /**
     * Specialized actual first-order reverse of {@code next=emit=h+x@w}. Incoming cotangents
     * must materialize once because both matrix products read them across coordinates. Separate
     * old/new banks retain the complete carry and K*N parameter sum. Each dot feeds its sole
     * represented result/addition directly; no per-step allocation or temporary product exists.
     * @param finalSeed borrowed B*N final seed
     * @param emissionSeed borrowed original-time T*B*N seeds
     * @param sequence borrowed primal T*B*K sequence
     * @param weight borrowed primal K*N parameter
     * @param zero borrowed single-cell parameter-sum initializer
     * @param initialGradient distinct B*N final carry-gradient publication
     * @param weightGradient distinct K*N parameter-gradient publication
     * @param sequenceGradient demanded T*B*K matrix-gradient frames
     * @param scratch run-owned 2*(B*N+K*N)*4-byte old/new banks
     * @param time positive step count
     * @param rows positive B extent
     * @param contraction positive K extent
     * @param columns positive N extent
     */
    public static void matmulReverse(MemorySegment finalSeed, MemorySegment emissionSeed, MemorySegment sequence,
            MemorySegment weight, MemorySegment zero, MemorySegment initialGradient, MemorySegment weightGradient,
            MemorySegment sequenceGradient, MemorySegment scratch, int time, int rows, int contraction, int columns) {
        var layout = ValueLayout.JAVA_FLOAT_UNALIGNED;
        long frame = rows*(long)columns*4, parameter = contraction*(long)columns*4, inputFrame = rows*(long)contraction*4;
        long lastCarry = 0, lastParameter = 0;
        for (int step = 0; step < time; step++) {
            int t = time-1-step;
            MemorySegment old = step == 0 ? finalSeed : scratch, oldParameter = step == 0 ? zero : scratch;
            long oldBase = step == 0 ? 0 : ((step-1)&1)*frame;
            long parameterBase = step == 0 ? 0 : 2*frame+((step-1)&1)*parameter;
            long next = (step&1)*frame, nextParameter = 2*frame+(step&1)*parameter;
            for (long p = 0; p < frame; p += 4) {
                float incoming = old.get(layout,oldBase+p)+emissionSeed.get(layout,t*frame+p);
                scratch.set(layout,next+p,incoming);
            }
            for (int k = 0; k < contraction; k++) for (int n = 0; n < columns; n++) {
                float sum = 0; long a = t*inputFrame+k*4L, b = next+n*4L;
                if (rows == 1) sum = sequence.get(layout,a)*scratch.get(layout,b);
                else for (int row = 0; row < rows; row++) {
                    float left = sequence.get(layout,a), right = scratch.get(layout,b);
                    sum = sum+left*right; a += contraction*4L; b += columns*4L;
                }
                long p = (k*(long)columns+n)*4;
                float accumulated = oldParameter.get(layout,parameterBase+(step == 0 ? 0 : p))+sum;
                scratch.set(layout,nextParameter+p,accumulated);
            }
            for (int row = 0; row < rows; row++) for (int k = 0; k < contraction; k++) {
                float sum = 0; long a = next+row*(long)columns*4, b = k*(long)columns*4;
                if (columns == 1) sum = scratch.get(layout,a)*weight.get(layout,b);
                else for (int n = 0; n < columns; n++) {
                    float left = scratch.get(layout,a), right = weight.get(layout,b);
                    sum = sum+left*right; a += 4; b += 4;
                }
                sequenceGradient.set(layout,t*inputFrame+(row*(long)contraction+k)*4,sum);
            }
            lastCarry = next; lastParameter = nextParameter;
        }
        for (long p = 0; p < frame; p += 4) initialGradient.set(layout,p,scratch.get(layout,lastCarry+p));
        for (long p = 0; p < parameter; p += 4) weightGradient.set(layout,p,scratch.get(layout,lastParameter+p));
    }

    /**
     * Optimal specialized first-order reverse of {@code next=emit=h+x*w}, with vector parameter
     * {@code w[H]} broadcast into {@code [B,H]}. The output-cell/selected-row loop evaluates all
     * represented incoming, product and sequence-gradient sites once, accumulating the vector
     * parameter contribution in one F32 local. Separate old/new banks preserve complete carry
     * versions; no contribution array, reduction scratch, input staging or per-step allocation
     * is required. The producing incoming and parameter-sum sites write directly to next banks.
     *
     * @param finalSeed borrowed dense final-carry seed
     * @param emissionSeed borrowed original-time dense emission seeds
     * @param sequence borrowed original-time primal sequence
     * @param weight borrowed dense H-cell parameter
     * @param zero borrowed one-cell exact typed accumulator initializer
     * @param initialGradient distinct B*H-cell final publication
     * @param weightGradient distinct H-cell final publication
     * @param sequenceGradient demanded stacked gradient emission
     * @param scratch independent run-owned 2*(B*H+H)*4-byte carry banks
     * @param time positive original step count
     * @param batch positive selected row count
     * @param width positive vector width
     */
    public static void broadcastReverse(MemorySegment finalSeed, MemorySegment emissionSeed,
            MemorySegment sequence, MemorySegment weight, MemorySegment zero, MemorySegment initialGradient,
            MemorySegment weightGradient, MemorySegment sequenceGradient, MemorySegment scratch,
            int time, int batch, int width) {
        var layout = ValueLayout.JAVA_FLOAT_UNALIGNED;
        long frameBytes = batch * (long)width * 4, vectorBytes = width * 4L;
        long lastCarry = 0, lastParameter = 0;
        for (int step = 0; step < time; step++) {
            int t = time-1-step;
            MemorySegment old = step == 0 ? finalSeed : scratch;
            MemorySegment oldParameter = step == 0 ? zero : scratch;
            long oldBase = step == 0 ? 0 : ((step-1)&1)*frameBytes;
            long oldParameterBase = step == 0 ? 0 : 2*frameBytes+((step-1)&1)*vectorBytes;
            long nextBase = (step&1)*frameBytes, nextParameterBase = 2*frameBytes+(step&1)*vectorBytes;
            long frame = t*frameBytes;
            for (int j = 0; j < width; j++) {
                float sum = 0;
                float parameter = weight.get(layout,j*4L);
                for (int b = 0; b < batch; b++) {
                    long coordinate = (b*(long)width+j)*4;
                    float incoming = old.get(layout,oldBase+coordinate)+emissionSeed.get(layout,frame+coordinate);
                    scratch.set(layout,nextBase+coordinate,incoming);
                    float contribution = incoming*sequence.get(layout,frame+coordinate);
                    float dx = incoming*parameter;
                    sequenceGradient.set(layout,frame+coordinate,dx);
                    sum = sum+contribution;
                }
                float accumulated = oldParameter.get(layout,oldParameterBase+(step == 0 ? 0 : j*4L))+sum;
                scratch.set(layout,nextParameterBase+j*4L,accumulated);
            }
            lastCarry = nextBase; lastParameter = nextParameterBase;
        }
        for (long address = 0; address < frameBytes; address += 4)
            initialGradient.set(layout,address,scratch.get(layout,lastCarry+address));
        for (long address = 0; address < vectorBytes; address += 4)
            weightGradient.set(layout,address,scratch.get(layout,lastParameter+address));
    }

    /**
     * Computes forward FLOAT32 producing sites using a demanded emission as exact carry backing.
     * @param initial borrowed dense initial values
     * @param sequence borrowed dense original-time steps
     * @param result distinct dense final publication
     * @param emissions demanded original-time next values
     * @param saved optional distinct demanded activation residual, null for inference
     * @param time non-negative time count
     * @param elements non-negative coordinates per step
     * @param reverse decreasing original-time traversal
     */
    public static void forward(MemorySegment initial, MemorySegment sequence, MemorySegment result,
            MemorySegment emissions, MemorySegment saved, long time, long elements, boolean reverse) {
        var layout = ValueLayout.JAVA_FLOAT_UNALIGNED;
        long bytes = elements * 4L;
        if (elements == 0) return;
        if (time == 0) {
            for (long address = 0; address < bytes; address += 4)
                result.set(layout, address, initial.get(layout, address));
            return;
        }
        long frame = reverse ? (time - 1) * bytes : 0;
        long advance = reverse ? -bytes : bytes;
        MemorySegment carry = initial;
        long carryBase = 0;
        for (long remaining = time; remaining > 0; remaining--) {
            long carryAddress = carryBase, sequenceAddress = frame, outputAddress = frame;
            for (long i = 0; i < elements; i++) {
                float old = carry.get(layout, carryAddress);
                float x = sequence.get(layout, sequenceAddress);
                float sum = old + x;
                float next = (float) StrictMath.tanh(sum);
                emissions.set(layout, outputAddress, next);
                if (saved != null) saved.set(layout, outputAddress, next);
                carryAddress += 4; sequenceAddress += 4; outputAddress += 4;
            }
            carry = emissions; carryBase = frame; frame += advance;
        }
        for (long address = 0; address < bytes; address += 4)
            result.set(layout, address, emissions.get(layout, carryBase + address));
    }

    /**
     * Computes the opposite traversal from exact saved outputs with both seed classes.
     * @param finalSeed borrowed dense final-carry seed
     * @param emissionSeeds borrowed stacked output seeds
     * @param saved exact original activation history, never recomputed
     * @param initialGradient distinct dense dInitial publication
     * @param sequenceGradient demanded stacked dSequence, also the exact cotangent carry backing
     * @param time non-negative time count
     * @param elements non-negative coordinates per step
     * @param originalReverse whether the original traversal was decreasing
     */
    public static void reverse(MemorySegment finalSeed, MemorySegment emissionSeeds, MemorySegment saved,
            MemorySegment initialGradient, MemorySegment sequenceGradient, long time, long elements,
            boolean originalReverse) {
        var layout = ValueLayout.JAVA_FLOAT_UNALIGNED;
        long bytes = elements * 4L;
        if (elements == 0) return;
        if (time == 0) {
            for (long address = 0; address < bytes; address += 4)
                initialGradient.set(layout, address, finalSeed.get(layout, address));
            return;
        }
        long frame = originalReverse ? 0 : (time - 1) * bytes;
        long advance = originalReverse ? bytes : -bytes;
        MemorySegment carry = finalSeed;
        long carryBase = 0;
        for (long remaining = time; remaining > 0; remaining--) {
            long carryAddress = carryBase, address = frame;
            for (long i = 0; i < elements; i++) {
                float old = carry.get(layout, carryAddress);
                float seed = emissionSeeds.get(layout, address);
                float y = saved.get(layout, address);
                float incoming = old + seed;
                float square = y * y;
                float factor = 1f - square;
                float next = incoming * factor;
                sequenceGradient.set(layout, address, next);
                carryAddress += 4; address += 4;
            }
            carry = sequenceGradient; carryBase = frame; frame += advance;
        }
        for (long address = 0; address < bytes; address += 4)
            initialGradient.set(layout, address, sequenceGradient.get(layout, carryBase + address));
    }
}
