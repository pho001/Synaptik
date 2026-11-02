package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuConcurrencyBudget;
import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuRecurrentScanExecutionTest.Fixture;
import java.lang.foreign.*;
import java.nio.ByteOrder;
import java.util.function.Consumer;
import jdk.incubator.vector.*;
import static java.lang.foreign.ValueLayout.*;

/**
 * Independent F3/H2 equations for the four CPU0074 realizations, not an IR interpreter or an
 * emitter transcription. The ordinary case differentiates sum(Y*Y), Y=tanh(XW+HU+b).
 * The Scan case consumes exact saved Y and previous H, and differentiates the joint final/emission
 * seeds. Ordered feature products, masked feature tails and separate multiply/add sites are explicit.
 * W/U must be immutable, and writable outputs/partials must not alias borrowed inputs.
 * Only row contributors regroup. Each active time completes its reduction before a distinct carry
 * ADD. Shared carry locals may remain live across times, but their contribution trees are never
 * flattened across time. No activation is replayed.
 *
 * <p>All storage, masks, row calls and worker adapters are bound cold. Parallel arms borrow the
 * fixture's existing workers and one exclusive reservation across the whole call. The caller must
 * keep borrowed segments and fixture alive and must not run the same oracle concurrently.</p>
 */
final class CpuRowFusionSimdParallelJavaOracle {
    private CpuRowFusionSimdParallelJavaOracle() { }

    /**
     * Borrowed dense native carriers for one specialized equation. All offsets below are logical.
     * @param x X[B,3] or X[T,B,3]
     * @param previous H[B,2] or saved previous H[T,B,2]
     * @param saved exact saved tanh Y[B,2] or Y[T,B,2]
     * @param w immutable W[3,2]
     * @param u immutable U[2,2]
     * @param lengths prevalidated INT64[B], used only for Scan
     * @param finalSeed Scan final seed[B,2], unused for ordinary
     * @param emissionSeed Scan emission seed[T,B,2], unused for ordinary
     * @param delta writable ordinary retained cotangent[B,2], unused for Scan
     * @param dx writable X cotangent
     * @param dh writable H cotangent[B,2]
     * @param dw writable W cotangent[3,2]
     * @param du writable U cotangent[2,2]
     * @param db writable Scan bias cotangent[2]; ordinary exact SUM is outside this region
     * @param batch nonnegative B
     * @param time nonnegative T for Scan, one for ordinary
     * @param scan whether to use the saved-primal opposite-traversal Scan equation
     * @param reverse original Scan traversal direction
     * @param seed represented ordinary scalar loss seed
     * @param derivative represented scalar tanh derivative constant, normally one
     * @param initial represented scalar shared-carry initial value, normally positive zero
     */
    record Data(MemorySegment x,MemorySegment previous,MemorySegment saved,MemorySegment w,MemorySegment u,
            MemorySegment lengths,MemorySegment finalSeed,MemorySegment emissionSeed,MemorySegment delta,
            MemorySegment dx,MemorySegment dh,MemorySegment dw,MemorySegment du,MemorySegment db,
            int batch,int time,boolean scan,boolean reverse,double seed,double derivative,double initial) { }

    /**
     * Binds a typed action and its private scratch; no generated code is consulted.
     * @param data borrowed exact native carriers and represented equation constants
     * @param doubles whether every arithmetic site is F64, otherwise F32
     * @param vector whether to use preferred-species masked feature vectors
     * @param parallel whether two fixed row ranges are requested (B less than two stays inline)
     * @param minimumElementsPerWorker positive cold row policy, identical to the selected unit
     * @param fixture borrowed existing workers/budget owner
     * @param arena caller-owned shared arena for fixed ten-cell transpose and private partials
     * @return nonconcurrent reusable action borrowing all arguments
     */
    static Runnable bind(Data data,boolean doubles,boolean vector,boolean parallel,long minimumElementsPerWorker,Fixture fixture,Arena arena) {
        if(minimumElementsPerWorker<1)throw new IllegalArgumentException("positive minimum row work required");
        return doubles ? new F64(data,vector,parallel,minimumElementsPerWorker,fixture,arena)
                : new F32(data,vector,parallel,minimumElementsPerWorker,fixture,arena);
    }

    /** One typed run owner; fields transport fixed range state, never runtime graph decisions. */
    private static final class F32 implements Runnable {
        private static final VectorSpecies<Float> S=FloatVector.SPECIES_PREFERRED;
        private static final ByteOrder ORDER=ByteOrder.nativeOrder();
        private static final VectorMask<Float> H=S.indexInRange(0,2);
        private static final VectorMask<Float> X=S.indexInRange(0,3);
        private static final VectorMask<Float> TAIL=S.indexInRange(0,1);
        private final Data d;
        private final boolean vector;
        private final int count,partialCells;
        private final MemorySegment transpose,partials;
        private final long[] valid;
        private final Runnable[] ranges;
        private final Consumer<CpuConcurrencyBudget.Lease> dispatch;
        private final CpuConcurrencyBudget budget;
        private int time;
        private float w00,w01,w10,w11,w20,w21,u00,u01,u10,u11;
        private final float seed,derivative,initial;
        private FloatVector vw0,vw1,vwt0,vwt1,vu0,vu1;
        F32(Data d,boolean vector,boolean parallel,long minimumElementsPerWorker,Fixture fixture,Arena arena) {
            this.d=d;this.vector=vector;partialCells=d.scan()?12:10;count=d.batch()==0?0:parallel?(int)Math.min(2,1+(d.batch()-1)/minimumElementsPerWorker):1;
            transpose=vector?arena.allocate(10L*4,4):MemorySegment.NULL;
            partials=count>1?arena.allocate((long)partialCells*count*4,4):MemorySegment.NULL;
            valid=count>1?new long[count]:null;ranges=count>1?new Runnable[count]:null;
            budget=fixture.rowOracleBudget();
            seed=(float)d.seed();derivative=(float)d.derivative();initial=(float)d.initial();
            for(int i=0;count>1&&i<count;i++) {
                int quotient=d.batch()/count,remainder=d.batch()%count;
                int id=i,begin=i*quotient+Math.min(i,remainder),end=begin+quotient+(i<remainder?1:0);
                ranges[i]=d.scan() ? vector?()->vectorScan(id,begin,end):()->scalarScan(id,begin,end)
                        : vector?()->vectorOrdinary(id,begin,end):()->scalarOrdinary(id,begin,end);
            }
            dispatch=count>1?fixture.bindRowOracleRanges(ranges):null;
        }
        @Override public void run() {
            try(var lease=budget==null?null:budget.acquire(Math.max(1,count))) {
                if(d.scan()) {
                    d.dh().copyFrom(d.finalSeed());d.dx().fill((byte)0);
                    long maximum=0;
                    for(int r=0;r<d.batch();r++)maximum=Math.max(maximum,d.lengths().getAtIndex(JAVA_LONG,r));
                    if(count<2) {
                        if(maximum>0&&d.time()>0)stage();
                        if(vector)inlineVectorScan(maximum);else inlineScalarScan(maximum);
                        return;
                    }
                    if(maximum>0&&d.time()>0)stage();
                    if(vector)parallelVectorScan(lease,maximum);else parallelScalarScan(lease,maximum);
                } else {
                    if(count<2) {
                        if(d.batch()>0)stage();
                        if(vector)inlineVectorOrdinary();else inlineScalarOrdinary();
                    } else {
                        stage();execute(lease);
                        if(vector)mergeVector(false);else mergeScalar(false);
                    }
                }
            }
        }
        private void execute(CpuConcurrencyBudget.Lease lease) {
            if(count>1)dispatch.accept(lease);else if(count==1)ranges[0].run();
        }
        private void stage() {
            w00=d.w().getAtIndex(JAVA_FLOAT,0);w01=d.w().getAtIndex(JAVA_FLOAT,1);
            w10=d.w().getAtIndex(JAVA_FLOAT,2);w11=d.w().getAtIndex(JAVA_FLOAT,3);
            w20=d.w().getAtIndex(JAVA_FLOAT,4);w21=d.w().getAtIndex(JAVA_FLOAT,5);
            u00=d.u().getAtIndex(JAVA_FLOAT,0);u01=d.u().getAtIndex(JAVA_FLOAT,1);
            u10=d.u().getAtIndex(JAVA_FLOAT,2);u11=d.u().getAtIndex(JAVA_FLOAT,3);
            if(vector) {
                transpose.setAtIndex(JAVA_FLOAT,0,w00);transpose.setAtIndex(JAVA_FLOAT,1,w10);transpose.setAtIndex(JAVA_FLOAT,2,w20);
                transpose.setAtIndex(JAVA_FLOAT,3,w01);transpose.setAtIndex(JAVA_FLOAT,4,w11);transpose.setAtIndex(JAVA_FLOAT,5,w21);
                transpose.setAtIndex(JAVA_FLOAT,6,u00);transpose.setAtIndex(JAVA_FLOAT,7,u10);
                transpose.setAtIndex(JAVA_FLOAT,8,u01);transpose.setAtIndex(JAVA_FLOAT,9,u11);
                // Immutable, nonaliasing coefficients are loaded once per invocation, not time.
                vu0=FloatVector.fromMemorySegment(S,transpose,24L,ORDER,H);
                vu1=FloatVector.fromMemorySegment(S,transpose,32L,ORDER,H);
                vw0=FloatVector.fromMemorySegment(S,transpose,0,ORDER,X);
                vw1=FloatVector.fromMemorySegment(S,transpose,12L,ORDER,X);
                vwt0=S.length()<3?FloatVector.fromMemorySegment(S,transpose,8L,ORDER,TAIL):FloatVector.zero(S);
                vwt1=S.length()<3?FloatVector.fromMemorySegment(S,transpose,20L,ORDER,TAIL):FloatVector.zero(S);
            }
        }

        private void scalarOrdinary(int id,int begin,int end) {
            float a00=0f,a01=0f,a10=0f,a11=0f,a20=0f,a21=0f;
            float c00=0f,c01=0f,c10=0f,c11=0f,b0=0f,b1=0f;
            boolean active=false;
            final float w00=this.w00,w01=this.w01,w10=this.w10,w11=this.w11,w20=this.w20,w21=this.w21;
            final float u00=this.u00,u01=this.u01,u10=this.u10,u11=this.u11;
            for(int r=begin;r<end;r++)  {
                active=true;int h=2*r,p=h,q=3*r;
                float y0=d.saved().getAtIndex(JAVA_FLOAT,p),y1=d.saved().getAtIndex(JAVA_FLOAT,p+1);
                float d0=(seed*y0+seed*y0)*(derivative-y0*y0),d1=(seed*y1+seed*y1)*(derivative-y1*y1);
                d.delta().setAtIndex(JAVA_FLOAT,h,d0);d.delta().setAtIndex(JAVA_FLOAT,h+1,d1);
                float x0=d.x().getAtIndex(JAVA_FLOAT,q),x1=d.x().getAtIndex(JAVA_FLOAT,q+1),x2=d.x().getAtIndex(JAVA_FLOAT,q+2);
                float h0=d.previous().getAtIndex(JAVA_FLOAT,p),h1=d.previous().getAtIndex(JAVA_FLOAT,p+1);
                d.dx().setAtIndex(JAVA_FLOAT,q,(0f+d0*w00)+d1*w01);
                d.dx().setAtIndex(JAVA_FLOAT,q+1,(0f+d0*w10)+d1*w11);
                d.dx().setAtIndex(JAVA_FLOAT,q+2,(0f+d0*w20)+d1*w21);
                d.dh().setAtIndex(JAVA_FLOAT,h,(0f+d0*u00)+d1*u01);
                d.dh().setAtIndex(JAVA_FLOAT,h+1,(0f+d0*u10)+d1*u11);
                a00+=x0*d0;a01+=x0*d1;a10+=x1*d0;a11+=x1*d1;a20+=x2*d0;a21+=x2*d1;
                c00+=h0*d0;c01+=h0*d1;c10+=h1*d0;c11+=h1*d1;
                
            }
            valid[id]=active?1:0;int p=id*partialCells;
            partials.setAtIndex(JAVA_FLOAT,p,a00);partials.setAtIndex(JAVA_FLOAT,p+1,a01);
            partials.setAtIndex(JAVA_FLOAT,p+2,a10);partials.setAtIndex(JAVA_FLOAT,p+3,a11);
            partials.setAtIndex(JAVA_FLOAT,p+4,a20);partials.setAtIndex(JAVA_FLOAT,p+5,a21);
            partials.setAtIndex(JAVA_FLOAT,p+6,c00);partials.setAtIndex(JAVA_FLOAT,p+7,c01);
            partials.setAtIndex(JAVA_FLOAT,p+8,c10);partials.setAtIndex(JAVA_FLOAT,p+9,c11);
            
        }
        private void vectorOrdinary(int id,int begin,int end) {
            var a0=FloatVector.zero(S);var a1=FloatVector.zero(S);var a2=FloatVector.zero(S);
            var c0=FloatVector.zero(S);var c1=FloatVector.zero(S);var b=FloatVector.zero(S);
            boolean active=false;
            var u0=vu0;var u1=vu1;var w0=vw0;var w1=vw1;var wt0=vwt0;var wt1=vwt1;
            // F64 may have only two preferred lanes: F3 then has a second masked feature chunk.
            for(int r=begin;r<end;r++)  {
                active=true;int h=2*r,p=h,q=3*r;
                var y=FloatVector.fromMemorySegment(S,d.saved(),(long)p*4,ORDER,H);
                var delta=y.mul(seed).add(y.mul(seed)).mul(FloatVector.broadcast(S,derivative).sub(y.mul(y)));
                delta.intoMemorySegment(d.delta(),(long)h*4,ORDER,H);
                float d0=delta.lane(0),d1=delta.lane(1);
                FloatVector.zero(S).add(w0.mul(d0)).add(w1.mul(d1)).intoMemorySegment(d.dx(),(long)q*4,ORDER,X);
                if(S.length()<3)FloatVector.zero(S).add(wt0.mul(d0)).add(wt1.mul(d1))
                        .intoMemorySegment(d.dx(),(long)(q+2)*4,ORDER,TAIL);
                FloatVector.zero(S).add(u0.mul(d0)).add(u1.mul(d1)).intoMemorySegment(d.dh(),(long)h*4,ORDER,H);
                float x0=d.x().getAtIndex(JAVA_FLOAT,q),x1=d.x().getAtIndex(JAVA_FLOAT,q+1),x2=d.x().getAtIndex(JAVA_FLOAT,q+2);
                var prev=FloatVector.fromMemorySegment(S,d.previous(),(long)p*4,ORDER,H);
                float h0=prev.lane(0),h1=prev.lane(1);
                a0=a0.add(delta.mul(x0));a1=a1.add(delta.mul(x1));a2=a2.add(delta.mul(x2));
                c0=c0.add(delta.mul(h0));c1=c1.add(delta.mul(h1));
                
            }
            valid[id]=active?1:0;long p=(long)id*partialCells*4;
            a0.intoMemorySegment(partials,p,ORDER,H);a1.intoMemorySegment(partials,p+8,ORDER,H);
            a2.intoMemorySegment(partials,p+16,ORDER,H);c0.intoMemorySegment(partials,p+24,ORDER,H);
            c1.intoMemorySegment(partials,p+32,ORDER,H);
            
        }

        private void scalarScan(int id,int begin,int end) {
            float a00=0f,a01=0f,a10=0f,a11=0f,a20=0f,a21=0f;
            float c00=0f,c01=0f,c10=0f,c11=0f,b0=0f,b1=0f;
            boolean active=false;
            final float w00=this.w00,w01=this.w01,w10=this.w10,w11=this.w11,w20=this.w20,w21=this.w21;
            final float u00=this.u00,u01=this.u01,u10=this.u10,u11=this.u11;
            for(int r=begin;r<end;r++) if(time<d.lengths().getAtIndex(JAVA_LONG,r)) {
                active=true;int h=2*r,p=2*(time*d.batch()+r),q=3*(time*d.batch()+r);
                float y0=d.saved().getAtIndex(JAVA_FLOAT,p),y1=d.saved().getAtIndex(JAVA_FLOAT,p+1);
                float d0=(d.dh().getAtIndex(JAVA_FLOAT,h)+d.emissionSeed().getAtIndex(JAVA_FLOAT,p))*(derivative-y0*y0);
                float d1=(d.dh().getAtIndex(JAVA_FLOAT,h+1)+d.emissionSeed().getAtIndex(JAVA_FLOAT,p+1))*(derivative-y1*y1);
                float x0=d.x().getAtIndex(JAVA_FLOAT,q),x1=d.x().getAtIndex(JAVA_FLOAT,q+1),x2=d.x().getAtIndex(JAVA_FLOAT,q+2);
                float h0=d.previous().getAtIndex(JAVA_FLOAT,p),h1=d.previous().getAtIndex(JAVA_FLOAT,p+1);
                d.dx().setAtIndex(JAVA_FLOAT,q,(0f+d0*w00)+d1*w01);
                d.dx().setAtIndex(JAVA_FLOAT,q+1,(0f+d0*w10)+d1*w11);
                d.dx().setAtIndex(JAVA_FLOAT,q+2,(0f+d0*w20)+d1*w21);
                d.dh().setAtIndex(JAVA_FLOAT,h,(0f+d0*u00)+d1*u01);
                d.dh().setAtIndex(JAVA_FLOAT,h+1,(0f+d0*u10)+d1*u11);
                a00+=x0*d0;a01+=x0*d1;a10+=x1*d0;a11+=x1*d1;a20+=x2*d0;a21+=x2*d1;
                c00+=h0*d0;c01+=h0*d1;c10+=h1*d0;c11+=h1*d1;
                if(d.batch()==1){b0=d0;b1=d1;}else{b0+=d0;b1+=d1;}
            }
            valid[id]=active?1:0;int p=id*partialCells;
            partials.setAtIndex(JAVA_FLOAT,p,a00);partials.setAtIndex(JAVA_FLOAT,p+1,a01);
            partials.setAtIndex(JAVA_FLOAT,p+2,a10);partials.setAtIndex(JAVA_FLOAT,p+3,a11);
            partials.setAtIndex(JAVA_FLOAT,p+4,a20);partials.setAtIndex(JAVA_FLOAT,p+5,a21);
            partials.setAtIndex(JAVA_FLOAT,p+6,c00);partials.setAtIndex(JAVA_FLOAT,p+7,c01);
            partials.setAtIndex(JAVA_FLOAT,p+8,c10);partials.setAtIndex(JAVA_FLOAT,p+9,c11);
            partials.setAtIndex(JAVA_FLOAT,p+10,b0);partials.setAtIndex(JAVA_FLOAT,p+11,b1);
        }
        private void vectorScan(int id,int begin,int end) {
            var a0=FloatVector.zero(S);var a1=FloatVector.zero(S);var a2=FloatVector.zero(S);
            var c0=FloatVector.zero(S);var c1=FloatVector.zero(S);var b=FloatVector.zero(S);
            boolean active=false;
            var u0=vu0;var u1=vu1;var w0=vw0;var w1=vw1;var wt0=vwt0;var wt1=vwt1;
            // F64 may have only two preferred lanes: F3 then has a second masked feature chunk.
            for(int r=begin;r<end;r++) if(time<d.lengths().getAtIndex(JAVA_LONG,r)) {
                active=true;int h=2*r,p=2*(time*d.batch()+r),q=3*(time*d.batch()+r);
                var y=FloatVector.fromMemorySegment(S,d.saved(),(long)p*4,ORDER,H);
                var delta=FloatVector.fromMemorySegment(S,d.dh(),(long)h*4,ORDER,H)
                        .add(FloatVector.fromMemorySegment(S,d.emissionSeed(),(long)p*4,ORDER,H)).mul(FloatVector.broadcast(S,derivative).sub(y.mul(y)));
                
                float d0=delta.lane(0),d1=delta.lane(1);
                FloatVector.zero(S).add(w0.mul(d0)).add(w1.mul(d1)).intoMemorySegment(d.dx(),(long)q*4,ORDER,X);
                if(S.length()<3)FloatVector.zero(S).add(wt0.mul(d0)).add(wt1.mul(d1))
                        .intoMemorySegment(d.dx(),(long)(q+2)*4,ORDER,TAIL);
                FloatVector.zero(S).add(u0.mul(d0)).add(u1.mul(d1)).intoMemorySegment(d.dh(),(long)h*4,ORDER,H);
                float x0=d.x().getAtIndex(JAVA_FLOAT,q),x1=d.x().getAtIndex(JAVA_FLOAT,q+1),x2=d.x().getAtIndex(JAVA_FLOAT,q+2);
                var prev=FloatVector.fromMemorySegment(S,d.previous(),(long)p*4,ORDER,H);
                float h0=prev.lane(0),h1=prev.lane(1);
                a0=a0.add(delta.mul(x0));a1=a1.add(delta.mul(x1));a2=a2.add(delta.mul(x2));
                c0=c0.add(delta.mul(h0));c1=c1.add(delta.mul(h1));
                b=d.batch()==1?delta:b.add(delta);
            }
            valid[id]=active?1:0;long p=(long)id*partialCells*4;
            a0.intoMemorySegment(partials,p,ORDER,H);a1.intoMemorySegment(partials,p+8,ORDER,H);
            a2.intoMemorySegment(partials,p+16,ORDER,H);c0.intoMemorySegment(partials,p+24,ORDER,H);
            c1.intoMemorySegment(partials,p+32,ORDER,H);
            b.intoMemorySegment(partials,p+40,ORDER,H);
        }

        private void mergeScalar(boolean carry) {
            for(int k=0;k<(carry?12:10);k++) {
                float sum=0f;boolean seen=false;
                for(int r=0;r<count;r++)if(valid[r]!=0) {
                    float part=partials.getAtIndex(JAVA_FLOAT,partialCells*r+k);
                    sum=seen?sum+part:part;seen=true;
                }
                MemorySegment out=k<6?d.dw():k<10?d.du():d.db();int index=k<6?k:k<10?k-6:k-10;
                out.setAtIndex(JAVA_FLOAT,index,carry?out.getAtIndex(JAVA_FLOAT,index)+sum:sum);
            }
        }
        private void mergeVector(boolean carry) {
            for(int k=0;k<(carry?12:10);k+=2) {
                var sum=FloatVector.zero(S);boolean seen=false;
                for(int r=0;r<count;r++)if(valid[r]!=0) {
                    var part=FloatVector.fromMemorySegment(S,partials,(long)(partialCells*r+k)*4,ORDER,H);
                    sum=seen?sum.add(part):part;seen=true;
                }
                MemorySegment out=k<6?d.dw():k<10?d.du():d.db();int index=k<6?k:k<10?k-6:k-10;
                if(carry)sum=FloatVector.fromMemorySegment(S,out,(long)index*4,ORDER,H).add(sum);
                sum.intoMemorySegment(out,(long)index*4,ORDER,H);
            }
        }
        /** Inline ordinary scalar reduction has no communication storage. */
        private void inlineScalarOrdinary() {
            if(d.batch()==0){d.dw().fill((byte)0);d.du().fill((byte)0);return;}
            float a00=0f,a01=0f,a10=0f,a11=0f,a20=0f,a21=0f;
            float c00=0f,c01=0f,c10=0f,c11=0f,b0=0f,b1=0f;
            final float w00=this.w00,w01=this.w01,w10=this.w10,w11=this.w11,w20=this.w20,w21=this.w21;
            final float u00=this.u00,u01=this.u01,u10=this.u10,u11=this.u11;
            for(int r=0;r<d.batch();r++)  {
int h=2*r,p=h,q=3*r;
                float y0=d.saved().getAtIndex(JAVA_FLOAT,p),y1=d.saved().getAtIndex(JAVA_FLOAT,p+1);
                float d0=(seed*y0+seed*y0)*(derivative-y0*y0),d1=(seed*y1+seed*y1)*(derivative-y1*y1);
                d.delta().setAtIndex(JAVA_FLOAT,h,d0);d.delta().setAtIndex(JAVA_FLOAT,h+1,d1);
                float x0=d.x().getAtIndex(JAVA_FLOAT,q),x1=d.x().getAtIndex(JAVA_FLOAT,q+1),x2=d.x().getAtIndex(JAVA_FLOAT,q+2);
                float h0=d.previous().getAtIndex(JAVA_FLOAT,p),h1=d.previous().getAtIndex(JAVA_FLOAT,p+1);
                d.dx().setAtIndex(JAVA_FLOAT,q,(0f+d0*w00)+d1*w01);
                d.dx().setAtIndex(JAVA_FLOAT,q+1,(0f+d0*w10)+d1*w11);
                d.dx().setAtIndex(JAVA_FLOAT,q+2,(0f+d0*w20)+d1*w21);
                d.dh().setAtIndex(JAVA_FLOAT,h,(0f+d0*u00)+d1*u01);
                d.dh().setAtIndex(JAVA_FLOAT,h+1,(0f+d0*u10)+d1*u11);
                a00+=x0*d0;a01+=x0*d1;a10+=x1*d0;a11+=x1*d1;a20+=x2*d0;a21+=x2*d1;
                c00+=h0*d0;c01+=h0*d1;c10+=h1*d0;c11+=h1*d1;
                
            }
            d.dw().setAtIndex(JAVA_FLOAT,0,a00);
            d.dw().setAtIndex(JAVA_FLOAT,1,a01);
            d.dw().setAtIndex(JAVA_FLOAT,2,a10);
            d.dw().setAtIndex(JAVA_FLOAT,3,a11);
            d.dw().setAtIndex(JAVA_FLOAT,4,a20);
            d.dw().setAtIndex(JAVA_FLOAT,5,a21);
            d.du().setAtIndex(JAVA_FLOAT,0,c00);
            d.du().setAtIndex(JAVA_FLOAT,1,c01);
            d.du().setAtIndex(JAVA_FLOAT,2,c10);
            d.du().setAtIndex(JAVA_FLOAT,3,c11);
        }

        /**
         * Retains twelve typed scalar carry locals across time, with a separate ADD at each active time.
         * @param maximum prevalidated maximum row length in [0,T], zero for an empty/all-inactive domain
         */
        private void inlineScalarScan(long maximum) {
            float carry0=initial,carry1=initial,carry2=initial,carry3=initial,carry4=initial,carry5=initial,carry6=initial,carry7=initial,carry8=initial,carry9=initial,carry10=initial,carry11=initial;
            for(int step=0;step<d.time();step++) {
                time=d.reverse()?step:d.time()-1-step;if(time>=maximum)continue;

            float a00=0f,a01=0f,a10=0f,a11=0f,a20=0f,a21=0f;
            float c00=0f,c01=0f,c10=0f,c11=0f,b0=0f,b1=0f;
            final float w00=this.w00,w01=this.w01,w10=this.w10,w11=this.w11,w20=this.w20,w21=this.w21;
            final float u00=this.u00,u01=this.u01,u10=this.u10,u11=this.u11;
            for(int r=0;r<d.batch();r++) if(time<d.lengths().getAtIndex(JAVA_LONG,r)) {
int h=2*r,p=2*(time*d.batch()+r),q=3*(time*d.batch()+r);
                float y0=d.saved().getAtIndex(JAVA_FLOAT,p),y1=d.saved().getAtIndex(JAVA_FLOAT,p+1);
                float d0=(d.dh().getAtIndex(JAVA_FLOAT,h)+d.emissionSeed().getAtIndex(JAVA_FLOAT,p))*(derivative-y0*y0);
                float d1=(d.dh().getAtIndex(JAVA_FLOAT,h+1)+d.emissionSeed().getAtIndex(JAVA_FLOAT,p+1))*(derivative-y1*y1);
                float x0=d.x().getAtIndex(JAVA_FLOAT,q),x1=d.x().getAtIndex(JAVA_FLOAT,q+1),x2=d.x().getAtIndex(JAVA_FLOAT,q+2);
                float h0=d.previous().getAtIndex(JAVA_FLOAT,p),h1=d.previous().getAtIndex(JAVA_FLOAT,p+1);
                d.dx().setAtIndex(JAVA_FLOAT,q,(0f+d0*w00)+d1*w01);
                d.dx().setAtIndex(JAVA_FLOAT,q+1,(0f+d0*w10)+d1*w11);
                d.dx().setAtIndex(JAVA_FLOAT,q+2,(0f+d0*w20)+d1*w21);
                d.dh().setAtIndex(JAVA_FLOAT,h,(0f+d0*u00)+d1*u01);
                d.dh().setAtIndex(JAVA_FLOAT,h+1,(0f+d0*u10)+d1*u11);
                a00+=x0*d0;a01+=x0*d1;a10+=x1*d0;a11+=x1*d1;a20+=x2*d0;a21+=x2*d1;
                c00+=h0*d0;c01+=h0*d1;c10+=h1*d0;c11+=h1*d1;
                if(d.batch()==1){b0=d0;b1=d1;}else{b0+=d0;b1+=d1;}
            }
            carry0+=a00;
            carry1+=a01;
            carry2+=a10;
            carry3+=a11;
            carry4+=a20;
            carry5+=a21;
            carry6+=c00;
            carry7+=c01;
            carry8+=c10;
            carry9+=c11;
            carry10+=b0;
            carry11+=b1;
            }
            d.dw().setAtIndex(JAVA_FLOAT,0,carry0);
            d.dw().setAtIndex(JAVA_FLOAT,1,carry1);
            d.dw().setAtIndex(JAVA_FLOAT,2,carry2);
            d.dw().setAtIndex(JAVA_FLOAT,3,carry3);
            d.dw().setAtIndex(JAVA_FLOAT,4,carry4);
            d.dw().setAtIndex(JAVA_FLOAT,5,carry5);
            d.du().setAtIndex(JAVA_FLOAT,0,carry6);
            d.du().setAtIndex(JAVA_FLOAT,1,carry7);
            d.du().setAtIndex(JAVA_FLOAT,2,carry8);
            d.du().setAtIndex(JAVA_FLOAT,3,carry9);
            d.db().setAtIndex(JAVA_FLOAT,0,carry10);
            d.db().setAtIndex(JAVA_FLOAT,1,carry11);
        }

        /** Inline ordinary reduction publishes locals directly; no partial buffer or merge. */
        private void inlineVectorOrdinary() {
            if(d.batch()==0){d.dw().fill((byte)0);d.du().fill((byte)0);return;}
            var a0=FloatVector.zero(S);var a1=FloatVector.zero(S);var a2=FloatVector.zero(S);
            var c0=FloatVector.zero(S);var c1=FloatVector.zero(S);var b=FloatVector.zero(S);
            var u0=vu0;var u1=vu1;var w0=vw0;var w1=vw1;var wt0=vwt0;var wt1=vwt1;
            // F64 may have only two preferred lanes: F3 then has a second masked feature chunk.
            for(int r=0;r<d.batch();r++)  {
int h=2*r,p=h,q=3*r;
                var y=FloatVector.fromMemorySegment(S,d.saved(),(long)p*4,ORDER,H);
                var delta=y.mul(seed).add(y.mul(seed)).mul(FloatVector.broadcast(S,derivative).sub(y.mul(y)));
                delta.intoMemorySegment(d.delta(),(long)h*4,ORDER,H);
                float d0=delta.lane(0),d1=delta.lane(1);
                FloatVector.zero(S).add(w0.mul(d0)).add(w1.mul(d1)).intoMemorySegment(d.dx(),(long)q*4,ORDER,X);
                if(S.length()<3)FloatVector.zero(S).add(wt0.mul(d0)).add(wt1.mul(d1))
                        .intoMemorySegment(d.dx(),(long)(q+2)*4,ORDER,TAIL);
                FloatVector.zero(S).add(u0.mul(d0)).add(u1.mul(d1)).intoMemorySegment(d.dh(),(long)h*4,ORDER,H);
                float x0=d.x().getAtIndex(JAVA_FLOAT,q),x1=d.x().getAtIndex(JAVA_FLOAT,q+1),x2=d.x().getAtIndex(JAVA_FLOAT,q+2);
                var prev=FloatVector.fromMemorySegment(S,d.previous(),(long)p*4,ORDER,H);
                float h0=prev.lane(0),h1=prev.lane(1);
                a0=a0.add(delta.mul(x0));a1=a1.add(delta.mul(x1));a2=a2.add(delta.mul(x2));
                c0=c0.add(delta.mul(h0));c1=c1.add(delta.mul(h1));
                
            }
            a0.intoMemorySegment(d.dw(),0L,ORDER,H);
            a1.intoMemorySegment(d.dw(),8L,ORDER,H);
            a2.intoMemorySegment(d.dw(),16L,ORDER,H);
            c0.intoMemorySegment(d.du(),0L,ORDER,H);
            c1.intoMemorySegment(d.du(),8L,ORDER,H);
        }

        /**
         * Retains twelve carry cells in six H2 vectors across time, without flattening time contributions.
         * @param maximum prevalidated maximum row length in [0,T], zero for an empty/all-inactive domain
         */
        private void inlineVectorScan(long maximum) {
            var cw0=FloatVector.broadcast(S,initial);var cw1=cw0;var cw2=cw0;var cu0=cw0;var cu1=cw0;var cb=cw0;
            for(int step=0;step<d.time();step++) {
                time=d.reverse()?step:d.time()-1-step;if(time>=maximum)continue;

            var a0=FloatVector.zero(S);var a1=FloatVector.zero(S);var a2=FloatVector.zero(S);
            var c0=FloatVector.zero(S);var c1=FloatVector.zero(S);var b=FloatVector.zero(S);
            var u0=vu0;var u1=vu1;var w0=vw0;var w1=vw1;var wt0=vwt0;var wt1=vwt1;
            // F64 may have only two preferred lanes: F3 then has a second masked feature chunk.
            for(int r=0;r<d.batch();r++) if(time<d.lengths().getAtIndex(JAVA_LONG,r)) {
int h=2*r,p=2*(time*d.batch()+r),q=3*(time*d.batch()+r);
                var y=FloatVector.fromMemorySegment(S,d.saved(),(long)p*4,ORDER,H);
                var delta=FloatVector.fromMemorySegment(S,d.dh(),(long)h*4,ORDER,H)
                        .add(FloatVector.fromMemorySegment(S,d.emissionSeed(),(long)p*4,ORDER,H)).mul(FloatVector.broadcast(S,derivative).sub(y.mul(y)));
                
                float d0=delta.lane(0),d1=delta.lane(1);
                FloatVector.zero(S).add(w0.mul(d0)).add(w1.mul(d1)).intoMemorySegment(d.dx(),(long)q*4,ORDER,X);
                if(S.length()<3)FloatVector.zero(S).add(wt0.mul(d0)).add(wt1.mul(d1))
                        .intoMemorySegment(d.dx(),(long)(q+2)*4,ORDER,TAIL);
                FloatVector.zero(S).add(u0.mul(d0)).add(u1.mul(d1)).intoMemorySegment(d.dh(),(long)h*4,ORDER,H);
                float x0=d.x().getAtIndex(JAVA_FLOAT,q),x1=d.x().getAtIndex(JAVA_FLOAT,q+1),x2=d.x().getAtIndex(JAVA_FLOAT,q+2);
                var prev=FloatVector.fromMemorySegment(S,d.previous(),(long)p*4,ORDER,H);
                float h0=prev.lane(0),h1=prev.lane(1);
                a0=a0.add(delta.mul(x0));a1=a1.add(delta.mul(x1));a2=a2.add(delta.mul(x2));
                c0=c0.add(delta.mul(h0));c1=c1.add(delta.mul(h1));
                b=d.batch()==1?delta:b.add(delta);
            }
            cw0=cw0.add(a0);
            cw1=cw1.add(a1);
            cw2=cw2.add(a2);
            cu0=cu0.add(c0);
            cu1=cu1.add(c1);
            cb=cb.add(b);
            }
            cw0.intoMemorySegment(d.dw(),0L,ORDER,H);
            cw1.intoMemorySegment(d.dw(),8L,ORDER,H);
            cw2.intoMemorySegment(d.dw(),16L,ORDER,H);
            cu0.intoMemorySegment(d.du(),0L,ORDER,H);
            cu1.intoMemorySegment(d.du(),8L,ORDER,H);
            cb.intoMemorySegment(d.db(),0L,ORDER,H);
        }

        /**
         * Merges valid ranges on the caller and keeps twelve typed carry cells local across T.
         * Each active time retains its own ADD; workers never update these shared carries.
         * @param lease live same-budget outer reservation, borrowed exclusively through join and not closed here
         * @param maximum prevalidated maximum row length in [0,T], zero for an empty/all-inactive domain
         */
        private void parallelScalarScan(CpuConcurrencyBudget.Lease lease,long maximum) {
            float carry0=initial,carry1=initial,carry2=initial,carry3=initial,carry4=initial,carry5=initial,carry6=initial,carry7=initial,carry8=initial,carry9=initial,carry10=initial,carry11=initial;
            for(int step=0;step<d.time();step++) {
                time=d.reverse()?step:d.time()-1-step;if(time>=maximum)continue;
                dispatch.accept(lease);
                float a00=0f,a01=0f,a10=0f,a11=0f,a20=0f,a21=0f,c00=0f,c01=0f,c10=0f,c11=0f,b0=0f,b1=0f;boolean seen=false;
                for(int r=0;r<count;r++)if(valid[r]!=0) {
                    float part0=partials.getAtIndex(JAVA_FLOAT,r*12+0);a00=seen?a00+part0:part0;
                    float part1=partials.getAtIndex(JAVA_FLOAT,r*12+1);a01=seen?a01+part1:part1;
                    float part2=partials.getAtIndex(JAVA_FLOAT,r*12+2);a10=seen?a10+part2:part2;
                    float part3=partials.getAtIndex(JAVA_FLOAT,r*12+3);a11=seen?a11+part3:part3;
                    float part4=partials.getAtIndex(JAVA_FLOAT,r*12+4);a20=seen?a20+part4:part4;
                    float part5=partials.getAtIndex(JAVA_FLOAT,r*12+5);a21=seen?a21+part5:part5;
                    float part6=partials.getAtIndex(JAVA_FLOAT,r*12+6);c00=seen?c00+part6:part6;
                    float part7=partials.getAtIndex(JAVA_FLOAT,r*12+7);c01=seen?c01+part7:part7;
                    float part8=partials.getAtIndex(JAVA_FLOAT,r*12+8);c10=seen?c10+part8:part8;
                    float part9=partials.getAtIndex(JAVA_FLOAT,r*12+9);c11=seen?c11+part9:part9;
                    float part10=partials.getAtIndex(JAVA_FLOAT,r*12+10);b0=seen?b0+part10:part10;
                    float part11=partials.getAtIndex(JAVA_FLOAT,r*12+11);b1=seen?b1+part11:part11;
                    seen=true;
                }
                carry0+=a00;carry1+=a01;carry2+=a10;carry3+=a11;carry4+=a20;carry5+=a21;carry6+=c00;carry7+=c01;carry8+=c10;carry9+=c11;carry10+=b0;carry11+=b1;
            }
            d.dw().setAtIndex(JAVA_FLOAT,0,carry0);
            d.dw().setAtIndex(JAVA_FLOAT,1,carry1);
            d.dw().setAtIndex(JAVA_FLOAT,2,carry2);
            d.dw().setAtIndex(JAVA_FLOAT,3,carry3);
            d.dw().setAtIndex(JAVA_FLOAT,4,carry4);
            d.dw().setAtIndex(JAVA_FLOAT,5,carry5);
            d.du().setAtIndex(JAVA_FLOAT,0,carry6);
            d.du().setAtIndex(JAVA_FLOAT,1,carry7);
            d.du().setAtIndex(JAVA_FLOAT,2,carry8);
            d.du().setAtIndex(JAVA_FLOAT,3,carry9);
            d.db().setAtIndex(JAVA_FLOAT,0,carry10);
            d.db().setAtIndex(JAVA_FLOAT,1,carry11);
        }

        /**
         * Merges valid ranges on the caller and keeps twelve typed carry cells local across T.
         * Each active time retains its own ADD; workers never update these shared carries.
         * @param lease live same-budget outer reservation, borrowed exclusively through join and not closed here
         * @param maximum prevalidated maximum row length in [0,T], zero for an empty/all-inactive domain
         */
        private void parallelVectorScan(CpuConcurrencyBudget.Lease lease,long maximum) {
            var carry0=FloatVector.broadcast(S,initial);var carry1=FloatVector.broadcast(S,initial);var carry2=FloatVector.broadcast(S,initial);var carry3=FloatVector.broadcast(S,initial);var carry4=FloatVector.broadcast(S,initial);var carry5=FloatVector.broadcast(S,initial);
            for(int step=0;step<d.time();step++) {
                time=d.reverse()?step:d.time()-1-step;if(time>=maximum)continue;
                dispatch.accept(lease);
                var a0=FloatVector.zero(S);var a1=FloatVector.zero(S);var a2=FloatVector.zero(S);var c0=FloatVector.zero(S);var c1=FloatVector.zero(S);var b=FloatVector.zero(S);boolean seen=false;
                for(int r=0;r<count;r++)if(valid[r]!=0) {
                    var part0=FloatVector.fromMemorySegment(S,partials,(long)(r*12+0)*4,ORDER,H);a0=seen?a0.add(part0):part0;
                    var part1=FloatVector.fromMemorySegment(S,partials,(long)(r*12+2)*4,ORDER,H);a1=seen?a1.add(part1):part1;
                    var part2=FloatVector.fromMemorySegment(S,partials,(long)(r*12+4)*4,ORDER,H);a2=seen?a2.add(part2):part2;
                    var part3=FloatVector.fromMemorySegment(S,partials,(long)(r*12+6)*4,ORDER,H);c0=seen?c0.add(part3):part3;
                    var part4=FloatVector.fromMemorySegment(S,partials,(long)(r*12+8)*4,ORDER,H);c1=seen?c1.add(part4):part4;
                    var part5=FloatVector.fromMemorySegment(S,partials,(long)(r*12+10)*4,ORDER,H);b=seen?b.add(part5):part5;
                    seen=true;
                }
                carry0=carry0.add(a0);carry1=carry1.add(a1);carry2=carry2.add(a2);carry3=carry3.add(c0);carry4=carry4.add(c1);carry5=carry5.add(b);
            }
            carry0.intoMemorySegment(d.dw(),0L,ORDER,H);
            carry1.intoMemorySegment(d.dw(),8L,ORDER,H);
            carry2.intoMemorySegment(d.dw(),16L,ORDER,H);
            carry3.intoMemorySegment(d.du(),0L,ORDER,H);
            carry4.intoMemorySegment(d.du(),8L,ORDER,H);
            carry5.intoMemorySegment(d.db(),0L,ORDER,H);
        }

    }

    /** One typed run owner; fields transport fixed range state, never runtime graph decisions. */
    private static final class F64 implements Runnable {
        private static final VectorSpecies<Double> S=DoubleVector.SPECIES_PREFERRED;
        private static final ByteOrder ORDER=ByteOrder.nativeOrder();
        private static final VectorMask<Double> H=S.indexInRange(0,2);
        private static final VectorMask<Double> X=S.indexInRange(0,3);
        private static final VectorMask<Double> TAIL=S.indexInRange(0,1);
        private final Data d;
        private final boolean vector;
        private final int count,partialCells;
        private final MemorySegment transpose,partials;
        private final long[] valid;
        private final Runnable[] ranges;
        private final Consumer<CpuConcurrencyBudget.Lease> dispatch;
        private final CpuConcurrencyBudget budget;
        private int time;
        private double w00,w01,w10,w11,w20,w21,u00,u01,u10,u11;
        private final double seed,derivative,initial;
        private DoubleVector vw0,vw1,vwt0,vwt1,vu0,vu1;
        F64(Data d,boolean vector,boolean parallel,long minimumElementsPerWorker,Fixture fixture,Arena arena) {
            this.d=d;this.vector=vector;partialCells=d.scan()?12:10;count=d.batch()==0?0:parallel?(int)Math.min(2,1+(d.batch()-1)/minimumElementsPerWorker):1;
            transpose=vector?arena.allocate(10L*8,8):MemorySegment.NULL;
            partials=count>1?arena.allocate((long)partialCells*count*8,8):MemorySegment.NULL;
            valid=count>1?new long[count]:null;ranges=count>1?new Runnable[count]:null;
            budget=fixture.rowOracleBudget();
            seed=d.seed();derivative=d.derivative();initial=d.initial();
            for(int i=0;count>1&&i<count;i++) {
                int quotient=d.batch()/count,remainder=d.batch()%count;
                int id=i,begin=i*quotient+Math.min(i,remainder),end=begin+quotient+(i<remainder?1:0);
                ranges[i]=d.scan() ? vector?()->vectorScan(id,begin,end):()->scalarScan(id,begin,end)
                        : vector?()->vectorOrdinary(id,begin,end):()->scalarOrdinary(id,begin,end);
            }
            dispatch=count>1?fixture.bindRowOracleRanges(ranges):null;
        }
        @Override public void run() {
            try(var lease=budget==null?null:budget.acquire(Math.max(1,count))) {
                if(d.scan()) {
                    d.dh().copyFrom(d.finalSeed());d.dx().fill((byte)0);
                    long maximum=0;
                    for(int r=0;r<d.batch();r++)maximum=Math.max(maximum,d.lengths().getAtIndex(JAVA_LONG,r));
                    if(count<2) {
                        if(maximum>0&&d.time()>0)stage();
                        if(vector)inlineVectorScan(maximum);else inlineScalarScan(maximum);
                        return;
                    }
                    if(maximum>0&&d.time()>0)stage();
                    if(vector)parallelVectorScan(lease,maximum);else parallelScalarScan(lease,maximum);
                } else {
                    if(count<2) {
                        if(d.batch()>0)stage();
                        if(vector)inlineVectorOrdinary();else inlineScalarOrdinary();
                    } else {
                        stage();execute(lease);
                        if(vector)mergeVector(false);else mergeScalar(false);
                    }
                }
            }
        }
        private void execute(CpuConcurrencyBudget.Lease lease) {
            if(count>1)dispatch.accept(lease);else if(count==1)ranges[0].run();
        }
        private void stage() {
            w00=d.w().getAtIndex(JAVA_DOUBLE,0);w01=d.w().getAtIndex(JAVA_DOUBLE,1);
            w10=d.w().getAtIndex(JAVA_DOUBLE,2);w11=d.w().getAtIndex(JAVA_DOUBLE,3);
            w20=d.w().getAtIndex(JAVA_DOUBLE,4);w21=d.w().getAtIndex(JAVA_DOUBLE,5);
            u00=d.u().getAtIndex(JAVA_DOUBLE,0);u01=d.u().getAtIndex(JAVA_DOUBLE,1);
            u10=d.u().getAtIndex(JAVA_DOUBLE,2);u11=d.u().getAtIndex(JAVA_DOUBLE,3);
            if(vector) {
                transpose.setAtIndex(JAVA_DOUBLE,0,w00);transpose.setAtIndex(JAVA_DOUBLE,1,w10);transpose.setAtIndex(JAVA_DOUBLE,2,w20);
                transpose.setAtIndex(JAVA_DOUBLE,3,w01);transpose.setAtIndex(JAVA_DOUBLE,4,w11);transpose.setAtIndex(JAVA_DOUBLE,5,w21);
                transpose.setAtIndex(JAVA_DOUBLE,6,u00);transpose.setAtIndex(JAVA_DOUBLE,7,u10);
                transpose.setAtIndex(JAVA_DOUBLE,8,u01);transpose.setAtIndex(JAVA_DOUBLE,9,u11);
                // Immutable, nonaliasing coefficients are loaded once per invocation, not time.
                vu0=DoubleVector.fromMemorySegment(S,transpose,48L,ORDER,H);
                vu1=DoubleVector.fromMemorySegment(S,transpose,64L,ORDER,H);
                vw0=DoubleVector.fromMemorySegment(S,transpose,0,ORDER,X);
                vw1=DoubleVector.fromMemorySegment(S,transpose,24L,ORDER,X);
                vwt0=S.length()<3?DoubleVector.fromMemorySegment(S,transpose,16L,ORDER,TAIL):DoubleVector.zero(S);
                vwt1=S.length()<3?DoubleVector.fromMemorySegment(S,transpose,40L,ORDER,TAIL):DoubleVector.zero(S);
            }
        }

        private void scalarOrdinary(int id,int begin,int end) {
            double a00=0d,a01=0d,a10=0d,a11=0d,a20=0d,a21=0d;
            double c00=0d,c01=0d,c10=0d,c11=0d,b0=0d,b1=0d;
            boolean active=false;
            final double w00=this.w00,w01=this.w01,w10=this.w10,w11=this.w11,w20=this.w20,w21=this.w21;
            final double u00=this.u00,u01=this.u01,u10=this.u10,u11=this.u11;
            for(int r=begin;r<end;r++)  {
                active=true;int h=2*r,p=h,q=3*r;
                double y0=d.saved().getAtIndex(JAVA_DOUBLE,p),y1=d.saved().getAtIndex(JAVA_DOUBLE,p+1);
                double d0=(seed*y0+seed*y0)*(derivative-y0*y0),d1=(seed*y1+seed*y1)*(derivative-y1*y1);
                d.delta().setAtIndex(JAVA_DOUBLE,h,d0);d.delta().setAtIndex(JAVA_DOUBLE,h+1,d1);
                double x0=d.x().getAtIndex(JAVA_DOUBLE,q),x1=d.x().getAtIndex(JAVA_DOUBLE,q+1),x2=d.x().getAtIndex(JAVA_DOUBLE,q+2);
                double h0=d.previous().getAtIndex(JAVA_DOUBLE,p),h1=d.previous().getAtIndex(JAVA_DOUBLE,p+1);
                d.dx().setAtIndex(JAVA_DOUBLE,q,(0d+d0*w00)+d1*w01);
                d.dx().setAtIndex(JAVA_DOUBLE,q+1,(0d+d0*w10)+d1*w11);
                d.dx().setAtIndex(JAVA_DOUBLE,q+2,(0d+d0*w20)+d1*w21);
                d.dh().setAtIndex(JAVA_DOUBLE,h,(0d+d0*u00)+d1*u01);
                d.dh().setAtIndex(JAVA_DOUBLE,h+1,(0d+d0*u10)+d1*u11);
                a00+=x0*d0;a01+=x0*d1;a10+=x1*d0;a11+=x1*d1;a20+=x2*d0;a21+=x2*d1;
                c00+=h0*d0;c01+=h0*d1;c10+=h1*d0;c11+=h1*d1;
                
            }
            valid[id]=active?1:0;int p=id*partialCells;
            partials.setAtIndex(JAVA_DOUBLE,p,a00);partials.setAtIndex(JAVA_DOUBLE,p+1,a01);
            partials.setAtIndex(JAVA_DOUBLE,p+2,a10);partials.setAtIndex(JAVA_DOUBLE,p+3,a11);
            partials.setAtIndex(JAVA_DOUBLE,p+4,a20);partials.setAtIndex(JAVA_DOUBLE,p+5,a21);
            partials.setAtIndex(JAVA_DOUBLE,p+6,c00);partials.setAtIndex(JAVA_DOUBLE,p+7,c01);
            partials.setAtIndex(JAVA_DOUBLE,p+8,c10);partials.setAtIndex(JAVA_DOUBLE,p+9,c11);
            
        }
        private void vectorOrdinary(int id,int begin,int end) {
            var a0=DoubleVector.zero(S);var a1=DoubleVector.zero(S);var a2=DoubleVector.zero(S);
            var c0=DoubleVector.zero(S);var c1=DoubleVector.zero(S);var b=DoubleVector.zero(S);
            boolean active=false;
            var u0=vu0;var u1=vu1;var w0=vw0;var w1=vw1;var wt0=vwt0;var wt1=vwt1;
            // F64 may have only two preferred lanes: F3 then has a second masked feature chunk.
            for(int r=begin;r<end;r++)  {
                active=true;int h=2*r,p=h,q=3*r;
                var y=DoubleVector.fromMemorySegment(S,d.saved(),(long)p*8,ORDER,H);
                var delta=y.mul(seed).add(y.mul(seed)).mul(DoubleVector.broadcast(S,derivative).sub(y.mul(y)));
                delta.intoMemorySegment(d.delta(),(long)h*8,ORDER,H);
                double d0=delta.lane(0),d1=delta.lane(1);
                DoubleVector.zero(S).add(w0.mul(d0)).add(w1.mul(d1)).intoMemorySegment(d.dx(),(long)q*8,ORDER,X);
                if(S.length()<3)DoubleVector.zero(S).add(wt0.mul(d0)).add(wt1.mul(d1))
                        .intoMemorySegment(d.dx(),(long)(q+2)*8,ORDER,TAIL);
                DoubleVector.zero(S).add(u0.mul(d0)).add(u1.mul(d1)).intoMemorySegment(d.dh(),(long)h*8,ORDER,H);
                double x0=d.x().getAtIndex(JAVA_DOUBLE,q),x1=d.x().getAtIndex(JAVA_DOUBLE,q+1),x2=d.x().getAtIndex(JAVA_DOUBLE,q+2);
                var prev=DoubleVector.fromMemorySegment(S,d.previous(),(long)p*8,ORDER,H);
                double h0=prev.lane(0),h1=prev.lane(1);
                a0=a0.add(delta.mul(x0));a1=a1.add(delta.mul(x1));a2=a2.add(delta.mul(x2));
                c0=c0.add(delta.mul(h0));c1=c1.add(delta.mul(h1));
                
            }
            valid[id]=active?1:0;long p=(long)id*partialCells*8;
            a0.intoMemorySegment(partials,p,ORDER,H);a1.intoMemorySegment(partials,p+16,ORDER,H);
            a2.intoMemorySegment(partials,p+32,ORDER,H);c0.intoMemorySegment(partials,p+48,ORDER,H);
            c1.intoMemorySegment(partials,p+64,ORDER,H);
            
        }

        private void scalarScan(int id,int begin,int end) {
            double a00=0d,a01=0d,a10=0d,a11=0d,a20=0d,a21=0d;
            double c00=0d,c01=0d,c10=0d,c11=0d,b0=0d,b1=0d;
            boolean active=false;
            final double w00=this.w00,w01=this.w01,w10=this.w10,w11=this.w11,w20=this.w20,w21=this.w21;
            final double u00=this.u00,u01=this.u01,u10=this.u10,u11=this.u11;
            for(int r=begin;r<end;r++) if(time<d.lengths().getAtIndex(JAVA_LONG,r)) {
                active=true;int h=2*r,p=2*(time*d.batch()+r),q=3*(time*d.batch()+r);
                double y0=d.saved().getAtIndex(JAVA_DOUBLE,p),y1=d.saved().getAtIndex(JAVA_DOUBLE,p+1);
                double d0=(d.dh().getAtIndex(JAVA_DOUBLE,h)+d.emissionSeed().getAtIndex(JAVA_DOUBLE,p))*(derivative-y0*y0);
                double d1=(d.dh().getAtIndex(JAVA_DOUBLE,h+1)+d.emissionSeed().getAtIndex(JAVA_DOUBLE,p+1))*(derivative-y1*y1);
                double x0=d.x().getAtIndex(JAVA_DOUBLE,q),x1=d.x().getAtIndex(JAVA_DOUBLE,q+1),x2=d.x().getAtIndex(JAVA_DOUBLE,q+2);
                double h0=d.previous().getAtIndex(JAVA_DOUBLE,p),h1=d.previous().getAtIndex(JAVA_DOUBLE,p+1);
                d.dx().setAtIndex(JAVA_DOUBLE,q,(0d+d0*w00)+d1*w01);
                d.dx().setAtIndex(JAVA_DOUBLE,q+1,(0d+d0*w10)+d1*w11);
                d.dx().setAtIndex(JAVA_DOUBLE,q+2,(0d+d0*w20)+d1*w21);
                d.dh().setAtIndex(JAVA_DOUBLE,h,(0d+d0*u00)+d1*u01);
                d.dh().setAtIndex(JAVA_DOUBLE,h+1,(0d+d0*u10)+d1*u11);
                a00+=x0*d0;a01+=x0*d1;a10+=x1*d0;a11+=x1*d1;a20+=x2*d0;a21+=x2*d1;
                c00+=h0*d0;c01+=h0*d1;c10+=h1*d0;c11+=h1*d1;
                if(d.batch()==1){b0=d0;b1=d1;}else{b0+=d0;b1+=d1;}
            }
            valid[id]=active?1:0;int p=id*partialCells;
            partials.setAtIndex(JAVA_DOUBLE,p,a00);partials.setAtIndex(JAVA_DOUBLE,p+1,a01);
            partials.setAtIndex(JAVA_DOUBLE,p+2,a10);partials.setAtIndex(JAVA_DOUBLE,p+3,a11);
            partials.setAtIndex(JAVA_DOUBLE,p+4,a20);partials.setAtIndex(JAVA_DOUBLE,p+5,a21);
            partials.setAtIndex(JAVA_DOUBLE,p+6,c00);partials.setAtIndex(JAVA_DOUBLE,p+7,c01);
            partials.setAtIndex(JAVA_DOUBLE,p+8,c10);partials.setAtIndex(JAVA_DOUBLE,p+9,c11);
            partials.setAtIndex(JAVA_DOUBLE,p+10,b0);partials.setAtIndex(JAVA_DOUBLE,p+11,b1);
        }
        private void vectorScan(int id,int begin,int end) {
            var a0=DoubleVector.zero(S);var a1=DoubleVector.zero(S);var a2=DoubleVector.zero(S);
            var c0=DoubleVector.zero(S);var c1=DoubleVector.zero(S);var b=DoubleVector.zero(S);
            boolean active=false;
            var u0=vu0;var u1=vu1;var w0=vw0;var w1=vw1;var wt0=vwt0;var wt1=vwt1;
            // F64 may have only two preferred lanes: F3 then has a second masked feature chunk.
            for(int r=begin;r<end;r++) if(time<d.lengths().getAtIndex(JAVA_LONG,r)) {
                active=true;int h=2*r,p=2*(time*d.batch()+r),q=3*(time*d.batch()+r);
                var y=DoubleVector.fromMemorySegment(S,d.saved(),(long)p*8,ORDER,H);
                var delta=DoubleVector.fromMemorySegment(S,d.dh(),(long)h*8,ORDER,H)
                        .add(DoubleVector.fromMemorySegment(S,d.emissionSeed(),(long)p*8,ORDER,H)).mul(DoubleVector.broadcast(S,derivative).sub(y.mul(y)));
                
                double d0=delta.lane(0),d1=delta.lane(1);
                DoubleVector.zero(S).add(w0.mul(d0)).add(w1.mul(d1)).intoMemorySegment(d.dx(),(long)q*8,ORDER,X);
                if(S.length()<3)DoubleVector.zero(S).add(wt0.mul(d0)).add(wt1.mul(d1))
                        .intoMemorySegment(d.dx(),(long)(q+2)*8,ORDER,TAIL);
                DoubleVector.zero(S).add(u0.mul(d0)).add(u1.mul(d1)).intoMemorySegment(d.dh(),(long)h*8,ORDER,H);
                double x0=d.x().getAtIndex(JAVA_DOUBLE,q),x1=d.x().getAtIndex(JAVA_DOUBLE,q+1),x2=d.x().getAtIndex(JAVA_DOUBLE,q+2);
                var prev=DoubleVector.fromMemorySegment(S,d.previous(),(long)p*8,ORDER,H);
                double h0=prev.lane(0),h1=prev.lane(1);
                a0=a0.add(delta.mul(x0));a1=a1.add(delta.mul(x1));a2=a2.add(delta.mul(x2));
                c0=c0.add(delta.mul(h0));c1=c1.add(delta.mul(h1));
                b=d.batch()==1?delta:b.add(delta);
            }
            valid[id]=active?1:0;long p=(long)id*partialCells*8;
            a0.intoMemorySegment(partials,p,ORDER,H);a1.intoMemorySegment(partials,p+16,ORDER,H);
            a2.intoMemorySegment(partials,p+32,ORDER,H);c0.intoMemorySegment(partials,p+48,ORDER,H);
            c1.intoMemorySegment(partials,p+64,ORDER,H);
            b.intoMemorySegment(partials,p+80,ORDER,H);
        }

        private void mergeScalar(boolean carry) {
            for(int k=0;k<(carry?12:10);k++) {
                double sum=0d;boolean seen=false;
                for(int r=0;r<count;r++)if(valid[r]!=0) {
                    double part=partials.getAtIndex(JAVA_DOUBLE,partialCells*r+k);
                    sum=seen?sum+part:part;seen=true;
                }
                MemorySegment out=k<6?d.dw():k<10?d.du():d.db();int index=k<6?k:k<10?k-6:k-10;
                out.setAtIndex(JAVA_DOUBLE,index,carry?out.getAtIndex(JAVA_DOUBLE,index)+sum:sum);
            }
        }
        private void mergeVector(boolean carry) {
            for(int k=0;k<(carry?12:10);k+=2) {
                var sum=DoubleVector.zero(S);boolean seen=false;
                for(int r=0;r<count;r++)if(valid[r]!=0) {
                    var part=DoubleVector.fromMemorySegment(S,partials,(long)(partialCells*r+k)*8,ORDER,H);
                    sum=seen?sum.add(part):part;seen=true;
                }
                MemorySegment out=k<6?d.dw():k<10?d.du():d.db();int index=k<6?k:k<10?k-6:k-10;
                if(carry)sum=DoubleVector.fromMemorySegment(S,out,(long)index*8,ORDER,H).add(sum);
                sum.intoMemorySegment(out,(long)index*8,ORDER,H);
            }
        }
        /** Inline ordinary scalar reduction has no communication storage. */
        private void inlineScalarOrdinary() {
            if(d.batch()==0){d.dw().fill((byte)0);d.du().fill((byte)0);return;}
            double a00=0d,a01=0d,a10=0d,a11=0d,a20=0d,a21=0d;
            double c00=0d,c01=0d,c10=0d,c11=0d,b0=0d,b1=0d;
            final double w00=this.w00,w01=this.w01,w10=this.w10,w11=this.w11,w20=this.w20,w21=this.w21;
            final double u00=this.u00,u01=this.u01,u10=this.u10,u11=this.u11;
            for(int r=0;r<d.batch();r++)  {
int h=2*r,p=h,q=3*r;
                double y0=d.saved().getAtIndex(JAVA_DOUBLE,p),y1=d.saved().getAtIndex(JAVA_DOUBLE,p+1);
                double d0=(seed*y0+seed*y0)*(derivative-y0*y0),d1=(seed*y1+seed*y1)*(derivative-y1*y1);
                d.delta().setAtIndex(JAVA_DOUBLE,h,d0);d.delta().setAtIndex(JAVA_DOUBLE,h+1,d1);
                double x0=d.x().getAtIndex(JAVA_DOUBLE,q),x1=d.x().getAtIndex(JAVA_DOUBLE,q+1),x2=d.x().getAtIndex(JAVA_DOUBLE,q+2);
                double h0=d.previous().getAtIndex(JAVA_DOUBLE,p),h1=d.previous().getAtIndex(JAVA_DOUBLE,p+1);
                d.dx().setAtIndex(JAVA_DOUBLE,q,(0d+d0*w00)+d1*w01);
                d.dx().setAtIndex(JAVA_DOUBLE,q+1,(0d+d0*w10)+d1*w11);
                d.dx().setAtIndex(JAVA_DOUBLE,q+2,(0d+d0*w20)+d1*w21);
                d.dh().setAtIndex(JAVA_DOUBLE,h,(0d+d0*u00)+d1*u01);
                d.dh().setAtIndex(JAVA_DOUBLE,h+1,(0d+d0*u10)+d1*u11);
                a00+=x0*d0;a01+=x0*d1;a10+=x1*d0;a11+=x1*d1;a20+=x2*d0;a21+=x2*d1;
                c00+=h0*d0;c01+=h0*d1;c10+=h1*d0;c11+=h1*d1;
                
            }
            d.dw().setAtIndex(JAVA_DOUBLE,0,a00);
            d.dw().setAtIndex(JAVA_DOUBLE,1,a01);
            d.dw().setAtIndex(JAVA_DOUBLE,2,a10);
            d.dw().setAtIndex(JAVA_DOUBLE,3,a11);
            d.dw().setAtIndex(JAVA_DOUBLE,4,a20);
            d.dw().setAtIndex(JAVA_DOUBLE,5,a21);
            d.du().setAtIndex(JAVA_DOUBLE,0,c00);
            d.du().setAtIndex(JAVA_DOUBLE,1,c01);
            d.du().setAtIndex(JAVA_DOUBLE,2,c10);
            d.du().setAtIndex(JAVA_DOUBLE,3,c11);
        }

        /**
         * Retains twelve typed scalar carry locals across time, with a separate ADD at each active time.
         * @param maximum prevalidated maximum row length in [0,T], zero for an empty/all-inactive domain
         */
        private void inlineScalarScan(long maximum) {
            double carry0=initial,carry1=initial,carry2=initial,carry3=initial,carry4=initial,carry5=initial,carry6=initial,carry7=initial,carry8=initial,carry9=initial,carry10=initial,carry11=initial;
            for(int step=0;step<d.time();step++) {
                time=d.reverse()?step:d.time()-1-step;if(time>=maximum)continue;

            double a00=0d,a01=0d,a10=0d,a11=0d,a20=0d,a21=0d;
            double c00=0d,c01=0d,c10=0d,c11=0d,b0=0d,b1=0d;
            final double w00=this.w00,w01=this.w01,w10=this.w10,w11=this.w11,w20=this.w20,w21=this.w21;
            final double u00=this.u00,u01=this.u01,u10=this.u10,u11=this.u11;
            for(int r=0;r<d.batch();r++) if(time<d.lengths().getAtIndex(JAVA_LONG,r)) {
int h=2*r,p=2*(time*d.batch()+r),q=3*(time*d.batch()+r);
                double y0=d.saved().getAtIndex(JAVA_DOUBLE,p),y1=d.saved().getAtIndex(JAVA_DOUBLE,p+1);
                double d0=(d.dh().getAtIndex(JAVA_DOUBLE,h)+d.emissionSeed().getAtIndex(JAVA_DOUBLE,p))*(derivative-y0*y0);
                double d1=(d.dh().getAtIndex(JAVA_DOUBLE,h+1)+d.emissionSeed().getAtIndex(JAVA_DOUBLE,p+1))*(derivative-y1*y1);
                double x0=d.x().getAtIndex(JAVA_DOUBLE,q),x1=d.x().getAtIndex(JAVA_DOUBLE,q+1),x2=d.x().getAtIndex(JAVA_DOUBLE,q+2);
                double h0=d.previous().getAtIndex(JAVA_DOUBLE,p),h1=d.previous().getAtIndex(JAVA_DOUBLE,p+1);
                d.dx().setAtIndex(JAVA_DOUBLE,q,(0d+d0*w00)+d1*w01);
                d.dx().setAtIndex(JAVA_DOUBLE,q+1,(0d+d0*w10)+d1*w11);
                d.dx().setAtIndex(JAVA_DOUBLE,q+2,(0d+d0*w20)+d1*w21);
                d.dh().setAtIndex(JAVA_DOUBLE,h,(0d+d0*u00)+d1*u01);
                d.dh().setAtIndex(JAVA_DOUBLE,h+1,(0d+d0*u10)+d1*u11);
                a00+=x0*d0;a01+=x0*d1;a10+=x1*d0;a11+=x1*d1;a20+=x2*d0;a21+=x2*d1;
                c00+=h0*d0;c01+=h0*d1;c10+=h1*d0;c11+=h1*d1;
                if(d.batch()==1){b0=d0;b1=d1;}else{b0+=d0;b1+=d1;}
            }
            carry0+=a00;
            carry1+=a01;
            carry2+=a10;
            carry3+=a11;
            carry4+=a20;
            carry5+=a21;
            carry6+=c00;
            carry7+=c01;
            carry8+=c10;
            carry9+=c11;
            carry10+=b0;
            carry11+=b1;
            }
            d.dw().setAtIndex(JAVA_DOUBLE,0,carry0);
            d.dw().setAtIndex(JAVA_DOUBLE,1,carry1);
            d.dw().setAtIndex(JAVA_DOUBLE,2,carry2);
            d.dw().setAtIndex(JAVA_DOUBLE,3,carry3);
            d.dw().setAtIndex(JAVA_DOUBLE,4,carry4);
            d.dw().setAtIndex(JAVA_DOUBLE,5,carry5);
            d.du().setAtIndex(JAVA_DOUBLE,0,carry6);
            d.du().setAtIndex(JAVA_DOUBLE,1,carry7);
            d.du().setAtIndex(JAVA_DOUBLE,2,carry8);
            d.du().setAtIndex(JAVA_DOUBLE,3,carry9);
            d.db().setAtIndex(JAVA_DOUBLE,0,carry10);
            d.db().setAtIndex(JAVA_DOUBLE,1,carry11);
        }

        /** Inline ordinary reduction publishes locals directly; no partial buffer or merge. */
        private void inlineVectorOrdinary() {
            if(d.batch()==0){d.dw().fill((byte)0);d.du().fill((byte)0);return;}
            var a0=DoubleVector.zero(S);var a1=DoubleVector.zero(S);var a2=DoubleVector.zero(S);
            var c0=DoubleVector.zero(S);var c1=DoubleVector.zero(S);var b=DoubleVector.zero(S);
            var u0=vu0;var u1=vu1;var w0=vw0;var w1=vw1;var wt0=vwt0;var wt1=vwt1;
            // F64 may have only two preferred lanes: F3 then has a second masked feature chunk.
            for(int r=0;r<d.batch();r++)  {
int h=2*r,p=h,q=3*r;
                var y=DoubleVector.fromMemorySegment(S,d.saved(),(long)p*8,ORDER,H);
                var delta=y.mul(seed).add(y.mul(seed)).mul(DoubleVector.broadcast(S,derivative).sub(y.mul(y)));
                delta.intoMemorySegment(d.delta(),(long)h*8,ORDER,H);
                double d0=delta.lane(0),d1=delta.lane(1);
                DoubleVector.zero(S).add(w0.mul(d0)).add(w1.mul(d1)).intoMemorySegment(d.dx(),(long)q*8,ORDER,X);
                if(S.length()<3)DoubleVector.zero(S).add(wt0.mul(d0)).add(wt1.mul(d1))
                        .intoMemorySegment(d.dx(),(long)(q+2)*8,ORDER,TAIL);
                DoubleVector.zero(S).add(u0.mul(d0)).add(u1.mul(d1)).intoMemorySegment(d.dh(),(long)h*8,ORDER,H);
                double x0=d.x().getAtIndex(JAVA_DOUBLE,q),x1=d.x().getAtIndex(JAVA_DOUBLE,q+1),x2=d.x().getAtIndex(JAVA_DOUBLE,q+2);
                var prev=DoubleVector.fromMemorySegment(S,d.previous(),(long)p*8,ORDER,H);
                double h0=prev.lane(0),h1=prev.lane(1);
                a0=a0.add(delta.mul(x0));a1=a1.add(delta.mul(x1));a2=a2.add(delta.mul(x2));
                c0=c0.add(delta.mul(h0));c1=c1.add(delta.mul(h1));
                
            }
            a0.intoMemorySegment(d.dw(),0L,ORDER,H);
            a1.intoMemorySegment(d.dw(),16L,ORDER,H);
            a2.intoMemorySegment(d.dw(),32L,ORDER,H);
            c0.intoMemorySegment(d.du(),0L,ORDER,H);
            c1.intoMemorySegment(d.du(),16L,ORDER,H);
        }

        /**
         * Retains twelve carry cells in six H2 vectors across time, without flattening time contributions.
         * @param maximum prevalidated maximum row length in [0,T], zero for an empty/all-inactive domain
         */
        private void inlineVectorScan(long maximum) {
            var cw0=DoubleVector.broadcast(S,initial);var cw1=cw0;var cw2=cw0;var cu0=cw0;var cu1=cw0;var cb=cw0;
            for(int step=0;step<d.time();step++) {
                time=d.reverse()?step:d.time()-1-step;if(time>=maximum)continue;

            var a0=DoubleVector.zero(S);var a1=DoubleVector.zero(S);var a2=DoubleVector.zero(S);
            var c0=DoubleVector.zero(S);var c1=DoubleVector.zero(S);var b=DoubleVector.zero(S);
            var u0=vu0;var u1=vu1;var w0=vw0;var w1=vw1;var wt0=vwt0;var wt1=vwt1;
            // F64 may have only two preferred lanes: F3 then has a second masked feature chunk.
            for(int r=0;r<d.batch();r++) if(time<d.lengths().getAtIndex(JAVA_LONG,r)) {
int h=2*r,p=2*(time*d.batch()+r),q=3*(time*d.batch()+r);
                var y=DoubleVector.fromMemorySegment(S,d.saved(),(long)p*8,ORDER,H);
                var delta=DoubleVector.fromMemorySegment(S,d.dh(),(long)h*8,ORDER,H)
                        .add(DoubleVector.fromMemorySegment(S,d.emissionSeed(),(long)p*8,ORDER,H)).mul(DoubleVector.broadcast(S,derivative).sub(y.mul(y)));
                
                double d0=delta.lane(0),d1=delta.lane(1);
                DoubleVector.zero(S).add(w0.mul(d0)).add(w1.mul(d1)).intoMemorySegment(d.dx(),(long)q*8,ORDER,X);
                if(S.length()<3)DoubleVector.zero(S).add(wt0.mul(d0)).add(wt1.mul(d1))
                        .intoMemorySegment(d.dx(),(long)(q+2)*8,ORDER,TAIL);
                DoubleVector.zero(S).add(u0.mul(d0)).add(u1.mul(d1)).intoMemorySegment(d.dh(),(long)h*8,ORDER,H);
                double x0=d.x().getAtIndex(JAVA_DOUBLE,q),x1=d.x().getAtIndex(JAVA_DOUBLE,q+1),x2=d.x().getAtIndex(JAVA_DOUBLE,q+2);
                var prev=DoubleVector.fromMemorySegment(S,d.previous(),(long)p*8,ORDER,H);
                double h0=prev.lane(0),h1=prev.lane(1);
                a0=a0.add(delta.mul(x0));a1=a1.add(delta.mul(x1));a2=a2.add(delta.mul(x2));
                c0=c0.add(delta.mul(h0));c1=c1.add(delta.mul(h1));
                b=d.batch()==1?delta:b.add(delta);
            }
            cw0=cw0.add(a0);
            cw1=cw1.add(a1);
            cw2=cw2.add(a2);
            cu0=cu0.add(c0);
            cu1=cu1.add(c1);
            cb=cb.add(b);
            }
            cw0.intoMemorySegment(d.dw(),0L,ORDER,H);
            cw1.intoMemorySegment(d.dw(),16L,ORDER,H);
            cw2.intoMemorySegment(d.dw(),32L,ORDER,H);
            cu0.intoMemorySegment(d.du(),0L,ORDER,H);
            cu1.intoMemorySegment(d.du(),16L,ORDER,H);
            cb.intoMemorySegment(d.db(),0L,ORDER,H);
        }

        /**
         * Merges valid ranges on the caller and keeps twelve typed carry cells local across T.
         * Each active time retains its own ADD; workers never update these shared carries.
         * @param lease live same-budget outer reservation, borrowed exclusively through join and not closed here
         * @param maximum prevalidated maximum row length in [0,T], zero for an empty/all-inactive domain
         */
        private void parallelScalarScan(CpuConcurrencyBudget.Lease lease,long maximum) {
            double carry0=initial,carry1=initial,carry2=initial,carry3=initial,carry4=initial,carry5=initial,carry6=initial,carry7=initial,carry8=initial,carry9=initial,carry10=initial,carry11=initial;
            for(int step=0;step<d.time();step++) {
                time=d.reverse()?step:d.time()-1-step;if(time>=maximum)continue;
                dispatch.accept(lease);
                double a00=0d,a01=0d,a10=0d,a11=0d,a20=0d,a21=0d,c00=0d,c01=0d,c10=0d,c11=0d,b0=0d,b1=0d;boolean seen=false;
                for(int r=0;r<count;r++)if(valid[r]!=0) {
                    double part0=partials.getAtIndex(JAVA_DOUBLE,r*12+0);a00=seen?a00+part0:part0;
                    double part1=partials.getAtIndex(JAVA_DOUBLE,r*12+1);a01=seen?a01+part1:part1;
                    double part2=partials.getAtIndex(JAVA_DOUBLE,r*12+2);a10=seen?a10+part2:part2;
                    double part3=partials.getAtIndex(JAVA_DOUBLE,r*12+3);a11=seen?a11+part3:part3;
                    double part4=partials.getAtIndex(JAVA_DOUBLE,r*12+4);a20=seen?a20+part4:part4;
                    double part5=partials.getAtIndex(JAVA_DOUBLE,r*12+5);a21=seen?a21+part5:part5;
                    double part6=partials.getAtIndex(JAVA_DOUBLE,r*12+6);c00=seen?c00+part6:part6;
                    double part7=partials.getAtIndex(JAVA_DOUBLE,r*12+7);c01=seen?c01+part7:part7;
                    double part8=partials.getAtIndex(JAVA_DOUBLE,r*12+8);c10=seen?c10+part8:part8;
                    double part9=partials.getAtIndex(JAVA_DOUBLE,r*12+9);c11=seen?c11+part9:part9;
                    double part10=partials.getAtIndex(JAVA_DOUBLE,r*12+10);b0=seen?b0+part10:part10;
                    double part11=partials.getAtIndex(JAVA_DOUBLE,r*12+11);b1=seen?b1+part11:part11;
                    seen=true;
                }
                carry0+=a00;carry1+=a01;carry2+=a10;carry3+=a11;carry4+=a20;carry5+=a21;carry6+=c00;carry7+=c01;carry8+=c10;carry9+=c11;carry10+=b0;carry11+=b1;
            }
            d.dw().setAtIndex(JAVA_DOUBLE,0,carry0);
            d.dw().setAtIndex(JAVA_DOUBLE,1,carry1);
            d.dw().setAtIndex(JAVA_DOUBLE,2,carry2);
            d.dw().setAtIndex(JAVA_DOUBLE,3,carry3);
            d.dw().setAtIndex(JAVA_DOUBLE,4,carry4);
            d.dw().setAtIndex(JAVA_DOUBLE,5,carry5);
            d.du().setAtIndex(JAVA_DOUBLE,0,carry6);
            d.du().setAtIndex(JAVA_DOUBLE,1,carry7);
            d.du().setAtIndex(JAVA_DOUBLE,2,carry8);
            d.du().setAtIndex(JAVA_DOUBLE,3,carry9);
            d.db().setAtIndex(JAVA_DOUBLE,0,carry10);
            d.db().setAtIndex(JAVA_DOUBLE,1,carry11);
        }

        /**
         * Merges valid ranges on the caller and keeps twelve typed carry cells local across T.
         * Each active time retains its own ADD; workers never update these shared carries.
         * @param lease live same-budget outer reservation, borrowed exclusively through join and not closed here
         * @param maximum prevalidated maximum row length in [0,T], zero for an empty/all-inactive domain
         */
        private void parallelVectorScan(CpuConcurrencyBudget.Lease lease,long maximum) {
            var carry0=DoubleVector.broadcast(S,initial);var carry1=DoubleVector.broadcast(S,initial);var carry2=DoubleVector.broadcast(S,initial);var carry3=DoubleVector.broadcast(S,initial);var carry4=DoubleVector.broadcast(S,initial);var carry5=DoubleVector.broadcast(S,initial);
            for(int step=0;step<d.time();step++) {
                time=d.reverse()?step:d.time()-1-step;if(time>=maximum)continue;
                dispatch.accept(lease);
                var a0=DoubleVector.zero(S);var a1=DoubleVector.zero(S);var a2=DoubleVector.zero(S);var c0=DoubleVector.zero(S);var c1=DoubleVector.zero(S);var b=DoubleVector.zero(S);boolean seen=false;
                for(int r=0;r<count;r++)if(valid[r]!=0) {
                    var part0=DoubleVector.fromMemorySegment(S,partials,(long)(r*12+0)*8,ORDER,H);a0=seen?a0.add(part0):part0;
                    var part1=DoubleVector.fromMemorySegment(S,partials,(long)(r*12+2)*8,ORDER,H);a1=seen?a1.add(part1):part1;
                    var part2=DoubleVector.fromMemorySegment(S,partials,(long)(r*12+4)*8,ORDER,H);a2=seen?a2.add(part2):part2;
                    var part3=DoubleVector.fromMemorySegment(S,partials,(long)(r*12+6)*8,ORDER,H);c0=seen?c0.add(part3):part3;
                    var part4=DoubleVector.fromMemorySegment(S,partials,(long)(r*12+8)*8,ORDER,H);c1=seen?c1.add(part4):part4;
                    var part5=DoubleVector.fromMemorySegment(S,partials,(long)(r*12+10)*8,ORDER,H);b=seen?b.add(part5):part5;
                    seen=true;
                }
                carry0=carry0.add(a0);carry1=carry1.add(a1);carry2=carry2.add(a2);carry3=carry3.add(c0);carry4=carry4.add(c1);carry5=carry5.add(b);
            }
            carry0.intoMemorySegment(d.dw(),0L,ORDER,H);
            carry1.intoMemorySegment(d.dw(),16L,ORDER,H);
            carry2.intoMemorySegment(d.dw(),32L,ORDER,H);
            carry3.intoMemorySegment(d.du(),0L,ORDER,H);
            carry4.intoMemorySegment(d.du(),16L,ORDER,H);
            carry5.intoMemorySegment(d.db(),0L,ORDER,H);
        }

    }
}

