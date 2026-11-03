package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import java.lang.foreign.MemorySegment;
import static java.lang.foreign.ValueLayout.*;

/**
 * Independently derived specialized Java equations, not a generated-program transcription.
 * Each method has one increasing-row traversal, typed producer locals, ordered products and
 * terminal accumulators. Outputs are caller allocated. There is no row-intermediate bank,
 * runtime dtype dispatch, allocation or helper arithmetic in a hot loop. The dense derivative
 * retains the two distinct multiplication sites for the two uses of Y in Y*Y. The Scan
 * derivative consumes saved primals; it never replays an activation. Shared carry additions
 * occur only after a complete per-time row reduction, and only at active times.
 */
final class CpuRowMapReductionJavaOracle {
    private CpuRowMapReductionJavaOracle() { }

    /**
     * Computes dense F2/J1/H1 row and parameter gradients of sum(Y*Y).
     * Exact db is deliberately not computed here: ordinary SUM remains a separate operation.
     * @param x borrowed [R,2] input
     * @param z borrowed [R,1] second input
     * @param w borrowed [2,1] weights
     * @param u borrowed [1,1] second weights
     * @param y borrowed exact saved [R,1] forward result
     * @param seed represented scalar loss cotangent
     * @param d writable [R,1] retained local cotangent for separate exact SUM
     * @param dx writable [R,2] first input cotangent
     * @param dz writable [R,1] second input cotangent
     * @param dw writable [2,1] completed parameter cotangent
     * @param du writable [1,1] completed second parameter cotangent
     */
    static void denseFloat32(float[] x,float[] z,float[] w,float[] u,float[] y,float seed,
            float[] d,float[] dx,float[] dz,float[] dw,float[] du) {
        float a = 0f, b = 0f, c = 0f;
        for (int r = 0; r < y.length; r++) {
            float left = seed*y[r], right = seed*y[r], delta = left+right;
            d[r] = delta;
            dx[2*r] = delta*w[0]; dx[2*r+1] = delta*w[1]; dz[r] = delta*u[0];
            if (y.length == 1) { a=x[2*r]*delta; b=x[2*r+1]*delta; c=z[r]*delta; }
            else { a += x[2*r]*delta; b += x[2*r+1]*delta; c += z[r]*delta; }
        }
        dw[0] = a; dw[1] = b; du[0] = c;
    }

    /**
     * Double-site counterpart, with no float working sites.
     * @param x borrowed [R,2] input
     * @param z borrowed [R,1] second input
     * @param w borrowed [2,1] weights
     * @param u borrowed [1,1] second weights
     * @param y borrowed exact saved [R,1] forward result
     * @param seed represented scalar loss cotangent
     * @param d writable [R,1] retained local cotangent for separate exact SUM
     * @param dx writable [R,2] first input cotangent
     * @param dz writable [R,1] second input cotangent
     * @param dw writable [2,1] completed parameter cotangent
     * @param du writable [1,1] completed second parameter cotangent
     */
    static void denseFloat64(double[] x,double[] z,double[] w,double[] u,double[] y,double seed,
            double[] d,double[] dx,double[] dz,double[] dw,double[] du) {
        double a = 0d, b = 0d, c = 0d;
        for (int r = 0; r < y.length; r++) {
            double left = seed*y[r], right = seed*y[r], delta = left+right;
            d[r] = delta;
            dx[2*r] = delta*w[0]; dx[2*r+1] = delta*w[1]; dz[r] = delta*u[0];
            if (y.length == 1) { a=x[2*r]*delta; b=x[2*r+1]*delta; c=z[r]*delta; }
            else { a += x[2*r]*delta; b += x[2*r+1]*delta; c += z[r]*delta; }
        }
        dw[0] = a; dw[1] = b; du[0] = c;
    }

    /**
     * F3/H2 opposite-traversal adjoint with one fused active-row traversal per time.
     * Fixed features are explicit scalar sites; T and B are loop bounds, never unrolled.
     * Bias SUM-to-shape moves a singleton geometry raw; one active row in larger B still
     * contributes to typed positive zero. The later shared-carry ADD remains a distinct site.
     * @param x borrowed [T,B,3] primal, potentially poisoned outside the active prefix
     * @param w borrowed [3,2] weights
     * @param u borrowed [2,2] recurrent weights
     * @param lengths borrowed prevalidated INT64[B]
     * @param fs borrowed [B,2] final seed, copied raw for inactive rows
     * @param es borrowed [T,B,2] emission seed
     * @param previous exact saved active previous carry [T,B,2]
     * @param saved exact saved activation [T,B,2]
     * @param dh writable [B,2] state cotangent
     * @param dx writable [T,B,3] sequence cotangent
     * @param dw writable [3,2] parameter carry
     * @param du writable [2,2] recurrent parameter carry
     * @param db writable [2] bias carry
     * @param time fixed nonnegative T
     * @param batch nonnegative invocation B
     * @param reverse original traversal direction
     */
    static void scanFloat32(MemorySegment x,MemorySegment w,MemorySegment u,MemorySegment lengths,
            MemorySegment fs,MemorySegment es,MemorySegment previous,MemorySegment saved,
            MemorySegment dh,MemorySegment dx,MemorySegment dw,MemorySegment du,MemorySegment db,
            int time,int batch,boolean reverse) {
        dh.copyFrom(fs); dx.fill((byte)0); dw.fill((byte)0); du.fill((byte)0); db.fill((byte)0);
        long maximum = 0;
        for (int r = 0; r < batch; r++) maximum = Math.max(maximum,lengths.getAtIndex(JAVA_LONG,r));
        for (int step = 0; step < time; step++) {
            int t = reverse ? step : time-1-step;
            if (t >= maximum) continue;
            float w00=0f,w01=0f,w10=0f,w11=0f,w20=0f,w21=0f,u00=0f,u01=0f,u10=0f,u11=0f,b0=0f,b1=0f;
            for (int r = 0; r < batch; r++) if (t < lengths.getAtIndex(JAVA_LONG,r)) {
                int h = 2*r, p = 2*(t*batch+r), q = 3*(t*batch+r);
                float y0=saved.getAtIndex(JAVA_FLOAT,p), y1=saved.getAtIndex(JAVA_FLOAT,p+1);
                float d0=(dh.getAtIndex(JAVA_FLOAT,h)+es.getAtIndex(JAVA_FLOAT,p))*(1f-y0*y0);
                float d1=(dh.getAtIndex(JAVA_FLOAT,h+1)+es.getAtIndex(JAVA_FLOAT,p+1))*(1f-y1*y1);
                float x0=x.getAtIndex(JAVA_FLOAT,q),x1=x.getAtIndex(JAVA_FLOAT,q+1),x2=x.getAtIndex(JAVA_FLOAT,q+2);
                float h0=previous.getAtIndex(JAVA_FLOAT,p),h1=previous.getAtIndex(JAVA_FLOAT,p+1);
                dx.setAtIndex(JAVA_FLOAT,q,(0f+d0*w.getAtIndex(JAVA_FLOAT,0))+d1*w.getAtIndex(JAVA_FLOAT,1));
                dx.setAtIndex(JAVA_FLOAT,q+1,(0f+d0*w.getAtIndex(JAVA_FLOAT,2))+d1*w.getAtIndex(JAVA_FLOAT,3));
                dx.setAtIndex(JAVA_FLOAT,q+2,(0f+d0*w.getAtIndex(JAVA_FLOAT,4))+d1*w.getAtIndex(JAVA_FLOAT,5));
                dh.setAtIndex(JAVA_FLOAT,h,(0f+d0*u.getAtIndex(JAVA_FLOAT,0))+d1*u.getAtIndex(JAVA_FLOAT,1));
                dh.setAtIndex(JAVA_FLOAT,h+1,(0f+d0*u.getAtIndex(JAVA_FLOAT,2))+d1*u.getAtIndex(JAVA_FLOAT,3));
                if (batch == 1) {
                    w00=x0*d0;w01=x0*d1;w10=x1*d0;w11=x1*d1;w20=x2*d0;w21=x2*d1;
                    u00=h0*d0;u01=h0*d1;u10=h1*d0;u11=h1*d1;
                } else {
                    w00+=x0*d0;w01+=x0*d1;w10+=x1*d0;w11+=x1*d1;w20+=x2*d0;w21+=x2*d1;
                    u00+=h0*d0;u01+=h0*d1;u10+=h1*d0;u11+=h1*d1;
                }
                if (batch == 1) { b0=d0;b1=d1; } else { b0+=d0;b1+=d1; }
            }
            dw.setAtIndex(JAVA_FLOAT,0,dw.getAtIndex(JAVA_FLOAT,0)+w00);dw.setAtIndex(JAVA_FLOAT,1,dw.getAtIndex(JAVA_FLOAT,1)+w01);
            dw.setAtIndex(JAVA_FLOAT,2,dw.getAtIndex(JAVA_FLOAT,2)+w10);dw.setAtIndex(JAVA_FLOAT,3,dw.getAtIndex(JAVA_FLOAT,3)+w11);
            dw.setAtIndex(JAVA_FLOAT,4,dw.getAtIndex(JAVA_FLOAT,4)+w20);dw.setAtIndex(JAVA_FLOAT,5,dw.getAtIndex(JAVA_FLOAT,5)+w21);
            du.setAtIndex(JAVA_FLOAT,0,du.getAtIndex(JAVA_FLOAT,0)+u00);du.setAtIndex(JAVA_FLOAT,1,du.getAtIndex(JAVA_FLOAT,1)+u01);
            du.setAtIndex(JAVA_FLOAT,2,du.getAtIndex(JAVA_FLOAT,2)+u10);du.setAtIndex(JAVA_FLOAT,3,du.getAtIndex(JAVA_FLOAT,3)+u11);
            db.setAtIndex(JAVA_FLOAT,0,db.getAtIndex(JAVA_FLOAT,0)+b0);db.setAtIndex(JAVA_FLOAT,1,db.getAtIndex(JAVA_FLOAT,1)+b1);
        }
    }

    /**
     * F64-site counterpart of the same saved-primal equations and fused traversal.
     * Singleton bias movement depends on full B geometry, not the active contributor count.
     * @param x borrowed [T,B,3] primal
     * @param w borrowed [3,2] weights
     * @param u borrowed [2,2] recurrent weights
     * @param lengths borrowed prevalidated INT64[B]
     * @param fs borrowed [B,2] final seed
     * @param es borrowed [T,B,2] emission seed
     * @param previous exact saved active previous carry
     * @param saved exact saved activation
     * @param dh writable [B,2] state cotangent
     * @param dx writable [T,B,3] sequence cotangent
     * @param dw writable [3,2] parameter carry
     * @param du writable [2,2] recurrent parameter carry
     * @param db writable [2] bias carry
     * @param time fixed nonnegative T
     * @param batch nonnegative invocation B
     * @param reverse original traversal direction
     */
    static void scanFloat64(MemorySegment x,MemorySegment w,MemorySegment u,MemorySegment lengths,
            MemorySegment fs,MemorySegment es,MemorySegment previous,MemorySegment saved,
            MemorySegment dh,MemorySegment dx,MemorySegment dw,MemorySegment du,MemorySegment db,
            int time,int batch,boolean reverse) {
        dh.copyFrom(fs); dx.fill((byte)0); dw.fill((byte)0); du.fill((byte)0); db.fill((byte)0);
        long maximum = 0;
        for (int r = 0; r < batch; r++) maximum = Math.max(maximum,lengths.getAtIndex(JAVA_LONG,r));
        for (int step = 0; step < time; step++) {
            int t = reverse ? step : time-1-step;
            if (t >= maximum) continue;
            double w00=0d,w01=0d,w10=0d,w11=0d,w20=0d,w21=0d,u00=0d,u01=0d,u10=0d,u11=0d,b0=0d,b1=0d;
            for (int r = 0; r < batch; r++) if (t < lengths.getAtIndex(JAVA_LONG,r)) {
                int h = 2*r, p = 2*(t*batch+r), q = 3*(t*batch+r);
                double y0=saved.getAtIndex(JAVA_DOUBLE,p), y1=saved.getAtIndex(JAVA_DOUBLE,p+1);
                double d0=(dh.getAtIndex(JAVA_DOUBLE,h)+es.getAtIndex(JAVA_DOUBLE,p))*(1d-y0*y0);
                double d1=(dh.getAtIndex(JAVA_DOUBLE,h+1)+es.getAtIndex(JAVA_DOUBLE,p+1))*(1d-y1*y1);
                double x0=x.getAtIndex(JAVA_DOUBLE,q),x1=x.getAtIndex(JAVA_DOUBLE,q+1),x2=x.getAtIndex(JAVA_DOUBLE,q+2);
                double h0=previous.getAtIndex(JAVA_DOUBLE,p),h1=previous.getAtIndex(JAVA_DOUBLE,p+1);
                dx.setAtIndex(JAVA_DOUBLE,q,(0d+d0*w.getAtIndex(JAVA_DOUBLE,0))+d1*w.getAtIndex(JAVA_DOUBLE,1));
                dx.setAtIndex(JAVA_DOUBLE,q+1,(0d+d0*w.getAtIndex(JAVA_DOUBLE,2))+d1*w.getAtIndex(JAVA_DOUBLE,3));
                dx.setAtIndex(JAVA_DOUBLE,q+2,(0d+d0*w.getAtIndex(JAVA_DOUBLE,4))+d1*w.getAtIndex(JAVA_DOUBLE,5));
                dh.setAtIndex(JAVA_DOUBLE,h,(0d+d0*u.getAtIndex(JAVA_DOUBLE,0))+d1*u.getAtIndex(JAVA_DOUBLE,1));
                dh.setAtIndex(JAVA_DOUBLE,h+1,(0d+d0*u.getAtIndex(JAVA_DOUBLE,2))+d1*u.getAtIndex(JAVA_DOUBLE,3));
                if (batch == 1) {
                    w00=x0*d0;w01=x0*d1;w10=x1*d0;w11=x1*d1;w20=x2*d0;w21=x2*d1;
                    u00=h0*d0;u01=h0*d1;u10=h1*d0;u11=h1*d1;
                } else {
                    w00+=x0*d0;w01+=x0*d1;w10+=x1*d0;w11+=x1*d1;w20+=x2*d0;w21+=x2*d1;
                    u00+=h0*d0;u01+=h0*d1;u10+=h1*d0;u11+=h1*d1;
                }
                if (batch == 1) { b0=d0;b1=d1; } else { b0+=d0;b1+=d1; }
            }
            dw.setAtIndex(JAVA_DOUBLE,0,dw.getAtIndex(JAVA_DOUBLE,0)+w00);dw.setAtIndex(JAVA_DOUBLE,1,dw.getAtIndex(JAVA_DOUBLE,1)+w01);
            dw.setAtIndex(JAVA_DOUBLE,2,dw.getAtIndex(JAVA_DOUBLE,2)+w10);dw.setAtIndex(JAVA_DOUBLE,3,dw.getAtIndex(JAVA_DOUBLE,3)+w11);
            dw.setAtIndex(JAVA_DOUBLE,4,dw.getAtIndex(JAVA_DOUBLE,4)+w20);dw.setAtIndex(JAVA_DOUBLE,5,dw.getAtIndex(JAVA_DOUBLE,5)+w21);
            du.setAtIndex(JAVA_DOUBLE,0,du.getAtIndex(JAVA_DOUBLE,0)+u00);du.setAtIndex(JAVA_DOUBLE,1,du.getAtIndex(JAVA_DOUBLE,1)+u01);
            du.setAtIndex(JAVA_DOUBLE,2,du.getAtIndex(JAVA_DOUBLE,2)+u10);du.setAtIndex(JAVA_DOUBLE,3,du.getAtIndex(JAVA_DOUBLE,3)+u11);
            db.setAtIndex(JAVA_DOUBLE,0,db.getAtIndex(JAVA_DOUBLE,0)+b0);db.setAtIndex(JAVA_DOUBLE,1,db.getAtIndex(JAVA_DOUBLE,1)+b1);
        }
    }
}
