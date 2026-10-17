package io.github.pho001.synaptik.backend.cpu.internal.reference;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPool3dIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPool3dLowering;
import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import org.junit.jupiter.api.Test;

class CpuPool3dReferenceTest {
    @Test void maximumUsesDepthHeightWidthFirstWinnerNanAndPositiveZero() {
        var g=geometry(CpuPool3dIr.Kind.MAX,DataType.FLOAT32,2,2,2,0);
        float[] out=new float[1];
        CpuPool3dReferenceKernel.evaluate(g,new float[]{-0f,+0f,1,1,Float.NaN,2,Float.NaN,2},out,0,1);
        assertTrue(Float.isNaN(out[0]));
        CpuPool3dReferenceKernel.evaluate(g,new float[]{-0f,+0f,-1,-1,-2,-2,-3,-3},out,0,1);
        assertEquals(0,Float.floatToRawIntBits(out[0]));
    }

    @Test void averageUsesFixedDivisorAndSignedZeroPolicy() {
        var g=geometry(CpuPool3dIr.Kind.AVERAGE,DataType.BFLOAT16,1,1,2,0);
        short negative=BFloat16Bits.fromFloat(-0f);short[] out=new short[1];
        CpuPool3dReferenceKernel.evaluate(g,new short[]{negative,negative},out,0,1);
        assertEquals(negative,out[0]);
        var padded=geometry(CpuPool3dIr.Kind.AVERAGE,DataType.BFLOAT16,1,1,2,1);
        CpuPool3dReferenceKernel.evaluate(padded,new short[]{negative,negative},out,0,1);
        assertEquals(BFloat16Bits.fromFloat(+0f),out[0]);
    }

    @Test void maximumRetainsFirstNanBitsAndDistinguishesRealInfinityFromPadding() {
        var g=geometry(CpuPool3dIr.Kind.MAX,DataType.FLOAT32,2,1,2,0);
        float first=Float.intBitsToFloat(0x7fc00011),second=Float.intBitsToFloat(0x7fc00022);
        float[] out=new float[1];
        CpuPool3dReferenceKernel.evaluate(g,new float[]{first,second,Float.POSITIVE_INFINITY,
                Float.NEGATIVE_INFINITY},out,0,1);
        assertEquals(0x7fc00011,Float.floatToRawIntBits(out[0]));
        var allPadding=geometry(CpuPool3dIr.Kind.MAX,DataType.FLOAT64,1,1,1,2);
        double[] paddedOut={17};
        CpuPool3dReferenceKernel.evaluate(allPadding,new double[]{Double.NEGATIVE_INFINITY},
                paddedOut,0,1);
        assertEquals(Double.doubleToRawLongBits(Double.NEGATIVE_INFINITY),
                Double.doubleToRawLongBits(paddedOut[0]));
    }

    @Test void averageUsesTypeSpecificOrderExceptionalValuesAndOneBfloat16Narrowing() {
        var f32=geometry(CpuPool3dIr.Kind.AVERAGE,DataType.FLOAT32,1,1,4,0);
        float[] out=new float[1];
        CpuPool3dReferenceKernel.evaluate(f32,
                new float[]{Float.POSITIVE_INFINITY,Float.NEGATIVE_INFINITY,1,-1},out,0,1);
        assertTrue(Float.isNaN(out[0]));
        CpuPool3dReferenceKernel.evaluate(f32,new float[]{Float.NaN,1,2,3},out,0,1);
        assertTrue(Float.isNaN(out[0]));
        CpuPool3dReferenceKernel.evaluate(f32,new float[]{1.0e20f,1,-1.0e20f,1},out,0,1);
        assertEquals(Float.floatToRawIntBits(.25f),Float.floatToRawIntBits(out[0]));

        var bf16=geometry(CpuPool3dIr.Kind.AVERAGE,DataType.BFLOAT16,1,1,3,0);
        short[] input={BFloat16Bits.fromFloat(1f),BFloat16Bits.fromFloat(1f),
                BFloat16Bits.fromFloat(2f)}, result=new short[1];
        CpuPool3dReferenceKernel.evaluate(bf16,input,result,0,1);
        assertEquals(BFloat16Bits.fromFloat(4f/3f),result[0]);
    }

    @Test void emptyOutputRangeTouchesNoCarrier() {
        var input=new CpuPool3dLowering.Layout(new long[]{0,1,2,2,2},0,
                new long[]{8,8,4,2,1});
        var output=new CpuPool3dLowering.Layout(new long[]{0,1,2,2,2},0,
                new long[]{8,8,4,2,1});
        var g=new CpuPool3dLowering.Geometry(CpuPool3dIr.Kind.AVERAGE,DataType.FLOAT64,
                input,output,1,1,1,1,1,1,0,0,0,1,1,1,1,0);
        CpuPool3dReferenceKernel.evaluate(g,new double[0],new double[0],0,0);
    }

    private static CpuPool3dLowering.Geometry geometry(CpuPool3dIr.Kind kind,DataType type,
            long d,long h,long w,long padding){
        long[] e={1,1,d,h,w};long[] s={d*h*w,d*h*w,h*w,w,1};
        long[] oe={1,1,1,1,1};long divisor=d*h*w;
        return new CpuPool3dLowering.Geometry(kind,type,new CpuPool3dLowering.Layout(e,0,s),
                new CpuPool3dLowering.Layout(oe,0,new long[]{1,1,1,1,1}),d,h,w,1,1,1,
                padding,padding,padding,1,1,1,divisor,1);
    }
}
