package io.github.pho001.synaptik.backend.cpu.internal.reference;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import java.util.List;
import java.util.Objects;

/**
 * Independent clean-Java semantic oracle for first-class batch-normalization training.
 *
 * <p>The oracle derives channel coordinates independently of production lowering and returns
 * dense logical arrays. It uses the selected three-pass formulas, biased saved variance,
 * unbiased running-variance transition, and result-format arithmetic boundaries; it is test
 * evidence and is never a Runtime fallback.</p>
 */
public final class CpuBatchNormTrainingReferenceKernel {
    private CpuBatchNormTrainingReferenceKernel() { }
    /**
     * Dense logical results in semantic output order.
     *
     * <p>The canonical constructor retains the supplied arrays and the accessors expose those
     * arrays directly. {@link #evaluate} creates a fresh array for every component before
     * constructing this carrier.</p>
     *
     * @param output normalized affine values in input logical order; retained directly
     * @param nextRunningMean explicit next running mean for each channel; retained directly
     * @param nextRunningVariance explicit next running variance for each channel; retained directly
     * @param savedBatchMean biased-forward batch mean saved for the occurrence; retained directly
     * @param savedInverseStandardDeviation reciprocal square root of biased variance plus epsilon;
     *     retained directly
     */
    public record Result(double[] output,double[] nextRunningMean,double[] nextRunningVariance,
            double[] savedBatchMean,double[] savedInverseStandardDeviation) { }

    /**
     * Evaluates the selected three-pass formulas into newly allocated dense outputs.
     *
     * @param inputTypes five semantic input types in occurrence order
     * @param resultType exact promoted result and computation type
     * @param momentum finite new-batch weight in {@code [0, 1]}
     * @param epsilon finite strictly positive value added only to biased variance before square
     *     root
     * @param inputs logical carrier values for input, scale, bias, running mean, and running
     *     variance; borrowed and not mutated
     * @param extents static extents for each input carrier view; borrowed and not mutated
     * @param offsets carrier-relative element offsets for the five input views
     * @param strides element strides for the five input views; borrowed and not mutated
     * @param axis normalized channel axis of the input view
     * @return newly allocated normalized output, next statistics, and saved statistics; never
     *     {@code null}
     * @throws NullPointerException if a required argument is {@code null}
     * @throws IllegalArgumentException if cardinality, type promotion, scalars, rank, axis, or
     *     non-empty channel-domain facts disagree
     * @throws ArithmeticException if a count or addressed index cannot be represented
     */
    public static Result evaluate(List<DataType> inputTypes,DataType resultType,double momentum,
            double epsilon,double[][] inputs,long[][] extents,long[] offsets,long[][] strides,int axis){
        inputTypes=List.copyOf(inputTypes);Objects.requireNonNull(resultType,"resultType");
        if(inputTypes.size()!=5||inputs.length!=5||extents.length!=5||offsets.length!=5||strides.length!=5)
            throw new IllegalArgumentException("training reference cardinality disagrees");
        DataType promoted=inputTypes.getFirst();for(int i=1;i<5;i++)promoted=DataTypePromotion.promoteFloating(promoted,inputTypes.get(i));
        if(promoted!=resultType||axis<0||axis>=extents[0].length||extents[0].length<2||!Double.isFinite(momentum)||momentum<0||momentum>1||!Double.isFinite(epsilon)||epsilon<=0)
            throw new IllegalArgumentException("training reference facts disagree");
        long channels=extents[0][axis],reduction=1;for(int i=0;i<extents[0].length;i++)if(i!=axis)reduction=Math.multiplyExact(reduction,extents[0][i]);
        if(channels>0&&reduction<2)throw new IllegalArgumentException("training domain is too small");
        int count=Math.toIntExact(Math.multiplyExact(channels,reduction));int c=Math.toIntExact(channels);
        double[] output=new double[count],nextMean=new double[c],nextVar=new double[c],meanOut=new double[c],invOut=new double[c];
        long[] coordinates=new long[extents[0].length];
        for(int channel=0;channel<c;channel++){
            double sum=0;boolean nonfinite=false;for(long ordinal=0;ordinal<reduction;ordinal++){decode(ordinal,channel,axis,extents[0],coordinates);double factor=represented(inputTypes.get(0),read(inputs[0],offsets[0],strides[0],coordinates));nonfinite|=!Double.isFinite(factor);sum+=factor;}
            if(nonfinite)sum=Double.NaN;
            double mean=arithmetic(resultType,sum,reduction,'/');double ds=0,dc=0,ss=0,sc=0;
            for(long ordinal=0;ordinal<reduction;ordinal++){decode(ordinal,channel,axis,extents[0],coordinates);double x=represented(inputTypes.get(0),read(inputs[0],offsets[0],strides[0],coordinates));double d=arithmetic(resultType,x,mean,'-');
                double y=d-dc,t=ds+y;dc=(t-ds)-y;ds=t;y=d*d-sc;t=ss+y;sc=(t-ss)-y;ss=t;}
            double numerator=ss-ds*ds/reduction;if(Double.isFinite(numerator)&&numerator<0)numerator=0;
            double biased=arithmetic(resultType,numerator,reduction,'/'),unbiased=arithmetic(resultType,numerator,reduction-1,'/');
            double inv=arithmetic(resultType,1,representedComputation(resultType,StrictMath.sqrt(arithmetic(resultType,biased,epsilon,'+'))),'/');
            double scale=promote(inputTypes.get(1),resultType,readVector(inputs[1],offsets[1],strides[1],channel));
            double bias=promote(inputTypes.get(2),resultType,readVector(inputs[2],offsets[2],strides[2],channel));
            double oldMean=promote(inputTypes.get(3),resultType,readVector(inputs[3],offsets[3],strides[3],channel));
            double oldVar=promote(inputTypes.get(4),resultType,readVector(inputs[4],offsets[4],strides[4],channel));
            double oneMinus=arithmetic(resultType,1,momentum,'-');nextMean[channel]=represented(resultType,arithmetic(resultType,arithmetic(resultType,oneMinus,oldMean,'*'),arithmetic(resultType,momentum,mean,'*'),'+'));
            nextVar[channel]=represented(resultType,arithmetic(resultType,arithmetic(resultType,oneMinus,oldVar,'*'),arithmetic(resultType,momentum,unbiased,'*'),'+'));meanOut[channel]=represented(resultType,mean);invOut[channel]=represented(resultType,inv);
            for(long ordinal=0;ordinal<reduction;ordinal++){decode(ordinal,channel,axis,extents[0],coordinates);double x=promote(inputTypes.get(0),resultType,read(inputs[0],offsets[0],strides[0],coordinates));double v=arithmetic(resultType,x,mean,'-');v=arithmetic(resultType,v,inv,'*');v=arithmetic(resultType,v,scale,'*');v=arithmetic(resultType,v,bias,'+');output[logical(coordinates,extents[0])]=represented(resultType,v);}
        }return new Result(output,nextMean,nextVar,meanOut,invOut);
    }
    private static double arithmetic(DataType t,double a,double b,char op){if(t==DataType.FLOAT64)return op=='+'?a+b:op=='-'?a-b:op=='*'?a*b:a/b;float x=(float)a,y=(float)b;return op=='+'?x+y:op=='-'?x-y:op=='*'?x*y:x/y;}
    private static double promote(DataType s,DataType r,double v){return representedComputation(r,represented(s,v));}
    private static double representedComputation(DataType t,double v){return t==DataType.FLOAT64?v:(float)v;}
    private static double represented(DataType t,double v){if(t==DataType.FLOAT64)return v;float f=(float)v;if(t==DataType.FLOAT32)return f;int b=Float.floatToRawIntBits(f);if((b&0x7fffffff)>0x7f800000)return Float.intBitsToFloat(0x7fc00000);b+=0x7fff+((b>>>16)&1);return Float.intBitsToFloat(b&0xffff0000);}
    private static double read(double[]v,long o,long[]s,long[]c){long a=o;for(int i=0;i<c.length;i++)a=Math.addExact(a,Math.multiplyExact(c[i],s[i]));return v[Math.toIntExact(a)];}
    private static double readVector(double[]v,long o,long[]s,long c){return v[Math.toIntExact(Math.addExact(o,Math.multiplyExact(c,s[0])))];}
    private static void decode(long ordinal,long channel,int axis,long[]e,long[]c){for(int i=e.length-1;i>=0;i--){if(i==axis){c[i]=channel;continue;}c[i]=ordinal%e[i];ordinal/=e[i];}}
    private static int logical(long[]c,long[]e){long x=0;for(int i=0;i<c.length;i++)x=Math.addExact(Math.multiplyExact(x,e[i]),c[i]);return Math.toIntExact(x);}
}
