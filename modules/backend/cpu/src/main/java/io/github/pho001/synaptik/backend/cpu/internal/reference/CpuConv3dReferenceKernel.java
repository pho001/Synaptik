package io.github.pho001.synaptik.backend.cpu.internal.reference;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/** Independent optimal clean-Java oracle for direct grouped NCDHW Conv3d. */
public final class CpuConv3dReferenceKernel {
    private CpuConv3dReferenceKernel() { }

    /**
     * Evaluates grouped NCDHW cross-correlation into a new dense logical output.
     *
     * @param inputTypes ordered input, weight, and optional bias types
     * @param resultType exact promoted result type
     * @param inputs semantic carriers represented as binary64 values
     * @param extents input Shapes
     * @param offsets non-negative element offsets
     * @param strides non-negative element strides
     * @param outputExtents exact NCDHW output Shape
     * @param attrs direct convolution geometry
     * @return newly allocated represented output in logical NCDHW order
     * @throws NullPointerException if a required value is {@code null}
     * @throws IllegalArgumentException if type, layout, Shape, or geometry facts disagree
     * @throws ArithmeticException if checked count or address arithmetic overflows
     */
    public static double[] evaluate(List<DataType> inputTypes, DataType resultType,
            double[][] inputs, long[][] extents, long[] offsets, long[][] strides,
            long[] outputExtents, Conv3dAttrs attrs) {
        inputTypes = List.copyOf(inputTypes);
        Objects.requireNonNull(resultType, "resultType");
        Objects.requireNonNull(inputs, "inputs"); Objects.requireNonNull(extents, "extents");
        Objects.requireNonNull(offsets, "offsets"); Objects.requireNonNull(strides, "strides");
        Objects.requireNonNull(outputExtents, "outputExtents"); Objects.requireNonNull(attrs, "attrs");
        boolean bias = inputTypes.size() == 3;
        DataType promoted = inputTypes.isEmpty() ? null : inputTypes.getFirst();
        for (int i = 1; i < inputTypes.size(); i++) promoted = DataTypePromotion.promoteFloating(promoted, inputTypes.get(i));
        if ((!bias && inputTypes.size() != 2) || inputs.length != inputTypes.size()
                || extents.length != inputs.length || offsets.length != inputs.length
                || strides.length != inputs.length || promoted != resultType
                || outputExtents.length != 5 || extents[0].length != 5
                || extents[1].length != 5 || bias && extents[2].length != 1) {
            throw new IllegalArgumentException("Conv3d reference facts disagree");
        }
        for (int i = 0; i < inputs.length; i++) validate(inputs[i], extents[i], offsets[i], strides[i]);
        long nCount=outputExtents[0], outChannels=outputExtents[1], outDepth=outputExtents[2];
        long outHeight=outputExtents[3], outWidth=outputExtents[4];
        long inChannels=extents[0][1], inDepth=extents[0][2], inHeight=extents[0][3], inWidth=extents[0][4];
        long channelsPerGroup=extents[1][1], kernelDepth=extents[1][2], kernelHeight=extents[1][3], kernelWidth=extents[1][4];
        if (extents[1][0] != outChannels || Math.multiplyExact(channelsPerGroup, attrs.groups()) != inChannels
                || inChannels % attrs.groups() != 0 || outChannels % attrs.groups() != 0
                || bias && extents[2][0] != outChannels) throw new IllegalArgumentException("Conv3d reference geometry disagrees");
        long count=count(outputExtents); if (count > Integer.MAX_VALUE) throw new IllegalArgumentException("reference output is too large");
        double[] output=new double[(int)count]; long ordinal=0, outputsPerGroup=outChannels/attrs.groups();
        for(long n=0;n<nCount;n++) for(long oc=0;oc<outChannels;oc++) {
            long inputChannelBase=(oc/outputsPerGroup)*channelsPerGroup;
            for(long od=0;od<outDepth;od++) for(long oh=0;oh<outHeight;oh++) for(long ow=0;ow<outWidth;ow++) {
                double sum=bias?promoted(inputTypes.get(2),resultType,read(inputs[2],offsets[2],strides[2],oc)):0.0;
                for(long ic=0;ic<channelsPerGroup;ic++) for(long kd=0;kd<kernelDepth;kd++) {
                    long id=od*attrs.strideDepth()-attrs.paddingDepth()+kd*attrs.dilationDepth();
                    for(long kh=0;kh<kernelHeight;kh++) {
                        long ih=oh*attrs.strideHeight()-attrs.paddingHeight()+kh*attrs.dilationHeight();
                        for(long kw=0;kw<kernelWidth;kw++) {
                            long iw=ow*attrs.strideWidth()-attrs.paddingWidth()+kw*attrs.dilationWidth();
                            double x=id<0||id>=inDepth||ih<0||ih>=inHeight||iw<0||iw>=inWidth?0.0:
                                    promoted(inputTypes.get(0),resultType,read(inputs[0],offsets[0],strides[0],n,inputChannelBase+ic,id,ih,iw));
                            double w=promoted(inputTypes.get(1),resultType,read(inputs[1],offsets[1],strides[1],oc,ic,kd,kh,kw));
                            sum=addProduct(resultType,sum,x,w);
                        }
                    }
                }
                output[Math.toIntExact(ordinal++)]=represented(resultType,sum);
            }
        }
        return output;
    }

    private static double addProduct(DataType type,double sum,double left,double right){return type==DataType.FLOAT64?sum+left*right:(float)((float)sum+(float)left*(float)right);}
    private static double promoted(DataType source,DataType result,double value){double represented=represented(source,value);return result==DataType.FLOAT64?represented:(float)represented;}
    private static double represented(DataType type,double value){if(type==DataType.FLOAT64)return value;float narrowed=(float)value;if(type==DataType.FLOAT32)return narrowed;int bits=Float.floatToRawIntBits(narrowed);if((bits&0x7fff_ffff)>0x7f80_0000)return Float.intBitsToFloat(0x7fc0_0000);bits+=0x7fff+((bits>>>16)&1);return Float.intBitsToFloat(bits&0xffff_0000);}
    private static double read(double[] values,long offset,long[] strides,long...coordinates){long address=offset;for(int i=0;i<coordinates.length;i++)address=Math.addExact(address,Math.multiplyExact(coordinates[i],strides[i]));return values[Math.toIntExact(address)];}
    private static void validate(double[] values,long[] extents,long offset,long[] strides){Objects.requireNonNull(values,"input");Objects.requireNonNull(extents,"extents");Objects.requireNonNull(strides,"strides");if(extents.length!=strides.length||offset<0||Arrays.stream(extents).anyMatch(v->v<0)||Arrays.stream(strides).anyMatch(v->v<0))throw new IllegalArgumentException("reference layout is invalid");if(count(extents)==0)return;long maximum=offset;for(int i=0;i<extents.length;i++)maximum=Math.addExact(maximum,Math.multiplyExact(extents[i]-1,strides[i]));if(maximum>=values.length)throw new IllegalArgumentException("reference span is too small");}
    private static long count(long[] extents){for(long extent:extents)if(extent==0)return 0;long result=1;for(long extent:extents)result=Math.multiplyExact(result,extent);return result;}
}
