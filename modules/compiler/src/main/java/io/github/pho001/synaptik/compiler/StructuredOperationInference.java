package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.attention.*;
import io.github.pho001.synaptik.model.operation.convolution.*;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.operation.loss.*;
import io.github.pho001.synaptik.model.operation.pooling.*;
import io.github.pho001.synaptik.model.operation.random.*;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.ArrayList;
import java.util.List;

/**
 * Derives descriptors and constraints for structured numeric, loss, and explicit graph-state
 * operation families.
 *
 * <p>Conv3d and Pool3d inference independently reconstruct complete NCDHW descriptors from
 * captured inputs and exact typed attributes. Candidate obligations are emitted in stable axis
 * order; the surrounding captured-graph pass proves, rejects, or retains them and repeats that
 * work for the final optimized graph. No inference route lowers an operation, selects execution
 * capability, or supplies a gradient rule.</p>
 */
final class StructuredOperationInference {
    private StructuredOperationInference() {}

    /**
     * Verifies one supported matrix, attention, convolution, pooling, loss, RNG, or dropout
     * occurrence.
     *
     * @param op non-null typed operation whose kind belongs to this family
     * @param in non-null ordered input descriptors supplied by the captured node
     * @param outputCount structurally validated stored output count, used only to select an
     *     accepted multi-output signature
     * @return immutable independently derived descriptors and ordered constraints; never
     *     {@code null}
     * @throws IllegalArgumentException if the kind is unsupported here or the occurrence violates
     *     its operand, Shape, state, output-role, or attribute contract
     */
    static CapturedGraphInference.InferenceResult infer(
            Operation op, List<TensorDescriptor> in, int outputCount) {
        if(op.kind() instanceof MatmulKind)return matmul(in);
        if(op.kind() instanceof ScaledDotProductAttentionKind)return attention((ScaledDotProductAttentionAttrs)op.attrs(),in,outputCount);
        if(op.kind() instanceof Conv2dKind)return conv((Conv2dAttrs)op.attrs(),in);
        if(op.kind() instanceof Conv3dKind)return conv3d(op,in,outputCount);
        if(op.kind() instanceof Pool2dKind k)return pool(k,op.attrs(),in);
        if(op.kind() instanceof Pool3dKind)return pool3d(op,in,outputCount);
        if(op.kind() instanceof LossKind k)return loss(k,op.attrs(),in);
        if(op.kind() instanceof GraphRngKind)return CapturedGraphInference.InferenceResult.of(ElementwiseInference.descriptor(DataType.INT64,Shape.of(2),false));
        if(op.kind() instanceof DropoutKind)return dropout(in);
        throw new IllegalArgumentException("unsupported structured kind");
    }
    private static CapturedGraphInference.InferenceResult matmul(List<TensorDescriptor>in){TensorDescriptor l=in.get(0),r=in.get(1);DataType t=DataTypePromotion.promoteNumeric(l.dataType(),r.dataType());Shape a=l.shape(),b=r.shape();if(a.rank()<1||b.rank()<1)throw new IllegalArgumentException("matmul rank must be at least one");List<CapturedGraphInference.ConstraintRequest>cs=new ArrayList<>();cs.add(new CapturedGraphInference.ConstraintRequest("matmul contraction",new DimensionEqual(a.dimension(a.rank()-1),b.dimension(b.rank()==1?0:b.rank()-2))));int ar=Math.max(0,a.rank()-2),br=Math.max(0,b.rank()-2),rr=Math.max(ar,br);List<Dimension>d=new ArrayList<>();for(int i=0;i<rr;i++){Dimension x=i<rr-ar?null:a.dimension(i-(rr-ar)),y=i<rr-br?null:b.dimension(i-(rr-br));d.add(batch(x,y,"matmul",i,cs));}if(a.rank()!=1)d.add(a.dimension(a.rank()-2));if(b.rank()!=1)d.add(b.dimension(b.rank()-1));return new CapturedGraphInference.InferenceResult(List.of(ElementwiseInference.descriptor(t,Shape.ofDimensions(d.toArray(Dimension[]::new)),l.requiresGrad()||r.requiresGrad())),cs);}
    private static CapturedGraphInference.InferenceResult attention(ScaledDotProductAttentionAttrs attrs,List<TensorDescriptor>in,int outputCount){TensorDescriptor q=in.get(0),k=in.get(1),v=in.get(2);ElementwiseInference.requireFloating(q,"query");ElementwiseInference.requireFloating(k,"key");ElementwiseInference.requireFloating(v,"value");if(q.shape().rank()<2||k.shape().rank()<2||v.shape().rank()<2)throw new IllegalArgumentException("attention rank must be at least two");DataType t=DataTypePromotion.promoteFloating(DataTypePromotion.promoteFloating(q.dataType(),k.dataType()),v.dataType());if(attrs.scale().isPresent()&&attrs.scale().orElseThrow().dataType()!=t)throw new IllegalArgumentException("scale data type mismatch");List<CapturedGraphInference.ConstraintRequest>cs=new ArrayList<>();Dimension qe=q.shape().dimension(q.shape().rank()-1),ke=k.shape().dimension(k.shape().rank()-1),ks=k.shape().dimension(k.shape().rank()-2),vs=v.shape().dimension(v.shape().rank()-2);cs.add(new CapturedGraphInference.ConstraintRequest("attention embedding",new DimensionEqual(qe,ke)));cs.add(new CapturedGraphInference.ConstraintRequest("attention embedding positive",new DimensionAtLeast(qe,1)));cs.add(new CapturedGraphInference.ConstraintRequest("attention key/value sequence",new DimensionEqual(ks,vs)));int qr=q.shape().rank()-2,kr=k.shape().rank()-2,vr=v.shape().rank()-2,rr=Math.max(qr,Math.max(kr,vr));List<Dimension>batch=new ArrayList<>();for(int i=0;i<rr;i++){Dimension a=aligned(q.shape(),qr,rr,i),b=aligned(k.shape(),kr,rr,i),c=aligned(v.shape(),vr,rr,i);Dimension selected=batch(a,b,"attention",i,cs);selected=batch(selected,c,"attention",i,cs);batch.add(selected);}List<Dimension>scores=new ArrayList<>(batch);scores.add(q.shape().dimension(q.shape().rank()-2));scores.add(ks);Shape score=Shape.ofDimensions(scores.toArray(Dimension[]::new));if(in.size()==4){TensorDescriptor mask=in.get(3);ElementwiseInference.requireBool(mask,"mask");if(mask.shape().rank()>score.rank())throw new IllegalArgumentException("mask rank exceeds score rank");int off=score.rank()-mask.shape().rank();for(int i=0;i<mask.shape().rank();i++)cs.add(new CapturedGraphInference.ConstraintRequest("attention mask axis "+i,new AnyOf(List.of(new DimensionEqual(mask.shape().dimension(i),new StaticDimension(1)),new DimensionEqual(mask.shape().dimension(i),score.dimension(off+i))))));}List<Dimension>out=new ArrayList<>(batch);out.add(q.shape().dimension(q.shape().rank()-2));out.add(v.shape().dimension(v.shape().rank()-1));TensorDescriptor output=ElementwiseInference.descriptor(t,Shape.ofDimensions(out.toArray(Dimension[]::new)),q.requiresGrad()||k.requiresGrad()||v.requiresGrad());List<TensorDescriptor>outs=new ArrayList<>();outs.add(output);if(outputCount==2)outs.add(ElementwiseInference.descriptor(t,score,q.requiresGrad()||k.requiresGrad()));return new CapturedGraphInference.InferenceResult(outs,cs);}
    private static CapturedGraphInference.InferenceResult conv(Conv2dAttrs a,List<TensorDescriptor>in){TensorDescriptor x=in.get(0),w=in.get(1),bias=in.size()==3?in.get(2):null;ElementwiseInference.requireFloating(x,"input");ElementwiseInference.requireFloating(w,"weight");if(x.shape().rank()!=4||w.shape().rank()!=4)throw new IllegalArgumentException("conv rank must be four");DataType t=DataTypePromotion.promoteFloating(x.dataType(),w.dataType());boolean grad=x.requiresGrad()||w.requiresGrad();if(bias!=null){ElementwiseInference.requireFloating(bias,"bias");if(bias.shape().rank()!=1)throw new IllegalArgumentException("bias rank must be one");t=DataTypePromotion.promoteFloating(t,bias.dataType());grad|=bias.requiresGrad();}if(!(w.shape().dimension(2)instanceof StaticDimension kh)||kh.size()==0||!(w.shape().dimension(3)instanceof StaticDimension kw)||kw.size()==0)throw new IllegalArgumentException("kernel dimensions must be positive static");List<CapturedGraphInference.ConstraintRequest>cs=new ArrayList<>();Dimension ic=x.shape().dimension(1),oc=w.shape().dimension(0),wc=w.shape().dimension(1);cs.add(new CapturedGraphInference.ConstraintRequest("input channels divisible by groups",new DimensionDivisible(ic,a.groups())));cs.add(new CapturedGraphInference.ConstraintRequest("output channels divisible by groups",new DimensionDivisible(oc,a.groups())));cs.add(new CapturedGraphInference.ConstraintRequest("weight channels",new DimensionEqual(DimensionExpressions.multiply(wc,a.groups()),ic)));if(bias!=null)cs.add(new CapturedGraphInference.ConstraintRequest("bias channels",new DimensionEqual(bias.shape().dimension(0),oc)));Dimension oh=spatial(x.shape().dimension(2),kh.size(),a.paddingHeight(),a.dilationHeight(),a.strideHeight(),false),ow=spatial(x.shape().dimension(3),kw.size(),a.paddingWidth(),a.dilationWidth(),a.strideWidth(),false);return new CapturedGraphInference.InferenceResult(List.of(ElementwiseInference.descriptor(t,Shape.ofDimensions(x.shape().dimension(0),oc,oh,ow),grad)),cs);}

    /**
     * Independently derives one exact grouped NCDHW convolution result.
     *
     * @param operation non-null exact Conv3d operation, including retained immutable attributes
     * @param inputs non-null ordered two- or three-role descriptor list
     * @param outputCount stored output cardinality, which must be exactly one
     * @return immutable promoted result descriptor and candidate constraints in task-defined order
     * @throws IllegalArgumentException if kind, attributes, cardinality, floating roles, ranks,
     *     static kernels, geometry, or output cardinality violates the Conv3d contract
     * @throws ArithmeticException if checked kernel, padding, channel, or spatial arithmetic
     *     overflows {@code long}
     */
    private static CapturedGraphInference.InferenceResult conv3d(
            Operation operation, List<TensorDescriptor> inputs, int outputCount) {
        if (operation.kind() != Conv3dKind.CONV3D
                || !(operation.attrs() instanceof Conv3dAttrs attrs)) {
            throw new IllegalArgumentException("unsupported conv3d kind or attributes");
        }
        if ((inputs.size() != 2 && inputs.size() != 3) || outputCount != 1) {
            throw new IllegalArgumentException(
                    "conv3d requires two or three inputs and exactly one output");
        }

        TensorDescriptor input = inputs.get(0);
        TensorDescriptor weight = inputs.get(1);
        TensorDescriptor bias = inputs.size() == 3 ? inputs.get(2) : null;
        ElementwiseInference.requireFloating(input, "conv3d input");
        ElementwiseInference.requireFloating(weight, "conv3d weight");
        if (bias != null) {
            ElementwiseInference.requireFloating(bias, "conv3d bias");
        }

        DataType resultType = DataTypePromotion.promoteFloating(
                input.dataType(), weight.dataType());
        if (bias != null) {
            resultType = DataTypePromotion.promoteFloating(resultType, bias.dataType());
        }

        Shape inputShape = input.shape();
        Shape weightShape = weight.shape();
        requireConv3dRank(inputShape, 5, "input");
        requireConv3dRank(weightShape, 5, "weight");
        if (bias != null) {
            requireConv3dRank(bias.shape(), 1, "bias");
        }
        long kernelDepth = requirePositiveStaticConv3dKernel(
                weightShape.dimension(2), "depth");
        long kernelHeight = requirePositiveStaticConv3dKernel(
                weightShape.dimension(3), "height");
        long kernelWidth = requirePositiveStaticConv3dKernel(
                weightShape.dimension(4), "width");

        Dimension inputChannels = inputShape.dimension(1);
        Dimension outputChannels = weightShape.dimension(0);
        Dimension weightChannelsPerGroup = weightShape.dimension(1);
        List<CapturedGraphInference.ConstraintRequest> constraints = new ArrayList<>();
        constraints.add(new CapturedGraphInference.ConstraintRequest(
                "conv3d input channels divisible by groups",
                new DimensionDivisible(inputChannels, attrs.groups())));
        constraints.add(new CapturedGraphInference.ConstraintRequest(
                "conv3d output channels divisible by groups",
                new DimensionDivisible(outputChannels, attrs.groups())));
        constraints.add(new CapturedGraphInference.ConstraintRequest(
                "conv3d weight channels per group",
                new DimensionEqual(
                        DimensionExpressions.multiply(weightChannelsPerGroup, attrs.groups()),
                        inputChannels)));
        if (bias != null) {
            constraints.add(new CapturedGraphInference.ConstraintRequest(
                    "conv3d bias channels",
                    new DimensionEqual(bias.shape().dimension(0), outputChannels)));
        }

        Conv3dSpatial depth = conv3dSpatial(
                inputShape.dimension(2), kernelDepth, attrs.paddingDepth(),
                attrs.dilationDepth(), attrs.strideDepth(), "depth");
        Conv3dSpatial height = conv3dSpatial(
                inputShape.dimension(3), kernelHeight, attrs.paddingHeight(),
                attrs.dilationHeight(), attrs.strideHeight(), "height");
        Conv3dSpatial width = conv3dSpatial(
                inputShape.dimension(4), kernelWidth, attrs.paddingWidth(),
                attrs.dilationWidth(), attrs.strideWidth(), "width");
        constraints.add(new CapturedGraphInference.ConstraintRequest(
                "conv3d depth numerator non-negative",
                new DimensionAtLeast(depth.numerator(), 0)));
        constraints.add(new CapturedGraphInference.ConstraintRequest(
                "conv3d height numerator non-negative",
                new DimensionAtLeast(height.numerator(), 0)));
        constraints.add(new CapturedGraphInference.ConstraintRequest(
                "conv3d width numerator non-negative",
                new DimensionAtLeast(width.numerator(), 0)));

        boolean requiresGrad = input.requiresGrad() || weight.requiresGrad()
                || bias != null && bias.requiresGrad();
        TensorDescriptor output = ElementwiseInference.descriptor(
                resultType,
                Shape.ofDimensions(
                        inputShape.dimension(0), outputChannels,
                        depth.output(), height.output(), width.output()),
                requiresGrad);
        return new CapturedGraphInference.InferenceResult(List.of(output), constraints);
    }

    /**
     * Requires one Conv3d semantic role to have its exact rank.
     *
     * @param shape non-null captured role Shape
     * @param expected required rank
     * @param role non-null diagnostic role name
     * @throws IllegalArgumentException if the actual rank differs
     */
    private static void requireConv3dRank(Shape shape, int expected, String role) {
        if (shape.rank() != expected) {
            throw new IllegalArgumentException(
                    "conv3d " + role + " rank must be " + expected + ": " + shape.rank());
        }
    }

    /**
     * Resolves one required positive static Conv3d kernel extent.
     *
     * @param dimension non-null captured kernel Dimension
     * @param axis non-null depth, height, or width diagnostic name
     * @return the positive static kernel extent
     * @throws IllegalArgumentException if the extent is unresolved or zero
     */
    private static long requirePositiveStaticConv3dKernel(Dimension dimension, String axis) {
        if (!(dimension instanceof StaticDimension staticDimension)) {
            throw new IllegalArgumentException(
                    "conv3d kernel " + axis + " must be static: " + dimension);
        }
        if (staticDimension.size() == 0) {
            throw new IllegalArgumentException(
                    "conv3d kernel " + axis + " must be positive: " + dimension);
        }
        return staticDimension.size();
    }

    /**
     * Derives one checked floor-mode spatial numerator and output Dimension.
     *
     * @param input non-null input spatial Dimension
     * @param kernel positive static kernel extent
     * @param padding non-negative symmetric padding per side
     * @param dilation positive kernel dilation
     * @param stride positive output stride
     * @param axis non-null diagnostic axis name
     * @return non-null numerator/output pair; an unresolved numerator remains available for the
     *     ordered non-negative constraint
     * @throws IllegalArgumentException if a static effective kernel does not fit
     * @throws ArithmeticException if checked geometry overflows {@code long}
     */
    private static Conv3dSpatial conv3dSpatial(
            Dimension input,
            long kernel,
            long padding,
            long dilation,
            long stride,
            String axis) {
        long effectiveKernel = Math.addExact(
                Math.multiplyExact(dilation, Math.subtractExact(kernel, 1)), 1);
        long doublePadding = Math.multiplyExact(2, padding);
        if (input instanceof StaticDimension staticInput) {
            long paddedInput = Math.addExact(staticInput.size(), doublePadding);
            long numerator = Math.subtractExact(paddedInput, effectiveKernel);
            if (numerator < 0) {
                throw new IllegalArgumentException(
                        "conv3d effective kernel does not fit padded " + axis + ": input="
                                + input + ", effectiveKernel=" + effectiveKernel
                                + ", padding=" + padding);
            }
            Dimension numeratorDimension = new StaticDimension(numerator);
            return new Conv3dSpatial(
                    numeratorDimension,
                    new StaticDimension(Math.addExact(numerator / stride, 1)));
        }
        long offset = Math.subtractExact(doublePadding, effectiveKernel);
        Dimension numerator = DimensionExpressions.addConstant(input, offset);
        Dimension output = DimensionExpressions.addConstant(
                DimensionExpressions.floorDivide(numerator, stride), 1);
        return new Conv3dSpatial(numerator, output);
    }

    /**
     * One axis's exact numerator obligation and derived output Dimension.
     *
     * @param numerator non-null pre-division padded-kernel numerator
     * @param output non-null floor-divided output extent plus one
     */
    private record Conv3dSpatial(Dimension numerator, Dimension output) {}
    private static CapturedGraphInference.InferenceResult pool(Pool2dKind kind,Object raw,List<TensorDescriptor>in){TensorDescriptor x=in.get(0);ElementwiseInference.requireFloating(x,"input");if(x.shape().rank()!=4)throw new IllegalArgumentException("pool input rank must be four");long kh,kw,ph,pw,dh,dw,sh,sw;boolean ceil;if(kind==Pool2dKind.MAX_POOL2D){MaxPool2dAttrs a=(MaxPool2dAttrs)raw;kh=a.kernelHeight();kw=a.kernelWidth();ph=a.paddingHeight();pw=a.paddingWidth();dh=a.dilationHeight();dw=a.dilationWidth();sh=a.strideHeight();sw=a.strideWidth();ceil=a.ceilMode();}else{AveragePool2dAttrs a=(AveragePool2dAttrs)raw;kh=a.kernelHeight();kw=a.kernelWidth();ph=a.paddingHeight();pw=a.paddingWidth();dh=a.dilationHeight();dw=a.dilationWidth();sh=a.strideHeight();sw=a.strideWidth();ceil=a.ceilMode();}Shape s=Shape.ofDimensions(x.shape().dimension(0),x.shape().dimension(1),spatial(x.shape().dimension(2),kh,ph,dh,sh,ceil),spatial(x.shape().dimension(3),kw,pw,dw,sw,ceil));return CapturedGraphInference.InferenceResult.of(ElementwiseInference.descriptor(x.dataType(),s,x.requiresGrad()));}

    /**
     * Independently derives one exact rank-five Pool3d result and its spatial obligations.
     *
     * @param operation non-null maximum- or average-Pool3d operation
     * @param inputs non-null ordered descriptor list containing exactly one input
     * @param outputCount structurally retained output count, required to be one
     * @return immutable result descriptor and depth, height, then width constraints
     * @throws IllegalArgumentException if kind, attributes, cardinality, floating type, rank, or
     *     static spatial geometry violates the Pool3d contract
     * @throws ArithmeticException if checked geometry arithmetic overflows {@code long}
     */
    private static CapturedGraphInference.InferenceResult pool3d(
            Operation operation, List<TensorDescriptor> inputs, int outputCount) {
        if (inputs.size() != 1 || outputCount != 1) {
            throw new IllegalArgumentException("pool3d requires exactly one input and one output");
        }
        Pool3dGeometry geometry;
        if (operation.kind() == Pool3dKind.MAX_POOL3D
                && operation.attrs() instanceof MaxPool3dAttrs attrs) {
            geometry = new Pool3dGeometry(
                    attrs.kernelDepth(), attrs.kernelHeight(), attrs.kernelWidth(),
                    attrs.strideDepth(), attrs.strideHeight(), attrs.strideWidth(),
                    attrs.paddingDepth(), attrs.paddingHeight(), attrs.paddingWidth(),
                    attrs.dilationDepth(), attrs.dilationHeight(), attrs.dilationWidth(),
                    attrs.ceilMode());
        } else if (operation.kind() == Pool3dKind.AVERAGE_POOL3D
                && operation.attrs() instanceof AveragePool3dAttrs attrs) {
            geometry = new Pool3dGeometry(
                    attrs.kernelDepth(), attrs.kernelHeight(), attrs.kernelWidth(),
                    attrs.strideDepth(), attrs.strideHeight(), attrs.strideWidth(),
                    attrs.paddingDepth(), attrs.paddingHeight(), attrs.paddingWidth(),
                    attrs.dilationDepth(), attrs.dilationHeight(), attrs.dilationWidth(),
                    attrs.ceilMode());
        } else {
            throw new IllegalArgumentException("unsupported pool3d kind or attributes");
        }
        TensorDescriptor input = inputs.getFirst();
        ElementwiseInference.requireFloating(input, "pool3d input");
        if (input.shape().rank() != 5) {
            throw new IllegalArgumentException("pool3d input rank must be five");
        }
        Pool3dSpatial depth = pool3dSpatial(
                input.shape().dimension(2), geometry.kernelDepth(), geometry.paddingDepth(),
                geometry.dilationDepth(), geometry.strideDepth(), geometry.ceilMode(), "depth");
        Pool3dSpatial height = pool3dSpatial(
                input.shape().dimension(3), geometry.kernelHeight(), geometry.paddingHeight(),
                geometry.dilationHeight(), geometry.strideHeight(), geometry.ceilMode(), "height");
        Pool3dSpatial width = pool3dSpatial(
                input.shape().dimension(4), geometry.kernelWidth(), geometry.paddingWidth(),
                geometry.dilationWidth(), geometry.strideWidth(), geometry.ceilMode(), "width");
        List<CapturedGraphInference.ConstraintRequest> constraints = List.of(
                new CapturedGraphInference.ConstraintRequest(
                        "pool3d depth numerator non-negative",
                        new DimensionAtLeast(depth.numerator(), 0)),
                new CapturedGraphInference.ConstraintRequest(
                        "pool3d height numerator non-negative",
                        new DimensionAtLeast(height.numerator(), 0)),
                new CapturedGraphInference.ConstraintRequest(
                        "pool3d width numerator non-negative",
                        new DimensionAtLeast(width.numerator(), 0)));
        Shape outputShape = Shape.ofDimensions(
                input.shape().dimension(0), input.shape().dimension(1),
                depth.output(), height.output(), width.output());
        return new CapturedGraphInference.InferenceResult(
                List.of(ElementwiseInference.descriptor(
                        input.dataType(), outputShape, input.requiresGrad())),
                constraints);
    }

    /**
     * Derives one checked Pool3d numerator and floor- or literal-ceil output extent.
     *
     * @param input non-null captured spatial Dimension
     * @param kernel positive kernel extent
     * @param padding non-negative symmetric padding per side
     * @param dilation positive kernel dilation
     * @param stride positive output stride
     * @param ceilMode whether to use literal ceiling division
     * @param axis non-null diagnostic axis name
     * @return exact numerator and output Dimensions
     * @throws IllegalArgumentException if a static numerator is negative
     * @throws ArithmeticException if checked geometry arithmetic overflows {@code long}
     */
    private static Pool3dSpatial pool3dSpatial(
            Dimension input, long kernel, long padding, long dilation, long stride,
            boolean ceilMode, String axis) {
        long effective = Math.addExact(Math.multiplyExact(dilation, kernel - 1), 1);
        long doubledPadding = Math.multiplyExact(2, padding);
        if (input instanceof StaticDimension staticInput) {
            long numerator = Math.subtractExact(
                    Math.addExact(staticInput.size(), doubledPadding), effective);
            if (numerator < 0) {
                throw new IllegalArgumentException(
                        "pool3d effective kernel does not fit padded " + axis);
            }
            return new Pool3dSpatial(
                    new StaticDimension(numerator),
                    new StaticDimension(Math.addExact(
                            numerator / stride
                                    + (ceilMode && numerator % stride != 0 ? 1 : 0),
                            1)));
        }
        Dimension numerator = DimensionExpressions.addConstant(
                input, Math.subtractExact(doubledPadding, effective));
        Dimension quotient = ceilMode
                ? DimensionExpressions.ceilingDivide(numerator, stride)
                : DimensionExpressions.floorDivide(numerator, stride);
        return new Pool3dSpatial(
                numerator, DimensionExpressions.addConstant(quotient, 1));
    }

    /**
     * Holds one Pool3d axis's pre-division numerator and derived output extent.
     *
     * @param numerator non-null padded-input-minus-effective-kernel Dimension
     * @param output non-null divided numerator plus one
     */
    private record Pool3dSpatial(Dimension numerator, Dimension output) {}

    /**
     * Normalizes the two exact Pool3d attribute records without erasing their operation identity.
     *
     * @param kernelDepth positive depth kernel extent
     * @param kernelHeight positive height kernel extent
     * @param kernelWidth positive width kernel extent
     * @param strideDepth positive depth stride
     * @param strideHeight positive height stride
     * @param strideWidth positive width stride
     * @param paddingDepth non-negative depth padding per side
     * @param paddingHeight non-negative height padding per side
     * @param paddingWidth non-negative width padding per side
     * @param dilationDepth positive depth dilation
     * @param dilationHeight positive height dilation
     * @param dilationWidth positive width dilation
     * @param ceilMode whether each output axis uses literal ceiling division
     */
    private record Pool3dGeometry(
            long kernelDepth, long kernelHeight, long kernelWidth,
            long strideDepth, long strideHeight, long strideWidth,
            long paddingDepth, long paddingHeight, long paddingWidth,
            long dilationDepth, long dilationHeight, long dilationWidth,
            boolean ceilMode) {}
    private static CapturedGraphInference.InferenceResult loss(LossKind kind,Object raw,List<TensorDescriptor>in){TensorDescriptor logits=in.get(0),target=in.get(1);ElementwiseInference.requireFloating(logits,"prediction/logits");List<CapturedGraphInference.ConstraintRequest>cs=new ArrayList<>();Shape result;DataType t;boolean grad;if(kind==LossKind.MEAN_SQUARED_ERROR){ElementwiseInference.requireFloating(target,"target");exactShape(logits.shape(),target.shape(),"MSE",cs);t=DataTypePromotion.promoteFloating(logits.dataType(),target.dataType());LossReduction r=((MeanSquaredErrorAttrs)raw).reduction();result=r==LossReduction.NONE?logits.shape():Shape.scalar();grad=logits.requiresGrad()||target.requiresGrad();}else if(kind==LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS){ElementwiseInference.requireFloating(target,"target");DenseCategoricalCrossEntropyWithLogitsAttrs a=(DenseCategoricalCrossEntropyWithLogitsAttrs)raw;axis(logits.shape(),a.axis());exactShape(logits.shape(),target.shape(),"dense target",cs);t=DataTypePromotion.promoteFloating(logits.dataType(),target.dataType());result=a.reduction()==LossReduction.NONE?remove(logits.shape(),a.axis()):Shape.scalar();grad=logits.requiresGrad()||target.requiresGrad();cs.add(classConstraint(logits.shape(),a.axis()));}else{ElementwiseInference.requireIndex(target,"target");IndexCategoricalCrossEntropyWithLogitsAttrs a=(IndexCategoricalCrossEntropyWithLogitsAttrs)raw;axis(logits.shape(),a.axis());Shape expected=remove(logits.shape(),a.axis());exactShape(expected,target.shape(),"index target",cs);if(a.ignoreIndex().isPresent()&&a.ignoreIndex().orElseThrow().dataType()!=target.dataType())throw new IllegalArgumentException("ignore index type mismatch");if(a.ignoreIndex().isEmpty())cs.add(classConstraint(logits.shape(),a.axis()));t=logits.dataType();result=a.reduction()==LossReduction.NONE?target.shape():Shape.scalar();grad=logits.requiresGrad();}return new CapturedGraphInference.InferenceResult(List.of(ElementwiseInference.descriptor(t,result,grad)),cs);}
    private static CapturedGraphInference.ConstraintRequest classConstraint(Shape s,int axis){return new CapturedGraphInference.ConstraintRequest("categorical class extent",new AnyOf(List.of(new ShapeElementCountValue(remove(s,axis),0,ShapeElementCountValue.Comparison.EQUAL),new DimensionAtLeast(s.dimension(axis),1))));}
    private static CapturedGraphInference.InferenceResult dropout(List<TensorDescriptor>in){TensorDescriptor x=in.get(0),state=in.get(1);ElementwiseInference.requireFloating(x,"input");if(state.dataType()!=DataType.INT64||!state.shape().equals(Shape.of(2))||state.requiresGrad())throw new IllegalArgumentException("invalid RNG state descriptor");return new CapturedGraphInference.InferenceResult(List.of(ElementwiseInference.descriptor(x.dataType(),x.shape(),x.requiresGrad()),ElementwiseInference.descriptor(DataType.BOOL,x.shape(),false),ElementwiseInference.descriptor(DataType.INT64,Shape.of(2),false)),List.of());}
    private static Dimension batch(Dimension a,Dimension b,String name,int axis,List<CapturedGraphInference.ConstraintRequest>cs){if(a==null)return b;if(b==null)return a;if(a.equals(b))return a;if(a instanceof StaticDimension sa&&sa.size()==1)return b;if(b instanceof StaticDimension sb&&sb.size()==1)return a;if(a instanceof StaticDimension&&b instanceof StaticDimension)throw new IllegalArgumentException(name+" batch mismatch");Dimension selected=a instanceof StaticDimension?a:b instanceof StaticDimension?b:null;if(selected==null)throw new IllegalArgumentException(name+" cannot select batch dimension");Dimension other=selected==a?b:a;cs.add(new CapturedGraphInference.ConstraintRequest(name+" batch axis "+axis,new AnyOf(List.of(new DimensionEqual(other,new StaticDimension(1)),new DimensionEqual(other,selected)))));return selected;}
    private static Dimension aligned(Shape s,int br,int rr,int axis){int off=rr-br;return axis<off?null:s.dimension(axis-off);}
    private static Dimension spatial(Dimension in,long kernel,long pad,long dilation,long stride,boolean ceil){long eff=Math.addExact(Math.multiplyExact(dilation,kernel-1),1),offset=Math.subtractExact(Math.multiplyExact(2,pad),eff);if(in instanceof StaticDimension s){long n=Math.addExact(s.size(),offset);if(n<0)throw new IllegalArgumentException("effective kernel does not fit");return new StaticDimension(Math.addExact(n/stride+(ceil&&n%stride!=0?1:0),1));}Dimension n=DimensionExpressions.addConstant(in,offset);return DimensionExpressions.addConstant(ceil?DimensionExpressions.ceilingDivide(n,stride):DimensionExpressions.floorDivide(n,stride),1);}
    private static void exactShape(Shape a,Shape b,String role,List<CapturedGraphInference.ConstraintRequest>cs){if(a.rank()!=b.rank())throw new IllegalArgumentException(role+" rank mismatch");for(int i=0;i<a.rank();i++)cs.add(new CapturedGraphInference.ConstraintRequest(role+" axis "+i,new DimensionEqual(a.dimension(i),b.dimension(i))));}
    private static Shape remove(Shape s,int axis){List<Dimension>d=new ArrayList<>(s.dimensions());d.remove(axis);return Shape.ofDimensions(d.toArray(Dimension[]::new));}
    private static void axis(Shape s,int a){if(a<0||a>=s.rank())throw new IllegalArgumentException("axis out of range");}
}
