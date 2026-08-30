package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.layout.*;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Derives descriptors and occurrence-local constraints for layout, Shape, composition, and
 * window transforms.
 *
 * <p>Binding-dependent {@code EXPAND} keeps the exact requested target Shape, emits one ordered
 * source-one-or-source-equal predicate for each unresolved aligned pair, and leaves layout
 * unresolved. Structural equality, a static source singleton, and fully resolved zero-stride
 * view geometry retain their existing behavior. This owner records obligations only; it does not
 * bind dimensions, select materialization, or authorize execution.</p>
 *
 * <p>Finite {@code SliceAttrs} regions derive exact static selected lengths. Each non-empty
 * signed coordinate sequence is proved within its source extent or retained as an occurrence-local
 * upper-bound constraint; an empty sequence needs no bound. {@code CropToShapeAttrs} extraction
 * returns the exact target Shape, while placement validates the update against that target and
 * returns the exact base Shape. Both retain one {@code prefix + target <= base} obligation per
 * axis when it cannot yet be proved.</p>
 *
 * <p>General-axis window transforms retain their existing static transformed-axis contract.
 * Two- and three-dimensional unfold and fold preserve symbolic batch, channel, and spatial
 * expressions and retain independent spatial-domain obligations. Three-dimensional fold also
 * proves the exact batch, flattened channel-kernel, and flattened grid relationships against its
 * explicit NCDHW target. A disproved relation fails inference; an undecidable spatial relation
 * remains typed compiler state. This class does not bind dimensions, inspect values, choose an
 * algorithm, materialize a view, lower a backend operation, or execute computation.</p>
 */
final class LayoutInference {
    private LayoutInference() {}

    /**
     * Verifies one supported layout or Shape-transform occurrence.
     *
     * @param op non-null typed operation whose kind belongs to this family
     * @param in non-null ordered input descriptors supplied by the captured node
     * @return immutable derived descriptors and ordered occurrence-local candidate constraints;
     *     never {@code null}
     * @throws IllegalArgumentException if the kind is unsupported here or the occurrence violates
     *     its rank, Shape, layout, or attribute contract, including a fully static incompatible
     *     expansion pair
     */
    static CapturedGraphInference.InferenceResult infer(Operation op,List<TensorDescriptor> in){
        if(op.kind() instanceof ContiguousKind)return contiguous(in);
        if(op.kind() instanceof ShapeTransformKind k)return shapeTransform(k,(TargetShapeAttrs)op.attrs(),in);
        if(op.kind() instanceof AxisTransformKind k)return axisTransform(k,op.attrs(),in);
        if(op.kind() instanceof SliceKind k)return slice(k,op.attrs(),in);
        if(op.kind() instanceof PadKind)return pad((PadAttrs)op.attrs(),in);
        if(op.kind() instanceof TileKind)return tile((TileAttrs)op.attrs(),in);
        if(op.kind() instanceof TensorCompositionKind k)return composition(k,(CompositionAxisAttrs)op.attrs(),in);
        if(op.kind() instanceof WindowTransformKind k)return window(k,op.attrs(),in);
        throw new IllegalArgumentException("unsupported layout kind");
    }
    private static CapturedGraphInference.InferenceResult contiguous(List<TensorDescriptor> in){TensorDescriptor x=in.get(0);Optional<LayoutDescriptor> l=x.shape().isFullyStatic()?Optional.of(LayoutDescriptor.contiguous(x.shape())):Optional.empty();return CapturedGraphInference.InferenceResult.of(new TensorDescriptor(x.dataType(),x.shape(),l,x.requiresGrad()));}
    private static CapturedGraphInference.InferenceResult shapeTransform(ShapeTransformKind kind,TargetShapeAttrs attrs,List<TensorDescriptor> in){
        TensorDescriptor x=in.get(0);Shape target=attrs.targetShape();Optional<LayoutDescriptor> layout=Optional.empty();List<CapturedGraphInference.ConstraintRequest> cs=new ArrayList<>();
        if(kind==ShapeTransformKind.RESHAPE){cs.add(new CapturedGraphInference.ConstraintRequest("reshape element count",new ShapeElementCountEqual(x.shape(),target)));if(target.isFullyStatic()&&x.layout().isPresent()&&x.layout().orElseThrow().isContiguous()){LayoutDescriptor c=LayoutDescriptor.contiguous(target);layout=Optional.of(LayoutDescriptor.of(target,c.strides(),x.layout().orElseThrow().storageOffset(),true));}}
        else {
            if(target.rank()<x.shape().rank())throw new IllegalArgumentException("expand target rank below input rank");
            int off=target.rank()-x.shape().rank();
            boolean bindingDependent=false;
            for(int i=0;i<x.shape().rank();i++){
                Dimension source=x.shape().dimension(i),targetDimension=target.dimension(off+i);
                if(source.equals(targetDimension)
                        || source instanceof StaticDimension singleton&&singleton.size()==1){
                    continue;
                }
                if(source instanceof StaticDimension&&targetDimension instanceof StaticDimension){
                    throw new IllegalArgumentException("incompatible expand dimension");
                }
                bindingDependent=true;
                cs.add(new CapturedGraphInference.ConstraintRequest(
                        "expand target axis "+(off+i),
                        new AnyOf(List.of(
                                new DimensionEqual(source,new StaticDimension(1)),
                                new DimensionEqual(source,targetDimension)))));
            }
            if(!bindingDependent&&target.isFullyStatic()&&x.layout().isPresent()){
                long[] strides=new long[target.rank()];
                for(int i=0;i<x.shape().rank();i++){
                    int j=off+i;
                    Dimension source=x.shape().dimension(i),targetDimension=target.dimension(j);
                    strides[j]=source instanceof StaticDimension singleton
                                    &&singleton.size()==1&&!source.equals(targetDimension)
                            ?0:x.layout().orElseThrow().stride(i);
                }
                layout=Optional.of(LayoutDescriptor.of(
                        target,strides,x.layout().orElseThrow().storageOffset(),true));
            }
        }
        return new CapturedGraphInference.InferenceResult(List.of(new TensorDescriptor(x.dataType(),target,layout,x.requiresGrad())),cs);
    }
    private static CapturedGraphInference.InferenceResult axisTransform(AxisTransformKind kind,Object raw,List<TensorDescriptor> in){
        TensorDescriptor x=in.get(0);Shape shape;Optional<LayoutDescriptor> layout=Optional.empty();
        if(kind==AxisTransformKind.PERMUTE){PermutationAttrs a=(PermutationAttrs)raw;if(a.axes().size()!=x.shape().rank())throw new IllegalArgumentException("permutation rank mismatch");Dimension[] d=new Dimension[a.axes().size()];long[] strides=new long[d.length];for(int i=0;i<d.length;i++){d[i]=x.shape().dimension(a.axes().get(i));if(x.layout().isPresent())strides[i]=x.layout().orElseThrow().stride(a.axes().get(i));}shape=Shape.ofDimensions(d);if(x.layout().isPresent())layout=Optional.of(LayoutDescriptor.of(shape,strides,x.layout().orElseThrow().storageOffset(),true));}
        else {AxisTransformAttrs a=(AxisTransformAttrs)raw;List<Dimension>d=new ArrayList<>(x.shape().dimensions());if(kind==AxisTransformKind.EXPAND_DIMS){if(a.axis()<0||a.axis()>d.size())throw new IllegalArgumentException("insertion axis out of range");d.add(a.axis(),new StaticDimension(1));shape=Shape.ofDimensions(d.toArray(Dimension[]::new));if(x.layout().isPresent()){long[] s=new long[shape.rank()];for(int i=0;i<a.axis();i++)s[i]=x.layout().orElseThrow().stride(i);s[a.axis()]=a.axis()==x.shape().rank()?1:Math.multiplyExact(x.layout().orElseThrow().stride(a.axis()),((StaticDimension)x.shape().dimension(a.axis())).size());for(int i=a.axis();i<x.shape().rank();i++)s[i+1]=x.layout().orElseThrow().stride(i);layout=Optional.of(LayoutDescriptor.of(shape,s,x.layout().orElseThrow().storageOffset(),true));}}
            else {axis(x.shape(),a.axis());if(!(x.shape().dimension(a.axis()) instanceof StaticDimension singleton&&singleton.size()==1))throw new IllegalArgumentException("squeeze axis must be singleton");d.remove(a.axis());shape=Shape.ofDimensions(d.toArray(Dimension[]::new));if(x.layout().isPresent()){long[]resultStrides=new long[shape.rank()];for(int i=0,j=0;i<x.shape().rank();i++)if(i!=a.axis())resultStrides[j++]=x.layout().orElseThrow().stride(i);layout=Optional.of(LayoutDescriptor.of(shape,resultStrides,x.layout().orElseThrow().storageOffset(),true));}}}
        return CapturedGraphInference.InferenceResult.of(new TensorDescriptor(x.dataType(),shape,layout,x.requiresGrad()));
    }
    private static CapturedGraphInference.InferenceResult slice(
            SliceKind kind, Object raw, List<TensorDescriptor> in) {
        TensorDescriptor base = in.getFirst();
        if (raw instanceof CropToShapeAttrs attrs) {
            if (attrs.targetShape().rank() != base.shape().rank()
                    || attrs.prefixShape().rank() != base.shape().rank()) {
                throw new IllegalArgumentException("crop rank mismatch");
            }
            List<CapturedGraphInference.ConstraintRequest> constraints = new ArrayList<>();
            for (int axis = 0; axis < base.shape().rank(); axis++) {
                Dimension region = DimensionExpressions.add(
                        attrs.prefixShape().dimension(axis),
                        attrs.targetShape().dimension(axis));
                constraints.add(new CapturedGraphInference.ConstraintRequest(
                        "crop axis " + axis,
                        FitsWithin.dimension(0, region, base.shape().dimension(axis))));
            }
            if (kind == SliceKind.SLICE) {
                return new CapturedGraphInference.InferenceResult(
                        List.of(ElementwiseInference.descriptor(
                                base.dataType(), attrs.targetShape(), base.requiresGrad())),
                        constraints);
            }
            TensorDescriptor update = in.get(1);
            if (update.dataType() != base.dataType()
                    || !update.shape().equals(attrs.targetShape())) {
                throw new IllegalArgumentException("slice update descriptor mismatch");
            }
            return new CapturedGraphInference.InferenceResult(
                    List.of(ElementwiseInference.descriptor(
                            base.dataType(),
                            base.shape(),
                            base.requiresGrad() || update.requiresGrad())),
                    constraints);
        }

        SliceAttrs attrs = (SliceAttrs) raw;
        List<Dimension> dimensions = new ArrayList<>(base.shape().dimensions());
        List<CapturedGraphInference.ConstraintRequest> constraints = new ArrayList<>();
        long offset = base.layout().map(LayoutDescriptor::storageOffset).orElse(0L);
        long[] strides = base.layout().map(LayoutDescriptor::strides).orElse(null);
        for (int index = 0; index < attrs.axes().size(); index++) {
            int selectedAxis = attrs.axes().get(index);
            axis(base.shape(), selectedAxis);
            long length = attrs.lengths().get(index);
            dimensions.set(selectedAxis, new StaticDimension(length));
            long step = attrs.steps().get(index);
            if (length > 0) {
                long start = attrs.starts().get(index);
                long last = Math.addExact(start, Math.multiplyExact(length - 1, step));
                if (start < 0 || last < 0) {
                    throw new IllegalArgumentException("slice coordinate below zero");
                }
                long upperBound = Math.addExact(Math.max(start, last), 1);
                constraints.add(new CapturedGraphInference.ConstraintRequest(
                        "slice axis " + selectedAxis,
                        FitsWithin.dimension(
                                0, new StaticDimension(upperBound),
                                base.shape().dimension(selectedAxis))));
            }
            if (strides != null && step > 0) {
                offset = Math.addExact(
                        offset, Math.multiplyExact(attrs.starts().get(index), strides[selectedAxis]));
                strides[selectedAxis] = Math.multiplyExact(strides[selectedAxis], step);
            } else if (step < 0) {
                strides = null;
            }
        }
        Shape result = Shape.ofDimensions(dimensions.toArray(Dimension[]::new));
        if (kind == SliceKind.SLICE_UPDATE) {
            TensorDescriptor update = in.get(1);
            if (update.dataType() != base.dataType() || !update.shape().equals(result)) {
                throw new IllegalArgumentException("slice update descriptor mismatch");
            }
            return new CapturedGraphInference.InferenceResult(
                    List.of(ElementwiseInference.descriptor(
                            base.dataType(),
                            base.shape(),
                            base.requiresGrad() || update.requiresGrad())),
                    constraints);
        }
        Optional<LayoutDescriptor> layout = strides == null
                        || result.knownElementCount().orElseThrow() == 0
                ? Optional.empty()
                : Optional.of(LayoutDescriptor.of(result, strides, offset, true));
        return new CapturedGraphInference.InferenceResult(
                List.of(new TensorDescriptor(
                        base.dataType(), result, layout, base.requiresGrad())),
                constraints);
    }
    private static CapturedGraphInference.InferenceResult pad(PadAttrs a,List<TensorDescriptor>in){TensorDescriptor x=in.get(0);if(a.before().size()!=x.shape().rank()||a.constantValue().dataType()!=x.dataType())throw new IllegalArgumentException("pad attributes mismatch");Dimension[]d=new Dimension[x.shape().rank()];for(int i=0;i<d.length;i++)d[i]=DimensionExpressions.addConstant(x.shape().dimension(i),Math.addExact(a.before().get(i),a.after().get(i)));return CapturedGraphInference.InferenceResult.of(ElementwiseInference.descriptor(x.dataType(),Shape.ofDimensions(d),x.requiresGrad()));}
    private static CapturedGraphInference.InferenceResult tile(TileAttrs a,List<TensorDescriptor>in){TensorDescriptor x=in.get(0);if(a.repeats().size()!=x.shape().rank())throw new IllegalArgumentException("tile repeat rank mismatch");Dimension[]d=new Dimension[x.shape().rank()];for(int i=0;i<d.length;i++)d[i]=DimensionExpressions.multiply(x.shape().dimension(i),a.repeats().get(i));return CapturedGraphInference.InferenceResult.of(ElementwiseInference.descriptor(x.dataType(),Shape.ofDimensions(d),x.requiresGrad()));}
    private static CapturedGraphInference.InferenceResult composition(TensorCompositionKind kind,CompositionAxisAttrs a,List<TensorDescriptor>in){TensorDescriptor first=in.get(0);boolean grad=false;Shape result;if(kind==TensorCompositionKind.STACK){if(a.axis()>first.shape().rank())throw new IllegalArgumentException("stack axis out of range");for(TensorDescriptor x:in){sameType(first,x);if(!x.shape().equals(first.shape()))throw new IllegalArgumentException("stack shape mismatch");grad|=x.requiresGrad();}List<Dimension>d=new ArrayList<>(first.shape().dimensions());d.add(a.axis(),new StaticDimension(in.size()));result=Shape.ofDimensions(d.toArray(Dimension[]::new));}else{axis(first.shape(),a.axis());Dimension sum=new StaticDimension(0);for(TensorDescriptor x:in){sameType(first,x);if(x.shape().rank()!=first.shape().rank())throw new IllegalArgumentException("concat rank mismatch");for(int i=0;i<first.shape().rank();i++)if(i!=a.axis()&&!x.shape().dimension(i).equals(first.shape().dimension(i)))throw new IllegalArgumentException("concat dimension mismatch");sum=DimensionExpressions.add(sum,x.shape().dimension(a.axis()));grad|=x.requiresGrad();}List<Dimension>d=new ArrayList<>(first.shape().dimensions());d.set(a.axis(),sum);result=Shape.ofDimensions(d.toArray(Dimension[]::new));}return CapturedGraphInference.InferenceResult.of(ElementwiseInference.descriptor(first.dataType(),result,grad));}
    private static CapturedGraphInference.InferenceResult window(
            WindowTransformKind kind, Object raw, List<TensorDescriptor> in) {
        TensorDescriptor input = in.getFirst();
        Shape result;
        List<CapturedGraphInference.ConstraintRequest> constraints = new ArrayList<>();
        if (kind == WindowTransformKind.UNFOLD_AXIS) {
            UnfoldAxisAttrs attrs = (UnfoldAxisAttrs) raw;
            axis(input.shape(), attrs.axis());
            if (!(input.shape().dimension(attrs.axis()) instanceof StaticDimension selected)) {
                throw new IllegalArgumentException("unfold axis extent must be static");
            }
            if (selected.size() < attrs.size()) {
                throw new IllegalArgumentException("window does not fit");
            }
            long count = Math.addExact(
                    Math.subtractExact(selected.size(), attrs.size()) / attrs.step(), 1);
            List<Dimension> dimensions = new ArrayList<>(input.shape().dimensions());
            dimensions.set(attrs.axis(), new StaticDimension(count));
            dimensions.add(new StaticDimension(attrs.size()));
            result = Shape.ofDimensions(dimensions.toArray(Dimension[]::new));
        } else if (kind == WindowTransformKind.FOLD_AXIS) {
            FoldAxisAttrs attrs = (FoldAxisAttrs) raw;
            if (input.shape().rank() < 2) {
                throw new IllegalArgumentException("fold input rank too small");
            }
            Shape target = Shape.ofDimensions(input.shape().dimensions()
                    .subList(0, input.shape().rank() - 1)
                    .toArray(Dimension[]::new));
            axis(target, attrs.axis());
            List<Dimension> dimensions = new ArrayList<>(input.shape().dimensions());
            Dimension window = dimensions.removeLast();
            if (!(window instanceof StaticDimension windowSize)
                    || !(dimensions.get(attrs.axis()) instanceof StaticDimension count)) {
                throw new IllegalArgumentException("fold geometry must be static");
            }
            long expectedCount;
            if (attrs.outputSize() == 0) {
                expectedCount = 0;
            } else {
                if (windowSize.size() > attrs.outputSize()) {
                    throw new IllegalArgumentException("fold window exceeds output");
                }
                expectedCount = Math.addExact(
                        Math.subtractExact(attrs.outputSize(), windowSize.size())
                                / attrs.step(),
                        1);
            }
            if (count.size() != expectedCount) {
                throw new IllegalArgumentException("fold geometry mismatch");
            }
            dimensions.set(attrs.axis(), new StaticDimension(attrs.outputSize()));
            result = Shape.ofDimensions(dimensions.toArray(Dimension[]::new));
        } else if (kind == WindowTransformKind.UNFOLD2D
                || kind == WindowTransformKind.FOLD2D) {
            Window2dAttrs window = raw instanceof Unfold2dAttrs attrs
                    ? attrs.window()
                    : raw instanceof Fold2dAttrs attrs
                            ? attrs.window()
                            : (Window2dAttrs) raw;
            if (kind == WindowTransformKind.UNFOLD2D) {
                if (input.shape().rank() != 4) {
                    throw new IllegalArgumentException("unfold2d rank must be four");
                }
                if (raw instanceof Unfold2dAttrs attrs
                        && attrs.paddingValue().dataType() != input.dataType()) {
                    throw new IllegalArgumentException("padding value type mismatch");
                }
                result = unfoldShape(input.shape(), window, constraints, "unfold2d");
            } else {
                Fold2dAttrs attrs = (Fold2dAttrs) raw;
                if (input.shape().rank() != 3 || attrs.outputShape().rank() != 4) {
                    throw new IllegalArgumentException("fold2d rank mismatch");
                }
                Shape expected = unfoldShape(
                        attrs.outputShape(), window, constraints, "fold2d output");
                if (!input.shape().equals(expected)) {
                    throw new IllegalArgumentException("fold2d columns mismatch");
                }
                result = attrs.outputShape();
            }
        } else if (kind == WindowTransformKind.UNFOLD3D) {
            if (in.size() != 1) {
                throw new IllegalArgumentException("unfold3d requires exactly one input");
            }
            Window3dAttrs attrs;
            if (raw instanceof Window3dAttrs direct) {
                attrs = direct;
            } else if (raw instanceof Unfold3dAttrs explicit) {
                attrs = explicit.window();
                if (explicit.paddingValue().dataType() != input.dataType()) {
                    throw new IllegalArgumentException("unfold3d padding value type mismatch");
                }
            } else {
                throw new IllegalArgumentException("unsupported unfold3d attributes");
            }
            ElementwiseInference.requireFloating(input, "unfold3d input");
            if (input.shape().rank() != 5) {
                throw new IllegalArgumentException("unfold3d input rank must be five");
            }
            result = unfold3dShape(input.shape(), attrs, constraints, "unfold3d");
        } else if (kind == WindowTransformKind.FOLD3D) {
            if (in.size() != 1 || !(raw instanceof Fold3dAttrs attrs)) {
                throw new IllegalArgumentException("unsupported fold3d occurrence");
            }
            ElementwiseInference.requireFloating(input, "fold3d input");
            if (input.shape().rank() != 3 || attrs.outputShape().rank() != 5) {
                throw new IllegalArgumentException("fold3d rank mismatch");
            }
            Shape target = attrs.outputShape();
            if (!input.shape().dimension(0).equals(target.dimension(0))) {
                throw new IllegalArgumentException("fold3d batch dimension mismatch");
            }
            Dimension expectedColumns = DimensionExpressions.multiply(
                    DimensionExpressions.multiply(
                            DimensionExpressions.multiply(
                                    target.dimension(1), attrs.window().kernelDepth()),
                            attrs.window().kernelHeight()),
                    attrs.window().kernelWidth());
            if (!input.shape().dimension(1).equals(expectedColumns)) {
                throw new IllegalArgumentException("fold3d channel/kernel columns mismatch");
            }
            Shape expected = unfold3dShape(
                    target, attrs.window(), constraints, "fold3d output");
            if (!input.shape().dimension(2).equals(expected.dimension(2))) {
                throw new IllegalArgumentException("fold3d spatial columns mismatch");
            }
            result = target;
        } else {
            throw new IllegalArgumentException("unsupported window kind");
        }
        return new CapturedGraphInference.InferenceResult(
                List.of(ElementwiseInference.descriptor(
                        input.dataType(), result, input.requiresGrad())),
                constraints);
    }

    private static Shape unfoldShape(
            Shape shape,
            Window2dAttrs window,
            List<CapturedGraphInference.ConstraintRequest> constraints,
            String subject) {
        Dimension outputHeight = windowDimension(
                shape.dimension(2),
                window.kernelHeight(),
                window.paddingHeight(),
                window.strideHeight(),
                window.dilationHeight(),
                window.ceilMode(),
                constraints,
                subject + " height");
        Dimension outputWidth = windowDimension(
                shape.dimension(3),
                window.kernelWidth(),
                window.paddingWidth(),
                window.strideWidth(),
                window.dilationWidth(),
                window.ceilMode(),
                constraints,
                subject + " width");
        return Shape.ofDimensions(
                shape.dimension(0),
                DimensionExpressions.multiply(
                        DimensionExpressions.multiply(
                                shape.dimension(1), window.kernelHeight()),
                        window.kernelWidth()),
                DimensionExpressions.multiply(outputHeight, outputWidth));
    }

    /**
     * Derives canonical volumetric columns from one exact NCDHW Shape.
     *
     * @param shape non-null rank-five NCDHW Shape
     * @param window non-null validated volumetric window geometry
     * @param constraints non-null ordered destination for depth, height, and width obligations
     * @param subject non-null diagnostic prefix
     * @return rank-three canonical columns Shape
     * @throws IllegalArgumentException if static geometry does not fit
     * @throws ArithmeticException if checked geometry arithmetic overflows {@code long}
     */
    private static Shape unfold3dShape(
            Shape shape,
            Window3dAttrs window,
            List<CapturedGraphInference.ConstraintRequest> constraints,
            String subject) {
        Dimension depth = window3dDimension(
                shape.dimension(2), window.kernelDepth(), window.paddingDepth(),
                window.strideDepth(), window.dilationDepth(), window.ceilMode(), constraints,
                subject + " depth");
        Dimension height = window3dDimension(
                shape.dimension(3), window.kernelHeight(), window.paddingHeight(),
                window.strideHeight(), window.dilationHeight(), window.ceilMode(), constraints,
                subject + " height");
        Dimension width = window3dDimension(
                shape.dimension(4), window.kernelWidth(), window.paddingWidth(),
                window.strideWidth(), window.dilationWidth(), window.ceilMode(), constraints,
                subject + " width");
        Dimension columns = DimensionExpressions.multiply(
                DimensionExpressions.multiply(
                        DimensionExpressions.multiply(
                                shape.dimension(1), window.kernelDepth()),
                        window.kernelHeight()),
                window.kernelWidth());
        Dimension positions = DimensionExpressions.multiply(
                DimensionExpressions.multiply(depth, height), width);
        return Shape.ofDimensions(shape.dimension(0), columns, positions);
    }

    /**
     * Derives one checked three-dimensional window extent and records its numerator domain.
     *
     * @param dimension non-null source spatial Dimension
     * @param kernel positive kernel extent
     * @param padding non-negative symmetric padding per side
     * @param stride positive output stride
     * @param dilation positive kernel dilation
     * @param ceil whether to use literal ceiling division
     * @param constraints non-null ordered constraint destination
     * @param subject non-null semantic axis name
     * @return exact static or canonical symbolic output Dimension
     * @throws IllegalArgumentException if a static numerator is negative
     * @throws ArithmeticException if checked geometry arithmetic overflows {@code long}
     */
    private static Dimension window3dDimension(
            Dimension dimension,
            long kernel,
            long padding,
            long stride,
            long dilation,
            boolean ceil,
            List<CapturedGraphInference.ConstraintRequest> constraints,
            String subject) {
        long effective = Math.addExact(Math.multiplyExact(dilation, kernel - 1), 1);
        long doubledPadding = Math.multiplyExact(2, padding);
        Dimension numerator;
        if (dimension instanceof StaticDimension staticDimension) {
            long staticNumerator = Math.subtractExact(
                    Math.addExact(staticDimension.size(), doubledPadding), effective);
            if (staticNumerator < 0) {
                throw new IllegalArgumentException("window does not fit");
            }
            numerator = new StaticDimension(staticNumerator);
        } else {
            numerator = DimensionExpressions.addConstant(
                    dimension, Math.subtractExact(doubledPadding, effective));
        }
        constraints.add(new CapturedGraphInference.ConstraintRequest(
                subject + " domain", new DimensionAtLeast(numerator, 0)));
        if (numerator instanceof StaticDimension staticNumerator) {
            return new StaticDimension(Math.addExact(
                    staticNumerator.size() / stride
                            + (ceil && staticNumerator.size() % stride != 0 ? 1 : 0),
                    1));
        }
        return DimensionExpressions.addConstant(
                ceil
                        ? DimensionExpressions.ceilingDivide(numerator, stride)
                        : DimensionExpressions.floorDivide(numerator, stride),
                1);
    }

    private static Dimension windowDimension(
            Dimension dimension,
            long kernel,
            long padding,
            long stride,
            long dilation,
            boolean ceil,
            List<CapturedGraphInference.ConstraintRequest> constraints,
            String subject) {
        long effective = Math.addExact(Math.multiplyExact(dilation, kernel - 1), 1);
        long doubledPadding = Math.multiplyExact(2, padding);
        Dimension padded = DimensionExpressions.addConstant(dimension, doubledPadding);
        constraints.add(new CapturedGraphInference.ConstraintRequest(
                subject + " domain",
                FitsWithin.dimension(0, new StaticDimension(effective), padded)));
        if (dimension instanceof StaticDimension staticDimension) {
            long numerator = Math.subtractExact(
                    Math.addExact(staticDimension.size(), doubledPadding), effective);
            if (numerator < 0) {
                throw new IllegalArgumentException("window does not fit");
            }
            return new StaticDimension(Math.addExact(
                    numerator / stride + (ceil && numerator % stride != 0 ? 1 : 0), 1));
        }
        Dimension numerator =
                DimensionExpressions.addConstant(
                        dimension, Math.subtractExact(doubledPadding, effective));
        return DimensionExpressions.addConstant(
                ceil
                        ? DimensionExpressions.ceilingDivide(numerator, stride)
                        : DimensionExpressions.floorDivide(numerator, stride),
                1);
    }
    private static void axis(Shape s,int a){if(a<0||a>=s.rank())throw new IllegalArgumentException("axis out of range");}
    private static void sameType(TensorDescriptor a,TensorDescriptor b){if(a.dataType()!=b.dataType())throw new IllegalArgumentException("data type mismatch");}
}
