package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.MethodTypeDesc;
import java.lang.constant.ConstantDescs;

/**
 * Emits direct typed grouped NCDHW Conv3d complete-output-cell loops.
 *
 * <p>The generated body traverses input channels within the selected group, kernel depth,
 * height, and width in increasing logical order. It loads every weight contribution, represents an
 * out-of-range input coordinate as positive zero before ordinary multiplication, accumulates in
 * FLOAT64 or FLOAT32 as selected by the output type, and narrows BFLOAT16 only at the final store.
 * No Synaptik method is called by generated hot work.</p>
 */
public final class CpuConv3dEmitter {
    /** Creates a stateless direct-convolution emitter. */
    public CpuConv3dEmitter() { }

    /**
     * Emits one direct scalar entry over the primitive half-open output-cell range.
     *
     * @param code non-null Class-File method builder to mutate
     * @param specialization non-null exact typed carrier specialization
     * @param ir non-null canonical Conv3d identity
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if boundary, type, strategy, or identity facts disagree
     */
    public void emit(CodeBuilder code, CpuKernelSpecialization specialization, CpuKernelIr ir) {
        String identity = ir.familyIdentity();
        boolean bias = identity.startsWith("conv3d:bias=true:");
        int boundaries = specialization.carrierPattern().size();
        if (!identity.startsWith("conv3d:") || boundaries != (bias ? 4 : 3)
                || specialization.executionStrategy().compute()
                    != io.github.pho001.synaptik.backend.cpu.internal.prepare
                        .CpuPartitionPreparationPlan.ExecutionStrategy.Compute.SCALAR) {
            throw new IllegalArgumentException("Conv3d generated facts disagree");
        }
        DataType inputType = specialization.boundaryDataTypes().get(0);
        DataType weightType = specialization.boundaryDataTypes().get(1);
        DataType biasType = bias ? specialization.boundaryDataTypes().get(2) : null;
        DataType resultType = specialization.boundaryDataTypes().getLast();
        boolean denseArrays=specialization.carrierPattern().stream().allMatch(access->access
                !=CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)&&ir.values().stream()
                .allMatch(value->value.accessPlan().regime()==io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Regime.DENSE_LINEAR);
        int outputBoundary = boundaries - 1;
        int geometry = boundaries;
        int start = geometry + 1;
        int end = start + 2;
        int inputExtents = boundaries;
        int inputStrides = inputExtents + 5;
        int weightExtents = inputStrides + 5;
        int weightStrides = weightExtents + 5;
        int outputExtents = weightStrides + 5 + (bias ? 2 : 0);
        int outputStrides = outputExtents + 5;
        long strideD = CpuNormEmitter.longAfter(identity, ":strideD=");
        long strideH = CpuNormEmitter.longAfter(identity, ":strideH=");
        long strideW = CpuNormEmitter.longAfter(identity, ":strideW=");
        long padD = CpuNormEmitter.longAfter(identity, ":padD=");
        long padH = CpuNormEmitter.longAfter(identity, ":padH=");
        long padW = CpuNormEmitter.longAfter(identity, ":padW=");
        long dilationD = CpuNormEmitter.longAfter(identity, ":dilationD=");
        long dilationH = CpuNormEmitter.longAfter(identity, ":dilationH=");
        long dilationW = CpuNormEmitter.longAfter(identity, ":dilationW=");
        long groups = CpuNormEmitter.longAfter(identity, ":groups=");
        var carriers = new CpuCarrierEmitter(code);
        int inputBase=geometryLocal(code,geometry,0),weightBase=geometryLocal(code,geometry,1);
        int biasBase=bias?geometryLocal(code,geometry,2):-1;
        int outputBase=geometryLocal(code,geometry,outputBoundary);
        int inD=geometryLocal(code,geometry,inputExtents+2),inH=geometryLocal(code,geometry,inputExtents+3),inW=geometryLocal(code,geometry,inputExtents+4);
        int inSN=geometryLocal(code,geometry,inputStrides),inSC=geometryLocal(code,geometry,inputStrides+1),inSD=geometryLocal(code,geometry,inputStrides+2),inSH=geometryLocal(code,geometry,inputStrides+3),inSW=geometryLocal(code,geometry,inputStrides+4);
        int channelsPerGroup=geometryLocal(code,geometry,weightExtents+1),kernelD=geometryLocal(code,geometry,weightExtents+2),kernelH=geometryLocal(code,geometry,weightExtents+3),kernelW=geometryLocal(code,geometry,weightExtents+4);
        int weightSO=geometryLocal(code,geometry,weightStrides),weightSC=geometryLocal(code,geometry,weightStrides+1),weightSD=geometryLocal(code,geometry,weightStrides+2),weightSH=geometryLocal(code,geometry,weightStrides+3),weightSW=geometryLocal(code,geometry,weightStrides+4);
        int biasStride=bias?geometryLocal(code,geometry,weightStrides+6):-1;
        int outC=geometryLocal(code,geometry,outputExtents+1),outD=geometryLocal(code,geometry,outputExtents+2),outH=geometryLocal(code,geometry,outputExtents+3),outW=geometryLocal(code,geometry,outputExtents+4);
        int outSN=geometryLocal(code,geometry,outputStrides),outSC=geometryLocal(code,geometry,outputStrides+1),outSD=geometryLocal(code,geometry,outputStrides+2),outSH=geometryLocal(code,geometry,outputStrides+3),outSW=geometryLocal(code,geometry,outputStrides+4);

        int cell = code.allocateLocal(TypeKind.LONG);
        int remaining = code.allocateLocal(TypeKind.LONG);
        int n = code.allocateLocal(TypeKind.LONG);
        int oc = code.allocateLocal(TypeKind.LONG);
        int od = code.allocateLocal(TypeKind.LONG);
        int oh = code.allocateLocal(TypeKind.LONG);
        int ow = code.allocateLocal(TypeKind.LONG);
        int group = code.allocateLocal(TypeKind.LONG);
        int channel = code.allocateLocal(TypeKind.LONG);
        int kd = code.allocateLocal(TypeKind.LONG);
        int kh = code.allocateLocal(TypeKind.LONG);
        int kw = code.allocateLocal(TypeKind.LONG);
        int id = code.allocateLocal(TypeKind.LONG);
        int ih = code.allocateLocal(TypeKind.LONG);
        int iw = code.allocateLocal(TypeKind.LONG);
        int inputAddress = code.allocateLocal(denseArrays?TypeKind.INT:TypeKind.LONG);
        int weightAddress = code.allocateLocal(denseArrays?TypeKind.INT:TypeKind.LONG);
        int outputAddress = code.allocateLocal(denseArrays?TypeKind.INT:TypeKind.LONG);
        int representedInput = representedLocal(code, inputType);
        int representedWeight = representedLocal(code, weightType);
        int representedBias = bias ? representedLocal(code, biasType) : -1;
        boolean binary64 = resultType == DataType.FLOAT64;
        TypeKind accumulationKind = binary64 ? TypeKind.DOUBLE : TypeKind.FLOAT;
        int inputValue = code.allocateLocal(accumulationKind);
        int weightValue = code.allocateLocal(accumulationKind);
        int accumulator = code.allocateLocal(accumulationKind);

        code.lload(start).lstore(cell);
        var cells = code.newLabel();
        var done = code.newLabel();
        code.labelBinding(cells).lload(cell).lload(end).lcmp().branch(Opcode.IFGE, done);
        code.lload(cell).lstore(remaining);
        decodeCoordinate(code, outW, remaining, ow);
        decodeCoordinate(code, outH, remaining, oh);
        decodeCoordinate(code, outD, remaining, od);
        decodeCoordinate(code, outC, remaining, oc);
        code.lload(remaining).lstore(n);
        code.lload(oc);
        code.lload(outC).loadConstant(groups).ldiv().ldiv().lstore(group);

        if (bias) {
            int biasAddress = code.allocateLocal(denseArrays?TypeKind.INT:TypeKind.LONG);
            if(denseArrays)code.lload(biasBase).l2i().lload(oc).l2i().iadd().istore(biasAddress);
            else code.lload(biasBase).lload(oc).lload(biasStride).lmul().ladd().lstore(biasAddress);
            load(code, carriers, specialization, biasType, resultType, 2, biasAddress,
                    representedBias, accumulator,denseArrays);
        } else {
            if (binary64) code.loadConstant(0.0).dstore(accumulator);
            else code.loadConstant(0.0f).fstore(accumulator);
        }

        code.loadConstant(0L).lstore(channel);
        var channels = code.newLabel(); var channelsDone = code.newLabel();
        code.labelBinding(channels).lload(channel);
        code.lload(channelsPerGroup).lcmp()
                .branch(Opcode.IFGE, channelsDone);
        code.loadConstant(0L).lstore(kd);
        var depths = code.newLabel(); var depthsDone = code.newLabel();
        code.labelBinding(depths).lload(kd);
        code.lload(kernelD).lcmp()
                .branch(Opcode.IFGE, depthsDone);
        code.lload(od).loadConstant(strideD).lmul().loadConstant(padD).lsub()
                .lload(kd).loadConstant(dilationD).lmul().ladd().lstore(id);
        code.loadConstant(0L).lstore(kh);
        var heights = code.newLabel(); var heightsDone = code.newLabel();
        code.labelBinding(heights).lload(kh);
        code.lload(kernelH).lcmp()
                .branch(Opcode.IFGE, heightsDone);
        code.lload(oh).loadConstant(strideH).lmul().loadConstant(padH).lsub()
                .lload(kh).loadConstant(dilationH).lmul().ladd().lstore(ih);
        code.loadConstant(0L).lstore(kw);
        var widths = code.newLabel(); var widthsDone = code.newLabel();
        code.labelBinding(widths).lload(kw);
        code.lload(kernelW).lcmp()
                .branch(Opcode.IFGE, widthsDone);
        code.lload(ow).loadConstant(strideW).lmul().loadConstant(padW).lsub()
                .lload(kw).loadConstant(dilationW).lmul().ladd().lstore(iw);

        if(denseArrays)weightAddressDense(code,weightBase,channelsPerGroup,kernelD,kernelH,kernelW,channel,kd,kh,kw,oc,weightAddress);
        else weightAddress(code,weightBase,weightSO,weightSC,weightSD,weightSH,weightSW,channel,kd,kh,kw,oc,weightAddress);
        load(code, carriers, specialization, weightType, resultType, 1, weightAddress,
                representedWeight, weightValue,denseArrays);
        var padded = code.newLabel(); var inputReady = code.newLabel();
        code.lload(id).loadConstant(0L).lcmp().branch(Opcode.IFLT, padded);
        code.lload(ih).loadConstant(0L).lcmp().branch(Opcode.IFLT, padded);
        code.lload(iw).loadConstant(0L).lcmp().branch(Opcode.IFLT, padded);
        code.lload(id).lload(inD).lcmp()
                .branch(Opcode.IFGE, padded);
        code.lload(ih).lload(inH).lcmp()
                .branch(Opcode.IFGE, padded);
        code.lload(iw).lload(inW).lcmp()
                .branch(Opcode.IFGE, padded);
        if(denseArrays)inputAddressDense(code,inputBase,channelsPerGroup,inD,inH,inW,groups,n,group,channel,id,ih,iw,inputAddress);
        else inputAddress(code,inputBase,inSN,inSC,inSD,inSH,inSW,n,group,channel,id,ih,iw,inputAddress,channelsPerGroup);
        load(code, carriers, specialization, inputType, resultType, 0, inputAddress,
                representedInput, inputValue,denseArrays);
        code.branch(Opcode.GOTO, inputReady).labelBinding(padded);
        if (binary64) code.loadConstant(0.0); else code.loadConstant(0.0f);
        if (binary64) code.dstore(inputValue); else code.fstore(inputValue);
        code.labelBinding(inputReady);
        accumulate(code, resultType, accumulator, inputValue, weightValue);
        code.lload(kw).loadConstant(1L).ladd().lstore(kw).branch(Opcode.GOTO, widths)
                .labelBinding(widthsDone);
        code.lload(kh).loadConstant(1L).ladd().lstore(kh).branch(Opcode.GOTO, heights)
                .labelBinding(heightsDone);
        code.lload(kd).loadConstant(1L).ladd().lstore(kd).branch(Opcode.GOTO, depths)
                .labelBinding(depthsDone);
        code.lload(channel).loadConstant(1L).ladd().lstore(channel)
                .branch(Opcode.GOTO, channels).labelBinding(channelsDone);

        if(denseArrays)outputAddressDense(code,outputBase,outC,outD,outH,outW,n,oc,od,oh,ow,outputAddress);
        else address(code,outputBase,outSN,outSC,outSD,outSH,outSW,n,oc,od,oh,ow,outputAddress);
        int stored = accumulator;
        if (!binary64) {
            stored = code.allocateLocal(TypeKind.DOUBLE);
            code.fload(accumulator).f2d().dstore(stored);
        }
        CpuNormEmitter.emitStore(code, carriers, specialization, resultType, outputBoundary,
                outputAddress, stored, denseArrays, true);
        code.lload(cell).loadConstant(1L).ladd().lstore(cell).branch(Opcode.GOTO, cells)
                .labelBinding(done);
    }

    private static int representedLocal(CodeBuilder code, DataType type) {
        return code.allocateLocal(type == DataType.FLOAT64 ? TypeKind.DOUBLE
                : type == DataType.FLOAT32 ? TypeKind.FLOAT : TypeKind.INT);
    }

    private static void load(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType type, DataType resultType,
            int boundary, int address,
            int represented, int decoded) {
        load(code,carriers,specialization,type,resultType,boundary,address,represented,decoded,false);
    }

    private static void load(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType type, DataType resultType,
            int boundary, int address, int represented, int decoded,boolean intAddress) {
        carriers.loadFrozen(type, specialization.carrierPattern().get(boundary), boundary,
                address, intAddress);
        if (type == resultType && type == DataType.FLOAT64) {
            code.dstore(decoded);
            return;
        }
        if (type == resultType && type == DataType.FLOAT32) {
            code.fstore(decoded);
            return;
        }
        if (type == DataType.FLOAT64) code.dstore(represented);
        else if (type == DataType.FLOAT32) code.fstore(represented);
        else code.istore(represented);
        if (resultType == DataType.FLOAT64) {
            CpuNormEmitter.decodeRepresented(code, type, represented, decoded);
        } else if (type == DataType.FLOAT32) {
            code.fload(represented).fstore(decoded);
        } else {
            code.iload(represented).loadConstant(16).ishl().invokestatic(
                    ClassDesc.of(Float.class.getName()), "intBitsToFloat",
                    MethodTypeDesc.of(ConstantDescs.CD_float, ConstantDescs.CD_int))
                    .fstore(decoded);
        }
    }

    private static void accumulate(CodeBuilder code, DataType result, int accumulator,
            int input, int weight) {
        if (result == DataType.FLOAT64) {
            code.dload(accumulator).dload(input).dload(weight).dmul().dadd().dstore(accumulator);
        } else {
            code.fload(accumulator).fload(input).fload(weight).fmul().fadd().fstore(accumulator);
        }
    }

    private static void decodeCoordinate(CodeBuilder code, int extent,
            int remaining, int coordinate) {
        code.lload(remaining).lload(extent).lrem()
                .lstore(coordinate);
        code.lload(remaining).lload(extent).ldiv()
                .lstore(remaining);
    }

    private static void weightAddress(CodeBuilder code,int base,int so,int sc,int sd,int sh,int sw,int channel,
            int kd, int kh, int kw, int oc, int target) {
        code.lload(base).lload(oc).lload(so).lmul().ladd()
                .lload(channel).lload(sc).lmul().ladd()
                .lload(kd).lload(sd).lmul().ladd()
                .lload(kh).lload(sh).lmul().ladd()
                .lload(kw).lload(sw).lmul().ladd()
                .lstore(target);
    }

    private static void inputAddress(CodeBuilder code,int base,int sn,int sc,int sd,int sh,int sw,int n,
            int group, int channel, int id, int ih, int iw, int target, int channelsPerGroup) {
        code.lload(base).lload(n).lload(sn).lmul().ladd()
                .lload(group).lload(channelsPerGroup)
                .lmul().lload(channel).ladd();
        code.lload(sc).lmul().ladd()
                .lload(id).lload(sd).lmul().ladd()
                .lload(ih).lload(sh).lmul().ladd()
                .lload(iw).lload(sw).lmul().ladd()
                .lstore(target);
    }

    private static void address(CodeBuilder code,int base,int sn,int sc,int sd,int sh,int sw,
            int n, int c, int d, int h, int w, int target) {
        code.lload(base).lload(n).lload(sn).lmul().ladd().lload(c).lload(sc).lmul().ladd()
                .lload(d).lload(sd).lmul().ladd().lload(h).lload(sh).lmul().ladd()
                .lload(w).lload(sw).lmul().ladd().lstore(target);
    }

    private static void weightAddressDense(CodeBuilder code,int base,int channels,int depth,int height,int width,int channel,int kd,int kh,int kw,int oc,int target){code.lload(base).l2i().lload(oc).l2i().lload(channels).l2i().imul().lload(channel).l2i().iadd().lload(depth).l2i().imul().lload(kd).l2i().iadd().lload(height).l2i().imul().lload(kh).l2i().iadd().lload(width).l2i().imul().lload(kw).l2i().iadd().iadd().istore(target);}
    private static void inputAddressDense(CodeBuilder code,int base,int channelsPerGroup,int depth,int height,int width,long groups,int n,int group,int channel,int id,int ih,int iw,int target){code.lload(base).l2i().lload(n).l2i().lload(channelsPerGroup).l2i().loadConstant(Math.toIntExact(groups)).imul().imul().lload(group).l2i().lload(channelsPerGroup).l2i().imul().lload(channel).l2i().iadd().iadd().lload(depth).l2i().imul().lload(id).l2i().iadd().lload(height).l2i().imul().lload(ih).l2i().iadd().lload(width).l2i().imul().lload(iw).l2i().iadd().iadd().istore(target);}
    private static void outputAddressDense(CodeBuilder code,int base,int channels,int depth,int height,int width,int n,int c,int d,int h,int w,int target){code.lload(base).l2i().lload(n).l2i().lload(channels).l2i().imul().lload(c).l2i().iadd().lload(depth).l2i().imul().lload(d).l2i().iadd().lload(height).l2i().imul().lload(h).l2i().iadd().lload(width).l2i().imul().lload(w).l2i().iadd().iadd().istore(target);}

    private static int geometryLocal(CodeBuilder code,int geometry,int index){int local=code.allocateLocal(TypeKind.LONG);CpuNormEmitter.geometry(code,geometry,index).lstore(local);return local;}
}
