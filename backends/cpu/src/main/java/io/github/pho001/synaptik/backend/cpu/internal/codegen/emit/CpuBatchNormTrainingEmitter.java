package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;

/**
 * Emits the direct generated body for one first-class batch-normalization training occurrence.
 *
 * <p>The generated outer range owns complete channels. For each channel it computes an exact
 * represented-value mean, corrected biased and unbiased variances, four channel-statistic
 * outputs, and the normalized affine output in three complete non-channel traversals. The body
 * reuses the invocation's exact-state slice and retains no state after invocation.</p>
 */
public final class CpuBatchNormTrainingEmitter {
    private static final ClassDesc MATH = ClassDesc.of(Math.class.getName());
    /** Creates a stateless generation-time emitter with no retained invocation resources. */
    public CpuBatchNormTrainingEmitter() { }

    /**
     * Appends one complete-channel scalar body to the method under construction.
     *
     * @param code mutable Class-File method builder that receives the generated instructions;
     *     must not be {@code null}
     * @param specialization immutable carrier, type, and scratch signature for the generated
     *     entry; must describe the boundaries encoded by {@code ir}
     * @param ir instruction-free batch-normalization training kernel identity; must encode five
     *     semantic inputs, five outputs, three passes, and compatible exact-state geometry
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if the IR family or specialization does not describe the
     *     supported batch-normalization training entry
     */
    public void emit(CodeBuilder code, CpuKernelSpecialization specialization, CpuKernelIr ir) {
        String identity = ir.familyIdentity();
        if (!identity.startsWith("batch-normalization-training:"))
            throw new IllegalArgumentException("training emitter requires training identity");
        int[] map = CpuLayerNormEmitter.map(identity);
        int boundaries = specialization.carrierPattern().size();
        int unique = boundaries - 5;
        boolean empty = Long.parseLong(CpuLayerNormEmitter.field(identity, ":slice=")) == 0;
        if (map.length != 5 || unique < 1 || specialization.scratchParameter() == empty
                || specialization.boundaryDataTypes().size() != boundaries)
            throw new IllegalArgumentException("training specialization disagrees");
        if (empty) return;
        DataType resultType = specialization.boundaryDataTypes().getLast();
        DataType exactType = resultType == DataType.FLOAT64 ? DataType.FLOAT64 : DataType.FLOAT32;
        int scratch = boundaries, geometry = boundaries + 1, start = geometry + 1, end = start + 2;
        int rank = Integer.parseInt(CpuLayerNormEmitter.field(identity, ":rank="));
        int axis = Integer.parseInt(CpuLayerNormEmitter.field(identity, ":axis="));
        int[] ranks = new int[boundaries], extents = new int[boundaries], strides = new int[boundaries];
        int cursor = 11 + boundaries;
        for (int b = 0; b < boundaries; b++) { ranks[b] = ir.values().get(b).accessPlan().iterationRank();
            extents[b] = cursor + 1; strides[b] = extents[b] + ranks[b]; cursor += 1 + 2 * ranks[b]; }
        var carriers = new CpuCarrierEmitter(code);
        int channel=code.allocateLocal(TypeKind.LONG), ordinal=code.allocateLocal(TypeKind.LONG);
        int remaining=code.allocateLocal(TypeKind.LONG), coordinate=code.allocateLocal(TypeKind.LONG);
        int address=code.allocateLocal(TypeKind.LONG);
        int inputRep=CpuLayerNormEmitter.represented(code,specialization.boundaryDataTypes().get(map[0]));
        int input=code.allocateLocal(TypeKind.DOUBLE), meanRep=code.allocateLocal(exactType==DataType.FLOAT64?TypeKind.DOUBLE:TypeKind.FLOAT);
        int mean=code.allocateLocal(TypeKind.DOUBLE), deviation=code.allocateLocal(TypeKind.DOUBLE);
        int devSum=code.allocateLocal(TypeKind.DOUBLE),devComp=code.allocateLocal(TypeKind.DOUBLE);
        int square=code.allocateLocal(TypeKind.DOUBLE),sqSum=code.allocateLocal(TypeKind.DOUBLE),sqComp=code.allocateLocal(TypeKind.DOUBLE);
        int temporary=code.allocateLocal(TypeKind.DOUBLE), numerator=code.allocateLocal(TypeKind.DOUBLE);
        int n=code.allocateLocal(TypeKind.DOUBLE),nm1=code.allocateLocal(TypeKind.DOUBLE);
        int biased=code.allocateLocal(TypeKind.DOUBLE),unbiased=code.allocateLocal(TypeKind.DOUBLE);
        int epsilon=code.allocateLocal(TypeKind.DOUBLE),momentum=code.allocateLocal(TypeKind.DOUBLE),one=code.allocateLocal(TypeKind.DOUBLE);
        int oneMinus=code.allocateLocal(TypeKind.DOUBLE),radicand=code.allocateLocal(TypeKind.DOUBLE),root=code.allocateLocal(TypeKind.DOUBLE),saved=code.allocateLocal(TypeKind.DOUBLE);
        int scale=code.allocateLocal(TypeKind.DOUBLE),bias=code.allocateLocal(TypeKind.DOUBLE),oldMean=code.allocateLocal(TypeKind.DOUBLE),oldVar=code.allocateLocal(TypeKind.DOUBLE);
        int left=code.allocateLocal(TypeKind.DOUBLE),right=code.allocateLocal(TypeKind.DOUBLE),nextMean=code.allocateLocal(TypeKind.DOUBLE),nextVar=code.allocateLocal(TypeKind.DOUBLE),result=code.allocateLocal(TypeKind.DOUBLE);
        int[] vectorRep=new int[4]; for(int p=1;p<5;p++)vectorRep[p-1]=CpuLayerNormEmitter.represented(code,specialization.boundaryDataTypes().get(map[p]));
        code.loadConstant(CpuLayerNormEmitter.epsilon(identity,resultType)).dstore(epsilon);
        code.loadConstant(scalar(identity,":momentum=",resultType)).dstore(momentum);
        code.loadConstant(1.0).dstore(one); code.loadConstant((double)Long.parseLong(CpuLayerNormEmitter.field(identity,":domain="))).dstore(n);
        code.dload(n).loadConstant(1.0).dsub().dstore(nm1);
        CpuExactSumEmitter exact=new CpuExactSumEmitter(code,exactType,true,true,scratch,geometry,Integer.parseInt(CpuLayerNormEmitter.field(identity,":limbs=")));
        code.lload(start).lstore(channel); var channels=code.newLabel(); var done=code.newLabel();
        code.labelBinding(channels).lload(channel).lload(end).lcmp().branch(Opcode.IFGE,done);
        exact.emitReset();
        loop(code,geometry,rank,axis,extents[map[0]],strides[map[0]],11+map[0],channel,ordinal,remaining,coordinate,address,()->{
            CpuLayerNormEmitter.load(code,carriers,specialization,map[0],inputRep,input,address);
            if(exactType==DataType.FLOAT64)code.dload(input).dstore(meanRep);else code.dload(input).d2f().fstore(meanRep);
            exact.emitFactor(meanRep);
        });
        exact.emitFinish(meanRep); CpuNormEmitter.decodeRepresented(code,exactType,meanRep,mean);
        code.loadConstant(0.0).dstore(devSum).loadConstant(0.0).dstore(devComp)
                .loadConstant(0.0).dstore(sqSum).loadConstant(0.0).dstore(sqComp);
        loop(code,geometry,rank,axis,extents[map[0]],strides[map[0]],11+map[0],channel,ordinal,remaining,coordinate,address,()->{
            CpuLayerNormEmitter.load(code,carriers,specialization,map[0],inputRep,input,address);
            CpuLayerNormEmitter.arithmetic(code,resultType,input,mean,Opcode.DSUB,deviation);
            code.dload(deviation).dstore(temporary);CpuNormEmitter.kahan(code,temporary,devSum,devComp,square);
            code.dload(deviation).dload(deviation).dmul().dstore(square);CpuNormEmitter.kahan(code,square,sqSum,sqComp,temporary);
        });
        code.dload(devSum).dload(devSum).dmul().dload(n).ddiv().dstore(temporary);
        code.dload(sqSum).dload(temporary).dsub().dstore(numerator);
        var nonnegative=code.newLabel();code.dload(numerator).loadConstant(0.0).dcmpg().branch(Opcode.IFGE,nonnegative)
                .loadConstant(0.0).dstore(numerator).labelBinding(nonnegative);
        CpuLayerNormEmitter.arithmetic(code,resultType,numerator,n,Opcode.DDIV,biased);
        CpuLayerNormEmitter.arithmetic(code,resultType,numerator,nm1,Opcode.DDIV,unbiased);
        CpuLayerNormEmitter.arithmetic(code,resultType,biased,epsilon,Opcode.DADD,radicand);
        code.dload(radicand).invokestatic(MATH,"sqrt",CpuNormEmitter.doubleUnary()).dstore(root);
        CpuLayerNormEmitter.arithmetic(code,resultType,one,root,Opcode.DDIV,saved);
        loadVector(code,carriers,specialization,map[1],vectorRep[0],scale,geometry,strides[map[1]],channel,address);
        loadVector(code,carriers,specialization,map[2],vectorRep[1],bias,geometry,strides[map[2]],channel,address);
        loadVector(code,carriers,specialization,map[3],vectorRep[2],oldMean,geometry,strides[map[3]],channel,address);
        loadVector(code,carriers,specialization,map[4],vectorRep[3],oldVar,geometry,strides[map[4]],channel,address);
        CpuLayerNormEmitter.arithmetic(code,resultType,one,momentum,Opcode.DSUB,oneMinus);
        CpuLayerNormEmitter.arithmetic(code,resultType,oneMinus,oldMean,Opcode.DMUL,left);
        CpuLayerNormEmitter.arithmetic(code,resultType,momentum,mean,Opcode.DMUL,right);
        CpuLayerNormEmitter.arithmetic(code,resultType,left,right,Opcode.DADD,nextMean);
        CpuLayerNormEmitter.arithmetic(code,resultType,oneMinus,oldVar,Opcode.DMUL,left);
        CpuLayerNormEmitter.arithmetic(code,resultType,momentum,unbiased,Opcode.DMUL,right);
        CpuLayerNormEmitter.arithmetic(code,resultType,left,right,Opcode.DADD,nextVar);
        int[] stats={nextMean,nextVar,mean,saved};
        for(int slot=1;slot<5;slot++){int b=unique+slot;vectorAddress(code,geometry,11+b,strides[b],channel,address);
            CpuNormEmitter.emitStore(code,carriers,specialization,resultType,b,address,stats[slot-1],false,true);}
        int out=unique;
        loopWithOutput(code,geometry,rank,axis,extents[map[0]],strides[map[0]],11+map[0],extents[out],strides[out],11+out,
                channel,ordinal,remaining,coordinate,address,code.allocateLocal(TypeKind.LONG),()->{
            CpuLayerNormEmitter.load(code,carriers,specialization,map[0],inputRep,input,address);
            CpuLayerNormEmitter.arithmetic(code,resultType,input,mean,Opcode.DSUB,result);
            CpuLayerNormEmitter.arithmetic(code,resultType,result,saved,Opcode.DMUL,result);
            CpuLayerNormEmitter.arithmetic(code,resultType,result,scale,Opcode.DMUL,result);
            CpuLayerNormEmitter.arithmetic(code,resultType,result,bias,Opcode.DADD,result);
        },carriers,specialization,resultType,out,result);
        code.lload(channel).loadConstant(1L).ladd().lstore(channel).branch(Opcode.GOTO,channels).labelBinding(done);
    }

    private static void loadVector(CodeBuilder c,CpuCarrierEmitter carriers,CpuKernelSpecialization s,int b,int rep,int value,int g,int stride,int channel,int address){
        vectorAddress(c,g,11+b,stride,channel,address);CpuLayerNormEmitter.load(c,carriers,s,b,rep,value,address);}
    private static void vectorAddress(CodeBuilder c,int g,int base,int stride,int channel,int address){CpuNormEmitter.geometry(c,g,base).lload(channel);CpuNormEmitter.geometry(c,g,stride).lmul().ladd().lstore(address);}
    private static void loop(CodeBuilder c,int g,int rank,int axis,int ext,int stride,int base,int channel,int ordinal,int remaining,int coordinate,int address,Runnable body){
        int[] coordinates=new int[rank];for(int a=0;a<rank;a++)if(a!=axis){coordinates[a]=c.allocateLocal(TypeKind.LONG);c.loadConstant(0L).lstore(coordinates[a]);}
        CpuNormEmitter.geometry(c,g,base).lload(channel);CpuNormEmitter.geometry(c,g,stride+axis).lmul().ladd().lstore(address);
        c.loadConstant(0L).lstore(ordinal); var loop=c.newLabel(); var done=c.newLabel(); c.labelBinding(loop).lload(ordinal);CpuNormEmitter.geometry(c,g,7).lcmp().branch(Opcode.IFGE,done);
        body.run();var advanced=c.newLabel();for(int a=rank-1;a>=0;a--){if(a==axis)continue;var wrapped=c.newLabel();c.lload(coordinates[a]).loadConstant(1L).ladd().lstore(coordinates[a]);c.lload(coordinates[a]);CpuNormEmitter.geometry(c,g,ext+a).lcmp().branch(Opcode.IFGE,wrapped);c.lload(address);CpuNormEmitter.geometry(c,g,stride+a).ladd().lstore(address);c.branch(Opcode.GOTO,advanced).labelBinding(wrapped);c.loadConstant(0L).lstore(coordinates[a]);c.lload(address);CpuNormEmitter.geometry(c,g,ext+a).loadConstant(1L).lsub();CpuNormEmitter.geometry(c,g,stride+a).lmul().lsub().lstore(address);}c.labelBinding(advanced);
        c.lload(ordinal).loadConstant(1L).ladd().lstore(ordinal).branch(Opcode.GOTO,loop).labelBinding(done);}
    private static void loopWithOutput(CodeBuilder c,int g,int rank,int axis,int inExt,int inStride,int inBase,int outExt,int outStride,int outBase,int channel,int ordinal,int remaining,int coordinate,int address,int outAddress,Runnable body,CpuCarrierEmitter carriers,CpuKernelSpecialization s,DataType type,int out,int result){
        int[] coordinates=new int[rank];for(int a=0;a<rank;a++)if(a!=axis){coordinates[a]=c.allocateLocal(TypeKind.LONG);c.loadConstant(0L).lstore(coordinates[a]);}
        CpuNormEmitter.geometry(c,g,inBase).lload(channel);CpuNormEmitter.geometry(c,g,inStride+axis).lmul().ladd().lstore(address);CpuNormEmitter.geometry(c,g,outBase).lload(channel);CpuNormEmitter.geometry(c,g,outStride+axis).lmul().ladd().lstore(outAddress);
        c.loadConstant(0L).lstore(ordinal); var loop=c.newLabel(); var done=c.newLabel(); c.labelBinding(loop).lload(ordinal);CpuNormEmitter.geometry(c,g,7).lcmp().branch(Opcode.IFGE,done);
        body.run();CpuNormEmitter.emitStore(c,carriers,s,type,out,outAddress,result,false,true);var advanced=c.newLabel();for(int a=rank-1;a>=0;a--){if(a==axis)continue;var wrapped=c.newLabel();c.lload(coordinates[a]).loadConstant(1L).ladd().lstore(coordinates[a]);c.lload(coordinates[a]);CpuNormEmitter.geometry(c,g,inExt+a).lcmp().branch(Opcode.IFGE,wrapped);c.lload(address);CpuNormEmitter.geometry(c,g,inStride+a).ladd().lstore(address);c.lload(outAddress);CpuNormEmitter.geometry(c,g,outStride+a).ladd().lstore(outAddress);c.branch(Opcode.GOTO,advanced).labelBinding(wrapped);c.loadConstant(0L).lstore(coordinates[a]);c.lload(address);CpuNormEmitter.geometry(c,g,inExt+a).loadConstant(1L).lsub();CpuNormEmitter.geometry(c,g,inStride+a).lmul().lsub().lstore(address);c.lload(outAddress);CpuNormEmitter.geometry(c,g,outExt+a).loadConstant(1L).lsub();CpuNormEmitter.geometry(c,g,outStride+a).lmul().lsub().lstore(outAddress);}c.labelBinding(advanced);
        c.lload(ordinal).loadConstant(1L).ladd().lstore(ordinal).branch(Opcode.GOTO,loop).labelBinding(done);}
    private static void tensorAddress(CodeBuilder c,int g,int rank,int axis,int ext,int stride,int base,int channel,int ordinal,int remaining,int coordinate,int address){
        CpuNormEmitter.geometry(c,g,base).lload(channel);CpuNormEmitter.geometry(c,g,stride+axis).lmul().ladd().lstore(address);c.lload(ordinal).lstore(remaining);
        for(int a=rank-1;a>=0;a--){if(a==axis)continue;c.lload(remaining);CpuNormEmitter.geometry(c,g,ext+a).lrem().lstore(coordinate);c.lload(remaining);CpuNormEmitter.geometry(c,g,ext+a).ldiv().lstore(remaining);c.lload(address).lload(coordinate);CpuNormEmitter.geometry(c,g,stride+a).lmul().ladd().lstore(address);}}
    private static double scalar(String identity,String marker,DataType type){long bits=Long.parseUnsignedLong(CpuLayerNormEmitter.field(identity,marker));return switch(type){case FLOAT64->Double.longBitsToDouble(bits);case FLOAT32->Float.intBitsToFloat((int)bits);case BFLOAT16->Float.intBitsToFloat((int)bits<<16);default->throw new IllegalArgumentException("type");};}
}
