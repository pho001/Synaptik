package io.github.pho001.synaptik.testing.conformance;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.metal.MetalCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.*;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutable;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLowering.Facts;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.*;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.operation.scan.*;
import io.github.pho001.synaptik.model.shape.*;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Model-built independent provider, numeric-resource and generated-execution witnesses. */
class CpuSymbolicScanConformanceTest {
    @Test void symbolicLengthDtypesHaveSeparateControlAndActualSizedGeneratedResources() throws Throwable {
        for (DataType type : List.of(DataType.FLOAT32,DataType.FLOAT64,DataType.BFLOAT16,DataType.FLOAT16,
                DataType.FLOAT8_E4M3,DataType.FLOAT8_E5M2)) {
            var q = symbolicLengthQuery(type); assertTrue(new CpuCapabilityProvider().supports(q)); assertFalse(new MetalCapabilityProvider().supports(q));
            var analysis = new CpuPartitionPreparer().analyze(CpuGeneralScanConformanceTest.context(q));
            var unit = analysis.plan().units().getFirst(); var facts = unit.runtimeFacts().generalScan().orElseThrow();
            var spec = unit.portablePlan().specialization(); var generator = new CpuClassFileKernelGenerator();
            byte[] bytes = generator.generateClassBytes(spec,unit.portablePlan().kernelIr());
            var entry = generator.defineClassBytes(spec, unit.portablePlan().kernelIr(), bytes).entryPoint();
            for (int batch : new int[]{32,7,1,32,0,3,4}) try (var arena = Arena.ofConfined()) {
                var args = new ArrayList<MemorySegment>();
                for (int p = 0; p < facts.batchBindings().size(); p++) args.add(arena.allocate(
                        facts.batchBindings().get(p).bind(batch).referencedElementSpan()*facts.ir().boundaryType(p).byteWidth(),facts.ir().boundaryType(p).byteWidth()));
                args.add(arena.allocate(facts.geometry().workspaceSize().byteSize(batch),8));
                var lengths = args.get(facts.ir().lengthRead()); assertEquals(batch*8L,lengths.byteSize());
                for (int row = 0; row < batch; row++) lengths.setAtIndex(ValueLayout.JAVA_LONG,row,row%5);
                for (int p = 0; p < 3; p++) for (long i = 0; i < args.get(p).byteSize()/type.byteWidth(); i++) put(type,args.get(p),i,p == 2 ? .25 : .5);
                long[] pack = facts.pack(batch); assertEquals(batch,pack[pack.length-1]);
                var bound = bindDirect(entry, facts, spec, args, batch);
                bound.execute();
                bound.execute(); // The same helper packs and scratch must reset for a repeated entry call.
                int reads = facts.ir().scalarReads().size();
                for (int row = 0; row < batch; row++) for (int col = 0; col < 3; col++) {
                    assertEquals(.5+(row%5)*.5,get(type,args.get(reads),row*3+col));
                    for (int t = 0; t < 4; t++) assertEquals(t < row%5 ? .5 : 0,get(type,args.get(reads+1),(t*batch+row)*3+col));
                }
            }
        }
    }
    static OperationCapabilityQuery symbolicLengthQuery(DataType type) {
        var q = query(4,false,type); var a = (ScanAttrs)q.operation().attrs(); var inputs = new ArrayList<>(q.inputs());
        inputs.add(descriptor(DataType.INT64,Shape.ofDimensions(new DynamicDimension("B"))));
        return new OperationCapabilityQuery(new Operation(ScanKind.SCAN,new ScanAttrs(a.timeCount(), a.carryCount(), a.sequenceCount(), a.invariantCount(), a.emissionCount(), a.residualCount(), a.reverse(), true, 1, List.of(OptionalInt.of(0)))),inputs,q.outputs(),q.scanBody(),q.scanActivityPlan());
    }
    private static void put(DataType type,MemorySegment s,long i,double value) {
        if (LowPrecisionValues.TYPES.contains(type)) LowPrecisionValues.put(type,s,i,(float)value);
        else if (type == DataType.FLOAT64) s.setAtIndex(ValueLayout.JAVA_DOUBLE,i,value);
        else s.setAtIndex(ValueLayout.JAVA_FLOAT,i,(float)value);
    }
    private static double get(DataType type,MemorySegment s,long i) {
        return LowPrecisionValues.TYPES.contains(type) ? LowPrecisionValues.decode(type,LowPrecisionValues.raw(type,s,i))
                : type == DataType.FLOAT64 ? s.getAtIndex(ValueLayout.JAVA_DOUBLE,i) : s.getAtIndex(ValueLayout.JAVA_FLOAT,i);
    }

    @Test void qualifiedFloat32UsesActualRowBytesAndIndependentSharedParameterShape() throws Throwable {
        for (boolean reverse : List.of(false,true)) for (int time : new int[] {0,3}) {
            var q = query(time,reverse,DataType.FLOAT32); var cpu = new CpuCapabilityProvider();
            assertTrue(cpu.supports(q)); assertFalse(new MetalCapabilityProvider().supports(q));
            assertTrue(cpu.supports(query(time, reverse, DataType.FLOAT64)));
            var analysis = new CpuPartitionPreparer().analyze(CpuGeneralScanConformanceTest.context(q));
            var unit = analysis.plan().units().getFirst(); var facts = unit.runtimeFacts().generalScan().orElseThrow();
            assertTrue(unit.runtimeFacts().workspaceDeclaration().isPresent(),"zero bytes never erase structural workspace membership");
            var generator = new CpuClassFileKernelGenerator(); var specialization = unit.portablePlan().specialization();
            var artifact = generator.defineClassBytes(specialization, unit.portablePlan().kernelIr(), generator.generateClassBytes(specialization, unit.portablePlan().kernelIr()));
            for (int batch : new int[] {64,40,1,64,0,3,4}) try (var arena = Arena.ofConfined()) {
                var args = new ArrayList<MemorySegment>();
                for (var binding : facts.batchBindings()) args.add(arena.allocate(binding.bind(batch).referencedElementSpan()*4,4));
                args.add(arena.allocate(facts.geometry().workspaceSize().byteSize(batch),8));
                assertEquals(48,args.get(2).byteSize(),"shared parameter backing is independent of batch");
                for (int p = 0; p < 3; p++) for (int i = 0; i < args.get(p).byteSize()/4; i++)
                    args.get(p).setAtIndex(ValueLayout.JAVA_FLOAT,i,(i%7-3)*(p+1)*.03125f);
                float[] expected = args.get(0).toArray(ValueLayout.JAVA_FLOAT); float[] history = new float[time*batch*3];
                for (int step = 0; step < time; step++) {
                    int t = reverse ? time-1-step : step;
                    for (int row = 0; row < batch; row++) for (int col = 0; col < 3; col++) {
                        float sum = 0; for (int k = 0; k < 4; k++) sum += args.get(1).getAtIndex(ValueLayout.JAVA_FLOAT,(t*batch+row)*4+k)
                                *args.get(2).getAtIndex(ValueLayout.JAVA_FLOAT,k*3+col);
                        expected[row*3+col] += sum; history[(t*batch+row)*3+col] = sum;
                    }
                }
                var bound = bindDirect(artifact.entryPoint(), facts, specialization, args, batch);
                long[] packed = facts.pack(batch); assertEquals(batch, packed[packed.length-1]);
                bound.execute();
                bound.execute();
                assertArrayEquals(expected,args.get(3).toArray(ValueLayout.JAVA_FLOAT));
                assertArrayEquals(history,args.get(4).toArray(ValueLayout.JAVA_FLOAT));
            }
        }
    }

    /**
     * Binds the generated owner's actual entry ABI, including ordinary helper packs and control storage.
     * The MATMUL/ADD body in these fixtures has no indexing or scatter validator cursors.
     * @param entry authenticated generated entry for {@code specialization}
     * @param facts retained actual-batch Scan geometry and ordinary helper recipes
     * @param specialization retained generated helper inventory
     * @param segments typed outer boundaries followed by the complete declared workspace
     * @param batch actual nonnegative batch coordinate for this invocation
     * @return exclusively owned entry arguments reusable without repacking or replacing scratch
     */
    private static DirectScan bindDirect(MethodHandle entry, Facts facts, CpuKernelSpecialization specialization,
            List<MemorySegment> segments, long batch) {
        if (facts.ir().composition().isEmpty()) return new DirectScan(
                entry.asSpreader(0, MemorySegment[].class, segments.size()), segments.toArray(MemorySegment[]::new),
                facts.pack(batch), null, null, null, facts.geometry().time(), batch);
        var composition = facts.ir().composition().orElseThrow();
        int boundaries = facts.batchBindings().size(), units = composition.units().size(), values = composition.values().size();
        var workspace = segments.getLast();
        long roots = composition.activity().map(a -> (long) a.carries().size() + a.seedFormals().size()).orElse(0L);
        long rows = composition.lengthAware()
                ? composition.boundaryMappings().get(composition.lengthRead()).bind(batch).extents().getFirst() : batch;
        var control = workspace.asSlice(composition.workspaceSize().byteSize(batch),
                Math.multiplyExact(1 + roots, 16 + rows));
        var operands = Arrays.copyOf(segments.toArray(MemorySegment[]::new), boundaries + units);
        var packs = new long[units + 3 * values + 4 * units][];
        Arrays.fill(packs, new long[0]);
        for (var value : composition.values()) {
            var access = value.access().bind(batch);
            int position = units + 3 * value.ordinal();
            packs[position] = access.extents().stream().mapToLong(Long::longValue).toArray();
            packs[position + 1] = access.effectiveStrides().stream().mapToLong(Long::longValue).toArray();
            packs[position + 2] = new long[access.extents().size()];
        }
        for (int u = 0; u < units; u++) {
            var unit = composition.units().get(u); var emission = specialization.compositionUnits().get(u);
            var ordinary = facts.ordinaryUnits().get(u);
            var lowered = ordinary.batchGeometry().isPresent() ? ordinary.batchGeometry().orElseThrow().bind(batch) : ordinary;
            long[] bases = new long[unit.boundaries().size()];
            for (int b = 0; b < bases.length; b++) {
                var value = composition.values().get(unit.boundaries().get(b));
                bases[b] = value.offset().byteSize(batch) / value.descriptor().dataType().byteWidth();
            }
            long[] geometry = CpuPreparedExecutable.packGeometry(lowered, emission.specialization(), bases, batch);
            packs[u] = Arrays.copyOf(geometry, geometry.length + emission.resets().size() + 1);
            for (int r = 0; r < emission.resets().size(); r++)
                packs[u][geometry.length + r] = geometry[emission.resets().get(r).cell()];
            packs[u][packs[u].length - 1] = lowered.elementCount();
            long start = emission.scratchOffset().byteSize(batch);
            long end = u + 1 == units ? workspace.byteSize() : specialization.compositionUnits().get(u + 1).scratchOffset().byteSize(batch);
            operands[boundaries + u] = workspace.asSlice(start, end - start);
        }
        return new DirectScan(entry, operands, null, packs, workspace, control, facts.geometry().time(), batch);
    }

    /** Invocation-owned resources for either the original ranged entry or the ordinary composite entry. */
    private record DirectScan(MethodHandle entry, MemorySegment[] operands, long[] geometry, long[][] packs,
            MemorySegment workspace, MemorySegment control, long time, long batch) {
        private void execute() throws Throwable {
            if (packs == null) entry.invokeExact(operands, geometry, 0L, geometry[1]);
            else entry.invokeExact(operands, packs, workspace, control, time, batch);
        }
    }

    private static OperationCapabilityQuery query(int time,boolean reverse,DataType type) {
        var b = new DynamicDimension("B"); var hshape = Shape.ofDimensions(b,new StaticDimension(3));
        var xshape = Shape.ofDimensions(b,new StaticDimension(4)); var wshape = Shape.of(4,3);
        var stacked = Shape.ofDimensions(new StaticDimension(time),b,new StaticDimension(3));
        var h = new ValueId(0); var x = new ValueId(1); var w = new ValueId(2); var product = new ValueId(3); var next = new ValueId(4);
        var values = List.of(new GraphValue(h,descriptor(type,hshape)),new GraphValue(x,descriptor(type,xshape)),new GraphValue(w,descriptor(type,wshape)),
                new GraphValue(product,descriptor(type,hshape)),new GraphValue(next,descriptor(type,hshape)));
        var mm = new CompiledNode(new NodeId(0),new Operation(MatmulKind.MATMUL,NoOperationAttrs.INSTANCE),List.of(x,w),List.of(product));
        var add = new CompiledNode(new NodeId(1),new Operation(BinaryArithmeticKind.ADD,NoOperationAttrs.INSTANCE),List.of(h,product),List.of(next));
        var retained = List.of(h,product,next);
        var graph = new CompiledGraphModel(values,List.of(mm,add),List.of(h,x,w),retained,Map.of(mm.id(),GraphPhase.FORWARD,add.id(),GraphPhase.FORWARD));
        var body = new CompiledScanBody(graph,List.of(next),List.of(product),retained,List.of(0,1));
        return new OperationCapabilityQuery(new Operation(ScanKind.SCAN,new ScanAttrs(time,1,1,1,1,3,reverse)),
                List.of(descriptor(type,hshape),descriptor(type,Shape.ofDimensions(new StaticDimension(time),b,new StaticDimension(4))),descriptor(type,wshape)),
                List.of(descriptor(type,hshape),descriptor(type,stacked),descriptor(type,stacked),descriptor(type,stacked),descriptor(type,stacked)),Optional.of(body),Optional.empty());
    }
    private static TensorDescriptor descriptor(DataType type,Shape shape) {
        return new TensorDescriptor(type,shape,shape.isFullyStatic() ? Optional.of(LayoutDescriptor.contiguous(shape)) : Optional.empty(),false);
    }
}
