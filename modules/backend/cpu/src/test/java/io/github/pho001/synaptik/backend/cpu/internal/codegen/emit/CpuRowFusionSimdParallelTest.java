package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;
import static io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionStrategy.*;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.CpuGeneralScanCapabilityTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.model.shape.*;
import io.github.pho001.synaptik.model.tensor.*;
import io.github.pho001.synaptik.prepare.GraphPreparation;
import io.github.pho001.synaptik.backend.cpu.internal.executable.*;
import io.github.pho001.synaptik.backend.cpu.internal.memory.*;
import io.github.pho001.synaptik.runtime.memory.*;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.run.*;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.ClassFile;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDescs;
import java.lang.constant.MethodTypeDesc;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.reflect.AccessFlag;
import java.util.List;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import org.junit.jupiter.api.Test;

/**
 * Unmeasured exec-v1 seam, strategy-selection and bounded-domain qualification. Tests assert
 * selected ordinary/Scan strategies, actual generated arithmetic and checked masked access,
 * temporal staging, empty/raw identities and retained scalar controls. Passing these focused
 * cases is not final module, integration or performance acceptance.
 */
class CpuRowFusionSimdParallelTest {
    /**
     * Demands both contraction boundaries so a later carry ADD cannot hide a wrong zero sign.
     * Feature K=1 assigns a product; terminal K is the full logical B, never a worker's range.
     * @throws Exception if compilation, binding or represented numerical comparison fails
     */
    @Test void selectedRegionsDistinguishSingletonFeaturesFromCompleteRowContractions() throws Exception {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean symbolic : List.of(false,true))
            for (boolean vector : List.of(false,true)) for (boolean parallel : List.of(false,true)) {
                if (symbolic && type == DataType.FLOAT64) continue;
                for (int fixedRows : symbolic ? List.of(1) : List.of(1,2)) {
                    Dimension rows = symbolic ? new DynamicDimension("B") : new StaticDimension(fixedRows);
                    Tensor x = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.ofDimensions(rows,new StaticDimension(1)));
                    Tensor w = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(1,3));
                    Tensor product = x.add(x).matmul(w), terminal = x.permute(1,0).matmul(product);
                    var compiled = CpuGeneralScanCapabilityTest.compile(List.of(product,terminal),java.util.Optional.empty(),false,new CpuCapabilityProvider());
                    var config = new CpuPartitionAnalysisInputs.PortableExecutionConfig(
                            vector ? CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE
                                    : CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.SCALAR,parallel ? 2 : 1,2,1);
                    var inputs = new CpuPartitionAnalysisInputs(false,List.of(),config);
                    var analysis = new CpuPartitionPreparer().analyze(GraphPreparation.project(compiled,compiled.partitions().getFirst(),inputs));
                    var selected = analysis.plan().units().stream().filter(u -> u.portablePlan().specialization().rowMapReductionIr().isPresent()).findFirst().orElseThrow();
                    assertEquals(parallel ? vector ? PARALLEL_VECTOR : PARALLEL_SCALAR : vector ? VECTOR : SCALAR,selected.executionStrategy());
                    for (int batch : symbolic ? List.of(0,1,2,1) : List.of(fixedRows))
                        try (var run = new CpuRecurrentScanExecutionTest.Fixture(compiled,inputs,analysis,1,true,true,
                                java.util.Map.of(),symbolic ? java.util.OptionalLong.of(batch) : java.util.OptionalLong.empty())) {
                            for (var input : compiled.constants().bindableInputBindings()) {
                                var storage = run.storage.get(input.valueId());
                                for (int cell = 0; cell < storage.byteSize()/type.byteWidth(); cell++)
                                    put(type,storage,cell,input.tensorId().equals(w.id()) ? -1d : 0d);
                            }
                            run.executable.bind(run.state).execute();
                            var rowOutput = run.storage.get(compiled.publication().forwardBindings().get(0).valueId());
                            for (int cell = 0; cell < rowOutput.byteSize()/type.byteWidth(); cell++)
                                assertEquals(Long.MIN_VALUE,Double.doubleToRawLongBits(get(type,rowOutput,cell)));
                            var terminalOutput = run.storage.get(compiled.publication().forwardBindings().get(1).valueId());
                            for (int cell = 0; cell < 3; cell++)
                                assertEquals(batch == 1 ? Long.MIN_VALUE : 0L,Double.doubleToRawLongBits(get(type,terminalOutput,cell)),
                                        "terminal uses complete logical B, including parallel singleton ranges");
                        }
                }
            }
    }

    /**
     * Qualifies the Scan regional binding with an observable row product and a distinct carry
     * ADD. Inactive poisoned rows execute neither product load, and parallel ranges cannot
     * substitute their size for the complete terminal contraction extent.
     */
    @Test void scanRegionsPreserveSingletonProductsBeforeSeparateCarryAdds() {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean symbolic : List.of(false,true))
            for (boolean vector : List.of(false,true)) for (boolean parallel : List.of(false,true))
                for (int fixedRows : symbolic ? List.of(1) : List.of(1,2)) {
                    Dimension b = symbolic ? new DynamicDimension("B") : new StaticDimension(fixedRows);
                    Shape rowShape = Shape.ofDimensions(b,new StaticDimension(1));
                    Tensor rowCarry = CpuSymbolicLengthScanFixture.tensor(type,false,rowShape);
                    Tensor carry = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(1,3));
                    Tensor x = CpuSymbolicLengthScanFixture.tensor(type,false,rowShape);
                    Tensor w = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(1,3));
                    Tensor product = x.add(x).matmul(w), terminal = x.permute(1,0).matmul(product);
                    var body = new ScanBody(List.of(rowCarry,carry),List.of(x),List.of(w),List.of(x,carry.add(terminal)),List.of(product));
                    Tensor initialRows = CpuSymbolicLengthScanFixture.tensor(type,false,rowShape);
                    Tensor initial = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(1,3));
                    Tensor sequence = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.ofDimensions(new StaticDimension(1),b,new StaticDimension(1)));
                    Tensor weight = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(1,3));
                    Tensor lengths = CpuSymbolicLengthScanFixture.tensor(DataType.INT64,false,Shape.ofDimensions(b));
                    var scan = Scan.apply(List.of(initialRows,initial),List.of(sequence),List.of(weight),lengths,1,
                            List.of(java.util.OptionalInt.of(0)),body,false);
                    var compiled = CpuGeneralScanCapabilityTest.compile(List.of(scan.finalCarry().get(1),scan.outputs().getFirst()),
                            java.util.Optional.empty(),false,new CpuCapabilityProvider());
                    var config = new CpuPartitionAnalysisInputs.PortableExecutionConfig(
                            vector ? CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE
                                    : CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.SCALAR,parallel ? 2 : 1,2,1);
                    var inputs = new CpuPartitionAnalysisInputs(false,List.of(),config);
                    var analysis = new CpuPartitionPreparer().analyze(GraphPreparation.project(compiled,compiled.partitions().getFirst(),inputs));
                    var selected = analysis.plan().units().stream().filter(u -> u.runtimeFacts().generalScan().isPresent()).findFirst().orElseThrow();
                    assertTrue(selected.runtimeFacts().generalScan().orElseThrow().ir().stages().orElseThrow().rowMapReductionIr().isPresent());
                    assertEquals(parallel ? vector ? PARALLEL_VECTOR : PARALLEL_SCALAR : vector ? VECTOR : SCALAR,selected.executionStrategy());
                    for (int batch : symbolic ? List.of(1,2,1) : List.of(fixedRows)) for (boolean active : List.of(false,true))
                        try (var run = new CpuRecurrentScanExecutionTest.Fixture(compiled,inputs,analysis,1,true,true,java.util.Map.of(),
                                symbolic ? java.util.OptionalLong.of(batch) : java.util.OptionalLong.empty())) {
                            for (var input : compiled.constants().bindableInputBindings()) {
                                var storage = run.storage.get(input.valueId());
                                if (input.tensorId().equals(lengths.id())) {
                                    for (int row = 0; row < batch; row++) storage.setAtIndex(java.lang.foreign.ValueLayout.JAVA_LONG,row,active ? 1 : 0);
                                } else for (int cell = 0; cell < storage.byteSize()/type.byteWidth(); cell++)
                                    put(type,storage,cell,input.tensorId().equals(initial.id()) ? -0d
                                            : input.tensorId().equals(weight.id()) ? -1d
                                            : input.tensorId().equals(sequence.id()) && !active ? Double.NaN : 0d);
                            }
                            run.executable.bind(run.state).execute();
                            var finalCarry = run.storage.get(compiled.publication().forwardBindings().get(0).valueId());
                            for (int cell = 0; cell < 3; cell++) assertEquals(!active || batch == 1 ? Long.MIN_VALUE : 0L,
                                    Double.doubleToRawLongBits(get(type,finalCarry,cell)),"separate carry ADD remains represented");
                            var output = run.storage.get(compiled.publication().forwardBindings().get(1).valueId());
                            for (int cell = 0; cell < output.byteSize()/type.byteWidth(); cell++) assertEquals(active ? Long.MIN_VALUE : 0L,
                                    Double.doubleToRawLongBits(get(type,output,cell)),"inactive NaN operands are not read");
                        }
                }
    }

    /**
     * A transposed [1,K] coefficient has a width-one vector, not permission to reload its scalar
     * cells per row. Both selected emitters must bind checked mask-one scratch loads before any
     * vector multiply. The unpadded short ordinary graph also traverses complete candidate scoring
     * and retained-identity validation in every strategy. The left input is consumed only as
     * scalars, exercising its live binding too.
     */
    @Test void widthOneTransposesArePreboundInOrdinaryAndScanVectorRegions() throws Exception {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean scan : List.of(false,true))
                for (boolean parallel : List.of(false,true)) for (boolean vector : List.of(false,true)) {
            Tensor x = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(7,2));
            Tensor q = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(1,2));
            Tensor actualQ = scan ? CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(1,2)) : q;
            Tensor z = x.matmul(q.permute(1,0)).neg();
            Tensor product = x.permute(1,0).matmul(z);
            Tensor initialSum = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(1));
            Tensor initialProduct = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(2,1));
            Tensor xs = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(2,7,2));
            List<Tensor> roots;
            if (scan) {
                Tensor sumCarry = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(1));
                Tensor productCarry = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(2,1));
                var body = new ScanBody(List.of(sumCarry,productCarry),List.of(x),List.of(q),
                        List.of(sumCarry.add(z.sumToShape(Shape.of(1))),productCarry.add(product)),List.of(z));
                var result = Scan.apply(List.of(initialSum,initialProduct),List.of(xs),List.of(actualQ),body,false);
                roots = List.of(result.finalCarry().get(0),result.finalCarry().get(1),result.outputs().getFirst());
            } else roots = List.of(z,product);
            var compiled = CpuGeneralScanCapabilityTest.compile(roots,java.util.Optional.empty(),false,new CpuCapabilityProvider());
            var config = new CpuPartitionAnalysisInputs.PortableExecutionConfig(
                    vector ? CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE
                            : CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.SCALAR,parallel ? 2 : 1,2,1);
            var analysisInputs = new CpuPartitionAnalysisInputs(false,List.of(),config);
            var context = GraphPreparation.project(compiled,compiled.partitions().getFirst(),analysisInputs);
            if (!scan) assertTrue(context.nodes().size() <= 8,"the ordinary regression must enter bounded candidate discovery");
            var analysis = new CpuPartitionPreparer().analyze(context);
            var unit = analysis.plan().units().stream().filter(u -> u.portablePlan() != null &&
                    (scan ? u.runtimeFacts().generalScan().isPresent() : u.portablePlan().specialization().rowMapReductionIr().isPresent()))
                    .findFirst().orElseThrow();
            assertEquals(parallel ? vector ? PARALLEL_VECTOR : PARALLEL_SCALAR : vector ? VECTOR : SCALAR,unit.executionStrategy());
            var spec = unit.portablePlan().specialization();
            if (!scan) {
                var candidates = analysis.plan().fusionDecisions().stream()
                        .filter(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFusionDecision.LegalCandidate.class::isInstance)
                        .map(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFusionDecision.LegalCandidate.class::cast).toList();
                assertFalse(candidates.isEmpty(),"do not bypass ordinary candidate scoring");
                for (var candidate : candidates) {
                    assertEquals(1,candidate.identity().units().size(),"the complete short row region stays indivisible");
                    assertEquals(0,candidate.facts().crossUnitMaterializedBytes(),"internal row-to-terminal uses are not cross-unit copies");
                }
                if (vector || parallel) for (var candidate : candidates) for (var identity : candidate.identity().units())
                    if (identity.specialization().rowFusionEmission().isPresent()) assertEquals(
                            io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFusionDecision.WorkspaceRole.ROW_REGION_STATE,
                            identity.workspace().orElseThrow().role());
            }
            var generator = new CpuClassFileKernelGenerator();
            byte[] bytes = generator.generateClassBytes(spec,unit.portablePlan().kernelIr());
            generator.defineClassBytes(spec, unit.portablePlan().kernelIr(), bytes);
            var methodName = parallel ? scan ? "scanRowRange" : "ordinaryRowRange" : "invoke";
            var code = ClassFile.of().parse(bytes).methods().stream().filter(m -> m.methodName().equalsString(methodName))
                    .findFirst().orElseThrow().code().orElseThrow();
            int loads = 0; boolean multiplied = false;
            for (var element : code) if (element instanceof java.lang.classfile.instruction.InvokeInstruction call
                    && call.owner().asInternalName().startsWith("jdk/incubator/vector/")) {
                if (call.name().equalsString("mul")) multiplied = true;
                if (call.name().equalsString("fromMemorySegment")) {
                    assertFalse(multiplied,"staged coefficients belong to the range preheader, not the row body");
                    assertTrue(call.type().stringValue().contains("VectorMask")); loads++;
                }
            }
            if (vector) {
                assertEquals(1,spec.rowFusionEmission().orElseThrow().invariantTransposeValues().size());
                var region = scan ? spec.generalScanIr().orElseThrow().stages().orElseThrow().rowMapReductionIr().orElseThrow()
                        : spec.rowMapReductionIr().orElseThrow();
                var staged = region.accesses().get(spec.rowFusionEmission().orElseThrow().invariantTransposeValues().getFirst());
                assertEquals(List.of(2L,1L),staged.extentConstants());
                assertNotEquals(1L,staged.strideConstants().getLast(),"the actual transposed RHS must require staging");
                assertEquals(2,loads,"one checked mask-one native preload per K coefficient");
                assertTrue(multiplied,"the selected row method must really execute vector arithmetic");
            }
            for (int carrier : scan ? List.of(1) : List.of(0,1)) try (var run = new CpuRecurrentScanExecutionTest.Fixture(
                    compiled,analysisInputs,analysis,carrier,true,true,java.util.Map.of())) {
                for (var binding : compiled.constants().bindableInputBindings()) {
                    var segment = run.storage.get(binding.valueId());
                    for (int c = 0; c < segment.byteSize()/type.byteWidth(); c++) put(type,segment,c,
                            binding.tensorId().equals(actualQ.id()) ? (c+1)*.25
                            : binding.tensorId().equals(x.id()) || binding.tensorId().equals(xs.id()) ? (c%7-3)*.125 : .25);
                }
                run.executable.bind(run.state).execute();
                var expectedRows = new double[scan ? 14 : 7]; var expectedProduct = new double[]{scan ? .25 : 0,scan ? .25 : 0};
                double expectedSum = .25;
                for (int t = 0; t < (scan ? 2 : 1); t++) {
                    double stepSum = 0; var stepProduct = new double[2];
                    for (int r = 0; r < 7; r++) {
                        double a = (((t*7+r)*2)%7-3)*.125, b = (((t*7+r)*2+1)%7-3)*.125;
                        double value = -(a*.25+b*.5); expectedRows[t*7+r] = value; stepSum += value;
                        stepProduct[0] += a*value; stepProduct[1] += b*value;
                    }
                    expectedSum += stepSum;
                    for (int c = 0; c < 2; c++) expectedProduct[c] += stepProduct[c];
                }
                var expected = scan ? List.of(new double[]{expectedSum},expectedProduct,expectedRows) : List.of(expectedRows,expectedProduct);
                for (int p = 0; p < expected.size(); p++) {
                    var actual = run.storage.get(compiled.publication().forwardBindings().get(p).valueId());
                    assertEquals(expected.get(p).length,actual.byteSize()/type.byteWidth());
                    for (int c = 0; c < expected.get(p).length; c++) assertEquals(expected.get(p)[c],get(type,actual,c),0,
                            type+" scan="+scan+" parallel="+parallel+" port="+p+" cell="+c);
                }
            }
        }
    }


    @Test void nonregionalSymbolicConstructorCannotBorrowWorkersOrAdmitASelectedPolicy() {
        for(var type:List.of(DataType.FLOAT32,DataType.FLOAT64)) {
            Tensor x=CpuSymbolicLengthScanFixture.tensor(type,false,Shape.ofDimensions(new DynamicDimension("B"),new StaticDimension(3)));
            var compiled=CpuGeneralScanCapabilityTest.compile(List.of(x.add(x)),java.util.Optional.empty(),false,new CpuCapabilityProvider());
            var analysis=new CpuPartitionPreparer().analyze(GraphPreparation.project(compiled,compiled.partitions().getFirst(),new CpuPartitionAnalysisInputs(false,List.of())));
            var unit=analysis.plan().units().getFirst();var route=unit.portablePlan();
            assertTrue(route.specialization().rowFusionEmission().isEmpty());
            var generator=new CpuClassFileKernelGenerator();var artifact=generator.defineClassBytes(route.specialization(), route.kernelIr(), generator.generateClassBytes(route.specialization(), route.kernelIr()));
            var buffers=analysis.plan().bufferDeclarations();
            var memory=new PreparedMemoryPlan(java.util.stream.IntStream.range(0,buffers.size()).mapToObj(i->new PreparedMemoryPlan.BufferEntry(new BufferSlot(i),buffers.get(i).size(),buffers.get(i).byteAlignment())).toList(),List.of(),true);
            var selections=unit.boundaryValues().stream().map(id->new PreparedExecutable.BufferSelection(buffers.stream().map(v->v.valueId()).toList().indexOf(id),0)).toList();
            var budget=new CpuConcurrencyBudget(2);
            try(var workers=new CpuWorkerGroup(2,budget)) {
                assertDoesNotThrow(()->new CpuPreparedExecutable(memory,selections,artifact,route,java.util.Optional.empty(),null));
                assertThrows(IllegalArgumentException.class,()->new CpuPreparedExecutable(memory,selections,artifact,route,java.util.Optional.empty(),budget,2,1,workers));
                assertThrows(IllegalArgumentException.class,()->new CpuPreparedExecutable(memory,selections,artifact,route,java.util.Optional.empty(),null,1,2,null));
            }
        }
    }

    @Test void selectedArrayAndSegmentHelpersPreserveSpecialValueSitesAndTailCanaries() throws Throwable {
        for(var type:List.of(DataType.FLOAT32,DataType.FLOAT64))for(var strategy:List.of(VECTOR,PARALLEL_SCALAR,PARALLEL_VECTOR))
            for(boolean arrays:List.of(false,true))for(int scenario:List.of(0,1,2)) {
            int rows=7,width=type.byteWidth(),maximum=strategy.equals(VECTOR)?1:2,slice=8+4*width,middle=(rows+1)/2;
            var ir=io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRowMapReductionLoweringTest.candidate(type,false,rows);
            var access=arrays?type==DataType.FLOAT64?CarrierAccess.DOUBLE_ARRAY:CarrierAccess.FLOAT_ARRAY:CarrierAccess.MEMORY_SEGMENT;
            int bits=strategy.equals(PARALLEL_SCALAR)?0:type==DataType.FLOAT64?DoubleVector.SPECIES_PREFERRED.vectorBitSize():FloatVector.SPECIES_PREFERRED.vectorBitSize();
            var spec=new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization(
                    io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint.fromHex(ir.structuralKey()),strategy,
                    java.util.Collections.nCopies(4,type),java.util.Collections.nCopies(4,access),bits,-1,List.of(),true,72,
                    java.util.Optional.empty(),java.util.Optional.empty(),false,false,java.util.Optional.of(ir),
                    java.util.Optional.of(new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.RowFusionEmission(maximum,List.of(),List.of(),0,0)));
            var generator=new CpuClassFileKernelGenerator();var kernel=generator.defineClassBytes(spec, ir.encodedKernelIr(), generator.generateClassBytes(spec, ir.encodedKernelIr()));
            try(var arena=java.lang.foreign.Arena.ofShared()) {
                var carriers=new Object[5];var segments=new MemorySegment[4];
                for(int p=0;p<4;p++) {
                    int cells=p==0||p==2?2*rows:4;
                    if(arrays) {carriers[p]=type==DataType.FLOAT64?new double[cells]:new float[cells];segments[p]=type==DataType.FLOAT64?MemorySegment.ofArray((double[])carriers[p]):MemorySegment.ofArray((float[])carriers[p]);}
                    else {segments[p]=arena.allocate((long)cells*width,width);carriers[p]=segments[p];}
                }
                double minimum=type==DataType.FLOAT64?Double.MIN_VALUE:Float.MIN_VALUE,maximumValue=type==DataType.FLOAT64?Double.MAX_VALUE:Float.MAX_VALUE;
                double[] values=scenario==0?new double[]{0.0,-0.0,minimum,-minimum,1,-1,minimum*16,-minimum*16}
                        :scenario==1?new double[]{maximumValue,-maximumValue,minimum,-minimum,1,-1,0.0,-0.0}
                        :new double[]{Double.NaN,Double.POSITIVE_INFINITY,Double.NEGATIVE_INFINITY,-0.0,1,-1};
                for(int k=0;k<2*rows;k++)put(type,segments[0],k,values[k%values.length]);
                for(int k=0;k<4;k++)put(type,segments[1],k,k==0||k==3?1:0);
                var guarded=arena.allocate(2L*slice+16,8);guarded.fill((byte)0x5a);carriers[4]=guarded.asSlice(8,2L*slice);
                long[] geometry={rows,0,0,0,0,maximum,0};
                if(maximum==1)invokeMixedCarrierControl(kernel.entryPoint(),carriers,geometry,0,1);
                else {
                    var range=kernel.ordinaryRowRangeEntryPoint();
                    invokeMixedCarrierControl(range,carriers,geometry,0,middle);
                    geometry[6]=slice;invokeMixedCarrierControl(range,carriers,geometry,middle,rows);geometry[6]=0;
                    invokeMixedCarrierControl(kernel.ordinaryRowMergeEntryPoint(),carriers,geometry,0,1);
                }
                for(int r=0;r<rows;r++)for(int j=0;j<2;j++) {
                    double expected=0;
                    for(int k=0;k<2;k++)expected=rounded(type,expected+rounded(type,rounded(type,get(type,segments[0],2*r+k)+get(type,segments[0],2*r+k))*(k==j?1:0)));
                    assertSiteClassAndBits(expected,get(type,segments[2],2*r+j));
                }
                for(int a=0;a<2;a++)for(int b=0;b<2;b++) {
                    double left=0,right=0;
                    for(int r=0;r<rows;r++) {
                        double product=rounded(type,get(type,segments[0],2*r+a)*rounded(type,get(type,segments[0],2*r+b)+get(type,segments[0],2*r+b)));
                        if(maximum==2&&r>=middle)right=rounded(type,right+product);else left=rounded(type,left+product);
                    }
                    assertSiteClassAndBits(maximum==2?rounded(type,left+right):left,get(type,segments[3],2*a+b));
                }
                for(int k=0;k<8;k++){assertEquals((byte)0x5a,guarded.get(java.lang.foreign.ValueLayout.JAVA_BYTE,k));assertEquals((byte)0x5a,guarded.get(java.lang.foreign.ValueLayout.JAVA_BYTE,guarded.byteSize()-8+k));}
            }
        }
    }

    /** Control-only heterogeneous carrier adaptation; never part of production or measured work. */
    private static void invokeMixedCarrierControl(java.lang.invoke.MethodHandle entry,Object[] carriers,long[] geometry,long begin,long end) throws Throwable {
        Object[] arguments=java.util.Arrays.copyOf(carriers,carriers.length+3);
        arguments[carriers.length]=geometry;arguments[carriers.length+1]=begin;arguments[carriers.length+2]=end;
        entry.invokeWithArguments(arguments);
    }

    /** NaN classification is separate; every other site retains exact represented class/sign/bits. */
    private static void assertSiteClassAndBits(double expected,double actual) {
        if(Double.isNaN(expected))assertTrue(Double.isNaN(actual));
        else assertEquals(Double.doubleToRawLongBits(expected),Double.doubleToRawLongBits(actual));
    }
    @Test void sumOnlySingletonNegationReallySelectsVectorAndPreservesRawMinusZero() {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean parallel : List.of(false,true)) {
            Tensor carry=CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(2));
            var rows=Shape.ofDimensions(new DynamicDimension("B"),new StaticDimension(2));
            Tensor x=CpuSymbolicLengthScanFixture.tensor(type,false,rows),negative=x.neg(),rowCarry=CpuSymbolicLengthScanFixture.tensor(type,false,rows);
            var body=new ScanBody(List.of(rowCarry,carry),List.of(x),List.of(),List.of(negative,carry.add(negative.sumToShape(Shape.of(2)))),List.of(negative));
            Tensor initial=CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(2)),initialRow=CpuSymbolicLengthScanFixture.tensor(type,false,rows);
            Tensor xs=CpuSymbolicLengthScanFixture.tensor(type,false,Shape.ofDimensions(new StaticDimension(1),new DynamicDimension("B"),new StaticDimension(2)));
            Tensor lengths=CpuSymbolicLengthScanFixture.tensor(DataType.INT64,false,Shape.ofDimensions(new DynamicDimension("B")));
            var scan=Scan.apply(List.of(initialRow,initial),List.of(xs),List.of(),lengths,1,List.of(java.util.OptionalInt.of(0)),body,false);
            var compiled=CpuGeneralScanCapabilityTest.compile(List.of(scan.finalCarry().get(1),scan.outputs().getFirst()),java.util.Optional.empty(),false,new CpuCapabilityProvider());
            var config=new CpuPartitionAnalysisInputs.PortableExecutionConfig(CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE,parallel?2:1,2,1);
            var analysisInputs=new CpuPartitionAnalysisInputs(false,List.of(),config);
            var analysis=new CpuPartitionPreparer().analyze(GraphPreparation.project(compiled,compiled.partitions().getFirst(),analysisInputs));
            assertEquals(parallel?PARALLEL_VECTOR:VECTOR,analysis.plan().units().getFirst().executionStrategy());
            try(var run=new CpuRecurrentScanExecutionTest.Fixture(compiled,analysisInputs,analysis,1,true,true,java.util.Map.of(),java.util.OptionalLong.of(1))) {
                for(var input:compiled.constants().bindableInputBindings()) {
                    var s=run.storage.get(input.valueId());
                    if(input.tensorId().equals(lengths.id())) {s.setAtIndex(java.lang.foreign.ValueLayout.JAVA_LONG,0,1);continue;}
                    for(int k=0;k<s.byteSize()/type.byteWidth();k++)put(type,s,k,input.tensorId().equals(initial.id())?-0.0:0.0);
                }
                run.executable.bind(run.state).execute();
                for(var output:compiled.publication().forwardBindings()) {
                    var s=run.storage.get(output.valueId());
                    for(int k=0;k<s.byteSize()/type.byteWidth();k++)assertEquals(Long.MIN_VALUE,Double.doubleToRawLongBits(get(type,s,k)));
                }
            }
        }
    }

    @Test void selectedActualReverseHandlesZeroTimeAllInactiveAndPoisonedPaddingWithoutReadingIt() {
        for(var type:List.of(DataType.FLOAT32,DataType.FLOAT64))for(boolean vector:List.of(false,true))for(boolean parallel:List.of(false,true))
            for(int scenario:List.of(0,1,2)) {
            int time=scenario==0?0:4,batch=7;
            var config=new CpuPartitionAnalysisInputs.PortableExecutionConfig(vector?CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE
                    :CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.SCALAR,parallel?2:1,2,1);
            var family=new CpuSymbolicLengthScanFixture.Family(type,false,time,3,config);
            try(var reference=new CpuSymbolicLengthScanFixture(family,batch)) {
                for(int r=0;r<batch;r++) {
                    long length=scenario==1?0:reference.lengths.getAtIndex(java.lang.foreign.ValueLayout.JAVA_LONG,r);
                    reference.lengths.setAtIndex(java.lang.foreign.ValueLayout.JAVA_LONG,r,length);
                    for(int t=0;t<time;t++)if(t>=length) {
                        for(int k=0;k<3;k++)put(type,reference.x,3*(t*batch+r)+k,Double.NaN);
                        for(int k=0;k<2;k++)put(type,reference.es,2*(t*batch+r)+k,Double.NaN);
                    }
                    if(scenario<2)for(int k=0;k<2;k++)put(type,reference.fs,2*r+k,k==0?-0.0:Double.NaN);
                }
                if(time==0) {
                    reference.expected.get(0).copyFrom(reference.initial);reference.expected.get(2).copyFrom(reference.fs);
                    for(int p:List.of(1,3,4,5,6))reference.expected.get(p).fill((byte)0);
                } else {reference.directForward.run();reference.directReverse.run();}
                reference.execute();
                var gradients=family.compiled.publication().gradientBindings();
                if(scenario<2)assertArrayEquals(reference.fs.toArray(java.lang.foreign.ValueLayout.JAVA_BYTE),
                        reference.actual.get(2).toArray(java.lang.foreign.ValueLayout.JAVA_BYTE),"inactive seed is raw identity");
                for(var gradient:gradients) {
                    var actual=reference.actual.get(gradient.targetIndex()+2);var expected=reference.expected.get(gradient.targetIndex()+2);
                    for(int k=0;k<actual.byteSize()/type.byteWidth();k++) {
                        double e=get(type,expected,k),a=get(type,actual,k);
                        if(Double.isNaN(e))assertTrue(Double.isNaN(a));else assertEquals(e,a,type==DataType.FLOAT64?2e-14:2e-6);
                    }
                }
            }
        }
    }
    @Test void externalSumVectorCacheIsDominatedBeforeSingletonAndAddAlternatives() {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (int rows : List.of(1,7)) for (boolean parallel : List.of(false,true)) {
            Tensor h = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(2)), p = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(2,2));
            Tensor x = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(rows,2)), q = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(rows,2));
            Tensor negative = x.neg();
            var body = new ScanBody(List.of(h,p),List.of(x,q),List.of(),
                    List.of(h.add(q.sumToShape(Shape.of(2))),p.add(x.permute(1,0).matmul(negative))),List.of(negative));
            Tensor initial = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(2)), initialProduct = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(2,2));
            Tensor xs = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(2,rows,2)), qs = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(2,rows,2));
            var scan = Scan.apply(List.of(initial,initialProduct),List.of(xs,qs),List.of(),body,false);
            var compiled = CpuGeneralScanCapabilityTest.compile(List.of(scan.finalCarry().get(0),scan.finalCarry().get(1),scan.outputs().getFirst()),
                    java.util.Optional.empty(),false,new CpuCapabilityProvider());
            var config = new CpuPartitionAnalysisInputs.PortableExecutionConfig(CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE,
                    parallel ? 2 : 1,2,1);
            var analysisInputs = new CpuPartitionAnalysisInputs(false,List.of(),config);
            var analysis = new CpuPartitionPreparer().analyze(GraphPreparation.project(compiled,compiled.partitions().getFirst(),analysisInputs));
            assertEquals(parallel ? PARALLEL_VECTOR : VECTOR,analysis.plan().units().getFirst().executionStrategy());
            try (var run = new CpuRecurrentScanExecutionTest.Fixture(compiled,analysisInputs,analysis,1,true,true,java.util.Map.of())) {
                for (var input : compiled.constants().bindableInputBindings()) {
                    var s = run.storage.get(input.valueId());
                    for (int c = 0; c < s.byteSize()/type.byteWidth(); c++) put(type,s,c,input.tensorId().equals(xs.id()) ? .125 : .25);
                }
                run.executable.bind(run.state).execute();
                for (int output = 0; output < 3; output++) {
                    var s = run.storage.get(compiled.publication().forwardBindings().get(output).valueId());
                    double expected = output == 0 ? .25+.5*rows : output == 1 ? .25-.03125*rows : -.125;
                    for (int c = 0; c < s.byteSize()/type.byteWidth(); c++) assertEquals(expected,get(type,s,c),0);
                }
            }
        }
    }
    @Test void requestedInlineScanPolicyPreservesAllNumericalOutputs() {
        var family = new CpuSymbolicLengthScanFixture.Family(DataType.FLOAT32,false,4,3,
                new CpuPartitionAnalysisInputs.PortableExecutionConfig(CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE,1,1,2));
        try (var reference = new CpuSymbolicLengthScanFixture(family,7)) { reference.verify(); }
    }
    @Test void oneSelectedPreparedOwnerSupportsConcurrentMixedBatchRuns() throws Exception {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean vector : List.of(false,true)) {
            var config = new CpuPartitionAnalysisInputs.PortableExecutionConfig(vector
                    ? CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE
                    : CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.SCALAR,4,4,2);
            var family = new CpuSymbolicLengthScanFixture.Family(type,false,4,3,config);
            try (var owner = new CpuRecurrentScanExecutionTest.Fixture(family.compiled,family.analysisInputs,family.analysis,1,true,true,java.util.Map.of(),java.util.OptionalLong.of(7))) {
                var references = new java.util.ArrayList<CpuSymbolicLengthScanFixture>();
                var runs = new java.util.ArrayList<CpuRecurrentScanExecutionTest.Fixture>();
                var threads = new java.util.ArrayList<Thread>();
                var begin = new java.util.concurrent.CountDownLatch(1);
                var failure = new java.util.concurrent.atomic.AtomicReference<Throwable>();
                try {
                    for (int b : List.of(0,1,7,32)) {
                        var reference = new CpuSymbolicLengthScanFixture(family,b); references.add(reference);
                        reference.directForward.run(); reference.directReverse.run();
                        var run = new CpuRecurrentScanExecutionTest.Fixture(family.compiled,family.analysisInputs,family.analysis,1,true,true,java.util.Map.of(),java.util.OptionalLong.of(b),owner); runs.add(run);
                        assertSame(owner.executable,run.executable); assertSame(owner.state.memoryPlan(),run.state.memoryPlan());
                        copyReferenceInputs(reference,run);
                        var invocation = run.executable.bind(run.state);
                        threads.add(Thread.ofPlatform().start(() -> {
                            try { begin.await(); invocation.execute(); }
                            catch (Throwable actual) { failure.compareAndSet(null,actual); }
                        }));
                    }
                    begin.countDown();
                    for (var thread : threads) { thread.join(5_000); assertFalse(thread.isAlive(),"mixed-B invocation must finish without nested admission"); }
                    assertNull(failure.get(),"child-thread failures must reach the test thread");
                    var outputs = new java.util.ArrayList<io.github.pho001.synaptik.model.graph.ValueId>();
                    family.compiled.publication().forwardBindings().forEach(p -> outputs.add(p.valueId()));
                    family.compiled.publication().gradientBindings().forEach(p -> outputs.add(p.valueId()));
                    for (int i = 0; i < runs.size(); i++) for (int p = 0; p < outputs.size(); p++) {
                        var expected = references.get(i).expected.get(p); var actual = runs.get(i).storage.get(outputs.get(p));
                        for (int c = 0; c < expected.byteSize()/type.byteWidth(); c++) assertEquals(get(type,expected,c),get(type,actual,c),type == DataType.FLOAT64 ? 2e-14 : 2e-6);
                    }
                } finally {
                    begin.countDown();
                    for (var thread : threads) if (thread.isAlive()) { thread.interrupt(); thread.join(5_000); }
                    for (var run : runs) run.close();
                    for (var reference : references) reference.close();
                }
            }
        }
    }
    @Test void selectedOrdinaryEmptyWorkRangeDoesNotStageDispatchOrPublish() {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (var strategy : List.of(VECTOR,PARALLEL_SCALAR,PARALLEL_VECTOR))
                for (int rows : List.of(0,7)) {
            var ir = io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRowMapReductionLoweringTest.candidate(type,false,rows);
            int maximum = strategy.equals(VECTOR) ? 1 : 2;
            int bits = strategy.equals(PARALLEL_SCALAR) ? 0 : type == DataType.FLOAT64 ? DoubleVector.SPECIES_PREFERRED.vectorBitSize() : FloatVector.SPECIES_PREFERRED.vectorBitSize();
            var values = ir.arithmetic().values().stream().filter(v -> v.kind() != io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr.Value.Kind.VIRTUAL).toList();
            var bindings = values.stream().map(v -> ir.accesses().get(v.ordinal()).bind(0)).toList();
            var pattern = java.util.Collections.nCopies(values.size(),CarrierAccess.MEMORY_SEGMENT);
            var spec = new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization(
                    io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint.fromHex(ir.structuralKey()),strategy,
                    java.util.Collections.nCopies(values.size(),type),pattern,bits,-1,List.of(),true,72,java.util.Optional.empty(),java.util.Optional.empty(),false,false,
                    java.util.Optional.of(ir),java.util.Optional.of(new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.RowFusionEmission(maximum,List.of(),List.of(),0,0)));
            var generator = new CpuClassFileKernelGenerator(); var artifact = generator.defineClassBytes(spec, ir.encodedKernelIr(), generator.generateClassBytes(spec, ir.encodedKernelIr()));
            long scratchBytes = maximum*(8+4L*type.byteWidth());
            var memory = new PreparedMemoryPlan(java.util.stream.IntStream.range(0,values.size()).mapToObj(i ->
                    new PreparedMemoryPlan.BufferEntry(new BufferSlot(i),bindings.get(i).referencedElementSpan()*type.byteWidth(),type.byteWidth())).toList(),
                    List.of(new PreparedMemoryPlan.WorkspaceEntry(new WorkspaceSlot(0),scratchBytes,8)));
            var selections = java.util.stream.IntStream.range(0,values.size()).mapToObj(i -> new PreparedExecutable.BufferSelection(i,0)).toList();
            var budget = new CpuConcurrencyBudget(2);
            try (var arena = java.lang.foreign.Arena.ofShared(); var workers = new CpuWorkerGroup(2,budget)) {
                var buffers = new java.util.ArrayList<List<BufferRepresentationBinding>>(); var segments = new java.util.ArrayList<MemorySegment>();
                for (var binding : bindings) {
                    var segment = arena.allocate(binding.referencedElementSpan()*type.byteWidth(),type.byteWidth()); segment.fill((byte)0x55); segments.add(segment);
                    var buffer = CpuBorrowedBuffer.borrow(new MemorySegmentStorage(type,binding.referencedElementSpan(),segment));
                    buffers.add(List.of(new BufferRepresentationBinding(buffer,RunResourceOwnership.BORROWED)));
                }
                var scratch = CpuContiguousWorkspace.allocate(scratchBytes,8); scratch.writableSegment().fill((byte)0x55);
                try (var state = new RunState(memory,buffers,List.of(scratch))) {
                    var before = segments.stream().map(s -> s.toArray(java.lang.foreign.ValueLayout.JAVA_BYTE)).toList();
                    var scratchBefore = scratch.writableSegment().toArray(java.lang.foreign.ValueLayout.JAVA_BYTE);
                    java.util.function.LongFunction<CpuPreparedExecutable> recipe = end -> new CpuPreparedExecutable(
                            memory,selections,artifact,bindings,pattern,pattern,0,end,maximum,1,maximum > 1 ? workers : null,
                            java.util.Optional.empty(),java.util.Optional.of(new PreparedExecutable.WorkspaceSelection(0)),null,
                            java.util.Optional.empty(),java.util.Optional.empty(),java.util.Optional.empty(),java.util.Optional.empty(),
                            java.util.Optional.empty(),java.util.Optional.empty(),java.util.Optional.empty(),java.util.Optional.empty(),
                            java.util.Optional.empty(),java.util.Optional.empty(),java.util.Optional.empty(),java.util.Optional.empty(),
                            java.util.Optional.empty(),java.util.Optional.empty(),java.util.Optional.empty(),java.util.Optional.empty(),
                            java.util.Optional.empty(),java.util.Optional.empty(),2);
                    var noOp = recipe.apply(0);
                    noOp.bind(state).execute();
                    for (int p = 0; p < segments.size(); p++) assertArrayEquals(before.get(p),segments.get(p).toArray(java.lang.foreign.ValueLayout.JAVA_BYTE));
                    assertArrayEquals(scratchBefore,scratch.writableSegment().toArray(java.lang.foreign.ValueLayout.JAVA_BYTE));
                    if (rows == 0) {
                        var full = recipe.apply(1);
                        full.bind(state).execute();
                        for (int c = 0; c < 4; c++) assertEquals(0,get(type,segments.getLast(),c),0,"R=0 full invocation still publishes terminal zeros");
                    }
                }
            }
        }
    }
    @Test void changingSharedCarryAndSequenceTransposesStageAtTheirActualTime() {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean history : List.of(false,true))
                for (boolean lengthAware : List.of(false,true)) for (boolean parallel : List.of(false,true)) {
            if (lengthAware && !history) continue;
            Tensor x = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(7,2));
            Tensor q = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(3,2));
            Tensor sum = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(3)), product = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(2,3));
            Tensor z = x.matmul(q.permute(1,0));
            var carries = new java.util.ArrayList<Tensor>(); var next = new java.util.ArrayList<Tensor>();
            Tensor row = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(7,2));
            if (lengthAware) { carries.add(row); next.add(x); }
            if (!history) { carries.add(q); next.add(q.add(q)); }
            carries.addAll(List.of(sum,product)); next.addAll(List.of(sum.add(z.sumToShape(Shape.of(3))),product.add(x.permute(1,0).matmul(z))));
            var sequences = history ? List.of(x,q) : List.of(x);
            var body = new ScanBody(carries,sequences,List.of(),next,List.of(z));
            var initial = carries.stream().map(t -> CpuSymbolicLengthScanFixture.tensor(type,false,t.descriptor().shape())).toList();
            Tensor xs = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(4,7,2));
            Tensor qs = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(4,3,2));
            Tensor lengths = CpuSymbolicLengthScanFixture.tensor(DataType.INT64,false,Shape.of(7));
            var actualSequences = history ? List.of(xs,qs) : List.of(xs);
            var scan = lengthAware ? Scan.apply(initial,actualSequences,List.of(),lengths,1,
                    List.of(java.util.OptionalInt.of(0),java.util.OptionalInt.empty()),body,false)
                    : Scan.apply(initial,actualSequences,List.of(),body,false);
            int sumPort = carries.size()-2;
            var roots = List.of(scan.finalCarry().get(sumPort),scan.finalCarry().get(sumPort+1),scan.outputs().getFirst());
            var compiled = CpuGeneralScanCapabilityTest.compile(roots,java.util.Optional.empty(),false,new CpuCapabilityProvider());
            var config = new CpuPartitionAnalysisInputs.PortableExecutionConfig(CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE,
                    parallel ? 4 : 1,4,2);
            var analysisInputs = new CpuPartitionAnalysisInputs(false,List.of(),config);
            var analysis = new CpuPartitionPreparer().analyze(GraphPreparation.project(compiled,compiled.partitions().getFirst(),analysisInputs));
            var unit = analysis.plan().units().stream().filter(u -> u.runtimeFacts().generalScan().isPresent()).findFirst().orElseThrow();
            assertEquals(parallel ? PARALLEL_VECTOR : VECTOR,unit.executionStrategy());
            var emission = unit.portablePlan().specialization().rowFusionEmission().orElseThrow();
            assertEquals(List.of(),emission.invariantTransposeValues()); assertEquals(1,emission.temporalTransposeValues().size());
            try (var run = new io.github.pho001.synaptik.backend.cpu.internal.executable.CpuRecurrentScanExecutionTest.Fixture(compiled,analysisInputs,analysis,1,true,true,java.util.Map.of())) {
                for (var input : compiled.constants().bindableInputBindings()) {
                    var s = run.storage.get(input.valueId());
                    if (input.tensorId().equals(lengths.id())) for (int r = 0; r < 7; r++) s.setAtIndex(java.lang.foreign.ValueLayout.JAVA_LONG,r,r%5);
                    else for (int c = 0; c < s.byteSize()/type.byteWidth(); c++) {
                        double value = input.tensorId().equals(xs.id()) ? (c%7-3)*.125
                                : history && input.tensorId().equals(qs.id()) ? .125*(1+c/6)*(1+c%6)
                                : !history && input.tensorId().equals(initial.get(0).id()) ? .125*(1+c) : .25;
                        put(type,s,c,value);
                    }
                }
                run.executable.bind(run.state).execute();
                var expectedSum = new double[]{.25,.25,.25}; var expectedProduct = new double[6]; java.util.Arrays.fill(expectedProduct,.25);
                var expectedRows = new double[84];
                for (int t = 0; t < 4; t++) {
                    var stepSum = new double[3]; var stepProduct = new double[6]; boolean active = false;
                    for (int r = 0; r < 7; r++) if (!lengthAware || t < r%5) {
                        active = true;
                        for (int j = 0; j < 3; j++) {
                            double value = 0;
                            for (int k = 0; k < 2; k++) value += (((t*7+r)*2+k)%7-3)*.125*(.125*(history ? t+1 : 1<<t)*(1+2*j+k));
                            expectedRows[(t*7+r)*3+j] = value; stepSum[j] += value;
                            for (int k = 0; k < 2; k++) stepProduct[3*k+j] += (((t*7+r)*2+k)%7-3)*.125*value;
                        }
                    }
                    if (active || !lengthAware) {
                        for (int c = 0; c < 3; c++) expectedSum[c] += stepSum[c];
                        for (int c = 0; c < 6; c++) expectedProduct[c] += stepProduct[c];
                    }
                }
                var expected = List.of(expectedSum,expectedProduct,expectedRows);
                for (int p = 0; p < 3; p++) {
                    var s = run.storage.get(compiled.publication().forwardBindings().get(p).valueId());
                    for (int c = 0; c < expected.get(p).length; c++) assertEquals(expected.get(p)[c],get(type,s,c),0,
                            type+" history="+history+" lengths="+lengthAware+" parallel="+parallel+" port="+p+" cell="+c);
                }
            }
        }
    }
    @Test void genuinelyFusedSharedCarryForwardExecutesSelectedRowStrategies() {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean symbolic : List.of(false,true))
                for (boolean lengthAware : List.of(false,true)) for (boolean parallel : List.of(false,true)) for (boolean vector : List.of(false,true)) {
            if (symbolic && !lengthAware && type == DataType.FLOAT64) continue;
            var preference = vector ? CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE
                    : CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.SCALAR;
            var config = new CpuPartitionAnalysisInputs.PortableExecutionConfig(preference,parallel ? 4 : 1,4,2);
            Dimension b = symbolic ? new DynamicDimension("B") : new StaticDimension(7);
            Tensor h = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(2)), p = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(2,2));
            Tensor x = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.ofDimensions(b,new StaticDimension(2)));
            Tensor negative = x.neg(), sum = negative.sumToShape(Shape.of(2)), product = x.permute(1,0).matmul(negative);
            var body = new ScanBody(List.of(h,p),List.of(x),List.of(),List.of(h.add(sum),p.add(product)),List.of(negative));
            Tensor rowCarry = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.ofDimensions(b,new StaticDimension(2)));
            if (lengthAware) body = new ScanBody(List.of(rowCarry,h,p),List.of(x),List.of(),List.of(x,h.add(sum),p.add(product)),List.of(negative));
            Tensor initial = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(2));
            Tensor initialProduct = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(2,2));
            Tensor sequence = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.ofDimensions(new StaticDimension(4),b,new StaticDimension(2)));
            Tensor lengths = CpuSymbolicLengthScanFixture.tensor(DataType.INT64,false,Shape.ofDimensions(b));
            Tensor initialRows = CpuSymbolicLengthScanFixture.tensor(type,false,rowCarry.descriptor().shape());
            var scan = lengthAware ? Scan.apply(List.of(initialRows,initial,initialProduct),List.of(sequence),List.of(),lengths,1,
                    List.of(java.util.OptionalInt.of(0)),body,false)
                    : Scan.apply(List.of(initial,initialProduct),List.of(sequence),List.of(),body,false);
            var roots = List.of(scan.finalCarry().get(lengthAware ? 1 : 0),scan.finalCarry().get(lengthAware ? 2 : 1),scan.outputs().getFirst());
            var compiled = CpuGeneralScanCapabilityTest.compile(roots,java.util.Optional.empty(),false,new CpuCapabilityProvider());
            var analysisInputs = new CpuPartitionAnalysisInputs(false,List.of(),config);
            var analysis = new CpuPartitionPreparer().analyze(GraphPreparation.project(compiled,compiled.partitions().getFirst(),analysisInputs));
            var unit = analysis.plan().units().stream().filter(u -> u.runtimeFacts().generalScan().isPresent()).findFirst().orElseThrow();
            var strategy = parallel ? vector ? PARALLEL_VECTOR : PARALLEL_SCALAR : vector ? VECTOR : SCALAR;
            assertTrue(unit.runtimeFacts().generalScan().orElseThrow().ir().stages().orElseThrow().rowMapReductionIr().isPresent(),"forward must genuinely fuse");
            assertEquals(strategy,unit.executionStrategy());
            for (int rows : symbolic ? List.of(0,1,7) : List.of(7)) try (var run =
                    new io.github.pho001.synaptik.backend.cpu.internal.executable.CpuRecurrentScanExecutionTest.Fixture(
                            compiled,analysisInputs,analysis,1,true,true,java.util.Map.of(),symbolic ? java.util.OptionalLong.of(rows) : java.util.OptionalLong.empty())) {
                for (var input : compiled.constants().bindableInputBindings()) {
                    var s = run.storage.get(input.valueId());
                    if (input.tensorId().equals(lengths.id())) for (int r = 0; r < rows; r++) s.setAtIndex(java.lang.foreign.ValueLayout.JAVA_LONG,r,r%5);
                    else for (int c = 0; c < s.byteSize()/type.byteWidth(); c++) put(type,s,c,input.tensorId().equals(sequence.id()) ? (c%9-4)*.125 : .25);
                }
                run.executable.bind(run.state).execute();
                var expectedSum = new double[]{.25,.25}; var expectedProduct = new double[]{.25,.25,.25,.25}; var expectedOutput = new double[8*rows];
                for (int t = 0; t < 4; t++) {
                    var stepSum = new double[2]; var stepProduct = new double[4]; boolean active = false;
                    for (int r = 0; r < rows; r++) if (!lengthAware || t < r%5) {
                        active = true;
                        for (int j = 0; j < 2; j++) {
                            double v = -(((t*rows+r)*2+j)%9-4)*.125;
                            expectedOutput[(t*rows+r)*2+j] = v; stepSum[j] += v;
                            for (int k = 0; k < 2; k++) stepProduct[2*k+j] += ((((t*rows+r)*2+k)%9-4)*.125)*v;
                        }
                    }
                    if (active || !lengthAware) {
                        for (int j = 0; j < 2; j++) expectedSum[j] += stepSum[j];
                        for (int j = 0; j < 4; j++) expectedProduct[j] += stepProduct[j];
                    }
                }
                var expected = List.of(expectedSum,expectedProduct,expectedOutput);
                for (int out = 0; out < 3; out++) {
                    var s = run.storage.get(compiled.publication().forwardBindings().get(out).valueId());
                    for (int c = 0; c < expected.get(out).length; c++) assertEquals(expected.get(out)[c],get(type,s,c),0,
                            type+" "+strategy+" symbolic="+symbolic+" lengths="+lengthAware+" B="+rows+" output="+out+" cell="+c);
                }
            }
        }
    }
    @Test void acceptedStaticAllValidFloat64ActualReverseExecutesAllStrategies() {
        var compiled = CpuConditionalScanFixture.conditionalFamily(DataType.FLOAT64,false);
        for (boolean parallel : List.of(false,true)) for (boolean vector : List.of(false,true)) {
            var preference = vector ? CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE
                    : CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.SCALAR;
            var config = new CpuPartitionAnalysisInputs.PortableExecutionConfig(preference,parallel ? 4 : 1,4,2);
            var analysisInputs = new CpuPartitionAnalysisInputs(false,List.of(),config);
            var analysis = new CpuPartitionPreparer().analyze(GraphPreparation.project(compiled,compiled.partitions().getFirst(),analysisInputs));
            var strategy = parallel ? vector ? PARALLEL_VECTOR : PARALLEL_SCALAR : vector ? VECTOR : SCALAR;
            try (var run = new io.github.pho001.synaptik.backend.cpu.internal.executable.CpuRecurrentScanExecutionTest.Fixture(
                    compiled,analysisInputs,analysis,1,true,true,java.util.Map.of())) {
                double[] x = new double[84], w = new double[6];
                for (int c = 0; c < x.length; c++) x[c] = .01*c;
                for (int c = 0; c < w.length; c++) w[c] = 1+.1*c;
                for (var binding : compiled.constants().bindableInputBindings()) {
                    var s = run.storage.get(binding.valueId()); int cells = (int)(s.byteSize()/8);
                    for (int c = 0; c < cells; c++) put(DataType.FLOAT64,s,c,cells == 84 ? x[c] : cells == 6 ? w[c] : 1);
                }
                run.executable.bind(run.state).execute();
                var initial = new double[14]; java.util.Arrays.fill(initial,5);
                var dx = new double[84]; var dw = new double[6];
                for (int t = 3; t >= 0; t--) {
                    double[] step = new double[6];
                    for (int b = 0; b < 7; b++) for (int k = 0; k < 3; k++) {
                        dx[(t*7+b)*3+k] = (0+(5-t)*w[2*k])+(5-t)*w[2*k+1];
                        for (int j = 0; j < 2; j++) step[2*k+j] += x[(t*7+b)*3+k]*(5-t);
                    }
                    for (int c = 0; c < 6; c++) dw[c] += step[c];
                }
                var expected = List.of(initial,dx,dw);
                for (var gradient : compiled.publication().gradientBindings()) {
                    var s = run.storage.get(gradient.valueId()); var values = expected.get(gradient.targetIndex());
                    for (int c = 0; c < values.length; c++) assertEquals(values[c],get(DataType.FLOAT64,s,c),1e-12,strategy+" target="+gradient.targetIndex()+" cell="+c);
                }
            }
        }
    }
    @Test void actualOrdinaryF3H2DerivativeExecutesAllStrategies() {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean symbolic : List.of(false,true)) {
            if (symbolic && type == DataType.FLOAT64) continue;
            for (boolean parallel : List.of(false,true)) for (boolean vector : List.of(false,true)) {
                var preference = vector ? CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE
                        : CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.SCALAR;
                var config = new CpuPartitionAnalysisInputs.PortableExecutionConfig(preference,parallel ? 4 : 1,4,2);
                Dimension r = symbolic ? new DynamicDimension("B") : new StaticDimension(7);
                Tensor x = CpuSymbolicLengthScanFixture.tensor(type,true,Shape.ofDimensions(r,new StaticDimension(3)));
                Tensor h = CpuSymbolicLengthScanFixture.tensor(type,true,Shape.ofDimensions(r,new StaticDimension(2)));
                Tensor w = CpuSymbolicLengthScanFixture.tensor(type,true,Shape.of(3,2));
                Tensor u = CpuSymbolicLengthScanFixture.tensor(type,true,Shape.of(2,2));
                Tensor bias = CpuSymbolicLengthScanFixture.tensor(type,true,Shape.of(2));
                Tensor y = x.matmul(w).add(h.matmul(u)).add(bias).tanh(), loss = y.mul(y).sum();
                var inputs = List.of(x,h,w,u,bias);
                var compiled = CpuGeneralScanCapabilityTest.compile(List.of(loss),java.util.Optional.of(
                        CpuGeneralScanCapabilityTest.request(List.of(loss),inputs)),false,new CpuCapabilityProvider());
                var analysisInputs = new CpuPartitionAnalysisInputs(false,List.of(),config);
                var analysis = new CpuPartitionPreparer().analyze(GraphPreparation.project(compiled,compiled.partitions().getFirst(),analysisInputs));
                var regions = analysis.plan().units().stream().filter(unit -> unit.portablePlan().specialization().rowMapReductionIr().isPresent()).toList();
                assertFalse(regions.isEmpty(),"actual ordinary F3/H2 derivative must fuse");
                var strategy = parallel ? vector ? PARALLEL_VECTOR : PARALLEL_SCALAR : vector ? VECTOR : SCALAR;
                assertTrue(regions.stream().anyMatch(unit -> unit.executionStrategy().equals(strategy)),"actual ordinary selected "+strategy);
                for (int b : symbolic ? List.of(0,1,7,32) : List.of(7)) for(int carrier:List.of(0,1)) try (var run =
                        new io.github.pho001.synaptik.backend.cpu.internal.executable.CpuRecurrentScanExecutionTest.Fixture(
                                compiled,analysisInputs,analysis,carrier,true,true,java.util.Map.of(),symbolic ? java.util.OptionalLong.of(b) : java.util.OptionalLong.empty())) {
                    var data = new java.util.ArrayList<double[]>();
                    for (var binding : compiled.constants().bindableInputBindings()) if (inputs.stream().noneMatch(t -> t.id().equals(binding.tensorId()))) {
                        var segment = run.storage.get(binding.valueId());
                        for (int c = 0; c < segment.byteSize()/type.byteWidth(); c++) put(type,segment,c,1);
                    }
                    for (var input : inputs) {
                        var binding = compiled.constants().bindableInputBindings().stream().filter(p -> p.tensorId().equals(input.id())).findFirst().orElseThrow();
                        var segment = run.storage.get(binding.valueId()); var cells = new double[(int)(segment.byteSize()/type.byteWidth())];
                        for (int c = 0; c < cells.length; c++) { cells[c] = rounded(type,(c%11-5)*.035); put(type,segment,c,cells[c]); }
                        data.add(cells);
                    }
                    run.executable.bind(run.state).execute();
                    var expected = ordinaryDerivative(type,b,data);
                    for (var gradient : compiled.publication().gradientBindings()) {
                        var actual = run.storage.get(gradient.valueId()); var target = expected.get(gradient.targetIndex());
                        assertEquals(target.length,actual.byteSize()/type.byteWidth());
                        for (int c = 0; c < target.length; c++) assertEquals(target[c],get(type,actual,c),type == DataType.FLOAT64 ? 2e-14 : 2e-6,
                                type+" "+strategy+" B="+b+" target="+gradient.targetIndex()+" cell="+c);
                    }
                }
            }
        }
    }

    /**
     * Independently differentiates squared TANH loss; no IR, generator or candidate outputs.
     * This semantic oracle is deliberately not the timed optimal Java arm.
     * @param type represented per-site rounding
     * @param rows actual B
     * @param data x,h,W,U,b in dense logical order
     * @return independent dx,dh,dW,dU,db arrays
     */
    private static List<double[]> ordinaryDerivative(DataType type,int rows,List<double[]> data) {
        var dx = new double[3*rows]; var dh = new double[2*rows]; var dw = new double[6]; var du = new double[4]; var db = new double[2];
        var x = data.get(0); var h = data.get(1); var w = data.get(2); var u = data.get(3); var bias = data.get(4);
        for (int r = 0; r < rows; r++) {
            var delta = new double[2];
            for (int j = 0; j < 2; j++) {
                double a = 0,c = 0;
                for (int k = 0; k < 3; k++) a = rounded(type,a+rounded(type,x[3*r+k]*w[2*k+j]));
                for (int k = 0; k < 2; k++) c = rounded(type,c+rounded(type,h[2*r+k]*u[2*k+j]));
                double y = rounded(type,StrictMath.tanh(rounded(type,rounded(type,a+c)+bias[j])));
                delta[j] = rounded(type,rounded(type,y+y)*rounded(type,1-rounded(type,y*y)));
                db[j] = rounded(type,db[j]+delta[j]);
                for (int k = 0; k < 3; k++) dw[2*k+j] = rounded(type,dw[2*k+j]+rounded(type,x[3*r+k]*delta[j]));
                for (int k = 0; k < 2; k++) du[2*k+j] = rounded(type,du[2*k+j]+rounded(type,h[2*r+k]*delta[j]));
            }
            for (int k = 0; k < 3; k++) for (int j = 0; j < 2; j++) dx[3*r+k] = rounded(type,dx[3*r+k]+rounded(type,delta[j]*w[2*k+j]));
            for (int k = 0; k < 2; k++) for (int j = 0; j < 2; j++) dh[2*r+k] = rounded(type,dh[2*r+k]+rounded(type,delta[j]*u[2*k+j]));
        }
        return List.of(dx,dh,dw,du,db);
    }
    @Test void actualCombinedReversePreservesAllRequestedPolicies() {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (int domain = 0; domain < 4; domain++)
                for (boolean parallel : List.of(false,true)) for (boolean vector : List.of(false,true)) {
            boolean symbolic = domain < 2, lengths = domain%2 == 0;
            var preference = vector ? io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE
                    : io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.SCALAR;
            var config = new io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig(preference,parallel ? 4 : 1,4,2);
            var family = new CpuSymbolicLengthScanFixture.Family(type,false,4,3,config,symbolic,lengths,7);
            for (int b : symbolic ? List.of(0,1,7,32) : List.of(7))
                try (var reference = new CpuSymbolicLengthScanFixture(family,b)) { reference.verify(); }
        }
    }
    @Test void selectedLayoutSnapshotsOrdinalsAndRejectsInvalidOrUnimplementedVectorFacts() {
        var ordinals = new java.util.ArrayList<>(List.of(1));
        var layout = new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.RowFusionEmission(
                2,ordinals,List.of(),0,0);
        ordinals.clear(); assertEquals(List.of(1),layout.invariantTransposeValues());
        assertThrows(UnsupportedOperationException.class,() -> layout.invariantTransposeValues().clear());
        for (int bad = 0; bad < 5; bad++) {
            int mutation = bad;
            assertThrows(IllegalArgumentException.class,() ->
                    new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.RowFusionEmission(
                            mutation == 0 ? 0 : 2,mutation == 1 ? List.of(1,1) : mutation == 2 ? List.of(-1) : List.of(1),
                            mutation == 3 ? List.of(1) : List.of(),mutation == 4 ? -1 : 0,0));
        }
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) {
            var ir = io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRowMapReductionLoweringTest.candidate(type,false,7);
            var strategy = io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR;
            int bits = type == DataType.FLOAT64 ? DoubleVector.SPECIES_PREFERRED.vectorBitSize()
                    : FloatVector.SPECIES_PREFERRED.vectorBitSize();
            var spec = new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization(
                    io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint.fromHex(ir.structuralKey()),strategy,
                    java.util.Collections.nCopies(4,type),java.util.Collections.nCopies(4,CarrierAccess.MEMORY_SEGMENT),bits,-1,List.of(),
                    true,72,java.util.Optional.empty(),java.util.Optional.empty(),false,false,java.util.Optional.of(ir),
                    java.util.Optional.of(new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.RowFusionEmission(
                            1,List.of(),List.of(),0,0)));
            var bytes = new CpuClassFileKernelGenerator().generateClassBytes(spec,ir.encodedKernelIr());
            assertTrue(ClassFile.of().parse(bytes).methods().stream().flatMap(m -> m.code().stream())
                    .flatMap(c -> c.elementStream()).anyMatch(e -> e instanceof java.lang.classfile.instruction.InvokeInstruction call
                        && call.owner().asInternalName().startsWith("jdk/incubator/vector/") && call.name().equalsString("mul")));
            assertTrue(new String(spec.compatibilityBytes(),java.nio.charset.StandardCharsets.US_ASCII)
                    .contains("|ordinary-row-map-multi-reduction-exec-v1="));
            assertTrue(new String(spec.classIdentityBytes(),java.nio.charset.StandardCharsets.US_ASCII)
                    .contains("|ordinary-row-map-multi-reduction-exec-v1="));
        }
    }

    @Test void ordinaryScalarRangeHelpersPublishPrivatePartialsAndMergeInRangeOrder() throws Throwable {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (int rows : List.of(0,1,7,32)) {
            var ir = io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRowMapReductionLoweringTest.candidate(type,false,rows);
            var boundaries = ir.arithmetic().values().stream().filter(v -> v.kind()
                    != io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr.Value.Kind.VIRTUAL).toList();
            var spec = new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization(
                    io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                    io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionStrategy.PARALLEL_SCALAR,
                    boundaries.stream().map(v -> v.dataType()).toList(),java.util.Collections.nCopies(4,CarrierAccess.MEMORY_SEGMENT),
                    0,-1,List.of(),true,72,java.util.Optional.empty(),java.util.Optional.empty(),false,false,java.util.Optional.of(ir),
                    java.util.Optional.of(new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.RowFusionEmission(
                            2,List.of(),List.of(),0,0)));
            var generator = new CpuClassFileKernelGenerator();
            var kernel = generator.defineClassBytes(spec, ir.encodedKernelIr(), generator.generateClassBytes(spec, ir.encodedKernelIr()));
            var range = kernel.ordinaryRowRangeEntryPoint().asSpreader(0,MemorySegment[].class,5);
            var merge = kernel.ordinaryRowMergeEntryPoint().asSpreader(0,MemorySegment[].class,5);
            assertEquals(3,ClassFile.of().parse(kernel.classBytes()).methods().size());
            int width = type.byteWidth(), slice = 8+4*width;
            try (var arena = java.lang.foreign.Arena.ofShared()) {
                var x = arena.allocate(2L*rows*width,width); var w = arena.allocate(4L*width,width);
                var out = arena.allocate(x.byteSize(),width); var terminal = arena.allocate(w.byteSize(),width);
                var scratch = arena.allocate(2L*slice,8); scratch.fill((byte)0x7f);
                for (int i = 0; i < 2*rows; i++) put(type,x,i,(i%9-4)*.25);
                for (int i = 0; i < 4; i++) put(type,w,i,i%3 == 0 ? 1 : 0);
                var operands = new MemorySegment[]{x,w,out,terminal,scratch};
                long[] first = {rows,0,0,0,0,2,0}, second = {rows,0,0,0,0,2,slice};
                int middle = (rows+1)/2;
                range.invokeExact(operands,first,0L,(long)middle);
                range.invokeExact(operands,second,(long)middle,(long)rows);
                assertEquals(middle > 0 ? 1L : 0L,scratch.get(java.lang.foreign.ValueLayout.JAVA_LONG,0));
                assertEquals(rows > middle ? 1L : 0L,scratch.get(java.lang.foreign.ValueLayout.JAVA_LONG,slice));
                // Invalid partials may contain poisoned data; the validity flag dominates reads.
                if (rows == 0 || rows == 1) for (int i = 0; i < 4; i++) put(type,scratch,(slice+8)/width+i,Double.NaN);
                merge.invokeExact(operands,first,0L,1L);
                for (int i = 0; i < 2*rows; i++) assertEquals(2*get(type,x,i),get(type,out,i));
                for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) {
                    double left = 0,right = 0;
                    for (int r = 0; r < middle; r++) left = rounded(type,left+rounded(type,get(type,x,2*r+a)*rounded(type,2*get(type,x,2*r+b))));
                    for (int r = middle; r < rows; r++) right = rounded(type,right+rounded(type,get(type,x,2*r+a)*rounded(type,2*get(type,x,2*r+b))));
                    double expected = rows > middle ? rounded(type,left+right) : left;
                    assertEquals(expected,get(type,terminal,2*a+b));
                }
            }
        }
    }

    /** Copies only authentic public inputs and compiled constants into another production invocation. */
    static void copyReferenceInputs(CpuSymbolicLengthScanFixture reference,CpuRecurrentScanExecutionTest.Fixture run) {
        var family=reference.family;
        var sources=List.of(reference.initial,reference.x,reference.w,reference.u,reference.bias,reference.lengths,reference.fs,reference.es);
        for (var binding:family.compiled.constants().bindableInputBindings()) {
            int position=family.inputs.indexOf(binding.tensorId());
            if (position>=0) run.storage.get(binding.valueId()).copyFrom(sources.get(position));
        }
    }

    /**
     * Stores a typed test operand without changing the production resource layout.
     * @param type homogeneous F32/F64 represented dtype
     * @param segment accessible writable test carrier
     * @param cell logical cell offset
     * @param value test value rounded once on F32 storage
     */
    private static void put(DataType type,MemorySegment segment,int cell,double value) {
        if (type == DataType.FLOAT64) segment.setAtIndex(java.lang.foreign.ValueLayout.JAVA_DOUBLE,cell,value);
        else segment.setAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT,cell,(float)value);
    }

    /**
     * Reads one represented test cell.
     * @param type homogeneous F32/F64 dtype
     * @param segment accessible test carrier
     * @param cell logical cell offset
     * @return exact represented test value widened only for assertion transport
     */
    private static double get(DataType type,MemorySegment segment,int cell) {
        return type == DataType.FLOAT64 ? segment.getAtIndex(java.lang.foreign.ValueLayout.JAVA_DOUBLE,cell)
                : segment.getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT,cell);
    }

    /**
     * Models one declared represented arithmetic site, independently of the emitter.
     * @param type producing dtype
     * @param value mathematical test-site result before producing conversion
     * @return one F32 rounding or the existing F64 result
     */
    private static double rounded(DataType type,double value) { return type == DataType.FLOAT32 ? (float)value : value; }


    @Test void generatedMaskedMathUsesActualFeaturesOnlyOnArraysAndSegments() throws Throwable {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) {
            int lanes = type == DataType.FLOAT64 ? DoubleVector.SPECIES_PREFERRED.length()
                    : FloatVector.SPECIES_PREFERRED.length();
            for (boolean segment : List.of(false,true)) for (boolean broadcast : List.of(false,true))
                for (int features : List.of(0,1,2,3,lanes-1,lanes)) {
                    if (features > lanes || features == 0 && broadcast) continue;
                    for (int mode = 0; mode < 3; mode++) {
                        byte[] bytes = seam(type,segment,broadcast,features,mode);
                        assertTrue(ClassFile.of().verify(bytes).isEmpty());
                        var lookup = MethodHandles.lookup().defineHiddenClass(bytes,true);
                        Class<?> carrier = segment ? MemorySegment.class
                                : type == DataType.FLOAT64 ? double[].class : float[].class;
                        var entry = lookup.findStatic(lookup.lookupClass(),"run",MethodType.methodType(void.class,
                                carrier,carrier,carrier,long.class,long.class));
                        if (type == DataType.FLOAT64) {
                            double[] a = new double[broadcast ? 1 : features], b = new double[features];
                            double[] out = new double[features+2]; java.util.Arrays.fill(out,-123.5);
                            for (int i = 0; i < a.length; i++) a[i] = (i-2)*.125;
                            for (int i = 0; i < b.length; i++) b[i] = (i+1)*.375;
                            if (mode == 2 && a.length > 0) a[0] = -0.0;
                            entry.invokeWithArguments(segment ? MemorySegment.ofArray(a) : a,
                                    segment ? MemorySegment.ofArray(b) : b,
                                    segment ? MemorySegment.ofArray(out) : out,0L,1L);
                            for (int i = 0; i < features; i++) {
                                double x = a[broadcast ? 0 : i], expected = mode == 0 ? (0d+x*b[i])+x*b[i]
                                        : mode == 1 ? (0d+x)+x : x;
                                assertEquals(Double.doubleToRawLongBits(expected),Double.doubleToRawLongBits(out[i]));
                            }
                            assertEquals(-123.5,out[features]); assertEquals(-123.5,out[features+1]);
                        } else {
                            float[] a = new float[broadcast ? 1 : features], b = new float[features];
                            float[] out = new float[features+2]; java.util.Arrays.fill(out,-123.5f);
                            for (int i = 0; i < a.length; i++) a[i] = (i-2)*.125f;
                            for (int i = 0; i < b.length; i++) b[i] = (i+1)*.375f;
                            if (mode == 2 && a.length > 0) a[0] = -0.0f;
                            entry.invokeWithArguments(segment ? MemorySegment.ofArray(a) : a,
                                    segment ? MemorySegment.ofArray(b) : b,
                                    segment ? MemorySegment.ofArray(out) : out,0L,1L);
                            for (int i = 0; i < features; i++) {
                                float x = a[broadcast ? 0 : i], expected = mode == 0 ? (0f+x*b[i])+x*b[i]
                                        : mode == 1 ? (0f+x)+x : x;
                                assertEquals(Float.floatToRawIntBits(expected),Float.floatToRawIntBits(out[i]));
                            }
                            assertEquals(-123.5f,out[features]); assertEquals(-123.5f,out[features+1]);
                        }
                        assertSeamStructure(bytes,mode);
                    }
                }
        }
    }

    /**
     * Emits test-only vector code through the production arithmetic/carrier seams. No runtime
     * callback, compiler-generated oracle, or production route selection is involved.
     * @param type separately tested represented floating dtype
     * @param segment whether all arguments are checked segments rather than typed arrays
     * @param broadcast whether the first operand is one scalar cell
     * @param features real cells in one preferred-species block, possibly zero
     * @param mode zero for product/add, one for represented SUM, two for raw singleton
     * @return verified-by-caller deterministic test ClassFile bytes
     */
    private static byte[] seam(DataType type, boolean segment, boolean broadcast, int features, int mode) {
        var primitive = type == DataType.FLOAT64 ? ConstantDescs.CD_double : ConstantDescs.CD_float;
        var vector = ClassDesc.of("jdk.incubator.vector."+(type == DataType.FLOAT64 ? "DoubleVector" : "FloatVector"));
        var species = ClassDesc.of("jdk.incubator.vector.VectorSpecies");
        var mask = ClassDesc.of("jdk.incubator.vector.VectorMask");
        var carrier = segment ? ClassDesc.of(MemorySegment.class.getName()) : primitive.arrayType();
        var access = segment ? CarrierAccess.MEMORY_SEGMENT
                : type == DataType.FLOAT64 ? CarrierAccess.DOUBLE_ARRAY : CarrierAccess.FLOAT_ARRAY;
        return ClassFile.of().build(ClassDesc.of(CpuRowFusionSimdParallelTest.class.getPackageName()+".Cpu0074Seam"),c -> {
            c.withVersion(ClassFile.JAVA_26_VERSION,0).withFlags(AccessFlag.FINAL);
            c.withMethod("run",MethodTypeDesc.of(ConstantDescs.CD_void,carrier,carrier,carrier,
                    ConstantDescs.CD_long,ConstantDescs.CD_long),AccessFlag.STATIC.mask(),m -> m.withCode(code -> {
                CpuCarrierEmitter.prepareSegmentLayouts(code,List.of(type,type,type),List.of(access,access,access),true);
                var emitter = new CpuCarrierEmitter(code);
                emitter.prepareVectorSpecies(type);
                if (segment) emitter.prepareVectorByteOrder();
                int address = code.allocateLocal(TypeKind.LONG), tail = code.allocateLocal(TypeKind.REFERENCE);
                int sum = code.allocateLocal(TypeKind.REFERENCE);
                code.loadConstant(0L).lstore(address).getstatic(vector,"SPECIES_PREFERRED",species)
                        .loadConstant(features == 64 ? -1L : (1L<<features)-1)
                        .invokestatic(mask,"fromLong",MethodTypeDesc.of(mask,species,ConstantDescs.CD_long)).astore(tail);
                Runnable left = () -> emitter.vectorLoad(type,access,0,address,broadcast,false,tail);
                Runnable right = () -> emitter.vectorLoad(type,access,1,address,false,false,tail);
                if (mode == 0) {
                    CpuMatmulEmitter.emitVectorInit(code,type,sum);
                    CpuMatmulEmitter.emitVectorContributorStep(code,type,sum,false,left,right);
                    CpuMatmulEmitter.emitVectorContributorStep(code,type,sum,false,left,right);
                    CpuMatmulEmitter.emitVectorFinish(code,type,sum,
                            local -> emitter.vectorStore(type,access,2,address,local,false,tail));
                } else {
                    CpuAggregateEmitter.emitRepresentedVectorInit(code,type,sum,mode == 2);
                    CpuAggregateEmitter.emitRepresentedVectorContributorStep(code,type,sum,mode == 2,left);
                    if (mode == 1) CpuAggregateEmitter.emitRepresentedVectorContributorStep(code,type,sum,false,left);
                    CpuAggregateEmitter.emitRepresentedVectorFinish(code,type,sum,
                            local -> emitter.vectorStore(type,access,2,address,local,false,tail));
                }
                code.return_();
            }));
        });
    }

    /**
     * Checks the seams preserve separate arithmetic sites and introduce no hidden allocation,
     * reflection, boxing, dispatch or horizontal reduction.
     * @param bytes actual generated test ClassFile
     * @param mode tested math algorithm
     */
    private static void assertSeamStructure(byte[] bytes,int mode) {
        int products = 0, additions = 0;
        for (var element : ClassFile.of().parse(bytes).methods().getFirst().code().orElseThrow()) {
            if (element instanceof java.lang.classfile.Instruction i)
                assertFalse(java.util.Set.of(java.lang.classfile.Opcode.NEW,java.lang.classfile.Opcode.NEWARRAY,
                        java.lang.classfile.Opcode.ANEWARRAY,java.lang.classfile.Opcode.INVOKEDYNAMIC).contains(i.opcode()));
            if (element instanceof java.lang.classfile.instruction.InvokeInstruction call) {
                String name = call.name().stringValue();
                assertFalse(name.toLowerCase(java.util.Locale.ROOT).contains("fma"));
                assertFalse(name.contains("reduceLanes"));
                if (name.equals("mul")) products++;
                if (name.equals("add")) additions++;
                assertFalse(call.owner().asInternalName().startsWith("io/github/pho001"),"no hidden runtime helper");
            }
        }
        assertEquals(mode == 0 ? 2 : 0,products);
        assertEquals(mode == 2 ? 0 : 2,additions);
    }
}
