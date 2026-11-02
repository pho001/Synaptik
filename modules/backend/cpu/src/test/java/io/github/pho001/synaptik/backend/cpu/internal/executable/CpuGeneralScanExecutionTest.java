package io.github.pho001.synaptik.backend.cpu.internal.executable;

import static org.junit.jupiter.api.Assertions.*;
import static io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLoweringTest.*;
import io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanGeneratedKernelTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.model.tensor.*;
import java.util.*;
import org.junit.jupiter.api.Test;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.CpuGeneralScanCapabilityTest;
import io.github.pho001.synaptik.compiler.FunctionalGradientRequest;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.foreign.*;

/** Real assigned-slot execution through one ordinary RunState, including temporal old carry. */
public class CpuGeneralScanExecutionTest {
    @Test void symbolicLengthMixedRowOneTanhSigmoidDerivativesUsePublicSpecialValuesAndSkipPadding() {
        for (DataType type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean reverse : List.of(false,true)) {
            var b = new io.github.pho001.synaptik.model.shape.DynamicDimension("B");
            var row = io.github.pho001.synaptik.model.shape.Shape.ofDimensions(b,new io.github.pho001.synaptik.model.shape.StaticDimension(2));
            var stack = io.github.pho001.synaptik.model.shape.Shape.ofDimensions(new io.github.pho001.synaptik.model.shape.StaticDimension(4),b,new io.github.pho001.synaptik.model.shape.StaticDimension(2));
            Tensor h = symbolic(type,true,row), x = symbolic(type,true,row), initial = symbolic(type,true,row), seq = symbolic(type,true,stack);
            Tensor t = h.add(x).permute(1,0).tanh(), s = t.sigmoid(), next = s.permute(1,0);
            var body = new ScanBody(List.of(h),List.of(x),List.of(),List.of(next),List.of(next));
            Tensor lengths = symbolic(DataType.INT64,false,io.github.pho001.synaptik.model.shape.Shape.ofDimensions(b));
            var scan = Scan.apply(List.of(initial),List.of(seq),List.of(),lengths,1,List.of(OptionalInt.of(0)),body,reverse);
            Tensor fs = symbolic(type,false,row), es = symbolic(type,false,stack);
            var roots = List.of(scan.finalCarry().getFirst(),scan.outputs().getFirst());
            var artifact = CpuGeneralScanCapabilityTest.compile(roots,Optional.of(gradientRequest(roots,List.of(fs,es),List.of(initial,seq))),false,new CpuCapabilityProvider());
            var context = io.github.pho001.synaptik.prepare.GraphPreparation.project(artifact,artifact.partitions().getFirst(),CpuPartitionAnalysisInputs.DEFAULT);
            var analysis = new CpuPartitionPreparer().analyze(context);
            try (var f = new CpuRecurrentScanExecutionTest.Fixture(artifact,CpuPartitionAnalysisInputs.DEFAULT,analysis,1,true,true,Map.of(),OptionalLong.of(4))) {
                for (var binding : artifact.constants().bindableInputBindings()) {
                    int id = Math.toIntExact(binding.valueId().value());
                    if (binding.tensorId().equals(lengths.id())) f.lengths(id,1,1,1,0);
                    else fillTyped(f,type,id,binding.tensorId().equals(fs.id()) ? 1 : 0);
                }
                double[] source = {.25,-0d,Double.NaN,0};
                for (var binding : artifact.constants().bindableInputBindings()) {
                    var segment=f.storage.get(binding.valueId());
                    if (binding.tensorId().equals(initial.id()))
                        for (int rowIndex=0; rowIndex<4; rowIndex++) for (int feature=0; feature<2; feature++)
                            put(type,segment,rowIndex*2+feature,source[rowIndex]);
                    if (binding.tensorId().equals(seq.id())) {
                        fillTyped(f,type,Math.toIntExact(binding.valueId().value()),Double.NaN);
                        for (int rowIndex=0; rowIndex<3; rowIndex++) for (int feature=0; feature<2; feature++)
                            put(type,segment,rowIndex*2+feature,0);
                    }
                }
                f.executable.bind(f.state).execute();
                var gradient = f.storage.get(artifact.publication().gradientBindings().getFirst().valueId());
                for (int rowIndex = 0; rowIndex < 4; rowIndex++) for (int feature = 0; feature < 2; feature++) {
                    double tv=work(type,Math.tanh(work(type,source[rowIndex]+0d)));
                    double sv=work(type,1d/(1d+Math.exp(-tv)));
                    double expected = rowIndex == 3 ? 1 : work(type,work(type,1-work(type,tv*tv))*work(type,sv*work(type,1-sv)));
                    double observed = type == DataType.FLOAT64 ? gradient.getAtIndex(ValueLayout.JAVA_DOUBLE,rowIndex*2+feature) : gradient.getAtIndex(ValueLayout.JAVA_FLOAT,rowIndex*2+feature);
                    assertEquals(expected,observed,type == DataType.FLOAT64 ? 1e-14 : 1e-6);
                }
            }
        }
    }

    private static Tensor symbolic(DataType type,boolean gradient,io.github.pho001.synaptik.model.shape.Shape shape) {
        return TensorFactory.create(new TensorDescriptor(type,shape,Optional.empty(),gradient));
    }

    @Test void symbolicLengthPreflightUsesExplicitBatchBehindZeroFeaturesAndRejectsInvalidLastBeforeWrites() {
        for (DataType type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (int time : new int[]{0,4}) for (int width : new int[]{0,2}) {
            var b = new io.github.pho001.synaptik.model.shape.DynamicDimension("rows");
            var feature = new io.github.pho001.synaptik.model.shape.StaticDimension(width);
            java.util.function.Function<io.github.pho001.synaptik.model.shape.Shape,Tensor> value = shape -> TensorFactory.create(
                    new TensorDescriptor(type,shape,shape.isFullyStatic() ? Optional.of(io.github.pho001.synaptik.model.layout.LayoutDescriptor.contiguous(shape)) : Optional.empty(),false));
            var row = io.github.pho001.synaptik.model.shape.Shape.ofDimensions(b,feature);
            var sequence = io.github.pho001.synaptik.model.shape.Shape.ofDimensions(new io.github.pho001.synaptik.model.shape.StaticDimension(time),b,feature);
            Tensor h = value.apply(row), x = value.apply(row), initial = value.apply(row), seq = value.apply(sequence);
            Tensor lengths = TensorFactory.create(new TensorDescriptor(DataType.INT64,io.github.pho001.synaptik.model.shape.Shape.ofDimensions(b),Optional.empty(),false));
            Tensor next = h.add(x);
            var scan = Scan.apply(List.of(initial),List.of(seq),List.of(),lengths,1,List.of(OptionalInt.of(0)),
                    new ScanBody(List.of(h),List.of(x),List.of(),List.of(next),List.of(next)),true);
            var roots = new ArrayList<>(scan.finalCarry()); roots.addAll(scan.outputs()); roots.addAll(scan.residuals());
            var analysis = new CpuPartitionPreparer().analyze(CpuGeneralScanPreparationTest.context(query(roots)));
            var unit = analysis.plan().units().getFirst(); var facts = unit.runtimeFacts().generalScan().orElseThrow();
            assertTrue(facts.ir().symbolic()); assertEquals(0,facts.ir().stages().orElseThrow().batch());
            for (int batch : new int[]{0,1,7}) try (var f = new CpuRecurrentScanExecutionTest.Fixture(analysis,1,true,true,Map.of(),OptionalLong.of(batch))) {
                int lengthId = Math.toIntExact(unit.boundaryValues().get(facts.ir().lengthRead()).value());
                long[] ls = new long[batch]; f.lengths(lengthId,ls);
                var writes = unit.boundaryValues().subList(facts.ir().scalarReads().size(),unit.boundaryValues().size());
                writes.forEach(id -> f.storage.get(id).fill((byte)0x5a));
                var before = writes.stream().map(id -> f.storage.get(id).toArray(ValueLayout.JAVA_BYTE)).toList();
                var bound = f.executable.bind(f.state); // Producers and value validation occur AFTER cold binding.
                if (batch > 0) {
                    ls[batch-1] = time+1; f.lengths(lengthId,ls);
                    var failure = assertThrows(IllegalArgumentException.class,bound::execute);
                    assertTrue(failure.getMessage().contains("row "+(batch-1)));
                    for (int p = 0; p < writes.size(); p++) assertArrayEquals(before.get(p),f.storage.get(writes.get(p)).toArray(ValueLayout.JAVA_BYTE));
                    ls[batch-1] = 0; f.lengths(lengthId,ls);
                }
                bound.execute();
                for (int p : facts.ir().demandedPorts()) if (p >= facts.ir().carryCount())
                    for (byte cell : f.storage.get(unit.boundaryValues().get(facts.ir().scalarReads().size()+facts.ir().demandedPorts().indexOf(p))).toArray(ValueLayout.JAVA_BYTE))
                        assertEquals(0,cell,"all-zero L produces raw +0, including histories");
            }
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void lengthZeroKActualReverseStillProducesNonemptyBiasGradientAndRowResults(DataType type) {
        Tensor h = tensor(type,true,3,3), x = tensor(type,true,3,0), w = tensor(type,true,0,3), b = tensor(type,true,3);
        Tensor initial = tensor(type,true,3,3), seq = tensor(type,true,5,3,0), weight = tensor(type,true,0,3), bias = tensor(type,true,3);
        Tensor lengths = tensor(DataType.INT64,false,3), seed = tensor(type,false,3,3);
        var body = new ScanBody(List.of(h),List.of(x),List.of(w,b),List.of(h.add(x.matmul(w)).add(b)),List.of());
        var scan = Scan.apply(List.of(initial),List.of(seq),List.of(weight,bias),lengths,1,List.of(OptionalInt.of(0)),body,false);
        var artifact = CpuGeneralScanCapabilityTest.compile(scan.finalCarry(),Optional.of(gradientRequest(scan.finalCarry(),List.of(seed),List.of(bias))),false,new CpuCapabilityProvider());
        var analysis = new CpuPartitionPreparer().analyze(io.github.pho001.synaptik.prepare.GraphPreparation.project(
                artifact,artifact.partitions().getFirst(),CpuPartitionAnalysisInputs.DEFAULT));
        try (var f = new CpuRecurrentScanExecutionTest.Fixture(artifact,CpuPartitionAnalysisInputs.DEFAULT,analysis,2,true,true,Map.of())) {
            for (var binding : artifact.constants().bindableInputBindings()) {
                int id = Math.toIntExact(binding.valueId().value());
                if (binding.tensorId().equals(lengths.id())) f.lengths(id,5,3,0);
                else fillTyped(f,type,id,binding.tensorId().equals(bias.id()) ? .25 : binding.tensorId().equals(seed.id()) ? 1 : 0);
            }
            f.executable.bind(f.state).execute();
            assertArrayEquals(new double[]{1.25,1.25,1.25,.75,.75,.75,0,0,0},readTyped(f,type,Math.toIntExact(artifact.publication().forwardBindings().getFirst().valueId().value())));
            assertArrayEquals(new double[]{8,8,8},readTyped(f,type,Math.toIntExact(artifact.publication().gradientBindings().getFirst().valueId().value())));
        }
    }
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void lengthInPlaceProofDoesNotCarryContributorAvailabilityIntoSumEpilogue(DataType type) {
        for (boolean reverse : List.of(false,true)) for (int t : List.of(1,2)) {
            Tensor h = tensor(type,false,2,1), x = tensor(type,false,2,1);
            Tensor r = h.add(x), sum = r.sum(1,true), next = sum.neg(), emit = next.add(h);
            var body = new ScanBody(List.of(h),List.of(x),List.of(),List.of(next),List.of(emit));
            var scan = Scan.apply(List.of(tensor(type,false,2,1)),List.of(tensor(type,false,t,2,1)),List.of(),
                    tensor(DataType.INT64,false,2),1,List.of(OptionalInt.of(0)),body,reverse);
            var analysis = new CpuPartitionPreparer().analyze(CpuGeneralScanPreparationTest.context(query(List.of(scan.finalCarry().getFirst(),scan.outputs().getFirst()))));
            var plan = analysis.plan().units().getFirst().runtimeFacts().generalScan().orElseThrow().ir().stages().orElseThrow();
            assertEquals(-1,plan.carryPorts().getFirst(),"epilogue reload requires separate old/new state");
            try (var f = new CpuRecurrentScanExecutionTest.Fixture(analysis,2,true,true,Map.of())) {
                fillTyped(f,type,0,1); fillTyped(f,type,1,Double.NaN);
                for (int time = 0; time < t; time++) put(type,f.segment(1),time*2,2);
                f.lengths(2,t,0); f.executable.bind(f.state).execute();
                assertArrayEquals(new double[]{t == 1 ? -3 : 1,1},readTyped(f,type,3));
                assertArrayEquals(t == 1 ? new double[]{-2,0} : new double[]{-2,0,-2,0},readTyped(f,type,4));
            }
        }
    }
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void lengthActualEmptyBatchReversePublishesNonemptyPositiveZeroSharedGradientWithoutReadingNaNWeight(DataType type) {
        Tensor h = tensor(type,true,0,3), x = tensor(type,true,0,2), w = tensor(type,true,2,3);
        Tensor initial = tensor(type,true,0,3), seq = tensor(type,true,5,0,2), weight = tensor(type,true,2,3), lengths = tensor(DataType.INT64,false,0);
        var body = new ScanBody(List.of(h),List.of(x),List.of(w),List.of(h.add(x.matmul(w))),List.of());
        var scan = Scan.apply(List.of(initial),List.of(seq),List.of(weight),lengths,1,List.of(OptionalInt.of(0)),body,true);
        Tensor seed = tensor(type,false,0,3);
        var artifact = CpuGeneralScanCapabilityTest.compile(scan.finalCarry(),Optional.of(gradientRequest(scan.finalCarry(),List.of(seed),List.of(initial,seq,weight))),false,new CpuCapabilityProvider());
        var analysis = new CpuPartitionPreparer().analyze(io.github.pho001.synaptik.prepare.GraphPreparation.project(
                artifact,artifact.partitions().getFirst(),CpuPartitionAnalysisInputs.DEFAULT));
        try (var f = new CpuRecurrentScanExecutionTest.Fixture(artifact,CpuPartitionAnalysisInputs.DEFAULT,analysis,2,true,true,Map.of())) {
            for (var binding : artifact.constants().bindableInputBindings()) if (binding.tensorId().equals(weight.id())) fillTyped(f,type,Math.toIntExact(binding.valueId().value()),Double.NaN);
            f.executable.bind(f.state).execute();
            var gradient = logicalNumbers(artifact,f,artifact.publication().gradientBindings().get(2).valueId());
            assertArrayEquals(new double[6],gradient);
            for (double value : gradient) assertEquals(0L,Double.doubleToRawLongBits(value));
        }
    }
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void lengthRawInactiveStateAndActualInitialCotangentsPreserveNaNPayloadZeroAndSubnormalBits(DataType type) {
        Tensor h = tensor(type,true,3,4), x = tensor(type,false,3,4), initial = tensor(type,true,3,4), seq = tensor(type,false,5,3,4);
        Tensor lengths = tensor(DataType.INT64,false,3), seed = tensor(type,false,3,4);
        var body = new ScanBody(List.of(h),List.of(x),List.of(),List.of(h.add(x)),List.of());
        var scan = Scan.apply(List.of(initial),List.of(seq),List.of(),lengths,1,List.of(OptionalInt.of(0)),body,false);
        var request = gradientRequest(scan.finalCarry(),List.of(seed),List.of(initial));
        var artifact = CpuGeneralScanCapabilityTest.compile(scan.finalCarry(),Optional.of(request),false,new CpuCapabilityProvider());
        var analysis = new CpuPartitionPreparer().analyze(io.github.pho001.synaptik.prepare.GraphPreparation.project(
                artifact,artifact.partitions().getFirst(),CpuPartitionAnalysisInputs.DEFAULT));
        try (var f = new CpuRecurrentScanExecutionTest.Fixture(artifact,CpuPartitionAnalysisInputs.DEFAULT,analysis,2,true,true,Map.of())) {
            MemorySegment initialStorage = null, seedStorage = null;
            for (var binding : artifact.constants().bindableInputBindings()) {
                int id = Math.toIntExact(binding.valueId().value());
                if (binding.tensorId().equals(lengths.id())) { f.lengths(id,5,0,0); continue; }
                if (binding.tensorId().equals(seq.id())) { fillTyped(f,type,id,Double.NaN); for (int t = 0; t < 5; t++) for (int j = 0; j < 4; j++) put(type,f.segment(id),t*12+j,1); continue; }
                var segment = f.segment(id);
                for (int i = 0; i < 12; i++) if (type == DataType.FLOAT64) segment.setAtIndex(ValueLayout.JAVA_LONG,i,new long[]{Long.MIN_VALUE,1,0x7ff8123456789abcL,0xfff8123456789abcL}[i%4]);
                else segment.setAtIndex(ValueLayout.JAVA_INT,i,new int[]{Integer.MIN_VALUE,1,0x7fc12345,0xffc12345}[i%4]);
                if (binding.tensorId().equals(initial.id())) initialStorage = segment;
                if (binding.tensorId().equals(seed.id())) seedStorage = segment;
            }
            f.executable.bind(f.state).execute();
            var forward = f.segment(Math.toIntExact(artifact.publication().forwardBindings().getFirst().valueId().value()));
            var gradient = f.segment(Math.toIntExact(artifact.publication().gradientBindings().getFirst().valueId().value()));
            assertEquals(-1,initialStorage.asSlice(4L*type.byteWidth()).mismatch(forward.asSlice(4L*type.byteWidth())));
            assertEquals(-1,seedStorage.mismatch(gradient),"actual identity reverse must not add positive zero to raw final seeds");
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void lengthTiedInvariantPositionsAndReusedOccurrencesAccumulateEachActualCompilerContribution(DataType type) {
        Tensor h = tensor(type,false,3,2), x = tensor(type,true,3,2), w = tensor(type,true,2), u = tensor(type,true,2);
        Tensor initial = tensor(type,false,3,2), seq = tensor(type,true,5,3,2), weight = tensor(type,true,2), lengths = tensor(DataType.INT64,false,3);
        var body = new ScanBody(List.of(h),List.of(x),List.of(w,u),List.of(h.add(x.mul(w)).add(x.mul(u))),List.of());
        var roots = new ArrayList<Tensor>();
        for (boolean reverse : List.of(false,true)) roots.addAll(Scan.apply(List.of(initial),List.of(seq),List.of(weight,weight),lengths,1,List.of(OptionalInt.of(0)),body,reverse).finalCarry());
        var seeds = List.of(tensor(type,false,3,2),tensor(type,false,3,2));
        var artifact = CpuGeneralScanCapabilityTest.compile(roots,Optional.of(gradientRequest(roots,seeds,List.of(seq,weight))),false,new CpuCapabilityProvider());
        var analysis = new CpuPartitionPreparer().analyze(io.github.pho001.synaptik.prepare.GraphPreparation.project(
                artifact,artifact.partitions().getFirst(),CpuPartitionAnalysisInputs.DEFAULT));
        try (var f = new CpuRecurrentScanExecutionTest.Fixture(artifact,CpuPartitionAnalysisInputs.DEFAULT,analysis,2,true,true,Map.of())) {
            for (var binding : artifact.constants().bindableInputBindings()) {
                int id = Math.toIntExact(binding.valueId().value());
                if (binding.tensorId().equals(lengths.id())) { f.lengths(id,5,3,0); continue; }
                double fill = binding.tensorId().equals(initial.id()) ? .125 : binding.tensorId().equals(seq.id()) ? .5 : binding.tensorId().equals(weight.id()) ? .25 : 1;
                fillTyped(f,type,id,fill);
            }
            f.executable.bind(f.state).execute();
            double[] dx = new double[30]; for (int t = 0; t < 5; t++) for (int b = 0; b < 3; b++) if (t < new int[]{5,3,0}[b]) Arrays.fill(dx,t*6+b*2,t*6+b*2+2,1);
            assertArrayEquals(dx,readTyped(f,type,Math.toIntExact(artifact.publication().gradientBindings().get(0).valueId().value())));
            assertArrayEquals(new double[]{16,16},readTyped(f,type,Math.toIntExact(artifact.publication().gradientBindings().get(1).valueId().value())));
        }
    }
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void lengthTypedCapacityAlignmentAndWritableOverlapRemainPrewriteGuards(DataType type) {
        var analysis = new CpuPartitionPreparer().analyze(CpuGeneralScanPreparationTest.context(lengthFixture(type,1,5,3,2,false,true)));
        try (var arena = Arena.ofConfined()) {
            var backing = arena.allocate(128,8);
            for (var lengths : List.of(backing.asSlice(0,16),backing.asSlice(4,24))) {
                var before = backing.toArray(ValueLayout.JAVA_BYTE);
                try (var f = new CpuRecurrentScanExecutionTest.Fixture(analysis,1,false,false,Map.of(new ValueId(3),lengths))) {
                    assertThrows(IllegalArgumentException.class,() -> f.executable.bind(f.state));
                }
                assertArrayEquals(before,backing.toArray(ValueLayout.JAVA_BYTE));
            }
            try (var f = new CpuRecurrentScanExecutionTest.Fixture(analysis,1,false,false,
                    Map.of(new ValueId(3),backing.asSlice(0,24),new ValueId(4),backing.asSlice(0,64)))) {
                assertThrows(IllegalArgumentException.class,() -> f.executable.bind(f.state));
            }
            // Physical read/read overlap is legal even across typed views; no formal is added
            // for lengths and selected writes remain distinct from the complete read capacities.
            backing.fill((byte)0);
            try (var f = new CpuRecurrentScanExecutionTest.Fixture(analysis,1,true,false,
                    Map.of(new ValueId(0),backing.asSlice(0,64),new ValueId(3),backing.asSlice(0,24)))) {
                f.executable.bind(f.state).execute(); assertArrayEquals(new double[6],readTyped(f,type,4));
            }
        }
    }
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void lengthTiedActualPositionsKeepFormalMappingSeparateFromUniqueReadsAndWrites(DataType type) {
        Tensor h = tensor(type,false,3,2), k = tensor(type,false,3,2), x = tensor(type,false,3,2), y = tensor(type,false,3,2);
        Tensor initial = tensor(type,false,3,2), seq = tensor(type,false,5,3,2), lengths = tensor(DataType.INT64,false,3);
        Tensor next = h.add(x).add(k).add(y);
        var body = new ScanBody(List.of(h,k),List.of(x,y),List.of(),List.of(next,next),List.of(next,next));
        var scan = Scan.apply(List.of(initial,initial),List.of(seq,seq),List.of(),lengths,2,List.of(OptionalInt.of(0),OptionalInt.of(0)),body,false);
        var roots = new ArrayList<>(scan.finalCarry()); roots.addAll(scan.outputs());
        var artifact = CpuGeneralScanCapabilityTest.compile(roots,Optional.empty(),false,new CpuCapabilityProvider());
        var analysis = new CpuPartitionPreparer().analyze(io.github.pho001.synaptik.prepare.GraphPreparation.project(
                artifact,artifact.partitions().getFirst(),CpuPartitionAnalysisInputs.DEFAULT));
        var unit = analysis.plan().units().getFirst(); var ir = unit.runtimeFacts().generalScan().orElseThrow().ir();
        assertEquals(List.of(0,0,1,1),ir.actualReads()); assertEquals(3,ir.scalarReads().size()); assertEquals(2,ir.lengthRead());
        assertEquals(DataType.INT64,unit.portablePlan().specialization().boundaryDataTypes().get(2));
        try (var f = new CpuRecurrentScanExecutionTest.Fixture(artifact,CpuPartitionAnalysisInputs.DEFAULT,analysis,2,true,true,Map.of())) {
            fillTyped(f,type,Math.toIntExact(unit.boundaryValues().get(0).value()),.125);
            fillTyped(f,type,Math.toIntExact(unit.boundaryValues().get(1).value()),.25);
            f.lengths(Math.toIntExact(unit.boundaryValues().get(2).value()),5,3,0);
            f.executable.bind(f.state).execute();
            double[] expected = new double[6]; for (int b = 0; b < 3; b++) {
                double state = .125; for (int t = 0; t < new int[]{5,3,0}[b]; t++) state = 2*state+.5;
                expected[b*2] = expected[b*2+1] = state;
            }
            for (int p = 0; p < 2; p++) assertArrayEquals(expected,readTyped(f,type,Math.toIntExact(unit.boundaryValues().get(3+p).value())));
            assertNotEquals(unit.boundaryValues().get(3),unit.boundaryValues().get(4),"tied roots keep distinct output storage");
        }
    }
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void lengthEmptyFirstFeatureAndBatchDoNotSuppressRequiredSharedIdentityOrActiveWork(DataType type) {
        for (int t : List.of(0,5)) for (int batch : List.of(0,3)) {
            Tensor h = tensor(type,false,batch,0), x = tensor(type,false,batch,0), old = tensor(type,false), w = tensor(type,false,2);
            var body = new ScanBody(List.of(h,old),List.of(x),List.of(w),List.of(h.add(x),old.add(w.sum())),List.of(h));
            var scan = Scan.apply(List.of(tensor(type,false,batch,0),tensor(type,false)),List.of(tensor(type,false,t,batch,0)),
                    List.of(tensor(type,false,2)),tensor(DataType.INT64,false,batch),1,List.of(OptionalInt.of(0)),body,true);
            var roots = new ArrayList<>(scan.finalCarry()); roots.addAll(scan.outputs()); roots.addAll(scan.residuals());
            var analysis = new CpuPartitionPreparer().analyze(CpuGeneralScanPreparationTest.context(query(roots)));
            try (var f = new CpuRecurrentScanExecutionTest.Fixture(analysis,2,true,true,Map.of())) {
                fillTyped(f,type,1,-0d); fillTyped(f,type,3,.125); f.lengths(4,batch == 0 ? new long[0] : new long[]{t,Math.min(3,t),0});
                f.executable.bind(f.state).execute();
                assertArrayEquals(new double[0],readTyped(f,type,5));
                assertArrayEquals(new double[]{t == 0 || batch == 0 ? -0d : t*.25},readTyped(f,type,6));
            }
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void lengthProducedControlIsCheckedAfterSamePartitionAndSeparatePreparedScheduleSteps(DataType type) {
        Tensor h = tensor(type,false,3,2), x = tensor(type,false,3,2), initial = tensor(type,false,3,2), seq = tensor(type,false,5,3,2);
        Tensor raw = tensor(type,false,3), produced = raw.cast(DataType.INT64);
        var body = new ScanBody(List.of(h),List.of(x),List.of(),List.of(h.add(x)),List.of(h));
        var scan = Scan.apply(List.of(initial),List.of(seq),List.of(),produced,1,List.of(OptionalInt.of(0)),body,false);
        var artifact = CpuGeneralScanCapabilityTest.compile(scan.finalCarry(),Optional.empty(),false,new CpuCapabilityProvider());
        var context = io.github.pho001.synaptik.prepare.GraphPreparation.project(
                artifact,artifact.partitions().getFirst(),CpuPartitionAnalysisInputs.DEFAULT);
        var analysis = new CpuPartitionPreparer().analyze(context);
        int rawId = Math.toIntExact(artifact.constants().bindableInputBindings().stream().filter(b -> b.tensorId().equals(raw.id())).findFirst().orElseThrow().valueId().value());
        var scanNode = artifact.graph().nodes().stream().filter(n -> n.scanBody().isPresent()).findFirst().orElseThrow();
        int controlId = Math.toIntExact(scanNode.inputs().getLast().value()), output = Math.toIntExact(scanNode.outputs().getFirst().value());
        for (boolean separate : List.of(false,true)) try (var f = new CpuRecurrentScanExecutionTest.Fixture(artifact,CpuPartitionAnalysisInputs.DEFAULT,analysis,2,false,false,Map.of())) {
            var composite = (CpuPreparedPartitionExecutable)f.executable;
            f.lengths(controlId,6,6,6); // stale pre-producer values must never be inspected by bind
            put(type,f.segment(rawId),0,5); put(type,f.segment(rawId),1,3); put(type,f.segment(rawId),2,6); f.segment(output).fill((byte)0x5a);
            var sentinel = f.segment(output).toArray(ValueLayout.JAVA_BYTE);
            List<io.github.pho001.synaptik.runtime.execution.BoundInvocation> bound;
            if (separate) {
                var schedule = new io.github.pho001.synaptik.runtime.schedule.PreparedSchedule(f.state.memoryPlan(),composite.children().stream()
                        .map(c -> (io.github.pho001.synaptik.runtime.schedule.PreparedSchedule.Step)new io.github.pho001.synaptik.runtime.schedule.PreparedSchedule.ExecutionStep(c)).toList());
                bound = schedule.steps().stream().map(s -> ((io.github.pho001.synaptik.runtime.schedule.PreparedSchedule.ExecutionStep)s).executable().bind(f.state)).toList();
            } else bound = List.of(composite.bind(f.state));
            assertThrows(IllegalArgumentException.class,() -> bound.forEach(io.github.pho001.synaptik.runtime.execution.BoundInvocation::execute));
            assertArrayEquals(new long[]{5,3,6},f.segment(controlId).toArray(ValueLayout.JAVA_LONG),"upstream writes are not rolled back");
            assertArrayEquals(sentinel,f.segment(output).toArray(ValueLayout.JAVA_BYTE));
            put(type,f.segment(rawId),2,0); bound.forEach(io.github.pho001.synaptik.runtime.execution.BoundInvocation::execute);
            assertArrayEquals(new long[]{5,3,0},f.segment(controlId).toArray(ValueLayout.JAVA_LONG));
        }
    }
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void lengthFusionNeverMovesCoincidentSharedFeatureSitesIntoRowActivity(DataType type) {
        for (boolean matmul : List.of(false,true)) for (boolean reverse : List.of(false,true)) {
            Tensor h = tensor(type,false,2,2), old = tensor(type,false,2), x = tensor(type,false,2,2), w = tensor(type,false,2,2);
            Tensor r = matmul ? x.matmul(w) : h.add(x), q = w.neg();
            Tensor a = r.sum(0,false), d = q.sum(0,false);
            Tensor contribution = matmul ? d.add(a) : a.add(d);
            var body = new ScanBody(List.of(h,old),List.of(x),List.of(w),List.of(r,old.add(contribution)),List.of(r));
            var scan = Scan.apply(List.of(tensor(type,false,2,2),tensor(type,false,2)),List.of(tensor(type,false,1,2,2)),
                    List.of(tensor(type,false,2,2)),tensor(DataType.INT64,false,2),1,List.of(OptionalInt.of(0)),body,reverse);
            var roots = new ArrayList<>(scan.finalCarry()); roots.addAll(scan.outputs()); roots.addAll(scan.residuals());
            var analysis = new CpuPartitionPreparer().analyze(CpuGeneralScanPreparationTest.context(query(roots)));
            try (var f = new CpuRecurrentScanExecutionTest.Fixture(analysis,2,true,true,Map.of())) {
                fillTyped(f,type,0,.125); fillTyped(f,type,1,-0d); fillTyped(f,type,2,Double.NaN); fillTyped(f,type,3,.25);
                put(type,f.segment(2),0,.5); put(type,f.segment(2),1,.5); f.lengths(4,1,0);
                f.executable.bind(f.state).execute();
                assertArrayEquals(new double[]{matmul ? .25 : .625,matmul ? .25 : .625,.125,.125},readTyped(f,type,5));
                assertArrayEquals(new double[]{matmul ? -.25 : .125,matmul ? -.25 : .125},readTyped(f,type,6));
                assertArrayEquals(new double[]{-.25,-.25,-.25,-.25},readTyped(f,type,5+3+body.retainedValues().indexOf(q)),
                        "SHARED feature-row1 remains computed/saved although batch-row1 is inactive");
            }
        }
    }
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void lengthActualCompilerReverseUsesActiveContributorsAndRawEmptySeedIdentity(DataType type) {
        for (boolean reverse : List.of(false,true)) for (int family = 0; family < 2; family++)
            for (boolean emissionSeed : List.of(false,true)) for (int t : List.of(0,5)) {
                final int batch = 3, features = 3;
                Tensor h = tensor(type,false,batch,features), x = tensor(type,true,batch,features);
                Tensor w = tensor(type,true,family == 0 ? new long[]{features} : new long[]{features,features});
                Tensor transformed = family == 0 ? h.add(x.mul(w)).tanh()
                        : h.matmul(w.permute(1,0)).add(x).permute(1,0).tanh().permute(1,0).add(h);
                var body = new ScanBody(List.of(h),List.of(x),List.of(w),List.of(transformed),List.of(transformed));
                Tensor initial = tensor(type,false,batch,features), seq = tensor(type,true,t,batch,features);
                Tensor weight = tensor(type,true,w.descriptor().shape().toLongArray()), lengths = tensor(DataType.INT64,false,batch);
                var scan = Scan.apply(List.of(initial),List.of(seq),List.of(weight),lengths,1,List.of(OptionalInt.of(0)),body,reverse);
                var roots = List.of(scan.finalCarry().getFirst(),scan.outputs().getFirst());
                Tensor finalSeed = tensor(type,false,batch,features), outputSeed = tensor(type,false,t,batch,features);
                var seeds = List.of(finalSeed,outputSeed);
                var request = new FunctionalGradientRequest(List.of(new FunctionalGradientRequest.Stage((emissionSeed ? roots : List.of(roots.getFirst())).stream()
                        .map(r -> (FunctionalGradientRequest.OutputReference)new FunctionalGradientRequest.ForwardTensorReference(r)).toList(),
                        emissionSeed ? List.of(Optional.of(finalSeed),Optional.of(outputSeed)) : List.of(Optional.of(finalSeed)),List.of(seq,weight),false,
                        t == 0 ? FunctionalGradientRequest.DisconnectedPolicy.ZERO : FunctionalGradientRequest.DisconnectedPolicy.ERROR)));
                var artifact = CpuGeneralScanCapabilityTest.compile(roots,Optional.of(request),false,new CpuCapabilityProvider());
                var context = io.github.pho001.synaptik.prepare.GraphPreparation.project(
                        artifact,artifact.partitions().getFirst(),CpuPartitionAnalysisInputs.DEFAULT);
                var analysis = new CpuPartitionPreparer().analyze(context);
                long[] lens = {t,Math.min(3,t),0};
                double[] init = new double[9], input = new double[t*9], weights = new double[family == 0 ? 3 : 9];
                double[] fs = new double[9], os = new double[t*9];
                for (int i = 0; i < 9; i++) { init[i] = work(type,(i-4)*.015625); fs[i] = work(type,(i+1)*.03125); }
                Arrays.fill(weights,.125); Arrays.fill(input,Double.NaN); Arrays.fill(os,Double.NaN);
                for (int time = 0; time < t; time++) for (int row = 0; row < batch; row++) if (time < lens[row])
                    for (int j = 0; j < features; j++) { input[time*9+row*3+j] = work(type,.0625*(1+time+j)); os[time*9+row*3+j] = emissionSeed ? .25 : 0; }
                double[] state = init.clone(), history = new double[t*9], previous = new double[t*9], activation = new double[t*9];
                for (int step = 0; step < t; step++) {
                    int time = reverse ? t-1-step : step;
                    for (int row = 0; row < batch; row++) if (time < lens[row]) {
                        for (int j = 0; j < features; j++) previous[time*9+row*3+j] = state[row*3+j];
                        for (int j = 0; j < features; j++) {
                            double z = 0;
                            if (family == 0) z = work(type,state[row*3+j]+work(type,input[time*9+row*3+j]*weights[j]));
                            else { for (int k = 0; k < features; k++) z = work(type,z+work(type,previous[time*9+row*3+k]*weights[j*3+k])); z = work(type,z+input[time*9+row*3+j]); }
                            double y = work(type,Math.tanh(z)); activation[time*9+row*3+j] = y;
                            state[row*3+j] = family == 0 ? y : work(type,y+previous[time*9+row*3+j]); history[time*9+row*3+j] = state[row*3+j];
                        }
                    }
                }
                double[] carry = fs.clone(), dx = new double[t*9], dw = new double[weights.length];
                for (int step = 0; step < t; step++) {
                    int time = reverse ? step : t-1-step; double[] dz = new double[9], incoming = new double[9];
                    for (int row = 0; row < batch; row++) if (time < lens[row]) for (int j = 0; j < features; j++) {
                        int at = time*9+row*3+j; incoming[row*3+j] = work(type,carry[row*3+j]+(emissionSeed ? os[at] : 0));
                        dz[row*3+j] = work(type,work(type,1-work(type,activation[at]*activation[at]))*incoming[row*3+j]);
                        dx[at] = family == 0 ? work(type,dz[row*3+j]*weights[j]) : dz[row*3+j];
                    }
                    for (int row = 0; row < batch; row++) if (time < lens[row]) for (int j = 0; j < features; j++) {
                        double next = family == 0 ? dz[row*3+j] : incoming[row*3+j];
                        if (family == 1) { double sum = 0; for (int k = 0; k < features; k++) sum = work(type,sum+work(type,dz[row*3+k]*weights[k*3+j])); next = work(type,next+sum); }
                        carry[row*3+j] = next;
                    }
                    boolean active = Arrays.stream(lens).anyMatch(l -> time < l);
                    if (active) for (int p = 0; p < weights.length; p++) {
                        double contribution = 0; int j = family == 0 ? p : p/3, k = p%3;
                        for (int row = 0; row < batch; row++) if (time < lens[row]) contribution = work(type,contribution+work(type,dz[row*3+j]
                                *(family == 0 ? input[time*9+row*3+j] : previous[time*9+row*3+k])));
                        dw[p] = work(type,dw[p]+contribution);
                    }
                }
                for (int mode = 0; mode < 3; mode++) try (var f = new CpuRecurrentScanExecutionTest.Fixture(artifact,CpuPartitionAnalysisInputs.DEFAULT,analysis,mode,true,true,Map.of())) {
                    var feeds = Map.of(initial,init,seq,input,weight,weights,finalSeed,fs,outputSeed,os);
                    for (var binding : artifact.constants().bindableInputBindings()) {
                        if (!f.storage.containsKey(binding.valueId())) continue; // No physical read after T=0 AD folding.
                        int id = Math.toIntExact(binding.valueId().value());
                        if (binding.tensorId().equals(lengths.id())) { f.lengths(id,lens); continue; }
                        var data = feeds.entrySet().stream().filter(e -> e.getKey().id().equals(binding.tensorId())).findFirst().orElseThrow().getValue();
                        for (int i = 0; i < data.length; i++) put(type,f.segment(id),i,data[i]);
                    }
                    f.executable.bind(f.state).execute();
                    double eps = type == DataType.FLOAT64 ? 5e-14 : 3e-6;
                    assertArrayEquals(state,logicalNumbers(artifact,f,artifact.publication().forwardBindings().get(0).valueId()),eps);
                    assertArrayEquals(history,logicalNumbers(artifact,f,artifact.publication().forwardBindings().get(1).valueId()),eps);
                    assertArrayEquals(dx,logicalNumbers(artifact,f,artifact.publication().gradientBindings().get(0).valueId()),eps);
                    assertArrayEquals(dw,logicalNumbers(artifact,f,artifact.publication().gradientBindings().get(1).valueId()),eps);
                }
            }
    }
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void lengthPrefixesSkipPoisonedPaddingPreserveRawStateAndRetainedSites(DataType type) {
        for (int family = 0; family < 3; family++) for (boolean reverse : List.of(false,true))
            for (int t : List.of(0,5)) for (int batch : List.of(0,3)) for (int features : List.of(0,2)) {
                var q = lengthFixture(type,family,t,batch,features,reverse,true);
                var analysis = new CpuPartitionPreparer().analyze(CpuGeneralScanPreparationTest.context(q));
                var ir = analysis.plan().units().getFirst().runtimeFacts().generalScan().orElseThrow().ir();
                for (int mode = 0; mode < 3; mode++) try (var f = new CpuRecurrentScanExecutionTest.Fixture(analysis,mode,true,true,Map.of())) {
                    fillTyped(f,type,0,-0d); fillTyped(f,type,1,Double.NaN); fillTyped(f,type,2,.25);
                    long[] lengths = batch == 0 ? new long[0] : new long[]{t,Math.min(3,t),0}; f.lengths(3,lengths);
                    for (int time = 0; time < t; time++) for (int row = 0; row < batch; row++) if (time < lengths[row])
                        for (int j = 0; j < features; j++) put(type,f.segment(1),(time*batch+row)*features+j,.5);
                    f.executable.bind(f.state).execute();
                    double[] expected = new double[batch*features]; Arrays.fill(expected,-0d);
                    double[] emissions = new double[t*batch*features], previous = new double[emissions.length];
                    for (int step = 0; step < t; step++) {
                        int time = reverse ? t-1-step : step;
                        for (int row = 0; row < batch; row++) if (time < lengths[row]) for (int j = 0; j < features; j++) {
                            int cell = row*features+j, at = time*batch*features+cell; previous[at] = expected[cell];
                            expected[cell] = work(type,expected[cell]+(family == 0 ? .5 : family == 1 ? .125 : .125*features));
                            emissions[at] = expected[cell];
                        }
                    }
                    assertArrayEquals(expected,readTyped(f,type,4)); assertArrayEquals(emissions,readTyped(f,type,5));
                    assertArrayEquals(previous,readTyped(f,type,6));
                    for (int w = 0; w < ir.demandedPorts().size(); w++) {
                        int p = ir.demandedPorts().get(w); if (p < 3) continue;
                        int v = ir.portOrdinals().get(p);
                        if (v == 0) assertArrayEquals(previous,readTyped(f,type,4+p));
                    }
                }
            }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void lengthValueValidationRunsAfterBindingBeforeAnyWriteAndAllowsRetry(DataType type) {
        for (int t : List.of(0,5)) for (int mode = 0; mode < 3; mode++) {
            var q = lengthFixture(type,1,t,3,2,false,true);
            var analysis = new CpuPartitionPreparer().analyze(CpuGeneralScanPreparationTest.context(q));
            try (var f = new CpuRecurrentScanExecutionTest.Fixture(analysis,mode,true,false,Map.of())) {
                f.lengths(3,0,0,-1); var bound = f.executable.bind(f.state); // values must NOT be read here
                var before = new LinkedHashMap<MemorySegment,byte[]>();
                for (var e : f.storage.entrySet()) if (e.getKey().value() >= 4) { e.getValue().fill((byte)0x5a); before.put(e.getValue(),e.getValue().toArray(ValueLayout.JAVA_BYTE)); }
                for (int slot = 0; slot < f.state.memoryPlan().workspaces().size(); slot++) {
                    var scratch = (io.github.pho001.synaptik.backend.cpu.internal.memory.CpuContiguousWorkspace)f.state.workspaceRepresentation(slot);
                    scratch.writableSegment().fill((byte)0x5a); before.put(scratch.writableSegment(),scratch.writableSegment().toArray(ValueLayout.JAVA_BYTE));
                }
                for (long invalid : new long[]{-1,t+1L}) {
                    f.lengths(3,0,0,invalid); assertThrows(IllegalArgumentException.class,bound::execute);
                    before.forEach((s,bytes) -> assertArrayEquals(bytes,s.toArray(ValueLayout.JAVA_BYTE)));
                }
                fillTyped(f,type,0,-0d); f.lengths(3,0,0,0); bound.execute();
                assertArrayEquals(new double[]{-0d,-0d,-0d,-0d,-0d,-0d},readTyped(f,type,4));
            }
        }
    }

    /**
     * Writes one represented test cell without narrowing FLOAT64.
     * @param type exact represented floating type
     * @param segment writable test storage
     * @param index logical element offset
     * @param value value rounded only to the declared type
     */
    private static void put(DataType type,MemorySegment segment,long index,double value) {
        if (type == DataType.FLOAT64) segment.setAtIndex(ValueLayout.JAVA_DOUBLE,index,value);
        else segment.setAtIndex(ValueLayout.JAVA_FLOAT,index,(float)value);
    }
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void denseReadViewsExecuteEveryFamilyOnHeapSegmentAndMixedCarriersWithFreshValues(DataType type) {
        for (int family = 0; family < 3; family++) for (int mode = 0; mode < 3; mode++) {
            var q = withActualReadViews(denseReadFixture(type,family,3,false),true);
            var analysis = new CpuPartitionPreparer().analyze(CpuGeneralScanPreparationTest.context(q));
            try (var f = new CpuRecurrentScanExecutionTest.Fixture(analysis,mode,true,true,Map.of())) {
                var bound = f.executable.bind(f.state);
                for (double input : new double[] {.0625,.1875}) {
                    fillTyped(f,type,0,.125); fillTyped(f,type,1,input); fillTyped(f,type,2,.25);
                    bound.execute();
                    double[] state = new double[6], history = new double[family == 1 ? 9 : 18]; Arrays.fill(state,.125);
                    for (int time = 0; time < 3; time++) {
                        double product = work(type,input*.25); if (family == 2) product = work(type,product+product);
                        for (int i = 0; i < state.length; i++) state[i] = work(type,Math.tanh(work(type,state[i]+product)));
                        if (family == 1) for (int i = 0; i < 3; i++) history[time*3+i] = work(type,state[i]+state[3+i]);
                        else System.arraycopy(state,0,history,time*6,6);
                    }
                    assertArrayEquals(state,readTyped(f,type,3),type == DataType.FLOAT64 ? 2e-15 : 2e-7);
                    assertArrayEquals(history,readTyped(f,type,4),type == DataType.FLOAT64 ? 2e-15 : 2e-7);
                }
            }
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void denseReadViewsKeepEmptyTimeRawCopyAndNonemptyWorkBesideAnEmptyCarry(DataType type) {
        var emptyTime = withActualReadViews(denseReadFixture(type,0,0,true),true);
        var analysis = new CpuPartitionPreparer().analyze(CpuGeneralScanPreparationTest.context(emptyTime));
        try (var f = new CpuRecurrentScanExecutionTest.Fixture(analysis,2,true,true,Map.of())) {
            // Byte-level comparison includes NaN payloads, signed zero and subnormal stored bits.
            if (type == DataType.FLOAT64) f.segment(0).copyFrom(MemorySegment.ofArray(new long[] {Long.MIN_VALUE,1,0x7ff8123456789abcL,0,2,-1}));
            else f.segment(0).copyFrom(MemorySegment.ofArray(new int[] {Integer.MIN_VALUE,1,0x7fc12345,0,2,-1}));
            f.executable.bind(f.state).execute(); assertEquals(-1,f.segment(0).mismatch(f.segment(3))); assertEquals(0,f.segment(4).byteSize());
        }
        Tensor h = tensor(type,false,0,3), s = tensor(type,false,3), x = tensor(type,false,0,2), w = tensor(type,false,2,3), v = tensor(type,false,3,3);
        Tensor product = x.matmul(w);
        var body = new ScanBody(List.of(h,s),List.of(x),List.of(w,v),List.of(h.add(product),s.add(w.sum(0,false))),List.of(product.sum(),w.matmul(v)));
        var scan = Scan.apply(List.of(tensor(type,false,0,3),tensor(type,false,3)),List.of(tensor(type,false,3,0,2)),
                List.of(tensor(type,false,2,3),tensor(type,false,3,3)),body,false);
        var roots = new ArrayList<>(scan.finalCarry()); roots.addAll(scan.outputs());
        var q = withActualReadViews(query(roots),true);
        var nonempty = new CpuPartitionPreparer().analyze(CpuGeneralScanPreparationTest.context(q));
        try (var f = new CpuRecurrentScanExecutionTest.Fixture(nonempty,2,true,false,Map.of())) {
            fillTyped(f,type,1,.125); fillTyped(f,type,3,.25); fillTyped(f,type,4,.5); f.executable.bind(f.state).execute();
            assertArrayEquals(new double[0],readTyped(f,type,5));
            assertArrayEquals(new double[] {1.625,1.625,1.625},readTyped(f,type,6),"nonempty parameter SUM executes despite first empty carry");
            assertArrayEquals(new double[3],readTyped(f,type,7),"empty product SUM publishes typed zero every step");
            double[] expectedProducts = new double[18]; Arrays.fill(expectedProducts,.375);
            assertArrayEquals(expectedProducts,readTyped(f,type,8),"nonempty invariant MATMUL still executes and emits at every original time");
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void denseReadViewsRetainExpandedScalarInitializersAndRankZeroReads(DataType type) {
        var dense = withActualReadViews(denseReadFixture(type,0,3,false),true);
        var scalarInputs = new ArrayList<>(dense.inputs());
        for (int p : List.of(0,2)) {
            var d = scalarInputs.get(p); scalarInputs.set(p,new TensorDescriptor(type,d.shape(),Optional.of(
                    io.github.pho001.synaptik.model.layout.LayoutDescriptor.of(d.shape(),new long[d.shape().rank()],0,true)),d.requiresGrad()));
        }
        var scalar = new io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery(dense.operation(), scalarInputs, dense.outputs(), dense.scanBody(), dense.scanActivityPlan());
        var facts = io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLowering.analyze(scalar);
        assertEquals(List.of(true,false,true),facts.ir().scalarReads()); assertEquals(List.of(1L,18L,1L),facts.spans().subList(0,3));
        List<double[]> expected = new ArrayList<>();
        for (var q : List.of(dense,scalar)) {
            var a = new CpuPartitionPreparer().analyze(CpuGeneralScanPreparationTest.context(q));
            try (var f = new CpuRecurrentScanExecutionTest.Fixture(a,2,true,false,Map.of())) {
                fillTyped(f,type,0,.125); fillTyped(f,type,1,.0625); fillTyped(f,type,2,.25); f.executable.bind(f.state).execute();
                if (expected.isEmpty()) { expected.add(readTyped(f,type,3)); expected.add(readTyped(f,type,4)); }
                else { assertArrayEquals(expected.get(0),readTyped(f,type,3)); assertArrayEquals(expected.get(1),readTyped(f,type,4)); }
            }
        }
        Tensor h = tensor(type,false), x = tensor(type,false), w = tensor(type,false);
        var body = new ScanBody(List.of(h),List.of(x),List.of(w),List.of(h.add(x.mul(w))),List.of());
        var scan = Scan.apply(List.of(tensor(type,false)),List.of(tensor(type,false,3)),List.of(tensor(type,false)),body,false);
        var q = withActualReadViews(query(scan.finalCarry()),true);
        var a = new CpuPartitionPreparer().analyze(CpuGeneralScanPreparationTest.context(q));
        try (var f = new CpuRecurrentScanExecutionTest.Fixture(a,0,true,false,Map.of())) {
            fillTyped(f,type,0,.125); fillTyped(f,type,1,.5); fillTyped(f,type,2,.25); f.executable.bind(f.state).execute();
            assertArrayEquals(new double[] {.5},readTyped(f,type,3));
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void denseReadViewsPreserveRepeatedExactActualIdentitiesAndPositionalRoles(DataType type) {
        Tensor h = tensor(type,false,2), x = tensor(type,false,2), w = tensor(type,false,2);
        var body = new ScanBody(List.of(h),List.of(x),List.of(w),List.of(h.mul(w).add(x)),List.of());
        var shape = io.github.pho001.synaptik.model.shape.Shape.of(2); var stacked = io.github.pho001.synaptik.model.shape.Shape.of(3,2);
        Tensor tied = TensorFactory.create(new TensorDescriptor(type,shape,Optional.of(
                io.github.pho001.synaptik.model.layout.LayoutDescriptor.of(shape,new long[] {1},0,true)),false));
        Tensor seq = TensorFactory.create(new TensorDescriptor(type,stacked,Optional.of(
                io.github.pho001.synaptik.model.layout.LayoutDescriptor.of(stacked,new long[] {2,1},0,true)),false));
        var scan = Scan.apply(List.of(tied),List.of(seq),List.of(tied),body,false);
        var artifact = CpuGeneralScanCapabilityTest.compile(scan.finalCarry(),Optional.empty(),false,new CpuCapabilityProvider());
        var context = io.github.pho001.synaptik.prepare.GraphPreparation.project(
                artifact,artifact.partitions().getFirst(),CpuPartitionAnalysisInputs.DEFAULT);
        var facts = new CpuPartitionPreparer().analyze(context).plan().units().getFirst().runtimeFacts().generalScan().orElseThrow();
        assertEquals(List.of(0,1,0),facts.ir().actualReads()); assertEquals(2,facts.ir().scalarReads().size());
        double[] expected = {.25,.5}; double[] sequence = {.125,.25,.375,.5,.625,.75};
        for (int t = 0; t < 3; t++) for (int i = 0; i < 2; i++) expected[i] = work(type,work(type,expected[i]*(i == 0 ? .25 : .5))+sequence[2*t+i]);
        if (type == DataType.FLOAT64) assertArrayEquals(expected,executeDoubleArtifact(artifact,Map.of(tied,new double[] {.25,.5},seq,sequence)).getFirst());
        else {
            float[] data = new float[6]; for (int i = 0; i < data.length; i++) data[i] = (float)sequence[i];
            float[] actual = executeArtifact(artifact,Map.of(tied,new float[] {.25f,.5f},seq,data)).getFirst();
            for (int i = 0; i < actual.length; i++) assertEquals(expected[i],actual[i]);
        }
        assertTrue(tied.descriptor().layout().orElseThrow().isView(),"lowering never clears source view metadata");
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void denseReadViewsKeepCapacityPermissionAliasScratchAndLifetimeGuards(DataType type) {
        var q = withActualReadViews(denseReadFixture(type,1,3,false),true);
        var analysis = new CpuPartitionPreparer().analyze(CpuGeneralScanPreparationTest.context(q));
        long width = type.byteWidth();
        try (var arena = Arena.ofConfined()) {
            var memory = arena.allocate(256*width,width); memory.fill((byte)0x5a);
            var cases = List.of(Map.of(new ValueId(1),memory.asSlice(0,17*width)),
                    Map.of(new ValueId(0),memory.asSlice(1,6*width)),
                    Map.of(new ValueId(3),memory.asSlice(0,6*width).asReadOnly()),
                    Map.of(new ValueId(0),memory.asSlice(0,12*width),new ValueId(3),memory.asSlice(6*width,6*width)),
                    Map.of(new ValueId(3),memory.asSlice(0,6*width),new ValueId(4),memory.asSlice(0,9*width)));
            for (var overrides : cases) {
                var before = memory.toArray(ValueLayout.JAVA_BYTE);
                try (var f = new CpuRecurrentScanExecutionTest.Fixture(analysis,1,false,false,overrides)) {
                    assertThrows(IllegalArgumentException.class,() -> f.executable.bind(f.state));
                    assertArrayEquals(before,memory.toArray(ValueLayout.JAVA_BYTE),"reject before any selected storage mutation");
                }
            }
            // Distinct positional reads may overlap; their complete capacities are read-only.
            var aliases = Map.of(new ValueId(0),memory.asSlice(0,6*width),new ValueId(2),memory.asSlice(0,3*width));
            try (var f = new CpuRecurrentScanExecutionTest.Fixture(analysis,1,true,false,aliases)) {
                fillTyped(f,type,0,.25); fillTyped(f,type,1,.125); var bound = f.executable.bind(f.state); bound.execute();
                double state = .25; for (int t = 0; t < 3; t++) state = work(type,Math.tanh(work(type,state+.03125)));
                double[] expected = new double[6]; Arrays.fill(expected,state); assertArrayEquals(expected,readTyped(f,type,3),type == DataType.FLOAT64 ? 2e-15 : 2e-7);
            }
        }
        try (var f = new CpuRecurrentScanExecutionTest.Fixture(analysis,1,false,false,Map.of())) {
            var scratch = (io.github.pho001.synaptik.backend.cpu.internal.memory.CpuContiguousWorkspace) f.state.workspaceRepresentation(0);
            var before = scratch.writableSegment().toArray(ValueLayout.JAVA_BYTE);
            var bindings = new ArrayList<List<io.github.pho001.synaptik.runtime.run.BufferRepresentationBinding>>();
            for (int p = 0; p < f.borrowed.size(); p++) {
                var representation = p == 3 ? io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBorrowedBuffer.borrow(
                        new io.github.pho001.synaptik.model.storage.MemorySegmentStorage(type,6,scratch.writableSegment().asSlice(0,6*width))) : f.borrowed.get(p);
                bindings.add(List.of(new io.github.pho001.synaptik.runtime.run.BufferRepresentationBinding(representation,
                        io.github.pho001.synaptik.runtime.run.RunResourceOwnership.BORROWED)));
            }
            try (var state = new io.github.pho001.synaptik.runtime.run.RunState(f.state.memoryPlan(),bindings,List.of(scratch))) {
                assertThrows(IllegalArgumentException.class,() -> f.executable.bind(state));
                assertArrayEquals(before,scratch.writableSegment().toArray(ValueLayout.JAVA_BYTE));
            }
        }
        var donor = Arena.ofShared();
        try (var f = new CpuRecurrentScanExecutionTest.Fixture(analysis,1,false,false,Map.of(new ValueId(0),donor.allocate(6*width,width)))) {
            var bound = f.executable.bind(f.state); donor.close();
            var before = f.segment(3).toArray(ValueLayout.JAVA_BYTE);
            assertThrows(IllegalArgumentException.class,() -> f.executable.bind(f.state));
            assertThrows(IllegalArgumentException.class,bound::execute);
            assertArrayEquals(before,f.segment(3).toArray(ValueLayout.JAVA_BYTE));
        } finally { if (donor.scope().isAlive()) donor.close(); }
    }

    /**
     * Initializes a selected typed fixture without rewriting its truthful descriptor.
     * @param f assigned-slot fixture with writable test setup access
     * @param type represented dtype
     * @param id selected value ordinal
     * @param value represented fill value
     */
    private static void fillTyped(CpuRecurrentScanExecutionTest.Fixture f, DataType type, int id, double value) {
        if (type == DataType.FLOAT64) f.fillDouble(id,value); else f.fill(id,(float)value);
    }

    /**
     * Reads represented test values without narrowing FLOAT64 or exposing writable storage.
     * @param f assigned-slot fixture
     * @param type represented dtype
     * @param id selected value ordinal
     * @return new logical dense array of represented values, never null
     */
    private static double[] readTyped(CpuRecurrentScanExecutionTest.Fixture f, DataType type, int id) {
        if (type == DataType.FLOAT64) return f.doubles(id);
        float[] data = f.floats(id); double[] result = new double[data.length];
        for (int i = 0; i < data.length; i++) result[i] = data[i]; return result;
    }

    /**
     * Preserves the primitive represented rounding site in a direct test-only oracle.
     * @param type represented dtype
     * @param value primitive arithmetic result before storage rounding
     * @return FLOAT32-rounded value or unchanged FLOAT64 value
     */
    private static double work(DataType type, double value) { return type == DataType.FLOAT64 ? value : (float)value; }

    @Test void float64OrdinarySpecialValuesAndScalarArithmeticRetainProducingSites() {
        for (boolean reverse : List.of(false, true)) {
            Tensor h = tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,false, 6), x = tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,false, 6);
            Tensor next = h.add(x).sub(io.github.pho001.synaptik.model.datatype.ScalarValue.float64(0))
                    .mul(io.github.pho001.synaptik.model.datatype.ScalarValue.float64(2)).div(x).neg();
            var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(next), List.of(next));
            var scan = Scan.apply(List.of(tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,false, 6)), List.of(tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,false, 3, 6)), List.of(), body, reverse);
            var analysis = new CpuPartitionPreparer().analyze(CpuGeneralScanPreparationTest.context(query(
                    List.of(scan.finalCarry().getFirst(), scan.outputs().getFirst()))));
            double[] initial = {Double.NaN, Double.POSITIVE_INFINITY, -0d, Double.MAX_VALUE, Double.MIN_VALUE, 1};
            double[] row = {1, Double.NEGATIVE_INFINITY, -0d, 2, .5, 0};
            try (var f = new CpuRecurrentScanExecutionTest.Fixture(analysis, 0, true, false, Map.of())) {
                f.doubles(0, initial); double[] sequence = new double[18];
                for (int t = 0; t < 3; t++) System.arraycopy(row, 0, sequence, t * 6, 6);
                f.doubles(1, sequence); f.executable.bind(f.state).execute();
                double[] expected = initial.clone();
                for (int t = 0; t < 3; t++) for (int i = 0; i < 6; i++) {
                    double add = expected[i] + row[i]; double sub = add - 0; double mul = sub * 2;
                    double div = mul / row[i]; expected[i] = -div;
                }
                assertArrayEquals(expected, f.doubles(2));
            }
        }
    }
    @Test void float64SubnormalsNeverAcquireDazFtzAtPrimitiveSumMatmulOrGeneratedGradientSites() {
        var type = io.github.pho001.synaptik.model.datatype.DataType.FLOAT64;
        Tensor h = tensor(type,false), x = tensor(type,false,2), w = tensor(type,false,2,1);
        Tensor half = x.mul(io.github.pho001.synaptik.model.datatype.ScalarValue.float64(.5));
        var body = new ScanBody(List.of(h),List.of(x),List.of(w),List.of(h),List.of(half,half.sum(),x.reshape(io.github.pho001.synaptik.model.shape.Shape.of(1,2)).matmul(w)));
        Tensor initial = tensor(type,false), seq = tensor(type,false,3,2), weight = tensor(type,false,2,1);
        var scan = Scan.apply(List.of(initial),List.of(seq),List.of(weight),body,false);
        double s = Double.MIN_NORMAL/2;
        double[] data = {s,s,-s,-s,Double.MIN_VALUE,-Double.MIN_VALUE};
        var result = executeDoubleArtifact(CpuGeneralScanCapabilityTest.compile(scan.outputs(),Optional.empty(),false,new CpuCapabilityProvider()),
                Map.of(initial,new double[] {-0d},seq,data,weight,new double[] {.5,.5}));
        for (int t = 0; t < 3; t++) {
            double sum = 0, product = 0;
            for (int i = 0; i < 2; i++) {
                double expected = data[t*2+i]*.5;
                assertEquals(Double.doubleToRawLongBits(expected),Double.doubleToRawLongBits(result.get(0)[t*2+i]));
                sum += expected; product += data[t*2+i]*.5;
            }
            assertEquals(Double.doubleToRawLongBits(sum),Double.doubleToRawLongBits(result.get(1)[t]));
            assertEquals(Double.doubleToRawLongBits(product),Double.doubleToRawLongBits(result.get(2)[t]));
        }
        Tensor gh = tensor(type,true,1), gx = tensor(type,true,1), gw = tensor(type,true,1);
        var gb = new ScanBody(List.of(gh),List.of(gx),List.of(gw),List.of(gh.add(gx.mul(gw))),List.of());
        Tensor ih = tensor(type,true,1), xs = tensor(type,true,3,1), ws = tensor(type,true,1), seed = tensor(type,false,1);
        var gs = Scan.apply(List.of(ih),List.of(xs),List.of(ws),gb,false);
        var request = gradientRequest(gs.finalCarry(),List.of(seed),List.of(ih,xs,ws));
        var ad = executeDoubleArtifact(CpuGeneralScanCapabilityTest.compile(gs.finalCarry(),Optional.of(request),false,new CpuCapabilityProvider()),
                Map.of(ih,new double[] {0},xs,new double[] {s,s,s},ws,new double[] {.5},seed,new double[] {s}));
        assertArrayEquals(new double[] {s},ad.get(1)); assertArrayEquals(new double[] {s*.5,s*.5,s*.5},ad.get(2));
        assertTrue(ad.get(2)[0] != 0,"representable subnormal gradient is not FTZ");
        assertArrayEquals(new double[1],ad.get(3),"s*s legitimately underflows at each product site");
    }

    @Test void float64RawIdentityCopiesPreservePayloadsSignsAndSubnormalsAtEveryTime() {
        var type = io.github.pho001.synaptik.model.datatype.DataType.FLOAT64;
        long[] raw = {0,Long.MIN_VALUE,1,Long.MIN_VALUE|1,0x7ff8123456789abcL,0xfff8123456789abcL};
        for (boolean reverse : List.of(false,true)) for (int t : List.of(0,3)) {
            Tensor h = tensor(type,false,2,3), x = tensor(type,false,2,3);
            var body = new ScanBody(List.of(h),List.of(x),List.of(),List.of(h.cast(type)),List.of(x.permute(1,0).contiguous(),x.sum(new int[0],false),x.sumToShape(x.descriptor().shape())));
            var scan = Scan.apply(List.of(tensor(type,false,2,3)),List.of(tensor(type,false,t,2,3)),List.of(),body,reverse);
            var roots = new ArrayList<>(scan.finalCarry()); roots.addAll(scan.outputs());
            var analysis = new CpuPartitionPreparer().analyze(CpuGeneralScanPreparationTest.context(query(roots)));
            try (var f = new CpuRecurrentScanExecutionTest.Fixture(analysis,2,true,true,Map.of())) {
                f.segment(0).copyFrom(java.lang.foreign.MemorySegment.ofArray(raw));
                long[] stacked = new long[t*6], transposed = new long[t*6];
                for (int time = 0; time < t; time++) for (int i = 0; i < 6; i++) { stacked[time*6+i] = raw[i]; transposed[time*6+(i%3)*2+i/3] = raw[i]; }
                f.segment(1).copyFrom(java.lang.foreign.MemorySegment.ofArray(stacked)); f.executable.bind(f.state).execute();
                assertArrayEquals(raw,f.segment(2).toArray(java.lang.foreign.ValueLayout.JAVA_LONG));
                assertArrayEquals(transposed,f.segment(3).toArray(java.lang.foreign.ValueLayout.JAVA_LONG));
                assertArrayEquals(stacked,f.segment(4).toArray(java.lang.foreign.ValueLayout.JAVA_LONG));
                assertArrayEquals(stacked,f.segment(5).toArray(java.lang.foreign.ValueLayout.JAVA_LONG));
            }
        }
    }

    @Test void float64BindingChecksWholeCapacityEightByteAlignmentPermissionAndAliasesBeforeWrites() {
        var q = withType(CpuGeneralScanGeneratedKernelTest.query(3,2,false,List.of(0,1)),io.github.pho001.synaptik.model.datatype.DataType.FLOAT64);
        var analysis = new CpuPartitionPreparer().analyze(CpuGeneralScanPreparationTest.context(q));
        try (var arena = java.lang.foreign.Arena.ofConfined()) {
            var memory = arena.allocate(160,8); memory.fill((byte)0x5a);
            var bad = List.of(Map.of(new ValueId(0),memory.asSlice(0,80),new ValueId(2),memory.asSlice(64,16)),
                    Map.of(new ValueId(2),memory.asSlice(4,16)),Map.of(new ValueId(2),memory.asSlice(0,16).asReadOnly()),
                    Map.of(new ValueId(1),memory.asSlice(0,40)));
            for (var overrides : bad) {
                byte[] before = memory.toArray(java.lang.foreign.ValueLayout.JAVA_BYTE);
                assertThrows(IllegalArgumentException.class,() -> { try (var f = new CpuRecurrentScanExecutionTest.Fixture(analysis,1,false,false,overrides)) { f.executable.bind(f.state); } });
                assertArrayEquals(before,memory.toArray(java.lang.foreign.ValueLayout.JAVA_BYTE));
            }
            var reads = Map.of(new ValueId(0),memory.asSlice(0,48),new ValueId(1),memory.asSlice(0,48));
            try (var f = new CpuRecurrentScanExecutionTest.Fixture(analysis,1,true,false,reads)) {
                f.fillDouble(1,.125+0x1p-40); assertDoesNotThrow(() -> f.executable.bind(f.state).execute());
            }
        }
    }
    @Test void float64MatmulBothTransposeViewsAndSavedPrestepCarryUseOriginalTimeIndices() {
        for (int flags = 0; flags < 8; flags++) for (boolean reverse : List.of(false,true)) for (int time : List.of(0,1,3)) {
            Tensor h = tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,false,2,2), x = tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,false,2,2);
            Tensor left = (flags&1) == 0 ? h : h.permute(1,0);
            Tensor right = (flags&2) == 0 ? x : x.permute(1,0);
            if ((flags&4) != 0) { left = left.contiguous(); right = right.contiguous(); }
            Tensor next = left.matmul(right);
            var body = new ScanBody(List.of(h),List.of(x),List.of(),List.of(next),List.of(next));
            var scan = Scan.apply(List.of(tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,false,2,2)),List.of(tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,false,time,2,2)),List.of(),body,reverse);
            var roots = new ArrayList<>(scan.finalCarry()); roots.addAll(scan.outputs());
            roots.add(scan.residuals().get(body.retainedValues().indexOf(h)));
            roots.add(scan.residuals().get(body.retainedValues().indexOf(next)));
            var analysis = new CpuPartitionPreparer().analyze(CpuGeneralScanPreparationTest.context(query(roots)));
            try (var f = new CpuRecurrentScanExecutionTest.Fixture(analysis,flags%3,true,false,Map.of())) {
                double[] carry = {.125,-.25,.375,.5}, seq = new double[time*4];
                for (int i = 0; i < seq.length; i++) seq[i] = (i%7-2)*.125;
                f.doubles(0,carry); f.doubles(1,seq); double[] old = new double[time*4], saved = new double[time*4];
                for (int step = 0; step < time; step++) {
                    int t = reverse ? time-1-step : step; System.arraycopy(carry,0,old,t*4,4);
                    double[] updated = new double[4];
                    for (int m = 0; m < 2; m++) for (int n = 0; n < 2; n++) {
                        double sum = 0;
                        for (int k = 0; k < 2; k++) sum += carry[(flags&1) == 0 ? m*2+k : k*2+m]
                                *seq[t*4+((flags&2) == 0 ? k*2+n : n*2+k)];
                        updated[m*2+n] = sum;
                    }
                    carry = updated; System.arraycopy(carry,0,saved,t*4,4);
                }
                f.executable.bind(f.state).execute(); assertArrayEquals(carry,f.doubles(2)); assertArrayEquals(saved,f.doubles(3));
                int oldPort = 2+2+body.retainedValues().indexOf(h), nextPort = 2+2+body.retainedValues().indexOf(next);
                assertArrayEquals(old,f.doubles(oldPort)); assertArrayEquals(saved,f.doubles(nextPort));
            }
        }
    }

    @Test void float64MatmulSpecialValuesHaveRepresentedProductsSavedBitsAndOrdinaryZeroDivision() {
        var rows = List.of(new double[] {9007199254740992d,1,-9007199254740992d,1}, new double[] {Double.NaN,1,2,3},
                new double[] {Double.POSITIVE_INFINITY,Double.NEGATIVE_INFINITY,1,2},new double[] {Double.MAX_VALUE,Double.MAX_VALUE,0,0},
                new double[] {Double.MIN_VALUE,-Double.MIN_VALUE,Double.MIN_NORMAL,-Double.MIN_NORMAL},new double[] {-0d,-0d,-0d,-0d});
        Tensor h = tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,false,1,1), x = tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,false,1,4), w = tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,false,4,1), zero = tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,false);
        Tensor product = x.matmul(w); var body = new ScanBody(List.of(h),List.of(x),List.of(w,zero),List.of(h),List.of(product,product.div(zero)));
        var scan = Scan.apply(List.of(tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,false,1,1)),List.of(tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,false,rows.size(),1,4)),List.of(tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,false,4,1),tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,false)),body,false);
        var roots = new ArrayList<>(scan.outputs()); roots.add(scan.residuals().get(body.retainedValues().indexOf(product)));
        var analysis = new CpuPartitionPreparer().analyze(CpuGeneralScanPreparationTest.context(query(roots)));
        try (var f = new CpuRecurrentScanExecutionTest.Fixture(analysis,2,false,false,Map.of())) {
            double[] seq = new double[rows.size()*4]; for (int t = 0; t < rows.size(); t++) System.arraycopy(rows.get(t),0,seq,t*4,4);
            f.fillDouble(0,Double.longBitsToDouble(0x7ff8123456789abcL)); f.doubles(1,seq); f.fillDouble(2,1); f.fillDouble(3,0);
            f.executable.bind(f.state).execute(); var products = f.doubles(5); var divided = f.doubles(6);
            var saved = f.doubles(4+1+body.emissions().size()+body.retainedValues().indexOf(product));
            for (int t = 0; t < rows.size(); t++) {
                double value = 0; for (double term : rows.get(t)) value += term;
                if (Double.isNaN(value)) assertTrue(Double.isNaN(products[t])); else assertEquals(Double.doubleToRawLongBits(value),Double.doubleToRawLongBits(products[t]));
                assertEquals(Double.doubleToRawLongBits(products[t]),Double.doubleToRawLongBits(saved[t]),"exact original saved product bits");
                if (Double.isNaN(value/0)) assertTrue(Double.isNaN(divided[t])); else assertEquals(value/0,divided[t]);
            }
        }
    }

    @Test void float64MatmulZeroTimeActualAdPreservesRawInitialAndSeedAndTypedZeroParameterSums() {
        Tensor h = tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,true,2,2), x = tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,true,2,2), w = tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,true,2,2);
        var body = new ScanBody(List.of(h),List.of(x),List.of(w),List.of(h.matmul(w).add(x).tanh()),List.of());
        Tensor initial = tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,true,2,2), sequence = tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,true,0,2,2), weight = tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,true,2,2), seed = tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,false,2,2);
        var scan = Scan.apply(List.of(initial),List.of(sequence),List.of(weight),body,true);
        var request = gradientRequest(scan.finalCarry(),List.of(seed),List.of(initial,sequence,weight),FunctionalGradientRequest.DisconnectedPolicy.ZERO);
        double[] raw = {Double.longBitsToDouble(0x7ff8123456789abcL),-0d,Double.MIN_VALUE,Double.NEGATIVE_INFINITY};
        var actual = executeDoubleArtifact(CpuGeneralScanCapabilityTest.compile(scan.finalCarry(),Optional.of(request),false,new CpuCapabilityProvider()),
                Map.of(initial,raw,sequence,new double[0],weight,new double[] {Double.NaN,Double.NaN,Double.NaN,Double.NaN},seed,raw));
        for (int p : List.of(0,1)) for (int i = 0; i < raw.length; i++) assertEquals(Double.doubleToRawLongBits(raw[i]),Double.doubleToRawLongBits(actual.get(p)[i]));
        assertArrayEquals(new double[0],actual.get(2)); assertArrayEquals(new double[4],actual.get(3));
    }

    @Test void float64MatmulReusedTemplateTiedActualRolesRepeatedRootsSumAcrossOccurrences() {
        Tensor h = tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,true,2,2), x = tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,true,2,2), w = tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,true,2,2), next = h.matmul(w).add(x).tanh();
        var body = new ScanBody(List.of(h),List.of(x),List.of(w),List.of(next),List.of(next,next));
        Tensor weight = tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,true,2,2), sequence = tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,true,3,2,2), seed = tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,false);
        var a = Scan.apply(List.of(weight),List.of(sequence),List.of(weight),body,false);
        var b = Scan.apply(List.of(weight),List.of(sequence),List.of(weight),body,true);
        Tensor root = a.finalCarry().getFirst().sum().add(b.finalCarry().getFirst().sum())
                .add(a.outputs().get(0).sum()).add(a.outputs().get(1).sum());
        var targets = List.of(weight,sequence);
        var feeds = Map.of(weight,new double[] {.1,.2,.3,.4},sequence,new double[] {.01,.02,.03,.04,.05,.06,.07,.08,.09,.1,.11,.12},seed,new double[] {1});
        var actual = executeDoubleArtifact(CpuGeneralScanCapabilityTest.compile(List.of(root),Optional.of(gradientRequest(List.of(root),List.of(seed),targets)),false,new CpuCapabilityProvider()),feeds);
        // Untie only the ordinary reference's initial role: ordinary MATMUL's fixed entry
        // requires distinct boundary identities. Sum both independently differentiated roles
        // afterward; the actual scan above keeps the exact tied actual identity throughout.
        Tensor referenceInitial = tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,true,2,2), forward = referenceInitial, reverse = referenceInitial; Tensor[] emitted = new Tensor[3];
        for (int t = 0; t < 3; t++) {
            forward = forward.matmul(weight).add(sequence.select(0,t)).tanh(); emitted[t] = forward;
            reverse = reverse.matmul(weight).add(sequence.select(0,2-t)).tanh();
        }
        Tensor stacked = Tensor.stack(0,emitted), reference = forward.sum().add(reverse.sum()).add(stacked.sum()).add(stacked.sum());
        var referenceFeeds = new IdentityHashMap<>(feeds); referenceFeeds.put(referenceInitial,feeds.get(weight));
        var expected = executeDoubleArtifact(CpuGeneralScanCapabilityTest.compile(List.of(reference),Optional.of(gradientRequest(List.of(reference),List.of(seed),
                List.of(referenceInitial,weight,sequence))),false,new CpuCapabilityProvider()),referenceFeeds);
        assertArrayEquals(expected.get(0),actual.get(0),2e-13);
        double[] tied = expected.get(1).clone(); for (int i = 0; i < tied.length; i++) tied[i] += expected.get(2)[i];
        assertArrayEquals(tied,actual.get(1),2e-13,"sum of both untied ordinary-AD roles equals tied actual scan gradient");
        assertArrayEquals(expected.get(3),actual.get(2),2e-13);
    }

    @Test void float64MatmulEmptyAxesStillExecuteNonemptyProductsAndReductionsOnEveryCarrier() {
        for (int[] dims : List.of(new int[] {2,3,0},new int[] {0,3,2},new int[] {2,0,3},new int[] {2,3,2}))
            for (int time : List.of(0,1,3)) for (boolean reverse : List.of(false,true)) for (int mode : List.of(0,1,2)) {
                int m = dims[0], n = dims[1], contraction = dims[2];
                Tensor h = tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,false,m,n), x = tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,false,m,contraction), w = tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,false,contraction,n);
                Tensor product = x.matmul(w), next = h.add(product);
                var body = new ScanBody(List.of(h),List.of(x),List.of(w),List.of(next),List.of(product,product.sum()));
                var scan = Scan.apply(List.of(tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,false,m,n)),List.of(tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,false,time,m,contraction)),List.of(tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,false,contraction,n)),body,reverse);
                var roots = new ArrayList<>(scan.finalCarry()); roots.addAll(scan.outputs());
                var analysis = new CpuPartitionPreparer().analyze(CpuGeneralScanPreparationTest.context(query(roots)));
                try (var f = new CpuRecurrentScanExecutionTest.Fixture(analysis,mode,true,false,Map.of())) {
                    f.fillDouble(0,.25); f.fillDouble(1,2); f.fillDouble(2,.5); f.executable.bind(f.state).execute();
                    double[] carry = new double[m*n]; Arrays.fill(carry,.25+time*contraction);
                    assertArrayEquals(carry,f.doubles(3));
                    double[] products = new double[time*m*n]; Arrays.fill(products,contraction);
                    assertArrayEquals(products,f.doubles(4));
                    double[] sums = new double[time]; Arrays.fill(sums,m*n*contraction);
                    assertArrayEquals(sums,f.doubles(5));
                }
            }
    }

    @Test void float64PermutedOldCarryAndDifferentShapeForwardingSurviveWholeCoordinateStages() {
        for (boolean reverse : List.of(false, true)) for (int mode = 0; mode < 3; mode++) {
            Tensor a = tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,false, 2, 2), b = tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,false, 2, 2), small = tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,false, 2), x = tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,false, 2, 2);
            Tensor next = b.permute(1, 0).add(x), smallNext = a.sum(0, false);
            var body = new ScanBody(List.of(a,b,small), List.of(x), List.of(),
                    List.of(next, a.permute(1, 0), smallNext), List.of(a, smallNext));
            var scan = Scan.apply(List.of(tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,false,2,2), tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,false,2,2), tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,false,2)),
                    List.of(tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,false,3,2,2)), List.of(), body, reverse);
            var roots = new ArrayList<>(scan.finalCarry()); roots.addAll(scan.outputs());
            var analysis = new CpuPartitionPreparer().analyze(CpuGeneralScanPreparationTest.context(query(roots)));
            try (var f = new CpuRecurrentScanExecutionTest.Fixture(analysis, mode, true, true, Map.of())) {
                double[] av = {1,2,4,8}, bv = {16,32,64,128}, sv = {0,0}, seq = {1,2,3,4,5,6,7,8,9,10,11,12};
                f.doubles(0,av); f.doubles(1,bv); f.doubles(2,sv); f.doubles(3,seq);
                double[] ah = new double[12], sh = new double[6];
                for (int k = 0; k < 3; k++) {
                    int t = reverse ? 2-k : k;
                    System.arraycopy(av,0,ah,t*4,4);
                    double[] na = {bv[0]+seq[t*4],bv[2]+seq[t*4+1],bv[1]+seq[t*4+2],bv[3]+seq[t*4+3]};
                    double[] nb = {av[0],av[2],av[1],av[3]};
                    sv = new double[] {av[0]+av[2],av[1]+av[3]}; System.arraycopy(sv,0,sh,t*2,2);
                    av = na; bv = nb;
                }
                f.executable.bind(f.state).execute();
                assertArrayEquals(av,f.doubles(4)); assertArrayEquals(bv,f.doubles(5)); assertArrayEquals(sv,f.doubles(6));
                assertArrayEquals(ah,f.doubles(7)); assertArrayEquals(sh,f.doubles(8));
            }
        }
    }

    @Test void float64ShapedBroadcastReductionAndActualReverseRetainRepresentedSites() {
        for (boolean reverse : List.of(false, true)) for (boolean optimized : List.of(false, true))
            for (int t : List.of(0, 1, 3)) for (int batch : List.of(0, 2)) {
                int width = 3, n = batch * width;
                Tensor h = tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,false, batch, width), x = tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,true, batch, width), w = tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,true, width);
                Tensor next = h.mul(w).add(x).tanh();
                var body = new ScanBody(List.of(h), List.of(x), List.of(w), List.of(next), List.of(next.sum(0, false)));
                Tensor initial = tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,false, batch, width), sequence = tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,true, t, batch, width), weight = tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,true, width);
                var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(weight), body, reverse);
                var roots = List.of(scan.finalCarry().getFirst(), scan.outputs().getFirst());
                Tensor hs = tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,false, batch, width), ys = tensor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT64,false, t, width);
                var request = new FunctionalGradientRequest(List.of(new FunctionalGradientRequest.Stage(roots.stream()
                        .map(r -> (FunctionalGradientRequest.OutputReference) new FunctionalGradientRequest.ForwardTensorReference(r)).toList(),
                        List.of(Optional.of(hs), Optional.of(ys)), List.of(sequence, weight), false,
                        t == 0 ? FunctionalGradientRequest.DisconnectedPolicy.ZERO : FunctionalGradientRequest.DisconnectedPolicy.ERROR)));
                var artifact = CpuGeneralScanCapabilityTest.compile(roots, Optional.of(request), optimized, new CpuCapabilityProvider());
                var context = io.github.pho001.synaptik.prepare.GraphPreparation.project(
                        artifact,artifact.partitions().getFirst(),CpuPartitionAnalysisInputs.DEFAULT);
                var analysis = new CpuPartitionPreparer().analyze(context);
                try (var f = new CpuRecurrentScanExecutionTest.Fixture(artifact,CpuPartitionAnalysisInputs.DEFAULT,analysis, 2, false, false, Map.of())) {
                    for (var input : artifact.constants().bindableInputBindings()) if (f.storage.containsKey(input.valueId())) {
                        double value = input.tensorId().equals(initial.id()) ? .2 : input.tensorId().equals(sequence.id()) ? .125
                                : input.tensorId().equals(weight.id()) ? batch == 0 ? Double.NaN : .75
                                : input.tensorId().equals(hs.id()) ? .5 : .25;
                        f.fillDouble(Math.toIntExact(input.valueId().value()),value);
                    }
                    f.executable.bind(f.state).execute();
                    double[] state = new double[n], old = new double[t*n], y = new double[t*n], emission = new double[t*width]; Arrays.fill(state, .2);
                    for (int k = 0; k < t; k++) { int time = reverse ? t-1-k : k;
                        for (int b = 0; b < batch; b++) for (int j = 0; j < width; j++) {
                            int i = b*width+j, p = time*n+i; old[p] = state[i];
                            double product = state[i]*.75; double sum = product+.125;
                            y[p] = state[i] = (double) StrictMath.tanh(sum); emission[time*width+j] += state[i];
                        }
                    }
                    double[] carry = new double[n], dx = new double[t*n], dw = new double[width]; Arrays.fill(carry, .5);
                    for (int k = 0; k < t; k++) { int time = reverse ? k : t-1-k;
                        double[] contribution = new double[width];
                        for (int b = 0; b < batch; b++) for (int j = 0; j < width; j++) {
                            int i = b*width+j, p = time*n+i;
                            double incoming = carry[i]+.25, square = y[p]*y[p], derivative = 1-square;
                            dx[p] = incoming*derivative; contribution[j] += dx[p]*old[p]; carry[i] = dx[p]*.75;
                        }
                        for (int j = 0; j < width; j++) dw[j] += contribution[j];
                    }
                    assertArrayEquals(state, logicalNumbers(artifact,f,artifact.publication().forwardBindings().get(0).valueId()));
                    assertArrayEquals(emission, logicalNumbers(artifact,f,artifact.publication().forwardBindings().get(1).valueId()));
                    assertArrayEquals(dx, logicalNumbers(artifact,f,artifact.publication().gradientBindings().get(0).valueId()));
                    assertArrayEquals(dw, logicalNumbers(artifact,f,artifact.publication().gradientBindings().get(1).valueId()));
                }
            }
    }
    @Test void float64CompleteActualReverseClosureMatchesOrdinaryUnrolledTensorAd() {
        var type = io.github.pho001.synaptik.model.datatype.DataType.FLOAT64;
        for (int family = 0; family < 6; family++) for (boolean reverse : List.of(false,true))
            for (boolean optimized : List.of(false,true)) for (int seedClass = 0; seedClass < 3; seedClass++) {
                boolean matrix = family >= 4, pointwise = family == 0;
                Tensor h = tensor(type,false,pointwise ? new long[] {3} : new long[] {2,3});
                Tensor k = tensor(type,true,matrix ? new long[] {3,2} : new long[] {3});
                Tensor x = tensor(type,true,pointwise ? new long[] {3} : matrix ? new long[] {2,2} : new long[] {2,1});
                Tensor w = tensor(type,true,matrix ? family == 5 ? new long[] {3,2} : new long[] {2,3} : new long[] {3});
                Tensor b = tensor(type,true,matrix ? new long[] {3} : new long[0]);
                var r = typedRecipe(h,k,x,w,b,family);
                var body = new ScanBody(List.of(h,k),List.of(x),List.of(w,b),r.subList(0,2),r.subList(2,4));
                Tensor initial = tensor(type,false,h.descriptor().shape().toLongArray()), initialK = tensor(type,true,k.descriptor().shape().toLongArray());
                long[] xs = x.descriptor().shape().toLongArray(), stacked = new long[xs.length+1]; stacked[0] = 3; System.arraycopy(xs,0,stacked,1,xs.length);
                Tensor seq = tensor(type,true,stacked), weight = tensor(type,true,w.descriptor().shape().toLongArray()), bias = tensor(type,true,b.descriptor().shape().toLongArray());
                var scan = Scan.apply(List.of(initial,initialK),List.of(seq),List.of(weight,bias),body,reverse);
                var all = new ArrayList<>(scan.finalCarry()); all.addAll(scan.outputs());
                var roots = seedClass == 0 ? all.subList(0,2) : seedClass == 1 ? all.subList(2,4) : all;
                var seeds = roots.stream().map(v -> tensor(type,false,v.descriptor().shape().toLongArray())).toList();
                var targets = List.of(initialK,seq,weight,bias);
                var policy = seedClass == 1 && (family == 2 || family == 3)
                        ? FunctionalGradientRequest.DisconnectedPolicy.ZERO : FunctionalGradientRequest.DisconnectedPolicy.ERROR;
                var actual = CpuGeneralScanCapabilityTest.compile(roots,Optional.of(gradientRequest(roots,seeds,targets,policy)),optimized,new CpuCapabilityProvider());
                var feeds = new IdentityHashMap<Tensor,double[]>();
                for (var v : List.of(initial,initialK,seq,weight,bias)) {
                    double[] data = new double[Math.toIntExact(v.descriptor().shape().knownElementCount().orElseThrow())];
                    for (int i = 0; i < data.length; i++) data[i] = .125+(i%7-3)*.03125+0x1p-38;
                    feeds.put(v,data);
                }
                for (var v : seeds) { double[] data = new double[Math.toIntExact(v.descriptor().shape().knownElementCount().orElseThrow())];
                    for (int i = 0; i < data.length; i++) data[i] = .25+(i%3)*.0625+0x1p-40; feeds.put(v,data); }
                Tensor uh = initial, uk = initialK; var e0 = new Tensor[3]; var e1 = new Tensor[3];
                for (int step = 0; step < 3; step++) {
                    int t = reverse ? 2-step : step; var u = typedRecipe(uh,uk,seq.select(0,t),weight,bias,family);
                    uh = u.get(0); uk = u.get(1); e0[t] = u.get(2); e1[t] = u.get(3);
                }
                var unrolled = List.of(uh,uk,Tensor.stack(0,e0),Tensor.stack(0,e1));
                var selected = seedClass == 0 ? unrolled.subList(0,2) : seedClass == 1 ? unrolled.subList(2,4) : unrolled;
                var reference = executeDoubleArtifact(CpuGeneralScanCapabilityTest.compile(unrolled,
                        Optional.of(gradientRequest(selected,seeds,targets,policy)),optimized,new CpuCapabilityProvider()),feeds);
                var expected = new ArrayList<double[]>();
                if (seedClass != 1) expected.addAll(reference.subList(0,2));
                if (seedClass != 0) expected.addAll(reference.subList(2,4));
                expected.addAll(reference.subList(4,reference.size()));
                var result = executeDoubleArtifact(actual,feeds); assertEquals(expected.size(),result.size());
                for (int p = 0; p < result.size(); p++) assertArrayEquals(expected.get(p),result.get(p),2e-13,
                        "F64 ordinary Tensor AD family="+family+" reverse="+reverse+" optimized="+optimized+" seeds="+seedClass+" port="+p);
            }
    }

    private static List<Tensor> typedRecipe(Tensor h, Tensor k, Tensor x, Tensor w, Tensor b, int family) {
        if (family >= 4) return matmulRecipe(h,k,x,w,b,family == 5);
        if (family > 0) {
            var result = shapedRecipe(h,k,x,w,b,family-1);
            return family == 2 ? result.stream().map(v -> v.cast(v.descriptor().dataType()).contiguous()).toList() : result;
        }
        Tensor next = h.mul(w).add(x).add(b).tanh(), nextK = k.add(next).sigmoid();
        return List.of(next,nextK,next,nextK);
    }

    /**
     * Executes the existing assigned-slot fixture with double data and logical affine reads.
     * @param artifact compiled whole occurrence, including actual reverse when requested
     * @param feeds exact source identities mapped to represented double storage values
     * @return fresh logical forward publications followed by target-ordered gradients
     */
    public static List<double[]> executeDoubleArtifact(io.github.pho001.synaptik.compiler.CompileArtifacts artifact, Map<Tensor,double[]> feeds) {
        var context = io.github.pho001.synaptik.prepare.GraphPreparation.project(
                artifact,artifact.partitions().getFirst(),CpuPartitionAnalysisInputs.DEFAULT);
        var analysis = new CpuPartitionPreparer().analyze(context);
        try (var f = new CpuRecurrentScanExecutionTest.Fixture(artifact,CpuPartitionAnalysisInputs.DEFAULT,analysis,2,true,true,Map.of())) {
            for (var input : artifact.constants().bindableInputBindings()) {
                if (!f.storage.containsKey(input.valueId())) continue; // Compiled inputs need not remain physical reads.
                var tensor = feeds.keySet().stream().filter(v -> v.id().equals(input.tensorId())).findFirst().orElseThrow();
                f.doubles(Math.toIntExact(input.valueId().value()),feeds.get(tensor));
            }
            f.executable.bind(f.state).execute();
            var ids = new ArrayList<ValueId>(); artifact.publication().forwardBindings().forEach(p -> ids.add(p.valueId()));
            artifact.publication().gradientBindings().forEach(p -> ids.add(p.valueId()));
            return ids.stream().map(id -> logicalNumbers(artifact,f,id)).toList();
        }
    }
    @Test void matmulEpilogueBlocksEarlierReductionMotionAndDeadProductDoesNotKeepIndependentTail() {
        Tensor scalar = tensor(false), h = tensor(false,2,2), x = tensor(false,2,2), w = tensor(false,2,2);
        Tensor sum = x.add(x).sum(), activated = h.matmul(w).tanh();
        Tensor next = activated.reshape(io.github.pho001.synaptik.model.shape.Shape.of(2,2)).add(x);
        var body = new ScanBody(List.of(scalar,h),List.of(x),List.of(w),List.of(sum,next),List.of());
        var scan = Scan.apply(List.of(tensor(false),tensor(false,2,2)),List.of(tensor(false,3,2,2)),List.of(tensor(false,2,2)),body,false);
        var analysis = new CpuPartitionPreparer().analyze(CpuGeneralScanPreparationTest.context(query(scan.finalCarry())));
        var plan = analysis.plan().units().getFirst().runtimeFacts().generalScan().orElseThrow().ir().stages().orElseThrow();
        int product = java.util.stream.IntStream.range(0,plan.order().size()).filter(i -> plan.order().get(i).right() >= 0).findFirst().orElseThrow();
        assertFalse(plan.order().get(product).epilogue().isEmpty());
        assertTrue(product+1 < plan.order().size(),"dependent reshape ADD must remain after MATMUL epilogue");
        try (var f = new CpuRecurrentScanExecutionTest.Fixture(analysis,2,false,false,Map.of())) {
            f.fill(0,0); f.fill(1,.25f); f.fill(2,.125f); f.fill(3,.5f); f.executable.bind(f.state).execute();
            assertArrayEquals(new float[] {1},f.floats(4));
            float value = .25f; for (int t = 0; t < 3; t++) value = (float)Math.tanh(value*.5f+value*.5f)+.125f;
            assertArrayEquals(new float[] {value,value,value,value},f.floats(5));
        }
        var deadBody = new ScanBody(List.of(h),List.of(x),List.of(w),List.of(h.matmul(w)),List.of(x.add(x)));
        var dead = Scan.apply(List.of(tensor(false,2,2)),List.of(tensor(false,3,2,2)),List.of(tensor(false,2,2)),deadBody,false);
        var deadAnalysis = new CpuPartitionPreparer().analyze(CpuGeneralScanPreparationTest.context(query(dead.outputs())));
        var facts = deadAnalysis.plan().units().getFirst().runtimeFacts().generalScan().orElseThrow();
        assertTrue(facts.ir().stages().orElseThrow().order().stream().noneMatch(s -> s.right() >= 0));
        assertEquals(0,facts.geometry().scratchBytes());
        try (var f = new CpuRecurrentScanExecutionTest.Fixture(deadAnalysis,2,false,false,Map.of())) {
            f.fill(0,Float.NaN); f.fill(1,.125f); f.fill(2,Float.NaN); f.executable.bind(f.state).execute();
            float[] expected = new float[12]; Arrays.fill(expected,.25f); assertArrayEquals(expected,f.floats(4));
        }
    }

    @Test void matmulActualReverseMatchesUnrolledNonSquareTwoCarryTensorAd() {
        for (boolean transpose : List.of(false,true)) for (boolean reverse : List.of(false,true))
            for (boolean optimized : List.of(false,true)) for (boolean initialGradient : List.of(false,true))
                for (int seedClass = 0; seedClass < 3; seedClass++) {
                    Tensor h = tensor(initialGradient,2,3), k = tensor(true,3,2), x = tensor(true,2,2);
                    Tensor w = tensor(true,transpose ? new long[] {3,2} : new long[] {2,3}), b = tensor(true,3);
                    var r = matmulRecipe(h,k,x,w,b,transpose);
                    var body = new ScanBody(List.of(h,k),List.of(x),List.of(w,b),r.subList(0,2),r.subList(2,4));
                    Tensor initial = tensor(initialGradient,2,3), initialK = tensor(true,3,2), seq = tensor(true,3,2,2);
                    Tensor weight = tensor(true,w.descriptor().shape().toLongArray()), bias = tensor(true,3);
                    var scan = Scan.apply(List.of(initial,initialK),List.of(seq),List.of(weight,bias),body,reverse);
                    var all = new ArrayList<>(scan.finalCarry()); all.addAll(scan.outputs());
                    var roots = seedClass == 0 ? all.subList(0,2) : seedClass == 1 ? all.subList(2,4) : all;
                    var seeds = roots.stream().map(v -> tensor(false,v.descriptor().shape().toLongArray())).toList();
                    var targets = new ArrayList<Tensor>(); if (initialGradient) targets.add(initial);
                    targets.addAll(List.of(initialK,seq,weight,bias));
                    var actual = CpuGeneralScanCapabilityTest.compile(roots,Optional.of(gradientRequest(roots,seeds,targets)),optimized,new CpuCapabilityProvider());
                    var feeds = new IdentityHashMap<Tensor,float[]>();
                    for (Tensor v : List.of(initial,initialK,seq,weight,bias)) {
                        float[] values = new float[Math.toIntExact(v.descriptor().shape().knownElementCount().orElseThrow())];
                        for (int i = 0; i < values.length; i++) values[i] = (i%7-3)*.03125f;
                        feeds.put(v,values);
                    }
                    for (Tensor v : seeds) { float[] values = new float[Math.toIntExact(v.descriptor().shape().knownElementCount().orElseThrow())];
                        for (int i = 0; i < values.length; i++) values[i] = .25f+(i%3)*.0625f; feeds.put(v,values); }
                    Tensor uh = initial, uk = initialK; Tensor[] e0 = new Tensor[3], e1 = new Tensor[3];
                    for (int step = 0; step < 3; step++) {
                        int time = reverse ? 2-step : step;
                        var u = matmulRecipe(uh,uk,seq.select(0,time),weight,bias,transpose);
                        uh = u.get(0); uk = u.get(1); e0[time] = u.get(2); e1[time] = u.get(3);
                    }
                    var unrolled = List.of(uh,uk,Tensor.stack(0,e0),Tensor.stack(0,e1));
                    var selected = seedClass == 0 ? unrolled.subList(0,2) : seedClass == 1 ? unrolled.subList(2,4) : unrolled;
                    var expected = executeArtifact(CpuGeneralScanCapabilityTest.compile(selected,
                            Optional.of(gradientRequest(selected,seeds,targets)),optimized,new CpuCapabilityProvider()),feeds);
                    var result = executeArtifact(actual,feeds);
                    assertEquals(expected.size(),result.size());
                    for (int p = 0; p < result.size(); p++) assertArrayEquals(expected.get(p),result.get(p),1e-5f,
                            "ordinary Tensor AD: transpose="+transpose+" reverse="+reverse+" optimized="+optimized+" h0grad="+initialGradient+" seeds="+seedClass+" port="+p);
                }
    }

    private static List<Tensor> matmulRecipe(Tensor h, Tensor k, Tensor x, Tensor w, Tensor b, boolean transpose) {
        Tensor weight = transpose ? w.permute(1,0) : w;
        Tensor product = h.matmul(k).matmul(weight);
        Tensor next = product.add(x.matmul(weight)).add(b).tanh();
        Tensor nextK = k.add(next.permute(1,0)).sigmoid();
        return List.of(next,nextK,next.sum(0,false),nextK.permute(1,0).contiguous());
    }

    private static FunctionalGradientRequest gradientRequest(List<Tensor> roots, List<Tensor> seeds, List<Tensor> targets) {
        return gradientRequest(roots,seeds,targets,FunctionalGradientRequest.DisconnectedPolicy.ERROR);
    }

    private static FunctionalGradientRequest gradientRequest(List<Tensor> roots, List<Tensor> seeds, List<Tensor> targets,
            FunctionalGradientRequest.DisconnectedPolicy policy) {
        return new FunctionalGradientRequest(List.of(new FunctionalGradientRequest.Stage(roots.stream()
                .map(v -> (FunctionalGradientRequest.OutputReference)new FunctionalGradientRequest.ForwardTensorReference(v)).toList(),
                seeds.stream().map(Optional::of).toList(),targets,false,policy)));
    }

    @Test void matmulBothTransposeViewsAndSavedPrestepCarryUseOriginalTimeIndices() {
        for (int flags = 0; flags < 8; flags++) for (boolean reverse : List.of(false,true)) for (int time : List.of(0,1,3)) {
            Tensor h = tensor(false,2,2), x = tensor(false,2,2);
            Tensor left = (flags&1) == 0 ? h : h.permute(1,0);
            Tensor right = (flags&2) == 0 ? x : x.permute(1,0);
            if ((flags&4) != 0) { left = left.contiguous(); right = right.contiguous(); }
            Tensor next = left.matmul(right);
            var body = new ScanBody(List.of(h),List.of(x),List.of(),List.of(next),List.of(next));
            var scan = Scan.apply(List.of(tensor(false,2,2)),List.of(tensor(false,time,2,2)),List.of(),body,reverse);
            var roots = new ArrayList<>(scan.finalCarry()); roots.addAll(scan.outputs());
            roots.add(scan.residuals().get(body.retainedValues().indexOf(h)));
            roots.add(scan.residuals().get(body.retainedValues().indexOf(next)));
            var analysis = new CpuPartitionPreparer().analyze(CpuGeneralScanPreparationTest.context(query(roots)));
            try (var f = new CpuRecurrentScanExecutionTest.Fixture(analysis,flags%3,true,false,Map.of())) {
                float[] carry = {.125f,-.25f,.375f,.5f}, seq = new float[time*4];
                for (int i = 0; i < seq.length; i++) seq[i] = (i%7-2)*.125f;
                f.floats(0,carry); f.floats(1,seq); float[] old = new float[time*4], saved = new float[time*4];
                for (int step = 0; step < time; step++) {
                    int t = reverse ? time-1-step : step; System.arraycopy(carry,0,old,t*4,4);
                    float[] updated = new float[4];
                    for (int m = 0; m < 2; m++) for (int n = 0; n < 2; n++) {
                        float sum = 0;
                        for (int k = 0; k < 2; k++) sum += carry[(flags&1) == 0 ? m*2+k : k*2+m]
                                *seq[t*4+((flags&2) == 0 ? k*2+n : n*2+k)];
                        updated[m*2+n] = sum;
                    }
                    carry = updated; System.arraycopy(carry,0,saved,t*4,4);
                }
                f.executable.bind(f.state).execute(); assertArrayEquals(carry,f.floats(2)); assertArrayEquals(saved,f.floats(3));
                int oldPort = 2+2+body.retainedValues().indexOf(h), nextPort = 2+2+body.retainedValues().indexOf(next);
                assertArrayEquals(old,f.floats(oldPort)); assertArrayEquals(saved,f.floats(nextPort));
            }
        }
    }

    @Test void matmulSpecialValuesHaveRepresentedProductsSavedBitsAndOrdinaryZeroDivision() {
        var rows = List.of(new float[] {16777216,1,-16777216,1}, new float[] {Float.NaN,1,2,3},
                new float[] {Float.POSITIVE_INFINITY,Float.NEGATIVE_INFINITY,1,2},new float[] {Float.MAX_VALUE,Float.MAX_VALUE,0,0},
                new float[] {Float.MIN_VALUE,-Float.MIN_VALUE,Float.MIN_NORMAL,-Float.MIN_NORMAL},new float[] {-0f,-0f,-0f,-0f});
        Tensor h = tensor(false,1,1), x = tensor(false,1,4), w = tensor(false,4,1), zero = tensor(false);
        Tensor product = x.matmul(w); var body = new ScanBody(List.of(h),List.of(x),List.of(w,zero),List.of(h),List.of(product,product.div(zero)));
        var scan = Scan.apply(List.of(tensor(false,1,1)),List.of(tensor(false,rows.size(),1,4)),List.of(tensor(false,4,1),tensor(false)),body,false);
        var roots = new ArrayList<>(scan.outputs()); roots.add(scan.residuals().get(body.retainedValues().indexOf(product)));
        var analysis = new CpuPartitionPreparer().analyze(CpuGeneralScanPreparationTest.context(query(roots)));
        try (var f = new CpuRecurrentScanExecutionTest.Fixture(analysis,2,false,false,Map.of())) {
            float[] seq = new float[rows.size()*4]; for (int t = 0; t < rows.size(); t++) System.arraycopy(rows.get(t),0,seq,t*4,4);
            f.fill(0,Float.intBitsToFloat(0x7fc12345)); f.floats(1,seq); f.fill(2,1); f.fill(3,0);
            f.executable.bind(f.state).execute(); var products = f.floats(5); var divided = f.floats(6);
            var saved = f.floats(4+1+body.emissions().size()+body.retainedValues().indexOf(product));
            for (int t = 0; t < rows.size(); t++) {
                float value = 0; for (float term : rows.get(t)) value += term;
                if (Float.isNaN(value)) assertTrue(Float.isNaN(products[t])); else assertEquals(Float.floatToRawIntBits(value),Float.floatToRawIntBits(products[t]));
                assertEquals(Float.floatToRawIntBits(products[t]),Float.floatToRawIntBits(saved[t]),"exact original saved product bits");
                if (Float.isNaN(value/0f)) assertTrue(Float.isNaN(divided[t])); else assertEquals(value/0f,divided[t]);
            }
        }
    }

    @Test void matmulZeroTimeActualAdPreservesRawInitialAndSeedAndTypedZeroParameterSums() {
        Tensor h = tensor(true,2,2), x = tensor(true,2,2), w = tensor(true,2,2);
        var body = new ScanBody(List.of(h),List.of(x),List.of(w),List.of(h.matmul(w).add(x).tanh()),List.of());
        Tensor initial = tensor(true,2,2), sequence = tensor(true,0,2,2), weight = tensor(true,2,2), seed = tensor(false,2,2);
        var scan = Scan.apply(List.of(initial),List.of(sequence),List.of(weight),body,true);
        var request = gradientRequest(scan.finalCarry(),List.of(seed),List.of(initial,sequence,weight),FunctionalGradientRequest.DisconnectedPolicy.ZERO);
        float[] raw = {Float.intBitsToFloat(0x7fc12345),-0f,Float.MIN_VALUE,Float.NEGATIVE_INFINITY};
        var actual = executeArtifact(CpuGeneralScanCapabilityTest.compile(scan.finalCarry(),Optional.of(request),false,new CpuCapabilityProvider()),
                Map.of(initial,raw,sequence,new float[0],weight,new float[] {Float.NaN,Float.NaN,Float.NaN,Float.NaN},seed,raw));
        for (int p : List.of(0,1)) for (int i = 0; i < raw.length; i++) assertEquals(Float.floatToRawIntBits(raw[i]),Float.floatToRawIntBits(actual.get(p)[i]));
        assertArrayEquals(new float[0],actual.get(2)); assertArrayEquals(new float[4],actual.get(3));
    }

    @Test void matmulReusedTemplateTiedActualRolesRepeatedRootsSumAcrossOccurrences() {
        Tensor h = tensor(true,2,2), x = tensor(true,2,2), w = tensor(true,2,2), next = h.matmul(w).add(x).tanh();
        var body = new ScanBody(List.of(h),List.of(x),List.of(w),List.of(next),List.of(next,next));
        Tensor weight = tensor(true,2,2), sequence = tensor(true,3,2,2), seed = tensor(false);
        var a = Scan.apply(List.of(weight),List.of(sequence),List.of(weight),body,false);
        var b = Scan.apply(List.of(weight),List.of(sequence),List.of(weight),body,true);
        Tensor root = a.finalCarry().getFirst().sum().add(b.finalCarry().getFirst().sum())
                .add(a.outputs().get(0).sum()).add(a.outputs().get(1).sum());
        var targets = List.of(weight,sequence);
        var feeds = Map.of(weight,new float[] {.1f,.2f,.3f,.4f},sequence,new float[] {.01f,.02f,.03f,.04f,.05f,.06f,.07f,.08f,.09f,.1f,.11f,.12f},seed,new float[] {1});
        var actual = executeArtifact(CpuGeneralScanCapabilityTest.compile(List.of(root),Optional.of(gradientRequest(List.of(root),List.of(seed),targets)),false,new CpuCapabilityProvider()),feeds);
        // Untie only the ordinary reference's initial role: ordinary MATMUL's fixed entry
        // requires distinct boundary identities. Sum both independently differentiated roles
        // afterward; the actual scan above keeps the exact tied actual identity throughout.
        Tensor referenceInitial = tensor(true,2,2), forward = referenceInitial, reverse = referenceInitial; Tensor[] emitted = new Tensor[3];
        for (int t = 0; t < 3; t++) {
            forward = forward.matmul(weight).add(sequence.select(0,t)).tanh(); emitted[t] = forward;
            reverse = reverse.matmul(weight).add(sequence.select(0,2-t)).tanh();
        }
        Tensor stacked = Tensor.stack(0,emitted), reference = forward.sum().add(reverse.sum()).add(stacked.sum()).add(stacked.sum());
        var referenceFeeds = new IdentityHashMap<>(feeds); referenceFeeds.put(referenceInitial,feeds.get(weight));
        var expected = executeArtifact(CpuGeneralScanCapabilityTest.compile(List.of(reference),Optional.of(gradientRequest(List.of(reference),List.of(seed),
                List.of(referenceInitial,weight,sequence))),false,new CpuCapabilityProvider()),referenceFeeds);
        assertArrayEquals(expected.get(0),actual.get(0),1e-5f);
        float[] tied = expected.get(1).clone(); for (int i = 0; i < tied.length; i++) tied[i] += expected.get(2)[i];
        assertArrayEquals(tied,actual.get(1),1e-5f,"sum of both untied ordinary-AD roles equals tied actual scan gradient");
        assertArrayEquals(expected.get(3),actual.get(2),1e-5f);
    }

    @Test void matmulEmptyAxesStillExecuteNonemptyProductsAndReductionsOnEveryCarrier() {
        for (int[] dims : List.of(new int[] {2,3,0},new int[] {0,3,2},new int[] {2,0,3},new int[] {2,3,2}))
            for (int time : List.of(0,1,3)) for (boolean reverse : List.of(false,true)) for (int mode : List.of(0,1,2)) {
                int m = dims[0], n = dims[1], contraction = dims[2];
                Tensor h = tensor(false,m,n), x = tensor(false,m,contraction), w = tensor(false,contraction,n);
                Tensor product = x.matmul(w), next = h.add(product);
                var body = new ScanBody(List.of(h),List.of(x),List.of(w),List.of(next),List.of(product,product.sum()));
                var scan = Scan.apply(List.of(tensor(false,m,n)),List.of(tensor(false,time,m,contraction)),List.of(tensor(false,contraction,n)),body,reverse);
                var roots = new ArrayList<>(scan.finalCarry()); roots.addAll(scan.outputs());
                var analysis = new CpuPartitionPreparer().analyze(CpuGeneralScanPreparationTest.context(query(roots)));
                try (var f = new CpuRecurrentScanExecutionTest.Fixture(analysis,mode,true,false,Map.of())) {
                    f.fill(0,.25f); f.fill(1,2); f.fill(2,.5f); f.executable.bind(f.state).execute();
                    float[] carry = new float[m*n]; Arrays.fill(carry,.25f+time*contraction);
                    assertArrayEquals(carry,f.floats(3));
                    float[] products = new float[time*m*n]; Arrays.fill(products,contraction);
                    assertArrayEquals(products,f.floats(4));
                    float[] sums = new float[time]; Arrays.fill(sums,m*n*contraction);
                    assertArrayEquals(sums,f.floats(5));
                }
            }
    }

    @Test void reductionEpilogueFanoutAndPermutedResultRetainNecessaryStageBoundaries() {
        for (boolean permuted : List.of(false,true)) for (boolean reverse : List.of(false,true))
            for (int t : List.of(0,1,3)) for (int history = 0; history < 3; history++) {
                Tensor h = tensor(false,permuted ? new long[] {2,2} : new long[] {1,2});
                Tensor x = tensor(false,permuted ? new long[] {2,2,2} : new long[] {1,2});
                Tensor p = permuted ? x.neg() : h.add(x);
                Tensor sum = p.sum(0,!permuted);
                Tensor next = permuted ? sum.permute(1,0).contiguous().add(h) : p.add(sum);
                var body = new ScanBody(List.of(h),List.of(x),List.of(),List.of(next),history == 0 ? List.of() : List.of(p,sum));
                Tensor initial = tensor(false,h.descriptor().shape().toLongArray());
                Tensor sequence = tensor(false,permuted ? new long[] {t,2,2,2} : new long[] {t,1,2});
                var scan = Scan.apply(List.of(initial),List.of(sequence),List.of(),body,reverse);
                var roots = new ArrayList<Tensor>(); roots.add(scan.finalCarry().getFirst());
                if (history > 0) roots.add(scan.outputs().get(history-1));
                var analysis = new CpuPartitionPreparer().analyze(CpuGeneralScanPreparationTest.context(query(roots)));
                var plan = analysis.plan().units().getFirst().runtimeFacts().generalScan().orElseThrow().ir().stages().orElseThrow();
                assertEquals(2,plan.order().size(),"unsafe epilogue keeps its own complete-domain stage");
                assertTrue(plan.order().stream().allMatch(s -> s.epilogue().isEmpty()));
                int sumOrdinal = plan.order().getFirst().output();
                assertTrue(plan.cuts().stream().anyMatch(c -> c.value() == sumOrdinal && c.death() > c.birth()),"SUM fully materialized before fanout/permuted tail");
                try (var f = new CpuRecurrentScanExecutionTest.Fixture(analysis,2,false,false,Map.of())) {
                    float[] carry = permuted ? new float[] {1,2,4,8} : new float[] {1,2};
                    float[] seq = new float[t*(permuted ? 8 : 2)];
                    for (int i = 0; i < seq.length; i++) seq[i] = (i+1)*.125f;
                    f.floats(0,carry); f.floats(1,seq);
                    float[] saved = new float[t*(history == 1 && permuted ? 8 : carry.length)];
                    for (int step = 0; step < t; step++) {
                        int time = reverse ? t-1-step : step;
                        float[] pv = new float[permuted ? 8 : 2], sv = new float[carry.length];
                        for (int i = 0; i < pv.length; i++) pv[i] = permuted ? -seq[time*8+i] : carry[i]+seq[time*2+i];
                        for (int i = 0; i < sv.length; i++) { float s = 0f; s += pv[i]; if (permuted) s += pv[i+4]; sv[i] = s; }
                        if (history > 0) { var values = history == 1 ? pv : sv; System.arraycopy(values,0,saved,time*values.length,values.length); }
                        float[] updated = new float[carry.length];
                        for (int i = 0; i < updated.length; i++) updated[i] = permuted ? sv[(i%2)*2+i/2]+carry[i] : pv[i]+sv[i];
                        carry = updated;
                    }
                    f.executable.bind(f.state).execute();
                    assertArrayEquals(carry,f.floats(2),"permuted="+permuted+" T="+t+" history="+history+" reverse="+reverse);
                    if (history > 0) assertArrayEquals(saved,f.floats(2+history));
                }
            }
    }

    @Test void shapedCapacityTailOverlapRejectsBeforeWritesAndIndependentRetrySucceeds() {
        Tensor h = tensor(false,2,2), x = tensor(false,2,2), w = tensor(false,2);
        var body = new ScanBody(List.of(h),List.of(x),List.of(w),List.of(h.add(x.mul(w))),List.of());
        var scan = Scan.apply(List.of(tensor(false,2,2)),List.of(tensor(false,3,2,2)),List.of(tensor(false,2)),body,false);
        var analysis = new CpuPartitionPreparer().analyze(CpuGeneralScanPreparationTest.context(query(scan.finalCarry())));
        try (var arena = java.lang.foreign.Arena.ofConfined()) {
            var memory = arena.allocate(64,4); memory.fill((byte)0x5a); var before = memory.toArray(java.lang.foreign.ValueLayout.JAVA_BYTE);
            // Logical input cells end at byte 16; the output aliases only its selected capacity tail.
            var overrides = Map.of(new ValueId(0),memory.asSlice(0,32),new ValueId(3),memory.asSlice(16,16));
            try (var f = new CpuRecurrentScanExecutionTest.Fixture(analysis,2,false,false,overrides)) {
                assertThrows(IllegalArgumentException.class,() -> f.executable.bind(f.state));
                assertArrayEquals(before,memory.toArray(java.lang.foreign.ValueLayout.JAVA_BYTE));
            }
        }
        try (var f = new CpuRecurrentScanExecutionTest.Fixture(analysis,2,false,false,Map.of())) {
            f.fill(0,1); f.fill(1,2); f.fill(2,.5f); f.executable.bind(f.state).execute();
            assertArrayEquals(new float[] {4,4,4,4},f.floats(3));
        }
    }

    @Test void shapedReusedTemplatesTiedRolesAndDuplicateRootsSumAcrossOccurrences() {
        Tensor h = tensor(false,2,3), k = tensor(true,3), x = tensor(true,2,1), w = tensor(true,3), b = tensor(true);
        var recipe = shapedRecipe(h,k,x,w,b,0);
        var body = new ScanBody(List.of(h,k),List.of(x),List.of(w,b),recipe.subList(0,2),List.of(recipe.get(1),recipe.get(1)));
        Tensor initial = tensor(false,2,3), weight = tensor(true,3), sequence = tensor(true,3,2,1), bias = tensor(true);
        var first = Scan.apply(List.of(initial,weight),List.of(sequence),List.of(weight,bias),body,false);
        var second = Scan.apply(List.of(initial,weight),List.of(sequence),List.of(weight,bias),body,true);
        Tensor root = first.finalCarry().getFirst().sum().add(second.finalCarry().getFirst().sum())
                .add(first.outputs().get(0).sum()).add(first.outputs().get(1).sum());
        Tensor seed = tensor(false); var targets = List.of(sequence,weight,bias);
        var feeds = new IdentityHashMap<Tensor,float[]>(); feeds.put(initial,new float[] {.1f,.2f,.3f,.4f,.5f,.6f});
        feeds.put(weight,new float[] {.5f,.75f,1.25f}); feeds.put(sequence,new float[] {.03125f,.0625f,.09375f,.125f,.15625f,.1875f}); feeds.put(bias,new float[] {.125f});
        // Use the same explicit seed object in both requests, not a hidden derivative constant.
        var request = new FunctionalGradientRequest(List.of(new FunctionalGradientRequest.Stage(List.of(new FunctionalGradientRequest.ForwardTensorReference(root)),
                List.of(Optional.of(seed)),targets,false,FunctionalGradientRequest.DisconnectedPolicy.ERROR)));
        var actual = CpuGeneralScanCapabilityTest.compile(List.of(root),Optional.of(request),false,new CpuCapabilityProvider()); feeds.put(seed,new float[] {1});
        Tensor forward = initial, reverse = initial, fk = weight, rk = weight;
        var emitted = new Tensor[3];
        for (int t = 0; t < 3; t++) {
            var f = shapedRecipe(forward,fk,sequence.select(0,t),weight,bias,0); forward = f.get(0); fk = f.get(1); emitted[t] = fk;
            var r = shapedRecipe(reverse,rk,sequence.select(0,2-t),weight,bias,0); reverse = r.get(0); rk = r.get(1);
        }
        Tensor savedSum = Tensor.stack(0,emitted).sum();
        Tensor referenceRoot = forward.sum().add(reverse.sum()).add(savedSum).add(savedSum);
        var referenceRequest = new FunctionalGradientRequest(List.of(new FunctionalGradientRequest.Stage(List.of(new FunctionalGradientRequest.ForwardTensorReference(referenceRoot)),
                List.of(Optional.of(seed)),targets,false,FunctionalGradientRequest.DisconnectedPolicy.ERROR)));
        var expected = executeArtifact(CpuGeneralScanCapabilityTest.compile(List.of(referenceRoot),Optional.of(referenceRequest),false,new CpuCapabilityProvider()),feeds);
        var result = executeArtifact(actual,feeds); assertEquals(expected.size(),result.size());
        for (int p = 0; p < result.size(); p++) assertArrayEquals(expected.get(p),result.get(p),5e-6f,"tied shaped role/cross-occurrence port="+p);
    }

    @Test void completeShapedClosureAndActualReverseMatchUnrolledOrdinaryTensorAd() {
        for (int variant = 0; variant < 3; variant++) for (boolean reverse : List.of(false,true))
            for (boolean optimized : List.of(false,true)) for (int t : List.of(1,3)) for (int seedClass = 0; seedClass < 3; seedClass++) {
                Tensor h = tensor(false,2,3), k = tensor(true,3), x = tensor(true,2,1), w = tensor(true,3), b = tensor(true);
                var recipe = shapedRecipe(h,k,x,w,b,variant);
                var body = new ScanBody(List.of(h,k),List.of(x),List.of(w,b),recipe.subList(0,2),recipe.subList(2,4));
                Tensor initial = tensor(false,2,3), initialK = tensor(true,3), sequence = tensor(true,t,2,1), weight = tensor(true,3), bias = tensor(true);
                var scan = Scan.apply(List.of(initial,initialK),List.of(sequence),List.of(weight,bias),body,reverse);
                var all = new ArrayList<>(scan.finalCarry()); all.addAll(scan.outputs());
                List<Tensor> roots = seedClass == 0 ? all.subList(0,2) : seedClass == 1 ? all.subList(2,4) : all;
                var seeds = roots.stream().map(r -> tensor(false,r.descriptor().shape().toLongArray())).toList();
                var targets = List.of(initialK,sequence,weight,bias);
                var stage = new FunctionalGradientRequest.Stage(roots.stream().map(r -> (FunctionalGradientRequest.OutputReference)
                        new FunctionalGradientRequest.ForwardTensorReference(r)).toList(),seeds.stream().map(Optional::of).toList(),targets,false,
                        seedClass == 1 && variant != 0 ? FunctionalGradientRequest.DisconnectedPolicy.ZERO : FunctionalGradientRequest.DisconnectedPolicy.ERROR);
                var actual = CpuGeneralScanCapabilityTest.compile(roots,Optional.of(new FunctionalGradientRequest(List.of(stage))),optimized,new CpuCapabilityProvider());
                var feeds = new IdentityHashMap<Tensor,float[]>();
                feeds.put(initial,new float[] {.1f,.2f,.3f,.4f,.5f,.6f}); feeds.put(initialK,new float[] {.2f,.3f,.4f});
                float[] seq = new float[t*2]; for (int i = 0; i < seq.length; i++) seq[i] = (i-2)*.03125f;
                feeds.put(sequence,seq); feeds.put(weight,new float[] {.5f,.75f,1.25f}); feeds.put(bias,new float[] {.125f});
                for (Tensor seed : seeds) { float[] s = new float[Math.toIntExact(seed.descriptor().shape().knownElementCount().orElseThrow())];
                    for (int i = 0; i < s.length; i++) s[i] = .25f + (i%3)*.03125f; feeds.put(seed,s); }
                var result = executeArtifact(actual,feeds);
                Tensor uh = initial, uk = initialK; var ea = new Tensor[t]; var eb = new Tensor[t];
                for (int step = 0; step < t; step++) {
                    int time = reverse ? t-1-step : step;
                    var r = shapedRecipe(uh,uk,sequence.select(0,time),weight,bias,variant);
                    uh = r.get(0); uk = r.get(1); ea[time] = r.get(2); eb[time] = r.get(3);
                }
                var unrolledAll = List.of(uh,uk,Tensor.stack(0,ea),Tensor.stack(0,eb));
                List<Tensor> unrolledRoots = seedClass == 0 ? unrolledAll.subList(0,2) : seedClass == 1 ? unrolledAll.subList(2,4) : unrolledAll;
                var referenceStage = new FunctionalGradientRequest.Stage(unrolledRoots.stream().map(r -> (FunctionalGradientRequest.OutputReference)
                        new FunctionalGradientRequest.ForwardTensorReference(r)).toList(),seeds.stream().map(Optional::of).toList(),targets,false,
                        seedClass == 1 && variant != 0 ? FunctionalGradientRequest.DisconnectedPolicy.ZERO : FunctionalGradientRequest.DisconnectedPolicy.ERROR);
                var fullReference = executeArtifact(CpuGeneralScanCapabilityTest.compile(unrolledAll,
                        Optional.of(new FunctionalGradientRequest(List.of(referenceStage))),optimized,new CpuCapabilityProvider()),feeds);
                // Keep real unselected forward roots in the reference inventory so a disconnected
                // target remains a legal requested zero; never manufacture x*0 graph membership.
                var expected = new ArrayList<float[]>();
                if (seedClass != 1) expected.addAll(fullReference.subList(0,2));
                if (seedClass != 0) expected.addAll(fullReference.subList(2,4));
                expected.addAll(fullReference.subList(4,fullReference.size()));
                assertEquals(expected.size(),result.size());
                for (int p = 0; p < result.size(); p++) {
                    assertEquals(expected.get(p).length,result.get(p).length,"shape variant="+variant+" seed="+seedClass+" port="+p+" reverse="+reverse+" T="+t);
                    for (int i = 0; i < result.get(p).length; i++) assertEquals(expected.get(p)[i],result.get(p)[i],
                            3e-6f*Math.max(1f,Math.abs(expected.get(p)[i])),"finite ordinary Tensor AD oracle, variant="+variant+" seed="+seedClass+" port="+p);
                }
            }
    }

    private static List<Tensor> shapedRecipe(Tensor h, Tensor k, Tensor x, Tensor w, Tensor b, int variant) {
        var quarter = h.descriptor().dataType() == io.github.pho001.synaptik.model.datatype.DataType.FLOAT64
                ? io.github.pho001.synaptik.model.datatype.ScalarValue.float64(.25+0x1p-38) : io.github.pho001.synaptik.model.datatype.ScalarValue.float32(.25f);
        var two = h.descriptor().dataType() == io.github.pho001.synaptik.model.datatype.DataType.FLOAT64
                ? io.github.pho001.synaptik.model.datatype.ScalarValue.float64(2d) : io.github.pho001.synaptik.model.datatype.ScalarValue.float32(2f);
        Tensor next;
        if (variant == 0) next = h.mul(w).add(x.mul(w)).add(b).tanh();
        else if (variant == 1) next = h.permute(1,0).contiguous().reshape(io.github.pho001.synaptik.model.shape.Shape.of(6))
                .reshape(io.github.pho001.synaptik.model.shape.Shape.of(2,3)).add(x.mul(w)).sub(quarter).mul(two).div(two).neg().sub(b).tanh();
        else next = h.add(x).sub(w).div(w.add(quarter)).add(b).sigmoid();
        Tensor nextK = variant == 1 ? k.add(next.sum(new int[] {0},true).squeeze(0)).neg()
                : k.add(next.sum(0,false)).sigmoid();
        Tensor e0 = variant == 0 ? next : variant == 1 ? next.sum(new int[] {0,1},true) : next.sum();
        Tensor e1 = variant == 0 ? nextK.expandDims(0).expand(io.github.pho001.synaptik.model.shape.Shape.of(2,3)).contiguous()
                : next.sumToShape(variant == 1 ? io.github.pho001.synaptik.model.shape.Shape.of(1,3) : io.github.pho001.synaptik.model.shape.Shape.of(2,1));
        return List.of(next,nextK,e0,e1);
    }

    private static List<float[]> executeArtifact(io.github.pho001.synaptik.compiler.CompileArtifacts artifact, Map<Tensor,float[]> feeds) {
        var context = io.github.pho001.synaptik.prepare.GraphPreparation.project(
                artifact,artifact.partitions().getFirst(),CpuPartitionAnalysisInputs.DEFAULT);
        var analysis = new CpuPartitionPreparer().analyze(context);
        try (var f = new CpuRecurrentScanExecutionTest.Fixture(artifact,CpuPartitionAnalysisInputs.DEFAULT,analysis,2,false,false,Map.of())) {
            for (var input : artifact.constants().bindableInputBindings()) {
                if (!f.storage.containsKey(input.valueId())) continue; // Compiled inputs need not remain physical reads.
                var tensor = feeds.keySet().stream().filter(v -> v.id().equals(input.tensorId())).findFirst().orElseThrow();
                f.floats(Math.toIntExact(input.valueId().value()),feeds.get(tensor));
            }
            f.executable.bind(f.state).execute();
            var result = new ArrayList<float[]>();
            artifact.publication().forwardBindings().forEach(p -> result.add(logicalFloats(artifact,f,p.valueId())));
            artifact.publication().gradientBindings().forEach(p -> result.add(logicalFloats(artifact,f,p.valueId())));
            return result;
        }
    }

    private static float[] logicalFloats(io.github.pho001.synaptik.compiler.CompileArtifacts artifact,
            CpuRecurrentScanExecutionTest.Fixture f, ValueId id) {
        var d = artifact.graph().values().stream().filter(v -> v.id().equals(id)).findFirst().orElseThrow().descriptor();
        var layout = d.layout().orElseGet(() -> io.github.pho001.synaptik.model.layout.LayoutDescriptor.contiguous(d.shape()));
        long[] shape = d.shape().toLongArray(), strides = layout.strides();
        float[] values = new float[Math.toIntExact(d.shape().knownElementCount().orElseThrow())];
        var segment = f.segment(Math.toIntExact(id.value()));
        for (int i = 0; i < values.length; i++) {
            long remaining = i, address = layout.storageOffset();
            for (int axis = shape.length-1; axis >= 0; axis--) { address += (remaining%shape[axis])*strides[axis]; remaining /= shape[axis]; }
            values[i] = segment.getAtIndex(ValueLayout.JAVA_FLOAT,address);
        }
        return values;
    }

    private static double[] logicalNumbers(io.github.pho001.synaptik.compiler.CompileArtifacts artifact,
            CpuRecurrentScanExecutionTest.Fixture f, ValueId id) {
        var d = artifact.graph().values().stream().filter(v -> v.id().equals(id)).findFirst().orElseThrow().descriptor();
        var layout = d.layout().orElseGet(() -> io.github.pho001.synaptik.model.layout.LayoutDescriptor.contiguous(d.shape()));
        long[] shape = d.shape().toLongArray(), strides = layout.strides();
        double[] values = new double[Math.toIntExact(d.shape().knownElementCount().orElseThrow())];
        var segment = f.segment(Math.toIntExact(id.value()));
        for (int i = 0; i < values.length; i++) {
            long remaining = i, address = layout.storageOffset();
            for (int axis = shape.length-1; axis >= 0; axis--) { address += (remaining%shape[axis])*strides[axis]; remaining /= shape[axis]; }
            values[i] = CpuRecurrentScanExecutionTest.number(d.dataType(),segment,address);
        }
        return values;
    }

    @Test void shapedRawMovementAndNoReductionPreserveStoredBitsWithoutArithmetic() {
        int[] raw = {0,0x80000000,1,0x80000001,0x7f800000,0xff800000,
                0x7fc12345,0xffc54321,0x7fa12345,0xffa54321,0x3f800000,0xbf800000};
        for (boolean reverse : List.of(false,true)) for (int t : List.of(0,1,3)) {
            Tensor h = tensor(false,2,3), x = tensor(false,2,3);
            var body = new ScanBody(List.of(h),List.of(x),List.of(),List.of(h),List.of(
                    x.permute(1,0).contiguous(),x.sum(new int[0],false),x.sumToShape(io.github.pho001.synaptik.model.shape.Shape.of(2,3))));
            var scan = Scan.apply(List.of(tensor(false,2,3)),List.of(tensor(false,t,2,3)),List.of(),body,reverse);
            var roots = new ArrayList<>(scan.finalCarry()); roots.addAll(scan.outputs());
            var analysis = new CpuPartitionPreparer().analyze(CpuGeneralScanPreparationTest.context(query(roots)));
            try (var f = new CpuRecurrentScanExecutionTest.Fixture(analysis,2,false,false,Map.of())) {
                for (int i = 0; i < 6; i++) f.segment(0).setAtIndex(java.lang.foreign.ValueLayout.JAVA_INT,i,raw[i+6]);
                int[] expected = new int[t*6], permuted = new int[t*6];
                for (int time = 0; time < t; time++) for (int i = 0; i < 6; i++) {
                    int bits = raw[(time*6+i)%raw.length]; expected[time*6+i] = bits;
                    f.segment(1).setAtIndex(java.lang.foreign.ValueLayout.JAVA_INT,time*6L+i,bits);
                    permuted[time*6+(i%3)*2+i/3] = bits;
                }
                f.executable.bind(f.state).execute();
                assertArrayEquals(Arrays.copyOfRange(raw,6,12),f.segment(2).toArray(java.lang.foreign.ValueLayout.JAVA_INT));
                assertArrayEquals(permuted,f.segment(3).toArray(java.lang.foreign.ValueLayout.JAVA_INT));
                assertArrayEquals(expected,f.segment(4).toArray(java.lang.foreign.ValueLayout.JAVA_INT));
                assertArrayEquals(expected,f.segment(5).toArray(java.lang.foreign.ValueLayout.JAVA_INT));
            }
        }
    }

    @Test void permutedOldCarryAndDifferentShapeForwardingSurviveWholeCoordinateStages() {
        for (boolean reverse : List.of(false, true)) for (int mode = 0; mode < 3; mode++) {
            Tensor a = tensor(false, 2, 2), b = tensor(false, 2, 2), small = tensor(false, 2), x = tensor(false, 2, 2);
            Tensor next = b.permute(1, 0).add(x), smallNext = a.sum(0, false);
            var body = new ScanBody(List.of(a,b,small), List.of(x), List.of(),
                    List.of(next, a.permute(1, 0), smallNext), List.of(a, smallNext));
            var scan = Scan.apply(List.of(tensor(false,2,2), tensor(false,2,2), tensor(false,2)),
                    List.of(tensor(false,3,2,2)), List.of(), body, reverse);
            var roots = new ArrayList<>(scan.finalCarry()); roots.addAll(scan.outputs());
            var analysis = new CpuPartitionPreparer().analyze(CpuGeneralScanPreparationTest.context(query(roots)));
            try (var f = new CpuRecurrentScanExecutionTest.Fixture(analysis, mode, true, true, Map.of())) {
                float[] av = {1,2,4,8}, bv = {16,32,64,128}, sv = {0,0}, seq = {1,2,3,4,5,6,7,8,9,10,11,12};
                f.floats(0,av); f.floats(1,bv); f.floats(2,sv); f.floats(3,seq);
                float[] ah = new float[12], sh = new float[6];
                for (int k = 0; k < 3; k++) {
                    int t = reverse ? 2-k : k;
                    System.arraycopy(av,0,ah,t*4,4);
                    float[] na = {bv[0]+seq[t*4],bv[2]+seq[t*4+1],bv[1]+seq[t*4+2],bv[3]+seq[t*4+3]};
                    float[] nb = {av[0],av[2],av[1],av[3]};
                    sv = new float[] {av[0]+av[2],av[1]+av[3]}; System.arraycopy(sv,0,sh,t*2,2);
                    av = na; bv = nb;
                }
                f.executable.bind(f.state).execute();
                assertArrayEquals(av,f.floats(4)); assertArrayEquals(bv,f.floats(5)); assertArrayEquals(sv,f.floats(6));
                assertArrayEquals(ah,f.floats(7)); assertArrayEquals(sh,f.floats(8));
            }
        }
    }

    @Test void shapedBroadcastReductionAndActualReverseRetainRepresentedSites() {
        for (boolean reverse : List.of(false, true)) for (boolean optimized : List.of(false, true))
            for (int t : List.of(0, 1, 3)) for (int batch : List.of(0, 2)) {
                int width = 3, n = batch * width;
                Tensor h = tensor(false, batch, width), x = tensor(true, batch, width), w = tensor(true, width);
                Tensor next = h.mul(w).add(x).tanh();
                var body = new ScanBody(List.of(h), List.of(x), List.of(w), List.of(next), List.of(next.sum(0, false)));
                Tensor initial = tensor(false, batch, width), sequence = tensor(true, t, batch, width), weight = tensor(true, width);
                var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(weight), body, reverse);
                var roots = List.of(scan.finalCarry().getFirst(), scan.outputs().getFirst());
                Tensor hs = tensor(false, batch, width), ys = tensor(false, t, width);
                var request = new FunctionalGradientRequest(List.of(new FunctionalGradientRequest.Stage(roots.stream()
                        .map(r -> (FunctionalGradientRequest.OutputReference) new FunctionalGradientRequest.ForwardTensorReference(r)).toList(),
                        List.of(Optional.of(hs), Optional.of(ys)), List.of(sequence, weight), false,
                        t == 0 ? FunctionalGradientRequest.DisconnectedPolicy.ZERO : FunctionalGradientRequest.DisconnectedPolicy.ERROR)));
                var artifact = CpuGeneralScanCapabilityTest.compile(roots, Optional.of(request), optimized, new CpuCapabilityProvider());
                var context = io.github.pho001.synaptik.prepare.GraphPreparation.project(
                        artifact,artifact.partitions().getFirst(),CpuPartitionAnalysisInputs.DEFAULT);
                var analysis = new CpuPartitionPreparer().analyze(context);
                try (var f = new CpuRecurrentScanExecutionTest.Fixture(artifact,CpuPartitionAnalysisInputs.DEFAULT,analysis, 2, false, false, Map.of())) {
                    for (var input : artifact.constants().bindableInputBindings()) if (f.storage.containsKey(input.valueId())) {
                        float value = input.tensorId().equals(initial.id()) ? .2f : input.tensorId().equals(sequence.id()) ? .125f
                                : input.tensorId().equals(weight.id()) ? batch == 0 ? Float.NaN : .75f
                                : input.tensorId().equals(hs.id()) ? .5f : .25f;
                        f.fill(Math.toIntExact(input.valueId().value()),value);
                    }
                    f.executable.bind(f.state).execute();
                    float[] state = new float[n], old = new float[t*n], y = new float[t*n], emission = new float[t*width]; Arrays.fill(state, .2f);
                    for (int k = 0; k < t; k++) { int time = reverse ? t-1-k : k;
                        for (int b = 0; b < batch; b++) for (int j = 0; j < width; j++) {
                            int i = b*width+j, p = time*n+i; old[p] = state[i];
                            float product = state[i]*.75f; float sum = product+.125f;
                            y[p] = state[i] = (float) StrictMath.tanh(sum); emission[time*width+j] += state[i];
                        }
                    }
                    float[] carry = new float[n], dx = new float[t*n], dw = new float[width]; Arrays.fill(carry, .5f);
                    for (int k = 0; k < t; k++) { int time = reverse ? k : t-1-k;
                        float[] contribution = new float[width];
                        for (int b = 0; b < batch; b++) for (int j = 0; j < width; j++) {
                            int i = b*width+j, p = time*n+i;
                            float incoming = carry[i]+.25f, square = y[p]*y[p], derivative = 1f-square;
                            dx[p] = incoming*derivative; contribution[j] += dx[p]*old[p]; carry[i] = dx[p]*.75f;
                        }
                        for (int j = 0; j < width; j++) dw[j] += contribution[j];
                    }
                    assertArrayEquals(state, logicalFloats(artifact,f,artifact.publication().forwardBindings().get(0).valueId()));
                    assertArrayEquals(emission, logicalFloats(artifact,f,artifact.publication().forwardBindings().get(1).valueId()));
                    assertArrayEquals(dx, logicalFloats(artifact,f,artifact.publication().gradientBindings().get(0).valueId()));
                    assertArrayEquals(dw, logicalFloats(artifact,f,artifact.publication().gradientBindings().get(1).valueId()));
                }
            }
    }

    @Test void emissionOnlyRetainsDirectAndTransitiveNextCarrySources() {
        for (boolean reverse : List.of(false, true)) for (int count : List.of(2, 3)) {
            var carries = new ArrayList<Tensor>();
            for (int c = 0; c < count; c++) carries.add(tensor(false, 1));
            Tensor x = tensor(false, 1);
            var next = new ArrayList<>(carries.subList(1, count));
            next.add(carries.getFirst().add(x));
            var body = new ScanBody(carries, List.of(x), List.of(), next, List.of(carries.getFirst()));
            var initial = carries.stream().map(ignored -> tensor(false, 1)).toList();
            var scan = Scan.apply(initial, List.of(tensor(false, 5, 1)), List.of(), body, reverse);
            var analysis = new CpuPartitionPreparer().analyze(CpuGeneralScanPreparationTest.context(query(scan.outputs())));
            try (var f = new CpuRecurrentScanExecutionTest.Fixture(analysis, 1, false, false, Map.of())) {
                float[] state = new float[count];
                for (int c = 0; c < count; c++) { state[c] = c == 0 ? 1 : c * 100; f.floats(c, state[c]); }
                float[] sequence = {2, 4, 8, 16, 32}, expected = new float[5];
                f.floats(count, sequence);
                for (int k = 0; k < 5; k++) {
                    int time = reverse ? 4 - k : k;
                    expected[time] = state[0];
                    float last = state[0] + sequence[time];
                    System.arraycopy(state, 1, state, 0, count - 1); state[count - 1] = last;
                }
                f.executable.bind(f.state).execute();
                assertArrayEquals(expected, f.floats(2 * count + 1), "carry chain=" + count + " reverse=" + reverse);
            }
        }
    }
    @Test void rawEmptyTimeCopiesPreservePayloadsAndWritesFailBeforeOverlapMutation() {
        var q = CpuGeneralScanGeneratedKernelTest.query(0, 4, false, List.of(0, 1));
        var analysis = new CpuPartitionPreparer().analyze(CpuGeneralScanPreparationTest.context(q));
        try (var f = new CpuRecurrentScanExecutionTest.Fixture(analysis, 2, false, true, Map.of())) {
            int[] raw = {0x80000000, 0x7fc12345, 1, 0x7f800000};
            for (int i = 0; i < raw.length; i++) f.segment(0).setAtIndex(java.lang.foreign.ValueLayout.JAVA_INT, i, raw[i]);
            f.executable.bind(f.state).execute();
            assertArrayEquals(raw, f.segment(2).toArray(java.lang.foreign.ValueLayout.JAVA_INT));
            assertEquals(0, f.segment(3).byteSize());
        }
        var live = new CpuPartitionPreparer().analyze(CpuGeneralScanPreparationTest.context(
                CpuGeneralScanGeneratedKernelTest.query(3, 2, false, List.of(0, 1))));
        try (var arena = java.lang.foreign.Arena.ofConfined()) {
            var memory = arena.allocate(64, 4); memory.fill((byte) 0x5a);
            var overrides = Map.of(new ValueId(0), memory.asSlice(0, 32), new ValueId(2), memory.asSlice(16, 8));
            try (var f = new CpuRecurrentScanExecutionTest.Fixture(live, 1, false, false, overrides)) {
                byte[] before = memory.toArray(java.lang.foreign.ValueLayout.JAVA_BYTE);
                assertThrows(IllegalArgumentException.class, () -> f.executable.bind(f.state));
                assertArrayEquals(before, memory.toArray(java.lang.foreign.ValueLayout.JAVA_BYTE));
            }
        }
    }

    @Test void ordinarySpecialValuesAndScalarArithmeticRetainProducingSites() {
        for (boolean reverse : List.of(false, true)) {
            Tensor h = tensor(false, 6), x = tensor(false, 6);
            Tensor next = h.add(x).sub(io.github.pho001.synaptik.model.datatype.ScalarValue.float32(0f))
                    .mul(io.github.pho001.synaptik.model.datatype.ScalarValue.float32(2f)).div(x).neg();
            var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(next), List.of(next));
            var scan = Scan.apply(List.of(tensor(false, 6)), List.of(tensor(false, 3, 6)), List.of(), body, reverse);
            var analysis = new CpuPartitionPreparer().analyze(CpuGeneralScanPreparationTest.context(query(
                    List.of(scan.finalCarry().getFirst(), scan.outputs().getFirst()))));
            float[] initial = {Float.NaN, Float.POSITIVE_INFINITY, -0f, Float.MAX_VALUE, Float.MIN_VALUE, 1f};
            float[] row = {1f, Float.NEGATIVE_INFINITY, -0f, 2f, .5f, 0f};
            try (var f = new CpuRecurrentScanExecutionTest.Fixture(analysis, 0, true, false, Map.of())) {
                f.floats(0, initial); float[] sequence = new float[18];
                for (int t = 0; t < 3; t++) System.arraycopy(row, 0, sequence, t * 6, 6);
                f.floats(1, sequence); f.executable.bind(f.state).execute();
                float[] expected = initial.clone();
                for (int t = 0; t < 3; t++) for (int i = 0; i < 6; i++) {
                    float add = expected[i] + row[i]; float sub = add - 0f; float mul = sub * 2f;
                    float div = mul / row[i]; expected[i] = -div;
                }
                assertArrayEquals(expected, f.floats(2));
            }
        }
    }
    @Test void actualCompilerSiblingRunsWithFalseInitialFlagAndTrainableSequenceAndWeight() {
        for (boolean reverse : List.of(false, true)) for (int t : List.of(0, 1, 3, 257)) {
            Tensor h = tensor(false, 1), x = tensor(true, 1), w = tensor(true, 1);
            Tensor next = h.mul(w).add(x).tanh();
            var body = new ScanBody(List.of(h), List.of(x), List.of(w), List.of(next), List.of(next));
            Tensor initial = tensor(false, 1), sequence = tensor(true, t, 1), weight = tensor(true, 1);
            var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(weight), body, reverse);
            var roots = List.of(scan.finalCarry().getFirst(), scan.outputs().getFirst());
            Tensor finalSeed = tensor(false, 1), outputSeed = tensor(false, t, 1);
            var request = new FunctionalGradientRequest(List.of(new FunctionalGradientRequest.Stage(roots.stream()
                    .map(r -> (FunctionalGradientRequest.OutputReference) new FunctionalGradientRequest.ForwardTensorReference(r)).toList(),
                    List.of(Optional.of(finalSeed), Optional.of(outputSeed)), List.of(sequence, weight), false,
                    t == 0 ? FunctionalGradientRequest.DisconnectedPolicy.ZERO : FunctionalGradientRequest.DisconnectedPolicy.ERROR)));
            var artifact = CpuGeneralScanCapabilityTest.compile(roots, Optional.of(request), false, new CpuCapabilityProvider());
            assertEquals(1, artifact.partitions().size());
            var context = io.github.pho001.synaptik.prepare.GraphPreparation.project(
                    artifact,artifact.partitions().getFirst(),CpuPartitionAnalysisInputs.DEFAULT);
            var analysis = new CpuPartitionPreparer().analyze(context);
            try (var f = new CpuRecurrentScanExecutionTest.Fixture(artifact,CpuPartitionAnalysisInputs.DEFAULT,analysis, 2, false, false, Map.of())) {
                for (var input : artifact.constants().bindableInputBindings()) if (f.storage.containsKey(input.valueId())) {
                    float value = input.tensorId().equals(initial.id()) ? .2f : input.tensorId().equals(sequence.id()) ? .125f
                            : input.tensorId().equals(weight.id()) ? .75f : input.tensorId().equals(finalSeed.id()) ? .5f : .25f;
                    f.fill(Math.toIntExact(input.valueId().value()),value);
                }
                f.executable.bind(f.state).execute();
                float state = .2f; float[] old = new float[t], y = new float[t];
                for (int k = 0; k < t; k++) { int time = reverse ? t - 1 - k : k;
                    old[time] = state; float product = state * .75f; float sum = product + .125f;
                    y[time] = state = (float) Math.tanh(sum); }
                float carry = .5f, dw = 0; float[] dx = new float[t];
                for (int k = 0; k < t; k++) { int time = reverse ? k : t - 1 - k;
                    float incoming = carry + .25f; float square = y[time] * y[time];
                    float derivative = 1f - square; float d = incoming * derivative;
                    dx[time] = d; carry = d * .75f; dw = dw + d * old[time]; }
                var gradientBindings = artifact.publication().gradientBindings();
                assertArrayEquals(dx, logicalFloats(artifact,f,gradientBindings.get(0).valueId()));
                assertArrayEquals(new float[] {dw}, logicalFloats(artifact,f,gradientBindings.get(1).valueId()));
            }
        }
    }
    @Test void assignedFrameBackedEntryRunsOnHeapNativeAndMixedSlices() {
        for (int mode = 0; mode < 3; mode++) for (boolean reverse : List.of(false, true)) {
            var analysis = new CpuPartitionPreparer().analyze(CpuGeneralScanPreparationTest.context(
                    CpuGeneralScanGeneratedKernelTest.query(3, 2, reverse, List.of(0, 1))));
            try (var f = new CpuRecurrentScanExecutionTest.Fixture(analysis, mode, true, true, Map.of())) {
                f.floats(0, .25f, -.25f); f.floats(1, .1f, .2f, .3f, .4f, .5f, .6f);
                f.executable.bind(f.state).execute();
                float[] state = {.25f, -.25f}, history = new float[6], x = {.1f,.2f,.3f,.4f,.5f,.6f};
                for (int k = 0; k < 3; k++) { int t = reverse ? 2 - k : k;
                    for (int i = 0; i < 2; i++) history[t * 2 + i] = state[i] = (float) Math.tanh(state[i] + x[t * 2 + i]); }
                assertArrayEquals(state, f.floats(2)); assertArrayEquals(history, f.floats(3));
            }
        }
    }

    @Test void oldCarrySurvivesAcrossThreeStepsInBothDirections() {
        for (boolean reverse : List.of(false, true)) {
            Tensor h = tensor(false, 1), old = tensor(false, 1), step = tensor(false, 1);
            var body = new ScanBody(List.of(h, old), List.of(step), List.of(), List.of(h.add(step), h), List.of());
            var scan = Scan.apply(List.of(tensor(false, 1), tensor(false, 1)), List.of(tensor(false, 3, 1)), List.of(), body, reverse);
            var q = query(scan.finalCarry());
            var analysis = new CpuPartitionPreparer().analyze(CpuGeneralScanPreparationTest.context(q));
            try (var f = new CpuRecurrentScanExecutionTest.Fixture(analysis, 2, true, true, Map.of())) {
                f.floats(0, 1); f.floats(1, 100); f.floats(2, 2, 4, 8);
                f.executable.bind(f.state).execute();
                assertArrayEquals(new float[] {15}, f.floats(3));
                assertArrayEquals(new float[] {reverse ? 13 : 7}, f.floats(4));
            }
        }
    }
}
