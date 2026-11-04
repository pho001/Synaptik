package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.*;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.model.datatype.*;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.operation.*;
import io.github.pho001.synaptik.model.operation.layout.*;
import io.github.pho001.synaptik.model.operation.reduction.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.math.BigInteger;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Exact count/geometry and provenance tests with metadata-only large domains. */
public class CpuStaticTypedOneCountFoldingTest {
    /** Checks represented-bit rounding and modular results without materializing large domains. */
    @Test void exactCountRoundingMatchesIndependentIntegerOracleWithoutLargeBuffers() {
        long[] counts = {0,1,255,256,257,258,259,2047,2048,2049,2050,2051,
                65503,65504,65519,65520,65521,16_777_215,16_777_216,16_777_217,16_842_753,
                (1L<<32)+3,(1L<<53)-1,1L<<53,(1L<<53)+1,(1L<<53)+3,Long.MAX_VALUE};
        for (var type : types()) for (long count : counts)
            assertEquals(oracle(type,count), CpuAggregateLowering.countImmediate(type,count).bits(), type+":"+count);
        var random = new Random(48);
        for (int i=0;i<2000;i++) for (var type: types()) {
            long count=random.nextLong() & Long.MAX_VALUE;
            assertEquals(oracle(type,count), CpuAggregateLowering.countImmediate(type,count).bits(), type+":"+count);
        }
        assertThrows(IllegalArgumentException.class,()->CpuAggregateLowering.countImmediate(DataType.INT64,-1));
    }

    /** Distinguishes the SUM working boundary from a direct integer-to-low cast at a double-rounding tie. */
    @Test void bfloatCountRoundsThroughFloatWorkingPrecisionBeforeFinalNarrowing() {
        long count = 16_842_753;
        int workingBits = (int) oracle(DataType.FLOAT32, count);
        float working = Float.intBitsToFloat(workingBits);
        assertEquals(16_842_752f, working);
        short expected = BFloat16Bits.fromFloat(working);
        assertEquals(oracle(DataType.BFLOAT16, count),
                CpuAggregateLowering.countImmediate(DataType.BFLOAT16, count).bits());
        assertEquals(Short.toUnsignedLong(expected),
                CpuAggregateLowering.countImmediate(DataType.BFLOAT16, count).bits());
        short directCast = io.github.pho001.synaptik.model.operation.elementwise.cast.CastValueConversions
                .convert(ScalarValue.int64(count), DataType.BFLOAT16).bfloat16Bits();
        assertNotEquals(directCast, expected, "direct casts must not erase the SUM FLOAT32 working boundary");
        assertEquals(16_777_216f, BFloat16Bits.toFloat(expected));
        assertEquals(16_908_288f, BFloat16Bits.toFloat(directCast));
        assertEquals(Short.toUnsignedLong(directCast),
                CpuAggregateLowering.countImmediate(DataType.BFLOAT16, count + 1).bits(),
                "the next FLOAT32 working value crosses the low rounding midpoint");
    }

    /** Distinguishes selected-empty, output-empty and no-axis domains across admitted forms. */
    @Test void allTypesFormsAndEmptyDomainsPreserveOutputMetadata() {
        for (var type:types()) for (int chain=0;chain<=2;chain++) {
            check(type,Shape.of(2,3,4),NoOperationAttrs.INSTANCE,Shape.scalar(),chain,24,1);
            check(type,Shape.of(2,3,4),new AxisReductionAttrs(1,true),Shape.of(2,1,4),chain,3,8);
            check(type,Shape.of(2,3,4),new MultiAxisReductionAttrs(List.of(2,0),false),Shape.of(3),chain,8,3);
            check(type,Shape.of(2,3,4),new SumToShapeAttrs(Shape.of(3,1)),Shape.of(3,1),chain,8,3);
            check(type,Shape.of(2,0,4),new AxisReductionAttrs(1,false),Shape.of(2,4),chain,0,8);
            check(type,Shape.of(0,3),new AxisReductionAttrs(1,false),Shape.of(0),chain,3,0);
            check(type,Shape.of(2,3),new MultiAxisReductionAttrs(List.of(),false),Shape.of(2,3),chain,1,6);
            check(type,Shape.scalar(),NoOperationAttrs.INSTANCE,Shape.scalar(),chain,1,1);
            check(type,Shape.of(1,3),new SumToShapeAttrs(Shape.of(1,3)),Shape.of(1,3),chain,1,3);
        }
    }

    /** Preserves byte overflow and affine prefix rejection even with a trailing zero extent. */
    @Test void largeMetadataCountsAndByteOverflowKeepCheckedAdmission() {
        check(DataType.BFLOAT16,Shape.of(16_842_753),NoOperationAttrs.INSTANCE,Shape.scalar(),2,16_842_753,1);
        check(DataType.INT32,Shape.of((1L<<32)+3),NoOperationAttrs.INSTANCE,Shape.scalar(),0,(1L<<32)+3,1);
        assertThrows(ArithmeticException.class,()->new CpuPartitionLowering().lower(context(
                DataType.FLOAT64,Shape.of(Long.MAX_VALUE/8+1),NoOperationAttrs.INSTANCE,Shape.scalar(),0,true)));
        assertThrows(ArithmeticException.class,()->Shape.of(Long.MAX_VALUE,2).knownElementCount());
        assertThrows(ArithmeticException.class,()->new CpuPartitionLowering().lower(context(
                DataType.FLOAT32,Shape.of(Long.MAX_VALUE,2,0),NoOperationAttrs.INSTANCE,Shape.scalar(),1,true)));
    }

    /** Rejects inferred, non-one or wrong-typed facts and unsupported shape-transform chains. */
    @Test void onlyExplicitOneFactsAndTheBoundedChainProveConstancy() {
        var base=context(DataType.FLOAT32,Shape.of(3),NoOperationAttrs.INSTANCE,Shape.scalar(),0,false);
        assertInstanceOf(CpuAggregateIr.class,new CpuPartitionLowering().lower(base).portableKernelIr());
        var wrong=new PrepareContext<>(base.partition(),base.nodes(),base.values(),base.memoryRequirements(),
                Map.of(new ValueId(0),ScalarValue.float32(2)),base.backendInputs());
        assertInstanceOf(CpuAggregateIr.class,new CpuPartitionLowering().lower(wrong).portableKernelIr());
        assertThrows(IllegalArgumentException.class,()->new PrepareContext<>(base.partition(),base.nodes(),base.values(),base.memoryRequirements(),
                Map.of(new ValueId(0),ScalarValue.int32(1)),base.backendInputs()));
        for (var kind:List.of(ShapeTransformKind.RESHAPE,ShapeTransformKind.EXPAND)) {
            var c=context(DataType.FLOAT32,Shape.of(1),NoOperationAttrs.INSTANCE,Shape.scalar(),1,true);
            var first=c.nodes().getFirst();
            var changed=new CompiledNode(first.id(),new Operation(kind,new TargetShapeAttrs(Shape.of(1))),first.inputs(),first.outputs());
            var nodes=List.of(changed,c.nodes().getLast());
            var modified=new PrepareContext<>(c.partition(),nodes,c.values(),c.memoryRequirements(),c.constants(),c.backendInputs());
            assertEquals(kind==ShapeTransformKind.EXPAND,new CpuAggregateLowering().foldStaticTypedOne(modified).isPresent());
        }
    }

    /** Keeps each affine stage materialized when publication or an outside consumer requires it. */
    @Test void publicationAndExternalConsumerAtEitherAbsorbedStageRetainFallback() {
        for (int stage=1;stage<=2;stage++) for (boolean external:List.of(false,true)) {
            var c=context(DataType.FLOAT32,Shape.of(2,3),NoOperationAttrs.INSTANCE,Shape.scalar(),2,true);
            var outside=new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,List.of(new NodeId(99)));
            int selected=stage;
            var memory=c.memoryRequirements().stream().map(r->r.valueId().equals(new ValueId(selected))
                    ? new LogicalMemoryRequirement(r.valueId(),r.descriptor(),r.producerPartition(),
                        external?List.of(c.partition(),outside):r.consumerPartitions(),!external):r).toList();
            var blocked=new PrepareContext<>(c.partition(),c.nodes(),c.values(),memory,c.constants(),c.backendInputs());
            assertTrue(new CpuAggregateLowering().foldStaticTypedOne(blocked).isEmpty());
            assertTrue(new CpuPartitionDagDecomposer().decompose(blocked,new CpuPartitionLowering()).stream()
                    .noneMatch(u->u.lowering().portableKernelIr() instanceof CpuConstantWriteIr));
        }
    }

    /**
     * Checks one metadata-only lowering against the independent represented-count oracle.
     * @param type non-null supported numeric source/result type
     * @param domain non-null fully static SUM source shape
     * @param attrs non-null admitted SUM attributes
     * @param output non-null exact result shape
     * @param chain zero for a direct splat, one for EXPAND, two for EXPAND/CONTIGUOUS
     * @param count expected nonnegative contributors per output cell
     * @param cells expected nonnegative output cardinality
     */
    private static void check(DataType type,Shape domain,OperationAttrs attrs,Shape output,int chain,long count,long cells) {
        var context=context(type,domain,attrs,output,chain,true);
        var lowered=new CpuPartitionLowering().lower(context);
        var ir=assertInstanceOf(CpuConstantWriteIr.class,lowered.portableKernelIr());
        assertEquals(oracle(type,count),ir.immediate().bits());
        assertEquals(cells,lowered.elementCount());
        assertEquals(1,lowered.boundaryValues().size());
        assertEquals(chain,lowered.virtualValues().size());
        assertEquals(output,context.values().getLast().descriptor().shape());
        assertTrue(lowered.aggregateGeometry().isEmpty());
        assertTrue(lowered.kernelIr().instructions().isEmpty());
        assertTrue(lowered.kernelIr().values().stream().noneMatch(v->v.kind()==CpuKernelIr.Value.Kind.INPUT));
    }

    /**
     * Builds an explicit splat followed by zero, one or two private affine nodes and SUM.
     * @param type non-null exact numeric source/result type
     * @param domain non-null fully static SUM domain
     * @param attrs non-null exact supported SUM attributes
     * @param output non-null exact SUM result shape
     * @param chain zero for direct splat, one for EXPAND, two for EXPAND/CONTIGUOUS
     * @param explicit whether to project the exact source-one fact
     * @return immutable cold fixture with public final output and no physical allocation
     */
    public static PrepareContext<CpuPartitionAnalysisInputs> context(DataType type,Shape domain,
            OperationAttrs attrs,Shape output,int chain,boolean explicit) {
        var descriptors=new ArrayList<io.github.pho001.synaptik.model.tensor.TensorDescriptor>();
        descriptors.add(CpuScatterLoweringTest.desc(type,chain==0?domain:Shape.scalar()));
        var nodes=new ArrayList<CompiledNode>();
        for(int i=0;i<chain;i++) {
            descriptors.add(i==0?CpuIndexingLoweringTest.descriptor(type,domain,
                    io.github.pho001.synaptik.model.layout.LayoutDescriptor.of(domain,new long[domain.rank()],0,true))
                    :CpuScatterLoweringTest.desc(type,domain));
            nodes.add(new CompiledNode(new NodeId(i),new Operation(i==0?ShapeTransformKind.EXPAND:ContiguousKind.CONTIGUOUS,
                    i==0?new TargetShapeAttrs(domain):NoOperationAttrs.INSTANCE),List.of(new ValueId(i)),List.of(new ValueId(i+1))));
        }
        descriptors.add(CpuScatterLoweringTest.desc(type,output));
        nodes.add(new CompiledNode(new NodeId(chain),new Operation(AggregateReductionKind.SUM,attrs),
                List.of(new ValueId(chain)),List.of(new ValueId(chain+1))));
        var c=CpuRecurrentScanLoweringTest.project(descriptors,nodes);
        var memory=c.memoryRequirements().stream().map(r->new LogicalMemoryRequirement(r.valueId(),r.descriptor(),
                r.producerPartition(),r.consumerPartitions(),r.valueId().equals(new ValueId(chain+1)))).toList();
        return new PrepareContext<>(c.partition(),c.nodes(),c.values(),memory,explicit?Map.of(new ValueId(0),one(type)):Map.of(),CpuPartitionAnalysisInputs.DEFAULT);
    }

    /**
     * Supplies the six separately checked represented count types in stable test order.
     * @return non-null immutable numeric count-type inventory
     */
    public static List<DataType> types() { return List.of(DataType.FLOAT32,DataType.FLOAT64,DataType.BFLOAT16,DataType.FLOAT16,DataType.INT32,DataType.INT64); }

    /**
     * Creates a typed immutable +1 fact without inspecting or allocating Tensor storage.
     * @param type non-null supported numeric type
     * @return non-null exact typed positive-one fact
     * @throws IllegalArgumentException if {@code type} is BOOL
     * @throws NullPointerException if {@code type} is {@code null}
     */
    public static ScalarValue one(DataType type) { return switch(type) {
        case FLOAT32->ScalarValue.float32(1); case FLOAT64->ScalarValue.float64(1);
        case BFLOAT16->ScalarValue.bfloat16(1); case FLOAT16->ScalarValue.float16(1);
        case INT32->ScalarValue.int32(1); case INT64->ScalarValue.int64(1); default->throw new IllegalArgumentException();
    }; }

    /**
     * Computes unbounded-integer working rounding, then the low storage boundary when present.
     * @param type non-null supported numeric result type
     * @param count exact nonnegative contributor count
     * @return represented result bits, or same-width modular integral bits
     */
    private static long oracle(DataType type,long count) {
        if(type==DataType.INT64) return count;
        if(type==DataType.INT32) return BigInteger.valueOf(count).and(BigInteger.valueOf(0xffff_ffffL)).longValueExact();
        if (type.byteWidth() < Float.BYTES) {
            float working = Float.intBitsToFloat((int) oracle(DataType.FLOAT32, count));
            return switch (type) {
                case BFLOAT16 -> Short.toUnsignedLong(io.github.pho001.synaptik.model.datatype.BFloat16Bits.fromFloat(working));
                case FLOAT16 -> Short.toUnsignedLong(Float.floatToFloat16(working));
                default -> throw new IllegalArgumentException("unsupported count oracle type");
            };
        }
        if(count==0) return 0;
        int fraction=type==DataType.FLOAT64?52:23;
        int bias=type==DataType.FLOAT64?1023:127;
        var exact=BigInteger.valueOf(count);int exponent=exact.bitLength()-1;
        int shift=exponent-fraction;
        BigInteger rounded;
        if(shift<=0) rounded=exact.shiftLeft(-shift);
        else {
            var qr=exact.divideAndRemainder(BigInteger.ONE.shiftLeft(shift));
            int cmp=qr[1].shiftLeft(1).compareTo(BigInteger.ONE.shiftLeft(shift));
            rounded=qr[0].add(cmp>0||cmp==0&&qr[0].testBit(0)?BigInteger.ONE:BigInteger.ZERO);
        }
        if(rounded.bitLength()>fraction+1) { rounded=rounded.shiftRight(1);exponent++; }
        if(exponent>bias) return ((1L<<(type==DataType.FLOAT64?11:8))-1)<<fraction;
        return ((long)(exponent+bias)<<fraction)|rounded.clearBit(fraction).longValueExact();
    }
}
