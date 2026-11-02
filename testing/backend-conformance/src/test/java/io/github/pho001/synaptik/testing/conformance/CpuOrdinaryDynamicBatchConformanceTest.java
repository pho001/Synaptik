package io.github.pho001.synaptik.testing.conformance;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.*;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import io.github.pho001.synaptik.config.compile.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.*;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.shape.*;
import io.github.pho001.synaptik.model.tensor.*;
import io.github.pho001.synaptik.prepare.GraphPreparation;
import io.github.pho001.synaptik.runtime.run.*;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Capability answers and the constant-byte symbolic Runtime boundary stay aligned. */
class CpuOrdinaryDynamicBatchConformanceTest {
    @Test void providerAdmitsOnlyBoundedHomogeneousRankTwoClosure() {
        var cpu=new CpuCapabilityProvider();
        var b=new DynamicDimension("B");var c=new DynamicDimension("C");
        var x=descriptor(DataType.FLOAT32,Shape.ofDimensions(b,new StaticDimension(4)));
        var w=descriptor(DataType.FLOAT32,Shape.of(4,3));
        var y=descriptor(DataType.FLOAT32,Shape.ofDimensions(b,new StaticDimension(3)));
        var mm=new Operation(MatmulKind.MATMUL,NoOperationAttrs.INSTANCE);
        assertTrue(cpu.supports(new OperationCapabilityQuery(mm,List.of(x,w),List.of(y))));
        assertFalse(cpu.supports(new OperationCapabilityQuery(mm,List.of(x,w),List.of(descriptor(DataType.FLOAT32,Shape.ofDimensions(c,new StaticDimension(3)))))));
        assertFalse(cpu.supports(new OperationCapabilityQuery(mm,List.of(descriptor(DataType.FLOAT64,x.shape()),w),List.of(y))));
        assertFalse(cpu.supports(new OperationCapabilityQuery(new Operation(UnaryElementwiseKind.RELU,NoOperationAttrs.INSTANCE),List.of(x),List.of(x))));
        assertFalse(cpu.supports(new OperationCapabilityQuery(mm,List.of(x,descriptor(DataType.FLOAT32,Shape.of(5,3))),List.of(y))));
        assertFalse(cpu.supports(new OperationCapabilityQuery(mm,List.of(descriptor(DataType.FLOAT32,Shape.ofDimensions(b,b)),w),List.of(y))));
        assertFalse(cpu.supports(new OperationCapabilityQuery(new Operation(AggregateReductionKind.SUM,NoOperationAttrs.INSTANCE),
                List.of(descriptor(DataType.FLOAT32,Shape.ofDimensions(b,new StaticDimension(2),new StaticDimension(2)))),List.of(descriptor(DataType.FLOAT32,Shape.scalar())))));
    }

    @Test void constantByteProjectionRetainsSymbolicModeAndNumericRecipes() {
        var x=descriptor(DataType.FLOAT32,Shape.ofDimensions(new DynamicDimension("B"),new StaticDimension(0)));
        var out=descriptor(DataType.FLOAT32,Shape.scalar());
        var inId=new ValueId(0);var outId=new ValueId(1);
        var node=new CompiledNode(new NodeId(0),new Operation(AggregateReductionKind.SUM,NoOperationAttrs.INSTANCE),List.of(inId),List.of(outId));
        var partition=new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,List.of(node.id()));
        var context=new PrepareContext<>(partition,List.of(node),List.of(new GraphValue(inId,x),new GraphValue(outId,out)),
                List.of(new LogicalMemoryRequirement(inId,x,Optional.empty(),List.of(partition),false),
                        new LogicalMemoryRequirement(outId,out,Optional.of(partition),List.of(),true)),Map.of(),CpuPartitionAnalysisInputs.DEFAULT);
        assertTrue(context.requiresBatchSize());
        var plan=new CpuPartitionPreparer().analyze(context).plan();
        assertTrue(plan.units().getFirst().portablePlan().specialization().ordinaryBatch());
        for(var declaration:plan.bufferDeclarations()) {
            assertTrue(declaration.size().fixedByteSize().isEmpty());
            assertEquals(declaration.size().byteSize(0),declaration.size().byteSize(64));
        }
        var aggregate=plan.units().getFirst().portablePlan().batchAggregate().orElseThrow();
        assertEquals(48,aggregate.byteSize(0));assertEquals(48,aggregate.byteSize(64));
    }
    private static TensorDescriptor descriptor(DataType type,Shape shape) { return new TensorDescriptor(type,shape,Optional.empty(),false); }
}
