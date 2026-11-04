package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool2dAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.*;
import java.lang.foreign.*;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Actual average-pool AD with fresh nonuniform cotangents and exact stored DIV sites. */
class EngineCpuAveragePoolBackwardIntegrationTest {
    @Test void actualAveragePoolBackwardReusesPreparationWithFreshInputAndNonuniformSeeds() {
        for(var type:List.of(DataType.FLOAT32,DataType.FLOAT64)) {
            try(var arena=Arena.ofShared();var engine=Engine.builder().takeOwnership(CpuBackendIntegration.open()).build()) {
                Shape shape=Shape.of(2,3,5,7),seedShape=Shape.of(2,3,4,4);
                var input=tensor(arena,type,shape,true,1);var seed=tensor(arena,type,seedShape,false,1);
                var output=input.averagePool2d(new AveragePool2dAttrs(2,3,2,2,1,1,1,1,true));
                var compiled=engine.compile(List.of(output),List.of(seed),List.of(input));
                try(var prepared=engine.prepare(compiled)) {
                    for(int run=1;run<=2;run++) {
                        var freshInput=tensor(arena,type,shape,true,run+1);
                        var freshSeed=tensor(arena,type,seedShape,false,run+2);
                        var values=compiled.inputs().stream().map(i->i.tensorId().equals(input.id())?freshInput:
                                i.tensorId().equals(seed.id())?freshSeed:null).toList();
                        assertFalse(values.contains(null));
                        try(var result=engine.runValues(prepared,values)) {
                            assertEquals(2,result.publications().size());
                            var bytes=result.materialize(result.publications().get(1),210L*type.byteWidth()).bytes();
                            double[] expected=materializedColumns(type,freshSeed);
                            for(double value:expected) {
                                double actual=type==DataType.FLOAT32?bytes.getFloat():bytes.getDouble();
                                if(type==DataType.FLOAT32)assertEquals(Float.floatToRawIntBits((float)value),Float.floatToRawIntBits((float)actual));
                                else assertEquals(Double.doubleToRawLongBits(value),Double.doubleToRawLongBits(actual));
                            }
                        }
                    }
                }
            }
        }
    }

    @Test void publicColumnsPreventAbsorptionAndRemainPublishedAcrossFreshRuns() {
        for(var type:List.of(DataType.FLOAT32,DataType.FLOAT64)) {
            try(var arena=Arena.ofShared();var engine=Engine.builder().takeOwnership(CpuBackendIntegration.open()).build()) {
                var source=tensor(arena,type,Shape.of(2,3,4),false,1);
                var columns=source.expandDims(2).expand(Shape.of(2,3,4,4)).contiguous().reshape(Shape.of(2,12,4)).contiguous();
                var output=columns.fold2d(Shape.of(2,3,3,3),new io.github.pho001.synaptik.model.operation.layout.Window2dAttrs(2,2,1,1,0,0,1,1,false));
                var compiled=engine.compile(List.of(columns,output));
                try(var prepared=engine.prepare(compiled)) {
                    for(int run=1;run<=2;run++) {
                        var fresh=tensor(arena,type,Shape.of(2,3,4),false,run+3);
                        try(var result=engine.runValues(prepared,List.of(fresh))) {
                            var bytes=result.materialize(result.publications().getFirst(),96L*type.byteWidth()).bytes();
                            var storage=((MemorySegmentStorage)fresh.hostStorage().orElseThrow()).segment();
                            for(int nc=0;nc<6;nc++)for(int k=0;k<4;k++)for(int p=0;p<4;p++)
                                assertEquals(read(type,storage,nc*4+p),type==DataType.FLOAT32?bytes.getFloat():bytes.getDouble());
                        }
                    }
                }
            }
        }
    }

    /**
     * Evaluates the independent split recipe: stores each typed DIV, physically repeats columns,
     * then scatters contributions in canonical q,p order. Temporary and result arrays belong to
     * this call; the borrowed seed Tensor and its host storage are only read and never retained.
     * @param type non-null F32 or F64 type matching the seed's represented storage
     * @param seed non-null dense [2,3,4,4] Tensor backed by live, current-thread-accessible
     *     MemorySegmentStorage with at least 96 represented cells
     * @return new non-null 210-cell [2,3,5,7] gradient, with F32 results widened exactly to double
     * @throws java.util.NoSuchElementException if host storage is absent
     * @throws IllegalStateException if host storage is closed or inaccessible
     */
    private static double[] materializedColumns(DataType type,Tensor seed) {
        var storage=((MemorySegmentStorage)seed.hostStorage().orElseThrow()).segment();
        double divisor=0;
        for(int i=0;i<6;i++)divisor=type==DataType.FLOAT32?(float)(divisor+1):divisor+1;
        double[] perPosition=new double[96],columns=new double[576],result=new double[210];
        for(int i=0;i<96;i++)perPosition[i]=type==DataType.FLOAT32?(float)((float)read(type,storage,i)/(float)divisor):read(type,storage,i)/divisor;
        for(int nc=0;nc<6;nc++)for(int q=0;q<6;q++)for(int p=0;p<16;p++)columns[(nc*6+q)*16+p]=perPosition[nc*16+p];
        for(int nc=0;nc<6;nc++)for(int q=0;q<6;q++)for(int p=0;p<16;p++) {
            int y=p/4*2-1+q/3,x=p%4*2-1+q%3;
            if(y<0||y>=5||x<0||x>=7)continue;
            int index=nc*35+y*7+x;double sum=result[index]+columns[(nc*6+q)*16+p];
            result[index]=type==DataType.FLOAT32?(float)sum:sum;
        }
        return result;
    }

    private static Tensor tensor(Arena arena,DataType type,Shape shape,boolean gradient,int scale) {
        long count=shape.knownElementCount().orElseThrow();var storage=arena.allocate(count*type.byteWidth(),type.byteWidth());
        for(int i=0;i<count;i++) {
            double value=scale*(i%19-9)/10d;
            if(type==DataType.FLOAT32)storage.setAtIndex(ValueLayout.JAVA_FLOAT,i,(float)value);
            else storage.setAtIndex(ValueLayout.JAVA_DOUBLE,i,value);
        }
        return TensorFactory.create(new TensorDescriptor(type,shape,Optional.of(LayoutDescriptor.contiguous(shape)),gradient),Optional.empty(),Optional.of(new MemorySegmentStorage(type,count,storage)));
    }
    private static double read(DataType t,MemorySegment s,int i) {return t==DataType.FLOAT32?s.getAtIndex(ValueLayout.JAVA_FLOAT,i):s.getAtIndex(ValueLayout.JAVA_DOUBLE,i);}
}
