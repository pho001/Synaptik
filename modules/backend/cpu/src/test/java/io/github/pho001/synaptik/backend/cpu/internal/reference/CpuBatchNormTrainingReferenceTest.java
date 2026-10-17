package io.github.pho001.synaptik.backend.cpu.internal.reference;
import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import org.junit.jupiter.api.Test;
class CpuBatchNormTrainingReferenceTest {
 @Test void locksModelExampleAndTransitionRoles(){double[][] in={{1,2,3},{2},{.5},{10},{4}};long[][] e={{1,1,3},{1},{1},{1},{1}};long[][] s={{3,3,1},{1},{1},{1},{1}};var r=CpuBatchNormTrainingReferenceKernel.evaluate(java.util.Collections.nCopies(5,DataType.FLOAT64),DataType.FLOAT64,.25,1e-5,in,e,new long[5],s,1);assertAll(()->assertEquals(2,r.savedBatchMean()[0],0),()->assertEquals(8,r.nextRunningMean()[0],0),()->assertEquals(3.25,r.nextRunningVariance()[0],1e-15),()->assertEquals(.5,r.output()[1],1e-15));}

 @Test void nonfiniteInputCanonicalizesEveryBatchOwnedFormula(){
  double[][] in={{1,Double.POSITIVE_INFINITY,3},{2},{.5},{10},{4}};long[][] e={{1,1,3},{1},{1},{1},{1}};long[][] s={{3,3,1},{1},{1},{1},{1}};
  var r=CpuBatchNormTrainingReferenceKernel.evaluate(java.util.Collections.nCopies(5,DataType.FLOAT64),DataType.FLOAT64,.25,1e-5,in,e,new long[5],s,1);
  assertAll(()->assertTrue(java.util.Arrays.stream(r.output()).allMatch(Double::isNaN)),
    ()->assertTrue(Double.isNaN(r.nextRunningMean()[0])),()->assertTrue(Double.isNaN(r.nextRunningVariance()[0])),
    ()->assertTrue(Double.isNaN(r.savedBatchMean()[0])),()->assertTrue(Double.isNaN(r.savedInverseStandardDeviation()[0])));
 }

 @Test void constantSignedZeroDomainProducesPositiveVarianceAndPreservesFormulaRoles(){
  double[][] in={{-0.0,-0.0},{1},{0},{Double.NaN},{-3}};long[][] e={{1,1,2},{1},{1},{1},{1}};long[][] s={{2,2,1},{1},{1},{1},{1}};
  var r=CpuBatchNormTrainingReferenceKernel.evaluate(java.util.Collections.nCopies(5,DataType.FLOAT64),DataType.FLOAT64,0.0,1e-5,in,e,new long[5],s,1);
  assertAll(()->assertEquals(0L,Double.doubleToRawLongBits(r.savedBatchMean()[0])),
    ()->assertEquals(0L,Double.doubleToRawLongBits(r.nextRunningVariance()[0]+3.0)),
    ()->assertTrue(Double.isNaN(r.nextRunningMean()[0])),
    ()->assertTrue(Double.doubleToRawLongBits(r.savedInverseStandardDeviation()[0])>0));
 }

 @Test void rejectsPositiveChannelReductionBelowTwo(){
  double[][] in={{1},{1},{0},{0},{1}};long[][] e={{1,1},{1},{1},{1},{1}};long[][] s={{1,1},{1},{1},{1},{1}};
  assertThrows(IllegalArgumentException.class,()->CpuBatchNormTrainingReferenceKernel.evaluate(
    java.util.Collections.nCopies(5,DataType.FLOAT32),DataType.FLOAT32,.25,1e-5,in,e,new long[5],s,0));
 }
}
