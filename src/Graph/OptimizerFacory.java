package Graph;

public class OptimizerFacory {
    public static OptimizationRule addFuseElementWise(){
        return new FuseElementWise();
    }
}
