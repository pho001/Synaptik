import Graph.optimizer.GraphOptimizer;
import Graph.optimizer.OptimizerFactory;
import Tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;



public class Main {
    public static void main(String[] args) {
        OptimizerBenchmark.run();
        Tensor T=new Tensor(new double[] {2.0,20.0},new ArrayList<>(),"nic");

    }
}
