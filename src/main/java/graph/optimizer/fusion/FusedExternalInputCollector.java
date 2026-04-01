package graph.optimizer.fusion;

import tensor.Tensor;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

public class FusedExternalInputCollector{
    public static List<Tensor> collect(List<Tensor> cluster) {
        Set<Tensor> clusterSet = new LinkedHashSet<>(cluster);
        Set<Tensor> external = new LinkedHashSet<>();

        for (Tensor t : cluster) {
            List<Tensor> parents = t.getPrevTensors();
            if (parents == null) {
                continue;
            }
            for (Tensor p : parents) {
                if (!clusterSet.contains(p)) {
                    external.add(p);
                }
            }
        }

        return new ArrayList<>(external);
    }
}

