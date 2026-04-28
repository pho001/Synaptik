package backend.cpu.fused.optimize;

import tensor.Tensor;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Internal helper for collecting external tensors consumed by a fused cluster.
 */
public class FusedExternalInputCollector{
    /**
     * Returns parent tensors used by the cluster but not included in the cluster itself.
     */
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
