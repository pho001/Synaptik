package Graph;

import Operations.FusedOperation;
import Operations.Operation;
import Operations.iFusedOperation;
import Tensor.Tensor;
import Utils.CustomClassLoader;

import java.io.FileOutputStream;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.util.*;
import java.util.stream.Collectors;

public class FuseElementWise implements OptimizationRule {

    @Override
    public List<Tensor> apply(Tensor root) {
        List<Tensor> sorted = root.topologicalSort();
        List<List<Tensor>> clusters = findClusters(sorted);
        Set<Tensor> removed = new HashSet<>();
        Map<Tensor, Tensor> replacements = new HashMap<>();

        for (List<Tensor> cluster : clusters) {
            if (cluster.size() <= 1) continue;

            Tensor last = cluster.get(cluster.size() - 1);
            List<Tensor> inputs = findInputTensors(cluster);

            // Fúzujeme operaci pro poslední tensor v clusteru

            //last.setOperation(new FusedOperation(cluster,root));
            byte[] byteCode = FusedOperationGenerator.generate(cluster,root);
            try (FileOutputStream fos = new FileOutputStream("fusedOperationClass.class")) {
                fos.write(byteCode);
            }
            catch (IOException e) {
                e.printStackTrace();
            }
            String className = "Operations.fusedOperationClass";
            CustomClassLoader loader = new CustomClassLoader();
            Class<?> customClass = loader.define(className, byteCode);
            //forward = customClass.getMethod("apply", double[][].class);
            Constructor<?> constructor;
            Operation instance;
            try {
                constructor = customClass.getConstructor(List.class);
                instance = (Operation) constructor.newInstance(cluster);
            }
            catch (Throwable e) {
                throw new RuntimeException(e);
            }

            last.setOperation(instance);
            last.setPrevTensors(inputs);

            // Nahrazujeme všechny tensory v clusteru (kromě posledního) novým tensoru
            for (Tensor t : cluster) {
                if (t != last) {
                    removed.add(t);
                    replacements.put(t, last);
                }
            }
        }

        // Použijeme updateReferences pro zajištění, že všechny odkazy budou aktualizovány
        List<Tensor> updatedGraph = updateReferences(sorted, replacements);

        // Vracíme graf, kde byly odstraněny všechny tensory z clusterů (kromě posledních)
        return updatedGraph.stream()
                .filter(t -> !removed.contains(t))
                .collect(Collectors.toList());

    }




    private List<Tensor> updateReferences(List<Tensor> graph, Map<Tensor, Tensor> replacements) {
        return graph.stream()
                .map(t -> {
                    if (t.getPrevTensors()==null) {
                        return t;
                    }

                    t.setPrevTensors(t.getPrevTensors().stream()
                            .map(prev -> replacements.getOrDefault(prev, prev))
                            .distinct()
                            .collect(Collectors.toList()));

                    return t;
                })
                .collect(Collectors.toList());
    }



    public List<List<Tensor>> findClusters(List<Tensor> sortedTensors) {
        List<List<Tensor>> clusters = new ArrayList<>();
        List<Tensor> currentCluster = null; // Odložená alokace

        for (Tensor tensor : sortedTensors) {
            if (tensor.getOperation() != null && tensor.getOperation().isElementWise()) {
                if (currentCluster == null) {
                    currentCluster = new ArrayList<>();
                }
                currentCluster.add(tensor);
            } else {
                if (currentCluster != null) { // Přidáme aktuální shluk, pokud existuje
                    clusters.add(currentCluster);
                    currentCluster = null; // Resetujeme
                }
                clusters.add(Collections.singletonList(tensor)); // Ne-elementwise tensor jako samostatný shluk
            }
        }

        // Přidáme poslední shluk, pokud existuje
        if (currentCluster != null) {
            clusters.add(currentCluster);
        }

        return clusters;
    }

    private List<Tensor> findInputTensors(List<Tensor> cluster) {
        Set<Tensor> allPrevTensors = new HashSet<>();
        for (Tensor tensor : cluster) {
            if (tensor.getPrevTensors() != null) {
                allPrevTensors.addAll(tensor.getPrevTensors());
            }
        }

        List<Tensor> inputTensors = new ArrayList<>();
        for (Tensor tensor : allPrevTensors) {
            if (!cluster.contains(tensor)) {
                inputTensors.add(tensor);
            }
        }
        return inputTensors;
    }

}