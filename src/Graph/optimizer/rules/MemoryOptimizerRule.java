package Graph.optimizer.rules;

import Graph.optimizer.OptimizationRule;
import Tensor.DataType;
import Tensor.Tensor;
import java.util.*;

public class MemoryOptimizerRule implements OptimizationRule {

    @Override
    public List<Tensor> apply(List<Tensor> sortedGraph) {
        if (sortedGraph.isEmpty()) return sortedGraph;
        for (Tensor t : sortedGraph) {
            if (t.getDataType() != DataType.FLOAT64) {
                return sortedGraph;
            }
        }

        // 1. Liveness Analysis: Kdy je uzel naposledy použit?
        Map<Tensor, Integer> lastUseMap = new HashMap<>();
        Map<Tensor, Integer> consumerCounts = new HashMap<>();
        for (int i = 0; i < sortedGraph.size(); i++) {
            Tensor t = sortedGraph.get(i);
            if (t.getPrevTensors() != null) {
                for (Tensor prev : t.getPrevTensors()) {
                    lastUseMap.put(prev, i);
                    consumerCounts.put(prev, consumerCounts.getOrDefault(prev, 0) + 1);
                }
            }
        }
        // Ochrana všech sink uzlů (nejen posledního v listu)
        for (Tensor t : sortedGraph) {
            if (consumerCounts.getOrDefault(t, 0) == 0) {
                lastUseMap.put(t, Integer.MAX_VALUE);
            }
        }

        Map<Integer, Stack<double[]>> freeBuffers = new HashMap<>();

        for (int i = 0; i < sortedGraph.size(); i++) {
            Tensor t = sortedGraph.get(i);
            if (t.getOperation() == null) continue;

            // --- IN-PLACE OPTIMALIZACE ---
            boolean inPlaceFound = false;
            if (t.getPrevTensors() != null) {
                for (Tensor input : t.getPrevTensors()) {
                    // Podmínky pro In-place:
                    // 1. Vstup má stejnou velikost jako potřebuje výstup (Broadcasting safe)
                    // 2. Vstup je v tomto kroku použit naposledy
                    // 3. Vstup není konstanta/váha (má operaci) - volitelné, ale bezpečnější
                    if (input.getFlatDataSize() == t.getFlatDataSize() &&
                            lastUseMap.get(input) == i &&
                            input.getOperation() != null) {

                        t.setData(input.getData());
                        inPlaceFound = true;
                        break; // Našli jsme vhodný buffer, končíme hledání pro tento uzel
                    }
                }
            }

            // --- STANDARDNÍ ALOKACE (pokud In-place nevyšlo) ---
            if (!inPlaceFound) {
                int size = t.getFlatDataSize();
                double[] buffer = getFromPool(size, freeBuffers);
                if (buffer != null) {
                    t.setData(buffer);
                } else {
                    t.setData(new double[size]);
                }
            }

            // 2. Uvolnění nepoužívaných bufferů do poolu
            if (t.getPrevTensors() != null) {
                for (Tensor input : t.getPrevTensors()) {
                    // Pokud vstup umírá a jeho data nebyla převzata In-place (nejsou stejná jako výstup)
                    if (lastUseMap.get(input) == i &&
                            input.getOperation() != null &&
                            input.getData() != t.getData()) {

                        releaseToPool(input.getData(), freeBuffers);
                    }
                }
            }
        }
        return sortedGraph;
    }

    private double[] getFromPool(int size, Map<Integer, Stack<double[]>> pool) {
        Stack<double[]> buffers = pool.get(size);
        return (buffers != null && !buffers.isEmpty()) ? buffers.pop() : null;
    }

    private void releaseToPool(double[] buffer, Map<Integer, Stack<double[]>> pool) {
        if (buffer == null) return;
        pool.computeIfAbsent(buffer.length, k -> new Stack<>()).push(buffer);
    }
}
