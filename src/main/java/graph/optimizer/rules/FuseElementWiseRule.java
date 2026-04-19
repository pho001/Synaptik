package graph.optimizer.rules;

import graph.optimizer.OptimizationRule;
import graph.optimizer.OptimizerGraphSupport;
import config.optimizer.FuseConfig;
import graph.codegen.FusedExpressionPlan;
import graph.codegen.FusedPlanBuilder;
import operations.fused.FusedOperation;
import operations.fused.FusedOperationFactory;
import operations.Operation;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import java.util.*;

public class FuseElementWiseRule implements OptimizationRule {
    private final FuseConfig config;

    public FuseElementWiseRule() {
        this(FuseConfig.trainingDefaults());
    }

    public FuseElementWiseRule(boolean preserveSharedExpensiveNodes) {
        this(FuseConfig.trainingDefaults().withPreserveSharedExpensiveNodes(preserveSharedExpensiveNodes));
    }

    public FuseElementWiseRule(FuseConfig config) {
        this.config = Objects.requireNonNull(config, "config cannot be null");
    }

    public FuseConfig config() {
        return config;
    }

    @Override
    public List<Tensor> apply(List<Tensor> sortedGraph) {
        // 1. KROK: Spočítáme konzumenty:
        // - allConsumer*: napříč celým spojeným grafem (fw+bw) pro liveness/materializaci
        // - samePhaseConsumer*: jen ve stejné fázi pro rozhodnutí o lokální fúzi
        Map<Tensor, Integer> allConsumerCounts = new HashMap<>();
        Map<Tensor, List<Tensor>> allConsumersMap = new HashMap<>();
        Map<Tensor, List<Tensor>> samePhaseConsumersMap = new HashMap<>();
        for (Tensor t : sortedGraph) {
            if (t.getPrevTensors() != null) {
                for (Tensor input : t.getPrevTensors()) {
                    allConsumerCounts.put(input, allConsumerCounts.getOrDefault(input, 0) + 1);
                    allConsumersMap.computeIfAbsent(input, k -> new ArrayList<>()).add(t);

                    if (input.isBackward() == t.isBackward()) {
                        samePhaseConsumersMap.computeIfAbsent(input, k -> new ArrayList<>()).add(t);
                    }
                }
            }
        }

        // 2. KROK: Najdeme uzly, které MUSÍ být v paměti zachovány (Materialization points)
        Set<Tensor> materializationPoints = new HashSet<>();
        for (Tensor t : sortedGraph) {
            boolean isElementWise = isFusedComputeCandidate(t.getOperation());
            boolean isCheap = t.getOperation() != null && t.getOperation().isCheap();
            boolean isFused = t.getOperation() != null && t.getOperation().opType() == Operation.OpType.FUSED;
            int consumersAll = allConsumerCounts.getOrDefault(t, 0);
            boolean hasNonElementWiseConsumer = false;
            for (Tensor c : samePhaseConsumersMap.getOrDefault(t, Collections.emptyList())) {
                if (!isFusedComputeCandidate(c.getOperation())) {
                    hasNonElementWiseConsumer = true;
                    break;
                }
            }
            boolean hasCrossPhaseConsumer = false;
            for (Tensor c : allConsumersMap.getOrDefault(t, Collections.emptyList())) {
                if (c.isBackward() != t.isBackward()) {
                    hasCrossPhaseConsumer = true;
                    break;
                }
            }

            // Uzel se musí materiálizovat pokud:
            // a) Je to vstup bez operace
            // b) Není to element-wise operace (MatMul)
            // c) Nemá žádného konzumenta v celém grafu (je to output/sink)
            // d) Má konzumenta, který není element-wise (hranice clusteru, např. noop kotva)
            // e) Má konzumenta v jiné fázi (hranice fw<->bw)
            // f) Má více konzumentů v celém grafu a ZÁROVEŇ není "cheap"
            // f) Je to už fused uzel (ten dál nefúzujeme; je to hranice clusteru)
            boolean sharedExpensive = consumersAll > 1 && !isCheap;
            if (isFused
                    || t.getOperation() == null
                    || !isElementWise
                    || consumersAll == 0
                    || hasNonElementWiseConsumer
                    || hasCrossPhaseConsumer
                    || (config.preserveSharedExpensiveNodes() && sharedExpensive)) {
                materializationPoints.add(t);
            }
        }

        // 3. KROK: Sestavení fúzí. Projdeme zachované uzly a postavíme pro ně clustery
        // retainedNodes určuje, co musí zůstat ve výsledném grafu po cleanupu.
        Set<Tensor> retainedNodes = new HashSet<>(materializationPoints);
        for (Tensor t : sortedGraph) {
            if (materializationPoints.contains(t)
                    && t.getOperation() != null
                    && isFusedComputeCandidate(t.getOperation())
                    && t.getOperation().opType() != Operation.OpType.FUSED) {

                // Postavíme cluster (nabalíme do něj vše, co není materializační bod)
                List<Tensor> cluster = buildCluster(t, materializationPoints);

                // Má smysl fúzovat jen tehdy, když jsme spolkli alespoň 2 operace
                if (cluster.size() > 1) {
                    // Najdeme externí vstupy pro tento nový obří uzel
                    List<Tensor> externalInputs = findOuterTensors(cluster);
                    if (!shouldFuseCluster(cluster, t, externalInputs, allConsumerCounts)) {
                        // Pokud cost model cluster odmítne, nesmíme jeho interní uzly zahodit.
                        // Kořen by jinak odkazoval na tensor, který už není v execution listu.
                        retainedNodes.addAll(cluster);
                        continue;
                    }
                    boolean containsBackward = cluster.stream().anyMatch(Tensor::isBackward);

                    // MAGIE: Zmutujeme aktuální uzel. Všechny reference zvenčí zůstanou zachovány!
                    FusedOperationFactory.Result fused = FusedOperationFactory.create(cluster, t, externalInputs);
                    TensorInternalAccess.setOperation(t, fused.operation());
                    TensorInternalAccess.setPrevTensors(t, fused.runtimeInputs());
                    if (containsBackward) {
                        TensorInternalAccess.setBackward(t, true);
                    }
                }
            }
        }

        // 4. KROK: Vyčistíme finální graf.
        // Uzly, které byly pohlceny fúzí, už v hlavním grafu nepotřebujeme.
        // Ony stále žijí uvnitř 'FusedOperation.cluster', ale MemoryPlanner už je neuvidí!
        List<Tensor> optimizedGraph = new ArrayList<>();
        for (Tensor t : sortedGraph) {
            if (retainedNodes.contains(t)) {
                optimizedGraph.add(t);
            }
        }

        return OptimizerGraphSupport.rebuildTopologicalClosureFromRoots(
                OptimizerGraphSupport.observableRoots(optimizedGraph)
        );
    }

    /**
     * Prohledává graf pozpátku (od kořene nahoru) a polyká uzly do clusteru.
     * Zastaví se ve chvíli, kdy narazí na jiný materializační bod.
     */
    private List<Tensor> buildCluster(Tensor root, Set<Tensor> materializationPoints) {
        List<Tensor> cluster = new ArrayList<>();
        Queue<Tensor> queue = new ArrayDeque<>();
        queue.add(root);

        while (!queue.isEmpty()) {
            Tensor curr = queue.poll();
            if (!cluster.contains(curr)) {
                cluster.add(curr);

                if (curr.getPrevTensors() != null) {
                    for (Tensor prev : curr.getPrevTensors()) {
                        // Pokud předek NENÍ materializační bod, spolkneme ho!
                        // Tady se přesně děje ten "Recompute" u isCheap operací.
                        if (!materializationPoints.contains(prev)
                                && prev.getOperation() != null
                                && isFusedComputeCandidate(prev.getOperation())
                                && prev.getOperation().opType() != Operation.OpType.FUSED
                                && prev.isBackward() == curr.isBackward()) {
                            queue.add(prev);
                        }
                    }
                }
            }
        }
        return cluster;
    }

    private List<Tensor> findOuterTensors(List<Tensor> cluster) {
        Set<Tensor> allPrevTensors = new LinkedHashSet<>();
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

    private boolean shouldFuseCluster(
            List<Tensor> cluster,
            Tensor root,
            List<Tensor> externalInputs,
            Map<Tensor, Integer> allConsumerCounts
    ) {
        int nodes = cluster.size();
        if (nodes < 2) return false;
        if (nodes > config.maxClusterNodes()) return false;
        FusedExpressionPlan previewPlan = FusedPlanBuilder.build(cluster, externalInputs, root);
        if (graph.optimizer.fusion.FusedCostModel.rejectBroadcastHeavySmallAffinePlan(previewPlan)) {
            return false;
        }

        Set<Tensor> clusterSet = new HashSet<>(cluster);
        int internalEdges = 0;
        int sharedExpensive = 0;
        int nonCheapNodes = 0;

        for (Tensor n : cluster) {
            Operation op = n.getOperation();
            if (op != null && !op.isCheap()) {
                nonCheapNodes++;
                if (allConsumerCounts.getOrDefault(n, 0) > 1) {
                    sharedExpensive++;
                }
            }

            List<Tensor> parents = n.getPrevTensors();
            if (parents == null) continue;
            for (Tensor p : parents) {
                if (clusterSet.contains(p)) {
                    internalEdges++;
                }
            }
        }

        double benefit = (nodes - 1)
                + config.internalEdgeBonus() * internalEdges
                + config.nonCheapBonus() * nonCheapNodes;
        double cost = config.externalInputPenalty() * externalInputs.size()
                + config.sharedExpensivePenalty() * sharedExpensive;
        cost += graph.optimizer.fusion.FusedCostModel.estimateFusionAccessPenalty(previewPlan);
        double score = benefit - cost;

        return score >= config.scoreThreshold();
    }

    private boolean isFusedComputeCandidate(Operation op) {
        if (op == null || op.opType() == null) {
            return false;
        }
        return op.opType().isFusable();
    }
}
