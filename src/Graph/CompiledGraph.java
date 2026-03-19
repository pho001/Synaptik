package Graph;

import Backend.ComputeEngine;
import Backend.ComputeBackend;
import Backend.kernels.cpu.CpuKernel;
import Backend.registry.CpuKernelRegistry;
import Operations.Operation;
import Tensor.Tensor;
import Graph.optimizer.GraphOptimizer;

import java.util.*;

public class CompiledGraph {
    private Tensor rootTensor; // Kořenový tensor grafu
    List<Tensor> finalGraph= new ArrayList<>();
    List<Tensor> forwardGraph = new ArrayList<>();
    List<Tensor> backwardGraph = new ArrayList<>();
    int backwardStartIndex = -1;
    private Tensor forwardOutput;
    int forwardEndIndex = -1;
    boolean inferenceMode = false;




    GraphOptimizer optimizer;


    public CompiledGraph(Tensor rootTensor, GraphOptimizer forwardOptimizer) {
        this.rootTensor = rootTensor;
        this.optimizer=forwardOptimizer;
        compile();
    }

    public void compile() {
        // 1. Forward graf ukotvený přes "yield" uzel
        this.forwardOutput = rootTensor.forwardOutput();
        this.forwardGraph.addAll(this.forwardOutput.topologicalSort());

        // Inference-only pipeline: pokud žádný leaf vstup nepožaduje gradient,
        // kompilujeme jen forward graf bez backward části.
        if (!hasTrainableLeafInputs()) {
            this.finalGraph = optimizer.optimize(new ArrayList<>(this.forwardGraph));
            preResolveCpuKernels();
            this.forwardEndIndex = this.finalGraph.indexOf(this.forwardOutput);
            if (this.forwardEndIndex == -1) {
                throw new IllegalStateException("Forward output node not found in inference finalGraph.");
            }
            this.backwardStartIndex = -1;
            return;
        }

        // 2. Seed gradientu kořene a sestavení backward uzlů pomocí lambda pravidel
        rootTensor.setGradient(Tensor.onesLike(rootTensor));
        for (int i = this.forwardGraph.size() - 1; i >= 0; i--) {
            this.forwardGraph.get(i).buildBackwardGraph();
        }

        // 3. Backward targets: gradienty leaf vstupů, které požadují gradient
        List<Tensor> backwardTargets = collectBackwardTargets();

        // Fallback: pokud není nic explicitně cíleno, sebereme všechny dostupné gradient uzly
        if (backwardTargets.isEmpty()) {
            for (Tensor t : forwardGraph) {
                if (t.getGradient() != null) {
                    backwardTargets.add(t.getGradient());
                }
            }
        }

        // 4. Označíme backward uzly (nezávisle na pořadí), aby šla po optimalizaci najít hranice fází.
        // Důležité pro fusion/CSE na spojeném grafu: bez tohoto značení se může smíchat fw+bw část.
        collectBackwardNodes();

        // 5. Super-root ukotví více výstupů pro optimalizaci (forward + všechny backward sinks)
        List<Tensor> targetsToSave = new ArrayList<>();
        //this.backwardGraph = backwardTargets;
        targetsToSave.add(this.forwardOutput);
        targetsToSave.addAll(backwardTargets);
        Tensor superRoot = new Tensor(new int[]{1}, targetsToSave, new Operations.noop(), "System_Super_Root");

        // 6. Sjednocený graf přes super-root topologii; super-root samotný ve finalGraph nechceme
        this.finalGraph = superRoot.topologicalSort();
        this.finalGraph.remove(superRoot);

        // 7. Optimalizace nad celým sjednoceným grafem
        this.finalGraph = optimizer.optimize(this.finalGraph);
        preResolveCpuKernels();
        this.forwardEndIndex = this.finalGraph.indexOf(this.forwardOutput);
        if (this.forwardEndIndex == -1) {
            throw new IllegalStateException("Forward output node not found in finalGraph.");
        }

    }

    private boolean hasTrainableLeafInputs() {
        for (Tensor t : forwardGraph) {
            if (t.getOperation() == null && t.getRequiresGrad()) {
                return true;
            }
        }
        return false;
    }

    public void setTrainingModeOn(){
        this.inferenceMode=false;
    }

    public void setTrainingModeOff(){
        this.inferenceMode=true;
    }

    // Nyní máme jen jednu exekuční metodu! Žádný forward a backward zvlášť.
    public void execute() {
        // 1) Vždy nejdřív spočítáme forward část včetně forwardOutput kotvy.
        for (int i = 0; i <= forwardEndIndex; i++) {
            Tensor tensor = finalGraph.get(i);
            if (tensor.getOperation() == null) {
                continue;
            }
            if (tensor.getPrevTensors() == null) {
                continue;
            }
            ComputeEngine.compute(tensor, tensor.getResolvedBackend());
        }

        // 2) Okamžitě synchronizujeme root výstup, aby ho případný backward/memory reuse
        // už nemohl přepsat před přečtením uživatelem.
        syncRootData();

        // 3) V training režimu dopočítáme zbytek (backward část sjednoceného grafu).
        if (!this.inferenceMode) {
            for (int i = forwardEndIndex + 1; i < finalGraph.size(); i++) {
                Tensor tensor = finalGraph.get(i);
                if (tensor.getOperation() == null) {
                    continue;
                }
                if (tensor.getPrevTensors() == null) {
                    continue;
                }
                ComputeEngine.compute(tensor, tensor.getResolvedBackend());
            }
        }
    }

    private void syncRootData() {
        // yieldNode je naše nezničitelná kotva.
        // Její 0-tý vstup je VŽDYCKY ten skutečný, zoptimalizovaný výsledek dopředného chodu!
        Tensor actualRoot = this.forwardOutput.getPrevTensors().get(0);

        // Pokud optimalizátor nahradil původní root něčím novým (fúzí), ukradneme data
        if (actualRoot != this.rootTensor && actualRoot.getData() != null) {
            System.arraycopy(actualRoot.getData(), 0, this.rootTensor.getData(), 0, actualRoot.getData().length);
        }
    }



    // Backward pass
    public void backward() {
        if (backwardStartIndex == -1) {
            System.out.println("Info: No gradients to compute.");
            return;
        }
        zeroGrad();
        if (rootTensor.getGradient() != null) {
            Arrays.fill(rootTensor.getGradient().getData(), 1.0);
        }

        for (int i = backwardStartIndex; i < finalGraph.size(); i++) {
            Tensor t = finalGraph.get(i);
            if (t.getOperation() != null && t.isBackward()) {
                ComputeEngine.compute(t, t.getResolvedBackend());
            }
        }
    }

    public void zeroGrad() {
        for (Tensor t : finalGraph) {
            if (t.getGradient() != null) {
                Arrays.fill(t.getGradient().getData(), 0.0);
            }
        }
    }


    private List<Tensor> collectBackwardNodes() {
        List<Tensor> backwardNodes = new ArrayList<>();
        Set<Tensor> visited = new HashSet<>();
        Set<Tensor> forwardSet = new HashSet<>(forwardGraph);

        // Procházíme dopředný graf od konce (od loss) k začátku.
        // Tím zajistíme, že začneme sbírat gradienty tam, kde vznikají jako první.
        for (int i = forwardGraph.size() - 1; i >= 0; i--) {
            Tensor fwdTensor = forwardGraph.get(i);
            Tensor gradTensor = fwdTensor.getGradient();

            // Pokud má dopředný tenzor přiřazený gradient (díky lambdě),
            // prozkoumáme celou historii jeho vzniku.
            if (gradTensor != null) {
                collectDFS(gradTensor, visited, backwardNodes, forwardSet);
            }
        }

        return backwardNodes;
    }

    private void collectDFS(Tensor tensor, Set<Tensor> visited, List<Tensor> sortedList, Set<Tensor> forwardSet) {
        // Základní podmínky: uzel jsme už viděli nebo neexistuje
        if (tensor == null || visited.contains(tensor)) {
            return;
        }

        visited.add(tensor);

        // Rekurze: Nejdříve prozkoumáme všechny předky (vstupy operace gradientu).
        // Např. u grad = gradA + gradB musíme mít nejdřív gradA a gradB.
        if (tensor.getPrevTensors() != null) {
            for (Tensor parent : tensor.getPrevTensors()) {
                collectDFS(parent, visited, sortedList, forwardSet);
            }
        }

        // "Post-order" přidání: Teprve až jsou zpracováni všichni rodiče,
        // přidáme samotný tenzor do seznamu pro výpočet.
        // Přidáváme pouze ty, které mají operaci (konstanty/listy se nepočítají).
        if (tensor.getOperation() != null && !forwardSet.contains(tensor)) {
            tensor.setBackward(true);
            sortedList.add(tensor);
        }
    }

    private List<Tensor> collectBackwardTargets() {
        List<Tensor> targets = new ArrayList<>();
        Set<Tensor> unique = new LinkedHashSet<>();

        for (Tensor t : forwardGraph) {
            // Leaf vstupy bez operace, které chtějí gradient
            if (t.getOperation() == null && t.getRequiresGrad() && t.getGradient() != null) {
                unique.add(t.getGradient());
            }
        }

        targets.addAll(unique);
        return targets;
    }




    public Tensor getRootTensor() {
        return rootTensor;
    }

    public List<Tensor> getCompiledGraphAsList() {
        return this.finalGraph;
    }

    private void preResolveCpuKernels() {
        for (Tensor tensor : finalGraph) {
            Operation operation = tensor.getOperation();
            ComputeBackend backend = tensor.resolveBackend();
            tensor.setResolvedBackend(backend);
            if (operation == null || backend != ComputeBackend.CPU) {
                tensor.setResolvedCpuKernel(null);
                continue;
            }
            CpuKernel kernel = CpuKernelRegistry.resolve(operation.opType());
            tensor.setResolvedCpuKernel(kernel);
        }
    }


}
