# Optimizer (Graph.optimizer)

Cíl
- Orchestrace optimalizace výpočetního grafu na úrovni Tensor uzlu a jeho předchůdců.
- Aplikace sekvence optimalizačních pravidel, která mohou graf transformovat (např. fúze element‑wise operací) před samotným výpočtem.

Hlavní moduly a soubory
- Jádro optimizeru
  - [src/Graph/optimizer/GraphOptimizer.java](src/Graph/optimizer/GraphOptimizer.java)
  - [src/Graph/optimizer/OptimizationRule.java](src/Graph/optimizer/OptimizationRule.java)
  - [src/Graph/optimizer/OptimizerFactory.java](src/Graph/optimizer/OptimizerFactory.java)
- Pravidla optimizeru
  - [src/Graph/optimizer/rules/FuseElementWise.java](src/Graph/optimizer/rules/FuseElementWise.java)
- Generování bytecode (sdíleno mezi pravidly a operacemi)
  - [src/Graph/codegen/FusedOperationGenerator.java](src/Graph/codegen/FusedOperationGenerator.java)

Datový tok a životní cyklus
- Optimalizace se spouští při kompilaci/počítání grafu:
  - [src/Graph/CompiledGraph.java](src/Graph/CompiledGraph.java) během `forward()` poprvé volá optimalizátor, aby vytvořil optimalizovaný pořadník uzlů (topologicky seřazený seznam po aplikaci pravidel).
  - [src/Tensor/Tensor.java](src/Tensor/Tensor.java) poskytuje metody `compute()` a `compute(GraphOptimizer optimizer)`, které propojí Tensor s kompilovaným grafem používajícím zadaný optimizer.
- Transformace grafu:
  - Každé pravidlo implementuje rozhraní `OptimizationRule` a přijímá kořenový Tensor. Vrací optimalizovaný seznam Tensorů (topologicky seřazený), který může nahradit podgrafy (např. fúzí více EW uzlů do jednoho).
- Výpočet:
  - Po vytvoření optimalizovaného seznamu Tensorů se dopředná fáze vypočítá přes `ComputeEngine` bez dalších změn API výpočtu.

Jádro API
- OptimizationRule (kontrakt)
  - Soubor: [src/Graph/optimizer/OptimizationRule.java](src/Graph/optimizer/OptimizationRule.java)
  - Účel: definovat jedno pravidlo transformace grafu.
  - Vstup: kořenový Tensor.
  - Výstup: topologicky seřazený seznam Tensorů po aplikaci transformace (může obsahovat náhrady uzlů).
- GraphOptimizer (orchestrátor)
  - Soubor: [src/Graph/optimizer/GraphOptimizer.java](src/Graph/optimizer/GraphOptimizer.java)
  - Uchovává seznam pravidel a aplikuje je (v daném pořadí) nad grafem.
  - Pokud nejsou registrována pravidla, vrací defaultní topologické seřazení (`Tensor.topologicalSort()`).
- OptimizerFactory (továrna)
  - Soubor: [src/Graph/optimizer/OptimizerFactory.java](src/Graph/optimizer/OptimizerFactory.java)
  - Pomocné metody pro instanciaci/sestavení předdefinovaných pravidel (např. přidání fúze EW).

Pravidla: FuseElementWise
- Soubor: [src/Graph/optimizer/rules/FuseElementWise.java](src/Graph/optimizer/rules/FuseElementWise.java)
- Účel:
  - Najde shluky (clustery) element‑wise operací v topologicky seřazeném grafu.
  - Vygeneruje fúzovanou implementaci (bytecode) pro výslednou operaci shluku.
  - Nahradí poslední uzel shluku fúzovanou operací a přepojí jeho vstupy na externí vstupy shluku.
  - Průběžně aktualizuje reference v grafu a odstraní mezilehlé uzly shluku (mimo posledního).
- Bytecode generace
  - Probíhá přes [src/Graph/codegen/FusedOperationGenerator.java](src/Graph/codegen/FusedOperationGenerator.java) a dynamické načtení třídy přes vlastní classloader.
- Poznámka:
  - Fúze je transparentní pro zbytek výpočtu – konečný uzel shluku nadále vystupuje jako jeden Tensor s vlastní operací.

Generátor fúzovaných operací (ASM)
- Soubor: [src/Graph/codegen/FusedOperationGenerator.java](src/Graph/codegen/FusedOperationGenerator.java)
- Role:
  - Vytváří na základě shluku EW uzlů konkrétní třídu implementující fúzovanou operaci (forward i gradient).
  - Zohledňuje architekturu (x86/ARM), vektorové instrukce (jdk.incubator.vector) a správu mezivýpočtů.
- Sdílené použití:
  - Pravidlo FuseElementWise generuje bytecode pro fúzovaný uzel.
  - Lze použít i mimo pravidla (např. přímo z operací) – viz [src/Operations/FusedOperation.java](src/Operations/FusedOperation.java).

Integrace v aplikaci
- Vstupní bod demonstrace:
  - [src/Main.java](src/Main.java)
  - Ukazuje, jak vytvořit optimizer, přidat pravidlo a spustit výpočet s optimalizací.
- Kompilace a běh grafu:
  - [src/Graph/CompiledGraph.java](src/Graph/CompiledGraph.java) – první dopředný běh provede optimalizaci a uloží optimalizovaný seznam uzlů.
- Výpočet a gradienty:
  - Po optimalizaci běží dopředná i zpětná fáze přes `ComputeEngine` beze změn API.

Jak přidat nové pravidlo
1) Vytvořte novou třídu v balíčku `Graph.optimizer.rules` a implementujte rozhraní z [src/Graph/optimizer/OptimizationRule.java](src/Graph/optimizer/OptimizationRule.java).
2) V metodě `apply(Tensor root)`:
   - Získejte topologické pořadí (`root.topologicalSort()`).
   - Najděte vhodné kandidáty (shluky, patterny).
   - Proveďte transformaci – nahraďte/úpravte uzly a jejich vstupy, zaktualizujte reference v grafe.
   - Vraťte nový topologicky seřazený seznam.
3) Přidejte tovární metodu do [src/Graph/optimizer/OptimizerFactory.java](src/Graph/optimizer/OptimizerFactory.java) pro snadnou registraci pravidla (volitelné).
4) Otestujte v separátním testu (analogicky k testu fúze).

Konvence a garance
- Transformace musí zachovat korektnost výstupů (dopředný běh) i možnost výpočtu gradientů (zpětný běh).
- Pravidla by měla být idempotentní v rámci jedné optimalizační fáze (opakované spuštění by nemělo dál měnit již transformovaný graf).
- Po transformaci vracejte topologicky seřazený seznam Tensorů, který odráží aktuální závislosti po změnách.

Poznámky k build/runtime
- Projekt používá ASM 9.6 a vektorové API (inkubační modul) – viz konfigurace v [build.gradle](build.gradle).
- Při spouštění Gradle doporučeno používat JDK 17/21 pro samotný build (Groovy/Gradle kompatibilita). Toolchain pro kompilaci zdrojů je nastaven v [build.gradle](build.gradle).

Související soubory
- [src/Graph/optimizer/GraphOptimizer.java](src/Graph/optimizer/GraphOptimizer.java)
- [src/Graph/optimizer/OptimizationRule.java](src/Graph/optimizer/OptimizationRule.java)
- [src/Graph/optimizer/OptimizerFactory.java](src/Graph/optimizer/OptimizerFactory.java)
- [src/Graph/optimizer/rules/FuseElementWise.java](src/Graph/optimizer/rules/FuseElementWise.java)
- [src/Graph/codegen/FusedOperationGenerator.java](src/Graph/codegen/FusedOperationGenerator.java)
- [src/Graph/CompiledGraph.java](src/Graph/CompiledGraph.java)
- [src/Tensor/Tensor.java](src/Tensor/Tensor.java)
- [src/Operations/FusedOperation.java](src/Operations/FusedOperation.java)
- [src/Main.java](src/Main.java)