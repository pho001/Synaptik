# Todo

Tento adresar drzi aktualni architektonicke ukoly a navrhy dalsich vetsich refaktoru.

## Ukoly

- [01-fused-operation-backend-refactor.md](./01-fused-operation-backend-refactor.md)
  - Dokonceny refaktor fused execution. `FusedOperation` je descriptor, codegen bezi pres `FusedExpressionPlan` a runtime executable vznikaji az v prepared metadata.

- [02-per-graph-autotune-architecture.md](./02-per-graph-autotune-architecture.md)
  - Pracovni navrh nove autotune vrstvy nad `Tensor` a `CompiledGraph`, zalozeny na implicitnich default profilech podle architektury/režimu a explicitnim per-graph autotuningu. Dokument zamerne pocita s tim, ze soucasny benchmark/autotune framework bude nahrazen cistym navrhem od nuly.

- [03-tuning-package-rewrite.md](./03-tuning-package-rewrite.md)
  - Navrh prepisu dnesni benchmark/autotune vrstvy do nove nadrazene struktury `tuning/`, kde budou oddelene performance benchmarky, numerics validace a autotune workflow.

- [04-tensor-compiledgraph-api-cleanup.md](./04-tensor-compiledgraph-api-cleanup.md)
  - Finalni navrh cisteho rozdeleni odpovednosti mezi `Tensor`, `CompiledGraph`, `PreparedExecution` a `ExecutionProfile`, vcetne odstraneni compile cache a optimizer overloadu z `Tensor` a navaznosti na autotune.
