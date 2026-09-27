import Task0069Binary32

namespace Task0069

structure Leaf where
  ordinal : Nat
  word : Word
  deriving DecidableEq, Repr

inductive Tree where
  | leaf (value : Leaf)
  | branch (left right : Tree)
  deriving Repr

/-- Contributor labels preserve source order and make multiplicity explicit. -/
def labelFrom : Nat → List Word → List Leaf
  | _, [] => []
  | ordinal, word :: rest =>
      { ordinal := ordinal, word := absWord word } :: labelFrom (ordinal + 1) rest

def leaves : Tree → List Leaf
  | .leaf value => [value]
  | .branch left right => leaves left ++ leaves right

def addNodes : Tree → Nat
  | .leaf _ => 0
  | .branch left right => addNodes left + addNodes right + 1

/-- The permitted tree used by the source is a left-associated ordinal chain. -/
def leftChainFrom : Tree → List Leaf → Tree
  | tree, [] => tree
  | tree, next :: rest => leftChainFrom (.branch tree (.leaf next)) rest

def sourceTree (first : Word) (rest : List Word) : Tree :=
  leftChainFrom
    (.leaf { ordinal := 0, word := absWord first })
    (labelFrom 1 rest)

inductive TreeEval (add : Word → Word → Word → Prop) : Tree → Word → Prop where
  | leaf (value : Leaf) : TreeEval add (.leaf value) value.word
  | branch {left right leftResult rightResult result} :
      TreeEval add left leftResult →
      TreeEval add right rightResult →
      add leftResult rightResult result →
      TreeEval add (.branch left right) result

/-- The source accumulator relation: no step exists before contributor ordinal one. -/
inductive SourceFold (add : Word → Word → Word → Prop) : Word → List Leaf → Word → Prop where
  | nil (current : Word) : SourceFold add current [] current
  | cons {current intermediate result : Word} {next : Leaf} {rest : List Leaf} :
      add current next.word intermediate →
      SourceFold add intermediate rest result →
      SourceFold add current (next :: rest) result

/-- Set-valued Model result over any labelled permitted binary tree. -/
def ModelResult (add : Word → Word → Word → Prop)
    (contributors : List Word) (result : Word) : Prop :=
  ∃ tree,
    leaves tree = labelFrom 0 contributors ∧
    TreeEval add tree result

@[simp] theorem labelFrom_length (start : Nat) (words : List Word) :
    (labelFrom start words).length = words.length := by
  induction words generalizing start with
  | nil => rfl
  | cons word rest induction =>
      simp [labelFrom, induction]

@[simp] theorem labelFrom_words (start : Nat) (words : List Word) :
    (labelFrom start words).map Leaf.word = words.map absWord := by
  induction words generalizing start with
  | nil => rfl
  | cons word rest induction =>
      simp [labelFrom, induction]

theorem leftChainFrom_leaves (tree : Tree) (rest : List Leaf) :
    leaves (leftChainFrom tree rest) = leaves tree ++ rest := by
  induction rest generalizing tree with
  | nil => simp [leftChainFrom]
  | cons next tail induction =>
      simp [leftChainFrom, induction, leaves, List.append_assoc]

theorem leftChainFrom_addNodes (tree : Tree) (rest : List Leaf) :
    addNodes (leftChainFrom tree rest) = addNodes tree + rest.length := by
  induction rest generalizing tree with
  | nil => simp [leftChainFrom]
  | cons next tail induction =>
      simp [leftChainFrom, induction, addNodes, Nat.add_assoc, Nat.add_comm]

@[simp] theorem sourceTree_leaves (first : Word) (rest : List Word) :
    leaves (sourceTree first rest) = labelFrom 0 (first :: rest) := by
  simp [sourceTree, leftChainFrom_leaves, labelFrom, leaves]

@[simp] theorem sourceTree_addNodes (first : Word) (rest : List Word) :
    addNodes (sourceTree first rest) = rest.length := by
  simp [sourceTree, leftChainFrom_addNodes, addNodes]

/-- A nonempty source tree has exactly N-1 actual addition nodes. -/
theorem sourceTree_has_n_minus_one_additions (first : Word) (rest : List Word) :
    addNodes (sourceTree first rest) = (first :: rest).length - 1 := by
  simp

/-- The N=1 path is exactly one raw ABS leaf and contains no addition. -/
theorem singleton_source_is_direct_abs (word : Word) :
    sourceTree word [] = .leaf { ordinal := 0, word := absWord word } ∧
    addNodes (sourceTree word []) = 0 := by
  exact ⟨rfl, rfl⟩

theorem fold_evaluates_left_chain
    {add : Word → Word → Word → Prop}
    {tree : Tree} {current result : Word} {rest : List Leaf}
    (treeEvaluation : TreeEval add tree current)
    (fold : SourceFold add current rest result) :
    TreeEval add (leftChainFrom tree rest) result := by
  induction fold generalizing tree with
  | nil current => simpa [leftChainFrom] using treeEvaluation
  | @cons current intermediate result next rest step tail induction =>
      apply induction
      exact TreeEval.branch treeEvaluation (TreeEval.leaf next) step

/--
Every source result is in the Model set: exact ABS leaves, identical labelled multiplicity,
the left-associated permitted tree, and one admitted primitive result at each add node.
-/
theorem source_result_is_model_result
    {add : Word → Word → Word → Prop}
    (first : Word) (rest : List Word) (result : Word)
    (fold : SourceFold add (absWord first) (labelFrom 1 rest) result) :
    ModelResult add (first :: rest) result := by
  refine ⟨sourceTree first rest, sourceTree_leaves first rest, ?_⟩
  apply fold_evaluates_left_chain (tree := .leaf { ordinal := 0, word := absWord first })
  · exact TreeEval.leaf _
  · exact fold

/-- Every L1 result is either NaN or has a nonnegative binary32 sign. -/
abbrev L1ClassInvariant (word : Word) : Prop :=
  L1Reachable word

def ValidNonNan (word : Word) : Prop :=
  word.sign = false ∧ rawClass word ≠ .nan

def PositiveInfinity (word : Word) : Prop :=
  word.sign = false ∧ rawClass word = .infinity

def AllLeaves (predicate : Leaf → Prop) : List Leaf → Prop
  | [] => True
  | next :: rest => predicate next ∧ AllLeaves predicate rest

def AnyLeaf (predicate : Leaf → Prop) : List Leaf → Prop
  | [] => False
  | next :: rest => predicate next ∨ AnyLeaf predicate rest

@[simp] theorem absWord_l1_class_invariant (word : Word) :
    L1ClassInvariant (absWord word) := by
  exact Or.inr rfl

@[simp] theorem labelFrom_l1_class_invariant (start : Nat) (words : List Word) :
    AllLeaves (fun leaf => L1ClassInvariant leaf.word) (labelFrom start words) := by
  induction words generalizing start with
  | nil => exact True.intro
  | cons word rest induction =>
      exact ⟨absWord_l1_class_invariant word, induction (start + 1)⟩

theorem addSite_l1_class_invariant
    {rne : Word → Word → Word}
    (contract : RneSpecialClassContract rne)
    {left right published : Word}
    (site : addSite rne left right published)
    (leftInvariant : L1ClassInvariant left)
    (rightInvariant : L1ClassInvariant right) :
    L1ClassInvariant published := by
  by_cases leftNan : rawClass left = .nan
  · exact Or.inl (addSite_nan_left contract site leftNan rightInvariant)
  by_cases rightNan : rawClass right = .nan
  · exact Or.inl (addSite_nan_right contract site rightNan leftInvariant)
  exact Or.inr (addSite_nonnegative contract site
    (leftInvariant.resolve_left leftNan)
    (rightInvariant.resolve_left rightNan)
    leftNan rightNan)

theorem sourceFold_l1_class_invariant
    {rne : Word → Word → Word}
    (contract : RneSpecialClassContract rne)
    {current result : Word} {rest : List Leaf}
    (fold : SourceFold (addSite rne) current rest result) :
    L1ClassInvariant current →
    AllLeaves (fun leaf => L1ClassInvariant leaf.word) rest →
    L1ClassInvariant result := by
  induction fold with
  | nil current =>
      intro currentInvariant _
      exact currentInvariant
  | @cons current intermediate result next rest step tail induction =>
      intro currentInvariant restInvariant
      rcases restInvariant with ⟨nextInvariant, remainingInvariant⟩
      exact induction
        (addSite_l1_class_invariant contract step currentInvariant nextInvariant)
        remainingInvariant

/-- The source fold's advertised finite-nonnegative/NaN class consequence. -/
theorem source_result_l1_class_contract
    {rne : Word → Word → Word}
    (contract : Binary32RneContract rne)
    (first : Word) (rest : List Word) (result : Word)
    (fold :
      SourceFold (binary32AddSite rne contract)
        (absWord first) (labelFrom 1 rest) result) :
    L1ClassInvariant result := by
  change SourceFold (addSite rne) (absWord first) (labelFrom 1 rest) result at fold
  exact sourceFold_l1_class_invariant contract.toRneSpecialClassContract fold
    (absWord_l1_class_invariant first)
    (labelFrom_l1_class_invariant 1 rest)

/-- A NaN at any reachable current-or-later contributor position is absorbing. -/
theorem sourceFold_nan_of_contains
    {rne : Word → Word → Word}
    (contract : RneSpecialClassContract rne)
    {current result : Word} {rest : List Leaf}
    (fold : SourceFold (addSite rne) current rest result) :
    L1ClassInvariant current →
    AllLeaves (fun leaf => L1ClassInvariant leaf.word) rest →
    rawClass current = .nan ∨ AnyLeaf (fun leaf => rawClass leaf.word = .nan) rest →
    rawClass result = .nan := by
  induction fold with
  | nil current =>
      intro _ _ contains
      simpa [AnyLeaf] using contains
  | @cons current intermediate result next rest step tail induction =>
      intro currentInvariant restInvariant contains
      rcases restInvariant with ⟨nextInvariant, remainingInvariant⟩
      have intermediateInvariant : L1ClassInvariant intermediate :=
        addSite_l1_class_invariant contract step currentInvariant nextInvariant
      have split :
          rawClass current = .nan ∨
            rawClass next.word = .nan ∨
              AnyLeaf (fun leaf => rawClass leaf.word = .nan) rest := by
        simpa [AnyLeaf] using contains
      rcases split with currentNan | nextNan | laterNan
      · exact induction intermediateInvariant remainingInvariant
          (Or.inl (addSite_nan_left contract step currentNan nextInvariant))
      · exact induction intermediateInvariant remainingInvariant
          (Or.inl (addSite_nan_right contract step nextNan currentInvariant))
      · exact induction intermediateInvariant remainingInvariant (Or.inr laterNan)

theorem sourceFold_positive_infinity_of_contains
    {rne : Word → Word → Word}
    (contract : RneSpecialClassContract rne)
    {current result : Word} {rest : List Leaf}
    (fold : SourceFold (addSite rne) current rest result) :
    ValidNonNan current →
    AllLeaves (fun leaf => ValidNonNan leaf.word) rest →
    (PositiveInfinity current ∨ AnyLeaf (fun leaf => PositiveInfinity leaf.word) rest) →
    PositiveInfinity result := by
  induction fold with
  | nil current =>
      intro _ _ contains
      simpa [AnyLeaf] using contains
  | @cons current intermediate result next rest step tail induction =>
      intro currentValid restValid contains
      rcases restValid with ⟨nextValid, remainingValid⟩
      have intermediateValid : ValidNonNan intermediate := ⟨
        addSite_nonnegative contract step
          currentValid.1 nextValid.1 currentValid.2 nextValid.2,
        addSite_non_nan contract step
          currentValid.1 nextValid.1 currentValid.2 nextValid.2⟩
      have split :
          PositiveInfinity current ∨
            PositiveInfinity next.word ∨
              AnyLeaf (fun leaf => PositiveInfinity leaf.word) rest := by
        simpa [AnyLeaf] using contains
      have intermediateContains :
          PositiveInfinity intermediate ∨
            AnyLeaf (fun leaf => PositiveInfinity leaf.word) rest := by
        rcases split with currentInfinity | nextInfinity | laterInfinity
        · exact Or.inl (addSite_positive_infinity_left
            contract step currentInfinity nextValid)
        · exact Or.inl (addSite_positive_infinity_right
            contract step currentValid nextInfinity)
        · exact Or.inr laterInfinity
      exact induction intermediateValid remainingValid intermediateContains

/-- Instantiation with a constrained bit-level RNE primitive plus complete DAZ/FTZ alternatives. -/
theorem source_result_is_binary32_model_result
    (rne : Word → Word → Word)
    (contract : Binary32RneContract rne)
    (first : Word) (rest : List Word) (result : Word)
    (fold :
      SourceFold (binary32AddSite rne contract)
        (absWord first) (labelFrom 1 rest) result) :
    ModelResult (binary32AddSite rne contract) (first :: rest) result := by
  change SourceFold (addSite rne) (absWord first) (labelFrom 1 rest) result at fold
  change ModelResult (addSite rne) (first :: rest) result
  exact source_result_is_model_result first rest result fold

/-- One update in the source list, before its source position is attached. -/
structure ScatterUpdate where
  target : Nat
  word : Word
  deriving DecidableEq, Repr

/-- One update carrying the only permitted ordinal: its zero-based source-list position. -/
structure ScatterOccurrence where
  ordinal : Nat
  target : Nat
  word : Word
  deriving DecidableEq, Repr

def labelScatterFrom : Nat → List ScatterUpdate → List ScatterOccurrence
  | _, [] => []
  | ordinal, update :: rest =>
      { ordinal := ordinal, target := update.target, word := update.word } ::
        labelScatterFrom (ordinal + 1) rest

def scatterOccurrences (updates : List ScatterUpdate) : List ScatterOccurrence :=
  labelScatterFrom 0 updates

def SourcePositionedFrom : Nat → List ScatterOccurrence → Prop
  | _, [] => True
  | position, occurrence :: rest =>
      occurrence.ordinal = position ∧ SourcePositionedFrom (position + 1) rest

@[simp] theorem labelScatterFrom_length (start : Nat) (updates : List ScatterUpdate) :
    (labelScatterFrom start updates).length = updates.length := by
  induction updates generalizing start with
  | nil => rfl
  | cons update rest induction =>
      simp [labelScatterFrom, induction]

/-- Every occurrence ordinal is constructed from its exact source-list position. -/
theorem labelScatterFrom_source_positions (start : Nat) (updates : List ScatterUpdate) :
    SourcePositionedFrom start (labelScatterFrom start updates) := by
  induction updates generalizing start with
  | nil => exact True.intro
  | cons update rest induction =>
      exact ⟨rfl, induction (start + 1)⟩

def OrdinalsGreaterThan : Nat → List ScatterOccurrence → Prop
  | _, [] => True
  | lower, occurrence :: rest =>
      lower < occurrence.ordinal ∧ OrdinalsGreaterThan lower rest

def StrictlyIncreasingOrdinals : List ScatterOccurrence → Prop
  | [] => True
  | occurrence :: rest =>
      OrdinalsGreaterThan occurrence.ordinal rest ∧ StrictlyIncreasingOrdinals rest

theorem ordinalsGreaterThan_mono
    {lower upper : Nat} (bound : lower ≤ upper) {occurrences : List ScatterOccurrence}
    (greater : OrdinalsGreaterThan upper occurrences) :
    OrdinalsGreaterThan lower occurrences := by
  induction occurrences with
  | nil => exact True.intro
  | cons occurrence rest induction =>
      rcases greater with ⟨head, tail⟩
      exact ⟨Nat.lt_of_le_of_lt bound head, induction tail⟩

theorem labelScatterFrom_greater
    (start : Nat) (updates : List ScatterUpdate) :
    OrdinalsGreaterThan start (labelScatterFrom (start + 1) updates) := by
  induction updates generalizing start with
  | nil => exact True.intro
  | cons update rest induction =>
      exact ⟨Nat.lt_succ_self start,
        ordinalsGreaterThan_mono (Nat.le_succ start) (induction (start + 1))⟩

theorem labelScatterFrom_strictly_increasing
    (start : Nat) (updates : List ScatterUpdate) :
    StrictlyIncreasingOrdinals (labelScatterFrom start updates) := by
  induction updates generalizing start with
  | nil => exact True.intro
  | cons update rest induction =>
      exact ⟨labelScatterFrom_greater start rest, induction (start + 1)⟩

theorem ordinalsGreaterThan_filter
    (lower : Nat) (predicate : ScatterOccurrence → Bool)
    (occurrences : List ScatterOccurrence)
    (greater : OrdinalsGreaterThan lower occurrences) :
    OrdinalsGreaterThan lower (occurrences.filter predicate) := by
  induction occurrences with
  | nil => exact True.intro
  | cons occurrence rest induction =>
      rcases greater with ⟨head, tail⟩
      by_cases keep : predicate occurrence
      · simp [List.filter, keep]
        exact ⟨head, induction tail⟩
      · simpa [List.filter, keep] using induction tail

theorem strictlyIncreasingOrdinals_filter
    (predicate : ScatterOccurrence → Bool)
    (occurrences : List ScatterOccurrence)
    (increasing : StrictlyIncreasingOrdinals occurrences) :
    StrictlyIncreasingOrdinals (occurrences.filter predicate) := by
  induction occurrences with
  | nil => exact True.intro
  | cons occurrence rest induction =>
      rcases increasing with ⟨headGreater, tailIncreasing⟩
      by_cases keep : predicate occurrence
      · simp [List.filter, keep]
        exact ⟨ordinalsGreaterThan_filter occurrence.ordinal predicate rest headGreater,
          induction tailIncreasing⟩
      · simpa [List.filter, keep] using induction tailIncreasing

/-- The exact stable filter traversed by one target thread. -/
def matchingOccurrences (target : Nat) (updates : List ScatterUpdate) :
    List ScatterOccurrence :=
  (scatterOccurrences updates).filter (fun update => update.target = target)

def matchingUpdates (target : Nat) (updates : List ScatterUpdate) : List ScatterUpdate :=
  updates.filter (fun update => update.target = target)

def scatterLeaf (update : ScatterOccurrence) : Leaf :=
  { ordinal := update.ordinal + 1, word := update.word }

def matchingLeaves (target : Nat) (updates : List ScatterUpdate) : List Leaf :=
  (matchingOccurrences target updates).map scatterLeaf

@[simp] theorem matchingLeaves_length (target : Nat) (updates : List ScatterUpdate) :
    (matchingLeaves target updates).length = (matchingOccurrences target updates).length := by
  simp [matchingLeaves]

theorem filteredLabelScatterFrom_length
    (target start : Nat) (updates : List ScatterUpdate) :
    ((labelScatterFrom start updates).filter
        (fun update => update.target = target)).length =
      (matchingUpdates target updates).length := by
  induction updates generalizing start with
  | nil => rfl
  | cons update rest induction =>
      by_cases sameTarget : update.target = target
      · simp [labelScatterFrom, matchingUpdates, sameTarget, induction (start + 1)]
      · simp [labelScatterFrom, matchingUpdates, sameTarget, induction (start + 1)]

/--
Filtering preserves duplicate multiplicity exactly: the occurrence count equals the source-update
count for the target, with no uniqueness or set conversion.
-/
theorem matchingOccurrences_duplicate_multiplicity
    (target : Nat) (updates : List ScatterUpdate) :
    (matchingOccurrences target updates).length =
      (matchingUpdates target updates).length := by
  exact filteredLabelScatterFrom_length target 0 updates

theorem filteredLabelScatterFrom_payloads
    (target start : Nat) (updates : List ScatterUpdate) :
    ((labelScatterFrom start updates).filter
        (fun update => update.target = target)).map
          (fun occurrence => (occurrence.target, occurrence.word)) =
      (matchingUpdates target updates).map
        (fun update => (update.target, update.word)) := by
  induction updates generalizing start with
  | nil => rfl
  | cons update rest induction =>
      by_cases sameTarget : update.target = target
      · simp [labelScatterFrom, matchingUpdates, sameTarget, induction (start + 1)]
      · simp [labelScatterFrom, matchingUpdates, sameTarget, induction (start + 1)]

/--
The stable occurrence filter preserves every source update target and raw word in order, so equal
targets with equal or different words retain their exact multiplicity and contributor payload.
-/
theorem matchingOccurrences_preserve_source_payloads
    (target : Nat) (updates : List ScatterUpdate) :
    (matchingOccurrences target updates).map
        (fun occurrence => (occurrence.target, occurrence.word)) =
      (matchingUpdates target updates).map
        (fun update => (update.target, update.word)) := by
  exact filteredLabelScatterFrom_payloads target 0 updates

/-- The filtered occurrence ordinals are strictly increasing source positions. -/
theorem matchingOccurrences_in_source_order
    (target : Nat) (updates : List ScatterUpdate) :
    StrictlyIncreasingOrdinals (matchingOccurrences target updates) := by
  exact strictlyIncreasingOrdinals_filter
    (fun update => update.target = target)
    (scatterOccurrences updates)
    (labelScatterFrom_strictly_increasing 0 updates)

/-- The filtered contributor words are exactly the matching update occurrences in source order. -/
@[simp] theorem matchingLeaves_words (target : Nat) (updates : List ScatterUpdate) :
    (matchingLeaves target updates).map Leaf.word =
      (matchingOccurrences target updates).map ScatterOccurrence.word := by
  simp [matchingLeaves, scatterLeaf]

/-- Source positions survive the stable filter and follow the distinguished base at ordinal zero. -/
@[simp] theorem matchingLeaves_ordinals (target : Nat) (updates : List ScatterUpdate) :
    (matchingLeaves target updates).map Leaf.ordinal =
      (matchingOccurrences target updates).map (fun update => update.ordinal + 1) := by
  simp [matchingLeaves, scatterLeaf]

def scatterTree (target : Nat) (base : Word) (updates : List ScatterUpdate) : Tree :=
  leftChainFrom
    (.leaf { ordinal := 0, word := base })
    (matchingLeaves target updates)

/-- Base plus the complete stable filtered occurrence list is the permitted target tree. -/
def ScatterModelResult (add : Word → Word → Word → Prop)
    (target : Nat) (base : Word) (updates : List ScatterUpdate) (result : Word) : Prop :=
  ∃ tree,
    leaves tree =
      { ordinal := 0, word := base } :: matchingLeaves target updates ∧
    TreeEval add tree result

@[simp] theorem scatterTree_leaves
    (target : Nat) (base : Word) (updates : List ScatterUpdate) :
    leaves (scatterTree target base updates) =
      { ordinal := 0, word := base } :: matchingLeaves target updates := by
  simp [scatterTree, leftChainFrom_leaves, leaves]

/--
An addressed source result is exactly the left chain of the raw base followed by every matching
update occurrence in increasing source-position order. No uniqueness filter is present.
-/
theorem scatter_source_result_is_model_result
    {add : Word → Word → Word → Prop}
    (target : Nat) (base result : Word) (updates : List ScatterUpdate)
    (fold : SourceFold add base (matchingLeaves target updates) result) :
    ScatterModelResult add target base updates result := by
  refine ⟨scatterTree target base updates, scatterTree_leaves target base updates, ?_⟩
  apply fold_evaluates_left_chain (tree := .leaf { ordinal := 0, word := base })
  · exact TreeEval.leaf _
  · exact fold

/-- Scatter instantiation with the all-raw-pairs signed binary32 RNE contract. -/
theorem scatter_source_result_is_general_binary32_model_result
    (rne : Word → Word → Word)
    (contract : GeneralBinary32RneContract rne)
    (target : Nat) (base result : Word) (updates : List ScatterUpdate)
    (fold :
      SourceFold (generalBinary32AddSite rne contract)
        base (matchingLeaves target updates) result) :
    ScatterModelResult
      (generalBinary32AddSite rne contract) target base updates result := by
  change SourceFold (addSite rne) base (matchingLeaves target updates) result at fold
  change ScatterModelResult (addSite rne) target base updates result
  exact scatter_source_result_is_model_result target base result updates fold

theorem filteredLabelScatterFrom_eq_nil_of_unaddressed
    (target start : Nat) (updates : List ScatterUpdate)
    (unaddressed : ∀ update ∈ updates, update.target ≠ target) :
    (labelScatterFrom start updates).filter
      (fun update => update.target = target) = [] := by
  induction updates generalizing start with
  | nil => rfl
  | cons update rest induction =>
      have head : update.target ≠ target := unaddressed update (by simp)
      have tail : ∀ item ∈ rest, item.target ≠ target := by
        intro item member
        exact unaddressed item (by simp [member])
      simp [labelScatterFrom, head, induction (start + 1) tail]

theorem matchingOccurrences_eq_nil_of_unaddressed
    (target : Nat) (updates : List ScatterUpdate)
    (unaddressed : ∀ update ∈ updates, update.target ≠ target) :
    matchingOccurrences target updates = [] := by
  exact filteredLabelScatterFrom_eq_nil_of_unaddressed target 0 updates unaddressed

/-- A target with no matching occurrence is a bit-for-bit raw base identity, not an arithmetic site. -/
theorem scatter_unaddressed_raw_identity
    {add : Word → Word → Word → Prop}
    (target : Nat) (base result : Word) (updates : List ScatterUpdate)
    (unaddressed : ∀ update ∈ updates, update.target ≠ target)
    (fold : SourceFold add base (matchingLeaves target updates) result) :
    result = base := by
  have filtered : matchingOccurrences target updates = [] :=
    matchingOccurrences_eq_nil_of_unaddressed target updates unaddressed
  have noLeaves : matchingLeaves target updates = [] := by
    simp [matchingLeaves, filtered]
  rw [noLeaves] at fold
  cases fold
  rfl

/-- The kernel ownership map assigns exactly one injective writer thread to each target. -/
def scatterWriterThread (target : Nat) : Nat :=
  target

theorem scatterWriterThread_injective : Function.Injective scatterWriterThread := by
  intro left right equality
  exact equality

theorem scatter_only_target_thread_writes
    (thread target : Nat) (owns : thread = scatterWriterThread target) :
    thread = target := by
  exact owns
