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
