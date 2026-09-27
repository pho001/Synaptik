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

/-- Instantiation with the complete exact/RNE/DAZ/FTZ primitive relation. -/
theorem source_result_is_binary32_model_result
    (rne : Word → Word → Word)
    (first : Word) (rest : List Word) (result : Word)
    (fold : SourceFold (addSite rne) (absWord first) (labelFrom 1 rest) result) :
    ModelResult (addSite rne) (first :: rest) result :=
  source_result_is_model_result first rest result fold

end Task0069
