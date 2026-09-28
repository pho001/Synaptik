import Task0069Binary32

namespace Task0071

open Task0069

/-- The only source-ordered operations admitted after an anchor in schema 18. -/
inductive Opcode where
  | scalarMul
  | add
  | relu
  | clamp
  deriving DecidableEq, Repr

/-- The bounded v1 epilogue description. `none` means that source site is absent. -/
structure Epilogue where
  scalar : Option (Word → Word)
  add : Option (Word → Word → Word)
  terminal : Option (Opcode × (Word → Word))

/-- Literal source order: scalar, external tensor add, then terminal. -/
def opcodes (epilogue : Epilogue) : List Opcode :=
  let scalarOps : List Opcode :=
    if epilogue.scalar.isSome then [Opcode.scalarMul] else []
  let addOps : List Opcode :=
    if epilogue.add.isSome then [Opcode.add] else []
  let terminalOps : List Opcode :=
    match epilogue.terminal with
    | some (opcode, _) => [opcode]
    | none => []
  scalarOps ++ addOps ++ terminalOps

/-- Evaluate the source grammar without reassociation or an intermediate publication. -/
def evaluate (epilogue : Epilogue) (anchor external : Word) : Word :=
  let afterScalar := epilogue.scalar.map (fun site => site anchor) |>.getD anchor
  let afterAdd := epilogue.add.map (fun site => site afterScalar external) |>.getD afterScalar
  epilogue.terminal.map (fun site => site.2 afterAdd) |>.getD afterAdd

/-- One native anchor step has one dispatch and reaches one final store. -/
structure PhysicalTrace where
  sourceSites : List Opcode
  dispatches : Nat
  intermediateStores : Nat
  finalStores : Nat

/-- The trace shape implemented by each Task-0071 kernel. -/
def physicalTrace (epilogue : Epilogue) : PhysicalTrace :=
  { sourceSites := opcodes epilogue
    dispatches := 1
    intermediateStores := 0
    finalStores := 1 }

@[simp] theorem one_physical_dispatch (epilogue : Epilogue) :
    (physicalTrace epilogue).dispatches = 1 := rfl

@[simp] theorem no_intermediate_store (epilogue : Epilogue) :
    (physicalTrace epilogue).intermediateStores = 0 := rfl

@[simp] theorem one_final_store (epilogue : Epilogue) :
    (physicalTrace epilogue).finalStores = 1 := rfl

/-- With all three sites present, evaluation is exactly scalar then add then terminal. -/
theorem scalar_add_terminal_source_order
    (scalarSite : Word → Word)
    (addSite : Word → Word → Word)
    (terminalSite : Word → Word)
    (anchor external : Word) :
    evaluate
      { scalar := some scalarSite, add := some addSite,
        terminal := some (.relu, terminalSite) }
      anchor external = terminalSite (addSite (scalarSite anchor) external) := by
  rfl

/-- Without scalar multiplication, the external add still precedes the terminal site. -/
theorem add_terminal_source_order
    (addSite : Word → Word → Word)
    (terminalSite : Word → Word)
    (anchor external : Word) :
    evaluate
      { scalar := none, add := some addSite,
        terminal := some (.clamp, terminalSite) }
      anchor external = terminalSite (addSite anchor external) := by
  rfl

/-- A terminal-only suffix receives the anchor value directly. -/
theorem terminal_only_source_order
    (terminalSite : Word → Word)
    (anchor external : Word) :
    evaluate
      { scalar := none, add := none, terminal := some (.relu, terminalSite) }
      anchor external = terminalSite anchor := by
  rfl

/-- One decoded NCHW output coordinate. -/
structure NchwCoordinate where
  batch : Nat
  channel : Nat
  height : Nat
  width : Nat

/-- Row-major NCHW linearization used by the output coordinate decoder. -/
def rowMajorNchwLinear
    (coordinate : NchwCoordinate)
    (channelExtent heightExtent widthExtent : Nat) : Nat :=
  (((coordinate.batch * channelExtent + coordinate.channel) * heightExtent
      + coordinate.height) * widthExtent) + coordinate.width

/--
The native right-aligned rank-one metadata gives all leading output axes zero stride and the last
axis the canonical addend stride. This is its resulting element offset.
-/
def rankOneRightAlignedOffset
    (addOffset addStride : Nat) (coordinate : NchwCoordinate) : Nat :=
  addOffset + coordinate.width * addStride

/-- A canonical zero-offset rank-one addend therefore indexes the NCHW width coordinate. -/
@[simp] theorem rank_one_add_is_width_broadcast (coordinate : NchwCoordinate) :
    rankOneRightAlignedOffset 0 1 coordinate = coordinate.width := by
  simp [rankOneRightAlignedOffset]

/-- The row-major linear decoder's final coordinate is the same width coordinate. -/
theorem row_major_linear_decodes_width
    (coordinate : NchwCoordinate)
    (channelExtent heightExtent widthExtent : Nat)
    (widthBound : coordinate.width < widthExtent) :
    rowMajorNchwLinear coordinate channelExtent heightExtent widthExtent % widthExtent =
      coordinate.width := by
  simp [rowMajorNchwLinear, Nat.mod_eq_of_lt widthBound]

/-- DAZ has only the exact input or its same-sign zero; it introduces no third value. -/
theorem daz_boundary_is_exhaustive (source presented : Word) :
    Task0069.daz source presented ↔
      presented = source ∨
        (Task0069.rawClass source = .subnormal ∧
          presented = Task0069.signedZero source.sign) :=
  Task0069.daz_complete source presented

/-- FTZ has only the rounded result or its same-sign zero; it introduces no third value. -/
theorem ftz_boundary_is_exhaustive (rounded published : Word) :
    Task0069.ftz rounded published ↔
      published = rounded ∨
        (Task0069.rawClass rounded = .subnormal ∧
          published = Task0069.signedZero rounded.sign) :=
  Task0069.ftz_complete rounded published

end Task0071
