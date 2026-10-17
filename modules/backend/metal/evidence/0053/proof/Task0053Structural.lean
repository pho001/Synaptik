import Lean

namespace Task0053

inductive RawClass where
  | nan
  | infinity
  | zero
  | subnormal
  | normal
  deriving DecidableEq, Repr

def classify (word : UInt32) : RawClass :=
  let magnitude := word &&& 0x7fffffff
  let exponent := magnitude >>> 23
  let fraction := magnitude &&& 0x007fffff
  if exponent == 0xff then
    if fraction == 0 then .infinity else .nan
  else if magnitude == 0 then .zero
  else if exponent == 0 then .subnormal
  else .normal

theorem classify_total (word : UInt32) :
    classify word = .nan ∨ classify word = .infinity ∨ classify word = .zero ∨
    classify word = .subnormal ∨ classify word = .normal := by
  unfold classify
  grind

theorem raw_partition_count :
    2 * ((2^23 : Nat) - 1) + 2 + 2 + 2 * ((2^23 : Nat) - 1) +
        2 * 254 * 2^23 = 2^32 := by
  decide

def linear (x y z width height : Nat) : Nat :=
  x + width * (y + height * z)

def coordinate (ordinal width height : Nat) : Nat × Nat × Nat :=
  (ordinal % width, (ordinal / width) % height, (ordinal / width) / height)

theorem coordinate_maps_back (ordinal width height : Nat) :
    let c := coordinate ordinal width height
    linear c.1 c.2.1 c.2.2 width height = ordinal := by
  simp [coordinate, linear, Nat.mod_add_div]

theorem coordinate_bounds (ordinal width height : Nat)
    (widthPositive : 0 < width) (heightPositive : 0 < height) :
    let c := coordinate ordinal width height
    c.1 < width ∧ c.2.1 < height := by
  simp [coordinate, Nat.mod_lt, widthPositive, heightPositive]

theorem coordinate_unique
    (ordinal width height x y z : Nat)
    (widthPositive : 0 < width)
    (heightPositive : 0 < height)
    (xBound : x < width)
    (yBound : y < height)
    (maps : linear x y z width height = ordinal) :
    coordinate ordinal width height = (x, y, z) := by
  have outer : ordinal / width = y + height * z ∧ ordinal % width = x :=
    (Nat.div_mod_unique widthPositive).2 ⟨by simpa [linear] using maps, xBound⟩
  have inner : (ordinal / width) / height = z ∧ (ordinal / width) % height = y :=
    (Nat.div_mod_unique heightPositive).2 ⟨outer.1.symm, yBound⟩
  simp [coordinate, outer.2, inner.2, inner.1]

def writes (ordinal elementCount : Nat) : Bool := ordinal < elementCount

theorem excess_thread_writes_nothing (ordinal elementCount : Nat)
    (excess : elementCount ≤ ordinal) : writes ordinal elementCount = false := by
  simp [writes, Nat.not_lt.mpr excess]

theorem admitted_thread_is_in_bounds (ordinal elementCount : Nat)
    (admitted : writes ordinal elementCount = true) : ordinal < elementCount := by
  simpa [writes] using admitted

def rankAdmitted (rank : Nat) : Bool := 1 ≤ rank && rank ≤ 16

theorem admitted_rank_bounds (rank : Nat) (admitted : rankAdmitted rank = true) :
    1 ≤ rank ∧ rank ≤ 16 := by
  simpa [rankAdmitted] using admitted

-- The numerical DOMAIN-PASS theorem is deliberately absent until the retained
-- transcendental/table/polynomial/reconstruction certificate is kernel checked.

end Task0053
