import Task0069Binary32

namespace Task0070Relu

open Task0069

def rawBits (word : Word) : Nat := (Task0069.rawWord word).val

/-- Literal raw-u32 semantics of the frozen MSL `relu_bits` helper. -/
def reluBitsSource (word : Word) : Nat :=
  if rawClass word = .nan then rawBits word
  else if word.sign then 0 else rawBits word

/-- Result-set model for RELU on every raw class. -/
def ReluModel (word : Word) (result : Nat) : Prop :=
  if rawClass word = .nan then
    result = rawBits word
  else
    result = if word.sign then 0 else rawBits word

theorem relu_universal_raw_certificate (word : Word) :
    ReluModel word (reluBitsSource word) := by
  unfold ReluModel reluBitsSource
  by_cases nan : rawClass word = .nan
  · simp [nan]
  · simp [nan]

theorem relu_permitted_iff (word : Word) (result : Nat) :
    ReluModel word result ↔ result = reluBitsSource word := by
  unfold ReluModel reluBitsSource
  by_cases nan : rawClass word = .nan
  · simp [nan]
  · simp [nan]

theorem relu_all_raw32_certificate (raw : Raw32) :
    ReluModel (wordOfRaw raw) (reluBitsSource (wordOfRaw raw)) :=
  relu_universal_raw_certificate (wordOfRaw raw)

/-- NaN payload/sign are returned bit-for-bit. -/
theorem relu_nan_raw_identity (word : Word) (nan : rawClass word = .nan) :
    reluBitsSource word = rawBits word := by
  simp [reluBitsSource, nan]

/-- Negative infinity and every negative finite/subnormal/zero word map to positive zero. -/
theorem relu_negative_non_nan_is_positive_zero
    (word : Word) (negative : word.sign = true) (notNan : rawClass word ≠ .nan) :
    reluBitsSource word = 0 := by
  simp [reluBitsSource, notNan, negative]

/-- Positive words, including +0 and +infinity, are returned bit-for-bit. -/
theorem relu_positive_raw_identity (word : Word) (positive : word.sign = false) :
    reluBitsSource word = rawBits word := by
  by_cases nan : rawClass word = .nan
  · simp [reluBitsSource, nan]
  · simp [reluBitsSource, nan, positive]

end Task0070Relu
