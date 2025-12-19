import Lake
open Lake DSL

package anspg where
  -- Remove LeanCopilot from args if not using it to prevent link errors
  moreLinkArgs := #["-lctranslate2"] 

-- Update tag to match the toolchain
require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.27.0-rc1"

@[default_target]
lean_lib ANSPG where
  srcDir := "src"
