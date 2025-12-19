import Lake
open Lake DSL

package anspg where
  -- Minimal configuration for testing
  moreLinkArgs := #[]

@[default_target]
lean_lib ANSPG_minimal where
  srcDir := "src"
