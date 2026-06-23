# KKT Boundary Principle Schematic

`kkt_boundary_schematic.png`

## CANVAS
- Square, 1:1 aspect ratio (2000x2000 px)
- Two triangular panels arranged diagonally: upper-left triangle = Panel A (concept), lower-right triangle = Panel B (strategies)
- Thin diagonal line separating them, from top-right to bottom-left
- White background, clean scientific illustration style

## PANEL A (upper-left triangle): "Constrained Optimum at Feasibility Boundary"

A 2D composition-like space showing the KKT principle:

- **Feasible region**: large area filled with very light teal (#E6F5F5), labeled "Feasible (FCC)"
- **Infeasible region**: area beyond a curved boundary, filled with very light red (#F5E6E6), labeled "Infeasible (sigma)"
- **Feasibility boundary**: smooth curved line separating the two regions, drawn as a thick dark teal line (#007A7A), labeled "Phase boundary g(x) = 0"
- **Objective contours**: 4-5 concentric elliptical contour lines in light gray, with an arrow labeled "Increasing strength" pointing toward the infeasible region. Contours should be roughly parallel to the boundary
- **Unconstrained optimum**: a hollow red circle (#C55A11) in the infeasible region, labeled "Unconstrained optimum" with a small "x" marker
- **Constrained optimum (KKT point)**: a filled dark teal circle (#007A7A) ON the boundary, labeled "Constrained optimum (KKT)"
- **Gradient arrows at KKT point**: two arrows originating from the KKT point:
  - One arrow labeled "nabla f" (objective gradient) pointing into infeasible region, colored maroon (#500000)
  - One arrow labeled "nabla g" (constraint gradient) pointing in same direction, colored dark teal (#007A7A)
  - Both arrows approximately parallel — this is the KKT alignment condition
- Label "(a)" in top-left corner

## PANEL B (lower-right triangle): "Three Acquisition Strategies"

Same 2D space as Panel A but showing three strategy outcomes:

- Same feasible/infeasible regions and boundary as Panel A
- **Hard Filter strategy**: 
  - A dashed exclusion zone near the boundary (gray hatching or light gray band)
  - Several blue dots (#1E5A8C) clustered far from boundary in the interior
  - Label: "Hard filter" with subtitle "Excludes boundary"
  - Small red "X" marks where BBC04/BBC02 would have been (never synthesized)

- **Blind BO strategy**:
  - Several orange dots (#C55A11) scattered across both feasible AND infeasible regions
  - Multiple dots in infeasible region (wasted experiments)
  - Label: "Blind BO" with subtitle "Wastes budget"

- **BRAVE strategy**:
  - Several green dots (#2E7D32) concentrated near but mostly inside the boundary
  - One or two dots right on the boundary (the best alloys)
  - Very few dots in infeasible region
  - Label: "BRAVE" with subtitle "Navigates boundary"

- Label "(b)" in bottom-right corner

## STYLE
- 2.5D stylized vector art with soft shadows, NOT photorealistic, NOT flat
- ARM MIP pastel palette: light fills (#E6F5F5, #F5E6E6, #C8E6C9, #D4E4F7, #FFE8D1) with dark accents (#007A7A, #500000, #2E7D32, #1E5A8C, #C55A11)
- Minimal text — keep labels to 2-3 words max
- Clean sans-serif font for labels
- Science/Nature publication quality
- No decorative elements, no 3D perspective — flat 2D composition space view

## ALTERNATIVE SIMPLIFIED PROMPT

Create a square scientific figure with two triangular panels separated by a diagonal line.

Upper-left panel (a): Shows a 2D design space with a curved feasibility boundary. Light teal feasible region, light red infeasible region. Gray objective contours increase toward the boundary. A hollow red circle marks the unconstrained optimum in the infeasible zone. A filled teal circle marks the constrained optimum ON the boundary (KKT point). Two gradient arrows at the KKT point (one maroon, one teal) point in the same direction into the infeasible region, showing gradient alignment.

Lower-right panel (b): Same space, but showing three strategies with colored dots. Blue dots far from boundary = "Hard filter" (misses best region). Orange dots scattered everywhere including infeasible = "Blind BO" (wastes budget). Green dots concentrated near boundary, mostly feasible = "BRAVE" (navigates boundary). 

Use ARM MIP color palette: teal #007A7A, maroon #500000, green #2E7D32, blue #1E5A8C, orange #C55A11 as accents on pastel backgrounds #E6F5F5 and #F5E6E6. Publication quality, 2.5D style with soft shadows, minimal text labels.
