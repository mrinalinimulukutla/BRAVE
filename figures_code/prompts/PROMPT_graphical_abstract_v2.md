# Graphical Abstract v2 — BRAVE Campaign 2 (Acta Materialia)

`graphical_abstract_v2.png`

## GOAL

Elsevier / Acta Materialia graphical abstract for the BRAVE (Bayesian
Risk-aware Alloy discoVery and Exploration) paper. Wide landscape
(16:9), high visual density, publication-quality infographic. Reads
left-to-right at journal-TOC size, communicates: 8-element FCC HEA
design space → risk-aware Bayesian optimization loop with feasibility
awareness → boundary-adjacent high-performance alloys.

Match the visual density and iconography level of the Digital Discovery
review figures (isometric/2.5D icons, blueprint-grid background, per-
panel accent color, embedded mini-charts, no photorealism, no cartoon
styling).

## ABSOLUTE REQUIREMENTS

1. STYLE: 2.5D stylized vector infographic with soft shadows, isometric
   feel where useful. NOT photorealistic. NOT flat. NOT hand-drawn.
2. FORMAT: 16:9 landscape. Safe margins. High readability at TOC
   thumbnail scale.
3. TEXT: Short labels only, 2–6 words. No equations. No paragraphs.
4. FLOW: Left-to-right progression with bold directional arrows between
   panels; thinner information-flow arrows inside each panel.
5. BACKGROUND: Very light blueprint-grid pattern (#F7F8FA base with
   faint #E5E7EB grid), NOT a decorative border.
6. EMPHASIS: Center panel (risk-aware BO loop) is the largest and most
   detailed; the two flanking panels frame it.

## LAYOUT

Three horizontal panels linked by two bold rightward decision-flow
arrows. Panel B (middle) is ~40% of the width; Panels A and C are ~30%
each. Each panel has a small header label at top-left in a colored
capsule.

### Panel A (Left) — "8-Element Design Space"

Header capsule: teal (#007A7A).

Domain content, all embedded:
- A stylized ternary/barycentric composition triangle occupying most of
  the panel. Pastel teal (#E6F5F5) fill for the feasible region and
  pastel red (#F5E6E6) fill for the infeasible sigma-phase region,
  separated by a curved dark-teal (#007A7A) phase-stability boundary
  line rendered with slight thickness variation.
- Overlay a small periodic-table strip along the top of the triangle
  listing the 8 elements: Al, V, Cr, Mn, Fe, Co, Ni, Cu — each in its
  own rounded cell colored by group (transition metals blue-gray, Al
  light yellow, Cu warm orange).
- Scatter ~20 tiny composition points on the diagram: solid teal
  circles inside feasible region, hollow maroon (#500000) circles
  inside infeasible region, and 3 highlighted larger teal-outlined
  circles ON the boundary line labeled "Boundary optima" with a thin
  callout line.
- Small V-axis arrow inside the triangle labeled "V ↑" pointing toward
  the infeasible corner.
- Bottom-of-panel micro-caption text: "27,240 CALPHAD-feasible
  candidates".
- Small CALPHAD icon in a corner: a miniature 2-phase diagram
  (temperature vs. composition) with FCC/sigma phase fields shaded.

### Panel B (Center) — "Risk-Aware Bayesian Optimization"

Header capsule: dark teal (#0F766E), largest.

A stylized closed-loop with a central hexagonal decision node and four
knowledge modules feeding it. Layout: hexagon at center; four rounded
module boxes arranged around it (top, right, bottom, left), each
connected by thick decision-flow arrows going clockwise.

CENTER: A dark-teal hexagon labeled "Risk-Weighted Acquisition" with a
small stylized brain-lattice or gears+dashboard icon inside.

MODULE 1 (TOP) — "GP Objective Surrogates"
- Pastel green (#C8E6C9) rounded box.
- Icon: 5 stacked mini-plots showing 5 GP posterior curves with
  uncertainty bands (YS, UTS/YS, εUTS, Hdyn/Hqs, DoP as label chips).
- Small text: "5 objectives".

MODULE 2 (RIGHT) — "Feasibility Classifier"
- Pastel red (#F5E6E6) rounded box.
- Icon: a 2D composition slice with a soft-edged pastel-red probability
  mask overlaying a feasible pocket; contour lines showing p(feasible)
  gradient.
- Small text: "GPC on phase outcomes".

MODULE 3 (BOTTOM) — "Diversity-Aware Batch"
- Pastel orange (#FFE8D1) rounded box.
- Icon: 3 grouped clusters of small dots (k-medoids clusters) with
  medoid points marked, and a "37 subsystems" label chip.
- Small text: "k-medoids, 16/iter".

MODULE 4 (LEFT) — "Experiment + Simulation"
- Pastel blue (#D4E4F7) rounded box.
- Icon: a small isometric row showing (a) an arc-melting crucible with
  a molten droplet, (b) a nanoindentation tip on a sample, (c) a
  Cowper–Symonds simulated impact schematic (sphere striking a target
  block).
- Small text: "3 iterations × 16 alloys".

Between hexagon and each module, a thin double-headed information-flow
arrow labeled with the shared quantity: "μ, σ" (to GP), "p_feas" (to
GPC), "batch" (to batch selector), "y_obs" (from experiment).

Bottom-of-panel micro-caption: "Feasibility-penalized EHVI on the
Pareto frontier".

### Panel C (Right) — "Boundary-Adjacent Optima"

Header capsule: maroon (#500000).

Three stacked mini-visualizations, all embedded:

TOP — Property landscape.
- A 2D V–Ni composition plane with an interpolated YS heatmap (light
  blue → deep teal gradient) and 3 highlighted stars marking the top
  performers (BBC04, BBC02, BBB01). A small "V = 24 at.%" annotation
  arrow points to the star cluster.

MIDDLE — Stress-strain curve.
- Compact chart with x-axis "Strain (%)" and y-axis "Stress (MPa)". Two
  curves: dark teal (#007A7A) "BRAVE alloy (BBC04)" reaching ~1480 MPa
  at ~50% strain; light gray "Reference HEA" curve reaching ~800 MPa.
  Fill area under the BRAVE curve with a translucent teal wash to
  emphasize toughness. Small callout: "UTS/YS > 4".

BOTTOM — KKT annotation.
- A tiny inset showing 5 elliptical objective contours with an orange
  arrow "unconstrained optimum" pointing to a large teal dot ON a
  boundary line, labeled "KKT-active optimum". Text below the inset:
  "Best alloys sit at the constraint boundary".

Bottom-of-panel micro-caption: "1480 MPa UTS, 50% elongation, single-
phase FCC".

## CONNECTING ARROWS BETWEEN PANELS

- A → B: thick pastel-teal (#007A7A) rightward arrow labeled "sample".
- B → C: thick pastel-maroon (#500000) rightward arrow labeled
  "measure + learn".

Each connecting arrow has a small feedback loop underneath (thin dashed
line) circling back from Panel C to Panel B labeled "y, phase, DoP".

## ICON SEMANTICS (STRICT, DO NOT SUBSTITUTE)

- CALPHAD icons: use a mini phase diagram (T vs. composition) with FCC
  / sigma phase fields — NOT a generic database, brain, or gear icon.
- Feasibility icons: use a soft probability mask over a composition
  region — NOT a lock, shield, or check-mark.
- Batch icons: use grouped scatter clusters with medoid highlights —
  NOT generic boxes or bar charts.
- Experiment icons: use domain-correct arc-melting crucible,
  nanoindentation tip, impact projectile — NOT generic beaker or
  test-tube icons.
- Property landscape: use a genuine 2D composition heatmap with V and
  Ni axes labeled — NOT a random gradient blob.
- Stress-strain: use true engineering-mechanics curve shape (elastic
  rise → plastic plateau → gentle strain hardening → necking) — NOT a
  monotone-linear line.

## LEGEND (SMALL, BOTTOM-RIGHT)

Compact legend in a tiny box:
- Thick arrow = decision flow
- Thin arrow = information flow
- Solid marker = feasible experiment
- Hollow marker = infeasible experiment

## COLOR PALETTE

- Background: #F7F8FA (with faint #E5E7EB blueprint grid).
- Primary text: #1F2937.
- Secondary text: #4B5563.
- Panel A accent: teal (#007A7A) with pastel teal (#E6F5F5) fill.
- Panel B accent: dark teal (#0F766E), with sub-modules using pastel
  green (#C8E6C9), pastel red (#F5E6E6), pastel orange (#FFE8D1), and
  pastel blue (#D4E4F7).
- Panel C accent: maroon (#500000) with pastel red (#F5E6E6) support.
- Connector arrows: #374151 or panel-accent-matched.

## TYPOGRAPHY

- Sans-serif, publication-safe (Inter / Helvetica / Arial family).
- Panel headers: semibold, ~14 pt effective.
- Node labels: regular, 2–6 words each.
- Micro-captions: light, one line.

## COMPOSITION CONSTRAINTS

- Generous whitespace between panels; no text overlap.
- All text legible when scaled to 200×500 px TOC thumbnail (test by
  imagining the panel headers must still be readable).
- No decorative frames, corner ornaments, or drop-shadow abuse.
- No equations, no long math symbols beyond μ, σ, V.
- Composition must read left-to-right at a glance.

---

## ALTERNATIVE SIMPLIFIED PROMPT

Create a wide 16:9 landscape publication-quality scientific graphical abstract for an Acta Materialia paper on risk-aware Bayesian optimization of high-entropy alloys. Three horizontal panels linked by bold rightward arrows on a very faint blueprint-grid background (#F7F8FA with #E5E7EB grid).

**Panel A (left, 30%) — "8-Element Design Space".** Barycentric ternary composition triangle with pastel-teal (#E6F5F5) feasible region and pastel-red (#F5E6E6) infeasible sigma-phase region separated by a curved dark-teal (#007A7A) phase-stability boundary. A small periodic-table strip along the top listing Al, V, Cr, Mn, Fe, Co, Ni, Cu in colored cells. Scatter ~20 composition points: solid teal circles in feasible region, hollow maroon (#500000) circles in infeasible region, 3 highlighted teal-outline circles labeled "Boundary optima" sitting on the boundary. A small "V ↑" arrow inside the triangle pointing to the infeasible side. Corner icon: mini T-vs-composition phase diagram with FCC/sigma fields. Caption: "27,240 CALPHAD-feasible candidates".

**Panel B (center, 40%, largest) — "Risk-Aware Bayesian Optimization".** Central dark-teal hexagon labeled "Risk-Weighted Acquisition". Four rounded module boxes arranged around it clockwise: (top, pastel green) "GP Objective Surrogates" with 5 stacked mini-GP-posterior plots and chip "5 objectives"; (right, pastel red) "Feasibility Classifier" with a soft probability mask over a composition slice; (bottom, pastel orange) "Diversity-Aware Batch" with k-medoids cluster icons and chip "16/iter"; (left, pastel blue) "Experiment + Simulation" with isometric icons of arc-melting crucible, nanoindentation tip, and impact-simulation projectile. Thick clockwise decision-flow arrows between hexagon and modules; thin double-headed information-flow arrows labeled μ, σ, p_feas, batch, y_obs. Caption: "Feasibility-penalized EHVI on the Pareto frontier".

**Panel C (right, 30%) — "Boundary-Adjacent Optima".** Three stacked mini-visualizations: (top) V–Ni composition plane with YS heatmap and 3 stars marking BBC04/BBC02/BBB01 with "V = 24 at.%" callout; (middle) compact stress-strain curve showing dark-teal "BRAVE alloy (BBC04)" reaching ~1480 MPa at ~50% strain with translucent teal fill under curve and callout "UTS/YS > 4", vs. light-gray "Reference HEA" at ~800 MPa; (bottom) KKT inset with elliptical objective contours, an orange arrow from an "unconstrained optimum" to a teal "KKT-active optimum" dot sitting on a boundary line, labeled "Best alloys at constraint boundary". Caption: "1480 MPa UTS, 50% elongation, single-phase FCC".

Connectors: thick teal (#007A7A) arrow A→B labeled "sample"; thick maroon (#500000) arrow B→C labeled "measure + learn"; thin dashed feedback loop from C back to B labeled "y, phase, DoP".

Icon rules (strict): CALPHAD = mini phase diagram not a database icon; feasibility = probability mask not a lock/shield; batch = grouped clusters with medoids not generic boxes; experiment = domain-correct crucible/indenter/impact glyphs not generic test tubes; property landscape = genuine V–Ni heatmap not a gradient blob; stress-strain = correct elastic-plastic-hardening curve shape.

Style: 2.5D stylized vector infographic with soft shadows, isometric feel where useful, publication-quality, high visual density but generous whitespace, no photorealism, no cartoon styling, no decorative frames, no equations, sans-serif labels 2-6 words each. Bottom-right compact legend: thick arrow = decision flow, thin arrow = information flow, solid marker = feasible, hollow marker = infeasible.
