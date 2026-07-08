# Graphical Abstract v4 — BRAVE Campaign 2 (Acta Materialia)

`graphical_abstract_v4.png`

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
7. NO TOP-LEVEL FIGURE TITLE. Do NOT add any header text like
   "Graphical abstract", "Acta Materialia", "BRAVE Campaign 2", or any
   overall title above the panels. The three panel-header capsules are
   the ONLY headline text in the figure. The area above the panels
   must be empty background.

## LAYOUT

Three horizontal panels linked by two bold rightward decision-flow
arrows. Panel B (middle) is ~40% of the width; Panels A and C are ~30%
each. Each panel has a small header label at top-left in a colored
capsule.

### Panel A (Left) — "8-Element Design Space"

MATCH THE VISUAL STYLE OF v2 EXACTLY for this panel — prominent
in-triangle text labels for Feasible Region, Infeasible Sigma-phase
Region, and Boundary optima with a leader line; large periodic-strip
cells at top; V↑ arrow inside the triangle; CALPHAD inset in the
bottom-right corner of the panel (NOT bottom-left).

Header capsule: teal (#007A7A).

Domain content, all embedded, laid out in this order:

1. Periodic-element strip along the TOP of the panel, above the
   composition triangle. Eight rounded rectangular cells side by side,
   each with the element symbol in large text: Al, V, Cr, Mn, Fe, Co,
   Ni, Cu. Color each cell by group: Al warm yellow (#FFE8D1), Cu warm
   orange (#F5D5B8), and the six transition metals (V, Cr, Mn, Fe, Co,
   Ni) muted blue-gray (#D4DDE7). Cells prominent and readable.

2. A large stylized ternary/barycentric composition triangle occupying
   the bulk of the panel below the periodic strip. Pastel teal
   (#E6F5F5) fill for the FCC-feasible region (majority, lower-left
   two-thirds) and pastel red (#F5E6E6) fill for the infeasible
   sigma-phase region (upper-right one-third), separated by a smoothly
   curved dark-teal (#007A7A) phase-stability boundary line with
   slight thickness variation.

3. Prominent in-triangle text labels (readable at TOC size):
   - "Feasible Region" in the lower-left teal area (dark-teal text,
     medium weight).
   - "Infeasible Sigma-phase Region" in the upper-right pastel-red
     area (dark-maroon text, medium weight).
   - "Boundary optima" positioned near the boundary line with a thin
     dark-teal leader line pointing to one of the highlighted boundary
     circles.

4. Scatter ~20 tiny composition points on the diagram: small solid
   teal circles inside feasible region, small hollow maroon (#500000)
   circles inside infeasible region, and 3 larger teal-outlined
   circles ON the boundary line (these are the "Boundary optima").

5. A small "V ↑" arrow inside the triangle, positioned in the middle
   of the panel, pointing from the feasible region toward the
   infeasible sigma-phase corner.

6. CALPHAD phase-diagram inset in the BOTTOM-RIGHT corner of the
   panel, sized about 22% of the panel width. This is a MINIATURE
   BINARY ISOPLETH (temperature vs. composition slice) and MUST be
   drawn as a correct phase diagram, NOT an abstract sketch.

   Required structure of the inset:
   - Rectangular plot with two thin dark-gray axes: horizontal x-axis
     labeled "at.% V" with ticks at 0 (left) and 30 (right);
     vertical y-axis labeled "T (°C)" with ticks near 800 (bottom)
     and 1400 (top).
   - Exactly TWO phase fields — nothing else:
     * UPPER field (approximately upper 55–60% of the plot area,
       tapering downward and to the right): filled pastel teal
       (#E6F5F5), labeled "FCC" in small dark-teal text. This is the
       single-phase FCC domain at high T.
     * LOWER-RIGHT field (approximately lower 40–45% of the plot area,
       expanding rightward): filled pastel red (#F5E6E6), labeled
       "FCC + σ" in small dark-maroon text. This is the two-phase
       field appearing at lower T and higher V.
   - A SINGLE smooth dark-teal solvus curve separating them: starts
     near the upper-left (low V, high T), curves gently downward and
     rightward, ends near the lower-right (high V, mid-low T). The
     curve is monotonic (always sloping down as V increases). No
     horizontal invariant lines, no vertical lines, no branches or
     intersections.
   - Explicitly DO NOT draw: Gibbs-energy convex parabolas, common
     tangents, ternary Gibbs triangles, multiple invariant reactions,
     additional phases beyond FCC and FCC+σ, or any decorative
     shading gradient.
   - Tiny caption above the inset: "CALPHAD isopleth" in small dark
     text.

7. Bottom-of-panel micro-caption text (below the triangle, outside
   any element): "27,240 CALPHAD-feasible candidates".

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

Create a wide 16:9 landscape publication-quality scientific graphical abstract on risk-aware Bayesian optimization of high-entropy alloys. Three horizontal panels linked by bold rightward arrows on a very faint blueprint-grid background (#F7F8FA with #E5E7EB grid). NO top-level title anywhere; do NOT write "Graphical abstract", "Acta Materialia", "BRAVE Campaign 2" or any figure title above the panels — the three panel-header capsules are the ONLY headline text; the area above the panels is empty background.

**Panel A (left, 30%) — "8-Element Design Space".** Match the v2 layout: prominent in-triangle labels for regions and boundary optima. Large periodic-strip cells at TOP listing Al (warm yellow), V, Cr, Mn, Fe, Co, Ni (blue-gray) and Cu (warm orange). Below the strip, a large stylized ternary composition triangle with pastel-teal (#E6F5F5) FCC-feasible region (lower-left two-thirds) and pastel-red (#F5E6E6) infeasible sigma-phase region (upper-right one-third) separated by a curved dark-teal (#007A7A) phase-stability boundary. Prominent in-triangle text labels: "Feasible Region" in lower-left teal area, "Infeasible Sigma-phase Region" in upper-right pastel-red area, and "Boundary optima" with a leader line pointing to a highlighted boundary circle. Scatter ~20 composition points: solid teal in feasible region, hollow maroon in infeasible region, 3 larger teal-outline circles ON the boundary. A "V ↑" arrow inside the triangle pointing toward the infeasible corner. In the BOTTOM-RIGHT corner of the panel, a miniature CALPHAD isopleth inset: rectangular T (°C) vs at.% V plot, x-axis 0–30 at.% V, y-axis 800–1400 °C, ONLY two phase fields — pastel-teal upper field labeled "FCC" occupying ~55% of the plot area (single-phase FCC domain at high T) and pastel-red lower-right field labeled "FCC + σ" occupying ~45% of the plot area (two-phase field appearing at lower T and higher V) — separated by a SINGLE smooth monotonically decreasing dark-teal solvus curve from upper-left to lower-right. NO Gibbs-energy convex curves, NO ternary Gibbs triangles, NO horizontal invariant lines, NO extra phases beyond FCC and FCC+σ. Tiny caption above the inset: "CALPHAD isopleth". Caption below the panel: "27,240 CALPHAD-feasible candidates".

**Panel B (center, 40%, largest) — "Risk-Aware Bayesian Optimization".** Central dark-teal hexagon labeled "Risk-Weighted Acquisition". Four rounded module boxes arranged around it clockwise: (top, pastel green) "GP Objective Surrogates" with 5 stacked mini-GP-posterior plots and chip "5 objectives"; (right, pastel red) "Feasibility Classifier" with a soft probability mask over a composition slice; (bottom, pastel orange) "Diversity-Aware Batch" with k-medoids cluster icons and chip "16/iter"; (left, pastel blue) "Experiment + Simulation" with isometric icons of arc-melting crucible, nanoindentation tip, and impact-simulation projectile. Thick clockwise decision-flow arrows between hexagon and modules; thin double-headed information-flow arrows labeled μ, σ, p_feas, batch, y_obs. Caption: "Feasibility-penalized EHVI on the Pareto frontier".

**Panel C (right, 30%) — "Boundary-Adjacent Optima".** Three stacked mini-visualizations: (top) V–Ni composition plane with YS heatmap and 3 stars marking BBC04/BBC02/BBB01 with "V = 24 at.%" callout; (middle) compact stress-strain curve showing dark-teal "BRAVE alloy (BBC04)" reaching ~1480 MPa at ~50% strain with translucent teal fill under curve and callout "UTS/YS > 4", vs. light-gray "Reference HEA" at ~800 MPa; (bottom) KKT inset with elliptical objective contours, an orange arrow from an "unconstrained optimum" to a teal "KKT-active optimum" dot sitting on a boundary line, labeled "Best alloys at constraint boundary". Caption: "1480 MPa UTS, 50% elongation, single-phase FCC".

Connectors: thick teal (#007A7A) arrow A→B labeled "sample"; thick maroon (#500000) arrow B→C labeled "measure + learn"; thin dashed feedback loop from C back to B labeled "y, phase, DoP".

Icon rules (strict): CALPHAD = mini phase diagram not a database icon; feasibility = probability mask not a lock/shield; batch = grouped clusters with medoids not generic boxes; experiment = domain-correct crucible/indenter/impact glyphs not generic test tubes; property landscape = genuine V–Ni heatmap not a gradient blob; stress-strain = correct elastic-plastic-hardening curve shape.

Style: 2.5D stylized vector infographic with soft shadows, isometric feel where useful, publication-quality, high visual density but generous whitespace, no photorealism, no cartoon styling, no decorative frames, no equations, sans-serif labels 2-6 words each. Bottom-right compact legend: thick arrow = decision flow, thin arrow = information flow, solid marker = feasible, hollow marker = infeasible.
