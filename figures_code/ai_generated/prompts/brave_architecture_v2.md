# BRAVE Framework Architecture Schematic v2

`brave_architecture_v2.png`

## ALTERNATIVE SIMPLIFIED PROMPT

Create a professional scientific workflow diagram for a research paper, showing ONE ITERATION of the BRAVE alloy discovery framework. The diagram should be visually rich with 2.5D stylized vector art (soft shadows, subtle gradients, rounded corners). NOT flat, NOT photorealistic.

**Overall layout:** Top-to-bottom flow, approximately 2400 × 1800 pixels, white background. Use a center-aligned vertical spine with branches left and right at the bifurcation.

**Visual elements to include:**

**1. Header banner** spanning full width: rounded rectangle with gradient fill (teal #007A7A to #009999), white text "ITERATION t", with a subtle circular arrow icon on the right suggesting iteration/loop.

**2. Synthesis block:** Large rounded box with light blue fill (#D4E4F7), dark blue border (#1E5A8C). Include a small icon of a melting crucible or furnace inside. Label: "Synthesize B Alloys (VAM)". Add small alloy ingot icons (3-4 small colored rectangles) below the label to represent the batch.

**3. Phase Verification block:** Medium box with light blue fill. Include a small XRD diffraction peaks icon (3 sharp peaks). Label: "Phase Verification (XRD)". 

**4. BIFURCATION — make this the visual centerpiece:**
- From Phase Verification, draw TWO thick diverging arrows: one going LEFT (green arrow with checkmark), one going RIGHT (red/coral arrow with X mark).
- LEFT: Green rounded box (#C8E6C9, border #2E7D32) with a checkmark icon. Label "FEASIBLE" in bold, subtitle "FCC confirmed" in smaller text.
- RIGHT: Coral rounded box (#F5E6E6, border #B71C1C) with an X icon. Label "INFEASIBLE" in bold, subtitle "Secondary phases" in smaller text. Add a "prohibition" or "stop" visual cue.
- Between the two boxes, add a dashed vertical line to emphasize the separation.

**5. LEFT branch (Feasible path) — stack of characterization steps:**
Four connected rounded boxes in light green (#C8E6C9), each with a small icon:
- Box 1: Small tensile specimen icon. "Tensile → YS, UTS/YS"
- Box 2: Small indenter tip icon. "NI + SHPB → Hdyn/Hqs"  
- Box 3: Small graph/curve icon. "CS Calibration"
- Box 4: Small FEM mesh icon. "FEM → DoP"
Connect with thin arrows between boxes.

**6. RIGHT branch (Infeasible path):**
A single faded/ghosted box with a "no data" icon (empty document with a slash). Label: "No Objective Data". This box should look visually diminished compared to the left branch — use lower opacity or lighter colors to convey that nothing useful comes from this path (except the phase label).

**7. Model Update row — two parallel boxes:**
- LEFT box: Purple (#E1D5F5, border #5E35B1) with a GP/curve icon. "Update GP Surrogates" subtitle "5 objectives"
- RIGHT box: Orange (#FFE8D1, border #C55A11) with a classifier/boundary icon. "Update GPC" subtitle "Feasibility"
- Draw arrows: Feasible path → GP Surrogates (thick green arrow), Feasible path → GPC (thin green arrow, both outcomes inform feasibility), Infeasible path → GPC (thick coral arrow).

**8. Convergence block — the EHVI × P_feas step:**
Wide rounded rectangle spanning both columns. Dark maroon (#500000) fill with white text. Mathematical notation: "EHVI(x) × P_feas(x)". Subtitle in lighter text: "n GP ensemble members". Add a subtle "merge" visual — two arrows coming from GP Surrogates and GPC converging into this box. Add a small multiplication symbol (×) between the two incoming arrow labels if possible.

**9. Selection block:**
Teal rounded box (#E6F5F5, border #007A7A) with a clustering icon (dots grouped into clusters). Label: "k-medoids → Next Batch B_{t+1}".

**10. Loop-back arrow:**
A large dashed curved arrow from the Selection block back up to the Synthesis block (on the right side of the diagram), with a small "t+1" label on the arrow.

**Design principles:**
- Each box should have soft drop shadows (2-3px offset, 50% opacity gray)
- Use rounded corners (8-12px radius) on all boxes
- Arrows should be thick (3-4px) with proper arrowheads
- The bifurcation (step 4) should be the most visually prominent element
- The convergence (step 8) should be the second most prominent
- Icons should be simple, monochrome line drawings inside each box (not photorealistic)
- Maintain consistent spacing between rows
- Keep text SHORT — 2-3 words per label maximum to prevent garbling
- The overall aesthetic should be: "publication-quality figure in a top materials science journal"

**Color palette (ARM MIP):**
- Fills: #E6F5F5, #F5E6E6, #C8E6C9, #D4E4F7, #FFE8D1, #E1D5F5
- Borders/accents: #007A7A, #500000, #2E7D32, #1E5A8C, #C55A11, #5E35B1
- Danger/infeasible: #B71C1C
- Text: #1A1A1A (dark) or #FFFFFF (on dark backgrounds)
