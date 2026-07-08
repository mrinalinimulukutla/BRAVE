# BRAVE Framework Architecture Schematic

`brave_architecture.png`

## Purpose
Schematic showing the data flow architecture of one iteration of the BRAVE (Bayesian Risk-Aware Alloy Discovery and Exploration) campaign. This figure accompanies Algorithm 1 in the paper and should be placed in the Computational Design Framework section (Section 2.6).

## Canvas
- Dimensions: 2400 × 1600 pixels (landscape, 3:2 aspect ratio)
- Background: white (#FFFFFF)
- Style: 2.5D stylized vector art with soft shadows. NOT photorealistic, NOT flat.

## ALTERNATIVE SIMPLIFIED PROMPT

Create a scientific workflow diagram showing ONE ITERATION of the BRAVE alloy discovery framework. The diagram flows top-to-bottom with a clear bifurcation in the middle.

**Layout (top to bottom):**

**Top bar:** A horizontal rounded rectangle labeled "Iteration t" spanning the full width, colored light teal (#E6F5F5) with dark teal border (#007A7A).

**Row 1 (Synthesis):** A single box "Synthesize B alloys (VAM)" in light blue (#D4E4F7) with dark blue border (#1E5A8C). Arrow pointing down.

**Row 2 (Verification):** A single box "Phase Verification (XRD)" in light blue (#D4E4F7). From this box, TWO arrows split left and right:

**Row 3 (Bifurcation) — THIS IS THE KEY VISUAL:**
- LEFT branch: Green box (#C8E6C9, border #2E7D32) labeled "Feasible" with subtitle "FCC confirmed"
- RIGHT branch: Red/coral box (#F5E6E6, border #500000) labeled "Infeasible" with subtitle "Secondary phases"

**Row 4 (Processing):**
- LEFT (under Feasible): Stack of 4 small boxes in light green:
  - "Tensile → YS, UTS/YS, strain"
  - "NI + HSRNI + SHPB → Hdyn/Hqs"
  - "Calibrate CS model"
  - "FEM → DoP"
- RIGHT (under Infeasible): A single box with an X or stop symbol: "No objective data"

**Row 5 (Model Updates) — TWO PARALLEL PATHS converging:**
- LEFT: Purple box (#E1D5F5, border #5E35B1) labeled "Update GP Surrogates" with subtitle "5 objective models"
- RIGHT: Orange box (#FFE8D1, border #C55A11) labeled "Update GPC Classifier" with subtitle "Feasibility model"
- Arrow from Feasible stack → GP Surrogates
- Arrow from Infeasible → GPC Classifier
- Arrow from Feasible stack → GPC Classifier (feasible outcomes also update GPC)

**Row 6 (Acquisition) — CONVERGENCE POINT:**
- A wide box spanning both columns, maroon (#500000) with white text: "EHVI(x) × P_feas(x)" with subtitle "n GP ensemble members"
- Arrows from both GP Surrogates AND GPC Classifier converge into this box

**Row 7 (Selection):**
- A box in light teal (#E6F5F5): "k-medoids → B_{t+1} candidates"
- Arrow loops back up to Row 1 (next iteration)

**Key design principles:**
- The bifurcation at Row 3 is the central visual feature — make it prominent with diverging arrows
- The convergence at Row 6 is the second key feature — both paths feed into one acquisition step
- Use consistent arrow styles: solid black arrows for data flow, dashed gray arrow for the loop-back
- Keep ALL text SHORT (2-3 words per label max, subtitles in smaller font)
- Add soft drop shadows behind each box for the 2.5D effect
- No decorative elements — every visual element carries information

**Color palette (ARM MIP pastel):**
- Light fills: #E6F5F5 (teal), #F5E6E6 (coral), #C8E6C9 (green), #D4E4F7 (blue), #FFE8D1 (orange), #E1D5F5 (purple)
- Dark accents/borders: #007A7A (teal), #500000 (maroon), #2E7D32 (green), #1E5A8C (blue), #C55A11 (orange), #5E35B1 (purple)
- Text: #1A1A1A (near-black) for labels, white for the maroon acquisition box

**Text to render (keep short to prevent Gemini garbling):**
- "Iteration t"
- "Synthesize"
- "Phase Check"
- "Feasible" / "Infeasible"
- "Tensile"
- "NI + SHPB"
- "CS Calibration"
- "FEM → DoP"
- "No Data"
- "GP Surrogates"
- "GPC Classifier"
- "EHVI × P_feas"
- "k-medoids → Next Batch"
