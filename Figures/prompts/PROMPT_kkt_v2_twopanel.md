# KKT Boundary Principle — Two Panel Version

`kkt_boundary_v2_twopanel.png`

## ALTERNATIVE SIMPLIFIED PROMPT

Create a square scientific figure (1:1 aspect ratio) with two panels stacked vertically, separated by a thin horizontal line. White background throughout.

The axes represent composition space (not property space). Label x-axis "Composition variable 1" and y-axis "Composition variable 2" subtly along the edges.

**The top half is labeled "(a) Constrained Optimum" in bold at the top.**

A simple 2D composition space. The left/lower region is light teal (#E6F5F5) labeled "Feasible (FCC)". The right/upper region is light red (#F5E6E6) labeled "Infeasible (sigma)". A single smooth curved boundary separates them (quarter-circle arc from upper-left to lower-right), drawn as a thick dark teal line (#007A7A). Label the boundary ONCE only as "Phase boundary g(x) = 0". IMPORTANT: exactly ONE boundary line, not two.

Four or five gray elliptical contour lines show an objective function (yield strength). The contour CENTER (highest YS) is in the infeasible region near the unconstrained optimum, but the contour lines EXTEND well into the feasible region — the outermost contour should reach about halfway into the feasible (teal) zone. A small arrow labeled "Increasing YS" shows the direction toward the infeasible region. Do NOT include any gray arrow pointing at the KKT star.

A hollow orange circle (#C55A11) in the infeasible region labeled "Unconstrained optimum". A filled dark teal star (#007A7A) exactly ON the boundary labeled "Constrained optimum (KKT)".

Gradient arrows: two bold arrows placed INSIDE the feasible (teal) region, away from the KKT star and unconstrained optimum — somewhere in the lower-left area of the feasible region where there is open space. The arrows originate from the same point in the feasible region and point toward the infeasible region (upper-right direction). They represent the local gradient directions of the objective and constraint. Make them different LENGTHS (nabla f shorter, nabla g longer) and at a 20-25 degree angle apart so both are clearly visible. One maroon arrow (#500000) labeled "nabla f" and one teal arrow (#007A7A) labeled "nabla g". Labels at arrow tips. The approximate parallelism shows gradient alignment throughout the space.

**The bottom half is labeled "(b) Three Strategies" in bold.**

Three small sub-panels side by side, each showing the EXACT SAME composition space with the EXACT SAME curved boundary shape as panel (a). 

CRITICAL CONSISTENCY RULES:
- Each sub-panel has exactly 20 dots total
- The hollow orange circle (unconstrained optimum) appears in ALL three sub-panels at the same location in the infeasible region
- The constrained optimum star appears in ALL three sub-panels ON the boundary at the same location, BUT with different rendering: in Hard filter and Blind BO the star is drawn as a HOLLOW outline only (empty star with teal border, no fill — showing the optimum exists but was NOT found by these strategies). In the BRAVE sub-panel the star is SOLID filled dark teal (#007A7A) — showing BRAVE is the only strategy that discovers the constrained optimum
- ALL dots in the feasible region are dark teal (#007A7A) in every panel — same color as the KKT point
- ALL dots in the infeasible region are orange (#C55A11) in every panel — same color as the unconstrained optimum
- The boundary curve is identical in all three sub-panels
- The SAME gray objective contour lines from panel (a) appear in ALL three sub-panels as faint background lines, showing this is the same optimization problem with different sampling strategies

Left sub-panel: "Hard filter" — The feasible region (lower-left, light teal) contains a gray shaded exclusion band running parallel to and on the FEASIBLE side of the teal boundary curve. This gray band represents the over-constrained buffer zone that excludes compositions near the boundary. The 20 teal dots are ALL in the light teal feasible region, clustered near the inner edge of the gray band, far from the actual phase boundary. ZERO dots in the light red infeasible region. The KKT star sits inside the gray exclusion zone on the boundary, showing it would be excluded by hard filtering. IMPORTANT: the light teal feasible region must be in the lower-left, matching panel (a) exactly. Label: "Over-constrained".

Middle sub-panel: "Blind BO" — no exclusion band. The 20 dots are clustered near the unconstrained optimum in the infeasible region. Most dots (about 12-14) are orange (infeasible), with only 6-8 teal dots in the feasible region scattered loosely. Label: "Budget wasted".

Right sub-panel: "BRAVE" — no exclusion band. A subtle teal gradient band along the boundary on the feasible side suggests the tunable exploration zone. Most dots (about 14-15) are teal and concentrated near the boundary on the feasible side. 3-4 dots are orange, just across the boundary in the infeasible region (accepted failures from boundary exploration). 1-2 teal dots in the interior. Label: "Boundary navigated".

Style: 2.5D with soft shadows, NOT photorealistic. ARM MIP palette. Minimal text, 2-3 words per label. Publication quality. No decorative elements. Clean sans-serif font. IMPORTANT: do NOT include any title text above the figure — no "KKT Boundary Principle" or similar heading. The figure starts directly with panel (a).
