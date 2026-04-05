# FORMA — Project Context & Decision Record
**Version:** 1.0  
**Purpose:** Provides the business thesis, founder constraints, competitive reasoning,
and architectural decision history that shaped the technical specs. Any reviewer should
read this document first before evaluating the spec files.

---

## The One-Sentence Pitch

Forma is a physics-based garment fit engine that lets online shoppers see exactly
how clothing will fit their body before they buy — not through statistical guessing,
but by running real cloth simulation on a 3D twin of the user.

---

## The Problem

Online apparel has a return rate of 20–30%, costing retailers over $100 billion
annually. The root cause is that shoppers cannot predict how a garment will fit
their specific body. Existing "fit recommendation" tools (True Fit, Virtusize, etc.)
use statistical models trained on purchase/return data — they correlate, they don't
measure. When a size chart says "chest: 96–100cm," it tells you nothing about how
a garment actually drapes on your particular shoulders, torso, and waist.

The insight behind Forma is that garment fit is a physics problem, not a statistics
problem. A T-shirt on a body obeys cloth mechanics: stretch stiffness, gravity,
collision. If you simulate that physics accurately, you can produce a millimeter-level
verdict on whether the garment fits — not a probability, a measurement.

**The agentic commerce gap.** A new class of AI shopping agents (Citrini, personal
shopping copilots, agentic commerce platforms) is emerging to reduce friction in
online shopping. These agents optimize for purchase conversion — finding products,
comparing prices, reducing clicks to checkout. But purchase conversion is a vanity
metric when 25% of apparel purchases come back. The agent looks smart at the moment
of sale and generates a return two weeks later. The retailer eats $15–30 in reverse
logistics, the consumer wastes time boxing it up, and nobody's happy.

Current agentic shopping is purely aesthetic- and price-based. It reduces *upfront*
friction (finding and paying) but completely ignores *downstream* cost (returns).
No existing agent can answer the question "will this garment actually fit this
person's body?" because that requires physics, not language modeling. This is
the gap Forma fills.

---

## Why Forma Wins

Every existing fit-tech company and shopping agent uses one of two approaches:

1. **Size chart matching** — map user measurements to brand size charts.
   Fails because size charts are inconsistent across brands and don't capture
   garment geometry.

2. **Statistical models** — train on purchase/return data to predict fit.
   Fails because correlation isn't causation, data is sparse for new products,
   and it can't explain *why* something doesn't fit (tight in chest? long in
   torso? loose in shoulders?).

Forma's approach is different: take the actual garment pattern (sewing panels,
seam layout, fabric properties), drape it on the user's actual body mesh using
XPBD cloth simulation, and measure the physical clearance between garment and
body at every region. The output isn't "you might like size M" — it's "size M
has 12mm of ease at the chest, 24mm at the shoulders, and is 8mm tight at the
waist."

This per-region measurement is what the architecture calls the `fit_verdict` —
a structured JSON output with signed clearance values in millimeters. It's
deterministic, explainable, and physically grounded.

**AI agents can't build this themselves.** Physics simulation of cloth on a body
mesh is a specialized pipeline — it's not something you bolt onto an LLM with a
tool call. Agents need Forma's API. This means Forma isn't competing with agentic
commerce platforms — Forma is infrastructure they depend on. That's the strongest
competitive position to be in.

---

## The Product — Two Layers

### Layer 1: Fit Verification API (fastest path to revenue)

The core product is a single API endpoint:

```
POST /verify
  body:    { body_profile_id, garment_id }
  returns: { fit: true/false, strain_map: [...], ease_map: [...], confidence: 1.0 }
```

This endpoint is what the entire Phase 1 physics engine builds toward. The moment
it returns a real verdict, it can be sold to three types of customers:

**AI shopping agents** call POST /verify before recommending a garment to their
user. One API call turns a guess into a physics-backed verdict. The agent's
return rate drops, its retailers are happier, and its users trust it more. The
pitch is dead simple: "Your agent already finds products and compares prices.
Add one API call to Forma before you recommend a garment, and your return rate
drops by 40–60%."

**Retailers** integrate the API into their product pages — a "Verified Fit" badge,
a per-region fit breakdown, or a "try before you buy" simulation widget. Retailers
pay because returns are their largest controllable cost in e-commerce.

**Other platforms** — Shopify apps, fashion marketplaces, personal styling services —
embed Forma's fit intelligence into their existing workflows.

### Layer 2: Consumer Shopping Interface (parallel workstream)

The longer-term consumer product is a shopping interface where users:

1. **Create a 3D body twin** — initially from photos (Phase 2) or standard size
   charts (Phase 1 uses synthetic mannequins), eventually from a precision
   body-scanning garment (long-term hardware roadmap).

2. **Browse and customize garments** — drag and drop garment types onto their
   mannequin, adjust fit preferences (slim, relaxed, oversized), specify materials,
   describe what they want via an LLM-powered "describe the garment" input.

3. **Get physics-verified recommendations** — Forma's engine simulates the actual
   garment on the user's body and surfaces real products from partner retailers
   that match the user's specified fit and style criteria.

The consumer UI is a near-term parallel workstream. It runs on the same physics
engine as the API — the only difference is the interface layer.

---

## Revenue Model

Three revenue streams, ordered by expected time-to-revenue:

**1. Fit Verification API (fastest).** Per-call or tiered usage pricing for AI
shopping agents and platforms that integrate Forma's POST /verify endpoint. This
is the same endpoint the physics engine produces in Week 4 of Phase 1 — no extra
product to build. Revenue starts the moment agent developers integrate.

**2. Retailer commission.** When a user finds and purchases a garment through
Forma's consumer interface, the partnering retailer pays a commission on the sale.
Requires the consumer shopping UI to be built (Layer 2).

**3. Consumer tiers.** Free tier with limited simulations per month. Pro tier
(paid subscription) with unlimited simulations, advanced customization tools,
saved fit profiles across retailers. Requires Layer 2.

The strategic insight: **revenue stream #1 requires only the physics engine and
an API layer — both of which are already in the Phase 1 build plan.** Streams #2
and #3 require the consumer UI, which is a larger effort. The API-first approach
means Forma can generate revenue and prove market demand while the consumer
product is still being built.

---

## Competitive Positioning

The fit-tech space is active and well-funded. The key players, as of early 2026:

**True Fit** is the most direct comparison and the current market leader. They
launched an agentic AI shopping agent in February 2026, built on 20 years of
purchase and return data covering $616 billion in transactions and 91,000 brands.
Their approach is purely statistical — they predict what size you'll keep based on
what millions of similar shoppers kept. They also offer their Fit Intelligence via
MCP, positioning themselves as "fit infrastructure" for other AI agents. True Fit
is Forma's clearest competitor and validates the market (up to 70% of questions
to AI shopping agents are about fit and sizing, per their own data). The critical
difference: True Fit correlates past outcomes. Forma simulates physics. True Fit
can tell you "shoppers like you kept size M." Forma can tell you "size M has 12mm
of ease at your chest and is 8mm tight at your waist." One is a prediction based
on population data. The other is a measurement of a specific garment on a specific
body. True Fit cannot explain *why* something doesn't fit or handle garments with
no purchase history.

**3DLook** is the leading mobile body scanning company. Their technology generates
3D body models and 80+ measurements from two smartphone photos (96–97% accuracy
per IEEE certification). They serve fashion, fitness, and made-to-measure verticals.
3DLook is a potential *complement* to Forma, not a competitor — their scanning
technology could feed body meshes into Forma's simulation pipeline. They do not
simulate cloth physics.

**Bold Metrics** uses AI to predict body measurements from minimal input (height,
weight, age, fit preference) and matches those to garment specifications. Like
True Fit, this is a statistical approach — it predicts measurements rather than
simulating fit.

**Fit Collective** raised £3M in late 2025 to help brands fix sizing at the
production stage. They analyze returns, fabric behavior, and sales data before
garments are produced. Their focus is upstream (design and manufacturing), not
downstream (consumer shopping). Different problem, different customer.

**Reactive Reality** and **Vybe** focus on visual try-on — AR overlays that show
how a garment *looks* on you, not whether it *fits*. They solve the aesthetic
question ("does this style suit me?") but not the dimensional question ("will
this be tight across my shoulders?").

**Kleep** and **Mirrorsize** offer smartphone-based body scanning with AI-powered
size recommendations. Similar to 3DLook — body measurement tools, not garment
simulation.

**No one in this space simulates actual cloth physics on a per-user body mesh.**
Every existing player either correlates statistical data (True Fit, Bold Metrics),
scans the body without simulating the garment (3DLook, Kleep), or visualizes
aesthetics without dimensional accuracy (Reactive Reality, Vybe). Forma is the
only approach that takes the actual garment pattern, drapes it on the actual body
using XPBD simulation, and produces a millimeter-level fit measurement. This is a
fundamentally different technical foundation — and it's the only one that can
produce the ground truth labels that statistical models need to train on.

---

## The Data Flywheel

Every call to POST /verify generates a structured, labeled data point:

```
(body_mesh, garment_pattern, fabric_params) → fit_verdict
```

This tuple contains the body geometry, the garment geometry, the material
properties, and the simulation outcome (per-region clearance in millimeters,
fit/no-fit, severity classification). Over time, this accumulates into a dataset
of body-garment-fit relationships that no other company possesses — because no
other company runs the physics simulation that produces them.

This dataset enables three things:

**1. ML fast-path.** Train an approximate model on simulation results to return
instant fit predictions (~50ms) for common body-garment pairs, falling back to
full XPBD simulation (~3–10s) for novel combinations. The physics engine produces
the training labels; the ML model provides the speed. This is how latency drops
from seconds to milliseconds at scale without sacrificing accuracy for novel
garments.

**2. Better recommendations.** With enough (body, garment, outcome) triples,
Forma can predict which garments from a catalog will fit a given body *without*
simulating each one — using the simulation corpus as ground truth for a
recommendation model. This is what makes the consumer shopping UI powerful:
instead of simulating 10,000 garments, simulate 200, train a model, and rank
the rest.

**3. Garment design feedback.** Aggregate fit outcomes across many body types
to show brands where their garments systematically fail — "your size M is tight
at the shoulders for 40% of male bodies in the 95th percentile chest range."
This is the Fit Collective use case (upstream design feedback), powered by
Forma's simulation data rather than return statistics.

The flywheel: more API calls → more simulation data → better ML models → faster
responses → more API calls. The physics engine is the moat because it generates
ground truth that statistical-only competitors cannot produce.

---

## Current Development Phase

Forma is in **Phase 1, pre-Week 1 — planning complete, no code written yet.**

The physics engine is being built first because it is the core technical risk.
The consumer shopping UI is a near-term parallel workstream but depends on the
engine producing trustworthy fit verdicts before it can surface real recommendations.

### Phase 1 Timeline (4 weeks)

| Week | Focus | Deliverable |
|------|-------|-------------|
| Week 1 | XPBD smoke test | 3 fit verdicts (S/M/XL on size M body) — go/no-go gate |
| Week 2 | Physics hardening | Extended sizes, second garment type, fabric sensitivity |
| Week 3 | Archivist (Supabase) | Persistent storage for body profiles, garments, verdicts |
| Week 4 | REST API (Vercel) | POST /verify live — first demo-ready endpoint |

### Beyond Phase 1

| Phase | Timeline | Focus |
|-------|----------|-------|
| Phase 2 (Weeks 5–10) | Real body scanning from photos, expanded garment library, Shopify integration POC |
| Phase 3 (Weeks 11–18) | Brand DXF ingestion, batch processing, shopping agent API, first retailer contract |

The shopping UI (mannequin customization, drag-and-drop garment tools, LLM
garment description, retailer recommendation surface) will begin development
alongside the physics engine, with the two converging when the engine can
produce reliable verdicts on real garments and real body scans.

---

## Founder Context

**Solo founder.** Finance undergraduate at San Diego State University, graduating
mid-May 2026. Begins full-time employment in early September 2026.

**Available build window:** Approximately 5 months of maximum focus (mid-May
through August 2026) between graduation and start of full-time work. Before
May, Forma is a side project with limited hours. After September, it becomes
an evenings-and-weekends effort.

**Technical approach:** The founder is not a software engineer by training.
The build strategy uses a two-layer AI orchestration model:

- **Layer 1 (Strategist):** Claude via claude.ai for architecture decisions,
  spec writing, problem diagnosis, and planning. Every major decision is resolved
  here as a written spec before any code is written.

- **Layer 2 (Executor):** Claude Code CLI for implementation. Takes spec
  documents from Layer 1 and produces committed, tested code. Can run
  semi-autonomously while the founder works on other tasks.

The founder's primary technical skill is spec writing — defining precise input/output
contracts, acceptance criteria, and test cases that the executor layer can implement
without ambiguity. The daily rhythm is: morning spec work (30 min), daytime
autonomous execution, evening review and redirect (30–60 min).

**No funding.** Operating costs for Phase 1 are approximately $45/month
(Supabase free tier, Google Colab student access, Anthropic API, Vercel free tier).

**No retailer conversations yet.** Outreach planned for when POST /verify
returns a real fit verdict — even on a synthetic mannequin, the demo shows
"size M fits a size M body, size S doesn't," which is exactly the right
proof point for retailers.

---

## Key Architectural Decisions and Why

### 1. Synthetic mannequin first, real body scanning later

**Decision:** Phase 1 uses a single SMPL-X generated body mesh (mathematically
perfect, confidence 1.0) instead of building the body scanning pipeline.

**Why:** Real body scanning from consumer photos (2-photo mesh fusion from
uncalibrated cameras) is the single biggest technical unknown in the entire
architecture. Building it first would block the physics pipeline for weeks on
an unsolved research problem. By using a synthetic mannequin, any error in
simulation output is the physics engine's fault — not scanning noise. The body
scanning pipeline drops in later with zero downstream changes (same .ply mesh
format, same body_profile.json schema, different `body_source` enum value).

**What was considered:** Building the Sculptor (body scanner) first, then
validating physics. Rejected because it couples two unknowns — if the fit
verdict is wrong, you can't tell whether it's a scanning bug or a physics bug.

### 2. pygarment + GarmentCode instead of custom pattern blocks

**Decision:** Use ETH Zurich's GarmentCode system (115,000 parametric patterns
via `pip install pygarment`) instead of building custom Master Parametric Blocks.

**Why:** Building pattern generation from scratch would take 2–3 weeks of
Phase 1 for a capability that already exists in a peer-reviewed, MIT-licensed
library. GarmentCode patterns are already simulation-ready with seam correctness
enforced by construction.

**What was considered:** Custom parametric blocks would give more control over
pattern geometry. Rejected for Phase 1 — pygarment gets us to the smoke test
in days instead of weeks.

### 3. NvidiaWarp-GarmentCode fork — BLOCKED, mainline Warp instead

**Original Decision:** Use the ETH Zurich fork of NVIDIA Warp for cloth simulation.

**What Changed (March 2026):** License audit confirmed the ETH fork
(NvidiaWarp-GarmentCode) is under NVSCL — non-commercial use only. The fork
is pinned to Warp v1.0.0-beta.6 and has not relicensed. This blocks commercial
use by Forma.

**Revised Decision:** Use mainline NVIDIA Warp (Apache 2.0, v1.12.0+) which
includes built-in `wp.sim.XPBDIntegrator` and `wp.sim.VBDIntegrator`. Garment-
specific features from the fork (body-part collision, attachment constraints,
self-collision) must be reimplemented in-house under a strict Clean Room Protocol
using only published academic papers — never referencing the fork's source code.
See FORMA_PHASE2_EXECUTOR_SPEC.md and CLAUDE.md for the full clean room rules.

### 4. trimesh + scipy instead of pytorch3d

**Decision:** Use lightweight mesh libraries for Phase 1 geometry operations
(KDTree clearance computation, mesh I/O, convex hull measurements) instead of
pytorch3d.

**Why:** pytorch3d is heavy, difficult to install (especially on Apple Silicon),
and requires PyTorch as a dependency. For Phase 1, we only need nearest-point
queries and basic mesh operations — trimesh and scipy handle this without the
installation complexity. pytorch3d is deferred to Phase 2 for mesh fusion.

### 5. Energy-based convergence instead of vertex-movement-only

**Decision:** Primary convergence criterion is total kinetic energy dropping
below a threshold, with vertex movement as a secondary check.

**Why:** The original spec used only vertex movement (fewer than 1.5% of
vertices moving more than 0.4mm per frame). In practice, XPBD cloth simulation
has a long tail where wrinkles keep shuffling even though global fit has
stabilized. Energy-based convergence is more robust and also catches cloth
explosion automatically (energy spike detection).

### 6. Week 2 is physics hardening, not Supabase

**Decision:** Moved Supabase integration from Week 2 to Week 3. Week 2 is now
physics hardening: extended size range, second garment type, fabric sensitivity.

**Why:** Supabase is trivial infrastructure with zero technical risk. The physics
pipeline is the technical risk. Testing it against edge cases (monotonic size
relationship, different garment topology, different material parameters) before
building infrastructure on top prevents building on a broken foundation.

### 7. Clearance, not strain

**Decision:** The `strain_map` field in fit_verdict.json measures signed
garment-to-body surface distance (clearance), not mechanical fabric strain.

**Why:** XPBD simulation outputs a deformed cloth mesh. "Strain" in physics
means triangle deformation ratio (dimensionless). What Forma measures is the
gap between the garment and the body — positive means loose, negative means
the garment is compressed against the body. The field is still named `strain_map`
for schema stability, but all documentation clarifies it as signed clearance
in millimeters.

### 8. Phase 2 GPU transition: vertex scaling before solver swap

**Decision (March 2026):** Phase 2 sequences the GPU migration as: (1) Warp
parity proof at 22K vertices, (2) vertex scaling to 1M before optimizing
subsystems, (3) collision + solver upgrades driven by profiling data, (4)
precompute matrix at full fidelity.

**Why:** Forma's value proposition is near-fidelity virtual try-on, which
requires million-vertex mesh density. Optimizing subsystems (collision, solver)
at 22K vertices would solve the wrong problems — bottlenecks are different at
scale. By pushing vertex count early, real bottlenecks reveal themselves and
drive targeted optimization. This is a "measure then fix" approach rather than
"guess and optimize."

**Key constraint:** All tech must be commercially licensable or built in-house.
NVIDIA Warp mainline (Apache 2.0) is the GPU foundation. The license-blocked
GarmentCode-Warp fork's features are reimplemented under a Clean Room Protocol.
See forma_phase2_roadmap_v2.docx for the full strategic roadmap.

---

## Spec File Index

Read these in order for full technical understanding:

| File | What It Covers |
|------|---------------|
| `FORMA_CONTEXT.md` | This document — business thesis, founder constraints, decision history |
| `forma-architecture-v1_2_1.html` | Visual architecture document — pipeline diagram, agent roster, risk register, timeline |
| `FORMA_WEEK1_SPEC.md` | Week 1 acceptance criteria, output schemas, test suite, run commands |
| `FORMA_GEOMETER_SPEC.md` | Geometry pipeline implementation — beta solver, region segmentation, clearance computation |
| `FORMA_SEAM_MANIFEST_SCHEMA.md` | Seam manifest JSON schema — contract between Tailor and Geometer |
| `FORMA_FABRIC_LIBRARY.md` | XPBD material parameters — replaces the original underspecified fabric params |
| `FORMA_WEEK2_SPEC_PHYSICS.md` | Week 2 physics hardening — extended sizes, second garment, fabric sensitivity |
| `FORMA_WEEK3_SPEC_SUPABASE.md` | Week 3 Supabase integration — body profiles, garment ledger, verdict persistence |
| `FORMA_Week1_A100_Validation.ipynb` | Google Colab notebook for final GPU validation of Week 1 smoke test |

---

## What Is Not Yet Specified

The following are known gaps that will need specs before their respective phases:

- **Shopping UI / consumer interface** — garment drag-and-drop, mannequin
  customization, LLM garment description, retailer recommendation surface.
  Near-term parallel workstream, spec not yet written.
- **AI Agent API tier** — pricing model (per-call vs tiered plans), rate
  limiting, authentication, documentation for third-party AI shopping agent
  integration. Same POST /verify endpoint, different access/billing layer.
- **Week 4 REST API spec** — POST /verify, GET /verdict, GET /garments on Vercel.
- **Phase 2 Sculptor** — photo-to-mesh pipeline (SAM 3D Body, confidence formula).
- **Phase 2 expanded garment library** — 5 garment categories, fabric library expansion.
- **Phase 3 brand DXF ingestion** — real factory patterns via Tailor + Gemini Flash.
- **Phase 3 Shopify integration** — "Verified Fit" badge on product pages.
- **Retailer commission model** — pricing, integration contracts, onboarding flow.

---

*FORMA Project Context v1.1 — March 2026 (updated for Phase 2 GPU transition)*
