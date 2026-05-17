# Piccolo-Spritesheet

> A latent world model specialized for pixel-art sprite animation generation, prediction, editing, and controllable motion synthesis.

---

# Vision

Modern diffusion and video models are optimized for photorealistic continuous imagery.

Pixel art is fundamentally different.

Pixel art animation is:

* discrete
* symbolic
* low-resolution
* temporally structured
* silhouette-constrained
* palette-constrained
* intentionally compressed

A human pixel artist does not think in pixels alone.

They think in:

* silhouette
* anticipation
* impact
* contact points
* motion arcs
* readability
* sprite identity
* animation phase
* gameplay semantics

This project explores whether a specialized latent world model can learn these dynamics directly.

The goal is not “generate pretty images”.

The goal is:

> Learn a compact predictive dynamics model for sprite motion.

---

# Core Thesis

Pixel sprite animation is closer to a tiny symbolic world simulator than to unconstrained image generation.

Unlike natural video:

* motion space is small
* state transitions are discrete
* actions are enumerable
* visual entropy is low
* animation rules are highly reusable

This makes sprite animation a uniquely suitable domain for:

* latent world models
* predictive representations
* action-conditioned dynamics
* compact JEPA-style training
* real-time inference on consumer hardware

---

# Inspiration

## LeCun / JEPA / World Models

Inspired by:

* V-JEPA
* LeWorldModel
* latent predictive architectures
* predictive representation learning

Key idea:

* predict future latent states
* not raw pixels

The model should learn:

* motion invariants
* temporal consistency
* sprite identity persistence
* controllable animation dynamics

rather than:

* exact pixel reconstruction

---

## SANA World Model

Inspired by:

* compact world simulation
* recursive latent prediction
* scalable simulation architectures

Important takeaway:

* intelligence may emerge from predictive latent dynamics
* not necessarily from autoregressive language

Potential adaptation:

* sprite latent simulation
* game-like state transitions
* low-resolution temporal modeling
* compressed animation physics

---

## Hunyuan World Model

Inspired by:

* structured video understanding
* long temporal coherence
* controllable visual generation

Important takeaway:

* action-conditioned temporal generation
* consistent identity across sequences
* hierarchical motion structure

Potential adaptation:

* animation phase control
* attack/walk/jump state machines
* controllable motion tokens
* direction-aware generation

---

# Problem Statement

Current image/video models struggle with pixel art because:

* pixels drift
* palettes mutate
* silhouettes collapse
* temporal consistency is weak
* identity preservation fails
* motion timing is poor
* animations lack intentionality

Current workflows:

* require manual cleanup
* are non-iterative
* are not gameplay-aware
* do not understand sprite semantics

We need:

* a specialized motion-native architecture
* not merely a style LoRA

---

# Long-Term Goal

Create a foundation model for:

* sprite animation
* game asset generation
* controllable motion synthesis
* procedural game worlds
* agentic sprite editing

Potential future capabilities:

* generate walk cycles
* generate attack combos
* interpolate animations
* pose transfer
* idle animation synthesis
* palette-aware editing
* animation repair
* gameplay-aware motion generation
* infinite procedural 2D game simulation

---

# Research Direction

## Phase 1 — Sprite Dynamics Model

Learn:

```text
frame_t + action + phase
→ frame_t+1
```

Focus:

* next-frame prediction
* short-loop prediction
* identity preservation
* palette consistency

---

## Phase 2 — Latent World Model

Learn:

```text
latent_t + control
→ latent_t+1
```

Add:

* latent planning
* temporal memory
* motion compression
* hierarchical representations

---

## Phase 3 — Action-Conditioned Simulation

Learn:

```text
character_state
+ environment_state
+ player_action
→ future sprite sequence
```

Possible applications:

* NPC behavior
* procedural animation
* game prototyping
* agent-controlled worlds

---

# Architectural Ideas

## Representation Encoder

Encode sprite frames into compact latent states.

Possible backbones:

* ViT
* ConvNet
* VQ-VAE
* tokenizer-based latent compression

Desired properties:

* preserve silhouette
* preserve palette structure
* preserve topology
* compress motion semantics

---

## Dynamics Model

Predict latent transitions.

Possible architectures:

* transformer
* Mamba
* recurrent state-space models
* JEPA-style predictor
* diffusion transformer

Input:

* previous latent states
* action tokens
* direction tokens
* phase embeddings

Output:

* future latent state

---

## Decoder

Convert latent states back into pixel art.

Requirements:

* pixel-grid stability
* palette discipline
* no anti-alias drift
* no blurry interpolation

Potential techniques:

* quantized decoding
* palette-conditioned generation
* nearest-neighbor-aware reconstruction

---

# Symbolic Motion Representation

Important insight:

Sprite animation is partially symbolic.

Examples:

* walk
* anticipation
* contact
* recoil
* airborne
* recovery

Potential representation:

```yaml
action: sword_attack
phase: impact
direction: east
foot_contact: left
velocity: 0.8
```

The model should eventually reason over:

* semantic motion states
* not only pixels

---

# Training Data

## Sources

Potential datasets:

* open-source game assets
* RPG Maker assets
* OpenGameArt
* itch.io asset packs
* manually curated sprite sheets
* animation tutorials
* pixel-art GIFs

---

## Preprocessing

Pipeline:

1. detect frames
2. separate animations
3. classify actions
4. normalize palette
5. extract silhouettes
6. estimate pose
7. infer phase labels

Potential derived supervision:

* motion vectors
* contact maps
* skeleton approximations
* silhouette masks

---

# Evaluation

Traditional image metrics are insufficient.

Need animation-native evaluation.

## Visual Metrics

* identity consistency
* palette consistency
* silhouette preservation
* temporal smoothness

## Motion Metrics

* loop closure quality
* contact stability
* motion readability
* anticipation timing
* impact clarity

## Pixel Metrics

* no subpixel drift
* no jaggies
* no blur collapse
* no palette explosion

---

# Key Insight

Photorealistic video models optimize:

* realism
* texture entropy
* continuous motion

Sprite models should optimize:

* readability
* controllability
* symbolic clarity
* gameplay semantics

This is a fundamentally different objective.

---

# Philosophical Direction

LLMs compress language.

World models compress dynamics.

Sprite world models may compress:

* gameplay motion
* animation grammar
* symbolic action structure

The goal is not merely generating frames.

The goal is learning:

> a predictive symbolic dynamics engine for 2D worlds.

---

# Future Research

Potential future exploration:

* recursive world simulation
* controllable procedural games
* agent-driven animation editing
* reinforcement-learned motion priors
* interactive world generation
* Hebbian-style local refinement
* taste-conditioned animation ranking
* latent planning for gameplay

---

# Non-Goals

This project is NOT:

* another generic diffusion model
* a simple style LoRA
* a photorealistic video model
* a single-shot image generator

This project aims to become:

> a motion-native world model for pixel-art dynamics.

---

# Initial MVP

First milestone:

* train on 32x32 / 64x64 sprites
* predict next animation frame
* preserve identity
* preserve palette
* generate stable walk cycles

Simple success criteria:

* generated loops are usable in games
* no manual cleanup required
* animation timing feels intentional

---

# Repository Principles

* small models first
* real-time inference preferred
* interpretable latent spaces
* controllable generation
* agent-compatible architecture
* iterative workflows over one-shot generation

---

# Working Name Ideas

* SpriteWorldModel
* PixelDynamics
* MotionSprite
* JEPA-Sprite
* TinyWorld2D
* SpriteWM
* PixelState
* LatentSprite
* InkyMotion
* InkyWorld

---

# Closing Thought

Pixel art is already a compressed symbolic representation of reality.

A sprite animation model is therefore not merely learning images.

It is learning:

* motion abstractions
* gameplay semantics
* symbolic visual language
* tiny world dynamics

The future direction is not:

> “bigger image models”

but:

> “smaller, structured, controllable world models specialized for symbolic visual domains.”

