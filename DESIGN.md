# Design System — Empath

## Product Context
- **What this is:** A journaling app that measures mental entropy (cognitive fragmentation, coherence, integration) and generates state-aware therapeutic insights using a closed-loop feedback system
- **Who it's for:** People who journal for self-understanding, not just logging. Likely already tried other journaling apps but felt the insights were generic
- **Space/industry:** Mental health / journaling / quantified self. Competitors: Rosebud, Reflectr, Jour, Day One. True reference products: Whoop, Oura, Linear (precision instruments, not wellness apps)
- **Project type:** Consumer landing page + web app

## Aesthetic Direction
- **Direction:** Dark Instrument — Linear meets Oura
- **Decoration level:** Intentional — subtle noise/grain texture on dark surfaces for warmth, one signature element (animated entropy waveform)
- **Mood:** Opening a precision instrument for the mind. Clinical but warm. Data-forward, not wellness-soft. The product IS the visual, not stock photos or illustrations
- **Reference sites:** linear.app (dark + data hero), whoop.com (black + performance), ouraring.com (premium instrument)
- **Anti-references:** Rosebud (white pastel, generic wellness), any app with nature photography heroes

## Typography
- **Display/Hero:** Satoshi (900, 700) — geometric, modern, confident. Not cold like Inter, not trendy like Clash. Clean enough for data, distinctive enough for headlines
- **Body:** Instrument Sans (400, 500, 600) — crisp, readable, slightly narrower than typical sans. Great at small sizes on dark backgrounds
- **UI/Labels:** Instrument Sans 600
- **Data/Scores:** Geist Mono (400, 500, 600, 700) — MES score rendered in monospace says "this is a real measurement." Must support tabular-nums for score alignment
- **Code:** Geist Mono
- **Loading:** Fontshare CDN for Satoshi + Instrument Sans, Google Fonts for Geist Mono
- **Scale:**
  - Hero: clamp(2rem, 5vw, 3.5rem) / Satoshi 900 / -0.03em tracking
  - Section title: clamp(1.5rem, 3vw, 2.25rem) / Satoshi 700 / -0.02em tracking
  - Body: 1rem (16px) / Instrument Sans 400 / 1.6-1.7 line-height
  - Small/Caption: 0.875rem / Instrument Sans 500
  - Mono labels: 0.7rem / Geist Mono / 0.1-0.15em tracking / uppercase
  - Score display: 4.5rem / Geist Mono 700 / -0.02em tracking

## Color
- **Approach:** Restrained — one bold accent on near-black. Color is rare and meaningful
- **Background:** `#0A0A0F` — near-black with blue undertone, warmer than pure black
- **Surface:** `#14141F` — cards, elevated elements, panels
- **Border:** `#1E1E2E` — subtle separation between elements
- **Primary text:** `#EAEAF0` — off-white, easier on eyes than pure white on dark
- **Muted text:** `#6B6B80` — secondary info, descriptions, timestamps
- **Accent:** `#4AE3B5` — luminous teal-green. The "signal" color. Used for MES score, active states, CTAs, progress bars, badges
- **Accent hover:** `#3CC9A0`
- **Accent glow:** `rgba(74, 227, 181, 0.15)` — radial glow behind focal elements
- **Accent glow strong:** `rgba(74, 227, 181, 0.3)` — hover states, emphasis
- **Semantic — Warning (high entropy/overwhelm):** `#E34A4A` — red
- **Semantic — Info (low entropy/rigidity):** `#4A7AE3` — blue
- **Semantic — Active (optimal/integration):** `#4AE3B5` — accent teal-green (same as accent)
- **Dark mode:** This IS dark mode. No light mode planned for landing page. App may add light mode later
- **Why teal-green:** Not purple (every AI app), not blue (every wellness app), not orange (Headspace owns it). Teal-green on dark reads as "alive, precise, measuring something real" — the color of a signal on an oscilloscope

## CSS Custom Properties
```css
:root {
  --bg: #0A0A0F;
  --surface: #14141F;
  --border: #1E1E2E;
  --text: #EAEAF0;
  --text-muted: #6B6B80;
  --accent: #4AE3B5;
  --accent-hover: #3CC9A0;
  --accent-glow: rgba(74, 227, 181, 0.15);
  --accent-glow-strong: rgba(74, 227, 181, 0.3);
  --warning: #E34A4A;
  --info: #4A7AE3;
}
```

## Spacing
- **Base unit:** 8px
- **Density:** Comfortable — landing page breathes with generous section spacing. Within data visualizations, tight (data-dense)
- **Scale:** 2xs(2px) xs(4px) sm(8px) md(16px) lg(24px) xl(32px) 2xl(48px) 3xl(64px) 4xl(96px)
- **Section padding:** 96px vertical (desktop), 64px (mobile)

## Layout
- **Approach:** Hybrid — hero section is creative-editorial (asymmetric, MES visualization dominates), explanation sections are grid-disciplined
- **Grid:** Single column centered (max-width 1100px) with content-specific grids: 3-col for steps, 5-col for dimensions, 2-col for comparisons
- **Max content width:** 1100px
- **Border radius:** sm: 4px, md: 8px, lg: 12px, full: 9999px (pills, badges, buttons)
- **First viewport:** Poster, not document. The MES score ring is the visual centerpiece

## Motion
- **Approach:** Intentional — measured, like breathing
- **MES score counter:** Animates on first view (counting up to the number)
- **Entropy waveform:** Subtle bar animation, 1.5s cycle, staggered delays
- **State badge:** Gentle pulse on the indicator dot (2s cycle)
- **Section transitions:** Subtle fade-up on scroll entry
- **Score ring progress:** 2s ease-out stroke animation
- **Easing:** enter: ease-out, exit: ease-in, move: ease-in-out
- **Duration:** micro: 50-100ms, short: 150-250ms, medium: 250-400ms, long: 400-700ms
- **Avoid:** bouncy, playful, spring physics. This is an instrument, not a toy

## Component Patterns

### Buttons
- **Primary:** Accent bg (#4AE3B5), dark text (#0A0A0F), pill shape (radius-full), font-body 500
- **Secondary:** Transparent bg, border (#1E1E2E), text color, pill shape
- **Ghost:** No bg, no border, muted text, hover to full text
- **Danger:** Warning red bg at 15% opacity, warning text, warning border at 20%

### State Badges
- **Active Integration:** Accent glow bg, accent border 20%, accent text, pulsing dot
- **Overwhelm:** Warning red 15% bg, warning border 20%, warning text
- **Rigidity:** Info blue 15% bg, info border 20%, info text
- **Baseline:** Neutral bg (text 8% opacity), muted text, border

### Cards
- Surface bg, border, radius-md (8px). Hover: border transitions to accent. Used for dimension cards, step cards, insight panels

### Score Display
- Geist Mono 700 at 4.5rem for the primary number. Radial accent glow behind. Ring visualization with 4px stroke, accent color, drop-shadow glow. "entropy score" label in Geist Mono 0.7rem uppercase muted

### Section Headers
- Label: Geist Mono 0.7rem, accent color, uppercase, 0.12em tracking
- Title: Satoshi 700, clamp sizing, -0.02em tracking
- Description: Instrument Sans 400, muted text, max-width 600px

## Noise Texture
Apply a subtle SVG noise overlay (fractalNoise, baseFrequency 0.9, opacity 0.03) as a fixed pseudo-element on body. Adds warmth to dark surfaces without feeling sterile. Pointer-events: none

## Landing Page Structure
1. **Nav:** Fixed, blurred bg, logo left, links + CTA right
2. **Hero:** Full viewport. "Your mind, measured." + MES score ring + waveform + record CTA
3. **Three Zones:** Spectrum bar (rigid/optimal/overwhelm) + 3-col explanation
4. **Five Dimensions:** 5-col cards with score, name, bar
5. **How It Works:** 3-col steps (Measure, Diagnose, Close the Loop)
6. **Insight Engine Demo:** 2-col state classification + generated insight
7. **Comparison:** 2-col "Every other AI journal" vs "Empath"
8. **CTA:** Centered, radial glow, "Start measuring."
9. **Footer:** Minimal, logo + version line

## Key Copy Lines
- "Your mind, measured."
- "Other apps read your words. Empath reads the state you're writing from."
- "Record 30 seconds. See what your mind looks like."
- "Mental entropy isn't a number to minimize."
- "A closed-loop system, not a chatbot."
- "Same words. Different state. Different insight."
- "Content is noise. State is signal."
- "Start measuring."

## Decisions Log
| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-01 | Initial design system created | /design-consultation based on competitive research (Rosebud, Whoop, Linear, Oura) and product positioning as precision instrument for the mind |
| 2026-04-01 | Dark mode only for landing page | Every journaling competitor is white/pastel. Dark immediately signals "instrument, not wellness blanket" |
| 2026-04-01 | Teal-green accent (#4AE3B5) | Not purple (AI), not blue (wellness), not orange (Headspace). Reads as oscilloscope signal on dark |
| 2026-04-01 | Geist Mono for MES score | Monospace at 80px+ says "real measurement" not "feel-good number" |
| 2026-04-01 | Interactive voice demo as hero | Unprecedented in category. Gives value before signup. Creates curiosity loop |
| 2026-04-01 | No photos, no illustrations | Only data visualizations and product UI. The Whoop move: the product is the aesthetic |
