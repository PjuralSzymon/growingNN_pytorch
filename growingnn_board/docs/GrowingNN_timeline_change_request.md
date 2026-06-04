# GrowingNN Board - Timeline UI Change Request

## Goal

The current implementation uses a single long linear timeline with generation divisions and epoch ticks. This should be changed to a horizontal row of compact generation cards, like the target mockup. Each generation should appear as its own rounded card. The selected/current generation should be highlighted with a blue border and light blue background. The timeline should be horizontally scrollable because there may be many generations.

## Current UI Problem

The current timeline shows all generations on one continuous axis. This makes the component wide, difficult to scan, and less useful when the number of generations grows. The current design also makes epochs look like small tick marks on a long scale, so the user has to interpret the whole axis instead of quickly seeing which generation is active.

## Desired UI

Replace the continuous timeline with a card-based generation selector/timeline.

The new timeline should contain:

- A legend at the top-left of the timeline card:
  - blue line = Epoch
  - dark line = Generation division
  - blue dot = Current position
- A horizontal row of generation cards.
- One card per generation.
- Each card should show:
  - generation number, for example `Generation 2`
  - epoch range, for example `(5 - 10)`
  - a small mini epoch chart inside the card
  - current epoch marker if the current position is inside that generation
- The selected/current generation should be visually highlighted.
- The whole row should have a visible horizontal scrollbar.
- The component should support many generations without breaking layout.

## Visual Layout

The final layout should look conceptually like this:

```text
Training timeline

Legend:
-- Epoch      -- Generation division      ● Current position

┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Generation 1 │ │ Generation 2 │ │ Generation 3 │ │ Generation 4 │
│ (0 - 5)      │ │ (5 - 10)     │ │ (10 - 15)    │ │ (15 - 20)    │
│ ▂ ▃ ▄ ▃ ▂    │ │ ▂ █ ▃ ▂ ▃    │ │ ▂ ▃ ▄ ▃ ▂    │ │ ▂ ▃ ▄ ▃ ▂    │
│              │ │   ●          │ │              │ │              │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

## Data Needed

The frontend can build this component from `main.generationTimeline` or from generation JSON files.

Recommended shape in `main.json`:

```json
{
  "generationTimeline": [
    {
      "generation": 1,
      "startEpoch": 0,
      "endEpoch": 5,
      "currentEpoch": null,
      "isCurrent": false,
      "isSelected": false,
      "epochValues": [0.2, 0.3, 0.4, 0.3, 0.35]
    },
    {
      "generation": 2,
      "startEpoch": 5,
      "endEpoch": 10,
      "currentEpoch": 7,
      "isCurrent": true,
      "isSelected": true,
      "epochValues": [0.25, 0.8, 0.3, 0.25, 0.32]
    }
  ]
}
```

If `epochValues` are not available, the frontend can render placeholder bars or use loss/accuracy values from training metrics for epochs inside that generation.

## Frontend Implementation

### 1. Add timeline container to `index.html`

Replace the old long-axis timeline block with:

```html
<section class="timeline-card">
  <div class="timeline-legend">
    <span><i class="legend-line epoch"></i> Epoch</span>
    <span><i class="legend-line generation"></i> Generation division</span>
    <span><i class="legend-dot current"></i> Current position</span>
  </div>

  <div id="generation-timeline" class="generation-timeline-scroll"></div>
</section>
```

### 2. Render generation cards in `app.js`

Add a function like:

```js
function renderGenerationTimeline(timeline, currentGeneration, currentEpoch) {
  const root = document.getElementById("generation-timeline");
  if (!root) return;

  root.innerHTML = "";

  for (const gen of timeline || []) {
    const isCurrent = gen.generation === currentGeneration;
    const card = document.createElement("button");
    card.className = `generation-card ${isCurrent ? "current" : ""}`;

    const values = gen.epochValues || [0.25, 0.35, 0.45, 0.3, 0.38];
    const bars = values.map((v, idx) => {
      const height = Math.max(8, Math.round(v * 40));
      const marker = isCurrent && gen.currentEpochIndex === idx
        ? '<span class="current-epoch-dot"></span>'
        : '';
      return `<span class="epoch-bar" style="height:${height}px">${marker}</span>`;
    }).join("");

    card.innerHTML = `
      <div class="generation-title">Generation ${gen.generation}</div>
      <div class="generation-range">(${gen.startEpoch} - ${gen.endEpoch})</div>
      <div class="generation-mini-chart">${bars}</div>
    `;

    card.onclick = () => loadGeneration(gen.generation);
    root.appendChild(card);
  }
}
```

Call this from `refreshMain()` after reading `main.trainingParameters`:

```js
renderGenerationTimeline(
  main.generationTimeline,
  tp.currentGeneration,
  tp.currentEpoch
);
```

### 3. Add CSS for the new timeline

Add or replace timeline CSS with:

```css
.timeline-card {
  background: #fff;
  border: 1px solid #e5e7eb;
  border-radius: 16px;
  padding: 20px;
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
}

.timeline-legend {
  display: flex;
  align-items: center;
  gap: 22px;
  margin-bottom: 18px;
  color: #475569;
  font-size: 14px;
}

.legend-line {
  display: inline-block;
  width: 28px;
  height: 2px;
  margin-right: 8px;
  vertical-align: middle;
}

.legend-line.epoch {
  background: #2563eb;
}

.legend-line.generation {
  background: #334155;
  height: 4px;
}

.legend-dot.current {
  display: inline-block;
  width: 9px;
  height: 9px;
  margin-right: 8px;
  border-radius: 999px;
  background: #2563eb;
}

.generation-timeline-scroll {
  display: flex;
  gap: 10px;
  overflow-x: auto;
  overflow-y: hidden;
  padding-bottom: 14px;
  scroll-snap-type: x proximity;
}

.generation-card {
  min-width: 132px;
  height: 96px;
  padding: 12px 14px;
  border: 1px solid #dbe3ef;
  border-radius: 12px;
  background: #fff;
  text-align: left;
  cursor: pointer;
  scroll-snap-align: start;
}

.generation-card.current {
  border-color: #2563eb;
  background: #eff6ff;
  box-shadow: 0 0 0 1px rgba(37, 99, 235, 0.2);
}

.generation-title {
  font-weight: 700;
  font-size: 13px;
  color: #334155;
}

.generation-card.current .generation-title,
.generation-card.current .generation-range {
  color: #2563eb;
}

.generation-range {
  margin-top: 2px;
  font-size: 12px;
  color: #64748b;
}

.generation-mini-chart {
  position: relative;
  display: flex;
  align-items: flex-end;
  gap: 4px;
  height: 32px;
  margin-top: 12px;
}

.epoch-bar {
  position: relative;
  display: inline-block;
  width: 16px;
  min-height: 8px;
  background: #cbd5e1;
  border-radius: 2px 2px 0 0;
}

.generation-card.current .epoch-bar.active,
.generation-card.current .epoch-bar:nth-child(2) {
  background: #2563eb;
}

.current-epoch-dot {
  position: absolute;
  left: 50%;
  bottom: -6px;
  width: 10px;
  height: 10px;
  transform: translateX(-50%);
  border-radius: 999px;
  background: #2563eb;
  border: 2px solid #fff;
}
```

## Behavior Requirements

- Clicking a generation card should load that generation's graph, charts, and simulation summary.
- Current generation should be highlighted automatically based on `trainingParameters.currentGeneration`.
- If the user manually selects a different generation, use the same selected style but keep the current-position dot only on the true current generation/epoch.
- The scrollbar must always be available when the number of generations exceeds available width.
- The timeline must not resize the whole page horizontally; only the timeline row should scroll.

## Files Likely Affected

```text
static/index.html
static/app.js
static/styles.css
schemas.py
```

Optional backend change:

```text
main.json generationTimeline can already be passed through because MainExperiment currently stores it as a list of dictionaries.
```

## Acceptance Criteria

The change is complete when:

- The old continuous axis timeline is removed.
- The new card-based generation timeline appears at the bottom of the main training board.
- Generations are shown as compact cards in one horizontal row.
- The current generation is highlighted blue.
- A blue dot shows the current epoch inside the current generation.
- Horizontal scrolling works when many generations exist.
- The legend clearly explains epoch bars, generation division, and current position.
