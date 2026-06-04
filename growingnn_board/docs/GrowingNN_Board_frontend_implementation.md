# GrowingNN Board Frontend Implementation Notes

## Goal

The GrowingNN Board is a read-only visual dashboard for a training process that is already running somewhere else. The board should not start training, stop training, change parameters, or write experiment files. It only selects a data folder, reads `main.json`, generation JSON files, simulation JSON files, and generated PDF graphs from that folder, then refreshes the UI about every 5 seconds.

The mock images define three screens:

1. **Start page**: choose experiment folder and show previously used training data directories with freshness status.
2. **Main training board**: show current training overview, a large PDF architecture/graph viewer, loss charts, last simulation card, and generation/epoch timeline.
3. **Simulation details board**: show simulation settings/results, candidate actions, selected generation, simulation PDF graph, and a bottom generation picker.

The visual style should be clean, white, spacious, rounded, and dashboard-like, using blue as the main accent color.

---

## Current Code Analysis

### Backend

The current backend is already close to the intended architecture.

`app.py` creates a FastAPI application, enables CORS, mounts the static frontend directory, includes the API router, and starts a background polling thread. The polling loop calls `cache.load(cache.path)` repeatedly when an experiment path is selected. This matches the desired app behavior where the board reads files periodically instead of controlling the training process.

Current behavior:

```python
while True:
    if cache.path is not None:
        cache.load(cache.path)
    time.sleep(settings.poll_interval_sec)
```

`config.py` already defines:

```python
poll_interval_sec = 5
```

This is correct for the board refresh requirement.

`file_reader.py` safely reads JSON files and returns `None` if a file is missing, incomplete, or temporarily invalid. This is good because the training process may be writing files while the board is reading them.

`directory_status()` already implements the start page freshness logic:

```text
< 1 hour   -> active
< 6 hours  -> recent
>= 6 hours -> inactive
```

`cache.py` reads:

```text
main.json
metrics/training.json
generations/generation_*.json
simulations/simulation_gen_*.json
```

This is a good directory structure for the board.

Important issue: `ExperimentCache.load()` currently calls `self.clear()` before reading `main.json`. If the file is temporarily invalid while the trainer is writing it, the previous valid state is lost. This should be changed so the board keeps the last valid cache and only replaces it after the new read succeeds.

Recommended pattern:

```python
def load(self, experiment_path: Path) -> None:
    new_state = read_all_files(experiment_path)
    if new_state.valid:
        self.path = experiment_path
        self.main = new_state.main
        self.training = new_state.training
        self.generations = new_state.generations
        self.simulations = new_state.simulations
    else:
        self.warnings.append("temporary invalid state; keeping last valid data")
```

### Schemas

`schemas.py` currently contains `MainExperiment`, `TrainingParameters`, and `TrainingMetrics`. This is a good start, but it does not yet contain all fields needed by the mock UI.

The mock UI needs these groups:

```text
experiment info
training parameters
current metrics
last simulation
files / graph paths
generation timeline
simulation settings
simulation results
candidate actions
simulation tree / graph data
```

The existing `TrainingParameters` should be expanded. It currently has `simulationAlgorithm`, `simulationTimeSec`, and `learningRateAlpha`, but the sidebar in the mock needs more general training fields such as optimizer, learning rate used, learning rate mode, batch size, weight decay, gradient clip, dropout, random seed, total generations, current generation, current epoch, total epochs, and epochs per generation.

### Frontend

The current frontend works as a basic prototype but does not yet match the mock images.

Current HTML has three views:

```text
home
training
simulation
```

That is correct.

However, the current UI still looks like a generic debug dashboard:

```text
header navigation at top
small centered max-width layout
basic table start page
basic training/sidebar layout
simulation generation select dropdown
```

The mock requires a more application-like layout:

```text
white full-page dashboard
large centered start card
left sidebar on board pages
wide main PDF area
cards with soft shadows
bottom generation selector row
no top debug-like nav buttons
```

The current `styles.css` has `main { max-width: 1200px; }`, which makes the dashboard too narrow. The mock needs almost full screen width, especially because PDF graphs can be huge.

Recommended:

```css
main {
  width: 100%;
  max-width: none;
  margin: 0;
  padding: 0;
}
```

The current PDF viewer uses:

```css
.pdf-viewport {
  overflow: auto;
}
```

For this app, scrollbars should always be visible because the PDF graphs are large and the user needs to know they can move around.

Recommended:

```css
.pdf-viewport {
  overflow: scroll;
}
```

The current JS has one global PDF state:

```js
let pdfDoc = null;
let pdfPage = 1;
let pdfScale = 1.2;
```

This can cause conflicts between the training PDF and simulation PDF. Use separate state objects per viewer.

Recommended:

```js
const pdfViewers = {
  training: { doc: null, page: 1, scale: 1.0 },
  simulation: { doc: null, page: 1, scale: 1.0 }
};
```

---

## Required Frontend Pages

## 1. Start Page

### Purpose

The start page is the first screen of the app. The user chooses the folder where the training algorithm writes board data. The folder must contain `main.json`, generated PDF graphs, and supporting JSON files.

### Layout

The page should be centered with a large title and a clean card-like structure.

Main elements:

```text
GrowingNN Board logo/title
Short subtitle
Folder path input
Folder picker button
Helper text
Previous training data directories table
Freshness legend
```

### Visual Requirements

Use a large white background with soft borders and shadows.

The directory rows should look like rounded cards, not a raw HTML table.

Each previous directory row should show:

```text
folder icon
path
last update
status dot
status label
```

Status rules:

```text
lastUpdate < 1 hour   -> green dot, Active
lastUpdate < 6 hours  -> yellow dot, Recent
lastUpdate >= 6 hours -> gray dot, Inactive
```

### Suggested HTML Structure

```html
<section id="view-home" class="start-page">
  <div class="start-card">
    <div class="brand-row">
      <div class="brand-icon">...</div>
      <h1>GrowingNN Board</h1>
    </div>

    <p class="subtitle">Visualize and analyze your GrowingNN training and simulation data.</p>

    <div class="folder-picker">
      <label>Choose a path to the training data directory:</label>
      <div class="path-input-row">
        <input id="path-input" placeholder="Select folder path..." />
        <button id="load-btn">folder icon</button>
      </div>
      <p>The folder must contain the main JSON file and generated graphs (PDF).</p>
    </div>

    <div class="recent-section">
      <h2>Your previous training data directories:</h2>
      <div id="recent-list"></div>
    </div>

    <div class="freshness-legend">
      <span class="dot green"></span> &lt; 1 hour
      <span class="dot yellow"></span> &lt; 6 hours
      <span class="dot gray"></span> ≥ 6 hours
    </div>
  </div>
</section>
```

---

## 2. Main Training Board

### Purpose

This is the main overview page. It shows what the algorithm is currently doing during training.

The user should immediately see:

```text
experiment parameters
large architecture PDF graph
loss in generation
global loss
last simulation summary
generation timeline divided into epochs
```

### Key Layout Rule

Do not add a right sidebar. The PDF graph needs as much horizontal space as possible.

Use this layout:

```text
left sidebar | main content
```

Inside main content:

```text
large PDF viewer
loss cards + last simulation card
training timeline
```

### Left Sidebar Content

The left sidebar should contain only general experiment and training parameters, not simulation details.

Good fields:

```text
Experiment started on
Training time elapsed
Model
Dataset
Device
Status
Total generations
Current generation
Current epoch
Total epochs
Batch size
Optimizer
Learning rate used
Learning rate mode
Weight decay
Gradient clip
Dropout
Random seed
```

Avoid putting simulation-specific fields here, such as:

```text
simulation max time
simulation max depth
UCB1 settings
simulation algorithm
simulation budget
action space size
```

Those belong on the simulation details page.

### PDF Viewer

The graph is a PDF, not an interactive network graph.

The PDF card title should be contextual:

```text
Simulation Graph (PDF)
```

or for generation-specific views:

```text
Simulation Graph for Generation: 5 (PDF)
```

The file path should be visible near the title:

```text
File path: /logs/exp_2024_05_18/graphs/gen_5/simulation_graph.pdf
```

The PDF viewer must always show both scrollbars:

```css
.pdf-viewport {
  overflow: scroll;
}
```

Toolbar requirements:

```text
page number
zoom percentage
zoom in/out
download icon
print icon
fullscreen icon
```

### Timeline

The timeline is important and should stay similar to the mock.

Rules:

```text
main division = generation
each generation is divided into epochs
show current position as blue dot
show selected generation in blue
show legend
```

Example:

```text
Generation 1 (0-20) | Generation 2 (20-40) | Generation 3 (40-60)
small ticks = epochs
big ticks = generation boundaries
blue dot = current position
```

Do not use a dropdown for selecting the generation on this page. The timeline itself can be clickable later, but the first implementation can be read-only.

---

## 3. Simulation Details Board

### Purpose

This page explains the simulation performed for one selected generation. It should answer:

```text
what structure was simulated?
which actions were tested?
which action was chosen?
what score did each action get?
what was the simulation depth?
what PDF graph belongs to this generation?
```

### Layout

Use this layout:

```text
left sidebar | main content
bottom generation picker
```

The bottom generation picker should be a single horizontal row at the footer, because there may be many generations.

Do not label it as "Pick simulation data".

Use:

```text
Pick generation number:
1 2 3 4 5 6 7 8 9 10 ... 80
```

The selected generation should be blue.

### Left Sidebar

The simulation details page sidebar should contain simulation-specific information.

Sections:

```text
Go to overview board button
Simulation Settings
Simulation Results (Current run)
```

Simulation settings:

```text
Simulation max time
Simulation max depth
UCB1 settings
Simulation algorithm
Exploration constant
Rollout policy
Max branching factor
```

Simulation results:

```text
Mean time of simulation run
Average score of chosen action
Time of chosen simulation
Depth of tree reached
Action chosen
Score of action chosen
```

### Top Simulation Candidate Area

Show the starting structure and candidate action cards.

Each candidate card should show:

```text
small structure icon
Action name
Score
Accuracy
Params
```

The chosen action should have a blue border/background.

### PDF Viewer Title

Use a generation-specific title:

```text
Simulation Graph for Generation: 5 (PDF)
```

Also show:

```text
File path: /logs/exp_2024_05_18/graphs/gen_5/simulation_graph.pdf
```

### Bottom Generation Picker

Implementation idea:

```html
<footer class="generation-picker">
  <strong>Pick generation number:</strong>
  <div class="generation-scroll-row" id="generation-buttons"></div>
</footer>
```

CSS:

```css
.generation-picker {
  position: sticky;
  bottom: 0;
  background: white;
  border-top: 1px solid var(--border);
  padding: 14px 24px;
  display: flex;
  align-items: center;
  gap: 18px;
}

.generation-scroll-row {
  display: flex;
  gap: 12px;
  overflow-x: auto;
  flex: 1;
}

.generation-button {
  min-width: 44px;
  height: 44px;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: white;
}

.generation-button.active {
  background: var(--accent);
  color: white;
  border-color: var(--accent);
}
```

---

## Recommended JSON Schemas for the Frontend

## `main.json`

This file should drive the training board and start page freshness.

```json
{
  "experimentId": "exp_2024_05_18_001",
  "experimentName": "GrowingNN GPT experiment",
  "lastUpdate": "2024-05-18T10:42:11Z",
  "experimentStartedOn": "2024-05-18T09:42:11Z",
  "trainingTimeElapsedSec": 12932,
  "status": "running",
  "dataset": "Custom Dataset v1.0",
  "device": "RTX 4090",
  "model": {
    "name": "GrowingNN",
    "version": "v2.3",
    "baseModel": "GPT-2 Small"
  },
  "trainingParameters": {
    "totalGenerations": 8,
    "currentGeneration": 5,
    "currentEpoch": 12,
    "epochsPerGeneration": 20,
    "totalEpochs": 160,
    "completedGlobalEpochs": 92,
    "batchSize": 64,
    "optimizer": "AdamW",
    "learningRateUsed": 0.000125,
    "learningRateMode": "progressive",
    "weightDecay": 0.01,
    "gradientClip": 1.0,
    "dropout": 0.1,
    "randomSeed": 42
  },
  "currentMetrics": {
    "generationLoss": 1.842,
    "globalLoss": 0.316,
    "accuracy": 0.7321
  },
  "lastSimulation": {
    "generation": 5,
    "actionsAnalyzed": 24,
    "treeDepth": 3,
    "executionTimeSec": 210,
    "actionChosen": "Add SeqLayer 3",
    "scoreChosen": 24123,
    "scoreMetric": "UCB1"
  },
  "graphs": {
    "latestSimplified": "graphs/gen_5_graph.pdf",
    "latestSimulation": "graphs/gen_5_simulation_graph.pdf"
  }
}
```

## `metrics/training.json`

This file should drive the two charts and timeline.

```json
{
  "lastUpdate": "2024-05-18T10:42:11Z",
  "epochs": [
    {
      "generation": 5,
      "epochInGeneration": 12,
      "globalEpoch": 92,
      "trainLoss": 1.91,
      "valLoss": 1.84,
      "globalLoss": 0.316,
      "accuracy": 0.732
    }
  ]
}
```

## `simulations/simulation_gen_5.json`

This file should drive the simulation details page.

```json
{
  "generation": 5,
  "simulationId": "sim_gen_5_001",
  "createdAt": "2024-05-18T10:35:42Z",
  "settings": {
    "simulationMaxTimeSec": 120,
    "simulationMaxDepth": 3,
    "ucb1Enabled": true,
    "explorationConstant": 1.4,
    "algorithm": "Monte Carlo",
    "rolloutPolicy": "Default",
    "maxBranchingFactor": 64
  },
  "results": {
    "meanSimulationRunTimeSec": 130,
    "averageChosenActionScore": 123,
    "timeOfChosenSimulationSec": 150,
    "depthReached": 3,
    "chosenAction": "Add SeqLayer 3",
    "chosenActionScore": 24123,
    "scoreMetric": "UCB1"
  },
  "startingStructure": {
    "totalParams": 12450000,
    "accuracy": 0.7321
  },
  "candidateActions": [
    {
      "actionId": "action_001",
      "name": "Add SeqLayer 3",
      "score": 24123,
      "scoreMetric": "UCB1",
      "accuracyAfter": 0.8421,
      "paramsAfter": 12780000,
      "isChosen": true
    }
  ],
  "files": {
    "simulationGraphPdf": "graphs/gen_5_simulation_graph.pdf"
  }
}
```

---

## Frontend Implementation Plan

## Step 1: Replace Generic Header with App Layout

The mock pages do not need top navigation buttons. Navigation should happen through UI buttons:

```text
Start page -> load experiment -> training board
Training board -> Check more simulation board -> simulation board
Simulation board -> Go to overview board -> training board
```

Keep the three view sections internally, but remove visible header nav.

## Step 2: Restyle Start Page

Replace the current `table` UI with card rows.

Use:

```js
function renderRecentExperiments(experiments) {
  const list = $("recent-list");
  list.innerHTML = "";
  for (const exp of experiments) {
    const row = document.createElement("button");
    row.className = "recent-row";
    row.innerHTML = `
      <span class="folder-icon">...</span>
      <span class="recent-path">${exp.path}</span>
      <span class="recent-update">${exp.lastUpdate}</span>
      <span class="recent-status"><span class="status-dot ${statusClass(exp.status)}"></span>${exp.status}</span>
    `;
    row.onclick = () => loadExperimentPath(exp.path);
    list.appendChild(row);
  }
}
```

## Step 3: Create Shared Sidebar Component

Use separate sidebar rendering for training and simulation.

Training sidebar:

```js
renderTrainingSidebar(main)
```

Simulation sidebar:

```js
renderSimulationSidebar(sim)
```

Do not mix simulation fields into the training sidebar.

## Step 4: Create Reusable PDF Viewer Component

Current PDF rendering should be refactored.

Recommended API:

```js
createPdfViewer({
  name: "training",
  canvasId: "training-pdf-canvas",
  viewportId: "training-pdf-viewport",
  pageInfoId: "training-pdf-page-info"
});
```

Each viewer should have its own state:

```js
const pdfState = {
  training: { doc: null, page: 1, scale: 1.0 },
  simulation: { doc: null, page: 1, scale: 1.0 }
};
```

## Step 5: Build the Training Timeline

Use `main.trainingParameters` and/or `main.generationTimeline`.

Minimal rendering:

```js
function renderTimeline(tp) {
  const totalGenerations = tp.totalGenerations;
  const epochsPerGeneration = tp.epochsPerGeneration;
  const currentGeneration = tp.currentGeneration;
  const currentEpoch = tp.currentEpoch;
}
```

The DOM can be:

```html
<div class="timeline-card">
  <div class="timeline-legend">...</div>
  <div class="timeline-track" id="timeline-track"></div>
</div>
```

Each generation block:

```html
<div class="generation-segment active">
  <div class="generation-label">Generation 5<br>(80 - 100)</div>
  <div class="epoch-ticks">...</div>
</div>
```

## Step 6: Replace Simulation Dropdown with Bottom Generation Picker

Remove:

```html
<select id="sim-generation"></select>
```

Add:

```html
<footer class="generation-picker">
  <strong>Pick generation number:</strong>
  <div id="generation-buttons" class="generation-scroll-row"></div>
</footer>
```

JS:

```js
function renderGenerationButtons(gens, selectedGen) {
  const box = $("generation-buttons");
  box.innerHTML = "";
  for (const gen of gens) {
    const btn = document.createElement("button");
    btn.className = "generation-button" + (gen === selectedGen ? " active" : "");
    btn.textContent = gen;
    btn.onclick = () => loadSimulation(gen);
    box.appendChild(btn);
  }
}
```

## Step 7: Match the Mock CSS

Use these design tokens:

```css
:root {
  --bg: #f6f8fb;
  --card: #ffffff;
  --text: #0f172a;
  --muted: #64748b;
  --border: #e5e7eb;
  --accent: #2563eb;
  --accent-soft: #eff6ff;
  --green: #16a34a;
  --yellow: #eab308;
  --gray: #9ca3af;
  --shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
  --radius: 14px;
}
```

Use a full-width dashboard shell:

```css
.board-shell {
  min-height: 100vh;
  background: var(--bg);
  display: grid;
  grid-template-columns: 320px minmax(0, 1fr);
  gap: 16px;
  padding: 10px;
}
```

Use large PDF viewport:

```css
.pdf-viewport {
  height: 420px;
  overflow: scroll;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: white;
}
```

Use a larger one for simulation:

```css
.simulation-pdf .pdf-viewport {
  height: 520px;
}
```

---

## Backend Changes Needed to Support the Mock

## 1. Add Missing Schema Fields

Update `TrainingParameters`:

```python
class TrainingParameters(BaseModel):
    totalGenerations: int = 0
    epochsPerGeneration: int = 0
    currentGeneration: int = 0
    currentEpoch: int = 0
    totalEpochs: int = 0
    completedGlobalEpochs: int = 0
    batchSize: int | None = None
    optimizer: str = ""
    learningRateUsed: float | None = None
    learningRateMode: str = ""
    weightDecay: float | None = None
    gradientClip: float | None = None
    dropout: float | None = None
    randomSeed: int | None = None
```

Simulation settings should not be inside `TrainingParameters`; they belong in simulation JSON.

## 2. Add Simulation Schema

```python
class SimulationSettings(BaseModel):
    simulationMaxTimeSec: float = 0
    simulationMaxDepth: int = 0
    algorithm: str = ""
    ucb1Enabled: bool = False
    explorationConstant: float | None = None
    rolloutPolicy: str = ""
    maxBranchingFactor: int | None = None

class SimulationResults(BaseModel):
    meanSimulationRunTimeSec: float | None = None
    averageChosenActionScore: float | None = None
    timeOfChosenSimulationSec: float | None = None
    depthReached: int | None = None
    chosenAction: str = ""
    chosenActionScore: float | None = None
    scoreMetric: str = ""

class CandidateAction(BaseModel):
    actionId: str = ""
    name: str = ""
    score: float | None = None
    scoreMetric: str = ""
    accuracyAfter: float | None = None
    paramsAfter: int | None = None
    isChosen: bool = False

class SimulationData(BaseModel):
    generation: int
    simulationId: str = ""
    createdAt: str = ""
    settings: SimulationSettings = Field(default_factory=SimulationSettings)
    results: SimulationResults = Field(default_factory=SimulationResults)
    startingStructure: dict[str, Any] = Field(default_factory=dict)
    candidateActions: list[CandidateAction] = Field(default_factory=list)
    files: dict[str, str] = Field(default_factory=dict)
```

## 3. Fix Cache Refresh

Do not clear the cache before successful validation. This prevents flicker and empty UI when files are temporarily half-written.

## 4. PDF Endpoint Safety

The PDF endpoint must only serve files from the currently loaded experiment directory.

Validation rule:

```text
resolved_pdf_path must be inside resolved_experiment_path
file extension must be .pdf
```

---

## Concrete UI Acceptance Checklist

### Start page

- [ ] Centered title and logo.
- [ ] Folder path input with folder icon button.
- [ ] Previous directories shown as rounded rows.
- [ ] Status dots: green, yellow, gray.
- [ ] Legend explains `lastUpdate` status rules.

### Main training board

- [ ] Left sidebar contains only general experiment/training parameters.
- [ ] No right sidebar.
- [ ] PDF graph uses most of page width.
- [ ] PDF viewer always shows horizontal and vertical scrollbars.
- [ ] Loss charts appear under PDF.
- [ ] Last simulation card appears beside charts.
- [ ] Timeline is divided by generations and epochs.
- [ ] Timeline has legend.

### Simulation board

- [ ] Left sidebar contains simulation settings/results.
- [ ] Top area shows starting structure and candidate actions.
- [ ] Chosen action card is highlighted blue.
- [ ] PDF title says `Simulation Graph for Generation: X (PDF)`.
- [ ] File path is visible near PDF title.
- [ ] Bottom footer says `Pick generation number:`.
- [ ] Generation numbers are in one horizontal scrollable row.
- [ ] Selected generation is highlighted blue.

---

## Suggested File Changes

### `index.html`

Replace the current simple sections with:

```text
view-home
view-training-board
view-simulation-board
```

Use semantic containers:

```text
start-page
board-shell
sidebar-card
content-card
pdf-card
metric-card
timeline-card
generation-picker
```

### `styles.css`

Replace the current generic style with mock-based layout styles:

```text
full width layout
large sidebar
rounded cards
soft shadows
PDF scroll container
bottom generation picker
large start page
```

### `app.js`

Refactor into render functions:

```js
renderStartPage()
renderTrainingBoard(main, training)
renderTrainingSidebar(main)
renderLastSimulation(main.lastSimulation)
renderTrainingCharts(training)
renderTimeline(main.trainingParameters)
renderSimulationBoard(sim)
renderSimulationSidebar(sim)
renderCandidateActions(sim.candidateActions)
renderGenerationPicker(gens, selectedGen)
renderPdfViewer(viewerName, pdfPath)
```

This will make the frontend much easier to maintain.

---

## Summary

The current implementation already has the correct foundation: FastAPI backend, static frontend, safe JSON reading, polling every 5 seconds, and PDF.js/Chart.js usage. The main work now is frontend restructuring and schema expansion. The frontend should move away from the basic prototype layout and implement the three mock screens directly: a centered start page, a wide main training dashboard with a large PDF viewer and timeline, and a simulation details dashboard with candidate actions and a footer generation picker. The app should remain read-only and should always treat the selected experiment directory as the source of truth.
