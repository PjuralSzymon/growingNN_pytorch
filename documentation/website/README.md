# GrowingNN Angular documentation

This website is an Angular 22 standalone application. Angular prerenders every route into static HTML, so Hostinger does not need Node.js or Python.

Python runs only before the Angular build. It converts the Obsidian vault to typed Angular content and generates the self-contained PyVis knowledge graph.

## View locally

Double-click `view_local.bat`.

The script installs portable Node.js, Angular dependencies, and PyVis when they are missing. It then opens `http://localhost:4200`.

Manual commands:

```text
python -m pip install -r documentation/website/requirements.txt
cd documentation/website/app
npm ci
npm start
```

`npm start` runs `scripts/generate_content.py` before Angular starts.

## Project structure

- `app/src/app/` contains Angular components and services.
- `app/src/styles.css` loads the shared visual design.
- `scripts/generate_content.py` converts Markdown and builds PyVis.
- `content/guides/` contains the algorithm introduction.
- `content/experiments/` contains sequential experiment reports.
- `../obsydian/growingNN/` contains all technical documentation.
- `app/src/app/generated/content.ts` is generated. Do not edit it.
- `app/public/assets/knowledge-graph.html` is generated. Do not edit it.

## Add an experiment

Copy the latest file under `content/experiments`. Give it the next three-digit number, for example `experiment-004-score-ablation.md`.

Use these sections:

1. Goal
2. Setup
3. Results
4. Finding
5. Next step

The next Angular build adds the page, route, navigation, search result, and prerendered HTML automatically.

## Production build

```text
cd documentation/website/app
npm run build
```

Static output is written to:

```text
documentation/website/app/dist/growingnn-docs/browser/
```

The build currently creates 53 routes: the homepage, documentation directory, graph, 45 vault pages, one algorithm guide, and four experiments.

## Deploy to Hostinger

`.github/workflows/deploy-documentation.yml` builds Angular and uploads the static browser output after documentation changes reach `main`.

Add these GitHub Actions repository secrets:

- `HOSTINGER_FTP_SERVER`
- `HOSTINGER_FTP_USERNAME`
- `HOSTINGER_FTP_PASSWORD`

Set the optional repository variable `HOSTINGER_DIRECTORY` when the site does not use `./public_html/`.

Use a dedicated domain or subdomain folder. Do not deploy over another live website.
