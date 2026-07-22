# GrowingNN documentation website

This folder builds the Obsidian vault and experiment reports into a static website. PyVis generates the interactive knowledge graph during the build. Hostinger only serves the static files created in `dist`, so it needs no Python runtime.

## View it locally

Double-click `view_local.bat`.

The script builds the website, opens `http://localhost:8000`, and starts a local server. Stop it with `Ctrl+C`.

You can also run:

```text
python -m pip install -r documentation/website/requirements.txt
python documentation/website/build.py
python -m http.server 8000 --directory documentation/website/dist
```

## Edit the website

- Edit the main algorithm introduction in `content/guides/algorithm-overview.md`.
- Edit normal technical pages in `documentation/obsydian/growingNN`.
- Add experiment reports in `content/experiments`.
- Change the visual style in `assets/site.css`.

Obsidian links such as `[[MCTS]]` become normal website links during the build. Every Markdown page in the vault is included automatically.

The documentation directory groups pages by their first Obsidian folder. The knowledge graph is generated from wiki links. It supports zooming, dragging, inspecting, and opening linked pages.

To add an experiment, copy the latest experiment file. Use the next three-digit number in its filename and title. Keep these short sections:

1. Goal
2. Setup
3. Results
4. Finding
5. Next step

The filename controls the order. For example, `experiment-004-score-ablation.md` follows Experiment 003.

## Deploy to Hostinger

The repository includes `.github/workflows/deploy-documentation.yml`. It builds and uploads the website after a documentation change is merged or pushed to `main`.

One-time setup:

1. In Hostinger hPanel, open `Websites`, choose the domain, then open `Files` and `FTP Accounts`.
2. Copy the FTP host and username. Set or copy the FTP password.
3. In GitHub, open the repository. Go to `Settings`, `Secrets and variables`, `Actions`.
4. Add these three repository secrets:
   - `HOSTINGER_FTP_SERVER`
   - `HOSTINGER_FTP_USERNAME`
   - `HOSTINGER_FTP_PASSWORD`
5. Under the `Variables` tab, add `HOSTINGER_DIRECTORY` only when the domain uses a different folder. Its usual value is `./public_html/`.
6. Open the GitHub `Actions` tab and run `Deploy documentation` once with `Run workflow`.

After this setup, each relevant merge to `main` deploys the current site. No cron job is required.

## Domain folder

The website expects to be served from the root of a domain or subdomain. A clean option is to create a documentation subdomain such as `docs.example.com` in Hostinger and deploy to that subdomain's `public_html` folder.

Do not point the workflow at a folder that contains another live website. Use the correct domain or subdomain folder in `HOSTINGER_DIRECTORY`.
