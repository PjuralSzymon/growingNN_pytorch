# Local test of regression CI

GitHub Actions cannot call `localhost`. Local test means: run the image on your PC, then POST a commit SHA yourself. The worker still talks to GitHub (checkout + PR comment). The timeline is at http://127.0.0.1:8080

If `build.bat` failed with `dockerDesktopLinuxEngine` / "Nie można odnaleźć określonego pliku", start **Docker Desktop**, wait until it says running, then continue.

## 1. One-time env

From `tools/regression-ci/`:

```bat
copy .env.example .env
```

Edit `.env`:

```text
CI_SHARED_SECRET=local-test-secret
GITHUB_TOKEN=<token with contents:read and pull-requests:write>
GITHUB_REPO=<owner>/<repo>
DASHBOARD_PASSWORD=local-pass
```

Same token style as Hostinger. Do not commit `.env`.

## 2. Start the stack

```bat
cd tools\regression-ci
.\local-test.bat
```

That builds (first time is slow), starts the container, and checks `/healthz`. You want `{"status":"ok"}`.

Open http://127.0.0.1:8080 and log in with `DASHBOARD_PASSWORD`. Queued and running jobs show under **Now**; finished jobs show under **Earlier**.

## 3. Open a real PR

The worker comments on a GitHub PR number. Create a branch and a PR to `main` (same repo, not a fork).

Copy:

- PR number, for example `72`
- Head SHA of that PR (GitHub PR page, or `git rev-parse HEAD` on the branch)

The SHA must exist on GitHub. The worker `git fetch`es it.

## 4. Start a job (this is what the GitHub Action does)

From `tools\regression-ci\`:

```bat
.\trigger-job.bat
```

It asks for the commit SHA and PR number, reads `CI_SHARED_SECRET` from `.env`, and POSTs the job. You should get `202` and `{"id":"...","state":"queued"}`.

Watch logs:

```bat
cd tools\regression-ci
docker compose logs -f
```

You should see git checkout, then `python tests/regression/ci/mnist.py`.

## 5. What "correct" looks like

| Check | Where |
|---|---|
| Checkout of that SHA | logs: `git fetch` / script start |
| Tests/training ran | logs from `mnist.py` |
| Results saved | after finish, refresh http://127.0.0.1:8080 — one row with dataset, acc, params, better/worse |
| Comment on the PR | GitHub PR conversation: `Regression CI — mnist` with seeds and baseline |

Full MNIST is long (about 45 min per seed × 2 on CPU). Leave the container running.

## Fast path (comment + timeline only)

To skip the long train, add a local-only file `tests/regression/ci/smoke.py` that prints one result line and exits. **Do not commit it**, or Hostinger will run it on every PR.

```python
print('REGRESSION_CI_RESULT {"dataset": "mnist", "seeds": [100, 101], "val_acc": [0.90, 0.91], "param_count": [1000, 1100]}', flush=True)
```

Temporarily move `mnist.py` out of that folder so only `smoke.py` runs. POST the job again. You should get a PR comment and a timeline row in a minute. Put `mnist.py` back and delete `smoke.py` before you push.

## Stop

```bat
cd tools\regression-ci
docker compose down
```

Saved runs live in the Docker volume `runs`. They survive `down` until you `docker compose down -v`.
