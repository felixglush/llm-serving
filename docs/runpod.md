# Runpod agent setup

Runpod's official agent plugin lets coding agents manage Pods, Serverless endpoints,
templates, and usage through the hosted Runpod MCP server (`https://mcp.runpod.io/`).
Setup follows <https://agents.runpod.io/>. Installs are per user, so each contributor
runs these steps once.

## Claude Code

`.claude/settings.json` registers the `runpod/runpod-plugins-official` marketplace and
enables `runpod@runpod`, so Claude Code offers to install the plugin when you trust this
repo. To install it by hand instead:

```sh
claude plugin marketplace add runpod/runpod-plugins-official
claude plugin install runpod@runpod
claude plugin list | grep -A2 'runpod@runpod'   # expect: enabled
```

Then run `/reload-plugins`, open `/mcp`, pick **runpod**, and choose
**Sign in with Runpod**. OAuth keeps the API key out of config files.

## Codex and other agents

Codex:

```sh
codex plugin marketplace add https://github.com/runpod/runpod-plugins-official.git
codex plugin add runpod@runpod
```

Other agents (Node.js 22.20+), with `<agent-id>` such as `cursor`, `gemini-cli`, or `opencode`:

```sh
npx -y skills add runpod/runpod-plugins-official --skill '*' --agent <agent-id> --yes --global
npx -y skills list --global
npx @runpod/mcp-server@latest add --client <agent-id>   # cursor, windsurf, vscode, opencode, claude-desktop
```

Sign in through the browser the first time a Runpod tool runs.

## Headless and CI

Without a person to complete OAuth, export `RUNPOD_API_KEY` from a secret store (see
`embedding-serving-lab/.env.example`; never commit or print the key) and register the server with a bearer header:

```sh
claude mcp add --transport http runpod -s user https://mcp.runpod.io/ \
  --header "Authorization: Bearer $RUNPOD_API_KEY"
```

Headless Claude Code also needs the skill allowed explicitly, for example
`--allowedTools "Skill(runpod:runpod-mcp)"`.

## Verify

Ask the agent to list your Pods. An empty list means the connection works. If you expected
Pods and see none, you are probably signed in to a different Runpod account.

`runpodctl` and the Flash SDK (Python Serverless deploys) are not installed up front; the
Runpod skills install them when a task first needs them.
