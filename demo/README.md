# demo/

The pages here are **generated, not committed** — each embeds a multi-megabyte trace, and a
committed copy is a claim that ages while a Makefile target is one that does not.

```bash
make race     # v1 vs v2 vs DCN through one maze -> demo/index.html
make demo     # one architecture, in more detail -> demo/single.html
make serve    # then open http://127.0.0.1:8080
```

**index.html — the race.** Three architectures, the same maze, the same breadth-first
planner, the same 900 steps. The only difference is the map each one decodes out of its own
internal state, so a difference in exits reached is a difference in what each remembers about
the walls it cannot currently see.

The page draws each architecture's *believed* map rather than the real maze, with the real
maze shown once underneath for comparison. It also prints the number that decides whether any
of it means anything: how well the same out-of-view walls decode straight from raw pixels. An
entrant below that line is not remembering anything the image did not already contain.

**single.html** is the older single-model view, kept because it shows the belief map
accumulating in more detail.

Both are self-contained: no network requests, no external assets, and they follow the
viewer's light or dark theme.
