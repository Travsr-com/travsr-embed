# Contributing

## Merge order with `travsr`

This repo path-depends on `travsr-plugin-protocol` and `travsr-plugin-sdk`
from `Travsr-com/travsr@master` (CI checks out the sibling repo and symlinks
it in - see `.github/workflows/ci.yml`).

If a PR here needs a travsr-side protocol change, that change must be merged
to `travsr` master **first**. Until then, CI on this PR stays red with an
unresolved-import error (the preflight step names the exact travsr SHA it
compiled against, so this is a one-look diagnosis, not a debugging session).
That is expected, not a bug in this repo - land the paired travsr PR, then
re-run CI here.
