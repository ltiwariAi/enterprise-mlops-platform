# Docs

Customer-facing materials for the Enterprise MLOps Platform.

## Presentation

`presentation.md` is a [Marp](https://marp.app/) slide deck. Render it with:

```bash
# PDF (best for sharing)
npx @marp-team/marp-cli@latest docs/presentation.md --pdf

# Self-contained HTML (best for web viewing)
npx @marp-team/marp-cli@latest docs/presentation.md --html

# PowerPoint (best for editing)
npx @marp-team/marp-cli@latest docs/presentation.md --pptx
```

Or preview live in VS Code with the **Marp for VS Code** extension.

Output is written next to the source file (`presentation.pdf`, `presentation.html`, `presentation.pptx`).
