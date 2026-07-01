# posts/ — blog posts + their assets

Posts bundle with their assets under `posts/`. One post = a dated markdown file
(`YYYY-MM-DD-slug.md`) plus shared `posts/assets/`.

## Assets layout

- `posts/assets/data/*.csv` — chart source data (checked in).
- `posts/assets/make_plots.py` — seaborn plot script. Run `./posts/assets/make_plots.py`
  (shebang'd `uv run --with seaborn`; no repo dependency). Add a chart = add a CSV +
  a function + one call.
- `posts/assets/plots/*.svg` — generated output, **gitignored** (regenerate on demand).

Charts use the dataviz-skill palette, validated. Static SVG for the site — no
hover/dark-mode/table-view layer (that's the interactive-HTML path).

## Attribution: link every specific number to its config page

Every specific number claim traces to the config page that produced it. In data
CSVs, carry the link as **its own column, one per number** — not an aggregate
`tags/model/#slug` page. A row with two claims (base + MTP) gets `base_url` +
`mtp_url`; a single-value row gets one `source_url`.

URLs are **relative to `gauravmm.github.io/autobench/`**: config pages are
`configs/<name>/` (permalink `/configs/:name/`, `:name` = the `_configs/*.md`
filename without extension).

**Finding the config for a number:** match the claim against `decode_toks` in
`_configs/`. Note `decode_toks` is stored to 2 decimals but the post rounds to 1
(e.g. `154.66` → `154.7`), so grep the 2-dp value or eyeball the model·quant·engine·conc run.
