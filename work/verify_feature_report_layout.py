import ast
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
nb = json.loads(Path('EPF/4_Features_select/4_feature_selection_report.ipynb').read_text(encoding='utf-8'))
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        compile(''.join(cell['source']), cell['id'], 'exec')
ns = {}
exec(''.join(nb['cells'][1]['source']), ns)
for node in ast.parse(''.join(nb['cells'][22]['source'])).body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SOURCE_DISPLAY_NAMES' for t in node.targets):
        names = list(ast.literal_eval(node.value).values())
ns['SOURCE_ORDER'] = names
ns['SOURCE_COLOURS'] = {n: matplotlib.colors.to_hex(matplotlib.colormaps['tab20'](i / 20)) for i, n in enumerate(names)}
ns['TOP_SOURCE_FEATURES'] = 100
exec(''.join(nb['cells'][23]['source']), ns)
np, pd = ns['np'], ns['pd']
values = np.random.default_rng(42).dirichlet(np.ones(len(names)), size=96)
collection = {o: pd.DataFrame(values, index=[f'h{i + 1}' for i in range(96)], columns=names) for o in ns['OBJECTIVES']}
original_savefig = matplotlib.figure.Figure.savefig
checks = []


def checked_savefig(fig, *args, **kwargs):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend = fig.legends[0].get_window_extent(renderer)
    ax = fig.axes[0]
    title = ax.title.get_window_extent(renderer)
    assert legend.y0 > title.y1, (legend, title)
    assert legend.x0 >= 0 and legend.x1 <= fig.bbox.width, legend
    assert ax.get_position().height > 0.4
    assert len(ax.patches) == 96 * len(names)
    checks.append(tuple(fig.get_size_inches()))
    return original_savefig(fig, *args, **kwargs)


matplotlib.figure.Figure.savefig = checked_savefig
captured = []
ns['display'] = lambda image: captured.append(image.data)
for weighting, ylabel in [('count-based', 'Share of top 100 features (%)'), ('importance-weighted', 'Share of top-100 feature gain (%)')]:
    for objective in ns['OBJECTIVES']:
        ns['show_source_composition'](objective, collection, weighting, ylabel)
Path('work/feature-source-portrait-preview.png').write_bytes(captured[0])
print(f'All notebook cells compile; verified {len(checks)} layouts at {checks[0]} inches, with legend above title and axes.')
print('Preview uses synthetic composition; real report inputs are incomplete locally.')
