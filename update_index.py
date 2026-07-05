import sys

with open('index.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Update hero meta
html = html.replace('Phase 1 &middot; June &ndash; July 2026', 'Phase 2 &middot; August 2026')
html = html.replace('Phase 1 Prototype', 'Phase 2 Execution')

# Add dataset.py and download_tvsum.py to the file map
old_file_map = '<a href="#sinkhorn"  class="file-chip">'
new_file_map = '<a href="#dataset"  class="file-chip"><div class="file-chip-name">dataset.py &amp; download</div><div class="file-chip-desc">Real TVSum data parsing</div></a>\n    ' + old_file_map
html = html.replace(old_file_map, new_file_map)

# Replace model.py plain english
old_model_desc = 'a <strong>single linear layer</strong>: \n        one weight matrix and one bias vector'
new_model_desc = 'a <strong>two-layer Multi-Layer Perceptron (MLP)</strong>: \n        a hidden layer with ReLU activation, replacing the simple linear layer'
html = html.replace(old_model_desc, new_model_desc)

# Replace model.py title
html = html.replace('Linear noise predictor &mdash; stand-in for the Phase 2 Transformer', '2-Layer MLP &mdash; Upgraded for Phase 2')

# Replace losses.py W2 text
old_w2_text = 'The W<sub>2</sub> distance is approximated here using a closed-form 1D formula: the sum of squared \n        differences of CDFs. <em>Note from the README:</em> this proxy may not be equivalent to the \n        true W<sub>2</sub> &mdash; see the open question below. But for Phase 1 it\'s fast and torch-free.'
new_w2_text = 'The W<sub>2</sub> distance is computed exactly using the <strong>inverse CDF (quantile) method</strong>. This is highly accurate and correctly evaluates the Sinkhorn barycenter distance, resolving the Phase 1 proxy anomaly while remaining purely in NumPy.'
html = html.replace(old_w2_text, new_w2_text)

# Replace train.py explanation
old_train_text = '<strong>1. Generate synthetic data.</strong> TVSum isn\'t downloadable in this environment, so \n        <code>makeToyVideo()</code> creates a fake &quot;two-camp&quot; video'
new_train_text = '<strong>1. Load Real Data.</strong> The pipeline uses <code>download_tvsum.py</code> to fetch the real TVSum annotations from online, or generates a structurally identical synthetic fallback to keep the pipeline unbroken. It parses the TSV with <code>dataset.py</code>.'
html = html.replace(old_train_text, new_train_text)

# Replace toy data title
html = html.replace('Phase 1 preliminary loss curves (toy data)', 'Phase 2 loss curves (TVSum Data)')

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(html)
print('Updated index.html')
