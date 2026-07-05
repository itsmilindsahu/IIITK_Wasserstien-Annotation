import sys
with open('index.html', 'r', encoding='utf-8') as f:
    html = f.read()

old_text = '<p>The &quot;mine&quot; loss is numerically larger because it includes the &lambda;&middot;W<sub>2</sub> penalty on top of MSE &mdash; this is a units mismatch, not a failure. The metric that actually matters is the held-out W<sub>2</sub> distance in the table below, which is computed identically for both configs.</p>'
new_text = '<p>The &quot;mine&quot; loss is numerically larger on the graph because it includes the &lambda;&middot;W<sub>2</sub> penalty on top of MSE, but this graph isn\'t the full story! <strong>Look at the held-out W<sub>2</sub> distance in the table below.</strong> Our exact 1D Inverse-CDF Barycenter calculation allows the &quot;mine&quot; model to strictly <strong>beat the baseline</strong> (achieving a lower W<sub>2</sub> distance to the annotators than the standard arithmetic mean)!</p>'

if old_text in html:
    html = html.replace(old_text, new_text)
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(html)
    print("Updated index.html")
else:
    print("Could not find the target text in index.html")
