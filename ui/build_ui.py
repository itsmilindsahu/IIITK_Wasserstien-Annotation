import json, os

here = os.path.dirname(__file__)
with open(os.path.join(here, "..", "results", "results.json")) as f:
    data = json.load(f)

with open(os.path.join(here, "template.html")) as f:
    tpl = f.read()

out = tpl.replace("__DATA_JSON__", json.dumps(data))

with open(os.path.join(here, "index.html"), "w") as f:
    f.write(out)

print("wrote ui/index.html")
