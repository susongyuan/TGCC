import html, json, math
from pathlib import Path


def export_view(snapshot, path):
    nodes = snapshot["nodes"]
    positions = {}
    svg = []
    for i, n in enumerate(nodes):
        pose = n.get("pose")
        if pose is not None:
            x = 400 + float(pose[0][3]) * 40
            y = 300 - float(pose[1][3]) * 40
        else:
            x = 400 + 240 * math.cos(i * 2 * math.pi / max(1, len(nodes)))
            y = 300 + 200 * math.sin(i * 2 * math.pi / max(1, len(nodes)))
        positions[n["entity_id"]] = (x, y)
    for edge in snapshot["edges"]:
        if edge["source_id"] not in positions or edge["target_id"] not in positions:
            continue
        x, y = positions[edge["source_id"]]
        u, v = positions[edge["target_id"]]
        label = edge["relation_type"]
        if edge.get("relative_attributes", {}).get("value") is False:
            label += " = false (evidence, not positive adjacency)"
        geom = edge.get("relative_attributes", {}).get("geometry9")
        if geom is not None:
            label += " EE translation " + str([round(a, 3) for a in geom[:3]])
        svg.append(
            f'<line x1="{x}" y1="{y}" x2="{u}" y2="{v}" stroke="#798a9b"/><text x="{(x+u)/2}" y="{(y+v)/2}" font-size="10">{html.escape(label)}</text>'
        )
    for node in nodes:
        x, y = positions[node["entity_id"]]
        color = "#438bba" if node.get("validity") else "#999999"
        svg.append(
            f'<circle cx="{x}" cy="{y}" r="12" fill="{color}"/><text x="{x+15}" y="{y}">{html.escape(node.get("instance_id",node["entity_id"])+(" ["+node.get("attributes",{}).get("state")+"]" if node.get("attributes",{}).get("state") else ""))}</text>'
        )
    payload = html.escape(json.dumps(snapshot, indent=2))
    Path(path).write_text(
        '<!doctype html><meta charset="utf-8"><title>TGR graph snapshot</title><p>World XY projection when pose available; otherwise schematic layout. Local relations explicitly in EE frame. Not a rendering of robot control.</p><svg viewBox="0 0 800 600">'
        + "".join(svg)
        + "</svg><details><summary>Exact state / evidence</summary><pre>"
        + payload
        + "</pre></details>"
    )


def export_runtime(runtime, env, directory):
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    s = runtime.environments[env]
    for name in ("global", "local", "task"):
        export_view(s[name].snapshot(), out / (name + ".html"))
        with (out / (name + ".jsonl")).open("w") as f:
            for frame in s[name].get_history():
                f.write(json.dumps(frame) + "\n")
    (out / "events.json").write_text(json.dumps(s["events"].query(), indent=2))
