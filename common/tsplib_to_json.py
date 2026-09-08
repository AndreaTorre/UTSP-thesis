# tsplib_to_json.py — .tsp EUC_2D -> schema nodi_*.json (distanze euclidee float).
import sys, json, math
def convert(tsp_path, out_path):
    name, coords, reading = None, {}, False
    for line in open(tsp_path, encoding="utf-8", errors="ignore"):
        s = line.strip()
        if s.startswith("NAME"):
            name = s.split(":", 1)[1].strip()
        elif s.startswith("NODE_COORD_SECTION"):
            reading = True
        elif s in ("EOF", "DISPLAY_DATA_SECTION", "") and reading:
            reading = False
        elif reading:
            p = s.split()
            if len(p) >= 3 and p[0].lstrip("-").isdigit():
                coords[int(p[0])] = [float(p[1]), float(p[2])]
    if not coords:
        raise ValueError(f"{tsp_path}: nessuna coordinata trovata")
    ids = sorted(coords)
    dist = {str(i): {str(j): math.dist(coords[i], coords[j]) for j in ids} for i in ids}
    json.dump({"source_instance": name or tsp_path, "num_nodes": len(ids),
               "node_ids": ids, "coordinates": {str(i): coords[i] for i in ids},
               "distance_matrix": dist}, open(out_path, "w"))
    return len(ids)
if __name__ == "__main__":
    print(convert(sys.argv[1], sys.argv[2]), "nodi")
