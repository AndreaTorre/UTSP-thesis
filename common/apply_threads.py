import re, sys
F = "gurobi_models.py"
src = open(F).read()
if "TESI_GRB_THREADS" in src:
    print("già applicato"); sys.exit(0)
repl = 'int(os.environ.get("TESI_GRB_THREADS", "1"))'  # NOTA: default 1 = tesi invariata
new, n1 = re.subn(r'(\.Params\.Threads\s*=\s*)1\b', r'\g<1>' + repl, src)
new, n2 = re.subn(r'(setParam\(\s*[\'"]Threads[\'"]\s*,\s*)1\b', r'\g<1>' + repl, new)
tot = n1 + n2
if tot == 0:
    print("NESSUN match — controlla la forma: grep -n Threads gurobi_models.py"); sys.exit(1)
if "import os" not in new:
    new = "import os\n" + new
open(F, "w").write(new)
print(f"patchato {tot} occorrenza/e")
