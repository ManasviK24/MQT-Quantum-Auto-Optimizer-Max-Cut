import glob, csv, os

def row_to_md(r):
    # make short strings for speed & correctness flags
    q = float(r["qaoa_cut"])
    s = float(r["sa_cut"])
    e = r["exact_cut"]
    ex = f"{float(e):.3f}" if e not in (None, "", "None") else "N/A"
    tag = "✅=opt" if e not in (None, "", "None") and abs(q - float(e)) < 1e-6 else ("≈" if e not in (None, "", "None") else "—")
    return f"| {r['file']} | {r['reps']} | {r['num_runs']} | {q:.3f} | {s:.3f} | {ex} | {tag} | {float(r['qaoa_time_s']):.2f} |"

def main():
    files = sorted(glob.glob("results/qaoa_sweep*.csv"))
    out = ["| file | reps | num_runs | qaoa_cut | sa_cut | exact_cut | note | qaoa_time_s |",
           "|---|---:|---:|---:|---:|---:|:--:|---:|"]
    for fpath in files:
        with open(fpath, newline="") as f:
            rd = csv.DictReader(f)
            for r in rd:
                r = dict(r)
                r["file"] = os.path.basename(fpath)
                out.append(row_to_md(r))
    os.makedirs("results", exist_ok=True)
    with open("results/summary.md", "w", encoding="utf-8") as f:
        f.write("# Experiment summary\n\n")
        f.write("\n".join(out))
        f.write("\n")
    print("Wrote results/summary.md with a markdown table.")

if __name__ == "__main__":
    main()
