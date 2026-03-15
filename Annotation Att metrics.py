#Put this in the notebook to get all accuracies for all annotation attributes:
import numpy as np, pathlib, pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = pathlib.Path("/kaggle/working/SMAT/output/test/tracking_results/mobilevitv2_track/mobilevitv2_256_128x1_ep300")
GT_DIR      = pathlib.Path("/kaggle/input/datasets/galaxythereal/uav123-tracking-dataset/anno/UAV123")
ATT_DIR     = GT_DIR / "att"

ATT_NAMES = ["SV", "ARC", "LR", "FM", "FOC", "POC", "OV", "BC", "IV", "VC", "CM", "SOB"]
ATT_FULL  = {
    "SV" : "Scale Variation",
    "ARC": "Aspect Ratio Change",
    "LR" : "Low Resolution",
    "FM" : "Fast Motion",
    "FOC": "Full Occlusion",
    "POC": "Partial Occlusion",
    "OV" : "Out-of-View",
    "BC" : "Background Clutter",
    "IV" : "Illumination Variation",
    "VC" : "Viewpoint Change",
    "CM" : "Camera Motion",
    "SOB": "Similar Object"
}

# Metric helpers
def compute_iou(pred, gt):
    px,py,pw,ph = pred[:,0],pred[:,1],pred[:,2],pred[:,3]
    gx,gy,gw,gh = gt[:,0], gt[:,1], gt[:,2], gt[:,3]
    ix1 = np.maximum(px,gx);  iy1 = np.maximum(py,gy)
    ix2 = np.minimum(px+pw,gx+gw); iy2 = np.minimum(py+ph,gy+gh)
    inter = np.maximum(0,ix2-ix1)*np.maximum(0,iy2-iy1)
    union = pw*ph+gw*gh-inter
    return np.where(union>0, inter/union, 0.0)

def center_error(pred, gt):
    pcx=pred[:,0]+pred[:,2]/2; pcy=pred[:,1]+pred[:,3]/2
    gcx=gt[:,0] +gt[:,2] /2;  gcy=gt[:,1] +gt[:,3] /2
    return np.sqrt((pcx-gcx)**2+(pcy-gcy)**2)

def success_auc(ious):
    return float(np.mean([np.mean(ious>=t) for t in np.linspace(0,1,101)]))

def precision_at(errors, thr=20):
    return float(np.mean(errors<=thr))

#  compute per-sequence metrics + load attributes
result_files = sorted([f for f in RESULTS_DIR.glob("*.txt") if "_time" not in f.name])

seq_data = []   # list of dicts: {seq, auc, prec, frames, att_flags}

for pred_file in result_files:
    seq_name   = pred_file.stem
    clean_name = seq_name.replace("uav_", "")

    gt_file  = GT_DIR / f"{clean_name}.txt"
    att_file = ATT_DIR / f"{clean_name}.txt"

    if not gt_file.exists():
        print(f"  GT missing  : {clean_name}")
        continue
    if not att_file.exists():
        print(f"  ATT missing : {clean_name}")
        continue

    try:
        # Load predictions
        try:    pred = np.loadtxt(str(pred_file), delimiter='\t').reshape(-1,4)
        except:
            try: pred = np.loadtxt(str(pred_file), delimiter=',').reshape(-1,4)
            except: pred = np.loadtxt(str(pred_file)).reshape(-1,4)

        # Load GT
        try:    gt = np.loadtxt(str(gt_file), delimiter=',').reshape(-1,4)
        except: gt = np.loadtxt(str(gt_file)).reshape(-1,4)

        # Load attribute flags (1 row x 12 cols)
        att_flags = np.loadtxt(str(att_file), delimiter=',').astype(int)
        # att_flags shape is (12,) — one binary value per attribute

        n = min(len(pred), len(gt))
        pred, gt = pred[:n], gt[:n]

        valid = (gt[:,2]>0) & (gt[:,3]>0)
        pred_v, gt_v = pred[valid], gt[valid]
        if len(pred_v) == 0:
            continue

        ious   = compute_iou(pred_v, gt_v)
        errors = center_error(pred_v, gt_v)

        seq_data.append({
            "sequence"   : seq_name,
            "auc"        : success_auc(ious),
            "prec"       : precision_at(errors),
            "frames"     : len(pred_v),
            "att_flags"  : att_flags,  
        })

    except Exception as e:
        print(f"  Error {seq_name}: {e}")

print(f"Loaded {len(seq_data)} sequences")

# per-attribute aggregation
att_results = []

for i, attr in enumerate(ATT_NAMES):
    # Sequences that HAVE this attribute
    seqs_with = [s for s in seq_data if s["att_flags"][i] == 1]
    # Sequences that DON'T have this attribute
    seqs_without = [s for s in seq_data if s["att_flags"][i] == 0]

    if len(seqs_with) == 0:
        continue

    # Mean AUC and Precision across sequences with this attribute
    auc_with  = np.mean([s["auc"]  for s in seqs_with])
    prec_with = np.mean([s["prec"] for s in seqs_with])
    auc_without  = np.mean([s["auc"]  for s in seqs_without]) if seqs_without else None
    prec_without = np.mean([s["prec"] for s in seqs_without]) if seqs_without else None

    att_results.append({
        "Attribute"      : attr,
        "Full Name"      : ATT_FULL[attr],
        "Sequences"      : len(seqs_with),
        "AUC (with)"     : round(auc_with,  4),
        "Prec (with)"    : round(prec_with, 4),
        "AUC (without)"  : round(auc_without,  4) if auc_without  else "-",
        "Prec (without)" : round(prec_without, 4) if prec_without else "-",
    })

df_att = pd.DataFrame(att_results).sort_values("AUC (with)", ascending=False)

# Print report 
print("\n" + "="*75)
print("  SMAT — UAV123 Per-Attribute Performance")
print("="*75)
print(f"  {'Attr':<6} {'Full Name':<22} {'Seqs':>5} {'AUC':>7} {'Prec@20':>8}  {'AUC(no att)':>11}")
print("  " + "-"*70)
for _, row in df_att.iterrows():
    print(f"  {row['Attribute']:<6} {row['Full Name']:<22} {row['Sequences']:>5} "
          f"{row['AUC (with)']:>7.4f} {row['Prec (with)']:>8.4f}  {str(row['AUC (without)']):>11}")
print("="*75)

# Save CSV
att_csv = "/kaggle/working/SMAT/output/UAV123_SMAT_attribute_results.csv"
df_att.to_csv(att_csv, index=False)
print(f"\n  CSV -> {att_csv}")

# Plot 
df_plot = df_att.sort_values("AUC (with)", ascending=True)
attrs   = df_plot["Attribute"].tolist()
aucs    = df_plot["AUC (with)"].tolist()
precs   = df_plot["Prec (with)"].tolist()
colors  = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(attrs)))

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("SMAT — UAV123 Per-Attribute Performance", fontsize=14, fontweight='bold')

ax = axes[0]
bars = ax.barh(attrs, aucs, color=colors)
ax.axvline(x=0.7002, color='navy', ls='--', lw=1.5, label="Overall AUC=0.700")
ax.set_xlabel("Success AUC", fontsize=12)
ax.set_title("Success AUC per Attribute", fontsize=13)
ax.set_xlim(0, 1)
ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
ax.legend(fontsize=10); ax.grid(True, axis='x', ls='--', alpha=0.5)

ax = axes[1]
bars = ax.barh(attrs, precs, color=colors)
ax.axvline(x=0.8670, color='navy', ls='--', lw=1.5, label="Overall Prec=0.867")
ax.set_xlabel("Precision @ 20px", fontsize=12)
ax.set_title("Precision per Attribute", fontsize=13)
ax.set_xlim(0, 1)
ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
ax.legend(fontsize=10); ax.grid(True, axis='x', ls='--', alpha=0.5)

plt.tight_layout()
plot_path = "/kaggle/working/SMAT/output/UAV123_SMAT_attributes.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Plot -> {plot_path}")
