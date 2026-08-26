import pandas as pd
import matplotlib
matplotlib.use("Agg")  # saves to file instead of opening a window
import matplotlib.pyplot as plt

df = pd.read_csv("data_utils/raw_data/synthetic_officers.csv")

# Pick one officer per archetype to visualize
sample_ids = df.groupby("archetype")["officer_id"].first()

fig, axes = plt.subplots(3, 1, figsize=(10, 12))

for ax, (archetype, officer_id) in zip(axes, sample_ids.items()):
    officer_data = df[df["officer_id"] == officer_id].sort_values("day")
    ax.plot(officer_data["day"], officer_data["d2_garage_concentration"], label="D2 - Garage Concentration")
    ax.plot(officer_data["day"], officer_data["d4_surveyor_diversity"] / 30, label="D4 - Surveyor Diversity (scaled /30)")
    ax.plot(officer_data["day"], officer_data["d6_after_hours_pct"], label="D6 - After Hours %")
    ax.set_title(f"{archetype} — {officer_id}")
    ax.set_xlabel("Day")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("notebooks/trajectory_sanity_check.png", dpi=100)
print("Saved plot to notebooks/trajectory_sanity_check.png")