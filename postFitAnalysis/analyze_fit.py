#!/usr/bin/env python3
"""
CharmingFit post-fit analysis with LaTeX axis titles from YAML.

Features:
  • Reads 1D + 2D YAML config files with LaTeX titles (xtitle, ytitle)
  • Produces weighted ROOT histograms (TH1D, TH2D)
  • Saves one PDF per histogram + ROOT file in an output directory
  • Removes stat boxes and histogram titles
  • Converts phases and CKM angles to degrees automatically
"""

import argparse
import math
import ROOT
import yaml
import os
import sys

# --- CMS Style ---
def set_cms_style():
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetCanvasBorderMode(0)
    ROOT.gStyle.SetFrameBorderMode(0)
    ROOT.gStyle.SetPadBorderMode(0)
    ROOT.gStyle.SetCanvasColor(0)
    ROOT.gStyle.SetPadColor(0)
    ROOT.gStyle.SetTitleFont(62, "XYZ")
    ROOT.gStyle.SetTitleSize(0.05, "XYZ")
    ROOT.gStyle.SetLabelFont(42, "XYZ")
    ROOT.gStyle.SetLabelSize(0.04, "XYZ")
    ROOT.gStyle.SetPadTopMargin(0.06)
    ROOT.gStyle.SetPadBottomMargin(0.13)
    ROOT.gStyle.SetPadLeftMargin(0.14)
    ROOT.gStyle.SetPadRightMargin(0.04)
    ROOT.gStyle.SetTitleXOffset(1.1)
    ROOT.gStyle.SetTitleYOffset(1.3)
    ROOT.gStyle.SetTickLength(0.03, "XY")
    ROOT.gStyle.SetEndErrorSize(2)
    ROOT.gStyle.SetLineWidth(2)
    ROOT.gStyle.SetFrameLineWidth(2)
    ROOT.gStyle.SetHistLineWidth(2)
    ROOT.gStyle.SetPadTickX(1)
    ROOT.gStyle.SetPadTickY(1)
    ROOT.gStyle.SetLegendBorderSize(0)
    ROOT.gStyle.SetLegendTextSize(0.035)

#def draw_cms_label():
#    latex = ROOT.TLatex()
#    latex.SetNDC(True)
#    latex.SetTextFont(62)
#    latex.SetTextSize(0.05)
#    latex.DrawLatex(0.16, 0.92, "CMS Preliminary")

# activate style
set_cms_style()




# ----------------- Utility functions -----------------

def evaluate_weight(expr, entry_dict):
    """Safely evaluate the weight expression on a given entry."""
    try:
        safe_math = {name: getattr(math, name) for name in dir(math) if not name.startswith("_")}
        safe_globals = {"__builtins__": {}, "math": math, **safe_math}
        return float(eval(expr, safe_globals, entry_dict))
    except Exception:
        return 0.0


def create_hist(hcfg):
    """Create a ROOT histogram (1D or 2D) from a YAML config dictionary."""
    if "nbins_x" in hcfg:  # 2D histogram
        h = ROOT.TH2D(
            hcfg["name"],
            "",
            hcfg["nbins_x"],
            hcfg["xmin"],
            hcfg["xmax"],
            hcfg["nbins_y"],
            hcfg["ymin"],
            hcfg["ymax"],
        )
    else:  # 1D histogram
        h = ROOT.TH1D(
            hcfg["name"],
            "",
            hcfg["nbins"],
            hcfg["xmin"],
            hcfg["xmax"],
        )

    h.SetDirectory(0)
    h.SetStats(0)
    return h


def load_yaml(filename):
    """Load YAML config if file exists, otherwise return empty dict."""
    if filename and os.path.exists(filename):
        with open(filename, "r") as yml:
            data = yaml.safe_load(yml)
        print(f"🧾 Loaded histogram configuration from {filename}")
        return data
    else:
        if filename:
            print(f"⚠️  Config file not found: {filename}")
        return {}


def merge_configs(cfg1, cfg2):
    """Merge two YAML configs by concatenating histogram lists per group."""
    merged = dict(cfg1)
    for key, items in cfg2.items():
        merged.setdefault(key, []).extend(items)
    return merged


def get_config_variables(config):
    """Collect the set of histogram variables already defined in the YAML config."""
    vars = set()
    for items in config.values():
        for hcfg in items:
            if "variable" in hcfg:
                vars.add(hcfg["variable"])
    return vars


def branch_mode_label(branch_name):
    """Convert a pred__ branch name into a LaTeX-friendly mode label."""
    labels = {
        "B0_K0K0bar": "B^{0}\\rightarrow K^{0}\\bar K^{0}",
        "B0_K0pi0": "B^{0}\\rightarrow K^{0}\\pi^{0}",
        "B0_KpKm": "B^{0}\\rightarrow K^{+}K^{-}",
        "B0_Kppim": "B^{0}\\rightarrow K^{+}\\pi^{-}",
        "B0_pi0pi0": "B^{0}\\rightarrow \\pi^{0}\\pi^{0}",
        "B0_pippim": "B^{0}\\rightarrow \\pi^{+}\\pi^{-}",
        "Bp_K0Kp": "B^{+}\\rightarrow K^{0}K^{+}",
        "Bp_K0barpip": "B^{+}\\rightarrow \\bar K^{0}\\pi^{+}",
        "Bp_Kppi0": "B^{+}\\rightarrow K^{+}\\pi^{0}",
        "Bp_pippi0": "B^{+}\\rightarrow \\pi^{+}\\pi^{0}",
        "Bs_K0K0bar": "B_{s}\\rightarrow K^{0}\\bar K^{0}",
        "Bs_Kmpip": "B_{s}\\rightarrow K^{-}\\pi^{+}",
        "Bs_KpKm": "B_{s}\\rightarrow K^{+}K^{-}",
    }
    mode = branch_name[len("pred__"):].rsplit("__", 1)[0]
    return labels.get(mode, mode.replace("_", " "))


def append_auto_observable_configs(tree, config):
    """Ensure all predicted observables in the TTree are represented by histogram configs."""
    existing = get_config_variables(config)
    auto_group = config.setdefault("Observables", [])
    branches = sorted([b.GetName() for b in tree.GetListOfBranches() if b.GetName().startswith("pred__")])

    for branch in branches:
        if branch in existing:
            continue
        observable_type = branch.rsplit("__", 1)[-1]
        if observable_type not in {"BR", "ACP", "S", "C"}:
            continue

        label = branch_mode_label(branch)
        if observable_type == "BR":
            hcfg = {
                "variable": branch,
                "name": f"h_{branch}",
                "xtitle": f"BR #times 10^{{6}} ({label})",
                "ytitle": "Weighted entries",
                "nbins": 100,
                "xmin": 0,
                "xmax": 100,
            }
        else:
            title = "A_{CP}" if observable_type == "ACP" else observable_type
            hcfg = {
                "variable": branch,
                "name": f"h_{branch}",
                "xtitle": f"{title}({label})",
                "ytitle": "Weighted entries",
                "nbins": 100,
                "xmin": -1,
                "xmax": 1,
            }

        auto_group.append(hcfg)
    return config


def summary_label(variable_name, xtitle):
    """Return a stable short observable label for the top-margin summary."""
    observable_type = variable_name.rsplit("__", 1)[-1] if "__" in variable_name else variable_name
    label_map = {
        "ACP": "A_{CP}",
        "BR": "BR",
        "S": "S",
        "C": "C",
    }
    if observable_type in label_map:
        return label_map[observable_type]

    if "(" in xtitle:
        return xtitle.split("(", 1)[0].strip()
    if " [" in xtitle:
        return xtitle.split(" [", 1)[0].strip()
    return xtitle.strip()


# ----------------- Main execution -----------------

def main():
    parser = argparse.ArgumentParser(description="Generate weighted ROOT histograms from CharmingFit output")
    parser.add_argument("--input", required=True, help="Input ROOT file containing the TTree")
    parser.add_argument("--tree", default="CharmingFit", help="Name of the TTree inside the ROOT file (default: CharmingFit)")
    parser.add_argument("--output", required=True, help="Output directory (must not exist)")
    parser.add_argument("--weight", default="weight", help="Weight expression (Python syntax)")
    parser.add_argument("--config", default="postFitAnalysis/plot_config_1D.yaml", help="YAML file specifying 1D histograms (default: postFitAnalysis/plot_config_1D.yaml)")
    parser.add_argument("--config2D", default="postFitAnalysis/plot_config_2D.yaml", help="YAML file specifying 2D histograms (default: postFitAnalysis/plot_config_2D.yaml)")
    args = parser.parse_args()

    # Force batch mode so PDF generation works in headless environments.
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)

    # --- Output directory handling ---
    if os.path.exists(args.output):
        print(f"❌ Output directory '{args.output}' already exists. Aborting to prevent overwrite.")
        sys.exit(1)
    os.makedirs(args.output)
    print(f"📂 Created output directory: {args.output}")

    root_outfile = os.path.join(args.output, "histograms.root")

    # --- Load YAML configurations ---
    config_1d = load_yaml(args.config)
    config_2d = load_yaml(args.config2D)
    config = merge_configs(config_1d, config_2d)

    if not config:
        raise RuntimeError("❌ No histogram configuration found. Please provide at least one YAML file.")

    # --- Open ROOT input file ---
    f_in = ROOT.TFile.Open(args.input)
    if not f_in or f_in.IsZombie():
        raise RuntimeError(f"❌ Cannot open input file: {args.input}")

    tree = f_in.Get(args.tree)
    if not tree:
        raise RuntimeError(f"❌ TTree '{args.tree}' not found in {args.input}")

    config = append_auto_observable_configs(tree, config)

    # --- Prepare histograms ---
    histograms = []
    for group, items in config.items():
        for hcfg in items:
            hist = create_hist(hcfg)
            histograms.append((group, hcfg, hist))

    # --- Create ROOT output file ---
    f_out = ROOT.TFile(root_outfile, "RECREATE")
    print(f"📁 Writing histograms to: {root_outfile}")

    # --- Event loop ---
    nentries = tree.GetEntries()
    print(f"🔍 Processing {nentries} entries...")
    sys.stdout.flush()

    try:
        for i, entry in enumerate(tree):
            if i % 5000 == 0:
                print(f"  → Entry {i}/{nentries}")
                sys.stdout.flush()

            entry_dict = {br.GetName(): getattr(entry, br.GetName()) for br in tree.GetListOfBranches()}
            w = evaluate_weight(args.weight, entry_dict)
            if w == 0:
                continue

            for group, hcfg, hist in histograms:
                var = hcfg["variable"]

                if "_2D" in var:
                    base = var.replace("_2D", "")
                    if f"{base}_abs" in entry_dict and f"{base}_phase" in entry_dict:
                        x = entry_dict[f"{base}_abs"]
                        y = math.degrees(entry_dict[f"{base}_phase"])
                        hist.Fill(x, y, w)
                else:
                    if var not in entry_dict:
                        continue
                    val = entry_dict[var]
                    if "BR" in var:
                        val *= 1e6  # Plot BR * 10^6
                    if "phase" in var or var in ["gamma", "beta", "beta_s"]:
                        val = math.degrees(val)
                    hist.Fill(val, w)
    except Exception as e:
        print(f"❌ Error during event loop: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise

    # --- Write histograms ---
    print("💾 Writing histograms to ROOT file...")
    sys.stdout.flush()
    
    for group in sorted(set(g for g, _, _ in histograms)):
        f_out.mkdir(group)
        f_out.cd(group)
        for g, _, h in histograms:
            if g == group:
                h.Write()
    f_out.Close()
    f_in.Close()
    print("✅ ROOT file written successfully.")
    sys.stdout.flush()

    # --- Draw and save PDFs ---
    print("🖨️  Generating PDFs ...")
    sys.stdout.flush()
    
    try:
        c = ROOT.TCanvas("c", "", 800, 600)
        c.SetLeftMargin(0.14)
        c.SetRightMargin(0.12)
        c.SetBottomMargin(0.13)
        c.SetTopMargin(0.10)

        for i, (group, hcfg, hist) in enumerate(histograms):
            if not hist:
                print(f"  ⚠️  Skipping empty histogram {hcfg['name']}")
                continue
            
            print(f"  📊 Generating PDF {i+1}/{len(histograms)}: {hcfg['name']}...", end="", flush=True)
            
            hist.SetStats(0)
            hist.SetTitle("")

            # Axis labels from YAML, fallback to variable names
            xtitle = hcfg.get("xtitle", hcfg["variable"])
            ytitle = hcfg.get("ytitle", "Weighted entries")
            hist.GetXaxis().SetTitle(xtitle)
            hist.GetYaxis().SetTitle(ytitle)

            c.Clear()
            if isinstance(hist, ROOT.TH2):
                hist.Draw("COLZ")
            else:
                hist.SetMinimum(0)
                hist.Draw("HIST")
            
            # Draw the summary label in the top margin, aligned with the frame's left edge.
            if not isinstance(hist, ROOT.TH2):
                mean = hist.GetMean()
                sigma = hist.GetStdDev()

                obs_short = summary_label(hcfg["variable"], xtitle)
                observable_type = hcfg["variable"].rsplit("__", 1)[-1] if "__" in hcfg["variable"] else hcfg["variable"]
                
                # Format with 1 digit precision: use format like ".1f" but handle rounding
                mean_str = f"{mean:.1f}"
                sigma_str = f"{sigma:.1f}"
                if observable_type == "BR":
                    summary_text = f"{obs_short} = ({mean_str} #pm {sigma_str}) #times 10^{{-6}}"
                else:
                    summary_text = f"{obs_short} = {mean_str} #pm {sigma_str}"
                
                # Create TLatex for statistics box
                latex = ROOT.TLatex()
                latex.SetNDC(True)
                latex.SetTextFont(42)
                latex.SetTextSize(0.035)
                x_left = c.GetLeftMargin()
                y_top = 1.0 - 0.5 * c.GetTopMargin()
                latex.SetTextAlign(13)
                latex.DrawLatex(x_left, y_top, summary_text)
            
            pdf_path = os.path.join(args.output, f"{hcfg['name']}.pdf")
            c.SaveAs(pdf_path)
            print(f" ✓")
            sys.stdout.flush()

        print(f"✅ All PDFs written in: {args.output}")
        print(f"🎯 Results summary:\n   ROOT: {root_outfile}\n   PDFs: {args.output}/*.pdf")
    except Exception as e:
        print(f"\n❌ Error during PDF generation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
