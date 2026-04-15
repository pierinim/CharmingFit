#!/usr/bin/env python3
"""
Inspect the structure of a ROOT file and list all TTrees and branches.
Usage:
    python inspect_tree.py myfile.root
"""

import argparse
import uproot

def main():
    parser = argparse.ArgumentParser(description="Inspect a ROOT file and list all TTree branches.")
    parser.add_argument("filename", help="Path to the ROOT file")
    args = parser.parse_args()

    try:
        f = uproot.open(args.filename)
    except Exception as e:
        print(f"❌ Could not open file {args.filename}: {e}")
        return

    print(f"📂 File: {args.filename}\n")
    for name, obj in f.items():
        if isinstance(obj, uproot.behaviors.TTree.TTree):
            print(f"🌳 TTree: {name}")
            branches = list(obj.keys())
            for br in branches:
                print(f"   • {br}")
            print()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Inspect the structure of a ROOT file and list all TTrees and branches.
Usage:
    python inspect_tree.py myfile.root
"""

import argparse
import uproot

def main():
    parser = argparse.ArgumentParser(description="Inspect a ROOT file and list all TTree branches.")
    parser.add_argument("filename", help="Path to the ROOT file")
    args = parser.parse_args()

    try:
        f = uproot.open(args.filename)
    except Exception as e:
        print(f"❌ Could not open file {args.filename}: {e}")
        return

    print(f"📂 File: {args.filename}\n")
    for name, obj in f.items():
        if isinstance(obj, uproot.behaviors.TTree.TTree):
            print(f"🌳 TTree: {name}")
            branches = list(obj.keys())
            for br in branches:
                print(f"   • {br}")
            print()

if __name__ == "__main__":
    main()
