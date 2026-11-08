import hashlib
import json
from pathlib import Path


ROOT = Path("/home/u202522802045/share/zpb")
OUT = ROOT / "task38_corrected_pbmc002_ablation"
FULL = ROOT / "final_dmode_b2full_updated_pbmc002"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    all_inputs = {}
    for fold in (f"target_D{i}" for i in range(1, 5)):
        cache = OUT / "cache" / fold
        manifest_path = cache / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        artifact = FULL / "validated_source_artifacts" / fold
        hroot = artifact / "harmonized"
        sources = manifest["sources"]
        assert manifest["gene_order_source"] == str(artifact / "gene_ids.npy")
        assert sha(artifact / "gene_ids.npy") == manifest["input_sha256"][str(artifact / "gene_ids.npy")]
        input_hashes = dict(manifest.get("input_sha256", {}))
        for path in [artifact / "B_D.npz", artifact / "gene_ids.npy", artifact / "manifest.json"]:
            input_hashes[str(path)] = sha(path)
        for path in [hroot / name / "rna.npy" for name in sources] + [hroot / name / "metadata.npz" for name in sources]:
            input_hashes[str(path)] = sha(path)
        for name in sources:
            for split in ("train", "val"):
                path = artifact / f"{name}_F_{split}.npy"
                assert path.is_file(), path
                input_hashes[str(path)] = sha(path)
        for name in ("gene_ids.npy", "target_rna_test.npy", "target_test_index.npy", "target_cells.tsv"):
            assert sha(cache / name) == manifest["cache_sha256"][name]
        manifest["source_harmonized_root"] = str(hroot)
        manifest["source_files_sha256"] = {
            str(p): input_hashes[str(p)]
            for name in sources
            for p in (hroot / name / "rna.npy", hroot / name / "metadata.npz")
        }
        write(manifest_path, manifest)

        fold_manifest = artifact / "fold_manifest.json"
        if not fold_manifest.exists():
            fold_manifest.write_text((artifact / "manifest.json").read_text())
        inputs = dict(input_hashes)
        inputs[str(fold_manifest)] = sha(fold_manifest)
        all_inputs.update(inputs)
        write(OUT / "data_manifest" / fold / "manifest.json", {
            "fold": fold,
            "target": manifest["target"],
            "sources": sources,
            "target_identity": manifest["target_identity"],
            "source_artifact_root": str(artifact),
            "source_harmonized_root": str(hroot),
            "source_files_sha256": inputs,
            "target_cache_manifest_sha256": sha(manifest_path),
            "target_atac_accessed": False,
            "target_labels_accessed": False,
        })
        print("ADAPTER_READY", fold, "sources", len(sources), "hashes", len(inputs), flush=True)
    write(OUT / "data_manifest.json", {
        "protocol": "corrected HDa quartet, fold-specific Full B2 source artifacts",
        "source_files_sha256": all_inputs,
        "target_atac_accessed": False,
        "target_labels_accessed_before_embedding_freeze": False,
    })


if __name__ == "__main__":
    main()
