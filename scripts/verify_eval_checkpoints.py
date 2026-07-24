#!/usr/bin/env python3
"""Verify every checkpoint required by the ego-pose evaluation plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickletools
import struct
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Checkpoint:
    group: str
    label: str
    relative_path: str
    size: int
    expected_hash: str | None = None
    gated: bool = False


C = Checkpoint


def _reviv_checkpoints() -> tuple[Checkpoint, ...]:
    common_sizes = {
        "reviv_tok_cam.pth": 718_878_062,
        "reviv_tok_gaze.pth": 718_826_027,
        "reviv_tok_lhand.pth": 720_085_992,
        "reviv_tok_rhand.pth": 720_086_056,
        "norm_stats/body_mean.npy": 632,
        "norm_stats/body_std.npy": 632,
        "norm_stats/cam_mean.npy": 200,
        "norm_stats/cam_std.npy": 200,
        "norm_stats/lhand_mean.npy": 152,
        "norm_stats/lhand_std.npy": 152,
        "norm_stats/rhand_mean.npy": 152,
        "norm_stats/rhand_std.npy": 152,
    }
    set_specific = {
        "metric_depth": {
            "reviv_main.pth": 2_574_931_428,
            "reviv_tok_body.pth": 720_220_907,
        },
        "reviv_500b": {
            "reviv_main.pth": 1_687_821_271,
            "reviv_tok_body.pth": 720_085_675,
        },
    }
    records: list[Checkpoint] = []
    for set_name, unique_sizes in set_specific.items():
        for filename, size in {**unique_sizes, **common_sizes}.items():
            records.append(
                C(
                    f"ReViV {set_name}",
                    filename,
                    f"reviv/{set_name}/{filename}",
                    size,
                )
            )
    return tuple(records)


REVIV_CHECKPOINTS = _reviv_checkpoints()
CHECKPOINTS = (
    C(
        "DA3 / DA3-Streaming",
        "DA3 1.1 README",
        "da3/DA3NESTED-GIANT-LARGE-1.1/README.md",
        4_976,
        "ae51281b510e7919abcbb6c65ddc51fec6fd5d91",
    ),
    C(
        "DA3 / DA3-Streaming",
        "DA3 1.1 config",
        "da3/DA3NESTED-GIANT-LARGE-1.1/config.json",
        3_113,
        "0c9ba5b7641a989502f845180f03b09c632f8f09",
    ),
    C(
        "DA3 / DA3-Streaming",
        "DA3 Nested Giant-Large 1.1",
        "da3/DA3NESTED-GIANT-LARGE-1.1/model.safetensors",
        6_759_558_100,
        "8ebe871a022ed58d2fc8fdfb2ebdb31d57b60fe39611c849095851a7b7c6020c",
    ),
    C(
        "LingBot-Map",
        "README",
        "lingbot-map/README.md",
        26_826,
        "985cb20e109a39ab2481b854d2740c11d0a146d4",
    ),
    C(
        "LingBot-Map",
        "long model",
        "lingbot-map/lingbot-map-long.pt",
        4_632_303_465,
        "832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409",
    ),
    C(
        "LingBot-Map",
        "sky segmentation",
        "lingbot-map/skyseg_batch.onnx",
        175_997_119,
        "b09c0f6cf79e1caa2591b946b659487bd7c8208caddd3f80680cbb169617e378",
    ),
    C(
        "VGGT-SLAM 2.0",
        "VGGT README",
        "vggt-slam/vggt-1b/README.md",
        2_060,
        "b8e9a3f570877961599dc5ca652630296467f7a5",
    ),
    C(
        "VGGT-SLAM 2.0",
        "VGGT config",
        "vggt-slam/vggt-1b/config.json",
        62,
        "303bf21400e2723e8ff9c0c7ceb6d86859b1ddeb",
    ),
    C(
        "VGGT-SLAM 2.0",
        "VGGT-1B",
        "vggt-slam/vggt-1b/model.pt",
        5_026_874_952,
        "d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0",
    ),
    C(
        "VGGT-SLAM 2.0",
        "SALAD loop closure",
        "vggt-slam/dino_salad.ckpt",
        352_040_378,
    ),
    C(
        "DROID-SLAM",
        "DROID model",
        "droid-slam/droid.pth",
        16_061_701,
        "46476ef64cde45a97504910d6f3de2eef7b398ec1c6e4e668815c29076024526",
    ),
    *REVIV_CHECKPOINTS,
    C(
        "ReViV metric_depth",
        "Cosmos-1.0 README",
        "reviv/cosmos/Cosmos-1.0-Tokenizer-DV8x16x16/README.md",
        30_903,
    ),
    C(
        "ReViV metric_depth",
        "Cosmos-1.0 autoencoder",
        "reviv/cosmos/Cosmos-1.0-Tokenizer-DV8x16x16/autoencoder.jit",
        223_576_773,
        gated=True,
    ),
    C(
        "ReViV metric_depth",
        "Cosmos-1.0 config",
        "reviv/cosmos/Cosmos-1.0-Tokenizer-DV8x16x16/config.json",
        54,
        gated=True,
    ),
    C(
        "ReViV metric_depth",
        "Cosmos-1.0 decoder",
        "reviv/cosmos/Cosmos-1.0-Tokenizer-DV8x16x16/decoder.jit",
        132_042_180,
        gated=True,
    ),
    C(
        "ReViV metric_depth",
        "Cosmos-1.0 encoder",
        "reviv/cosmos/Cosmos-1.0-Tokenizer-DV8x16x16/encoder.jit",
        92_292_848,
        gated=True,
    ),
    C(
        "ReViV metric_depth",
        "Cosmos-1.0 model config",
        "reviv/cosmos/Cosmos-1.0-Tokenizer-DV8x16x16/model_config.yaml",
        92,
        gated=True,
    ),
    C(
        "ViPE 1.2.0",
        "DA3Metric README",
        "vipe/da3metric-large/README.md",
        4_789,
        "0856df0c9e59168474ff856705849ca6738cb713",
    ),
    C(
        "ViPE 1.2.0",
        "DA3Metric config",
        "vipe/da3metric-large/config.json",
        847,
        "3f50fc09637d19fdf0998946aa383bc377de9ade",
    ),
    C(
        "ViPE 1.2.0",
        "DA3Metric model",
        "vipe/da3metric-large/model.safetensors",
        1_336_734_448,
        "bbea5b0b3ee389849cffa7ddae89de064a90abd2b055fc5aa99aac68db324776",
    ),
    C(
        "ViPE 1.2.0",
        "DA3-GIANT README",
        "vipe/da3-giant/README.md",
        4_881,
        "f7738bb73d0709ede4af55b239b16c097b5599ba",
    ),
    C(
        "ViPE 1.2.0",
        "DA3-GIANT config",
        "vipe/da3-giant/config.json",
        1_880,
        "eea36247d758867adc7f4b4ac0d7d7d79105cc50",
    ),
    C(
        "ViPE 1.2.0",
        "DA3-GIANT model",
        "vipe/da3-giant/model.safetensors",
        5_422_814_644,
        "1e47a08338ca73a6d6a21d37fd060b26b993b672bc6ddf6295fe474df2592001",
    ),
    C(
        "ViPE 1.2.0",
        "GeoCalib pinhole",
        "vipe/geocalib/geocalib-pinhole.tar",
        116_074_121,
    ),
    C(
        "ViPE 1.2.0",
        "SAM ViT-B",
        "vipe/track-anything/sam_vit_b_01ec64.pth",
        375_042_383,
    ),
    C(
        "ViPE 1.2.0",
        "DeAOT",
        "vipe/track-anything/R50_DeAOTL_PRE_YTB_DAV.pth",
        236_513_959,
    ),
    C(
        "ViPE 1.2.0",
        "GroundingDINO Swin-T",
        "vipe/grounding-dino/groundingdino_swint_ogc.pth",
        693_997_677,
        "3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799",
    ),
    C(
        "ViPE 1.2.0",
        "BERT config",
        "vipe/bert-base-uncased/config.json",
        570,
        "45a2321a7ecfdaaf60a6c1fd7f5463994cc8907d",
    ),
    C(
        "ViPE 1.2.0",
        "BERT weights",
        "vipe/bert-base-uncased/model.safetensors",
        440_449_768,
        "68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3",
    ),
    C(
        "ViPE 1.2.0",
        "BERT tokenizer",
        "vipe/bert-base-uncased/tokenizer.json",
        466_062,
        "949a6f013d67eb8a5b4b5b46026217b888021b88",
    ),
    C(
        "ViPE 1.2.0",
        "BERT tokenizer config",
        "vipe/bert-base-uncased/tokenizer_config.json",
        48,
        "e5c73d8a50df1f56fb5b0b8002d7cf4010afdccb",
    ),
    C(
        "ViPE 1.2.0",
        "BERT vocabulary",
        "vipe/bert-base-uncased/vocab.txt",
        231_508,
        "fb140275c155a9c7c5a3b3e0e77a9e839594a938",
    ),
    C(
        "EgoM2P",
        "base model",
        "egom2p/checkpoint-main.pth",
        4_818_565_952,
    ),
    C(
        "EgoM2P",
        "camera tokenizer",
        "egom2p/checkpoint-cam.pth",
        2_156_231_716,
    ),
    C(
        "EgoM2P / ReViV-256",
        "Cosmos README",
        "egom2p/cosmos-tokenizer/README.md",
        21_412,
        "d7338fbf966ac341b466f5d1d10c0e62a67421b6",
    ),
    C(
        "EgoM2P / ReViV-256",
        "Cosmos config",
        "egom2p/cosmos-tokenizer/config.json",
        54,
        "bcad561de5279b772db7dd4b76b11d07ddc7ced1",
    ),
    C(
        "EgoM2P / ReViV-256",
        "Cosmos model config",
        "egom2p/cosmos-tokenizer/model_config.yaml",
        92,
        "5be0900a5551d62cb295248f76186f2f665c51d0",
    ),
    C(
        "EgoM2P / ReViV-256",
        "Cosmos autoencoder",
        "egom2p/cosmos-tokenizer/autoencoder.jit",
        211_093_069,
    ),
    C(
        "EgoM2P / ReViV-256",
        "Cosmos encoder",
        "egom2p/cosmos-tokenizer/encoder.jit",
        86_641_076,
        "9a0e8459ab5e0ecfd0c00f215571de43e368f090c16adeb1a69fa835177bdea6",
    ),
    C(
        "EgoM2P / ReViV-256",
        "Cosmos decoder",
        "egom2p/cosmos-tokenizer/decoder.jit",
        125_210_440,
        "a6b82dd6f4d489bbeb728e54c828d5a676f17e6eba9b9dfe2dc7839928bee73f",
    ),
    C(
        "EgoEgo",
        "gravity network",
        "egoego/stage1_gravitynet_2000.pt",
        31_814_605,
    ),
    C(
        "EgoEgo",
        "ARES head network",
        "egoego/stage1_headnet_ares_250.pt",
        52_191_441,
    ),
    C(
        "EgoEgo",
        "GIMO head network",
        "egoego/stage1_headnet_gimo_1000.pt",
        52_191_377,
    ),
    C(
        "EgoEgo",
        "KinPoly head network",
        "egoego/stage1_headnet_kinpoly_1000.pt",
        52_191_377,
    ),
    C(
        "EgoEgo",
        "stage-2 diffusion model",
        "egoego/stage2_diffusion_4.pt",
        88_417_189,
    ),
    C(
        "MegaSaM",
        "camera tracker",
        "mega-sam/megasam_final.pth",
        20_812_149,
        "750ba60eb19054829263f03163d96120e28134c4",
    ),
    C(
        "MegaSaM",
        "Depth Anything V1",
        "mega-sam/depth-anything-v1/checkpoints/depth_anything_vitl14.pth",
        1_341_401_882,
        "6c6a383e33e51c5fdfbf31e7ebcda943973a9e6a1cbef1564afe58d7f2e8fe63",
    ),
    C(
        "MegaSaM",
        "UniDepth README",
        "mega-sam/unidepth-v2-vitl14/README.md",
        397,
        "e024a3978f71577a6cc1db48f6a31d63b94d50c2",
    ),
    C(
        "MegaSaM",
        "UniDepth config",
        "mega-sam/unidepth-v2-vitl14/config.json",
        1_329,
        "9161783235097a46641d2770ff857db20f9016fe",
    ),
    C(
        "MegaSaM",
        "UniDepth model",
        "mega-sam/unidepth-v2-vitl14/model.safetensors",
        1_452_916_608,
        "13952af59d28d21d4de5873f8b8bf6d679c5dd031dbf7f9e5818cbeca579f6af",
    ),
    C(
        "MegaSaM",
        "RAFT",
        "mega-sam/raft-things.pth",
        21_108_000,
    ),
    C(
        "HaWoR",
        "hand detector",
        "hawor/external/detector.pt",
        53_582_271,
        "5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033",
    ),
    C(
        "HaWoR",
        "masked DROID",
        "hawor/external/droid.pth",
        16_061_701,
        "46476ef64cde45a97504910d6f3de2eef7b398ec1c6e4e668815c29076024526",
    ),
    C(
        "HaWoR",
        "Metric3D",
        "hawor/external/metric_depth_vit_large_800k.pth",
        1_647_972_663,
        "15328ffc42b528b95f188687418f6f03b3f123eb34ccdbd686c112abbea6d972",
    ),
    C(
        "HaWoR",
        "main model",
        "hawor/hawor/checkpoints/hawor.ckpt",
        3_267_481_572,
        "4d1cc43853c190d6f2c10d9b6295c73109f0faf9ef41ac817a2b31d94b4823f2",
    ),
    C(
        "HaWoR",
        "infiller",
        "hawor/hawor/checkpoints/infiller.pt",
        418_603_497,
        "30715e7e72e91d4e164bb762c7ea613dcff5448dbda5fabf40b4054e408cc5c2",
    ),
    C(
        "HaWoR",
        "model config",
        "hawor/hawor/model_config.yaml",
        2_743,
        "6796f59855c75391c503c47ed166fe28941db2ac",
    ),
    C(
        "HaWoR / MANO",
        "MANO left-hand model",
        "MANO/MANO_LEFT.pkl",
        3_821_391,
    ),
    C(
        "HaWoR / MANO",
        "MANO right-hand model",
        "MANO/MANO_RIGHT.pkl",
        3_821_356,
    ),
    C(
        "VGGT-Omega",
        "VGGT-Omega-1B-512",
        "vggt-omega/VGGT-Omega-1B-512/vggt_omega_1b_512.pt",
        4_576_706_117,
        "c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934",
        gated=True,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "ckpts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON result path (default: ROOT/verification.json)",
    )
    parser.add_argument(
        "--hash",
        action="store_true",
        help="Compute local SHA-256 and compare every published upstream hash.",
    )
    return parser.parse_args()


def format_error(path: Path, file_size: int) -> str | None:
    try:
        if path.name in {"MANO_LEFT.pkl", "MANO_RIGHT.pkl"}:
            last_opcode = None
            for opcode, _argument, _position in pickletools.genops(path.read_bytes()):
                last_opcode = opcode.name
            if last_opcode != "STOP":
                return "MANO pickle does not terminate with STOP"
            return None

        if path.suffix != ".safetensors":
            return None
        with path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                return "truncated safetensors length"
            (header_length,) = struct.unpack("<Q", raw_length)
            if header_length <= 2 or 8 + header_length > file_size:
                return f"invalid safetensors header length {header_length}"
            header = json.loads(handle.read(header_length))
            if not isinstance(header, dict):
                return "safetensors header is not a JSON object"
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return str(error)
    return None


def hashes(path: Path, size: int) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1()
    git_blob.update(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            sha256.update(block)
            git_blob.update(block)
    return sha256.hexdigest(), git_blob.hexdigest()


def state_files(path: Path) -> list[str]:
    suffixes = (".aria2", ".range.part", ".range.done", ".range.json")
    return [str(Path(f"{path}{suffix}")) for suffix in suffixes if Path(f"{path}{suffix}").exists()]


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output = (args.output or root / "verification.json").resolve()
    results: list[dict[str, object]] = []

    for checkpoint in CHECKPOINTS:
        path = root / checkpoint.relative_path
        states = state_files(path)
        actual_size = path.stat().st_size if path.is_file() else None
        if actual_size == checkpoint.size and not states:
            status = "complete"
        elif checkpoint.gated and actual_size is None and not states:
            status = "gated-missing"
        elif actual_size is None and not states:
            status = "missing"
        elif states:
            status = "partial"
        else:
            status = "wrong-size"

        format_error_message = None
        local_sha256 = None
        local_git_blob_sha1 = None
        expected_hash_kind = None
        hash_match = None

        if status == "complete":
            format_error_message = format_error(path, checkpoint.size)
            if format_error_message:
                status = "invalid-format"
            if args.hash:
                local_sha256, local_git_blob_sha1 = hashes(path, checkpoint.size)
                if checkpoint.expected_hash:
                    expected_hash_kind = (
                        "sha256" if len(checkpoint.expected_hash) == 64
                        else "git-blob-sha1"
                    )
                    actual_hash = (
                        local_sha256
                        if expected_hash_kind == "sha256"
                        else local_git_blob_sha1
                    )
                    hash_match = actual_hash == checkpoint.expected_hash
                    if not hash_match:
                        status = "hash-mismatch"

        results.append(
            {
                "group": checkpoint.group,
                "label": checkpoint.label,
                "path": str(path),
                "expected_size": checkpoint.size,
                "actual_size": actual_size,
                "status": status,
                "gated": checkpoint.gated,
                "state_files": states,
                "expected_hash": checkpoint.expected_hash,
                "expected_hash_kind": expected_hash_kind,
                "local_sha256": local_sha256,
                "local_git_blob_sha1": local_git_blob_sha1,
                "hash_match": hash_match,
                "format_error": format_error_message,
            }
        )

    counts: dict[str, int] = {}
    for result in results:
        status = str(result["status"])
        counts[status] = counts.get(status, 0) + 1

    manifest = {
        "root": str(root),
        "hashes_computed": args.hash,
        "counts": counts,
        "files": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    for result in results:
        print(
            f"{str(result['status']):14} "
            f"{str(result['group']):20} "
            f"{result['path']}"
        )
    print(f"\nSummary: {json.dumps(counts, sort_keys=True)}")
    print(f"Manifest: {output}")

    non_gated_failures = [
        result
        for result in results
        if result["status"] != "complete" and not result["gated"]
    ]
    gated_failures = [
        result
        for result in results
        if result["status"] != "complete" and result["gated"]
    ]
    if non_gated_failures:
        return 1
    if gated_failures:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
