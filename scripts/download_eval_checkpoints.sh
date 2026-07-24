#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="/data/aigc/cyb/zxgu/env/worldsearcher"
PYTHON="${ENV_ROOT}/bin/python"
HF="${ENV_ROOT}/bin/hf"
ARIA2="/usr/bin/aria2c"
RANGE_DOWNLOADER="${PROJECT_ROOT}/scripts/range_download.py"
CKPT_ROOT="${PROJECT_ROOT}/ckpts"
SLEEP_SEC="${SLEEP_SEC:-30}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-32}"
ARIA2_CONNECTIONS="${ARIA2_CONNECTIONS:-8}"
POLYBOX_WORKERS="${POLYBOX_WORKERS:-32}"

# huggingface_hub >= 1.x uses Xet rather than the deprecated hf_transfer path.
export HF_XET_HIGH_PERFORMANCE=1
unset HF_HUB_ENABLE_HF_TRANSFER

# Bulk artifacts explicitly bypass the host's preconfigured localhost proxy
# because a measured direct transfer is substantially faster.

log() {
    printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

require_file() {
    if [[ ! -x "$1" ]]; then
        log "Required executable is missing: $1"
        exit 1
    fi
}

download_hf_file() {
    local label="$1"
    local repo="$2"
    local revision="$3"
    local filename="$4"
    local destination_root="$5"
    local expected_size="$6"
    local repo_type="${7:-model}"
    local url_prefix=""

    if [[ "$repo_type" == "space" ]]; then
        url_prefix="spaces/"
    fi

    download_url \
        "$label" \
        "https://huggingface.co/${url_prefix}${repo}/resolve/${revision}/${filename}?download=true" \
        "${destination_root}/${filename}" \
        "$expected_size"
}

download_hf_once() {
    local label="$1"
    local repo="$2"
    local revision="$3"
    local destination="$4"
    shift 4

    mkdir -p "$destination"
    log "HF access check: ${label} (${repo}@${revision})"
    env \
        -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
        -u http_proxy -u https_proxy -u all_proxy \
        "$HF" download "$repo" "$@" \
            --revision "$revision" \
            --max-workers "$HF_MAX_WORKERS" \
            --local-dir "$destination"
}

download_url() {
    local label="$1"
    local url="$2"
    local destination="$3"
    local expected_size="$4"
    local destination_dir
    local destination_name
    local actual_size

    destination_dir="$(dirname "$destination")"
    destination_name="$(basename "$destination")"
    mkdir -p "$destination_dir"

    if [[ -f "$destination" ]]; then
        actual_size="$(stat -c '%s' "$destination")"
        if [[ "$actual_size" == "$expected_size" ]]; then
            log "Skip complete: ${label} (${actual_size} bytes)"
            return 0
        fi
        if [[ -f "${destination}.aria2" ]]; then
            log "Resume partial: ${label} (${actual_size}/${expected_size} bytes)"
        else
            log "Existing file has wrong size: ${destination}; expected ${expected_size}, got ${actual_size}"
            return 1
        fi
    fi

    while true; do
        log "URL start/resume: ${label}"
        if env \
            -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
            -u http_proxy -u https_proxy -u all_proxy \
            "$ARIA2" \
            --continue=true \
            --max-connection-per-server="$ARIA2_CONNECTIONS" \
            --split="$ARIA2_CONNECTIONS" \
            --min-split-size=1M \
            --file-allocation=none \
            --auto-file-renaming=false \
            --allow-overwrite=false \
            --max-tries=5 \
            --retry-wait=5 \
            --connect-timeout=30 \
            --timeout=120 \
            --console-log-level=warn \
            --summary-interval=30 \
            --dir="$destination_dir" \
            --out="$destination_name" \
            "$url"; then
            actual_size="$(stat -c '%s' "$destination")"
            if [[ "$actual_size" == "$expected_size" ]]; then
                log "URL done: ${label} (${actual_size} bytes)"
                return 0
            fi
            log "Size check failed: ${label}; expected ${expected_size}, got ${actual_size}"
            return 1
        fi
        log "URL failed: ${label}; retrying in ${SLEEP_SEC}s"
        sleep "$SLEEP_SEC"
    done
}

download_webdav_file() {
    local label="$1"
    local path="$2"
    local destination="$3"
    local expected_size="$4"
    local destination_dir
    local destination_name
    local actual_size
    local url="https://polybox.ethz.ch/public.php/webdav/${path}"

    destination_dir="$(dirname "$destination")"
    destination_name="$(basename "$destination")"
    mkdir -p "$destination_dir"

    if [[ -f "$destination" ]]; then
        actual_size="$(stat -c '%s' "$destination")"
        if [[ "$actual_size" == "$expected_size" && ! -f "${destination}.aria2" ]]; then
            log "Skip complete: ${label} (${actual_size} bytes)"
            return 0
        fi
        if [[ ! -f "${destination}.aria2" ]]; then
            log "Existing file has wrong size: ${destination}; expected ${expected_size}, got ${actual_size}"
            return 1
        fi
    fi

    while true; do
        log "Polybox start/resume: ${label}"
        if env \
            -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
            -u http_proxy -u https_proxy -u all_proxy \
            "$ARIA2" \
            --continue=true \
            --max-connection-per-server=1 \
            --split=1 \
            --file-allocation=none \
            --auto-file-renaming=false \
            --allow-overwrite=false \
            --max-tries=5 \
            --retry-wait=5 \
            --connect-timeout=30 \
            --timeout=120 \
            --console-log-level=warn \
            --summary-interval=30 \
            --http-user="LHz64M2YnRo3CpL" \
            --http-passwd="" \
            --dir="$destination_dir" \
            --out="$destination_name" \
            "$url"; then
            actual_size="$(stat -c '%s' "$destination")"
            if [[ "$actual_size" == "$expected_size" ]]; then
                log "Polybox done: ${label} (${actual_size} bytes)"
                return 0
            fi
            log "Size check failed: ${label}; expected ${expected_size}, got ${actual_size}"
            return 1
        fi
        log "Polybox failed: ${label}; retrying in ${SLEEP_SEC}s"
        sleep "$SLEEP_SEC"
    done
}

download_da3_and_streaming() {
    local destination="${CKPT_ROOT}/da3/DA3NESTED-GIANT-LARGE-1.1"
    download_hf_file \
        "DA3 Nested Giant-Large 1.1 README" \
        "depth-anything/DA3NESTED-GIANT-LARGE-1.1" \
        "b2359bdf726fb44ef62acca04d629dcf158053e7" \
        "README.md" "$destination" "4976"
    download_hf_file \
        "DA3 Nested Giant-Large 1.1 config" \
        "depth-anything/DA3NESTED-GIANT-LARGE-1.1" \
        "b2359bdf726fb44ef62acca04d629dcf158053e7" \
        "config.json" "$destination" "3113"
    download_hf_file \
        "DA3 Nested Giant-Large 1.1 weights (also DA3-Streaming)" \
        "depth-anything/DA3NESTED-GIANT-LARGE-1.1" \
        "b2359bdf726fb44ef62acca04d629dcf158053e7" \
        "model.safetensors" "$destination" "6759558100"
}

download_lingbot_and_vggt_slam() {
    local lingbot_destination="${CKPT_ROOT}/lingbot-map"
    local vggt_destination="${CKPT_ROOT}/vggt-slam/vggt-1b"

    download_hf_file \
        "LingBot-Map README" \
        "robbyant/lingbot-map" \
        "204754b72bb24f561f8d7e7e1e4e4cd9e809adf9" \
        "README.md" "$lingbot_destination" "26826"
    download_hf_file \
        "LingBot-Map long weights" \
        "robbyant/lingbot-map" \
        "204754b72bb24f561f8d7e7e1e4e4cd9e809adf9" \
        "lingbot-map-long.pt" "$lingbot_destination" "4632303465"
    download_hf_file \
        "LingBot-Map sky segmentation model" \
        "robbyant/lingbot-map" \
        "204754b72bb24f561f8d7e7e1e4e4cd9e809adf9" \
        "skyseg_batch.onnx" "$lingbot_destination" "175997119"

    download_hf_file \
        "VGGT-SLAM 2.0 VGGT README" \
        "facebook/VGGT-1B" \
        "860abec7937da0a4c03c41d3c269c366e82abdf9" \
        "README.md" "$vggt_destination" "2060"
    download_hf_file \
        "VGGT-SLAM 2.0 VGGT config" \
        "facebook/VGGT-1B" \
        "860abec7937da0a4c03c41d3c269c366e82abdf9" \
        "config.json" "$vggt_destination" "62"
    download_hf_file \
        "VGGT-SLAM 2.0 VGGT-1B weights" \
        "facebook/VGGT-1B" \
        "860abec7937da0a4c03c41d3c269c366e82abdf9" \
        "model.pt" "$vggt_destination" "5026874952"

    download_url \
        "VGGT-SLAM 2.0 SALAD loop-closure model" \
        "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt" \
        "${CKPT_ROOT}/vggt-slam/dino_salad.ckpt" \
        "352040378"
}

download_vipe_and_droid() {
    download_url \
        "DROID-SLAM official model (shared with ViPE)" \
        "https://drive.usercontent.google.com/download?id=1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh&export=download&confirm=t" \
        "${CKPT_ROOT}/droid-slam/droid.pth" \
        "16061701"

    local metric_destination="${CKPT_ROOT}/vipe/da3metric-large"
    local giant_destination="${CKPT_ROOT}/vipe/da3-giant"
    local grounding_destination="${CKPT_ROOT}/vipe/grounding-dino"
    local bert_destination="${CKPT_ROOT}/vipe/bert-base-uncased"

    download_hf_file \
        "ViPE DA3Metric README" \
        "depth-anything/DA3METRIC-LARGE" \
        "4010e39f3634a45bc60553321fb49fb760bd594e" \
        "README.md" "$metric_destination" "4789"
    download_hf_file \
        "ViPE DA3Metric config" \
        "depth-anything/DA3METRIC-LARGE" \
        "4010e39f3634a45bc60553321fb49fb760bd594e" \
        "config.json" "$metric_destination" "847"
    download_hf_file \
        "ViPE DA3Metric weights" \
        "depth-anything/DA3METRIC-LARGE" \
        "4010e39f3634a45bc60553321fb49fb760bd594e" \
        "model.safetensors" "$metric_destination" "1336734448"

    # ViPE 1.2.0 hard-codes this upstream checkpoint for its official dav3
    # pipeline. It is intentionally separate from the main DA3 1.1 baseline.
    download_hf_file \
        "ViPE DA3-GIANT README" \
        "depth-anything/DA3-GIANT" \
        "7cd62ae9315b9dff094d2d300e4ad012640607dd" \
        "README.md" "$giant_destination" "4881"
    download_hf_file \
        "ViPE DA3-GIANT config" \
        "depth-anything/DA3-GIANT" \
        "7cd62ae9315b9dff094d2d300e4ad012640607dd" \
        "config.json" "$giant_destination" "1880"
    download_hf_file \
        "ViPE DA3-GIANT weights" \
        "depth-anything/DA3-GIANT" \
        "7cd62ae9315b9dff094d2d300e4ad012640607dd" \
        "model.safetensors" "$giant_destination" "5422814644"

    download_url \
        "ViPE GeoCalib pinhole model" \
        "https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar" \
        "${CKPT_ROOT}/vipe/geocalib/geocalib-pinhole.tar" \
        "116074121"

    download_url \
        "ViPE Segment Anything ViT-B" \
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" \
        "${CKPT_ROOT}/vipe/track-anything/sam_vit_b_01ec64.pth" \
        "375042383"

    download_url \
        "ViPE DeAOT tracker" \
        "https://drive.usercontent.google.com/download?id=1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ&export=download&confirm=t" \
        "${CKPT_ROOT}/vipe/track-anything/R50_DeAOTL_PRE_YTB_DAV.pth" \
        "236513959"

    download_hf_file \
        "ViPE GroundingDINO Swin-T" \
        "ShilongLiu/GroundingDINO" \
        "a94c9b567a2a374598f05c584e96798a170c56fb" \
        "groundingdino_swint_ogc.pth" "$grounding_destination" "693997677"

    download_hf_file \
        "ViPE BERT config" \
        "bert-base-uncased" \
        "86b5e0934494bd15c9632b12f734a8a67f723594" \
        "config.json" "$bert_destination" "570"
    download_hf_file \
        "ViPE BERT weights" \
        "bert-base-uncased" \
        "86b5e0934494bd15c9632b12f734a8a67f723594" \
        "model.safetensors" "$bert_destination" "440449768"
    download_hf_file \
        "ViPE BERT tokenizer" \
        "bert-base-uncased" \
        "86b5e0934494bd15c9632b12f734a8a67f723594" \
        "tokenizer.json" "$bert_destination" "466062"
    download_hf_file \
        "ViPE BERT tokenizer config" \
        "bert-base-uncased" \
        "86b5e0934494bd15c9632b12f734a8a67f723594" \
        "tokenizer_config.json" "$bert_destination" "48"
    download_hf_file \
        "ViPE BERT vocabulary" \
        "bert-base-uncased" \
        "86b5e0934494bd15c9632b12f734a8a67f723594" \
        "vocab.txt" "$bert_destination" "231508"
}

download_egom2p() {
    "$PYTHON" "$RANGE_DOWNLOADER" \
        --url "https://polybox.ethz.ch/index.php/s/3byirHr9ka7gs8W/download" \
        --output "${CKPT_ROOT}/egom2p/checkpoint-main.pth" \
        --expected-size "4818565952" \
        --workers "$POLYBOX_WORKERS" &
    local pid_egom2p_main=$!

    "$PYTHON" "$RANGE_DOWNLOADER" \
        --url "https://polybox.ethz.ch/index.php/s/eqcEQNkZ7e8iQs3/download" \
        --output "${CKPT_ROOT}/egom2p/checkpoint-cam.pth" \
        --expected-size "2156231716" \
        --workers "$POLYBOX_WORKERS" &
    local pid_egom2p_camera=$!

    local egom2p_status=0
    if ! wait "$pid_egom2p_main"; then
        egom2p_status=1
    fi
    if ! wait "$pid_egom2p_camera"; then
        egom2p_status=1
    fi
    if (( egom2p_status != 0 )); then
        return "$egom2p_status"
    fi

    local cosmos_destination="${CKPT_ROOT}/egom2p/cosmos-tokenizer"
    download_hf_file \
        "EgoM2P Cosmos tokenizer README" \
        "nvidia/Cosmos-0.1-Tokenizer-DV4x8x8" \
        "76642c98e6068c02f6e897c46d4ab84cb176d84d" \
        "README.md" "$cosmos_destination" "21412"
    download_hf_file \
        "EgoM2P Cosmos tokenizer config" \
        "nvidia/Cosmos-0.1-Tokenizer-DV4x8x8" \
        "76642c98e6068c02f6e897c46d4ab84cb176d84d" \
        "config.json" "$cosmos_destination" "54"
    download_hf_file \
        "EgoM2P Cosmos model config" \
        "nvidia/Cosmos-0.1-Tokenizer-DV4x8x8" \
        "76642c98e6068c02f6e897c46d4ab84cb176d84d" \
        "model_config.yaml" "$cosmos_destination" "92"
    download_hf_file \
        "EgoM2P/ReViV Cosmos autoencoder" \
        "nvidia/Cosmos-0.1-Tokenizer-DV4x8x8" \
        "76642c98e6068c02f6e897c46d4ab84cb176d84d" \
        "autoencoder.jit" "$cosmos_destination" "211093069"
    download_hf_file \
        "EgoM2P Cosmos RGB encoder" \
        "nvidia/Cosmos-0.1-Tokenizer-DV4x8x8" \
        "76642c98e6068c02f6e897c46d4ab84cb176d84d" \
        "encoder.jit" "$cosmos_destination" "86641076"
    download_hf_file \
        "EgoM2P Cosmos RGB decoder" \
        "nvidia/Cosmos-0.1-Tokenizer-DV4x8x8" \
        "76642c98e6068c02f6e897c46d4ab84cb176d84d" \
        "decoder.jit" "$cosmos_destination" "125210440"
}

download_reviv_set() {
    local set_name="$1"
    local main_size="$2"
    local body_size="$3"
    local destination="${CKPT_ROOT}/reviv/${set_name}"
    local -a pids=()
    local status=0
    local spec

    for spec in \
        "reviv_main.pth:${main_size}" \
        "reviv_tok_body.pth:${body_size}" \
        "reviv_tok_cam.pth:718878062" \
        "reviv_tok_gaze.pth:718826027" \
        "reviv_tok_lhand.pth:720085992" \
        "reviv_tok_rhand.pth:720086056"; do
        download_webdav_file \
            "ReViV ${set_name} ${spec%%:*}" \
            "${set_name}/${spec%%:*}" \
            "${destination}/${spec%%:*}" \
            "${spec##*:}" &
        pids+=("$!")
    done

    local stat_name
    local stat_size
    for spec in \
        "body_mean.npy:632" \
        "body_std.npy:632" \
        "cam_mean.npy:200" \
        "cam_std.npy:200" \
        "lhand_mean.npy:152" \
        "lhand_std.npy:152" \
        "rhand_mean.npy:152" \
        "rhand_std.npy:152"; do
        stat_name="${spec%%:*}"
        stat_size="${spec##*:}"
        download_webdav_file \
            "ReViV ${set_name} norm_stats/${stat_name}" \
            "${set_name}/norm_stats/${stat_name}" \
            "${destination}/norm_stats/${stat_name}" \
            "$stat_size" &
        pids+=("$!")
    done

    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            status=1
        fi
    done
    return "$status"
}

download_reviv_and_egoego() {
    local status=0
    download_reviv_set "metric_depth" "2574931428" "720220907" &
    local metric_pid=$!
    download_reviv_set "reviv_500b" "1687821271" "720085675" &
    local reviv_500b_pid=$!

    local egoego_destination="${CKPT_ROOT}/egoego"
    download_url \
        "EgoEgo gravity network" \
        "https://drive.usercontent.google.com/download?id=1rTTlwlUYhKzfjsvot7QExskn6RvGmtOx&export=download&confirm=t" \
        "${egoego_destination}/stage1_gravitynet_2000.pt" \
        "31814605"
    download_url \
        "EgoEgo ARES head network" \
        "https://drive.usercontent.google.com/download?id=1vhOL3tk19eiASGZg2Cd4QCuM6Dz4iPx3&export=download&confirm=t" \
        "${egoego_destination}/stage1_headnet_ares_250.pt" \
        "52191441"
    download_url \
        "EgoEgo GIMO head network" \
        "https://drive.usercontent.google.com/download?id=1gD7H6AdCYExP4UP3DUV5NgTGihip2iRC&export=download&confirm=t" \
        "${egoego_destination}/stage1_headnet_gimo_1000.pt" \
        "52191377"
    download_url \
        "EgoEgo KinPoly head network" \
        "https://drive.usercontent.google.com/download?id=1V7gMRPN7Q45gpkDmb-HIdpLT3C5PFO1T&export=download&confirm=t" \
        "${egoego_destination}/stage1_headnet_kinpoly_1000.pt" \
        "52191377"
    download_url \
        "EgoEgo stage-2 diffusion model" \
        "https://drive.usercontent.google.com/download?id=18LQnbj_O809h8UJ60rpUuEKXIA15SKkX&export=download&confirm=t" \
        "${egoego_destination}/stage2_diffusion_4.pt" \
        "88417189"

    if ! wait "$metric_pid"; then
        status=1
    fi
    if ! wait "$reviv_500b_pid"; then
        status=1
    fi

    local cosmos_destination="${CKPT_ROOT}/reviv/cosmos/Cosmos-1.0-Tokenizer-DV8x16x16"
    download_hf_file \
        "ReViV Cosmos-1.0 tokenizer README" \
        "nvidia/Cosmos-1.0-Tokenizer-DV8x16x16" \
        "62e95d83acc55dc4344af1cad4848bdd7f7e1d97" \
        "README.md" "$cosmos_destination" "30903"

    return "$status"
}

download_megasam_and_hawor() {
    download_url \
        "MegaSaM camera tracker" \
        "https://raw.githubusercontent.com/mega-sam/mega-sam/a27b4e633c5cc0828a62ed943ef9f6505705fd3f/checkpoints/megasam_final.pth" \
        "${CKPT_ROOT}/mega-sam/megasam_final.pth" \
        "20812149"

    download_hf_file \
        "MegaSaM relative depth prior" \
        "LiheYoung/Depth-Anything" \
        "7f1457e21e74e7aa001c88fc15da5c74598aa3fa" \
        "checkpoints/depth_anything_vitl14.pth" \
        "${CKPT_ROOT}/mega-sam/depth-anything-v1" \
        "1341401882" \
        "space"

    local unidepth_destination="${CKPT_ROOT}/mega-sam/unidepth-v2-vitl14"
    download_hf_file \
        "MegaSaM UniDepth README" \
        "lpiccinelli/unidepth-v2-vitl14" \
        "1d0d3c52f60b5164629d279bb9a7546458e6dcc4" \
        "README.md" "$unidepth_destination" "397"
    download_hf_file \
        "MegaSaM UniDepth config" \
        "lpiccinelli/unidepth-v2-vitl14" \
        "1d0d3c52f60b5164629d279bb9a7546458e6dcc4" \
        "config.json" "$unidepth_destination" "1329"
    download_hf_file \
        "MegaSaM UniDepth weights" \
        "lpiccinelli/unidepth-v2-vitl14" \
        "1d0d3c52f60b5164629d279bb9a7546458e6dcc4" \
        "model.safetensors" "$unidepth_destination" "1452916608"

    download_url \
        "MegaSaM RAFT flow model" \
        "https://drive.usercontent.google.com/download?id=1MqDajR89k-xLV0HIrmJ0k-n8ZpG6_suM&export=download&confirm=t" \
        "${CKPT_ROOT}/mega-sam/raft-things.pth" \
        "21108000"

    local hawor_destination="${CKPT_ROOT}/hawor"
    download_hf_file \
        "HaWoR hand detector" \
        "ThunderVVV/HaWoR" \
        "da6335f47f9806308992d5ae1002a4cc5f7252c2" \
        "external/detector.pt" "$hawor_destination" "53582271"
    download_hf_file \
        "HaWoR masked DROID" \
        "ThunderVVV/HaWoR" \
        "da6335f47f9806308992d5ae1002a4cc5f7252c2" \
        "external/droid.pth" "$hawor_destination" "16061701"
    download_hf_file \
        "HaWoR Metric3D" \
        "ThunderVVV/HaWoR" \
        "da6335f47f9806308992d5ae1002a4cc5f7252c2" \
        "external/metric_depth_vit_large_800k.pth" "$hawor_destination" "1647972663"
    download_hf_file \
        "HaWoR main weights" \
        "ThunderVVV/HaWoR" \
        "da6335f47f9806308992d5ae1002a4cc5f7252c2" \
        "hawor/checkpoints/hawor.ckpt" "$hawor_destination" "3267481572"
    download_hf_file \
        "HaWoR infiller" \
        "ThunderVVV/HaWoR" \
        "da6335f47f9806308992d5ae1002a4cc5f7252c2" \
        "hawor/checkpoints/infiller.pt" "$hawor_destination" "418603497"
    download_hf_file \
        "HaWoR model config" \
        "ThunderVVV/HaWoR" \
        "da6335f47f9806308992d5ae1002a4cc5f7252c2" \
        "hawor/model_config.yaml" "$hawor_destination" "2743"
}

require_file "$PYTHON"
require_file "$HF"
require_file "$ARIA2"
require_file "$RANGE_DOWNLOADER"
mkdir -p "$CKPT_ROOT"

log "Checkpoint root: ${CKPT_ROOT}"
log "Python: $("$PYTHON" --version 2>&1)"

download_da3_and_streaming &
pid_da3=$!
download_lingbot_and_vggt_slam &
pid_geometry=$!
download_vipe_and_droid &
pid_vipe=$!
download_egom2p &
pid_egom2p=$!
download_megasam_and_hawor &
pid_specialists=$!
download_reviv_and_egoego &
pid_ego_specialists=$!

public_status=0
for pid in \
    "$pid_da3" \
    "$pid_geometry" \
    "$pid_vipe" \
    "$pid_egom2p" \
    "$pid_specialists" \
    "$pid_ego_specialists"; do
    if ! wait "$pid"; then
        public_status=1
    fi
done

# VGGT-Omega is gated. Try once so the exit status accurately records whether
# this machine's existing Hugging Face login has since been approved.
omega_status=0
if ! download_hf_once \
    "VGGT-Omega-1B-512" \
    "facebook/VGGT-Omega" \
    "05654241adc2f218dfb089c373a011f8a7040576" \
    "${CKPT_ROOT}/vggt-omega/VGGT-Omega-1B-512" \
    vggt_omega_1b_512.pt; then
    omega_status=2
    log "VGGT-Omega remains gated. Request access at https://huggingface.co/facebook/VGGT-Omega"
fi

cosmos_status=0
if ! download_hf_once \
    "ReViV Cosmos-1.0 tokenizer (NVIDIA license)" \
    "nvidia/Cosmos-1.0-Tokenizer-DV8x16x16" \
    "62e95d83acc55dc4344af1cad4848bdd7f7e1d97" \
    "${CKPT_ROOT}/reviv/cosmos/Cosmos-1.0-Tokenizer-DV8x16x16" \
    autoencoder.jit config.json decoder.jit encoder.jit model_config.yaml; then
    cosmos_status=2
    log "Cosmos-1.0 remains gated. Accept the NVIDIA license at https://huggingface.co/nvidia/Cosmos-1.0-Tokenizer-DV8x16x16"
fi

if (( public_status != 0 )); then
    log "At least one public checkpoint group failed."
    exit "$public_status"
fi

if (( omega_status != 0 || cosmos_status != 0 )); then
    exit 2
fi

log "All planned model checkpoints downloaded successfully."
