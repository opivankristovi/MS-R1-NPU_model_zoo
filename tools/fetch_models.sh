#!/usr/bin/env bash
# Fetch every model the zoo uses from the CIX AI Model Hub 25Q3 (ModelScope), without git-lfs.
# Needs the pointer-file clone at $HUB (default $MODELS/hub, MODELS=/mnt/data/npu-models):
#   GIT_LFS_SKIP_SMUDGE=1 git clone https://www.modelscope.cn/cix/ai_model_hub_25_Q3.git $HUB
# Idempotent: files already downloaded are skipped. ~6 GB in total.
set -euo pipefail
HUB=${HUB:-${MODELS:-/mnt/data/npu-models}/hub}
BASE=https://www.modelscope.cn/models/cix/ai_model_hub_25_Q3/resolve/master
CV=models/ComputeVision; GEN=models/Generative_AI/Image_to_Text
FILES=(
  $CV/Object_Detection/onnx_yolov8_l/yolov8_l.cix
  $CV/Object_Detection/onnx_yolov12_m/yolov12_m.cix
  $CV/Object_Detection/onnx_yolox_m/yolox_m.cix
  $CV/Object_Detection/onnx_yolox_l/yolox_l.cix
  $CV/Object_Detection/onnx_yolov8l_worldv2/yolov8l-worldv2.cix
  $CV/Face_Recognition/onnx_scrfd_arcface/scrfd.cix
  $CV/Face_Recognition/onnx_scrfd_arcface/arcface.cix
  $CV/Face_Detection/onnx_centerface/centerface.cix
  $CV/Face_Detection/onnx_rfb_320/rfb_320.cix
  $CV/Body_Analysis/onnx_age_googlenet/age_googlenet.cix
  $CV/Body_Analysis/onnx_gender_googlenet/gender_googlenet.cix
  $CV/Semantic_Segmentation/onnx_yolov8_segmentation/yolov8l_seg.cix
  $CV/Semantic_Segmentation/onnx_sam2_1_tiny/sam2_hiera_tiny_encoder_sim.cix
  $CV/Semantic_Segmentation/onnx_sam2_1_tiny/model/decoder.onnx
  $CV/Semantic_Segmentation/onnx_u2net/u2net_human_seg.cix
  $CV/Semantic_Segmentation/onnx_pphumanseg/pphuanseg.cix
  $CV/Pose_Estimation/onnx_yolov8s_pose/yolov8s-pose.cix
  $CV/Pose_Estimation/onnx_handpose/handpose_resnet50_size256.cix
  $CV/Depth_Estimation/onnx_depth_anything_v2/depth_anything_v2.cix
  $CV/Depth_Estimation/onnx_MiDaS_v2/MiDaS_v2.cix
  $CV/Super_Resolution/onnx_real_esrgan/real_esrgan.cix
  $CV/Super_Resolution/onnx_super_resolution/super_resolution_10.cix
  $CV/Super_Resolution/onnx_vdsr/vdsr.cix
  $CV/Image_Enhancement/onnx_AGLLNet/AGLLNet.cix
  $CV/Image_Enhancement/onnx_pairlie/pairlie.cix
  $CV/Image_Enhancement/onnx_nighttime_dehaze/nighttime_dehaze.cix
  $GEN/onnx_clip/clip_visual.cix
  $GEN/onnx_clip/clip_txt.cix
  $GEN/onnx_clip/model/clip_text_model_vitb32.onnx
  $GEN/onnx_siglip-so400m-patch14-384/siglip_img.cix
  $GEN/onnx_siglip-so400m-patch14-384/siglip-so400m-patch14-384/spiece.model
  $GEN/onnx_siglip-so400m-patch14-384/model/siglip-so400m-patch14-384-txt.onnx
)
cd "$HUB"
for f in "${FILES[@]}"; do
  [ -f "$f" ] || { echo "missing in clone: $f"; exit 1; }
  grep -q '^version https://git-lfs' "$f" 2>/dev/null || { echo "ok    $f"; continue; }
  want=$(awk '/^size/{print $2}' "$f")
  echo "fetch $f ($((want/1000000)) MB)"
  curl -fL --retry 3 -o "$f.part" "$BASE/$f"
  [ "$(stat -c %s "$f.part")" = "$want" ] || { echo "size mismatch $f"; rm -f "$f.part"; exit 1; }
  mv "$f.part" "$f"
done
echo "all models present"
