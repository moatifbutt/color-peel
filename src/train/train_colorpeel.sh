MODEL_NAME="CompVis/stable-diffusion-v1-4"
Out_dir="models/exp"

CUDA_VISIBLE_DEVICES=3 CUDA_LAUNCH_BLOCKING=1 python src/train/train_colorpeel.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir=$Out_dir \
  --concepts_list=./src/concept_json/instances_3d.json \
  --resolution=512  \
  --train_batch_size=1  \
  --learning_rate=1e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=1500 \
  --cos_weight=0.2 \
  --scale_lr --hflip  \
  --modifier_token "<s1*>+<s2*>+<c1*>+<c2*>+<c3*>+<c4*>" \
  --initializer_token "cone+sphere+red+green+blue+yellow"

python src/test/test.py --exp exp
