import torch
from diffusers import DiffusionPipeline
import os
from datetime import datetime
import argparse

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("--exp", type=str, default=None)
	parser.add_argument("--inf_steps", type=int, default=100)
	parser.add_argument("--seeds", type=int, default=42)
	parser.add_argument("--scale", type=float, default=6.0)
	parser.add_argument("--samples", type=int, default=20)

	args = parser.parse_args()
	
	prompts = [
		"a photo of <s*> pine in the desert"
		]

	model_path = f'model/{args.exp}/'
	LOG_DIR=f''

	model_id = f"{model_path+LOG_DIR}"

	out_path = f'Results/{args.exp}_{args.inf_steps}_{args.seeds}_{args.scale}_{args.samples}/'

	isExist = os.path.exists(out_path)
	if not isExist:
  		os.makedirs(out_path)
		
	osExist = os.path.exists(f'{out_path}/{LOG_DIR}')

	if not osExist:
  		os.makedirs(f'{out_path}/{LOG_DIR}')
	
	pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", low_cpu_mem_usage=False, torch_dtype=torch.float16).to("cuda:5")
	pipe.unet.load_attn_procs(f"{model_id}", weight_name="pytorch_custom_diffusion_weights.bin")
	pipe.load_textual_inversion(f"{model_id}", weight_name="<s*>.bin")
	pipe.load_textual_inversion(f"{model_id}", weight_name="<c1*>.bin")
	pipe.load_textual_inversion(f"{model_id}", weight_name="<c2*>.bin")
	pipe.load_textual_inversion(f"{model_id}", weight_name="<c3*>.bin")
	pipe.load_textual_inversion(f"{model_id}", weight_name="<c4*>.bin")

	for prompt in prompts:
		n_samples = args.samples
		path = f'{out_path}'

		for ind in range(n_samples):
			out = pipe(prompt)
			out.images[0].save(f"{path}/{prompt}_{ind}.png")

if __name__ == "__main__":
    main()