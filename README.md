# ColorPeel: Color Prompt Learning with Diffusion Models via  Color and Shape Disentanglement [ECCV 2024]

![teaser](assets/teaser_4.jpg)

***TL;DB*** Given the RGB triplets or color coordinates, ColorPeel generates basic 2D or 3D geometries with target colors for color learning. This facilitates the disentanglement of color and shape concepts, allowing for personalized color usage in image generation.

## ColorPeel: Color Prompt Learning with Diffusion Models via  Color and Shape Disentanglement
[Muhammad Atif Butt](https://scholar.google.com/citations?user=vf7PeaoAAAAJ&hl=en), [Kai Wang](https://scholar.google.com/citations?user=j14vd0wAAAAJ&hl=en), [Javier Vazquez-Corral](https://scholar.google.com/citations?user=gjnuPMoAAAAJ&hl=en),  [Joost van de Weijer](https://scholar.google.com/citations?user=Gsw2iUEAAAAJ&hl=en)

[[Paper](http://arxiv.org/pdf/2407.07197)] [[arXiv](http://arxiv.org/abs/2407.07197)] [[Project](https://moatifbutt.github.io/colorpeel/)] [[Poster](https://github.com/moatifbutt/color-peel/blob/main/assets/ECCV2024_ColorPeel_.pdf)]

<hr>

## Installations (for local execution with PyTorch)
Before running the scripts, make sure to install diffusers from source. Note that ColorPeel is developed on **Diffusers 0.17.0**.
To install diffusers from source, do the following steps:

```sh
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
```

After successful installation, download/clone the **ColorPeel** repoistory.

```sh
https://github.com/moatifbutt/color-peel.git
cd color-peel
pip install -r requirements.txt
```

And initialize an 🤗Accelerate environment with:

```sh
accelerate config
```

Or for a default accelerate configuration without answering questions about your environment.

```sh
accelerate config default
```

## Dataset
We provide two sample datasets for learning colors from 2D and 3D shapes. These datasets are available in data repository along with the `src/concept_json/instances_3d.json` which contain information regarding the class images and their corresponding conditioning prompts.

**Custom Dataset (2D)**: Users can also create their own 2D dataset using the following script.

```sh
python src/draw_shapes.py 512 --shapes circle rectangle --rgb_values "235,33,33" "33,235,33" "33,33,235" "235,235,33" --out data/dataset
```
**Custom Dataset (3D)**: We design our 3D shapes in blender. The rendering script will be released soon.

## Train
Now, we are all set to start training. After setting up the paths in `train/train.sh`, run the following.

```sh
./src/train/train_colorpeel.sh
```

## Test
After completing the training, the model will be saved in `models` directory. Run the following command for inference.

```sh
python src/test.py --exp model_name
```

# Future Work
We have experimented with mapping various colors from color spaces into color prompt embeddings. However, we encountered convergence issues that we are currently unable to resolve. For those interested in learning multiple colors for practical applications, we suggest developing a training scheme based on Textual Inversion, which has demonstrated satisfactory performance. As for the task of mapping color spaces into text embeddings, we leave this as a future research direction for the community to explore.


# Citation

If you like our work, please cite our paper:

```
@inproceedings{butt2024colorpeel, 
    title={ColorPeel: Color Prompt Learning with Diffusion Models via Color and Shape Disentanglement}, 
    author={Muhammad Atif Butt and Kai Wang and Javier Vazquez-Corral and Joost van de Weijer},
    booktitle={European Conference on Computer Vision}, 
    year={2024}
}
```
