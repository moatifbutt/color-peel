# ColorPeel: Color Prompt Learning with Diffusion Models via  Color and Shape Disentanglement [ECCV 2024]
[Muhammad Atif Butt](https://scholar.google.com/citations?user=vf7PeaoAAAAJ&hl=en), [Kai Wang](https://scholar.google.com/citations?user=j14vd0wAAAAJ&hl=en), [Javier Vazquez-Corral](https://scholar.google.com/citations?user=gjnuPMoAAAAJ&hl=en), and [Joost van de Weijer](https://scholar.google.com/citations?user=Gsw2iUEAAAAJ&hl=en)

![teaser](assets/teaser_4.jpg)
Given the RGB triplets or color coordinates, ColorPeel generates basic 2D or 3D geometries with target colors for color learning. This facilitates the disentanglement of color and shape concepts, allowing for personalized color usage in image generation.

<hr>

## Installations

```sh
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
```
Then cd in the example folder and run

```sh
pip install -r requirements.txt
pip install clip-retrieval
```
And initialize an 🤗Accelerate environment with:
```sh
accelerate config
```
Or for a default accelerate configuration without answering questions about your environment
```sh
accelerate config default
```
