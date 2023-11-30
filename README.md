# Towards Realistic Generative 3D Face Models (AlbedoGAN)

Aashish Rai, Hiresh Gupta*, Ayush Pandey*,  Francisco Vicente Carrasco, Shingo Jason Takagi, Amaury Aubel, Daeil Kim, Aayush Prakash, Fernando de la Torre

### Carnegie Mellon University, Facebook/Meta

### WACV 2024 

[[Project Page](https://aashishrai3799.github.io/Towards-Realistic-Generative-3D-Face-Models)] [[Arxiv](https://arxiv.org/pdf/2304.12483.pdf)]

We propose a 3D face generative model that generates high-quality albedo and precise 3D shape by leveraging StyleGAN2, resulting in a photo-realistic rendered image.


![](figure_1.png)

![](supp_image.png)


## Testing

Conda environment: Refer environment.yml

Download pre-trained models and put in the respective folders. 

Follow [[MICA](https://github.com/Zielon/MICA)] to download insightface and MICA pre-trained models. Put the weights in 'insightface' and 'data/mica_pretrained' folders, respectively.
Follow [[DECA](https://github.com/yfeng95/DECA)] to download DECA pre-trained weights. Put them in the 'data' folder.

Download AlbedoGAN modified weights from the following [[LINK](gdrive)]. Put these modified ArcFace backbone and DECA weights to generate better reconstruction results.

- Generate Random 3D Faces (mesh and texture)
    ```
    python demos/demo_generate.py
    ```
    
- Reconstruct 3D Faces from 2D Images
    ```
    python demos/demo_reconstruct.py
    ```
