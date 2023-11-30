# Towards Realistic Generative 3D Face Models (AlbedoGAN)

Aashish Rai, Hiresh Gupta*, Ayush Pandey*,  Francisco Vicente Carrasco, Shingo Jason Takagi, Amaury Aubel, Daeil Kim, Aayush Prakash, Fernando de la Torre

### Carnegie Mellon University, Meta Reality Labs

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

Download AlbedoGAN modified weights from the following [[LINK](https://drive.google.com/drive/folders/1nJw8rUBTLcyhvCMTDohE_KcKKtFI6Orm?usp=sharing)]. Put these modified ArcFace backbone and DECA weights to generate better reconstruction results.

- Generate Random 3D Faces (mesh and texture)
    ```
    python demos/demo_generate.py
    ```
    
- Reconstruct 3D Faces from 2D Images
    ```
    python demos/demo_reconstruct.py
    ```

## Training code will be released at the earliest convenience. 

    
## License

The code is available under X11 License. Please read the license terms available at [[Link](https://github.com/aashishrai3799/Towards-Realistic-Generative-3D-Face-Models/blob/main/LICENSE)]. Quick summary available at [[Link](https://www.tldrlegal.com/l/x11)].

## Citation

If you use find this paper/code useful, please consider citing:

```
@article{rai2023towards,
  		title={Towards Realistic Generative 3D Face Models},
  		author={Rai, Aashish and Gupta, Hiresh and Pandey, Ayush and Carrasco, Francisco Vicente and Takagi, Shingo Jason and Aubel, Amaury and Kim, Daeil and Prakash, Aayush and De la Torre, Fernando},
  		journal={arXiv preprint arXiv:2304.12483},
  		year={2023}
 		}
```
