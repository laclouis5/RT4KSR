# Towards Real-Time 4K Image Super-Resolution

**[Eduard Zamfir](https://scholar.google.com/citations?hl=en&user=5-FIWKoAAAAJ), [Marcos V. Conde](https://scholar.google.com/citations?user=NtB1kjYAAAAJ&hl=en), [Radu Timofte](https://scholar.google.com/citations?user=u3MwH5kAAAAJ&hl=en)**

[Computer Vision Lab, CAIDAS, University of Würzburg](https://www.informatik.uni-wuerzburg.de/computervision/home/)

Work part of the [NTIRE Real-Time 4K Super-Resolution](https://cvlai.net/ntire/2023/) Challenge @ CVPR 2023 in Vancouver

---

<img src="assets/rt4ksr_teaser.png" width="1000" />

## Abstract

Over the past few years, high-definition videos and images in 720p (HD), 1080p (FHD), and 4K (UHD) resolution have become standard. While higher resolutions offer improved visual quality for users, they pose a significant chal- lenge for super-resolution networks to achieve real-time performance on commercial GPUs. This paper presents a comprehensive analysis of super-resolution model designs and techniques aimed at efficiently upscaling images from 720p and 1080p resolutions to 4K. We begin with a simple, effective baseline architecture and gradually modify its design by focusing on extracting important high-frequency details efficiently. This allows us to subsequently downscale the resolution of deep feature maps, reducing the overall computational footprint, while maintaining high reconstruction fidelity. We enhance our method by incorporating pixel-unshuffling, a simplified and speed-up reinterpretation of the basic block proposed by NAFNet, along with structural re-parameterization. We assess the performance of the fastest version of our method in the new [NTIRE Real-Time 4K Super-Resolution](https://cvlai.net/ntire/2023/) challenge and demonstrate its potential in comparison with state-of-the-art efficient super-resolution models when scaled up. Our method was tested successfully on high-quality content from photography, digital art, and gaming content.

---

<a href="https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/html/Zamfir_Towards_Real-Time_4K_Image_Super-Resolution_CVPRW_2023_paper.html"><img src="assets/paper.png" width="200" border="0"></a> <a href="https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/html/Conde_Efficient_Deep_Models_for_Real-Time_4K_Image_Super-Resolution._NTIRE_2023_CVPRW_2023_paper.html"><img src="assets/report.png" width="200" border="0"></a>

&ensp;&ensp;&emsp;&emsp;&emsp;&emsp;[Paper](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/html/Zamfir_Towards_Real-Time_4K_Image_Super-Resolution_CVPRW_2023_paper.html)
&ensp;&ensp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[Challenge Report](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/html/Conde_Efficient_Deep_Models_for_Real-Time_4K_Image_Super-Resolution._NTIRE_2023_CVPRW_2023_paper.html)

---

## Installation

Install the repository using uv:

```shell
uv sync
```

PyTorch 2.5 may not be compatible with Python 3.13 at the time of writing. If an error occurs during installation, switch to Python 3.12 or bellow.

---

## Usage

### Export ONNX Model

Export the ONNX FP32 and FP16 models using (x2 model):

```shell
uv run python code/export.py --checkpoint-id rt4ksr_x2 --scale 2 --arch rt4ksr_rep --is-train --rep
```

---

## Citation

```
@InProceedings{Zamfir_2023_CVPR,
    author    = {Zamfir, Eduard and Conde, Marcos V. and Timofte, Radu},
    title     = {Towards Real-Time 4K Image Super-Resolution},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {1522-1532}
}
```
