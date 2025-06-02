# MatmulFree-cpu
CPU MatmulFree LLM

If you like our project, please give us a star ⭐ on GitHub for the latest updates.

## Quick Start

### Install & Run
Run:
```bash
git clone https://github.com/MengAiDev/MatmulFree-cpu
cd MatmulFree-cpu
pip install -e .
python generate.py
```

## Usage

You can use the `generate.py` script to generate text with the pre-trained models. The script supports various command-line arguments to customize the generation process. Change `name` variable in `generate.py` to use different models.

### Pre-trained Model Zoo
| Model Size     | Layer | Hidden dimension  | Trained tokens |
|:----------------|:------------:|:----------------:|:------------------:|
| [370M](https://huggingface.co/ridger/MMfreeLM-370M)  | 24  | 1024 | 15B  |
| [1.3B](https://huggingface.co/ridger/MMfreeLM-1.3B)  | 24 | 2048 | 100B  |
| [2.7B](https://huggingface.co/ridger/MMfreeLM-2.7B)  | 32  | 2560 | 100B  |

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
Part of the code is used from [MatmulFree](https://github.com/ridgerchu/matmulfreellm)

```bib
@article{zhu2024scalable,
title={Scalable MatMul-free Language Modeling},
author={Zhu, Rui-Jie and Zhang, Yu and Sifferman, Ethan and Sheaves, Tyler and Wang, Yiqiao and Richmond, Dustin and Zhou, Peng and Eshraghian, Jason K},
journal={arXiv preprint arXiv:2406.02528},
year={2024}
}
```