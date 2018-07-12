# reproduce-chexnet
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/jrzech/reproduce-chexnet/master?filepath=Explore_Predictions.ipynb)

Provides Python code to reproduce model training, predictions, and heatmaps from the [CheXNet paper](https://arxiv.org/pdf/1711.05225) that predicted 14 common diagnoses using convolutional neural networks in over 100,000 NIH chest x-rays.

![Illustration](illustration.png?raw=true "Illustration")


## Getting Started:
Click on the `launch binder` button at the top of this `README` to launch a remote instance in your browser using [binder](https://mybinder.org/). This requires no local configuration and lets you get started immediately. Open `Explore_Predictions.ipynb`, run all cells, and follow the instructions provided to review a selection of included [chest x-rays from NIH](https://arxiv.org/pdf/1705.02315.pdf).

To configure your own local instance (assumes [Anaconda is installed](https://www.anaconda.com/download/); can be run on Amazon EC2 p2.xlarge instance if you do not have a GPU):

```git clone https://www.github.com/jrzech/reproduce-chexnet.git
cd reproduce-chexnet
conda env create -f environment.yml
source postBuild
source activate reproduce-chexnet
```

## Replicated results:
This reproduction achieved average test set AUC 0.836 across 14 findings compared to 0.841 reported in original paper:

<div>
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>retrained auc</th>
      <th>chexnet auc</th>
    </tr>
    <tr>
      <th>label</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Atelectasis</th>
      <td>0.8161</td>
      <td>0.8094</td>
    </tr>
    <tr>
      <th>Cardiomegaly</th>
      <td>0.9105</td>
      <td>0.9248</td>
    </tr>
    <tr>
      <th>Consolidation</th>
      <td>0.8008</td>
      <td>0.7901</td>
    </tr>
    <tr>
      <th>Edema</th>
      <td>0.8979</td>
      <td>0.8878</td>
    </tr>
    <tr>
      <th>Effusion</th>
      <td>0.8839</td>
      <td>0.8638</td>
    </tr>
    <tr>
      <th>Emphysema</th>
      <td>0.9227</td>
      <td>0.9371</td>
    </tr>
    <tr>
      <th>Fibrosis</th>
      <td>0.8293</td>
      <td>0.8047</td>
    </tr>
    <tr>
      <th>Hernia</th>
      <td>0.9010</td>
      <td>0.9164</td>
    </tr>
    <tr>
      <th>Infiltration</th>
      <td>0.7077</td>
      <td>0.7345</td>
    </tr>
    <tr>
      <th>Mass</th>
      <td>0.8308</td>
      <td>0.8676</td>
    </tr>
    <tr>
      <th>Nodule</th>
      <td>0.7748</td>
      <td>0.7802</td>
    </tr>
    <tr>
      <th>Pleural_Thickening</th>
      <td>0.7860</td>
      <td>0.8062</td>
    </tr>
    <tr>
      <th>Pneumonia</th>
      <td>0.7651</td>
      <td>0.7680</td>
    </tr>
    <tr>
      <th>Pneumothorax</th>
      <td>0.8739</td>
      <td>0.8887</td>
    </tr>
  </tbody>
</table>
</div>

## Results available in pretrained folder:
- `aucs.csv`: test AUCs of retrained model vs original ChexNet reported results
- `checkpoint`: saved model checkpoint
- `log_train`: log of train and val loss by epoch
- `preds.csv`: individual probabilities for each finding in each test set image predicted by retrained model

## NIH Dataset
To explore the full dataset, [download images from NIH (large, ~40gb compressed)](https://nihcc.app.box.com/v/ChestXray-NIHCC),
extract all `tar.gz` files to a single folder, and provide path as needed in code.

## Train your own model!
Please note: a GPU is required to train the model. You will encounter errors if you do not have a GPU available and CUDA installed and you attempt to retrain. With a GPU, you can retrain the model with `retrain.py`. Make sure you download the full NIH dataset before trying this. If you run out of GPU memory, reduce `BATCH_SIZE` from its default setting of 16.

If you do not have a GPU, but wish to retrain the model yourself to verify performance, you can replicate the model using Amazon EC2's p2.xlarge instance ($0.90/hr at time of writing) with an AMI that has CUDA installed (e.g. Deep Learning AMI (Ubuntu) Version 8.0 - ami-dff741a0). After [creating and ssh-ing into the EC2 instance](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html), follow the instructions in Getting Started above to configure your environment. If you have no experience with Amazon EC2, [fast.ai's tutorial is a good place to start](http://course.fast.ai/lessons/aws.html)

## Note on training
I use SGD+momentum rather than the Adam optimizer as described in the original [paper](https://arxiv.org/pdf/1711.05225.pdf). I achieved better results with SGD+momentum, as has been reported in [other work](https://arxiv.org/pdf/1705.08292.pdf).

## Note on data
A sample of 621 test NIH chest x-rays enriched for positive pathology is included with the repo to faciliate immediate use and exploration in the `Explore Predictions.ipynb` notebook. The [full NIH dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC) is required for model retraining.

## Use and citation
My goal in releasing this code is to increase transparency and replicability of deep learning models in radiology. I encourage you to use this code to start your own projects. If you do, please cite the repo:

```@misc{Zech2018,
  author = {Zech, J.},
  title = {reproduce-chexnet},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/jrzech/reproduce-chexnet}}
}
```

## Acknowledgements
With deep gratitude to researchers and developers at PyTorch, NIH, Stanford, and Project Jupyter, on whose generous work this project relies. With special thanks to Sasank Chilamkurthy, whose demonstration code was incorporated into this project. PyTorch is an incredible contribution to the research community.
