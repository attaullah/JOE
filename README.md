# [WIP] code 
code for experimenting with jointly optimising embeddings.
## Getting started
Start with cloning the repo:
```bash
git clone https://github.com/attaullah/JOE.git
cd JOE/
```
### Environment setup
Required packages
```bash
tensorflow
scikit-learn
absl-py
scipy
```
and pip packages
```bash
tensorflow-addons
tensorflow-datasets
```

### Data preparation
MNIST, Fashion-MNIST, SVHN, and CIFAR-10 datasets are loaded using   [TensorFlow  datasets](https://www.tensorflow.org/datasets). 
package.
By default, tensorflow_datasets package will save datasets at `~/ tensorflow_datasets/` directory.

For PlantVillage dataset [1] please follow instructions at
 [plant-disease-dataset](https://github.com/attaullah/downsampled-plant-disease-dataset). The downloaded files are 
expected to be saved in the `data/` directory. 



## Example usage
Training can be started using the `train.py` script. Details of self-explanatory command-line 
arguments can be seen by passing `--helpfull` to it.


```bash
 python joe.py --helpfull
 
       USAGE: joe.py [flags]
flags:
  --batch_size: size of mini-batch
    (default: '128')
    (an integer)
  --confidence_measure: <1-nn>: confidence measure for pseudo-label selection.
    (default: '1-nn')
  --dataset: <cifar10|svhn|plant32|plant64|plant96>: dataset name
    (default: 'cifar10')
  --epochs: initial training epochs
    (default: '200')
    (an integer)
  --epochs_per_m_iteration: number of epochs per meta-iterations
    (default: '200')
    (an integer)
  --gpu: gpu id
    (default: '0')
  --lbl: <knn|lda|rf|lr>: shallow classifiers for labelling for metric learning losses
    (default: 'knn')
  --lr: learning_rate
    (default: '0.001')
    (a number)
  --lt: <cross-entropy|triplet>: loss_type: cross-entropy, triplet,  arcface or contrastive.
    (default: 'cross-entropy')
  --margin: margin for triplet loss calculation
    (default: '1.0')
    (a number)
  --meta_iterations: number of meta_iterations
    (default: '25')
    (an integer)
  --network: <wrn-28-2|resnet18|vgg16|resnet34|resnet20|resnet50>: network architecture.
    (default: 'wrn-28-2')
  --opt: <adam|sgd|sgdw|rmsprop>: optimizer.
    (default: 'adam')
  --pre: prefix for log directory
    (default: '')
  --[no]self_training: apply self-training
    (default: 'false')
  --[no]semi: True: N-labelled training, False: All-labelled training
    (default: 'true')
  --verbose: verbose
    (default: '1')
    (an integer)
  --[no]weights: random or ImageNet pretrained weights 
 ```