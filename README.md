# MiniFold
A mini version of Deep Learning for Protein Structure Prediction inspired by [DeepMind AlphaFold](https://deepmind.com/blog/alphafold/) algorithm.

## Summary


## Proposed Architecture 

The methods implemented are inspired by the DeepMind original post. Two different residual neural networks (ResNets) are used to predict **angles** between adjacent aminoacids (AAs) and **distance** between every pair of AAs of a protein. 

<div style="text-align:center">
	<img src="https://storage.googleapis.com/deepmind-live-cms/images/Origami-CASP-181127-r01_fig4-method.width-400.png" width="500" height="300">
</div>

Image from DeepMind's original blogpost.

### Distance prediction

The ResNet for distance prediction is built as a 2D-ResNet and takes as input tensors of shape LxLxN (a normal image would be LxLx3). The window length is set to 200 (we only train and predict proteins of less than 200 AAs) and smaller proteins are padded to match the window size. No larger proteins nor crops of larger proteins are used.

The 41 channels of the input are distributed as follows: 20 for AAs in one-hot encoding (LxLx20), 1 for the Van der Waals radius of the AA encoded previously and 20 channels for the Position Specific Scoring Matrix).

The network is comprised of packs of residual blocks with the architecture below illustrated with blocks cycling through 1,2,4 and 8 strides plus a first normal convolutional layer and the last convolutional layer where a Softmax activation function is applied to get an output of LxLx7 (6 classes for different distance + 1 trash class for the padding that is less penalized).

<div style="text-align:center">
	<img src="imgs/elu_resnet_2d.png">
</div>

Architecture of the residual block used. A mini version of the block in [this description](http://predictioncenter.org/casp13/doc/presentations/Pred_CASP13-DeepLearning-AlphaFold-Senior.pdf)

The network was trained with 134 proteins and evaluated with 16 more. Clearly unsufficient data, but memory constraints didn't allow for more. Comparably, AlphaFold was trained with 29k proteins.




## Future
The future directions of the project as well as planned/work-in-progress improvements are extensively exposed in the [FUTURE.md](FUTURE.md) file.

*"Science is a Work In Progress."*


## Limitations
This project has been developed in one week by 1 person and,, therefore, many limitations have appeared.
They will be listed below in order to give a sense about what this project is and what it's not.

* **No usage of Multiple Sequence Alignments (MSA)**: The methods developed in this project don't use [MSA](https://en.wikipedia.org/wiki/Multiple_sequence_alignment) nor MSA-based features as input. 
* **Computing power/memory**: Development of the project has taken part in a computer with the following specs: Intel i7-6700k, 8gb RAM, NVIDIA GTX-1060Ti 6gb and 256gb of storage. The capacity for data exploration, processing, training and evaluating the models is limited.
* **GPU/TPUs for training**: The models were trained and evaluated on a single GPU. No cloud servers were used. 
* **Time**: One week of development during spare time. Ideas that might be worth testing in the future are described [here]().
* **Domain expertise**: No experts in the field. The author knows the basics of Biochemistry and Deep Learnning.
* **Data**: The average paper about Protein Structure Prediction uses a personalized dataset acquired from the Protein Data Bank (PDB). No such dataset was used. Instead, we used a subset of the [ProteinNet](https://github.com/aqlaboratory/proteinnet) dataset from CASP7. Our models are rained with 150 proteins (distance prediction) and 600 proteins (angles prediction). 

Due to these limitations and/or constraints, the precission/accuracy the methods here developed can achieve is limited when compared against SOTA algorithms.


## References
* [DeepMind original blog post](https://deepmind.com/blog/alphafold/)
* [AlphaFold @ CASP13: “What just happened?”](https://moalquraishi.wordpress.com/2018/12/09/alphafold-casp13-what-just-happened/#s2.2)
* []()
* []()
* []()
* []()
* []()
* []()
* []()

## Contribute
Hey there! New ideas are welcome: open/close issues, fork the repo and share your code with a Pull Request.
Clone this project to your computer:
 
`git clone https://github.com/EricAlcaide/MiniFold`
 
By participating in this project, you agree to abide by the thoughtbot [code of conduct](https://thoughtbot.com/open-source-code-of-conduct)
 
## Meta
 
* **Author's GitHub Profile**: [Eric Alcaide](https://github.com/EricAlcaide/)
* **Twitter**: [@eric_alcaide](https://twitter.com/eric_alcaide)
* **Email**: ericalcaide1@gmail.com