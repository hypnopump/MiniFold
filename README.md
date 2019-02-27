# MiniFold
A mini version of Deep Learning for Protein Structure Prediction inspired by [DeepMind AlphaFold](https://deepmind.com/blog/alphafold/) algorithm.

## Summary


## Proposed Architecture 

The methods implemented are inspired by the DeepMind original post. Two different residual neural networks (ResNets) are used to predict a.) angles between adjacent aminoacids (AAs) and b.) distance between every pair of AAs of a protein. 

<img src="https://storage.googleapis.com/deepmind-live-cms/images/Origami-CASP-181127-r01_fig4-method.width-400.png" width="500" height="300">
Image from DeepMind's original blogpost.


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
* **DeepMind's original blog post**: []()
* ****: []()
* ****: []()
* ****: []()
* ****: []()

## Contribute
Hey there! New ideas are welcome: open/close issues, fork the repo and share your code with a Pull Request.
Clone this project to your computer:
 
`git clone https://github.com/EricAlcaide/MiniFold`
 
By participating in this project, you agree to abide by the thoughtbot [code of conduct](https://thoughtbot.com/open-source-code-of-conduct)
 
## Meta
 
* **Author's GitHub Profile**: [Eric Alcaide](https://github.com/EricAlcaide/)
* **Twitter**: [@eric_alcaide](https://twitter.com/eric_alcaide)
* **Email**: ericalcaide1@gmail.com