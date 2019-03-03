# Repository for Google's AI program by DeepMind

The forked repository is of DeepMind's new Artificial Intelligence based program that allows protein structure prediction. 
The hypothesis of the program is that Protein structure prediction not just based on amino acid sequence but also between the
spatial folding of the amino acids in environment and the program using AI strategies to tackle this issue.

# Strategy of prediction

The strategy of prediction of protein structures from sequences starts with replacing the known intermediate sequences with known homology fragment models. DeepMind has developed a neural network that allows the generation of a dataset that has distance and angle predictions (more about the code further). The dataset once obtained is later posed through a gradient descent method to find the best possible structure based on the predictive dataset.

![GitHub Logo](/images/logo.png)
Format: ![Alt Text](url)
