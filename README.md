# Invariant Weighted Frames for the Symmetric Group

This repository contains a demonstration of continuous invariant weighted frames (robust frames) for deep learning with permutation invariance. The definition and theoretical properties of robust frames, in addition to the construction of efficient (polynomial sized) robust frames for the actions of permutations, rotations, and orthogonal transformations on point clouds can be found in the paper:

- Nadav Dym, Hannah Lawrence, and Jonathan W. Siegel. "[Equivariant Frames and the Impossibility of Continuous Canonicalization.](https://arxiv.org/abs/2402.16077)" arXiv preprint arXiv:2402.16077 (2024).

In our experiments, we test different approaches to enforcing invariance on the following toy classification problem. Starting with the MNIST dataset, we process each digit into a two-dimensional randomly ordered point cloud consisting of $100$ points each. This dataset is generated and saved by the file **generate_point_cloud_MNIST.py**.

The goal is to learn to classify the digits based only upon this two-dimensional point cloud data. Since the point clouds are unordered, this classification problem is permutation invariant. We enforce the permutation invariance in five different ways: no invariance, invariance via a discontinuous canonicalization (sorting along the x-axis), invariance using two robust frames introduced in the paper (implemented via stochastic averaging), and invariance via stochastic averaging over the entire symmetric group.

Our experiments can be reproduced by running **point_cloud_mnist_classifier.py** after generating the point cloud data. From these experiments, we see that the testing accuracy is very bad if no invariance is enforced or if we average over the whole symmetric group. This is because the symmetric group is so large that both of these approaches effectively don't enforce any invariance, and without permutation invariance there is not nearly enough data to learn to classify the point clouds. The testing accuracy is much better when invariance is enforced by sorting along the x-axis, and an even further significant improvement is obtained by using the robust frames introduced in the paper. This shows that enforcing symmetry is essential to this problem, but that sacrificing continuity significantly lowers the accuracy of the model. The best approach requires both enforcing symmetry and also preserving continuity, which is achieved by the robust frames we introduce.

 ## Citation

 
    @article{dym2024equivariant,
      title={Equivariant Frames and the Impossibility of Continuous Canonicalization},
      author={Dym, Nadav and Lawrence, Hannah and Siegel, Jonathan W},
      journal={arXiv preprint arXiv:2402.16077},
      year={2024}
    }
