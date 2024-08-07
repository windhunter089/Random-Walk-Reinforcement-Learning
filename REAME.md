# Project 1: Temporal Difference Learning (TD(λ))

## Overview

This project replicates the Random Walk experiment conducted by Richard Sutton to demonstrate the efficiency of Temporal Difference (TD) Learning. The report details the implementation and results of applying TD(λ) to solve the random walk problem, highlighting the dependency of the learning model on parameters such as λ and the learning rate.

## Contents

- `Project_1_report_tpham328.docx`: Detailed project report.
- `random_walk_experiment.py`: Python script used to replicate Sutton's Random Walk experiment.


## Abstract

Richard Sutton's discussion about Temporal Difference Learning emphasizes its efficiency and lower memory requirements compared to traditional supervised learning methods. This project replicates Sutton's Random Walk experiment using TD(λ) on generated data. It explores the impact of various parameters like λ and the learning rate on the learning model.

## Problem Description

The random walk problem consists of 7 states arranged alphabetically from A to G. The person starts at the center state D and can move left or right with equal probability (50%). The game ends when the person reaches either state A (reward = 0) or state G (reward = 1). Non-terminal states (B, C, D, E, F) are represented as column identity vectors.

## Temporal Difference Learning

For each observation, the learner generates a sequence of predictions to estimate the reward. The weight vector is updated using the TD learning rule. The Widrow-Hoff supervised learning procedure is a special case of TD(λ) where λ = 1.

## Experiments

### Experiment 1

The weight vector was updated after each sequence, accumulated over 10 sequences, until convergence. The root mean square error (RMS) of the predicted weight and the ideal weight was used as the error metric. Results showed that TD(1) produced worse results compared to TD methods with λ < 1, matching the observations in Sutton's paper.

### Experiment 2

A pair-wise value of λ and α was used, with each training set presented only once. The learning rate significantly impacted the error. The best λ for TD learners was either 0.2 or 0.3, and the results matched Sutton's graph shape closely.

## Results

The project successfully replicated Sutton's Random Walk experiment, demonstrating the importance of selecting appropriate learning parameters. The shape of the generated graphs closely matched Sutton's, confirming the effectiveness of TD learning.

## Conclusion

The replication of Sutton's experiment highlighted the efficiency of TD learning over supervised learning methods. The project's results were consistent with Sutton's findings, emphasizing the significance of parameter selection and the impact of different training sets.

## References

1. Richard S Sutton and Andrew G Barto. Reinforcement Learning: An Introduction (2020)
2. Richard Sutton. “Learning to Predict by the Method of Temporal Differences”. Machine Learning, 3 (Aug 1988), pp 9-44

## Contact

For any inquiries, please contact Trung Pham at [trungpham89@gmail.com](mailto:trungpham89@gmail.com).
