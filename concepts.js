const MLConcepts = {
  concepts: {
    feature: "x-axis", // ---> the inputted data
    label: "y-axis", // -----> the predicted output / we predict Labels !
    models: [
      {
        name: "regression",
        useCases: [
          "What is the value of a house in California?",
          "What is the probability that a user will click on this ad?",
        ],
        equation: "y = b + wx", // y = label; x = feature; b = bias; w = weight of the feature(aka slope)
        keywords: [
          "empirical risk minimization",
          "square loss functions",
          "iterative process/convergence of loss functions",
          "plot of (w) vs (loss) === convex parabola graph || gradient/slope descent algorithm",
          "learning rate; hyperparameter || gradient magnitude * learning rate ~ convergence rate",
          "stochastic gradient descent"
        ],
      },
      {
        name: "classification",
        useCases: [
          "Is a given email message spam or not spam?",
          "Is this an image of a dog, a cat, or a hamster?",
        ],
      },
    ],
  },
};
