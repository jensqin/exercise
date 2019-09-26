// The input data is a vector 'y' of length 'N'.
data {
  int<lower=0> N;
  int<lower=0> k;
  vector[N] y;
  matrix[N, k] X;
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  vector[k] b;
  real<lower=0> sigma;
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  y ~ normal(X * b, sigma);
}

