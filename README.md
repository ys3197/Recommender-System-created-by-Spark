# Recommender-System-created-by-Spark

The recommendation model uses Spark's alternating least squares (ALS) method to learn latent factor representations for users and items.  This model has some hyper-parameters that you should tune to optimize performance on the validation set, notably: 

  - the *rank* (dimension) of the latent factors,
  - the *regularization* parameter, and
  - *alpha*, the scaling parameter for handling implicit feedback (count) data.
