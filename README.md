# Recommender-System-created-by-Spark

The recommendation model uses Spark's alternating least squares (ALS) method to learn latent factor representations for users and items.  This model has some hyper-parameters that could tune to optimize performance on the validation set, notably: 

  - the *rank* (dimension) of the latent factors,
  - the *regularization* parameter, and
  - *alpha*, the scaling parameter for handling implicit feedback (count) data.
  
The user and item identifiers (strings) are also transfomred into numerical index representations to work properly with Spark's ALS model. The team also conducted a thorough evaluation of different modification strategies (e.g., log compression, or dropping low count values) and their impact on overall accuracy.
