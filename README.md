# simclr

###default setting  

-For data augmentation, we use random crop and resize,  
color distortions, and Gaussian blur.  

-we use ResNet-50 as the base encoder network

-a 2  layer  MLP projectrion head to project the  representation to  a 128-dimensional latent  space

-As the  loss, we use NT-Xent, optimized using  LARS with  linear learning  rate  scaling (LearningRate =  0.3 X BatchSize/256) and weight decay 10^-6

-we  train at  batch size  4096 for 100 epochs 

-we use linear  warmup for the  first 10 epochs, and decay the  learning rate  with  the  cosine  decay schedule without  restarts 
