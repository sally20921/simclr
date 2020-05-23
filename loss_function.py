'''
the  contrastive prediction task 

aims to identify x_j in x_k for a given x_i

for convenience, we term it NT-Xent 
(the  normalized temperature-scaled cross entropy loss)
(optimized using LARS with linear learning rate  scaling 

LearningRate = 0.3 * BatchSize/256
weight decay of 10**-6)


'''
