'''
projection head g 
that maps representations 
to the  space  where  contrastive  loss is applied. 
(128-dimensional latent space) 

2-layer MLP projection head  

to obtain z_i = W*ReLU(W*h_i)


self.projector = nn.Sequential(
    nn.Linear()
    nn.ReLU()
    nn.Linear()
'''
