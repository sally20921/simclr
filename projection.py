'''
a small neural network projection head g 
that maps representations to the space  
where contrastive loss is applied.
We use a MLP with one hidden layer to obtain
z_i = g(h_i) = W_2*ReLU(W_1 * h_i)

we find it beneficial to define contrastive loss 
on z_i rather  than h_i
'''
