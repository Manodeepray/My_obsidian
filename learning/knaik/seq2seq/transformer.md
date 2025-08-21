
![[Pasted image 20250319013318.png]]


### positional Encoding 
In this work, we use sine and cosine functions of different frequencies: 

P E(pos,2i)     =  sin(pos/10000^(2i/dmodel)  ) 
P E(pos,2i+1) = cos(pos/10000^(2i/dmodel)  )

where pos is the position and i is the dimension. That is, each dimension of the positional encoding corresponds to a sinusoid. 

The wavelengths form a geometric progression from 2π to 10000 · 2π. 

We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset k, P Epos+k can be represented as a linear function of P Epos.
We also experimented with using learned positional embeddings [8] instead, and found that the two versions produced nearly identical results (see Table 3 row (E)). We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training

![[Pasted image 20250319160130.png]]

![[Pasted image 20250319160221.png]]