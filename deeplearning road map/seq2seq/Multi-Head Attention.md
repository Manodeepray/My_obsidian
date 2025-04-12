**Multi-head Attention** is a module for attention mechanisms which runs through an attention mechanism several times in parallel. The independent attention outputs are then concatenated and linearly transformed into the expected dimension. Intuitively, multiple attention heads allows for attending to parts of the sequence differently (e.g. longer-term dependencies versus shorter-term dependencies).
```
MultiHead(Q,K,V)=[head1,…,headh]W0

where headi=Attention(QWiQ,KWiK,VWiV)
```

Above W are all learnable parameter matrices.

Note that [scaled dot-product attention](https://paperswithcode.com/method/scaled) is most commonly used in this module, although in principle it can be swapped out for other types of attention mechanism

![[Pasted image 20250319005736.png]]