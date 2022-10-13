# FeatureDescriptor-Reimplementation

- BDIP: 
  - v0.0: $O(n^2)$
  - v0.1: $O((\frac{n}{b})^2)$ b: block_size
  - v0.2: $O((log_b(n)^2)$ b: block_size
- BVLC: 
  - v0.0: $O(4n^2)$
  - v0.1: $O(4(\frac{n - 4s}{b})^2)$ b: block_size, s: stride
  - v0.2: $O(4(log_{b}{(n - 4s)})^2)$ b: block_size, s: stride
