# FeatureDescriptor-Reimplementation

- BDIP: 
  - v0.0: $O(n^2)$
  - v0.1: $O((\frac{n}{b})^2)$ b: block_size
  - v0.2: $O((log_b(n)^2)$ b: block_size
- BVLC: 
  - v0.0: $O(4n^2)$
  - v0.1: $O(4(\frac{n - 4s}{b})^2)$ b: block_size, s: stride
  - v0.2: $O(4(log_{b}{(n - 4s)})^2)$ b: block_size, s: stride

| | | |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/ScaleMind-C9308A/FeatureDescriptor-Reimplementation/blob/main/ExampleImage/CHGastro_Abnormal_037.png">  Original Image |  <img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/ScaleMind-C9308A/FeatureDescriptor-Reimplementation/blob/main/ExampleImage/BDIP_CHGastro_Abnormal_037.png"> BDIP Image |<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/ScaleMind-C9308A/FeatureDescriptor-Reimplementation/blob/main/ExampleImage/BVLC_CHGastro_Abnormal_037.png"> BVLC Image |
|<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/ScaleMind-C9308A/FeatureDescriptor-Reimplementation/blob/main/ExampleImage/CHGastro_Normal_047.png"> Original Image |  <img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/ScaleMind-C9308A/FeatureDescriptor-Reimplementation/blob/main/ExampleImage/BDIP_CHGastro_Normal_047.png"> BDIP Image |<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/ScaleMind-C9308A/FeatureDescriptor-Reimplementation/blob/main/ExampleImage/BVLC_CHGastro_Normal_047.png"> BVLC Image |
