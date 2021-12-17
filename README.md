# Python Implementation for Linde-Buzo-Gray / Generalized Lloyd Algorithm

This is a small set of Python functions that implement the
[Generalized-Lloyd or Linde-Buzo-Gray Algorithm](https://en.wikipedia.org/wiki/Linde%E2%80%93Buzo%E2%80%93Gray_algorithm)
for vector quantization. It allows clustering of vectors of any dimension. This is helpful for example for
image classification when using the SIFT or SURF algorithms where you can cluster the feature vectors. It might
be also useful if you want to cluster a large amount of points on a map.

See also: http://mkonrad.net/projects/gen_lloyd.html

See also my original [Java implementation](https://github.com/internaut/JGenLloydCluster).

The repository also contains an IPython notebook to visualize how this algorithm works.

The source-code is provided under [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).
