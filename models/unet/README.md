# U-net models

## Residual U-net

- This configuration tries to be the midpoint between traditional U-net literature and Chen et al. 2021

- Changes:
  1. They use single convolutions we use double convolutions (with a residual connection)
     - our model is a little deeper
     - the increased depth could be alleviated by the residual connections
  2. The shapes don't quite match
     - they're using unpadded convolutions
     - it's a bit nicer to keep powers of 2 (especially for reconstruction) so we're using padded convolutions