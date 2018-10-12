### Notes:

- Dataset Conversion
    - Create binary and instance training images using source image and labels from tuSimple dataset
- Pytorch Dataset
    - Create a Pytorch dataset from tuSimple dataset
    - Image preprocessing for `__getitem__`
        - Use nearest neighbor interpolation since linear interpolation changes pixel values near segmentation edges
- Encoder
    - Use VGG16 to encode input RGB image
    - Output the result of the last 3 pooling layer (as said in the paper)
- Decoder
    - Use conv and deconv layers
    - Output should be pixel embeddings and binary segmentation
    