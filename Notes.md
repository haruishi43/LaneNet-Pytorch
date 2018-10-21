Lane keep technology

More like lane detection or drivable area detection

KTTI dataset


“Towards End-to-End Lane Detection: an Instance Segmentation approach”


ABSTRACT:

- Traditional lane detection methods rely on a combination of highly-specialized, hand-crafted features and heuristics, usually followed by post-processing techniques, that are computationally expensive and prone to scalability due to road scene variations
- Recent approaches
    - Deep learning models
        - Pixel wise lane segmentation
            - Useful since they have big receptive field
        - Cannot cope with lane changes (ego-lanes)
- This paper goes beyond the limitations and propose to cast the lane detection problem as an instance segmentation problem
- Learned perspective transform
    - Contact to bird’s eye view transformation
- Fast lane detection algorithm that runs at 50 fps
    - Can handle a variable number of lanes and cope with lane changes
-  Used tuSimple dataset


INTRODUCTION:

- Overview:
    - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.233.1475&rep=rep1&type=pdf
    - Traditional methods are prone to robustness issues due to road scene variations that can not be easily modeled by such model-based systems
    - Recently:
        - Handcrafted feature detectors with deep networks to learn dense predictions
            - Pixel-wise lane segmentations (Gopalan et al.)
        - CNN with RANSAC algorithm
        - DualViewCNN (DVCNN)
            - Front view and top view image simultaneously to exclude false detections and remove non-club-shaped structure respectively
        - CNN and RNN
            - CNN to detect geometric lane attributes
            - RNN to detect lane
        - Big receptive field
            - Allows them to find lanes even when no markings are present in the image
    - Binary segmentation images needs to be divided into different lane instances
- With the success of dense prediction networks in semantic segmentation and instance segmentation tasks
    - Instance segmentation
        - identify object outlines at the pixel level
        - Object level semantic segmentation that accounts for instances
- Lane segmentation branch
    - Outputs:
        - Background or lane (2 classes)
- Lane embedding branch
    - Further disentangles the segmented lane pixels into different lane instances
- Having these two branches can fully utilize the power of the lane segmentation branch without it having to assign different classes to different lanes
    - Lane embedding branch, which is trained using a clustering loss function, assigns a lane id to each pixel rom the lane segmentation branch while ignoring the background pixels
- After estimating lane instances (which pixels belongs to which lane)
    - convert each of them to parametric description
    - Traditional methods: “bird’s eye view”
    - Our method:
        - Apply perspective transform before fitting a curve
        - Train the perspective transformation matrix


METHOD:

- Train network end-to-end for lane detection
    - Lane detection -> instance segmentation problem
    - This network is called LaneNet
- LaneNet outputs a collection of pixels per lane
- Have to fit a curve through these pixels to get the lane parametrization


LANENET:
- Two parts in the instance segmentation:
    - Segmentation
        - Binary segmentation:
            - Trained to output a binary segmentation map
            - Indicate which pixel belongs to a lane and which is not
            - Trained with standard cross-entropy loss function
            - 2 classes {lane/background} -> highly unbalanced
                - Used inverse class weighting
        - Instance segmentation:
            - https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46
            - Lane instance embedding
            - One-shot method based on distance metric learning
                - Can easily be integrated with standard feed-forward networks
            - Trained to output an embedding for each lane pixel 
                - the distance between pixel embeddings belonging to the same lane is small
                - The distance between pixel embeddings belonging to the different lane is maximized
            - Pixel embeddings of the same lane will cluster together
                - unique cluster per lane
            - Two terms
                - Variance term (L_var)
                    - Applies a pull force on each embedding towards the mean embedding of a lane
                - Distance term (L_dist)
                    - Pushes the cluster centers away from each other
                - Both terms a hinged by threshold
                - Total loss = L_var + L_dist
    - Clustering
        - Iterative procedure
        - All embeddings are assigned to a lane
        - Use mean shift to shift closer to eh cluster center and then do the thresholding
    - Network architecture
        - Encoder-decoder network Net
            - Consequently modified into a two-branched network
            - Shares the first two stages out of three for each branch
            - Last layer output 
                - 1 channel image for binary segmentation
                - N channel image with N as the embedding dimension for instance segmentation
        - Honestly, what is an ENet?
        - 
    - Two parts are jointly trained in a multi-task network


RESULTS:

- Dataset
    - 3626 training images
- LaneNet:
    - Embedding N =4
    - delta_v = 0.5
    - delta_d = 3
    - 512x256
    - Adam
    - Batch size = 8
    - Lr = 5e-4