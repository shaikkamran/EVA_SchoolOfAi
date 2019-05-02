## What are Channels and Kernels?

A *kernel* is a feature extractor from image which extracts features like edges,patterns,colors ,gradients through the process of Convolution. 
Before the deep learning and CNNs came into existence, ComputerVision experts use to design these kernels by hand , for Image enhancement and all the other Image processing work,
Now these kernels are randomly initialized for tarining the CNNs and as the network trains these feature extractors nudge themselves in a way to extract better and hence reducing the overall loss.

## Why should we only ue 3x3 Kernels?

The reasons we use 3*3 kernels is:
- We can get same receptive field by using lesser number of parameters. For eg a 11*11 filter will be 121 parameters but for getting
the same receptive field we can use five 3*3 filters i.e., 45 parameters.
- Our Nvidia gpus are much optimized to work faster/ faster matrix multiplications with 3*3 filters, and even though if we feed with bigger flters 
the gpus break down those into 3*3 and perform multiplications.
- A 3*3 filter provides symmetry ,there is an anchor point which is important to detect changes in gradient/edges which cant be possible with even no of filter size.

## How many times do we need to perform 3x3 convolution operation to reach 1x1 from 199x199?

We need *99* convolution operations

199 | 3x3 | 197 | 3x3 | 195 | 3x3 | 193 | 3x3 | 191 | 3x3 | 189 | 3x3 | 187 | 3x3 | 185 | 3x3 | 183 | 3x3 | 181 | 3x3 | 179 | 3x3 | 177 | 3x3 | 175 | 3x3 | 173 | 3x3 | 171 | 3x3 | 169 | 3x3 | 167 | 3x3 | 165 | 3x3 | 163 | 3x3 | 161 | 3x3 | 159 | 3x3 | 157 | 3x3 | 155 | 3x3 | 153 | 3x3 | 151 | 3x3 | 149 | 3x3 | 147 | 3x3 | 145 | 3x3 | 143 | 3x3 | 141 | 3x3 | 139 | 3x3 | 137 | 3x3 | 135 | 3x3 | 133 | 3x3 | 131 | 3x3 | 129 | 3x3 | 127 | 3x3 | 125 | 3x3 | 123 | 3x3 | 121 | 3x3 | 119 | 3x3 | 117 | 3x3 | 115 | 3x3 | 113 | 3x3 | 111 | 3x3 | 109 | 3x3 | 107 | 3x3 | 105 | 3x3 | 103 | 3x3 | 101 | 3x3 | 99 | 3x3 | 97 | 3x3 | 95 | 3x3 | 93 | 3x3 | 91 | 3x3 | 89 | 3x3 | 87 | 3x3 | 85 | 3x3 | 83 | 3x3 | 81 | 3x3 | 79 | 3x3 | 77 | 3x3 | 75 | 3x3 | 73 | 3x3 | 71 | 3x3 | 69 | 3x3 | 67 | 3x3 | 65 | 3x3 | 63 | 3x3 | 61 | 3x3 | 59 | 3x3 | 57 | 3x3 | 55 | 3x3 | 53 | 3x3 | 51 | 3x3 | 49 | 3x3 | 47 | 3x3 | 45 | 3x3 | 43 | 3x3 | 41 | 3x3 | 39 | 3x3 | 37 | 3x3 | 35 | 3x3 | 33 | 3x3 | 31 | 3x3 | 29 | 3x3 | 27 | 3x3 | 25 | 3x3 |  23 | 3x3 | 21 | 3x3 | 19 | 3x3 | 17 | 3x3 | 15 | 3x3 | 13 | 3x3 | 11 | 3x3 | 9 | 3x3 | 7 | 3x3 | 5 | 3x3 | 3 | 3x3 | 1
