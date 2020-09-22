# KHD_2020
image classification for paranasal sinuses data for Naver Korea Health Datathon 2020

## Sources
- baseline : [[github]](https://github.com/KYBiMIL/KHD_2020/tree/master/pytorch)
- Issues : [[github]](https://github.com/Korea-Health-Datathon/KHD2020)
- QnA : [[link]](https://app.sli.do/event/th7tsarn/live/questions)
- ranking : https://ai.nsml.navercorp.com/ranking
- papers
  * paranasal sinus X-ray paper [[paper]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6629570/pdf/qims-09-06-942.pdf)
  * cbam [[blog]](https://blog.lunit.io/2018/08/30/bam-and-cbam-self-attention-modules-for-cnn/) [[github]](https://github.com/arp95/cbam_cnn_architectures_image_classification/blob/master/notebooks/resnet50_cbam.ipynb)
  * anomaly detection [[paper]](http://s-space.snu.ac.kr/handle/10371/161931)

## Experiments
|NSML Session No.|Data Augmentation|Model|Checkpoint|Performance|Info|
|:-:|:-:|:-:|:-:|:-:|:-:|
|42|Split left/right + Resize 256 + RandomCrop 224 + Normalize with Session No. 26|Conv2d(1, 3, 3) + BN + ReLU + ResNet50 + cbam + 3 FC (1000, 1000, 128, 4)|38|0.1424815478|ratio 0.1, batch 32, Adam lr 1e-5|
|19|Resize 256|-|-|-|Get mean/std of images|
|26|Split left/right + Resize 256|-|-|-|Get mean/std of images|
|31|Split left/right + Resize 256 + RandomCrop 224 + Normalize with Session No. 26|Conv2d(1, 3, 3) + BN + ReLU + ResNet50 + 3 FC (1000, 1000, 128, 4)|9|0.2058235431|ratio 0.1, batch 32, Adam lr 1e-5|
|36|Split left/right + Resize 256 + RandomCrop 224 + Normalize with Session No. 26|Conv2d(1, 3, 3) + BN + ReLU + ResNet152 + 3 FC (1000, 1000, 128, 4)|27|0.1154561488|ratio 0.1, batch 32, Adam lr 1e-5|
|38|Split left/right + Resize 256 + RandomCrop 224 + Normalize with Session No. 26|Conv2d(1, 3, 3) + BN + ReLU + ResNet34 + 3 FC (1000, 1000, 128, 4)|12|0.1667924528|ratio 0.1, batch 32, Adam lr 1e-5|
