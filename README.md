# Pytorch implementation of Rajaraman's work
### Rajaraman, S.; Zamzmi, G.; Folio, L.; Alderson, P.; Antani, S. Chest X-Ray Bone Suppression for Improving Classification of Tuberculosis-Consistent Findings. Diagnostics 2021, 11, 840. https://doi.org/10.3390/diagnostics11050840

A pre-trained model with the weights of the ResNet-BS bone suppression model is included for direct use. Run the model using **analysis_script.ipynb** on 256 x 256 grayscale CXR image to generate a soft-tissue image with suppressed bone shadows.

pytorch-msssim is sourced from: https://github.com/jorge-pessoa/pytorch-msssim
