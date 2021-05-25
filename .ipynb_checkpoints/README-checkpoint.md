# Pytorch implementation of Rajaraman's work
### Rajaraman, S.; Zamzmi, G.; Folio, L.; Alderson, P.; Antani, S. Chest X-Ray Bone Suppression for Improving Classification of Tuberculosis-Consistent Findings. Diagnostics 2021, 11, 840. https://doi.org/10.3390/diagnostics11050840

We train several bone-suppression models with varying architecture on the Japanese Society of Radiological Technology (JSRT) CXR dataset and its bone-suppressed counterpart. 
The performance of the trained models is tested using the cross-institutional National Institutes of Health (NIH) clinical center (CC) dual-energy subtraction (DES) CXR dataset. The best-performing model is used to suppress bones in the Shenzhen and Montgomery TB collections. We then compare the performance of the CXR-retrained VGG-16 models trained with the non-bone-suppressed and bone-suppressed Montgomery TB datasets using several performance metrics and analyzed them for a statistically significant difference. The predictions of the non-bone-suppressed and bone-suppressed models are interpreted through class-selective relevance maps (CRM).



We have also included the weights of the best-performing ResNet-BS bone suppression model for direct use. Run the model on 256 x 256 grayscale CXR image to geneate a soft-tissue image with suppressed bone shadows. 


