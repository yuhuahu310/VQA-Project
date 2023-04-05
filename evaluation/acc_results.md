| Model Name                        | Mode  | Overall Accuracy | Accuracy by Answer Type (Other) | Accuracy by Answer Type (Yes/No) | Accuracy by Answer Type (Unanswerable) | Accuracy by Answer Type (Number) |
|-----------------------------------|-------|------------------|---------------------------------|----------------------------------|---------------------------------------|---------------------------------|
| LSTM                              | Train | 0.3750           | 0.3106                          | 0.2839                           | 0.5451                                | 0.4751                          |
| LSTM                              | Val   | 0.1987           | 0.1277                          | 0.1236                           | 0.3510                                | 0.0917                          |
| T5                                | Train | 0.4736           | 0.2935                          | 0.8929                           | 0.8600                                | 0.2561                          |
| T5                                | Val   | 0.4690           | 0.2766                          | 0.7421                           | 0.8100                                | 0.3125                          |
| ResNet                            | Train | 0.9252           | 0.8934                          | 0.9834                           | 0.9925                                | 0.9581                          |
| ResNet                            | Val   | 0.3635           | 0.2339                          | 0.3231                           | 0.6277                                | 0.1687                          |
| ViT                               | Train | 0.7439           | 0.6553                          | 0.5947                           | 0.9923                                | 0.6910                          |
| ViT                               | Val   | 0.3877           | 0.2276                          | 0.2969                           | 0.7187                                | 0.1750                          |
| CLIP                              | Train | 0.9193           | 0.8825                          | 0.9841                           | 0.9980                                | 0.9419                          |
| CLIP                              | Val   | 0.4520           | 0.2615                          | 0.6031                           | 0.8055                                | 0.3167                          |
| CompetitiveBaseline_CrossAttention| Train | 0.6658           | 0.6801                          | 0.9771                           | 0.5704                                | 0.7764                          |
| CompetitiveBaseline_CrossAttention| Val   | 0.6257           | 0.6490                          | 0.9446                           | 0.5313                                | 0.7521                          |
| CompetitiveBaseline_CLIP          | Train | 0.7794           | 0.8589                          | 0.8001                           | 0.5692                                | 0.9522                          |
| CompetitiveBaseline_CLIP          | Val   | 0.7336           | 0.8425                          | 0.7138                           | 0.5165                                | 0.9708                          |
| ViLT                              | Train | 0.1783           | 0.1881                          | 0.7009                           | 0.0692                                | 0.0751                          |
| ViLT                              | Val   | 0.1777           | 0.1931                          | 0.6749                           | 0.0825                                | 0.0396                          |


### Raw outputs:

Model name: LSTM
        Mode: train
        Overall accuracy: 0.3750
        Accuracy by answer type: other: 0.3106, yes/no: 0.2839, unanswerable: 0.5451, number: 0.4751
Model name: LSTM
        Mode: val
        Overall accuracy: 0.1987
        Accuracy by answer type: unanswerable: 0.3510, other: 0.1277, yes/no: 0.1236, number: 0.0917
Model name: T5
        Mode: train
        Overall accuracy: 0.4736
        Accuracy by answer type: other: 0.2935, yes/no: 0.8929, unanswerable: 0.8600, number: 0.2561
Model name: T5
        Mode: val
        Overall accuracy: 0.4690
        Accuracy by answer type: unanswerable: 0.8100, other: 0.2766, yes/no: 0.7421, number: 0.3125
Model name: resnet
        Mode: train
        Overall accuracy: 0.9252
        Accuracy by answer type: number: 0.9581, other: 0.8934, unanswerable: 0.9925, yes/no: 0.9834
Model name: resnet
        Mode: val
        Overall accuracy: 0.3635
        Accuracy by answer type: unanswerable: 0.6277, other: 0.2339, yes/no: 0.3231, number: 0.1687
Model name: vit
        Mode: train
        Overall accuracy: 0.7439
        Accuracy by answer type: other: 0.6553, unanswerable: 0.9923, yes/no: 0.5947, number: 0.6910
Model name: vit
        Mode: val
        Overall accuracy: 0.3877
        Accuracy by answer type: unanswerable: 0.7187, other: 0.2276, yes/no: 0.2969, number: 0.1750
Model name: clip
        Mode: train
        Overall accuracy: 0.9193
        Accuracy by answer type: other: 0.8825, number: 0.9419, unanswerable: 0.9980, yes/no: 0.9841
Model name: clip
        Mode: val
        Overall accuracy: 0.4520
        Accuracy by answer type: unanswerable: 0.8055, other: 0.2615, yes/no: 0.6031, number: 0.3167
Model name: vit_bert_attn
        Mode: train
        Overall accuracy: 0.0000
        Accuracy by answer type: other: 0.0000, unanswerable: 0.0000, number: 0.0000, yes/no: 0.0000
Model name: vit_bert_attn
        Mode: val
        Overall accuracy: 0.0000
        Accuracy by answer type: unanswerable: 0.0000, other: 0.0000, yes/no: 0.0000, number: 0.0000
Model name: vit_bert
        Mode: train
        Overall accuracy: 0.8676
        Accuracy by answer type: other: 0.8129, unanswerable: 0.9875, yes/no: 0.9541, number: 0.8877
Model name: vit_bert
        Mode: val
        Overall accuracy: 0.3593
        Accuracy by answer type: unanswerable: 0.6569, other: 0.2132, yes/no: 0.2995, number: 0.2063
Model name: CompetitiveBaseline_CrossAttention
        Mode: train
        Overall accuracy: 0.6658
        Accuracy by answer type: other: 0.6801, yes/no: 0.9771, unanswerable: 0.5704, number: 0.7764
Model name: CompetitiveBaseline_CrossAttention
        Mode: val
        Overall accuracy: 0.6257
        Accuracy by answer type: unanswerable: 0.5313, other: 0.6490, yes/no: 0.9446, number: 0.7521
Model name: CompetitiveBaseline_CLIP
        Mode: train
        Overall accuracy: 0.7794
        Accuracy by answer type: other: 0.8589, yes/no: 0.8001, unanswerable: 0.5692, number: 0.9522
Model name: CompetitiveBaseline_CLIP
        Mode: val
        Overall accuracy: 0.7336
        Accuracy by answer type: unanswerable: 0.5165, other: 0.8425, yes/no: 0.7138, number: 0.9708
Model name: ViLT
        Mode: train
        Overall accuracy: 0.1783
        Accuracy by answer type: other: 0.1881, yes/no: 0.7009, unanswerable: 0.0692, number: 0.0751
Model name: ViLT
        Mode: val
        Overall accuracy: 0.1777
        Accuracy by answer type: unanswerable: 0.0825, other: 0.1931, yes/no: 0.6749, number: 0.0396