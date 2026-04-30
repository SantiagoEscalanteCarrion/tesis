# Antecedente [3] — Zhang et al. (2023) ScolioNets

## Referencia completa
Zhang T, Zhu C, Zhao Y, et al. "Deep Learning Model to Classify and Monitor
Idiopathic Scoliosis in Adolescents Using a Single Smartphone Photograph."
*JAMA Network Open*. 2023;6(8):e2330617.
doi:10.1001/jamanetworkopen.2023.30617

## Arquitectura original (ScolioNets)
- **Backbone**: CNN multicapa con mecanismos de atención (224×224 input)
- **Atención**: Attention Branch Network — Fukui et al., CVPR 2019 (ref [29])
- **Tarea original**: Clasificación 3 clases de severidad (ángulo de Cobb)
- **Dataset original**: 1780 pacientes, clínica Hong Kong, radiografías + fotos
- **AUC reportado**: 0.839–0.902 según tarea

## Reproducción en este trabajo

### Arquitectura implementada
- **Backbone**: ResNet50 preentrenado en ImageNet (más cercano al ABN original)
- **Atención**: Attention Branch Network (fiel a ref [29])
- **Input**: 224×224×3 (especificado en el paper)
- **Fine-tuning**: conv5_block de ResNet50 (equivalente a block6/7 de EfficientNet)

### Adaptaciones documentadas
| Aspecto | Original | Esta reproducción | Justificación |
|---------|----------|------------------|---------------|
| Backbone exacto | No publicado en texto principal (eAppendix 4) | ResNet50 | Arquitectura estándar para ABN, compatible con 224×224 |
| Tarea | 3 clases (leve/moderada/severa) | Binaria (scoliosis_yes/no) | Dataset sin ángulo de Cobb medido |
| Video matting | Modelo propietario AlignProCARE | Omitido | Modelo no publicado, no reproducible |

### Protocolo de evaluación
- 5-Fold Cross-Validation sobre imágenes originales (mismo protocolo que E1/E2/E3)
- Train fold: orig + aug_* del grupo train
- Test fold: solo orig del grupo test
- Métricas: Accuracy, F1-macro, AUC-ROC (media ± std)

## Cómo ejecutar
```powershell
cd Antecedentes/zhang_2023_scolionets
python evaluate.py
```
