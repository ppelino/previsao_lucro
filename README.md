# Projeto de Machine Learning

## Definição do Problema

Neste projeto, vamos construir um modelo de Machine Learning para analisar uma projeção de lucro na venda de sorvetes.

## Etapas do Projeto

### Etapa 1: Importação das Bibliotecas

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
