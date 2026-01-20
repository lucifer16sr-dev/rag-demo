# Machine Learning Basics

Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.

## Types of Machine Learning

1. **Supervised Learning**: Learning from labeled training data
   - Classification: Predicting categories
   - Regression: Predicting continuous values

2. **Unsupervised Learning**: Finding patterns in unlabeled data
   - Clustering: Grouping similar data points
   - Dimensionality reduction

3. **Reinforcement Learning**: Learning through trial and error with rewards

## Python Libraries

Popular Python libraries for machine learning include:
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms
- **TensorFlow**: Deep learning framework
- **PyTorch**: Deep learning framework

## Example: Linear Regression

```python
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1], [2], [3]])
y = np.array([2, 4, 6])

model = LinearRegression()
model.fit(X, y)
predictions = model.predict([[4]])
```
