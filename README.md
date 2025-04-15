# LuciaDoval-fcc_sms_text_classification

This project uses a machine learning model to classify SMS messages as **"spam"** (unwanted or advertising messages) or **"ham"** (normal messages, usually from known contacts). A neural network model with a bidirectional LSTM layer is trained to capture the context of the text messages.

---

## 🧠 Technologies Used

- Python
- TensorFlow / Keras
- Pandas
- Numpy
- Matplotlib (optional for visualization)
- TensorFlow Datasets (for future integrations)

---

## 📂 Project Structure

```
├── fcc_sms_text_classification.ipynb       
├── README.md                    # This file
```

---

## ⚙️ Training the Model

The model is a neural network with the following architecture:

- Embedding Layer (`input_dim=10000`, `output_dim=64`)
- Bidirectional LSTM Layer
- Dense Layer (ReLU activation)
- Output Layer with sigmoid activation

```python
model = keras.Sequential([
    keras.layers.Embedding(10000, 64, input_length=max_length),
    keras.layers.Bidirectional(keras.layers.LSTM(64)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
```

Compiled with:
```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---

## 🧪 Using the Model

Once the model is trained, you can predict new messages with the following function:

```python
predict_message("Win a free iPhone now!!!")
# Result: [0.973, 'spam']
```

This function returns a list with:
1. The probability of being **spam**
2. The predicted label: `'spam'` or `'ham'`

---

## 📊 Example Evaluation

After training the model, an accuracy of over 95% is typically achieved on the validation set, depending on the parameters and preprocessing used.

---

## 📌 Notes

- The data is sourced from the [SMS Spam Collection Dataset](https://www.dt.fee.unicamp.br/~tiago/smsspamcollection/).
- Tokenization and padding are performed using Keras' `Tokenizer` and `pad_sequences`.

---

## 🚀 Requirements

- Python ≥ 3.7
- TensorFlow ≥ 2.8
- Pandas
- NumPy

To install dependencies:
```bash
pip install tensorflow pandas numpy
```

---

## 📬 Contact

If you have any questions or suggestions, feel free to open an "issue" or contact me directly. Thanks for reviewing this project!
