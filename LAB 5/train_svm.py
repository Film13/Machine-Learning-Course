import matplotlib.pyplot as plt
import pandas as pd
from data_loader import load_and_explore_data
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


def main():
    # 1. โหลดข้อมูล
    df = load_and_explore_data()

    # 2. ทำ Preprocessing ข้อมูล
    data = df[['Age', 'Gender', 'Size', 'Breed']].dropna()
    X = data[['Age', 'Size', 'Breed']]
    y = data['Gender']

    # 3. กำหนด Pipeline สำหรับ Preprocessor
    categorical_features = ['Age', 'Size', 'Breed']
    preprocessor = ColumnTransformer([
        (
            'categorical',
            Pipeline([
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False)),
            ]),
            categorical_features,
        )
    ])

    # 4. แบ่ง Data สำหรับ Train / Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 5. เทรน SVM ในแต่ละ Kernel
    kernels = ['linear', 'poly', 'rbf']
    results = {}
    models = {}

    for kernel in kernels:
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('svm', SVC(kernel=kernel)),
        ])

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        models[kernel] = model
        results[kernel] = accuracy
        print(f'{kernel.capitalize()} Kernel Accuracy: {accuracy:.4f}')

    # 6. เปรียบเทียบผลลัพธ์
    results_df = pd.DataFrame(
        {'Kernel': list(results.keys()), 'Accuracy': list(results.values())}
    )
    print('\nComparison Results:\n', results_df)

    plt.bar(results_df['Kernel'], results_df['Accuracy'])
    plt.title('SVM Kernel Accuracy Comparison')
    plt.xlabel('Kernel')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig('svm_accuracy_output.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 7. ทดสอบ Prediction ตัวอย่างข้อมูลใหม่
    prediction_model = models['linear']
    sample_data = pd.DataFrame({
        'Age': ['Adult', 'Young', 'Senior'],
        'Size': ['Medium', 'Small', 'Large'],
        'Breed': ['Persian', 'Mixed Breed', 'Domestic Short Hair'],
    })

    predictions = prediction_model.predict(sample_data)
    prediction_result = sample_data.copy()
    prediction_result['Predicted Gender'] = predictions
    print('\nPrediction Example:\n', prediction_result)


if __name__ == '__main__':
    main()