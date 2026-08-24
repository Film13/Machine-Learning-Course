import matplotlib.pyplot as plt
import pandas as pd
from config import DATA_PATH


def load_and_explore_data():
    df = pd.read_csv(DATA_PATH, header=None)
    df.columns = [
        'RecordID',
        'PetID',
        'URL',
        'AnimalType',
        'Age',
        'Gender',
        'Size',
        'Unnamed',
        'Breed',
        'PhotoData',
        'PhotoURLs',
    ]

    print('Dataset shape:', df.shape)
    print('\nMissing values:\n', df.isnull().sum())
    print('\nGender distribution:\n', df['Gender'].value_counts())

    # พล็อต กราฟแสดง Gender Distribution
    df['Gender'].value_counts().plot(kind='bar')
    plt.title('Gender Distribution')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.show()

    return df


if __name__ == '__main__':
    load_and_explore_data()