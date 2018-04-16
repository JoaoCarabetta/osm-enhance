import osmnx as ox
import pandas as pd
import pickle
import re
import numpy as np
import glob
import networkx as nx
from copy import deepcopy
from sklearn import preprocessing
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import warnings



class Train:

    def __init__(self, addresses):

        self.addresses = addresses

        self.save_path = 'app/app/resources/'

        # From methods
        self.raw_data = None
        self.data = None
        self.ready_data = None
        self.final_data = None
        self.encoders = None
        self.models = dict()

    def fit(self):

        print('Downloading...')
        self.raw_data = self.download_data()

        print('% of complete lanes: ', len(self.raw_data.dropna(subset=['lanes'])) / len(self.raw_data) * 100)
        print('% of complete maxspeed: ', len(self.raw_data.dropna(subset=['maxspeed'])) / len(self.raw_data) * 100)
        print('% of complete highway: ', len(self.raw_data[self.raw_data['highway'] != 'unclassified']) /
                                             len(self.raw_data) * 100)
        print('\nTreating...')
        self.data = self.treat_data(self.raw_data)

        print('\nPreprocessing...')
        self.ready_data = self.preprocess_data(self.data)
        self.ready_data.to_csv(self.save_path + 'ready_data/ready.csv')

        print('\nTraining...')
        data = self.ready_data
        for column in ['lanes', 'maxspeed', 'highway']:
            print(column)
            data, model = self.train_predict(data, column, self.encoders)
            self.models[column] = model
        self.final_data = data
        self.final_data.to_csv(self.save_path + 'final_data/encoded.csv')

        print('\nPickling')
        pickle.dump(self.models['highway'].fitted_pipeline_, open(self.save_path + 'models/model.p', 'wb'))

        print('\nDecoding')
        self.final_data = self.restore_data(self.decode(self.final_data))
        self.final_data.to_csv(self.save_path + 'final_data/final.csv')

    def check_data_exists(self, address):

        files = glob.glob(self.save_path + '/addresses/*')
        files = [file.split('.')[0].split('/')[-1] for file in files]
        print(address, files)
        if address in files:
            return True
        else:
            return False

    def download_data(self):

        all_g = []

        for address in self.addresses:

            print('Address: ', address)

            if self.check_data_exists(address):
                G = pickle.load(open(self.save_path + '/addresses/{}.p'.format(address), 'rb'))
            else:
                G = ox.graph_from_place(address, network_type='drive')
                G = ox.project_graph(G)
                pickle.dump(G, open(self.save_path + '/addresses/{}.p'.format(address), 'wb'))

            all_g.append(G)

        for i, g in enumerate(all_g):

            if i == 0:
                raw_data = pd.DataFrame([data for u, v, key, data in g.edges(keys=True, data=True)])

            else:
                d = nx.to_pandas_edgelist(g)
                raw_data = pd.concat(d, raw_data)

        raw_data.to_csv(self.save_path + '/raw_data/raw.csv')

        return raw_data

    ### Treat data

    def split_data_frame_list(self,
                              df,
                              target_column,
                              output_type=float):
        ''' df = dataframe to split,
        target_column = the column containing the values to split
        separator = the symbol used to perform the split
        returns: a dataframe with each entry for the target column separated, with each element moved into a new row.
        The values in the other columns are duplicated across the newly divided rows.
        '''
        row_accumulator = []

        def split_list_to_rows(row):
            split_row = row[target_column]
            if isinstance(split_row, list):
                for s in split_row:
                    new_row = row.to_dict()
                    new_row[target_column] = output_type(s)
                    row_accumulator.append(new_row)
            else:
                new_row = row.to_dict()
                try:
                    new_row[target_column] = float(split_row)
                except:
                    new_row[target_column] = output_type(split_row)
                row_accumulator.append(new_row)

        df.apply(split_list_to_rows, axis=1)
        new_df = pd.DataFrame(row_accumulator)

        return new_df

    def clean_speed_number(self, x):

        try:
            return float(re.findall(r'\d+', x)[0])

        except:
            return float(x)

    def treat_data(self, df):

        df = self.split_data_frame_list(df, 'lanes', output_type=float)

        df = self.split_data_frame_list(df, 'highway', output_type=str)

        df = self.split_data_frame_list(df, 'maxspeed', output_type=str)

        df = self.split_data_frame_list(df, 'tunnel', output_type=str)

        df['maxspeed'] = df['maxspeed'].apply(self.clean_speed_number)

        df = df.replace('nan', np.nan)

        return df

    #  Prepare to Fit
    def rescale(self, df):

        columns = df.columns
        min_max_scaler = preprocessing.MinMaxScaler()
        df = min_max_scaler.fit_transform(df)

        df = pd.DataFrame(df, columns=columns)
        return df

    def fill_na(self, df):
        for column in df.columns:
            if df[column].dtype == 'O':
                df[column] = df[column].fillna('missing')
                df[column] = df[column].replace('unclassified', 'missing')

            elif df[column].dtype == 'float64':
                df[column] = df[column].fillna(0)

        return df

    def label_encoder(self, df):

        le = preprocessing.LabelEncoder()

        encoders = dict()

        for column in df.columns:
            if (df[column].dtype == 'O') or (df[column].dtype == 'bool'):
                print(column)
                le.fit(df[column])
                df[column] = le.transform(df[column])
                print(le.classes_)
                encoders[column] = deepcopy(le)

        return df, encoders

    def preprocess_data(self, df):

        df = df.reset_index()

        df = df[['index',
                  'access',
                  'bridge',
                  'highway',
                  'junction',
                  'lanes',
                  'length',
                  'maxspeed',
                  'oneway',
                  'tunnel',]]

        df = self.fill_na(df)
        df, self.encoders = self.label_encoder(df)
        # df = rescale(df)

        return df


    #  Training

    def select_missing(self, encoders, target_column):

        print(target_column)
        if target_column in encoders.keys():

            missing = encoders[target_column].transform(['missing'])[0]

        else:

            missing = 0

        return missing

    def train(self, df, target_column, encoders):

        missing = self.select_missing(encoders, target_column)

        known = df[df[target_column] != missing]

        print('Dataset Size: ', len(known))

        y = known.pop(target_column).values
        X = known.values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        model = TPOTClassifier(generations=1, population_size=1, verbosity=2, n_jobs=-1)
        model.fit(X_train, y_train)
        print('Test Score: ', model.score(X_test, y_test))

        return model

    def apply_model_to_dataset(self, df, target_column, model, encoders):

        missing = self.select_missing(encoders, target_column)

        unkown = df[df[target_column] == missing]

        unkown.pop(target_column)

        unkown[target_column] = model.predict(unkown.values)

        df.iloc[unkown.index] = unkown

        return df

    def train_predict(self, df, target_column, encoders):

        model = self.train(df, target_column, encoders)

        df = self.apply_model_to_dataset(df, target_column, model, encoders)

        self.save_model(target_column, model)

        return df, model

    def save_model(self, target_column, model):

        filename = self.save_path + 'pipelines/model_{}.py'.format(target_column)

        model.export(filename)

    def decode(self, df):
        for column in df.columns:
            if column in self.encoders.keys():
                df[column] = df[column].apply(self.encoders[column].inverse_transform)

        return df

    def restore_data(self, df):

        for col in self.data.columns:
            if col not in df.columns:
                df[col] = self.data[col]
        return df

    def to_osm(self, df):

        return nx.from_pandas_edgelist(df, source='source', target='target', edge_attr=True)

if __name__ == "__main__":
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    t = Train([
        'Manhattan Island, New York City, New York, USA'])
    t.fit()
